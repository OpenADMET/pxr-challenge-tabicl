"""Scoring: the challenge's metrics, seed ensembling, and the paired bootstrap.

The metric definitions here are transcribed from the previous generation of
this project, ``evaluate_predictions.py``'s ``compute_metrics``, so that a
number produced now is comparable with one already published. The definition
that matters most is RAE, which has no single convention: the denominator is
the total absolute deviation of the truth about its own mean, making RAE the
error of the model relative to the error of a constant mean predictor on the
same compounds. A value of 1 means the model is no better than predicting the
mean, and the reference implementation returns NaN rather than infinity when
the truth is constant.

An entry on the leaderboard was an ensemble, not a single seed, so a comparison
against it is only honest if our side is ensembled the same way: average the
per-compound predictions across seeds first, then score once.

Uncertainty here is over compounds, not over seeds. The evaluation set is 260
compounds, and :func:`paired_bootstrap` resamples those compounds with
replacement, recomputing both predictors' metrics on the same resample so the
difference is paired and the compound-to-compound noise common to both cancels.
It says nothing about how much a result moves when the training seed changes.
Seed spread is a separate source of variation, measured across seed-wise runs
and reported alongside the interval rather than folded into it.

Everything in this module is a pure function over arrays: no I/O, no caching,
no plotting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

# metrics that metrics() reports and paired_bootstrap can resample
METRIC_NAMES = ("mae", "rmse", "rae", "r2", "kendall_tau", "spearman_rho")


@dataclass(frozen=True)
class BootstrapResult:
    """One paired-bootstrap comparison of two predictors on the same compounds.

    Attributes
    ----------
    metric : str
        Name of the resampled metric, one of :data:`METRIC_NAMES`.
    observed : float
        The metric for ``pred_a`` minus the metric for ``pred_b``, computed on
        the compounds as observed rather than on a resample. For an error
        metric a negative value means ``pred_a`` is the better predictor.
    ci_low : float
        Lower bound of the percentile interval on that difference.
    ci_high : float
        Upper bound of the percentile interval on that difference.
    level : float
        Coverage of the interval, for example 0.95.
    n_resamples : int
        Number of compound resamples the interval was taken from.
    """

    metric: str
    observed: float
    ci_low: float
    ci_high: float
    level: float
    n_resamples: int


def metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute the challenge's metrics for one set of aligned predictions.

    Definitions follow ``compute_metrics`` in the previous generation's
    ``evaluate_predictions.py``:

    - ``mae``, ``rmse``: mean and root-mean-square of the residual
      ``y_pred - y_true``.
    - ``rae``: ``sum(|y_pred - y_true|) / sum(|y_true - mean(y_true)|)``, the
      relative absolute error against a constant mean predictor. NaN when the
      truth is constant.
    - ``r2``: ``1 - sum(residual ** 2) / sum((y_true - mean(y_true)) ** 2)``,
      NaN when the truth is constant. This is the coefficient of determination
      of the predictions, not a squared correlation.
    - ``kendall_tau``, ``spearman_rho``: ``scipy.stats.kendalltau`` (tau-b) and
      ``scipy.stats.spearmanr``, NaN with fewer than two compounds.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values, one per compound.
    y_pred : array-like
        Predicted values, aligned elementwise with `y_true`.

    Returns
    -------
    dict of str to float
        Keys ``n`` plus every name in :data:`METRIC_NAMES`. ``n`` is the
        compound count as a float, so the mapping is uniformly numeric.

    Raises
    ------
    ValueError
        If the two arrays do not have the same one-dimensional shape.

    Examples
    --------
    >>> scores = metrics([1.0, 2.0, 3.0, 4.0], [1.5, 1.5, 3.5, 5.0])
    >>> round(scores["mae"], 4), round(scores["rae"], 4), round(scores["r2"], 4)
    (0.625, 0.625, 0.65)
    """
    true, pred = _as_aligned(y_true, y_pred)
    if true.size == 0:
        return {"n": 0.0, **dict.fromkeys(METRIC_NAMES, float("nan"))}

    scores = {name: float(_METRICS[name](true, pred)) for name in METRIC_NAMES}
    return {"n": float(true.size), **scores}


def ensemble_mean(predictions: Sequence[ArrayLike] | ArrayLike) -> NDArray[np.float64]:
    """Average per-compound predictions across seeds.

    The leaderboard entry these results are compared against was an ensemble,
    so a like-for-like comparison averages the seed-wise predictions per
    compound and scores the average once, rather than averaging seed-wise
    scores.

    Parameters
    ----------
    predictions : sequence of array-like or array-like
        One row per seed, one column per compound; equivalently a sequence of
        equal-length per-seed prediction vectors.

    Returns
    -------
    ndarray
        The per-compound mean, of length equal to the compound count.

    Raises
    ------
    ValueError
        If the input is empty, is not two-dimensional, or holds seeds of
        differing length.

    Examples
    --------
    >>> ensemble_mean([[1.0, 2.0], [3.0, 4.0]])
    array([2., 3.])
    """
    if isinstance(predictions, Sequence) and not isinstance(predictions, str | bytes):
        lengths = {np.asarray(row).shape for row in predictions}
        if len(lengths) > 1:
            raise ValueError(f"seeds have differing shapes {sorted(map(str, lengths))}")

    stacked = np.asarray(predictions, dtype=np.float64)
    if stacked.ndim != 2:
        raise ValueError(f"expected a (n_seeds, n_compounds) block, got shape {stacked.shape}")
    if stacked.shape[0] == 0 or stacked.shape[1] == 0:
        raise ValueError(f"cannot ensemble an empty block of shape {stacked.shape}")
    return np.asarray(stacked.mean(axis=0), dtype=np.float64)


def paired_bootstrap(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    *,
    metric: str = "mae",
    n_resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap the metric difference between two predictors over compounds.

    Each resample draws compounds (rows) with replacement and scores both
    predictors on that same draw, so the difference is paired: variation from
    which compounds happen to be easy or hard is common to both predictors and
    cancels. The interval is the percentile interval of those differences, and
    it describes sampling of compounds only. Variation across training seeds is
    a separate quantity and belongs beside this interval, not inside it.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values, one per compound.
    pred_a : array-like
        First predictor's values, aligned with `y_true`.
    pred_b : array-like
        Second predictor's values, aligned with `y_true`.
    metric : str, optional
        Which of :data:`METRIC_NAMES` to resample. Default is ``mae``.
    n_resamples : int, optional
        Number of compound resamples. Default is 10000.
    level : float, optional
        Interval coverage, strictly between 0 and 1. Default is 0.95.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Default is 0.

    Returns
    -------
    BootstrapResult
        The observed difference, the interval bounds, and the resample count.

    Raises
    ------
    ValueError
        If the arrays disagree in shape, the metric is unknown, `n_resamples`
        is not positive, `level` is outside (0, 1), or every resample is
        degenerate (a metric undefined on constant truth, such as RAE or R^2,
        when resampling repeatedly draws a single distinct compound).
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {list(_METRICS)}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level}")

    true, a = _as_aligned(y_true, pred_a)
    _, b = _as_aligned(y_true, pred_b)
    if true.size == 0:
        raise ValueError("cannot bootstrap an empty compound set")

    # one index row per resample, drawn once so both predictors see the same
    # compounds in the same multiplicity, which is what makes the test paired
    rng = np.random.default_rng(seed)
    index = rng.integers(0, true.size, size=(n_resamples, true.size))
    score = _METRICS[metric]
    differences = np.asarray(score(true[index], a[index]) - score(true[index], b[index]))

    usable = np.isfinite(differences)
    if not usable.any():
        raise ValueError(
            f"every resample left {metric!r} undefined; the truth is likely near-constant"
        )

    tail = (1.0 - level) / 2.0
    ci_low, ci_high = np.quantile(differences[usable], [tail, 1.0 - tail])
    observed = float(score(true, a) - score(true, b))
    return BootstrapResult(
        metric=metric,
        observed=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        level=level,
        n_resamples=n_resamples,
    )


def excludes_zero(result: BootstrapResult) -> bool:
    """Return whether a bootstrap interval separates the two predictors.

    Separation is the interval lying wholly on one side of zero. An interval
    that straddles zero is not evidence that the predictors are equivalent, only
    that this evaluation set does not distinguish them.

    Parameters
    ----------
    result : BootstrapResult
        A comparison from :func:`paired_bootstrap`.

    Returns
    -------
    bool
        True when zero lies outside ``[ci_low, ci_high]``.
    """
    return result.ci_low > 0.0 or result.ci_high < 0.0


def _as_aligned(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return both inputs as float arrays, refusing to broadcast a mismatch."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError(f"expected one-dimensional arrays, got {true.shape} and {pred.shape}")
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true is {true.shape}, predictions are {pred.shape}")
    return true, pred


def _mae(true: NDArray[np.float64], pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the mean absolute residual along the last axis."""
    return np.asarray(np.mean(np.abs(pred - true), axis=-1))


def _rmse(true: NDArray[np.float64], pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the root-mean-square residual along the last axis."""
    return np.asarray(np.sqrt(np.mean((pred - true) ** 2, axis=-1)))


def _rae(true: NDArray[np.float64], pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return absolute error relative to a constant mean predictor's absolute error."""
    deviation = true - np.mean(true, axis=-1, keepdims=True)
    denominator = np.sum(np.abs(deviation), axis=-1)
    numerator = np.sum(np.abs(pred - true), axis=-1)
    return _divide_or_nan(numerator, denominator)


def _r2(true: NDArray[np.float64], pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the coefficient of determination along the last axis."""
    deviation = true - np.mean(true, axis=-1, keepdims=True)
    total = np.sum(deviation**2, axis=-1)
    residual = np.sum((pred - true) ** 2, axis=-1)
    return np.asarray(1.0 - _divide_or_nan(residual, total))


def _kendall_tau(true: NDArray[np.float64], pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Kendall's tau-b along the last axis."""
    return _rank_statistic(stats.kendalltau, true, pred)


def _spearman_rho(true: NDArray[np.float64], pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Spearman's rho along the last axis."""
    return _rank_statistic(stats.spearmanr, true, pred)


def _divide_or_nan(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Divide elementwise, yielding NaN wherever the denominator vanishes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = np.where(denominator > 0.0, numerator / denominator, np.nan)
    return np.asarray(quotient, dtype=np.float64)


def _rank_statistic(
    correlate: Callable[..., Any],
    true: NDArray[np.float64],
    pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a scipy rank correlation row by row, returning NaN below two points."""
    if true.shape[-1] < 2:
        return np.full(true.shape[:-1], np.nan, dtype=np.float64)
    if true.ndim == 1:
        # argument order matches the reference implementation; both are symmetric
        return np.asarray(correlate(pred, true).statistic, dtype=np.float64)
    rows = [correlate(p, t).statistic for t, p in zip(true, pred, strict=True)]
    return np.asarray(rows, dtype=np.float64)


_METRICS: dict[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]] = {
    "mae": _mae,
    "rmse": _rmse,
    "rae": _rae,
    "r2": _r2,
    "kendall_tau": _kendall_tau,
    "spearman_rho": _spearman_rho,
}
