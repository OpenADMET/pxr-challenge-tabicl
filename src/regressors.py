"""Stage three: one uniform adapter over the seven regressors the sweep fits.

Every regressor here answers the same three questions: what parameters it runs
under, how to construct it, and how to get a prediction with an uncertainty out
of it. The sweep never learns that TabPFN pins a checkpoint version, that TabICL
takes ``None`` rather than ``"auto"`` for its device, or that TabFM wants a
separately loaded weights object handed to its constructor. Those differences
live in this module and nowhere else.

Parameters are resolved before anything is constructed. ``resolved_params``
returns the complete, JSON-serialisable set a fit would run under, which is what
goes into the cache key and the provenance record, so a cached prediction can be
read back and matched against the settings that produced it. Nothing that
changes the numbers is left implicit: where a library default matters, it is
written out here rather than inherited silently, so a future version of that
library bumping its own default renames the artifact instead of quietly changing
it.

Semantics are versioned by hand, on the same convention as ``features.BLOCKS``.
Bump a regressor's ``version`` when the meaning of its output changes, never for
a change that cannot alter the numbers.

Where a model exposes a predictive distribution, ``fit_predict`` returns a
spread alongside the point estimate. TabPFN's bar-distribution criterion gives
an exact predictive variance; TabICL gives quantiles, from which a
normal-equivalent scale is taken. The gradient-boosted models and TabFM expose a
point estimate only and report ``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# TabPFN documents its pretraining regime up to this many features and refuses
# wider matrices unless explicitly told to proceed; the widest featuresets in
# this sweep run to roughly 2,300 columns, so the flag has to be derived
PRETRAINING_FEATURE_LIMIT = 2000

# tabicl 2.1.1's own default. Written out rather than inherited because peak
# memory does not move monotonically with it across feature widths: at 130
# columns and 4,392 training rows this value OOMs a 24 GB GPU and batch_size=2
# fits, while at 258 and 386 columns the reverse holds and only this value
# fits. There is no batch size that is safe everywhere, so callers must choose
# per featureset and the choice must be visible in the record.
TABICL_BATCH_SIZE = 8

# TabFM's in-context rows enter the same between-items attention TabPFN's do.
# The full 4,392-row training context OOMs this GPU, and so did caps of 2,000
# and 1,000; 500 is the largest that fitted.
TABFM_MAX_ROWS = 500

# tabfm 1.0.1's own default feature cap, past which it subsamples columns.
# Recorded because subsampling changes the numbers.
TABFM_MAX_FEATURES = 500

# TabFM's default ensemble size
TABFM_N_ESTIMATORS = 32

# quantile levels a normal-equivalent scale is taken between, and the width of
# that interval in standard deviations of a normal distribution
SPREAD_QUANTILES = (0.1, 0.9)
SPREAD_QUANTILE_WIDTH = 2.0 * 1.2815515655446004

# names whose estimator is a version-pinned TabPFN checkpoint
TABPFN_NAMES = ("tabpfn-v2.5", "tabpfn-v2.6", "tabpfn-v3")


class RegressorError(RuntimeError):
    """A regressor could not be built or fitted under the parameters it was given."""


@dataclass(frozen=True)
class Prediction:
    """A regressor's output on a test matrix.

    Attributes
    ----------
    mean : ndarray
        Point prediction, one value per test row.
    std : ndarray or None
        Predictive spread, one value per test row, or None where the model
        exposes no distribution. This is an exact predictive standard deviation
        for the TabPFN checkpoints and a normal-equivalent scale taken from the
        10th and 90th predictive percentiles for TabICL, so it is comparable
        within a model but is a width, not a moment, across models.
    """

    mean: np.ndarray
    std: np.ndarray | None


@dataclass(frozen=True)
class _Spec:
    """A named regressor: how to build it, and what changes its meaning.

    Attributes
    ----------
    version : int
        Semantic version, bumped by hand when the meaning of the output changes.
    defaults : dict
        Parameters the regressor runs under unless a caller overrides them.
    construct : callable
        Turns a fully resolved parameter set into an unfitted estimator.
    resolve : callable
        Derives the parameters that depend on the seed or the matrix width.
    """

    version: int
    defaults: dict[str, Any]
    construct: Callable[[dict[str, Any]], Any]
    resolve: Callable[..., dict[str, Any]]


def resolved_params(
    name: str,
    *,
    seed: int,
    n_features: int,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete parameter set a fit would run under.

    The result is JSON-serialisable and holds everything that changes the
    numbers, so it can go straight into a cache key and a provenance record.
    A configured device of ``"auto"`` stays ``"auto"``: which accelerator that
    resolves to belongs to the environment record, not to the specification.

    Parameters
    ----------
    name : str
        Regressor name, one of the keys of ``REGRESSORS``.
    seed : int
        Seed threaded into the estimator's ``random_state``.
    n_features : int
        Width of the matrix the regressor will be fitted on. Only the TabPFN
        checkpoints use it, to decide whether their pretraining limits apply.
    params : mapping, optional
        Overrides merged over the regressor's defaults.

    Returns
    -------
    dict
        The resolved parameters.

    Raises
    ------
    KeyError
        If the regressor name is not known.
    """
    spec = _spec(name)
    merged = {**spec.defaults, **(params or {})}
    return spec.resolve(merged, seed=seed, n_features=n_features)


def build(
    name: str,
    *,
    seed: int,
    n_features: int,
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Construct an unfitted estimator with sklearn ``fit``/``predict``.

    Parameters
    ----------
    name : str
        Regressor name, one of the keys of ``REGRESSORS``.
    seed : int
        Seed threaded into the estimator's ``random_state``.
    n_features : int
        Width of the matrix the regressor will be fitted on.
    params : mapping, optional
        Overrides merged over the regressor's defaults.

    Returns
    -------
    object
        The unfitted estimator.

    Raises
    ------
    KeyError
        If the regressor name is not known.
    """
    spec = _spec(name)
    resolved = resolved_params(name, seed=seed, n_features=n_features, params=params)
    return spec.construct(resolved)


def fit_predict(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    params: Mapping[str, Any] | None = None,
) -> Prediction:
    """Fit a regressor on the training matrix and predict the test matrix.

    An out-of-memory failure is re-raised naming the parameters that produced
    it rather than retried under quieter settings, so a sweep never records a
    number obtained under a configuration nobody asked for.

    Parameters
    ----------
    name : str
        Regressor name, one of the keys of ``REGRESSORS``.
    x_train, y_train : ndarray
        Training matrix and target.
    x_test : ndarray
        Matrix to predict.
    seed : int
        Seed threaded into the estimator's ``random_state``.
    params : mapping, optional
        Overrides merged over the regressor's defaults.

    Returns
    -------
    Prediction
        Point predictions, and a predictive spread where the model has one.

    Raises
    ------
    KeyError
        If the regressor name is not known.
    RegressorError
        If the fit ran out of accelerator memory.
    """
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError(
            f"{name}: test matrix has {x_test.shape[1]} columns, training matrix has "
            f"{x_train.shape[1]}"
        )

    resolved = resolved_params(name, seed=seed, n_features=x_train.shape[1], params=params)
    logger.info(
        "%s: fitting on %d x %d, predicting %d rows",
        name,
        *x_train.shape,
        x_test.shape[0],
    )

    # a sweep fits thousands of models in one process, so the accelerator cache
    # is released after every fit whether or not it succeeded
    try:
        model = _spec(name).construct(resolved)
        model.fit(x_train, y_train)
        return _predict(name, model, x_test)
    except MemoryError as err:
        raise RegressorError(f"{name}: out of memory under {resolved}") from err
    except RuntimeError as err:
        if not _is_out_of_memory(err):
            raise
        raise RegressorError(f"{name}: out of memory under {resolved}") from err
    finally:
        _empty_accelerator_cache()


def _spec(name: str) -> _Spec:
    """Look a regressor up, failing with the names that do exist."""
    if name not in REGRESSORS:
        raise KeyError(f"unknown regressor {name!r}; known: {sorted(REGRESSORS)}")
    return REGRESSORS[name]


def _predict(name: str, model: Any, x_test: np.ndarray) -> Prediction:
    """Read a point estimate, and a spread where the model exposes a distribution."""
    if name in TABPFN_NAMES:
        # the full output carries the bar-distribution criterion and its
        # logits, already mapped back to the target's own scale, so the
        # variance it reports is a predictive variance in the target's units
        output = model.predict(x_test, output_type="full")
        variance = output["criterion"].variance(output["logits"]).detach().cpu().numpy()
        spread = np.sqrt(np.clip(np.asarray(variance, dtype=np.float64), 0.0, None))
        return Prediction(np.asarray(output["mean"], dtype=np.float64), spread)

    if name == "tabicl":
        # TabICL reports quantiles rather than moments; the 10th-to-90th
        # percentile span converted to a normal-equivalent scale is the closest
        # comparable width
        output = model.predict(
            x_test, output_type=["mean", "quantiles"], alphas=list(SPREAD_QUANTILES)
        )
        quantiles = np.asarray(output["quantiles"], dtype=np.float64)
        spread = (quantiles[:, 1] - quantiles[:, 0]) / SPREAD_QUANTILE_WIDTH
        return Prediction(np.asarray(output["mean"], dtype=np.float64), spread)

    return Prediction(np.asarray(model.predict(x_test), dtype=np.float64), None)


def _resolve_seeded(params: dict[str, Any], *, seed: int, n_features: int) -> dict[str, Any]:
    """Thread the seed through; most regressors derive nothing from the width."""
    del n_features
    return {**params, "random_state": seed}


def _resolve_tabpfn(params: dict[str, Any], *, seed: int, n_features: int) -> dict[str, Any]:
    """Thread the seed through and decide whether the pretraining limits apply."""
    resolved = {**params, "random_state": seed}

    # an explicit caller value wins, so a run can be forced either way
    resolved.setdefault("ignore_pretraining_limits", n_features > PRETRAINING_FEATURE_LIMIT)
    return resolved


def _construct_tabpfn(version_name: str) -> Callable[[dict[str, Any]], Any]:
    """Return a constructor pinned to one TabPFN checkpoint version."""

    def construct(params: dict[str, Any]) -> Any:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        # the checkpoint is always pinned: the library default moves between
        # releases, and three of these entries exist precisely to compare
        # checkpoints against each other
        return TabPFNRegressor.create_default_for_version(
            getattr(ModelVersion, version_name), **params
        )

    return construct


def _construct_tabicl(params: dict[str, Any]) -> Any:
    """Construct TabICL, translating the device convention this module uses."""
    from tabicl import TabICLRegressor

    params = dict(params)

    # TabICL spells "choose a device yourself" as None, not "auto"
    device = params.pop("device")
    return TabICLRegressor(device=None if device == "auto" else device, **params)


def _construct_tabfm(params: dict[str, Any]) -> Any:
    """Load TabFM's pretrained weights and construct a regressor around them.

    The weights are separately licensed (``tabfm-non-commercial-v1.0``, distinct
    from the Apache-licensed package code) and download from Hugging Face
    (``google/tabfm-1.0.0-pytorch``) on first use.
    """
    from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch

    params = dict(params)

    # unlike the other libraries TabFM needs a concrete device at load time
    device = _concrete_device(params.pop("device"))
    model = tabfm_v1_0_0_pytorch.load(model_type="regression", device=device)
    return TabFMRegressor(model=model, **params)


def _construct_lgbm(params: dict[str, Any]) -> Any:
    """Construct a LightGBM regressor."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params)


def _construct_xgboost(params: dict[str, Any]) -> Any:
    """Construct an XGBoost regressor."""
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


def _concrete_device(device: str) -> str:
    """Turn a configured device into the one a fit would actually land on."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_out_of_memory(err: RuntimeError) -> bool:
    """Report whether a runtime error is an accelerator out-of-memory failure."""
    import torch

    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


def _empty_accelerator_cache() -> None:
    """Release cached accelerator memory so the next fit in the sweep starts clean."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Bump a regressor's version when the meaning of its output changes; never for
# a change that cannot alter the numbers.
REGRESSORS: dict[str, _Spec] = {
    "lgbm": _Spec(1, {"verbose": -1}, _construct_lgbm, _resolve_seeded),
    "xgboost": _Spec(1, {}, _construct_xgboost, _resolve_seeded),
    "tabpfn-v2.5": _Spec(
        1,
        {"device": "auto", "memory_saving_mode": True, "fit_mode": "low_memory"},
        _construct_tabpfn("V2_5"),
        _resolve_tabpfn,
    ),
    "tabpfn-v2.6": _Spec(
        1,
        {"device": "auto", "memory_saving_mode": True, "fit_mode": "low_memory"},
        _construct_tabpfn("V2_6"),
        _resolve_tabpfn,
    ),
    "tabpfn-v3": _Spec(
        1,
        {"device": "auto", "memory_saving_mode": True, "fit_mode": "low_memory"},
        _construct_tabpfn("V3"),
        _resolve_tabpfn,
    ),
    "tabicl": _Spec(
        1,
        {"device": "auto", "batch_size": TABICL_BATCH_SIZE},
        _construct_tabicl,
        _resolve_seeded,
    ),
    "tabfm": _Spec(
        1,
        {
            "device": "auto",
            "n_estimators": TABFM_N_ESTIMATORS,
            "max_num_rows": TABFM_MAX_ROWS,
            "max_num_features": TABFM_MAX_FEATURES,
        },
        _construct_tabfm,
        _resolve_seeded,
    ),
}
