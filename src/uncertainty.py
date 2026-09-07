"""Whether a model's predicted spread tracks the error it actually makes.

Two quantities are called uncertainty here and they are not the same thing.

The model's own spread is what a regressor reports alongside its point
prediction, which for the tabular foundation models is the spread of their
in-context predictive distribution. Not every regressor reports one.

The ensemble spread is the standard deviation across the five training seeds'
point predictions for a compound. Every configuration has one, including the
graph networks, which report no spread of their own. It measures disagreement
between models rather than any model's own belief.

Both are scored the same two ways. A rank correlation against absolute
residual asks the weak question, whether more uncertain predictions are more
often wrong, and answers it as a degree rather than a yes or no. The
miscalibration area asks the calibrated question: treating each prediction as
Gaussian, the fraction of compounds falling inside the central p interval
should be p for every p, and the area between the observed curve and that
diagonal is how far off it is. Zero is perfect; a model that is uniformly
overconfident and one that is uniformly underconfident both score above zero,
and the signed gap says which.

An affine calibration rescales predictions, so it rescales the spread with
them: under ``calibrated = slope * raw + intercept`` the predictive standard
deviation becomes ``|slope| * sigma``. Applying the map to the mean while
leaving the spread alone would report a different distribution from the one
the calibration implies, so the scaling is applied here and recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)

# confidence levels the calibration curve is evaluated on, excluding the
# endpoints where every model is trivially right
QUANTILE_GRID = np.linspace(0.01, 0.99, 99)


class UncertaintyError(ValueError):
    """An uncertainty diagnostic could not be computed from what it was given."""


@dataclass(frozen=True)
class Diagnostic:
    """How well one spread tracked one set of errors.

    Attributes
    ----------
    source : str
        ``model`` for a regressor's own reported spread, ``ensemble`` for the
        standard deviation across seeds.
    n : int
        Compounds scored.
    spearman, pearson : float
        Correlation between the spread and the absolute residual. Spearman is
        the one to quote: it asks whether the ordering is right without
        assuming the relationship is linear.
    miscalibration_area : float
        Area between the observed coverage curve and the diagonal, in [0, 1].
    mean_sigma, mean_abs_residual : float
        The two scales, so a reader can see whether the spread is the right
        size at all rather than only the right shape.
    signed_gap : float
        Mean of observed coverage minus expected. Negative is overconfident,
        the intervals being too narrow; positive is underconfident.
    """

    source: str
    n: int
    spearman: float
    pearson: float
    miscalibration_area: float
    mean_sigma: float
    mean_abs_residual: float
    signed_gap: float

    def as_dict(self) -> dict[str, Any]:
        """Return the diagnostic as plain data for a record."""
        return {
            "source": self.source,
            "n": self.n,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "miscalibration_area": self.miscalibration_area,
            "mean_sigma": self.mean_sigma,
            "mean_abs_residual": self.mean_abs_residual,
            "signed_gap": self.signed_gap,
        }


def coverage_curve(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    levels: NDArray[np.float64] = QUANTILE_GRID,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return expected against observed coverage, treating each prediction as Gaussian.

    Parameters
    ----------
    observed, predicted, sigma : ndarray
        True values, point predictions, and predictive standard deviations.
    levels : ndarray, optional
        Central-interval probabilities to evaluate.

    Returns
    -------
    expected, observed_fraction : ndarray
        Both of length ``len(levels)``. ``observed_fraction[i]`` is the share
        of compounds whose true value falls inside the central
        ``levels[i]`` interval of its own predicted distribution.

    Raises
    ------
    UncertaintyError
        If the arrays disagree in shape or every spread is zero, which leaves
        coverage undefined rather than merely bad.
    """
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    sd = np.asarray(sigma, dtype=np.float64)
    if not obs.shape == pred.shape == sd.shape:
        raise UncertaintyError(f"shape mismatch: {obs.shape}, {pred.shape}, {sd.shape}")
    if obs.size == 0:
        raise UncertaintyError("no compounds to score")
    if not np.any(sd > 0):
        raise UncertaintyError("every predicted spread is zero, so coverage is undefined")

    # a zero spread would make the standardized residual infinite; the compound
    # is either exactly right or infinitely wrong, and the second is the honest
    # reading, so it is kept rather than dropped
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(obs - pred) / sd
    z = np.where(sd > 0, z, np.inf)

    # the central interval of probability p is |z| <= the (1+p)/2 quantile
    thresholds = stats.norm.ppf(0.5 + levels / 2.0)
    fractions = np.array([float(np.mean(z <= t)) for t in thresholds])
    return np.asarray(levels, dtype=np.float64), fractions


def miscalibration_area(
    expected: NDArray[np.float64], observed_fraction: NDArray[np.float64]
) -> float:
    """Return the area between the coverage curve and the diagonal.

    Trapezoidal over the expected axis, so the value is comparable across
    different grids and lies in [0, 1].
    """
    return float(np.trapezoid(np.abs(observed_fraction - expected), expected))


def diagnose(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    source: str,
) -> Diagnostic:
    """Score one spread against the residuals it is supposed to track."""
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    sd = np.asarray(sigma, dtype=np.float64)
    residual = np.abs(obs - pred)

    if np.allclose(sd, sd.flat[0]):
        # a constant spread has no ordering to correlate, and reporting a
        # correlation of nan is clearer than reporting zero
        spearman = pearson = float("nan")
    else:
        spearman = float(stats.spearmanr(sd, residual).statistic)
        pearson = float(stats.pearsonr(sd, residual).statistic)

    expected, fractions = coverage_curve(obs, pred, sd)
    return Diagnostic(
        source=source,
        n=int(obs.size),
        spearman=spearman,
        pearson=pearson,
        miscalibration_area=miscalibration_area(expected, fractions),
        mean_sigma=float(sd.mean()),
        mean_abs_residual=float(residual.mean()),
        signed_gap=float(np.mean(fractions - expected)),
    )


def ensemble_spread(stacked: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the per-compound standard deviation across seeds.

    Parameters
    ----------
    stacked : ndarray
        Shape ``(n_seeds, n_compounds)``, one row per seed.

    Returns
    -------
    ndarray
        One standard deviation per compound, with the sample convention
        (``ddof=1``) since the seeds are a sample of training runs rather than
        the population of them.

    Raises
    ------
    UncertaintyError
        If fewer than two seeds are given, which leaves the spread undefined.
    """
    rows = np.asarray(stacked, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise UncertaintyError(f"need at least two seeds to have a spread, got {rows.shape}")
    return rows.std(axis=0, ddof=1)


def rescale_sigma(sigma: NDArray[np.float64], slope: float) -> NDArray[np.float64]:
    """Return the predictive spread under an affine calibration.

    ``calibrated = slope * raw + intercept`` maps the whole predictive
    distribution, not only its mean, so the standard deviation scales by the
    absolute slope. Leaving it unscaled would report a distribution the
    calibration does not imply.
    """
    return np.abs(float(slope)) * np.asarray(sigma, dtype=np.float64)
