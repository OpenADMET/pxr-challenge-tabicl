"""Split a predicted spread into measurement noise and model error, and simulate both.

A calibrated predictive standard deviation sigma describes how far a measured
value lands from the prediction. That distance has two parts with different
behavior under repetition. Measurement noise, with standard deviation s, is
drawn afresh every time a compound is measured. Model error, the gap between
the prediction and the compound's true value, is fixed for a given compound
and model, with standard deviation sqrt(sigma^2 - s^2) across compounds.
Replicate measurements average away the first part and leave the second.

Two kinds of measurement-noise model are compared. A fixed-share model gives
every compound the same fraction of sigma^2 as measurement noise, from 0 (all
model error, the case where replicates change nothing) to 1 (all measurement
noise, the case where replicates converge on a perfect ranking). The two
realistic models use the curve-fit standard error the challenge reports for
each pEC50 and a constant literature-style value.

The module also scores a predicted spread for each residual by its Gaussian
negative log-likelihood, which compares the spread sigma predicts with one
that adds the curve-fit standard error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from data import CANONICAL_COL, RAW_DIR, load_test_curve_fit_se

# constant run-to-run pEC50 noise assumed by the literature-style noise model
LITERATURE_NOISE = 0.3

# shares of sigma^2 assigned to measurement noise by the fixed-share models
NOISE_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
SHARE_PREFIX = "share_"
REALISTIC_NOISE_MODELS = ("curve_fit", "literature")
NOISE_MODELS = tuple(f"{SHARE_PREFIX}{share:g}" for share in NOISE_SHARES) + REALISTIC_NOISE_MODELS


def load_points(points_path: Path, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return a per-compound uncertainty table with each compound's curve-fit standard error.

    Parameters
    ----------
    points_path : path-like
        The ``points.csv`` run/06_uncertainty.py writes.
    raw_dir : path-like, optional
        Directory holding the raw challenge files. Defaults to ``data/raw``.

    Returns
    -------
    pandas.DataFrame
        The table, with the curve-fit standard error added as ``curve_fit_se``.

    Raises
    ------
    FileNotFoundError
        If the table does not exist.
    ValueError
        If the table has no ``model_sigma`` column or a compound has no
        curve-fit standard error.
    """
    if not points_path.exists():
        raise FileNotFoundError(f"no {points_path}; run run/06_uncertainty.py first")
    points = pd.read_csv(points_path)
    if "model_sigma" not in points:
        raise ValueError(f"{points_path}: the regressor reports no spread of its own")

    # Join the curve-fit standard errors by canonical structure
    points["curve_fit_se"] = points[CANONICAL_COL].map(load_test_curve_fit_se(raw_dir))
    if points["curve_fit_se"].isna().any():
        raise ValueError(f"{points_path}: some compounds have no curve-fit standard error")
    return points


def measurement_noise(
    model: str, sigma: NDArray[np.float64], curve_fit_se: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return each compound's measurement-noise standard deviation under one noise model.

    The result is capped at sigma, since measurement noise cannot exceed the
    total spread it is part of.

    Parameters
    ----------
    model : str
        One of ``NOISE_MODELS``: ``"share_<f>"`` assigns the fraction ``f`` of
        each compound's sigma^2 to measurement noise, ``"curve_fit"`` uses the
        reported curve-fit standard error, and ``"literature"`` uses
        ``LITERATURE_NOISE`` for every compound.
    sigma : array-like
        Predicted standard deviation of each compound.
    curve_fit_se : array-like
        Curve-fit standard error of each compound's measured pEC50.

    Returns
    -------
    numpy.ndarray
        Measurement-noise standard deviation per compound, at most sigma.

    Raises
    ------
    ValueError
        If ``model`` is not one of ``NOISE_MODELS``.
    """
    if model.startswith(SHARE_PREFIX) and model in NOISE_MODELS:
        return np.sqrt(float(model.removeprefix(SHARE_PREFIX))) * sigma
    if model == "curve_fit":
        return np.minimum(curve_fit_se, sigma)
    if model == "literature":
        return np.minimum(np.full_like(sigma, LITERATURE_NOISE), sigma)
    raise ValueError(f"unknown noise model {model!r}; expected one of {NOISE_MODELS}")


def spearman_rows(x: NDArray[np.float64], rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Spearman rho between a fixed vector and each row of a matrix.

    Parameters
    ----------
    x : array-like
        Values shared by every comparison, shape ``(n,)``.
    rows : array-like
        One set of values per row, shape ``(m, n)``.

    Returns
    -------
    numpy.ndarray
        One rho per row, shape ``(m,)``.
    """
    # Pearson correlation of ranks, with ties given their average rank
    row_ranks = stats.rankdata(rows, axis=1)
    x_ranks = stats.rankdata(x)
    row_centered = row_ranks - row_ranks.mean(axis=1, keepdims=True)
    x_centered = x_ranks - x_ranks.mean()
    scale = np.linalg.norm(row_centered, axis=1) * np.linalg.norm(x_centered)
    return (row_centered @ x_centered) / scale


def simulate_replicates(
    sigma: NDArray[np.float64],
    noise: NDArray[np.float64],
    n_replicates: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return rho between sigma and replicate-averaged absolute error, per simulation.

    Each simulation draws one model error per compound, then ``n_replicates``
    measurements around the true value, and averages the absolute distance of
    those measurements from the prediction.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound, assumed correct.
    noise : array-like
        Measurement-noise standard deviation of each compound, at most sigma.
    n_replicates : int
        Measurements simulated per compound.
    n_simulations : int
        Independent simulations to run.
    rng : numpy.random.Generator
        Source of the normal draws.

    Returns
    -------
    numpy.ndarray
        One rho per simulation, shape ``(n_simulations,)``.
    """
    # Fix one model error per compound, then draw replicate measurements around it
    model_sd = np.sqrt(sigma**2 - noise**2)
    model_error = rng.standard_normal((n_simulations, 1, sigma.size)) * model_sd
    replicates = rng.standard_normal((n_simulations, n_replicates, sigma.size)) * noise
    mean_error = np.abs(model_error + replicates).mean(axis=1)
    return spearman_rows(sigma, mean_error)


def gaussian_nll(residual: NDArray[np.float64], spread: NDArray[np.float64]) -> float:
    """Return the mean negative log-likelihood of residuals under zero-centered normals.

    Parameters
    ----------
    residual : array-like
        Signed residual, measurement minus prediction, of each compound.
    spread : array-like
        Predicted standard deviation of each residual, all positive.

    Returns
    -------
    float
        Mean of ``-log N(residual; 0, spread^2)`` across compounds, in nats.
    """
    return float(np.mean(0.5 * np.log(2 * np.pi * spread**2) + 0.5 * (residual / spread) ** 2))


def pair_ranking_probability(ratio: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return how often a correct sigma ranks one pair of compounds' errors correctly.

    For two compounds whose errors are single normal draws with standard
    deviations in the given ratio (larger over smaller), this is the
    probability that the compound with the larger sigma has the larger
    absolute error, even when the true values are known exactly.

    Parameters
    ----------
    ratio : array-like
        Larger sigma divided by smaller sigma, each at least 1.

    Returns
    -------
    numpy.ndarray
        Probability of the correct order, from 0.5 at a ratio of 1 toward 1.

    Notes
    -----
    The ratio of two independent half-normal draws with unit scale has the
    cumulative distribution (2/pi) arctan(r), which gives the probability.
    """
    return 2 / np.pi * np.arctan(ratio)
