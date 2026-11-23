"""Split a predicted spread into measurement noise and model error, and simulate both.

A calibrated predictive standard deviation sigma describes how far a measured
value lands from the prediction. That distance has two parts with different
behavior under repetition. Measurement noise, with standard deviation s, is
drawn afresh every time a compound is measured. Model error, the gap between
the prediction and the compound's true value, is fixed for a given compound
and model, with standard deviation sqrt(sigma^2 - s^2) across compounds.
Replicate measurements average away the first part and leave the second.

Three measurement-noise models are compared: all of sigma (the case where
replicates converge on a perfect ranking), the curve-fit standard error the
challenge reports for each pEC50, and a constant literature-style value.
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

NOISE_MODELS = ("all", "curve_fit", "literature")


def load_points(points_path: Path, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return a per-compound uncertainty table with each compound's curve-fit standard error.

    Parameters
    ----------
    points_path : path-like
        The ``points.csv`` run/07_uncertainty.py writes.
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
        raise FileNotFoundError(f"no {points_path}; run run/07_uncertainty.py first")
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
        One of ``NOISE_MODELS``: ``"all"`` treats all of sigma as measurement
        noise, ``"curve_fit"`` uses the reported curve-fit standard error, and
        ``"literature"`` uses ``LITERATURE_NOISE`` for every compound.
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
    if model == "all":
        return sigma.copy()
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


def folded_normal_mean(
    offset: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return E|offset + scale * z| for a standard normal z.

    This is the mean absolute error infinitely many replicates would converge
    on, for a fixed model error ``offset`` and measurement noise ``scale``.

    Parameters
    ----------
    offset : array-like
        Fixed shift of each normal.
    scale : array-like
        Standard deviation of each normal; zero gives ``|offset|``.

    Returns
    -------
    numpy.ndarray
        The mean of each folded normal.
    """
    # Avoid dividing by a zero scale, where the folded mean is just |offset|
    safe = np.where(scale > 0, scale, 1.0)
    spread = safe * np.sqrt(2 / np.pi) * np.exp(-(offset**2) / (2 * safe**2))
    shift = offset * (1 - 2 * stats.norm.cdf(-offset / safe))
    return np.where(scale > 0, spread + shift, np.abs(offset))
