"""Test whether resampling each measurement around itself can sharpen the ranking.

The proposal under test treats each measured pEC50 as the center of a normal
distribution whose width is the curve-fit standard error s, draws a new value
from it for every compound, recomputes Spearman rho between sigma and the
absolute residual, and repeats. The draws are y* = y + s z, so they center on
the one measurement rather than on the true value, and every draw inherits that
measurement's error.

Four things are written to label_noise.json beside the per-compound table:

- how the curve-fit standard errors are distributed and how they relate to
  sigma, pEC50 and the absolute residual
- the resampled rho on the real test set, under the curve-fit standard errors
  and under a constant literature-style noise
- a simulation in which the true values are known, comparing rho against the
  true values, against one measurement, and after resampling, under each
  measurement-noise model (see src/label_noise.py)
- rho within the better-fitted and worse-fitted halves of the test set, each
  against the band a correct sigma produces for that half
- the limit a correct sigma faces even with exact true values: rho when all
  of sigma is model error, and how often one pair of compounds is ranked
  correctly at a given ratio of their sigma
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gates  # noqa: E402
import label_noise  # noqa: E402
import manifest as manifest_module  # noqa: E402
import provenance  # noqa: E402

DESCRIPTION = "Test whether resampling each measurement around itself sharpens the ranking."

UNCERTAINTY_DIR = Path("results/uncertainty")
RESAMPLING_NOISE_MODELS = ("curve_fit", "literature")
N_RESAMPLES = 2000
N_DATASETS = 1000
N_INNER_RESAMPLES = 20
N_NULL = 2000
SE_BIN_WIDTH = 0.025
RHO_BIN_WIDTH = 0.01
PAIR_RATIOS = (1.5, 2.0, 4.0, 10.0)
SEED = 0

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--slug", help="tabular configuration; defaults to the gated winner")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the random draws")
    return parser.parse_args(argv)


def _interval(values: NDArray[np.float64]) -> dict[str, float]:
    """Summarize draws as a mean and central 95% interval."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def _histogram(values: NDArray[np.float64], width: float) -> dict[str, Any]:
    """Bin values on a fixed grid of the given width, anchored at zero."""
    start = np.floor(values.min() / width) * width
    stop = np.ceil(values.max() / width) * width + width / 2
    counts, edges = np.histogram(values, bins=np.arange(start, stop, width))
    return {"edges": np.round(edges, 4).tolist(), "counts": counts.tolist()}


def _partial_rank_correlation(
    x: NDArray[np.float64], y: NDArray[np.float64], control: NDArray[np.float64]
) -> float:
    """Correlate the ranks of x and y after regressing the ranks of a control out of each."""
    ranks = [stats.rankdata(v) for v in (x, y, control)]
    residuals = [r - np.polyval(np.polyfit(ranks[2], r, 1), ranks[2]) for r in ranks[:2]]
    return float(np.corrcoef(*residuals)[0, 1])


def describe_se(
    curve_fit_se: NDArray[np.float64],
    sigma: NDArray[np.float64],
    observed: NDArray[np.float64],
    abs_residual: NDArray[np.float64],
) -> dict[str, Any]:
    """Summarize the curve-fit standard errors and their rank correlations.

    Parameters
    ----------
    curve_fit_se : array-like
        Curve-fit standard error of each measured pEC50.
    sigma : array-like
        Predicted standard deviation of each compound.
    observed : array-like
        Measured pEC50 of each compound.
    abs_residual : array-like
        Absolute residual of each compound.

    Returns
    -------
    dict
        Quantiles, the root mean square, a histogram, Spearman rho of the
        standard error against sigma, pEC50 and the absolute residual, Spearman
        rho of sigma against pEC50, and the rank correlation of sigma and the
        standard error with pEC50 partialled out.
    """
    p5, p50, p95 = np.percentile(curve_fit_se, [5, 50, 95])
    return {
        "median": float(p50),
        "p5": float(p5),
        "p95": float(p95),
        "rms": float(np.sqrt(np.mean(curve_fit_se**2))),
        "mean_sigma": float(sigma.mean()),
        "n_above_sigma": int((curve_fit_se > sigma).sum()),
        "histogram": _histogram(curve_fit_se, SE_BIN_WIDTH),
        "rho_sigma": float(stats.spearmanr(sigma, curve_fit_se).statistic),
        "rho_pec50": float(stats.spearmanr(observed, curve_fit_se).statistic),
        "rho_abs_residual": float(stats.spearmanr(abs_residual, curve_fit_se).statistic),
        "rho_sigma_pec50": float(stats.spearmanr(sigma, observed).statistic),
        "partial_rho_sigma_given_pec50": _partial_rank_correlation(sigma, curve_fit_se, observed),
    }


def resample_real(
    sigma: NDArray[np.float64],
    residual: NDArray[np.float64],
    noise: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return rho after resampling each measurement around itself, per resample.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound.
    residual : array-like
        Signed residual, measurement minus prediction, of each compound.
    noise : array-like
        Width of the normal each measurement is resampled from.
    rng : numpy.random.Generator
        Source of the normal draws.

    Returns
    -------
    numpy.ndarray
        One rho per resample, shape ``(N_RESAMPLES,)``.
    """
    resampled = residual + rng.standard_normal((N_RESAMPLES, sigma.size)) * noise
    return label_noise.spearman_rows(sigma, np.abs(resampled))


def simulate_known_truth(
    sigma: NDArray[np.float64], noise: NDArray[np.float64], rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """Compare rho against true values, one measurement, and resampling, with truth known.

    Each simulated dataset draws a model error per compound with standard
    deviation sqrt(sigma^2 - noise^2), then one measurement around the true
    value with standard deviation ``noise``. The resampled rho of a dataset is
    the mean over ``N_INNER_RESAMPLES`` resamples of its measurements.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound, assumed correct.
    noise : array-like
        Measurement-noise standard deviation of each compound, at most sigma.
    rng : numpy.random.Generator
        Source of the normal draws.

    Returns
    -------
    dict
        For each of ``oracle`` (against the true values), ``single`` (against
        one measurement), ``resampled`` and ``infinite_replicates`` (against the
        mean absolute error infinitely many replicates converge on), the mean
        and central 95% interval across datasets.
    """
    # Draw the fixed model errors and one measurement per compound per dataset
    model_sd = np.sqrt(sigma**2 - noise**2)
    model_error = rng.standard_normal((N_DATASETS, sigma.size)) * model_sd
    measured = model_error + rng.standard_normal((N_DATASETS, sigma.size)) * noise

    # Resample every dataset's measurements around themselves and average the rho
    resampled = np.empty(N_DATASETS)
    for k in range(N_DATASETS):
        draws = measured[k] + rng.standard_normal((N_INNER_RESAMPLES, sigma.size)) * noise
        resampled[k] = label_noise.spearman_rows(sigma, np.abs(draws)).mean()

    limit = label_noise.folded_normal_mean(model_error, np.broadcast_to(noise, model_error.shape))
    return {
        "oracle": _interval(label_noise.spearman_rows(sigma, np.abs(model_error))),
        "single": _interval(label_noise.spearman_rows(sigma, np.abs(measured))),
        "resampled": _interval(resampled),
        "infinite_replicates": _interval(label_noise.spearman_rows(sigma, limit)),
    }


def split_by_fit_quality(
    sigma: NDArray[np.float64],
    abs_residual: NDArray[np.float64],
    curve_fit_se: NDArray[np.float64],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Return rho and its correct-sigma band in the better- and worse-fitted halves.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound.
    abs_residual : array-like
        Absolute residual of each compound.
    curve_fit_se : array-like
        Curve-fit standard error of each compound, split at its median.
    rng : numpy.random.Generator
        Source of the normal draws for the band.

    Returns
    -------
    dict
        For ``low_se`` and ``high_se``, the compound count, observed rho, and
        the mean and central 95% interval of rho under a correct sigma.
    """
    cut = np.median(curve_fit_se)
    halves = {"low_se": curve_fit_se <= cut, "high_se": curve_fit_se > cut}
    summary = {}
    for name, mask in halves.items():
        null = label_noise.spearman_rows(
            sigma[mask], np.abs(rng.standard_normal((N_NULL, int(mask.sum())))) * sigma[mask]
        )
        band = _interval(null)
        summary[name] = {
            "n": int(mask.sum()),
            "rho": float(stats.spearmanr(sigma[mask], abs_residual[mask]).statistic),
            "null_mean": band["mean"],
            "null_lo": band["lo"],
            "null_hi": band["hi"],
        }
    return summary


def exact_truth_limit(sigma: NDArray[np.float64], rng: np.random.Generator) -> dict[str, Any]:
    """Summarize how well a correct sigma ranks errors when the true values are known.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound, assumed correct and
        entirely model error.
    rng : numpy.random.Generator
        Source of the normal draws.

    Returns
    -------
    dict
        The mean and central 95% interval of rho against the true values, the
        probability of ranking a pair correctly at each of ``PAIR_RATIOS``,
        and, over every pair of test compounds, the median sigma ratio, the
        share of pairs with a ratio below 1.5, and the mean probability of
        ranking a pair correctly.
    """
    oracle = label_noise.spearman_rows(
        sigma, np.abs(rng.standard_normal((N_NULL, sigma.size))) * sigma
    )
    upper = np.triu_indices(sigma.size, k=1)
    ratios = np.maximum(sigma[upper[0]], sigma[upper[1]]) / np.minimum(
        sigma[upper[0]], sigma[upper[1]]
    )
    probabilities = label_noise.pair_ranking_probability(np.array(PAIR_RATIOS))
    return {
        "oracle": _interval(oracle),
        "pair_probability": {
            f"{r:g}": float(p) for r, p in zip(PAIR_RATIOS, probabilities, strict=True)
        },
        "median_pair_ratio": float(np.median(ratios)),
        "share_pairs_below_1_5": float(np.mean(ratios < 1.5)),
        "mean_pair_probability": float(label_noise.pair_ranking_probability(ratios).mean()),
    }


def main() -> None:
    """Run every comparison and write label_noise.json."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # Read sigma, residuals and curve-fit standard errors for the target configuration
    config = gates.resolve_target(manifest_module.load(), args.slug)
    out_dir = UNCERTAINTY_DIR / config.slug
    points = label_noise.load_points(out_dir / "points.csv")
    sigma = points["model_sigma"].to_numpy(dtype=np.float64)
    residual = (points["observed"] - points["predicted"]).to_numpy(dtype=np.float64)
    abs_residual = np.abs(residual)
    curve_fit_se = points["curve_fit_se"].to_numpy(dtype=np.float64)
    observed = points["observed"].to_numpy(dtype=np.float64)

    record: dict[str, Any] = {
        "n": int(sigma.size),
        "observed_rho": float(stats.spearmanr(sigma, abs_residual).statistic),
        "literature_noise": label_noise.LITERATURE_NOISE,
        "curve_fit_se": describe_se(curve_fit_se, sigma, observed, abs_residual),
        "resampled_real": {},
        "known_truth": {},
        "noise_share": {},
    }

    # Resample the real measurements, and simulate with the truth known, per noise model
    for index, model in enumerate(RESAMPLING_NOISE_MODELS):
        noise = label_noise.measurement_noise(model, sigma, curve_fit_se)
        record["noise_share"][model] = float(np.mean(noise**2) / np.mean(sigma**2))
        rho = resample_real(sigma, residual, noise, np.random.default_rng([args.seed, 0, index]))
        record["resampled_real"][model] = _interval(rho) | {
            "histogram": _histogram(rho, RHO_BIN_WIDTH)
        }
        record["known_truth"][model] = simulate_known_truth(
            sigma, noise, np.random.default_rng([args.seed, 1, index])
        )
        logger.info(
            "%s: resampled real rho %.3f; known truth %s",
            model,
            rho.mean(),
            {k: round(v["mean"], 3) for k, v in record["known_truth"][model].items()},
        )

    record["exact_truth"] = exact_truth_limit(sigma, np.random.default_rng([args.seed, 3]))
    record["fit_quality_halves"] = split_by_fit_quality(
        sigma, abs_residual, curve_fit_se, np.random.default_rng([args.seed, 2])
    )

    # Write the record beside the per-compound points
    out_path = out_dir / "label_noise.json"
    with provenance.atomic(out_path) as tmp:
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    logger.info("%s: wrote %s", config.slug, out_path)


if __name__ == "__main__":
    main()
