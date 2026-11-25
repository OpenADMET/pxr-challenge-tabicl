"""Summarize the curve-fit standard errors and test a spread that includes them.

The challenge reports the standard error s of each test pEC50's dose-response
fit. If the model's predicted standard deviation sigma describes its own error
and s describes the measurement's noise, a measurement should land around the
prediction with spread sqrt(sigma^2 + s^2), the convolution of the two
normals. This script scores that combined spread against the measurements and
compares it with sigma, with sigma rescaled by a constant, and with sigma
combined with a constant noise.

The following are written to label_noise.json beside the per-compound table:

- how the curve-fit standard errors are distributed and how they relate to
  sigma, pEC50 and the absolute residual
- the average share of sigma^2 each measurement-noise model assigns to noise
- the probability that a correct sigma ranks one pair of compounds correctly
- for each predicted spread, the mean negative log-likelihood, the root mean
  square of the standardized residuals, the miscalibration area, Spearman rho
  against the absolute residual, and the band of rho a correct spread produces
  with one residual per compound
- paired bootstrap intervals for the likelihood and rho gains of the combined
  spread
- the mean absolute residual and the mean absolute residual each spread
  expects, within quintiles of the curve-fit standard error
- the same spread scores without the compounds measured below the assay
  floor, whose pEC50 values are censored, the share of the combined spread's
  likelihood gain those compounds account for, and how well sigma separates
  them from the rest
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
import uncertainty  # noqa: E402

DESCRIPTION = "Summarize the curve-fit standard errors and test a spread that includes them."

UNCERTAINTY_DIR = Path("results/uncertainty")
NOISE_SHARE_MODELS = ("curve_fit", "literature")
PAIR_SIGMAS = (0.6, 0.4)
N_NULL = 2000
N_BOOTSTRAP = 2000
N_SE_GROUPS = 5

# pEC50 below which a measured value is censored at the assay floor
CENSOR_FLOOR = 4.0
SE_BIN_WIDTH = 0.025
SEED = 0

# mean absolute value of a zero-centered normal per unit standard deviation
HALF_NORMAL_MEAN = float(np.sqrt(2 / np.pi))

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


def _quantiles(values: NDArray[np.float64]) -> dict[str, float]:
    """Summarize draws as a median and central 95% interval."""
    lo, median, hi = np.percentile(values, [2.5, 50, 97.5])
    return {"lo": float(lo), "median": float(median), "hi": float(hi)}


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


def predicted_spreads(
    sigma: NDArray[np.float64], residual: NDArray[np.float64], curve_fit_se: NDArray[np.float64]
) -> dict[str, NDArray[np.float64]]:
    """Return each predicted spread compared, keyed by name.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound.
    residual : array-like
        Signed residual, measurement minus prediction, of each compound.
    curve_fit_se : array-like
        Curve-fit standard error of each measured pEC50.

    Returns
    -------
    dict
        ``sigma``; ``sigma_rescaled``, sigma times the constant that brings the
        root mean square standardized residual to 1; ``sigma_constant``, sigma
        combined with the literature-style constant noise; and
        ``sigma_curve_fit``, sigma combined with the curve-fit standard error.
    """
    scale = np.sqrt(np.mean((residual / sigma) ** 2))
    return {
        "sigma": sigma,
        "sigma_rescaled": scale * sigma,
        "sigma_constant": np.sqrt(sigma**2 + label_noise.LITERATURE_NOISE**2),
        "sigma_curve_fit": np.sqrt(sigma**2 + curve_fit_se**2),
    }


def score_spread(
    spread: NDArray[np.float64], residual: NDArray[np.float64], rng: np.random.Generator
) -> dict[str, float]:
    """Score one predicted spread against the residuals.

    Parameters
    ----------
    spread : array-like
        Predicted standard deviation of each residual.
    residual : array-like
        Signed residual, measurement minus prediction, of each compound.
    rng : numpy.random.Generator
        Source of the normal draws for the band of rho.

    Returns
    -------
    dict
        Mean negative log-likelihood, root mean square standardized residual,
        miscalibration area, Spearman rho against the absolute residual, and
        the mean, central 95% interval and percentile of the observed rho
        among the rho values a correct spread produces with one residual per
        compound.
    """
    abs_residual = np.abs(residual)
    rho = float(stats.spearmanr(spread, abs_residual).statistic)
    null = label_noise.spearman_rows(
        spread, np.abs(rng.standard_normal((N_NULL, spread.size))) * spread
    )
    band = _interval(null)
    expected, observed = uncertainty.coverage_curve(residual, np.zeros_like(residual), spread)
    return {
        "nll": label_noise.gaussian_nll(residual, spread),
        "rms_z": float(np.sqrt(np.mean((residual / spread) ** 2))),
        "miscalibration_area": uncertainty.miscalibration_area(expected, observed),
        "rho": rho,
        "null_mean": band["mean"],
        "null_lo": band["lo"],
        "null_hi": band["hi"],
        "null_percentile": float(np.mean(null <= rho) * 100),
    }


def bootstrap_gains(
    spreads: dict[str, NDArray[np.float64]], residual: NDArray[np.float64], rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """Return paired bootstrap intervals for the gains of the curve-fit combined spread.

    Parameters
    ----------
    spreads : dict
        Predicted spreads from ``predicted_spreads``.
    residual : array-like
        Signed residual, measurement minus prediction, of each compound.
    rng : numpy.random.Generator
        Source of the bootstrap resamples of compounds.

    Returns
    -------
    dict
        ``nll_vs_rescaled``, the rescaled sigma's negative log-likelihood minus
        the combined spread's, and ``rho_vs_sigma``, the combined spread's rho
        minus sigma's, each as a median and central 95% interval.
    """
    combined, rescaled, sigma = (
        spreads["sigma_curve_fit"],
        spreads["sigma_rescaled"],
        spreads["sigma"],
    )
    nll_gain = np.empty(N_BOOTSTRAP)
    rho_gain = np.empty(N_BOOTSTRAP)
    for k in range(N_BOOTSTRAP):
        idx = rng.integers(0, residual.size, residual.size)
        r = residual[idx]
        nll_gain[k] = label_noise.gaussian_nll(r, rescaled[idx]) - label_noise.gaussian_nll(
            r, combined[idx]
        )
        rho_gain[k] = (
            stats.spearmanr(combined[idx], np.abs(r)).statistic
            - stats.spearmanr(sigma[idx], np.abs(r)).statistic
        )
    return {"nll_vs_rescaled": _quantiles(nll_gain), "rho_vs_sigma": _quantiles(rho_gain)}


def by_se_group(
    sigma: NDArray[np.float64], residual: NDArray[np.float64], curve_fit_se: NDArray[np.float64]
) -> list[dict[str, float]]:
    """Compare observed and expected absolute residuals within quintiles of the standard error.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound.
    residual : array-like
        Signed residual, measurement minus prediction, of each compound.
    curve_fit_se : array-like
        Curve-fit standard error of each measured pEC50.

    Returns
    -------
    list of dict
        One entry per quintile, lowest standard errors first, with its
        standard-error range, compound count, mean standard error, mean sigma,
        mean absolute residual, and the mean absolute residual that sigma and
        the combined spread each expect.
    """
    edges = np.quantile(curve_fit_se, np.linspace(0, 1, N_SE_GROUPS + 1))
    group = np.clip(np.searchsorted(edges, curve_fit_se, side="right") - 1, 0, N_SE_GROUPS - 1)
    combined = np.sqrt(sigma**2 + curve_fit_se**2)
    rows = []
    for g in range(N_SE_GROUPS):
        mask = group == g
        rows.append(
            {
                "se_lo": float(edges[g]),
                "se_hi": float(edges[g + 1]),
                "n": int(mask.sum()),
                "mean_se": float(curve_fit_se[mask].mean()),
                "mean_sigma": float(sigma[mask].mean()),
                "mean_abs_residual": float(np.abs(residual[mask]).mean()),
                "expected_sigma": float(HALF_NORMAL_MEAN * sigma[mask].mean()),
                "expected_combined": float(HALF_NORMAL_MEAN * combined[mask].mean()),
            }
        )
    return rows


def describe_censored(
    sigma: NDArray[np.float64],
    residual: NDArray[np.float64],
    curve_fit_se: NDArray[np.float64],
    predicted: NDArray[np.float64],
    censored: NDArray[np.bool_],
) -> dict[str, float]:
    """Summarize the censored compounds and their share of the combined spread's gain.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound.
    residual : array-like
        Signed residual, measurement minus prediction, of each compound.
    curve_fit_se : array-like
        Curve-fit standard error of each measured pEC50.
    predicted : array-like
        Predicted pEC50 of each compound.
    censored : array-like of bool
        Whether each compound was measured below the assay floor.

    Returns
    -------
    dict
        Compound count, mean sigma and median standard error inside and outside
        the censored group, the median prediction for censored compounds, the
        area under the ROC curve of sigma for separating censored compounds,
        and the share of the combined spread's negative log-likelihood gain
        over sigma that comes from censored compounds.
    """
    combined = np.sqrt(sigma**2 + curve_fit_se**2)
    gain = _nll_terms(residual, sigma) - _nll_terms(residual, combined)
    auc = stats.mannwhitneyu(sigma[censored], sigma[~censored]).statistic / (
        censored.sum() * (~censored).sum()
    )
    return {
        "floor": CENSOR_FLOOR,
        "n": int(censored.sum()),
        "mean_sigma": float(sigma[censored].mean()),
        "mean_sigma_rest": float(sigma[~censored].mean()),
        "median_se": float(np.median(curve_fit_se[censored])),
        "median_se_rest": float(np.median(curve_fit_se[~censored])),
        "median_predicted": float(np.median(predicted[censored])),
        "auc_sigma": float(auc),
        "nll_gain_share": float(gain[censored].sum() / gain.sum()),
    }


def _nll_terms(residual: NDArray[np.float64], spread: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return each compound's negative log-likelihood under a zero-centered normal."""
    return 0.5 * np.log(2 * np.pi * spread**2) + 0.5 * (residual / spread) ** 2


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
    curve_fit_se = points["curve_fit_se"].to_numpy(dtype=np.float64)
    observed = points["observed"].to_numpy(dtype=np.float64)

    # Summarize the standard errors and the noise share each noise model implies
    record: dict[str, Any] = {
        "n": int(sigma.size),
        "observed_rho": float(stats.spearmanr(sigma, np.abs(residual)).statistic),
        "literature_noise": label_noise.LITERATURE_NOISE,
        "curve_fit_se": describe_se(curve_fit_se, sigma, observed, np.abs(residual)),
        "noise_share": {
            model: float(
                np.mean(label_noise.measurement_noise(model, sigma, curve_fit_se) ** 2)
                / np.mean(sigma**2)
            )
            for model in NOISE_SHARE_MODELS
        },
        "pair_probability": float(
            label_noise.pair_ranking_probability(np.array(PAIR_SIGMAS[0] / PAIR_SIGMAS[1]))
        ),
    }

    # Score each predicted spread, then the paired gains and the standard-error quintiles
    spreads = predicted_spreads(sigma, residual, curve_fit_se)
    record["spreads"] = {
        name: score_spread(spread, residual, np.random.default_rng([args.seed, index]))
        for index, (name, spread) in enumerate(spreads.items())
    }
    record["spread_scale"] = float(spreads["sigma_rescaled"][0] / sigma[0])
    record["bootstrap"] = bootstrap_gains(spreads, residual, np.random.default_rng([args.seed, 9]))
    record["by_se_group"] = by_se_group(sigma, residual, curve_fit_se)

    # Repeat the spread scores without the censored compounds, and describe those compounds
    predicted = points["predicted"].to_numpy(dtype=np.float64)
    censored = observed < CENSOR_FLOOR
    kept = ~censored
    kept_spreads = predicted_spreads(sigma[kept], residual[kept], curve_fit_se[kept])
    record["spreads_uncensored"] = {
        name: score_spread(spread, residual[kept], np.random.default_rng([args.seed, 20 + index]))
        for index, (name, spread) in enumerate(kept_spreads.items())
    }
    record["censored"] = describe_censored(sigma, residual, curve_fit_se, predicted, censored)
    for name, score in record["spreads"].items():
        logger.info("%s: nll %.3f, rho %.3f", name, score["nll"], score["rho"])

    # Write the record beside the per-compound points
    out_path = out_dir / "label_noise.json"
    with provenance.atomic(out_path) as tmp:
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    logger.info("%s: wrote %s", config.slug, out_path)


if __name__ == "__main__":
    main()
