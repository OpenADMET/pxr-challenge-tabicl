"""Simulate how Spearman rho between sigma and error grows with replicate count.

Treats the model's predicted spread as correct: each simulated measurement of a
compound misses its prediction by a normal draw scaled by that compound's
sigma. Averaging the absolute misses over R replicates per compound and ranking
them against sigma gives the rho a correct sigma would reach if each compound
were measured R times. At R = 1 this is the single-measurement case the test
set is actually in, which is why rho stays well below 1 even when sigma is
right.

Reads the per-compound table run/07_uncertainty.py writes and writes
replicates.csv beside it, one row per replicate count, which the walkthrough's
replicate curve plots.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402
import provenance  # noqa: E402

DESCRIPTION = "Simulate Spearman rho under a correct sigma at several replicate counts."

UNCERTAINTY_DIR = Path("results/uncertainty")
REPLICATE_COUNTS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100)
N_SIMULATIONS = 500
SEED = 0

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--slug", help="tabular configuration; defaults to the gated winner")
    parser.add_argument(
        "--simulations", type=int, default=N_SIMULATIONS, help="simulations per replicate count"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the random draws")
    return parser.parse_args(argv)


def simulate_rho(
    sigma: NDArray[np.float64], n_replicates: int, n_simulations: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return Spearman rho between sigma and replicate-averaged error, per simulation.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviation of each compound, assumed correct.
    n_replicates : int
        Measurements simulated per compound, whose absolute errors are averaged.
    n_simulations : int
        Independent simulations to run.
    rng : numpy.random.Generator
        Source of the normal draws.

    Returns
    -------
    numpy.ndarray
        One rho per simulation, shape ``(n_simulations,)``.
    """
    # Average absolute normal errors scaled by each compound's sigma
    draws = rng.standard_normal((n_simulations, n_replicates, sigma.size))
    mean_error = np.abs(draws * sigma).mean(axis=1)

    # Correlate ranks, which is Spearman rho for untied continuous values
    error_ranks = stats.rankdata(mean_error, axis=1)
    sigma_ranks = stats.rankdata(sigma)
    error_centered = error_ranks - error_ranks.mean(axis=1, keepdims=True)
    sigma_centered = sigma_ranks - sigma_ranks.mean()
    covariance = error_centered @ sigma_centered
    scale = np.linalg.norm(error_centered, axis=1) * np.linalg.norm(sigma_centered)
    return covariance / scale


def main() -> None:
    """Simulate every replicate count and write the summary table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # Read the model's predicted sigma for the target configuration
    config = gates.resolve_target(manifest_module.load(), args.slug)
    out_dir = UNCERTAINTY_DIR / config.slug
    points_path = out_dir / "points.csv"
    if not points_path.exists():
        raise SystemExit(f"{config.slug}: no {points_path}; run run/07_uncertainty.py first")
    points = pd.read_csv(points_path)
    if "model_sigma" not in points:
        raise SystemExit(f"{config.slug}: the regressor reports no spread of its own")
    sigma = points["model_sigma"].to_numpy(dtype=np.float64)

    # Summarize the simulated rho at each replicate count
    rng = np.random.default_rng(args.seed)
    rows = []
    for n_replicates in REPLICATE_COUNTS:
        rho = simulate_rho(sigma, n_replicates, args.simulations, rng)
        p5, p95 = np.percentile(rho, [5, 95])
        rows.append({"replicates": n_replicates, "mean": rho.mean(), "p5": p5, "p95": p95})
        logger.info(
            "R = %d: mean rho %.4f (5th-95th %.4f to %.4f)", n_replicates, rho.mean(), p5, p95
        )

    # Write the table beside the per-compound points
    with provenance.atomic(out_dir / "replicates.csv") as tmp:
        pd.DataFrame(rows).round(4).to_csv(tmp, index=False)
    logger.info("%s: wrote %s", config.slug, out_dir / "replicates.csv")


if __name__ == "__main__":
    main()
