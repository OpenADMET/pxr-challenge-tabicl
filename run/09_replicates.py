"""Simulate how Spearman rho between sigma and error grows with replicate count.

Treats the model's predicted spread as correct and splits it into measurement
noise, which each replicate draws afresh, and model error, which is fixed per
compound (see src/label_noise.py). Averaging the absolute misses over R
replicates per compound and ranking them against sigma gives the rho a correct
sigma would reach if each compound were measured R times. At R = 1 this is the
single-measurement case the test set is actually in.

How far replicates can raise rho depends on how much of sigma is measurement
noise, so the simulation runs under each noise model: all of sigma, the
curve-fit standard error the challenge reports, and a constant literature-style
value. Only the first lets rho approach 1.

Reads the per-compound table run/07_uncertainty.py writes, joins the curve-fit
standard errors from data/raw, and writes replicates.csv beside the table, one
row per noise model and replicate count, which the walkthrough's replicate
curve plots.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gates  # noqa: E402
import label_noise  # noqa: E402
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


def main() -> None:
    """Simulate every noise model and replicate count and write the summary table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # Read the predicted sigma and curve-fit standard error for the target configuration
    config = gates.resolve_target(manifest_module.load(), args.slug)
    points = label_noise.load_points(UNCERTAINTY_DIR / config.slug / "points.csv")
    sigma = points["model_sigma"].to_numpy(dtype=np.float64)
    curve_fit_se = points["curve_fit_se"].to_numpy(dtype=np.float64)

    # Summarize the simulated rho per noise model and replicate count
    rows = []
    for index, model in enumerate(label_noise.NOISE_MODELS):
        noise = label_noise.measurement_noise(model, sigma, curve_fit_se)
        rng = np.random.default_rng([args.seed, index])
        for n_replicates in REPLICATE_COUNTS:
            rho = label_noise.simulate_replicates(sigma, noise, n_replicates, args.simulations, rng)
            p5, p95 = np.percentile(rho, [5, 95])
            rows.append(
                {
                    "noise": model,
                    "replicates": n_replicates,
                    "mean": rho.mean(),
                    "p5": p5,
                    "p95": p95,
                }
            )
            logger.info(
                "%s, R = %d: mean rho %.4f (5th-95th %.4f to %.4f)",
                model,
                n_replicates,
                rho.mean(),
                p5,
                p95,
            )

    # Write the table beside the per-compound points
    out_path = UNCERTAINTY_DIR / config.slug / "replicates.csv"
    with provenance.atomic(out_path) as tmp:
        pd.DataFrame(rows).round(4).to_csv(tmp, index=False)
    logger.info("%s: wrote %s", config.slug, out_path)


if __name__ == "__main__":
    main()
