"""Score a configuration's predicted spread against the errors it actually made.

Runs on whatever the gates settled unless told otherwise, so the thing
diagnosed is the thing the sweep chose. --slug takes any tabular configuration
that has run and --cell any graph-network cell, which is what lets this be
applied off-gate.

Two spreads are scored wherever both exist. The model's own is the regressor's
reported predictive spread, which the tabular foundation models supply and the
graph networks do not; the ensemble's is the standard deviation across the five
training seeds, which every configuration has. A graph-network cell is
therefore scored on the ensemble spread alone, and says so in its record rather
than reporting a missing number as zero.

Where a calibration exists for the configuration, both spreads are scored again
after it. The affine map rescales the predictive distribution rather than only
its mean, so the spread is multiplied by the absolute slope; see
src/uncertainty.py.

Writes the per-compound table and the coverage curve alongside the summary, so
the miscalibration-area figure is drawn from a file rather than recomputed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aggregate  # noqa: E402
import evaluate  # noqa: E402
import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402
import provenance  # noqa: E402
import sweep  # noqa: E402
import uncertainty  # noqa: E402
from data import CANONICAL_COL  # noqa: E402

DESCRIPTION = "Score a configuration's predicted spread against its actual errors."

UNCERTAINTY_DIR = Path("results/uncertainty")
CALIBRATION_DIR = Path("results/calibration")
VERSION = 1

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--slug", help="tabular configuration; defaults to the gated winner")
    parser.add_argument("--cell", help="graph-network cell, scored on its ensemble spread alone")
    parser.add_argument(
        "--results", type=Path, default=sweep.RESULTS_DIR, help="root holding the run directories"
    )
    parser.add_argument(
        "--no-calibrated",
        action="store_true",
        help="skip the post-calibration diagnostics even where a calibration exists",
    )
    parser.add_argument("--force", action="store_true", help="recompute a diagnostic on disk")
    return parser.parse_args(argv)


def _stack_with_spread(
    run_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return stacked predictions, observed values, smiles, and the mean model spread.

    The model spread is averaged over seeds rather than combined in quadrature:
    it is each seed's own statement about its own prediction, and the ensemble
    spread is the separate quantity that measures disagreement between them.
    """
    stacked, observed = aggregate.stack_predictions(run_dirs)
    frames = [
        pd.read_csv(d / "predictions.csv", dtype={CANONICAL_COL: str}).sort_values(CANONICAL_COL)
        for d in run_dirs
    ]
    smiles = frames[0][CANONICAL_COL].to_numpy()

    if any("predicted_std" not in f.columns for f in frames):
        return stacked, observed, smiles, None
    spread = np.vstack([f["predicted_std"].to_numpy(dtype=np.float64) for f in frames])
    if not np.any(np.isfinite(spread) & (spread > 0)):
        return stacked, observed, smiles, None
    return stacked, observed, smiles, spread.mean(axis=0)


def _load_calibration(slug: str) -> dict[str, Any] | None:
    """Return the fitted affine map for a configuration, if one has been written."""
    path = CALIBRATION_DIR / slug / "calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["calibration"]


def main() -> None:
    """Diagnose one configuration's uncertainty and write what it found."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    started = time.perf_counter()

    if args.slug and args.cell:
        raise SystemExit("--slug and --cell name different kinds of run; give one")

    spec = manifest_module.load()
    if args.cell:
        cells = {c.id: c for c in spec.gnn_cells}
        if args.cell not in cells:
            raise SystemExit(f"no graph-network cell named {args.cell!r}")
        label, kind = args.cell, "gnn"
        run_dirs = [cells[args.cell].run_dir(s, args.results) for s in spec.seeds]
    else:
        config = gates.resolve_target(spec, args.slug)
        label, kind = config.slug, "tabular"
        run_dirs = [config.run_dir(s, args.results) for s in spec.seeds]

    missing = [d for d in run_dirs if not (d / "predictions.csv").exists()]
    if missing:
        raise SystemExit(f"{label}: {len(missing)} of {len(run_dirs)} seeds have no predictions")

    out_dir = UNCERTAINTY_DIR / label
    record_path = out_dir / "uncertainty.json"
    if record_path.exists() and not args.force:
        logger.info("%s: already diagnosed", label)
        return

    stacked, observed, smiles, model_sigma = _stack_with_spread(run_dirs)
    predicted = evaluate.ensemble_mean(stacked)
    ensemble_sigma = uncertainty.ensemble_spread(stacked)

    spreads: dict[str, np.ndarray] = {"ensemble": ensemble_sigma}
    if model_sigma is not None:
        spreads["model"] = model_sigma
    else:
        logger.info("%s: no per-run predictive spread, scoring the ensemble spread alone", label)

    diagnostics = {
        name: uncertainty.diagnose(observed, predicted, s, source=name).as_dict()
        for name, s in spreads.items()
    }

    curves = []
    for name, s in spreads.items():
        expected, fraction = uncertainty.coverage_curve(observed, predicted, s)
        curves.append(
            pd.DataFrame(
                {"source": name, "stage": "raw", "expected": expected, "observed": fraction}
            )
        )

    calibrated_block: dict[str, Any] | None = None
    fitted = None if args.no_calibrated else _load_calibration(label)
    if fitted is not None and "slope" not in fitted:
        # a monotone map is affine only locally, and where it is flat the
        # spread it implies is zero. Coverage under a spread of zero is a
        # statement about the map rather than about the model, so it is not
        # drawn here: the affine calibrations are the ones with a spread to
        # rescale
        logger.warning(
            "%s: calibration is %s, which implies no single factor on the spread; "
            "reporting the raw diagnostics only",
            label,
            fitted.get("method", "not affine"),
        )
        fitted = None
    if fitted is not None:
        slope, intercept = float(fitted["slope"]), float(fitted["intercept"])
        cal_pred = slope * predicted + intercept
        cal_spreads = {n: uncertainty.rescale_sigma(s, slope) for n, s in spreads.items()}
        calibrated_block = {
            "slope": slope,
            "intercept": intercept,
            "diagnostics": {
                n: uncertainty.diagnose(observed, cal_pred, s, source=n).as_dict()
                for n, s in cal_spreads.items()
            },
        }
        for name, s in cal_spreads.items():
            expected, fraction = uncertainty.coverage_curve(observed, cal_pred, s)
            curves.append(
                pd.DataFrame(
                    {
                        "source": name,
                        "stage": "calibrated",
                        "expected": expected,
                        "observed": fraction,
                    }
                )
            )

    record: dict[str, Any] = {
        "version": VERSION,
        "label": label,
        "kind": kind,
        "n_seeds": len(run_dirs),
        "n_test": int(observed.size),
        "has_model_spread": model_sigma is not None,
        "diagnostics": diagnostics,
        "calibrated": calibrated_block,
        "wall_clock_s": round(time.perf_counter() - started, 3),
        "environment": provenance.environment(),
        "written_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    points = pd.DataFrame(
        {
            CANONICAL_COL: smiles,
            "observed": observed,
            "predicted": predicted,
            "abs_residual": np.abs(observed - predicted),
            "ensemble_sigma": ensemble_sigma,
        }
    )
    if model_sigma is not None:
        points["model_sigma"] = model_sigma
    with provenance.atomic(out_dir / "points.csv") as tmp:
        points.to_csv(tmp, index=False)
    with provenance.atomic(out_dir / "coverage.csv") as tmp:
        pd.concat(curves, ignore_index=True).to_csv(tmp, index=False)
    with provenance.atomic(record_path) as tmp:
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True))

    for name, d in diagnostics.items():
        logger.info(
            "%s [%s]: spearman %.3f, miscalibration area %.4f, gap %+.4f",
            label,
            name,
            d["spearman"],
            d["miscalibration_area"],
            d["signed_gap"],
        )
    logger.info("wrote %s", record_path)


if __name__ == "__main__":
    main()
