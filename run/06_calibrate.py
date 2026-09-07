"""Calibrate a configuration's predictions with the report's affine map.

The map is fitted on out-of-fold predictions over the whole fit set, weighted
by a train-versus-test density ratio, and applied to the configuration's
seed-ensembled test predictions. See src/calibration.py for the method and why
it is not the monotone map on one held-out slice that this pipeline used to
declare.

By default it runs on whatever configuration the gates settled, so the thing
calibrated is the thing the sweep chose rather than one named on the command
line. --slug calibrates any configuration that has run instead, which is what
makes the script a tool rather than a single-purpose step.

Out-of-fold predictions cost k refits of the model. Those refits are the whole
expense here, and --oof-seeds decides how many seeds each fold is fitted at:
one by default, which is five fits, against five which is twenty-five and
matches the ensemble the map is applied to. That mismatch is real and recorded
in the run's record rather than hidden.
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
import calibration  # noqa: E402
import evaluate  # noqa: E402
import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402
import provenance  # noqa: E402
import regressors  # noqa: E402
import sweep  # noqa: E402
from data import CANONICAL_COL, TARGET_COL  # noqa: E402

DESCRIPTION = "Calibrate a configuration with the report's density-ratio affine map."

CALIBRATION_DIR = Path("results/calibration")
VERSION = 1

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--slug",
        help=(
            "configuration to calibrate; defaults to whatever the gates settled, "
            "which is the point of the default"
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=calibration.N_FOLDS,
        help="folds the out-of-fold predictions are generated over",
    )
    parser.add_argument(
        "--oof-seeds",
        type=int,
        default=1,
        help=(
            "seeds each fold is fitted at. One is five fits; five matches the "
            "ensemble the map is applied to, at twenty-five"
        ),
    )
    parser.add_argument(
        "--clip",
        nargs=2,
        type=float,
        default=list(calibration.WEIGHT_CLIP),
        metavar=("LOW", "HIGH"),
        help="bounds on the density-ratio weights",
    )
    parser.add_argument(
        "--radius", type=int, default=calibration.MORGAN_RADIUS, help="Morgan fingerprint radius"
    )
    parser.add_argument(
        "--bits", type=int, default=calibration.MORGAN_BITS, help="Morgan fingerprint length"
    )
    parser.add_argument(
        "--method",
        default="affine",
        choices=list(calibration.METHODS),
        help="which map to fit: the report's affine map, a scale factor through "
        "the origin, or a non-decreasing isotonic map",
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="fit the map without the density ratio, the ablation that says what it bought",
    )
    parser.add_argument(
        "--results", type=Path, default=sweep.RESULTS_DIR, help="root holding the run directories"
    )
    parser.add_argument("--force", action="store_true", help="recompute a calibration on disk")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    return parser.parse_args(argv)


def out_of_fold_predictions(
    config: manifest_module.TabularConfig,
    spec: manifest_module.Manifest,
    partitions: sweep.Partitions,
    *,
    n_folds: int,
    n_seeds: int,
) -> np.ndarray:
    """Predict every fit compound from a model that did not train on it.

    Each fold fits on the other folds and predicts the held-out one, at
    ``n_seeds`` seeds averaged. The blocks are built once against the whole fit
    set, matching how the sweep builds them, so the reduction a fold sees is
    the one the scored model saw.
    """
    fit = partitions.fit
    fit_smiles = fit[CANONICAL_COL].tolist()
    y = fit[TARGET_COL].to_numpy(dtype=np.float64)

    # the reductions are fitted on the whole fit partition, exactly as the
    # sweep fits them. Refitting PCA per fold would measure a different
    # pipeline from the one being calibrated
    artifacts = sweep.reduced_blocks(config, spec, seed=0, fit_smiles=fit_smiles)
    x = sweep.assemble(artifacts, fit_smiles)

    predictions = np.zeros(len(fit_smiles), dtype=np.float64)
    for fold, (train_rows, held_rows) in enumerate(
        calibration.fold_indices(len(fit_smiles), n_folds=n_folds)
    ):
        per_seed = []
        for seed in range(n_seeds):
            logger.info(
                "fold %d/%d seed %d: fitting on %d, predicting %d",
                fold + 1,
                n_folds,
                seed,
                len(train_rows),
                len(held_rows),
            )
            result = regressors.fit_predict(
                config.regressor, x[train_rows], y[train_rows], x[held_rows], seed=seed
            )
            per_seed.append(result.mean)
        predictions[held_rows] = np.mean(per_seed, axis=0)
    return predictions


def ensembled_test_predictions(
    config: manifest_module.TabularConfig, spec: manifest_module.Manifest, results_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the seed-ensembled test predictions for a configuration.

    Raises
    ------
    SystemExit
        If the configuration has not been run at every seed, since calibrating
        a partial ensemble would compare it against runs that are not partial.
    """
    run_dirs = [config.run_dir(seed, results_dir) for seed in spec.seeds]
    missing = [d for d in run_dirs if not (d / "predictions.csv").exists()]
    if missing:
        raise SystemExit(
            f"{config.slug}: {len(missing)} of {len(run_dirs)} seeds have no predictions; "
            "run the sweep for it first"
        )

    stacked, observed = aggregate.stack_predictions(run_dirs)
    frame = pd.read_csv(run_dirs[0] / "predictions.csv", dtype={CANONICAL_COL: str}).sort_values(
        CANONICAL_COL
    )
    return (
        evaluate.ensemble_mean(stacked),
        observed,
        frame[CANONICAL_COL].to_numpy(),
        len(run_dirs),
    )


def main() -> None:
    """Fit the affine map for one configuration and write what it did."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    started = time.perf_counter()

    spec = manifest_module.load()
    config = gates.resolve_target(spec, args.slug)
    clip = (float(args.clip[0]), float(args.clip[1]))

    # the report's map keeps the configuration's own directory, since it is
    # what the uncertainty stage reads and what the figures were drawn from.
    # Another method writes beside it rather than over it, so the three can be
    # compared without one of them silently replacing the others
    out_dir = CALIBRATION_DIR / config.slug
    if args.method != "affine":
        out_dir = CALIBRATION_DIR / f"{config.slug}__{args.method}"
    record_path = out_dir / "calibration.json"
    if record_path.exists() and not args.force:
        logger.info("%s: already calibrated", config.slug)
        return

    partitions = sweep.load_partitions()
    logger.info(
        "%s: %d folds x %d seed(s) = %d fits over %d compounds",
        config.slug,
        args.folds,
        args.oof_seeds,
        args.folds * args.oof_seeds,
        len(partitions.fit),
    )
    if args.dry_run:
        return

    raw, observed_test, test_smiles, n_seeds = ensembled_test_predictions(
        config, spec, args.results
    )
    out_of_fold = out_of_fold_predictions(
        config, spec, partitions, n_folds=args.folds, n_seeds=args.oof_seeds
    )

    fitted = calibration.calibrate(
        out_of_fold,
        partitions.fit[TARGET_COL].to_numpy(dtype=np.float64),
        partitions.fit[CANONICAL_COL].tolist(),
        partitions.test[CANONICAL_COL].tolist(),
        radius=args.radius,
        n_bits=args.bits,
        clip=clip,
        method=args.method,
        weighted=not args.unweighted,
    )
    calibrated = fitted.apply(raw)

    before = evaluate.metrics(observed_test, raw)
    after = evaluate.metrics(observed_test, calibrated)
    record: dict[str, Any] = {
        "version": VERSION,
        "slug": config.slug,
        "config": config.as_dict(),
        "from_gates": args.slug is None,
        "calibration": fitted.as_dict(),
        "settings": {
            "method": args.method,
            "n_folds": args.folds,
            "oof_seeds": args.oof_seeds,
            "morgan_radius": args.radius,
            "morgan_bits": args.bits,
            "weight_clip": list(clip),
            "weighted": not args.unweighted,
        },
        # the map is fitted on predictions from oof_seeds models and applied to
        # an ensemble of n_seeds, so the two differ in variance unless they match
        "ensemble_mismatch": {
            "oof_seeds": args.oof_seeds,
            "applied_to_seeds": n_seeds,
            "matched": args.oof_seeds == n_seeds,
        },
        "metrics_before": before,
        "metrics_after": after,
        "n_test": int(observed_test.size),
        "wall_clock_s": round(time.perf_counter() - started, 3),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    # the out-of-fold predictions are the whole cost of this step, so they are
    # kept: refitting the map unweighted, or under another clip, is then free
    with provenance.atomic(out_dir / "out_of_fold.csv") as tmp:
        pd.DataFrame(
            {
                CANONICAL_COL: partitions.fit[CANONICAL_COL].to_numpy(),
                "observed": partitions.fit[TARGET_COL].to_numpy(dtype=np.float64),
                "out_of_fold": out_of_fold,
            }
        ).to_csv(tmp, index=False)

    predictions = pd.DataFrame(
        {
            CANONICAL_COL: test_smiles,
            "observed": observed_test,
            "raw": raw,
            "calibrated": calibrated,
        }
    )
    with provenance.atomic(out_dir / "predictions.csv") as tmp:
        predictions.to_csv(tmp, index=False)
    with provenance.atomic(record_path) as tmp:
        stamped = {
            **record,
            "environment": provenance.environment(),
            "written_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        tmp.write_text(json.dumps(stamped, indent=2, sort_keys=True))

    logger.info(
        "%s: MAE %.4f -> %.4f, slope %.4f, intercept %.4f",
        config.slug,
        before["mae"],
        after["mae"],
        fitted.slope,
        fitted.intercept,
    )
    logger.info("wrote %s", record_path)


if __name__ == "__main__":
    main()
