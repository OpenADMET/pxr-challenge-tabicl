"""Stage three: assemble a featureset, fit a regressor, and write one run.

A run is one configuration at one seed. It reads reduced blocks from stage two,
joins them into a feature matrix, fits on the 4,392 fit rows, predicts the 260
phase-2 rows, and writes its predictions, its metrics and a record of exactly
what produced it.

Skipping is decided by that record rather than by the directory existing. A run
whose recorded specification matches the one being asked for is complete and is
left alone; a run whose inputs have changed has a different specification and is
redone. So a sweep can be interrupted and restarted at any point, and changing
an upstream block does not silently leave stale results behind.

The calibration arm costs a second fit. An isotonic map has to be fitted on
predictions the model has not already seen, so the arm trains a second model on
the training partition alone, predicts the held-out validation partition, fits
the map there, and applies it to the phase-2 predictions of the model fitted on
everything. Fitting the map on predictions the model was trained on would flatter
it, which is the whole failure mode calibration is supposed to detect.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import evaluate
import features
import provenance
import reduce as reduction
import regressors
from data import CANONICAL_COL, SPLIT_DIR, TARGET_COL
from manifest import GnnCell, Manifest, TabularConfig

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

# bumped when the meaning of a run changes, never for a cosmetic edit
VERSION = 1

# the order feature blocks are joined in, so a matrix's columns are a function
# of the configuration and not of dictionary iteration order
BLOCK_ORDER = ("embedding", "readout", "descriptors")

# what the vendored architecture must supply: axes, seed and partitions in,
# test-partition predictions out, in that partition's own row order
TrainGnn = Callable[[dict[str, Any], int, "Partitions"], np.ndarray]


class SweepError(RuntimeError):
    """A run could not be assembled or scored."""


@dataclass(frozen=True)
class Partitions:
    """The compound sets a run fits on, calibrates on, and is scored against."""

    fit: pd.DataFrame
    fit_train: pd.DataFrame
    fit_val: pd.DataFrame
    test: pd.DataFrame


def load_partitions(split_dir: Path = SPLIT_DIR) -> Partitions:
    """Read the four split resource files."""

    def read(name: str) -> pd.DataFrame:
        frame = pd.read_csv(split_dir / name, dtype={CANONICAL_COL: str})
        return frame.loc[frame[TARGET_COL].notna()].reset_index(drop=True)

    return Partitions(
        fit=read("fit_all.csv"),
        fit_train=read("fit_train.csv"),
        fit_val=read("fit_val.csv"),
        test=read("test_phase2.csv"),
    )


def reduced_blocks(
    config: TabularConfig,
    manifest: Manifest,
    seed: int,
    *,
    fit_smiles: Sequence[str],
    feature_cache: Path = features.CACHE_DIR,
    reduced_cache: Path = reduction.CACHE_DIR,
    force: bool = False,
) -> list[provenance.Artifact]:
    """Return the reduced block artifacts a configuration draws on, in join order.

    Parameters
    ----------
    config : TabularConfig
        The configuration whose blocks are wanted.
    manifest : Manifest
        Supplies the axis vocabulary mapping a level to its raw blocks and width.
    seed : int
        Threaded into the seed-dependent blocks and into PCA's solver.
    fit_smiles : sequence of str
        The partition every reduction is fitted on.
    feature_cache, reduced_cache : path-like, optional
        Cache roots for stages one and two.
    force : bool, optional
        Recompute rather than reading from cache.

    Returns
    -------
    list of Artifact
        One reduced artifact per present block group, in ``BLOCK_ORDER``.
    """
    artifacts: list[provenance.Artifact] = []
    for group in BLOCK_ORDER:
        level = getattr(config, group)
        if level == "none":
            continue
        spec = manifest.axes[group][level]

        names = spec.get("blocks", [spec["block"]] if "block" in spec else [])
        width = config.descriptor_pca if group == "descriptors" else spec.get("pca")

        raw = [
            features.build(
                name,
                params=_seed_params(name, seed),
                cache_dir=feature_cache,
                force=force,
            )
            for name in names
        ]
        artifacts.append(
            reduction.build(
                raw,
                width=width,
                fit_smiles=fit_smiles,
                seed=seed,
                cache_dir=reduced_cache,
                force=force,
            )
        )
    return artifacts


def assemble(artifacts: Sequence[provenance.Artifact], smiles: Sequence[str]) -> np.ndarray:
    """Join reduced blocks and select rows, in the order the compounds are given.

    Parameters
    ----------
    artifacts : sequence of Artifact
        Reduced blocks, already in join order.
    smiles : sequence of str
        Canonical SMILES of the compounds wanted, in the order wanted.

    Returns
    -------
    ndarray
        The feature matrix, one row per compound given.

    Raises
    ------
    SweepError
        If a block does not cover every compound asked for.
    """
    frames = []
    for artifact in artifacts:
        frame = reduction.load(artifact)
        missing = set(smiles) - set(frame.index)
        if missing:
            raise SweepError(
                f"{artifact.path.name}: reduced block is missing {len(missing)} compounds"
            )
        frames.append(frame.loc[list(smiles)])
    return pd.concat(frames, axis=1).to_numpy(dtype=np.float64)


def run_spec(config: TabularConfig, seed: int, artifacts: Sequence[provenance.Artifact]) -> dict:
    """Return the specification that identifies a run."""
    return provenance.block_spec(
        "tabular_run",
        VERSION,
        params={**config.as_dict(), "seed": seed},
        inputs=list(artifacts),
        regressor_version=regressors.REGRESSORS[config.regressor].version,
    )


def is_complete(run_dir: Path, spec: dict) -> bool:
    """Whether a run directory already holds a finished run of this specification."""
    record_path = run_dir / "run.json"
    if not (record_path.exists() and (run_dir / "predictions.csv").exists()):
        return False
    try:
        recorded = json.loads(record_path.read_text())
    except json.JSONDecodeError:
        return False
    return recorded.get("key") == provenance.spec_key(spec)


def run_one(
    config: TabularConfig,
    manifest: Manifest,
    seed: int,
    *,
    partitions: Partitions | None = None,
    results_dir: Path = RESULTS_DIR,
    feature_cache: Path = features.CACHE_DIR,
    reduced_cache: Path = reduction.CACHE_DIR,
    force: bool = False,
) -> Path:
    """Fit one configuration at one seed and write its run directory.

    Parameters
    ----------
    config : TabularConfig
        What to fit.
    manifest : Manifest
        Supplies the axis vocabulary and the split paths.
    seed : int
        The replicate seed, threaded into the encoder blocks, PCA's solver and
        the regressor.
    partitions : Partitions, optional
        Preloaded split frames, to avoid rereading them per run.
    results_dir : path-like, optional
        Root the run directory hangs off.
    feature_cache, reduced_cache : path-like, optional
        Cache roots for stages one and two.
    force : bool, optional
        Refit even when a matching run is already present.

    Returns
    -------
    Path
        The run directory.
    """
    partitions = partitions or load_partitions()
    fit_smiles = sorted(set(partitions.fit[CANONICAL_COL]))

    artifacts = reduced_blocks(
        config,
        manifest,
        seed,
        fit_smiles=fit_smiles,
        feature_cache=feature_cache,
        reduced_cache=reduced_cache,
        force=force,
    )
    spec = run_spec(config, seed, artifacts)
    run_dir = config.run_dir(seed, results_dir)

    if is_complete(run_dir, spec) and not force:
        logger.info("%s seed=%d: already complete", config.slug, seed)
        return run_dir

    # the model that is scored sees the whole fit partition
    x_fit = assemble(artifacts, partitions.fit[CANONICAL_COL].tolist())
    y_fit = partitions.fit[TARGET_COL].to_numpy(dtype=np.float64)
    x_test = assemble(artifacts, partitions.test[CANONICAL_COL].tolist())
    y_test = partitions.test[TARGET_COL].to_numpy(dtype=np.float64)

    logger.info("%s seed=%d: fitting on %d x %d", config.slug, seed, *x_fit.shape)
    prediction = regressors.fit_predict(config.regressor, x_fit, y_fit, x_test, seed=seed)
    predicted = prediction.mean

    calibrator = None
    if config.calibration == "isotonic_fitval":
        calibrator = _fit_calibrator(config, artifacts, partitions, seed)
        predicted = calibrator.predict(predicted)

    scores = write_run(
        run_dir,
        spec=spec,
        seed=seed,
        smiles=partitions.test[CANONICAL_COL].to_numpy(),
        observed=y_test,
        predicted=predicted,
        std=prediction.std,
        record={
            "slug": config.slug,
            "config": config.as_dict(),
            "regressor_params": regressors.resolved_params(
                config.regressor, seed=seed, n_features=int(x_fit.shape[1])
            ),
            "inputs": [
                {"key": a.key, "block": a.spec["block"], "path": str(a.path)} for a in artifacts
            ],
            "n_features": int(x_fit.shape[1]),
            "n_fit": int(x_fit.shape[0]),
            "calibrated": calibrator is not None,
        },
    )
    logger.info("%s seed=%d: mae=%.4f -> %s", config.slug, seed, scores["mae"], run_dir)
    return run_dir


def write_run(
    run_dir: Path,
    *,
    spec: dict,
    seed: int,
    smiles: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    std: np.ndarray | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Score predictions and write a run directory.

    Both the tabular and the graph-network paths write through here, so a run
    directory has one shape whatever produced it, and aggregation has one thing
    to read.

    Parameters
    ----------
    run_dir : path-like
        Directory to write into; created if absent.
    spec : dict
        The run's specification, whose key decides whether a later sweep can
        skip it.
    seed : int
        The replicate seed.
    smiles : ndarray
        Canonical SMILES of the scored compounds, in prediction order.
    observed, predicted : ndarray
        Truth and prediction for those compounds.
    std : ndarray, optional
        Predicted standard deviation, where the model gives one.
    record : dict, optional
        Extra fields for the run record, such as the configuration behind it.

    Returns
    -------
    dict of str to float
        The metrics written.
    """
    scores = evaluate.metrics(observed, predicted)

    run_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({CANONICAL_COL: smiles, "observed": observed, "predicted": predicted})
    if std is not None:
        frame["predicted_std"] = std
    frame.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "key": provenance.spec_key(spec),
                "seed": seed,
                "spec": spec,
                **(record or {}),
                "n_test": int(len(observed)),
                "metrics": scores,
                "environment": provenance.environment(),
                "written_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return scores


def run_gnn_one(
    cell: GnnCell,
    manifest: Manifest,
    seed: int,
    *,
    train: TrainGnn,
    partitions: Partitions | None = None,
    results_dir: Path = RESULTS_DIR,
    force: bool = False,
) -> Path:
    """Train one graph-network cell at one seed and write its run directory.

    The architecture itself is vendored separately and injected here, so this
    module owns the run's identity, its skip rule and its record while knowing
    nothing about message passing.

    Parameters
    ----------
    cell : GnnCell
        The configuration to train, from the manifest.
    manifest : Manifest
        Supplies the seeds and the split paths.
    seed : int
        The replicate seed.
    train : callable
        Takes the cell's axes, the seed and the partitions, and returns
        predictions for the test partition in its row order.
    partitions : Partitions, optional
        Preloaded split frames.
    results_dir : path-like, optional
        Root the run directory hangs off.
    force : bool, optional
        Retrain even when a matching run is present.

    Returns
    -------
    Path
        The run directory.

    Raises
    ------
    SweepError
        If the architecture returns a different number of predictions than
        there are compounds to score.
    """
    partitions = partitions or load_partitions()
    spec = provenance.block_spec(
        "gnn_run",
        VERSION,
        params={"cell": cell.id, "axes": cell.axes, "seed": seed},
        split=provenance.spec_key(
            {
                "fit_train": sorted(partitions.fit_train[CANONICAL_COL]),
                "test": sorted(partitions.test[CANONICAL_COL]),
            }
        ),
    )
    run_dir = cell.run_dir(seed, results_dir)

    if is_complete(run_dir, spec) and not force:
        logger.info("%s seed=%d: already complete", cell.id, seed)
        return run_dir

    logger.info("%s seed=%d: training", cell.id, seed)
    predicted = np.asarray(train(cell.axes, seed, partitions), dtype=np.float64)
    observed = partitions.test[TARGET_COL].to_numpy(dtype=np.float64)
    if predicted.shape != observed.shape:
        raise SweepError(
            f"{cell.id} seed={seed}: got {predicted.shape} predictions for "
            f"{observed.shape} test compounds"
        )

    scores = write_run(
        run_dir,
        spec=spec,
        seed=seed,
        smiles=partitions.test[CANONICAL_COL].to_numpy(),
        observed=observed,
        predicted=predicted,
        record={"cell": cell.id, "axes": cell.axes, "n_fit": int(len(partitions.fit_train))},
    )
    logger.info("%s seed=%d: mae=%.4f -> %s", cell.id, seed, scores["mae"], run_dir)
    return run_dir


def run_stage(
    manifest: Manifest,
    stage_id: str,
    resolved: dict[str, Any] | None = None,
    *,
    seeds: Iterable[int] | None = None,
    results_dir: Path = RESULTS_DIR,
    force: bool = False,
) -> list[Path]:
    """Run every configuration and seed of one stage."""
    partitions = load_partitions()
    seeds = list(seeds if seeds is not None else manifest.seeds)
    configs = manifest.expand(stage_id, resolved)
    logger.info("%s: %d configurations x %d seeds", stage_id, len(configs), len(seeds))

    written = []
    for config in configs:
        for seed in seeds:
            written.append(
                run_one(
                    config,
                    manifest,
                    seed,
                    partitions=partitions,
                    results_dir=results_dir,
                    force=force,
                )
            )
    return written


def _fit_calibrator(
    config: TabularConfig,
    artifacts: Sequence[provenance.Artifact],
    partitions: Partitions,
    seed: int,
) -> IsotonicRegression:
    """Fit an isotonic map on predictions the model has not been trained on."""
    x_train = assemble(artifacts, partitions.fit_train[CANONICAL_COL].tolist())
    y_train = partitions.fit_train[TARGET_COL].to_numpy(dtype=np.float64)
    x_val = assemble(artifacts, partitions.fit_val[CANONICAL_COL].tolist())
    y_val = partitions.fit_val[TARGET_COL].to_numpy(dtype=np.float64)

    held_out = regressors.fit_predict(config.regressor, x_train, y_train, x_val, seed=seed)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(held_out.mean, y_val)
    return calibrator


def _seed_params(block: str, seed: int) -> dict[str, Any]:
    """Return the seed parameter for a block that has one, or nothing."""
    spec = features.BLOCKS.get(block)
    if spec is not None and getattr(spec, "seeded", False):
        return {"seed": seed}
    return {}
