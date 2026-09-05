"""Resolve a stage's gate into a recorded decision the next stage reads.

A staged sweep only reproduces if the decision between two stages is an
artifact. Declaring the chain in the manifest is not enough: without this
module the winning axis values reach the next stage as ``--fix`` arguments
typed by whoever ran it, so the chain is carried by shell history, a typo pins
the sweep to a configuration nothing chose, and reproducing the work means
rerunning the aggregation and reading a winner off a table by eye.

So a gate is resolved once, from the completed runs of the stage that gates it,
and written to ``results/gates/<gate_id>.json``. The file carries the chosen
axis values, the rule that chose them, the ranking that rule was applied to,
and the keys of the runs it read. Later stages load it instead of being told.

Ties are decided rather than hidden, and by two bars rather than one. The
ranking is by ensemble MAE. A difference smaller than compound sampling noise
is not a result, so the leader is compared to every other configuration by a
paired bootstrap over the 260 phase-2 compounds, and the ones it does not
separate from are recorded as tied.

That bar alone is too weak to decide on. At 260 compounds the bootstrap fails
to separate configurations differing by ten times the run-to-run noise, so
taking the cheapest thing it waves through would trade real accuracy for
columns. A second bar decides: a saving counts as free only when it moves the
metric less than changing the training seed does, measured by the leader's own
spread across seeds. The cheapest configuration clearing both wins, cheap
meaning fewer feature blocks and then fewer columns, and the file records the
tied set, the affordable subset, the budget and whether cost decided it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import aggregate
import evaluate
import manifest as manifest_module
import provenance
from manifest import Manifest, TabularConfig

logger = logging.getLogger(__name__)

GATES_DIR = Path("results/gates")

# bumped when the meaning of a decision changes, never for a cosmetic edit
VERSION = 1

# the metric every gate rule ranks on
RANK_METRIC = "mae"

# stands in for a block whose column count the manifest does not declare, so an
# undeclared block kept whole is treated as the most expensive rather than the
# cheapest thing on its axis
_WIDEST = 1 << 30


class GateError(RuntimeError):
    """A gate cannot be resolved, or a stage's decision is missing."""


def gate_path(gate_id: str, gates_dir: Path | None = None) -> Path:
    """Where one gate's decision is written."""
    return _dir(gates_dir) / f"{gate_id}.json"


def _dir(gates_dir: Path | None) -> Path:
    """Resolve the gate directory at call time, not at import time."""
    return GATES_DIR if gates_dir is None else gates_dir


def read(gate_id: str, gates_dir: Path | None = None) -> dict[str, Any]:
    """Load a gate's recorded decision.

    Raises
    ------
    GateError
        If the gate has not been resolved yet.
    """
    path = gate_path(gate_id, gates_dir)
    if not path.exists():
        raise GateError(
            f"gate {gate_id!r} has not been resolved; run its stage, then "
            f"resolve it, so {path} exists"
        )
    return json.loads(path.read_text())


def settled(manifest: Manifest, stage_id: str, gates_dir: Path | None = None) -> dict[str, Any]:
    """Return the axis values a stage inherits from the gates before it.

    Parameters
    ----------
    manifest : Manifest
        The parsed coverage spec.
    stage_id : str
        The stage about to run.
    gates_dir : path-like, optional
        Where decisions are written.

    Returns
    -------
    dict
        Axis name mapped to the level a gate chose. Empty for a stage that
        depends on no gate.

    Notes
    -----
    A stage lists its gates in the order they were decided, and a later one
    supersedes an earlier one on any axis both chose. That is deliberate rather
    than a conflict: the descriptor probe settles which descriptor block to
    carry, and the featureset sweep afterwards may find that carrying none of
    it wins. The override is logged so it is visible rather than silent.

    Raises
    ------
    GateError
        If a gate the stage depends on has not been resolved.
    """
    resolved: dict[str, Any] = {}
    for gate_id in manifest.stage(stage_id).depends_on:
        for axis, value in read(gate_id, gates_dir)["chosen"].items():
            if axis in resolved and resolved[axis] != value:
                logger.info(
                    "%s: %s supersedes %r with %r on %s",
                    stage_id,
                    gate_id,
                    resolved[axis],
                    value,
                    axis,
                )
            resolved[axis] = value
    return resolved


def all_chosen(manifest: Manifest, gates_dir: Path | None = None) -> dict[str, Any]:
    """Return every axis value the resolved gates have chosen so far.

    Figures read this rather than restating a winner, so a selection cannot
    drift from the decision it is supposed to follow. The gates are walked in
    the order their stages run, so where two chose the same axis the later one
    stands, exactly as it does for a stage reading its own dependencies.
    """
    chosen: dict[str, Any] = {}
    for stage in manifest.stages:
        if stage.gate is None:
            continue
        path = gate_path(stage.gate.id, gates_dir)
        if path.exists():
            chosen.update(json.loads(path.read_text())["chosen"])
    return chosen


def cost(config: TabularConfig, axes: dict[str, Any]) -> tuple[int, int]:
    """Rank configurations by expense, for breaking a statistical tie.

    Fewer blocks first, then fewer columns in total. Comparing tied
    configurations on cost rather than on a difference the data does not
    support keeps the sweep from chasing noise into a wider featureset.

    A block kept whole costs its own column count, not the sentinel that
    records it: keeping RDKit's 217 descriptors is the widest option on that
    axis, not the narrowest, and ordering on the raw sentinel would rank it
    cheaper than every reduction. Widths are summed rather than compared axis
    by axis, so the ordering does not depend on which axis is looked at first.
    """
    total = 0
    for width_axis, block_axis in manifest_module.WIDTH_AXES.items():
        width = getattr(config, width_axis)
        if width == manifest_module.NATIVE:
            columns = manifest_module.native_width(axes, block_axis, getattr(config, block_axis))
            width = columns if columns is not None else _WIDEST
        total += width
    return (config.n_blocks, total)


def resolve(
    manifest: Manifest,
    stage_id: str,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Apply a stage's gate rule to its completed runs and return the decision.

    Parameters
    ----------
    manifest : Manifest
        The parsed coverage spec.
    stage_id : str
        The stage whose gate is being decided.
    results_dir : path-like, optional
        Root holding the run directories.
    gates_dir : path-like, optional
        Where the gates this stage depends on were written.
    n_resamples : int, optional
        Compound resamples per paired bootstrap.

    Returns
    -------
    dict
        The decision, ready to be written.

    Raises
    ------
    GateError
        If the stage gates nothing, or if any planned run is missing.
    """
    stage = manifest.stage(stage_id)
    if stage.gate is None:
        raise GateError(f"stage {stage_id!r} has no gate to resolve")

    with provenance.timed() as elapsed:
        configs = manifest.expand(stage_id, settled(manifest, stage_id, gates_dir))
        scored = _score(configs, manifest, results_dir)
        ranked = sorted(scored, key=lambda row: row[RANK_METRIC])
        leader = ranked[0]

        # a lead smaller than compound sampling noise is not a result, so
        # everything the leader does not separate from is a candidate on cost
        tied = _tied_with(leader, ranked[1:], n_resamples=n_resamples)

    # the bootstrap is a weak bar at 260 compounds: it fails to separate
    # configurations differing by far more than run-to-run noise. A saving
    # only counts as free if it moves the metric less than changing the
    # training seed does
    budget = leader["seed_spread"]
    affordable = [row for row in tied if row[RANK_METRIC] - leader[RANK_METRIC] <= budget]
    candidates = [leader, *affordable]
    chosen_row = min(candidates, key=lambda row: cost(row["config"], manifest.axes))
    chosen = {axis: getattr(chosen_row["config"], axis) for axis in stage.gate.chooses}

    return {
        "gate": stage.gate.id,
        "stage": stage_id,
        "version": VERSION,
        "rule": stage.gate.rule,
        "metric": RANK_METRIC,
        "chooses": list(stage.gate.chooses),
        "chosen": chosen,
        "chosen_slug": chosen_row["config"].slug,
        "leader_slug": leader["config"].slug,
        "tie_broken_on_cost": chosen_row is not leader,
        "tied_with_leader": [row["config"].slug for row in tied],
        "within_seed_spread": [row["config"].slug for row in affordable],
        "seed_spread_budget": budget,
        "ranking": [_public(row) for row in ranked],
        "n_resamples": n_resamples,
        "wall_clock_s": elapsed(),
        "environment": provenance.environment(),
    }


def write(decision: dict[str, Any], gates_dir: Path | None = None) -> Path:
    """Write a resolved decision, replacing any earlier one for that gate."""
    resolved = _dir(gates_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    path = gate_path(decision["gate"], resolved)
    with provenance.atomic(path) as partial:
        partial.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    logger.info("gate %s chose %s", decision["gate"], decision["chosen"])
    return path


def _score(
    configs: list[TabularConfig], manifest: Manifest, results_dir: Path
) -> list[dict[str, Any]]:
    """Score every configuration's seed ensemble, requiring a complete stage."""
    rows: list[dict[str, Any]] = []
    for config in configs:
        run_dirs = [config.run_dir(seed, results_dir) for seed in manifest.seeds]
        missing = [d for d in run_dirs if not (d / "predictions.csv").exists()]
        if missing:
            raise GateError(
                f"cannot decide on a partial stage: {len(missing)} of "
                f"{len(run_dirs)} runs missing for {config.slug}, first {missing[0]}"
            )

        # the ensemble is the quantity the leaderboard entry reports, so gates
        # rank on it rather than on a mean of seed-wise scores
        stacked, observed = aggregate.stack_predictions(run_dirs)
        pooled = evaluate.ensemble_mean(stacked)

        # how far the metric moves when only the training seed changes, which
        # is the yardstick a cost saving has to come in under
        per_seed = [evaluate.metrics(observed, row)[RANK_METRIC] for row in stacked]
        rows.append(
            {
                "config": config,
                "observed": observed,
                "prediction": pooled,
                "seed_spread": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                "n_seeds": len(run_dirs),
                **evaluate.metrics(observed, pooled),
            }
        )
    return rows


def _tied_with(
    leader: dict[str, Any], others: list[dict[str, Any]], *, n_resamples: int
) -> list[dict[str, Any]]:
    """Return the configurations a paired bootstrap does not separate from the leader."""
    tied = []
    for row in others:
        result = evaluate.paired_bootstrap(
            leader["observed"],
            leader["prediction"],
            row["prediction"],
            metric=RANK_METRIC,
            n_resamples=n_resamples,
        )
        if not evaluate.excludes_zero(result):
            tied.append(row)
    return tied


def _public(row: dict[str, Any]) -> dict[str, Any]:
    """Strip the arrays off a scored row, leaving what belongs in the record."""
    return {
        "slug": row["config"].slug,
        "config": row["config"].as_dict(),
        "n_seeds": int(row["n_seeds"]),
        "seed_spread": float(row["seed_spread"]),
        **{name: float(row[name]) for name in evaluate.METRIC_NAMES if name in row},
    }
