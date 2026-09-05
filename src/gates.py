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

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import aggregate
import evaluate
import manifest as manifest_module
import provenance
from manifest import (
    NOT_REDUCED,
    TABULAR_AXES,
    Manifest,
    ManifestError,
    TabularConfig,
    normalize_widths,
)

logger = logging.getLogger(__name__)

GATES_DIR = Path("results/gates")

# bumped when the meaning of a decision changes, never for a cosmetic edit
VERSION = 1

# the metric every gate rule ranks on
RANK_METRIC = "mae"

# false discovery rate for the all-pairwise comparison, the error a candidate
# set wants controlled rather than the family-wise rate
FDR = 0.05

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


def trained_encoders(config: TabularConfig, axes: dict[str, Any]) -> int:
    """Return how many encoders a configuration has to train before it can run.

    Distinct encoders, not blocks: a prefix's embedding and readout come off
    one trained network, so a configuration using both pays for one. A block
    needing no training, such as the frozen CheMeleon embedding or any
    descriptor set, costs nothing here.

    Parameters
    ----------
    config : TabularConfig
        The configuration to price.
    axes : dict
        The axis vocabulary, mapping a level to the raw blocks behind it.

    Returns
    -------
    int
        Encoders to train, each of which is trained once per replicate seed.
    """
    import features

    names: set[str] = set()
    for axis in ("embedding", "readout", "descriptors"):
        level = (axes.get(axis) or {}).get(getattr(config, axis))
        if not level:
            continue
        for block in level.get("blocks", [level["block"]] if "block" in level else []):
            spec = features.BLOCKS.get(block)
            encoder = None if spec is None else getattr(spec, "encoder", None)
            if encoder is not None:
                names.add(encoder)
    return len(names)


def cost(config: TabularConfig, axes: dict[str, Any]) -> tuple[int, int, int]:
    """Rank configurations by expense, for breaking a statistical tie.

    Fewer blocks, then fewer encoders to train, then fewer columns in total.
    Comparing tied configurations on cost rather than on a difference the data
    does not support keeps the sweep from chasing noise into a wider featureset.

    The middle term is what separates two featuresets of the same shape and
    different provenance. A frozen foundation embedding and one fine-tuned on
    log2FC are both a single 256-column block, and only the second has to be
    trained, five times over. Ordering on width alone calls them equal and
    picks whichever happens to score lower, which is how the more expensive of
    two indistinguishable configurations wins.

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
    return (config.n_blocks, trained_encoders(config, axes), total)


def evidence(
    manifest: Manifest,
    stage_id: str,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Measure a stage's completed runs, without deciding anything.

    Returns what a decision should be made against: every configuration scored
    on the seed mean with its spread and its ensemble, every pair tested, and
    the cost facts. It chooses nothing, because at the top of these tables the
    configurations are statistically tied and any tie-break is a preference
    rather than a finding. A preference stated in prose can be argued with; the
    same preference expressed as a cost ordering cannot, and invites tuning the
    ordering until it returns the answer already believed.

    Parameters
    ----------
    manifest : Manifest
        The parsed coverage spec.
    stage_id : str
        The stage whose gate is being measured.
    results_dir : path-like, optional
        Root holding the run directories.
    gates_dir : path-like, optional
        Where the gates this stage depends on were written.
    n_resamples : int, optional
        Compound resamples for the family bootstrap.

    Returns
    -------
    dict
        The ranking, the significance evidence, and the stage's question.

    Raises
    ------
    GateError
        If the stage gates nothing, or if any planned run is missing.
    """
    stage = manifest.stage(stage_id)
    if stage.gate is None:
        raise GateError(f"stage {stage_id!r} has no gate to decide")

    with provenance.timed() as elapsed:
        configs = manifest.expand(stage_id, settled(manifest, stage_id, gates_dir))
        scored = _score(configs, manifest, results_dir)
        ranked = sorted(scored, key=lambda row: row[RANK_METRIC])
        separated, significance = _separated(ranked, n_resamples=n_resamples)

    tied = [row["config"].slug for row in ranked[1:] if row not in separated]
    return {
        "gate": stage.gate.id,
        "stage": stage_id,
        "version": VERSION,
        "question": stage.gate.question,
        "chooses": list(stage.gate.chooses),
        "metric": RANK_METRIC,
        "leader_slug": ranked[0]["config"].slug,
        "indistinguishable_from_leader": tied,
        "ranking": [_public(row, manifest) for row in ranked],
        "n_resamples": n_resamples,
        "significance": significance,
        "wall_clock_s": elapsed(),
    }


def confirm(
    manifest: Manifest,
    stage_id: str,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Check a stage's declared decision against its runs, and record both.

    The decision is read from the manifest rather than computed, because at the
    top of these tables the configurations are statistically tied and any
    tie-break is a preference. Declaring it keeps the preference legible and
    the pipeline autonomous: this verifies it and writes the evidence beside
    it, and never chooses.

    A choice the evidence now separates from the leader is still honoured, and
    logged as a warning. That is the case worth seeing rather than silently
    overriding: the runs have moved and the declared preference may no longer
    be defensible, which is a judgement for whoever reads the log.

    Parameters
    ----------
    manifest : Manifest
        The parsed coverage spec, carrying the declared decision.
    stage_id : str
        The stage whose gate is being confirmed.
    results_dir, gates_dir, n_resamples : optional
        Passed to :func:`evidence`.

    Returns
    -------
    dict
        The decision, its reason, and the evidence behind it.

    Raises
    ------
    GateError
        If the gate declares no decision, the decision has no reason, it does
        not cover exactly the axes the gate settles, or it names a
        configuration the stage did not run.
    """
    stage = manifest.stage(stage_id)
    if stage.gate is None:
        raise GateError(f"stage {stage_id!r} has no gate")
    if not stage.gate.decision:
        raise GateError(
            f"{stage.gate.id}: undecided. Declare the choice under this gate's "
            f"decision: in the manifest, with a reason, then run this again"
        )
    if not stage.gate.reason.strip():
        raise GateError(f"{stage.gate.id}: a decision needs a reason; it is its only defence")

    chosen = dict(stage.gate.decision)
    expected = set(stage.gate.chooses)
    if set(chosen) != expected:
        raise GateError(
            f"{stage.gate.id}: decision must cover exactly {sorted(expected)}, got {sorted(chosen)}"
        )

    measured = evidence(
        manifest,
        stage_id,
        results_dir=results_dir,
        gates_dir=gates_dir,
        n_resamples=n_resamples,
    )
    matching = [
        row
        for row in measured["ranking"]
        if all(row["config"][axis] == value for axis, value in chosen.items())
    ]
    if not matching:
        raise GateError(f"{stage.gate.id}: no configuration this stage ran matches {chosen}")

    slug = matching[0]["slug"]
    separated = (
        slug not in measured["indistinguishable_from_leader"] and slug != measured["leader_slug"]
    )
    if separated:
        logger.warning(
            "%s: the declared choice %s is now separated from the leader %s; "
            "the runs may have moved since it was written",
            stage.gate.id,
            slug,
            measured["leader_slug"],
        )

    return {
        **measured,
        "chosen": chosen,
        "chosen_slug": slug,
        "reason": stage.gate.reason.strip(),
        "is_leader": slug == measured["leader_slug"],
        "separated_from_leader": separated,
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

        # the subject is a single model, so a configuration is scored by the
        # mean of its seeds and not by the ensemble of them. Seeds are
        # replicates here, for statistical power and a variance estimate,
        # rather than a way to build a better predictor. The ensemble is
        # carried alongside because the leaderboard anchor is one
        stacked, observed = aggregate.stack_predictions(run_dirs)
        per_seed = [evaluate.metrics(observed, row) for row in stacked]
        seed_mean = {name: float(np.mean([s[name] for s in per_seed])) for name in per_seed[0]}
        ensemble = evaluate.metrics(observed, evaluate.ensemble_mean(stacked))

        rows.append(
            {
                "config": config,
                "observed": observed,
                "stacked": stacked,
                # how far the metric moves when only the training seed changes
                "seed_spread": float(np.std([s[RANK_METRIC] for s in per_seed], ddof=1))
                if len(per_seed) > 1
                else 0.0,
                "n_seeds": len(run_dirs),
                "ensemble": ensemble,
                **seed_mean,
            }
        )
    return rows


def _separated(
    ranked: list[dict[str, Any]], *, n_resamples: int, fdr: float = FDR
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the configurations separated from the leader, and the evidence.

    Every pair is tested, not only each against the leader. Two reasons. The
    family a reader will use is every pair, because a figure of nineteen bars
    invites comparing any two of them, and a later stage asks questions that
    are not about the leader at all. And under a step-up procedure the larger
    family is the more powerful one here rather than the less: the extra
    comparisons mostly separate, which lifts the rank of a borderline one and
    with it the threshold it must clear.

    The whole family is resampled on one set of draws, so every difference is
    paired and the cost is one bootstrap per configuration rather than per
    pair.

    Parameters
    ----------
    ranked : list of dict
        Scored configurations, leader first.
    n_resamples : int
        Compound resamples.
    fdr : float, optional
        False discovery rate. Defaults to :data:`FDR`.

    Returns
    -------
    separated : list of dict
        The configurations the procedure distinguishes from the leader.
    evidence : dict
        The family size, the level, and each comparison against the leader with
        its p-value and the threshold it was judged against, so the verdict can
        be checked without rerunning it.
    """
    # each configuration's resampled score is the mean over its seeds, matching
    # what it is ranked on: resampling the ensemble would test a quantity the
    # gate does not decide on
    observed = ranked[0]["observed"]
    resampled = np.stack(
        [
            np.mean(
                evaluate.bootstrap_family(
                    observed, list(row["stacked"]), metric=RANK_METRIC, n_resamples=n_resamples
                ),
                axis=0,
            )
            for row in ranked
        ]
    )
    pairs = list(itertools.combinations(range(len(ranked)), 2))
    p_values = [evaluate.difference_p_value(resampled, i, j) for i, j in pairs]
    rejected = evaluate.benjamini_hochberg(p_values, fdr=fdr)

    order = np.argsort(p_values)
    rank_of = {int(index): position + 1 for position, index in enumerate(order)}
    separated, against_leader = [], []
    for position, ((i, j), p_value, is_separated) in enumerate(
        zip(pairs, p_values, rejected, strict=True)
    ):
        if i != 0:
            continue
        if is_separated:
            separated.append(ranked[j])
        against_leader.append(
            {
                "slug": ranked[j]["config"].slug,
                "p_value": p_value,
                "bh_threshold": rank_of[position] / len(pairs) * fdr,
                "separated": bool(is_separated),
            }
        )

    evidence = {
        "procedure": "all-pairwise paired bootstrap, Benjamini-Hochberg",
        "fdr": fdr,
        "n_comparisons": len(pairs),
        "n_separated_pairs": int(rejected.sum()),
        "against_leader": against_leader,
    }
    return separated, evidence


def _public(row: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    """Strip the arrays off a scored row, leaving what belongs in the record.

    Cost appears here as three facts about a configuration and not as an
    ordering: how many blocks it joins, how many encoders it has to train, and
    how many columns it carries. Whether any of that is worth a difference in
    score is a judgement, and judgements live in a decision's reason.
    """
    blocks, encoders, columns = cost(row["config"], manifest.axes)
    return {
        "slug": row["config"].slug,
        "config": row["config"].as_dict(),
        "n_seeds": int(row["n_seeds"]),
        "seed_spread": float(row["seed_spread"]),
        "n_blocks": blocks,
        "n_encoders_to_train": encoders,
        "n_columns": columns,
        "ensemble": {k: float(v) for k, v in row["ensemble"].items()},
        **{name: float(row[name]) for name in evaluate.METRIC_NAMES if name in row},
    }


def gated_config(manifest: Manifest) -> TabularConfig:
    """Return the configuration the settled gates point at.

    Raises
    ------
    SystemExit
        If a gate the winning configuration depends on has not been decided,
        since guessing at one would calibrate something no stage chose.
    """
    settled = all_chosen(manifest)
    missing = [axis for axis in ("embedding", "readout", "descriptors") if axis not in settled]
    if missing:
        raise SystemExit(
            f"no gated winner yet: {', '.join(missing)} unsettled. "
            "Run the ingredient stage and aggregate it, or name a --slug."
        )

    # the regressor gate may not have run; the manifest's fixed value stands in
    # and the record says which of the two it was
    levels = dict(settled)
    levels.setdefault("regressor", manifest.stage("ingredients").fixed["regressor"])
    levels.setdefault("calibration", "none")
    for axis in ("embedding_pca", "descriptor_pca"):
        levels.setdefault(axis, NOT_REDUCED)

    config = TabularConfig(**{a: levels[a] for a in TABULAR_AXES})
    # a later gate can supersede an earlier one on a block while leaving its
    # width behind: the featureset gate may drop descriptors that the width
    # probe settled at 128, and an unnormalized width would name a run
    # directory no stage ever wrote
    return normalize_widths(config, manifest.axes)


def find_config(manifest: Manifest, slug: str) -> TabularConfig:
    """Return the configuration a slug names, from every stage the manifest declares."""
    settled = all_chosen(manifest)
    for stage in manifest.stages:
        try:
            for config in manifest.expand(stage.id, settled):
                if config.slug == slug:
                    return config
        except ManifestError:
            continue
    raise SystemExit(f"no configuration in the manifest has slug {slug!r}")


def resolve_target(manifest: Manifest, slug: str | None = None) -> TabularConfig:
    """Return the configuration to act on: the one a slug names, or the gated winner.

    Parameters
    ----------
    manifest : Manifest
        The coverage spec.
    slug : str or None, optional
        A configuration to act on instead of the gated winner. None resolves
        the winner, which is the point of the default: a post-hoc step acts on
        what the sweep chose rather than on what was typed.

    Returns
    -------
    TabularConfig
    """
    return find_config(manifest, slug) if slug else gated_config(manifest)
