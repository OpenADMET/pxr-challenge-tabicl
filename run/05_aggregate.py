"""Read every run into results.parquet and report coverage against the manifest.

The tidy table holds one row per run. Coverage is reported per stage, since a
stage's ``unplanned`` set is relative to that stage alone: runs another stage
called for show up there, so read the stages together to account for a whole
sweep. A stage that waits on an earlier gate reads that gate's recorded
decision, the same way the sweep does; ``--fix`` overrides it.

Gates are also resolved here, because deciding one means reading every run of
the stage that gates it, which is what this module already does. Each stage
whose runs are complete has its rule applied and its decision written to
``results/gates/<id>.json``; a stage that is unfinished, or that waits on a gate
not yet decided, is reported and skipped. An existing decision is left alone
unless ``--regate`` is passed, so a settled gate cannot move under a sweep that
has already been run against it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aggregate  # noqa: E402
import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402

DESCRIPTION = "Read every run into results.parquet and report coverage against the manifest."

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--results",
        type=Path,
        default=aggregate.RESULTS_DIR,
        help="root holding the tabular/ and gnn/ run directories",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="ID",
        help="report coverage for this stage; repeat, or omit for every stage",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="AXIS=VALUE",
        help="an axis an earlier gate settled, needed by the stages that wait on it",
    )
    parser.add_argument(
        "--decide",
        metavar="STAGE",
        help="record a decision for this stage's gate, with --choose and --reason",
    )
    parser.add_argument(
        "--choose",
        action="append",
        default=[],
        metavar="AXIS=VALUE",
        help="the level chosen, one per axis the gate decides; repeat as needed",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="why, in prose. Recorded verbatim: it is the decision's only defence",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="how many leading ensemble rows to print, by mean absolute error",
    )
    return parser.parse_args(argv)


def _parse_choice(assignments: list[str]) -> dict[str, Any]:
    """Read repeated ``axis=value`` arguments into a gate's chosen levels."""
    if not assignments:
        raise SystemExit("--decide needs at least one --choose axis=value")
    return parse_fixed(assignments)


def parse_fixed(assignments: list[str]) -> dict[str, Any]:
    """Read repeated ``axis=value`` arguments into resolved axis levels."""
    resolved: dict[str, Any] = {}
    for assignment in assignments:
        axis, separator, value = assignment.partition("=")
        if not separator:
            raise SystemExit(f"expected axis=value, got {assignment!r}")
        if axis not in manifest_module.TABULAR_AXES:
            raise SystemExit(f"unknown axis {axis!r}; known: {list(manifest_module.TABULAR_AXES)}")
        resolved[axis] = int(value) if value.lstrip("-").isdigit() else value
    return resolved


def report_coverage(
    spec: manifest_module.Manifest,
    stage_ids: list[str],
    resolved: dict[str, Any],
    results_root: Path,
) -> None:
    """Print a coverage report for each stage, plus the graph-network cells."""
    for stage_id in stage_ids:
        stage = spec.stage(stage_id)
        if stage.reuses_runs:
            print(f"{stage_id}: reuses earlier runs and adds none of its own")
            continue
        try:
            settled = resolved or gates.settled(spec, stage_id)
            coverage = aggregate.stage_coverage(spec, stage_id, settled, results_root)
        except (gates.GateError, manifest_module.ManifestError) as err:
            # a stage waiting on a gate cannot be expanded until that gate is
            # decided, which is a state to report rather than an error to raise
            print(f"{stage_id}: not expandable yet ({err})")
            continue
        print(aggregate.coverage_report(coverage, stage_id))

    print(aggregate.coverage_report(aggregate.gnn_coverage(spec, results_root), "gnn"))


def report_gates(spec: manifest_module.Manifest, results_root: Path) -> None:
    """Say which gates are decided and which are waiting on a decision.

    Nothing is decided here. A gate is a judgement about configurations the
    evidence cannot separate, so it is made by a person and recorded with its
    reason; this only says where one is missing and how to write it.
    """
    for stage in spec.stages:
        if stage.gate is None:
            continue

        if gates.gate_path(stage.gate.id).exists():
            decision = gates.read(stage.gate.id)
            who = decision.get("decided_by", "unknown")
            print(f"{stage.gate.id}: decided by {who}, chose {decision['chosen']}")
            print(f"    {decision.get('reason', '(no reason recorded)')}")
            continue

        try:
            measured = gates.evidence(spec, stage.id, results_dir=results_root)
        except (gates.GateError, manifest_module.ManifestError) as err:
            print(f"{stage.gate.id}: not ready ({err})")
            continue

        tied = len(measured["indistinguishable_from_leader"]) + 1
        print(
            f"{stage.gate.id}: undecided. {len(measured['ranking'])} configurations ranked, "
            f"{tied} indistinguishable from the leader {measured['leader_slug']}"
        )
        print(
            f"    decide it with: python run/05_aggregate.py --decide {stage.id} "
            f"{'='.join([measured['chooses'][0], '<level>'])} --reason '...'"
        )


def record_decision(
    spec: manifest_module.Manifest, results_root: Path, stage_id: str, chosen: dict, reason: str
) -> None:
    """Write one gate decision, with the evidence it was made against."""
    decision = gates.decide(
        spec,
        stage_id,
        chosen,
        reason,
        decided_by=_whoami(),
        results_dir=results_root,
    )
    gates.write(decision)
    print(f"{decision['gate']}: chose {decision['chosen']}")
    print(f"    {decision['reason']}")


def _whoami() -> str:
    """Return who is recording a decision, for the record to attribute it to."""
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def main() -> None:
    """Write the tidy table, print the leading ensemble rows, and report coverage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    spec = manifest_module.load()
    frame = aggregate.tidy(args.results)
    aggregate.write_table(frame, args.results)

    if not frame.empty and args.top > 0:
        summary = aggregate.summarize(frame)
        ensembles = summary.loc[summary["aggregation"] == aggregate.ENSEMBLE]
        leaders = ensembles.nsmallest(args.top, "mae")
        print(f"leading ensembles by mae ({len(ensembles)} configurations):")
        print(leaders[["config_id", "n_seeds", "mae", "rmse", "r2"]].to_string(index=False))

    if args.decide:
        record_decision(spec, args.results, args.decide, _parse_choice(args.choose), args.reason)
    else:
        report_gates(spec, args.results)

    stage_ids = args.stage or [stage.id for stage in spec.stages]
    report_coverage(spec, stage_ids, parse_fixed(args.fix), args.results)


if __name__ == "__main__":
    main()
