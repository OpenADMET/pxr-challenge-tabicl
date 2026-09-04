"""Read every run into results.parquet and report coverage against the manifest.

The tidy table holds one row per run. Coverage is reported per stage, since a
stage's ``unplanned`` set is relative to that stage alone: runs another stage
called for show up there, so read the stages together to account for a whole
sweep. A stage that waits on an earlier gate needs those axes passed with
``--fix``, the same way the sweep takes them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aggregate  # noqa: E402
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
        "--top",
        type=int,
        default=5,
        help="how many leading ensemble rows to print, by mean absolute error",
    )
    return parser.parse_args(argv)


def parse_fixed(assignments: list[str]) -> dict[str, Any]:
    """Read repeated ``axis=value`` arguments into resolved axis levels."""
    resolved: dict[str, Any] = {}
    for assignment in assignments:
        axis, separator, value = assignment.partition("=")
        if not separator:
            raise SystemExit(f"--fix expects axis=value, got {assignment!r}")
        if axis not in manifest_module.TABULAR_AXES:
            raise SystemExit(
                f"--fix names unknown axis {axis!r}; known: {list(manifest_module.TABULAR_AXES)}"
            )
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
            coverage = aggregate.stage_coverage(spec, stage_id, resolved, results_root)
        except manifest_module.ManifestError as err:
            # a stage waiting on a gate cannot be expanded until that gate is
            # decided, which is a state to report rather than an error to raise
            print(f"{stage_id}: not expandable yet ({err})")
            continue
        print(aggregate.coverage_report(coverage, stage_id))

    print(aggregate.coverage_report(aggregate.gnn_coverage(spec, results_root), "gnn"))


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

    stage_ids = args.stage or [stage.id for stage in spec.stages]
    report_coverage(spec, stage_ids, parse_fixed(args.fix), args.results)


if __name__ == "__main__":
    main()
