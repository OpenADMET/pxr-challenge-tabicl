"""Run one stage of the tabular sweep: every configuration it varies, at every seed.

A stage sweeps the dimensions it names and holds fixed what an earlier gate
already settled on this split. Those settled axes are read from the gate files
``run/05_aggregate.py`` writes, so running a chain of stages needs no values
carried between commands by hand and a replicator cannot mistype one. ``--fix
axis=value`` overrides a gate for exploration, and the override is recorded in
the run's own specification, so a hand-pinned run is never mistaken for a gated
one.

The graph-network cells are not a stage, since they sweep no shared axis and
depend on no gate. ``--stage gnn`` trains every cell the manifest enumerates,
optionally narrowed with ``--cells``.

Completed runs are skipped by their recorded specification rather than by the
directory existing, so an interrupted sweep can be restarted and a changed
upstream block is redone. ``--force`` is the only way to overwrite a run whose
specification still matches.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import encoders  # noqa: E402
import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402
import sweep  # noqa: E402

DESCRIPTION = "Run one stage of the sweep, at every configuration and seed."

# not a manifest stage: the graph-network cells sweep no shared axis and wait
# on no gate, so they are enumerated rather than expanded
GNN_STAGE = "gnn"

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--stage",
        required=True,
        help="which stage of the manifest to run, or 'gnn' for the graph-network cells",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        metavar="ID",
        help="restrict a gnn run to these cells; defaults to every cell",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="AXIS=VALUE",
        help="override an axis a gate settled; repeat for each one",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="N",
        help="replicate seeds to run; defaults to the manifest's seeds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="refit even when a matching run is already present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and stop, without fitting anything",
    )
    return parser.parse_args(argv)


def parse_fixed(assignments: list[str]) -> dict[str, Any]:
    """Read repeated ``axis=value`` arguments into resolved axis levels.

    Parameters
    ----------
    assignments : list of str
        Arguments of the form ``embedding=chemeleon``.

    Returns
    -------
    dict
        Axis name mapped to its level, with a whole-number level read as an
        integer so ``descriptor_pca=128`` matches the axis it belongs to.

    Raises
    ------
    SystemExit
        If an argument is not an assignment, or names an axis that does not
        exist.
    """
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


def _run_gnn(spec: manifest_module.Manifest, args: argparse.Namespace, seeds: list[int]) -> None:
    """Print the graph-network plan and, unless this is a dry run, train it."""
    wanted = set(args.cells) if args.cells else None
    unknown = wanted - {cell.id for cell in spec.gnn_cells} if wanted else set()
    if unknown:
        raise SystemExit(f"unknown gnn cells: {sorted(unknown)}")

    selected = [c for c in spec.gnn_cells if wanted is None or c.id in wanted]
    print(f"stage gnn: {len(selected)} cells x {len(seeds)} seeds")
    print(f"  seeds: {seeds}")
    for cell in selected:
        print(f"  {cell.id}")
    print(f"total runs: {len(selected) * len(seeds)}")

    if args.dry_run:
        return

    written = sweep.run_gnn_stage(spec, seeds=seeds, cells=args.cells, force=args.force)
    logger.info("gnn: %d runs in %s", len(written), sweep.RESULTS_DIR)


def main() -> None:
    """Expand the stage, print its plan, and run every configuration and seed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    encoders.register()
    spec = manifest_module.load()
    seeds = args.seeds if args.seeds is not None else list(spec.seeds)

    if args.stage == GNN_STAGE:
        _run_gnn(spec, args, seeds)
        return

    overrides = parse_fixed(args.fix)
    settled = gates.settled(spec, args.stage)
    resolved = {**settled, **overrides}
    configs = spec.expand(args.stage, resolved)

    print(f"stage {args.stage}: {len(configs)} configurations x {len(seeds)} seeds")
    print(f"  from gates: {settled or 'nothing (the stage settles its own axes)'}")
    if overrides:
        print(f"  overridden by hand: {overrides}")
    print(f"  seeds: {seeds}")
    for config in configs:
        print(f"  {config.slug}")
    print(f"total runs: {len(configs) * len(seeds)}")

    if args.dry_run:
        return

    written = sweep.run_stage(spec, args.stage, resolved, seeds=seeds, force=args.force)
    logger.info("%s: %d runs in %s", args.stage, len(written), sweep.RESULTS_DIR)


if __name__ == "__main__":
    main()
