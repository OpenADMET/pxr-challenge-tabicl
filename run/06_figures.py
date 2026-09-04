"""Draw the manifest's figures from the runs on disk.

Every figure is a slice of the one completed table, so this reads the runs once
and then renders whichever figures were asked for. A figure whose slice holds no
runs is skipped and named at the end rather than written as a blank chart, which
is how a partial sweep reads as partial instead of as a measured absence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

# figures are a build product written to disk, never shown, so the interactive
# backends are not wanted and would fail on a headless machine
matplotlib.use("Agg")

import aggregate  # noqa: E402
import evaluate  # noqa: E402
import figures  # noqa: E402
import manifest as manifest_module  # noqa: E402

DESCRIPTION = "Draw the manifest's figures from the runs on disk."

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
        "--out",
        type=Path,
        default=figures.FIGURES_DIR,
        help="directory the rendered figures are written into",
    )
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        metavar="ID",
        help="render this figure; repeat, or omit for every figure the manifest declares",
    )
    parser.add_argument(
        "--metric",
        default=figures.DEFAULT_METRIC,
        choices=list(evaluate.METRIC_NAMES),
        help="which metric the spread panels are drawn on",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=figures.DEFAULT_RESAMPLES,
        help="compound resamples behind the paired-bootstrap panels of figures 3 and 4",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=figures.DEFAULT_DPI,
        help="resolution of the rendered images",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Read the runs, render the requested figures, and report what was skipped."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    spec = manifest_module.load()
    frame = aggregate.tidy(args.results)
    if frame.empty:
        raise SystemExit(f"no runs under {args.results}, so there is nothing to draw")

    written, skipped = figures.render_all(
        spec,
        frame,
        args.out,
        args.figure or None,
        metric=args.metric,
        n_resamples=args.resamples,
        dpi=args.dpi,
    )

    for figure_id, path in written.items():
        print(f"{figure_id}: {path}")
    for entry in skipped:
        print(f"{entry.figure_id}: skipped, {entry.reason}")


if __name__ == "__main__":
    main()
