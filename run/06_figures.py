"""Draw the figures from the runs on disk.

Every panel is measured the same way a gate is, by the all-pairwise paired
bootstrap under Benjamini-Hochberg, so a figure and the decision recorded
beside it cannot disagree. What each figure shows is declared in ``panels``;
this reads the runs, measures them, and writes a page per figure.

The bootstrap is the expensive part, so ``--figure`` is worth using while
iterating on one panel.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aggregate  # noqa: E402
import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402
import panels  # noqa: E402

DESCRIPTION = "Draw the figures from the runs on disk."

# the panel drawn from the uncertainty artifacts rather than from the run table
UNCERTAINTY = "fig6"

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
        default=panels.FIGURES_DIR,
        help="directory the rendered figures are written into",
    )
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        metavar="ID",
        help="render this figure only; repeat, or omit for all of them",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=panels.DEFAULT_RESAMPLES,
        help="compound resamples behind every panel's paired bootstrap",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Measure the panels the manifest declares and write each one out."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    spec = manifest_module.load()
    wanted = set(args.figure)

    written = {}
    if wanted != {UNCERTAINTY}:
        built = panels.build(spec, results_dir=args.results, n_resamples=args.resamples)
        # figure 2 is one page of two panels, so it is asked for by the page's
        # name rather than by either panel's
        if wanted:
            built = [
                panel
                for panel in built
                if panel.id in wanted or (panel.id.startswith("fig2") and "fig2" in wanted)
            ]
        written |= panels.render(built, args.out)

    if not wanted or UNCERTAINTY in wanted:
        target = gates.resolve_target(spec)
        figure = panels.uncertainty_figure(target.slug, results_dir=args.results)
        path = Path(args.out) / f"{UNCERTAINTY}.html"
        figure.write_html(path, include_plotlyjs="cdn", full_html=True)
        written[UNCERTAINTY] = path
        logger.info("wrote %s", path)

    for figure_id, path in sorted(written.items()):
        print(f"{figure_id}: {path}")
    if wanted - set(written) - {"fig2a", "fig2b"}:
        raise SystemExit(f"no such figure: {sorted(wanted - set(written))}")


if __name__ == "__main__":
    main()
