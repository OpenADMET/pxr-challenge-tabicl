"""Precompute the stage-two reductions the manifest's axes declare.

Optional. The sweep builds a reduction on demand and caches it under the same
key, so this script only moves that work forward: run it to fill the reduction
cache before a sweep, or skip it and let the first run of each configuration
pay for its own blocks.

Every reduction is fitted on the fit partition alone and then applied to all
molecules, test set included, which is the one place in the pipeline that can
leak and so the one place the fit rows are passed explicitly.
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
import features  # noqa: E402
import manifest as manifest_module  # noqa: E402
import reduce as reduction  # noqa: E402

DESCRIPTION = "Precompute the stage-two reductions the manifest's axes declare."

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        metavar="N",
        help="PCA widths to build; defaults to those each width axis declares",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        metavar="NAME",
        help=(
            "restrict to reductions drawing only on these raw blocks; use it to "
            "prepare a stage without training the encoders it does not need"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="N",
        help="replicate seeds to build for; defaults to the manifest's seeds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even when a cached reduction is already present",
    )
    return parser.parse_args(argv)


def planned_reductions(
    spec: manifest_module.Manifest,
    widths: list[int] | None = None,
    blocks: list[str] | None = None,
) -> list[tuple[list[str], int | None]]:
    """Return every (block names, width) reduction the axes call for.

    A block level declaring ``reduce: true`` is built at every level of the
    width axis that belongs to it; anything else is imputed but not rotated.
    The widths come from the manifest rather than from here, so this script
    cannot drift from what the sweep will ask for.

    Parameters
    ----------
    spec : Manifest
        The coverage spec, whose axis levels name their raw blocks.
    widths : list of int or None
        Override the widths to build. Defaults to each width axis's own levels.
    blocks : list of str or None
        Keep only reductions drawing entirely on these raw blocks. Defaults to
        every block, which includes the ones a trained encoder produces.

    Returns
    -------
    list of (list of str, int or None)
        The blocks to join and the width to project them to, a width of None
        meaning imputed but not rotated.
    """
    axes: dict[str, Any] = spec.axes
    wanted: list[tuple[list[str], int | None]] = []

    for group in ("embedding", "readout", "descriptors"):
        width_axis = manifest_module.WIDTH_OF.get(group)
        available = widths if widths is not None else list(axes.get(width_axis or "", []))
        for level in axes[group].values():
            if level is None:
                continue
            names = list(level.get("blocks", [level["block"]] if "block" in level else []))
            if not (width_axis and level.get("reduce")):
                wanted.append((names, None))
                continue

            # a width at or above the block's own size is not a reduction, and
            # the decomposition refuses it; the manifest declares the size
            columns = level.get("n_features")
            wanted.extend(
                (names, width) for width in available if columns is None or width < columns
            )

    if blocks is None:
        return wanted
    allowed = set(blocks)
    return [(names, width) for names, width in wanted if allowed.issuperset(names)]


def _seed_params(name: str, seed: int) -> dict[str, Any]:
    """Return the seed parameter for a block that has one, matching the sweep's rule."""
    return {"seed": seed} if getattr(features.BLOCKS[name], "seeded", False) else {}


def main() -> None:
    """Build every declared reduction, for each seed, from the stage-one blocks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    encoders.register()
    spec = manifest_module.load()
    seeds = args.seeds if args.seeds is not None else list(spec.seeds)
    fit_smiles = reduction.fit_molecules()
    wanted = planned_reductions(spec, args.widths, args.blocks)
    logger.info(
        "%d reductions x %d seeds, fitted on %d molecules",
        len(wanted),
        len(seeds),
        len(fit_smiles),
    )

    for seed in seeds:
        for names, width in wanted:
            raw = [
                features.build(name, params=_seed_params(name, seed), force=args.force)
                for name in names
            ]
            artifact = reduction.build(
                raw, width=width, fit_smiles=fit_smiles, seed=seed, force=args.force
            )
            logger.info("%s w=%s seed=%d -> %s", "+".join(names), width, seed, artifact.path)


if __name__ == "__main__":
    main()
