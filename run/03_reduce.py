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
        help="descriptor PCA widths to build; defaults to those the manifest declares",
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
    spec: manifest_module.Manifest, widths: list[int] | None
) -> list[tuple[list[str], int | None]]:
    """Return every (block names, width) reduction the axes call for.

    Parameters
    ----------
    spec : Manifest
        The coverage spec, whose axis levels name their raw blocks and widths.
    widths : list of int or None
        Descriptor widths to build. Defaults to the ``descriptor_pca`` axis.

    Returns
    -------
    list of (list of str, int or None)
        The blocks to join and the width to project them to, a width of None
        meaning imputed but not rotated.
    """
    axes: dict[str, Any] = spec.axes
    wanted: list[tuple[list[str], int | None]] = []

    for group in ("embedding", "readout"):
        for level in axes[group].values():
            if level is None:
                continue
            wanted.append(([level["block"]], level.get("pca")))

    descriptor_widths = widths if widths is not None else list(axes["descriptor_pca"])
    for level in axes["descriptors"].values():
        if level is None:
            continue
        wanted.extend((list(level["blocks"]), width) for width in descriptor_widths)

    return wanted


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
    wanted = planned_reductions(spec, args.widths)
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
