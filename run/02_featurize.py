"""Compute the stage-one feature blocks over every molecule in the split.

Each block is a per-molecule function of structure alone, so one table spanning
the fit and test partitions carries no information between them. Blocks that
come from a trained encoder are seeded and are computed once per replicate
seed; the rest are computed once. A cached block is left alone unless
``--force`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import encoders  # noqa: E402
import features  # noqa: E402
import manifest as manifest_module  # noqa: E402

DESCRIPTION = "Compute the stage-one feature blocks over every molecule in the split."

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--blocks",
        nargs="+",
        metavar="NAME",
        help="which blocks to build; defaults to every registered block",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="N",
        help="replicate seeds for the seeded blocks; defaults to the manifest's seeds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even when a cached block is already present",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Build every requested block, one artifact per block and seed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # the trained-encoder blocks live in their own module and are installed
    # into the stage-one registry before anything dispatches on a block name
    encoders.register()

    spec = manifest_module.load()
    seeds = args.seeds if args.seeds is not None else list(spec.seeds)
    names = args.blocks if args.blocks is not None else sorted(features.BLOCKS)

    unknown = [name for name in names if name not in features.BLOCKS]
    if unknown:
        raise SystemExit(f"unknown blocks {unknown}; known: {sorted(features.BLOCKS)}")

    for name in names:
        # the seeded rule matches the one sweep applies when it looks a block
        # up, so what is precomputed here is what the sweep later finds cached
        block_seeds = seeds if getattr(features.BLOCKS[name], "seeded", False) else [None]
        for seed in block_seeds:
            params = {} if seed is None else {"seed": seed}
            artifact = features.build(name, params=params, force=args.force)
            logger.info("%s seed=%s -> %s", name, seed, artifact.path)


if __name__ == "__main__":
    main()
