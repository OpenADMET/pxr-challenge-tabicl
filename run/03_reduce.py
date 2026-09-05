"""Precompute the stage-two reductions the sweep will ask for.

Optional. The sweep builds a reduction on demand and caches it under the same
key, so this script only moves that work forward: run it to fill the reduction
cache before a sweep, or skip it and let the first run of each configuration
pay for its own blocks.

What it builds is derived from the stages, with the settled gates applied, and
not from the width axes' declared levels. The two differ once a gate has
chosen: the embedding width axis declares six levels because the width probe
sweeps all six on the frozen CheMeleon block, but every later stage reads the
one width the gate settled, so the fine-tuned embeddings are only ever asked
for at that width. Planning from the axes would build the other five for them,
and planning from the stages does not. The consequence worth stating is that
running this with no arguments builds what the sweep will read and nothing
else, so no width has to be named on the command line; --widths and --blocks
remain as ways to ask for more than that, never as a way to get what is
needed.

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
import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402
import reduce as reduction  # noqa: E402

DESCRIPTION = "Precompute the stage-two reductions the sweep will ask for."

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        metavar="N",
        help="PCA widths to build; defaults to those the stages actually read",
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


def _reductions_for(config: Any, axes: dict[str, Any]) -> list[tuple[list[str], int | None]]:
    """Return the reductions one configuration reads, as (block names, width).

    A width of None means imputed but not rotated, which is what a block gets
    when it is passed through whole: either it declares no reduction, or the
    configuration keeps it native.
    """
    wanted: list[tuple[list[str], int | None]] = []
    for group in ("embedding", "readout", "descriptors"):
        level_name = getattr(config, group)
        level = axes[group].get(level_name)
        if level is None:
            continue

        names = list(level.get("blocks", [level["block"]] if "block" in level else []))
        width_axis = manifest_module.WIDTH_OF.get(group)
        width = getattr(config, width_axis) if width_axis else manifest_module.NOT_REDUCED

        # NATIVE is a block kept whole and NOT_REDUCED is one with no width to
        # keep; a level that declares no reduction has no width either way
        if not level.get("reduce") or width in (
            manifest_module.NATIVE,
            manifest_module.NOT_REDUCED,
        ):
            wanted.append((names, None))
            continue
        wanted.append((names, width))
    return wanted


def planned_reductions(
    spec: manifest_module.Manifest,
    widths: list[int] | None = None,
    blocks: list[str] | None = None,
    resolved: dict[str, Any] | None = None,
) -> list[tuple[list[str], int | None]]:
    """Return every (block names, width) reduction the stages call for.

    The plan is the union over every tabular stage of what its configurations
    read, with the settled gates applied, so it is exactly what the sweep will
    look for in the cache and nothing besides. Planning from the width axes
    instead would build every declared width for every reducible block, which
    over-builds any block a gate has already narrowed.

    Parameters
    ----------
    spec : Manifest
        The coverage spec, whose stages name the configurations that run.
    widths : list of int or None
        Keep only these widths, of those the stages read. Unreduced blocks are
        kept whatever this says, since they carry no width to filter on.
    blocks : list of str or None
        Keep only reductions drawing entirely on these raw blocks. Defaults to
        every block, which includes the ones a trained encoder produces.
    resolved : dict or None
        Gate decisions to apply, as axis name to chosen level. Defaults to
        whatever ``results/gates`` holds. A stage whose gates are not among
        them is skipped rather than guessed at, so the plan grows as the gates
        are decided.

    Returns
    -------
    list of (list of str, int or None)
        The blocks to join and the width to project them to, a width of None
        meaning imputed but not rotated. Deduplicated, in the order the stages
        first ask for them.
    """
    settled = gates.all_chosen(spec) if resolved is None else resolved

    # a gate counts as decided when every axis it chooses is in the resolved
    # set, which keeps the plan a function of its arguments rather than of what
    # results/gates happens to hold
    decided = {
        stage.gate.id
        for stage in spec.stages
        if stage.gate and all(axis in settled for axis in stage.gate.chooses)
    }
    axes: dict[str, Any] = spec.axes

    wanted: list[tuple[list[str], int | None]] = []
    for stage in spec.stages:
        # a stage whose gates have not been decided cannot say what it reads,
        # so it is left for the next run of this script rather than guessed at.
        # Nothing is lost: its reductions are built when its gate exists, and
        # the ones it shares with an earlier stage are already cached by then
        pending = [gate for gate in stage.depends_on if gate not in decided]
        if pending:
            logger.info("stage %s: waiting on %s", stage.id, ", ".join(pending))
            continue
        for config in spec.expand(stage.id, settled):
            wanted.extend(_reductions_for(config, axes))

    # a stage's configurations repeat a block's reduction across every other
    # axis, and stages overlap besides, so the union is taken by hand to keep
    # the order the stages ask in
    unique: list[tuple[list[str], int | None]] = []
    seen: set[tuple[tuple[str, ...], int | None]] = set()
    for names, width in wanted:
        key = (tuple(names), width)
        if key not in seen:
            seen.add(key)
            unique.append((names, width))

    if widths is not None:
        allowed_widths = set(widths)
        unique = [(n, w) for n, w in unique if w is None or w in allowed_widths]
    if blocks is not None:
        allowed = set(blocks)
        unique = [(n, w) for n, w in unique if allowed.issuperset(n)]
    return unique


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
