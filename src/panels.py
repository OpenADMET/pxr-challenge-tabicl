"""What each numbered figure shows, and where its statistics come from.

The figures are not the stages. A stage is a sweep with a question and a gate;
a figure is the order the result is told in, which runs from the graph networks
through the feature blocks to the regressors and ends on what the winning
model knows about its own errors. Two stages ran first and are reported late,
and one stage is read twice, once for its single blocks and once whole.

Each panel declares three things. Its family, which is the set of runs the
all-pairwise correction is computed over and therefore what the verdicts in it
mean. Its references, which are scored on the same compounds but left out of
that family, so a figure can carry a point of comparison without changing the
question it asks. And its identities, the configurations that keep one colour
wherever they appear, so a reader can follow the winner of one figure into the
next and read its neighbours.

Nothing here decides anything. The gates were decided in the manifest and
recorded from the runs; a panel reads those records and draws them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

import aggregate
import gates
import plots
from manifest import Manifest, Reference, TabularConfig

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("results/figures")

# the references the tabular figures carry, declared in the manifest and named
# here in the order they are drawn
CARRIED = ("best_gnn", "chemeleon_baseline", "best_single")

# the two panels drawn on one page, since they ask one question of two blocks
PAIRED = ("fig2a", "fig2b")

# how many compound resamples every panel's bootstrap draws
DEFAULT_RESAMPLES = 10000


class PanelError(ValueError):
    """A panel cannot be built from the runs on disk."""


@dataclass
class Panel:
    """One figure: what it shows, and what was measured to show it."""

    id: str
    title: str
    question: str
    evidence: dict[str, Any]
    references: list[dict[str, Any]] = field(default_factory=list)
    named: dict[str, str] = field(default_factory=dict)
    home: str | None = None

    def frame(self, winners: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
        """Turn what this panel measured into rows ready to draw."""
        return plots.comparison_frame(
            self.evidence,
            plots.gate_winners() if winners is None else winners,
            self.named,
            self.references,
            self.home,
        )


def gnn_panel(
    manifest: Manifest,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Panel:
    """Figure 1: where a fine-tuned message-passing network lands.

    The cells are enumerated rather than crossed and no gate reads them, so
    this family has no decision attached to it. Two of its rows are declared
    references and are named here, which is checked against what the runs say
    rather than read off them.
    """
    cells = list(manifest.gnn_cells)
    labels = {cell.slug: plots.gnn_label(cell.as_dict()) for cell in cells}
    evidence = gates.measure(
        cells, manifest, results_dir=results_dir, n_resamples=n_resamples, labels=labels
    )
    named = {
        resolve(manifest, name).slug: name
        for name in ("best_gnn", "chemeleon_baseline")
        if _declared(manifest, name)
    }
    _check_leader(manifest, evidence, "best_gnn")
    return Panel(
        id="fig1",
        title="Graph-network baselines",
        question="Where does a fine-tuned message-passing network land on the challenge split?",
        evidence=evidence,
        named=named,
        home="best_gnn",
    )


def width_panels(
    manifest: Manifest,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> tuple[Panel, Panel]:
    """Figure 2: the two width probes, read against each other.

    They ask one question of two different blocks, so they are drawn side by
    side rather than in sequence. Each keeps its own family, and so its own
    correction; only the axis is shared.
    """
    common = {"results_dir": results_dir, "gates_dir": gates_dir, "n_resamples": n_resamples}
    return (
        Panel(
            id="fig2a",
            title="Descriptor blocks and how far they reduce",
            question="Which descriptor block wins, and how much does reducing it cost?",
            evidence=gates.evidence(manifest, "descriptor_width", **common),
        ),
        Panel(
            id="fig2b",
            title="Embedding PCA width",
            question="How much does reducing the CheMeleon embedding cost?",
            evidence=gates.evidence(manifest, "embedding_width", **common),
        ),
    )


def ingredient_panels(
    manifest: Manifest,
    references: list[dict[str, Any]],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> tuple[Panel, Panel]:
    """Figures 3 and 4: the blocks alone, then every combination of them.

    The stage is read twice. Figure 3 asks how far each block gets on its own
    and is corrected over the single-block rows only, which is the family that
    question is about. Figure 4 is the whole stage, exactly the family the
    featureset gate was decided on, so the figure showing the decision and the
    record of it cannot disagree.
    """
    settled = gates.settled(manifest, "ingredients", gates_dir)
    configs = manifest.expand("ingredients", settled)
    singles = [config for config in configs if config.n_blocks == 1]
    if not singles:
        raise PanelError("the ingredients stage expands to no single-block configurations")

    alone = gates.measure(singles, manifest, results_dir=results_dir, n_resamples=n_resamples)
    whole = gates.evidence(
        manifest,
        "ingredients",
        results_dir=results_dir,
        gates_dir=gates_dir,
        n_resamples=n_resamples,
    )
    _check_leader(manifest, alone, "best_single", gates_dir=gates_dir)
    named = {resolve(manifest, "best_single", gates_dir=gates_dir).slug: "best_single"}
    return (
        Panel(
            id="fig3",
            title="Single-ingredient tabular features",
            question="How far does each feature block get on its own?",
            evidence=alone,
            references=references,
            named=named,
            home="best_single",
        ),
        Panel(
            id="fig4",
            title="Featureset combinations",
            question="Which combination of blocks wins?",
            evidence=whole,
            references=references,
            named=named,
        ),
    )


def regressor_panel(
    manifest: Manifest,
    references: list[dict[str, Any]],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Panel:
    """Figure 5: the regressors, on the one featureset the stage before settled.

    Every row here carries the same columns, so this figure cannot say whether
    the ranking would survive a different featureset.
    """
    return Panel(
        id="fig5",
        title="Regressor comparison",
        question="Which regressor wins on the featureset this split chose?",
        evidence=gates.evidence(
            manifest,
            "regressor",
            results_dir=results_dir,
            gates_dir=gates_dir,
            n_resamples=n_resamples,
        ),
        references=references,
    )


def resolve(manifest: Manifest, name: str, *, gates_dir: Path | None = None) -> TabularConfig | Any:
    """Return the configuration a declared reference names.

    Raises
    ------
    PanelError
        If nothing is declared under that name, or what it names was not run.
    """
    reference = _declared(manifest, name)
    if reference is None:
        raise PanelError(
            f"no reference {name!r} is declared; add it under references: in the "
            f"manifest, with a reason, or stop carrying it"
        )
    if reference.cell:
        for cell in manifest.gnn_cells:
            if cell.id == reference.cell:
                return cell
        raise PanelError(f"reference {name!r} names cell {reference.cell!r}, which is not run")

    settled = gates.settled(manifest, reference.stage, gates_dir)
    matching = [
        config
        for config in manifest.expand(reference.stage, settled)
        if all(str(getattr(config, axis)) == str(level) for axis, level in reference.config.items())
    ]
    if len(matching) != 1:
        raise PanelError(
            f"reference {name!r} matches {len(matching)} configurations of "
            f"{reference.stage!r}; it has to name exactly one"
        )
    return matching[0]


def carried(
    manifest: Manifest,
    names: tuple[str, ...] = CARRIED,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Score the declared references the tabular figures are read against.

    They are scored on the same compounds as everything else and shown wherever
    the story goes next, and stay out of those figures' families.
    """
    configs = [resolve(manifest, name, gates_dir=gates_dir) for name in names]
    # a reference stands outside any family here, so it is named by its own
    # blocks rather than by the axes some family varies
    labels = {
        config.slug: plots.label_for(config.as_dict(), plots.BLOCK_AXES)
        if isinstance(config, TabularConfig)
        else plots.gnn_label(config.as_dict())
        for config in configs
    }
    rows = gates.scores(configs, manifest, results_dir=results_dir, labels=labels)
    identities = {config.slug: name for config, name in zip(configs, names, strict=True)}
    return [row | {"identity": identities[row["slug"]]} for row in rows]


def anchor(manifest: Manifest) -> dict[str, Any]:
    """Return the published leaderboard score, as a row every figure carries.

    It is drawn without an interval, like any reference, and for a stronger
    reason than the others: the report publishes one number and no per-compound
    predictions, so there is nothing here to resample and no test against it is
    possible. It is also an ensemble of nine models where these rows are single
    models averaged over seeds, which is a different quantity. It marks where
    the leaderboard sat, and supports no claim of significance either way.
    """
    published = manifest.anchor
    return {
        "slug": "anchor",
        # the full name is a sentence and would set the width of every
        # figure's label column, so the row is named by who it was and what
        # kind of thing it is
        "label": f"{published['name'].split(',')[0]} leaderboard, ensemble",
        "mae": float(published["mae_ensemble"]),
        "seed_spread": float("nan"),
        "ensemble": {"mae": float(published["mae_ensemble"])},
        "identity": "anchor",
    }


def uncertainty_figure(slug: str, *, results_dir: Path = aggregate.RESULTS_DIR) -> go.Figure:
    """Figure 6: what the winning model knows about its own errors.

    Read from the uncertainty stage's own artifacts rather than from the run
    table, since the quantities are per compound and per nominal level rather
    than one score per configuration.
    """
    directory = Path(results_dir) / "uncertainty" / slug
    points = directory / "points.csv"
    coverage = directory / "coverage.csv"
    if not points.exists() or not coverage.exists():
        raise PanelError(f"no uncertainty artifacts under {directory}; run the uncertainty stage")
    return plots.uncertainty_figure(pd.read_csv(points), pd.read_csv(coverage))


def build(
    manifest: Manifest,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> list[Panel]:
    """Measure every comparison panel, in the order the story is told.

    Every panel carries the leaderboard anchor, so each figure says where it
    stands against the one absolute number this project has. The tabular
    figures also carry the graph networks, and the regressor figure carries the
    best single block, which is in the other two figures' families already.

    Figure 6 is not here: it is not a comparison of configurations and is drawn
    from the uncertainty artifacts by :func:`uncertainty_figure`.
    """
    common = {"results_dir": results_dir, "gates_dir": gates_dir, "n_resamples": n_resamples}
    published = anchor(manifest)
    graphs = carried(
        manifest,
        ("best_gnn", "chemeleon_baseline"),
        results_dir=results_dir,
        gates_dir=gates_dir,
    )
    single = carried(manifest, ("best_single",), results_dir=results_dir, gates_dir=gates_dir)

    gnn = gnn_panel(manifest, results_dir=results_dir, n_resamples=n_resamples)
    gnn.references = [published]

    left, right = width_panels(manifest, **common)
    left.references = [published]
    right.references = [published]

    singles, combinations = ingredient_panels(manifest, [published, *graphs], **common)
    regressors = regressor_panel(manifest, [published, *graphs, *single], **common)
    return [gnn, left, right, singles, combinations, regressors]


def render(
    panels: list[Panel],
    out_dir: Path = FIGURES_DIR,
    *,
    winners: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write each panel to its own page, and figure 2 as one page of two.

    Returns
    -------
    dict
        Figure id mapped to the file written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    winners = plots.gate_winners() if winners is None else winners
    by_id = {panel.id: panel for panel in panels}

    written: dict[str, Path] = {}
    for panel in panels:
        if panel.id in PAIRED:
            continue
        figure = plots.comparison_figure(panel.frame(winners))
        written[panel.id] = _write(figure, out_dir / f"{panel.id}.html")

    pair = [by_id[name] for name in PAIRED if name in by_id]
    if len(pair) == len(PAIRED):
        figure = plots.comparison_figure(
            *[panel.frame(winners) for panel in pair],
            subtitles=tuple(panel.title for panel in pair),
        )
        written["fig2"] = _write(figure, out_dir / "fig2.html")
    return written


def _declared(manifest: Manifest, name: str) -> Reference | None:
    """Return the manifest's declaration of a reference, if it has one."""
    for reference in manifest.references:
        if reference.id == name:
            return reference
    return None


def _check_leader(
    manifest: Manifest,
    evidence: dict[str, Any],
    name: str,
    *,
    gates_dir: Path | None = None,
) -> None:
    """Warn when a declared reference is no longer its family's leader.

    The declaration stands: it was a judgement, and the figures keep drawing
    what it names. But a reference declared as the best of something that the
    runs now separate from their leader is worth seeing rather than silently
    honouring, in the same way a gate's declared choice is.
    """
    config = resolve(manifest, name, gates_dir=gates_dir)
    if config.slug == evidence["leader_slug"]:
        return
    separated = config.slug not in evidence["indistinguishable_from_leader"]
    logger.warning(
        "reference %s names %s, which is no longer the leader of its family (%s); it is %s",
        name,
        config.slug,
        evidence["leader_slug"],
        "separated from it" if separated else "not separated from it",
    )


def _write(figure: go.Figure, path: Path) -> Path:
    """Write one figure as a standalone page."""
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)
    logger.info("wrote %s", path)
    return path
