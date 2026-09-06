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

# an epoch budget: a body held this long is never released
FROZEN_AT = 30

# how many compound resamples every panel's bootstrap draws
DEFAULT_RESAMPLES = 10000


# where the feature blocks are written, read for their true column counts
FEATURES_DIR = Path("data/features")

# the readout blocks are two predicted columns and are never reduced, so their
# width is a property of what they are rather than of any choice
NATIVE = -1


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


def annotations(
    configs: list[Any],
    dims: dict[tuple[str, str], int],
    *,
    axes: tuple[str, ...] = (),
    widths: tuple[str, ...] = ("descriptors",),
    freeze: bool = False,
    width_separator: str = " ",
    titles: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Name and describe each configuration, the way one figure wants it.

    A label carries what varies in the figure it is in and nothing else, so the
    same configuration is named differently in the figure asking what reducing
    costs than in the one asking which blocks to join. Everything left out of
    the label goes into the detail, which is read on hover.

    A reference carried into a figure takes its declared title instead. Away
    from the family that established it, what matters about it is what it is,
    and a row labelled by its blocks among rows that differ in exactly those
    blocks reads as one of them.
    """
    named: dict[str, dict[str, str]] = {}
    for config in configs:
        flat = config.as_dict()
        if isinstance(config, TabularConfig):
            label = plots.label_for(
                flat,
                axes or plots.BLOCK_AXES,
                dims=dims,
                widths=widths,
                width_separator=width_separator,
            )
            detail = tabular_detail(flat, dims)
        else:
            label, detail = plots.gnn_label(flat, freeze=freeze), gnn_detail(flat, dims)
        title = (titles or {}).get(config.slug)
        named[config.slug] = {
            "label": title or label,
            # the configuration is still worth reading, so a titled row says
            # what it is made of on hover instead of in the column
            "detail": f"<b>{label}</b><br>{detail}" if title else detail,
        }
    return named


def best_over_freeze(manifest: Manifest, *, results_dir: Path = aggregate.RESULTS_DIR) -> list[Any]:
    """Return the graph-network cells, reporting the best over the freeze axis.

    How many epochs the body is held before it is released is a training
    setting rather than a question figure 1 asks, in the same class as the
    gradient clip. Six of the cells are that axis crossed with the head width,
    and showing all six spends most of the figure on it. So the axis is
    maximised over: cells that differ in nothing else are collapsed to their
    best, and the label stops naming a level that no longer varies.

    Never releasing the body is a different thing and is not collapsed, since
    it is a question about whether the body needs to adapt at all.

    Taking a maximum over three flatters the cells it applies to, by about what
    picking the best of three seeds would. The comparison it feeds is against
    the tabular models, which the graph networks lose, so the bias runs against
    the conclusion rather than towards it.
    """
    rows = gates.scores(list(manifest.gnn_cells), manifest, results_dir=results_dir)
    scored = {row["slug"]: row["mae"] for row in rows}
    groups: dict[tuple, Any] = {}
    for cell in manifest.gnn_cells:
        flat = cell.as_dict()
        if int(flat["freeze_epochs"]) >= FROZEN_AT:
            groups[(cell.id,)] = cell
            continue
        key = tuple(
            sorted((axis, str(value)) for axis, value in flat.items() if axis != "freeze_epochs")
        )
        if key not in groups or scored[cell.slug] < scored[groups[key].slug]:
            groups[key] = cell
    kept = [cell for cell in manifest.gnn_cells if cell in groups.values()]
    dropped = len(list(manifest.gnn_cells)) - len(kept)
    if dropped:
        logger.info("figure 1 reports the best over the freeze axis, collapsing %d cells", dropped)
    return kept


def gnn_panel(
    manifest: Manifest,
    dims: dict[tuple[str, str], int],
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
    cells = best_over_freeze(manifest, results_dir=results_dir)
    evidence = gates.measure(
        cells,
        manifest,
        results_dir=results_dir,
        n_resamples=n_resamples,
        annotations=annotations(cells, dims, titles=_titles(manifest, "best_gnn")),
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
    dims: dict[tuple[str, str], int],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> tuple[Panel, Panel]:
    """Figure 2: the two width probes, read against each other.

    They ask one question of two different blocks, so they are drawn side by
    side rather than in sequence. Each keeps its own family, and so its own
    correction; only the axis is shared. These are the two figures where a
    width belongs in a label, since the width is what they are about.
    """
    common = {"results_dir": results_dir, "gates_dir": gates_dir, "n_resamples": n_resamples}
    panels = []
    # the descriptor panel varies which blocks as well as how wide, so its
    # names read as one phrase; the embedding panel varies only the width,
    # which then sits under a block name repeated down the column
    for stage_id, axes, separator, title, question in (
        (
            "descriptor_width",
            ("descriptors", "descriptor_pca"),
            " ",
            "Descriptor blocks and how far they reduce",
            "Which descriptor block wins, and how much does reducing it cost?",
        ),
        (
            "embedding_width",
            ("embedding", "embedding_pca"),
            "<br>",
            "Embedding PCA width",
            "How much does reducing the CheMeleon embedding cost?",
        ),
    ):
        evidence = gates.evidence(manifest, stage_id, **common)
        configs = manifest.expand(stage_id, gates.settled(manifest, stage_id, gates_dir))
        _annotate(
            evidence,
            annotations(
                configs,
                dims,
                axes=axes,
                widths=(axes[0],),
                width_separator=separator,
            ),
        )
        panels.append(
            Panel(id=f"fig2{'ab'[len(panels)]}", title=title, question=question, evidence=evidence)
        )
    return panels[0], panels[1]


def ingredient_panels(
    manifest: Manifest,
    dims: dict[tuple[str, str], int],
    carried_configs: list[Any],
    references: list[dict[str, Any]],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> tuple[Panel, Panel]:
    """Figures 3 and 4: the blocks alone, then every combination of them.

    The stage is read twice. Figure 3 asks how far each block gets on its own
    and is corrected over the single-block rows, which is the family that
    question is about. Figure 4 is the whole stage.

    The carried rows are in both families rather than beside them. A reference
    with no interval cannot be told apart from one that is merely tied, and
    what a reader wants to know about the graph networks is exactly whether
    the tabular models separate from them.
    """
    settled = gates.settled(manifest, "ingredients", gates_dir)
    configs = manifest.expand("ingredients", settled)
    singles = [config for config in configs if config.n_blocks == 1]
    if not singles:
        raise PanelError("the ingredients stage expands to no single-block configurations")

    # the embedding's width was settled in figure 2 and says nothing here; the
    # descriptors keep theirs, since 217 columns reduced to 128 is what that
    # block is
    label = {"axes": ("embedding", "readout", "descriptors")}
    common = {"results_dir": results_dir, "n_resamples": n_resamples}
    alone = gates.measure(
        [*singles, *carried_configs],
        manifest,
        annotations=annotations(
            [*singles, *carried_configs],
            dims,
            titles=_titles(manifest, "best_single", gates_dir),
            **label,
        ),
        **common,
    )
    whole = gates.measure(
        [*configs, *carried_configs],
        manifest,
        annotations=annotations(
            [*configs, *carried_configs], dims, titles=_titles(manifest, None, gates_dir), **label
        ),
        **common,
    )
    whole["gate"] = manifest.stage("ingredients").gate.id

    _check_leader(manifest, alone, "best_single", gates_dir=gates_dir)
    named = _identities(manifest, gates_dir)
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
    dims: dict[tuple[str, str], int],
    carried_configs: list[Any],
    references: list[dict[str, Any]],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Panel:
    """Figure 5: the regressors, on the one featureset the stage before settled.

    Every row of the family here carries the same columns, so this figure
    cannot say whether the ranking would survive a different featureset. The
    carried rows do not share that featureset and are named by their own blocks.
    """
    configs = manifest.expand("regressor", gates.settled(manifest, "regressor", gates_dir))
    named_rows = annotations(configs, dims, axes=("regressor",)) | annotations(
        carried_configs, dims, titles=_titles(manifest, None, gates_dir)
    )
    evidence = gates.measure(
        [*configs, *carried_configs],
        manifest,
        results_dir=results_dir,
        n_resamples=n_resamples,
        annotations=named_rows,
    )
    evidence["gate"] = manifest.stage("regressor").gate.id
    return Panel(
        id="fig5",
        title="Regressor comparison",
        question="Which regressor wins on the featureset this split chose?",
        evidence=evidence,
        references=references,
        named=_identities(manifest, gates_dir),
    )


def _annotate(evidence: dict[str, Any], named: dict[str, dict[str, str]]) -> None:
    """Add each row's name and detail to a measured family, in place."""
    for row in evidence["ranking"]:
        row |= named.get(row["slug"], {})


def _titles(manifest: Manifest, home: str | None, gates_dir: Path | None = None) -> dict[str, str]:
    """Map each carried reference's slug to the title it is drawn under.

    The figure that established a reference names it by its configuration,
    since that is what that figure is asking about. Everywhere else it is a
    landmark and is named as one.
    """
    return {
        resolve(manifest, reference.id, gates_dir=gates_dir).slug: reference.title
        for reference in manifest.references
        if reference.id != home
    }


def _identities(manifest: Manifest, gates_dir: Path | None = None) -> dict[str, str]:
    """Map each identity a panel shows to the name it keeps.

    The declared references, and every gate's winner. These are named here
    rather than matched on axes because a family carrying rows of other kinds
    has no single set of axes to match against: the reference rows differ from
    the stage's rows on axes the stage holds fixed, which is exactly what the
    matching rule treats as disqualifying.
    """
    identities = {
        resolve(manifest, name, gates_dir=gates_dir).slug: name
        for name in CARRIED
        if _declared(manifest, name)
    }

    # every gate's winner, named by the slug the stage that decided it ran. A
    # later stage holds those axes fixed, so every one of its rows agrees with
    # the decision and matching on axes would claim them all; the winner is the
    # one row that is also the configuration the earlier stage scored
    for stage in manifest.stages:
        if stage.gate is None or not stage.gate.decision:
            continue
        chosen = stage.gate.decision
        for config in manifest.expand(stage.id, gates.settled(manifest, stage.id, gates_dir)):
            if all(str(getattr(config, axis)) == str(level) for axis, level in chosen.items()):
                identities[config.slug] = stage.gate.id
                break
    return identities


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
        # figure's label column, so the row is named by who it was and the
        # detail says what it is
        "label": published["name"].split(",")[0],
        "detail": (
            f"{published['name']}<br>{published['n']} compounds, nine-model ensemble"
            f"<br>no per-compound predictions published, so it cannot be tested against"
        ),
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

    Every panel carries the leaderboard anchor as a reference, since it is the
    one absolute number this project has and the one thing no figure can test
    against. The tabular figures carry the graph networks and the best single
    block inside their families instead, so those rows are tested where they
    are drawn.

    Figure 6 is not here: it is not a comparison of configurations and is drawn
    from the uncertainty artifacts by :func:`uncertainty_figure`.
    """
    dims = block_dims(manifest)
    published = [anchor(manifest)]
    common = {"results_dir": results_dir, "gates_dir": gates_dir, "n_resamples": n_resamples}

    gnn = gnn_panel(manifest, dims, results_dir=results_dir, n_resamples=n_resamples)
    gnn.references = published

    left, right = width_panels(manifest, dims, **common)
    left.references = published
    right.references = published

    graphs = [resolve(manifest, name, gates_dir=gates_dir) for name in CARRIED[:2]]
    single = resolve(manifest, "best_single", gates_dir=gates_dir)
    singles, combinations = ingredient_panels(manifest, dims, graphs, published, **common)
    regressors = regressor_panel(manifest, dims, [*graphs, single], published, **common)
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
        # no titles over the panels: what each asks is a caption's job, and a
        # title over a column of block names reads as one of them
        figure = plots.comparison_figure(*[panel.frame(winners) for panel in pair])
        written["fig2"] = _write(figure, out_dir / "fig2.html")
    return written


def block_dims(manifest: Manifest, features_dir: Path = FEATURES_DIR) -> dict[tuple[str, str], int]:
    """Return each block level's true column count, read off the features.

    The manifest declares a count for the blocks it reduces, and a figure wants
    one for every block it names: what a native block actually is, and what a
    reduction reduced from. They are read from the written features rather than
    from the declaration, so a label cannot claim a width the data does not
    have. The declared counts are checked against these by a test.

    Returns
    -------
    dict
        ``(axis, level)`` mapped to the block's own column count. A level whose
        features have not been built is absent rather than guessed at.
    """
    dims: dict[tuple[str, str], int] = {}
    for axis in ("embedding", "readout", "descriptors"):
        levels = manifest.axes.get(axis) or {}
        for level, spec in levels.items():
            if not spec:
                continue
            blocks = spec.get("blocks") or ([spec["block"]] if spec.get("block") else [level])
            widths = [_columns(features_dir / block) for block in blocks]
            if all(width is not None for width in widths):
                dims[(axis, str(level))] = sum(width for width in widths if width is not None)
    return dims


def _columns(directory: Path) -> int | None:
    """Return how many feature columns a block holds, or None if unbuilt."""
    written = sorted(directory.glob("*.parquet"))
    if not written:
        return None
    import pyarrow.parquet as pq

    # one column is the compound key the block is joined on
    return len(pq.ParquetFile(written[0]).schema.names) - 1


def tabular_detail(config: dict[str, Any], dims: dict[tuple[str, str], int]) -> str:
    """Describe one tabular configuration: its blocks, their widths, its total.

    The labels carry only what varies in a figure, so everything a reader would
    otherwise look up in a table goes here: what each block is, what it was
    reduced from and to, and how many columns reach the regressor in the end.

    A row of one block is named by it already, so the block is not repeated
    here; a row joining several needs saying which line belongs to which.
    """
    present = sum(
        1 for axis in ("embedding", "readout", "descriptors") if config.get(axis, "none") != "none"
    )
    lines, total, known = [], 0, True
    for axis, width_axis in (
        ("embedding", "embedding_pca"),
        ("readout", None),
        ("descriptors", "descriptor_pca"),
    ):
        level = str(config.get(axis, "none"))
        if level == "none":
            continue
        native = dims.get((axis, level))
        width = int(config.get(width_axis, 0)) if width_axis else 0
        name = plots.PRETTY.get(level, level)
        prefix = f"{name} {axis}: " if present > 1 else ""
        if width > 0:
            lines.append(f"{prefix}PCA {native or '?'} \u2192 {width}")
            total += width
        else:
            lines.append(f"{prefix}{native or '?'}")
            total += native or 0
            known = known and native is not None
    if config.get("regressor"):
        lines.append(f"regressor: {plots.PRETTY.get(config['regressor'], config['regressor'])}")
    if known:
        lines.append(f"ndims into the regressor: {total}")
    return "<br>".join(lines)


def gnn_detail(config: dict[str, Any], dims: dict[tuple[str, str], int]) -> str:
    """Describe one graph-network cell: its body, its head, its auxiliary arm.

    Every cell is a Chemprop D-MPNN with a feed-forward predictor on top, and
    those are two different sizes that one integer in a label cannot keep
    apart. What differs between the cells is where the body's weights came
    from and how wide that body therefore is: the CheMeleon checkpoint brings
    its own, and the log2FC body is the one this project trained from scratch.
    The head width is a figure-1 axis and is the same choice whichever body it
    sits on.
    """
    frozen = int(config["freeze_epochs"]) >= FROZEN_AT
    started = plots.BODY.get(str(config["encoder_init"]), str(config["encoder_init"]))
    finetune = str(config["finetune_target"])

    # the body's own width, read off the block that same network writes rather
    # than restated here
    level = "chemeleon" if str(config["encoder_init"]) == "chemeleon" else "chemprop_log2fc"
    width = dims.get(("embedding", level))
    lines = [
        "model: Chemprop D-MPNN with a feed-forward predictor",
        f"MPNN body from: {started}" + (f", width {width}" if width else ""),
        f"MPNN body: {'frozen' if frozen else 'free'}",
        f"predictor head: {config['ffn_hidden_dim']} wide",
        f"fine-tuned on: {plots.PRETTY.get(finetune, finetune)}",
    ]
    if str(config.get("aux_target", "none")) != "none":
        halves = []
        if config.get("aux_embedding"):
            halves.append(f"embedding ({dims.get(('embedding', 'chemeleon_log2fc'), '?')})")
        if config.get("aux_readout"):
            halves.append(f"readout ({dims.get(('readout', 'chemeleon_log2fc'), '?')})")
        target = plots.PRETTY.get(str(config["aux_target"]), str(config["aux_target"]))
        lines.append(f"auxiliary {target}: {' + '.join(halves)}")
    else:
        lines.append("auxiliary encoder: none")
    return "<br>".join(lines)


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
