"""What each numbered figure shows, and where its statistics come from.

The figures are not the stages. A stage is a sweep with a question and a gate;
a figure is the order the result is told in, which runs from the graph networks
through the feature blocks to the regressors, then to what the winning model
knows about its own errors and how much of it is ensembling. One stage is read
twice, once for its single blocks and once whole.

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

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import aggregate
import evaluate
import gates
import plots
import regressors
import tukey
import uncertainty
from data import CANONICAL_COL
from manifest import Manifest, Reference, TabularConfig

logger = logging.getLogger(__name__)

# hovering a row's axis label opens that row's tooltip, which plotly has no
# native event for; appended to every page written
ROW_HOVER = Path(__file__).with_name("row_hover.js")

# the gap between two panels is a fraction of the figure and the labels drawn
# in it are pixels wide, so the gap is settled in the page rather than here
PANEL_GAP = Path(__file__).with_name("panel_gap.js")

FIGURES_DIR = Path("results/figures")

# the references the tabular figures carry, declared in the manifest and named
# here in the order they are drawn
CARRIED = ("best_gnn", "chemeleon_baseline", "best_single")

# the two panels drawn on one page, since they ask one question of two blocks
PAIRED = ("fig2a", "fig2b")

# an epoch budget: a body held this long is never released
FROZEN_AT = 30

# Where each identity was established, and how a later figure refers back to
# it. Colour says a reader has seen a row before; this says where, which is the
# whole of what carrying a colour forward is for. Both are suppressed in the
# figure that established the identity, where the row is one of the things
# being compared rather than a landmark brought in. The anchor is absent on
# purpose: it was established in the report rather than in any figure, and its
# tooltip says so at length.
ESTABLISHED_IN = {
    "best_gnn": ("fig1", "established in figure 1"),
    "chemeleon_baseline": ("fig1", "the baseline, from figure 1"),
    "canonical_descriptors": ("fig2", "chosen in figure 2"),
    "embedding_reduction": ("fig2", "chosen in figure 2"),
    "best_single": ("fig3", "established in figure 3"),
    "best_featureset": ("fig4", "chosen in figure 4"),
    "best_regressor": ("fig5", "chosen in figure 5"),
}


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
    # family members this figure did not sweep, which are drawn as carried
    carried: set[str] = field(default_factory=set)
    # identities settled elsewhere that this figure does not carry forward
    hidden: tuple[str, ...] = ()
    # the page this panel is drawn on, which is not its id for the two that
    # share one. An identity established here is not carried in
    figure: str = ""

    def frame(self, winners: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
        """Turn what this panel measured into rows ready to draw."""
        resolved = plots.gate_winners() if winners is None else winners
        return plots.comparison_frame(
            self.evidence,
            {gate: axes for gate, axes in resolved.items() if gate not in self.hidden},
            {slug: name for slug, name in self.named.items() if name not in self.hidden},
            self.references,
            self.home,
            self.carried,
            {
                name: phrase
                for name, (where, phrase) in ESTABLISHED_IN.items()
                if where != (self.figure or self.id)
            },
        )


def annotations(
    configs: list[Any],
    dims: dict[tuple[str, str], int],
    manifest: Manifest,
    *,
    axes: tuple[str, ...] = (),
    widths: tuple[str, ...] = ("descriptors",),
    freeze: bool = False,
    width_separator: str = " ",
    titles: dict[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Name and describe each configuration, the way one figure wants it.

    A label carries what varies in the figure it is in and nothing else, so the
    same configuration is named differently in the figure asking what reducing
    costs than in the one asking which blocks to join. Everything left out of
    the label goes into the detail, which is read on hover.

    A reference carried into a figure takes its declared title instead. Away
    from the family that established it, what matters about it is what it is,
    and a row labelled by its blocks among rows that differ in exactly those
    blocks reads as one of them. Its configuration goes into the line that
    says where a reader met it, which is the line that has to name it anyway.
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
            detail = tabular_detail(flat, dims, manifest)
        else:
            label, detail = plots.gnn_label(flat, freeze=freeze), gnn_detail(flat, dims)
        title, page = (titles or {}).get(config.slug, ("", ""))
        named[config.slug] = {"label": title or label, "detail": detail}
        if title:
            # the row is drawn under its title, so the line saying where a
            # reader met it carries the configuration too: one line naming what
            # it is and where it came from, rather than a second bold heading
            # competing with the row's own
            named[config.slug]["origin"] = (
                f"{plots.spelled(label)}, from {page.replace('fig', 'figure ')}"
            )
    return named


def held(
    manifest: Manifest, figure_id: str = "fig1", *, frozen_at: int = FROZEN_AT
) -> tuple[list[Any], dict[str, Any]]:
    """Return the graph-network cells a figure shows, and what it holds fixed.

    A figure holds a setting rather than showing it when the setting was swept
    but is not what the figure asks. The levels are declared in the manifest
    and read here, never chosen from the runs: which warmup to standardise on
    is a judgement, and one the numbers alone would decide differently from the
    way the rest of the table needs it decided.

    A body held at or above the epoch budget is never released, which is a
    different question from how long a warmup runs even though the two share a
    field. Those cells are kept whatever the warmup is held at.

    Returns
    -------
    cells : list
        The cells that carry every held level.
    hold : dict
        Axis mapped to the level held, for the check and for the record.

    Raises
    ------
    PanelError
        If holding the declared levels leaves nothing to draw.
    """
    hold = dict(declaration(manifest, figure_id).get("hold") or {})
    kept = []
    for cell in manifest.gnn_cells:
        axes = cell.as_dict()
        frozen = int(axes["freeze_epochs"]) >= frozen_at
        if all(
            frozen and axis == "freeze_epochs" or str(axes.get(axis)) == str(level)
            for axis, level in hold.items()
        ):
            kept.append(cell)
    if not kept:
        raise PanelError(f"{figure_id}: holding {hold} leaves no graph-network cell to draw")
    dropped = len(list(manifest.gnn_cells)) - len(kept)
    if dropped:
        logger.info("%s holds %s, leaving out %d cells", figure_id, hold, dropped)
    return kept, hold


def marginal(
    manifest: Manifest,
    axis: str,
    hold: dict[str, Any],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    frozen_at: int = FROZEN_AT,
) -> dict[str, float]:
    """Return each level of an axis, averaged over the architectures that carry it.

    The number a level is pinned on. It is a mean of per-cell seed means with
    one weight per architecture, not a mean over runs: the cells differ by a
    factor of five in seed spread, and pooling runs would let the noisiest of
    them decide a training setting.

    Only architectures carrying every level counted are included, so the levels
    are compared over the same set. A level one architecture happens to have
    and another does not says nothing about the level.

    Returns
    -------
    dict
        Level mapped to its mean, empty where no architecture carries more than
        one level of the axis.
    """
    groups = _families(manifest, axis, hold, frozen_at=frozen_at)
    if not groups:
        return {}

    levels = set.intersection(*({str(cell.as_dict()[axis]) for cell in g} for g in groups))
    if len(levels) < 2:
        return {}

    scored = {
        row["slug"]: row["mae"]
        for row in gates.scores(
            [cell for group in groups for cell in group], manifest, results_dir=results_dir
        )
    }
    means: dict[str, float] = {}
    for level in sorted(levels):
        at = [
            scored[cell.slug]
            for group in groups
            for cell in group
            if str(cell.as_dict()[axis]) == level
        ]
        means[level] = float(np.mean(at))
    return means


def _families(
    manifest: Manifest, axis: str, hold: dict[str, Any], *, frozen_at: int = FROZEN_AT
) -> list[list[Any]]:
    """Group the cells that differ in nothing but one axis.

    The only place a level of that axis can be judged. Cells that merely share
    the other held levels differ in what the figure is about, so the best of
    those says nothing about the setting.
    """
    candidates = [
        cell
        for cell in manifest.gnn_cells
        if int(cell.as_dict()["freeze_epochs"]) < frozen_at
        and all(
            str(cell.as_dict().get(name)) == str(value)
            for name, value in hold.items()
            if name != axis
        )
    ]
    groups: dict[tuple, list[Any]] = {}
    for cell in candidates:
        rest = tuple(sorted((k, str(v)) for k, v in cell.as_dict().items() if k != axis))
        groups.setdefault(rest, []).append(cell)
    return [group for group in groups.values() if len({str(c.as_dict()[axis]) for c in group}) > 1]


def check_held(
    manifest: Manifest,
    hold: dict[str, Any],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    n_resamples: int = DEFAULT_RESAMPLES,
    frozen_at: int = FROZEN_AT,
) -> None:
    """Warn when a level held is one the runs separate from the best of its axis.

    Two comparisons are logged and they are not the same one. The marginal
    averages a level over every architecture carrying it, which is the axis
    taken as a whole. The rest are per family, a family being the cells that
    differ in nothing but this axis, and a level can lead the margin while
    losing inside an individual family. Each message names the family it is
    about, since the two disagreeing is ordinary rather than a contradiction.

    Held is not the same as best, and does not have to be: standardising on a
    level the whole table shares is worth more than a difference the test
    cannot see. It is worth seeing when the level held is one the runs can tell
    apart from the best, though, which is a judgement for whoever reads the log
    rather than something to override here.

    The comparison is made among cells that differ in nothing but the axis
    under test, which is the only place a level of it can be judged. Cells that
    merely share the other held levels differ in what the figure is about, so
    the best of those says nothing about the setting.
    """
    for axis, level in hold.items():
        means = marginal(manifest, axis, hold, results_dir=results_dir, frozen_at=frozen_at)
        if means:
            best = min(means, key=lambda name: means[name])
            logger.info(
                "figure 1: %s averaged over the architectures carrying it, %s; best %s, held %s",
                axis,
                ", ".join(f"{name} {value:.4f}" for name, value in means.items()),
                best,
                level,
            )

        for family in _families(manifest, axis, hold, frozen_at=frozen_at):
            if len({str(cell.as_dict()[axis]) for cell in family}) < 2:
                continue
            measured = gates.measure(
                family, manifest, results_dir=results_dir, n_resamples=n_resamples
            )
            chosen = next(
                (cell for cell in family if str(cell.as_dict()[axis]) == str(level)), None
            )
            if chosen is None or chosen.slug == measured["leader_slug"]:
                continue
            if chosen.slug in measured["indistinguishable_from_leader"]:
                logger.info(
                    "figure 1 holds %s at %s, which %s does not prefer (%s leads that family); "
                    "the runs do not separate the two",
                    axis,
                    level,
                    chosen.slug,
                    measured["leader_slug"],
                )
                continue
            logger.warning(
                "figure 1 holds %s at %s, which the runs separate from %s within the family %s "
                "belongs to; the level held may no longer be defensible",
                axis,
                level,
                measured["leader_slug"],
                chosen.slug,
            )


def gnn_panel(
    manifest: Manifest,
    dims: dict[tuple[str, str], int],
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Panel:
    """Figure 1: where a fine-tuned message-passing network lands.

    The cells are enumerated rather than crossed and no gate reads them, so
    this family has no decision attached to it. Two settings are held rather
    than shown, and two of the rows are declared references; all four are
    declared in the manifest and checked against the runs here rather than read
    off them.
    """
    cells, hold = held(manifest)
    check_held(manifest, hold, results_dir=results_dir, n_resamples=n_resamples)
    evidence = gates.measure(
        cells,
        manifest,
        results_dir=results_dir,
        n_resamples=n_resamples,
        annotations=annotations(cells, dims, manifest, titles=_titles(manifest, "fig1")),
    )
    named = {
        resolve(manifest, name).slug: name
        for name in ("best_gnn", "chemeleon_baseline")
        if _declared(manifest, name)
    }
    _check_leader(manifest, evidence, "best_gnn")
    said = declaration(manifest, "fig1")
    return Panel(
        id="fig1",
        title=said["title"],
        question=said["question"],
        evidence=evidence,
        named=named,
        home="best_gnn",
        figure="fig1",
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
    said = declaration(manifest, "fig2")
    for stage_id, axes, separator in (
        ("descriptor_width", ("descriptors", "descriptor_pca"), " "),
        ("embedding_width", ("embedding", "embedding_pca"), "<br>"),
    ):
        evidence = gates.evidence(manifest, stage_id, **common)
        configs = manifest.expand(stage_id, gates.settled(manifest, stage_id, gates_dir))
        _annotate(
            evidence,
            annotations(
                configs,
                dims,
                manifest,
                axes=axes,
                widths=(axes[0],),
                width_separator=separator,
            ),
        )
        panels.append(
            Panel(
                id=f"fig2{'ab'[len(panels)]}",
                title=said["title"],
                question=said["question"],
                evidence=evidence,
                figure="fig2",
            )
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
            manifest,
            titles=_titles(manifest, "fig3", gates_dir),
            **label,
        ),
        **common,
    )
    whole = gates.measure(
        [*configs, *carried_configs],
        manifest,
        annotations=annotations(
            [*configs, *carried_configs],
            dims,
            manifest,
            titles=_titles(manifest, "fig4", gates_dir),
            **label,
        ),
        **common,
    )
    whole["gate"] = manifest.stage("ingredients").gate.id

    _check_leader(manifest, alone, "best_single", gates_dir=gates_dir)
    named = _identities(manifest, gates_dir)
    carried_slugs = {config.slug for config in carried_configs}
    return (
        Panel(
            id="fig3",
            title=declaration(manifest, "fig3")["title"],
            question=declaration(manifest, "fig3")["question"],
            evidence=alone,
            references=references,
            named=named,
            home="best_single",
            carried=carried_slugs,
            figure="fig3",
        ),
        Panel(
            id="fig4",
            title=declaration(manifest, "fig4")["title"],
            question=declaration(manifest, "fig4")["question"],
            evidence=whole,
            references=references,
            named=named,
            carried=carried_slugs,
            figure="fig4",
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
    named_rows = annotations(configs, dims, manifest, axes=("regressor",)) | annotations(
        carried_configs, dims, manifest, titles=_titles(manifest, "fig5", gates_dir)
    )
    evidence = gates.measure(
        [*configs, *carried_configs],
        manifest,
        results_dir=results_dir,
        n_resamples=n_resamples,
        annotations=named_rows,
    )
    evidence["gate"] = manifest.stage("regressor").gate.id
    said = declaration(manifest, "fig5")
    return Panel(
        id="fig5",
        title=said["title"],
        question=said["question"],
        evidence=evidence,
        references=references,
        named=_identities(manifest, gates_dir),
        carried={config.slug for config in carried_configs},
        figure="fig5",
    )


def _annotate(evidence: dict[str, Any], named: dict[str, dict[str, str]]) -> None:
    """Add each row's name and detail to a measured family, in place."""
    for row in evidence["ranking"]:
        row |= named.get(row["slug"], {})


def _titles(
    manifest: Manifest, figure: str, gates_dir: Path | None = None
) -> dict[str, tuple[str, str]]:
    """Map each carried reference's slug to the title and the page it came from.

    The figure that established a reference names it by its configuration,
    since that is what that figure is asking about. Everywhere else it is a
    landmark, drawn under its title with the configuration moved into the
    line that says where a reader met it.
    """
    return {
        resolve(manifest, reference.id, gates_dir=gates_dir).slug: (
            reference.title,
            ESTABLISHED_IN[reference.id][0],
        )
        for reference in manifest.references
        if ESTABLISHED_IN.get(reference.id, ("", ""))[0] != figure
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
        # what it is; that it cannot be tested is the verdict line's job
        "detail": f"{published['name']}<br>{published['n']} compounds, nine-model ensemble",
        "mae": float(published["mae_ensemble"]),
        "seed_spread": float("nan"),
        "ensemble": {"mae": float(published["mae_ensemble"])},
        "identity": "anchor",
        "verdict": "no per-compound predictions, so not tested",
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
    block_by_seed: bool | None = True,
) -> list[Panel]:
    """Measure every comparison panel, in the order the story is told.

    Every panel carries the leaderboard anchor as a reference, since it is the
    one absolute number this project has and the one thing no figure can test
    against. The tabular figures carry the graph networks and the best single
    block inside their families instead, so those rows are tested where they
    are drawn.

    Figure 6 is not here: it is not a comparison of configurations and is drawn
    from the uncertainty artifacts by :func:`uncertainty_figure`.

    What the panels are tested by is not what the gates decide by. Every panel
    is measured with the paired compound bootstrap first, because that is what
    a gate records and what the costs and the ranking are read off, and then
    its verdicts are re-taken under Tukey HSD, which is the procedure the
    comparison geometry belongs to. The gate records on disk are untouched. A
    caption has to say which of the two a figure is showing, since they answer
    different questions and disagree on some rows.

    Parameters
    ----------
    block_by_seed : bool or None, optional
        Whether Tukey takes its error term from the blocked model. None leaves
        the bootstrap verdicts in place, which is how the figures were drawn
        before Tukey and is kept for comparison rather than for use.
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
    panels = [gnn, left, right, singles, combinations, regressors]
    if block_by_seed is None:
        return panels

    for panel in panels:
        panel.evidence = tukey.retest(
            panel.evidence, list(manifest.seeds), block_by_seed=block_by_seed, label=panel.id
        )
    return panels


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


# an indent inside a tooltip: HTML eats ordinary spaces, so the two that set a
# stanza's lines under its heading have to be non-breaking
INDENT = "\u00a0\u00a0"


def trained_as(from_foundation: Any, target: str | None) -> str:
    """Name a network by where it started and what it was trained on.

    Adjectival, so it reads as a name rather than as a history: it sits after
    ``init:`` or ``encoder:`` without competing with the line below that says
    what this run trained. A checkpoint used unmodified was trained on nothing
    here and says so instead.
    """
    if target is None:
        return "published CheMeleon"
    origin = "CheMeleon" if from_foundation else "Chemprop"
    return f"{plots.PRETTY.get(target, target)}-trained {origin}"


def provenance(axis: str, level: str, manifest: Manifest) -> str:
    """Say which network produced a feature block.

    A block name says what a column is called and not what produced it, and
    four of these blocks are outputs of networks this project trained.
    "CheMeleon log2FC" and "Chemprop log2FC" differ in exactly that and in
    nothing a label shows.

    Read from the encoder specifications rather than restated here, so a block
    that changes what it trains cannot keep an old description. A block nothing
    trains says nothing: descriptors are computed from structure, and the
    foundation embedding is the checkpoint as published.
    """
    spec = (manifest.axes.get(axis) or {}).get(level) or {}
    blocks = spec.get("blocks") or ([spec["block"]] if spec.get("block") else [])
    if not blocks:
        return ""

    # imported here rather than at module scope: the encoders pull in torch and
    # chemprop, which a figure has no other use for
    from encoders import BLOCK_SPECS

    described = []
    for block in blocks:
        declared = BLOCK_SPECS.get(block)
        if declared is None:
            described.append(trained_as(None, None) if block == "chemeleon" else "")
            continue
        described.append(trained_as(declared.defaults.get("from_foundation"), declared.target))
    return ", ".join(part for part in described if part)


def tabular_detail(
    config: dict[str, Any], dims: dict[tuple[str, str], int], manifest: Manifest
) -> str:
    """Describe one tabular configuration: its blocks, then what reads them.

    Every block says the same three things in the same order, whichever figure
    it appears in and whether it stands alone or is joined to others: what
    produced it, how many columns it has, and what it was reduced to. A row
    joining several says them once per block under the kind of block it is, so
    a combination reads as the ingredients it was built from rather than as a
    new thing.

    The regressor comes last, after a blank line, because it reads what is
    above it, and the total beside it is what reaches it.
    """
    order = (("embedding", "embedding_pca"), ("readout", None), ("descriptors", "descriptor_pca"))
    present = [
        (axis, width_axis) for axis, width_axis in order if str(config.get(axis, "none")) != "none"
    ]

    blocks, total, known = [], 0, True
    for axis, width_axis in present:
        level = str(config[axis])
        native = dims.get((axis, level))
        width = int(config.get(width_axis, 0)) if width_axis else 0
        total += width if width > 0 else (native or 0)
        known = known and (width > 0 or native is not None)

        if axis == "descriptors":
            # blocks rather than a descriptor: rdkit_mordred is two of them,
            # and the singular under a plural heading read as a slip
            head = f"blocks: {plots.PRETTY.get(level, level)}"
        else:
            head = f"encoder: {provenance(axis, level, manifest)}"
        # a reduction is reported where one happened. The only line in the set
        # that reported a thing not happening was this one saying none, and its
        # absence says the same more quietly
        lines = [head, f"dims: {native or '?'}"] + ([f"PCA: {width}"] if width > 0 else [])
        if len(present) > 1:
            blocks.append(f"{axis}:<br>" + "<br>".join(INDENT + line for line in lines))
        else:
            blocks.append("<br>".join(lines))

    detail = list(blocks) if len(present) < 2 else [b for block in blocks for b in ("", block)][1:]
    regressor = config.get("regressor")
    if regressor:
        detail += ["", f"regressor: {plots.PRETTY.get(regressor, regressor)}"]
        if known:
            detail.append(f"ndims: {total}")
    return "<br>".join(detail)


def gnn_detail(config: dict[str, Any], dims: dict[tuple[str, str], int]) -> str:
    """Describe one graph-network cell: its body, its head, its auxiliary arm.

    Every cell is a Chemprop network with a feed-forward predictor on top, and
    those are two different sizes that one integer in a label cannot keep
    apart. What differs between the cells is where the body's weights came
    from and how wide that body therefore is: the CheMeleon checkpoint brings
    its own, and the log2FC body is the one this project trained from scratch.
    The head width is a figure-1 axis and is the same choice whichever body it
    sits on.
    """
    frozen = int(config["freeze_epochs"]) >= FROZEN_AT
    init = str(config["encoder_init"])
    finetune = str(config["finetune_target"])

    # the body's own width, read off the block that same network writes rather
    # than restated here. A body with no checkpoint behind it writes no block
    # and carries its width itself
    # the body's width is the width of the block that same network writes,
    # except for a body with no checkpoint behind it, which writes no block and
    # carries its width itself
    block = {
        "chemeleon": "chemeleon",
        "log2fc_checkpoint": "chemprop_log2fc",
        "chemeleon_log2fc_checkpoint": "chemeleon_log2fc",
    }.get(init)
    width = config.get("message_hidden_dim") if block is None else dims.get(("embedding", block))
    started = {
        "chemeleon": trained_as(None, None),
        "log2fc_checkpoint": trained_as(False, "log2fc"),
        "chemeleon_log2fc_checkpoint": trained_as(True, "log2fc"),
        # no checkpoint at all, which is the whole of what this cell is
        "scratch": "none",
    }[init]
    lines = [
        f"init: {started}",
        f"trained on: {plots.PRETTY.get(finetune, finetune)}",
        # released is the ordinary case and the label says frozen when it is
        # not, so only the exception is worth a word here
        f"MPNN dim: {width or '?'}" + (", frozen" if frozen else ""),
        f"FFN dim: {config['ffn_hidden_dim']}",
    ]
    if str(config.get("aux_target", "none")) == "none":
        lines.append("auxiliary: none")
        return "<br>".join(lines)

    # the auxiliary arm is a block like any other and says so the same way. It
    # is also the network behind the CheMeleon log2FC blocks of figures 3 to 5,
    # which is worth being able to read off both places in the same words
    arm = [f"encoder: {trained_as(True, str(config['aux_target']))}"]
    if config.get("aux_embedding"):
        arm.append(f"embedding dims: {dims.get(('embedding', 'chemeleon_log2fc'), '?')}")
    if config.get("aux_readout"):
        arm.append(f"readout dims: {dims.get(('readout', 'chemeleon_log2fc'), '?')}")
    lines.append("auxiliary:<br>" + "<br>".join(INDENT + line for line in arm))
    return "<br>".join(lines)


def declaration(manifest: Manifest, figure_id: str) -> dict[str, Any]:
    """Return what the manifest says a figure is.

    The title and the question are declared beside the stages rather than
    written here, so what a figure claims to answer and what it draws cannot
    drift apart, and a figure nothing declares cannot be built.

    Raises
    ------
    PanelError
        If the manifest declares no such figure.
    """
    for figure in manifest.figures:
        if figure["id"] == figure_id:
            return figure
    raise PanelError(
        f"the manifest declares no figure {figure_id!r}; "
        f"it has {[figure['id'] for figure in manifest.figures]}"
    )


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
    """Write one figure as a standalone page, with the row labels made hoverable.

    The page also carries the script that widens the gap between two panels to
    fit the labels drawn in it, which does nothing to a page of one panel.

    The scripts are appended rather than passed as plotly's ``post_script``,
    which formats the string it is given and would take every brace in the
    JavaScript for a placeholder.
    """
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)

    page = path.read_text()
    scripts = "\n".join(
        f"<script>{source.read_text()}</script>" for source in (ROW_HOVER, PANEL_GAP)
    )
    closing = "</body>"
    if closing not in page:
        raise PanelError(f"{path} has no body to attach the page scripts to")
    path.write_text(page.replace(closing, f"{scripts}\n{closing}", 1))
    logger.info("wrote %s", path)
    return path


def write_standalone(figure: go.Figure, path: Path) -> Path:
    """Write a figure whose axes carry numbers rather than one row per configuration.

    Figures 6 and 7 have no row labels, so there is nothing for a hover on a
    tick to open and the label script is left off rather than attached and
    left to report that it matched nothing.
    """
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)
    logger.info("wrote %s", path)
    return path


# the stage whose regressor levels are one checkpoint at a fixed member count
ENSEMBLE_STAGE = "tabpfn_ensemble"


def _member_count(slug: str, config: TabularConfig) -> int:
    """Return how many ensemble members a configuration's regressor was held at."""
    name = str(config.regressor)
    size = regressors.ENSEMBLE_SIZE_OF.get(name)
    if size is None:
        raise PanelError(f"{slug}: regressor {name!r} names no ensemble size")
    return size


def ensemble_rows(
    manifest: Manifest,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
) -> pd.DataFrame:
    """Score every ensemble size, as one model and as the seeds ensembled.

    Two errors per size, because the question needs both. The seed mean is one
    fitted model's error, averaged over the five replicates, and is what every
    ranking in this project is read on. The seed ensemble averages those five
    models' per-compound predictions and scores that once. The gap between them
    is what ensembling the seeds buys on top of whatever the model's own
    ensemble already did, which is the comparison the sweep exists to make.

    The spread diagnostics follow the uncertainty stage exactly, so a number
    here and a number in figure 6 mean the same thing: the model spread is
    averaged over seeds rather than combined in quadrature, and both spreads
    are scored against the residuals of the ensembled prediction.

    Parameters
    ----------
    manifest : Manifest
        Supplies the seeds and the stage's configurations.
    results_dir : path-like, optional
        Root holding the run directories.
    gates_dir : path-like, optional
        Where the gates this stage inherits from were recorded.

    Returns
    -------
    DataFrame
        One row per member count, ordered by it.

    Raises
    ------
    PanelError
        If a configuration has no runs on disk.
    """
    settled = gates.settled(manifest, ENSEMBLE_STAGE, gates_dir)
    rows = []
    for config in manifest.expand(ENSEMBLE_STAGE, settled):
        run_dirs = [config.run_dir(seed, results_dir) for seed in manifest.seeds]
        missing = [d for d in run_dirs if not (d / "predictions.csv").exists()]
        if missing:
            raise PanelError(
                f"{config.slug}: {len(missing)} of {len(run_dirs)} runs missing, first {missing[0]}"
            )

        stacked, observed = aggregate.stack_predictions(run_dirs)
        per_seed = [evaluate.metrics(observed, row)[gates.RANK_METRIC] for row in stacked]
        ensembled = evaluate.ensemble_mean(stacked)

        row = {
            "slug": config.slug,
            "n_estimators": _member_count(config.slug, config),
            "mae": float(np.mean(per_seed)),
            "seed_spread": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
            "ensemble_mae": float(evaluate.metrics(observed, ensembled)[gates.RANK_METRIC]),
            "n_seeds": len(run_dirs),
        }

        # both spreads are scored against the same prediction, so the only
        # thing that differs between them is what they claim to know
        spreads = {"ensemble": uncertainty.ensemble_spread(stacked)}
        model_spread = _model_spread(run_dirs)
        if model_spread is not None:
            spreads["model"] = model_spread
        for source, sigma in spreads.items():
            row[f"spearman_{source}"] = uncertainty.diagnose(
                observed, ensembled, sigma, source=source
            ).spearman
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_estimators").reset_index(drop=True)


def _model_spread(run_dirs: list[Path]) -> np.ndarray | None:
    """Return the per-compound model spread, averaged over seeds, or None.

    Averaged rather than combined in quadrature, matching the uncertainty
    stage: each seed's spread is that seed's statement about its own
    prediction, and disagreement between seeds is the other spread.
    """
    frames = [
        pd.read_csv(d / "predictions.csv", dtype={CANONICAL_COL: str}).sort_values(CANONICAL_COL)
        for d in run_dirs
    ]
    if any("predicted_std" not in frame.columns for frame in frames):
        return None
    spread = np.vstack([f["predicted_std"].to_numpy(dtype=np.float64) for f in frames])
    if not np.any(np.isfinite(spread) & (spread > 0)):
        return None
    return spread.mean(axis=0)


def ensemble_figure(
    manifest: Manifest,
    *,
    results_dir: Path = aggregate.RESULTS_DIR,
    gates_dir: Path | None = None,
) -> go.Figure:
    """Figure 7: how much of the tabular foundation model's advantage is ensembling.

    The references are the scores this project already has, drawn as rules
    rather than as rows: nothing in the sweep is a comparison to them, and the
    only reason they are here is to say whether a one-member model has already
    cleared what the rest of the work achieved.
    """
    frame = ensemble_rows(manifest, results_dir=results_dir, gates_dir=gates_dir)
    references = {}
    for name in ("best_gnn", "chemeleon_baseline"):
        if not _declared(manifest, name):
            continue
        config = resolve(manifest, name, gates_dir=gates_dir)
        scored = gates.scores([config], manifest, results_dir=results_dir)
        references[plots.spelled(name.replace("_", " "))] = scored[0][gates.RANK_METRIC]
    return plots.ensemble_figure(frame, references=references)
