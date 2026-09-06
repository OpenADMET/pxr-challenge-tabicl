"""Comparison plots, drawn from a gate's recorded evidence.

One chart shape answers most of this project's questions: several
configurations scored on the same compounds, and which of the differences
between them the evidence supports. It is the plot a multiple-comparison
procedure calls for, and the arrangement is the one statsmodels draws for
Tukey's HSD, a point per configuration with an interval and the comparisons
marked against a reference.

What differs is where the significance comes from. Tukey's assumes independent
groups with equal variances and takes its replicates from the seeds, which
here number five and are not the quantity of interest. This uses the project's
own test: every pair compared by a paired bootstrap over the 260 evaluation
compounds, resampled on one shared set of draws so each difference is paired,
with the family corrected by Benjamini-Hochberg. A configuration is marked
separated when that procedure separates it from the leader, and the interval
drawn is the comparison half width the same resampling produced, so two
intervals touch exactly when the pair is not separated. Interval and verdict
therefore come from one computation, and a figure cannot contradict the
decision recorded beside it.

Nothing here decides anything. A gate's chosen configuration is marked because
it was chosen, not because the plot found it.

Colour and position carry different things, and neither is redundant. Position
is the score, so the best configuration is the top row and needs no marking of
its own. Colour is either a verdict, blue for a configuration the procedure
does not separate from the best and grey for one it does, or an identity: a
gate's winner keeps one colour wherever it appears again. An identity colour
says which configuration a row is, not how it stands, so a row carried into a
figure for recognition is not making a significance claim it was not tested
for. Shape answers a third question, which is where the row came from: the
configuration this figure establishes is a star, one carried in from another
family is a diamond, and one this figure's own sweep produced is a circle.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# a row is either indistinguishable from the best configuration or set apart
# from it. The best is the top row rather than a colour of its own, since the
# figure is sorted and position already says it
TIED = "not separated from the leader"
SEPARATED = "separated from the leader"

# A verdict is drawn in grey, dark for a row the procedure does not separate
# from the leader and light for one it does. Colour is then spent entirely on
# identity: a hue means a particular configuration and nothing else, and a row
# with no hue has earned no name rather than having been assigned a reading.
VERDICT_COLOUR = {TIED: "#57606a", SEPARATED: "#c6cbd1"}

# Plotly's default qualitative palette, named by position so the source of a
# colour is legible and nothing is invented. Ten colours, of which eight are
# spent: the assignment below is the one that leaves the closest pair inside
# any single panel as far apart as this palette allows, which is 21.8 in
# CIELAB, between its blue and its purple in figure 4. Those two are the
# palette's own nearest neighbours, and no assignment separates them: pinning
# them to identities that never share a panel leaves the remaining five to be
# drawn from four hue families.
PALETTE = px.colors.qualitative.Plotly

# Each gate's winner takes one colour and keeps it wherever that configuration
# appears again, so a reader can follow it across the figures and read its
# neighbours: a grey neighbourhood means it was not among the best where it
# reappears. Matched on the axes the gate settled rather than on a slug,
# because a slug also names the axes a figure holds fixed and those differ
# from stage to stage.
GATE_COLOUR = {
    "canonical_descriptors": PALETTE[0],
    "embedding_reduction": PALETTE[4],
    "best_featureset": PALETTE[3],
    "best_regressor": PALETTE[9],
}

# Not every configuration worth following is a gate's winner. The GNN sweep
# settles no axis of the tabular manifest, so its best cell and the CheMeleon
# baseline it is read against have no gate to take a colour from, and the best
# single block is a row of a stage whose gate chose a combination. All three
# are carried through later figures as reference points, and are named here so
# they hold their colour the way a gate's winner does. The figure carrying one
# names the row by slug rather than matching axes, since a graph network in a
# tabular family shares no axes to be matched on.
NAMED_COLOUR = {
    # the published leaderboard score, which is a landmark rather than a
    # competitor: it has no per-compound predictions here to test against
    "anchor": PALETTE[1],
    "best_gnn": PALETTE[8],
    "chemeleon_baseline": PALETTE[5],
    "best_single": PALETTE[7],
}

IDENTITY_COLOUR = {**GATE_COLOUR, **NAMED_COLOUR}

X_RANGE = (0.30, 0.75)

# axis labels are read at a glance and against a dense column of markers
BOLD = {"weight": "bold"}
METRIC_LABEL = "MAE"

# Shape says how a row got into the figure, colour says what it is. A row this
# figure's own sweep produced is a circle whatever it is called elsewhere; a
# row carried in from another family is a diamond; the configuration the figure
# establishes is a star. Shape and colour answer different questions, so a
# carried row keeps its identity colour and a swept one keeps its circle even
# when it is somebody's winner.
CHOSEN, CARRIED, SCORED = "chosen here", "carried", "scored"
SYMBOL = {CHOSEN: "star", CARRIED: "diamond", SCORED: "circle"}
SIZE = {CHOSEN: 15, CARRIED: 11, SCORED: 9}

# names that read as prose rather than as identifiers
# Names as prose. The subscripts are markup rather than unicode because plotly
# renders <sub> in labels and tooltips alike, and a unicode subscript would not
# survive a copy into a caption
LOG2FC = "log<sub>2</sub>FC"
PEC50 = "pEC<sub>50</sub>"

PRETTY = {
    "rdkit_mordred": "RDKit + Mordred",
    "chemprop_log2fc": f"Chemprop {LOG2FC}",
    "chemeleon_log2fc": f"CheMeleon {LOG2FC}",
    "chemeleon_pec50": f"CheMeleon {PEC50}",
    "chemeleon": "CheMeleon",
    "log2fc": LOG2FC,
    "pec50": PEC50,
    "rdkit": "RDKit",
    "mordred": "Mordred",
    "tabpfn-v2.6": "TabPFN v2.6",
    "tabpfn-v3": "TabPFN v3",
    "tabicl": "TabICL",
    "tabfm": "TabFM",
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
}


class PlotError(ValueError):
    """A figure could not be built from the evidence it was given."""


def comparison_frame(
    evidence: dict[str, Any],
    winners: dict[str, dict[str, Any]] | None = None,
    named: dict[str, str] | None = None,
    references: list[dict[str, Any]] | None = None,
    home: str | None = None,
    carried: set[str] | None = None,
    origins: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Turn a gate's evidence into one row per configuration, ready to plot.

    Each row is drawn as its paired difference from the best configuration,
    positioned at the best configuration's score and offset by the best
    configuration's own half width, so a row's interval reaches the best one's
    exactly when the two are not separated. That is the geometry statsmodels
    gives Tukey's intervals, each comparison split between the two intervals
    it joins, and it leaves the best configuration with an interval of its own
    rather than a bare point.

    A marginal interval is the wrong thing to draw in its place. It carries
    the compound variance the paired test cancels, so two of them overlap
    while the test separates them and the figure contradicts the verdict
    beside it.

    Parameters
    ----------
    evidence : dict
        A :func:`gates.evidence` or :func:`gates.confirm` result.
    winners : dict, optional
        Gate winners from :func:`gate_winners`, each keeping its own colour
        wherever it appears.
    named : dict, optional
        Slug mapped to a name from :data:`NAMED_COLOUR`, colouring a row by
        identity where no gate decided it. A row carried into a figure is
        picked out by slug rather than by axis levels, since a graph network
        carried into a tabular family shares no axes to be matched on.
    references : list of dict, optional
        Scored rows to show without testing, from :func:`gates.scores`, each
        naming its identity under ``identity``. A reference is drawn where its
        score puts it and given no interval, because it was not part of the
        family the correction was computed over and a whisker would claim it
        was. This is how a figure carries a point of comparison without
        changing the question it asks.
    home : str, optional
        The identity this figure establishes, drawn as the chosen one.
        Defaults to the gate the evidence came from.
    carried : set of str, optional
        Slugs of family members this figure did not sweep, drawn as carried.
        A row the figure's own stage ran is drawn as swept however it is named
        elsewhere, since shape reports where a row came from and not what it
        is called.
    origins : dict, optional
        Identity mapped to where it was established, which the tooltip says on
        every row carrying an identity from elsewhere. Colour tells a reader
        they have met a row before; this tells them where.

    Returns
    -------
    DataFrame
        Sorted best first, one row per configuration.

    Raises
    ------
    PlotError
        If the evidence carries no ranking, or predates the comparison
        intervals and would have to be drawn with a substitute.
    """
    ranking = evidence.get("ranking")
    if not ranking:
        raise PlotError("evidence carries no ranking")

    significance = evidence.get("significance") or {}
    intervals = significance.get("comparison_intervals")
    if not intervals or "leader_halfwidth" not in significance:
        raise PlotError(
            "evidence carries no comparison intervals; re-measure the gate rather "
            "than drawing a marginal interval in their place, which would overlap "
            "where the paired test separates"
        )

    against = significance.get("against_leader", [])
    separated = {row["slug"] for row in against if row["separated"]}
    p_values = {row["slug"]: row["p_value"] for row in against}
    halfwidth = significance["leader_halfwidth"]
    leader = evidence["leader_slug"]
    best = min(row["mae"] for row in ranking)
    this_gate = home or evidence.get("gate")
    named = named or {}
    carried = carried or set()
    origins = origins or {}

    # a family may mix kinds, and rows of different kinds share no axes to be
    # compared on. The axes are read off the largest group that shares a shape,
    # and a row outside it is named and coloured by what the caller says it is
    signature = Counter(tuple(sorted(row["config"])) for row in ranking).most_common(1)[0][0]
    axes = varying_axes([row for row in ranking if tuple(sorted(row["config"])) == signature])

    rows = []
    for row in ranking:
        slug = row["slug"]
        matched = tuple(sorted(row["config"])) == signature
        identity = winner_of(row["config"], winners or {}, axes) if matched else None
        gate = named.get(slug) or identity
        low, high = intervals[slug]
        if slug == leader:
            err_minus = err_plus = halfwidth
        else:
            err_minus = max(0.0, row["mae"] - (best + low + halfwidth))
            err_plus = max(0.0, (best + high + halfwidth) - row["mae"])
        is_separated = slug in separated
        colour = (
            IDENTITY_COLOUR.get(gate or "") or VERDICT_COLOUR[SEPARATED if is_separated else TIED]
        )
        rows.append(
            {
                "slug": slug,
                # a family of one kind is named by the axes that vary in it; a
                # mixed one has none, so its rows carry their own names
                "label": row.get("label") or label_for(row["config"], axes),
                "mae": row["mae"],
                "err_minus": err_minus,
                "err_plus": err_plus,
                "seed_spread": row["seed_spread"],
                "ensemble": (row.get("ensemble") or {}).get("mae"),
                "gate": gate,
                # an identity colour where the row is somebody's winner, a
                # verdict colour otherwise
                "colour": colour,
                # the interval says how the row stands whatever its marker is
                # coloured, so an identity keeps its colour and still reads as
                # separated
                "whisker": VERDICT_COLOUR[SEPARATED] if is_separated else colour,
                "role": (CHOSEN if gate == this_gate else (CARRIED if slug in carried else SCORED)),
                "p_value": p_values.get(slug),
                "detail": row.get("detail", ""),
                "origin": "" if gate == this_gate else origins.get(gate or "", ""),
                # the leader has nothing to be compared against and is the top
                # row, which says it without a line of its own
                "verdict": (
                    ""
                    if slug == leader
                    else f"{'' if is_separated else 'not '}leader-separated: "
                    f"p={p_values.get(slug, 0):.4f}"
                ),
            }
        )

    for reference in references or []:
        identity = reference["identity"]
        rows.append(
            {
                "slug": reference["slug"],
                "label": reference["label"],
                "mae": reference["mae"],
                # no interval: this row was not in the family the correction
                # was computed over, so it has no verdict to draw
                "err_minus": 0.0,
                "err_plus": 0.0,
                "seed_spread": reference.get("seed_spread"),
                "ensemble": (reference.get("ensemble") or {}).get("mae"),
                "gate": identity,
                "colour": IDENTITY_COLOUR[identity],
                "whisker": IDENTITY_COLOUR[identity],
                "role": CHOSEN if identity == this_gate else CARRIED,
                "p_value": None,
                "detail": reference.get("detail", ""),
                "origin": "" if identity == this_gate else origins.get(identity, ""),
                "verdict": reference.get("verdict", "not tested here"),
            }
        )

    frame = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    frame["hover"] = [_hover(row) for _, row in frame.iterrows()]
    return frame


def _hover(row: pd.Series) -> str:
    """Render one row's tooltip: how it scored, how it stands, and what it is.

    Everything a reader would otherwise have to look up in a table, since the
    labels carry only what varies. A row with no seeds behind it, the published
    anchor, is not offered statistics it does not have.

    Written with the characters themselves rather than HTML entities: a
    hovertemplate is not parsed as markup beyond its tags, so an entity would
    be shown as it was typed.
    """
    lines = [f"<b>{_visible(row['label'])}</b>"]
    if row["origin"]:
        lines.append(f"<i>{row['origin']}</i>")
    lines.append(f"MAE {row['mae']:.4f}")
    if pd.notna(row["seed_spread"]):
        lines[-1] += f" \u00b1 {row['seed_spread']:.4f} over seeds"
    if pd.notna(row["ensemble"]) and row["ensemble"] != row["mae"]:
        lines.append(f"seed ensemble MAE: {row['ensemble']:.4f}")
    if row["verdict"]:
        lines.append(row["verdict"])
    if row["detail"]:
        lines.append(f"<br>{row['detail']}")
    return "<br>".join(lines)


def gate_winners(gates_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return each recorded gate's chosen axes, for colouring them everywhere.

    Reads the gate files rather than being told, so a figure cannot disagree
    with a decision that has been recorded.

    Returns
    -------
    dict
        Gate id mapped to the axis levels it settled.
    """
    import gates as gates_module

    resolved = gates_dir or gates_module.GATES_DIR
    winners: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(resolved).glob("*.json")):
        record = json.loads(path.read_text())
        chosen = record.get("chosen")
        if chosen:
            winners[record["gate"]] = chosen
    return winners


# the level an axis takes when whatever it names is absent
ABSENT = {"none", "0", "-1"}

# widths, which qualify a block rather than being blocks themselves
WIDTH_AXES = {"embedding_pca", "descriptor_pca"}

# what a tabular configuration is named by when it stands on its own, outside a
# family whose varying axes would otherwise say what to call it
BLOCK_AXES = ("embedding", "embedding_pca", "readout", "descriptors", "descriptor_pca")


def winner_of(
    config: dict[str, Any], selectors: dict[str, dict[str, Any]], varying: tuple[str, ...]
) -> str | None:
    """Return the identity this row carries, if any.

    An identity is a gate's winner, given as the axis levels that pick it
    out. A row qualifies only when it is that configuration and nothing else:
    it agrees on every axis the selector names, and every other axis this
    figure varies is absent.
    Agreeing on the named axes alone is far too weak, since a selector naming
    one axis would then claim every row sharing that level, and one whose axes
    the figure holds fixed would claim every row at once.

    Later identities are checked first, so a row satisfying several is named by
    the latest, which is the decision a reader is following.
    """
    for gate in reversed(list(IDENTITY_COLOUR)):
        chosen = selectors.get(gate)
        if not chosen:
            continue
        if not set(chosen) & set(varying):
            # this figure holds the selector's axes fixed, so nothing here is
            # that configuration rather than everything being it
            continue
        if any(str(config.get(axis)) != str(level) for axis, level in chosen.items()):
            continue
        # a width belongs to its block rather than standing on its own, so it
        # is not something a row can carry "as well as" the winner
        others = [axis for axis in varying if axis not in chosen and axis not in WIDTH_AXES]
        if all(str(config.get(axis)) in ABSENT for axis in others):
            return gate
    return None


def varying_axes(ranking: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return the axes that differ across a ranking, in manifest order.

    A figure holds most axes fixed, and naming those in every row says nothing
    while hiding what does differ. The regressor stage varies one axis and
    fixes the featureset; the width probes do the reverse. Reading it off the
    data means a label never has to be told which stage it is drawing.
    """
    if not ranking:
        return ()
    axes = list(ranking[0]["config"])
    return tuple(axis for axis in axes if len({str(row["config"][axis]) for row in ranking}) > 1)


# a block whose width is not shown is named by what kind of block it is, since
# the level alone ("CheMeleon") does not say whether it is an embedding, a
# readout or a descriptor set
BLOCK_NOUN = {"embedding": "emb.", "descriptors": "descriptors"}


def label_for(
    config: dict[str, Any],
    axes: tuple[str, ...],
    *,
    dims: dict[tuple[str, str], int] | None = None,
    widths: tuple[str, ...] = ("embedding", "descriptors"),
    width_separator: str = " ",
) -> str:
    """Return a short name for a configuration, over the axes that name it.

    Parameters
    ----------
    config : dict
        One configuration's axis levels.
    axes : tuple of str
        The axes to name, usually from :func:`varying_axes`. Empty names
        everything, which is the degenerate case of a single-row figure.
    dims : dict, optional
        Native column count per ``(axis, level)``, which is what a block kept
        whole is called: "RDKit 217 (native)" rather than "RDKit native".
    widths : tuple of str, optional
        The block axes whose width belongs in the name. A width says something
        where it varies or where it was chosen; the embedding's is dropped
        once it is settled, since an integer beside a foundation model reads
        as ambiguous between the reduction and the network's own size.
    width_separator : str, optional
        What comes between a block and its width. ``"<br>"`` sets the width on
        its own line, for a figure whose rows differ in nothing else.

    Returns
    -------
    str
    """
    named = axes or tuple(config)
    parts: list[str] = []

    # a width belongs to its block rather than standing on its own, so it is
    # folded into the block's name wherever both are being shown
    width_axes = {"embedding": "embedding_pca", "descriptors": "descriptor_pca"}
    for axis in named:
        if axis in width_axes.values():
            continue
        level = config.get(axis, "none")
        if level == "none":
            continue
        width_axis = width_axes.get(axis)
        width = int(config.get(width_axis, 0)) if width_axis else 0
        shown = axis in widths
        suffix = ""
        if shown and width_axis and (width_axis in named or width > 0):
            if width > 0:
                suffix = f"{width_separator}{width}"
            else:
                native = (dims or {}).get((axis, str(level)))
                whole = f"{native} (native)" if native else "native"
                suffix = f"{width_separator}{whole}"
        name = PRETTY.get(str(level), str(level))
        # a readout is two predicted columns, not an embedding, and the two are
        # produced by the same networks, so the name has to say which it is
        if axis == "readout":
            name = f"{name} r.o."
        elif not shown and axis in BLOCK_NOUN:
            name = f"{name} {BLOCK_NOUN[axis]}"
        parts.append(f"{name}{suffix}")

    if not parts:
        # every named axis is a width, so the widths are the whole story
        for axis in named:
            if axis in width_axes.values():
                value = int(config[axis])
                parts.append("native" if value < 0 else str(value))
    return " + ".join(parts) or "none"


# What a cell's message-passing network was initialised from. Both are Chemprop
# D-MPNNs: one starts from the CheMeleon foundation checkpoint, the other from
# this project's own checkpoint pretrained on log2FC, which is what a network
# was trained to predict and never a body of its own
BODY = {"chemeleon": "CheMeleon", "log2fc_checkpoint": f"Chemprop {LOG2FC}"}


def wrap_label(label: str, width: int = 34) -> str:
    """Break a long name over two lines, at a join rather than mid-phrase.

    A figure of combinations has rows naming three blocks, and a label column
    wide enough for them leaves little of the axis. The break goes at a ``+``,
    so each line is a whole ingredient and the plus stays with the line it
    joins from.

    Parameters
    ----------
    label : str
        A name built by :func:`label_for`.
    width : int, optional
        Characters above which a name is worth breaking.

    Returns
    -------
    str
        The name, with at most one line break.
    """
    if len(_visible(label)) <= width or " + " not in label:
        return label

    # split where the two lines come closest to even, so neither is a stub
    parts = label.split(" + ")
    joins = [len(_visible(" + ".join(parts[: i + 1]))) for i in range(len(parts) - 1)]
    target = len(_visible(label)) / 2
    at = min(range(len(joins)), key=lambda i: abs(joins[i] - target))
    head, tail = " + ".join(parts[: at + 1]), " + ".join(parts[at + 1 :])
    return f"{head} +<br>{tail}"


def _visible(label: str) -> str:
    """Return a label as it reads on one line, without its markup.

    A line break stands for the space it replaced, so a wrapped name read back
    as a tooltip's title does not run two words together.
    """
    for tag in ("<sub>", "</sub>", "<b>", "</b>"):
        label = label.replace(tag, "")
    return label.replace("<br>", " ")


def gnn_label(config: dict[str, Any], *, frozen_at: int = 30, freeze: bool = True) -> str:
    """Return a short name for one graph-network cell.

    The cells are enumerated rather than crossed, so they share no single
    varying axis to be named by the way a tabular family is. The name is built
    from what actually differs between them: what the body started as, whether
    it was allowed to adapt, the head width, and which halves of the auxiliary
    arm reach the predictor.

    Parameters
    ----------
    config : dict
        A cell's flattened axes, from :meth:`manifest.GnnCell.as_dict`.
    frozen_at : int, optional
        The epoch budget. A freeze at or above it never releases the body,
        which is what "frozen" means here rather than a number of epochs.
    freeze : bool, optional
        Whether the number of warmup epochs belongs in the name. It does not
        once a figure reports the best over that axis, though never releasing
        the body is a different thing and is always named.

    Returns
    -------
    str
    """
    held = int(config["freeze_epochs"]) >= frozen_at
    name = BODY.get(str(config["encoder_init"]), str(config["encoder_init"]))

    # the auxiliary arm is a second network feeding the same predictor, so it
    # is joined with a circled plus rather than the plain one that joins the
    # feature blocks of a tabular row: nothing is concatenated into a table
    # here, a second network's output is folded into the predictor's input. Its
    # two halves belong to it and are named under it, where a bare "readout"
    # would read as a third ingredient
    if config.get("aux_target", "none") != "none":
        target = PRETTY.get(str(config["aux_target"]), str(config["aux_target"]))
        halves = []
        if config.get("aux_embedding"):
            halves.append("embedding")
        if config.get("aux_readout"):
            halves.append("readout")
        name += f" \u2295 {target} ({', '.join(halves)})"

    # the head's width is a property of the predictor rather than something
    # added to the inputs, so it follows them behind a comma
    parts = [name, f"FFN {config['ffn_hidden_dim']}"]
    if held:
        parts.append("frozen")
    elif freeze:
        parts.append(f"freeze {int(config['freeze_epochs'])}")
    return ", ".join(parts)


def darken(colour: str, factor: float = 0.62) -> str:
    """Return a darker shade of a hex colour, for a marker's own outline.

    One neutral outline on every marker flattens the palette: a light fill
    reads as unfinished and the identity colours stop being distinguishable at
    a glance. An outline of the fill's own hue keeps the marker one object.
    """
    value = colour.lstrip("#")
    channels = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{int(channel * factor):02x}" for channel in channels)


def row_traces(frame: pd.DataFrame) -> list[go.Scatter]:
    """Build one panel's traces: the whiskers, then the markers over them.

    The whiskers are drawn as plain lines rather than as error bars, because
    plotly renders a trace's error bars above the marker layer whatever the
    trace order, so an error bar runs through the shape it belongs to. A line
    trace is covered by the markers added after it. There is one trace per
    whisker colour, since a line takes a single colour for its whole trace.

    Parameters
    ----------
    frame : DataFrame
        One panel's rows, from :func:`comparison_frame`.

    Returns
    -------
    list of plotly.graph_objects.Scatter
        In draw order.
    """
    whiskers = []
    for colour, group in frame.groupby("whisker", sort=False):
        # a null between rows breaks the line rather than joining one row's
        # whisker to the next
        xs: list[float | None] = []
        ys: list[str | None] = []
        for _, row in group.iterrows():
            if not row["err_minus"] and not row["err_plus"]:
                # a row shown without being tested here
                continue
            xs += [row["mae"] - row["err_minus"], row["mae"] + row["err_plus"], None]
            ys += [row["label"], row["label"], None]
        if not xs:
            continue
        whiskers.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line={"color": colour, "width": 2.4},
                # the end caps, one on each point the line was built from
                marker={
                    "color": colour,
                    "symbol": "line-ns-open",
                    "size": 8,
                    "line": {"width": 2.0, "color": colour},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    markers = []
    for colour, group in frame.groupby("colour", sort=False):
        markers.append(
            go.Scatter(
                x=group["mae"],
                y=group["label"],
                mode="markers",
                marker={
                    "color": colour,
                    "symbol": [SYMBOL[role] for role in group["role"]],
                    "size": [SIZE[role] for role in group["role"]],
                    "line": {"color": darken(colour), "width": 1.1},
                },
                customdata=group[["hover"]].to_numpy(),
                hovertemplate="%{customdata[0]}<extra></extra>",
                showlegend=False,
            )
        )
    return whiskers + markers


def comparison_figure(
    *frames: pd.DataFrame,
    subtitles: tuple[str, ...] = (),
    x_range: tuple[float, float] = X_RANGE,
    metric_label: str = METRIC_LABEL,
) -> go.Figure:
    """Draw one or more comparison panels on a shared axis.

    Every comparison figure in the project is this one function. A panel is a
    frame from :func:`comparison_frame`, and what differs between figures is
    the rows in it, not how they are drawn: the same shapes mean the same
    things everywhere, and a configuration keeps its colour from one figure to
    the next. Two frames are drawn side by side, for questions asked of two
    families that are read against each other rather than in sequence, and
    each keeps its own family and so its own correction.

    Parameters
    ----------
    *frames : DataFrame
        One panel each, from :func:`comparison_frame`.
    subtitles : tuple of str, optional
        A short name over each panel. Wanted when there is more than one, since
        nothing else in the figure says which is which.
    x_range : tuple of float, optional
        Axis limits, the same for every figure so a position can be carried
        from one to the next.
    metric_label : str, optional
        The x-axis label, naming the metric and what it was measured on.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    PlotError
        If no frame is given, or the subtitles do not match the panels.
    """
    if not frames:
        raise PlotError("a figure needs at least one panel")
    if subtitles and len(subtitles) != len(frames):
        raise PlotError(f"{len(frames)} panels but {len(subtitles)} subtitles")

    figure = make_subplots(
        rows=1,
        cols=len(frames),
        subplot_titles=list(subtitles) or None,
        horizontal_spacing=0.06,
    )
    for column, frame in enumerate(frames, start=1):
        for trace in row_traces(frame):
            figure.add_trace(trace, row=1, col=column)
        # the y axis is pinned to the sorted labels rather than left to follow
        # trace order, which is what keeps the column sorted by score however
        # the traces come out, and reversed so the best sits at the top
        figure.update_yaxes(
            title_text="",
            type="category",
            categoryorder="array",
            categoryarray=frame["label"].tolist(),
            autorange="reversed",
            tickfont=BOLD,
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text=f"<b>{metric_label}</b>", range=list(x_range), row=1, col=column
        )

    # one key for the whole figure, built from every panel so a shape used on
    # only one side is still explained
    for trace in _legend(pd.concat(frames, ignore_index=True)):
        figure.add_trace(trace, row=1, col=1)

    figure.update_layout(**_layout(max(len(frame) for frame in frames)))
    figure.update_xaxes(tickfont=BOLD)
    figure.update_annotations(font={"size": 13})
    return figure


def _layout(n_rows: int) -> dict[str, Any]:
    """Return the layout every comparison panel shares."""
    return {
        "template": "plotly_white",
        "height": max(300, 32 * n_rows + 140),
        "legend": {
            "title_text": "",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "right",
            "x": 1.0,
        },
        "margin": {"l": 10, "r": 30, "t": 50, "b": 60},
    }


# the legend describes the encoding, not the rows: the two verdict colours and
# the two shapes. An identity colour names one configuration, so it would put
# a figure's own contents in the key and grow with them, and what it means is
# better said in a caption
_LEGEND_NEUTRAL = "#57606a"


def _legend(frame: pd.DataFrame) -> list[go.Scatter]:
    """Build the key: what the two verdict colours and the two shapes mean.

    Only entries the figure uses are drawn, so a panel with nothing carried
    into it does not advertise a shape it has no example of. The traces hold
    no points, which keeps them out of the axis ranges and the hover.
    """
    entries: list[tuple[str, str, str]] = []
    colours = set(frame["colour"])
    for verdict, colour in VERDICT_COLOUR.items():
        if colour in colours:
            entries.append((verdict, colour, "circle"))
    roles = set(frame["role"])
    for role, label in ((CHOSEN, "chosen here"), (CARRIED, "carried in")):
        if role in roles:
            entries.append((label, _LEGEND_NEUTRAL, SYMBOL[role]))

    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=name,
            marker={
                "color": colour,
                "symbol": symbol,
                "size": 10,
                "line": {"color": "#24292f", "width": 0.8},
            },
            hoverinfo="skip",
            showlegend=True,
        )
        for name, colour, symbol in entries
    ]


# the two kinds of spread a run reports, which are not the same quantity and
# are never pooled: what the model itself claims, and how far the seeds of one
# configuration disagree
SPREAD_COLOUR = {"model": "#ff2e63", "ensemble": "#00c2d1"}

# the scatter is one quantity rather than a comparison of two, so it takes the
# palette's neutral working colour rather than either curve's
SCATTER_COLOUR = "#00c2d1"
STAGE_DASH = {"raw": "solid", "calibrated": "dot"}


def uncertainty_figure(
    points: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    source: str = "model",
    metric_label: str = "predicted standard deviation",
) -> go.Figure:
    """Draw whether a predicted uncertainty tracks the error it is about.

    Two panels, because the question has two halves that no single chart
    answers. On the left, one point per compound: does a larger predicted
    spread go with a larger error? That is a ranking question, and a cloud
    with no slope answers it in the negative however well calibrated the
    intervals are. On the right, the coverage curve: of the compounds an
    interval of nominal width should contain, how many does it? That is a
    calibration question, and a model can pass it while failing the first by
    predicting the same width everywhere.

    The diagonal is where a perfectly calibrated model's curve lies. Above it
    the intervals are too wide, below it too narrow.

    Parameters
    ----------
    points : DataFrame
        One row per compound, with ``abs_residual`` and a sigma column per
        source, as written by the uncertainty stage.
    coverage : DataFrame
        Long form, with ``source``, ``stage``, ``expected`` and ``observed``.
    source : str, optional
        Which spread the scatter is drawn on. The coverage panel shows every
        source it was given, since comparing them is the point of it.
    metric_label : str, optional
        The scatter's x-axis label.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    column = f"{source}_sigma"
    if column not in points:
        raise PlotError(f"points carry no {column!r} column, so there is no spread to draw")

    figure = make_subplots(rows=1, cols=2, horizontal_spacing=0.09)
    figure.add_trace(
        go.Scatter(
            x=points[column],
            y=points["abs_residual"],
            mode="markers",
            marker={
                "color": SCATTER_COLOUR,
                "size": 6,
                "opacity": 0.7,
                "line": {"color": darken(SCATTER_COLOUR), "width": 0.6},
            },
            hovertemplate=("sigma %{x:.3f}<br>|error| %{y:.3f}<extra></extra>"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # the diagonal first, so the curves are drawn over it
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "#9aa4ae", "width": 1, "dash": "dash"},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    for (curve_source, stage), group in coverage.groupby(["source", "stage"], sort=False):
        ordered = group.sort_values("expected")
        figure.add_trace(
            go.Scatter(
                x=ordered["expected"],
                y=ordered["observed"],
                mode="lines",
                name=f"{curve_source}, {stage}",
                line={
                    "color": SPREAD_COLOUR.get(str(curve_source), "#0969da"),
                    "width": 1.8,
                    "dash": STAGE_DASH.get(str(stage), "solid"),
                },
                hovertemplate="nominal %{x:.2f}<br>actual %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    figure.update_xaxes(title_text=f"<b>{metric_label}</b>", row=1, col=1)
    figure.update_yaxes(title_text="<b>absolute error</b>", row=1, col=1)
    figure.update_xaxes(title_text="<b>nominal coverage</b>", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="<b>observed coverage</b>", range=[0, 1], row=1, col=2)
    figure.update_xaxes(tickfont=BOLD)
    figure.update_yaxes(tickfont=BOLD)
    # both panels carry their own axis titles, so the margin has to hold one
    layout = {**_layout(6), "height": 420}
    layout["margin"] = {"l": 70, "r": 30, "t": 20, "b": 60}
    figure.update_layout(**layout)
    return figure
