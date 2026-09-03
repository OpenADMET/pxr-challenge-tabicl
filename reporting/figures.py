"""Render the PXR result bar figures from the run index and reference table.

Reads two data files and nothing else:

- ``reporting/results.parquet`` : every run we tried, one row per
  (base_dir, seed), every spec and metric column resolved from the run's own
  artifacts by ``build_run_provenance.py``.
- ``reporting/reference.csv`` : borrowed external numbers with no run of ours
  behind them (the N283T challenge report figures).

Every MAE, whisker, bar width, panel membership, row order, divider position,
and role color is computed here from that data. No number is hand-entered, and
no panel enumerates its rows. Each ``PanelSpec`` in ``build_panels`` names a
swept run family by a spec predicate (never a base_dir string), the axis it
varies along, and a few borrowed comparison rows drawn from computed role
holders; membership, order, the reused-bar flag, and role color all fall out of
the data. The one thing not computed is label text, which encodes distinctions
the provenance columns do not (for example "head only" versus "encoder + head");
it lives in the declared ``LABELS`` registry, resolved per run so the renderer
never invents it.

Run ``python reporting/figures.py`` to rewrite ``figures/figure-01.html`` ..
``figures/figure-06.html``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd

# axis range shared by every panel, in MAE units
AXIS_LOW = 0.30
AXIS_HIGH = 0.70

# only these seeds are real re-runs; the unseeded originals (seed -1) carry a
# different, sometimes mis-resolved spec and a different MAE, so they never draw
SEEDS = (0, 1, 2, 3, 4)

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.parquet"
REFERENCE_PATH = HERE / "reference.csv"
OUT_DIR = HERE / "figures"

CSS_STYLE = """<style>.pxr-post {
    --bg: #282a36;
    --bg-raised: #343746;
    --ink: #f8f8f2;
    --ink-soft: #a4a8c7;
    --line: #44475a;
    --accent: #ffb86c;
    --accent-ink: #ffcb94;
    --sage: #50fa7b;
    --gray: #b8bcc8;
    --plum: #bd93f9;
    --track: #21222c;
    --target: #ff5555;
    --target-ink: #ff8080;
    --blue: #8be9fd;
    --magenta: #ff6ac1;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono", Consolas, monospace;
    --sans: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
  }

  @media (prefers-color-scheme: dark) {.pxr-post {
      --bg: #282a36;
      --bg-raised: #343746;
      --ink: #f8f8f2;
      --ink-soft: #a4a8c7;
      --line: #44475a;
      --accent: #ffb86c;
      --accent-ink: #ffcb94;
      --sage: #50fa7b;
      --gray: #b8bcc8;
      --plum: #bd93f9;
      --track: #21222c;
      --target: #ff5555;
      --target-ink: #ff8080;
      --blue: #8be9fd;
      --magenta: #ff6ac1;
    }
  }.pxr-post[data-theme="dark"] {
    --bg: #282a36;
    --bg-raised: #343746;
    --ink: #f8f8f2;
    --ink-soft: #a4a8c7;
    --line: #44475a;
    --accent: #ffb86c;
    --accent-ink: #ffcb94;
    --sage: #50fa7b;
    --gray: #b8bcc8;
    --plum: #bd93f9;
    --track: #21222c;
    --target: #ff5555;
    --target-ink: #ff8080;
    --blue: #8be9fd;
    --magenta: #ff6ac1;
  }.pxr-post[data-theme="light"] {
    --bg: #282a36;
    --bg-raised: #343746;
    --ink: #f8f8f2;
    --ink-soft: #a4a8c7;
    --line: #44475a;
    --accent: #ffb86c;
    --accent-ink: #ffcb94;
    --sage: #50fa7b;
    --gray: #b8bcc8;
    --plum: #bd93f9;
    --track: #21222c;
    --target: #ff5555;
    --target-ink: #ff8080;
    --blue: #8be9fd;
    --magenta: #ff6ac1;
  }.pxr-post, .pxr-post * { box-sizing: border-box; }.pxr-post {
    background: var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.55;
    margin: 0;
    padding: 4.5rem 1.5rem 6rem;
  }.pxr-post {
    max-width: 780px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 3.5rem;
  }.pxr-post header {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    border-bottom: 1px solid var(--line);
    padding-bottom: 2rem;
  }.pxr-post .eyebrow {
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sage);
  }.pxr-post h1 {
    font-size: 2rem;
    font-weight: 650;
    margin: 0;
    letter-spacing: -0.01em;
    text-wrap: balance;
  }.pxr-post .subhead {
    color: var(--ink-soft);
    font-size: 1.05rem;
  }.pxr-post .meta {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--ink-soft);
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.4rem;
  }.pxr-post section {
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
  }.pxr-post .section-head {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }.pxr-post .section-num {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--sage);
    letter-spacing: 0.06em;
  }.pxr-post h2 {
    font-size: 1.35rem;
    margin: 0;
    text-wrap: balance;
  }.pxr-post .lede {
    color: var(--ink-soft);
    font-size: 0.98rem;
  }.pxr-post .chart {
    background: var(--bg-raised);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.6rem 1.6rem 1.3rem;
  }.pxr-post .axis-ref {
    display: grid;
    grid-template-columns: 15rem 1fr 4.2rem;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
  }.pxr-post .axis-ticks {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--ink-soft);
  }.pxr-post .rows {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
  }.pxr-post .row {
    display: grid;
    grid-template-columns: 15rem 1fr 4.2rem;
    align-items: center;
    gap: 0.6rem;
    /* Fixed to the two-line (label + sub) height so bar-to-bar spacing
       stays even whether or not a given row's sub label is present; the
       label column is widened to keep every sub on one line, so no row
       needs a third line */
    min-height: 1.9rem;
  }.pxr-post .row-label {
    font-size: 0.86rem;
    text-align: right;
    color: var(--ink);
  }.pxr-post .row-label .sub {
    display: block;
    font-size: 0.72rem;
    color: var(--ink-soft);
  }.pxr-post .track {
    position: relative;
    height: 1.6rem;
    background: var(--track);
    border-radius: 4px;
    overflow: hidden;
  }.pxr-post .bar {
    position: absolute;
    inset: 0 auto 0 0;
    background-color: var(--gray);
    border-radius: 4px;
  }.pxr-post .bar.win {
    background-color: var(--accent);
  }.pxr-post .bar.reused {
    background-image: repeating-linear-gradient(
      135deg,
      rgba(0, 0, 0, 0.32) 0 6px,
      transparent 6px 12px
    );
  }.pxr-post /* Real measured range only: seed-to-seed min-max, .pxr-post or a split-to-split range; absent means single-run */
  .err-whisker {
    position: absolute;
    top: 0.15rem;
    bottom: 0.15rem;
    border-left: 1.5px solid rgb(255, 255, 255);
    border-right: 1.5px solid rgb(255, 255, 255);
    opacity: 0.9;
  }.pxr-post .err-whisker::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    border-top: 1.5px solid rgb(255, 255, 255);
  }.pxr-post .mini-track .err-whisker {
    top: 0.1rem;
    bottom: 0.1rem;
  }.pxr-post .row.winner .row-label {
    font-weight: 650;
    color: var(--accent-ink);
  }.pxr-post .mae-val {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.86rem;
    text-align: right;
    color: var(--ink);
  }.pxr-post .row.winner .mae-val {
    font-weight: 650;
    color: var(--accent-ink);
  }.pxr-post .row.ref-divider {
    padding-bottom: 0.55rem;
    margin-bottom: 0.1rem;
    border-bottom: 1px dashed var(--line);
  }.pxr-post .bar.target-bar {
    background-color: var(--target);
  }.pxr-post .row.target .row-label {
    font-style: italic;
    color: var(--target-ink);
  }.pxr-post .row.target .mae-val {
    color: var(--target-ink);
    font-weight: 650;
  }.pxr-post .bar.context-bar {
    background: repeating-linear-gradient(
      135deg,
      var(--target) 0 6px,
      color-mix(in srgb, var(--target) 70%, transparent) 6px 12px
    );
  }.pxr-post .row.context .row-label {
    font-style: italic;
    color: var(--target-ink);
  }.pxr-post .row.context .mae-val {
    color: var(--target-ink);
    font-weight: 650;
  }.pxr-post .bar.floor-bar {
    background-color: var(--plum);
  }.pxr-post .row.floor .row-label {
    font-style: italic;
    color: var(--plum);
  }.pxr-post .row.floor .mae-val {
    color: var(--plum);
    font-weight: 650;
  }.pxr-post .row.single_best .row-label {
    color: var(--blue);
  }.pxr-post .row.single_best .mae-val {
    color: var(--blue);
    font-weight: 650;
  }.pxr-post .row.sweepbest .row-label {
    color: var(--sage);
  }.pxr-post .row.sweepbest .mae-val {
    color: var(--sage);
    font-weight: 650;
  }.pxr-post .row.chemeleon .row-label {
    color: var(--magenta);
  }.pxr-post .row.chemeleon .mae-val {
    color: var(--magenta);
    font-weight: 650;
  }.pxr-post .chips {
    display: flex;
    gap: 0.3rem;
    justify-content: flex-end;
    margin-top: 0.15rem;
  }.pxr-post .chip {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.02em;
    padding: 0.05rem 0.36rem;
    border-radius: 3px;
    border: 1px solid var(--line);
    color: var(--ink-soft);
  }.pxr-post .chip.on {
    background: color-mix(in srgb, var(--accent) 22%, transparent);
    border-color: var(--accent);
    color: var(--accent-ink);
    font-weight: 600;
  }.pxr-post .callout {
    font-size: 0.88rem;
    color: var(--ink-soft);
    border-left: 2px solid var(--sage);
    padding: 0.15rem 0 0.15rem 0.9rem;
  }.pxr-post .callout strong { color: var(--ink); }.pxr-post .support-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.4rem;
  }
  @media (max-width: 620px) {.pxr-post .support-grid { grid-template-columns: 1fr; }.pxr-post .row { grid-template-columns: 10rem 1fr 3.6rem; }.pxr-post .row-label { font-size: 0.78rem; }.pxr-post .axis-ref { grid-template-columns: 10rem 1fr 3.6rem; }
  }.pxr-post .limitation-list {
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
  }.pxr-post .limitation {
    border-left: 2px solid var(--line);
    padding: 0.1rem 0 0.1rem 0.9rem;
  }.pxr-post .limitation h3 {
    font-size: 0.94rem;
    margin: 0 0 0.3rem;
    color: var(--ink);
  }.pxr-post .limitation p {
    font-size: 0.88rem;
    color: var(--ink-soft);
    margin: 0;
  }.pxr-post .scatter-wrap {
    background: var(--bg-raised);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
  }.pxr-post .scatter-svg {
    width: 100%;
    height: auto;
    display: block;
  }.pxr-post .scatter-grid {
    stroke: var(--line);
    stroke-width: 1;
  }.pxr-post .scatter-axis {
    stroke: var(--ink-soft);
    stroke-width: 1;
  }.pxr-post .scatter-tick {
    font-family: var(--mono);
    font-size: 9px;
    font-variant-numeric: tabular-nums;
    fill: var(--ink-soft);
  }.pxr-post .scatter-axis-label {
    font-family: var(--sans);
    font-size: 10px;
    letter-spacing: 0.03em;
    fill: var(--ink-soft);
  }.pxr-post .scatter-points circle {
    fill: color-mix(in srgb, var(--plum) 55%, transparent);
  }.pxr-post .calib-diagonal {
    stroke: var(--ink-soft);
    stroke-width: 1;
    stroke-dasharray: 4 3;
  }.pxr-post .calib-area {
    fill: color-mix(in srgb, var(--target) 22%, transparent);
    stroke: none;
  }.pxr-post .calib-curve {
    fill: none;
    stroke: var(--target);
    stroke-width: 2;
  }.pxr-post .calib-points circle {
    fill: var(--target);
  }.pxr-post .mini-chart {
    background: var(--bg-raised);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
  }.pxr-post .mini-title {
    font-size: 0.92rem;
    font-weight: 650;
    margin: 0 0 0.9rem;
  }.pxr-post .mini-row {
    display: grid;
    grid-template-columns: 5.2rem 1fr 3.4rem;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }.pxr-post .mini-row:last-child { margin-bottom: 0; }.pxr-post .mini-label {
    font-size: 0.8rem;
    color: var(--ink);
  }.pxr-post .mini-track {
    height: 1.1rem;
    background: var(--track);
    border-radius: 3px;
    position: relative;
    overflow: hidden;
  }.pxr-post .mini-bar {
    position: absolute;
    inset: 0 auto 0 0;
    border-radius: 3px;
  }.pxr-post .mini-tick {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 2px;
    background: var(--ink);
  }.pxr-post .mini-val {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.78rem;
    text-align: right;
  }.pxr-post table.grid-table {
    border-collapse: collapse;
    font-size: 0.85rem;
    width: 100%;
  }.pxr-post table.grid-table caption {
    text-align: left;
    font-size: 0.92rem;
    font-weight: 650;
    margin-bottom: 0.7rem;
  }.pxr-post table.grid-table th, .pxr-post table.grid-table td {
    border: 1px solid var(--line);
    padding: 0.5rem 0.7rem;
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    text-align: center;
  }.pxr-post table.grid-table th:first-child, .pxr-post table.grid-table td:first-child {
    text-align: left;
    font-family: var(--sans);
  }.pxr-post table.grid-table td.best {
    background: color-mix(in srgb, var(--accent) 20%, transparent);
    font-weight: 650;
    color: var(--accent-ink);
  }.pxr-post footer {
    border-top: 1px solid var(--line);
    padding-top: 1.6rem;
    font-size: 0.82rem;
    color: var(--ink-soft);
    font-family: var(--mono);
  }

  .pxr-post { padding: 0; max-width: none; margin: 0; }
</style>"""

# chart-panel wrapper, byte-identical to the hand-authored cards
CHART_OPEN = """<div class="pxr-post">
<div class="chart">
 <div class="axis-ref">
  <div>
  </div>
  <div class="axis-ticks">
   <span>
    0.30
   </span>
   <span>
    0.50
   </span>
   <span>
    0.70
   </span>
  </div>
  <div>
  </div>
 </div>
 <div class="rows">
"""
CHART_CLOSE = """ </div>
</div>
</div>"""

# role -> (row css class, bar css class, inline bar background)
ROLE_STYLE: dict[str, tuple[str, str, str]] = {
    "winner": ("winner", "win", ""),
    "floor": ("floor", "floor-bar", ""),
    "single_best": ("single_best", "", "var(--blue)"),
    "sweepbest": ("sweepbest", "", "var(--sage)"),
    "chemeleon": ("chemeleon", "", "var(--magenta)"),
    "context": ("context", "context-bar", ""),
    "target": ("target", "target-bar", ""),
    "plain": ("", "", ""),
}


def pct(mae: float) -> float:
    """Map an MAE onto the shared 0.30-0.70 axis as a percentage."""
    return (mae - AXIS_LOW) / (AXIS_HIGH - AXIS_LOW) * 100.0


def fmt_pct(value: float) -> str:
    """Format an axis percentage to one decimal, matching the hand-authored cards."""
    return f"{value:.1f}"


# ---------------------------------------------------------------------------
# data loading and categories
# ---------------------------------------------------------------------------


def load_configs() -> pd.DataFrame:
    """Aggregate the run index to one row per config over the real seeds.

    Returns a frame indexed by ``base_dir`` carrying the resolved spec columns
    plus ``mae_mean``, ``mae_min``, ``mae_max`` and ``n_seeds`` over the seeds
    that produced a metric. Configs with no evaluated seed are dropped, which is
    how the 17 unevaluated members of the GNN sweep fall away on their own.
    """
    runs = pd.read_parquet(RESULTS_PATH)
    # keep the canonical 5-seed sweep runs, dropping stray unpinned duplicates;
    # additionally admit the anvil-recipe baseline, which is a single run with no
    # seed sweep (seed=UNPINNED_SEED) and so would otherwise fall through the filter
    runs = runs[runs["seed"].isin(SEEDS) | (runs["spec_source"] == "anvil_recipe")]
    scored = runs[runs["mae"].notna()].copy()

    spec_cols = [
        "encoder_family",
        "encoder_init",
        "encoder_target",
        "has_embedding",
        "has_readout",
        "has_descriptors",
        "descriptor_sources",
        "descriptor_pca_width",
        "embedding_pca_dim",
        "regressor",
        "train_data",
        "n_features",
        "freeze_epochs",
    ]
    agg = scored.groupby("base_dir").agg(
        **{col: (col, "first") for col in spec_cols},
        mae_mean=("mae", "mean"),
        mae_min=("mae", "min"),
        mae_max=("mae", "max"),
        n_seeds=("mae", "count"),
    )
    agg["category"] = [classify(row) for _, row in agg.iterrows()]
    return agg


def classify(spec: pd.Series) -> str:
    """Assign a config to a category from its spec columns alone.

    Categories drive the computed role extrema (floor, single_best, sweepbest)
    and the never-drawn accounting; they never decide which rows appear.
    """
    reg = spec["regressor"]
    is_end2end = reg == "N/A" or pd.isna(reg)
    ingredients = (
        int(bool(spec["has_embedding"]))
        + int(bool(spec["has_readout"]))
        + int(bool(spec["has_descriptors"]))
    )

    # single-stage CheMeleon baseline: the CheMeleon-init encoder trained
    # end-to-end straight to pEC50 (no log2FC pretraining, no downstream
    # regressor). Checked before the gnn branch, which would otherwise claim it,
    # and gated on is_end2end so the tabular pEC50-on-CheMeleon variants (which
    # regress a frozen embedding) do not fall in here.
    if (
        is_end2end
        and spec["encoder_init"] == "chemeleon_pretrained"
        and spec["encoder_target"] == "pec50"
    ):
        return "chemeleon_baseline"

    # end-to-end GNN readout heads: the concatenation sweep winner and the e4 encoders
    if (
        is_end2end
        and not bool(spec["has_readout"])
        and not bool(spec["has_descriptors"])
    ):
        return "gnn"

    # tabular regressors on a canonical featureset
    if reg == "tabpfn" and spec["encoder_target"] != "pec50":
        if ingredients == 1:
            return "single_ingredient"
        canonical_desc = (not bool(spec["has_descriptors"])) or (
            spec["descriptor_sources"] == "mordred"
            and spec["descriptor_pca_width"] == 128
        )
        if ingredients >= 2 and canonical_desc:
            return "combination"

    return "other"


# ---------------------------------------------------------------------------
# selection by spec predicate
# ---------------------------------------------------------------------------

# the spec columns that make two runs the same experiment; runs identical across
# all of these are launch-duplicates and collapse to their lowest-MAE representative
SIGNATURE_COLS = [
    "encoder_family",
    "encoder_init",
    "encoder_target",
    "has_embedding",
    "has_readout",
    "has_descriptors",
    "descriptor_sources",
    "descriptor_pca_width",
    "embedding_pca_dim",
    "regressor",
    "train_data",
    "n_features",
    "freeze_epochs",
    "_calibrated",
]


def match(configs: pd.DataFrame, predicate: dict[str, object]) -> pd.DataFrame:
    """Return every config matching each ``column == value`` in the predicate."""
    mask = pd.Series(True, index=configs.index)
    for col, want in predicate.items():
        mask &= configs[col] == want
    return configs[mask]


def dedup_min(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse spec-identical launch-duplicates to their lowest-MAE representative.

    A config launched twice produces two rows with the same spec signature and
    different MAEs; the better run draws and its twin stays unplotted, matching
    what a lowest-MAE predicate resolution did when panels named single slots.
    """
    keep = (
        frame.sort_values("mae_mean")
        .reset_index()
        .drop_duplicates(subset=SIGNATURE_COLS, keep="first")
    )
    return frame.loc[keep["base_dir"]]


# ---------------------------------------------------------------------------
# panel composition and the declared label layer (the only editorial surface)
# ---------------------------------------------------------------------------


@dataclass
class Native:
    """The swept run family a panel is built around.

    Every config matching ``pred`` is drawn in ascending-MAE order, minus any
    ``exclude`` variant that sits outside the panel's story, with launch
    duplicates collapsed. ``sweep_axis`` names the column or columns the family
    is meant to vary along; the resolver asserts it genuinely varies so a panel
    cannot silently collapse to a single bar.
    """

    pred: dict[str, object]
    sweep_axis: str | list[str]
    exclude: list[dict[str, object]] = field(default_factory=list)


@dataclass
class Context:
    """A comparison row borrowed from outside the swept family.

    Either a computed role holder (``role``) or a predicate match (``pred`` with
    ``mode`` "min" for the lowest-MAE match, "all" for every match). ``native``
    marks whether the row draws solid (fresh in this panel) or hatched (reused
    from another panel).
    """

    role: str | None = None
    pred: dict[str, object] | None = None
    mode: str = "min"
    native: bool = False


@dataclass
class Head:
    """A pinned row above the sort boundary: a reference id or the global winner."""

    ref_id: str | None = None
    winner: bool = False


@dataclass
class PanelSpec:
    """A figure declared by what it compares, not by an enumerated row list."""

    out_name: str
    native: Native | None = None
    context: list[Context] = field(default_factory=list)
    head: list[Head] = field(default_factory=list)
    divider_after: int = -1
    winner_in_body: bool = False  # global winner sorts into the body, not the head
    accent_native_min: bool = False  # style the family's own minimum as a winner
    kind: str = "bars"  # 'bars' | 'mini' | 'calibration'
    title: str = ""  # mini-chart heading


@dataclass
class Row:
    """One resolved, ordered draw row, ready to render."""

    key: str
    label: str
    sub: str
    role: str
    native: bool
    is_cfg: bool
    divider: bool
    data: pd.Series
    source: str = (
        "panel"  # provenance that picks the label registry: reference | role | panel
    )


# reference rows carry a fixed role from their identity, not the data extrema
REF_ROLES = {"n283t_ensemble": "context", "n283t_target": "target"}

# The declared label layer. Composition (which runs draw, in what order, in what
# role, solid or reused) is computed; the text is not, because a label encodes
# distinctions the spec columns do not (a column count, "head only" versus
# "encoder + head"). The one text a row shows is fixed by how it entered its
# panel, so there is exactly one lookup path and no overrides to keep in sync:
#   - a reference row reads from REF_LABELS, keyed by reference id
#   - a row borrowed as a computed role (the pinned winner, or a floor / single-
#     ingredient / CheMeleon comparison) reads from ROLE_LABELS, keyed by role,
#     so the same borrow reads the same everywhere it appears
#   - a native family member or panel-specific constituent reads from
#     PANEL_LABELS[panel], keyed by run, so a panel names each run by the axis it
#     sweeps (a regressor name in the regressor panel, a featureset in the
#     combination panel), even for a run that is native in two panels
# A run therefore reads by its role where borrowed and by its identity where
# native; the two never collide because provenance, not the run alone, selects
# the registry.
REF_LABELS: dict[str, tuple[str, str]] = {
    "n283t_ensemble": ("N283T ensemble", "context only, not our pipeline"),
    "n283t_target": ("N283T report target", "context only, not our pipeline"),
}

ROLE_LABELS: dict[str, tuple[str, str]] = {
    "winner": (
        "Our best overall",
        "CheMeleon embedding + log<sub>2</sub>FC readout, TabICL 2.1.1",
    ),
    "floor": (
        "Concatenation architecture",
        "best of 18-config freeze/width/clip sweep",
    ),
    "single_best": ("Best single-ingredient", "log<sub>2</sub>FC embedding"),
    "chemeleon": ("CheMeleon baseline", "encoder + head, dose-response only"),
}

PANEL_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "figure-01.html": {
        "freeze2_hd512_clipoff": (
            "Concatenation architecture",
            "best of 18-config freeze/width/clip sweep",
        ),
        "e4_finetune": (
            "Fine-tuned log<sub>2</sub>FC encoder",
            "encoder + head, dose-response + primary screen",
        ),
        "e4_finetune_drc_only": (
            "Fine-tuned log<sub>2</sub>FC encoder",
            "encoder + head, dose-response only",
        ),
        "e4_frozen": (
            "Frozen log<sub>2</sub>FC encoder",
            "head only, dose-response + primary screen",
        ),
        "e4_frozen_drc_only": (
            "Frozen log<sub>2</sub>FC encoder",
            "head only, dose-response only",
        ),
    },
    "figure-02.html": {
        "tabpfn_embed_only_no_readout": (
            "log<sub>2</sub>FC embedding",
            "Chemprop encoder, log<sub>2</sub>FC-trained",
        ),
        "tabpfn_chemeleon_embed_only": (
            "CheMeleon embedding",
            "pretrained, 2048 → 256 columns",
        ),
        "tabpfn_rdkit_only_pca128_no_embed_no_readout": (
            "RDKit descriptors",
            "217 → 128 columns",
        ),
        "tabpfn_readout_only_no_embed": (
            "log<sub>2</sub>FC readout",
            "Chemprop head, log<sub>2</sub>FC-trained",
        ),
        "tabpfn_mordred_only_pca128_no_embed_no_readout": (
            "Mordred descriptors",
            "~1600 → 128 columns",
        ),
    },
    "figure-03.html": {
        "tabpfn_chemeleon_readout_descriptors": (
            "Embedding + log<sub>2</sub>FC readout + descriptors",
            "CheMeleon embedding",
        ),
        "tabpfn_chemeleon_readout_only": (
            "Embedding + log<sub>2</sub>FC readout",
            "CheMeleon embedding",
        ),
        "tabpfn_readout_mordred_pca128_no_embed": (
            "log<sub>2</sub>FC readout + descriptors",
            "",
        ),
        "tabpfn_embed_readout_mordred_pca128": (
            "Embedding + log<sub>2</sub>FC readout + descriptors",
            "log<sub>2</sub>FC embedding",
        ),
        "tabpfn_embed_mordred_pca128_no_readout": (
            "Embedding + descriptors",
            "log<sub>2</sub>FC embedding",
        ),
        "tabpfn_small_embed": (
            "Embedding + log<sub>2</sub>FC readout",
            "log<sub>2</sub>FC embedding",
        ),
        "tabpfn_chemeleon_descriptors_only": (
            "Embedding + descriptors",
            "CheMeleon embedding",
        ),
        # the two fresh constituents read like their figure-02 identities, minus
        # the "alone" and negation text a single-ingredient row already implies
        "tabpfn_readout_only_no_embed": ("log<sub>2</sub>FC readout", ""),
        "tabpfn_mordred_only_pca128_no_embed_no_readout": ("Mordred descriptors", ""),
    },
    "figure-04.html": {
        "tabicl_chemeleon_readout_descriptors": ("TabICL v2.1.1", ""),
        "tabpfn-v3_chemeleon_readout_descriptors": ("TabPFN v3", ""),
        "tabpfn_chemeleon_readout_descriptors": ("TabPFN v2.5", ""),
        "tabpfn-v2.6_chemeleon_readout_descriptors": ("TabPFN v2.6", ""),
        "tabfm_n32_chemeleon_readout_descriptors": ("TabFM v1.0.0", "max_num_rows=500"),
        "lgbm_chemeleon_readout_descriptors": ("LightGBM", ""),
        "xgboost_chemeleon_readout_descriptors": ("XGBoost", ""),
    },
    "figure-05.html": {
        "tabpfn_embed_readout_mordred_pca128": ("Mordred, 128", ""),
        "tabpfn_concat_small_embed": ("RDKit + Mordred, 128", ""),
        "tabpfn_embed_readout_mordred_pca256": ("Mordred, 256", ""),
        "tabpfn_concat_pca256": ("RDKit + Mordred, 256", ""),
        "tabpfn_concat_pca64": ("RDKit + Mordred, 64", ""),
        "tabpfn_embed_readout_mordred_pca64": ("Mordred, 64", ""),
    },
    "figure-06.html": {
        "tabicl_chemeleon_readout_only": (
            "Uncalibrated",
            "CheMeleon embedding + log<sub>2</sub>FC readout, TabICL 2.1.1",
        ),
        "tabicl_chemeleon_readout_only_calibrated": (
            "Calibrated",
            "isotonic map fit on 5-fold OOF predictions",
        ),
    },
}


def label_for(source: str, role: str, out_name: str, key: str) -> tuple[str, str]:
    """Resolve a row's label and sub from its provenance, one registry per source."""
    if source == "reference":
        return REF_LABELS[key]
    if source == "role":
        return ROLE_LABELS[role]
    panel = PANEL_LABELS.get(out_name, {})
    if key not in panel:
        raise ValueError(f"{out_name}: no panel label declared for run {key!r}")
    return panel[key]


def build_panels() -> list[PanelSpec]:
    """Return the six panels in output order, each declared by what it compares.

    A panel names a swept family (a category or spec predicate plus the axis it
    varies along) and a few comparison rows drawn from computed roles or
    predicates. Membership, order, role color, the reused-bar flag, and the
    reference divider all fall out of the data; only the family, its axis, and
    the context rows are editorial.
    """
    head3 = [
        Head(ref_id="n283t_ensemble"),
        Head(ref_id="n283t_target"),
        Head(winner=True),
    ]
    head2 = [Head(ref_id="n283t_ensemble"), Head(ref_id="n283t_target")]

    # graph baselines: the GNN encoder sweep, against the CheMeleon reference
    panel_gnn = PanelSpec(
        "figure-01.html",
        native=Native({"category": "gnn"}, sweep_axis="train_data"),
        context=[Context(role="chemeleon", native=True)],
        head=head3,
        divider_after=2,
    )

    # single ingredients: each lone featureset, against the floor and CheMeleon
    panel_ingredients = PanelSpec(
        "figure-02.html",
        native=Native(
            {"category": "single_ingredient"},
            sweep_axis=[
                "has_embedding",
                "has_readout",
                "has_descriptors",
                "descriptor_sources",
            ],
        ),
        context=[Context(role="floor"), Context(role="chemeleon")],
        head=head3,
        divider_after=2,
    )

    # combining ingredients: the two-embedding combination family. The third
    # embedding hybrid (CheMeleon-init and then log2FC-trained) sits outside the
    # story and is excluded; the embedding-alone rung is the single-ingredient
    # best, so it enters as the reused single_best pointer, not a fresh bar.
    panel_combine = PanelSpec(
        "figure-03.html",
        native=Native(
            {"category": "combination"},
            sweep_axis=[
                "encoder_init",
                "has_embedding",
                "has_readout",
                "has_descriptors",
            ],
            exclude=[
                {"encoder_init": "chemeleon_pretrained", "encoder_target": "log2fc"}
            ],
        ),
        context=[
            Context(
                pred={
                    "category": "single_ingredient",
                    "encoder_init": "scratch",
                    "has_readout": True,
                    "has_descriptors": False,
                },
                mode="all",
                native=True,
            ),
            Context(
                pred={
                    "category": "single_ingredient",
                    "encoder_init": "scratch",
                    "descriptor_sources": "mordred",
                },
                mode="all",
                native=True,
            ),
            Context(role="single_best"),
            Context(role="floor"),
            Context(role="chemeleon"),
        ],
        head=head3,
        divider_after=2,
    )

    # regressor sweep: the featureset fixed at figure-03's 386-column combination,
    # varying only the tabular regressor; the family's own minimum is accented and
    # the global winner drops into the sorted body for scale
    panel_regressor = PanelSpec(
        "figure-04.html",
        native=Native(
            {
                "encoder_init": "chemeleon_pretrained",
                "has_embedding": True,
                "has_readout": True,
                "has_descriptors": True,
                "n_features": 386,
            },
            sweep_axis="regressor",
        ),
        context=[
            Context(role="single_best"),
            Context(role="floor"),
            Context(role="chemeleon"),
        ],
        head=head2,
        divider_after=1,
        winner_in_body=True,
        accent_native_min=True,
    )

    # PCA-width mini-chart: embedding + readout + descriptors, scratch/log2FC,
    # varying the descriptor source and PCA width
    panel_pca = PanelSpec(
        "figure-05.html",
        native=Native(
            {
                "has_embedding": True,
                "has_readout": True,
                "has_descriptors": True,
                "encoder_init": "scratch",
                "encoder_target": "log2fc",
                "regressor": "tabpfn",
            },
            sweep_axis=["descriptor_sources", "descriptor_pca_width"],
        ),
        kind="mini",
        title="Embedding + log<sub>2</sub>FC readout (fixed, raw) + descriptors, by PCA width",
    )

    # calibration: the global winner and its isotonic-calibrated counterpart
    panel_calibration = PanelSpec("figure-06.html", kind="calibration")

    return [
        panel_gnn,
        panel_ingredients,
        panel_combine,
        panel_regressor,
        panel_pca,
        panel_calibration,
    ]


# ---------------------------------------------------------------------------
# role verification against computed extrema
# ---------------------------------------------------------------------------


def role_holders(configs: pd.DataFrame) -> dict[str, str]:
    """Compute the base_dir that legitimately holds each singular role."""

    def argmin_in(category: str) -> str:
        members = configs[configs["category"] == category]
        if members.empty:
            raise ValueError(f"category {category!r} has no config")
        return str(members["mae_mean"].idxmin())

    return {
        "winner": str(configs["mae_mean"].idxmin()),
        "floor": argmin_in("gnn"),
        "single_best": argmin_in("single_ingredient"),
        "sweepbest": argmin_in("combination"),
        "chemeleon": argmin_in("chemeleon_baseline"),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_config_row(row: Row) -> str:
    """Render one config bar with computed width, whisker, and role color."""
    stats = row.data
    row_cls, bar_cls, bg = ROLE_STYLE[row.role]
    classes = ["row"]
    if row_cls:
        classes.append(row_cls)
    if row.divider:
        classes.append("ref-divider")
    reused = " reused" if not row.native else ""

    width = fmt_pct(pct(stats["mae_mean"]))
    bar_classes = "bar"
    if bar_cls:
        bar_classes += f" {bar_cls}"
    bar_classes += reused
    style = f"width: {width}%;"
    if bg:
        style += f" background-color: {bg};"

    left = fmt_pct(pct(stats["mae_min"]))
    span = fmt_pct(pct(stats["mae_max"]) - pct(stats["mae_min"]))
    whisker = f'<div class="err-whisker" style="left: {left}%; width: {span}%;"></div>'

    label = _label_html(row.label, row.sub)
    val = f"{stats['mae_mean']:.4f}"
    return (
        f'      <div class="{" ".join(classes)}">\n'
        f'        <div class="row-label">{label}</div>\n'
        f'        <div class="track"><div class="{bar_classes}" style="{style}"></div>{whisker}</div>\n'
        f'        <div class="mae-val">{val}</div>\n'
        f"      </div>"
    )


def render_reference_row(row: Row) -> str:
    """Render an external reference bar: width from its MAE, no whisker."""
    ref = row.data
    row_cls, bar_cls, _ = ROLE_STYLE[row.role]
    classes = ["row", row_cls]
    if row.divider:
        classes.append("ref-divider")
    width = fmt_pct(pct(float(ref["mae"])))
    label = _label_html(row.label, row.sub)
    return (
        f'      <div class="{" ".join(classes)}">\n'
        f'        <div class="row-label">{label}</div>\n'
        f'        <div class="track"><div class="bar {bar_cls}" style="width: {width}%;"></div></div>\n'
        f'        <div class="mae-val">{ref["mae_display"]}</div>\n'
        f"      </div>"
    )


def _label_html(label: str, sub: str) -> str:
    """Compose the label cell, omitting the sub span when there is no sub."""
    if sub:
        return f'{label}<span class="sub">{sub}</span>'
    return label


# ---------------------------------------------------------------------------
# panel resolution: composition derived from the data, labels looked up
# ---------------------------------------------------------------------------


def _series(frame: pd.DataFrame, key: str) -> pd.Series:
    """One row of a frame by label as a Series (scalar-label ``.loc``)."""
    return cast(pd.Series, frame.loc[key])


def _make_row(
    spec: PanelSpec,
    key: str,
    role: str,
    native: bool,
    is_cfg: bool,
    data: pd.Series,
    source: str,
) -> Row:
    """Build a draw row, looking up its label from its provenance source."""
    label, sub = label_for(source, role, spec.out_name, key)
    return Row(
        key=key,
        label=label,
        sub=sub,
        role=role,
        native=native,
        is_cfg=is_cfg,
        divider=False,
        data=data,
        source=source,
    )


def _axis_cols(sweep_axis: str | list[str]) -> list[str]:
    """The sweep axis as a list, whether declared as one column or several."""
    return [sweep_axis] if isinstance(sweep_axis, str) else list(sweep_axis)


def _native_family(native: Native, configs: pd.DataFrame) -> pd.DataFrame:
    """The swept family: matches minus excludes, duplicates collapsed, MAE-sorted.

    Asserts the sweep axis genuinely varies so a panel cannot silently collapse
    to a single bar when a predicate is too narrow.
    """
    frame = match(configs, native.pred)
    for excluded in native.exclude:
        frame = frame.drop(index=match(frame, excluded).index)
    frame = dedup_min(frame)
    if not any(frame[col].nunique() > 1 for col in _axis_cols(native.sweep_axis)):
        raise ValueError(
            f"sweep axis {native.sweep_axis!r} does not vary across the family"
        )
    return frame.sort_values("mae_mean")


def _context_keys(
    ctx: Context, configs: pd.DataFrame, holders: dict[str, str], out_name: str
) -> list[str]:
    """The base_dir(s) a context row draws: a role holder, or predicate match(es)."""
    if ctx.role is not None:
        return [holders[ctx.role]]
    if ctx.pred is None:
        raise ValueError(f"{out_name}: context row declares neither role nor pred")
    hits = match(configs, ctx.pred)
    if hits.empty:
        raise ValueError(f"{out_name}: context predicate {ctx.pred} matched nothing")
    if ctx.mode == "all":
        return [str(k) for k in hits.index]
    return [str(hits["mae_mean"].idxmin())]


def _check_labels_unique(rows: list[Row], out_name: str) -> None:
    """Reject a panel that would render two rows with identical label text."""
    seen: set[str] = set()
    for row in rows:
        rendered = _label_html(row.label, row.sub)
        if rendered in seen:
            raise ValueError(f"duplicate rendered label {rendered!r} in {out_name}")
        seen.add(rendered)


def _resolve_calibration(
    spec: PanelSpec, configs: pd.DataFrame, holders: dict[str, str]
) -> list[Row]:
    """The global winner and its isotonic-calibrated twin, in that fixed order."""
    winner = holders["winner"]
    rows: list[Row] = []
    for idx, key in enumerate((winner, f"{winner}_calibrated")):
        rows.append(
            _make_row(
                spec,
                key,
                role="winner" if idx == 0 else "plain",
                native=True,
                is_cfg=True,
                data=_series(configs, key),
                source="panel",
            )
        )
        rows[-1].divider = idx == spec.divider_after
    return rows


def resolve(
    spec: PanelSpec,
    configs: pd.DataFrame,
    references: pd.DataFrame,
    holders: dict[str, str],
) -> list[Row]:
    """Resolve a bars or calibration panel to its ordered draw rows.

    Head rows (references and, when pinned, the global winner) stay above the
    sort boundary; the swept family and the borrowed context rows sort together
    by MAE below it. Role color, the reused flag, and the divider all follow from
    the data; only label text is looked up.
    """
    if spec.kind == "calibration":
        return _resolve_calibration(spec, configs, holders)
    if spec.native is None:
        raise ValueError(f"{spec.out_name}: a bars panel needs a native family")

    global_winner = holders["winner"]

    def role_of(key: str) -> str:
        for role, holder in holders.items():
            if holder == key:
                return role
        return "plain"

    # pinned head: reference bars and, unless it sorts into the body, the winner
    head: list[Row] = []
    for pinned in spec.head:
        if pinned.ref_id is not None:
            head.append(
                _make_row(
                    spec,
                    pinned.ref_id,
                    REF_ROLES[pinned.ref_id],
                    True,
                    False,
                    _series(references, pinned.ref_id),
                    source="reference",
                )
            )
        elif pinned.winner:
            head.append(
                _make_row(
                    spec,
                    global_winner,
                    "winner",
                    True,
                    True,
                    _series(configs, global_winner),
                    source="role",
                )
            )

    # swept family, then borrowed context rows, all sorted together by MAE
    family = _native_family(spec.native, configs)
    body: list[Row] = [
        _make_row(
            spec,
            str(key),
            role_of(str(key)),
            True,
            True,
            _series(configs, str(key)),
            source="panel",
        )
        for key in family.index
    ]
    if spec.accent_native_min:
        min(body, key=lambda row: float(row.data["mae_mean"])).role = "winner"
    if spec.winner_in_body:
        body.append(
            _make_row(
                spec,
                global_winner,
                "winner",
                False,
                True,
                _series(configs, global_winner),
                source="role",
            )
        )

    # a borrowed context row reads by its role (role source); a predicate context
    # row reads by its native identity in this panel (panel source)
    for ctx in spec.context:
        source = "role" if ctx.role is not None else "panel"
        for key in _context_keys(ctx, configs, holders, spec.out_name):
            body.append(
                _make_row(
                    spec,
                    key,
                    role_of(key),
                    ctx.native,
                    True,
                    _series(configs, key),
                    source=source,
                )
            )
    body.sort(key=lambda row: float(row.data["mae_mean"]))

    rows = head + body
    for idx, row in enumerate(rows):
        row.divider = idx == spec.divider_after
    _check_labels_unique(rows, spec.out_name)
    return rows


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_panel(
    spec: PanelSpec,
    configs: pd.DataFrame,
    references: pd.DataFrame,
    holders: dict[str, str],
) -> str:
    """Render a full bars or calibration panel to its standalone card HTML."""
    rendered = [
        render_config_row(row) if row.is_cfg else render_reference_row(row)
        for row in resolve(spec, configs, references, holders)
    ]
    body = "\n\n".join(rendered)
    return f"{CSS_STYLE}\n{CHART_OPEN}{body}\n{CHART_CLOSE}"


def render_mini(spec: PanelSpec, configs: pd.DataFrame) -> str:
    """Render the PCA-width mini-chart, sorted by MAE with the minimum accented."""
    if spec.native is None:
        raise ValueError(f"{spec.out_name}: a mini panel needs a native family")
    family = _native_family(spec.native, configs)
    resolved = [
        (label_for("panel", "plain", spec.out_name, str(key))[0], configs.loc[key])
        for key in family.index
    ]
    floor = min(float(cfg["mae_mean"]) for _, cfg in resolved)

    grid = 'style="grid-template-columns: 9rem 1fr 3.4rem;"'
    lines = [
        '<div class="pxr-post">',
        '    <div class="mini-chart">',
        f'      <p class="mini-title">{spec.title}</p>',
        f'      <div class="axis-ref" {grid}><div></div><div class="axis-ticks"><span>0.30</span><span>0.50</span><span>0.70</span></div><div></div></div>',
    ]
    for label, cfg in resolved:
        width = fmt_pct(pct(float(cfg["mae_mean"])))
        bg = "var(--accent)" if float(cfg["mae_mean"]) == floor else "var(--gray)"
        left = fmt_pct(pct(float(cfg["mae_min"])))
        span = fmt_pct(pct(float(cfg["mae_max"])) - pct(float(cfg["mae_min"])))
        lines.append(f'      <div class="mini-row" {grid}>')
        lines.append(f'        <span class="mini-label">{label}</span>')
        lines.append(
            f'        <div class="mini-track"><div class="mini-bar" style="width: {width}%; background: {bg};"></div>'
            f'<div class="err-whisker" style="left: {left}%; width: {span}%;"></div></div>'
        )
        lines.append(
            f'        <span class="mini-val">{float(cfg["mae_mean"]):.4f}</span>'
        )
        lines.append("      </div>")
    lines.append("    </div>")
    lines.append("</div>")
    return f"{CSS_STYLE}\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# calibration variant handling
# ---------------------------------------------------------------------------


def attach_calibrated(configs: pd.DataFrame) -> pd.DataFrame:
    """Add a ``_calibrated`` boolean so the calibration slot resolves by spec.

    Calibration is a post-hoc isotonic map; it does not touch the featureset, so
    the calibrated run shares its parent's spec and is only distinguishable by
    the ``_calibrated`` suffix on its base_dir.
    """
    configs = configs.copy()
    configs["_calibrated"] = [
        str(name).endswith("_calibrated") for name in configs.index
    ]
    return configs


def main() -> None:
    """Render figure-01 .. figure-06 from the run index and reference table."""
    configs = attach_calibrated(load_configs())
    references = pd.read_csv(REFERENCE_PATH).set_index("id")
    holders = role_holders(configs)

    panels = build_panels()
    OUT_DIR.mkdir(exist_ok=True)

    # the calibration panel resolves on the _calibrated column; every other panel
    # selects non-calibrated runs, so drop the calibrated twin there
    plain_configs = configs[~configs["_calibrated"]]

    for spec in panels:
        source = configs if spec.kind == "calibration" else plain_configs
        if spec.kind == "mini":
            html = render_mini(spec, source)
        else:
            html = render_panel(spec, source, references, holders)
        (OUT_DIR / spec.out_name).write_text(html)
        print(f"wrote {spec.out_name}")


if __name__ == "__main__":
    main()
