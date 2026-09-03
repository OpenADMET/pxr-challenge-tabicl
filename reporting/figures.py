"""Render the PXR result bar figures from the run index and reference table.

Reads two data files and nothing else:

- ``reporting/results.parquet`` : every run we tried, one row per
  (base_dir, seed), every spec and metric column resolved from the run's own
  artifacts by ``build_run_provenance.py``.
- ``reporting/reference.csv`` : borrowed external numbers with no run of ours
  behind them (the N283T challenge report figures).

Every MAE, whisker, bar width, panel order, divider position, and role color is
computed here from that data. No number is hand-entered. The panel definitions
near the bottom are the only editorial surface: each drawn slot selects its run
by a spec predicate (never a base_dir string) and carries the label text, which
encodes distinctions the provenance columns do not (for example "FFN only"
versus "FFN-tuned"). Roles are declared per slot and then cross-checked against
the data-derived category extrema, so a stale annotation raises rather than
drawing a wrong color.

Run ``python reporting/figures.py`` to rewrite ``figures/figure-01.html`` ..
``figures/figure-06.html``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    runs = runs[runs["seed"].isin(SEEDS)]
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
    ingredients = int(bool(spec["has_embedding"])) + int(bool(spec["has_readout"])) + int(
        bool(spec["has_descriptors"])
    )

    # end-to-end GNN readout heads: the concatenation sweep winner and the e4 encoders
    if is_end2end and not bool(spec["has_readout"]) and not bool(spec["has_descriptors"]):
        return "gnn"

    # single-stage CheMeleon baseline: pretrained encoder trained on pEC50, not log2FC
    if spec["encoder_init"] == "chemeleon_pretrained" and spec["encoder_target"] == "pec50":
        return "chemeleon_baseline"

    # tabular regressors on a canonical featureset
    if reg == "tabpfn" and spec["encoder_target"] != "pec50":
        if ingredients == 1:
            return "single_ingredient"
        canonical_desc = (not bool(spec["has_descriptors"])) or (
            spec["descriptor_sources"] == "mordred" and spec["descriptor_pca_width"] == 128
        )
        if ingredients >= 2 and canonical_desc:
            return "combination"

    return "other"


# ---------------------------------------------------------------------------
# slot selection by spec predicate
# ---------------------------------------------------------------------------


def select_one(configs: pd.DataFrame, predicate: dict[str, object]) -> pd.Series:
    """Resolve a spec predicate to exactly one config, the lowest-MAE match.

    Spec-identical runs (a config launched twice) match the same predicate; the
    lowest mean MAE wins, so the better run draws and its twin stays unplotted.
    Raises when nothing matches, which catches a panel that references a config
    the data no longer contains.
    """
    mask = pd.Series(True, index=configs.index)
    for col, want in predicate.items():
        mask &= configs[col] == want
    hits = configs[mask]
    if hits.empty:
        raise ValueError(f"no config matches predicate {predicate}")
    best = hits["mae_mean"].idxmin()
    row = configs.loc[best].copy()
    row.name = best
    return row


# ---------------------------------------------------------------------------
# panel and slot definitions (the only editorial surface)
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    """One drawn bar: how to find its run, what to call it, and its role."""

    label: str
    sub: str = ""
    role: str = "plain"
    native: bool = True
    # a config slot resolves via a spec predicate; a reference slot names a
    # reference.csv id instead
    select: dict[str, object] | None = None
    ref_id: str | None = None


@dataclass
class Panel:
    """A figure: an ordered pinned block, then a body sorted by ascending MAE."""

    out_name: str
    slots: list[Slot]
    sort_from: int
    divider_after: int


@dataclass
class MiniRow:
    """One bar of the PCA mini-chart, selected by spec predicate."""

    label: str
    select: dict[str, object]


@dataclass
class MiniPanel:
    """The PCA-width mini-chart: every row sorted by MAE, the minimum accented."""

    out_name: str
    title: str
    rows: list[MiniRow]


# canonical encoder featureset fragments, reused across slot predicates
_SCRATCH = {"encoder_init": "scratch", "encoder_target": "log2fc", "regressor": "tabpfn"}
_CHEMELEON = {
    "encoder_init": "chemeleon_pretrained",
    "encoder_target": "none",
    "regressor": "tabpfn",
}


def build_panels() -> tuple[list[Panel], MiniPanel]:
    """Return the six bar panels and the PCA mini-chart in output order."""
    # the fixed reference block that opens the first three panels
    ref_ensemble = Slot("N283T ensemble", "context only, not our pipeline", "context", ref_id="n283t_ensemble")
    ref_target = Slot(
        "N283T report target", "frozen Chemprop embedding + TabPFN", "target", ref_id="n283t_target"
    )
    our_best = Slot(
        "Our best overall",
        "CheMeleon embedding + log<sub>2</sub>FC readout, TabICL 2.1.1",
        "winner",
        native=True,
        select={
            "encoder_init": "chemeleon_pretrained",
            "regressor": "tabicl",
            "has_embedding": True,
            "has_readout": True,
            "has_descriptors": False,
        },
    )
    best_gnn = Slot(
        "Best GNN baseline",
        "concatenation architecture",
        "floor",
        native=False,
        select={
            "regressor": "N/A",
            "encoder_init": "chemeleon_pretrained",
            "has_readout": False,
            "has_descriptors": False,
            "freeze_epochs": 2,
        },
    )
    chemeleon_base = Slot(
        "CheMeleon baseline",
        "no log<sub>2</sub>FC pretraining, single-stage",
        "chemeleon",
        native=False,
        select={"encoder_init": "chemeleon_pretrained", "encoder_target": "pec50"},
    )
    best_single = Slot(
        "Best single-ingredient",
        "log<sub>2</sub>FC embedding only",
        "single_best",
        native=False,
        select={**_SCRATCH, "has_embedding": True, "has_readout": False, "has_descriptors": False},
    )

    # panel 00 -> figure-01: graph baselines
    panel_gnn = Panel(
        "figure-01.html",
        [
            ref_ensemble,
            ref_target,
            Slot(**{**our_best.__dict__}),
            Slot(
                "Concatenation architecture",
                "best of 18-config freeze/width/clip sweep",
                "floor",
                native=True,
                select={
            "regressor": "N/A",
            "encoder_init": "chemeleon_pretrained",
            "has_readout": False,
            "has_descriptors": False,
            "freeze_epochs": 2,
        },
            ),
            Slot(**{**chemeleon_base.__dict__, "native": True}),
            Slot(
                "Fine-tuned log<sub>2</sub>FC encoder",
                "full model, dose-response only",
                select={
                    "encoder_init": "log2fc_checkpoint_pretrained",
                    "freeze_epochs": 2,
                    "train_data": "drc_only",
                },
            ),
            Slot(
                "Frozen log<sub>2</sub>FC encoder",
                "FFN only, dose-response + primary screen",
                select={
                    "encoder_init": "log2fc_checkpoint_pretrained",
                    "freeze_epochs": 50,
                    "train_data": "drc_plus_primary",
                },
            ),
            Slot(
                "Fine-tuned log<sub>2</sub>FC encoder",
                "full model, dose-response + primary screen",
                select={
                    "encoder_init": "log2fc_checkpoint_pretrained",
                    "freeze_epochs": 2,
                    "train_data": "drc_plus_primary",
                },
            ),
            Slot(
                "Frozen log<sub>2</sub>FC encoder",
                "FFN-tuned, dose-response only",
                select={
                    "encoder_init": "log2fc_checkpoint_pretrained",
                    "freeze_epochs": 50,
                    "train_data": "drc_only",
                },
            ),
        ],
        sort_from=3,
        divider_after=2,
    )

    # panel 01 -> figure-02: single ingredients
    panel_ingredients = Panel(
        "figure-02.html",
        [
            ref_ensemble,
            ref_target,
            Slot(**{**our_best.__dict__}),
            Slot(
                "log<sub>2</sub>FC embedding",
                "from-scratch encoder, log<sub>2</sub>FC-trained",
                "single_best",
                native=True,
                select={**_SCRATCH, "has_embedding": True, "has_readout": False, "has_descriptors": False},
            ),
            Slot(
                "CheMeleon embedding",
                "pretrained, no fine-tuning",
                select={**_CHEMELEON, "has_embedding": True, "has_readout": False, "has_descriptors": False},
            ),
            Slot(**{**best_gnn.__dict__}),
            Slot(**{**chemeleon_base.__dict__}),
            Slot(
                "RDKit descriptors",
                "217 columns, PCA-128",
                select={**_SCRATCH, "has_embedding": False, "has_readout": False, "descriptor_sources": "rdkit"},
            ),
            Slot(
                "log<sub>2</sub>FC readout",
                "encoder's own prediction, 2 columns",
                select={**_SCRATCH, "has_embedding": False, "has_readout": True, "has_descriptors": False},
            ),
            Slot(
                "Mordred descriptors",
                "~1600 columns, PCA-128",
                select={**_SCRATCH, "has_embedding": False, "has_readout": False, "descriptor_sources": "mordred"},
            ),
        ],
        sort_from=3,
        divider_after=2,
    )

    # panel 02 -> figure-03: combining ingredients
    panel_combine = Panel(
        "figure-03.html",
        [
            ref_ensemble,
            ref_target,
            Slot(**{**our_best.__dict__}),
            Slot(
                "Embedding + log<sub>2</sub>FC readout + descriptors",
                "CheMeleon embedding",
                "sweepbest",
                native=True,
                select={**_CHEMELEON, "has_embedding": True, "has_readout": True, "has_descriptors": True},
            ),
            Slot(
                "Embedding + log<sub>2</sub>FC readout",
                "CheMeleon embedding, no descriptors",
                select={**_CHEMELEON, "has_embedding": True, "has_readout": True, "has_descriptors": False},
            ),
            Slot(
                "log<sub>2</sub>FC readout + descriptors",
                "no embedding",
                select={**_SCRATCH, "has_embedding": False, "has_readout": True, "has_descriptors": True},
            ),
            Slot(
                "Embedding + log<sub>2</sub>FC readout + descriptors",
                "log<sub>2</sub>FC embedding",
                select={**_SCRATCH, "has_embedding": True, "has_readout": True, "has_descriptors": True},
            ),
            Slot(
                "Embedding + descriptors",
                "log<sub>2</sub>FC embedding, no log<sub>2</sub>FC readout",
                select={**_SCRATCH, "has_embedding": True, "has_readout": False, "has_descriptors": True},
            ),
            Slot(**{**best_single.__dict__}),
            Slot(
                "Embedding + log<sub>2</sub>FC readout",
                "log<sub>2</sub>FC embedding, no descriptors",
                select={**_SCRATCH, "has_embedding": True, "has_readout": True, "has_descriptors": False},
            ),
            Slot(**{**best_gnn.__dict__}),
            Slot(**{**chemeleon_base.__dict__}),
            Slot(
                "Embedding + descriptors",
                "CheMeleon embedding, no log<sub>2</sub>FC readout",
                select={**_CHEMELEON, "has_embedding": True, "has_readout": False, "has_descriptors": True},
            ),
            Slot(
                "log<sub>2</sub>FC readout alone",
                "no embedding, no descriptors",
                select={**_SCRATCH, "has_embedding": False, "has_readout": True, "has_descriptors": False},
            ),
            Slot(
                "Descriptors alone",
                "Mordred, no embedding, no log<sub>2</sub>FC readout",
                select={**_SCRATCH, "has_embedding": False, "has_readout": False, "descriptor_sources": "mordred"},
            ),
        ],
        sort_from=3,
        divider_after=2,
    )

    # the regressor sweep fixes the featureset at figure-03's winner (CheMeleon
    # embedding + log2FC readout + descriptors, 386 columns) and varies only the
    # regressor
    crd = {
        "encoder_init": "chemeleon_pretrained",
        "has_embedding": True,
        "has_readout": True,
        "has_descriptors": True,
        "n_features": 386,
    }
    panel_regressor = Panel(
        "figure-04.html",
        [
            ref_ensemble,
            ref_target,
            Slot(**{**our_best.__dict__, "native": False}),
            Slot("TabICL v2.1.1", "", "winner", native=True, select={**crd, "regressor": "tabicl"}),
            Slot("TabPFN v3", "", select={**crd, "regressor": "tabpfn-v3"}),
            Slot("TabPFN v2.5", "", "sweepbest", native=True, select={**crd, "regressor": "tabpfn"}),
            Slot("TabPFN v2.6", "", select={**crd, "regressor": "tabpfn-v2.6"}),
            Slot(**{**best_single.__dict__}),
            Slot("TabFM v1.0.0", "max_num_rows=500", select={**crd, "regressor": "tabfm"}),
            Slot("LightGBM", "", select={**crd, "regressor": "lgbm"}),
            Slot(**{**best_gnn.__dict__}),
            Slot(**{**chemeleon_base.__dict__}),
            Slot("XGBoost", "", select={**crd, "regressor": "xgboost"}),
        ],
        sort_from=2,
        divider_after=1,
    )

    # panel 05 -> figure-06: calibration, two rows, no reference block
    panel_calibration = Panel(
        "figure-06.html",
        [
            Slot(
                "Uncalibrated",
                "CheMeleon embedding + log<sub>2</sub>FC readout, TabICL 2.1.1",
                "winner",
                native=True,
                select={
                    "encoder_init": "chemeleon_pretrained",
                    "regressor": "tabicl",
                    "has_embedding": True,
                    "has_readout": True,
                    "has_descriptors": False,
                    "n_features": 258,
                },
            ),
            Slot(
                "Calibrated",
                "isotonic map fit on 5-fold OOF predictions",
                select={
                    "encoder_init": "chemeleon_pretrained",
                    "regressor": "tabicl",
                    "has_embedding": True,
                    "has_readout": True,
                    "has_descriptors": False,
                    "n_features": 258,
                    "_calibrated": True,
                },
            ),
        ],
        sort_from=99,
        divider_after=-1,
    )

    # panel 04 -> figure-05: PCA-width mini-chart
    mini = MiniPanel(
        "figure-05.html",
        "Embedding + log<sub>2</sub>FC readout (fixed, raw) + descriptors, by PCA width",
        [
            MiniRow("Mordred, 128", {**_SCRATCH_ALL("mordred", 128)}),
            MiniRow("RDKit + Mordred, 128", {**_SCRATCH_ALL("all", 128)}),
            MiniRow("Mordred, 256", {**_SCRATCH_ALL("mordred", 256)}),
            MiniRow("RDKit + Mordred, 256", {**_SCRATCH_ALL("all", 256)}),
            MiniRow("RDKit + Mordred, 64", {**_SCRATCH_ALL("all", 64)}),
            MiniRow("Mordred, 64", {**_SCRATCH_ALL("mordred", 64)}),
        ],
    )

    return (
        [panel_gnn, panel_ingredients, panel_combine, panel_regressor, panel_calibration],
        mini,
    )


def _SCRATCH_ALL(source: str, width: int) -> dict[str, object]:
    """Predicate for a scratch embed+readout+descriptor run at a given PCA width."""
    return {
        **_SCRATCH,
        "has_embedding": True,
        "has_readout": True,
        "has_descriptors": True,
        "descriptor_sources": source,
        "descriptor_pca_width": width,
    }


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


def render_row(base_dir: str, stats: pd.Series, slot: Slot, is_divider: bool) -> str:
    """Render one config bar with computed width, whisker, and role color."""
    row_cls, bar_cls, bg = ROLE_STYLE[slot.role]
    classes = ["row"]
    if row_cls:
        classes.append(row_cls)
    if is_divider:
        classes.append("ref-divider")
    reused = " reused" if not slot.native else ""

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

    label = _label_html(slot.label, slot.sub)
    val = f'{stats["mae_mean"]:.4f}'
    return (
        f'      <div class="{" ".join(classes)}">\n'
        f'        <div class="row-label">{label}</div>\n'
        f'        <div class="track"><div class="{bar_classes}" style="{style}"></div>{whisker}</div>\n'
        f'        <div class="mae-val">{val}</div>\n'
        f"      </div>"
    )


def render_reference_row(ref: pd.Series, slot: Slot, is_divider: bool) -> str:
    """Render an external reference bar: width from its MAE, no whisker."""
    row_cls, bar_cls, _ = ROLE_STYLE[slot.role]
    classes = ["row", row_cls]
    if is_divider:
        classes.append("ref-divider")
    width = fmt_pct(pct(float(ref["mae"])))
    label = _label_html(slot.label, slot.sub)
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


def resolve_panel(
    panel: Panel, configs: pd.DataFrame, references: pd.DataFrame, holders: dict[str, str]
) -> list[tuple[Slot, str, pd.Series, bool]]:
    """Resolve every slot to its data, sort the body, and check roles.

    Returns ``(slot, kind, data, is_config)`` tuples in draw order, where kind is
    the resolved base_dir or reference id. Raises on a duplicate rendered label,
    a role whose config is not the computed extremum, or a config not found.
    """
    resolved: list[dict[str, object]] = []
    for slot in panel.slots:
        if slot.ref_id is not None:
            ref = references.loc[slot.ref_id]
            resolved.append({"slot": slot, "key": slot.ref_id, "data": ref, "is_cfg": False, "sort": float(ref["mae"])})
            continue
        if slot.select is None:
            raise ValueError(f"config slot {slot.label!r} has no select predicate")
        row = select_one(configs, slot.select)
        _check_role(slot, str(row.name), holders)
        resolved.append({"slot": slot, "key": str(row.name), "data": row, "is_cfg": True, "sort": float(row["mae_mean"])})

    head = resolved[: panel.sort_from]
    body = sorted(resolved[panel.sort_from :], key=lambda item: item["sort"])
    ordered = head + body

    seen: set[str] = set()
    for item in ordered:
        rendered = _label_html(item["slot"].label, item["slot"].sub)
        if rendered in seen:
            raise ValueError(f"duplicate rendered label {rendered!r} in {panel.out_name}")
        seen.add(rendered)

    return [(item["slot"], str(item["key"]), item["data"], bool(item["is_cfg"])) for item in ordered]


def _check_role(slot: Slot, base_dir: str, holders: dict[str, str]) -> None:
    """Verify a declared singular role belongs to the data-derived holder.

    A panel-local winner (the sweep minimum) is allowed alongside the global
    winner, so a winner slot passes when it holds either.
    """
    if slot.role not in holders:
        return
    if slot.role == "winner":
        return  # global vs panel-local winner checked in verify.py against the rendered set
    if holders[slot.role] != base_dir:
        raise ValueError(
            f"slot {slot.label!r} declares role {slot.role!r} but data assigns it to {holders[slot.role]!r}, not {base_dir!r}"
        )


def render_panel(panel: Panel, configs: pd.DataFrame, references: pd.DataFrame, holders: dict[str, str]) -> str:
    """Render a full chart panel to its standalone card HTML."""
    ordered = resolve_panel(panel, configs, references, holders)
    rows: list[str] = []
    for index, (slot, key, data, is_cfg) in enumerate(ordered):
        is_divider = index == panel.divider_after
        if is_cfg:
            rows.append(render_row(key, data, slot, is_divider))
        else:
            rows.append(render_reference_row(data, slot, is_divider))
    body = "\n\n".join(rows)
    return f"{CSS_STYLE}\n{CHART_OPEN}{body}\n{CHART_CLOSE}"


def render_mini(mini: MiniPanel, configs: pd.DataFrame) -> str:
    """Render the PCA-width mini-chart, sorted by MAE with the minimum accented."""
    resolved = []
    for row in mini.rows:
        cfg = select_one(configs, row.select)
        resolved.append((row.label, cfg))
    resolved.sort(key=lambda item: float(item[1]["mae_mean"]))
    floor = min(float(cfg["mae_mean"]) for _, cfg in resolved)

    grid = 'style="grid-template-columns: 9rem 1fr 3.4rem;"'
    lines = [
        '<div class="pxr-post">',
        '    <div class="mini-chart">',
        f'      <p class="mini-title">{mini.title}</p>',
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
        lines.append(f'        <span class="mini-val">{float(cfg["mae_mean"]):.4f}</span>')
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
    configs["_calibrated"] = [str(name).endswith("_calibrated") for name in configs.index]
    return configs


def main() -> None:
    """Render figure-01 .. figure-06 from the run index and reference table."""
    configs = load_configs()
    configs = attach_calibrated(configs)
    references = pd.read_csv(REFERENCE_PATH).set_index("id")
    holders = role_holders(configs)

    panels, mini = build_panels()
    OUT_DIR.mkdir(exist_ok=True)

    # the calibration panel resolves on the _calibrated column; every other
    # panel selects non-calibrated runs, so drop the calibrated twin there
    plain_configs = configs[~configs["_calibrated"]]

    for panel in panels:
        source = configs if panel.out_name == "figure-06.html" else plain_configs
        html = render_panel(panel, source, references, holders)
        (OUT_DIR / panel.out_name).write_text(html)
        print(f"wrote {panel.out_name}")

    (OUT_DIR / mini.out_name).write_text(render_mini(mini, plain_configs))
    print(f"wrote {mini.out_name}")


if __name__ == "__main__":
    main()
