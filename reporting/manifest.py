"""Row-level metadata for the panel 00/01/02 bar charts in the ablation artifact.

Each entry names one experiment shown on the page. `build_results_table.py`
joins this manifest against `results/<base_dir>_seed{0..4}/eval_out.csv` to
compute the mean/min/max MAE each row's bar and whisker are drawn from, so a
new seed sweep only requires re-running the build script, never a hand edit
to the HTML.

`role` records the deliberate story-level anchors (winner, floor, blue
"best single ingredient") that persist across panels. These are asserted
against the computed means at build time rather than picked automatically,
because "floor" here means "best of the concatenation-GNN-baseline family
specifically," a narrower, human-decided grouping than the flat `category`
enum can express (the pEC50-direct GNN row often beats it but is not part of
that family). If a re-run flips which row actually has the lowest mean
within a role's group, the build script raises instead of silently drawing
the wrong row in gold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Category = str  # one of: "n283t", "gnn", "tabular-single", "tabular-multiple"


@dataclass(frozen=True)
class Experiment:
    """One row's identity: which results directory backs it, and how it's drawn."""

    key: str
    base_dir: str | None  # None for fixed external-benchmark rows (no results/ dir)
    label: str
    sub: str
    category: Category
    panels: tuple[str, ...]
    role: str | None = None  # "winner" | "floor" | "single_best" | "chemeleon" | "context" | "target" | None
    fixed_mae: float | None = None  # set only for external-benchmark rows with no seed data
    extra_sub: str | None = None  # footnote marker text shown after the sub label, if any
    single_run: bool = False  # read `{base_dir}/eval_out.csv` directly, no `_seed{N}` sweep exists
    native_panels: tuple[str, ...] = ()  # panels whose own sweep this row is a genuine member of;
    # empty means "not tracked" (never hatched). A row can be native to more than
    # one panel at once (e.g. a feature-combination winner that's also a value of
    # a later panel's regressor sweep on that same featureset) — it's only
    # "reused for reference" in panels outside this set.


@dataclass(frozen=True)
class MiniChart:
    """A `.mini-chart` bar block: several rows on a shared axis, one highlighted accent bar."""

    panel: str
    title: str
    rows: tuple[tuple[str, str], ...]  # (experiment key, mini-label) pairs, in display order
    extra_sub: str | None = None
    accent: str = "min"  # "min" picks the lowest mean automatically, or an explicit experiment key
    show_whiskers: bool = True  # False for single-run sweeps with no seed data to whisker
    sort_by_mean: bool = False  # re-order `rows` by ascending mean MAE instead of manifest order
    label_width: str = "5.2rem"  # mini-row's label column width, widened for longer row labels


@dataclass(frozen=True)
class GridTableCell:
    key: str
    is_best: bool = False


@dataclass(frozen=True)
class GridTable:
    """A `.grid-table` 2D comparison: row/column headers over experiment keys."""

    panel: str
    caption: str
    col_headers: tuple[str, ...]
    row_headers: tuple[str, ...]
    cells: tuple[tuple[str, ...], ...]  # cells[row_idx][col_idx] = experiment key


EXPERIMENTS: list[Experiment] = [
    # External benchmarks: not our runs, no seed data, fixed values.
    Experiment(
        key="n283t_ensemble",
        base_dir=None,
        label="N283T ensemble",
        sub="context only, not our pipeline",
        category="n283t",
        panels=("00", "01", "02"),
        role="context",
        fixed_mae=0.408,
    ),
    Experiment(
        key="n283t_target",
        base_dir=None,
        label="N283T report target",
        sub="frozen ChemProp embedding + TabPFN",
        category="n283t",
        panels=("00", "01", "02"),
        role="target",
        fixed_mae=0.437,
    ),
    # Our best overall (TabICL regressor), anchor row in panels 00, 01, 02, 03.
    # Its own category ("tabular-best-overall" rather than "tabular-multiple")
    # keeps it out of panel 02's sweepbest peer comparison (it's a different
    # regressor track, not part of the TabPFN-only combo sweep). Earlier we
    # believed TabICL OOMs on panel 02's winning featureset
    # (chemeleon_readout_descriptors, 386 columns, descriptors included), so
    # panel 03's regressor sweep ran on the leaner chemeleon_readout_only (258
    # columns) instead. That premise was wrong on the hardware this project
    # actually runs on: a complete, valid 5-seed GPU run at 386 columns
    # exists (mean MAE 0.4382), so panel 03 now runs every regressor,
    # including TabICL, on chemeleon_readout_descriptors, matching panel 02's
    # own winning combo directly. TabICL only OOMs there on CPU (confirmed via
    # a CPU rerun, growing to ~30 GB RSS before the kernel OOM-killer stops
    # it), which doesn't affect the GPU runs this page reports. TabICL still
    # wins that apples-to-apples sweep. Separately, the single best number
    # found anywhere in this project remains this row: TabICL on the leaner
    # chemeleon_readout_only featureset (0.4356), which is why it's sourced
    # from there rather than from panel 03's own 386-column sweep, and why
    # panel 03 shows it too, as a hashed reference row outside that sweep.
    Experiment(
        key="our_best_overall",
        base_dir="tabicl_chemeleon_readout_only",
        label="Our best overall",
        sub="CheMeleon embedding + log<sub>2</sub>FC readout, TabICL 2.1.1",
        category="tabular-best-overall",
        panels=("00", "01", "02"),
        role="winner",
        native_panels=("00", "01", "02"),
    ),
    # Panel 00: pre-tabular-foundation-model reference points.
    # "CheMeleon baseline" is the point of departure from a pretrained,
    # non-fine-tuned CheMeleon encoder to a fully fine-tuned one (no log2FC
    # pretraining at all): flagged with the "chemeleon" role and propagated to
    # panels 01/02/03 as a separate reused row below, so this transition
    # stays visible everywhere on the page.
    Experiment(
        key="fine_tuned_pec50_direct",
        base_dir="tabpfn_pec50_chemeleon",
        label="CheMeleon baseline",
        sub="no log<sub>2</sub>FC pretraining, single-stage",
        category="gnn",
        panels=("00",),
        role="chemeleon",
    ),
    # The actual E3 concatenation architecture (auxiliary log2FC encoder,
    # concatenated into the primary network before the final FFN;
    # moal_aux_plan_config.yaml), swept over freeze_epochs x ffn_hidden_dim x
    # gradient_clip_val (18 configs, 5 seeds each). This is the best of that
    # 18-config sweep; the other 17 aren't shown individually.
    Experiment(
        key="concat_architecture",
        base_dir="freeze2_hd512_clipoff",
        label="Concatenation architecture",
        sub="best of 18-config freeze/width/clip sweep",
        category="gnn",
        panels=("00",),
        role="floor",
    ),
    Experiment(
        key="fine_tuned_log2fc_drc_only",
        base_dir="e4_finetune_drc_only",
        label="Fine-tuned log<sub>2</sub>FC encoder",
        sub="full model, dose-response only",
        category="gnn",
        panels=("00",),
    ),
    Experiment(
        key="frozen_log2fc_full",
        base_dir="e4_frozen",
        label="Frozen log<sub>2</sub>FC encoder",
        sub="FFN only, dose-response + primary screen",
        category="gnn",
        panels=("00",),
    ),
    Experiment(
        key="fine_tuned_log2fc_full",
        base_dir="e4_finetune",
        label="Fine-tuned log<sub>2</sub>FC encoder",
        sub="full model, dose-response + primary screen",
        category="gnn",
        panels=("00",),
    ),
    Experiment(
        key="frozen_log2fc_drc_only",
        base_dir="e4_frozen_drc_only",
        label="Frozen log<sub>2</sub>FC encoder",
        sub="FFN-tuned, dose-response only",
        category="gnn",
        panels=("00",),
    ),
    # Panels 01/02: single-ingredient tabular-foundation-model rows.
    Experiment(
        key="log2fc_embedding",
        base_dir="tabpfn_embed_only_no_readout",
        label="log<sub>2</sub>FC embedding",
        sub="from-scratch encoder, log<sub>2</sub>FC-trained",
        category="tabular-single",
        panels=("01", "02"),
        role="single_best",
    ),
    Experiment(
        key="chemeleon_embedding",
        base_dir="tabpfn_chemeleon_embed_only",
        label="CheMeleon embedding",
        sub="pretrained, no fine-tuning",
        category="tabular-single",
        panels=("01",),
    ),
    Experiment(
        key="rdkit_descriptors",
        base_dir="tabpfn_rdkit_only_pca128_no_embed_no_readout",
        label="RDKit descriptors",
        sub="217 columns, PCA-128",
        category="tabular-single",
        panels=("01",),
    ),
    Experiment(
        key="predicted_readout",
        base_dir="tabpfn_readout_only_no_embed",
        label="Predicted log<sub>2</sub>FC readout",
        sub="encoder's own prediction, 2 columns",
        category="tabular-single",
        panels=("01",),
    ),
    Experiment(
        key="mordred_descriptors",
        base_dir="tabpfn_mordred_only_pca128_no_embed_no_readout",
        label="Mordred descriptors",
        sub="~1600 columns, PCA-128",
        category="tabular-single",
        panels=("01",),
    ),
    # Panel 02: all seven combinations of embedding / readout / descriptors (Mordred, TabPFN v2.5).
    Experiment(
        key="readout_descriptors",
        base_dir="tabpfn_readout_mordred_pca128_no_embed",
        label="Readout + descriptors",
        sub="no embedding",
        category="tabular-multiple",
        panels=("02",),
    ),
    Experiment(
        key="embed_readout_descriptors",
        base_dir="tabpfn_embed_readout_mordred_pca128",
        label="Embedding + readout + descriptors",
        sub="log<sub>2</sub>FC embedding",
        category="tabular-multiple",
        panels=("02",),
    ),
    Experiment(
        key="embed_descriptors",
        base_dir="tabpfn_embed_mordred_pca128_no_readout",
        label="Embedding + descriptors",
        sub="log<sub>2</sub>FC embedding, no readout",
        category="tabular-multiple",
        panels=("02",),
    ),
    Experiment(
        key="embed_readout",
        base_dir="tabpfn_small_embed",
        label="Embedding + readout",
        sub="log<sub>2</sub>FC embedding, no descriptors",
        category="tabular-multiple",
        panels=("02",),
    ),
    # Side experiment: same three combos, but with the log2FC-pretrained
    # encoder's own embedding swapped for CheMeleon's untouched, off-the-shelf
    # pretrained embedding (PCA-256, matching panel 01's own CheMeleon-
    # embedding treatment). CheMeleon is never fine-tuned here, so it has no
    # predicted-readout head of its own; the two combos below that need a
    # readout block borrow it from the same from-scratch encoder cache the
    # log2FC-embedding rows above use, isolating the embedding swap as the
    # only changed variable.
    Experiment(
        key="chemeleon_readout_descriptors",
        base_dir="tabpfn_chemeleon_readout_descriptors",
        label="Embedding + readout + descriptors",
        sub="CheMeleon embedding",
        category="tabular-multiple",
        panels=("02",),
        role="sweepbest",
        native_panels=("02", "03"),
    ),
    Experiment(
        key="chemeleon_descriptors",
        base_dir="tabpfn_chemeleon_descriptors_only",
        label="Embedding + descriptors",
        sub="CheMeleon embedding, no readout",
        category="tabular-multiple",
        panels=("02",),
    ),
    Experiment(
        key="chemeleon_readout",
        base_dir="tabpfn_chemeleon_readout_only",
        label="Embedding + readout",
        sub="CheMeleon embedding, no descriptors",
        category="tabular-multiple",
        panels=("02",),
        native_panels=("02",),
    ),
]

# Panel 02 reuses several panel 00/01 rows under different labels (same
# underlying experiment, same results directory, different display text).
EXPERIMENTS.extend(
    [
        Experiment(
            key="best_gnn_baseline",
            base_dir="freeze2_hd512_clipoff",
            label="Best GNN baseline",
            sub="concatenation architecture",
            category="gnn",
            panels=("01", "02", "03"),
            role="floor",
            native_panels=("00",),
        ),
        Experiment(
            key="chemeleon_baseline",
            base_dir="tabpfn_pec50_chemeleon",
            label="CheMeleon baseline",
            sub="no log<sub>2</sub>FC pretraining, single-stage",
            category="gnn",
            panels=("01", "02", "03"),
            role="chemeleon",
            native_panels=("00",),
        ),
        Experiment(
            key="best_single_ingredient",
            base_dir="tabpfn_embed_only_no_readout",
            label="Best single-ingredient",
            sub="log<sub>2</sub>FC embedding only",
            category="tabular-single",
            panels=("02", "03"),
            role="single_best",
            native_panels=("01",),
        ),
        Experiment(
            key="readout_alone",
            base_dir="tabpfn_readout_only_no_embed",
            label="Readout alone",
            sub="no embedding, no descriptors",
            category="tabular-single",
            panels=("02",),
            native_panels=("01", "02"),
        ),
        Experiment(
            key="descriptors_alone",
            base_dir="tabpfn_mordred_only_pca128_no_embed_no_readout",
            label="Descriptors alone",
            sub="Mordred, no embedding, no readout",
            category="tabular-single",
            panels=("02",),
            native_panels=("01", "02"),
        ),
        # Panel 03: every regressor here runs on chemeleon_readout_descriptors
        # (embedding + readout + descriptors, 386 columns), panel 02's own
        # best combo, TabICL included. TabICL was previously thought to OOM
        # at this width; a complete, valid 5-seed GPU run at 386 columns
        # (mean MAE 0.4382) shows it doesn't, so every regressor here shares
        # panel 02's own winning featureset directly, an apples-to-apples
        # regressor comparison on the featureset the rest of the page already
        # crowns. TabICL still wins. Its even-better result on the leaner
        # chemeleon_readout_only featureset (0.4356, our_best_overall) is
        # shown here too, as a hashed reference row outside this sweep.
        Experiment(
            key="tabicl_v211",
            base_dir="tabicl_chemeleon_readout_descriptors",
            label="TabICL v2.1.1",
            sub="",
            category="tabular-multiple",
            panels=("03",),
        ),
        Experiment(
            key="tabpfn_v26",
            base_dir="tabpfn-v2.6_chemeleon_readout_descriptors",
            label="TabPFN v2.6",
            sub="",
            category="tabular-multiple",
            panels=("03",),
        ),
        Experiment(
            key="tabpfn_v3",
            base_dir="tabpfn-v3_chemeleon_readout_descriptors",
            label="TabPFN v3",
            sub="",
            category="tabular-multiple",
            panels=("03",),
        ),
        Experiment(
            key="tabfm_v1",
            base_dir="tabfm_n32_chemeleon_readout_descriptors",
            label="TabFM v1.0.0",
            sub="max_num_rows=500",
            category="tabular-multiple",
            panels=("03",),
        ),
        Experiment(
            key="lgbm",
            base_dir="lgbm_chemeleon_readout_descriptors",
            label="LightGBM",
            sub="",
            category="tabular-multiple",
            panels=("03",),
        ),
        Experiment(
            key="xgboost",
            base_dir="xgboost_chemeleon_readout_descriptors",
            label="XGBoost",
            sub="",
            category="tabular-multiple",
            panels=("03",),
        ),
        # Panel 03: encoder init x training-target grid (from-scratch/CheMeleon x log2FC/pEC50).
        Experiment(
            key="pec50_scratch",
            base_dir="tabpfn_pec50_scratch",
            label="From-scratch, pEC<sub>50</sub>-trained",
            sub="",
            category="tabular-multiple",
            panels=("03",),
        ),
        Experiment(
            key="chemeleon_log2fc",
            base_dir="tabpfn_chemeleon_log2fc_mordred_pca128",
            label="CheMeleon-init, log<sub>2</sub>FC-trained",
            sub="",
            category="tabular-multiple",
            panels=("03",),
        ),
        # Panel 04: Mordred-only descriptor PCA-width sweep (embed + readout fixed, raw).
        Experiment(
            key="mordred_pca64",
            base_dir="tabpfn_embed_readout_mordred_pca64",
            label="64 components",
            sub="",
            category="tabular-multiple",
            panels=("04",),
        ),
        Experiment(
            key="mordred_pca256",
            base_dir="tabpfn_embed_readout_mordred_pca256",
            label="256 components",
            sub="",
            category="tabular-multiple",
            panels=("04",),
        ),
        # Panel 04: RDKit+Mordred combined PCA-width sweep, 5-seed mean each.
        Experiment(
            key="rdkit_mordred_pca64",
            base_dir="tabpfn_concat_pca64",
            label="64 components",
            sub="",
            category="tabular-multiple",
            panels=("04",),
        ),
        Experiment(
            key="rdkit_mordred_pca128",
            base_dir="tabpfn_concat_small_embed",
            label="128 components",
            sub="",
            category="tabular-multiple",
            panels=("04",),
        ),
        Experiment(
            key="rdkit_mordred_pca256",
            base_dir="tabpfn_concat_pca256",
            label="256 components",
            sub="",
            category="tabular-multiple",
            panels=("04",),
        ),
        # Panel 05: blind-set calibration against the actual current winner
        # (tabicl_chemeleon_readout_only, 5-seed sweep). Each seed's isotonic
        # map is fit via tabpfn_chemeleon_calibrate_predictions.py's 5-fold OOF
        # loop (retraining the log2FC readout encoder per fold to avoid
        # leaking the fixed embedding cache), then applied to that seed's own
        # blind predictions; the calibrated row below joins those 5 rescored
        # blind-set results the same way every other seed-swept row here does.
        # The OOF mini-chart was dropped: the OOF MAE is computed on raw
        # per-record DRC values, not per-compound pEC50, so it isn't the same
        # metric as the rest of the page and can't back a row here.
        Experiment(
            key="calib_blind_uncalibrated",
            base_dir="tabicl_chemeleon_readout_only",
            label="Uncalibrated",
            sub="",
            category="tabular-multiple",
            panels=("05",),
        ),
        Experiment(
            key="calib_blind_calibrated",
            base_dir="tabicl_chemeleon_readout_only_calibrated",
            label="Calibrated",
            sub="",
            category="tabular-multiple",
            panels=("05",),
        ),
    ]
)

MINI_CHARTS: list[MiniChart] = [
    MiniChart(
        panel="04",
        title="Embed + readout (fixed, raw) + descriptors, by PCA width",
        rows=(
            ("mordred_pca64", "Mordred, 64"),
            ("embed_readout_descriptors", "Mordred, 128"),
            ("mordred_pca256", "Mordred, 256"),
            ("rdkit_mordred_pca64", "RDKit + Mordred, 64"),
            ("rdkit_mordred_pca128", "RDKit + Mordred, 128"),
            ("rdkit_mordred_pca256", "RDKit + Mordred, 256"),
        ),
        sort_by_mean=True,
        label_width="9rem",
    ),
]

GRID_TABLES: list[GridTable] = []

# Visual divider (a bottom border) marking the end of the reference/anchor
# block before each panel's actual comparison rows begin. Purely positional
# (which row happens to sit last in that block), not a property of the
# experiment itself, so it's keyed by panel rather than stored on Experiment.
PANEL_DIVIDER_AFTER: dict[str, str] = {
    "00": "our_best_overall",
    "01": "our_best_overall",
    "02": "our_best_overall",
    "03": "n283t_target",
}

# Each panel's reference/anchor block stays fixed in manifest order;
# everything after the given index is sorted by ascending mean MAE at build
# time so "best of" context rows (Best GNN baseline, Best single-ingredient)
# interleave with the sweep rows by rank rather than being appended after
# them, and manifest insertion order (e.g. adding a new combo row) can never
# desync the display order from actual MAE rank. Panels 00-02's fixed block
# is n283t x2 + our_best_overall (index 3 onward is the sweep); panel 03's
# is n283t x2 only (index 2 onward, since its own winner is part of the
# sweep there, not a separate anchor row). Every panel must have an entry
# here, or its post-anchor rows silently keep raw manifest order instead of
# being sorted (the bug this comment used to miss for panels 00 and 01).
PANEL_SORT_FROM: dict[str, int] = {
    "00": 3,
    "01": 3,
    "02": 3,
    "03": 2,
}

# Cosmetic-only role recoloring for rows reused in a second panel with a
# different meaning there than their Experiment.role implies. Not checked by
# check_roles (that reads exp.role, not this map): these aren't "beats its
# peers" claims. Colors are identity tags, not per-panel medals: once a
# configuration earns a color as some panel's winner, every later reuse of
# that exact configuration keeps the color, whether or not it wins the panel
# it's reused in. chemeleon_readout_descriptors needs no override here: its
# panel 03 appearance is the same 386-column combo that won panel 02, so it
# stays green (sweepbest) via its own declared role, matching Figure 2.
# tabicl_v211 is overridden to "winner" (gold, the color already used
# elsewhere for our_best_overall) since it's a genuinely new win here, on a
# featureset none of the other gold or green rows share.
ROLE_OVERRIDE: dict[tuple[str, str], str] = {
    ("03", "tabicl_v211"): "winner",
    ("05", "calib_blind_uncalibrated"): "winner",
}

# Cosmetic-only label override for panel 03's TabPFN v2.5 row.
# chemeleon_readout_descriptors reads as "TabPFN v2.5" alongside its
# regressor-sweep siblings ("TabPFN v2.6", "LightGBM", ...), with its
# original panel 02 label ("Embedding + readout + descriptors") replaced so
# the row reads as a regressor name like its neighbors; its sub is blank
# there too, same as theirs. tabicl_v211 needs no override here: it now runs
# on the same featureset as every other panel 03 row, so its own label/sub
# (set on the Experiment itself) already read like its neighbors.
# our_best_overall needs no override either: its own label ("Our best
# overall") and sub already explain why it's a hashed outlier here.
LABEL_OVERRIDE: dict[tuple[str, str], tuple[str, str]] = {
    ("03", "chemeleon_readout_descriptors"): ("TabPFN v2.5", ""),
    ("05", "calib_blind_uncalibrated"): (
        "Uncalibrated",
        "CheMeleon embedding + log<sub>2</sub>FC readout, TabICL 2.1.1",
    ),
    ("05", "calib_blind_calibrated"): (
        "Calibrated",
        "isotonic map fit on 5-fold OOF predictions",
    ),
}

PANEL_ORDER: dict[str, list[str]] = {
    "00": [
        "n283t_ensemble",
        "n283t_target",
        "our_best_overall",
        "fine_tuned_pec50_direct",
        "concat_architecture",
        "frozen_log2fc_full",
        "frozen_log2fc_drc_only",
        "fine_tuned_log2fc_full",
        "fine_tuned_log2fc_drc_only",
    ],
    "01": [
        "n283t_ensemble",
        "n283t_target",
        "our_best_overall",
        "log2fc_embedding",
        "chemeleon_embedding",
        "rdkit_descriptors",
        "best_gnn_baseline",
        "chemeleon_baseline",
        "predicted_readout",
        "mordred_descriptors",
    ],
    "02": [
        "n283t_ensemble",
        "n283t_target",
        "our_best_overall",
        "readout_descriptors",
        "embed_readout_descriptors",
        "chemeleon_readout_descriptors",
        "embed_descriptors",
        "chemeleon_descriptors",
        "best_single_ingredient",
        "embed_readout",
        "chemeleon_readout",
        "best_gnn_baseline",
        "chemeleon_baseline",
        "readout_alone",
        "descriptors_alone",
    ],
    # Rows after index 2 (the n283t x2 header, see PANEL_SORT_FROM) are
    # sorted by ascending mean MAE at build time; their order here is
    # irrelevant except as a stable tie-break.
    "03": [
        "n283t_ensemble",
        "n283t_target",
        "tabicl_v211",
        "chemeleon_readout_descriptors",
        "tabpfn_v3",
        "tabpfn_v26",
        "tabfm_v1",
        "lgbm",
        "xgboost",
        "our_best_overall",
        "best_gnn_baseline",
        "chemeleon_baseline",
        "best_single_ingredient",
    ],
    "05": [
        "calib_blind_uncalibrated",
        "calib_blind_calibrated",
    ],
}
