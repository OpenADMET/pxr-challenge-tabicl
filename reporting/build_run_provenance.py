"""Build an artifact-grounded provenance table for every run in `results/`.

Two independent evidence sources are consulted for each run directory:

1. **In-run artifacts** (`config_used.yaml`, `eval_out.csv`) written into the
   run directory itself by the moal chemprop active-learning pipeline. This
   is the only source the first version of this script used.
2. **Launch scripts**: `run_seed_sweep*.sh`, `run_sweep.sh`,
   `run_cpu_oom_reproductions.sh`, `run_gnn_seed_sweep.py`, and
   `run_tabfm_param_sweep.py` in the repo root hard-code, for a given
   `results/<output-dir>`, exactly which Python feature/training script ran,
   with exactly which `configs/*.yaml` file and exactly which CLI flags. Each
   underlying script's argparse definitions pin what a flag actually controls
   (`--no-readout`, `--descriptor-source`, `--regressor`, an unset flag's
   documented default, ...). `LAUNCH_INVOCATIONS` below is an exhaustive,
   hand-verified transcription of every invocation in those launch scripts
   (each one read in full, not sampled), turned into structured records so
   this script can join them against the actual run directories deterministically.

3. **Convention** (`CONVENTION_INVOCATIONS` below): a lower-trust layer for
   result directories no current launch script emits verbatim but whose full
   feature spec is still recoverable, from NEXT_STEPS.md's running experiment
   log, the naming convention it documents, the invoked script's argparse
   defaults, and the axes of a seeded sibling already in the launch table.
   This covers the pre-seed-sweep single runs (unseeded originals) and the
   rewritten-recipe runs (tabicl and pinned-tabpfn-version variants later
   recipe edits dropped). Kept separate from the launch tier so
   `LAUNCH_INVOCATIONS` stays a pure verbatim transcription; these rows are
   stamped `spec_source="convention"`.

Where an in-run artifact and a launch-script resolution both exist for a row,
the in-run artifact is preferred as the row's value, but the two are compared
and any disagreement is recorded in `launch_vs_inrun_disagreements` and
surfaced in the report rather than silently dropped. Where only a
launch-script resolution exists, it fills the axis and `spec_source` records
`launch_script`. The convention layer is consulted only for rows the launch
layer does not match at all, and stamps `spec_source="convention"`. Where no
source resolves an axis, it stays null and is recorded in `unknown_axes` for
that row, same as before.

Run it from the repo root:

    python reporting/build_run_provenance.py

Output lands at `results/run_provenance.parquet` (or `.csv` if pyarrow is
unavailable).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"
OUTPUT_PARQUET = RESULTS_DIR / "run_provenance.parquet"
OUTPUT_CSV = RESULTS_DIR / "run_provenance.csv"

# The CheMeleon->pEC50 baseline runs use the anvil-recipe pipeline (openadmet
# standard models) instead of this challenge's own seeded sweep scripts, so
# they carry no config_used.yaml/eval_out.csv and are skipped by the normal
# per-dir loop. They are not dropped: their axes come from anvil_recipe.yaml and
# their metrics from regression_metrics.json, ingested as one combined row by
# build_anvil_baseline_spec (the two dirs are one model scored on the two blind
# test phases separately, recombined here into the same overall split the sweep
# rows report). Anything genuinely outside the table stays in EXCLUDED_DIRS.
ANVIL_BASELINE_DIRS = ("pxr_baseline_predictions_phase1", "pxr_baseline_predictions_phase2")
EXCLUDED_DIRS = set(ANVIL_BASELINE_DIRS)

SEED_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")

# The critical-run-component columns, in the fixed order used for fingerprinting.
CRITICAL_COLUMNS = [
    "encoder_family",
    "encoder_init",
    "encoder_target",
    "embedding_native_dim",
    "embedding_pca_dim",
    "has_embedding",
    "has_readout",
    "readout_dim",
    "has_descriptors",
    "descriptor_sources",
    "descriptor_pca_width",
    "regressor",
    "train_data",
    "n_features",
]

METRIC_COLUMNS = ["mae", "n", "rmse", "rae", "r2", "kendall_tau", "spearman_rho"]

# Third-pass additions (directive 2): moal/chemprop-schema hyperparameters
# that are real run facts (FFN hidden width, frozen-epoch count,
# gradient-clip value) but sit outside the tabular-pipeline vocabulary
# CRITICAL_COLUMNS was built around. Tracked as their own columns rather than
# folded into CRITICAL_COLUMNS/the fingerprint, since they only apply to one
# pipeline family and adding them to the fingerprint would retroactively
# change every already-resolved fingerprint's meaning.
#
# aux_use_observed_readout/aux_use_predicted_readout added this pass: the
# feat_observed_2task / feat_observed_and_predicted_2task /
# feat_predicted_native_2task trio of moal-schema configs is defined entirely
# by which combination of these two auxiliary_model flags is set (True/False,
# True/True, False/True — verified against results/*/config_used.yaml), so
# without these columns those three runs are indistinguishable in this table.
# encoder_depth/aux_encoder_depth (model.depth/auxiliary_model.depth in
# config_used.yaml) were checked too and dropped: every run that carries the
# field has it pinned to 3, so it is not an axis anything in results/ varies
# on, only ffn_num_layers/depth is genuinely constant repo-wide.
MOAL_HYPERPARAM_COLUMNS = [
    "ffn_hidden_dim",
    "freeze_epochs",
    "gradient_clip_val",
    "aux_use_observed_readout",
    "aux_use_predicted_readout",
]

# The sentinel used for `seed` when no script/CLI invocation pins an explicit
# seed for a run (directive 1). A real int64 value, never NaN/None, so it
# survives parquet round-trips without coercing the column to float.
UNPINNED_SEED = -1

# All configs referenced by the launch scripts use this input CSV, which has
# no "drc_only" marker, so every launch-script-resolved row's train_data is
# "drc_plus_primary" (mirrors resolve_moal_config_axes's own input_csv rule).
_TRAIN_DATA = "drc_plus_primary"

# The from-scratch auxiliary encoder's readout head predicts one value per
# log2fc_columns entry; every config referenced by the launch scripts lists
# exactly 2 (see configs/*.yaml's data.plan.log2fc_columns).
_READOUT_DIM = 2

# CheMeleon's native pooled-embedding width is fixed regardless of config
# (documented in every CheMeleon-embedding script's docstring/argparse help).
_CHEMELEON_NATIVE_DIM = 2048


@dataclass
class RunSpec:
    """Everything resolved (or explicitly left unresolved) for one run directory."""

    base_dir: str
    seed: int
    run_dir: str
    encoder_family: str | None = None
    encoder_init: str | None = None
    encoder_target: str | None = None
    embedding_native_dim: int | None = None
    embedding_pca_dim: int | None = None
    has_embedding: bool | None = None
    has_readout: bool | None = None
    readout_dim: int | None = None
    has_descriptors: bool | None = None
    descriptor_sources: str | None = None
    descriptor_pca_width: int | None = None
    regressor: str | None = None
    train_data: str | None = None
    n_features: int | None = None
    ffn_hidden_dim: int | None = None
    freeze_epochs: int | None = None
    gradient_clip_val: float | None = None
    aux_use_observed_readout: bool | None = None
    aux_use_predicted_readout: bool | None = None
    spec_source: str | None = None
    # Set True for the moal/chemprop end-to-end pipeline: it has no
    # concatenated final feature vector for a downstream regressor (there is
    # no downstream regressor), so n_features is a category error for these
    # rows even once embedding_native_dim/readout_dim resolve, not merely an
    # unresolved value compute_n_features could derive if it tried.
    n_features_inapplicable: bool = False
    unknown_axes: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)


def discover_run_dirs() -> list[Path]:
    """List every candidate run directory under results/, skipping non-run entries."""
    dirs = []
    for p in sorted(RESULTS_DIR.iterdir()):
        if not p.is_dir():
            continue
        if p.name in EXCLUDED_DIRS:
            continue
        dirs.append(p)
    return dirs


def split_base_and_seed(dir_name: str) -> tuple[str, int]:
    """Split a results/ directory name into (base_dir, seed).

    `_seed{N}` with N in 0..4 (the only seed range any launch script's
    5-seed sweep convention ever uses) is the accepted explicit seed pin.
    Everything else gets `UNPINNED_SEED` (-1): an unsuffixed directory, or a
    `_seed{N}` suffix with N outside 0..4 (e.g. the out-of-range
    `tabicl_embed_readout_mordred_pca128_seed5`, whose "5" does not correspond
    to any actual 5-seed-sweep invocation and so is not a genuine pin, just a
    directory name that happens to parse). This is an explicit
    "no script/CLI invocation pins a seed for this run" sentinel, not a
    guess at what seed the run might have used (directive 1, third pass —
    do not infer seed=42 from a config/argparse default fallback here).
    """
    m = SEED_RE.match(dir_name)
    if m:
        seed = int(m.group("seed"))
        if 0 <= seed <= 4:
            return m.group("base"), seed
        # Out-of-range suffix (e.g. "_seed5"): still strip it for base_dir
        # grouping (it is unmistakably the same sweep family), but the seed
        # itself is not a genuine pin.
        return m.group("base"), UNPINNED_SEED
    return dir_name, UNPINNED_SEED


def load_config_used(run_dir: Path) -> dict[str, Any] | None:
    """Load config_used.yaml if present in this run directory."""
    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


def load_eval_metrics(run_dir: Path) -> dict[str, float | None]:
    """Read the 'overall' subset row of eval_out.csv, if present."""
    eval_path = run_dir / "eval_out.csv"
    if not eval_path.exists():
        return {col: None for col in METRIC_COLUMNS}
    df = pd.read_csv(eval_path)
    overall = df.loc[df["subset"] == "overall"]
    if overall.empty:
        return {col: None for col in METRIC_COLUMNS}
    row = overall.iloc[0]
    return {col: (row[col] if col in row.index else None) for col in METRIC_COLUMNS}


# Synthetic base_dir for the combined CheMeleon->pEC50 baseline. It is not a
# directory on disk: the two per-phase anvil runs recombine into this one row.
ANVIL_BASELINE_BASE_DIR = "pxr_baseline_chemeleon_pec50"


def _resolve_anvil_axes(recipe: dict[str, Any]) -> dict[str, Any]:
    """Resolve the critical axes for an anvil-recipe ChemProp/CheMeleon baseline run.

    Only the CheMeleon->pEC50 baseline shape is understood here; a recipe that
    does not match it raises, so a new anvil run cannot be ingested with silently
    wrong axes.
    """
    model = recipe["procedure"]["model"]
    params = model["params"]
    from_foundation = params.get("from_foundation")

    # guard: this resolver only knows the single-task CheMeleon->pEC50 baseline
    if model["type"] != "ChemPropModel" or from_foundation != "chemeleon":
        raise ValueError(f"unrecognized anvil model shape: type={model['type']!r} from_foundation={from_foundation!r}")
    if recipe["data"]["target_cols"] != ["pEC50"] or params.get("n_tasks") != 1:
        raise ValueError(f"unexpected anvil target: {recipe['data']['target_cols']!r} n_tasks={params.get('n_tasks')!r}")

    # ChemProp end-to-end model initialized from the CheMeleon foundation and
    # trained straight to the pEC50 endpoint: no downstream tabular regressor,
    # so the feature-composition flags mirror the freeze/width/clip sweep's
    # end-to-end rows (embedding present, no separate readout, no descriptors)
    # and n_features is a category error (n_features_inapplicable).
    return {
        "encoder_family": "chemprop",
        "encoder_init": "chemeleon_pretrained",
        "encoder_target": "pec50",
        "has_embedding": True,
        "has_readout": False,
        "has_descriptors": False,
        "descriptor_sources": "",
        "regressor": "N/A",
        # dose-response pEC50 targets only, with no primary-screen/log2FC augmentation
        "train_data": "drc_only",
        "ffn_hidden_dim": int(params["ffn_hidden_dim"]),
        # freeze_weights null means the encoder trains from the first epoch (full
        # fine-tune), the freeze_epochs=0 end of the sweep's freeze schedule
        "freeze_epochs": 0 if model.get("freeze_weights") is None else None,
        "gradient_clip_val": float(recipe["procedure"]["train"]["params"]["gradient_clip_val"]),
    }


def _combine_baseline_metrics(phase_dirs: list[Path]) -> dict[str, float | None]:
    """Recombine per-phase anvil metrics into the sweep's combined 'overall' split.

    MAE and RMSE recombine exactly from the per-phase summaries (they are means of
    absolute and squared errors, so the pooled value is the count-weighted mean).
    The rank and relative-error metrics (r2, rae, kendall_tau, spearman_rho) need
    per-compound predictions, which the anvil runs do not persist, so they are left
    null rather than approximated.
    """
    total_n = 0
    sum_abs = 0.0
    sum_sq = 0.0
    for d in phase_dirs:
        metrics = json.loads((d / "regression_metrics.json").read_text())
        target = next(k for k in metrics if k != "tag")
        n = len(pd.read_csv(d / "data" / "y_test.csv"))
        sum_abs += n * metrics[target]["mae"]["value"]
        sum_sq += n * metrics[target]["mse"]["value"]
        total_n += n
    return {
        "mae": sum_abs / total_n,
        "n": total_n,
        "rmse": (sum_sq / total_n) ** 0.5,
        "rae": None,
        "r2": None,
        "kendall_tau": None,
        "spearman_rho": None,
    }


def build_anvil_baseline_spec() -> RunSpec | None:
    """Build the single combined RunSpec for the CheMeleon->pEC50 anvil baseline.

    Returns None if the baseline directories are absent. Requires every present
    phase to share one model recipe, so a divergent recipe fails loudly rather
    than collapsing two different models into one row.
    """
    phase_dirs = [RESULTS_DIR / name for name in ANVIL_BASELINE_DIRS if (RESULTS_DIR / name).is_dir()]
    if not phase_dirs:
        return None

    axes = None
    for d in phase_dirs:
        recipe = yaml.safe_load((d / "anvil_recipe.yaml").read_text())
        resolved = _resolve_anvil_axes(recipe)
        if axes is None:
            axes = resolved
        elif resolved != axes:
            raise ValueError(f"anvil baseline phases disagree on axes: {d.name} -> {resolved} vs {axes}")

    spec = RunSpec(
        base_dir=ANVIL_BASELINE_BASE_DIR,
        seed=UNPINNED_SEED,
        run_dir=str(phase_dirs[0]),
        spec_source="anvil_recipe",
        n_features_inapplicable=True,
    )
    for col, value in axes.items():
        setattr(spec, col, value)
    spec.metrics = _combine_baseline_metrics(phase_dirs)
    return spec


# Directive 3, third pass: base_dirs whose header comment in configs/<stem>.yaml
# is the *only* place a pretrain->fine-tune lineage is recorded (no code/config
# field states it). Read from every moal-schema config's header comment (see
# `_moal_header_pretrain_lineage`); populated once, at import time, below.
# Maps base_dir -> "yaml_comment"-sourced encoder_target/encoder_init overrides.
_YAML_COMMENT_LINEAGE: dict[str, dict[str, str]] = {}


def _moal_header_comment(config_stem: str) -> str:
    """Return the leading '#'-comment block of `configs/<config_stem>.yaml`, or "" if none."""
    cfg_path = CONFIGS_DIR / f"{config_stem}.yaml"
    if not cfg_path.exists():
        return ""
    lines = []
    for line in cfg_path.read_text().splitlines():
        if line.startswith("#"):
            lines.append(line)
        elif line.strip() == "":
            continue
        else:
            break
    return "\n".join(lines)


def _discover_yaml_comment_lineage() -> None:
    """Scan every moal-schema config's header comment for a pretrain->fine-tune lineage.

    Directive 3 (third pass): a human-authored header comment is a legitimate
    provenance source when, and only when, no code or config field encodes the
    axis at all. `e4_frozen.yaml`'s header is the worked example: it names a
    stage-1 log2FC pretrain (`e4_log2fc_pretrain/log2fc_pretrain.yaml`,
    converted via `convert_to_foundation_checkpoint.py`) that
    `model.from_foundation` only names as an opaque checkpoint path
    (`e4_log2fc_pretrain/log2fc_mp.pt`) — the path alone does not say what
    that checkpoint was pretrained on, only the comment does. Every other
    moal-schema config (`freeze*`, `feat_*`) was read too (see
    `resolve_moal_config_axes`'s module-level scan below) and none carries a
    header comment at all, so this dict only ever gains e4_* entries.
    """
    stems = ["e4_finetune", "e4_finetune_drc_only", "e4_frozen", "e4_frozen_drc_only"]
    comments = {stem: _moal_header_comment(stem) for stem in stems}
    lineage_value = {
        # encoder_target reflects what the encoder (MPNN body) was itself
        # fit to; here that is the frozen/warmup-frozen body's stage-1
        # objective (log2FC), not the readout's own pEC50 fine-tune target,
        # matching this column's meaning everywhere else in this table (e.g.
        # tabpfn_concat_features.py rows, whose auxiliary encoder is likewise
        # log2FC-trained regardless of what a downstream regressor predicts).
        "encoder_target": "log2fc",
        "encoder_init": "log2fc_checkpoint_pretrained",
    }
    for stem in stems:
        if "log2FC-pretrained" in comments[stem] or "pretrained on log2FC" in comments[stem]:
            _YAML_COMMENT_LINEAGE[stem] = lineage_value

    # e4_finetune_drc_only.yaml's own header never restates the lineage
    # directly ("Same as e4_finetune.yaml, except the main model trains on
    # DRC records only" — no "log2FC-pretrained" wording of its own), so the
    # direct-phrase scan above misses it. It shares e4_finetune's identical
    # from_foundation checkpoint path and its comment explicitly names
    # e4_finetune.yaml as the base it's "same as", so the lineage established
    # for e4_finetune.yaml applies here too by that cross-reference.
    same_as_re = re.compile(r"Same as (\S+)\.yaml")
    for stem in stems:
        if stem in _YAML_COMMENT_LINEAGE:
            continue
        m = same_as_re.search(comments[stem])
        if m and m.group(1) in _YAML_COMMENT_LINEAGE:
            _YAML_COMMENT_LINEAGE[stem] = lineage_value


_discover_yaml_comment_lineage()


def resolve_moal_config_axes(spec: RunSpec, cfg: dict[str, Any]) -> None:
    """Resolve axes for the moal/chemprop active-learning pipeline (has config_used.yaml).

    This pipeline trains a chemprop MPNN encoder end to end with an FFN head,
    driven by an active-learning loop config. It has no descriptor stage and
    no interchangeable regressor stage in the tabpfn/xgboost/lgbm sense
    (directive 2, third pass: a real end-to-end message-passing network has a
    built-in FFN readout, not a swappable regressor, so `regressor` is set to
    the resolved value "N/A" here, not left unknown).
    """
    spec.spec_source = "parsed_config"
    spec.encoder_family = "chemprop"

    model_cfg = cfg.get("model") or {}
    from_foundation = model_cfg.get("from_foundation")
    lineage = _YAML_COMMENT_LINEAGE.get(spec.base_dir)
    if lineage is not None:
        # Directive 3: header-comment-sourced lineage overrides the generic
        # from_foundation rule below for this base_dir's encoder_init, since
        # from_foundation here is a checkpoint path (neither "chemeleon" nor
        # false) that the generic rule alone cannot classify.
        spec.encoder_init = lineage["encoder_init"]
        spec.encoder_target = lineage["encoder_target"]
        if spec.spec_source == "parsed_config":
            spec.spec_source = "parsed_config+yaml_comment"
    elif from_foundation == "chemeleon":
        spec.encoder_init = "chemeleon_pretrained"
        spec.unknown_axes.append("encoder_target")
    elif from_foundation is False or from_foundation is None:
        spec.encoder_init = "scratch"
        spec.unknown_axes.append("encoder_target")
    else:
        # A custom checkpoint path with no header-comment lineage to explain
        # it — do not guess which foundation it derives from.
        spec.unknown_axes.append("encoder_init")
        spec.unknown_axes.append("encoder_target")

    # Directive 2: hd{512,1024} in the base_dir name / model.ffn_hidden_dim in
    # the config is the FFN readout's hidden width. The MPNN body's own
    # pooled-embedding width is not separately recorded in this schema
    # (message_hidden_dim is a message-passing hidden-layer width, not a
    # documented pooled-output width), so ffn_hidden_dim is the only width
    # this schema records for either the embedding or the readout — resolve
    # both from it, per the e4_frozen.yaml worked example.
    ffn_hidden_dim = model_cfg.get("ffn_hidden_dim")
    spec.ffn_hidden_dim = ffn_hidden_dim
    spec.freeze_epochs = model_cfg.get("freeze_epochs")
    trainer_cfg = cfg.get("trainer") or {}
    spec.gradient_clip_val = trainer_cfg.get("gradient_clip_val")

    spec.has_embedding = True
    if ffn_hidden_dim is not None:
        spec.embedding_native_dim = ffn_hidden_dim
        spec.readout_dim = ffn_hidden_dim
    else:
        spec.unknown_axes.append("embedding_native_dim")
    # No PCA-compression concept exists in this pipeline.
    spec.embedding_pca_dim = None

    aux_cfg = cfg.get("auxiliary_model")
    if isinstance(aux_cfg, dict):
        spec.has_readout = bool(aux_cfg.get("use_observed_readout") or aux_cfg.get("use_predicted_readout"))
        # Third pass: the two flags separately, not just their OR — the
        # feat_observed_2task/feat_observed_and_predicted_2task/
        # feat_predicted_native_2task trio only differs by which of these is
        # set, so has_readout alone (True for all three) collapses them.
        spec.aux_use_observed_readout = bool(aux_cfg.get("use_observed_readout", False))
        spec.aux_use_predicted_readout = bool(aux_cfg.get("use_predicted_readout", False))
    else:
        spec.has_readout = False
        spec.aux_use_observed_readout = False
        spec.aux_use_predicted_readout = False

    # This pipeline's "regressor" is a built-in FFN readout, not one of the
    # interchangeable tabular regressors this table's vocabulary covers — a
    # resolved fact ("N/A"), not an unknown (directive 2, third pass).
    spec.regressor = "N/A"

    # No descriptor stage anywhere in this schema.
    spec.has_descriptors = False
    spec.descriptor_sources = ""
    spec.descriptor_pca_width = None

    data_cfg = cfg.get("data") or {}
    plan_cfg = data_cfg.get("plan") or {}
    input_csv = plan_cfg.get("input_csv")
    if isinstance(input_csv, str) and input_csv:
        spec.train_data = "drc_only" if "drc_only" in input_csv else "drc_plus_primary"
    else:
        spec.unknown_axes.append("train_data")

    # n_features has no meaning for this pipeline even where
    # embedding_native_dim/readout_dim resolve: there is no concatenated
    # final feature vector fed to a downstream regressor (there is no
    # downstream regressor) for compute_n_features to sum widths toward.
    spec.n_features_inapplicable = True
    spec.unknown_axes.append("n_features")


# Third pass, directive 4: base_dir names of the form "freeze{F}_hd{HD}_clipX"
# encode freeze_epochs/ffn_hidden_dim/gradient_clip_val in the directory name
# itself, independently of config_used.yaml, so this is a second evidence
# source (like LAUNCH_MAP) worth cross-checking rather than trusting blindly.
_FREEZE_HD_CLIP_RE = re.compile(r"^freeze(?P<freeze>\d+)_hd(?P<hd>\d+)_clip(?P<clip>[0-9.]+|off)$")


def _cross_check_freeze_hd_clip_naming(spec: RunSpec) -> None:
    """Cross-check freeze/hd/clip values parsed from base_dir against the resolved config axes.

    Only applies to base_dirs matching the "freeze{F}_hd{HD}_clip{C}" naming
    convention (the freeze0/1/2 x hd512/1024 x clip1.0/5.0/off sweep). Any
    mismatch between what the name encodes and what config_used.yaml/the
    launch-script resolution actually produced is recorded in
    `spec.disagreements`, the same channel `_apply_launch_axes` uses, rather
    than silently trusting either source.
    """
    m = _FREEZE_HD_CLIP_RE.match(spec.base_dir)
    if m is None:
        return
    name_freeze = int(m.group("freeze"))
    name_hd = int(m.group("hd"))
    name_clip = None if m.group("clip") == "off" else float(m.group("clip"))

    if spec.freeze_epochs is not None and spec.freeze_epochs != name_freeze:
        spec.disagreements.append(
            f"freeze_epochs: resolved={spec.freeze_epochs!r} vs base_dir-name={name_freeze!r}"
        )
    if spec.ffn_hidden_dim is not None and spec.ffn_hidden_dim != name_hd:
        spec.disagreements.append(
            f"ffn_hidden_dim: resolved={spec.ffn_hidden_dim!r} vs base_dir-name={name_hd!r}"
        )
    if spec.gradient_clip_val is not None or name_clip is not None:
        if spec.gradient_clip_val != name_clip:
            spec.disagreements.append(
                f"gradient_clip_val: resolved={spec.gradient_clip_val!r} vs base_dir-name={name_clip!r}"
            )


def resolve_axes_no_config(spec: RunSpec) -> None:
    """Mark every critical axis unresolved: no in-run config artifact survives for this run."""
    spec.spec_source = None
    for col in CRITICAL_COLUMNS:
        spec.unknown_axes.append(col)


# ---------------------------------------------------------------------------
# Launch-script evidence: every invocation in run_seed_sweep*.sh,
# run_sweep.sh, run_cpu_oom_reproductions.sh, run_gnn_seed_sweep.py, and
# run_tabfm_param_sweep.py, transcribed by reading each script in full.
# ---------------------------------------------------------------------------


def _yaml_from_foundation(config_stem: str, block: str) -> str | bool | None:
    """Read `<block>.from_foundation` out of `configs/<config_stem>.yaml`."""
    with (CONFIGS_DIR / f"{config_stem}.yaml").open() as fh:
        cfg = yaml.safe_load(fh)
    return (cfg.get(block) or {}).get("from_foundation")


def _yaml_message_hidden_dim(config_stem: str, block: str) -> int | None:
    """Read `<block>.message_hidden_dim` out of `configs/<config_stem>.yaml`, if present."""
    with (CONFIGS_DIR / f"{config_stem}.yaml").open() as fh:
        cfg = yaml.safe_load(fh)
    return (cfg.get(block) or {}).get("message_hidden_dim")


def _tabpfn_concat_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for a `tabpfn_concat_features.py` invocation.

    Encoder is the config's `auxiliary_model` block (log2FC multi-task
    objective, per the script's own docstring and `tabpfn_chemeleon_log2fc_features.py`'s
    docstring, which calls out that both scripts pretrain with "the same
    log2FC multi-task objective"). `--no-embedding`/`--no-readout` gate the
    embedding/readout blocks; `--descriptor-source`/`--pca-components` gate
    the descriptor block (always present unless the descriptor-source flag
    is unreachable, which this script never leaves unset).
    """
    no_embedding = bool(flags.get("no-embedding", False))
    no_readout = bool(flags.get("no-readout", False))
    native_dim = _yaml_message_hidden_dim(config_stem, "auxiliary_model")
    from_foundation = _yaml_from_foundation(config_stem, "auxiliary_model")
    encoder_init = (
        "chemeleon_pretrained" if from_foundation == "chemeleon" else "scratch" if from_foundation is False else None
    )
    return {
        "encoder_family": "chemprop",
        "encoder_init": encoder_init,
        "encoder_target": "log2fc",
        "embedding_native_dim": native_dim,
        "embedding_pca_dim": None,  # this script never PCA-compresses the embedding block
        "has_embedding": not no_embedding,
        "has_readout": not no_readout,
        "readout_dim": _READOUT_DIM if not no_readout else 0,
        "has_descriptors": True,
        "descriptor_sources": flags.get("descriptor-source", "all"),
        "descriptor_pca_width": int(flags.get("pca-components", 128)),
        "regressor": flags.get("regressor", "tabpfn"),
        "train_data": _TRAIN_DATA,
    }


def _tabpfn_frozen_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for a `tabpfn_frozen_features.py` invocation (embedding/readout only, no descriptors)."""
    no_embedding = bool(flags.get("no-embedding", False))
    no_readout = bool(flags.get("no-readout", False))
    native_dim = _yaml_message_hidden_dim(config_stem, "auxiliary_model")
    from_foundation = _yaml_from_foundation(config_stem, "auxiliary_model")
    encoder_init = (
        "chemeleon_pretrained" if from_foundation == "chemeleon" else "scratch" if from_foundation is False else None
    )
    return {
        "encoder_family": "chemprop",
        "encoder_init": encoder_init,
        "encoder_target": "log2fc",
        "embedding_native_dim": native_dim,
        "embedding_pca_dim": None,
        "has_embedding": not no_embedding,
        "has_readout": not no_readout,
        "readout_dim": _READOUT_DIM if not no_readout else 0,
        "has_descriptors": False,
        "descriptor_sources": "",
        "descriptor_pca_width": None,
        "regressor": "tabpfn",  # script hard-codes TabPFNRegressor(), no --regressor flag
        "train_data": _TRAIN_DATA,
    }


def _tabpfn_chemeleon_log2fc_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for a `tabpfn_chemeleon_log2fc_features.py` invocation.

    Fine-tunes a CheMeleon-initialized auxiliary encoder on the same log2FC
    objective as `tabpfn_concat_features.py`'s from-scratch cell.
    """
    no_readout = bool(flags.get("no-readout", False))
    return {
        "encoder_family": "chemprop",
        "encoder_init": "chemeleon_pretrained",
        "encoder_target": "log2fc",
        "embedding_native_dim": _CHEMELEON_NATIVE_DIM,
        "embedding_pca_dim": int(flags.get("embedding-pca-components", 128)),
        "has_embedding": True,
        "has_readout": not no_readout,
        "readout_dim": _READOUT_DIM if not no_readout else 0,
        "has_descriptors": True,
        "descriptor_sources": flags.get("descriptor-source", "mordred"),
        "descriptor_pca_width": int(flags.get("descriptor-pca-components", 128)),
        "regressor": "tabpfn",  # script hard-codes TabPFNRegressor(), no --regressor flag
        "train_data": _TRAIN_DATA,
    }


def _tabpfn_pec50_chemeleon_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for a `tabpfn_pec50_chemeleon_features.py` invocation.

    Fine-tunes `cfg.model` (the same chemprop block the moal pipeline trains
    end to end) directly on pEC50 and pulls its pooled embedding via
    `embed_smiles`, so encoder_target is "pec50", not "log2fc". Config-driven
    via `from_foundation`, despite the filename (per the script's own
    docstring/comments), so this same resolver applies whether the config is
    CheMeleon-initialized (freeze1_hd512_clip5.0.yaml) or from-scratch
    (tabpfn_pec50_scratch.yaml).
    """
    from_foundation = _yaml_from_foundation(config_stem, "model")
    encoder_init = (
        "chemeleon_pretrained" if from_foundation == "chemeleon" else "scratch" if from_foundation is False else None
    )
    return {
        "encoder_family": "chemprop",
        "encoder_init": encoder_init,
        "encoder_target": "pec50",
        "embedding_native_dim": _CHEMELEON_NATIVE_DIM,
        "embedding_pca_dim": int(flags.get("embedding-pca-components", 256)),
        "has_embedding": True,
        "has_readout": False,  # this script has no readout concept at all
        "readout_dim": None,
        "has_descriptors": True,
        "descriptor_sources": flags.get("descriptor-source", "all"),
        "descriptor_pca_width": int(flags.get("descriptor-pca-components", 256)),
        "regressor": "tabpfn",  # script hard-codes TabPFNRegressor(), no --regressor flag
        "train_data": _TRAIN_DATA,
    }


def _tabpfn_chemeleon_static_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for a `tabpfn_chemeleon_static_features.py` invocation.

    No fine-tuning at all: CheMeleon's untouched pretrained weights, embedded
    via `embed_smiles` without ever calling `refit`.
    """
    skip_descriptors = bool(flags.get("skip-descriptors", False))
    return {
        "encoder_family": "chemeleon",
        "encoder_init": "chemeleon_pretrained",
        "encoder_target": "none",  # never fine-tuned
        "embedding_native_dim": _CHEMELEON_NATIVE_DIM,
        "embedding_pca_dim": int(flags.get("embedding-pca-components", 256)),
        "has_embedding": True,
        "has_readout": False,  # this script has no readout concept
        "readout_dim": None,
        "has_descriptors": not skip_descriptors,
        "descriptor_sources": "" if skip_descriptors else "all",  # combined RDKit+Mordred, no --descriptor-source flag
        "descriptor_pca_width": None if skip_descriptors else int(flags.get("descriptor-pca-components", 256)),
        "regressor": "tabpfn",  # script hard-codes TabPFNRegressor(), no --regressor flag
        "train_data": _TRAIN_DATA,
    }


def _tabpfn_chemeleon_concat_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for a `tabpfn_chemeleon_concat_features.py` invocation.

    Also no fine-tuning of the CheMeleon embedding itself (docstring: "Loading
    pretrained CheMeleon weights (no fine-tuning)"); the readout block, when
    included, is *borrowed* from the from-scratch encoder's own cache, not
    produced by this run's encoder.
    """
    no_descriptors = bool(flags.get("no-descriptors", False))
    no_readout = bool(flags.get("no-readout", False))
    return {
        "encoder_family": "chemeleon",
        "encoder_init": "chemeleon_pretrained",
        "encoder_target": "none",  # never fine-tuned
        "embedding_native_dim": _CHEMELEON_NATIVE_DIM,
        "embedding_pca_dim": int(flags.get("embedding-pca-components", 256)),
        "has_embedding": True,
        "has_readout": not no_readout,  # borrowed from a from-scratch log2FC encoder's cache
        "readout_dim": _READOUT_DIM if not no_readout else 0,
        "has_descriptors": not no_descriptors,
        "descriptor_sources": "" if no_descriptors else "mordred",  # this script hard-codes mordred-only
        "descriptor_pca_width": None if no_descriptors else int(flags.get("pca-components", 128)),
        "regressor": flags.get("regressor", "tabpfn"),
        "train_data": _TRAIN_DATA,
    }


def _tabpfn_uncertainty_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for the single `tabpfn_uncertainty_analysis.py` invocation in its own docstring.

    Reads the canonical (non-seed-specific) embedding cache produced from
    `configs/tabpfn_small_embed.yaml`'s from-scratch auxiliary encoder;
    embedding+readout+descriptors, regressor hard-coded to TabPFN.
    """
    return {
        "encoder_family": "chemprop",
        "encoder_init": "scratch",
        "encoder_target": "log2fc",
        "embedding_native_dim": _yaml_message_hidden_dim(config_stem, "auxiliary_model"),
        "embedding_pca_dim": None,
        "has_embedding": True,
        "has_readout": True,
        "readout_dim": _READOUT_DIM,
        "has_descriptors": True,
        "descriptor_sources": flags.get("descriptor-source", "mordred"),
        "descriptor_pca_width": int(flags.get("pca-components", 128)),
        "regressor": "tabpfn",
        "train_data": _TRAIN_DATA,
    }


def _moal_plan_axes(config_stem: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Axes for a `moal plan` invocation (run_sweep.sh / run_gnn_seed_sweep.py).

    Covers only the axes the moal config schema itself pins (mirrors
    `resolve_moal_config_axes`'s own rules), so this is a cross-check against
    the in-run `config_used.yaml` these runs already carry, not a source of
    new axes: `encoder_family`, `encoder_init`, and `train_data` are the only
    keys returned.
    """
    from_foundation = _yaml_from_foundation(config_stem, "model")
    encoder_init = (
        "chemeleon_pretrained" if from_foundation == "chemeleon" else "scratch" if from_foundation is False else None
    )
    with (CONFIGS_DIR / f"{config_stem}.yaml").open() as fh:
        cfg = yaml.safe_load(fh)
    input_csv = ((cfg.get("data") or {}).get("plan") or {}).get("input_csv", "")
    train_data = "drc_only" if "drc_only" in input_csv else "drc_plus_primary"
    axes: dict[str, Any] = {"encoder_family": "chemprop", "train_data": train_data}
    if encoder_init is not None:
        axes["encoder_init"] = encoder_init
    return axes


_AXES_RESOLVERS = {
    "moal plan": _moal_plan_axes,
    "tabpfn_concat_features.py": _tabpfn_concat_axes,
    "tabpfn_frozen_features.py": _tabpfn_frozen_axes,
    "tabpfn_chemeleon_log2fc_features.py": _tabpfn_chemeleon_log2fc_axes,
    "tabpfn_pec50_chemeleon_features.py": _tabpfn_pec50_chemeleon_axes,
    "tabpfn_chemeleon_static_features.py": _tabpfn_chemeleon_static_axes,
    "tabpfn_chemeleon_concat_features.py": _tabpfn_chemeleon_concat_axes,
    "tabpfn_uncertainty_analysis.py": _tabpfn_uncertainty_axes,
}


def _seeded(base: str, seeds: range = range(5)) -> list[str]:
    """Expand a base output-dir name into its 5-seed replicate names."""
    return [f"{base}_seed{s}" for s in seeds]


# Each entry: (output_dir, script, config_stem, flags). `flags` uses argparse
# long-option names (without leading --) as keys; boolean store_true flags
# are present with value True only when the invocation actually passes them.
LAUNCH_INVOCATIONS: list[tuple[str, str, str, dict[str, Any]]] = []


def _add(output_dirs: list[str], script: str, config_stem: str, flags: dict[str, Any]) -> None:
    for out in output_dirs:
        LAUNCH_INVOCATIONS.append((out, script, config_stem, flags))


# --- run_seed_sweep.sh, recipe 1: from-scratch log2FC encoder ---
for _name, _extra in [
    ("tabpfn_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabpfn"}),
    ("tabpfn_embed_readout_mordred_pca256", {"descriptor-source": "mordred", "pca-components": 256, "regressor": "tabpfn"}),
    ("tabpfn_embed_readout_mordred_pca64", {"descriptor-source": "mordred", "pca-components": 64, "regressor": "tabpfn"}),
    ("tabicl_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabicl"}),
    ("tabfm_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabfm"}),
    ("lgbm_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "lgbm"}),
    ("xgboost_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "xgboost"}),
]:
    _add(_seeded(_name), "tabpfn_concat_features.py", "tabpfn_small_embed", _extra)
_add(_seeded("tabpfn_frozen_features"), "tabpfn_frozen_features.py", "tabpfn_small_embed", {})

# --- run_seed_sweep.sh, recipe 2: CheMeleon-init log2FC encoder ---
_add(
    _seeded("tabpfn_chemeleon_log2fc_mordred_pca128"),
    "tabpfn_chemeleon_log2fc_features.py",
    "tabpfn_chemeleon_log2fc",
    {"descriptor-source": "mordred"},
)

# --- run_seed_sweep_recipe3.sh: pEC50-direct CheMeleon encoder, rerun after OOM. ---
# Supersedes run_seed_sweep.sh's own recipe-3 block (256/256 PCA) at the same
# output dirs: recipe3.sh's pass 2 is the one that actually fits and writes
# TabPFN's predictions, at 128/128, so that is this table's resolution.
_add(
    _seeded("tabpfn_pec50_chemeleon"),
    "tabpfn_pec50_chemeleon_features.py",
    "freeze1_hd512_clip5.0",
    {"embedding-pca-components": 128, "descriptor-pca-components": 128},
)

# --- run_seed_sweep_recipe4.sh: from-scratch/pEC50-trained cell ---
_add(
    _seeded("tabpfn_pec50_scratch"),
    "tabpfn_pec50_chemeleon_features.py",
    "tabpfn_pec50_scratch",
    {"embedding-pca-components": 128, "descriptor-pca-components": 128},
)

# --- run_seed_sweep_recipe5.sh: panel 01/02 single-ingredient/pairwise rows ---
_add(_seeded("tabpfn_embed_only_no_readout"), "tabpfn_frozen_features.py", "tabpfn_small_embed", {"no-readout": True})
_add(_seeded("tabpfn_readout_only_no_embed"), "tabpfn_frozen_features.py", "tabpfn_small_embed", {"no-embedding": True})
_add(_seeded("tabpfn_small_embed"), "tabpfn_frozen_features.py", "tabpfn_small_embed", {})
_add(
    _seeded("tabpfn_mordred_only_pca128_no_embed_no_readout"),
    "tabpfn_concat_features.py",
    "tabpfn_small_embed",
    {"no-embedding": True, "no-readout": True, "descriptor-source": "mordred", "pca-components": 128},
)
_add(
    _seeded("tabpfn_rdkit_only_pca128_no_embed_no_readout"),
    "tabpfn_concat_features.py",
    "tabpfn_small_embed",
    {"no-embedding": True, "no-readout": True, "descriptor-source": "rdkit", "pca-components": 128},
)
_add(
    _seeded("tabpfn_embed_mordred_pca128_no_readout"),
    "tabpfn_concat_features.py",
    "tabpfn_small_embed",
    {"no-readout": True, "descriptor-source": "mordred", "pca-components": 128},
)
_add(
    _seeded("tabpfn_readout_mordred_pca128_no_embed"),
    "tabpfn_concat_features.py",
    "tabpfn_small_embed",
    {"no-embedding": True, "descriptor-source": "mordred", "pca-components": 128},
)
_add(_seeded("tabpfn_chemeleon_static"), "tabpfn_chemeleon_static_features.py", "freeze1_hd512_clip5.0", {})
_add(
    _seeded("tabpfn_chemeleon_embed_only"),
    "tabpfn_chemeleon_static_features.py",
    "freeze1_hd512_clip5.0",
    {"skip-descriptors": True},
)

# --- run_seed_sweep_recipe6.sh: RDKit+Mordred combined-descriptor PCA-width sweep ---
for _name, _pca in [("tabpfn_concat_pca64", 64), ("tabpfn_concat_small_embed", 128), ("tabpfn_concat_pca256", 256)]:
    _add(
        _seeded(_name),
        "tabpfn_concat_features.py",
        "tabpfn_small_embed",
        {"descriptor-source": "all", "pca-components": _pca, "regressor": "tabpfn"},
    )

# --- run_seed_sweep_recipe7.sh: regressor sweep on readout+descriptors ---
for _name, _regressor, _extra in [
    ("tabpfn-v2.6_readout_mordred_pca128_no_embed", "tabpfn-v2.6", {}),
    ("tabpfn-v3_readout_mordred_pca128_no_embed", "tabpfn-v3", {}),
    ("tabfm_n32_readout_mordred_pca128_no_embed", "tabfm", {"tabfm-n-estimators": 32}),
    ("lgbm_readout_mordred_pca128_no_embed", "lgbm", {}),
    ("xgboost_readout_mordred_pca128_no_embed", "xgboost", {}),
]:
    _add(
        _seeded(_name),
        "tabpfn_concat_features.py",
        "tabpfn_small_embed",
        {"no-embedding": True, "descriptor-source": "mordred", "pca-components": 128, "regressor": _regressor, **_extra},
    )

# --- run_seed_sweep_recipe8.sh: off-the-shelf CheMeleon embedding side experiment ---
_add(_seeded("tabpfn_chemeleon_readout_descriptors"), "tabpfn_chemeleon_concat_features.py", "freeze1_hd512_clip5.0", {})
_add(
    _seeded("tabpfn_chemeleon_descriptors_only"),
    "tabpfn_chemeleon_concat_features.py",
    "freeze1_hd512_clip5.0",
    {"no-readout": True},
)
_add(
    _seeded("tabpfn_chemeleon_readout_only"),
    "tabpfn_chemeleon_concat_features.py",
    "freeze1_hd512_clip5.0",
    {"no-descriptors": True},
)

# --- run_seed_sweep_recipe9.sh: regressor sweep on chemeleon_readout_descriptors ---
for _name, _regressor, _extra in [
    ("tabpfn-v2.6_chemeleon_readout_descriptors", "tabpfn-v2.6", {}),
    ("tabpfn-v3_chemeleon_readout_descriptors", "tabpfn-v3", {}),
    ("tabfm_n32_chemeleon_readout_descriptors", "tabfm", {"tabfm-n-estimators": 32}),
    ("lgbm_chemeleon_readout_descriptors", "lgbm", {}),
    ("xgboost_chemeleon_readout_descriptors", "xgboost", {}),
]:
    _add(
        _seeded(_name),
        "tabpfn_chemeleon_concat_features.py",
        "freeze1_hd512_clip5.0",
        {"regressor": _regressor, **_extra},
    )

# --- run_seed_sweep_recipe10.sh: TabICL on chemeleon_readout_only ---
_add(
    _seeded("tabicl_chemeleon_readout_only"),
    "tabpfn_chemeleon_concat_features.py",
    "freeze1_hd512_clip5.0",
    {"regressor": "tabicl", "no-descriptors": True},
)

# --- run_seed_sweep_recipe11.sh: regressor sweep on chemeleon_readout_only ---
for _name, _regressor, _extra in [
    ("tabpfn-v2.6_chemeleon_readout_only", "tabpfn-v2.6", {}),
    ("tabpfn-v3_chemeleon_readout_only", "tabpfn-v3", {}),
    ("tabfm_n32_chemeleon_readout_only", "tabfm", {"tabfm-n-estimators": 32}),
    ("lgbm_chemeleon_readout_only", "lgbm", {}),
    ("xgboost_chemeleon_readout_only", "xgboost", {}),
]:
    _add(
        _seeded(_name),
        "tabpfn_chemeleon_concat_features.py",
        "freeze1_hd512_clip5.0",
        {"regressor": _regressor, "no-descriptors": True, **_extra},
    )

# --- run_tabfm_param_sweep.py: TabFM ensemble-size sweep ---
for _n_est in (16, 32, 64):
    _add(
        _seeded(f"tabfm_n{_n_est}_embed_readout_mordred_pca128"),
        "tabpfn_concat_features.py",
        "tabpfn_small_embed",
        {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabfm", "tabfm-n-estimators": _n_est},
    )

# --- tabpfn_uncertainty_analysis.py: single invocation, its own docstring ---
_add(["tabpfn_uncertainty_mordred_pca128"], "tabpfn_uncertainty_analysis.py", "tabpfn_small_embed", {"descriptor-source": "mordred", "pca-components": 128})

# --- run_cpu_oom_reproductions.sh: two single seed-0 CPU repro cases ---
_add(
    ["cpu_oom_repro_tabicl_chemeleon_readout_descriptors"],
    "tabpfn_chemeleon_concat_features.py",
    "freeze1_hd512_clip5.0",
    {"regressor": "tabicl", "device": "cpu"},
)
_add(
    ["cpu_oom_repro_tabfm_full_rows"],
    "tabpfn_chemeleon_concat_features.py",
    "freeze1_hd512_clip5.0",
    {"regressor": "tabfm", "tabfm-max-rows": 0, "tabfm-n-estimators": 4, "device": "cpu"},
)

# --- run_sweep.sh: `moal plan` against every configs/*.yaml, output dir = stem ---
# Runs unconditionally over every config in configs/, skipping only ones that
# already have a predictions CSV; this also explains why the un-seeded
# results/tabpfn_small_embed/ directory carries the moal-schema
# config_used.yaml (its own name notwithstanding) while its
# tabpfn_small_embed_seed{0..4} siblings, produced instead by
# tabpfn_frozen_features.py via recipe5.sh, do not.
for _config_path in sorted(CONFIGS_DIR.glob("*.yaml")):
    _add([_config_path.stem], "moal plan", _config_path.stem, {})

# --- run_gnn_seed_sweep.py: `moal plan` against seed-variant copies of the
# freeze*/e4* configs, 5 seeds each, output dir = "<config-stem>_seed{N}" ---
_GNN_CONFIG_STEMS = [
    f"freeze{f}_hd{hd}_clip{clip}"
    for f in (0, 1, 2)
    for hd in (512, 1024)
    for clip in ("1.0", "5.0", "off")
] + ["e4_finetune", "e4_finetune_drc_only", "e4_frozen", "e4_frozen_drc_only"]
for _stem in _GNN_CONFIG_STEMS:
    _add(_seeded(_stem), "moal plan", _stem, {})

LAUNCH_MAP: dict[str, tuple[str, str, dict[str, Any]]] = {}
for _out, _script, _cfg, _flags in LAUNCH_INVOCATIONS:
    LAUNCH_MAP[_out] = (_script, _cfg, _flags)


def resolve_launch_axes(run_dir_name: str) -> dict[str, Any] | None:
    """Resolve every critical axis for `run_dir_name` from its launch-script invocation, if any."""
    entry = LAUNCH_MAP.get(run_dir_name)
    if entry is None:
        return None
    script, config_stem, flags = entry
    resolver = _AXES_RESOLVERS.get(script)
    if resolver is None:
        return None
    return resolver(config_stem, flags)


# ---------------------------------------------------------------------------
# Convention evidence: a lower-trust third layer for result directories that
# no current launch script emits verbatim, but whose full feature spec is still
# recoverable. Two populations land here: pre-seed-sweep single runs (the
# unseeded originals whose _seed0..4 replicates the launch table already
# covers) and rewritten-recipe runs (tabicl and pinned-tabpfn-version variants
# that later recipe edits dropped, e.g. recipe 7/9's tabicl exclusion). Each
# entry is resolved from NEXT_STEPS.md's running log, the naming convention it
# documents, the invoked script's argparse defaults, and the axes of a seeded
# sibling already in the launch table, run through the SAME axis resolvers.
# Kept separate from LAUNCH_INVOCATIONS so the "launch_script" tier stays a
# pure verbatim transcription; these rows are stamped spec_source="convention".
# ---------------------------------------------------------------------------

CONVENTION_INVOCATIONS: list[tuple[str, str, str, dict[str, Any]]] = []


def _add_conv(output_dirs: list[str], script: str, config_stem: str, flags: dict[str, Any]) -> None:
    for out in output_dirs:
        CONVENTION_INVOCATIONS.append((out, script, config_stem, flags))


# From-scratch log2FC concat one-offs (fixed-embedding cache from
# configs/tabpfn_small_embed.yaml), pre-seed-sweep originals from NEXT_STEPS.md's
# "Where we landed" table. Axes match their recipe-1/5/6 seeded siblings.
for _name, _flags in [
    ("tabpfn_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabpfn"}),
    ("tabpfn_embed_readout_mordred_pca256", {"descriptor-source": "mordred", "pca-components": 256, "regressor": "tabpfn"}),
    ("tabpfn_embed_readout_mordred_pca64", {"descriptor-source": "mordred", "pca-components": 64, "regressor": "tabpfn"}),
    ("tabicl_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabicl"}),
    ("tabfm_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabfm"}),
    ("lgbm_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "lgbm"}),
    ("xgboost_embed_readout_mordred_pca128", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "xgboost"}),
    # RDKit-only descriptors (NEXT_STEPS.md L44, renamed from tabpfn_concat_rdkit_only)
    ("tabpfn_embed_readout_rdkit_pca128", {"descriptor-source": "rdkit", "pca-components": 128, "regressor": "tabpfn"}),
    # RDKit+Mordred combined descriptor block (NEXT_STEPS.md L38/40/42)
    ("tabpfn_concat_pca256", {"descriptor-source": "all", "pca-components": 256, "regressor": "tabpfn"}),
    ("tabpfn_concat_pca64", {"descriptor-source": "all", "pca-components": 64, "regressor": "tabpfn"}),
    ("tabpfn_concat_small_embed", {"descriptor-source": "all", "pca-components": 128, "regressor": "tabpfn"}),
    # single-block ablations
    ("tabpfn_embed_mordred_pca128_no_readout", {"no-readout": True, "descriptor-source": "mordred", "pca-components": 128, "regressor": "tabpfn"}),
    ("tabpfn_readout_mordred_pca128_no_embed", {"no-embedding": True, "descriptor-source": "mordred", "pca-components": 128, "regressor": "tabpfn"}),
    ("tabpfn_mordred_only_pca128_no_embed_no_readout", {"no-embedding": True, "no-readout": True, "descriptor-source": "mordred", "pca-components": 128, "regressor": "tabpfn"}),
    ("tabpfn_rdkit_only_pca128_no_embed_no_readout", {"no-embedding": True, "no-readout": True, "descriptor-source": "rdkit", "pca-components": 128, "regressor": "tabpfn"}),
]:
    _add_conv([_name], "tabpfn_concat_features.py", "tabpfn_small_embed", _flags)

# Regressor-swept concat runs the current recipes dropped: recipe 7 moved the
# pinned-TabPFN-version sweep off the embed+readout featureset onto the no_embed
# one, and excluded tabicl entirely (OOM); these are the earlier seeded runs on
# the panel-03 original featureset, plus the tabicl no_embed run recipe 7 left out.
_add_conv(_seeded("tabpfn-v2.6_embed_readout_mordred_pca128"), "tabpfn_concat_features.py", "tabpfn_small_embed", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabpfn-v2.6"})
_add_conv(_seeded("tabpfn-v3_embed_readout_mordred_pca128"), "tabpfn_concat_features.py", "tabpfn_small_embed", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabpfn-v3"})
_add_conv(_seeded("tabicl_readout_mordred_pca128_no_embed"), "tabpfn_concat_features.py", "tabpfn_small_embed", {"no-embedding": True, "descriptor-source": "mordred", "pca-components": 128, "regressor": "tabicl"})
# One manual replicate beyond the 0..4 sweep range; identical featureset to the
# recipe-1 tabicl_embed_readout_mordred_pca128_seed0..4 siblings.
_add_conv(["tabicl_embed_readout_mordred_pca128_seed5"], "tabpfn_concat_features.py", "tabpfn_small_embed", {"descriptor-source": "mordred", "pca-components": 128, "regressor": "tabicl"})

# Frozen embedding/readout one-offs (no descriptors), recipe-5 seeded siblings.
_add_conv(["tabpfn_embed_only_no_readout"], "tabpfn_frozen_features.py", "tabpfn_small_embed", {"no-readout": True})
_add_conv(["tabpfn_readout_only_no_embed"], "tabpfn_frozen_features.py", "tabpfn_small_embed", {"no-embedding": True})

# CheMeleon-init log2FC one-off (recipe-2 seeded sibling); NEXT_STEPS.md L36.
_add_conv(["tabpfn_chemeleon_log2fc_mordred_pca128"], "tabpfn_chemeleon_log2fc_features.py", "tabpfn_chemeleon_log2fc", {"descriptor-source": "mordred"})

# pEC50-direct one-offs. NEXT_STEPS.md pins the widths: L52 (chemeleon, RDKit+Mordred
# PCA(128)+PCA(128)) and L47 (from-scratch, Mordred PCA-128).
_add_conv(["tabpfn_pec50_chemeleon"], "tabpfn_pec50_chemeleon_features.py", "freeze1_hd512_clip5.0", {"embedding-pca-components": 128, "descriptor-pca-components": 128})
_add_conv(["tabpfn_pec50_scratch_mordred_pca128"], "tabpfn_pec50_chemeleon_features.py", "tabpfn_pec50_scratch", {"embedding-pca-components": 128, "descriptor-pca-components": 128, "descriptor-source": "mordred"})
# Incomplete run: only embed_cache.npz on disk, no eval output — the 256+256
# CheMeleon OOM documented in NEXT_STEPS.md L276-291. Intended axes resolve
# (script defaults, 256/256, all); its metric columns stay null legitimately.
_add_conv(["tabpfn_pec50_chemeleon_full"], "tabpfn_pec50_chemeleon_features.py", "freeze1_hd512_clip5.0", {})

# Off-the-shelf CheMeleon (never fine-tuned) one-offs. NEXT_STEPS.md L53 pins
# tabpfn_chemeleon_static at PCA(128)+PCA(128) (its recipe-5 seeded siblings use
# the script's 256/256 defaults instead, a genuine pre-fix width difference).
# tabpfn_chemeleon_embed_only carries no independent width log; its embedding
# PCA width is inferred from its static companion (128) and is the least-supported
# axis in this table.
_add_conv(["tabpfn_chemeleon_static"], "tabpfn_chemeleon_static_features.py", "freeze1_hd512_clip5.0", {"embedding-pca-components": 128, "descriptor-pca-components": 128})
_add_conv(["tabpfn_chemeleon_embed_only"], "tabpfn_chemeleon_static_features.py", "freeze1_hd512_clip5.0", {"skip-descriptors": True, "embedding-pca-components": 128})

# TabICL on CheMeleon concat featuresets, which recipe 9 excluded (386-col OOM).
# These seeded runs completed (each has eval_out.csv); the _pca64 variant narrows
# the descriptor block to 64 to fit under the ceiling.
_add_conv(_seeded("tabicl_chemeleon_readout_descriptors"), "tabpfn_chemeleon_concat_features.py", "freeze1_hd512_clip5.0", {"regressor": "tabicl"})
_add_conv(_seeded("tabicl_chemeleon_readout_descriptors_pca64"), "tabpfn_chemeleon_concat_features.py", "freeze1_hd512_clip5.0", {"regressor": "tabicl", "pca-components": 64})
# recipe-10's tabicl_chemeleon_readout_only featureset (CheMeleon embed + borrowed
# readout, no descriptors) put through OOF-isotonic calibration
# (tabpfn_chemeleon_calibrate_predictions.py). Calibration is a monotonic post-map
# on predictions, so the feature/regressor/encoder axes are unchanged.
_add_conv(_seeded("tabicl_chemeleon_readout_only_calibrated"), "tabpfn_chemeleon_concat_features.py", "freeze1_hd512_clip5.0", {"regressor": "tabicl", "no-descriptors": True})

CONVENTION_MAP: dict[str, tuple[str, str, dict[str, Any]]] = {}
for _out, _script, _cfg, _flags in CONVENTION_INVOCATIONS:
    if _out in LAUNCH_MAP:
        raise ValueError(f"convention entry {_out!r} collides with a launch-script entry; the launch tier already resolves it")
    CONVENTION_MAP[_out] = (_script, _cfg, _flags)


def resolve_convention_axes(run_dir_name: str) -> dict[str, Any] | None:
    """Resolve every critical axis for `run_dir_name` from the convention layer, if any."""
    entry = CONVENTION_MAP.get(run_dir_name)
    if entry is None:
        return None
    script, config_stem, flags = entry
    resolver = _AXES_RESOLVERS.get(script)
    if resolver is None:
        return None
    return resolver(config_stem, flags)


def compute_n_features(spec: RunSpec) -> None:
    """Derive n_features from resolved embedding/readout/descriptor widths, if possible."""
    if spec.n_features_inapplicable:
        spec.unknown_axes.append("n_features")
        return
    if spec.has_embedding is None or spec.has_readout is None or spec.has_descriptors is None:
        spec.unknown_axes.append("n_features")
        return
    total = 0
    if spec.has_embedding:
        width = spec.embedding_pca_dim if spec.embedding_pca_dim is not None else spec.embedding_native_dim
        if width is None:
            spec.unknown_axes.append("n_features")
            return
        total += width
    if spec.has_readout:
        if spec.readout_dim is None:
            spec.unknown_axes.append("n_features")
            return
        total += spec.readout_dim
    if spec.has_descriptors:
        if spec.descriptor_pca_width is None:
            spec.unknown_axes.append("n_features")
            return
        total += spec.descriptor_pca_width
    spec.n_features = total
    if "n_features" in spec.unknown_axes:
        spec.unknown_axes.remove("n_features")


def fingerprint(spec: RunSpec) -> str:
    """Stable sha256 (truncated) of the critical-column values in fixed key order."""
    payload = {col: getattr(spec, col) for col in CRITICAL_COLUMNS}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _apply_resolved_axes(spec: RunSpec, resolved_axes: dict[str, Any], source_label: str) -> None:
    """Fill unresolved axes from `resolved_axes`; record any disagreement with an already-resolved value.

    `source_label` names the evidence tier doing the filling ("launch_script"
    for a verbatim launch-script line, "convention" for the NEXT_STEPS.md +
    naming-convention + sibling-analogy resolution of a row no current launch
    script emits verbatim). It is stamped into `spec_source` and any recorded
    disagreement so the trust tier of every filled axis stays legible.
    """
    had_in_run_source = spec.spec_source is not None
    filled_any = False
    for col in CRITICAL_COLUMNS:
        if col == "n_features":
            continue  # derived after this, from the other axes
        if col not in resolved_axes:
            continue  # this resolver doesn't cover this axis at all
        resolved_value = resolved_axes[col]
        current = getattr(spec, col)
        if col in spec.unknown_axes:
            setattr(spec, col, resolved_value)
            spec.unknown_axes.remove(col)
            filled_any = True
        elif current != resolved_value:
            spec.disagreements.append(f"{col}: in-run={current!r} vs {source_label}={resolved_value!r}")
    if filled_any:
        spec.spec_source = f"run_provenance+{source_label}" if had_in_run_source else source_label


def build_run_spec(run_dir: Path) -> RunSpec:
    """Build one RunSpec from a results/<...>/ directory's in-run artifacts and launch-script evidence."""
    base_dir, seed = split_base_and_seed(run_dir.name)
    spec = RunSpec(base_dir=base_dir, seed=seed, run_dir=str(run_dir))

    cfg = load_config_used(run_dir)
    if cfg is not None and "model" in cfg and "active_learning_loop" in cfg:
        resolve_moal_config_axes(spec, cfg)
    else:
        # No in-run artifact of a recognized schema pins any critical axis;
        # the launch-script route below is this row's only chance.
        resolve_axes_no_config(spec)

    launch_axes = resolve_launch_axes(run_dir.name)
    if launch_axes is not None:
        _apply_resolved_axes(spec, launch_axes, "launch_script")
    else:
        # No verbatim launch-script line emits this dir name. Fall back to the
        # convention layer (NEXT_STEPS.md + naming convention + script defaults
        # + seeded-sibling analogy) for the pre-sweep one-offs and the rewritten-
        # recipe runs (tabicl/pinned-tabpfn variants the current scripts dropped).
        convention_axes = resolve_convention_axes(run_dir.name)
        if convention_axes is not None:
            _apply_resolved_axes(spec, convention_axes, "convention")

    _cross_check_freeze_hd_clip_naming(spec)
    compute_n_features(spec)
    spec.metrics = load_eval_metrics(run_dir)
    return spec


def spec_to_row(spec: RunSpec) -> dict[str, Any]:
    """Flatten a RunSpec into one output row, including its fingerprint."""
    row: dict[str, Any] = {
        "base_dir": spec.base_dir,
        "seed": spec.seed,
        "run_dir": spec.run_dir,
    }
    for col in CRITICAL_COLUMNS:
        row[col] = getattr(spec, col)
    for col in MOAL_HYPERPARAM_COLUMNS:
        row[col] = getattr(spec, col)
    row["config_fingerprint"] = fingerprint(spec)
    row["spec_source"] = spec.spec_source
    row["unknown_axes"] = ",".join(dict.fromkeys(spec.unknown_axes))  # de-duplicate, preserve order
    row["launch_vs_inrun_disagreements"] = "; ".join(spec.disagreements)
    row.update(spec.metrics)
    return row


def build_table() -> pd.DataFrame:
    """Build the full one-row-per-run provenance dataframe."""
    run_dirs = discover_run_dirs()
    specs = [build_run_spec(d) for d in run_dirs]
    baseline_spec = build_anvil_baseline_spec()
    if baseline_spec is not None:
        specs.append(baseline_spec)
    rows = [spec_to_row(s) for s in specs]
    df = pd.DataFrame(rows)
    # seed is never null: split_base_and_seed always assigns either an
    # explicit 0..4 pin or the UNPINNED_SEED (-1) sentinel (directive 1,
    # third pass), so it gets a real int64 dtype, not a nullable one — a
    # silent coercion back to float/NaN here would defeat the sentinel.
    if df["seed"].isna().any():
        raise ValueError("seed column has null values; UNPINNED_SEED sentinel assignment is broken")
    df["seed"] = df["seed"].astype("int64")
    # Nullable integer dtypes for the width/dim columns so NA survives round-trips.
    for col in [
        "embedding_native_dim",
        "embedding_pca_dim",
        "readout_dim",
        "descriptor_pca_width",
        "n_features",
        "n",
        "ffn_hidden_dim",
        "freeze_epochs",
    ]:
        df[col] = df[col].astype("Int64")
    df["gradient_clip_val"] = df["gradient_clip_val"].astype("Float64")
    for col in [
        "has_embedding",
        "has_readout",
        "has_descriptors",
        "aux_use_observed_readout",
        "aux_use_predicted_readout",
    ]:
        df[col] = df[col].astype("boolean")
    return df


def write_table(df: pd.DataFrame) -> Path:
    """Write the dataframe to parquet if pyarrow is available, else CSV."""
    try:
        import pyarrow  # noqa: F401

        df.to_parquet(OUTPUT_PARQUET, index=False)
        return OUTPUT_PARQUET
    except ImportError:
        df.to_csv(OUTPUT_CSV, index=False)
        return OUTPUT_CSV


def main() -> None:
    df = build_table()
    out_path = write_table(df)
    print(f"Wrote {len(df)} rows to {out_path}")
    n_fully_resolved = (df["unknown_axes"] == "").sum()
    dir_names = df["run_dir"].apply(lambda p: Path(p).name)
    n_launch_matched = dir_names.isin(LAUNCH_MAP).sum()
    n_convention_matched = dir_names.isin(CONVENTION_MAP).sum()
    print(f"{n_fully_resolved}/{len(df)} rows fully resolved")
    print(f"{n_launch_matched}/{len(df)} rows matched a launch-script invocation")
    print(f"{n_convention_matched}/{len(df)} rows matched a convention-layer resolution")
    n_disagree = (df["launch_vs_inrun_disagreements"] != "").sum()
    print(f"{n_disagree} rows have an in-run-vs-launch-script disagreement")


if __name__ == "__main__":
    main()
