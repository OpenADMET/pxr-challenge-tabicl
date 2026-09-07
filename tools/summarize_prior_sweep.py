"""Summarize the prior (pre-rebuild) sweep into per-config evidence for the manifest.

The prior sweep lives on the `main` branch as `reporting/results.parquet`, an
aggregation of 423 run rows produced by `reporting/build_run_provenance.py`.
It is read here straight out of git at a pinned commit so the extraction is
reproducible from any worktree, and collapsed to one row per configuration.

Three properties of that parquet drive what this script does.

`config_fingerprint` under-discriminates. The 18-cell concatenation-architecture
grid collapses into two fingerprints because the fingerprint ignores
`freeze_epochs`, `ffn_hidden_dim` and `gradient_clip_val`, and the three TabFM
ensemble sizes share one fingerprint because the regressor is recorded as plain
`tabfm`. The run directory name (`base_dir`) is the true configuration unit, so
that is what this script groups on, asserting the recorded axes agree within a
group.

Rows with `seed == -1` are runs whose seed the provenance extractor could not
recover. Where a configuration also has real seeded runs, its unseeded row is a
superseded earlier run that sometimes carries different axes (`tabpfn_small_embed`
seed -1 is a different architecture entirely from the tabular runs sharing that
name), so it is dropped from the group. Where a configuration has only an
unseeded row, that row is its single real result and is kept, flagged as such.

Two axes the parquet does not record are reconstructed here. `encoder_target`
is blank for the 113 concatenation-architecture rows and mixes semantics
elsewhere, so `encoder_finetune_target` and `aux_encoder_target` are filled from
the launch configs on `main`; `calibration` is read off the run-directory
suffix. Every reconstruction names the file it came from in `axes_source`.

The metrics are the prior sweep's own evaluation: phase 1 and phase 2 pooled,
n = 513. They rank configurations against each other and nothing else. They are
not comparable to this rebuild's phase-2 scores and never set a threshold.

Run with:
    python tools/summarize_prior_sweep.py
"""

from __future__ import annotations

import argparse
import io
import logging
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# the prior sweep aggregation, pinned to the commit that produced it
PRIOR_REV = "00be37e"
PRIOR_PATH = "reporting/results.parquet"

DEFAULT_OUT = Path("experiments/prior_summary.csv")

# the prior evaluation pooled both phases; recorded so no reader mistakes these
# numbers for phase-2 scores
PRIOR_EVAL = "phase1+phase2 pooled"

# axes copied through from the parquet, asserted constant within a config
RECORDED_AXES = (
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
    "ffn_hidden_dim",
    "freeze_epochs",
    "gradient_clip_val",
    "aux_use_observed_readout",
    "aux_use_predicted_readout",
)

METRICS = ("mae", "rmse", "rae", "r2", "kendall_tau", "spearman_rho")

# run-directory prefixes whose runs are the concatenation architecture: a main
# A Chemprop model fine-tuned on pEC50 (`value_column: value`) alongside an
# auxiliary encoder trained on log2FC, per configs/freeze*.yaml and
# configs/feat_*.yaml on main
_CONCAT_PREFIXES = ("freeze", "feat_")

# number of log2FC concentration tasks the auxiliary encoder was trained on;
# the grid and every feat_* variant but one use the trimmed 2-task set
# (generate_feature_sweep_configs.py)
_FOUR_TASK_DIRS = frozenset({"feat_observed_4task"})


def summarize(prior: pd.DataFrame) -> pd.DataFrame:
    """Collapse the prior sweep's run rows to one evidence row per configuration.

    Parameters
    ----------
    prior : DataFrame
        The prior sweep aggregation, one row per run.

    Returns
    -------
    DataFrame
        One row per run-directory configuration, sorted by mean RAE with
        unscored configurations last, carrying the recorded axes, the
        reconstructed axes, and the mean and spread of each metric.

    Raises
    ------
    ValueError
        If a configuration's retained rows disagree on a recorded axis.
    """
    retained = _drop_superseded_singletons(prior)

    rows = [_summarize_config(str(name), group) for name, group in retained.groupby("base_dir")]
    summary = pd.DataFrame(rows)

    # unscored configurations sort last so the file reads best-first
    return summary.sort_values(["rae_mean", "config"], na_position="last", ignore_index=True)


def load_prior(rev: str = PRIOR_REV, path: str = PRIOR_PATH) -> pd.DataFrame:
    """Read the prior sweep parquet out of git at a pinned revision.

    Parameters
    ----------
    rev : str, optional
        Commit-ish holding the file. Defaults to the commit that produced it.
    path : str, optional
        Repository-relative path to the parquet.

    Returns
    -------
    DataFrame
        The parquet's contents, one row per prior run.

    Raises
    ------
    RuntimeError
        If the object is not reachable from this repository.
    """
    try:
        blob = subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as err:
        message = err.stderr.decode().strip()
        raise RuntimeError(f"cannot read {rev}:{path} from this repository: {message}") from err
    return pd.read_parquet(io.BytesIO(blob))


def _drop_superseded_singletons(prior: pd.DataFrame) -> pd.DataFrame:
    """Drop seed -1 rows from configurations that also have real seeded runs."""
    has_seeded = prior.groupby("base_dir")["seed"].transform(lambda s: (s >= 0).any())
    return prior.loc[~(has_seeded & (prior["seed"] < 0))].copy()


def _summarize_config(name: str, group: pd.DataFrame) -> dict[str, object]:
    """Build one evidence row from a configuration's retained runs."""
    axes = _constant_axes(name, group)
    scored = group.loc[group["rae"].notna()]

    row: dict[str, object] = {
        "config": name,
        "model_kind": "gnn" if axes["regressor"] == "N/A" else "tabular",
        "n_seeds": len(scored),
        "seeds": ",".join(str(s) for s in sorted(scored["seed"].tolist())),
        "unseeded_only": bool((group["seed"] < 0).all()),
        "n_eval": _single_value(scored["n"]),
        "prior_eval": PRIOR_EVAL,
    }
    row.update(axes)
    row.update(_reconstructed_axes(name, axes))

    # mean and spread across whatever seeds actually scored; both are NaN for a
    # configuration that produced no metrics at all
    for metric in METRICS:
        values = scored[metric]
        row[f"{metric}_mean"] = values.mean() if len(values) else pd.NA
        row[f"{metric}_sd"] = values.std() if len(values) > 1 else pd.NA

    row["config_fingerprint"] = _single_value(group["config_fingerprint"])
    row["spec_source"] = ",".join(sorted(set(group["spec_source"])))
    return row


def _constant_axes(name: str, group: pd.DataFrame) -> dict[str, object]:
    """Return the recorded axes of a configuration, requiring them constant."""
    axes: dict[str, object] = {}
    for axis in RECORDED_AXES:
        values = {v for v in group[axis].tolist() if not pd.isna(v)}
        if len(values) > 1:
            raise ValueError(f"{name}: runs disagree on {axis}: {sorted(map(str, values))}")
        axes[axis] = values.pop() if values else pd.NA
    return axes


def _reconstructed_axes(name: str, axes: dict[str, object]) -> dict[str, object]:
    """Fill the axes the prior parquet leaves blank or records ambiguously.

    `encoder_target` conflates what an encoder was pretrained on with what it
    was fine-tuned on, and is blank for every concatenation-architecture run.
    The launch configs settle both: every GNN run fine-tunes on pEC50, and the
    concatenation runs pair that with a log2FC auxiliary encoder.
    """
    is_concat = name.startswith(_CONCAT_PREFIXES)
    is_gnn = axes["regressor"] == "N/A"

    if is_concat:
        source = "configs/freeze*.yaml, configs/feat_*.yaml @ main"
        return {
            "encoder_finetune_target": "pec50",
            "aux_encoder_target": "log2fc",
            "aux_readout_tasks": 4 if name in _FOUR_TASK_DIRS else 2,
            "calibration": _calibration(name),
            "axes_source": source,
        }

    if is_gnn:
        # e4_*: a body pretrained on log2FC, fine-tuned on pEC50, no auxiliary
        # encoder at all, so the auxiliary readout flags the extractor defaulted
        # to False do not apply
        return {
            "encoder_finetune_target": "pec50",
            "aux_encoder_target": "none",
            "aux_readout_tasks": pd.NA,
            "calibration": _calibration(name),
            "axes_source": "configs/e4_*.yaml @ main",
        }

    # tabular runs: the recorded encoder_target is already the fine-tune target
    # of the encoder that produced the embedding
    return {
        "encoder_finetune_target": axes["encoder_target"],
        "aux_encoder_target": "none",
        "aux_readout_tasks": pd.NA,
        "calibration": _calibration(name),
        "axes_source": "recorded",
    }


def _calibration(name: str) -> str:
    """Return the calibration a run directory's name declares."""
    return "isotonic_oof" if name.endswith("_calibrated") else "none"


def _single_value(values: pd.Series) -> object:
    """Return the one distinct non-null value in a series, or NA if there is none."""
    distinct = {v for v in values.tolist() if not pd.isna(v)}
    return distinct.pop() if len(distinct) == 1 else pd.NA


def main() -> None:
    """Write the per-configuration prior evidence CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default=PRIOR_REV, help="commit holding the prior parquet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV to write")
    args = parser.parse_args()

    prior = load_prior(rev=args.rev)
    summary = summarize(prior)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    scored = int(summary["n_seeds"].gt(0).sum())
    logger.info(
        "%d configs (%d scored, %d unscored) from %d prior runs -> %s",
        len(summary),
        scored,
        len(summary) - scored,
        len(prior),
        args.out,
    )


if __name__ == "__main__":
    main()
