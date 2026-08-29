"""Evaluate `moal plan` pEC50 predictions against the unblinded PXR test set.

The blinded TEST set (`pxr-challenge_TEST_BLINDED.csv`, used as the
`moal plan` inference targets) has since been unblinded and published as two
phases on the Hugging Face Hub:

    hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv
    hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv

This script joins a `moal plan` annotated output CSV (which carries a
`predicted_pec50` column) against those two phases by canonical SMILES and
reports MAE, RMSE, RAE, R-squared, Kendall's tau, and Spearman's rho overall,
per phase, and for the potent-compound subset (`pEC50 >= activity_threshold`)
called out in issue #36 as a known failure mode of this approach.

Run with:
    python evaluate_predictions.py results/pxr_aux_predictions/pxr_aux_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from moal.preprocessing import SMILESPreprocessor

logger = logging.getLogger(__name__)

_HF_DATASET = "hf://datasets/openadmet/pxr-challenge-train-test"
_CACHE_DIR = Path("data")
_PHASES = (1, 2)


def _cached_unblinded_phase(phase: int, *, refresh: bool) -> pd.DataFrame:
    """Load one unblinded test phase, caching it to `data/` after first fetch.

    Parameters
    ----------
    phase : int
        Phase number (1 or 2).
    refresh : bool
        When True, re-download from Hugging Face even if a local cache exists.

    Returns
    -------
    pd.DataFrame
        Raw phase data with a `phase` column.
    """
    cache_path = _CACHE_DIR / f"pxr-challenge_TEST_PHASE_{phase}_UNBLINDED.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    remote_path = f"{_HF_DATASET}/pxr-challenge_TEST_PHASE_{phase}_UNBLINDED.csv"
    logger.info("Fetching %s", remote_path)
    df = pd.read_csv(remote_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_unblinded_test(preprocessor: SMILESPreprocessor, *, refresh: bool = False) -> pd.DataFrame:
    """Load and combine both unblinded test phases with canonical SMILES.

    Parameters
    ----------
    preprocessor : SMILESPreprocessor
        Canonicalizer shared with the main moal pipeline, so join keys match
        `moal plan`'s own canonicalized SMILES.
    refresh : bool, optional
        When True, re-download from Hugging Face instead of using the local
        cache in `data/`. Default is False.

    Returns
    -------
    pd.DataFrame
        Columns: canonical, pEC50, phase.
    """
    phase_frames = [_cached_unblinded_phase(phase, refresh=refresh) for phase in _PHASES]
    combined = pd.concat(phase_frames, ignore_index=True)
    combined["canonical"] = combined["SMILES"].astype(str).map(preprocessor.canonicalize)
    n_rejected = int(combined["canonical"].isna().sum())
    if n_rejected:
        logger.warning("Rejected %d unblinded row(s) that failed SMILES canonicalization.", n_rejected)
    combined = combined[combined["canonical"].notna()]
    return combined[["canonical", "pEC50", "phase"]].rename(columns={"pEC50": "true_pec50"})


def load_predictions(predictions_csv: Path, preprocessor: SMILESPreprocessor) -> pd.DataFrame:
    """Load a `moal plan` annotated output CSV and canonicalize its SMILES.

    Parameters
    ----------
    predictions_csv : Path
        Path to a `moal plan` output CSV containing `smiles` and
        `predicted_pec50` columns.
    preprocessor : SMILESPreprocessor
        Canonicalizer shared with the main moal pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: canonical, predicted_pec50. Rows with no prediction (e.g.
        training-only rows) are dropped.

    Raises
    ------
    ValueError
        If `predictions_csv` has no `predicted_pec50` column.
    """
    df = pd.read_csv(predictions_csv)
    if "predicted_pec50" not in df.columns:
        raise ValueError(
            f"{predictions_csv} has no 'predicted_pec50' column; re-run moal plan "
            "with a moal build that writes it."
        )
    df = df[df["predicted_pec50"].notna()].copy()
    df["canonical"] = df["smiles"].astype(str).map(preprocessor.canonicalize)
    n_rejected = int(df["canonical"].isna().sum())
    if n_rejected:
        logger.warning("Rejected %d prediction row(s) that failed SMILES canonicalization.", n_rejected)
    df = df[df["canonical"].notna()]
    return df[["canonical", "predicted_pec50"]]


def compute_metrics(true_pec50: np.ndarray, predicted_pec50: np.ndarray) -> dict[str, float]:
    """Compute regression and rank-correlation metrics for one subset.

    Parameters
    ----------
    true_pec50 : np.ndarray
        Ground-truth pEC50 values.
    predicted_pec50 : np.ndarray
        Model-predicted pEC50 values, aligned with `true_pec50`.

    Returns
    -------
    dict[str, float]
        Keys: n, mae, rmse, rae, r2, kendall_tau, spearman_rho. Rank
        correlations are NaN when fewer than two samples are present; RAE
        and R-squared are NaN when `true_pec50` is constant (zero variance).
    """
    n = len(true_pec50)
    if n == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "rae": float("nan"),
            "r2": float("nan"),
            "kendall_tau": float("nan"),
            "spearman_rho": float("nan"),
        }

    residual = predicted_pec50 - true_pec50
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))

    true_deviation = true_pec50 - np.mean(true_pec50)
    denom = np.sum(np.abs(true_deviation))
    rae = float(np.sum(np.abs(residual)) / denom) if denom > 0 else float("nan")

    ss_tot = np.sum(true_deviation**2)
    r2 = float(1.0 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else float("nan")

    if n >= 2:
        kendall_tau = float(stats.kendalltau(predicted_pec50, true_pec50).statistic)
        spearman_rho = float(stats.spearmanr(predicted_pec50, true_pec50).statistic)
    else:
        kendall_tau = float("nan")
        spearman_rho = float("nan")

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "rae": rae,
        "r2": r2,
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
    }


def evaluate(
    predictions_csv: Path, *, activity_threshold: float, refresh: bool = False
) -> pd.DataFrame:
    """Join predictions against the unblinded test set and score every subset.

    Parameters
    ----------
    predictions_csv : Path
        Path to a `moal plan` output CSV.
    activity_threshold : float
        pEC50 threshold defining the potent-compound subset (see issue #36's
        documented potent-compound tail-compression limitation).
    refresh : bool, optional
        Forwarded to :func:`load_unblinded_test`. Default is False.

    Returns
    -------
    pd.DataFrame
        One row per subset (overall, phase 1, phase 2, potent, non-potent)
        with the metrics from :func:`compute_metrics`.

    Raises
    ------
    ValueError
        If no predicted compound matches the unblinded test set by canonical
        SMILES.
    """
    preprocessor = SMILESPreprocessor()
    truth = load_unblinded_test(preprocessor, refresh=refresh)
    predictions = load_predictions(predictions_csv, preprocessor)

    merged = predictions.merge(truth, on="canonical", how="inner")
    if merged.empty:
        raise ValueError(
            f"No predicted compound in {predictions_csv} matched the unblinded test set."
        )
    n_unmatched = len(predictions) - len(merged)
    if n_unmatched:
        logger.warning(
            "%d predicted compound(s) had no match in the unblinded test set.", n_unmatched
        )

    subsets = {
        "overall": merged,
        "phase_1": merged[merged["phase"] == 1],
        "phase_2": merged[merged["phase"] == 2],
        f"potent (>= {activity_threshold})": merged[merged["true_pec50"] >= activity_threshold],
        f"non_potent (< {activity_threshold})": merged[merged["true_pec50"] < activity_threshold],
    }

    rows = []
    for name, subset in subsets.items():
        metrics = compute_metrics(
            subset["true_pec50"].to_numpy(dtype=np.float64),
            subset["predicted_pec50"].to_numpy(dtype=np.float64),
        )
        rows.append({"subset": name, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    """Parse CLI arguments, run the evaluation, and print the results table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path, help="moal plan annotated output CSV")
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=7.0,
        help="pEC50 threshold for the potent-compound subset (default: 7.0, matching "
        "moal_aux_plan_config.yaml's oracle.activity_threshold)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download the unblinded test set instead of using the local cache in data/",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to save the metrics table")
    args = parser.parse_args()

    results = evaluate(
        args.predictions_csv,
        activity_threshold=args.activity_threshold,
        refresh=args.refresh_cache,
    )
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results.to_string(index=False))

    if args.output_csv is not None:
        results.to_csv(args.output_csv, index=False)
        logger.info("Wrote metrics to %s", args.output_csv)


if __name__ == "__main__":
    main()
