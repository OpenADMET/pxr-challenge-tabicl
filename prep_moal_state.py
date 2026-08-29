"""Build the unified moal plan campaign-state CSV for the PXR challenge data.

Combines three raw PXR challenge files into the single CSV `moal plan`
expects (columns: smiles, relation, value, weight, log2fc_*):

- `pxr-challenge_TRAIN.csv`: dose-response curves -> exact ("==") DRC records.
- `pxr-challenge_single_concentration_TRAIN.csv`: single-concentration hit
  calls at four concentrations -> primary-screen ("<" / ">=") records, plus
  the raw log2FC readouts used as auxiliary-encoder input (issue #36).
- `pxr-challenge_TEST_BLINDED.csv`: unqueried inference targets.

moal's primary-screen schema is single-threshold: every PS row's `value`
must equal a fixed `oracle.ps_threshold` (relation encodes hit vs. miss, not
magnitude). A compound is called a hit if it hit at any tested
concentration; by monotonicity, a hit at a more stringent (lower)
concentration implies a hit at the threshold too. The per-concentration
resolution this discards is preserved separately in the log2fc_* columns
for the auxiliary encoder.

Run with `python prep_moal_state.py` from the pxr-challenge repo root.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from moal.preprocessing import SMILESPreprocessor

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "moal_plan_state.csv"

FDR_CUTOFF = 1.301  # -log10(0.05): significance threshold for PS hit calling
CONCENTRATIONS_M = (9.803e-07, 8.251e-06, 3.300e-05, 9.901e-05)
LOG2FC_COLUMNS = [f"log2fc_{conc:.3e}" for conc in CONCENTRATIONS_M]

# moal's PS schema needs one fixed threshold value shared by every PS row.
PS_THRESHOLD_PCONC = 4.0

# PS hit calls are noisier than DRC-confirmed misses: INTERVAL (hit) weight
# reflects (1 - false-positive rate) / (1 - false-negative rate) against DRC
# ground truth, scaled by hit-call confidence (neg_log10_fdr); LEFT (miss)
# records keep a fixed weight of 1.0.
_PS_RELIABILITY_RATIO = 0.485 / 0.978
_NEG_LOG10_FDR_CAP = 15.0  # cap near-zero FDR values (neg_log10 -> inf)


def _canonicalize_column(df: pd.DataFrame, smiles_column: str, preprocessor: SMILESPreprocessor) -> pd.DataFrame:
    """Add a `canonical` column and drop rows that fail canonicalization.

    Parameters
    ----------
    df : pd.DataFrame
        Frame containing a raw SMILES column.
    smiles_column : str
        Name of the raw SMILES column.
    preprocessor : SMILESPreprocessor
        Canonicalizer shared with the main moal pipeline.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with a `canonical` column, rejected rows removed.
    """
    result = df.copy()
    result["canonical"] = result[smiles_column].astype(str).map(preprocessor.canonicalize)
    n_rejected = int(result["canonical"].isna().sum())
    if n_rejected:
        logger.warning("Rejected %d row(s) that failed SMILES canonicalization.", n_rejected)
    return result[result["canonical"].notna()]


def build_drc_rows(train_csv: Path, preprocessor: SMILESPreprocessor) -> pd.DataFrame:
    """Build exact ("==") DRC records from the dose-response training data.

    Parameters
    ----------
    train_csv : Path
        Path to `pxr-challenge_TRAIN.csv`.
    preprocessor : SMILESPreprocessor
        Canonicalizer shared with the main moal pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: smiles, relation, value, weight, canonical.
    """
    df = _canonicalize_column(pd.read_csv(train_csv), "SMILES", preprocessor)
    return pd.DataFrame(
        {
            "smiles": df["canonical"],
            "relation": "==",
            "value": df["pEC50"].astype(float),
            "weight": 1.0,
            "canonical": df["canonical"],
        }
    )


def _pivot_log2fc(sc_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-concentration log2FC readings to one column per concentration.

    Parameters
    ----------
    sc_df : pd.DataFrame
        Single-concentration measurements with `canonical`, `concentration_M`,
        and `log2_fc_estimate` columns.

    Returns
    -------
    pd.DataFrame
        Indexed by `canonical`, one column per `LOG2FC_COLUMNS` entry;
        replicate measurements at the same concentration are median-pooled.
    """
    pivot = sc_df.groupby(["canonical", "concentration_M"])["log2_fc_estimate"].median().unstack()
    pivot = pivot.reindex(columns=list(CONCENTRATIONS_M))
    pivot.columns = LOG2FC_COLUMNS
    return pivot


def _defining_hit_weight(hits: pd.DataFrame, mean_defining_nlf: float) -> float:
    """Weight for an INTERVAL (hit) PS record, from its most stringent hit.

    Parameters
    ----------
    hits : pd.DataFrame
        Rows called as hits for one compound; must be non-empty.
    mean_defining_nlf : float
        Population mean of the defining (most stringent hit's) capped
        neg_log10_fdr, used to normalize confidence around 1.0.

    Returns
    -------
    float
        Raw (pre-normalization) PS record weight.
    """
    defining_row = hits.loc[hits["pconc"].idxmax()]
    defining_nlf = float(defining_row["nlf_clamped"])
    return _PS_RELIABILITY_RATIO * (defining_nlf / mean_defining_nlf)


def load_single_concentration_data(
    single_concentration_csv: Path, preprocessor: SMILESPreprocessor
) -> pd.DataFrame:
    """Load and annotate the single-concentration hit-calling data.

    Parameters
    ----------
    single_concentration_csv : Path
        Path to `pxr-challenge_single_concentration_TRAIN.csv`.
    preprocessor : SMILESPreprocessor
        Canonicalizer shared with the main moal pipeline.

    Returns
    -------
    pd.DataFrame
        Library-class rows with `canonical`, `is_hit`, `pconc`, and
        `nlf_clamped` columns added; covers every screened compound
        regardless of DRC coverage.
    """
    sc_df = _canonicalize_column(pd.read_csv(single_concentration_csv), "SMILES", preprocessor)
    sc_df = sc_df[sc_df["compound_class"] == "Library"]
    return sc_df.assign(
        is_hit=(sc_df["neg_log10_fdr"] > FDR_CUTOFF) & (sc_df["log2_fc_estimate"] > 0.0),
        pconc=-sc_df["concentration_M"].map(math.log10),
        nlf_clamped=sc_df["neg_log10_fdr"].clip(upper=_NEG_LOG10_FDR_CAP),
    )


def build_ps_rows(sc_df: pd.DataFrame, *, exclude_canonical: set[str]) -> pd.DataFrame:
    """Build primary-screen records and log2FC auxiliary columns.

    Parameters
    ----------
    sc_df : pd.DataFrame
        Output of :func:`load_single_concentration_data`.
    exclude_canonical : set[str]
        Canonical SMILES already covered by a DRC record; skipped here since
        moal drops the PS record for DRC-upgraded compounds at refit time.

    Returns
    -------
    pd.DataFrame
        Columns: smiles, relation, value, weight, plus `LOG2FC_COLUMNS`.
    """
    log2fc_by_compound = _pivot_log2fc(sc_df)

    defining_nlf_per_compound = (
        sc_df[sc_df["is_hit"]]
        .sort_values("pconc")
        .groupby("canonical")["nlf_clamped"]
        .last()
    )
    mean_defining_nlf = float(defining_nlf_per_compound.mean())
    logger.info("Mean defining neg_log10_fdr across PS hits: %.3f", mean_defining_nlf)

    rows: list[dict[str, object]] = []
    n_skipped_drc = 0
    for canonical, group in sc_df.groupby("canonical"):
        if canonical in exclude_canonical:
            n_skipped_drc += 1
            continue
        hits = group[group["is_hit"]]
        if hits.empty:
            relation, weight = "<", 1.0
        else:
            relation = ">="
            weight = _defining_hit_weight(hits, mean_defining_nlf)

        row: dict[str, object] = {
            "smiles": canonical,
            "relation": relation,
            "value": PS_THRESHOLD_PCONC,
            "weight": weight,
        }
        if canonical in log2fc_by_compound.index:
            readouts = log2fc_by_compound.loc[canonical]
            row.update({col: readouts[col] for col in LOG2FC_COLUMNS if pd.notna(readouts[col])})
        rows.append(row)

    logger.info(
        "PS rows: %d (skipped %d already covered by a DRC record)", len(rows), n_skipped_drc
    )
    ps_df = pd.DataFrame(rows).reindex(
        columns=["smiles", "relation", "value", "weight", *LOG2FC_COLUMNS]
    )
    logger.info(
        "  LEFT (miss): %d  INTERVAL (hit): %d",
        int((ps_df["relation"] == "<").sum()),
        int((ps_df["relation"] == ">=").sum()),
    )
    return ps_df


def attach_log2fc_to_drc(drc_df: pd.DataFrame, log2fc_by_compound: pd.DataFrame) -> pd.DataFrame:
    """Copy log2FC readouts onto DRC rows for compounds also screened via PS.

    A compound can be DRC-confirmed and still carry single-concentration
    log2FC measurements (e.g. later upgraded from a PS hit); those readouts
    are useful auxiliary-encoder input even though the compound has no
    separate PS record in the output CSV.

    Parameters
    ----------
    drc_df : pd.DataFrame
        Output of :func:`build_drc_rows`.
    log2fc_by_compound : pd.DataFrame
        Output of :func:`_pivot_log2fc` over the full single-concentration
        data (covers screened compounds regardless of DRC coverage).

    Returns
    -------
    pd.DataFrame
        `drc_df` with `LOG2FC_COLUMNS` populated where available.
    """
    result = drc_df.copy()
    for col in LOG2FC_COLUMNS:
        result[col] = pd.NA
    overlap = result["canonical"].isin(log2fc_by_compound.index)
    result.loc[overlap, LOG2FC_COLUMNS] = log2fc_by_compound.reindex(
        result.loc[overlap, "canonical"]
    ).to_numpy()
    logger.info("DRC rows with >=1 log2fc readout: %d", int(overlap.sum()))
    return result


def build_unqueried_rows(
    test_csv: Path, preprocessor: SMILESPreprocessor, *, labeled_canonical: set[str]
) -> pd.DataFrame:
    """Build blank-relation inference rows from the blinded test set.

    Parameters
    ----------
    test_csv : Path
        Path to `pxr-challenge_TEST_BLINDED.csv`.
    preprocessor : SMILESPreprocessor
        Canonicalizer shared with the main moal pipeline.
    labeled_canonical : set[str]
        Canonical SMILES already present as training records; used only to
        report cross-partition overlap (`parse_campaign_state` rejects it).

    Returns
    -------
    pd.DataFrame
        Columns: smiles, relation, value, weight (relation/value/weight
        blank).
    """
    df = _canonicalize_column(pd.read_csv(test_csv), "SMILES", preprocessor)
    overlap = int(df["canonical"].isin(labeled_canonical).sum())
    if overlap:
        logger.warning(
            "%d TEST_BLINDED compound(s) overlap with labeled training data; "
            "moal plan will reject the combined CSV.",
            overlap,
        )
    return pd.DataFrame(
        {"smiles": df["canonical"], "relation": "", "value": "", "weight": ""}
    )


def main() -> None:
    """Build the unified campaign-state CSV and write it to `OUTPUT_PATH`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    preprocessor = SMILESPreprocessor()

    drc_df = build_drc_rows(DATA_DIR / "pxr-challenge_TRAIN.csv", preprocessor)
    logger.info("DRC rows: %d", len(drc_df))

    train_canonical = set(drc_df["canonical"])
    sc_df = load_single_concentration_data(
        DATA_DIR / "pxr-challenge_single_concentration_TRAIN.csv", preprocessor
    )
    ps_df = build_ps_rows(sc_df, exclude_canonical=train_canonical)
    drc_df = attach_log2fc_to_drc(drc_df, _pivot_log2fc(sc_df))

    labeled_canonical = train_canonical | set(ps_df["smiles"])
    test_df = build_unqueried_rows(
        DATA_DIR / "pxr-challenge_TEST_BLINDED.csv",
        preprocessor,
        labeled_canonical=labeled_canonical,
    )
    logger.info("TEST (unqueried) rows: %d", len(test_df))

    columns = ["smiles", "relation", "value", "weight", *LOG2FC_COLUMNS]
    combined = pd.concat(
        [drc_df.drop(columns="canonical"), ps_df, test_df], ignore_index=True
    )[columns]
    combined.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(combined), OUTPUT_PATH)


if __name__ == "__main__":
    main()
