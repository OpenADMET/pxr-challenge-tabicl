"""Save TabPFN's full predicted distribution per blind compound and check its calibration.

TabPFN represents its regression target as a discretized (Riemann) distribution
over bins, not a point estimate, and its own true distribution is available via
`TabPFNRegressor.predict(..., output_type="full")` (mean, quantiles, and the
underlying `FullSupportBarDistribution` criterion). This reads the current
best config's features (`tabpfn_embed_readout_mordred_pca128`: canonical
embedding + predicted log2FC readout + Mordred descriptors, PCA-128, all from
`data/auxiliary_embedding_cache.parquet`), fits TabPFN on all DRC training
data, predicts on the 513-compound blind set with the full distribution, and
checks whether that distribution tracks the actual residual: does |residual|
correlate with the predicted standard deviation, and does empirical quantile
coverage match the nominal quantile level (over- or under-confident)?

Run with:
    python tabpfn_uncertainty_analysis.py configs/tabpfn_small_embed.yaml \
        --descriptor-source mordred --pca-components 128 \
        --output-dir results/tabpfn_uncertainty_mordred_pca128
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNRegressor

from evaluate_predictions import load_unblinded_test
from moal.config import PipelineConfig
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import QueryType
from reporting.figures import OUT_DIR, render_reliability, render_scatter
from tabpfn_concat_features import (
    _embedding_features,
    _load_descriptor_cache,
    _load_embedding_cache,
    _raw_descriptors,
)

logger = logging.getLogger(__name__)

_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    """Fit the best config on cached features, predict full distributions, and check calibration."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="moal plan config supplying data/auxiliary_model settings",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tabpfn_uncertainty")
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Embedding cache to read (default: the canonical seed-42 cache); pass a "
        "seed-specific cache from precompute_auxiliary_embeddings.py --seed to hold "
        "the encoder fixed within a 5-seed replicate comparison",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Must match the config being analyzed",
    )
    parser.add_argument(
        "--descriptor-source", choices=["all", "rdkit", "mordred"], default="mordred"
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if cfg.auxiliary_model is None:
        raise ValueError(f"{args.config} has no auxiliary_model block")

    descriptor_cache = _load_descriptor_cache(args.descriptor_source)

    preprocessor = SMILESPreprocessor()
    state_df = pd.read_csv(cfg.data.plan.input_csv)
    state = parse_campaign_state(
        state_df,
        cost_ps=cfg.oracle.cost_ps,
        cost_drc=cfg.oracle.cost_drc,
        upper_bound=cfg.oracle.upper_bound,
        preprocessor=preprocessor,
        smiles_column=cfg.data.plan.smiles_column,
        relation_column=cfg.data.plan.relation_column,
        value_column=cfg.data.plan.value_column,
        weight_column=cfg.data.plan.weight_column,
        log2fc_columns=cfg.data.plan.log2fc_columns,
        is_canonical=cfg.data.plan.is_canonical,
        expected_ps_threshold=cfg.oracle.ps_threshold,
    )
    fit_records = training_records_for_refit(state.training_records)
    drc_records = [r for r in fit_records if r.fidelity == QueryType.DOSE_RESPONSE]

    train_smiles = [r.canonical_smiles for r in drc_records]
    train_true = np.array([r.value for r in drc_records], dtype=np.float64)
    test_smiles = [smi for _, smi in state.unqueried_rows]

    embedding_cache = _load_embedding_cache(args.embedding_cache)
    train_embed = _embedding_features(embedding_cache, train_smiles, True, True)
    test_embed = _embedding_features(embedding_cache, test_smiles, True, True)

    train_descriptors_raw = _raw_descriptors(train_smiles, descriptor_cache)
    test_descriptors_raw = _raw_descriptors(test_smiles, descriptor_cache)
    imputer = SimpleImputer(strategy="median").fit(train_descriptors_raw)
    pca = PCA(n_components=args.pca_components, random_state=cfg.seed).fit(
        imputer.transform(train_descriptors_raw)
    )
    train_descriptors = pca.transform(imputer.transform(train_descriptors_raw))
    test_descriptors = pca.transform(imputer.transform(test_descriptors_raw))

    train_features = np.concatenate([train_embed, train_descriptors], axis=1)
    test_features = np.concatenate([test_embed, test_descriptors], axis=1)
    logger.info(
        "Extracted %s / %s train / blind feature matrices",
        train_features.shape,
        test_features.shape,
    )

    regressor = TabPFNRegressor(
        random_state=cfg.seed,
        memory_saving_mode=True,
        fit_mode="low_memory",
        ignore_pretraining_limits=train_features.shape[1] > 2000,
    )
    regressor.fit(train_features, train_true)

    full_output = regressor.predict(
        test_features, output_type="full", quantiles=_QUANTILE_LEVELS
    )
    predicted_mean = full_output["mean"]
    predicted_std = (
        full_output["criterion"].variance(full_output["logits"]).sqrt().cpu().numpy()
    )
    quantile_preds = np.stack(full_output["quantiles"], axis=1)  # (n_test, n_quantiles)

    results_df = pd.DataFrame(
        {
            "smiles": test_smiles,
            "predicted_pec50": predicted_mean,
            "predicted_std": predicted_std,
        }
    )
    for level, col in zip(_QUANTILE_LEVELS, quantile_preds.T, strict=True):
        results_df[f"q{level:.1f}"] = col

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "pxr_tabpfn_distributions.csv"
    results_df.to_csv(raw_csv, index=False)
    logger.info("Wrote per-compound distributions to %s", raw_csv)

    unblinded = load_unblinded_test(preprocessor)
    results_df["canonical"] = (
        results_df["smiles"].astype(str).map(preprocessor.canonicalize)
    )
    merged = results_df.merge(unblinded, on="canonical", how="inner")
    logger.info(
        "Matched %d/%d blind predictions to unblinded ground truth",
        len(merged),
        len(results_df),
    )

    merged["residual"] = merged["predicted_pec50"] - merged["true_pec50"]
    merged["abs_residual"] = merged["residual"].abs()
    merged.to_csv(args.output_dir / "pxr_tabpfn_distributions_scored.csv", index=False)

    spearman = stats.spearmanr(merged["abs_residual"], merged["predicted_std"])
    logger.info(
        "Spearman rho(|residual|, predicted_std) = %.4f (p = %.4g, n = %d)",
        spearman.statistic,
        spearman.pvalue,
        len(merged),
    )

    print(
        "\nQuantile coverage (nominal level vs. empirical fraction of true values below the predicted quantile):"
    )
    coverage_rows = []
    for level in _QUANTILE_LEVELS:
        col = f"q{level:.1f}"
        empirical = float((merged["true_pec50"] <= merged[col]).mean())
        coverage_rows.append(
            {"nominal": level, "empirical": empirical, "gap": empirical - level}
        )
    coverage_df = pd.DataFrame(coverage_rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(coverage_df.to_string(index=False))
    coverage_df.to_csv(args.output_dir / "quantile_coverage.csv", index=False)

    n = len(merged)
    scatter_html = render_scatter(
        merged["predicted_std"].tolist(),
        merged["abs_residual"].tolist(),
        title=f"|residual| vs. predicted std, {n} blind compounds",
        aria_label=(
            "Scatter plot of predicted standard deviation versus absolute "
            f"residual for {n} blind compounds"
        ),
        x_label="predicted std",
        y_label="|residual|",
    )
    (OUT_DIR / "figure-07.html").write_text(scatter_html)
    logger.info("Wrote %s", OUT_DIR / "figure-07.html")

    reliability_html = render_reliability(
        coverage_df["nominal"].tolist(),
        coverage_df["empirical"].tolist(),
        title=f"Miscalibration area: empirical coverage vs. nominal level, all {len(coverage_df)} quantiles",
        aria_label=(
            "Reliability diagram plotting empirical coverage against nominal "
            "quantile level, with the gap to perfect calibration shaded"
        ),
    )
    (OUT_DIR / "figure-08.html").write_text(reliability_html)
    logger.info("Wrote %s", OUT_DIR / "figure-08.html")


if __name__ == "__main__":
    main()
