"""Frozen, off-the-shelf CheMeleon embedding + descriptors (+ borrowed readout), through TabPFN.

Side experiment against section 02 of the ablation artifact: repeats that
panel's three embedding-inclusive combos (embedding + readout + descriptors,
embedding + descriptors, embedding + readout), but swaps section 02's
from-scratch, log2FC-pretrained encoder embedding for CheMeleon's untouched
pretrained embedding, PCA-compressed exactly the way
`tabpfn_chemeleon_static_features.py` already does it for panel 01's
"CheMeleon embedding" row. This is the off-the-shelf embedding a community
user could drop in without training an encoder of their own.

CheMeleon was never fine-tuned on log2FC here, so it has no predicted-readout
head to borrow from. Where a combo calls for a readout block, this script
reads the 2-column predicted-readout out of the existing from-scratch
encoder's cache (`data/auxiliary_embedding_cache_seed{N}.parquet`, the same
cache section 02's own rows read), so the only feature that changes between
this experiment and section 02 is the embedding itself.

Descriptors are Mordred-only, PCA-128, matching section 02's own descriptor
treatment exactly (not this repo's other CheMeleon scripts, which default to
combined RDKit+Mordred at PCA-256); the CheMeleon embedding is PCA-compressed
to 256 components, the same width panel 01's own CheMeleon-embedding row
uses, and close to the from-scratch encoder's native (un-PCA'd) 256-dim
width, so total column counts stay in the same range as section 02's 386.

Run with:
    python tabpfn_chemeleon_concat_features.py configs/freeze1_hd512_clip5.0.yaml \
        --embedding-cache data/auxiliary_embedding_cache_seed0.parquet --seed 0 \
        --output-dir results/tabpfn_chemeleon_readout_mordred_pca128_seed0
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from evaluate_predictions import evaluate
from moal.config import PipelineConfig
from moal.model import ChemPropLightningModule
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import QueryType
from tabpfn_concat_features import _build_regressor, _load_descriptor_cache, _load_embedding_cache

logger = logging.getLogger(__name__)


def _pca_compress(
    train_raw: np.ndarray, test_raw: np.ndarray, n_components: int, seed: int, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on the training split and apply it to both splits.

    Parameters
    ----------
    train_raw : np.ndarray
        Training-split feature block, shape ``(n_train, n_features)``.
    test_raw : np.ndarray
        Blind-set feature block, shape ``(n_test, n_features)``.
    n_components : int
        Number of PCA components to keep.
    seed : int
        Random seed for PCA's randomized solver.
    label : str
        Human-readable name for the feature block, used only in the log line.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_compressed, test_compressed)``, each with ``n_components`` columns.
    """
    pca = PCA(n_components=n_components, random_state=seed).fit(train_raw)
    logger.info(
        "Compressed %s from %d to %d PCA component(s) (%.1f%% variance explained)",
        label,
        train_raw.shape[1],
        n_components,
        100 * pca.explained_variance_ratio_.sum(),
    )
    return pca.transform(train_raw), pca.transform(test_raw)


def _readout_features(embedding_cache: pd.DataFrame, canonical_smiles: list[str]) -> np.ndarray:
    """Look up the borrowed from-scratch encoder's 2-column predicted-readout block."""
    rows = embedding_cache.loc[canonical_smiles]
    return rows[[c for c in rows.columns if c.startswith("readout_")]].to_numpy(dtype=np.float32)


def main() -> None:
    """Extract CheMeleon embeddings, join borrowed readout/descriptors, fit TabPFN, score."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="moal plan config supplying data/model settings")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tabpfn_chemeleon_concat"))
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        required=True,
        help="From-scratch encoder cache to borrow the predicted-readout block from "
        "(e.g. data/auxiliary_embedding_cache_seed0.parquet); ignored if --no-readout",
    )
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument(
        "--embedding-pca-components",
        type=int,
        default=256,
        help="Components to compress the 2048-dim CheMeleon embedding to before TabPFN",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Components to compress the Mordred descriptor block to before TabPFN",
    )
    parser.add_argument(
        "--no-descriptors",
        action="store_true",
        help="Drop the descriptor block (keep the CheMeleon embedding + borrowed readout, if enabled)",
    )
    parser.add_argument(
        "--no-readout",
        action="store_true",
        help="Drop the borrowed predicted-readout block (keep the CheMeleon embedding + descriptors, if enabled)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed (PCA and regressor random state) for a seed-sweep replicate",
    )
    parser.add_argument(
        "--regressor",
        choices=["tabpfn", "tabpfn-v2.6", "tabpfn-v3", "lgbm", "xgboost", "tabicl", "tabfm"],
        default="tabpfn",
        help="Final regressor fit on the assembled feature matrix; see tabpfn_concat_features.py",
    )
    parser.add_argument(
        "--tabfm-n-estimators",
        type=int,
        default=32,
        help="Ensemble size for --regressor tabfm (ignored otherwise)",
    )
    parser.add_argument(
        "--tabicl-batch-size",
        type=int,
        default=None,
        help="batch_size for --regressor tabicl (ignored otherwise); default keeps the "
        "library default (8), which this featureset's width needs unlike the leaner "
        "no-embedding featureset tabpfn_concat_features.py defaults to batch_size=2 for",
    )
    parser.add_argument(
        "--tabfm-max-rows",
        type=int,
        default=500,
        help="max_num_rows for --regressor tabfm (ignored otherwise); 0 means uncapped "
        "(fits on the full training context), reproducing the blog's row-count OOM",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for both the CheMeleon embedding extraction and the tabular-foundation-"
        "model regressor (ignored for lgbm/xgboost); 'auto' is the library default (GPU if "
        "visible). Use 'cpu' to reproduce the blog's GPU-OOM claims on CPU",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    descriptor_cache = None if args.no_descriptors else _load_descriptor_cache("mordred")

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

    logger.info("Loading pretrained CheMeleon weights (no fine-tuning)")
    model = ChemPropLightningModule(
        ffn_hidden_dim=cfg.model.ffn_hidden_dim,
        ffn_num_layers=cfg.model.ffn_num_layers,
        message_hidden_dim=cfg.model.message_hidden_dim,
        depth=cfg.model.depth,
        freeze_epochs=cfg.model.freeze_epochs,
        mpnn_lr=cfg.model.mpnn_lr,
        ffn_lr=cfg.model.ffn_lr,
        mpnn_weight_decay=cfg.model.mpnn_weight_decay,
        ffn_weight_decay=cfg.model.ffn_weight_decay,
        sigma=cfg.model.sigma,
        w_drc=cfg.model.w_drc,
        w_ps=cfg.model.w_ps,
        learnable_sigma=cfg.model.learnable_sigma,
        from_foundation=cfg.model.from_foundation,
    )
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda:
        model = model.to("cuda")

    train_embed_raw = model.embed_smiles(train_smiles)
    test_embed_raw = model.embed_smiles(test_smiles)
    logger.info("Extracted %s raw CheMeleon embedding for training", train_embed_raw.shape)

    # Embedding extraction is the last use of the untouched model; free its
    # GPU memory before TabPFN's own pass claims the device
    del model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    train_embed, test_embed = _pca_compress(
        train_embed_raw,
        test_embed_raw,
        args.embedding_pca_components,
        cfg.seed,
        "CheMeleon embedding",
    )

    blocks_train = [train_embed]
    blocks_test = [test_embed]

    if not args.no_readout:
        embedding_cache = _load_embedding_cache(args.embedding_cache)
        blocks_train.append(_readout_features(embedding_cache, train_smiles))
        blocks_test.append(_readout_features(embedding_cache, test_smiles))

    if descriptor_cache is not None:
        train_descriptors_raw = descriptor_cache.loc[train_smiles].to_numpy(dtype=np.float64)
        test_descriptors_raw = descriptor_cache.loc[test_smiles].to_numpy(dtype=np.float64)
        imputer = SimpleImputer(strategy="median").fit(train_descriptors_raw)
        train_descriptors, test_descriptors = _pca_compress(
            imputer.transform(train_descriptors_raw),
            imputer.transform(test_descriptors_raw),
            args.pca_components,
            cfg.seed,
            "Mordred descriptors",
        )
        blocks_train.append(train_descriptors)
        blocks_test.append(test_descriptors)

    train_features = np.concatenate(blocks_train, axis=1)
    test_features = np.concatenate(blocks_test, axis=1)
    logger.info("Extracted %s concatenated feature matrix for training", train_features.shape)
    logger.info("Extracted %s concatenated feature matrix for blind targets", test_features.shape)

    regressor = _build_regressor(
        args.regressor,
        cfg.seed,
        train_features.shape[1],
        tabfm_n_estimators=args.tabfm_n_estimators,
        tabicl_batch_size=args.tabicl_batch_size,
        device=args.device,
        tabfm_max_rows=None if args.tabfm_max_rows == 0 else args.tabfm_max_rows,
    )
    regressor.fit(train_features, train_true)
    test_predictions = regressor.predict(test_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = args.output_dir / "pxr_tabpfn_predictions.csv"
    pd.DataFrame({"smiles": test_smiles, "predicted_pec50": test_predictions}).to_csv(
        predictions_csv, index=False
    )
    logger.info("Wrote predictions to %s", predictions_csv)

    results = evaluate(predictions_csv, activity_threshold=args.activity_threshold)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results.to_string(index=False))
    results.to_csv(args.output_dir / "eval_out.csv", index=False)


if __name__ == "__main__":
    main()
