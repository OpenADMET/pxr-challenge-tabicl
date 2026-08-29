"""pEC50-fine-tuned CheMeleon embedding + RDKit/Mordred descriptors fed through TabPFN.

Every prior TabPFN run in this repo (`tabpfn_frozen_features.py`,
`tabpfn_concat_features.py`) draws its embedding from the `auxiliary_model`
block: an encoder trained on log2FC readouts, either from scratch or
CheMeleon-initialized. None of them use a CheMeleon encoder fine-tuned
directly against the pEC50 target itself, which is what the `model` block in
the `freeze*` sweep configs trains end-to-end (with its own FFN head, never
exposed as a feature vector).

This script isolates that missing cell: fine-tune a plain
`ChemPropLightningModule` (`cfg.model`, `from_foundation: chemeleon`) on DRC
records the same way the `freeze*` sweep does, then pull its pooled
structural embedding via the new `embed_smiles` method (mirrors
`AuxiliaryEncoderModule.embed_smiles`) instead of its scalar pEC50 output.
CheMeleon's native embedding width is fixed at 2048 regardless of config, so
unlike the log2FC path's 256-dim from-scratch encoder, this embedding needs
its own PCA compression before concatenation with the (also PCA-compressed)
descriptor block, to stay under TabPFN's feature-count OOM ceiling
(~2000 total columns on this GPU).

Run with:
    python tabpfn_pec50_chemeleon_features.py configs/freeze1_hd512_clip5.0.yaml \
        --output-dir results/tabpfn_pec50_chemeleon
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNRegressor

from evaluate_predictions import evaluate
from moal.config import PipelineConfig
from moal.model import ChemPropLightningModule
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import QueryType

logger = logging.getLogger(__name__)

_DESCRIPTOR_CACHE = Path("data/descriptor_cache.parquet")


def _load_descriptor_cache(source: str) -> pd.DataFrame:
    """Load the precomputed descriptor block, numeric columns only.

    Parameters
    ----------
    source : str
        Which descriptor family to keep: ``"all"`` for RDKit+Mordred,
        ``"rdkit"`` or ``"mordred"`` to ablate the other family out entirely.
    """
    cache = pd.read_parquet(_DESCRIPTOR_CACHE)
    numeric = cache.drop(columns=["canonical_smiles"]).select_dtypes(include=[np.number])
    if source in {"rdkit", "mordred"}:
        prefix = f"{source}_"
        numeric = numeric[[c for c in numeric.columns if c.startswith(prefix)]]
    all_nan = numeric.columns[numeric.isna().all()]
    if len(all_nan):
        logger.info("Dropping %d all-NaN descriptor column(s)", len(all_nan))
        numeric = numeric.drop(columns=all_nan)
    return pd.concat([cache[["canonical_smiles"]], numeric], axis=1).set_index("canonical_smiles")


def _raw_descriptors(canonical_smiles: list[str], descriptor_cache: pd.DataFrame) -> np.ndarray:
    """Look up the raw (uncompressed) descriptor block for a list of compounds."""
    return descriptor_cache.loc[canonical_smiles].to_numpy(dtype=np.float64)


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


def main() -> None:
    """Fine-tune CheMeleon on pEC50, join descriptors, fit TabPFN, and score the blind set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="moal plan config supplying data/model/trainer settings"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tabpfn_pec50_chemeleon"))
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument(
        "--embedding-pca-components",
        type=int,
        default=256,
        help="Components to compress the 2048-dim CheMeleon embedding to before TabPFN",
    )
    parser.add_argument(
        "--descriptor-pca-components",
        type=int,
        default=256,
        help="Components to compress the RDKit+Mordred descriptor block to before TabPFN",
    )
    parser.add_argument(
        "--descriptor-source",
        choices=["all", "rdkit", "mordred"],
        default="all",
        help="'all' for RDKit+Mordred, or 'rdkit'/'mordred' to keep only that family",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help=(
            "Path to an .npz file of raw CheMeleon embeddings. If it exists, load "
            "embeddings from it and skip fine-tuning entirely. If it does not exist, "
            "fine-tune, write embeddings there, and exit before running TabPFN, so the "
            "GPU-heavy fine-tuning process and the TabPFN process never share one HIP "
            "context (fine-tuning's allocator fragmentation otherwise starves TabPFN's "
            "between-items attention pass of a large contiguous block, even after "
            "`del model; gc.collect(); torch.cuda.empty_cache()`)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for encoder fine-tuning, train/val split, PCA, and TabPFN; "
        "overrides the config's own seed/split_seed. Default: use the config's seed.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    L.seed_everything(seed, workers=True, verbose=False)

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

    if args.embedding_cache is not None and args.embedding_cache.exists():
        cached = np.load(args.embedding_cache)
        train_embed_raw = cached["train_embed_raw"]
        test_embed_raw = cached["test_embed_raw"]
        logger.info(
            "Loaded cached raw CheMeleon embedding %s from %s (skipping fine-tuning)",
            train_embed_raw.shape,
            args.embedding_cache,
        )
    else:
        logger.info("Fine-tuning CheMeleon on %d DRC record(s), target pEC50", len(drc_records))
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
        datamodule_kwargs = cfg.trainer.to_datamodule_kwargs()
        datamodule_kwargs["seed"] = seed
        model.refit(
            drc_records,
            trainer_kwargs=cfg.trainer.to_dict(),
            datamodule_kwargs=datamodule_kwargs,
            output_dir=args.output_dir,
        )

        train_embed_raw = model.embed_smiles(train_smiles)
        test_embed_raw = model.embed_smiles(test_smiles)
        logger.info("Extracted %s raw CheMeleon embedding for training", train_embed_raw.shape)

        # Embedding extraction is the last use of the fine-tuned model; free its
        # GPU memory before TabPFN's own pass claims the device. gc.collect()
        # first: CheMeleon's unfrozen encoder is large enough that Lightning's
        # Trainer/DataModule references from refit() must actually be collected
        # before empty_cache() can release their CUDA allocations, unlike the
        # smaller from-scratch auxiliary encoder in tabpfn_concat_features.py.
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if args.embedding_cache is not None:
            args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.embedding_cache,
                train_embed_raw=train_embed_raw,
                test_embed_raw=test_embed_raw,
            )
            logger.info(
                "Wrote raw CheMeleon embedding to %s; rerun this command in a fresh "
                "process to load it and fit TabPFN without fine-tuning's GPU memory "
                "footprint in the same HIP context",
                args.embedding_cache,
            )
            return

    train_embed, test_embed = _pca_compress(
        train_embed_raw,
        test_embed_raw,
        args.embedding_pca_components,
        seed,
        "CheMeleon embedding",
    )

    train_descriptors_raw = _raw_descriptors(train_smiles, descriptor_cache)
    test_descriptors_raw = _raw_descriptors(test_smiles, descriptor_cache)
    imputer = SimpleImputer(strategy="median").fit(train_descriptors_raw)
    train_descriptors, test_descriptors = _pca_compress(
        imputer.transform(train_descriptors_raw),
        imputer.transform(test_descriptors_raw),
        args.descriptor_pca_components,
        seed,
        "RDKit+Mordred descriptors",
    )

    train_features = np.concatenate([train_embed, train_descriptors], axis=1)
    test_features = np.concatenate([test_embed, test_descriptors], axis=1)
    logger.info("Extracted %s concatenated feature matrix for training", train_features.shape)
    logger.info("Extracted %s concatenated feature matrix for blind targets", test_features.shape)

    regressor = TabPFNRegressor(
        random_state=seed,
        memory_saving_mode=True,
        fit_mode="low_memory",
        ignore_pretraining_limits=train_features.shape[1] > 2000,
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
