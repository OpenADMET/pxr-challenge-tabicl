"""CheMeleon-initialized, log2FC-finetuned embedding + Mordred descriptors fed through TabPFN.

Completes the 2x2 grid of {CheMeleon-initialized, from-scratch} x
{log2FC-trained, pEC50-trained} embeddings. The from-scratch/log2FC cell is
the existing backbone (`tabpfn_concat_features.py`); the CheMeleon/pEC50 cell
is `tabpfn_pec50_chemeleon_features.py`. This script is the missing
CheMeleon/log2FC cell: pretrain `AuxiliaryEncoderModule` (same log2FC
multi-task objective as the backbone) with `auxiliary_model.from_foundation:
chemeleon`, so the message-passing encoder starts from CheMeleon's
pretrained weights instead of random initialization.

CheMeleon's native embedding width is fixed at 2048 regardless of config, so
unlike the from-scratch route's 256-dim embedding, this needs its own PCA
compression before concatenation with the (also PCA-compressed) descriptor
block, mirroring `tabpfn_pec50_chemeleon_features.py`'s approach, to avoid
the TabPFN memory ceiling observed with raw CheMeleon-scale embeddings.

Run with:
    python tabpfn_chemeleon_log2fc_features.py configs/tabpfn_chemeleon_log2fc.yaml \
        --output-dir results/tabpfn_chemeleon_log2fc_mordred_pca128
"""

from __future__ import annotations

import argparse
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
from moal.auxiliary_encoder import pretrain_auxiliary_encoder
from moal.config import PipelineConfig
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
    """Fit PCA on the training split and apply it to both splits."""
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
    """Pretrain the CheMeleon-initialized log2FC encoder, join descriptors, fit TabPFN, and score."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="moal plan config supplying data/auxiliary_model settings"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tabpfn_chemeleon_log2fc")
    )
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument(
        "--embedding-pca-components",
        type=int,
        default=128,
        help="Components to compress the 2048-dim CheMeleon embedding to before TabPFN",
    )
    parser.add_argument(
        "--descriptor-pca-components",
        type=int,
        default=128,
        help="Components to compress the descriptor block to before TabPFN",
    )
    parser.add_argument(
        "--descriptor-source",
        choices=["all", "rdkit", "mordred"],
        default="mordred",
        help="'all' for RDKit+Mordred, or 'rdkit'/'mordred' to keep only that family",
    )
    parser.add_argument(
        "--no-readout",
        action="store_true",
        help="Drop the predicted-log2FC-readout block; use the raw embedding alone",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for encoder weight init, train/val split, PCA, and TabPFN; "
        "overrides the config's own seed/split_seed. Default: use the config's seed.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if cfg.auxiliary_model is None:
        raise ValueError(f"{args.config} has no auxiliary_model block; TabPFN baseline needs one")

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

    datamodule_kwargs = cfg.auxiliary_trainer.to_datamodule_kwargs()
    datamodule_kwargs["seed"] = seed

    logger.info(
        "Pretraining CheMeleon-initialized auxiliary encoder (seed=%d) on %d readout-bearing record(s)",
        seed,
        len(fit_records),
    )
    aux_encoder = pretrain_auxiliary_encoder(
        fit_records,
        cfg.auxiliary_model,
        trainer_kwargs=cfg.auxiliary_trainer.to_dict(),
        datamodule_kwargs=datamodule_kwargs,
    )

    train_embed_raw = aux_encoder.embed_smiles(train_smiles)
    test_embed_raw = aux_encoder.embed_smiles(test_smiles)
    logger.info("Extracted %s raw CheMeleon embedding for training", train_embed_raw.shape)

    if not args.no_readout:
        train_readout = aux_encoder.predict_smiles(train_smiles)
        test_readout = aux_encoder.predict_smiles(test_smiles)
    else:
        train_readout = np.empty((len(train_smiles), 0), dtype=np.float32)
        test_readout = np.empty((len(test_smiles), 0), dtype=np.float32)

    # Embedding extraction is the last use of the aux encoder; free its GPU
    # memory before TabPFN's own pass claims the device, since the Lightning
    # module otherwise stays resident on GPU after training
    aux_encoder.to("cpu")
    torch.cuda.empty_cache()

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
        f"{args.descriptor_source} descriptors",
    )

    train_features = np.concatenate([train_embed, train_readout, train_descriptors], axis=1)
    test_features = np.concatenate([test_embed, test_readout, test_descriptors], axis=1)
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
