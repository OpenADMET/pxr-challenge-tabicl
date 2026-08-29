"""Fit isotonic calibration on out-of-fold predictions for the CheMeleon+TabICL winner, then apply it to blind predictions.

Adapts `tabpfn_calibrate_predictions.py`'s OOF-calibration idea to the actual
current best config (`tabpfn_chemeleon_concat_features.py`'s off-the-shelf
CheMeleon embedding + borrowed log2FC readout, no descriptors, TabICL
regressor; recipe 10 in `run_seed_sweep_recipe10.sh`), not the from-scratch
embedding + TabPFN pipeline the existing calibration script targets.

The CheMeleon embedding is frozen and off-the-shelf: it was never fit on any
label, so it can be extracted once, up front, for every compound (train and
blind alike), with no leakage risk. The borrowed readout block is different:
it comes from an auxiliary log2FC encoder that WAS trained on labels, so
reusing the fixed `auxiliary_embedding_cache_seed{N}.parquet` (trained on all
compounds) directly for OOF evaluation on those same compounds would leak.
This script instead retrains that auxiliary encoder per fold, excluding the
fold's held-out compounds from its training pool, exactly as
`tabpfn_calibrate_predictions.py` already does for its own OOF loop.

Run with:
    python tabpfn_chemeleon_calibrate_predictions.py configs/freeze1_hd512_clip5.0.yaml \
        results/tabicl_chemeleon_readout_only_seed0/pxr_tabpfn_predictions.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from evaluate_predictions import evaluate
from moal.auxiliary_encoder import pretrain_auxiliary_encoder
from moal.config import PipelineConfig
from moal.model import ChemPropLightningModule
from moal.types import LabelRecord
from tabpfn_calibrate_predictions import _load_fit_and_drc_records
from tabpfn_chemeleon_concat_features import _pca_compress
from tabpfn_concat_features import _build_regressor, _live_embedding_features

# TabICL's flash-attention kernel on this ROCm/Navi31 GPU is experimental and,
# without this flag, silently returns all-NaN predictions for some fold sizes
# (reproduced on a 5-fold OOF split: 4/5 folds fine, 1/5 fold entirely NaN)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

logger = logging.getLogger(__name__)


def _embed_all_compounds(
    model: ChemPropLightningModule, canonical_smiles: list[str]
) -> np.ndarray:
    """Extract raw CheMeleon embeddings for every compound, once, up front.

    Safe to compute outside the fold loop: the embedding comes from a frozen,
    never-fine-tuned CheMeleon checkpoint, so it was never fit on any label
    and carries no leakage risk across folds.
    """
    return model.embed_smiles(canonical_smiles)


def collect_oof_predictions(
    cfg: PipelineConfig,
    fit_records: list[LabelRecord],
    drc_records: list[LabelRecord],
    embed_by_smiles: dict[str, np.ndarray],
    embedding_pca_components: int,
    n_folds: int,
    seed: int,
    tabicl_batch_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrain the auxiliary readout encoder per fold and collect OOF (predicted, true) pairs.

    Parameters
    ----------
    embed_by_smiles : dict[str, np.ndarray]
        Raw (uncompressed) CheMeleon embedding per canonical SMILES, extracted
        once up front (see `_embed_all_compounds`); PCA is still refit per
        fold on that fold's training split only.
    """
    if cfg.auxiliary_model is None:
        raise ValueError("collect_oof_predictions requires cfg.auxiliary_model to be set")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_predicted: list[float] = []
    oof_true: list[float] = []

    for fold_idx, (train_idx, held_out_idx) in enumerate(kfold.split(drc_records)):
        logger.info("Fold %d/%d: %d train, %d held out", fold_idx + 1, n_folds, len(train_idx), len(held_out_idx))

        train_drc = [drc_records[i] for i in train_idx]
        held_out_drc = [drc_records[i] for i in held_out_idx]
        held_out_ids = {id(r) for r in held_out_drc}
        aux_fit_records = [r for r in fit_records if id(r) not in held_out_ids]

        aux_encoder = pretrain_auxiliary_encoder(
            aux_fit_records,
            cfg.auxiliary_model,
            trainer_kwargs=cfg.auxiliary_trainer.to_dict(),
            datamodule_kwargs=cfg.auxiliary_trainer.to_datamodule_kwargs(),
        )

        train_smiles = [r.canonical_smiles for r in train_drc]
        held_out_smiles = [r.canonical_smiles for r in held_out_drc]
        train_true = np.array([r.value for r in train_drc], dtype=np.float64)
        held_out_true = np.array([r.value for r in held_out_drc], dtype=np.float64)

        train_readout = _live_embedding_features(aux_encoder, train_smiles, False, True)
        held_out_readout = _live_embedding_features(aux_encoder, held_out_smiles, False, True)

        # Readout extraction is the last use of this fold's aux encoder; free
        # its GPU memory before TabICL's own pass claims the device
        aux_encoder.to("cpu")
        torch.cuda.empty_cache()

        train_embed_raw = np.stack([embed_by_smiles[s] for s in train_smiles])
        held_out_embed_raw = np.stack([embed_by_smiles[s] for s in held_out_smiles])
        train_embed, held_out_embed = _pca_compress(
            train_embed_raw, held_out_embed_raw, embedding_pca_components, seed, "CheMeleon embedding"
        )

        train_features = np.concatenate([train_embed, train_readout], axis=1)
        held_out_features = np.concatenate([held_out_embed, held_out_readout], axis=1)

        regressor = _build_regressor(
            "tabicl", seed, train_features.shape[1], tabicl_batch_size=tabicl_batch_size
        )
        regressor.fit(train_features, train_true)
        fold_predictions = regressor.predict(held_out_features)
        if np.isnan(fold_predictions).any():
            raise RuntimeError(
                f"Fold {fold_idx + 1}: TabICL returned "
                f"{int(np.isnan(fold_predictions).sum())} NaN prediction(s)"
            )

        oof_predicted.extend(fold_predictions.tolist())
        oof_true.extend(held_out_true.tolist())

    return np.asarray(oof_predicted, dtype=np.float64), np.asarray(oof_true, dtype=np.float64)


def main() -> None:
    """Fit OOF isotonic calibration on the CheMeleon+TabICL pipeline, apply it to a blind predictions CSV, and re-score."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="moal plan config supplying data/auxiliary_model settings")
    parser.add_argument("predictions_csv", type=Path, help="Existing blind-set predictions CSV to calibrate")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of OOF folds (default: 5)")
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument(
        "--embedding-pca-components",
        type=int,
        default=256,
        help="Components to compress the CheMeleon embedding to (must match the original run)",
    )
    parser.add_argument(
        "--tabicl-batch-size",
        type=int,
        default=None,
        help="batch_size for TabICLRegressor; None keeps the library default (8), "
        "matching recipe 10's choice for this featureset",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path for the calibrated predictions CSV (default: alongside predictions_csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed (fold split, PCA, and regressor random state) for a seed-sweep replicate",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    fit_records, drc_records = _load_fit_and_drc_records(cfg)

    predictions_df = pd.read_csv(args.predictions_csv)
    blind_smiles = predictions_df["smiles"].tolist()
    all_smiles = sorted({*(r.canonical_smiles for r in fit_records), *blind_smiles})

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
    if torch.cuda.is_available():
        model = model.to("cuda")

    all_embed_raw = _embed_all_compounds(model, all_smiles)
    embed_by_smiles = dict(zip(all_smiles, all_embed_raw))
    logger.info("Extracted raw CheMeleon embedding for %d compound(s)", len(all_smiles))

    # Embedding extraction is the last use of the untouched model; free its
    # GPU memory before the per-fold auxiliary-encoder retraining claims it
    del model
    gc.collect()
    torch.cuda.empty_cache()

    oof_predicted, oof_true = collect_oof_predictions(
        cfg,
        fit_records,
        drc_records,
        embed_by_smiles,
        args.embedding_pca_components,
        args.n_folds,
        cfg.seed,
        args.tabicl_batch_size,
    )
    oof_mae_before = float(np.mean(np.abs(oof_predicted - oof_true)))
    logger.info("Collected %d OOF predictions, uncalibrated OOF MAE = %.4f", len(oof_true), oof_mae_before)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_predicted, oof_true)
    oof_calibrated = calibrator.predict(oof_predicted)
    oof_mae_after = float(np.mean(np.abs(oof_calibrated - oof_true)))
    logger.info("Calibrated OOF MAE = %.4f (uncalibrated %.4f)", oof_mae_after, oof_mae_before)

    has_prediction = predictions_df["predicted_pec50"].notna()
    predictions_df.loc[has_prediction, "predicted_pec50"] = calibrator.predict(
        predictions_df.loc[has_prediction, "predicted_pec50"].to_numpy(dtype=np.float64)
    )

    output_csv = args.output_csv or args.predictions_csv.with_stem(args.predictions_csv.stem + "_calibrated")
    predictions_df.to_csv(output_csv, index=False)
    logger.info("Wrote calibrated predictions to %s", output_csv)

    logger.info("Uncalibrated blind-set metrics:")
    uncalibrated = evaluate(args.predictions_csv, activity_threshold=args.activity_threshold)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(uncalibrated.to_string(index=False))

    logger.info("Calibrated blind-set metrics:")
    calibrated = evaluate(output_csv, activity_threshold=args.activity_threshold)
    print(calibrated.to_string(index=False))


if __name__ == "__main__":
    main()
