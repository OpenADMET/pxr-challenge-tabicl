"""Fit isotonic calibration on out-of-fold TabPFN predictions, then apply it to blind predictions.

Adapts `calibrate_predictions.py`'s OOF-calibration idea to the frozen-feature
+ TabPFN pipeline (`tabpfn_concat_features.py`) instead of the fine-tuned-GNN
concatenation architecture: k-fold split the DRC training records, per fold
retrain the auxiliary encoder on a pool that excludes the held-out fold, fit
the descriptor imputer and PCA on that fold's training split only, fit a fresh
`TabPFNRegressor` on the fold's assembled features, and predict on the held-out
DRC records. Fit an isotonic map from the collected out-of-fold predictions to
their true pEC50 values, then apply that map to an existing blind-test
predictions CSV and re-score with `evaluate_predictions.py`.

Run with:
    python tabpfn_calibrate_predictions.py configs/tabpfn_small_embed.yaml \
        results/tabpfn_embed_readout_mordred_pca128/pxr_tabpfn_predictions.csv \
        --descriptor-source mordred --pca-components 128
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

from evaluate_predictions import evaluate
from moal.auxiliary_encoder import pretrain_auxiliary_encoder
from moal.config import PipelineConfig
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import LabelRecord, QueryType
from tabpfn_concat_features import _live_embedding_features, _load_descriptor_cache, _raw_descriptors

logger = logging.getLogger(__name__)


def _load_fit_and_drc_records(cfg: PipelineConfig) -> tuple[list[LabelRecord], list[LabelRecord]]:
    """Parse the campaign state CSV and split into (all fit records, DRC-only records)."""
    state_df = pd.read_csv(cfg.data.plan.input_csv)
    preprocessor = SMILESPreprocessor()
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
    return fit_records, drc_records


def collect_oof_predictions(
    cfg: PipelineConfig,
    descriptor_cache: pd.DataFrame,
    pca_components: int,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrain the frozen-feature + TabPFN pipeline per fold and collect OOF (predicted, true) pairs."""
    if cfg.auxiliary_model is None:
        raise ValueError("collect_oof_predictions requires cfg.auxiliary_model to be set")

    fit_records, drc_records = _load_fit_and_drc_records(cfg)
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

        train_embed = _live_embedding_features(aux_encoder, train_smiles, True, True)
        held_out_embed = _live_embedding_features(aux_encoder, held_out_smiles, True, True)

        # Feature extraction is the last use of the aux encoder; free its GPU
        # memory before TabPFN's own pass claims the device, since the Lightning
        # module otherwise stays resident on GPU after training
        aux_encoder.to("cpu")
        torch.cuda.empty_cache()

        train_descriptors_raw = _raw_descriptors(train_smiles, descriptor_cache)
        held_out_descriptors_raw = _raw_descriptors(held_out_smiles, descriptor_cache)

        imputer = SimpleImputer(strategy="median").fit(train_descriptors_raw)
        pca = PCA(n_components=pca_components, random_state=seed).fit(imputer.transform(train_descriptors_raw))
        train_descriptors = pca.transform(imputer.transform(train_descriptors_raw))
        held_out_descriptors = pca.transform(imputer.transform(held_out_descriptors_raw))

        train_features = np.concatenate([train_embed, train_descriptors], axis=1)
        held_out_features = np.concatenate([held_out_embed, held_out_descriptors], axis=1)

        regressor = TabPFNRegressor(
            random_state=seed,
            memory_saving_mode=True,
            fit_mode="low_memory",
            ignore_pretraining_limits=train_features.shape[1] > 2000,
        )
        regressor.fit(train_features, train_true)
        fold_predictions = regressor.predict(held_out_features)

        oof_predicted.extend(fold_predictions.tolist())
        oof_true.extend(held_out_true.tolist())

    return np.asarray(oof_predicted, dtype=np.float64), np.asarray(oof_true, dtype=np.float64)


def main() -> None:
    """Fit OOF isotonic calibration on the TabPFN pipeline, apply it to a blind predictions CSV, and re-score."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="moal plan config supplying data/auxiliary_model settings")
    parser.add_argument("predictions_csv", type=Path, help="Existing blind-set predictions CSV to calibrate")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of OOF folds (default: 5)")
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Components to compress the descriptor block to before TabPFN (must match the original run)",
    )
    parser.add_argument(
        "--descriptor-source",
        choices=["all", "rdkit", "mordred"],
        default="mordred",
        help="Descriptor family to use (must match the original run)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path for the calibrated predictions CSV (default: alongside predictions_csv)",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    descriptor_cache = _load_descriptor_cache(args.descriptor_source)

    oof_predicted, oof_true = collect_oof_predictions(
        cfg, descriptor_cache, args.pca_components, args.n_folds, cfg.seed
    )
    oof_mae_before = float(np.mean(np.abs(oof_predicted - oof_true)))
    logger.info("Collected %d OOF predictions, uncalibrated OOF MAE = %.4f", len(oof_true), oof_mae_before)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_predicted, oof_true)
    oof_calibrated = calibrator.predict(oof_predicted)
    oof_mae_after = float(np.mean(np.abs(oof_calibrated - oof_true)))
    logger.info("Calibrated OOF MAE = %.4f (uncalibrated %.4f)", oof_mae_after, oof_mae_before)

    predictions_df = pd.read_csv(args.predictions_csv)
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
