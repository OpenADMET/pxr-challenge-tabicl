"""Fit isotonic calibration on out-of-fold DRC predictions and apply it to blind predictions.

Per the N283T model report, a post-hoc isotonic calibration step (fit on
training out-of-fold predictions, applied to the blind test set) closed a
similar-sized gap (~0.441 -> ~0.408 MAE) to several of their architecture
ablations. This script reproduces the OOF-fitting half of that idea for a
single (non-ensembled) moal plan config: k-fold split the DRC training
records, retrain the concatenation architecture (auxiliary encoder + main
model) from scratch on each fold's training split, predict on that fold's
held-out DRC records, and fit an isotonic map from those out-of-fold
predictions to their true pEC50 values. The auxiliary encoder is retrained
per fold on a pool that excludes the held-out DRC records, so no compound
ever contributes to both a fold's own training signal and its OOF prediction.

The map is then applied to an existing blind-test predictions CSV (already
produced by `moal plan`) and the calibrated predictions are re-scored with
`evaluate_predictions.py`'s metrics.

Run with:
    python calibrate_predictions.py configs/freeze2_hd512_clip1.0.yaml \
        results/freeze2_hd512_clip1.0/pxr_aux_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from evaluate_predictions import evaluate
from moal.auxiliary_encoder import AuxiliaryEncoderModule, pretrain_auxiliary_encoder
from moal.concatenation_model import ConcatenationChemPropLightningModule, concatenation_feature_dim
from moal.config import PipelineConfig
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import LabelRecord, QueryType

logger = logging.getLogger(__name__)


def _build_concatenation_model(
    cfg: PipelineConfig, aux_encoder: AuxiliaryEncoderModule
) -> ConcatenationChemPropLightningModule:
    """Instantiate a fold-local concatenation model, mirroring moal.cli's builder."""
    feature_dim = concatenation_feature_dim(len(aux_encoder.task_names), aux_encoder.embedding_dim)
    return ConcatenationChemPropLightningModule(
        concat_feature_dim=feature_dim,
        use_observed_readout=cfg.auxiliary_model.use_observed_readout,
        use_predicted_readout=cfg.auxiliary_model.use_predicted_readout,
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
        learnable_sigma=cfg.model.learnable_sigma,
        from_foundation=cfg.model.from_foundation,
    )


def _load_drc_records(cfg: PipelineConfig) -> tuple[list[LabelRecord], list[LabelRecord]]:
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
    cfg: PipelineConfig, n_folds: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retrain per fold and collect out-of-fold (predicted, true) pEC50 pairs.

    Parameters
    ----------
    cfg : PipelineConfig
        Loaded plan config; must have `auxiliary_model` set (concatenation
        architecture).
    n_folds : int
        Number of KFold splits over the DRC training records.
    seed : int
        Random seed for the KFold shuffle and per-fold model seeding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (oof_predicted_pec50, oof_true_pec50), concatenated across folds.
    """
    if cfg.auxiliary_model is None:
        raise ValueError("collect_oof_predictions requires cfg.auxiliary_model to be set")

    fit_records, drc_records = _load_drc_records(cfg)
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

        model = _build_concatenation_model(cfg, aux_encoder)
        model.refit(
            train_drc,
            aux_encoder=aux_encoder,
            trainer_kwargs=cfg.trainer.to_dict(),
            datamodule_kwargs=cfg.trainer.to_datamodule_kwargs(),
        )

        held_out_smiles = [r.canonical_smiles for r in held_out_drc]
        held_out_readouts = [r.raw_ps_readouts for r in held_out_drc]
        fold_predictions = model.predict_smiles(held_out_smiles, held_out_readouts, aux_encoder)

        oof_predicted.extend(fold_predictions.tolist())
        oof_true.extend(r.value for r in held_out_drc)

    return np.asarray(oof_predicted, dtype=np.float64), np.asarray(oof_true, dtype=np.float64)


def main() -> None:
    """Fit OOF isotonic calibration, apply it to a blind predictions CSV, and re-score."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="moal plan config used to produce predictions_csv")
    parser.add_argument("predictions_csv", type=Path, help="Existing moal plan blind-set predictions CSV")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of OOF folds (default: 5)")
    parser.add_argument("--activity-threshold", type=float, default=6.0, help="Potent-subset threshold (default: 6.0)")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path for the calibrated predictions CSV (default: alongside predictions_csv)",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)

    oof_predicted, oof_true = collect_oof_predictions(cfg, args.n_folds, cfg.seed)
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
