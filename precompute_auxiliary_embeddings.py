"""Mint an auxiliary-encoder embedding, reused by every downstream ablation that shares its seed.

`tabpfn_concat_features.py` and `tabpfn_uncertainty_analysis.py` each called
`pretrain_auxiliary_encoder` independently, so every PCA width, descriptor
family, regressor choice, and diagnostic pass fit its own from-scratch,
randomly initialized 256-dim encoder. That confounds any comparison across
those runs: a MAE difference could reflect encoder-init variance rather than
the feature or regressor choice actually being ablated. This script trains the
encoder from a config (e.g. `configs/tabpfn_small_embed.yaml`) exactly once
per `--seed`, saves its weights as a checkpoint, and precomputes the
embedding + predicted-log2FC-readout block for every compound any of those
scripts touches (all dose-response training records plus the blind unqueried
set), so every downstream run at a given seed reads the identical embedding
instead of retraining it.

The encoder's weight initialization was previously uncontrolled (this script
never called `L.seed_everything`), so every invocation already produced a
different embedding by accident. `--seed` makes that variation deliberate and
reproducible: running the same seed twice reproduces the same embedding, and
running 5 distinct seeds gives 5 independent draws suitable for reporting a
distribution (mean +/- spread) on downstream MAE, rather than a single point
estimate that silently bakes in one arbitrary draw of encoder-init noise.

`tabpfn_calibrate_predictions.py` deliberately retrains the encoder per k-fold
split (excluding the held-out fold, to stay leakage-free) and must keep doing
so; it does not read this cache.

Run with:
    python precompute_auxiliary_embeddings.py configs/tabpfn_small_embed.yaml --seed 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch

from moal.auxiliary_encoder import pretrain_auxiliary_encoder, save_auxiliary_encoder_checkpoint
from moal.config import PipelineConfig
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import QueryType

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path("data")
_EMBEDDING_CACHE_DIR = Path("data")


def _checkpoint_path(seed: int) -> Path:
    """Return the seed-specific encoder checkpoint path."""
    return _CHECKPOINT_DIR / f"auxiliary_encoder_256d_seed{seed}.pt"


def _embedding_cache_path(seed: int) -> Path:
    """Return the seed-specific embedding cache path."""
    return _EMBEDDING_CACHE_DIR / f"auxiliary_embedding_cache_seed{seed}.parquet"


def main() -> None:
    """Train the auxiliary encoder once per seed and cache its embedding+readout output to parquet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="moal plan config supplying data/auxiliary_model settings")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for encoder weight init and train/val split; overrides the config's own seed/split_seed",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if cfg.auxiliary_model is None:
        raise ValueError(f"{args.config} has no auxiliary_model block")

    L.seed_everything(args.seed, workers=True, verbose=False)

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
    test_smiles = [smi for _, smi in state.unqueried_rows]
    # De-duplicate while keeping a stable, reproducible row order for the cache
    all_smiles = list(dict.fromkeys(train_smiles + test_smiles))

    datamodule_kwargs = cfg.auxiliary_trainer.to_datamodule_kwargs()
    datamodule_kwargs["seed"] = args.seed

    logger.info(
        "Pretraining auxiliary encoder (seed=%d) on %d readout-bearing record(s)",
        args.seed,
        len(fit_records),
    )
    aux_encoder = pretrain_auxiliary_encoder(
        fit_records,
        cfg.auxiliary_model,
        trainer_kwargs=cfg.auxiliary_trainer.to_dict(),
        datamodule_kwargs=datamodule_kwargs,
    )

    checkpoint_path = _checkpoint_path(args.seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_auxiliary_encoder_checkpoint(aux_encoder, checkpoint_path)
    logger.info("Saved encoder checkpoint to %s", checkpoint_path)

    embed = aux_encoder.embed_smiles(all_smiles)
    readout = aux_encoder.predict_smiles(all_smiles)
    aux_encoder.to("cpu")
    torch.cuda.empty_cache()

    embed_cols = [f"embed_{i}" for i in range(embed.shape[1])]
    readout_cols = [f"readout_{i}" for i in range(readout.shape[1])]
    cache = pd.DataFrame(
        np.concatenate([embed, readout], axis=1),
        columns=embed_cols + readout_cols,
    )
    cache.insert(0, "canonical_smiles", all_smiles)
    embedding_cache_path = _embedding_cache_path(args.seed)
    embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(embedding_cache_path, index=False)
    logger.info(
        "Wrote %d compound(s), %d embed + %d readout column(s) to %s",
        len(cache),
        len(embed_cols),
        len(readout_cols),
        embedding_cache_path,
    )


if __name__ == "__main__":
    main()
