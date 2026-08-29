"""Baseline: frozen log2FC-encoder features fed through TabPFN instead of a fine-tuned GNN.

Isolates one variable in the gap to the N283T report: their winning single-
pathway number (0.437 MAE) comes from a frozen log2fc-pretrained encoder's
embedding fed into TabPFN, not from an end-to-end fine-tuned GNN (E3/E4 in
this repo's terminology). This script reproduces that shape directly: read the canonical, frozen
auxiliary-encoder features (embedding + predicted readout) from
`data/auxiliary_embedding_cache.parquet` (`precompute_auxiliary_embeddings.py`,
one shared encoder trained once and reused across every script that needs
these features) and fit a plain TabPFNRegressor on them against pEC50. No GNN
fine-tuning happens after log2FC pretraining at all.

This is a variable-isolation baseline, not a replacement for the concatenation
architecture: it does not use molecular descriptors (Mordred/RDKit/Boltz-2)
the way the report's actual winning ensemble member did, so it answers "does
TabPFN help over our fine-tuned GNN, holding features fixed" rather than
"can we match the report's full pipeline."

The auxiliary encoder must produce a feature vector under TabPFN's 2000-
feature limit and small enough to fit alongside TabPFN's attention pass on
the GPU. `configs/freeze2_hd512_clip1.0.yaml`'s CheMeleon-initialized encoder
fixes the embedding at CheMeleon's native 2048 dims regardless of
`auxiliary_model.message_hidden_dim`, so use a config with
`auxiliary_model.from_foundation: false` instead, which lets
`message_hidden_dim` set the embedding size directly (e.g.
`configs/tabpfn_small_embed.yaml`, 256 dims, matching the report's own
ChemProp-256 frozen-embedding ensemble member).

Run with:
    python tabpfn_frozen_features.py configs/tabpfn_small_embed.yaml \
        --output-dir results/tabpfn_frozen_features
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor

from evaluate_predictions import evaluate
from moal.config import PipelineConfig
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import QueryType
from tabpfn_concat_features import _embedding_features, _load_embedding_cache

logger = logging.getLogger(__name__)


def main() -> None:
    """Read the cached frozen-encoder features, fit TabPFN, and score the blind set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="moal plan config supplying data/auxiliary_model settings")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tabpfn_frozen_features"))
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Embedding cache to read (default: the canonical seed-42 cache); pass a "
        "seed-specific cache from precompute_auxiliary_embeddings.py --seed to hold "
        "the encoder fixed within a 5-seed replicate comparison",
    )
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument(
        "--no-readout",
        action="store_true",
        help="Drop the predicted-log2FC-readout columns; use the raw embedding alone",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Drop the raw embedding columns; use the predicted-readout alone",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed (regressor random state) for a seed-sweep replicate",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    if cfg.auxiliary_model is None:
        raise ValueError(f"{args.config} has no auxiliary_model block; TabPFN baseline needs one")

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

    embedding_cache = _load_embedding_cache(args.embedding_cache)

    train_smiles = [r.canonical_smiles for r in drc_records]
    train_true = np.array([r.value for r in drc_records], dtype=np.float64)
    train_features = _embedding_features(
        embedding_cache, train_smiles, not args.no_embedding, not args.no_readout
    )
    logger.info("Extracted %s frozen feature matrix for training", train_features.shape)

    test_smiles = [smi for _, smi in state.unqueried_rows]
    test_features = _embedding_features(
        embedding_cache, test_smiles, not args.no_embedding, not args.no_readout
    )
    logger.info("Extracted %s frozen feature matrix for blind targets", test_features.shape)

    # "auto" memory_saving_mode misjudged the free HIP memory and OOM'd trying
    # to allocate the full (unchunked) between-items attention tensor for
    # 4139 training rows; forcing memory_saving_mode=True makes it always
    # take the chunked path instead. fit_mode="low_memory" preprocesses
    # on-demand at predict time instead of caching the transformer's
    # key-value state, trading speed for the lowest possible GPU footprint
    # CheMeleon's 2048-dim embedding plus the 2-dim predicted readout exceeds
    # TabPFN's officially-supported 2000-feature limit
    regressor = TabPFNRegressor(
        random_state=cfg.seed,
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
