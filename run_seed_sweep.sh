#!/usr/bin/env bash
# 5-seed replicate sweep: mints 5 seed-varied embeddings per tabular encoder
# recipe and reruns the load-bearing downstream comparisons against each,
# so MAE differences already argued in the artifact can be checked against
# encoder-init/split noise instead of a single point estimate.
set -euo pipefail

SEEDS=(0 1 2 3 4)

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Recipe 1: from-scratch log2FC encoder (256-dim) ---
for seed in "${SEEDS[@]}"; do
    log "precompute: tabpfn_small_embed seed=$seed"
    python precompute_auxiliary_embeddings.py configs/tabpfn_small_embed.yaml --seed "$seed"

    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "concat: embed_readout_mordred_pca128 seed=$seed"
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --descriptor-source mordred --pca-components 128 --regressor tabpfn \
        --output-dir "results/tabpfn_embed_readout_mordred_pca128_seed${seed}"

    log "concat: embed_readout_mordred_pca256 seed=$seed"
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --descriptor-source mordred --pca-components 256 --regressor tabpfn \
        --output-dir "results/tabpfn_embed_readout_mordred_pca256_seed${seed}"

    log "concat: embed_readout_mordred_pca64 seed=$seed"
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --descriptor-source mordred --pca-components 64 --regressor tabpfn \
        --output-dir "results/tabpfn_embed_readout_mordred_pca64_seed${seed}"

    log "concat: tabicl embed_readout_mordred_pca128 seed=$seed"
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --descriptor-source mordred --pca-components 128 --regressor tabicl \
        --output-dir "results/tabicl_embed_readout_mordred_pca128_seed${seed}"

    log "concat: tabfm embed_readout_mordred_pca128 seed=$seed"
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --descriptor-source mordred --pca-components 128 --regressor tabfm \
        --output-dir "results/tabfm_embed_readout_mordred_pca128_seed${seed}"

    log "concat: lgbm embed_readout_mordred_pca128 seed=$seed"
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --descriptor-source mordred --pca-components 128 --regressor lgbm \
        --output-dir "results/lgbm_embed_readout_mordred_pca128_seed${seed}"

    log "concat: xgboost embed_readout_mordred_pca128 seed=$seed"
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --descriptor-source mordred --pca-components 128 --regressor xgboost \
        --output-dir "results/xgboost_embed_readout_mordred_pca128_seed${seed}"

    log "frozen: embed_readout, no descriptors, seed=$seed"
    python tabpfn_frozen_features.py configs/tabpfn_small_embed.yaml \
        --embedding-cache "$cache" \
        --output-dir "results/tabpfn_frozen_features_seed${seed}"
done

# --- Recipe 2: CheMeleon-init log2FC encoder (retrains internally per run; no shared cache) ---
for seed in "${SEEDS[@]}"; do
    log "chemeleon_log2fc: embed_readout_mordred_pca128 seed=$seed"
    python tabpfn_chemeleon_log2fc_features.py configs/tabpfn_chemeleon_log2fc.yaml \
        --seed "$seed" --descriptor-source mordred \
        --output-dir "results/tabpfn_chemeleon_log2fc_mordred_pca128_seed${seed}"
done

# --- Recipe 3: CheMeleon-init, pEC50-direct-trained encoder ---
for seed in "${SEEDS[@]}"; do
    log "pec50_chemeleon: seed=$seed"
    python tabpfn_pec50_chemeleon_features.py configs/freeze1_hd512_clip5.0.yaml \
        --seed "$seed" \
        --output-dir "results/tabpfn_pec50_chemeleon_seed${seed}"
done

log "Seed sweep (tabular recipes) complete."
