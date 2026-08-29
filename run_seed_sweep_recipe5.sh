#!/usr/bin/env bash
# 5-seed replicate sweep for panel 01/02's single-ingredient and pairwise rows,
# the last group in the artifact still resting on a single run each. Reuses
# the same from-scratch encoder caches (data/auxiliary_embedding_cache_seed*.parquet)
# recipe 1 already minted, so embedding-derived rows share encoder-init noise
# with every other 5-seed row. The two CheMeleon-static rows have no encoder
# to re-init (CheMeleon's weights are pretrained and frozen here), so their
# --seed only varies PCA's randomized solver and the regressor's random
# state, not encoder init; their whisker will be narrower for that reason,
# not because the row is more stable.
set -euo pipefail

SEEDS=(0 1 2 3 4)
CONFIG=configs/tabpfn_small_embed.yaml
CHEMELEON_CONFIG=configs/freeze1_hd512_clip5.0.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "embed_only_no_readout seed=$seed"
    python tabpfn_frozen_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-readout \
        --output-dir "results/tabpfn_embed_only_no_readout_seed${seed}"

    log "readout_only_no_embed seed=$seed"
    python tabpfn_frozen_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-embedding \
        --output-dir "results/tabpfn_readout_only_no_embed_seed${seed}"

    log "small_embed (embedding + readout, no descriptors) seed=$seed"
    python tabpfn_frozen_features.py "$CONFIG" \
        --embedding-cache "$cache" \
        --output-dir "results/tabpfn_small_embed_seed${seed}"

    log "mordred_only_pca128_no_embed_no_readout seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --no-embedding --no-readout \
        --descriptor-source mordred --pca-components 128 --seed "$seed" \
        --output-dir "results/tabpfn_mordred_only_pca128_no_embed_no_readout_seed${seed}"

    log "rdkit_only_pca128_no_embed_no_readout seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --no-embedding --no-readout \
        --descriptor-source rdkit --pca-components 128 --seed "$seed" \
        --output-dir "results/tabpfn_rdkit_only_pca128_no_embed_no_readout_seed${seed}"

    log "embed_mordred_pca128_no_readout seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-readout \
        --descriptor-source mordred --pca-components 128 --seed "$seed" \
        --output-dir "results/tabpfn_embed_mordred_pca128_no_readout_seed${seed}"

    log "readout_mordred_pca128_no_embed seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-embedding \
        --descriptor-source mordred --pca-components 128 --seed "$seed" \
        --output-dir "results/tabpfn_readout_mordred_pca128_no_embed_seed${seed}"

    log "chemeleon_static (embedding + descriptors) seed=$seed"
    python tabpfn_chemeleon_static_features.py "$CHEMELEON_CONFIG" \
        --seed "$seed" \
        --output-dir "results/tabpfn_chemeleon_static_seed${seed}"

    log "chemeleon_embed_only seed=$seed"
    python tabpfn_chemeleon_static_features.py "$CHEMELEON_CONFIG" \
        --seed "$seed" --skip-descriptors \
        --output-dir "results/tabpfn_chemeleon_embed_only_seed${seed}"
done

log "Recipe 5 (panel 01/02 single-ingredient) seed sweep complete."
