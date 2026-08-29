#!/usr/bin/env bash
# Recipe 6: RDKit+Mordred combined descriptor block, PCA-width sweep
# (panel 04's second mini-chart), 5 seeds each. Reuses the same
# auxiliary_embedding_cache_seed{0..4}.parquet files recipe 1 already
# minted, so this sweep's encoder-init noise matches every other 5-seed
# row on the page, and its widths line up with the Mordred-only sweep's
# same three PCA widths for a like-for-like comparison.
set -euo pipefail

SEEDS=(0 1 2 3 4)
CONFIG=configs/tabpfn_small_embed.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "concat_pca64 (RDKit+Mordred) seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" \
        --descriptor-source all --pca-components 64 --regressor tabpfn \
        --output-dir "results/tabpfn_concat_pca64_seed${seed}"

    log "concat_small_embed (RDKit+Mordred, PCA-128) seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" \
        --descriptor-source all --pca-components 128 --regressor tabpfn \
        --output-dir "results/tabpfn_concat_small_embed_seed${seed}"

    log "concat_pca256 (RDKit+Mordred) seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" \
        --descriptor-source all --pca-components 256 --regressor tabpfn \
        --output-dir "results/tabpfn_concat_pca256_seed${seed}"
done

log "Recipe 6 (RDKit+Mordred PCA-width sweep) seed sweep complete."
