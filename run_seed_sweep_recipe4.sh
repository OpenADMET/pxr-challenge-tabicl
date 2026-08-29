#!/usr/bin/env bash
# Recipe 4: from-scratch/pEC50-trained cell of the encoder-init x
# training-target grid, added to match Recipe 3's 128/128 PCA width fix.
#
# Same script as Recipe 3 (tabpfn_pec50_chemeleon_features.py is config-
# driven via from_foundation, not CheMeleon-specific despite the filename),
# pointed at configs/tabpfn_pec50_scratch.yaml (from_foundation: false)
# instead of the CheMeleon config. Without this, the grid's from-scratch/
# pEC50 cell would stay at the old 256/256 (512-feature) width while every
# other cell moves to 128/128, breaking cross-cell comparability.
#
# Two-pass --embedding-cache pattern per seed, as in Recipe 3: pass 1
# fine-tunes and writes the raw (pre-PCA) embedding to disk and exits
# before touching TabPFN; pass 2 loads it fresh and fits TabPFN alone.
# Caching the raw embedding also means this cache is reusable across any
# future PCA-width sweep for this cell without re-fine-tuning or
# reintroducing fine-tune-seed variance.
set -euo pipefail

SEEDS=(0 1 2 3 4)
log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/pec50_scratch_embed_cache_seed${seed}.npz"

    log "pec50_scratch: seed=$seed, pass 1 (fine-tune, write cache)"
    python tabpfn_pec50_chemeleon_features.py configs/tabpfn_pec50_scratch.yaml \
        --seed "$seed" \
        --embedding-cache "$cache" \
        --output-dir "results/tabpfn_pec50_scratch_seed${seed}"

    log "pec50_scratch: seed=$seed, pass 2 (load cache, fit TabPFN)"
    python tabpfn_pec50_chemeleon_features.py configs/tabpfn_pec50_scratch.yaml \
        --seed "$seed" \
        --embedding-cache "$cache" \
        --embedding-pca-components 128 \
        --descriptor-pca-components 128 \
        --output-dir "results/tabpfn_pec50_scratch_seed${seed}"
done

log "Recipe 4 seed sweep complete."
