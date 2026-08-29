#!/usr/bin/env bash
# Recipe 8: side experiment swapping section 02's from-scratch, log2FC-
# pretrained encoder embedding for CheMeleon's untouched, off-the-shelf
# pretrained embedding (PCA-256, same treatment as panel 01's CheMeleon-
# embedding row), while keeping everything else matched to section 02:
# Mordred-only descriptors at PCA-128, and TabPFN v2.5 (the unpinned
# default) as the regressor. CheMeleon is never fine-tuned here, so it has
# no predicted-readout head of its own; where a combo needs a readout
# block, this borrows the from-scratch encoder's own 2-column predicted-
# readout from the same auxiliary_embedding_cache_seed{0..4}.parquet files
# recipe 1 minted, isolating the embedding swap as the only changed
# variable relative to section 02's equivalent rows.
set -euo pipefail

SEEDS=(0 1 2 3 4)
CONFIG=configs/freeze1_hd512_clip5.0.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "chemeleon readout+descriptors seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" \
        --output-dir "results/tabpfn_chemeleon_readout_descriptors_seed${seed}"

    log "chemeleon descriptors only (no readout) seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --no-readout \
        --output-dir "results/tabpfn_chemeleon_descriptors_only_seed${seed}"

    log "chemeleon readout only (no descriptors) seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --no-descriptors \
        --output-dir "results/tabpfn_chemeleon_readout_only_seed${seed}"
done

log "Recipe 8 (off-the-shelf CheMeleon embedding side experiment) seed sweep complete."
