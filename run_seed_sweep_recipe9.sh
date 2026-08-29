#!/usr/bin/env bash
# Recipe 9: "same features, different regressor" sweep on
# chemeleon_readout_descriptors (recipe 8's off-the-shelf CheMeleon
# embedding + borrowed from-scratch readout + Mordred descriptors, PCA-128
# / PCA-256), the row that unexpectedly beat our_best_overall's TabPFN v2.5
# result within panel 02. Mirrors recipe 7's regressor sweep over the
# log2FC-embedding equivalent featureset, so this checks whether TabPFN
# v2.5 stays on top on the CheMeleon-embedding featureset or another
# regressor wins here instead.
# TabICL is left out: this featureset is 386 columns (256 CheMeleon PCA +
# 2 readout + 128 descriptor PCA), the same width as recipe 7's excluded
# embed_readout_descriptors row, and it OOMs identically here (HIP OOM
# trying to allocate ~12GiB at seed 0), confirming recipe 7's note that the
# failure tracks training row count/width, not which embedding is used.
set -euo pipefail

SEEDS=(0 1 2 3 4)
CONFIG=configs/freeze1_hd512_clip5.0.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "tabpfn-v2.6 chemeleon_readout_descriptors seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor tabpfn-v2.6 \
        --output-dir "results/tabpfn-v2.6_chemeleon_readout_descriptors_seed${seed}"

    log "tabpfn-v3 chemeleon_readout_descriptors seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor tabpfn-v3 \
        --output-dir "results/tabpfn-v3_chemeleon_readout_descriptors_seed${seed}"

    log "tabfm_n32 chemeleon_readout_descriptors seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor tabfm --tabfm-n-estimators 32 \
        --output-dir "results/tabfm_n32_chemeleon_readout_descriptors_seed${seed}"

    log "lgbm chemeleon_readout_descriptors seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor lgbm \
        --output-dir "results/lgbm_chemeleon_readout_descriptors_seed${seed}"

    log "xgboost chemeleon_readout_descriptors seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor xgboost \
        --output-dir "results/xgboost_chemeleon_readout_descriptors_seed${seed}"
done

log "Recipe 9 (chemeleon_readout_descriptors regressor sweep) seed sweep complete."
