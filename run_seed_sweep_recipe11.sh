#!/usr/bin/env bash
# Recipe 11: "same features, different regressor" sweep on
# chemeleon_readout_only (258 columns: 256 CheMeleon embedding PCA + 2
# borrowed from-scratch readout, no descriptors), the featureset recipe 10
# found TabICL winning on decisively (0.4356 mean MAE, beating every other
# combo tried so far including the richer CheMeleon+descriptors rows and
# the log2FC-embedding our_best_overall). This fills out panel 03's
# regressor comparison for that featureset, mirroring recipes 7/9.
# TabICL itself is already covered by recipe 10 (--tabicl-batch-size unset,
# the library default, which this featureset needs; batch_size=2 OOMs
# harder here, see tabpfn_concat_features.py's _build_regressor docstring).
set -euo pipefail

SEEDS=(0 1 2 3 4)
CONFIG=configs/freeze1_hd512_clip5.0.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "tabpfn-v2.6 chemeleon_readout_only seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor tabpfn-v2.6 --no-descriptors \
        --output-dir "results/tabpfn-v2.6_chemeleon_readout_only_seed${seed}"

    log "tabpfn-v3 chemeleon_readout_only seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor tabpfn-v3 --no-descriptors \
        --output-dir "results/tabpfn-v3_chemeleon_readout_only_seed${seed}"

    log "tabfm_n32 chemeleon_readout_only seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor tabfm --tabfm-n-estimators 32 --no-descriptors \
        --output-dir "results/tabfm_n32_chemeleon_readout_only_seed${seed}"

    log "lgbm chemeleon_readout_only seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor lgbm --no-descriptors \
        --output-dir "results/lgbm_chemeleon_readout_only_seed${seed}"

    log "xgboost chemeleon_readout_only seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor xgboost --no-descriptors \
        --output-dir "results/xgboost_chemeleon_readout_only_seed${seed}"
done

log "Recipe 11 (chemeleon_readout_only regressor sweep) seed sweep complete."
