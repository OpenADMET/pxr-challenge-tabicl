#!/usr/bin/env bash
# Recipe 7: "same features, different regressor" sweep re-run on the
# "Readout + descriptors" featureset (no raw embedding), which won panel
# 02's post-seed-sweep comparison over "Embedding + readout + descriptors"
# (the featureset panel 03 originally swept regressors on). TabPFN v2.5
# already has this featureset 5-seed swept as
# results/tabpfn_readout_mordred_pca128_no_embed_seed{0..4} (recipe 5); this
# adds five more regressors so panel 03's regressor comparison can be
# checked against the featureset that actually wins post-seed-sweep.
# tabicl OOMs on this GPU regardless of feature width (its attention scales
# with training row count, not column count, so trimming the embedding
# columns doesn't help); left out here pending separate debugging.
# Reuses the same auxiliary_embedding_cache_seed{0..4}.parquet files recipe 1
# minted, so encoder-init noise matches every other 5-seed row on the page.
set -euo pipefail

SEEDS=(0 1 2 3 4)
CONFIG=configs/tabpfn_small_embed.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "tabpfn-v2.6 readout_mordred_pca128_no_embed seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-embedding \
        --descriptor-source mordred --pca-components 128 --regressor tabpfn-v2.6 \
        --output-dir "results/tabpfn-v2.6_readout_mordred_pca128_no_embed_seed${seed}"

    log "tabpfn-v3 readout_mordred_pca128_no_embed seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-embedding \
        --descriptor-source mordred --pca-components 128 --regressor tabpfn-v3 \
        --output-dir "results/tabpfn-v3_readout_mordred_pca128_no_embed_seed${seed}"

    log "tabfm_n32 readout_mordred_pca128_no_embed seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-embedding \
        --descriptor-source mordred --pca-components 128 --regressor tabfm --tabfm-n-estimators 32 \
        --output-dir "results/tabfm_n32_readout_mordred_pca128_no_embed_seed${seed}"

    log "lgbm readout_mordred_pca128_no_embed seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-embedding \
        --descriptor-source mordred --pca-components 128 --regressor lgbm \
        --output-dir "results/lgbm_readout_mordred_pca128_no_embed_seed${seed}"

    log "xgboost readout_mordred_pca128_no_embed seed=$seed"
    python tabpfn_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --no-embedding \
        --descriptor-source mordred --pca-components 128 --regressor xgboost \
        --output-dir "results/xgboost_readout_mordred_pca128_no_embed_seed${seed}"
done

log "Recipe 7 (readout+descriptors regressor sweep, tabicl excluded) seed sweep complete."
