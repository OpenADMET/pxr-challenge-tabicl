#!/usr/bin/env bash
# Recipe 10: TabICL on the CheMeleon "embedding + readout" featureset (258
# columns: 256 CheMeleon embedding PCA + 2 borrowed from-scratch readout, no
# descriptors), which recipe 8 found nearly tied with the full
# embedding+readout+descriptors combo (0.4444 vs 0.4437 mean MAE). TabICL
# OOMs on both 386-column CheMeleon combos (recipe 9) at the batch_size=2
# tabpfn_concat_features.py otherwise defaults to for TabICL; testing here
# showed the OOM is peak-memory-direction-sensitive per featureset, not a
# fixed batch_size=2-is-safer rule: batch_size=2 OOMs harder on this
# featureset than the library default (8) does, so this sweep explicitly
# passes --tabicl-batch-size unset (library default) rather than reusing
# tabpfn_concat_features.py's batch_size=2 default for TabICL.
set -euo pipefail

SEEDS=(0 1 2 3 4)
CONFIG=configs/freeze1_hd512_clip5.0.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/auxiliary_embedding_cache_seed${seed}.parquet"

    log "tabicl chemeleon_readout_only seed=$seed"
    python tabpfn_chemeleon_concat_features.py "$CONFIG" \
        --embedding-cache "$cache" --seed "$seed" --regressor tabicl --no-descriptors \
        --output-dir "results/tabicl_chemeleon_readout_only_seed${seed}"
done

log "Recipe 10 (TabICL on CheMeleon embedding+readout) seed sweep complete."
