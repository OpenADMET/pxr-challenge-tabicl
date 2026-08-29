#!/usr/bin/env bash
# Recipe 3 only (CheMeleon pEC50-direct), rerun after an OOM mid-sweep.
#
# expandable_segments isn't supported on this ROCm platform (confirmed: the
# retry with it set hit the identical OOM), so this isn't allocator
# fragmentation within one process. The script's own --embedding-cache
# comment already documents the real cause: running fine-tuning and TabPFN's
# predict in the same HIP context starves TabPFN's between-items attention
# of a large contiguous block, even after explicit del/gc.collect/empty_cache.
# So each seed runs as two separate process invocations: pass 1 fine-tunes
# and writes the raw embedding to an .npz cache and exits before touching
# TabPFN; pass 2 (fresh process) loads that cache and runs TabPFN alone.
#
# Also dropped from the 256/256 defaults to 128/128 PCA components (256
# total features instead of 512) after repeated OOMs at 512 that two
# targeted fixes (allocator flag, context-splitting) didn't resolve.
set -euo pipefail

SEEDS=(0 1 2 3 4)
log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in "${SEEDS[@]}"; do
    cache="data/pec50_chemeleon_embed_cache_seed${seed}.npz"

    log "pec50_chemeleon: seed=$seed, pass 1 (fine-tune, write cache)"
    python tabpfn_pec50_chemeleon_features.py configs/freeze1_hd512_clip5.0.yaml \
        --seed "$seed" \
        --embedding-cache "$cache" \
        --embedding-pca-components 128 \
        --descriptor-pca-components 128 \
        --output-dir "results/tabpfn_pec50_chemeleon_seed${seed}"

    log "pec50_chemeleon: seed=$seed, pass 2 (load cache, fit TabPFN)"
    python tabpfn_pec50_chemeleon_features.py configs/freeze1_hd512_clip5.0.yaml \
        --seed "$seed" \
        --embedding-cache "$cache" \
        --embedding-pca-components 128 \
        --descriptor-pca-components 128 \
        --output-dir "results/tabpfn_pec50_chemeleon_seed${seed}"
done

log "Recipe 3 seed sweep complete."
