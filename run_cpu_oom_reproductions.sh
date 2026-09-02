#!/usr/bin/env bash
# Reproduces, on CPU, the two GPU out-of-memory (OOM) cases the blog post
# cites as tabular-foundation-model limitations, to check whether CPU has
# headroom the GPU used for these runs didn't. Each case is a single
# seed-0 run, not a 5-seed sweep: the question here is whether a fit
# completes or OOMs, not its accuracy. Both cases use the same 386-column
# CheMeleon embedding+readout+descriptors featureset, PCA-128 on the
# descriptor block exactly as every other analysis in this repo does it;
# nothing here runs on raw, uncompressed descriptors
#
# case 1: TabICL on the 386-column featureset; manifest.py documents TabICL
#   alone OOMing on this width, which is why its own headline result is
#   instead read from the narrower embedding+readout-only featureset
# case 2: TabFM on the full ~4139-row training context (no max_num_rows
#   cap), same featureset. Ensemble size is cut to 4 (from the default 32)
#   to bound CPU wall-clock: whether a single member's full-context
#   attention pass OOMs doesn't depend on ensemble size
set -uo pipefail

CHEMELEON_CONFIG=configs/freeze1_hd512_clip5.0.yaml
CACHE_SEED0=data/auxiliary_embedding_cache_seed0.parquet

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_case() {
    local label="$1"
    shift
    log "=== $label ==="
    if "$@"; then
        log "=== $label: completed (no OOM on CPU) ==="
    else
        log "=== $label: FAILED (exit $?) ==="
    fi
}

run_case "case 1: TabICL, 386-column CheMeleon readout+descriptors, CPU" \
    python tabpfn_chemeleon_concat_features.py "$CHEMELEON_CONFIG" \
        --embedding-cache "$CACHE_SEED0" --seed 0 \
        --regressor tabicl --device cpu \
        --output-dir results/cpu_oom_repro_tabicl_chemeleon_readout_descriptors

run_case "case 2: TabFM, full training context (no row cap), CPU" \
    python tabpfn_chemeleon_concat_features.py "$CHEMELEON_CONFIG" \
        --embedding-cache "$CACHE_SEED0" --seed 0 \
        --regressor tabfm --tabfm-max-rows 0 --tabfm-n-estimators 4 --device cpu \
        --output-dir results/cpu_oom_repro_tabfm_full_rows

log "CPU OOM reproduction run complete."
