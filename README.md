# PXR challenge: are tabular foundation models all you need?

A rebuild of the OpenADMET PXR (pregnane X receptor) blind-challenge analysis,
run on the challenge's own evaluation split: train on the dose-response
training set plus phase 1, score on phase 2 alone.

The earlier analysis scored on phase 1 and phase 2 pooled, which is not the
split the challenge leaderboard used and is not comparable to the phase-2
numbers participants reported. This rebuild corrects that, emits run
provenance forward instead of reconstructing it after the fact, and is laid
out so a reader can clone and rerun the whole thing.

See [`PLAN.md`](PLAN.md) for the tracked rebuild checklist.

## Reproduce

Filled in as the runners land. Target flow: `uv sync`, then the numbered
scripts under `run/` in order, pausing at each figure's decision gate to
select the configuration that advances.
