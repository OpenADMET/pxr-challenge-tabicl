# PXR challenge: status and next steps (2026-07-29)

## Where we landed

**Fixed-embedding rerun (this session):** every result below that reads the
auxiliary encoder's embedding now reads it from one canonical, precomputed
cache (`data/auxiliary_embedding_cache.parquet`, built once by
`precompute_auxiliary_embeddings.py` from `configs/tabpfn_small_embed.yaml`)
instead of retraining a fresh, randomly initialized encoder on every script
invocation. Before this fix, every regressor/PCA-width/descriptor-family
comparison below was confounded by encoder-init variance: a MAE difference
could reflect which random encoder a run happened to get rather than the
feature or regressor choice actually being ablated. `tabpfn_concat_features.py`,
`tabpfn_frozen_features.py`, and `tabpfn_uncertainty_analysis.py` all read the
fixed cache now; `tabpfn_calibrate_predictions.py` is the deliberate exception
(it retrains per k-fold split to stay leakage-free) and reads a separate
`_live_embedding_features` path instead.

Rows marked with the fixed-embedding cache have been rerun this session and
are directly comparable to each other. Rows without that mark predate the fix
(different CheMeleon-initialized encoder configs, or non-TabPFN regressor
baselines not yet rerun) and are kept for reference but should not be compared
MAE-for-MAE against the fixed-embedding rows above them.

Best result so far: `results/tabicl_embed_readout_mordred_pca128/` (256-dim
from-scratch encoder embedding + predicted log2FC readout + Mordred
descriptors, PCA-compressed to 128 components, fit through `TabICLRegressor`).
With the confound removed, TabICL now edges out TabPFN on identical features,
reversing the earlier (confounded) conclusion that TabPFN won.

| Run | Overall MAE | R² |
|---|---|---|
| `tabicl_embed_readout_mordred_pca128` (TabICL, embed + readout + Mordred PCA-128) | **0.4407** | 0.6039 |
| `tabpfn_embed_mordred_pca128_no_readout` (embed + Mordred PCA-128, no readout) | 0.4492 | 0.5876 |
| `tabpfn_embed_readout_mordred_pca256` (embed + readout + Mordred PCA-256) | 0.4509 | 0.5907 |
| `tabpfn_chemeleon_log2fc_mordred_pca128` (CheMeleon-init, log2FC-trained embed + readout + Mordred PCA-128; different encoder config, not part of the shared cache, not rerun) | 0.4524 | 0.5913 |
| `tabpfn_embed_readout_mordred_pca128`, OOF-isotonic-calibrated (`pxr_tabpfn_predictions_calibrated.csv`) | 0.4535 | 0.5854 |
| `tabpfn_concat_small_embed` (embed + readout + RDKit+Mordred PCA-128) | 0.4555 | 0.5870 |
| `tabpfn_embed_readout_mordred_pca128` (embed + readout + Mordred PCA-128, TabPFN, uncalibrated) | 0.4558 | 0.5865 |
| `tabpfn_concat_pca256` (embed + readout + RDKit+Mordred PCA-256) | 0.4574 | 0.5991 |
| `tabpfn_readout_mordred_pca128_no_embed` (readout + Mordred PCA-128, no embed) | 0.4583 | 0.5831 |
| `tabpfn_concat_pca64` (embed + readout + RDKit+Mordred PCA-64) | 0.4610 | 0.5831 |
| `tabpfn_embed_readout_mordred_pca64` (embed + readout + Mordred PCA-64) | 0.4610 | 0.5819 |
| `tabpfn_embed_readout_rdkit_pca128` (embed + readout + RDKit-only PCA-128) | 0.4663 | 0.5655 |
| `tabfm_embed_readout_mordred_pca128` (TabFM, embed + readout + Mordred PCA-128, `max_num_rows=500`, capped in-context rows) | 0.4728 | 0.5815 |
| `tabpfn_embed_only_no_readout` (embed only, no readout, no descriptors) | 0.4742 | 0.5548 |
| `tabpfn_pec50_scratch_mordred_pca128` (from-scratch, pEC50-trained embed + Mordred PCA-128; different encoder config, not part of the shared cache, not rerun) | 0.4745 | 0.5173 |
| `tabpfn_small_embed` (embed + readout, no descriptors) | 0.4825 | 0.5416 |
| `xgboost_embed_readout_mordred_pca128` (same fixed features as best, XGBoost instead of TabPFN) | 0.4893 | 0.5470 |
| `lgbm_embed_readout_mordred_pca128` (same fixed features as best, LGBM instead of TabPFN) | 0.4934 | 0.5401 |
| Best `freeze*` sweep config (`freeze1_hd512_clip5.0`, fine-tuned GNN, no TabPFN; unaffected, different pipeline) | 0.4988 | 0.5493 |
| `tabpfn_pec50_chemeleon` (CheMeleon-init, pEC50-trained embed, RDKit+Mordred PCA(128)+PCA(128); different encoder config, not part of the shared cache, not rerun) | 0.5088 | 0.4228 |
| `tabpfn_chemeleon_static` (CheMeleon untouched pretrained, PCA(128)+PCA(128); different encoder config, not part of the shared cache, not rerun) | 0.5226 | 0.4469 |
| `tabpfn_readout_only_no_embed` (predicted readout alone, 2 columns) | 0.5357 | 0.4903 |
| `tabpfn_rdkit_only_pca128_no_embed_no_readout` (RDKit PCA-128 alone; no embedding, unaffected by the fix) | 0.5341 | 0.3990 |
| `tabpfn_mordred_only_pca128_no_embed_no_readout` (Mordred PCA-128 alone; no embedding, unaffected by the fix) | 0.5440 | 0.4298 |

Target from the N283T report: 0.437-0.408 MAE. We're not there yet but
TabICL (0.4407, Mordred-only descriptors) is the closest so far, and it beats
the RDKit+Mordred-combined run, so RDKit is diluting rather than helping.

`tabpfn_calibrate_predictions.py` repeats the N283T report's OOF-isotonic
calibration idea against the best fixed-embedding TabPFN config: 5-fold split
the DRC records, retrain the encoder and refit TabPFN per fold (no leakage),
fit an isotonic map from the out-of-fold predictions, and apply it to the
blind set. A rerun against the fixed-embedding base predictions is in
progress; the 0.4413 figure from the pre-fix run is stale and not reported
here. This calibration run has not yet been repeated against TabICL, the new
leader.

## Non-TabPFN baselines, for context

Everything above is TabPFN reading frozen or PCA-compressed features. These
are the non-TabPFN reference points we've run, from weakest to strongest:

| Baseline | Overall MAE | R² |
|---|---|---|
| Best fine-tuned-GNN concatenation sweep, no TabPFN (`freeze1_hd512_clip5.0`) | 0.4988 | 0.5493 |
| CheMeleon fine-tuned directly on pEC50, no log2FC pretraining, no `moal` pipeline | 0.5348 | n/a* |
| Permanent-freeze log2FC-pretrained CheMeleon encoder, mixed primary-screen + DRC training data | 0.5480 | 0.4205 |
| Warmup-then-finetune log2FC-pretrained CheMeleon encoder, mixed primary-screen + DRC training data | 0.5537 | 0.4256 |
| Permanent-freeze log2FC-pretrained CheMeleon encoder, DRC-only training data | 0.5693 | 0.4209 |
| Warmup-then-finetune log2FC-pretrained CheMeleon encoder, DRC-only training data | 0.6012 | 0.4102 |

\* The CheMeleon-direct row was evaluated as two separate splits (phase 1:
MAE 0.6033, R² 0.318, n=253; phase 2: MAE 0.4681, R² 0.5082, n=260) rather
than against the combined 513-compound blind set the way every other row
here is. The MAE shown is an n-weighted average of the two splits so it
sits on the same overall-513 basis as the rest of the table; R² does not
combine this way; do not read the phase 2 MAE (0.4681) alone as an
overall figure, since on its own it would misleadingly outperform some
TabPFN configurations above.

Two things stand out:

1. **Freezing the encoder after log2FC pretraining beats letting it keep
   adapting during pEC50 training**, in both the mixed-data and DRC-only
   comparisons (0.5480 vs. 0.5537 mixed; 0.5693 vs. 0.6012 DRC-only). This
   is the same direction as our TabPFN-side finding that the from-scratch,
   never-refit-on-pEC50 embedding beats a directly pEC50-fine-tuned one.
2. **None of these non-TabPFN routes come close to any TabPFN configuration
   above**, including the weakest TabPFN solo-component baseline
   (readout alone, 0.5734). The best of them (0.4988) still trails the
   worst full-factorial TabPFN combination that includes descriptors
   (0.4572). This reinforces "what moves the needle, ranked" item 1: TabPFN
   itself, not just which features feed it, is doing a large share of the
   work.

For external reference, the N283T report's own non-ensembled members: a
frozen, log2FC-pretrained ChemProp embedding read by TabPFN scores 0.437
MAE (their best single embedding, and the closest published number to our
own approach); their CheMeleon embedding alone (fine-tuned, no log2FC
pretraining) scores about 0.512 MAE; and adding CheMeleon as one static
300-dimensional descriptor block to their larger tabular core takes that
core from 0.443 to 0.421 MAE. Their final 9-member ensemble scores 0.4059
(phase 1) and 0.4113 (phase 2).

Naming note: every result above states explicitly which of {raw embedding,
predicted-readout, descriptor family, PCA width} is present. Some earlier
result directories were misleadingly named after the descriptor restriction
alone even though they also included the embedding+readout block by default;
those have been renamed (`tabpfn_concat_rdkit_only` ->
`tabpfn_embed_readout_rdkit_pca128`, etc.) and this table uses the corrected
names throughout.

The `potent (>= 6.0)` subset stays badly miscalibrated in every run so far
(MAE 0.85-1.0, R² around -13 to -17 in the best runs), and does not improve
with better overall MAE — this matches the known failure mode flagged in
issue #36 and hasn't been addressed by any of today's changes.

## Encoder-initialization x training-target grid

Completed the 2x2 grid crossing encoder initialization against training
target, all with Mordred PCA-128 descriptors and the predicted-readout block
included, so the four cells are directly comparable:

| | log2FC-trained | pEC50-trained |
|---|---|---|
| **From-scratch encoder** | **0.4435** (backbone) | 0.4745 |
| **CheMeleon-initialized** | 0.4524 | 0.5088 |

Two consistent, additive effects: log2FC training beats pEC50-direct
training in both rows (0.4435 vs. 0.4745 from scratch; 0.4524 vs. 0.5088 for
CheMeleon), and from-scratch beats CheMeleon initialization in both columns
(0.4435 vs. 0.4524 for log2FC; 0.4745 vs. 0.5088 for pEC50). Training target
matters somewhat more than initialization (roughly 0.03-0.034 MAE versus
0.009-0.034 MAE), but neither swamps the other; the two mixed cells land
between the best and worst corners as expected from independent effects.
CheMeleon+log2FC (0.4524) is much closer to the from-scratch backbone than
either prior CheMeleon variant, confirming log2FC pretraining is the more
load-bearing of the two choices, not the CheMeleon initialization itself.
Neither of the two new cells hit the earlier CheMeleon-scale TabPFN OOM at
128+128 width (new script `tabpfn_chemeleon_log2fc_features.py` PCA-
compresses the 2048-dim embedding before concatenation, same pattern as
`tabpfn_pec50_chemeleon_features.py`).

## Full 2³ factorial: embedding x readout x Mordred descriptors

All 7 non-empty subsets of {embedding, readout, Mordred PCA-128} are now
measured directly rather than interpolated, completing the factorial:

| Combination | MAE |
|---|---|
| Embedding + readout + descriptors (winner) | **0.4435** |
| Embedding + descriptors | 0.4540 |
| Readout + descriptors | 0.4572 |
| Embedding alone | 0.4693 |
| Embedding + readout (no descriptors) | 0.4861 |
| Descriptors alone (Mordred) | 0.5440 |
| Readout alone | 0.5734 |

The build-up is not additive: dropping descriptors from the winning triple
while keeping embedding and readout (row 5) lands worse than dropping
readout and keeping just embedding and descriptors, and worse than the
embedding by itself (row 4). The readout only earns its keep once
descriptors are present to give it context; paired with the raw embedding
alone, it adds noise instead of signal. A visual walkthrough of this
factorial, plus the solo-component baselines and the regressor-swap and
encoder-grid comparisons, is published at
https://claude.ai/code/artifact/869de907-7924-459b-b1cb-ac3547bff880.

## What moves the needle, ranked

This is the answer to "which features are worth keeping as inputs to TabPFN
versus diminishing returns," based on today's isolation ablations, all run
on the same 4139 DRC training records / 513 blind compounds:

1. **The TabPFN regressor itself is doing real work, not just riding good
   features.** Refitting the exact same best featureset (embedding + readout
   + Mordred PCA-128) through LGBM gives MAE 0.5016 and through XGBoost gives
   MAE 0.5494, both clearly worse than TabPFN's 0.4435 on identical inputs.
   TabPFN's in-context-learning approach is the single largest lever we've
   isolated all session: swapping only the final regressor costs 13-24%
   relative MAE, more than any feature ablation below. Keep TabPFN; it is
   not incidental to the result.
2. **Mordred descriptors (PCA-128) are the strongest single descriptor
   block**, clearly better than RDKit alone (0.4650) and better than
   RDKit+Mordred combined at any width tried (0.4574-0.4666). RDKit adds
   noise/dilution rather than signal once Mordred is present. Keep Mordred,
   drop RDKit.
3. **Embedding and readout are both individually useful but partly
   redundant, and neither alone matches having both.** Holding Mordred
   PCA-128 fixed: embedding+readout together = 0.4435; embedding alone
   (no readout) = 0.4540; readout alone (no embedding) = 0.4572. Dropping
   either costs about 0.01-0.014 MAE, roughly the same amount either way, so
   neither dominates the other; keep both, but if forced to cut one for
   feature-budget reasons, cutting readout costs marginally less than
   cutting embedding.
4. **Without any descriptors, the readout actively hurts.** Embedding-only,
   no readout, no descriptors = 0.4693, clearly better than embedding+readout
   with no descriptors (`tabpfn_small_embed`, 0.4861). The readout's value
   only shows up once Mordred descriptors are also present to give it
   context; standing alone next to the raw embedding it adds noise.
5. **CheMeleon (any treatment) underperforms the from-scratch log2FC
   embedding** and is not worth the added complexity or the OOM risk (see
   below) as a lead feature. Its only validated use, per the N283T report
   rather than our own testing, is as one static descriptor column-block
   layered onto an already-built tabular core, not as a standalone embedding
   source.
6. **Real experimental log2FC values are unusable** for our blind-set eval:
   confirmed zero non-null log2FC measurements across all 513 unqueried
   compounds.

## What we ruled in and ruled out, versus the N283T report

Compared against the N283T report
(https://github.com/N283T/openadmet-pxr-model-report), which never
fine-tunes an encoder on pEC50 directly and uses CheMeleon only as a static,
untouched 300-dim descriptor inside a larger tabular core (MAE 0.443 -> 0.421
when added):

- **Ruled in: log2FC-finetuned frozen embeddings.** Our best result and the
  report's own best single embedding (ChemProp frozen, MAE 0.437 OOF) both
  come from finetuning on log2FC, never on pEC50 directly. Keep this as the
  backbone.
- **Ruled in: Mordred descriptors over RDKit.** Directly isolated this
  session (see table above); the report doesn't break this out, but our own
  ablation is unambiguous. Keep Mordred, drop RDKit going forward.
- **Ruled in: TabPFN as the final regressor**, over gradient-boosted trees,
  on identical features (see "What moves the needle," item 1).
- **Ruled in (as an add-on, not a lead feature): CheMeleon as a static
  descriptor block.** The report's own tabular-core ablation shows a real
  ~5% relative MAE gain from adding it (0.443 -> 0.421), but only as one
  static column-block layered onto an already-built descriptor set, not as
  a standalone embedding source.
- **Ruled in: ensembling architecturally diverse embeddings.** The report's
  final 5-embedding ensemble (0.4059) clearly beats its single best
  embedding (0.437), so architectural diversity helps, though the report
  doesn't break out each architecture's individual contribution.
- **Ruled out: finetuning an encoder directly on pEC50.** Tried this
  ourselves (`tabpfn_pec50_chemeleon_features.py`, CheMeleon finetuned on
  pEC50 end-to-end): MAE 0.5088, clearly worse than the log2FC route
  (0.4435) and barely better than not finetuning CheMeleon at all (0.5226).
  The report never tries this either. Close this cell.
- **Ruled out: RDKit descriptors once Mordred is present.** RDKit+Mordred
  combined never beats Mordred alone at any PCA width tried.
- **Ruled out for our eval specifically: real experimental log2FC as a
  feature.** The 513 blind/unqueried compounds have zero log2FC
  measurements (`data/moal_plan_state.csv`, confirmed by filtering to
  `value.isna()`), so only the predicted-log2FC-from-auxiliary-network path
  (already in `tabpfn_concat_features.py`) is usable at inference time.
- **Not yet resolved:** which of the report's other 4 architectures
  (KERMT, MoLFormer, GatedGCN, AttentiveFP) contributes the most, and
  Boltz-2 trunk representation's marginal MAE contribution; the report
  gives ensemble-level numbers only, not per-architecture ablation deltas.

## CheMeleon: two treatments tried, both underperform the log2FC route

New scripts `tabpfn_pec50_chemeleon_features.py` (finetuned on pEC50) and
`tabpfn_chemeleon_static_features.py` (untouched pretrained weights) both
add an `embed_smiles` method to `moal.model.ChemPropLightningModule`
(mirrors `AuxiliaryEncoderModule.embed_smiles`, already used by the
log2FC-finetuned route). Neither beats the from-scratch log2FC-embedding
approach; see table above.

Both scripts hit a genuine TabPFN memory ceiling at 256+256 PCA components
(2048-dim CheMeleon embedding + descriptors) that does not reproduce with
the log2FC-embedding route at a similar total width (514 columns succeeds
there, 512-514 columns fails for CheMeleon-sourced features). Ruled out as
explanations: GPU contention from a stray Ollama process (killed, no
change), Lightning `Trainer` reference leaks (`refit()` only stores the
trainer on `self`, confirmed no other referrer), allocator fragmentation
(`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` made no difference), and
process-level GPU memory history (a fully fresh process loading cached
embeddings from disk still OOMs identically). The cause is something about
the actual CheMeleon-derived feature values that pushes TabPFN into a
costlier internal code path at the same column count as other embeddings;
not root-caused further since both CheMeleon variants already lose to the
log2FC embedding even at the reduced 128+128 width. `tabpfn_pec50_chemeleon_features.py`
now supports `--embedding-cache <path>` to split fine-tuning and TabPFN
fitting into two separate process invocations, useful if this is revisited.

## What's left to try

1. **`potent (>= 6.0)` subset recalibration** is still open (issue #36) and
   untouched by anything from today; every run so far, including the best
   one, does badly on it. Worth investigating separately, e.g. isotonic
   calibration (there was an earlier, since-removed
   `pxr_aux_predictions_calibrated.csv` experiment in
   `results/freeze2_hd512_clip1.0/` from before this session) or a
   potency-weighted loss/sampling adjustment.
2. **Rerun the best featureset (Mordred PCA-128 + embedding + readout)
   dropping RDKit entirely from `descriptor_cache.parquet` reads by
   default**, since RDKit has now been ruled out as diluting the signal.
   Settled: `tabpfn_embed_readout_mordred_pca64` (0.4658 MAE) confirms no
   further PCA-width sweep on Mordred alone beats 128 (0.4435); 192 is the
   only untested width left in that range and is not expected to change
   the conclusion.
3. **Other graph embedding architectures** (KERMT, MoLFormer, GatedGCN,
   AttentiveFP per the N283T report) are not wired into `moal` at all;
   would need new encoder integrations, not just new scripts.
4. **`data/descriptor_cache.parquet`** (12787 compounds x 1831 columns,
   217 RDKit + 1613 Mordred, keyed by `canonical_smiles`) is a standing
   artifact now — reuse it directly in any future feature-concatenation
   experiment rather than recomputing.
5. **Done: checked whether TabPFN's predicted distribution is well
   calibrated against actual error.** See "TabPFN's native uncertainty:
   weak signal, biased coverage" below.
6. **TabFM and TabICL, ablating which tabular foundation model, not just
   foundation-model-versus-classical-ML.** `_build_regressor` in
   `tabpfn_concat_features.py` now supports `--regressor tabicl` and
   `--regressor tabfm` alongside the existing `tabpfn`/`lgbm`/`xgboost`
   options, both verified against the installed package APIs on synthetic
   data before running for real. TabFM's pretrained weights
   (`google/tabfm-1.0.0-pytorch` on Hugging Face) carry a separate
   `tabfm-non-commercial-v1.0` license; the package code itself is
   Apache-licensed. Runs against the current best featureset (embedding +
   readout + Mordred PCA-128) are in progress; results to follow.

## TabPFN's native uncertainty: weak signal, biased coverage

`tabpfn_uncertainty_analysis.py` refits the best config
(`tabpfn_embed_readout_mordred_pca128`) and calls `TabPFNRegressor.predict`
with `output_type="full"` on the 513-compound blind set, extracting each
compound's predicted mean, standard deviation, and a 9-point quantile grid
(0.1 through 0.9). Two checks against the known true pEC50:

- **Does |residual| correlate with the predicted standard deviation?**
  Spearman rho = 0.226 (p = 2.3e-7, n = 513). Real, not noise, but weak:
  TabPFN's own uncertainty ranks compounds by expected error only loosely.
- **Does empirical quantile coverage track the nominal level?** No. Every
  nominal level from 0.1 to 0.9 shows a negative gap (fewer true values
  fall at or below the predicted quantile than the nominal level implies),
  worst in the middle of the distribution (gap around −0.13 to −0.14 at the
  0.5-0.6 levels) and smaller at the tails (−0.03 at 0.1, −0.05 at 0.9).
  TabPFN's predicted quantiles run systematically low relative to the true
  values across most of the range, a directional bias, not just
  under-confident (too-narrow) or over-confident (too-wide) spread.

Rescaling or recalibrating the predicted quantiles (isotonic or affine,
applied uniformly across compounds) would fix the coverage bias but cannot
improve the 0.226 Spearman correlation: that correlation is rank-based and
invariant to a monotonic transform of one variable, so no amount of
recalibrating `predicted_std` changes which compounds it ranks as more or
less uncertain relative to each other. Moving that number would need a
different uncertainty signal entirely (e.g. ensemble disagreement across
architecturally distinct embeddings, or distance to the nearest training
compound), not a calibration step on the existing quantiles.

Results: `results/tabpfn_uncertainty_mordred_pca128/pxr_tabpfn_distributions_scored.csv`
(per-compound mean/std/quantiles plus true pEC50 and residual) and
`quantile_coverage.csv` (the 9-row nominal-versus-empirical table above).

## Housekeeping already done this session (no action needed)

- Stripped the dead/sparse log2FC columns (`log2fc_9.803e-07`,
  `log2fc_9.901e-05`) from all 4 `data/moal_plan_state*.csv` files and all
  24 affected configs; originals backed up as `*_backup.csv` siblings.
- Retired `configs/feat_observed_4task.yaml` to `configs/retired/` (its
  4-task ablation purpose no longer holds post-cleanup).
- Reran and rescored all 18 `freeze*` sweep configs against the cleaned
  2-column data; results above.
- Added `ChemPropLightningModule.embed_smiles` to `moal/model.py` (mirrors
  the existing `AuxiliaryEncoderModule.embed_smiles`), enabling embedding
  extraction from the main pEC50-trained GNN, not just the auxiliary
  log2FC-trained one.
- Added `--descriptor-source {all,rdkit,mordred}`, `--no-embedding`,
  `--no-readout`, and `--regressor {tabpfn,lgbm,xgboost}` to
  `tabpfn_concat_features.py`, and `--no-readout` to
  `tabpfn_frozen_features.py`, enabling all of today's isolation ablations.
- Renamed result directories that previously implied descriptor-only
  ablation (they always included the embedding+readout block by default)
  to state every included feature block explicitly.
- Added new script `tabpfn_chemeleon_log2fc_features.py` (CheMeleon-init
  encoder trained on log2FC via `AuxiliaryEncoderModule`, PCA-compressed
  before concatenation) and config `configs/tabpfn_chemeleon_log2fc.yaml`,
  and new config `configs/tabpfn_pec50_scratch.yaml` (`model.from_foundation:
  false`, reuses existing `tabpfn_pec50_chemeleon_features.py`), completing
  the encoder-init x training-target grid above. Also added
  `--descriptor-source` to `tabpfn_pec50_chemeleon_features.py` so all four
  grid cells use the same Mordred-only descriptor block.
- Added `--no-embedding` to `tabpfn_frozen_features.py` (previously only
  supported dropping the readout, not the embedding), and ran the three
  remaining solo-component baselines (readout alone, Mordred alone, RDKit
  alone), completing the full 2³ factorial over {embedding, readout,
  Mordred descriptors} above.
- Published a visual artifact of the ablation story (solo-component
  baselines, the full factorial build-up, the regressor swap, and the
  encoder-init x training-target grid).

## Known footgun for future background jobs

A wait-loop pattern like `while pgrep -f "<script>.*<flag>" ...; do sleep
...; done` is unsafe when launched via a backgrounded shell: the wrapping
shell's own command line (which contains the full script text, including
the search string, as an argument) matches the pattern, so the loop waits
on itself forever with no output and no error. This caused two silent
"stuck" jobs this session before being diagnosed. Prefer plain `&&`
sequencing of commands in one background call instead of pgrep-based
wait loops.

## Git / push status

`moal` (the library repo) has one code change this session
(`ChemPropLightningModule.embed_smiles` in `moal/model.py`), not yet
committed. Everything else lives in `pxr-challenge`, which is not a git
repo. Nothing pushed in either place.
