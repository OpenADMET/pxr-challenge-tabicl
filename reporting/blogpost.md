*PXR induction pEC<sub>50</sub> · feature ablation study*

# Are tabular foundation models all you need?

Lessons from the PXR Blind Challenge.

n = 513 blind compounds &middot; 4139 dose-response training records &middot; lower MAE (mean absolute error) is better &middot; same train/blind split throughout

---

**00 — before tabular foundation models**

## Where the non-tabular-foundation-model routes land, for scale

Every reference point below reads features through something other than a tabular foundation model: a directly fine-tuned CheMeleon encoder, or a graph neural network (GNN) whose body is initialized from a log<sub>2</sub>FC-pretraining checkpoint (E4, Buterez et al. 2024) rather than staying a tabular foundation model downstream. The four log<sub>2</sub>FC rows below cross frozen vs. fine-tuned body against dose-response-only vs. dose-response-plus-primary-screen training data; none of them uses the separate auxiliary-encoder concatenation architecture (E3) that a fifth config in this repo implements, since that run was never completed. None of these routes approaches even the weakest tabular-foundation-model configuration shown in the panels that follow.

*(paste `reporting/figures/figure-00.html` here as an HTML card)*

> "Concatenation architecture" is the best-of-18 winner (freeze_epochs=2, 512-dim hidden, no gradient clipping) from a sweep over freeze_epochs × hidden width × gradient clip on the E3 auxiliary-encoder-plus-concatenation design (Buterez et al. 2024): a 5-seed mean of 0.5176 (range 0.4919-0.5339), now the floor for this whole panel. Every other log<sub>2</sub>FC-encoder row below it is CheMeleon-initialized but skips that auxiliary concatenation step entirely (E4: the log<sub>2</sub>FC-pretrained checkpoint is reused as the main model's init, nothing more). Among those, the "CheMeleon baseline" skips log<sub>2</sub>FC pretraining too and fits straight to pEC<sub>50</sub>: a 5-seed mean of 0.5215 (range 0.5058-0.5407), beating the fine-tuned-dose-response-only row (0.5215 vs. 0.5339) but not the concatenation architecture. On the primary-screen + dose-response data, freezing the encoder beats letting it keep adapting (0.5464 vs. 0.5573). On dose-response-only data, the direction is opposite: fine-tuning beats freezing (0.5339 vs. 0.5849). Freezing the encoder is not consistently better; it depends on the training-data regime. Even so, every row on this panel trails the weakest tabular-foundation-model combination that includes descriptors by at least 0.0566 MAE (0.5176 vs. 0.4610, panel 2). "Dose-response-only" and "primary-screen + dose-response" above describe only what the main model is directly supervised on: the frozen, CheMeleon-initialized log<sub>2</sub>FC encoder is always pretrained on primary-screen data, and its 2-column predicted log<sub>2</sub>FC readout reaches the main model in both variants, so primary-screen information is never fully excluded.

---

**01 — solo components**

## Does each ingredient carry signal on its own?

Four candidate feature blocks, each fit through a tabular foundation model completely alone, with nothing else concatenated alongside it. The N283T report benchmarks its own candidate embedding architectures the same way, reading each one alone through a tabular foundation model before building any ensemble or concatenation. We follow that practice here across our own candidate blocks (the raw embedding, its predicted log<sub>2</sub>FC readout, RDKit descriptors, and Mordred descriptors), checking which one carries signal on its own before testing any combination. Every row here is read by TabPFN v2.5, the regressor used throughout this comparison, except "our best overall": that row is read by TabICL 2.1.1 instead, the winner of the separate regressor sweep in section 03, so it isn't on equal footing with the other rows in this panel. "CheMeleon embedding" below is the off-the-shelf embedding alone, nothing else concatenated; it is not the same experiment as any CheMeleon-based row in sections 02 or 03, which always add the log<sub>2</sub>FC readout block (and, in section 03, descriptors too) on top of this embedding.

*(paste `reporting/figures/figure-01.html` here as an HTML card)*

> The raw embedding is the clearest single-ingredient win (0.4780). RDKit descriptors, not the predicted log<sub>2</sub>FC readout, is the strongest descriptor-only block (0.5312 vs. 0.5410), and Mordred descriptors alone is the worst row on this panel (0.5784, range 0.5751-0.5804), worse even than the best GNN baseline floor (0.5176). Mordred's PCA (principal component analysis) fit is unstable enough on unscaled descriptors (see panel 04) that its 5-seed range alone spans roughly 0.005 MAE, wider than the 0.0098 MAE gap separating the RDKit-descriptor and readout rows on this panel. The pretrained CheMeleon embedding, run through TabPFN with no fine-tuning at all (0.5002), still sits worse than our from-scratch, log<sub>2</sub>FC-trained embedding (0.4780) but better than every descriptor-only row, so the encoder's log<sub>2</sub>FC pretraining is still adding signal on top of what CheMeleon's own general-purpose embedding already carries.

---

**02 — the path to the best performer**

## All seven combinations of embedding, readout, and descriptors

Every non-empty subset of the three ingredients was measured directly, holding the descriptor family fixed at Mordred (the stronger of the two 2D-descriptor blocks), and repeated with both the from-scratch, log<sub>2</sub>FC-trained embedding and CheMeleon's untouched, off-the-shelf pretrained embedding (PCA-256, disambiguated in each row's sub-label), so all seven non-empty subsets are covered under each embedding source. No path was interpolated or assumed. Every subset row here is read by TabPFN v2.5; "our best overall" is shown above the dashed line for scale, but it does not read any combination shown on this panel: TabICL can't fit the 386-column combination this panel's own sweep crowns best (CheMeleon embedding + readout + descriptors, 0.4437), so its own result instead comes from a leaner, near-tied featureset (CheMeleon embedding + readout, 0.4356), which beats this panel's own winner in turn. Section 03 reruns this panel's own winning combination (CheMeleon embedding + readout + descriptors) through five other regressors, plus TabICL on that leaner featureset as a flagged exception.

*(paste `reporting/figures/figure-02.html` here as an HTML card)*

> Embedding + readout + descriptors, with CheMeleon's off-the-shelf embedding, is the best subset read by TabPFN (0.4437), edging out the same three-ingredient combination built on the from-scratch log<sub>2</sub>FC embedding (0.4574) and the no-embedding readout + descriptors row (0.4531); swapping in TabICL (0.4356) beats every TabPFN-read subset on this panel, though it reads a different, leaner combination rather than this panel's own winner (see panel 03). The top rows overlap once ranges are drawn (embedding + readout + descriptors [CheMeleon] 0.4381-0.4476, readout + descriptors 0.4485-0.4590, embedding + readout + descriptors [log<sub>2</sub>FC] 0.4460-0.4722), so "CheMeleon wins" is a soft claim: the gap sits inside seed-to-seed noise. Descriptors alone (Mordred) is the worst row here (0.5784), worse than the fine-tuned GNN floor, because that recipe's PCA fit has a 5-seed range of roughly 0.005 MAE on unscaled descriptors (panel 04). Once the from-scratch embedding is in the mix, the readout looks close to redundant with what the embedding already carries (0.4574 with it vs. 0.4610 without), a gap well within seed noise; for CheMeleon's off-the-shelf embedding, dropping the readout costs far more than dropping descriptors does (0.4444 embedding + readout vs. 0.5290 embedding + descriptors, vs. 0.4437 with all three).

---

**03 — same features, different regressor**

## Which tabular foundation model reads the featureset best?

Not a feature ablation: every row here holds a featureset fixed and swaps only the final tabular model reading it. That featureset is panel 02's own combo-sweep winner, CheMeleon embedding + readout + descriptors (386 columns), for every row except one: TabICL OOMs (runs out of memory) at that width, so it alone is read from a leaner, near-tied featureset instead (CheMeleon embedding + readout, no descriptors, 0.4356 vs. 0.4437 under TabPFN v2.5 on the full combination, a gap well inside seed noise). TabPFN v2.5, TabPFN v2.6, TabPFN v3, TabFM, LightGBM, and XGBoost all see identical 386-column inputs, the same data backing panel 02's green row. TabICL still comes out ahead even from its leaner featureset, which is why sections 00-02 read their "our best overall" row through TabICL + CheMeleon-embedding-and-readout. It's shown here in-sweep rather than repeated above the dashed line, since it would otherwise be an exact duplicate of the "our best overall" row. Best GNN baseline and Best single-ingredient are repeated from sections 01/02 for scale.

*(paste `reporting/figures/figure-03.html` here as an HTML card)*

> TabICL v2.1.1 is the best tabular foundation model overall, a 5-seed mean of 0.4356, but it alone reads a leaner featureset here (no descriptors) because it OOMs on the 386-column combination every other row uses. Among the rows that do share that combination, TabPFN v2.5 (0.4437) and TabPFN v3 (0.4436) are a statistical dead heat, a 0.0001 gap that is noise, not a ranking; TabPFN v2.6 trails both (0.4571), and TabFM and the two gradient-boosted baselines (0.4795, 0.4987, 0.5311) trail well behind all three TabPFN releases. TabFM v1.0.0 is capped at <code>max_num_rows=500</code>: the full 4,139-row training set OOMs its attention pass on this GPU (graphics processing unit), so its in-context set is subsampled per ensemble member while TabPFN and TabICL see all 4,139 rows, making its 0.4795 not fully apples-to-apples against the other regressors here. TabICL's own mean (0.4356) still beats every one of these, including on the leaner featureset that gave it its OOM workaround, which is the basis for reading "our best overall" through TabICL rather than through TabPFN v2.5 or v3.

---

**04 — the PCA compression sweep**

## Why the descriptor block is PCA-compressed at all

Raw Mordred descriptors run to about 1,613 columns and RDKit adds another 217; CheMeleon's raw embedding is 2,048-dimensional. Feeding any of these to TabPFN uncompressed OOMs its between-items attention pass, even with <code>memory_saving_mode</code> and <code>fit_mode="low_memory"</code> set, because that pass scales with feature count as well as row count. The runs below solve the embedding side of that by using a small, from-scratch 256-dimensional encoder instead of CheMeleon's native 2,048. PCA compression is applied only to the descriptor block, fit on the training split only and kept well under TabPFN's officially supported 2,000-feature hard limit, with roughly 500 total features as the practical headroom on this GPU. The embedding and predicted-readout columns themselves stay raw and uncompressed throughout every sweep here. PCA here runs directly on median-imputed, unstandardized descriptor values (no per-column scaling), so a handful of high-magnitude raw descriptors dominate the covariance: every width tested, on both descriptor sources, already captures 99.998-100.000% of that raw variance. That number describes how much of the unscaled descriptor covariance survives compression, not how much predictive signal does, which is why MAE keeps changing with PCA width even though explained variance is already saturated at 64 components.

*(paste `reporting/figures/figure-04.html` here as an HTML card)*

> 128 components is the lowest mean in both descriptor sources (Mordred only: 0.4574, range 0.4460-0.4722; RDKit + Mordred: 0.4581, range 0.4537-0.4624), with 64 worst and 256 in between in both cases, not a monotonic trend, and the three within-source ranges overlap enough that "128 is best" is a soft claim in either. Mordred-only 128 and RDKit + Mordred 128 differ by 0.0007, far inside both rows' seed-to-seed ranges: adding RDKit descriptors on top of Mordred buys no measurable accuracy here. This page uses Mordred alone as the descriptor block for every concatenation panel (00-02), a simplicity argument (one fewer descriptor source, no RDKit dependency) rather than a demonstrated accuracy win over the combined block. The embedding and readout blocks are identical, fixed, and uncompressed across every row here; only the descriptor source and PCA width change.

---

**05 — out-of-fold isotonic calibration**

## Repeating the N283T report's post-hoc calibration step

Per the N283T report, fitting an isotonic map on out-of-fold predictions and applying it to the blind set closed a similar-sized gap for a subset of their architecture ablations (~0.441 → ~0.408 MAE). This repeats that idea against our current best config (CheMeleon embedding + log<sub>2</sub>FC readout, TabICL): for each of the 5 seeds, 5-fold split the dose-response training records, retrain the log<sub>2</sub>FC readout encoder per fold (the CheMeleon embedding itself is frozen and off-the-shelf, so it's extracted once up front with no leakage risk) so no compound informs both a fold's own training and its held-out prediction, fit isotonic regression on the pooled out-of-fold predictions, then apply that map to the blind set. The out-of-fold MAE itself isn't shown here: it's computed on raw per-record dose-response values, not per-compound pEC<sub>50</sub> like every other number on this page, so it isn't comparable to the blind-set MAE below and would misrepresent the calibration step's effect if plotted alongside it.

*(paste `reporting/figures/figure-05.html` here as an HTML card)*

> Calibration makes this config worse (0.4356 → 0.4507 mean MAE across the same 5 seeds), the opposite of the N283T report's ~0.441 → ~0.408 jump for their ablations. The <code>potent (≥ 6.0)</code> subset gets worse too (MAE 0.6882 → 0.7402 mean), and stays badly miscalibrated either way (R² around −9 to −9.4). TabICL's own predictions on this featureset are already close to well-calibrated, so fitting an isotonic map on a 5-fold OOF split adds fold-to-fold noise (a different, per-fold-retrained readout encoder each time) without a systematic bias left to correct. Reserve this step for configs that show a demonstrated calibration gap, not apply it by default.

---

**06 — is TabPFN's uncertainty trustworthy?**

## Checking the predicted distribution against actual error

TabPFN doesn't just return a point estimate: <code>output_type="full"</code> exposes a per-compound predicted distribution (a discretized Riemann distribution, not a Gaussian), with a mean, a standard deviation, and a quantile grid. Refitting the best config and predicting on the 513 blind compounds this way lets us check whether that spread tracks error, or is decoration. The left chart plots each compound's absolute residual against its predicted standard deviation, testing whether a higher predicted std means a bigger typical error. The right chart is a reliability diagram: it plots empirical coverage against nominal quantile level across all 9 predicted quantiles, testing whether, say, the predicted 0.4 quantile actually has 40% of true values at or below it.

*(paste `reporting/figures/figure-06.html` here as an HTML card)*

> <strong>Spearman ρ(|residual|, predicted std) = 0.2472</strong> (n = 513): a weak positive correlation. Coverage is worse: the curve sits below the dashed diagonal at every one of the 9 quantile levels, worst around the median (gap around −0.10 to −0.11 at the 0.4-0.6 levels), so TabPFN's predicted quantiles run systematically low rather than just too narrow or too wide. Recalibrating the quantiles would fix that directional bias but can't raise the 0.2472 correlation, since Spearman rho is rank-based and invariant to any monotonic rescaling of <code>predicted_std</code>.

---

**07 — limitations**

## Where this comparison could be misleading us

Below are the places where the setup itself, not the model, could be driving the headline number, ordered by how much fixing each one would move it.

**Our encoder's hyperparameters were never tuned against this objective**

The N283T report's ChemProp encoder, at the same 256-dim embedding width we use, had its hyperparameters chosen with Optuna directly against downstream pEC<sub>50</sub> OOF MAE. Ours were carried over from earlier stages of this project and never searched end-to-end against the metric we're comparing on. Given matched embedding width, this is the leading candidate for the remaining gap to their single-pipeline target (0.437): ~0.020 MAE for our from-scratch/log<sub>2</sub>FC TabPFN backbone, a 5-seed mean of 0.4574 (that same recipe is also our best PCA-width variant, 128 components, so these are two views of one recipe, not two separate results), not anything downstream of the embedding.

**The potent (≥ 6.0) subset is small, unstable, and unaddressed**

Only 31 of 513 blind compounds fall in this subset, and every configuration we've tried, including calibration, scores badly on it (MAE 0.8-1.0, R² around −12 to −17). At n=31, its MAE is sensitive to a handful of outliers and could overstate or understate a weakness; either way it's a known open problem (issue #36) that nothing here has fixed.

**PCA compression is a memory workaround, not a modeling choice**

Every descriptor block on this page is compressed because the raw columns OOM TabPFN's attention pass on this GPU, not because compression was found to help. Some signal in the discarded variance could be recoverable on hardware with more memory headroom, a chunked/tiled attention implementation, or a regressor without TabPFN's feature-count ceiling.

**Only one embedding architecture was tried, though it's the best one on its own**

Every tabular-foundation-model result here reads a ChemProp-style encoder's embedding. In the N283T report's own single-embedding comparison, ChemProp scored best (0.437) ahead of KERMT (0.448), GatedGCN (0.474), MoLFormer (0.475), and AttentiveFP (0.484), so swapping architectures alone is unlikely to close much of the remaining gap. Their jump to ~0.408 came from ensembling all of these (plus two Boltz-2 structural members) together, which is out of scope here by design; this limitation is more a note on undertested breadth than an expected source of headline gains.

---

*data: results/tabicl_embed_readout_mordred_pca128 (winner) · tabpfn_embed_readout_mordred_pca128 · tabpfn_embed_only_no_readout · tabpfn_readout_only_no_embed · tabpfn_mordred_only_pca128_no_embed_no_readout · tabpfn_rdkit_only_pca128_no_embed_no_readout · tabpfn_embed_mordred_pca128_no_readout · tabpfn_readout_mordred_pca128_no_embed · tabpfn_small_embed · lgbm_embed_readout_mordred_pca128 · xgboost_embed_readout_mordred_pca128 · tabfm_embed_readout_mordred_pca128 · tabpfn_uncertainty_mordred_pca128 · tabpfn_chemeleon_log2fc_mordred_pca128 · tabpfn_pec50_scratch_mordred_pca128 · tabpfn_pec50_chemeleon · freeze1_hd512_clip5.0 · e4_frozen · e4_finetune · e4_frozen_drc_only · e4_finetune_drc_only · pxr_baseline_predictions_phase1 · pxr_baseline_predictions_phase2 — PXR challenge, pxr-challenge repo*
