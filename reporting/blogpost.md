*PXR induction pEC<sub>50</sub> · feature ablation study*

# Are tabular foundation models all you need?

What the PXR blind challenge taught us about featurization, and about one paper that helped, but not as much as we initially thought.

**The OpenADMET PXR (pregnane X receptor) blind challenge has closed, and the top of the leaderboard was crowded. At the Phase 1 handover, a paired-bootstrap analysis could not separate the top ten entries. Our own model catalog was not among them, so we did what any curious team does after a challenge: we read the reports of the participants who beat us, looking for an idea worth implementing.**

---

We ran the challenge as the OMSF (Open Molecular Software Foundation) contingent of OpenADMET, from April through July 2026, providing every team the same PXR induction assay data, holding out a 513-compound analog set, and scoring predicted pEC<sub>50</sub> against the unblinded truth. The CheMeleon baselines implemented in `openadmet-models` land around 0.51 to 0.52 MAE (mean absolute error) on that blind set (panel 00), below the roughly 0.41 the leading entries reached.

We were not solely prioritizing the single lowest score. A blind challenge rewards whatever squeezes out the last hundredth of an MAE, and the recipes near the top tend to be large ensembles tuned over many submissions. We wanted something we could fold into `openadmet-models` as a reproducible, YAML-defined pipeline: a method disclosed in enough detail to rebuild, and simple enough to maintain after the dust settled. One top-of-the-leaderboard entry, for instance, blended nine members (five frozen graph encoders, two Boltz-2 structural models, and two tabular models) with a Caruana-weighted forward selection. Nine members, each with its own weights and featurization to package, version, and rerun, is a lot to maintain for an edge smaller than the leaderboard's own bootstrap noise, a standard deviation of roughly 0.02 MAE.

That same fourth-place entry, from participant **N283T** ([model report](https://n283t.github.io/openadmet-pxr-model-report/)), is the one we ended up learning the most from, and not for its ensemble. Their report finished at 0.4113 MAE on Phase 2 and documented the pipeline clearly, and it credited the core idea to **Buterez et al. 2024** ([Nature Communications](https://www.nature.com/articles/s41467-024-45566-8)), a multi-fidelity transfer-learning paper. The premise is an appealing one: cheap, abundant low-fidelity labels can act as a proxy to sharpen predictions on the scarce, expensive high-fidelity target you actually care about. It maps cleanly onto the PXR assay funnel, where thousands of single-concentration log<sub>2</sub> fold change (log<sub>2</sub>FC) primary-screen readings sit upstream of a few thousand dose-response pEC<sub>50</sub> values. We assumed the paper's architecture was carrying the gain, and set out to reproduce it.

That assumption did not hold. Reading the report closely and rebuilding the pieces ourselves, we found the Buterez paper contributed one component: an auxiliary encoder that turns primary-screen data into a predicted log<sub>2</sub>FC readout. In our hands, the paper's architectural configurations, the frozen and fine-tuned graph-network bodies we sweep in panel 00, did not produce the improvement, and they land below the tabular-foundation-model configurations we tried later. The gain came instead from what happened to those predicted log<sub>2</sub>FC columns next. They went into a tabular foundation model (TabPFN first, then TabICL) alongside a molecular embedding. N283T reached the same conclusion in their own analysis, finding the predicted log<sub>2</sub>FC readout to be their single strongest feature. The paper gave us a reliable way to construct that feature, and the tabular model made good use of it.

That combination, careful featurization feeding a tabular foundation model, moved our numbers from roughly 0.51 to roughly 0.44 MAE on this split (a 5-seed mean of 0.4356 at its best), a step large enough that we wanted to support it as a first-class configuration in `openadmet-models`. We work through what each ingredient is worth on its own, how the ingredients combine into the best performer, which tabular model performs best, why the descriptor block needs compressing, whether the model's own uncertainty can be trusted, and where the comparison could still be misleading. Every panel is scored on the same blind set of 513 compounds the leaderboard used, so these numbers are directly comparable to the challenge entries.

---

**00: before tabular foundation models**

## Where the graph-network baselines land

We evaluated several approaches that use a neural encoder rather than a tabular foundation model, following the architectures from Buterez et al. 2024: a directly fine-tuned CheMeleon encoder, a graph neural network (GNN) whose body is initialized from a log<sub>2</sub>FC-pretraining checkpoint, and an auxiliary-encoder-plus-concatenation design. The four log<sub>2</sub>FC-encoder rows below cross frozen versus fine-tuned body against dose-response-only versus dose-response-plus-primary-screen training data. Only the concatenation architecture edges past the plain CheMeleon baseline here, and none of these routes reach the accuracy N283T reported. Freezing the encoder helps on primary-screen + dose-response data (0.5464 versus 0.5573 fine-tuned) but hurts on dose-response-only data, where fine-tuning wins (0.5339 versus 0.5849). The "dose-response-only" and "primary-screen + dose-response" labels describe only what the main model is directly supervised on. The frozen, CheMeleon-initialized log<sub>2</sub>FC encoder is always pretrained on primary-screen data, and its 2-column predicted log<sub>2</sub>FC readout is input into the main model in both variants, so primary-screen information is never fully excluded.

*(paste `reporting/figures/figure-00.html` here as an HTML card)*

> The panel floor is the concatenation architecture, the best-of-18 winner (freeze_epochs=2, 512-dim hidden, no gradient clipping) from a sweep over freeze_epochs × hidden width × gradient clip on the auxiliary-encoder-plus-concatenation design (Buterez et al. 2024): a 5-seed mean of 0.5176 (range 0.4919-0.5339). The "CheMeleon baseline" skips log<sub>2</sub>FC pretraining and fits straight to pEC<sub>50</sub> (0.5215, range 0.5058-0.5407). The four log<sub>2</sub>FC-encoder rows are CheMeleon-initialized but skip the auxiliary concatenation step, reusing the log<sub>2</sub>FC-pretrained checkpoint as the main model's init: frozen and fine-tuned on primary-screen + dose-response data (0.5464, 0.5573), and on dose-response-only data (0.5339, 0.5849).

---

**01: solo components**

## Does each ingredient carry signal on its own?

We tested each of four candidate feature blocks individually (the raw embedding, its predicted log<sub>2</sub>FC readout, RDKit descriptors, and Mordred descriptors) before evaluating any combination, following the same practice as the N283T report. Every row here is input into TabPFN v2.5, the regressor used throughout this comparison, except "our best overall": that row uses TabICL 2.1.1 instead, the winner of the separate regressor sweep in section 03, so it isn't on equal footing with the other rows in this panel. "CheMeleon embedding" below is the off-the-shelf embedding alone, nothing else concatenated. It is not the same experiment as any CheMeleon-based row in sections 02 or 03, which always add the log<sub>2</sub>FC readout block (and, in section 03, descriptors too) on top of this embedding. On its own, the raw embedding is the clearest single-ingredient win (0.4780). Among the descriptor-only rows, RDKit (0.5312) edges the predicted log<sub>2</sub>FC readout (0.5410), and Mordred alone is the weakest featureset on the panel (0.5784), below even the graph-network floor (0.5176). That ordering is soft. Mordred's PCA (principal component analysis) fit on unscaled descriptors (panel 04) has a 5-seed range of roughly 0.005 MAE, wider than the 0.0098 MAE separating the RDKit and readout rows. The pretrained CheMeleon embedding run through TabPFN with no fine-tuning (0.5002) lands below our from-scratch, log<sub>2</sub>FC-trained embedding (0.4780) but above every descriptor-only row, so the encoder's log<sub>2</sub>FC pretraining adds signal on top of what CheMeleon's general-purpose embedding already carries.

*(paste `reporting/figures/figure-01.html` here as an HTML card)*

> Single-ingredient MAE: raw embedding 0.4780 (best), pretrained CheMeleon embedding 0.5002, RDKit descriptors 0.5312, predicted log<sub>2</sub>FC readout 0.5410, and Mordred descriptors 0.5784 (worst, range 0.5751-0.5804). The best GNN baseline floor (0.5176) and "our best overall" are shown for reference.

---

**02: the path to the best performer**

## All seven combinations of embedding, readout, and descriptors

Every non-empty subset of the three ingredients was measured directly, holding the descriptor family fixed at Mordred (the stronger of the two 2D-descriptor blocks), and repeated with both the from-scratch, log<sub>2</sub>FC-trained embedding and CheMeleon's untouched, off-the-shelf pretrained embedding (compressed to 256 components by PCA to fit under TabPFN's feature limit, labeled PCA-256 in each row's sub-label), so all seven non-empty subsets are covered under each embedding source. No path was interpolated or assumed. Every subset row here is input into TabPFN v2.5. The best subset under TabPFN is embedding + readout + descriptors on CheMeleon's off-the-shelf embedding (0.4437), just ahead of the same three ingredients on the from-scratch embedding (0.4574) and the no-embedding readout + descriptors row (0.4531). TabICL on a leaner featureset (0.4356) beats every TabPFN subset here, though from a different combination (panel 03). That CheMeleon's embedding wins is a soft claim. The top rows' ranges overlap (embedding + readout + descriptors [CheMeleon] 0.4381-0.4476, readout + descriptors 0.4485-0.4590, embedding + readout + descriptors [log<sub>2</sub>FC] 0.4460-0.4722), so the gap sits inside seed-to-seed noise. Two patterns are steadier than the ranking. Once the from-scratch embedding is present, the readout is close to redundant with what it already carries (0.4574 with the readout versus 0.4610 without, within seed noise). For CheMeleon's off-the-shelf embedding, dropping the readout costs far more than dropping descriptors (0.4444 for embedding + readout versus 0.5290 for embedding + descriptors, against 0.4437 with all three). Mordred alone is again the weakest row (0.5784), unstable for the same PCA reason as panel 04.

*(paste `reporting/figures/figure-02.html` here as an HTML card)*

> The best subset under TabPFN is embedding + readout + descriptors on CheMeleon's off-the-shelf embedding (0.4437, range 0.4381-0.4476), with readout + descriptors (0.4531) and the from-scratch three-ingredient row (0.4574) close behind. Mordred descriptors alone is the worst (0.5784). TabICL (0.4356) is shown above the dashed line for reference.

---

**03: same features, different regressor**

## Evaluating tabular foundation models

Every row here fixes the featureset and swaps only the final tabular model. That featureset is CheMeleon embedding + readout + descriptors (386 columns) for every row except one. TabICL OOMs (runs out of memory) at that width, so it alone uses a leaner, near-tied featureset instead (CheMeleon embedding + readout, no descriptors, 0.4356 versus 0.4437 under TabPFN v2.5 on the full combination, a gap well inside seed noise). TabPFN v2.5, TabPFN v2.6, TabPFN v3, TabFM, LightGBM, and XGBoost all take identical 386-column inputs, the same data backing panel 02's green row. TabICL comes out ahead even from its leaner featureset, which is why we report it as "our best overall" in sections 00-02. Among the rows on the full 386-column featureset, TabPFN v2.5 (0.4437) and TabPFN v3 (0.4436) are a dead heat, a 0.0001 gap that is noise rather than a ranking, with v2.6 behind them (0.4571). TabFM (0.4795) and the two gradient-boosted baselines (LightGBM 0.4987, XGBoost 0.5311) trail all three TabPFN releases. TabFM's number is not fully comparable. <code>max_num_rows</code> is capped at 500 in TabFM v1.0.0 because the full 4,139-row training set OOMs its attention pass on this GPU (graphics processing unit), so its in-context set is subsampled per ensemble member while TabPFN and TabICL take all 4,139 rows. Best GNN baseline and Best single-ingredient are repeated from sections 01/02 for reference.

*(paste `reporting/figures/figure-03.html` here as an HTML card)*

> MAE by regressor on the fixed featureset: TabICL v2.1.1 0.4356 (best, on a leaner featureset with no descriptors, since it OOMs on the 386-column combination the others use), then TabPFN v3 0.4436, TabPFN v2.5 0.4437, TabPFN v2.6 0.4571, TabFM 0.4795, LightGBM 0.4987, and XGBoost 0.5311.

---

**04: the PCA compression sweep**

## Why is the descriptor block PCA-compressed at all?

Raw Mordred descriptors run to about 1,613 columns and RDKit adds another 217. CheMeleon's raw embedding is 2,048-dimensional. Feeding any of these to TabPFN uncompressed OOMs its between-items attention pass, even with <code>memory_saving_mode</code> and <code>fit_mode="low_memory"</code> set, because that pass scales with feature count as well as row count. For the embedding, the runs below use a small, from-scratch 256-dimensional encoder instead of CheMeleon's native 2,048 dimensions. PCA compression is applied only to the descriptor block, fit on the training split only and kept well under TabPFN's officially supported 2,000-feature hard limit, with roughly 500 total features as the practical headroom on this GPU. The embedding and predicted-readout columns themselves stay raw and uncompressed throughout every sweep here. PCA here runs directly on median-imputed, unstandardized descriptor values (no per-column scaling), so a handful of high-magnitude raw descriptors dominate the covariance. Every width tested, on both descriptor sources, already captures 99.998-100.000% of that raw variance. That number describes how much of the unscaled descriptor covariance survives compression, not how much predictive signal does, which is why MAE keeps changing with PCA width even though explained variance is already saturated at 64 components. Across widths, 128 components gives the lowest mean for both descriptor sources (Mordred only 0.4574, RDKit + Mordred 0.4581), with 64 worst and 256 in between, not a monotonic trend. The within-source ranges overlap enough that "128 is best" is a soft claim either way. Mordred-only and RDKit + Mordred at 128 differ by 0.0007, well inside both rows' seed ranges, so adding RDKit on top of Mordred buys no measurable accuracy. Every concatenation panel (00-02) therefore uses Mordred alone, a simplicity choice (one fewer descriptor source, no RDKit dependency) rather than a demonstrated accuracy win.

*(paste `reporting/figures/figure-04.html` here as an HTML card)*

> Lowest mean at 128 components for both sources (Mordred only: 0.4574, range 0.4460-0.4722; RDKit + Mordred: 0.4581, range 0.4537-0.4624), with 64 worst and 256 between. The embedding and readout blocks are identical, fixed, and uncompressed across every row. Only the descriptor source and PCA width change.

---

**05: out-of-fold isotonic calibration**

## Does repeating the N283T report's calibration step help?

Per the N283T report, fitting an isotonic map on out-of-fold (OOF) predictions and applying it to the blind set closed a similar-sized gap for a subset of their architecture ablations (~0.441 → ~0.408 MAE). We applied the same calibration step to our current best config (CheMeleon embedding + log<sub>2</sub>FC readout, TabICL), but without the ensembling their pipeline uses, which could itself account for the difference in outcome. For each of the 5 seeds, we 5-fold split the dose-response training records and retrained the log<sub>2</sub>FC readout encoder on each fold's training portion, holding out that fold for prediction (the CheMeleon embedding itself is frozen and off-the-shelf, so it is extracted once up front with no leakage risk), fit isotonic regression on the pooled OOF predictions, then applied that map to the blind set. In our hands the step made accuracy worse rather than better, the opposite of what the N283T report saw. Mean MAE rose from 0.4356 to 0.4507 across the same 5 seeds, and the potent (≥ 6.0) subset degraded too. We did not test the mechanism directly, but the likely explanation is that TabICL's own predictions on this featureset are already close to well-calibrated, so fitting an isotonic map on a 5-fold OOF split adds fold-to-fold noise (a different, per-fold-retrained readout encoder each time) with no systematic bias left to correct. The absence of ensembling in our setup could contribute as well. We reserve this step for configs that show a demonstrated calibration gap, rather than applying it by default. The OOF MAE itself isn't shown here. It is computed on raw per-record dose-response values, not per-compound pEC<sub>50</sub> like every other number on this page, so it isn't comparable to the blind-set MAE below and would misrepresent the calibration step's effect if plotted alongside it.

*(paste `reporting/figures/figure-05.html` here as an HTML card)*

> Calibration makes this config worse (0.4356 → 0.4507 mean MAE across the same 5 seeds). The <code>potent (≥ 6.0)</code> subset gets worse too (MAE 0.6882 → 0.7402 mean), and stays badly miscalibrated either way (R² around −9 to −9.4).

---

**06: is TabPFN's uncertainty trustworthy?**

## Checking the predicted distribution against actual error

TabPFN returns more than a point estimate. <code>output_type="full"</code> exposes a per-compound predicted distribution (a discretized Riemann distribution, not a Gaussian), with a mean, a standard deviation, and a quantile grid. Refitting the best config and predicting on the 513 blind compounds this way lets us check how well that spread tracks actual error. The left chart plots each compound's absolute residual against its predicted standard deviation, testing whether a higher predicted std means a bigger typical error. The right chart is a reliability diagram. It plots empirical coverage against nominal quantile level across all 9 predicted quantiles, testing whether, say, the predicted 0.4 quantile actually has 40% of true values at or below it. The spread does track error, but only weakly: the Spearman rank correlation between predicted standard deviation and absolute residual is 0.2472 (n = 513). Coverage is the weaker of the two. The reliability curve sits below the diagonal at every quantile level, so the predicted quantiles run systematically low rather than simply too narrow or too wide. Recalibrating the quantiles would correct that directional bias, but it could not raise the 0.2472 correlation, because Spearman rho is rank-based and invariant to any strictly monotonic rescaling of the predicted standard deviation (a non-strict map could introduce ties and shift it slightly).

*(paste `reporting/figures/figure-06.html` here as an HTML card)*

> <strong>Spearman ρ(|residual|, predicted std) = 0.2472</strong> (n = 513). The reliability curve sits below the dashed diagonal at every one of the 9 quantile levels, worst around the median (gap around −0.10 to −0.11 at the 0.4-0.6 levels).

---

**07: limitations**

## Where this comparison could be misleading us

Below are the places where the setup itself, not the model, could be driving the headline number, ordered by likely impact.

**Our encoder's hyperparameters were never tuned against this objective**

The N283T report's ChemProp encoder, at the same 256-dim embedding width we use, had its hyperparameters chosen with Optuna directly against downstream pEC<sub>50</sub> OOF MAE. Ours were carried over from earlier stages of this project and never searched end-to-end against the metric we're comparing on. Given matched embedding width, this is the leading candidate for the remaining gap to their single-pipeline target (0.437): ~0.020 MAE for our from-scratch/log<sub>2</sub>FC TabPFN backbone, a 5-seed mean of 0.4574 (that same recipe is also our best PCA-width variant, 128 components, so these are two views of one recipe, not two separate results), not anything downstream of the embedding.

**PCA compression is a memory workaround, not a modeling choice**

Every descriptor block on this page is compressed because the raw columns OOM TabPFN's attention pass on this GPU, not because compression was found to help. Some signal in the discarded variance could be recoverable on hardware with more memory headroom, a chunked/tiled attention implementation, or a regressor without TabPFN's feature-count ceiling.

**Only one embedding architecture was tried, though it's the best one on its own**

Every tabular-foundation-model result here uses a ChemProp-style encoder's embedding. In the N283T report's single-embedding comparison, ChemProp scored best (0.437) ahead of KERMT (0.448), GatedGCN (0.474), MoLFormer (0.475), and AttentiveFP (0.484), so swapping architectures alone is unlikely to close much of the remaining gap. Their jump to ~0.408 came from ensembling all of these (plus two Boltz-2 structural members) together, which is out of scope here by design. This limitation is more a note on undertested breadth than an expected source of headline gains.

---

*data: results/tabicl_embed_readout_mordred_pca128 (winner) · tabpfn_embed_readout_mordred_pca128 · tabpfn_embed_only_no_readout · tabpfn_readout_only_no_embed · tabpfn_mordred_only_pca128_no_embed_no_readout · tabpfn_rdkit_only_pca128_no_embed_no_readout · tabpfn_embed_mordred_pca128_no_readout · tabpfn_readout_mordred_pca128_no_embed · tabpfn_small_embed · lgbm_embed_readout_mordred_pca128 · xgboost_embed_readout_mordred_pca128 · tabfm_embed_readout_mordred_pca128 · tabpfn_uncertainty_mordred_pca128 · tabpfn_chemeleon_log2fc_mordred_pca128 · tabpfn_pec50_scratch_mordred_pca128 · tabpfn_pec50_chemeleon · freeze1_hd512_clip5.0 · e4_frozen · e4_finetune · e4_frozen_drc_only · e4_finetune_drc_only · pxr_baseline_predictions_phase1 · pxr_baseline_predictions_phase2 (PXR challenge, pxr-challenge repo)*
