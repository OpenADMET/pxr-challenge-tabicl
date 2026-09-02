# Are tabular foundation models all you need?

**The OpenADMET PXR (pregnane X receptor) blind challenge has closed, and the top of the leaderboard was crowded. At the Phase 1 handover, a paired-bootstrap analysis could not separate the top ten entries. Our own model catalog was not among them, so we did what any curious team does after a challenge: we read the reports of the top participants, looking for ideas to implement.**

---

We ran the PXR blind challenge from April through July 2026, concluding with X submissions from Y teams, 10 of which were statistically indistinguishable at the top of the leaderboard. Our CheMeleon baselines implemented in `openadmet-models` were starting to look a little long in the tooth by comparison, landing around 0.51 to 0.52 MAE (mean absolute error), well below the ~0.41 the leading entries achieved. 

In our search, we were not solely prioritizing the lowest score. Large ensembles tuned over many submissions might nab the top leaderboard slot, but we wanted something we could fold into `openadmet-models` as a reproducible, YAML-defined pipeline: a method disclosed in enough detail to rebuild, and simple enough to maintain after the dust settled. Numerous unique ensemble members, each with its own weights and featurization to package, version, and rerun, is a lot to maintain...

The entry from participant **N283T** ([model report](https://n283t.github.io/openadmet-pxr-model-report/)) did both: a full-fledged nine-member ensemble (five frozen graph encoders, two Boltz-2 structural models, and two tabular models) with a Caruana-weighted forward selection *and* a fairly performant single-model solution (well, technically two, as we'll learn). Their report finished at 0.4113 MAE ("single" model at 0.437 MAE) on Phase 2 and documented the pipeline clearly, crediting the core idea to **Buterez et al. 2024** ([Nature Communications](https://www.nature.com/articles/s41467-024-45566-8)), a multi-fidelity transfer-learning paper. The premise is an appealing one: cheap, abundant low-fidelity labels can act as a proxy to sharpen predictions on the scarce, expensive high-fidelity target you actually care about. It maps cleanly onto the PXR assay funnel, where thousands of single-concentration log<sub>2</sub> fold change (log<sub>2</sub>FC) primary-screen readings sit upstream of a few thousand dose-response pEC<sub>50</sub> values. We assumed the Buterez paper's architecture was driving the performance, and set out to reproduce it.

Upon reading the report closely and rebuilding the pieces ourselves, however, we found the Buterez paper contributed a single, albeit important, component: an auxiliary encoder trained on primary-screen data to predict a log<sub>2</sub>FC readout. The paper's architectural configurations, the frozen and fine-tuned graph-network configurations we compared in Figure 1, did not produce the sought after improvement. Instead, the gain came from feeding those predicted log<sub>2</sub>FC columns, alongside a molecular embedding, into a *tabular foundation model*. N283T reached the same conclusion in their own analysis, finding the log<sub>2</sub>FC-trained Chemprop embedding to be their single strongest feature, as input into a tabular foundation model. 

That combination, careful featurization feeding a tabular foundation model, reduced our MAE from ~0.51 to ~0.44 on this split (a 5-seed mean of 0.4356 at its best), a step large enough that we wanted to support it as a first-class configuration in `openadmet-models`. We work through what each ingredient contributes in isolation, how the ingredients combine into the best performer, which tabular foundation model performs best, why the descriptor block needs compressing, whether the model's own uncertainty can be trusted, and where the comparison could still be misleading. Every figure was scored on the same blind set of 513 compounds the leaderboard used, so these numbers are directly comparable to the challenge entries.

---

## Graph-network baselines

We evaluated several approaches that use a neural encoder rather than a tabular foundation model, following the architectures from Buterez et al. 2024: a directly fine-tuned CheMeleon encoder, a graph neural network (GNN) whose body is initialized from a log<sub>2</sub>FC-pretraining checkpoint, and an auxiliary-encoder-plus-concatenation design. The four log<sub>2</sub>FC-encoder rows below cross frozen versus fine-tuned body against dose-response-only versus dose-response-plus-primary-screen training data. The concatenation architecture did surpass the plain CheMeleon baseline, but none of these experiments reached the accuracy N283T reported. 

*(paste `reporting/figures/figure-00.html` here as an HTML card)*

> The figure floor is the concatenation architecture, the best-of-18 winner (freeze_epochs=2, 512-dim hidden, no gradient clipping) from a sweep over freeze_epochs × hidden width × gradient clip on the auxiliary-encoder-plus-concatenation design (Buterez et al. 2024): a 5-seed mean of 0.5176 MAE (range 0.4919-0.5339). The "CheMeleon baseline" forgoes log<sub>2</sub>FC pretraining and fits straight to pEC<sub>50</sub> (0.5215 MAE, range 0.5058-0.5407). The four log<sub>2</sub>FC-encoder rows are CheMeleon-initialized without the auxiliary concatenation step, reusing the log<sub>2</sub>FC-pretrained checkpoint as the main model's initialization: frozen and fine-tuned on primary-screen + dose-response data (0.5464, 0.5573 MAE), and on dose-response-only data (0.5339, 0.5849 MAE).

---

## Individual component contributions

We tested each of four candidate feature blocks individually (the raw embedding, its predicted log<sub>2</sub>FC readout, RDKit descriptors, and Mordred descriptors) before evaluating any combination, following the same practice as the N283T report. Features were input into TabPFN v2.5 with one exception: "our best overall" used TabICL 2.1.1 instead, from a separate regressor sweep in a later section, included for reference. The embedding from log<sub>2</sub>FC-trained Chemprop achieved the best single-ingredient result here. Among the descriptor-only entries, RDKit (0.5312 MAE) edged the predicted log<sub>2</sub>FC readout (0.5410 MAE), and Mordred was the weakest featureset on the figure (0.5784 MAE), worse than the graph-network floor (0.5176 MAE). The pretrained CheMeleon embedding run through TabPFN with no log<sub>2</sub>FC fine-tuning (0.5002 MAE) scored worse than the log<sub>2</sub>FC-trained Chemprop embedding (0.4780), but above every descriptor-only entry.

*(paste `reporting/figures/figure-01.html` here as an HTML card)*

> Single-ingredient MAE: log<sub>2</sub>FC-trained Chemprop embedding 0.4780 (best), pretrained CheMeleon embedding 0.5002, RDKit descriptors 0.5312, predicted log<sub>2</sub>FC readout 0.5410, and Mordred descriptors 0.5784 (worst, range 0.5751-0.5804). The best GNN baseline floor (0.5176) and "our best overall" are shown for reference.

---

## Combining embedding, log<sub>2</sub>FC readout, and descriptors

Every subset of the embedding, log<sub>2</sub>FC readout, and descriptor set was measured directly, holding the descriptor family fixed at Mordred (the stronger of the two 2D-descriptor blocks), and repeated with both the log<sub>2</sub>FC-trained Chemprop embedding and CheMeleon's off-the-shelf pretrained embedding (compressed to 256 components by PCA to fit under TabPFN's feature limit). The top configuration was embedding + readout + descriptors on CheMeleon's off-the-shelf embedding (0.4437 MAE), just ahead of the Chemprop log<sub>2</sub>FC-trained embedding (0.4574 MAE) and the no-embedding log<sub>2</sub>FC readout + descriptors (0.4531 MAE). However, the top rows' ranges overlapped with seed-to-seed variability. Once the log<sub>2</sub>FC-trained Chemprop embedding was present, the log<sub>2</sub>FC readout was close to redundant with what its representation already contained (0.4574 MAE with the readout versus 0.4610 MAE without, within seed variability). For CheMeleon's off-the-shelf embedding, dropping the log<sub>2</sub>FC readout cost far more than dropping descriptors (0.4444 MAE for embedding + readout versus 0.5290 MAE for embedding + descriptors, against 0.4437 MAE with all three). Mordred alone was again the weakest row (0.5784 MAE).

*(paste `reporting/figures/figure-02.html` here as an HTML card)*

> The best subset under TabPFN is embedding + readout + descriptors on CheMeleon's off-the-shelf embedding (0.4437, range 0.4381-0.4476), with readout + descriptors (0.4531) and the from-scratch three-ingredient row (0.4574) close behind. Mordred descriptors alone is the worst (0.5784). TabICL (0.4356) is shown above the dashed line for reference.

---

## Evaluating tabular foundation models

Next we fixed the featureset (CheMeleon embedding + readout + descriptors) and swapped only the tabular model, with one exception. TabICL ran out of memory (OOM) at that width, so it used a leaner, near-tied featureset instead (CheMeleon embedding + readout, no descriptors, 0.4356 versus 0.4437 under TabPFN v2.5 on the full combination). TabPFN v2.5, TabPFN v2.6, TabPFN v3, TabFM, LightGBM, and XGBoost all took identical 386-column inputs, the same data backing Figure 3's green row. TabICL came out ahead even from its leaner featureset, which is why we reported it as "our best overall" in previous sections. Among the rows on the full 386-column featureset, TabPFN v2.5 (0.4437) and TabPFN v3 (0.4436) were a dead heat, a 0.0001 gap that is just noise, with v2.6 behind them (0.4571). TabFM (0.4795) and the two gradient-boosted baselines (LightGBM 0.4987, XGBoost 0.5311) trailed all three TabPFN releases. We note that TabFM's number is not fully comparable: <code>max_num_rows</code> is capped at 500 in TabFM v1.0.0 because the full 4,139-row training set OOMs its attention pass on the GPU (graphics processing unit) used here, so its in-context set is subsampled per ensemble member while TabPFN and TabICL took all 4,139 rows. Best GNN baseline and Best single-ingredient are repeated from prior sections for reference.

*(paste `reporting/figures/figure-03.html` here as an HTML card)*

> MAE by regressor on the fixed featureset: TabICL v2.1.1 0.4356 (best, on a leaner featureset with no descriptors, since it OOMs on the 386-column combination the others use), then TabPFN v3 0.4436, TabPFN v2.5 0.4437, TabPFN v2.6 0.4571, TabFM 0.4795, LightGBM 0.4987, and XGBoost 0.5311.

---

## Why is the descriptor block PCA-compressed?

Raw Mordred descriptors are comprised of 1,613 columns, RDKit adds another 217, and CheMeleon's embedding is 2,048-dimensional. Feeding any of these to TabPFN uncompressed OOMs its between-items attention pass, even with <code>memory_saving_mode</code> and <code>fit_mode="low_memory"</code> set, because that pass scales with feature count *and* row count. For the embedding, the runs below used a small, Chemprop-based 256-dimensional encoder instead of CheMeleon's native 2,048 dimensions. PCA compression was applied only to the descriptor block, fit on the training split only and kept well under TabPFN's officially supported 2,000-feature hard limit, with roughly 500 total features as the practical headroom on the GPU used. The embedding and predicted-readout columns themselves stayed raw and uncompressed throughout every sweep here. PCA here ran directly on median-imputed, unstandardized descriptor values (no per-column scaling), so a handful of high-magnitude raw descriptors dominated the covariance. [ADD EXPLAINED VARIANCE] Across widths, 128 components gave the lowest nominal mean for both descriptor sources (Mordred only 0.4574 MAE, RDKit + Mordred 0.4581 MAE), with 64 worst and 256 in between, not a monotonic trend. The within-source ranges overlapped enough that "128 is best" is a soft claim either way. Mordred-only and RDKit + Mordred at 128 differed by 0.0007, well inside both rows' seed ranges, so adding RDKit on top of Mordred did not meaningfully improve accuracy. Given the insensitivity here, all previous analyses used Mordred alone as a simplicity choice (one fewer descriptor source, no RDKit dependency).

*(paste `reporting/figures/figure-04.html` here as an HTML card)*

> Lowest mean at 128 components for both sources (Mordred only: 0.4574, range 0.4460-0.4722; RDKit + Mordred: 0.4581, range 0.4537-0.4624), with 64 worst and 256 between. The embedding and readout blocks are identical, fixed, and uncompressed across every row. Only the descriptor source and PCA width change.

---

## Does repeating the N283T report's calibration step help?

Per the N283T report, fitting an isotonic map on out-of-fold (OOF) predictions and applying it to the blind set closed a similar-sized gap for a subset of their architecture ablations (~0.441 → ~0.408 MAE). We applied the same calibration step to our current best config (CheMeleon embedding + log<sub>2</sub>FC readout, TabICL), but without the ensembling their pipeline uses. For each of the 5 seeds, we 5-fold split the dose-response training records and retrained the log<sub>2</sub>FC readout encoder on each fold's training portion, holding out that fold for prediction (the CheMeleon embedding itself was frozen and off-the-shelf, so it was extracted once up front with no leakage risk), fit isotonic regression on the pooled OOF predictions, then applied that map to the blind set. In our testing, the calibration step *degraded* performance, opposite the N283T report result. Mean MAE rose from 0.4356 to 0.4507 across the same 5 seeds. The absence of ensembling in our setup is the most likely explanation here, though this remains untested.

*(paste `reporting/figures/figure-05.html` here as an HTML card)*

> Calibration makes this config worse (0.4356 → 0.4507 mean MAE across the same 5 seeds). The <code>potent (≥ 6.0)</code> subset gets worse too (MAE 0.6882 → 0.7402 mean), and stays badly miscalibrated either way (R² around −9 to −9.4).

---

## Checking the predicted distribution against actual error

Tabular foundation models return more than a point estimate. <code>output_type="full"</code> exposes a per-compound predicted distribution (a discretized Riemann distribution, not a Gaussian), with a mean, a standard deviation, and a quantile grid. Refitting the best config and predicting on the 513 blind compounds this way let us check how well that spread tracked actual error. The left chart plots each compound's absolute residual against its predicted standard deviation, testing whether a predicted standard deviation correlates with error. The right chart is a reliability diagram. It plots empirical coverage against nominal quantile level across all 9 predicted quantiles, testing whether, say, the predicted 0.4 quantile actually has 40% of true values at or below it. The spread did track error, but only weakly: the Spearman rank correlation between predicted standard deviation and absolute residual was 0.2472 (n = 513). The reliability curve sat below the diagonal at every quantile level, so the predicted quantiles ran systematically low rather than simply too narrow or too wide. Recalibrating the quantiles would correct that directional bias, but it could not raise the 0.2472 correlation, because Spearman rho is rank-based and invariant to any strictly monotonic rescaling of the predicted standard deviation.

*(paste `reporting/figures/figure-06.html` here as an HTML card)*

> <strong>Spearman ρ(|residual|, predicted std) = 0.2472</strong> (n = 513). The reliability curve sits below the dashed diagonal at every one of the 9 quantile levels, worst around the median (gap around −0.10 to −0.11 at the 0.4-0.6 levels).

---

## Limitations

**Our encoder's hyperparameters were never tuned against this objective**

The N283T report's ChemProp encoder, at the same 256-dim embedding width we used, had its hyperparameters chosen with Optuna directly against downstream pEC<sub>50</sub> OOF MAE. Ours were carried over from earlier stages of this project and never searched end-to-end against the metric we compared on. Given matched embedding width, this is the leading candidate for the remaining gap to their single-pipeline target (0.437): ~0.020 MAE for our from-scratch/log<sub>2</sub>FC TabPFN backbone, a 5-seed mean of 0.4574 (that same recipe is also our best PCA-width variant, 128 components, so these are two views of one recipe, not two separate results), not anything downstream of the embedding.

**PCA compression is a memory workaround, not a modeling choice**

Every descriptor block on this page was compressed because the raw columns OOMed TabPFN's attention pass on the GPU we used, not because compression was found to help. Some signal in the discarded variance could be recoverable on hardware with more memory headroom, a chunked/tiled attention implementation, or a regressor without TabPFN's feature-count ceiling.

**Only one embedding architecture was tried, though it's the best one on its own**

Every tabular-foundation-model result here used a ChemProp-style encoder's embedding. In the N283T report's single-embedding comparison, ChemProp scored best (0.437) ahead of KERMT (0.448), GatedGCN (0.474), MoLFormer (0.475), and AttentiveFP (0.484), so swapping architectures alone is unlikely to close much of the remaining gap. Their jump to ~0.408 came from ensembling all of these (plus two Boltz-2 structural members) together, which is intentionally out of scope here. This limitation is more a note on undertested breadth than an expected source of performance gains.
