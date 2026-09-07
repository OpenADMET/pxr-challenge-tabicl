# Are tabular foundation models all you need?

**The OpenADMET PXR (pregnane X receptor) blind challenge has closed, and the top of the leaderboard was ripe with highly performant submissions. At challenge end, a paired-bootstrap analysis could not separate the top entries. Our own model catalog was not among them, so we did what any curious team does after a challenge: we read the reports of the top participants, looking for ideas to implement.**

---

We ran the PXR blind challenge from April through July 2026, concluding with thousands of submissions from over 350 unique participants, 28 of which were statistically indistinguishable at the top of the leaderboard. Our CheMeleon baselines implemented in `openadmet-models` were starting to look a little long in the tooth by comparison, landing around 0.50 MAE (mean absolute error), well below the ~0.41 MAE the leading entries achieved. 

In our search, we were not solely prioritizing the lowest score. Large ensembles tuned over many submissions might nab the top leaderboard slot, but we wanted something we could fold into `openadmet-models` as a reproducible, YAML-defined pipeline: a method disclosed in enough detail to rebuild, and simple enough to maintain after the dust settled. Numerous unique ensemble members, each with its own weights and featurization to package, version, and rerun, is a lot to maintain...

The entry from participant **N283T** ([model report](https://n283t.github.io/openadmet-pxr-model-report/)) did both: a full-fledged nine-member ensemble (five frozen graph encoders, two Boltz-2 structural models, and two tabular models) with a Caruana-weighted forward selection *and* a fairly performant non-ensemble solution (well, technically two models, as we'll learn). Their leaderboard entry finished at 0.4113 MAE on the blinded 260-compound Phase 2 set, which is the only score in the report measured on the same compounds we evaluate on and so the only absolute reference here and documented the pipeline clearly, crediting the core idea to **Buterez et al. 2024** ([Nature Communications](https://www.nature.com/articles/s41467-024-45566-8)), a multi-fidelity transfer-learning paper. The premise is an appealing one: cheap, abundant low-fidelity labels can act as a proxy to sharpen predictions on the scarce, expensive high-fidelity target we actually care about. It also maps cleanly onto the PXR assay funnel, where thousands of single-concentration log<sub>2</sub> fold change (log<sub>2</sub>FC) primary-screen readings sit upstream of a few thousand dose-response pEC<sub>50</sub> values. We assumed the Buterez paper's architecture was driving the performance, and set out to reproduce it.

Upon reading the report closely and rebuilding the pieces ourselves, however, we found the Buterez paper contributed a single, albeit important, component: an auxiliary encoder trained on primary-screen data to predict a log<sub>2</sub>FC readout. The paper's architectural configurations, the frozen and fine-tuned graph-network configurations we compared in **Figure 1**, did not produce the sought after improvement. Instead, the gain came from feeding those log<sub>2</sub>FC readout columns, alongside a molecular embedding, into a *tabular foundation model*. N283T reached the same conclusion in their own analysis, finding the log<sub>2</sub>FC-trained Chemprop embedding to be their single strongest feature, as input into a tabular foundation model. 

That combination, careful featurization feeding a tabular foundation model, reduced our MAE from ~0.50 to ~0.43 on this split (a 5-seed mean of 0.4269 MAE at its best), a step large enough that we wanted to support it as a first-class configuration in `openadmet-models`. In this post, we work through what each ingredient contributes in isolation, how the ingredients combine into the best performer, and which tabular foundation model performs best. We then circle back to explain why the descriptor set needs compressing, whether the model's own uncertainty can be trusted, and where the comparison could still be misleading. [add section on the ensemble-size sweep here (Figure 7): how much of the tabular foundation model's advantage is its internal ensemble]

---

## Graph-network baselines

We evaluated several approaches that use a neural encoder rather than a tabular foundation model, following the architectures from Buterez et al. 2024: a directly fine-tuned CheMeleon encoder, a graph neural network (GNN) whose body is initialized from a log<sub>2</sub>FC-pretraining checkpoint, and an auxiliary-encoder-plus-concatenation design. [STALE: this passage described a two-by-two grid over encoder treatment crossed with dose-response-only versus dose-response-plus-primary-screen training data. This work fits every model on dose-response pEC<sub>50</sub> alone, so the primary-screen arm does not exist and the grid it describes is gone.] The best of these was a Chemprop message-passing network initialized from the log<sub>2</sub>FC checkpoint and fine-tuned (0.4902 MAE), ahead of the single-stage CheMeleon baseline (0.5017 MAE). Nothing on the panel is separated from the best row: the Tukey HSD critical distance across the eleven configurations is 0.0983 MAE and the closest comparison is *p* = 0.616, so this family cannot be ranked at 5 seeds. None of these experiments reached the accuracy N283T reported. [STALE: the concatenation architecture was reported as beating the CheMeleon baseline at *p* = 0.038 under a two-sample *t*-test. On this split the concatenation rows run 0.5043 to 0.5229 against the baseline's 0.5017, so the claim inverts, and the two-sample test has been replaced throughout by Tukey HSD over the whole panel, blocked on seed.]

*(paste `results/figures/fig1.html` here as an HTML card)*

**Figure 1. Graph neural network implementation performance.** The panel reports test set mean absolute error (MAE) for the graph-encoder configurations evaluated in this work. The CheMeleon baseline fits the off-the-shelf CheMeleon encoder directly to pEC<sub>50</sub> in a single stage. Each point is the mean MAE across 5 random seeds. [NEEDS REWRITE: the rest of this caption describes a figure that no longer exists. The two-by-two encoder-treatment grid is gone with the primary-screen arm; the whiskers are Tukey HSD comparison intervals, not minimum-to-maximum ranges, so two intervals touch exactly when the panel does not separate the pair; the dashed divider has been replaced by a shaded band spanning the best row's interval; and there is one reference row, N283T at 0.4113, rather than three. The caption also needs to say that position encodes score, colour encodes identity or verdict, and shape encodes whether a row was swept here, carried from another figure, or chosen.]

---

## Individual ingredient contributions

We then tested each candidate ingredient individually before evaluating any combination, following the same practice as the N283T report. Features were input into TabPFN v3. The embedding from a CheMeleon encoder fine-tuned on log<sub>2</sub>FC achieved the best single-ingredient result here (0.4594 MAE). The off-the-shelf CheMeleon embedding (0.4659 MAE, *p* = 1.0) and the same encoder fine-tuned on pEC<sub>50</sub> (0.4784 MAE, *p* = 0.85) are not separated from it, nor is the best graph network (0.4902 MAE, *p* = 0.27). The single-stage CheMeleon baseline is separated (0.5017 MAE, *p* = 0.033), as is the RDKit descriptor block on its own (0.5309 MAE, *p* = 3e-05) and the log<sub>2</sub>FC readout columns on their own (0.5308 MAE, *p* = 3e-05). [STALE: this paragraph previously ranked Mordred against RDKit here. Mordred is no longer on this panel: the descriptor probe that now precedes it (Figure 2) settles the descriptor block first, and RDKit beats Mordred at every width, so only RDKit is carried forward. TabPFN v2.5 is also gone, excluded for returning NaN on this split.]

*(paste `results/figures/fig3.html` here as an HTML card)*

**Figure 3. Single ingredient tabular foundation model performance.** The panel reports test set mean absolute error (MAE) for each candidate feature in isolation, fed to TabPFN v3. The RDKit block is PCA-reduced to 128 components and the CheMeleon embedding to 256 (from 217 and 2,048 columns respectively), both widths settled by the probes in Figure 2. The **Best graph network** and **CheMeleon baseline** rows repeat from Figure 1 for scale. As in Figure 1, whiskers are Tukey HSD comparison intervals across the 5 seeds; the N283T row is a published single value and has none. The **N283T** leaderboard entry is reproduced for reference.

---

## Combining embedding, log<sub>2</sub>FC readout, and descriptors

We evaluated every subset of embedding, log<sub>2</sub>FC readout, and descriptor set, holding the descriptor set fixed at RDKit reduced to 128 components, and repeated across the embeddings available: CheMeleon off the shelf, CheMeleon fine-tuned on log<sub>2</sub>FC, CheMeleon fine-tuned on pEC<sub>50</sub>, and Chemprop trained from scratch on either target. The top configuration was the log<sub>2</sub>FC-fine-tuned CheMeleon embedding + log<sub>2</sub>FC readout + descriptors (0.4325 MAE). Sixteen of the thirty-six rows are not separated from it, including the two-block configuration that drops descriptors entirely (0.4358 MAE, *p* = 1.0), so the descriptor block adds nothing detectable on top of an embedding and a readout. An embedding on its own is separated (0.4659 MAE, *p* = 0.031), as is the best graph network (0.4902 MAE, *p* = 5e-08) and the single-stage CheMeleon baseline (0.5017 MAE, *p* = 2e-11). [STALE: the closing claim here was that a CheMeleon-initialized encoder fine-tuned on log<sub>2</sub>FC was worse than both embeddings it was built from and was therefore omitted. It is now the best embedding on this panel and the leader's first block, so the reasoning inverts rather than needing a number swapped.]

*(paste `results/figures/fig4.html` here as an HTML card)*

**Figure 4. Combining embedding, log<sub>2</sub>FC readout, and descriptors.** The panel reports test set MAE for every subset of the three ingredients (embedding, log<sub>2</sub>FC readout, and descriptors), all fed to TabPFN v3. The descriptor set is RDKit reduced to 128 components throughout. The **Best single ingredient**, **Best graph network**, and **CheMeleon baseline** rows repeat from earlier figures for scale. As in Figure 1, whiskers are Tukey HSD comparison intervals across the 5 seeds. The **N283T** leaderboard entry is reproduced for reference. [NEEDS REWRITE: the caption still describes two candidate embeddings, both from one Chemprop model. The panel now carries five, and the sentence explaining that the embedding and readout come from the same model no longer holds for the rows where they come from different ones.]

---

## Evaluating tabular foundation models

We then fixed the featureset at **Figure 4**'s chosen configuration, CheMeleon embedding + log<sub>2</sub>FC readout (258 columns), and swapped only the regressor, to check whether a different tabular (or gradient boosting) model beats TabPFN v3 on the same inputs. TabICL comes out ahead (0.4269 MAE), with TabPFN v3 (0.4358 MAE) the only other regressor not separated from it (*p* = 1.0). TabFM (0.4800 MAE, *p* = 0.002), LightGBM (0.4809 MAE, *p* = 0.002), XGBoost (0.4941 MAE, *p* = 7e-05) and TabPFN v2.6 (0.5007 MAE, *p* = 1e-05) are all separated from the leader. We note that TabFM's number is not fully comparable: <code>max_num_rows</code> is capped at 500 in TabFM v1.0.0 because the full 4,392-row training set OOMs its attention pass on the GPU used here, so its in-context set is subsampled per ensemble member while TabPFN and TabICL took all 4,392 rows. [STALE: TabPFN v2.5 was previously swept and is now excluded, having returned NaN for every compound at this width on this hardware.]

TabICL leads nominally, 0.4269 against 0.4358 on the seed mean, and the pair is split on licence. TabICL is BSD 3-Clause. TabPFN carries the Prior Labs Licence, Apache 2.0 with an added provision requiring "Built with PriorLabs-TabPFN" displayed on any related website, blog post or product documentation, and a TabPFN name prefix on models derived from it. This work ends in a public write-up, so that is a live obligation rather than a hypothetical one. The cost is stated rather than hidden: TabICL does not fit this GPU and runs on CPU at about 160 seconds a fit, against roughly 20 on the accelerator for TabPFN v3, which also has the steadier seeds. **Best graph network** and **Best single ingredient** are also repeated from prior sections for reference.

*(paste `results/figures/fig5.html` here as an HTML card)*

**Figure 5. Tabular foundation model comparison.** The panel reports test set MAE with the featureset held fixed (CheMeleon embedding + log<sub>2</sub>FC readout, 258 columns), varying only the tabular foundation regressor. The swept rows cover two TabPFN releases (v2.6 and v3), TabICL, TabFM, and two gradient-boosted baselines (LightGBM and XGBoost). TabFM's row is not strictly comparable: its in-context set is capped at 500 rows to fit memory, while every other regressor saw all 4,392 training rows. As in Figure 1, whiskers are Tukey HSD comparison intervals across the 5 seeds. The **Best single ingredient**, **Best graph network**, and **CheMeleon baseline** rows repeat from earlier figures for scale. The **N283T** leaderboard entry is reproduced for reference.

---

## Why is the descriptor set PCA-compressed?

[ORDERING: this section now carries Figure 2 and the probes in it settle both the descriptor block and the embedding width that every later figure holds fixed, so it runs before the single-ingredient section rather than after the regressor comparison. Moving it is a cut and paste; the figure numbers already assume it.]

Raw Mordred descriptors are comprised of 1,613 columns, RDKit adds another 217, and CheMeleon's embedding is 2,048-dimensional. Feeding any of these to TabPFN uncompressed OOMs its between-items attention pass, even with <code>memory_saving_mode</code> and <code>fit_mode="low_memory"</code> set, because that pass scales with feature count *and* row count. For the embedding, the runs below reduce CheMeleon's native 2,048 dimensions to 256 components, a width the right-hand panel settles. PCA compression was applied only to the descriptor set, fit on the training split only and kept well under TabPFN's officially supported 2,000-feature hard limit, with roughly 500 total features as the practical headroom on the GPU used. The embedding and log<sub>2</sub>FC readout columns themselves stayed raw and uncompressed throughout every sweep here. PCA here ran directly on median-imputed, unstandardized descriptor values (no per-column scaling), so a handful of high-magnitude raw descriptors dominated the covariance. Explained variance exceeds 99% with just 16 components. RDKit beats Mordred and the concatenation of the two at every width. Keeping all 217 RDKit columns unreduced is nominally best (0.5300 MAE), and 128 components are not separated from it (0.5309 MAE, *p* = 1.0) at a little over half the width, while 64 is where reducing starts to cost (0.5405 MAE, *p* = 0.002). Mordred at its own best width is separated from the RDKit leader (0.5483 MAE, *p* = 2e-08), as is the concatenation of the two (0.5388 MAE, *p* = 0.015). The embedding was reduced on the same principle: 384 components lead nominally (0.4641 MAE), 256 are not separated from them (0.4659 MAE, *p* = 0.96) at two thirds of the columns, and 128 is separated and worse (0.4930 MAE, *p* = 6e-10). Later sections therefore carry RDKit at 128 components and the embedding at 256. [STALE: the previous version chose Mordred alone as a simplicity choice on the grounds that the two descriptor sets were indistinguishable. They are not indistinguishable on this split, and the choice is now RDKit on accuracy rather than Mordred on convenience, so every mention of Mordred as the carried descriptor set elsewhere in the post inverts.]

*(paste `results/figures/fig2.html` here as an HTML card)*

**Figure 2. How far the feature blocks reduce.** Two panels. On the left, the descriptor probe: test set MAE varying the descriptor set (RDKit, Mordred, or both) and its PCA width (32, 64, 128, 256 components, or the block kept whole), with no embedding and no log<sub>2</sub>FC readout, so the question is about the descriptors alone. On the right, the embedding-reduction probe: the CheMeleon embedding alone at 32 to 512 components. Each panel is corrected over its own family. As in Figure 1, whiskers are Tukey HSD comparison intervals across the 5 seeds.

---

## Does repeating the N283T report's calibration step help?

[STALE: this section reported an isotonic-calibration arm as a swept comparison with its own figure. Calibration is no longer swept. It is applied post hoc to a single chosen configuration and has no figure of its own, so the section needs either re-running against the current best configuration or cutting. The numbers below are from the previous generation's pooled 513-compound evaluation and describe a comparison this work does not make.]

Per the N283T report, fitting an isotonic map on out-of-fold (OOF) predictions and applying it to the test set closed a similar-sized gap for a subset of their architecture ablations (~0.441 → ~0.408 MAE). We applied the same calibration step to our current best config (CheMeleon embedding + log<sub>2</sub>FC readout, TabICL), but without the ensembling their pipeline uses. For each of the 5 seeds, we 5-fold split the dose-response training records and retrained the log<sub>2</sub>FC readout encoder on each fold's training portion, holding out that fold for prediction (the CheMeleon embedding itself was frozen and off-the-shelf, so it was extracted once up front with no leakage risk), fit isotonic regression on the pooled OOF predictions, then applied that map to the test set. In our testing, the calibration step *degraded* performance, opposite the N283T report result. Mean MAE rose from 0.4356 to 0.4507 across the same 5 seeds, a significant difference under a paired *t*-test (*p* = 0.010). The absence of ensembling in our setup is the most likely explanation here, though this remains untested.

[REMOVED: there is no calibration figure in the current set. Figure 6 is now the uncertainty diagnostic below.]

---

## Checking the predicted distribution against actual error

Tabular foundation models return more than a point estimate. <code>output_type="full"</code> exposes a per-compound predicted distribution (a discretized Riemann distribution, not a Gaussian), with a mean, a standard deviation, and a quantile grid.
We use the chosen configuration from **Figure 5**, CheMeleon embedding + log<sub>2</sub>FC readout fed to TabICL. The left chart plots each compound's absolute residual against its predicted standard deviation, testing whether a predicted standard deviation correlates with error. The right chart is a reliability diagram, plotting empirical coverage against nominal quantile level across all 9 predicted quantiles, testing whether, say, the predicted 0.4 quantile actually has 40% of true values at or below it (see our previous post on uncertainty [here](https://openadmet.ghost.io/concerning-uncertainty/)). The spread did track error, but only weakly: the Spearman rank correlation between predicted standard deviation and absolute residual was 0.2616. Recalibrating the quantiles would not raise that correlation, because Spearman rho is rank-based and invariant to any strictly monotonic rescaling of the predicted standard deviation. [STALE: the reliability reading has flipped sign. The model's intervals now come out slightly *under*confident on this split, with mean observed coverage 0.036 above nominal and a miscalibration area of 0.041, where the previous version reported the curve below the diagonal at every level and called the model overconfident. The direction of the conclusion needs rewriting, not just the number.] Thus, in this case, tabular foundation models unfortunately do not solve the uncertainty quantification problem for us.

*(paste `results/figures/fig6.html` here as an HTML card)*

**Figure 6. Predicted uncertainty versus actual error, TabICL.** Left: each of the 260 phase-2 test compounds' predicted standard deviation, plotted against its absolute residual. Right: the reliability diagram, empirical coverage plotted against nominal level, with the gap to the diagonal (perfect calibration) shaded.

## Is the tabular foundation model's advantage just ensembling?

[add section on the ensemble-size sweep here (Figure 7): TabPFN aggregates over permutations of one in-context model, and every result above used its default of 8 members. Sweeping 1 to 128 members on the chosen featureset, only the first doubling moves anything. A single member scores 0.4443 MAE and 128 members score 0.4358, with every count from 2 upward tied with the leader and only 1 member separated (*p* = 4e-06). A single member already beats the best single block (0.4594), the best graph network (0.4902), and the CheMeleon baseline (0.5017), so ensembling accounts for under a fifth of the margin over the next best approach. The seed-ensemble gap shrinks from 0.0101 to 0.0039 across the sweep, and five seeds of one-member models score 0.4343, ahead of a single eight-member model at 0.4362.]

*(paste `results/figures/fig7.html` here as an HTML card)*

[add caption for Figure 7 here: ensemble size on a log axis against MAE on the left and against the Spearman correlation between predicted spread and absolute residual on the right, with one model and the five seeds ensembled drawn on both panels.]

---

## Applying the method to the CYP inhibition blind challenge

OpenADMET's next blind challenge, modeling cytochrome P450 (CYP) inhibition across 4 isoforms, [kicked off on August 17, 2026](https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/). Of course we wanted to apply the newly minted PCA-transformed-CheMeleon-embeddings-concatenated-with-/log<sub>2</sub>FC-predictions-input-into-tabular-foundation-model approach. This was our **TabICL-baseline**, which held the top spot on the regression live leaderboard (0.6755 MA-ST-RAE) for... a few days. As of this writing, the top performers have reduced that error by ~36%. Still, the new method improves upon previous `openadmet-model` baselines by ~20% in this challenge, well worth the effort to adopt.

---

## Limitations

**Our encoder's hyperparameters were never tuned against this objective**

The N283T report's Chemprop encoder, at the same 256-dim embedding width we used, had its hyperparameters chosen with Optuna directly against downstream pEC<sub>50</sub> OOF MAE. Ours were carried over from earlier stages of this project and never searched end-to-end against the metric we compared on. [NEEDS REWRITE: this limitation measured a gap to N283T's single-model out-of-fold number. That comparison has been dropped, and the only reference is now their leaderboard entry on the same phase-2 compounds, 0.4113 against our 0.4269. The point about untuned encoder hyperparameters still stands and needs restating against that 0.0156 MAE gap.]

**PCA compression is a memory workaround, not a modeling choice**

Descriptor sets were compressed because the raw columns OOMed TabPFN's attention pass on the GPU we used, not because compression was found to help. Some signal in the discarded variance could be recoverable on hardware with more memory headroom, a chunked/tiled attention implementation, or a regressor without TabPFN's feature-count ceiling. [STALE: on this split the RDKit block fits unreduced at 217 columns and is nominally the best descriptor row, and the chosen featureset carries no descriptors at all, so the memory ceiling no longer binds where this limitation says it does.]

**Only one embedding architecture was tried**

[STALE: this work now sweeps five embeddings, CheMeleon off the shelf and fine-tuned on either target, and Chemprop trained from scratch on either target, and the winner is a fine-tuned CheMeleon rather than a Chemprop. The limitation as written no longer holds, though the narrower point that no non-message-passing architecture was tried still does.]

Every tabular-foundation-model result here used a Chemprop-style encoder's embedding. In the N283T report's single-embedding comparison, Chemprop scored best (0.437 MAE) ahead of KERMT (0.448 MAE), GatedGCN (0.474 MAE), MoLFormer (0.475 MAE), and AttentiveFP (0.484 MAE), so swapping architectures alone is unlikely to close much of the remaining gap. Their jump to ~0.408 MAE came from ensembling all of these (plus two Boltz-2 structural members) together, which is intentionally out of scope here.

## Reproducibility

[STALE: this section describes exploratory branches across two repositories and a partial carry-over. The work behind this post is now a single declared pipeline: a manifest that states every sweep, gate and figure before it runs, numbered entry points from featurizing to figures, and content-addressed artifacts with provenance on every run. The section needs rewriting around what actually exists, and the release link is still a placeholder.]

## Acknowledgements

We would like to thank our funders for their support of OpenADMET, in particular ARPAH, Radial (part of the [Astera Institute](https://ror.org/00ydx1s47)), Schrodinger Inc,  and the Gates Foundation.  We would also like to thank our partners Enamine, HuggingFace, OpenEye, CDD Vault, Discovery Life Sciences and the beamline staff at NSLS-II for their support. 

This work is supported by the Advanced Research Projects Agency for Health (ARPA-H) under AVOID-OME, and Award Number 1AY1AX000035. The contents are those of the authors. They may not reflect the policies of the Department of Health and Human Services or the U.S. government. The content is solely the responsibility of the authors and does not necessarily represent the official views of the Advanced Research Projects Agency for Health.