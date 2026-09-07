# Are tabular foundation models all you need?

**The OpenADMET PXR (pregnane X receptor) blind challenge has closed, and the top of the leaderboard was ripe with highly performant submissions. At challenge end, a paired-bootstrap analysis could not separate the top entries. Our own model catalog was not among them, so we did what any curious team does after a challenge: we read the reports of the top participants, looking for ideas to implement.**

---

We ran the PXR blind challenge from April through July 2026, concluding with thousands of submissions from over 350 unique participants, 28 of which were statistically indistinguishable at the top of the leaderboard. Our CheMeleon baselines implemented in `openadmet-models` were starting to look a little long in the tooth by comparison, landing around 0.50 MAE (mean absolute error), well below the ~0.41 MAE the leading entries achieved.

In our search, we were not solely prioritizing the lowest score. Large ensembles tuned over many submissions might nab the top leaderboard slot, but we wanted something we could fold into `openadmet-models` as a reproducible pipeline defined in YAML (yet another markup language): a method disclosed in enough detail to rebuild, and simple enough to maintain after the dust settled. Numerous unique ensemble members, each with its own weights and featurization to package, version, and rerun, is a lot to maintain...

The entry from participant **N283T** ([model report](https://n283t.github.io/openadmet-pxr-model-report/)) did both: a full-fledged nine-member ensemble (five frozen graph encoders, two Boltz-2 structural models, and two tabular models) with a Caruana-weighted forward selection *and* a fairly performant non-ensemble solution (well, technically two models, as we'll learn). Their leaderboard entry finished at 0.4113 MAE on the blinded 260-compound Phase 2 set, crediting the core idea to **Buterez et al. 2024** ([Nature Communications](https://www.nature.com/articles/s41467-024-45566-8)), a multi-fidelity transfer-learning paper. The premise is an appealing one: cheap, abundant low-fidelity labels can act as a proxy to sharpen predictions on the scarce, expensive high-fidelity target we actually care about. It also maps cleanly onto the PXR assay funnel, where thousands of single-concentration log<sub>2</sub> fold change (log<sub>2</sub>FC) primary-screen readings sit upstream of a few thousand dose-response pEC<sub>50</sub> values. We assumed the Buterez paper's architecture was driving the performance, and set out to reproduce it.

Upon reading the report closely and rebuilding the pieces ourselves, however, we found the Buterez paper contributed a single, albeit important, component: an auxiliary encoder trained on primary-screen data to predict a log<sub>2</sub>FC readout. The paper's architectural configurations, the frozen and fine-tuned graph-network configurations we compared in **Figure 1**, did not produce the sought-after improvement. Instead, the gain came from feeding those log<sub>2</sub>FC readout columns, alongside a molecular embedding, into a *tabular foundation model*. **N283T** reached the same conclusion in their own analysis, finding the log<sub>2</sub>FC-trained Chemprop embedding to be their single strongest feature, as input into a tabular foundation model.

That combination, careful featurization feeding a tabular foundation model, reduced our MAE from ~0.50 to ~0.43 on this split (a 5-seed mean of 0.4269 MAE at its best), a step large enough that we wanted to support it as a first-class configuration in `openadmet-models`. In this post, we work through what each ingredient contributes in isolation, how the ingredients combine into the best performer, and which tabular foundation model performs best.

---

## Graph-network baselines

We evaluated several approaches that use a neural encoder rather than a tabular foundation model, following the architectures from Buterez et al. 2024: a directly fine-tuned CheMeleon encoder, a graph neural network (GNN) whose body is initialized from a log<sub>2</sub>FC-pretraining checkpoint, and an auxiliary-encoder-plus-concatenation design, with a Chemprop network trained from scratch as a control. The best of these was a Chemprop message-passing network initialized from the log<sub>2</sub>FC checkpoint and fine-tuned (0.4902 MAE), nominally ahead of the single-stage **CheMeleon baseline** (0.5017 MAE) and the best concatenation configuration (0.5043 MAE). Nothing on the panel is separated from the best row: the Tukey Honestly Significant Difference (HSD) critical distance across the ten configurations is 0.0983 MAE and the closest comparison is *p* = 0.616. None of these experiments reached the accuracy **N283T** reported.

*(paste `results/figures/fig1.html` here as an HTML card)*

**Figure 1. Graph neural network implementation performance.** Test set MAE in ascending order for the graph-encoder configurations evaluated in this work, each point the mean across 5 random seeds. Whiskers are Tukey HSD comparison intervals, blocked on seed. Overlapping whiskers signal a pair cannot be distinguished. The shaded band spans the best row's interval. The **N283T** row is a published leaderboard value, drawn without an interval and not tested against. Hover on a point or row label to see detailed configuration information.

---

## What are tabular foundation models?

A bit discouraged by the GNN results, we turned to the next potential source of improvement: the use of a tabular foundation model as regressor. But what are they, other than "a foundation model for tabular data"? Prior to this effort, Karim Ben Hicham gave [an excellent overview](https://www.youtube.com/watch?v=e1XdacPHlJg) in our OpenADMET seminar series, based on his work detailed in [this paper](https://doi.org/10.48550/arXiv.2604.16123). Summarizing, TabPFN and the models following it are pretrained on millions of synthetic tables whose data-generating process is known by construction, learning to approximate Bayesian inference. Exact Bayesian inference is intractable because it integrates over every model that could have produced the data. As such, approximating it in a single forward pass, with the training rows held in context and nothing fitted, is an elegant saving. But does that pretraining generalize to the relationships under study here?

---

## Reducing dimensionality

The next sections move into tabular foundation model evaluation, which are notoriously memory intensive. Descriptor sets run to 217 dimensions for RDKit and 1,613 for Mordred, a CheMeleon embedding to 2,048, and their concatenations compose. We swept principal component analysis (PCA) dimensionality reductions over descriptors and fed them to TabPFN in isolation to see if we could safely reduce feature count without degrading performance (**Figure 2, left**). Native-dimension RDKit topped the comparison at 0.5300 MAE, but was statistically indistinguishable from PCA-reducing it to 128 (0.5309 MAE, *p* = 1.0). Mordred and RDKit+Mordred trailed in all configurations, and RDKit at 128 components separated from every one of them except the 256-component concatenation (0.5378 MAE, *p* = 0.13). We thus proceeded with RDKit descriptors PCA-reduced to 128 dimensions as our canonical descriptor set.

We conducted a similar sweep with the 2,048-dim CheMeleon embedding (**Figure 2, right**). We could not evaluate dimensions above 512 without hitting memory limits. PCA reduction to 256, 384, and 512 were all statistically indistinguishable (0.4659, 0.4641 and 0.4703 MAE, *p* = 0.96 and *p* = 0.10 against the 384 leader), while 128 and below were separated and worse (0.4930 MAE, *p* = 6e-10), so we took the smallest as our canonical CheMeleon embedding.

*(paste `results/figures/fig2.html` here as an HTML card)*

**Figure 2. Dimensionality reduction impact.** Left, descriptors: test set MAE varying the descriptor set (RDKit, Mordred, or both) and its PCA width (32, 64, 128, 256 components, or native). Right, embedding: the CheMeleon embedding at 32 to 512 components. Overlapping whiskers signal a pair cannot be distinguished. The shaded band spans the best row's interval. The **N283T** row is a published leaderboard value, drawn without an interval and not tested against. Hover on a point or row label to see detailed configuration information.

---

## Individual ingredient contributions

We then tested each candidate ingredient individually with TabPFN v3 as the regressor, following the same practice as the **N283T** report (**Figure 3**). The embedding from a CheMeleon encoder fine-tuned on log<sub>2</sub>FC achieved the best single-ingredient result here (0.4594 MAE). The off-the-shelf CheMeleon embedding (0.4659 MAE, *p* = 1.0), the same encoder fine-tuned on pEC<sub>50</sub> (0.4784 MAE, *p* = 0.85), and the log<sub>2</sub>FC-trained Chemprop embedding (0.4817 MAE, *p* = 0.69) are not separated from it, nor is the **Best graph network** (0.4902 MAE, *p* = 0.27). The single-stage **CheMeleon baseline** is separated (0.5017 MAE, *p* = 0.033), as are the **RDKit 128** descriptor block on its own (0.5309 MAE, *p* = 3e-05) and either encoder's log<sub>2</sub>FC readout columns (0.5308 and 0.5411 MAE, *p* = 3e-05 and 2e-06).

*(paste `results/figures/fig3.html` here as an HTML card)*

**Figure 3. Single ingredient tabular foundation model performance.** Test set mean absolute error (MAE) for each candidate feature in isolation, fed to TabPFN v3. Overlapping whiskers signal a pair cannot be distinguished. The shaded band spans the best row's interval. The **N283T** row is a published leaderboard value, drawn without an interval and not tested against. Hover on a point or row label to see detailed configuration information.

---

## Combining embedding, log<sub>2</sub>FC readout, and descriptors

We evaluated every subset of embedding, log<sub>2</sub>FC readout, and descriptor set, across five embeddings (CheMeleon off the shelf, CheMeleon fine-tuned on log<sub>2</sub>FC, CheMeleon fine-tuned on pEC<sub>50</sub>, and Chemprop trained from scratch on either target) and the two log<sub>2</sub>FC readouts those encoders supply. The top configuration was the **log<sub>2</sub>FC-fine-tuned CheMeleon embedding + Chemprop log<sub>2</sub>FC readout + descriptors** (0.4325 MAE, **Figure 4**). Fifteen of the thirty-six rows are not separated from it, including the two-block configuration that drops descriptors entirely (0.4358 MAE, *p* = 1.0). In other words, the descriptor block adds nothing detectable on top of an embedding and a readout. An embedding on its own is separated (0.4659 MAE, *p* = 0.031), as are the **Best graph network** (0.4902 MAE, *p* = 5e-08) and the single-stage **CheMeleon baseline** (0.5017 MAE, *p* = 2e-11). We therefore carried the off-the-shelf **CheMeleon embedding + Chemprop log<sub>2</sub>FC readout** (0.4358 MAE), which ties the leader while maintaining a much simpler implementation.

*(paste `results/figures/fig4.html` here as an HTML card)*

**Figure 4. Combining embedding, log<sub>2</sub>FC readout, and descriptors.** Test set MAE for every subset of the three ingredients (embedding, log<sub>2</sub>FC readout, and descriptors), all fed to TabPFN v3. Overlapping whiskers signal a pair cannot be distinguished. The shaded band spans the best row's interval. The **N283T** row is a published leaderboard value, drawn without an interval and not tested against. Hover on a point or row label to see detailed configuration information.

---

## Evaluating tabular foundation models

We then fixed the featureset at off-the-shelf **CheMeleon embedding + log<sub>2</sub>FC readout** (258 columns), and swapped only the regressor, to check whether a different tabular (or gradient boosting) model beats TabPFN v3 on the same inputs. TabICL comes out ahead (0.4269 MAE, **Figure 5**), with TabPFN v3 (0.4358 MAE) the only other regressor not separated from it (*p* = 1.0). TabFM (0.4800 MAE, *p* = 0.002), LightGBM (0.4809 MAE, *p* = 0.002), XGBoost (0.4941 MAE, *p* = 7e-05) and TabPFN v2.6 (0.5007 MAE, *p* = 1e-05) are all separated from the leader. We note that TabFM's number is not fully comparable: <code>max_num_rows</code> is capped at 500 in TabFM v1.0.0 because the full 4,392-row training set hits memory limits during its attention pass on the graphics processing unit (GPU) used here, so its in-context set is subsampled per ensemble member, whereas TabPFN and TabICL could accept all 4,392 rows.

As noted above, TabICL only leads TabPFN v3 nominally, 0.4269 against 0.4358 on the seed mean. However, TabICL is BSD 3-Clause, whereas TabPFN carries the Prior Labs Licence, Apache 2.0 with an added provision requiring "Built with PriorLabs-TabPFN" displayed on any related website, blog post or product documentation, and a TabPFN name prefix on models derived from it (as it applies to this very post). As the "Open" of OpenADMET signals, we prefer the more permissive license, all else equal. TabICL is not without its own downsides, though: it did not fit within our 24 GB GPU memory buffer, falling back to the central processing unit (CPU) at about 184 seconds per run, against 14.6 seconds for TabPFN v3 on the GPU for the same task. TabICL at 128 dimensions *does* fit on the GPU, at about 13 seconds per run.

*(paste `results/figures/fig5.html` here as an HTML card)*

**Figure 5. Tabular foundation model comparison.** Test set MAE with the featureset held fixed (**CheMeleon embedding + log<sub>2</sub>FC readout**, 258 columns), varying only the tabular foundation regressor. The swept rows cover two TabPFN releases (v2.6 and v3), TabICL, TabFM, and two gradient-boosted baselines (LightGBM and XGBoost). TabFM's row is not strictly comparable: its in-context set is capped at 500 rows to fit memory, while every other regressor saw all 4,392 training rows. Overlapping whiskers signal a pair cannot be distinguished. The shaded band spans the best row's interval. The **N283T** row is a published leaderboard value, drawn without an interval and not tested against. Hover on a point or row label to see detailed configuration information.

---

## Do tabular foundation models quantify uncertainty well?

Tabular foundation models also return a per-compound predicted distribution. TabICL reports it as quantiles, from which we took the 10th-to-90th percentile span converted to a normal-equivalent standard deviation. We used the off-the-shelf **CheMeleon embedding + log<sub>2</sub>FC readout** fed to TabICL to assess whether these predicted distributions inform uncertainty quantification, the subject of [a previous OpenADMET post](https://openadmet.ghost.io/concerning-uncertainty/). **Figure 6 (left)** plots each compound's absolute residual against its predicted standard deviation, testing whether a predicted standard deviation correlates with error. **Figure 6 (right)** plots empirical coverage against nominal level across 99 quantile levels to test whether, say, the predicted 0.4 quantile actually has 40% of true values at or below it. The spread did track error, but only weakly: the Spearman rank correlation between predicted standard deviation and absolute residual was 0.2616. Recalibrating the quantiles would not raise that correlation, because Spearman rho is rank-based and invariant to any strictly monotonic rescaling of the predicted standard deviation. The intervals themselves are close to the right width, running 3.6 percentage points above nominal coverage on average with a miscalibration area of 0.041, so the model is slightly underconfident rather than over. Thus, in this case, tabular foundation models give us intervals of roughly the right width, but not a reliable ranking of which compounds are hard, which is the part we would need.

*(paste `results/figures/fig6.html` here as an HTML card)*

**Figure 6. Predicted uncertainty versus actual error.** Left: each of the test compounds' predicted standard deviation, plotted against its absolute error. Right: the reliability diagram, observed coverage against nominal coverage across 99 levels, with the dashed diagonal marking perfect calibration.

---

## Applying the method to the CYP inhibition blind challenge

OpenADMET's next blind challenge, modeling cytochrome P450 (CYP) inhibition across 4 isoforms, [kicked off on August 17, 2026](https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/). Of course we wanted to apply the newly minted approach. This was our **TabICL-baseline**, which held the top spot on the regression live leaderboard (0.6755 mean-averaged, standardized target, relative absolute error, or MA-ST-RAE) for... a few days. As of this writing, the top performers have reduced that error by ~36% (against 0.4326 MA-ST-RAE). Still, the new method improves upon previous `openadmet-models` baselines by ~20% in this challenge (see the leaderboard's **CheMeleon-baseline** at 0.8335 MA-ST-RAE), well worth the effort to adopt.

---

## Updating `openadmet-models`

In order to realize PCA-transformed-CheMeleon-embeddings-concatenated-with-log<sub>2</sub>FC-predictions-input-into-tabular-foundation-model (exhale) approach from a YAML specification using `openadmet-models`, we implemented several new features.
- A `CheMeleonFeaturizer`, to produce the off-the-shelf, 2,048-dim CheMeleon embeddings.
- A PCA transformer that can be applied to all, or groups of, features listed in the featurization section of an anvil YAML. Or pass `None` for a given featurizer to enable direct passthrough of a subset.
- A `PretrainedModelFeaturizer` to read a trained model back in with another workflow to use that network's predictions as features. For example, train a Chemprop model on primary screen log<sub>2</sub>FC data, whose output then gets read into the main tabular foundation model as predicted log<sub>2</sub>FC features.
- A `TabICLRegressor` (and `TabICLClassifier`), to use TabICL for final predictions.

We have example configs up on [`optimus-prime`](https://github.com/OpenADMET/optimus-prime). [MAKE SURE THIS LINK GETS UPDATED TO TAKE THEM DIRECTLY TO THE RIGHT CONFIGS].

---

## Reproducibility

All supporting code lives in the [blog post repository](https://github.com/OpenADMET/pxr-challenge-tabicl). `experiments/manifest.yaml` is the spine: it declares every sweep, gate, and figure before anything runs, along with the axis levels this work deliberately excluded and why. `src/manifest.py` reads it, `src/gates.py` records the decisions, `src/regressors.py` and `src/encoders.py` build the models, and `src/panels.py` and `src/plots.py` produce every figure in this post. `src/tukey.py` computes the comparison intervals the figures draw. The numbered scripts in `run/` are the entry points, in the order they run.

To reproduce, install the dependencies with [`uv`](https://docs.astral.sh/uv/) and run from the repository root:

```shell
uv sync

# Data, split, features, and reductions (all cached and content-addressed)
python run/00_fetch_data.py
python run/01_build_split.py
python run/02_featurize.py
python run/03_reduce.py

# Each stage sweeps, then aggregation writes the gate the next stage reads
for stage in descriptor_width embedding_width ingredients regressor tabpfn_ensemble; do
    python run/04_sweep.py --stage "$stage"
    python run/05_aggregate.py
done

# The graph networks depend on no gate and can run at any point
python run/04_sweep.py --stage gnn

# Uncertainty diagnostics on the chosen configuration, then every figure
python run/07_uncertainty.py
python run/08_figures.py
```

## Acknowledgements

We would like to thank our funders for their support of OpenADMET, in particular ARPA-H, Radial (part of the [Astera Institute](https://ror.org/00ydx1s47)), Schrodinger Inc, and the Gates Foundation. We would also like to thank our partners Enamine, HuggingFace, OpenEye, CDD Vault, Discovery Life Sciences and the beamline staff at NSLS-II for their support.

This work is supported by the Advanced Research Projects Agency for Health (ARPA-H) under AVOID-OME, and Award Number 1AY1AX000035. The contents are those of the authors. They may not reflect the policies of the Department of Health and Human Services or the U.S. government. The content is solely the responsibility of the authors and does not necessarily represent the official views of the Advanced Research Projects Agency for Health.
