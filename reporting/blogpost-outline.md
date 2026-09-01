# Blogpost outline: are tabular foundation models all you need?

Rebuild scaffold for `blogpost.md`. Captures the narrative arc, each panel's job,
and the load-bearing numbers, so the post can be rewritten from scratch without
re-deriving the facts. Every metric is MAE on the challenge's own blind set (513
compounds), lower is better, unless noted.

## Through-line

The winning idea was not the architecture we set out to reproduce; it was where
one predicted feature ended up. We read the reports that beat us, chased a paper
one entry cited, found the paper contributed only an auxiliary log<sub>2</sub>FC
encoder, and learned the gain came from feeding that predicted readout into a
tabular foundation model. The post is the ablation that convinced us to support
the configuration in `openadmet-models`.

## Intro (narrative, first person)

- Bold lead: the OpenADMET PXR blind challenge closed; top of the leaderboard was
  crowded (paired-bootstrap could not separate the top ten at Phase 1 handover);
  our catalog was not among them, so we read the reports that beat us.
- We ran the challenge (OMSF contingent of OpenADMET), April to July 2026: same
  PXR induction assay data to every team, held-out 513-compound analog set, score
  predicted pEC<sub>50</sub> against unblinded truth. CheMeleon baselines land
  around 0.51 to 0.52 (panel 00) vs. roughly 0.41 for the leaders.
- What we wanted: not the lowest score, but a method reproducible as a YAML
  pipeline in `openadmet-models`. Contrast the fourth-place nine-member ensemble
  (five frozen graph encoders, two Boltz-2, two tabular, Caruana forward
  selection): a lot to package and rerun for an edge inside the leaderboard's
  bootstrap noise (SD roughly 0.02 MAE).
- N283T (fourth, 0.4113 MAE Phase 2, clear report) credited Buterez et al. 2024,
  a multi-fidelity transfer-learning paper: cheap low-fidelity labels as a proxy
  to sharpen scarce high-fidelity predictions. Maps onto the PXR assay funnel
  (single-concentration log<sub>2</sub>FC primary screen upstream of a few
  thousand dose-response pEC<sub>50</sub> values). We assumed the architecture
  carried the gain and set out to reproduce it.
- The turn ("It was not."): the paper informed exactly one component, an
  auxiliary encoder producing a predicted log<sub>2</sub>FC readout. None of the
  paper's own architectural variants (panel 00) carried the gain. The improvement
  came from feeding those predicted log<sub>2</sub>FC columns into a tabular
  foundation model (TabPFN first, then TabICL) alongside a molecular embedding.
  N283T reached the same conclusion (readout their single strongest feature).
- Why write it up: the combination moved us from roughly 0.51 to roughly 0.44
  (5-seed mean 0.4356 at best), simple enough for a first-class config. Roadmap
  the panels. Every panel scored on the challenge's own blind set, same scale as
  the leaderboard.

## Panels

Each panel: bold label, sentence-case H2 question, setup paragraph, figure card
(`reporting/figures/figure-0N.html`), findings blockquote. Panel 07 is prose-only
(no figure).

### 00: before tabular foundation models — where the non-TFM routes land, for scale
- Setup: reference points that read features through something other than a
  tabular foundation model (directly fine-tuned CheMeleon encoder; GNN initialized
  from a log<sub>2</sub>FC-pretraining checkpoint, following Buterez et al. 2024).
  Four log<sub>2</sub>FC rows cross frozen vs. fine-tuned body against
  dose-response-only vs. dose-response-plus-primary-screen data.
- Key numbers: concatenation architecture (best-of-18: freeze_epochs=2, 512-dim
  hidden, no clip) 0.5176 (range 0.4919-0.5339), the panel floor. CheMeleon
  baseline (no log<sub>2</sub>FC pretraining, straight to pEC<sub>50</sub>) 0.5215
  (range 0.5058-0.5407). Primary-screen + dose-response: freezing beats
  fine-tuning (0.5464 vs. 0.5573). Dose-response-only: fine-tuning beats freezing
  (0.5339 vs. 0.5849).
- Takeaway: freezing is not consistently better, it depends on the training-data
  regime; every row trails the weakest descriptor-including TFM combination by at
  least 0.0566 MAE (0.5176 vs. 0.4610, panel 02).

### 01: solo components — does each ingredient carry signal on its own?
- Setup: four feature blocks each read alone through a TFM (raw embedding, its
  predicted log<sub>2</sub>FC readout, RDKit descriptors, Mordred descriptors),
  mirroring N283T's single-embedding benchmarking. TabPFN v2.5 throughout except
  the "our best overall" reference row (TabICL 2.1.1, not on equal footing).
- Key numbers: raw embedding best single ingredient (0.4780). RDKit block, not the
  readout, is strongest descriptor-only row (0.5312 vs. 0.5410). Mordred block
  alone worst (0.5784, range 0.5751-0.5804). Off-the-shelf CheMeleon embedding
  (no fine-tuning) 0.5002, worse than from-scratch log<sub>2</sub>FC embedding but
  better than every descriptor-only row.
- Takeaway: the encoder's log<sub>2</sub>FC pretraining adds signal on top of
  CheMeleon's general-purpose embedding.

### 02: the path to the best performer — all seven combinations of embedding, readout, descriptors
- Setup: every non-empty subset of the three ingredients, descriptors fixed at
  Mordred, under both the from-scratch log<sub>2</sub>FC embedding and CheMeleon's
  off-the-shelf embedding (PCA-256). All TabPFN v2.5; "our best overall" (TabICL)
  shown for scale but reads a different, leaner featureset.
- Key numbers: best TabPFN subset is embedding + readout + descriptors [CheMeleon]
  0.4437, over the log<sub>2</sub>FC-embedding version 0.4574 and no-embedding
  readout + descriptors 0.4531. TabICL 0.4356 beats every TabPFN row here.
  Top rows overlap on ranges (0.4381-0.4476, 0.4485-0.4590, 0.4460-0.4722), so
  "CheMeleon wins" is soft. Mordred block alone worst (0.5784). For CheMeleon's
  embedding, dropping the readout costs far more than dropping descriptors (0.4444
  vs. 0.5290, vs. 0.4437 with all three); with the from-scratch embedding the
  readout looks near-redundant (0.4574 vs. 0.4610).

### 03: same features, different regressor — which tabular model reads the featureset best?
- Setup: not a feature ablation; hold panel 02's 386-column winner (CheMeleon
  embedding + readout + descriptors) fixed, swap the final tabular model. TabICL
  OOMs at 386 columns, so it alone reads a leaner near-tied featureset (CheMeleon
  embedding + readout, 0.4356 vs. 0.4437 under TabPFN v2.5 on the full set).
- Key numbers: TabICL v2.1.1 best overall (0.4356). Among the shared-featureset
  rows, TabPFN v2.5 (0.4437) and v3 (0.4436) a dead heat; v2.6 trails (0.4571);
  TabFM 0.4795, LightGBM 0.4987, XGBoost 0.5311. TabFM capped at max_num_rows=500
  (full 4,139-row set OOMs its attention pass), so not fully apples-to-apples.
- Takeaway: basis for reading "our best overall" through TabICL, not TabPFN.

### 04: the PCA compression sweep — why the descriptor block is PCA-compressed at all
- Setup: raw Mordred about 1,613 columns, RDKit adds 217, CheMeleon embedding
  2,048-dim; feeding any uncompressed OOMs TabPFN's between-items attention (scales
  with feature count and row count). Embedding side solved with a 256-dim
  from-scratch encoder; PCA applied only to descriptors, fit on train split, kept
  under TabPFN's 2,000-feature limit (roughly 500 features practical headroom).
  PCA on median-imputed, unstandardized values, so a few high-magnitude descriptors
  dominate covariance; every width captures 99.998-100.000% of raw variance
  (variance saturated by 64 components, but MAE keeps changing with width).
- Key numbers: 128 components lowest mean in both sources (Mordred-only 0.4574,
  range 0.4460-0.4722; RDKit + Mordred 0.4581, range 0.4537-0.4624); 64 worst, 256
  between; not monotonic, ranges overlap so "128 best" is soft. Mordred-only 128
  and RDKit + Mordred 128 differ by 0.0007, so adding RDKit buys no measurable
  accuracy.
- Takeaway: the page uses Mordred alone across concatenation panels (00-02) as a
  simplicity argument, not a demonstrated accuracy win.

### 05: out-of-fold isotonic calibration — repeating the N283T report's post-hoc step
- Setup: N283T fit an isotonic map on OOF predictions, closing a similar gap for a
  subset of their ablations (~0.441 to ~0.408). Repeat against our best config
  (CheMeleon embedding + log<sub>2</sub>FC readout, TabICL): per seed, 5-fold split
  the dose-response records, retrain the readout encoder per fold (embedding frozen
  and extracted once, no leakage), fit isotonic on pooled OOF, apply to blind set.
  OOF MAE not shown (computed on raw per-record dose-response, not per-compound
  pEC<sub>50</sub>, so not comparable).
- Key numbers: calibration makes this config worse (0.4356 to 0.4507). potent
  (>= 6.0) subset worse too (0.6882 to 0.7402), stays miscalibrated (R^2 about -9
  to -9.4).
- Takeaway: TabICL already near-calibrated here, so the map adds fold-to-fold
  noise. Reserve calibration for configs with a demonstrated gap.

### 06: is TabPFN's uncertainty trustworthy? — checking predicted distribution against error
- Setup: output_type="full" exposes a per-compound predicted distribution
  (discretized Riemann, not Gaussian): mean, std, quantile grid. Refit best config,
  predict on 513 blind compounds. Left chart: absolute residual vs. predicted std.
  Right chart: reliability diagram (empirical coverage vs. nominal quantile across
  9 quantiles).
- Key numbers: Spearman rho(|residual|, predicted std) = 0.2472 (n = 513), weak
  positive. Coverage below diagonal at all 9 levels, worst near median (gap about
  -0.10 to -0.11 at 0.4-0.6), so quantiles run systematically low.
- Takeaway: recalibration fixes the directional bias but can't raise the rank-based
  0.2472 correlation.

### 07: limitations — where this comparison could be misleading us (prose only)
Ordered by how much fixing each would move the headline number:
- Our encoder's hyperparameters were never tuned against this objective (N283T
  tuned theirs with Optuna against downstream pEC<sub>50</sub> OOF MAE at the same
  256-dim width). Leading candidate for the ~0.020 gap to their single-pipeline
  target (0.437 vs. our 0.4574).
- The potent (>= 6.0) subset is small, unstable, unaddressed: 31 of 513, every
  config scores badly (MAE 0.8-1.0, R^2 about -12 to -17), sensitive to a few
  outliers; open problem (issue #36).
- PCA compression is a memory workaround, not a modeling choice: raw columns OOM
  TabPFN's attention on this GPU; some discarded-variance signal could be
  recoverable with more memory, tiled attention, or a regressor without the
  feature-count ceiling.
- Only one embedding architecture tried, though it's the best on its own: in
  N283T's single-embedding comparison ChemProp best (0.437) over KERMT (0.448),
  GatedGCN (0.474), MoLFormer (0.475), AttentiveFP (0.484); their jump to ~0.408
  came from ensembling all plus two Boltz-2 members, out of scope here.

## Footer

Data provenance: list the result directories backing each panel (winner
`results/tabicl_embed_readout_mordred_pca128`, etc.), attributed to the PXR
challenge, pxr-challenge repo.

## Voice and hygiene reminders (from the rules)

- Bold thesis opening, first-person "we", numbers woven into the claims they
  support, descriptive sentence-case headings, inline links wrapped in the named
  noun. Explain the mechanism; no quips; no concluding summary.
- Subscript potency and readout notation (pEC<sub>50</sub>, log<sub>2</sub>FC);
  keep flat forms only in code spans and identifiers. Name the blind split.
- No em-dashes, en-dashes, or sequential hyphens; a colon introduces a list or a
  restatement, not an "and here's why" joint. Quantify every intensifier.
