# PXR challenge rebuild plan

Tracked checklist for rebuilding the PXR blog pipeline onto the correct
evaluation split, with forward-emitted provenance and a self-contained,
reproducible layout. We commit to this list and tick boxes as work lands on
this branch.

## Why

The published comparison trains on the dose-response training set and scores
on phase 1 and phase 2 pooled (513 compounds). The challenge's
end-of-challenge leaderboard trained on `train + phase 1` and scored on
phase 2 alone (260 compounds), so N283T's 0.4113 and 0.437 are phase-2
numbers our pooled results were never comparable to. This rebuild moves every
experiment onto `train + phase 1` then `phase 2`, which makes the N283T
comparison sound and is the reason for the redo.

Every result on `main`, including the recent 5-seed CheMeleon baseline, is on
the pooled split and is superseded here. No old number carries over.

## Decisions locked

- **Split**: `train + phase 1` for every fit (imputers, PCA, encoders,
  regressors), `phase 2` for scoring. Enforced declaratively by each anvil
  recipe's train and test resource paths, which point at a `train + phase 1`
  CSV and a `phase 2` CSV built once. Expected sizes: train 4,392, test 260.
- **Dependencies**: depend on `openadmet-models` at the
  `preview/chemeleon-pca-tabicl` branch, pinned to a commit SHA. It now
  provides the anvil recipe system and the `openadmet anvil` runner, every
  featurizer the featuresets need (CheMeleon embedding, RDKit and Mordred
  descriptors, PCA reduction, trained-model embedding, feature
  concatenation), all four Figure-4 regressors (TabPFN, TabICL, LightGBM,
  XGBoost), and regression and uncertainty eval. Most sweeps become YAML
  recipes run through `openadmet anvil`. The `moal` dependency is dropped:
  canonicalization is anvil-native, and the log2FC-trained embedding becomes a
  ChemProp model trained on the log2FC target and read back through
  `TrainedModelFeaturizer`. No dependency manifest exists in the repo today, so
  a real `pyproject.toml` and lockfile are authored from scratch declaring the
  pinned `openadmet-models` git dependency.
- **Vendored**: only the Figure-1 concatenation architecture, the
  auxiliary-encoder graph network with a jointly trained log2FC-readout head
  and a freeze, hidden-width, and gradient-clip schedule, has no anvil
  equivalent. It is vendored as self-contained Lightning code under `src/pxr/`.
- **Encoders**: retrain the log2FC encoders from scratch on this branch (via
  anvil for the tabular embedding ingredient, via the vendored architecture
  for Figure 1), even though the log2FC pretraining is split-agnostic.
- **Seeds**: seed 0 first as a sanity smoke test that also counts as a real
  member, then seeds 1-4, for five per configuration. Every published bar has
  five seeds and whiskers.
- **Significance at gates**: a paired bootstrap over the 260 phase-2 compounds
  decides between candidate configurations, alongside the five-seed spread.

## Checklist

### Phase 0: scaffold and correctness spine
- [x] Orphan branch `rebuild` cut from `main` (root: `.gitignore`,
  `challenge.pdf`)
- [x] This tracking PR opened against `rebuild`
- [ ] `pyproject.toml` and lockfile authored from scratch, declaring
  `openadmet-models @ preview/chemeleon-pca-tabicl` pinned to a commit SHA
  (which pulls tabpfn 8.2.0, scikit-learn 1.6.1, and the rest of the runtime)
- [ ] `fetch_data.py`: pull the training set, both test phases, and the
  primary-screen log2FC set from the Hugging Face dataset
- [ ] `build_split.py`: write the two resource CSVs the recipes point at, a
  `train + phase 1` file and a `phase 2` file, canonicalized once
- [ ] Leakage guard: assert train and test SMILES are disjoint and no phase-2
  pEC<sub>50</sub> reaches any fit
- [ ] Verification test for the split contract: train n = 4,392, test n = 260,
  the scorer only ever sees phase 2

### Phase 1: recipes, provenance, and the one vendored architecture
- [ ] Vendor the Figure-1 concatenation architecture (auxiliary-encoder graph
  network, freeze, hidden-width, and gradient-clip schedule) as self-contained
  Lightning code under `src/pxr/`
- [ ] Recipe templates under `recipes/`, one per featureset, regressor, and
  PCA-width axis, parametrized by seed and by the train and test resource CSVs
- [ ] `aggregate.py`: read each anvil run directory's recipe and metrics into
  `results.parquet`, with no directory-name archaeology
- [ ] Port the figure renderer to read the aggregated parquet
- [ ] `select.py`: rank a sweep (five-seed mean plus a paired bootstrap over
  the 260 phase-2 compounds) and record the advancing configuration in
  `decisions.yaml`, which downstream recipes read

### Phase 2: figures
Each figure: run seed 0 across the sweep as a sanity check, backfill seeds 1-4,
then pause at a decision gate to record the configuration that advances.
- [ ] Figure 1: graph-network baselines, plus the CheMeleon baseline retrained
  on the new split
- [ ] Figure 2: single-ingredient tabular features (embedding, readout, RDKit,
  Mordred)
- [ ] Figure 3: combinations of embedding, log2FC readout, and descriptors.
  Gate: the winning featureset
- [ ] Figure 4: regressor comparison at Figure 3's featureset. Gate: "our best
  overall"
- [ ] Figure 5: descriptor PCA width sweep
- [ ] Figure 6: isotonic calibration on "our best overall"
- [ ] Figure 7: predicted uncertainty versus actual error for the best
  configuration

### Phase 3: publication
- [ ] Rewrite `blogpost.md` numbers from the new `results.parquet`; the N283T
  comparison is now phase-2 against phase-2
- [ ] Commit the final small `results.parquet` and figure HTML (`results/`
  stays gitignored)
- [ ] Top-level README: the exact runner order and decision gates for a fresh
  clone
- [ ] Retire the archaeology artifacts (`NEXT_STEPS.md`,
  `run_provenance_report.md`, `blogpost_verification.md`, `moal.ipynb`)
