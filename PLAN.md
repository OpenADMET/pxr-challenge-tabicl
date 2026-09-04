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
  regressors), `phase 2` for scoring, enforced in one `data.py` chokepoint.
  Expected sizes: train 4,392, test 260.
- **Dependencies**: vendor the minimal SMILES preprocessing and auxiliary
  encoder into `src/pxr/`, dropping the `moal` and `openadmet-models` imports.
  There is no dependency manifest in the repo today (the environment is
  unmanaged), so a real `pyproject.toml` and lockfile are authored from
  scratch. The CheMeleon baseline (Figure 1) is the heaviest vendored piece
  (a ChemProp foundation-model load); its exact approach is confirmed when we
  reach Figure 1.
- **Encoders**: retrain all five seeded auxiliary encoders from scratch on
  this branch, even though the log2FC pretraining is split-agnostic.
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
- [ ] `pyproject.toml` and lockfile authored from scratch (pinned: tabpfn
  8.2.0, scikit-learn 1.6.1, and the rest of the runtime)
- [ ] `fetch_data.py`: pull the training set, both test phases, and the
  primary-screen log2FC set from the Hugging Face dataset
- [ ] `src/pxr/data.py`: the single split point returning
  (train = training set + phase 1, test = phase 2), canonicalized once
- [ ] Leakage guard: assert train and test SMILES are disjoint and no phase-2
  pEC<sub>50</sub> reaches any fit
- [ ] Verification test for the split contract: train n = 4,392, test n = 260,
  the scorer only ever sees phase 2

### Phase 1: shared library
- [ ] Vendor `SMILESPreprocessor` (canonicalization) into `src/pxr/`
- [ ] Vendor the auxiliary log2FC encoder (training and embedding) into
  `src/pxr/`
- [ ] `features.py`: embedding, log2FC readout, RDKit, and Mordred builders,
  each fitting its imputer and PCA on the training split only
- [ ] `models.py`: regressor registry (TabPFN, TabICL, LightGBM, XGBoost)
- [ ] `sweep.py`: one parametrized runner writing a per-run `manifest.json`
  (spec axes, metrics, seed, git SHA, timestamp)
- [ ] `aggregate.py`: glob the manifests into `results.parquet`, replacing the
  directory-name provenance reconstructor
- [ ] Port the figure renderer to read the manifest-derived parquet
- [ ] `select.py`: rank a sweep and record the advancing configuration in
  `decisions.yaml`, which downstream sweeps read

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
