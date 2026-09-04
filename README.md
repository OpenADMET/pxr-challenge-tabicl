# PXR challenge: are tabular foundation models all you need?

A rebuild of the OpenADMET PXR (pregnane X receptor) blind-challenge analysis,
run on the challenge's own evaluation split. It sweeps tabular featuresets and
regressors, including the tabular foundation models, against fine-tuned
message-passing graph networks, and scores every configuration the way the
leaderboard did.

The tracked rebuild checklist lives on the pull request:
<https://github.com/OpenADMET/pxr-challenge-tabicl/pull/2>.

## The split

Fit on the dose-response training set plus phase 1 (4,392 compounds), score on
phase 2 alone (260 compounds). That is the split the challenge used, so a
number produced here is comparable to what participants reported on the
leaderboard.

The earlier analysis scored on phase 1 and phase 2 pooled (513 compounds).
Those numbers rank configurations against each other and cannot be read against
the leaderboard, which is why `experiments/prior_summary.csv` is kept as
context and decides nothing about what runs.

Reductions (imputation and PCA) are fitted on the fit partition alone. Isotonic
calibration is fitted on a held-out slice of the fit partition, never on
predictions the model was trained on.

## Environment

The modelling stack is not reproducible with `uv sync`. The tabular foundation
models, chemprop and the CheMeleon weights live in a shared virtual environment
beside this checkout:

```
/home/sean/projects/openadmet/.venv/bin/python
```

Use that interpreter for everything below. Modules resolve from `src/` without
an installed package, so tests and ad-hoc scripts need `PYTHONPATH=src`; the
`run/` entry points bootstrap it themselves.

```bash
PYTHONPATH=src /home/sean/projects/openadmet/.venv/bin/python -m pytest
```

## Run order

Each script writes into a content-addressed cache or a run directory named by
what produced it. A rerun skips work that is already complete and matches its
recorded specification, so an interrupted sweep can be restarted safely, and a
changed upstream block is redone automatically because it has a different
specification. `--force` is the only way to overwrite work that is already
complete.

| Step | Produces | Cost |
|---|---|---|
| `run/00_fetch_data.py` | the raw challenge CSVs in `data/raw/` | seconds, one download |
| `run/01_build_split.py` | the fit, fit_train, fit_val and test_phase2 CSVs in `data/splits/` | seconds |
| `run/02_featurize.py` | stage-one feature blocks in `data/features/`, one table per block over every molecule in the split | minutes for RDKit and Mordred; the CheMeleon pass and the trained-encoder blocks want a GPU and are the expensive part, since a seeded block is trained once per seed |
| `run/03_reduce.py` | stage-two reductions in `data/reduced/`, fitted on the fit partition | minutes; optional, because the sweep builds any reduction it needs on demand |
| `run/04_sweep.py` | one run directory per configuration and seed under `results/tabular/`, each holding `predictions.csv`, `metrics.json` and `run.json` | the bulk of the compute: stage one alone is 273 configurations at 5 seeds, and the foundation-model regressors need a GPU |
| `run/05_aggregate.py` | `results/results.parquet` and a coverage report | seconds |

Stage one sweeps the featureset and the regressor together. Later stages hold
fixed what an earlier gate settled, and take those axes on the command line:

```bash
python run/04_sweep.py --stage featureset_and_regressor --dry-run
python run/04_sweep.py --stage featureset_and_regressor
python run/05_aggregate.py --stage featureset_and_regressor

python run/04_sweep.py --stage descriptor_width \
    --fix embedding=chemeleon --fix readout=none \
    --fix descriptors=rdkit_mordred --fix regressor=tabicl
```

The `--fix` values above are placeholders for whatever the gate actually
chooses; `experiments/manifest.yaml` states each gate's rule.

## Reading the results

`results/results.parquet` holds one row per run: the six configuration axes,
the seed, every metric, the input block keys and the commit that produced it.
`src/aggregate.py` summarizes it two ways, and they answer different questions.
The seed summary gives each configuration's mean and spread across seeds, which
says how much a result moves when the training seed changes. The ensemble row
averages the per-compound predictions across seeds and scores that once, which
is what the leaderboard entry was and so the only row comparable to the anchor
in `experiments/manifest.yaml`. Ensemble rows carry `aggregation = "ensemble"`
and are never merged with the seed means.

`run/05_aggregate.py` also reports coverage against the manifest: which planned
runs are missing, and which runs on disk no stage called for. A partial sweep
reads as partial. A run directory whose `run.json` is missing or unreadable
raises rather than being skipped, because a silently dropped run looks exactly
like a configuration that was never planned.
