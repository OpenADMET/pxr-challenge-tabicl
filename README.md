# PXR challenge: are tabular foundation models all you need?

A rebuild of the OpenADMET PXR (pregnane X receptor) blind-challenge analysis,
run on the challenge's own evaluation split. It sweeps tabular featuresets and
regressors, including the tabular foundation models, against fine-tuned
message-passing graph networks, and scores every configuration the way the
leaderboard did.

The tracked checklist lives on the pull request:
<https://github.com/OpenADMET/pxr-challenge-tabicl/pull/2>.

## The split

Fit on the dose-response training set plus phase 1 (4,392 compounds), score on
phase 2 alone (260 compounds). That is the split the challenge used, so a number
produced here is comparable to what participants reported on the leaderboard.

The earlier analysis scored phase 1 and phase 2 pooled (513 compounds). Those
numbers rank configurations against each other and cannot be read against the
leaderboard, which is why `experiments/prior_summary.csv` is kept as context and
decides nothing about what runs.

Imputation and PCA are fitted on the fit partition alone and then applied to
every molecule, test set included. That is the one place in the pipeline where a
leak could enter, so it is the one place the fit rows are passed explicitly
rather than inferred. The isotonic calibration arm is fitted on a held-out slice
of the fit partition rather than on predictions the model was trained on; that
arm is declared but not yet run, and the manifest says why.

Compounds are matched across every stage by a canonical SMILES computed once, so
a block joined to another block is joined on structure rather than on the string
a file happened to carry.

## Environment

```bash
uv sync
```

That installs everything, including the modelling stack: PyTorch, chemprop and
the tabular foundation models TabPFN, TabICL and TabFM. The result-affecting
packages are pinned to exact versions rather than floored, because they decide
the numbers; every artifact records the versions it was built under, and a
resolver free to take a newer minor would hand you a stack that never produced
the results here.

One thing `uv sync` cannot decide for you is which PyTorch build your
accelerator needs. The lockfile carries the default `torch` wheel from PyPI. On
an AMD card, install the ROCm build over it:

```bash
uv pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch==2.5.1
```

The CheMeleon weights download on first use and cache under `~/.chemprop`.

Run everything below from the repository root. No `PYTHONPATH` is needed: the
`run/` scripts add `src/` to the path themselves, and pytest is configured with
it in `pyproject.toml`.

```bash
python -m pytest                 # fast tests
python -m pytest -m slow         # trains models; minutes
python -m pytest -m gpu          # fits foundation models; needs an accelerator
ruff check . && ruff format --check .
```

## Run order

Every script writes into a content-addressed cache or a run directory named by a
hash of the specification that produced it. A rerun skips work whose recorded
specification already matches, so an interrupted sweep restarts safely, and a
change upstream renames everything downstream and rebuilds it. `--force` is the
only way to overwrite completed work.

| Step | Produces | Cost |
|---|---|---|
| `run/00_fetch_data.py` | raw challenge CSVs in `data/raw/`, from the `openadmet/pxr-challenge-train-test` dataset on Hugging Face | seconds |
| `run/01_build_split.py` | the four split CSVs in `data/splits/`, the only tracked contents of `data/` | seconds |
| `run/02_featurize.py` | feature blocks in `data/features/`, one table per block | RDKit and Mordred take minutes on CPU; the four encoder-derived blocks train a network per seed and want a GPU |
| `run/03_reduce.py` | reductions in `data/reduced/`, each fitted on the fit partition | minutes, and optional: the sweep builds any reduction it needs on demand |
| `run/04_sweep.py` | a run directory per configuration and seed under `results/`, each holding predictions, metrics and a provenance record | the bulk of the compute |
| `run/05_aggregate.py` | `results/results.parquet`, a coverage report, and any gate the completed stage settles | seconds |
| `run/06_figures.py` | the manifest's figures, each a slice of the completed table | seconds |

`run/02_featurize.py` and `run/04_sweep.py` both take `--seeds`, which is how a
run is spread across processes. Split by seed and never by block or cell: seeds
share no artifact, whereas two processes on different cells of the same seed
would contend for one cached encoder.

```bash
seq 0 4 | xargs -P 5 -I{} sh -c \
  'python run/04_sweep.py --stage gnn --seeds {} > /tmp/gnn-seed{}.log 2>&1'
```

Do not aggregate while a sweep is in flight. `run/05_aggregate.py` reads every
run directory and refuses one whose record is missing, which is exactly what a
run in progress looks like.

## Stages and gates

`experiments/manifest.yaml` is the single source of what runs. It declares the
axes and the stages that sweep them rather than enumerating configurations,
because the cross product is a better description than several hundred
hand-written entries.

`experiments/prior_summary.csv` sits beside it as a check that nothing was
forgotten, and a test pins that its rows are the pooled 513-compound evaluation
so they cannot be mistaken for numbers comparable to these. That check is read
by hand rather than enforced: two prior configurations vary a training-data
axis this rebuild does not have, and no cell reproduces them.

Each stage sweeps every level of the dimensions it names and holds fixed only
what an earlier stage settled. A stage that settles something writes a **gate**:
a record in `results/gates/` naming the chosen level, the ranking behind it, the
configurations a paired bootstrap could not separate from the leader, and the
budget the tie-break used. Later stages read that file. Gates are tracked in git,
because they are what a later stage runs against and a manifest cannot be
reviewed without them.

| Stage | Sweeps | Runs | Settles |
|---|---|---|---|
| `descriptor_width` | descriptor block against width | 60 | `canonical_descriptors` |
| `embedding_width` | CheMeleon embedding width | 30 | `embedding_reduction` |
| `ingredients` | which blocks, and which combination | 95 | `best_featureset` |
| `regressor` | seven regressors, on the winning featureset | 35 | `best_regressor` |
| `calibration` | isotonic against none | 10 | held; see the manifest |
| `uncertainty` | nothing; reads runs already written | 0 | |
| `gnn` | graph-network cells | 50 | no gate |

Run order is not figure order. The two width probes run first because they need
no trained encoder, and they are reported late.

```bash
python run/04_sweep.py --stage ingredients --dry-run   # print the plan
python run/04_sweep.py --stage ingredients
python run/05_aggregate.py
```

A gate is decided from completed runs, not chosen by hand. `--fix AXIS=VALUE`
overrides one for a deliberate off-plan run, and `--regate` re-decides a gate
that already has a record. Neither is part of the normal path.

The gate rule is two bars, because one is not enough. A paired bootstrap over
the 260 phase-2 compounds says which configurations the leader does not separate
from, but at that sample size it fails to separate differences ten times the
run-to-run noise. So a cheaper configuration may displace the leader only if its
gap is also no larger than the leader's own standard deviation across seeds: the
saving has to cost less than changing the training seed does.

## Reading the results

`results/results.parquet` holds one row per run: the configuration axes, the
seed, every metric, the input block keys, and the commit that produced it.
`src/aggregate.py` summarizes it two ways, answering different questions.

The **seed summary** gives each configuration's mean and spread across seeds,
which says how much a result moves when only the training seed changes. The
**ensemble row** averages the per-compound predictions across seeds and scores
that once, which is what a leaderboard entry was, and so the only row comparable
to the anchor recorded in `experiments/manifest.yaml`. Ensemble rows carry
`aggregation = "ensemble"` and are never merged with the seed means.

`run/05_aggregate.py` also reports coverage against the manifest: which planned
runs are missing, and which runs on disk no stage called for. A partial sweep
reads as partial. A run directory whose record is missing or unreadable raises
rather than being skipped, because a silently dropped run looks exactly like a
configuration that was never planned.

## Layout

```
data/raw/          downloaded CSVs (ignored)
data/splits/       the four split CSVs (tracked)
data/features/     feature blocks, content-addressed (ignored)
data/reduced/      reductions, content-addressed (ignored)
data/encoders/     trained encoders and body checkpoints (ignored)
results/gates/     gate decisions (tracked)
results/           run directories and results.parquet (ignored)
experiments/       the manifest, the prior summary, open questions
src/               modules, resolved without an installed package
run/               numbered entry points
tests/             pytest suite
```
