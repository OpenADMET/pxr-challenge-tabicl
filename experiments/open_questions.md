# Decisions taken without review, and questions still open

Written while building the pipeline unattended. Everything here is either a call
made to keep moving that deserves a second look, or a question that needs an
answer before the sweep is worth launching. Nothing here has been run.

## Decisions taken

**The first stage crosses featureset with regressor rather than sequencing them.**
Sweeping dimensions one at a time needs a starting point, and neither the
featureset nor the regressor can be fixed first without pre-committing to the
other. The only available basis for that pre-commitment is the prior sweep,
which is the one thing this rebuild will not lean on. Crossing them costs seven
times a single-regressor sweep, 273 configurations and 1,365 runs, and removes
the ordering problem outright. Figures 2, 3 and 4 then read that one table three
ways. Later stages do sequence, because width and calibration can be settled
against a configuration this split has actually chosen.

**The calibration arm runs in its own stage, not across the grid.** Across all
273 configurations it would double the tabular sweep, since an honest isotonic
map needs a second fit on held-out predictions. Confined to the configuration
the earlier gates settle, it costs ten runs. The trade is that calibration is
measured on the winner rather than characterised across the space.

**Descriptor width is held at 128 during stage one.** It matches the width the
previous generation used for nearly every descriptor row, so the stage-one table
is comparable to the prior's shape, and width gets its own sweep in stage two.

**RDKit and Mordred combine into one block before reduction.** A combined
descriptor featureset means the two concatenated and then projected once, which
is what the prior's 386-column featureset was: 256 embedding, 2 readout, 128
descriptors. Projecting them separately and gluing would be a different thing
with the same name.

**PCA's solver is seeded per replicate seed.** So seed-to-seed spread includes
the randomized solver's contribution rather than holding it artificially fixed.
It costs one reduction per seed instead of one overall.

**Rows with no pEC50 are dropped when a partition is loaded.** They are kept in
the split files, since featurization still wants the structures.

## Open questions

**TabICL's memory behaviour is not monotonic in batch size, and one setting does
not fit every featureset.** Measured at the production data shape: 130 columns
runs out of memory at the library default, while 258 columns fits in 13.5 s and
386 in 19.8 s on that same setting. The previous generation found 130 columns
needs batch size 2. So the setting has to be found per featureset. You said to
park this and revisit with memory flags or a CPU fallback; the adapter currently
takes batch size as an ordinary parameter, does not retry, and re-raises an
out-of-memory failure naming the resolved parameters, so nothing is hidden. The
sweep will stop on the first featureset that does not fit until this is settled.

**TabFM sees a fraction of the training set the other regressors see.** Its
attention would not fit 4,392 rows on this GPU, so in-context rows are capped at
500. That is a handicap to report next to its result rather than a property of
the model, and it means its row in the regressor comparison is not quite
like-for-like. Worth deciding whether to say so in the figure or only in the
text.

**Figure 5 has nothing to sweep if the winning configuration carries no
descriptor block.** The manifest says the stage is skipped in that case. The
alternative is to sweep width on the best descriptor-bearing configuration
instead and say so. Currently unresolved because it depends on a result.

**The `openadmet-models` version string disagrees with its checkout.** The
installed editable package reports `0.2.0+3.gba3d506` while the working tree is
on `4a8df806`, the pinned commit. Provenance records the commit as well as the
version, so results are attributable, but anyone reading the version alone would
be misled.

## Not yet verified

The encoder blocks and the vendored concatenation architecture were built in
parallel and their reports are not yet folded in. Their wall-clock per run is
what sets figure 1's cost, and figure 1 is the expensive half of the sweep.
