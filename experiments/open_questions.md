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

**The graph networks refit on the whole fit set, in two passes inside one run.**
The first pass trains on `fit_train` and lets early stopping choose an epoch
count; the second reinitialises from the same seed and retrains on `fit_train`
plus `fit_val` for exactly that many epochs, with no validation partition and no
early stopping, and the scored predictions come from that model. Both passes are
recorded. This removes the 20% handicap the graph networks carried against the
tabular models, and it matches what the challenge entry did: it retrained on the
training set plus the released phase-1 compounds before submitting.

The same rule now applies to every model in the pipeline that early-stops: the
two log2FC encoders, the pEC50 encoder, and the auxiliary encoder inside the
concatenation architecture. For the log2FC encoders it is the cleanest of the
lot, since their labels come from a different assay and no pEC50 value is
involved on either side, so the extra rows are free of any leakage argument.

The graph network's second pass reuses its auxiliary encoder rather than
retraining it, so the refit adds only the main model's cost, far less than a
doubling.

Two details worth knowing. The second pass runs the same *number of epochs* on
25% more data, so it takes more gradient steps than the first; matching steps
instead would be the alternative, and the record carries both the epoch count
and the row count so the choice is visible rather than implied.
`TrainingConfig(refit_on_all=False)` reproduces the single-pass behaviour.

## Open questions

Three decisions are deferred and tracked on PR #2 rather than here: TabICL's
per-featureset batch size, how TabFM's capped row count is reported, and whether
to reproduce the N283T report's own calibration. They are listed there so this
file holds only what is still unassigned.

**The pEC50 encoder refits on everything too, under a uniform rule.** The rule
adopted is that anything deciding how many epochs to run does so on held-out
rows and then retrains on everything for that count. That covers the pEC50
encoder, and it is worth being explicit that this makes its embedding a
function of every training label rather than 80% of them, which strengthens a
caveat already on record: cross-validation *inside the fit set* using that block
reads optimistically. Scoring against phase 2 is unaffected either way, because
no phase-2 label ever reaches the encoder. `EncoderConfig(refit_on_all=False)`
reverses it for that block alone if the trade turns out not to be wanted.

**Figure 5 has nothing to sweep if the winning configuration carries no
descriptor block.** The manifest says the stage is skipped in that case. The
alternative is to sweep width on the best descriptor-bearing configuration
instead and say so. Currently unresolved because it depends on a result.

**The `openadmet-models` version string disagrees with its checkout.** The
installed editable package reports `0.2.0+3.gba3d506` while the working tree is
on `4a8df806`, the pinned commit. Provenance records the commit as well as the
version, so results are attributable, but anyone reading the version alone would
be misled.

**The auxiliary encoder has seen some validation structures under a different
measurement.** The log2FC screen covers 565 of the 878 compounds in `fit_val`.
That is not pEC50 leakage, since the screen is a different, public assay and no
phase-2 compound is in it at all, but it does mean the auxiliary encoder has
been trained on structures the main model then early-stops on. The prior design
did the same. Scoring against phase 2 is unaffected; the question is whether
the validation loss the main model stops on is slightly optimistic, and whether
that is worth saying in the write-up.

**Predictions now come from the best-validation epoch, not the last.** The prior
ran without checkpointing and predicted from wherever training happened to stop,
several epochs past its best. That is a real improvement rather than a
reproduction, so it is flagged rather than absorbed;
`TrainingConfig(restore_best=False)` reproduces the old behaviour exactly if a
like-for-like comparison is ever wanted.

**Figure 1's cost is dominated by retraining the auxiliary encoder.** Measured
at roughly 200 s per concatenation run, so 125 runs is about 7 hours serial, and
most of each run is the auxiliary encoder, which is retrained per cell because it
depends on the seed. Caching it per seed and task set across cells would cut
around 150 s from each of the 22 concatenation cells, saving roughly an hour.
Not done, because it trades a simple story for a faster one.

**The N283T report's cross-validation was a UMAP cluster split**, Morgan
fingerprints through UMAP into KMeans, not a random carve-out. If a validation
partition here only picks an epoch count, a random split is defensible. If it is
ever read as an estimate of generalization, a cluster split is the stricter and
more comparable choice.

**Their phase-2 submission retrained on train plus the released phase-1
compounds**, about 4,393, which is exactly this repository's `fit_all` at 4,392.
That settles what a production run should train on, and it is the argument for
removing the graph networks' 20% handicap rather than living with it.

## Early numbers, not results

One seed of three graph-network cells was run to measure wall clock:
`chemeleon_pec50` 0.518, `concat_freeze2_hd512_clipoff` 0.504,
`concat_predicted_readout` 0.542 phase-2 MAE, against a training-mean baseline of
0.959. These are one seed each, they are not a sweep, and they must not be read
against the leaderboard anchor.
