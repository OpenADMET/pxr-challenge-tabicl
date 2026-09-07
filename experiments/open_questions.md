# Open questions

A staging area for questions raised while building, before they have an answer
or a home. An item belongs here only while it is genuinely unassigned. Once it
is answered or turned into planned work it moves to the pull request carrying that
work, which is the single source of truth for decisions and scope, and it is deleted
from this file rather than duplicated there.

## The calibration reproduces a method the report describes incompletely

The report gives the shape of its calibration and not its settings. Three
choices were made here to fill the gaps, all recorded in every calibration
record so a reader sees what was assumed rather than inferring it:

- **Morgan radius 2 and 2,048 bits.** The report names neither. These are the
  conventional defaults and `--radius` and `--bits` change them.
- **Logistic regression at C = 1.0 with balanced class weights** as the
  train-versus-test separator. The report names no classifier, but warns that
  "too strong a classifier lets a handful of test-like compounds pull the fit",
  which argues for a linear model. The classes are imbalanced 4,392 against
  260; without balancing every probability is pushed toward zero and every
  weight to the lower clip, which would make the ratio a statement about how
  few test compounds exist rather than about which compounds look test-like.
- **Plain k-fold rather than nested cross-validation.** The report says "5-fold
  nested CV". Nesting exists to select hyperparameters inside the outer loop,
  and nothing is selected here, so the inner loop would do no work. If a later
  stage adds per-fold model selection this has to be revisited.

Open: whether any of these three materially changes the fitted map. The
`--unweighted` flag gives the ablation for the weighting as a whole, but not
for the classifier's own settings.

## The calibration is fitted on fewer seeds than it is applied to

Out-of-fold predictions cost a refit per fold. At the chosen default of one
seed per fold that is five fits, and the resulting map is applied to a
five-seed ensemble. Ensembling shrinks prediction variance, so a slope fitted
on single-model residuals is mis-scaled for ensemble predictions, in the
direction of being too small. `--oof-seeds 5` removes the mismatch at
twenty-five fits, and every record carries `ensemble_mismatch.matched` saying
which was run.

Open: whether the mismatch moves the fitted slope enough to matter. One run of
each on the winning configuration would answer it and has not been done.

## The report's calibration numbers are not comparable to ours

The report improved MAE from 0.4209 to 0.4077 on the 513-compound pooled
evaluation. This pipeline applies the same method to the 260-compound phase-2
set. The direction may agree and the magnitudes cannot be set beside each
other, which the manifest's calibration note states.

Open: nothing to decide, but any writeup quoting both has to say it.

## Uncertainty has two spreads and they answer different questions

A regressor's own predictive spread and the standard deviation across the five
training seeds are both reported, and `run/07_uncertainty.py` scores both where
both exist. A graph-network cell has only the second, since the architecture
reports no spread of its own.

Open: which one a figure should lead with. The model spread is the one a single
deployed model could offer; the ensemble spread is the one every configuration
here has, so it is the only basis on which the graph networks and the tabular
models can be compared at all.

## Whether the uncertainty diagnostic should be run before calibration at all

`run/07_uncertainty.py` scores the spread before and after calibration when a
calibration exists. The affine map rescales the predictive distribution, so the
spread is multiplied by the absolute slope rather than left alone; a slope
above one widens the intervals and mechanically improves coverage without the
model having learned anything about its own reliability.

Open: whether the post-calibration miscalibration area is therefore worth
reporting, or whether it only restates the slope. The rank correlation between
spread and residual is unaffected by the rescaling, since it is monotone, so
that half of the diagnostic is safe either way.
