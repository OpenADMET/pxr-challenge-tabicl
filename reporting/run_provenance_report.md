# Run provenance audit

Built by `reporting/build_run_provenance.py`, output at `results/run_provenance.parquet`
(419 rows: one per `results/<base_dir>_seed{n}/` sweep directory, plus one combined row for
the anvil-recipe CheMeleon->pEC50 baseline, ingested from `pxr_baseline_predictions_phase{1,2}`
(see "Fifth pass" below). `seed` is `-1` for a run with no explicit seed pin, see
"Third pass changes" below).

## Fourth pass: convention-layer orphan resolution

The user asked for a hand retriage of the 54 "all axes unknown" orphans against the launch
bash scripts, their supporting Python scripts, `NEXT_STEPS.md`, and the configs (comments
included), to resolve them rather than leave them quarantined. Every one of the 54 resolves.
They were never unresolvable: they orphaned only because their output-dir names are not emitted
verbatim by any *current* launch script. Two populations account for all 54:

- **Pre-seed-sweep single runs** (24 rows, unseeded originals at `seed = -1`, plus the one
  manual `tabicl_embed_readout_mordred_pca128_seed5` replicate beyond the 0..4 range). Each is
  the single run whose later `_seed0..4` replicates the launch table already resolves; each is
  described by name in `NEXT_STEPS.md`'s "Where we landed" table.
- **Rewritten-recipe runs** (30 rows, `_seed0..4`). Later edits to the recipe scripts dropped
  these invocations: recipe 7 moved the pinned-TabPFN-version regressor sweep off the
  embed+readout featureset onto the no-embed one and excluded TabICL (GPU OOM), and recipe 9
  excluded TabICL on the 386-column CheMeleon featureset. The earlier runs on disk (each with a
  completed `eval_out.csv`) predate those edits.

Resolution uses a new, explicitly lower-trust **convention** evidence layer
(`CONVENTION_INVOCATIONS` in the builder), kept separate from `LAUNCH_INVOCATIONS` so the
launch tier stays a pure verbatim transcription. Each convention entry is resolved from
`NEXT_STEPS.md`, the naming convention it documents, the invoked script's argparse defaults,
and the axes of a seeded sibling already in the launch table, then run through the **same**
axis resolvers as the launch tier. These rows are stamped `spec_source = "convention"`.

**Resolution rate: 226/417 -> 281/418 fully resolved (67%).** All 54 formerly-orphaned rows now
resolve every critical axis (`unknown_axes == ""`); the "all 14 axes unknown" group is empty and
no row has a null `spec_source`. `spec_source` now breaks down as launch_script 227,
parsed_config 113, convention 54, parsed_config+yaml_comment 24, anvil_recipe 1 (the fifth-pass
baseline, which brings the current totals to 282/419 fully resolved). Zero in-run-vs-launch
disagreements remain.

Three honest caveats travel with the convention tier:

- `tabpfn_pec50_chemeleon_full` never completed: only `embed_cache.npz` is on disk, no eval
  output, the 256+256 CheMeleon OOM documented in `NEXT_STEPS.md`. Its intended axes resolve
  (script defaults, embed PCA 256 + descriptor PCA 256, `n_features = 512`); its metric columns
  stay null because no run finished.
- `tabpfn_chemeleon_static` is pinned to PCA(128)+PCA(128) by `NEXT_STEPS.md` L53, a genuine
  pre-fix width that differs from its recipe-5 seeded siblings' 256/256 script defaults (so it
  carries a distinct fingerprint, correctly).
- `tabpfn_chemeleon_embed_only` has no independent width log; its embedding PCA width is
  inferred from its `tabpfn_chemeleon_static` companion (128) and is the single least-supported
  axis in the whole table.

The `## Quarantine list`, the four-pattern quarantine table, and the "54 true orphans" and
`bdb8b96a88b8d409` fingerprint discussion below are **superseded** by this pass and retained
only as the record of the prior state.

## Third pass changes

The user reviewed the second pass and gave three binding directives, applied as follows.

**Directive 1 (seed sentinel).** `seed` is no longer `NA`/nullable. `split_base_and_seed` now
returns the sentinel `-1` (a real `int64` value) for any directory with no `_seed{N}` suffix,
and also for the one out-of-range suffix in the table
(`tabicl_embed_readout_mordred_pca128_seed5`, whose "5" does not correspond to any actual
5-seed-sweep invocation). 52 of 417 rows now carry `seed = -1`. The prior pass's proposed
`--seed` default -> `cfg.seed` -> `seed: 42` fallback chain was **not** used, per the user's
explicit instruction; no row's seed is inferred from a config default. `build_table` now casts
`seed` to plain `int64` (not nullable `Int64`) and raises if any null slips through, so a
silent float/NaN coercion would fail loudly rather than passing unnoticed.

**Directive 2 (moal/chemprop regressor + hd/freeze/clip axes).** All 137 moal-schema rows
(`e4_*`, `freeze*`, `feat_*`) now get `regressor = "N/A"` instead of `regressor` unknown: an
end-to-end MPNN with a built-in FFN readout has no swappable regressor, which is a resolved
fact, not a gap. `unknown_axes` no longer lists `regressor` for any of these 137 rows.

Three new columns, `ffn_hidden_dim`, `freeze_epochs`, `gradient_clip_val`, are added and
backfilled for all 137 moal-schema rows from `model.ffn_hidden_dim`, `model.freeze_epochs`, and
`trainer.gradient_clip_val` in each row's config (`config_used.yaml` in-run, or the matching
`configs/<stem>.yaml` for the 226 rows resolved only via launch script). They sit outside
`CRITICAL_COLUMNS`/the fingerprint (adding them there would retroactively change every already
-resolved fingerprint's meaning for one pipeline family only). Per the user's e4_frozen.yaml
worked example ("if both apply, resolve both"), `ffn_hidden_dim` also fills
`embedding_native_dim` and `readout_dim` for every moal-schema row: this schema records no
separate MPNN pooled-embedding width, so `ffn_hidden_dim` is the only width it records for
either the embedding or the readout stage. This resolves `embedding_native_dim`/`readout_dim`
for all 137 rows, removing both from `unknown_axes` everywhere they previously appeared.
`n_features` stays unresolved for every moal-schema row on purpose: this pipeline has no
concatenated final feature vector fed to a downstream regressor (there is no downstream
regressor), so summing embedding/readout/descriptor widths the way the tabular pipeline does
would produce a number with no real referent, not a genuine resolution. A new
`n_features_inapplicable` flag on `RunSpec` forces this rather than letting the generic
width-summing logic fire now that the two widths above resolve.

**Directive 3 (YAML header comments as a provenance source).** `e4_finetune.yaml`,
`e4_finetune_drc_only.yaml`, `e4_frozen.yaml`, and `e4_frozen_drc_only.yaml` are the only
moal-schema configs with a header comment at all (`freeze*_hd*_clip*.yaml` and `feat_*.yaml`
start directly with `data:`, confirmed by reading every one of them, not just the worked
example). Three of the four e4_* headers state the log2FC-pretrain lineage directly
("...on top of a message-passing body pretrained on log2FC..."); the fourth
(`e4_finetune_drc_only.yaml`) only says "Same as e4_finetune.yaml, except..." without restating
it, so a follow-up cross-reference pass (`Same as (\S+)\.yaml` matched against already-resolved
stems) picks it up too, since it shares `e4_finetune.yaml`'s identical `from_foundation`
checkpoint path. For all four, `encoder_init` resolves to a new third category,
`log2fc_checkpoint_pretrained` (an MPNN checkpoint pretrained in-repo, distinct from
`chemeleon_pretrained` and `scratch`), and `encoder_target` resolves to `"log2fc"` (what the
frozen/warmup-frozen MPNN body was itself fit to in stage 1, consistent with how this column is
used everywhere else in the table — the encoder's own training target, not the downstream
readout's fine-tune target). `spec_source` for these 24 rows is `parsed_config+yaml_comment`,
kept distinct from plain `parsed_config` so the comment-sourced resolution is auditable.

**e4_log2fc_pretrain cross-check: verified, passes.** (Correction: an earlier draft of this
report claimed `e4_log2fc_pretrain/` and its contents did not exist anywhere in this repo,
because the `find . -iname ...` check that produced that claim ran from a subdirectory rather
than the repo root and searched the wrong subtree. Re-run from the repo root, the directory and
all three referenced files are present, predating this table: `e4_log2fc_pretrain/log2fc_mp.pt`,
`log2fc_pretrain.yaml`, `results/`, and `convert_to_foundation_checkpoint.py`, all with
filesystem timestamps from 2026-07-27/28.) `e4_log2fc_pretrain/log2fc_pretrain.yaml`'s own
`ffn_hidden_dim: 512` matches `configs/e4_frozen.yaml`'s `ffn_hidden_dim: 512` exactly. The
stage-1 recipe's own header comment independently corroborates the lineage `e4_frozen.yaml`
claims: it describes pretraining a CheMeleon-initialized ChemProp encoder on log2FC and
extracting the message-passing weights via `convert_to_foundation_checkpoint.py` into
`log2fc_mp.pt`, matching `e4_frozen.yaml`'s `from_foundation: e4_log2fc_pretrain/log2fc_mp.pt`.
The `encoder_target="log2fc"` / `encoder_init="log2fc_checkpoint_pretrained"` resolution for
`e4_finetune`, `e4_frozen`, `e4_finetune_drc_only`, and `e4_frozen_drc_only` is now a verified
cross-check, not an unverified single-source claim.

**Resolution rate.** 226/417 rows were "fully resolved" (`unknown_axes == ""`) before and after
this pass — that headline number is unchanged, because `n_features` is now deliberately
unresolvable for all 137 moal-schema rows (directive 2 above), so no moal row can reach 0
unknown axes. The real movement is inside `unknown_axes`'s composition for those 137 rows:
- 24 rows (the four e4_* base_dirs) went from 5 unknown axes each
  (`encoder_init, encoder_target, embedding_native_dim, regressor, n_features`) to 1
  (`n_features` only) — every other axis directive 2/3 could reach is now resolved.
- 113 rows (`freeze*`, `feat_*`) went from 4-5 unknown axes each down to 2
  (`encoder_target, n_features`) — `regressor`, `embedding_native_dim`, and (where applicable)
  `readout_dim` are now resolved; `encoder_target` stays unknown because none of these configs'
  header comments name a pretrain lineage the way the e4_* ones do (confirmed: no header
  comment on any of them at all).
- The 54-row "all axes unknown" orphan group and the 226 launch-script-resolved rows are
  unchanged in resolution status; only their `seed` values changed (directive 1).

## Third pass changes (axis-widening continuation)

Two more columns and one cross-check were added on top of the three numbered directives above.

**`aux_use_observed_readout` / `aux_use_predicted_readout`.** Sourced from
`auxiliary_model.use_observed_readout` and `auxiliary_model.use_predicted_readout` in
`config_used.yaml`, resolved by `resolve_moal_config_axes` alongside the existing
`aux_cfg.get(...)`-derived `has_readout`. These genuinely vary: querying every
`results/*/config_used.yaml` for the pair returns six distinct combinations, including
`(True, False)`, `(True, True)`, and `(False, True)` across the
`feat_observed_2task` / `feat_observed_and_predicted_2task` / `feat_predicted_native_2task`
trio. Before this change, those three runs were indistinguishable in this table: `has_readout`
is `True` for all three (it is just the two flags' `or`), so nothing separated "observed only"
from "predicted only" from "both". Backfilled for every row `resolve_moal_config_axes` touches;
`None` (nullable `boolean` dtype) elsewhere, same as the other `has_*` columns.

`model.depth`/`auxiliary_model.depth` (candidate `encoder_depth`/`aux_encoder_depth` columns)
were checked against every `results/*/config_used.yaml` and dropped: every row that carries the
field has it pinned to `3`, so nothing in `results/` varies on it. `ffn_num_layers` (checked
across every `configs/*.yaml` too) is likewise constant (`3` for `model`, `2` for
`auxiliary_model`) everywhere it appears. Adding either as a column would add width, not
information.

**`_cross_check_freeze_hd_clip_naming`.** New function, called from `build_run_spec` after the
launch-script fill-in. `freeze{F}_hd{HD}_clip{C}` base_dirs encode `freeze_epochs`,
`ffn_hidden_dim`, and `gradient_clip_val` in the directory name itself, independent of
`config_used.yaml`; this parses that name and compares it against the resolved values, appending
to `spec.disagreements` (the same channel `_apply_launch_axes` uses) on any mismatch, so a
directory whose name drifted from its config would surface rather than silently pass. Run
against all 417 rows: **0 disagreements** for every `freeze*_hd*_clip*` base_dir, including the
`clipoff` variants (`gradient_clip_val: null` in the config, `None` parsed from `"clipoff"` in
the name).

**Resolution-rate delta from this pass: none.** `aux_use_observed_readout` and
`aux_use_predicted_readout` are not members of `CRITICAL_COLUMNS`, and `resolve_moal_config_axes`
always assigns them a concrete `bool` (never leaves them in `unknown_axes`), so no row's
`unknown_axes` or resolved status changed. `_cross_check_freeze_hd_clip_naming` only appends to
`disagreements`, likewise untouched. Resolution rate stays **226/417 (54.2%)**, matching before
this pass; the two new columns and the cross-check widen what the table records per row without
moving the headline number.

## Correction notice

This is a correction of a prior version of this table, which reported **0 of 417 rows
fully resolved** because it consulted only artifacts sitting *inside* each run directory
(`config_used.yaml`, `eval_out.csv`). That conclusion understated what this repo actually
records: the repo root's `run_seed_sweep*.sh`, `run_sweep.sh`, `run_cpu_oom_reproductions.sh`,
`run_gnn_seed_sweep.py`, and `run_tabfm_param_sweep.py` launch scripts hard-code, for a given
`results/<output-dir>`, exactly which Python script ran, with exactly which `configs/*.yaml`
file and exactly which CLI flags. Reading those scripts in full (every invocation, not a
sample) and each invoked script's own argparse definitions resolves every critical axis for
**226 of the 417 rows** that carried no in-run artifact at all. Combined with the 137 rows
already resolved from `config_used.yaml`, **226 of 417 rows (54%) are now fully resolved**,
up from 0.

## Fifth pass: CheMeleon->pEC50 baseline ingestion

The `pxr_baseline_predictions_phase{1,2}` directories are the CheMeleon->pEC50 baseline
(openadmet standard `ChemPropModel`, anvil-recipe pipeline), previously excluded because they
carry `anvil_recipe.yaml` + `regression_metrics.json` instead of the sweep's
`config_used.yaml` + `eval_out.csv`. Excluding them left the figures with no genuine baseline
row, so the CheMeleon-baseline slot was resolving to the unrelated `tabpfn_pec50_chemeleon`
run (a TabPFN regressor over a frozen CheMeleon embedding). This pass ingests the real baseline.

Axes resolve from `anvil_recipe.yaml` (`_resolve_anvil_axes`): `from_foundation: chemeleon`
gives `encoder_init = chemeleon_pretrained` and `encoder_family = chemprop`; the single-task
`pEC50` target gives `encoder_target = pec50`; the end-to-end `ChemPropModel` (no downstream
regressor) gives `regressor = N/A`, `has_embedding = True`, no readout, no descriptors, and
`n_features` inapplicable, mirroring the freeze/width/clip sweep's end-to-end rows;
`freeze_weights: null` (encoder trains from the first epoch) maps to `freeze_epochs = 0`;
`ffn_hidden_dim = 1024` and `gradient_clip_val = 0.5` come straight from the recipe;
`train_data = drc_only` records that it trains on the dose-response pEC50 targets with no
primary-screen augmentation. A recipe that does not match this shape raises rather than
ingesting with wrong axes.

The two directories are one model scored on the two blind test phases separately (phase1 n=253,
phase2 n=260), so `_combine_baseline_metrics` recombines them into the same overall split the
sweep rows report (n=513). MAE and RMSE recombine exactly as count-weighted means of the
per-phase values; the rank and relative-error metrics (`r2`, `rae`, `kendall_tau`,
`spearman_rho`) need per-compound predictions the anvil runs do not persist, so they are left
null rather than approximated. The result is one row, `base_dir = pxr_baseline_chemeleon_pec50`,
`seed = -1`, `spec_source = anvil_recipe`, MAE 0.5348, all critical axes resolved.

## Evidence sources consulted

| source | what it pins | rows it applies to |
|---|---|---|
| in-run `config_used.yaml` | moal/chemprop active-learning config schema | 137 rows (`e4_*`, `freeze{0,1,2}_hd*_clip*`, `feat_observed_*`, `feat_predicted_native_2task`, one `tabpfn_small_embed` dir) |
| in-run `eval_out.csv` | metrics only (`mae`, `n`, `rmse`, `rae`, `r2`, `kendall_tau`, `spearman_rho`) | 312 rows |
| launch script + config + argparse defaults | every critical axis: encoder family/init/target, embedding/readout/descriptor composition, regressor, train_data, n_features | 226 rows (see below) |
| launch script (`moal plan` route) | encoder_family, encoder_init, train_data — a cross-check against the 137 `config_used.yaml` rows, not a new source for them | 362 rows total matched a launch-script invocation (226 tabular + 136 moal cross-check overlapping the `config_used.yaml` set, 2 OOM cases never produced a run dir) |

`results/pxr_baseline_predictions_phase1` and `results/pxr_baseline_predictions_phase2` are the
anvil-recipe CheMeleon->pEC50 baseline, now ingested as one combined row (see "Fifth pass"
below) rather than excluded; `.DS_Store` is skipped.

## Launch scripts parsed

Every invocation in each of these was read and transcribed (not sampled):

- `run_seed_sweep.sh` (3 recipe blocks, 5 seeds each)
- `run_seed_sweep_recipe3.sh` through `run_seed_sweep_recipe11.sh` (9 files)
- `run_tabfm_param_sweep.py` (subprocess-invoked `tabpfn_concat_features.py`, 3 ensemble sizes x 5 seeds)
- `run_cpu_oom_reproductions.sh` (2 single seed-0 cases; both OOM'd, so neither produced a `results/` directory — genuinely absent, not unresolved)
- `run_gnn_seed_sweep.py` (`moal plan` against seed-variant copies of the freeze*/e4* configs)
- `run_sweep.sh` (`moal plan` against every `configs/*.yaml`, output dir = the config stem)
- `tabpfn_uncertainty_analysis.py`'s own docstring `Run with:` invocation (its only recorded launch)

Each underlying Python script's argparse block (`tabpfn_concat_features.py`,
`tabpfn_chemeleon_concat_features.py`, `tabpfn_chemeleon_log2fc_features.py`,
`tabpfn_pec50_chemeleon_features.py`, `tabpfn_frozen_features.py`,
`tabpfn_chemeleon_static_features.py`) was read for what each flag controls and what its
default is when a launch script leaves it unset — argparse defaults are real behavior and are
recorded as resolved values, tagged as a default rather than an explicit flag where relevant
(e.g. `--regressor`'s default `"tabpfn"` tracks the unpinned TabPFN v2.5 the package ships,
not a value any launch script passes explicitly for most rows).

One correction folded in while parsing: `run_seed_sweep_recipe3.sh`'s comment says it reruns
after an OOM, and its own two-pass invocation for `results/tabpfn_pec50_chemeleon_seed{N}`
supersedes `run_seed_sweep.sh`'s original 256/256-PCA attempt at the same output directory
(recipe3.sh's pass 2 is the one that actually fits TabPFN and writes predictions, at 128/128).
This table uses recipe3.sh's values, not the superseded ones, for those 5 rows.

## What newly resolved, in aggregate

For the 226 rows that had **no** in-run artifact and are now resolved via
launch script + config + argparse:

| axis | resolved via |
|---|---|
| `encoder_family` | `chemprop` (from-scratch/CheMeleon-fine-tuned auxiliary encoder) or `chemeleon` (untouched pretrained, never fine-tuned) — which script ran |
| `encoder_init` | config's `from_foundation` field (`chemeleon` -> `chemeleon_pretrained`, `false` -> `scratch`) |
| `encoder_target` | which script ran: `log2fc` (`tabpfn_concat_features.py`/`tabpfn_frozen_features.py`/`tabpfn_chemeleon_log2fc_features.py`), `pec50` (`tabpfn_pec50_chemeleon_features.py`), or `none` (`tabpfn_chemeleon_static_features.py`/`tabpfn_chemeleon_concat_features.py`, both explicitly skip fine-tuning per their own docstrings) |
| `embedding_native_dim` | config's `auxiliary_model.message_hidden_dim` (from-scratch) or the fixed CheMeleon width (2048), documented in every CheMeleon-embedding script |
| `embedding_pca_dim` | `--embedding-pca-components` value/default, or `null` for scripts that never PCA-compress the embedding block |
| `has_embedding` / `has_readout` / `has_descriptors` | presence/absence of `--no-embedding` / `--no-readout` / `--no-descriptors` / `--skip-descriptors` |
| `readout_dim` | fixed at 2 (every referenced config's `data.plan.log2fc_columns` has 2 entries) when a readout block is included |
| `descriptor_sources` | `--descriptor-source` value/default, or the script's hard-coded family (e.g. `tabpfn_chemeleon_concat_features.py` is Mordred-only, no flag) |
| `descriptor_pca_width` | `--pca-components` / `--descriptor-pca-components` value/default |
| `regressor` | `--regressor` value, or the script's hard-coded regressor (`tabpfn_frozen_features.py`, `tabpfn_chemeleon_static_features.py`, `tabpfn_chemeleon_log2fc_features.py`, `tabpfn_pec50_chemeleon_features.py`, `tabpfn_uncertainty_analysis.py` all call `TabPFNRegressor()` directly with no `--regressor` flag) |
| `train_data` | every referenced config's `data.plan.input_csv` is `data/moal_plan_state.csv` (no `drc_only` marker) -> `drc_plus_primary` for all 226 rows |
| `n_features` | derived once the three block widths above resolve |

This moves `regressor`, `encoder_target`, and `n_features` from "never resolvable from any
artifact anywhere in `results/`" (the prior report's finding) to resolved for 226 rows — the
prior finding was an artifact of only looking inside run directories, not a real absence of
provenance.

## Cross-check: launch script vs. in-run artifact

136 of the 137 `config_used.yaml` rows also matched a `moal plan` launch-script invocation
(`run_sweep.sh` for the unseeded `feat_*`/single-config rows, `run_gnn_seed_sweep.py` for the
seeded `freeze*`/`e4_*` rows); the remaining one (`feat_observed_4task`) has no matching
`configs/*.yaml` file and so was reached by neither `run_sweep.sh` nor
`run_gnn_seed_sweep.py` — it still resolves from its own in-run `config_used.yaml`, just with
no independent launch-script corroboration. For the axes both routes can determine
(`encoder_family`, `encoder_init`, `train_data`), **0 disagreements** were found: every
in-run `config_used.yaml` matches what its launch script's config file says it should. This is
a genuine cross-check, not a trivial one — a mismatch here would mean a run directory's config
diverged from what the launch script intended (e.g. a stale rerun with an edited config), and
none did.

## Systemic finding: what's still genuinely unresolvable

**191 of 417 rows remain quarantined**, in four patterns (breakdown below reflects the
third-pass directives: `regressor` is now the resolved value `"N/A"`, not an unknown, for
every moal row, since an end-to-end message-passing network has a built-in FFN readout, not
a swappable regressor stage; `embedding_native_dim`/`readout_dim` now resolve from
`ffn_hidden_dim` for the 131 rows whose config carries it, and `encoder_target`/`encoder_init`
now resolve for `e4_finetune`/`e4_frozen`/their `_drc_only` variants via the `configs/e4_*.yaml`
header-comment cross-check described below):

| `unknown_axes` pattern | row count | why |
|---|---|---|
| all 14 critical columns unknown | 54 | no in-run artifact **and** no matching launch-script invocation found anywhere (true orphans, see below) |
| `encoder_target, n_features` | 113 | `freeze{0,1,2}_hd*_clip*` and `feat_*` configs: no header comment states a training-target lineage the way `e4_*.yaml` does, so `encoder_target` stays unresolved; `n_features` has no source anywhere in this schema (moal never logs an input-matrix width) |
| `n_features` | 18 | `e4_finetune`/`e4_frozen`/`_drc_only` variants: `encoder_target`/`encoder_init` now resolve via the `configs/e4_*.yaml` header-comment cross-check (see below); only `n_features` remains unresolved |
| `encoder_init, encoder_target, n_features` | 6 | `e4_finetune_drc_only`'s config header comment does not state a pretrain lineage the way `e4_frozen`/`e4_finetune`/`e4_frozen_drc_only` do, so `encoder_init`/`encoder_target` stay unresolved for this one variant |

`regressor` and `embedding_native_dim`/`readout_dim` are no longer unknown for any moal row.
`n_features` remains unresolved for all 137, and is a real schema limitation: moal never logs
an input-matrix column count anywhere this table can reach.

**The 54 "all axes unknown" rows are true orphans**, found in neither `results/` in-run
artifacts nor any launch script read for this table:

`feat_observed_4task` (config_used.yaml *is* present in-run and resolves it via
`parsed_config`, so it is not actually in this 54; see the cross-check section), and, with no
in-run artifact and no launch-script match: `lgbm_embed_readout_mordred_pca128`,
`tabfm_embed_readout_mordred_pca128`, `tabicl_embed_readout_mordred_pca128`,
`tabpfn_chemeleon_embed_only`, `tabpfn_chemeleon_log2fc_mordred_pca128`,
`tabpfn_chemeleon_static`, `tabpfn_concat_pca{64,256}`, `tabpfn_concat_small_embed`,
`tabpfn_embed_mordred_pca128_no_readout`, `tabpfn_embed_only_no_readout`,
`tabpfn_embed_readout_mordred_pca{64,128,256}`, `tabpfn_embed_readout_rdkit_pca128`,
`tabpfn_mordred_only_pca128_no_embed_no_readout`, `tabpfn_pec50_chemeleon`,
`tabpfn_pec50_chemeleon_full`, `tabpfn_pec50_scratch_mordred_pca128`,
`tabpfn_rdkit_only_pca128_no_embed_no_readout`, `tabpfn_readout_mordred_pca128_no_embed`,
`tabpfn_readout_only_no_embed`, `tabicl_chemeleon_readout_{descriptors,descriptors_pca64,only,only_calibrated}`,
`tabpfn-v2.6_embed_readout_mordred_pca128`, `tabpfn-v3_embed_readout_mordred_pca128`,
`xgboost_embed_readout_mordred_pca128`, and one stray `tabicl_embed_readout_mordred_pca128_seed5`.

Every one of these is an **unseeded** directory whose name matches a *seeded* recipe this
table does resolve (e.g. `tabpfn_embed_readout_mordred_pca128` alongside its own
`_seed{0..4}` siblings, or `tabicl_chemeleon_readout_descriptors_pca64_seed{0..4}` alongside
`tabicl_chemeleon_readout_descriptors_seed{0..4}`), plus the one out-of-range `_seed5`
directory. No launch script's `Run with:` docstring or shell invocation targets these exact
directory names; they read as earlier, single-run predecessors of the later 5-seed sweep
(the same shape the prior report's `tabpfn_small_embed` base_dir already flagged), or
one-off manual invocations never captured in any committed script. This is a real remaining
gap, not something this pass could close from the evidence available: without a further
artifact (a shell history, a notebook, a commit message tying a specific invocation to one of
these directories) their provenance cannot be pinned without guessing.

## Quarantine list

Full per-row list (`run_dir`, `unknown_axes`) is in `results/run_provenance.parquet` /
`.csv`.

## Fingerprint consistency check

Grouping by `config_fingerprint` and checking `nunique()==1` on every critical column within
each group: **0 violations**, same as before (fingerprints are hashed directly from those
column values, so a violation would indicate a hashing bug, not found here).

## `base_dir` → `config_fingerprint` mapping (against `reporting/manifest.py`)

33 distinct `base_dir` values appear in `manifest.py`'s `EXPERIMENTS` list; **all 33 have
matching run directories** in this table (no manifest base_dir was missing).

**21 base_dirs now resolve to 2 distinct fingerprints each** (up from 1 flagged case before),
all following the same shape: an unresolved, unseeded directory (one of the 54 orphans above)
sitting alongside its resolved, seeded siblings under the same base_dir name once the seed
suffix is stripped. This is not a new inconsistency this pass introduced; it is the same
`tabpfn_small_embed` pattern the prior report already flagged, now showing up for every base
name that has both an orphaned pre-sweep run and a resolved 5-seed sweep. The root cause is
now explained rather than merely flagged: `tabpfn_small_embed`'s own two fingerprints come
from `run_sweep.sh` running `moal plan` against `configs/tabpfn_small_embed.yaml` for the
unseeded directory (moal schema, misleadingly named) while `run_seed_sweep_recipe5.sh` runs
`tabpfn_frozen_features.py` for the seeded siblings (TabPFN-tabular schema) — two different
launch scripts target the same base name for genuinely different pipelines.

**Fingerprints shared across more than one `base_dir` (informational only):**
- `033566ac240e21ca`: `e4_finetune`, `e4_frozen` — identical under this table's resolution power (config schema doesn't distinguish them on any axis this table extracts).
- `faeae1ffbc5dbc45`: `e4_finetune_drc_only`, `e4_frozen_drc_only` — same situation.
- `6d3641f3ab94b8b7`: all 18 `freeze{0,1,2}_hd*_clip*` base_dirs plus the unseeded `tabpfn_small_embed` — all moal-schema, `from_foundation: chemeleon`, no readout/descriptors (this table's axes don't yet distinguish `hd`/`clip` hyperparameters, which sit outside `CRITICAL_COLUMNS`).
- `cbc6bca30161123e`: `feat_observed_2task`, `feat_observed_4task`, `feat_observed_and_predicted_2task`, `feat_predicted_native_2task` — same reason.
- `1911bbe88da77c90`: the three `tabfm_n{16,32,64}_embed_readout_mordred_pca128` base_dirs plus the unseeded `tabfm_embed_readout_mordred_pca128` orphan — `n_estimators` isn't one of this table's critical columns, so the ensemble-size sweep collapses to one fingerprint (by design; it's a hyperparameter sweep at fixed feature/regressor axes).
- `8fa7abaf5b2eb8dd`: `tabpfn_embed_readout_mordred_pca128`'s seeded rows plus `tabpfn_uncertainty_mordred_pca128` — same resolved featureset (embedding+readout+Mordred-PCA128, `tabpfn` regressor), different downstream analysis script.
- `d48fc6b90c079cf8`: `tabpfn_frozen_features`, `tabpfn_small_embed` (seeded rows) — both resolve to embedding+readout, no descriptors, `tabpfn` regressor: identical axes by this table's vocabulary even though they're separate recipe entries.
- `bdb8b96a88b8d409`: the "all axes unknown" fingerprint, shared by the 20 orphaned base_dirs listed above (`lgbm_embed_readout_mordred_pca128` through `xgboost_embed_readout_mordred_pca128`) — expected shape of the orphan finding, not a real identity collision; every one of these rows has an identical all-null fingerprint because nothing distinguishes them from this table's point of view yet.

## Cross-check against `NEXT_STEPS.md`

`NEXT_STEPS.md` (2026-07-29, the running experiment log) supplies three narrative
cross-checks against this table. None changed a resolved value; all three confirmed what the
table already computed.

**Renaming history.** The doc states some result directories were renamed because their
original names implied descriptor-only ablation when they always included the
embedding+readout block by default (its example: `tabpfn_concat_rdkit_only` ->
`tabpfn_embed_readout_rdkit_pca128`). The old name `tabpfn_concat_rdkit_only` does not appear
anywhere in `build_run_provenance.py`'s `LAUNCH_INVOCATIONS` or in this report's orphan list;
every reference uses the corrected name. Confirms the table already reflects the renamed,
corrected directory names, no change made.

**Fixed-embedding-cache confound fix.** The doc flags four base_dirs as predating this
session's fix (each reads its own per-invocation encoder, not the shared
`data/auxiliary_embedding_cache.parquet`, and is "not part of the shared cache, not rerun"):
`tabpfn_pec50_scratch_mordred_pca128`, `tabpfn_pec50_chemeleon`, `tabpfn_chemeleon_static`,
`tabpfn_chemeleon_log2fc_mordred_pca128`. Querying `results/run_provenance.parquet` for each:

- `tabpfn_pec50_scratch_mordred_pca128` and `tabpfn_chemeleon_static` and
  `tabpfn_pec50_chemeleon` and `tabpfn_chemeleon_log2fc_mordred_pca128` (the unseeded
  directory names) all carry the all-null orphan fingerprint `bdb8b96a88b8d409`, distinct from
  their resolved `_seed{0..4}` siblings' fingerprints. `tabpfn_pec50_scratch_mordred_pca128`
  has no seeded sibling at all under that exact name (the seeded sweep for that cell runs under
  `tabpfn_pec50_scratch`, a different base_dir string).
- No column collapses a pre-fix and post-fix run of the same base_dir into one identity: this
  table's existing `base_dir` + `config_fingerprint` pairing already separates them, because
  the pre-fix orphan rows have zero resolved axes (hence a shared null fingerprint) while the
  post-fix rows resolve every critical axis (a distinct fingerprint per config). No new column
  (e.g. `embedding_source_stable`) is needed on top of this.
- Separately, all four of these base_dirs are produced by `tabpfn_pec50_chemeleon_features.py`
  or `tabpfn_chemeleon_log2fc_features.py`/`tabpfn_chemeleon_static_features.py`, none of which
  the doc lists among the three scripts the fix touched
  (`tabpfn_concat_features.py`, `tabpfn_frozen_features.py`,
  `tabpfn_uncertainty_analysis.py`). So even their resolved, seeded rows are, by construction,
  outside the shared cache; this is already implied by `encoder_init`/`encoder_target`
  resolving through a different script's axes resolver, not something this table's schema was
  missing.

**Encoder-init x training-target 2x2 grid.** Queried `encoder_init`/`encoder_target` for the
seeded rows of all four grid cells the doc names:

| base_dir | table's `encoder_init` | table's `encoder_target` | doc's cell |
|---|---|---|---|
| `tabpfn_embed_readout_mordred_pca128` | `scratch` | `log2fc` | from-scratch, log2FC-trained |
| `tabpfn_pec50_scratch` | `scratch` | `pec50` | from-scratch, pEC50-trained |
| `tabpfn_chemeleon_log2fc_mordred_pca128` | `chemeleon_pretrained` | `log2fc` | CheMeleon-init, log2FC-trained |
| `tabpfn_pec50_chemeleon` | `chemeleon_pretrained` | `pec50` | CheMeleon-init, pEC50-trained |

All four agree with the doc's implied labels; 0 disagreements.

## n_features cross-check

No run directory's own artifacts record an actual input-matrix column count (no logged
`X.shape`, feature-cache shape, or saved feature parquet was found anywhere under `results/`),
so there is still nothing to cross-check the 226 launch-script-derived `n_features` values
against directly. They are, however, internally consistent: each is the sum of the same
embedding/readout/descriptor widths the corresponding script's own log lines describe (e.g.
`tabpfn_chemeleon_readout_descriptors`'s 386 columns = 256 CheMeleon PCA + 2 borrowed readout +
128 Mordred PCA, matching `run_seed_sweep_recipe9.sh`'s own comment). `n_features` remains
unresolved for all 191 quarantined rows, for the same reasons their other axes are unresolved.
