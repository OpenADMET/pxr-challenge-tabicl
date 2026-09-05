from pathlib import Path

import pytest

import manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "experiments" / "manifest.yaml"
PRIOR_SUMMARY = REPO_ROOT / "experiments" / "prior_summary.csv"

# what the two width probes settle before the ingredient stage runs
SETTLED = {"descriptors": "mordred", "descriptor_pca": 128, "embedding_pca": 256}

WINNER = {
    **SETTLED,
    "embedding": "chemeleon",
    "readout": "chemprop_log2fc",
    "regressor": "tabicl",
    "calibration": "none",
}


@pytest.fixture(scope="module")
def spec() -> manifest.Manifest:
    return manifest.load(MANIFEST, PRIOR_SUMMARY)


def test_manifest_loads(spec):
    assert spec.seeds == (0, 1, 2, 3, 4)
    assert [s.id for s in spec.stages] == [
        "descriptor_width",
        "embedding_width",
        "ingredients",
        "calibration",
        "uncertainty",
    ]


def test_the_width_probes_lead_and_wait_on_nothing(spec):
    # both run before any gate exists, so neither may depend on one
    for stage_id in ("descriptor_width", "embedding_width"):
        assert spec.stage(stage_id).depends_on == ()


def test_the_descriptor_probe_crosses_every_block_with_every_width_it_can(spec):
    configs = spec.expand("descriptor_width")
    # 3 blocks x 4 widths, less RDKit at 256: it has 217 columns, so that is
    # not a reduction and the decomposition refuses it
    assert len(configs) == 11
    assert sorted({c.descriptor_pca for c in configs}) == [32, 64, 128, 256]
    assert {c.descriptors for c in configs} == {"mordred", "rdkit", "rdkit_mordred"}
    assert {c.regressor for c in configs} == {"tabpfn-v3"}
    assert {c.embedding for c in configs} == {"none"}


def test_a_width_that_would_not_reduce_its_block_is_not_a_configuration(spec):
    widths = {
        c.descriptors: sorted(
            x.descriptor_pca
            for x in spec.expand("descriptor_width")
            if x.descriptors == c.descriptors
        )
        for c in spec.expand("descriptor_width")
    }

    assert widths["rdkit"] == [32, 64, 128]
    assert widths["mordred"] == [32, 64, 128, 256]
    assert widths["rdkit_mordred"] == [32, 64, 128, 256]


def test_the_embedding_probe_sweeps_width_on_the_frozen_embedding(spec):
    configs = spec.expand("embedding_width")
    assert sorted(c.embedding_pca for c in configs) == [32, 64, 128, 256]
    # a fine-tuned embedding would need an encoder, and the probe runs before
    # any encoder is trained
    assert {c.embedding for c in configs} == {"chemeleon"}
    assert {c.regressor for c in configs} == {"tabpfn-v3"}


def test_the_probes_carry_no_width_for_a_block_they_do_not_use(spec):
    assert {c.embedding_pca for c in spec.expand("descriptor_width")} == {manifest.NOT_REDUCED}
    assert {c.descriptor_pca for c in spec.expand("embedding_width")} == {manifest.NOT_REDUCED}


def test_no_configuration_is_empty(spec):
    for stage in ("descriptor_width", "embedding_width"):
        assert all(c.n_blocks > 0 for c in spec.expand(stage))
    assert all(c.n_blocks > 0 for c in spec.expand("ingredients", SETTLED))


def test_the_ingredient_stage_crosses_featureset_with_regressor(spec):
    configs = spec.expand("ingredients", SETTLED)
    n_regressors = len(spec.axes["regressor"])
    # 5 embeddings x 2 readouts x (no descriptors, or the settled block), less
    # the empty featureset
    assert len(configs) == 19 * n_regressors
    assert len({c.slug for c in configs}) == len(configs)


def test_the_ingredient_stage_holds_what_the_probes_settled(spec):
    configs = spec.expand("ingredients", SETTLED)
    assert {c.descriptors for c in configs} == {"none", "mordred"}
    assert {c.descriptor_pca for c in configs} == {manifest.NOT_REDUCED, 128}
    assert {c.calibration for c in configs} == {"none"}


def test_the_ingredient_stage_reruns_nothing_the_probe_already_did(spec):
    probe = {c.slug for c in spec.expand("descriptor_width")}
    ingredients = {c.slug for c in spec.expand("ingredients", SETTLED)}
    # the settled block at the settled width under the probe's own regressor is
    # the same run, and shares its directory rather than being fitted twice
    assert probe & ingredients == {"emb-none__ro-none__desc-mordred128__reg-tabpfn-v3__cal-none"}


def test_a_gated_stage_refuses_to_expand_before_its_gates_are_decided(spec):
    with pytest.raises(manifest.ManifestError, match="none were supplied"):
        spec.expand("ingredients")


def test_a_restricted_axis_needs_the_gate_it_points_at(spec):
    # every axis but the one the restriction dereferences
    partial = {"embedding_pca": 256, "descriptor_pca": 128}
    with pytest.raises(manifest.ManifestError, match="descriptors"):
        spec.expand("ingredients", partial)


def test_calibration_stage_sweeps_both_arms(spec):
    configs = spec.expand("calibration", WINNER)
    assert sorted(c.calibration for c in configs) == ["isotonic_fitval", "none"]


def test_the_uncertainty_stage_adds_no_runs(spec):
    assert spec.expand("uncertainty", WINNER) == []


def test_run_directories_are_unique_and_one_per_seed(spec):
    runs = spec.planned_runs("ingredients", SETTLED)
    configs = spec.expand("ingredients", SETTLED)
    assert len(runs) == len(configs) * len(spec.seeds)
    assert len(set(runs)) == len(runs)


def test_a_slug_describes_its_configuration():
    config = manifest.TabularConfig(
        embedding="chemeleon",
        embedding_pca=256,
        readout="none",
        descriptors="mordred",
        descriptor_pca=128,
        regressor="tabicl",
        calibration="none",
    )
    assert config.slug == "emb-chemeleon256__ro-none__desc-mordred128__reg-tabicl__cal-none"
    assert config.run_dir(3).parts[-2:] == (config.slug, "seed3")


def test_an_inapplicable_width_is_normalized_away_rather_than_hidden(spec):
    # two widths over a featureset with no descriptor block are one
    # configuration, and it must not merely look like one: sharing a run
    # directory while carrying different specifications would make each run
    # read the other's results as stale
    configs = spec.expand("embedding_width")
    assert {c.descriptor_pca for c in configs} == {manifest.NOT_REDUCED}
    assert len({c.slug for c in configs}) == len(
        {tuple(sorted(c.as_dict().items())) for c in configs}
    )


def test_graph_network_cells_expand_the_declared_grid(spec):
    ids = [c.id for c in spec.gnn_cells]
    # 3 freeze settings x 2 widths x 3 clip settings, plus 7 named variants
    assert len(ids) == 18 + 7
    assert len(set(ids)) == len(ids)
    assert "concat_freeze2_hd512_clipoff" in ids
    assert all(cell.axes for cell in spec.gnn_cells)


def test_graph_network_runs_are_five_seeds_each(spec):
    assert len(spec.gnn_runs()) == len(spec.gnn_cells) * len(spec.seeds)


def test_nothing_the_prior_explored_is_inexpressible(spec):
    # the prior decides nothing, but a family it ran that this vocabulary cannot
    # even name would be an oversight rather than a decision
    assert manifest.uncovered_prior_levels(spec) == {}


def test_every_prior_configuration_a_cell_names_exists(spec):
    for cell in spec.gnn_cells:
        if cell.prior is not None:
            assert cell.prior in spec.prior.index, cell.id


def test_every_regressor_level_is_implemented(spec):
    import regressors

    assert set(spec.axes["regressor"]) == set(regressors.REGRESSORS)


def test_coverage_reports_missing_and_unplanned_runs(spec):
    planned = spec.planned_runs("descriptor_width", WINNER)
    stray = Path("results/tabular/invented/seed0")

    assert spec.coverage(planned, planned).is_complete

    partial = spec.coverage(planned, [*planned[1:], stray])
    assert partial.missing == (planned[0],)
    assert partial.unplanned == (stray,)


def test_split_files_the_manifest_names_exist(spec):
    for key in ("fit", "fit_train", "fit_val", "test"):
        assert (REPO_ROOT / spec.split[key]).exists()
    assert spec.split["n_fit"] == 4392
    assert spec.split["n_test"] == 260


def test_anchor_is_the_phase_two_number(spec):
    assert spec.anchor["n"] == 260
    assert spec.anchor["mae_ensemble"] == pytest.approx(0.4113)


def test_prior_summary_is_the_pooled_evaluation(spec):
    # a 260-row eval here would mean the summary was rebuilt from the wrong source
    assert set(spec.prior["n_eval"].dropna().unique()) == {513}
