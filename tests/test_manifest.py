from pathlib import Path

import pytest

import manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "experiments" / "manifest.yaml"
PRIOR_SUMMARY = REPO_ROOT / "experiments" / "prior_summary.csv"

WINNER = {
    "embedding": "chemeleon",
    "readout": "chemprop_log2fc",
    "descriptors": "mordred",
    "regressor": "tabicl",
    "calibration": "none",
}


@pytest.fixture(scope="module")
def spec() -> manifest.Manifest:
    return manifest.load(MANIFEST, PRIOR_SUMMARY)


def test_manifest_loads(spec):
    assert spec.seeds == (0, 1, 2, 3, 4)
    assert [s.id for s in spec.stages] == [
        "featureset_and_regressor",
        "descriptor_width",
        "calibration",
        "uncertainty",
    ]


def test_first_stage_crosses_every_featureset_with_every_regressor(spec):
    configs = spec.expand("featureset_and_regressor")
    n_regressors = len(spec.axes["regressor"])
    # 5 embeddings x 2 readouts x 4 descriptor options, less the empty featureset
    assert len(configs) == 39 * n_regressors
    assert len({c.slug for c in configs}) == len(configs)


def test_no_configuration_is_empty(spec):
    for stage in ("featureset_and_regressor",):
        assert all(c.n_blocks > 0 for c in spec.expand(stage))


def test_first_stage_holds_width_and_calibration_fixed(spec):
    configs = spec.expand("featureset_and_regressor")
    assert {c.descriptor_pca for c in configs} == {128}
    assert {c.calibration for c in configs} == {"none"}


def test_a_gated_stage_refuses_to_expand_before_its_gate_is_decided(spec):
    with pytest.raises(manifest.ManifestError, match="none were supplied"):
        spec.expand("descriptor_width")


def test_width_stage_sweeps_every_width_on_the_settled_configuration(spec):
    configs = spec.expand("descriptor_width", WINNER)
    assert sorted(c.descriptor_pca for c in configs) == [64, 128, 256]
    assert {c.regressor for c in configs} == {"tabicl"}


def test_width_stage_collapses_when_the_winner_has_no_descriptors(spec):
    winner = {**WINNER, "descriptors": "none"}
    configs = spec.expand("descriptor_width", winner)
    # three widths over a featureset with no descriptor block would be one
    # configuration under three names
    assert len(configs) == 1


def test_calibration_stage_sweeps_both_arms(spec):
    configs = spec.expand("calibration", {**WINNER, "descriptor_pca": 128})
    assert sorted(c.calibration for c in configs) == ["isotonic_fitval", "none"]


def test_the_uncertainty_stage_adds_no_runs(spec):
    assert spec.expand("uncertainty", WINNER) == []


def test_run_directories_are_unique_and_one_per_seed(spec):
    runs = spec.planned_runs("featureset_and_regressor")
    configs = spec.expand("featureset_and_regressor")
    assert len(runs) == len(configs) * len(spec.seeds)
    assert len(set(runs)) == len(runs)


def test_a_slug_describes_its_configuration(spec):
    config = manifest.TabularConfig("chemeleon", "none", "mordred", 128, "tabicl", "none")
    assert config.slug == "emb-chemeleon__ro-none__desc-mordred128__reg-tabicl__cal-none"
    assert config.run_dir(3).parts[-2:] == (config.slug, "seed3")


def test_a_width_is_absent_from_the_slug_when_there_are_no_descriptors():
    a = manifest.TabularConfig("chemeleon", "none", "none", 64, "tabicl", "none")
    b = manifest.TabularConfig("chemeleon", "none", "none", 256, "tabicl", "none")
    assert a.slug == b.slug


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
