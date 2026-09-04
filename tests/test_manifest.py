from pathlib import Path

import pytest
import yaml

import manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "experiments" / "manifest.yaml"
PRIOR_SUMMARY = REPO_ROOT / "experiments" / "prior_summary.csv"


@pytest.fixture(scope="module")
def spec() -> manifest.Manifest:
    return manifest.load(MANIFEST, PRIOR_SUMMARY)


def test_manifest_loads_and_validates(spec):
    assert spec.seeds == (0, 1, 2, 3, 4)
    assert spec.figures
    assert len(spec.cells) == sum(len(f.cells) for f in spec.figures)


def test_every_figure_in_the_plan_is_present(spec):
    assert [f.id for f in spec.figures] == [f"fig{n}" for n in range(1, 8)]


def test_pruning_obeys_its_own_margin(spec):
    assert manifest.prune_violations(spec) == []


def test_every_scored_prior_config_is_accounted_for(spec):
    claimed = manifest.prior_configs_claimed(spec)
    unclaimed = set(spec.prior.index) - claimed
    assert unclaimed == set()


def test_every_cell_gives_a_reason(spec):
    assert all(cell.reason.strip() for cell in spec.cells.values())


def test_pruned_cells_cite_evidence_or_a_reason_kind(spec):
    for cell in spec.cells.values():
        if cell.status != "pruned":
            continue
        cites_rival = cell.compared_to is not None
        assert cites_rival or cell.reason_kind in manifest.REASON_KINDS, cell.id


def test_run_cells_either_own_axes_or_share_runs(spec):
    for cell in spec.cells.values():
        if cell.status != "run":
            continue
        assert bool(cell.axes) ^ bool(cell.shares_runs_with), cell.id


def test_shared_runs_resolve_to_a_cell_that_owns_them(spec):
    for cell in spec.cells.values():
        seen = set()
        current = cell
        while current.shares_runs_with is not None:
            assert current.id not in seen, cell.id
            seen.add(current.id)
            current = spec.cells[current.shares_runs_with]
        if cell.status == "run":
            assert current.owns_runs, cell.id


def test_planned_runs_are_five_seeds_of_every_owning_cell(spec):
    owners = [c for c in spec.cells.values() if c.owns_runs]
    planned = spec.planned_runs()
    assert len(planned) == len(owners) * len(spec.seeds)
    assert len(set(planned)) == len(planned)


def test_coverage_reports_missing_and_unplanned_runs(spec):
    planned = spec.planned_runs()
    stray = Path("results/fig9/invented/seed0")

    complete = spec.coverage(planned)
    assert complete.is_complete

    partial = spec.coverage([*planned[1:], stray])
    assert partial.missing == (planned[0],)
    assert partial.unplanned == (stray,)


def test_split_matches_the_committed_resource_files(spec):
    for key in ("fit", "fit_train", "fit_val", "test"):
        assert (REPO_ROOT / spec.split[key]).exists()


def test_declared_regressors_are_registered_upstream(spec):
    from openadmet.models._registry_loader import load_group
    from openadmet.models.architecture.model_base import models

    load_group("models")
    for name in spec.regressors["available"]:
        assert name in models, name


def test_prior_summary_carries_no_phase2_metrics(spec):
    # every prior metric is the pooled n=513 evaluation; a 260-row eval here
    # would mean the summary had been rebuilt against the wrong source
    assert set(spec.prior["n_eval"].dropna().unique()) == {513}


def test_anchor_is_recorded_as_phase2(spec):
    assert spec.anchor["n"] == 260
    assert spec.anchor["mae_ensemble"] == pytest.approx(0.4113)


def test_manifest_copies_no_metrics():
    # metrics live in the summary CSV alone; a number in the manifest would be a
    # second source of truth able to drift from it
    raw = yaml.safe_load(MANIFEST.read_text())
    for figure in raw["figures"]:
        for cell in figure["cells"]:
            assert set(cell.get("prior") or {}) <= {"config", "compared_to"}, cell["id"]
