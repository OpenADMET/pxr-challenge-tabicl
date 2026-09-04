import json

import pandas as pd
import pytest

import features
import manifest as manifest_module
import sweep
from data import CANONICAL_COL, SPLIT_DIR, TARGET_COL

REPO_ROOT = SPLIT_DIR.parent.parent


@pytest.fixture(scope="module")
def spec():
    return manifest_module.load()


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    """Build a miniature split and feature cache, so a run exercises the real path cheaply."""

    def head(name: str, n: int) -> pd.DataFrame:
        frame = pd.read_csv(SPLIT_DIR / name, dtype={CANONICAL_COL: str})
        return frame.loc[frame[TARGET_COL].notna()].head(n).reset_index(drop=True)

    fit = head("fit_all.csv", 40)
    test = head("test_phase2.csv", 12)
    partitions = sweep.Partitions(
        fit=fit, fit_train=fit.head(30), fit_val=fit.tail(10).reset_index(drop=True), test=test
    )
    molecules = sorted(set(fit[CANONICAL_COL]) | set(test[CANONICAL_COL]))
    monkeypatch.setattr(features, "unique_molecules", lambda *a, **k: molecules)
    return partitions, tmp_path


def _config(**overrides) -> manifest_module.TabularConfig:
    base = {
        "embedding": "none",
        "readout": "none",
        "descriptors": "rdkit",
        "descriptor_pca": 4,
        "regressor": "lgbm",
        "calibration": "none",
    }
    return manifest_module.TabularConfig(**{**base, **overrides})


def _run(config, spec, partitions, tmp_path, **kwargs):
    return sweep.run_one(
        config,
        spec,
        seed=0,
        partitions=partitions,
        results_dir=tmp_path / "results",
        feature_cache=tmp_path / "features",
        reduced_cache=tmp_path / "reduced",
        **kwargs,
    )


def test_a_run_writes_predictions_metrics_and_a_record(spec, tiny):
    partitions, tmp_path = tiny
    run_dir = _run(_config(), spec, partitions, tmp_path)

    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert list(predictions.columns[:3]) == [CANONICAL_COL, "observed", "predicted"]
    assert len(predictions) == len(partitions.test)

    scores = json.loads((run_dir / "metrics.json").read_text())
    assert {"mae", "rmse", "rae", "r2", "kendall_tau", "spearman_rho"} <= set(scores)

    record = json.loads((run_dir / "run.json").read_text())
    assert record["config"]["regressor"] == "lgbm"
    assert record["n_features"] == 4
    assert record["n_test"] == len(partitions.test)
    assert record["calibrated"] is False
    # the record names the reduced blocks it consumed, by content key
    assert [i["block"] for i in record["inputs"]] == ["rdkit"]
    assert record["environment"]["repo"]["commit"]


def test_the_run_directory_is_named_by_the_configuration(spec, tiny):
    partitions, tmp_path = tiny
    config = _config()
    run_dir = _run(config, spec, partitions, tmp_path)
    assert run_dir.parts[-2:] == (config.slug, "seed0")


def test_a_completed_run_is_not_repeated(spec, tiny):
    partitions, tmp_path = tiny
    run_dir = _run(_config(), spec, partitions, tmp_path)
    stamp = (run_dir / "run.json").stat().st_mtime_ns
    _run(_config(), spec, partitions, tmp_path)
    assert (run_dir / "run.json").stat().st_mtime_ns == stamp


def test_forcing_redoes_a_completed_run(spec, tiny):
    partitions, tmp_path = tiny
    run_dir = _run(_config(), spec, partitions, tmp_path)
    stamp = (run_dir / "run.json").stat().st_mtime_ns
    _run(_config(), spec, partitions, tmp_path, force=True)
    assert (run_dir / "run.json").stat().st_mtime_ns != stamp


def test_a_changed_input_makes_the_run_incomplete(spec, tiny):
    partitions, tmp_path = tiny
    run_dir = _run(_config(), spec, partitions, tmp_path)
    recorded = json.loads((run_dir / "run.json").read_text())["spec"]
    # a run whose upstream block moved has a different specification, so the
    # existing directory must not be mistaken for a finished run
    moved = {**recorded, "inputs": ["something else"]}
    assert sweep.is_complete(run_dir, recorded)
    assert not sweep.is_complete(run_dir, moved)


def test_the_calibration_arm_fits_on_held_out_predictions(spec, tiny):
    partitions, tmp_path = tiny
    run_dir = _run(_config(calibration="isotonic_fitval"), spec, partitions, tmp_path)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["calibrated"] is True
    assert record["config"]["calibration"] == "isotonic_fitval"


def test_calibration_changes_the_predictions(spec, tiny):
    partitions, tmp_path = tiny
    plain = _run(_config(), spec, partitions, tmp_path)
    calibrated = _run(_config(calibration="isotonic_fitval"), spec, partitions, tmp_path)
    a = pd.read_csv(plain / "predictions.csv")["predicted"].to_numpy()
    b = pd.read_csv(calibrated / "predictions.csv")["predicted"].to_numpy()
    assert not (a == b).all()


def test_assembling_a_block_that_misses_compounds_is_refused(spec, tiny):
    partitions, tmp_path = tiny
    config = _config()
    fit_smiles = sorted(set(partitions.fit[CANONICAL_COL]))
    artifacts = sweep.reduced_blocks(
        config,
        spec,
        0,
        fit_smiles=fit_smiles,
        feature_cache=tmp_path / "features",
        reduced_cache=tmp_path / "reduced",
    )
    with pytest.raises(sweep.SweepError, match="missing"):
        sweep.assemble(artifacts, ["not-a-real-smiles"])


def test_blocks_are_joined_in_a_fixed_order():
    assert sweep.BLOCK_ORDER == ("embedding", "readout", "descriptors")


def _stub_train(offset: float):
    """Return a stand-in architecture, so the run path is testable without training one."""

    def train(axes, seed, partitions):
        del axes, seed
        predictions = partitions.test[TARGET_COL].to_numpy(dtype=float) + offset
        return predictions, {"epochs": 1}

    return train


def test_a_graph_network_run_writes_the_same_shape_as_a_tabular_one(spec, tiny):
    partitions, tmp_path = tiny
    cell = spec.gnn_cells[0]
    run_dir = sweep.run_gnn_one(
        cell,
        spec,
        seed=1,
        train=_stub_train(0.25),
        partitions=partitions,
        results_dir=tmp_path / "results",
    )
    assert run_dir.parts[-3:] == ("gnn", cell.id, "seed1")

    record = json.loads((run_dir / "run.json").read_text())
    assert record["cell"] == cell.id
    assert record["axes"] == cell.axes
    assert record["seed"] == 1
    assert record["training"] == {"epochs": 1}
    # aggregation identifies a run by its directory and cross-checks the record
    assert record["cell"] == run_dir.parent.name

    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert list(predictions.columns) == [CANONICAL_COL, "observed", "predicted"]
    scores = json.loads((run_dir / "metrics.json").read_text())
    assert scores["mae"] == pytest.approx(0.25)


def test_a_completed_graph_network_run_is_not_repeated(spec, tiny):
    partitions, tmp_path = tiny
    cell = spec.gnn_cells[0]
    kwargs = {
        "train": _stub_train(0.25),
        "partitions": partitions,
        "results_dir": tmp_path / "results",
    }
    run_dir = sweep.run_gnn_one(cell, spec, seed=1, **kwargs)
    stamp = (run_dir / "run.json").stat().st_mtime_ns
    sweep.run_gnn_one(cell, spec, seed=1, **kwargs)
    assert (run_dir / "run.json").stat().st_mtime_ns == stamp


def test_wrong_number_of_predictions_is_refused(spec, tiny):
    partitions, tmp_path = tiny

    def short(axes, seed, partitions):
        del axes, seed
        return partitions.test[TARGET_COL].to_numpy(dtype=float)[:-1], {}

    with pytest.raises(sweep.SweepError, match="predictions for"):
        sweep.run_gnn_one(
            spec.gnn_cells[0],
            spec,
            seed=0,
            train=short,
            partitions=partitions,
            results_dir=tmp_path / "results",
        )


def test_aggregation_reads_a_graph_network_run(spec, tiny):
    import aggregate

    partitions, tmp_path = tiny
    results = tmp_path / "results"
    for seed in (0, 1):
        sweep.run_gnn_one(
            spec.gnn_cells[0],
            spec,
            seed=seed,
            train=_stub_train(0.25),
            partitions=partitions,
            results_dir=results,
        )
    table = aggregate.tidy(results)
    assert len(table) == 2
    assert set(table["kind"]) == {"gnn"}
    assert set(table["config_id"]) == {spec.gnn_cells[0].id}
