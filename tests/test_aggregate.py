import json
from pathlib import Path

import pandas as pd
import pytest

import aggregate
import manifest as manifest_module
from data import CANONICAL_COL

SMILES = ["CCO", "CCN", "CCC", "CCCl"]
OBSERVED = [0.0, 1.0, 2.0, 3.0]

# two seeds whose residuals cancel: each is wrong by a constant magnitude, and
# the seed-averaged prediction is closer than either, so an ensemble row cannot
# be confused with a mean of seed-wise scores
SEED_PREDICTIONS = {
    0: [1.0, 0.0, 3.0, 2.0],
    1: [-2.0, 3.0, 0.0, 5.0],
}

# hand-computed for the predictions above: mae and rmse are the constant
# residual magnitude, rae is that over the truth's mean absolute deviation of
# 1.0, and r2 is 1 - sum(residual^2) / 5
SEED_METRICS = {
    0: {
        "n": 4.0,
        "mae": 1.0,
        "rmse": 1.0,
        "rae": 1.0,
        "r2": 0.2,
        "kendall_tau": 1 / 3,
        "spearman_rho": 0.6,
    },
    1: {
        "n": 4.0,
        "mae": 2.0,
        "rmse": 2.0,
        "rae": 2.0,
        "r2": -2.2,
        "kendall_tau": 2 / 3,
        "spearman_rho": 0.8,
    },
}

SLUG = "emb-none__ro-none__desc-rdkit128__reg-lgbm__cal-none"

CONFIG = {
    "embedding": "none",
    "readout": "none",
    "descriptors": "rdkit",
    "descriptor_pca": 128,
    "regressor": "lgbm",
    "calibration": "none",
}


@pytest.fixture(scope="module")
def spec():
    return manifest_module.load()


def write_run(
    results_root: Path,
    slug: str,
    seed: int,
    *,
    kind: str = "tabular",
    config: dict | None = None,
    commit: str = "abc123",
) -> Path:
    """Write one synthetic run directory matching the sweep's contract."""
    run_dir = results_root / kind / slug / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = SEED_METRICS[seed]

    pd.DataFrame(
        {
            CANONICAL_COL: SMILES,
            "observed": OBSERVED,
            "predicted": SEED_PREDICTIONS[seed],
        }
    ).to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(scores))
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "key": f"{slug}-{seed}",
                "slug": slug,
                "seed": seed,
                "spec": {"name": "tabular_run", "version": 1},
                "config": CONFIG if config is None else config,
                "regressor_params": {"seed": seed},
                "inputs": [{"key": "deadbeef0000", "block": "rdkit", "path": "data/reduced/x"}],
                "n_features": 128,
                "n_fit": 4392,
                "n_test": len(SMILES),
                "metrics": scores,
                "calibrated": False,
                "environment": {"repo": {"commit": commit, "dirty": False}},
                "written_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    return run_dir


@pytest.fixture
def two_seeds(tmp_path):
    """Build a results root holding one configuration at two seeds."""
    for seed in SEED_PREDICTIONS:
        write_run(tmp_path, SLUG, seed)
    return tmp_path


def test_every_run_becomes_one_row_with_the_contracted_columns(two_seeds):
    frame = aggregate.tidy(two_seeds)

    assert list(frame.columns) == list(aggregate.ROW_COLUMNS)
    assert len(frame) == 2
    assert set(frame["seed"]) == {0, 1}
    assert frame["kind"].unique().tolist() == ["tabular"]
    assert frame["regressor"].unique().tolist() == ["lgbm"]
    assert frame["descriptor_pca"].unique().tolist() == [128]
    assert frame["input_blocks"].unique().tolist() == ["rdkit"]
    assert frame["commit"].unique().tolist() == ["abc123"]


def test_the_tidy_table_round_trips_through_parquet(two_seeds):
    frame = aggregate.tidy(two_seeds)

    path = aggregate.write_table(frame, two_seeds)

    assert path == two_seeds / "results.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)


def test_a_configuration_is_summarized_across_its_seeds(two_seeds):
    summary = aggregate.seed_summary(aggregate.tidy(two_seeds))

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_seeds"] == 2
    # the seeds score 1.0 and 2.0, so the mean is 1.5 and the sample sd is sqrt(0.5)
    assert row["mae"] == pytest.approx(1.5)
    assert row["mae_std"] == pytest.approx(0.5**0.5)
    assert row["aggregation"] == aggregate.SEED_MEAN


def test_the_ensemble_row_scores_averaged_predictions_and_is_labeled_apart(two_seeds):
    summary = aggregate.summarize(aggregate.tidy(two_seeds))

    ensembles = summary.loc[summary["aggregation"] == aggregate.ENSEMBLE]
    assert len(ensembles) == 1
    row = ensembles.iloc[0]
    # the seed-averaged prediction is wrong by 0.5 everywhere, well inside the
    # 1.5 mean of the seed-wise scores, so the two rows cannot be interchanged
    assert row["mae"] == pytest.approx(0.5)
    assert pd.isna(row["mae_std"])
    assert len(summary) == 2


def test_a_configuration_with_one_seed_has_no_spread(tmp_path):
    write_run(tmp_path, SLUG, 0)

    summary = aggregate.seed_summary(aggregate.tidy(tmp_path))

    assert summary.iloc[0]["n_seeds"] == 1
    assert pd.isna(summary.iloc[0]["mae_std"])


def test_coverage_names_both_a_missing_planned_run_and_an_unplanned_one(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_run(tmp_path, configs[0].slug, 0, config=configs[0].as_dict())
    unplanned = write_run(tmp_path, "emb-none__ro-none__desc-nowhere__reg-lgbm__cal-none", 0)

    coverage = aggregate.stage_coverage(spec, "descriptor_width", results_root=tmp_path)

    assert configs[0].run_dir(0, tmp_path) not in coverage.missing
    assert configs[0].run_dir(1, tmp_path) in coverage.missing
    assert configs[1].run_dir(0, tmp_path) in coverage.missing
    assert unplanned in coverage.unplanned
    assert not coverage.is_complete


def test_a_partial_sweep_reads_as_partial(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_run(tmp_path, configs[0].slug, 0, config=configs[0].as_dict())

    coverage = aggregate.stage_coverage(spec, "descriptor_width", results_root=tmp_path)
    report = aggregate.coverage_report(coverage, "descriptor_width")

    assert report.startswith(f"descriptor_width: 1/{len(coverage.expected)} planned runs")
    assert "missing:" in report


def test_a_run_without_a_record_is_refused_rather_than_skipped(two_seeds):
    (two_seeds / "tabular" / SLUG / "seed0" / "run.json").unlink()

    with pytest.raises(aggregate.AggregateError, match="no run.json"):
        aggregate.tidy(two_seeds)


def test_a_malformed_record_is_refused_rather_than_skipped(two_seeds):
    record = two_seeds / "tabular" / SLUG / "seed1" / "run.json"
    record.write_text("{not json at all")

    with pytest.raises(aggregate.AggregateError, match="not valid JSON"):
        aggregate.tidy(two_seeds)


def test_a_record_naming_another_configuration_is_refused(tmp_path):
    run_dir = write_run(tmp_path, SLUG, 0)
    record = json.loads((run_dir / "run.json").read_text())
    (run_dir / "run.json").write_text(json.dumps({**record, "slug": "somewhere-else"}))

    with pytest.raises(aggregate.AggregateError, match="has been moved"):
        aggregate.tidy(tmp_path)


def test_ensembling_seeds_that_cover_different_compounds_is_refused(two_seeds):
    path = two_seeds / "tabular" / SLUG / "seed1"
    frame = pd.read_csv(path / "predictions.csv")
    frame.loc[0, CANONICAL_COL] = "c1ccccc1"
    frame.to_csv(path / "predictions.csv", index=False)

    with pytest.raises(aggregate.AggregateError, match="different compounds"):
        aggregate.ensemble_summary(aggregate.tidy(two_seeds))


def test_an_empty_results_root_gives_a_typed_empty_table(tmp_path):
    frame = aggregate.tidy(tmp_path)

    assert frame.empty
    assert list(frame.columns) == list(aggregate.ROW_COLUMNS)
    assert aggregate.summarize(frame).empty
