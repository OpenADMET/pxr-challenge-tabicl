import dataclasses
import json
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import aggregate  # noqa: E402
import evaluate  # noqa: E402
import figures  # noqa: E402
import manifest as manifest_module  # noqa: E402
from data import CANONICAL_COL  # noqa: E402
from manifest import TabularConfig  # noqa: E402

# the challenge's own phase-2 count, so the anchor applies to these slices
N_COMPOUNDS = 260
COMPOUNDS = [f"C{index:03d}" for index in range(N_COMPOUNDS)]
OBSERVED = np.linspace(4.0, 8.0, N_COMPOUNDS)

SEEDS = (0, 1, 2, 3, 4)

# a bootstrap over 260 compounds is the point, but the interval's precision is
# not what these tests check, so the resample count stays small
RESAMPLES = 200

# one single-block embedding configuration, one single-block descriptor one,
# and one that combines them, so a block-count selection has something to
# include and something to exclude
EMBEDDING_ONLY = TabularConfig("chemeleon", "none", "none", 128, "lgbm", "none")
DESCRIPTORS_ONLY = TabularConfig("none", "none", "rdkit", 128, "lgbm", "none")
COMBINED = TabularConfig("chemeleon", "none", "rdkit", 128, "lgbm", "none")

# the same descriptor block at another width, and the calibrated arm of it
WIDER_DESCRIPTORS = TabularConfig("none", "none", "rdkit", 256, "lgbm", "none")
CALIBRATED = TabularConfig("none", "none", "rdkit", 128, "lgbm", "isotonic_fitval")

# the two regressors that report a spread, and the kinds differ between them
TABPFN = TabularConfig("chemeleon", "none", "none", 128, "tabpfn-v2.5", "none")
TABICL = TabularConfig("chemeleon", "none", "none", 128, "tabicl", "none")

GNN_CELL = "chemeleon_pec50"


@pytest.fixture(scope="module")
def spec():
    return manifest_module.load()


def predictions_for(key: str, offset: float, seed: int) -> np.ndarray:
    """Return predictions displaced from the truth by a configuration-specific amount.

    The displacement varies compound by compound, deterministically per
    configuration, so that a paired bootstrap over compounds has something to
    resample; a constant offset would make every resample identical.
    """
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    per_compound = 0.6 + 0.8 * rng.random(N_COMPOUNDS)
    wobble = 0.01 * seed * np.cos(np.arange(N_COMPOUNDS))
    return OBSERVED + offset * per_compound + wobble


def write_run(
    root: Path,
    config: TabularConfig,
    seed: int,
    *,
    offset: float = 0.4,
    with_std: bool = False,
    compounds: list[str] | None = None,
) -> Path:
    """Write one synthetic tabular run directory."""
    compounds = compounds or COMPOUNDS
    predicted = predictions_for(config.slug, offset, seed)[: len(compounds)]
    observed = OBSERVED[: len(compounds)]
    run_dir = root / "tabular" / config.slug / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame({CANONICAL_COL: compounds, "observed": observed, "predicted": predicted})
    if with_std:
        # a spread that tracks the error loosely, so a diagnostic figure has
        # something other than noise to draw
        frame["predicted_std"] = np.abs(predicted - observed) * 1.2 + 0.05
    frame.to_csv(run_dir / "predictions.csv", index=False)

    scores = evaluate.metrics(observed, predicted)
    (run_dir / "metrics.json").write_text(json.dumps(scores))
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "slug": config.slug,
                "seed": seed,
                "config": config.as_dict(),
                "inputs": [{"key": "deadbeef", "block": "rdkit", "path": "data/reduced/x"}],
                "n_features": 128,
                "n_fit": 4392,
                "n_test": len(compounds),
                "metrics": scores,
                "calibrated": config.calibration != "none",
                "environment": {"repo": {"commit": "abc123", "dirty": False}},
            }
        )
    )
    return run_dir


def write_gnn_run(root: Path, cell_id: str, seed: int, *, offset: float = 0.6) -> Path:
    """Write one synthetic graph-network run directory."""
    predicted = predictions_for(cell_id, offset, seed)
    run_dir = root / "gnn" / cell_id / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({CANONICAL_COL: COMPOUNDS, "observed": OBSERVED, "predicted": predicted}).to_csv(
        run_dir / "predictions.csv", index=False
    )

    scores = evaluate.metrics(OBSERVED, predicted)
    (run_dir / "metrics.json").write_text(json.dumps(scores))
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "cell": cell_id,
                "seed": seed,
                "config": {},
                "inputs": [],
                "n_test": N_COMPOUNDS,
                "metrics": scores,
                "environment": {"repo": {"commit": "abc123", "dirty": False}},
            }
        )
    )
    return run_dir


@pytest.fixture
def results_root(tmp_path):
    """Build a results root spanning every axis the seven figures select on."""
    offsets = {
        EMBEDDING_ONLY: 0.45,
        DESCRIPTORS_ONLY: 0.52,
        COMBINED: 0.40,
        WIDER_DESCRIPTORS: 0.55,
        CALIBRATED: 0.47,
    }
    for config, offset in offsets.items():
        for seed in SEEDS:
            write_run(tmp_path, config, seed, offset=offset)
    for config, offset in ((TABPFN, 0.60), (TABICL, 0.50)):
        for seed in SEEDS:
            write_run(tmp_path, config, seed, offset=offset, with_std=True)
    for seed in SEEDS:
        write_gnn_run(tmp_path, GNN_CELL, seed)
    return tmp_path


@pytest.fixture
def frame(results_root):
    """Read the synthetic runs into the tidy table the figures are drawn from."""
    return aggregate.tidy(results_root)


def test_figure_one_selects_the_graph_network_runs_and_no_tabular_ones(spec, frame):
    data = figures.slice_for(spec, "fig1", frame)

    assert set(data["kind"]) == {"gnn"}
    assert set(data["config_id"]) == {GNN_CELL}
    assert len(data) == len(SEEDS)


def test_figure_two_selects_only_single_block_configurations(spec, frame):
    data = figures.slice_for(spec, "fig2", frame)

    assert set(data["n_blocks"]) == {1}
    assert COMBINED.slug not in set(data["config_id"])
    assert EMBEDDING_ONLY.slug in set(data["config_id"])
    assert DESCRIPTORS_ONLY.slug in set(data["config_id"])


def test_figure_two_excludes_other_widths_calibrated_arms_and_graph_networks(spec, frame):
    data = figures.slice_for(spec, "fig2", frame)

    selected = set(data["config_id"])
    assert WIDER_DESCRIPTORS.slug not in selected
    assert CALIBRATED.slug not in selected
    assert GNN_CELL not in selected


def test_figure_three_keeps_the_combined_featureset_the_single_block_figure_drops(spec, frame):
    combinations = figures.slice_for(spec, "fig3", frame)
    single_blocks = figures.slice_for(spec, "fig2", frame)

    assert COMBINED.slug in set(combinations["config_id"])
    # a combination is two blocks or more, so the two figures partition the
    # space rather than one containing the other
    assert set(single_blocks["config_id"]).isdisjoint(set(combinations["config_id"]))
    assert EMBEDDING_ONLY.slug in set(single_blocks["config_id"])


def test_figure_five_selects_only_descriptor_bearing_configurations(spec, frame):
    data = figures.slice_for(spec, "fig5", frame)

    assert set(data["descriptors"]) == {"rdkit"}
    assert EMBEDDING_ONLY.slug not in set(data["config_id"])
    assert WIDER_DESCRIPTORS.slug in set(data["config_id"])
    assert CALIBRATED.slug not in set(data["config_id"])


def test_figure_six_selects_both_calibration_arms(spec, frame):
    data = figures.slice_for(spec, "fig6", frame)

    assert set(data["calibration"]) == {"none", "isotonic_fitval"}
    # the calibrated configuration's own uncalibrated twin has to be there for
    # the arms to be comparable
    assert CALIBRATED.slug in set(data["config_id"])
    uncalibrated = TabularConfig(**{**CALIBRATED.as_dict(), "calibration": "none"})
    assert uncalibrated.slug in set(data["config_id"])


def test_figure_seven_selects_only_regressors_that_report_a_spread(spec, frame):
    data = figures.slice_for(spec, "fig7", frame)

    assert set(data["regressor"]) == {"tabpfn-v2.5", "tabicl"}
    assert EMBEDDING_ONLY.slug not in set(data["config_id"])


def test_an_unknown_selector_is_refused_rather_than_silently_ignored(frame):
    with pytest.raises(figures.FigureError, match="not a column"):
        figures.select_rows(figures.annotate(frame), {"nonexistent_axis": 1})


def test_an_empty_slice_is_skipped_rather_than_drawn(spec, tmp_path):
    for seed in SEEDS:
        write_run(tmp_path, COMBINED, seed)
    out_dir = tmp_path / "figures"

    path = figures.render(spec, "fig2", aggregate.tidy(tmp_path), out_dir)

    assert path is None
    assert not (out_dir / "fig2.png").exists()


def test_render_all_names_every_figure_it_skipped(spec, tmp_path):
    for seed in SEEDS:
        write_run(tmp_path, COMBINED, seed)
    out_dir = tmp_path / "figures"

    written, skipped = figures.render_all(
        spec, aggregate.tidy(tmp_path), out_dir, ["fig1", "fig2", "fig3"], n_resamples=RESAMPLES
    )

    assert set(written) == {"fig3"}
    assert {entry.figure_id for entry in skipped} == {"fig1", "fig2"}
    assert all(entry.reason for entry in skipped)


def test_a_figure_writes_the_file_it_claims_to(spec, frame, tmp_path):
    out_dir = tmp_path / "figures"

    path = figures.render(spec, "fig2", frame, out_dir)

    assert path is not None
    assert path == out_dir / "fig2.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_render_all_writes_every_file_it_reports(spec, frame, tmp_path):
    out_dir = tmp_path / "figures"

    written, _ = figures.render_all(spec, frame, out_dir, n_resamples=RESAMPLES)

    assert set(written) == {"fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7"}
    assert all(path.exists() for path in written.values())


def test_the_anchor_values_come_from_the_manifest(spec, frame):
    edited = dataclasses.replace(
        spec,
        anchor={"name": "somewhere else", "n": 260, "mae_ensemble": 0.9},
    )

    lines = figures.anchor_lines(edited, figures.slice_for(edited, "fig2", frame))

    assert [value for _, value in lines] == [0.9]
    assert all("somewhere else" in label for label, _ in lines)


def test_only_the_phase_two_score_is_drawn_as_an_anchor(spec, frame):
    # the report's other quoted figure, 0.437, is an out-of-fold score over
    # training compounds; putting it on an axis of phase-2 results would
    # compare different quantities
    lines = figures.anchor_lines(spec, figures.slice_for(spec, "fig2", frame))

    assert [value for _, value in lines] == [spec.anchor["mae_ensemble"]]
    assert 0.437 not in {value for _, value in lines}


def test_the_anchor_is_drawn_at_the_value_the_manifest_carries(spec, frame):
    edited = dataclasses.replace(
        spec,
        anchor={"name": "somewhere else", "n": 260, "mae_ensemble": 0.9},
    )

    figure = figures.figure_2(edited, frame)

    drawn = {
        round(float(np.asarray(line.get_xdata())[0]), 6)
        for line in figure.axes[0].lines
        if str(line.get_label()).startswith("somewhere else")
    }
    plt.close(figure)
    assert drawn == {0.9}


def test_the_anchor_is_withheld_when_the_slice_scores_other_compounds(spec, tmp_path):
    for seed in SEEDS:
        write_run(tmp_path, EMBEDDING_ONLY, seed, compounds=COMPOUNDS[:100])

    lines = figures.anchor_lines(spec, figures.slice_for(spec, "fig2", aggregate.tidy(tmp_path)))

    assert lines == []


def test_the_anchor_is_withheld_on_a_metric_it_does_not_describe(spec, frame):
    lines = figures.anchor_lines(spec, figures.slice_for(spec, "fig2", frame), metric="r2")

    assert lines == []


def test_every_seed_reaches_the_panel_rather_than_only_a_mean(spec, frame):
    figure = figures.figure_1(spec, frame)

    plotted = np.concatenate(
        [np.asarray(collection.get_offsets()) for collection in figure.axes[0].collections]
    )
    plt.close(figure)
    seed_maes = sorted(
        round(value, 6) for value in frame.loc[frame["kind"] == "gnn", "mae"].tolist()
    )
    assert len(set(seed_maes)) == len(SEEDS)
    assert set(seed_maes) <= {round(float(x), 6) for x in plotted[:, 0]}


def test_the_categories_on_the_axis_are_the_featuresets_in_the_slice(spec, frame):
    data = figures.slice_for(spec, "fig2", frame)

    figure = figures.figure_2(spec, frame)

    labels = {text.get_text() for text in figure.axes[0].get_yticklabels()}
    plt.close(figure)
    assert labels == set(data["featureset"].dropna())


def test_the_paired_bootstrap_ranks_the_leader_by_its_ensemble_score(spec, frame):
    data = figures.slice_for(spec, "fig4", frame)

    comparisons = figures.paired_intervals(data, group="regressor", n_resamples=RESAMPLES)

    # lgbm's best configuration sits 0.40 from the truth and no other
    # regressor's best is nearer, so lgbm leads every comparison
    assert {c.leader for c in comparisons} == {COMBINED.slug}
    assert all(c.result.observed > 0 for c in comparisons)


def test_a_clear_gap_separates_and_a_near_tie_does_not(spec, tmp_path):
    close = TabularConfig("chemeleon", "none", "mordred", 128, "lgbm", "none")
    far = TabularConfig("chemeleon", "chemprop_log2fc", "rdkit", 128, "lgbm", "none")
    for seed in SEEDS:
        write_run(tmp_path, COMBINED, seed, offset=0.40)
        write_run(tmp_path, close, seed, offset=0.4001)
        write_run(tmp_path, far, seed, offset=0.80)
    data = figures.slice_for(spec, "fig3", aggregate.tidy(tmp_path))

    by_challenger = {c.challenger: c for c in figures.paired_intervals(data, n_resamples=RESAMPLES)}

    # whichever of the two near-identical configurations leads, the other is
    # its challenger, and the interval between them has to span zero
    near = by_challenger.get(close.slug) or by_challenger[COMBINED.slug]
    assert not near.separates
    assert by_challenger[far.slug].separates


def test_the_uncertainty_points_keep_the_kinds_apart(spec, frame):
    data = figures.slice_for(spec, "fig7", frame)

    points = figures.uncertainty_points(data)

    kinds = points.groupby("regressor")["uncertainty_kind"].nunique()
    assert set(points["regressor"]) == {"tabpfn-v2.5", "tabicl"}
    assert (kinds == 1).all()
    assert (
        points.loc[points["regressor"] == "tabicl", "uncertainty_kind"].iloc[0]
        != (points.loc[points["regressor"] == "tabpfn-v2.5", "uncertainty_kind"].iloc[0])
    )
    assert len(points) == 2 * len(SEEDS) * N_COMPOUNDS


def test_figure_seven_draws_one_labelled_panel_per_spread_reporting_regressor(spec, frame):
    figure = figures.figure_7(spec, frame)

    titles = [ax.get_title() for ax in figure.axes]
    plt.close(figure)
    assert len(titles) == 2
    assert any("tabicl" in title and "p10" in title for title in titles)
    assert any("tabpfn-v2.5" in title and "exact" in title for title in titles)


def test_a_run_without_predictions_is_reported_rather_than_ensembled(tmp_path):
    run_dir = write_run(tmp_path, COMBINED, 0)
    (run_dir / "predictions.csv").unlink()

    with pytest.raises(figures.FigureError, match="no predictions.csv"):
        figures.ensemble_predictions([run_dir])


def test_ensembling_seeds_over_different_compounds_is_refused(tmp_path):
    first = write_run(tmp_path, COMBINED, 0)
    second = write_run(tmp_path, COMBINED, 1, compounds=COMPOUNDS[:100])

    with pytest.raises(figures.FigureError, match="different compounds"):
        figures.ensemble_predictions([first, second])


def test_the_block_count_is_missing_rather_than_zero_for_a_graph_network_run(frame):
    annotated = figures.annotate(frame)

    gnn_rows = annotated.loc[annotated["kind"] == "gnn"]
    assert gnn_rows["n_blocks"].isna().all()
    assert not gnn_rows["has_descriptors"].fillna(False).any()
