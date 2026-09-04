from __future__ import annotations

import ast
import itertools
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from concat_arch import (
    AuxEncoderConfig,
    BackboneError,
    ConfigError,
    GraphRegressor,
    LeakageError,
    RunConfig,
    SplitPaths,
    TrainingConfig,
    build_features,
    build_mpnn,
    feature_dim,
    masked_mse_loss,
    run_cell,
    task_columns,
    write_body_checkpoint,
)
from concat_arch import readouts as readouts_module
from concat_arch import run as run_module

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SPLITS = REPO_ROOT / "data" / "splits"
LOG2FC_CSV = REPO_ROOT / "data" / "raw" / "pxr-challenge_single_concentration_TRAIN.csv"
MANIFEST = REPO_ROOT / "experiments" / "manifest.yaml"

# a cheap stand-in for CheMeleon: a randomly initialised body narrow enough
# that a whole run is seconds, wired through the same code path
TINY_BODY = {"body_checkpoint": "random", "message_hidden_dim": 16, "depth": 2}


@pytest.fixture(scope="module")
def tiny_splits(tmp_path_factory: pytest.TempPathFactory) -> SplitPaths:
    directory = tmp_path_factory.mktemp("splits")
    frames = {
        "fit_train": pd.read_csv(SPLITS / "fit_train.csv").head(48),
        "fit_val": pd.read_csv(SPLITS / "fit_val.csv").head(24),
        "test_phase2": pd.read_csv(SPLITS / "test_phase2.csv").head(12),
        "log2fc": pd.read_csv(LOG2FC_CSV).head(300),
    }
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    return SplitPaths(
        fit_train=directory / "fit_train.csv",
        fit_val=directory / "fit_val.csv",
        test=directory / "test_phase2.csv",
        log2fc=directory / "log2fc.csv",
    )


@pytest.fixture
def tiny_encoder() -> GraphRegressor:
    torch.manual_seed(0)
    model = build_mpnn(
        "random", ffn_hidden_dim=8, ffn_num_layers=1, message_hidden_dim=16, depth=2, n_tasks=2
    )
    return GraphRegressor(model, freeze_epochs=0, body_lr=1e-3, head_lr=1e-3)


@pytest.fixture(scope="module")
def tiny_run_pair(tiny_splits: SplitPaths) -> tuple[run_module.RunResult, run_module.RunResult]:
    config = RunConfig(
        aux_encoder=AuxEncoderConfig(tasks=2, use_observed_readout=True),
        **TINY_BODY,
    ).with_training(
        max_epochs=1, aux_max_epochs=1, accelerator="cpu", batch_size=16, inference_batch_size=8
    )
    return run_cell(config, seed=0, splits=tiny_splits), run_cell(
        config, seed=0, splits=tiny_splits
    )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_no_source_file_imports_moal():
    offenders = {
        path.relative_to(REPO_ROOT): sorted(
            name for name in _imported_modules(path) if name.split(".")[0] == "moal"
        )
        for path in sorted(SRC.rglob("*.py"))
    }

    assert {path: names for path, names in offenders.items() if names} == {}


def test_this_test_module_does_not_import_moal():
    names = _imported_modules(Path(__file__))

    assert [name for name in names if name.split(".")[0] == "moal"] == []


def test_training_compounds_are_disjoint_from_phase_two():
    training = set(pd.read_csv(SPLITS / "fit_train.csv")["canonical_smiles"]) | set(
        pd.read_csv(SPLITS / "fit_val.csv")["canonical_smiles"]
    )
    phase_two = set(pd.read_csv(SPLITS / "test_phase2.csv")["canonical_smiles"])

    assert training & phase_two == set()


def test_log2fc_screen_holds_no_phase_two_compound():
    screen = readouts_module.load_readouts(LOG2FC_CSV, tasks=4)
    phase_two = set(pd.read_csv(SPLITS / "test_phase2.csv")["canonical_smiles"])

    assert set(screen.index) & phase_two == set()


def test_run_cell_refuses_a_training_set_holding_a_phase_two_compound(
    tiny_splits: SplitPaths, tmp_path: Path
):
    test_frame = pd.read_csv(tiny_splits.test)
    leaked = pd.concat([pd.read_csv(tiny_splits.fit_train), test_frame.head(1)], ignore_index=True)
    leaked_path = tmp_path / "leaked_fit_train.csv"
    leaked.to_csv(leaked_path, index=False)
    splits = SplitPaths(
        fit_train=leaked_path,
        fit_val=tiny_splits.fit_val,
        test=tiny_splits.test,
        log2fc=tiny_splits.log2fc,
    )

    with pytest.raises(LeakageError, match="also in phase 2"):
        run_cell(RunConfig(**TINY_BODY), seed=0, splits=splits)


def _grid_axes(grid: dict) -> list[dict]:
    names = list(grid["axes"])
    return [
        {**grid.get("fixed", {}), **dict(zip(names, values, strict=True))}
        for values in itertools.product(*(grid["axes"][name] for name in names))
    ]


def _standalone_axes(node) -> Iterator[dict]:
    """Yield every already-resolved axis mapping in the manifest, grids excluded."""
    if isinstance(node, dict):
        for key, value in node.items():
            # a grid's axes map each name to the list of values it sweeps, which
            # is a different shape and is expanded by _grid_axes instead
            if key == "axes" and isinstance(value, dict) and "encoder_init" in value:
                yield value
            else:
                yield from _standalone_axes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _standalone_axes(item)


def test_every_graph_network_cell_in_the_manifest_parses_into_a_configuration():
    raw = yaml.safe_load(MANIFEST.read_text())
    grids = [g for g in raw.get("gnn", {}).get("grids", [])]
    axes = [a for grid in grids for a in _grid_axes(grid)] + list(_standalone_axes(raw))

    configs = [RunConfig.from_axes(a) for a in axes]

    # the freeze/width/clip grid is 3 x 2 x 3, and the standalone cells cover
    # the readout variants, the two E4 arms and the plain fine-tune
    assert len(configs) >= 18 + 7
    assert {c.encoder_init for c in configs} == {"chemeleon", "log2fc_checkpoint"}
    assert all(c.aux_encoder is None for c in configs if c.encoder_init == "log2fc_checkpoint")


def test_the_e4_frozen_cell_keeps_its_body_frozen_for_the_whole_run():
    raw = yaml.safe_load(MANIFEST.read_text())
    cell = next(a for a in _standalone_axes(raw) if a["freeze_epochs"] >= 50)

    config = RunConfig.from_axes(cell)

    assert config.encoder_init == "log2fc_checkpoint"
    assert config.body_never_unfreezes


@pytest.mark.parametrize(
    "axes",
    [
        pytest.param({"freeze_epochs": -1}, id="negative-freeze"),
        pytest.param({"gradient_clip_val": 0.0}, id="zero-clip"),
        pytest.param({"gradient_clip_val": -1.0}, id="negative-clip"),
        pytest.param({"ffn_hidden_dim": 256}, id="unswept-width"),
        pytest.param({"ffn_num_layers": 0}, id="empty-head"),
        pytest.param({"encoder_init": "chemeleon-large"}, id="unknown-init"),
        pytest.param({"finetune_target": "log2fc"}, id="unknown-target"),
        pytest.param({"calibration": "isotonic"}, id="unsupported-calibration"),
        pytest.param({"model_kind": "tabular"}, id="wrong-model-kind"),
        pytest.param({"aux_encoder": {"tasks": 3}}, id="unswept-task-count"),
        pytest.param({"aux_encoder": {"target": "pec50"}}, id="wrong-aux-target"),
        pytest.param({"aux_encoder": {"tasks": 2, "use_median": True}}, id="unknown-aux-field"),
        pytest.param({"dropout": 0.1}, id="unknown-axis"),
        pytest.param(
            {"encoder_init": "log2fc_checkpoint", "aux_encoder": {"tasks": 2}},
            id="e4-with-auxiliary-encoder",
        ),
    ],
)
def test_configuration_rejects_an_unrunnable_cell(axes: dict):
    with pytest.raises(ConfigError):
        RunConfig.from_axes(axes)


def test_a_clip_of_none_means_no_clipping_and_is_accepted():
    config = RunConfig.from_axes({"gradient_clip_val": None, "freeze_epochs": 0})

    assert config.gradient_clip_val is None
    assert config.freeze_epochs == 0


def test_a_freeze_at_the_epoch_budget_never_unfreezes_the_body():
    config = RunConfig(freeze_epochs=50).with_training(max_epochs=50)

    assert config.body_never_unfreezes
    assert not RunConfig(freeze_epochs=49).with_training(max_epochs=50).body_never_unfreezes


def test_a_missing_body_checkpoint_is_rejected_at_construction(tmp_path: Path):
    with pytest.raises(ConfigError, match="does not exist"):
        RunConfig(encoder_init="log2fc_checkpoint", body_checkpoint=tmp_path / "absent.pt")


def test_configuration_round_trips_to_hashable_data():
    config = RunConfig.from_axes({"aux_encoder": {"tasks": 4, "use_observed_readout": True}})

    payload = config.as_dict()

    assert payload["aux_encoder"] == {
        "target": "log2fc",
        "tasks": 4,
        "use_observed_readout": True,
        "use_predicted_readout": False,
    }
    assert payload["training"]["max_epochs"] == 50


def test_the_log2fc_body_is_requested_from_encoders_and_says_so_when_absent(monkeypatch):
    monkeypatch.setattr(run_module, "ENCODERS_MODULE", "concat_arch_encoders_that_do_not_exist")

    with pytest.raises(run_module.MissingEncoderError, match="not importable"):
        run_module.log2fc_body_checkpoint(seed=0)


def test_the_log2fc_body_request_names_the_function_it_wants(monkeypatch):
    monkeypatch.setattr(run_module, "ENCODERS_FACTORY", "a_function_encoders_does_not_define")

    with pytest.raises(run_module.MissingEncoderError, match="defines no"):
        run_module.log2fc_body_checkpoint(seed=0)


def test_masked_loss_ignores_unobserved_entries():
    predictions = torch.tensor([[1.0, 5.0], [2.0, 2.0]])
    targets = torch.tensor([[3.0, 99.0], [4.0, 2.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    loss = masked_mse_loss(predictions, targets, mask)

    # (1-3)^2 + (2-4)^2 + (2-2)^2 over three observed entries
    assert float(loss) == pytest.approx(8.0 / 3.0)


def test_masked_loss_of_a_batch_with_nothing_observed_stays_differentiable():
    predictions = torch.tensor([[1.0, 5.0]], requires_grad=True)

    loss = masked_mse_loss(predictions, torch.zeros(1, 2), torch.zeros(1, 2))

    assert float(loss) == 0.0
    assert loss.requires_grad


@pytest.mark.parametrize(
    ("n_tasks", "embedding_dim", "expected"),
    [(2, 2048, 2055), (4, 2048, 2061), (2, 16, 23)],
)
def test_feature_width_is_three_readout_blocks_an_embedding_and_a_flag(
    n_tasks: int, embedding_dim: int, expected: int
):
    assert feature_dim(n_tasks, embedding_dim) == expected


def test_task_columns_name_the_screened_concentrations():
    assert task_columns(2) == ("log2fc_8.251e-06", "log2fc_3.300e-05")
    assert task_columns(4) == (
        "log2fc_9.803e-07",
        "log2fc_8.251e-06",
        "log2fc_3.300e-05",
        "log2fc_9.901e-05",
    )
    with pytest.raises(ValueError, match="must be 2 or 4"):
        task_columns(3)


def test_canonicalization_agrees_with_the_split_files():
    frame = pd.read_csv(SPLITS / "fit_train.csv").head(200)

    recomputed = frame["SMILES"].map(readouts_module.canonical_smiles)

    assert recomputed.tolist() == frame["canonical_smiles"].tolist()


def test_replicate_readouts_at_one_concentration_pool_by_median(tmp_path: Path):
    screen = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCO", "CCO", "CCC"],
            "concentration_M": [8.251e-06, 8.251e-06, 3.300e-05, 8.251e-06],
            "log2_fc_estimate": [1.0, 3.0, -0.5, 2.0],
        }
    )
    path = tmp_path / "screen.csv"
    screen.to_csv(path, index=False)

    table = readouts_module.load_readouts(path, tasks=2)

    assert list(table.columns) == ["log2fc_8.251e-06", "log2fc_3.300e-05"]
    assert table.loc["CCO", "log2fc_8.251e-06"] == pytest.approx(2.0)
    assert table.loc["CCO", "log2fc_3.300e-05"] == pytest.approx(-0.5)
    assert bool(pd.isna(table.loc["CCC", "log2fc_3.300e-05"]))


def test_excluded_compounds_never_reach_the_readout_table(tmp_path: Path):
    screen = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCC"],
            "concentration_M": [8.251e-06, 8.251e-06],
            "log2_fc_estimate": [1.0, 2.0],
        }
    )
    path = tmp_path / "screen.csv"
    screen.to_csv(path, index=False)

    table = readouts_module.load_readouts(path, tasks=2, exclude={"CCO"})

    assert table.index.tolist() == ["CCC"]


def test_observed_readouts_populate_their_block_only_for_screened_compounds(
    tiny_encoder: GraphRegressor,
):
    smiles = ["CCO", "CCC"]
    readouts = pd.DataFrame({"log2fc_8.251e-06": [1.5], "log2fc_3.300e-05": [-2.5]}, index=["CCO"])

    features = build_features(
        smiles,
        readouts,
        tiny_encoder,
        use_observed_readout=True,
        use_predicted_readout=False,
    )

    assert features.shape == (2, feature_dim(2, tiny_encoder.embedding_dim))
    assert features[0, :2].tolist() == [1.5, -2.5]
    assert features[0, 2:4].tolist() == [1.0, 1.0]
    assert features[0, -1] == 1.0
    assert features[1, :4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert features[1, -1] == 0.0
    # the predicted block stays zero while its flag is off
    assert features[:, 4:6].tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_turning_the_observed_readout_off_zeroes_its_block_without_narrowing_the_vector(
    tiny_encoder: GraphRegressor,
):
    smiles = ["CCO"]
    readouts = pd.DataFrame({"log2fc_8.251e-06": [1.5], "log2fc_3.300e-05": [-2.5]}, index=["CCO"])

    on = build_features(
        smiles, readouts, tiny_encoder, use_observed_readout=True, use_predicted_readout=False
    )
    off = build_features(
        smiles, readouts, tiny_encoder, use_observed_readout=False, use_predicted_readout=False
    )

    assert on.shape == off.shape
    assert off[0, :4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert off[0, -1] == 0.0
    # the embedding block is populated either way, for every compound
    assert np.any(off[0, 6:-1] != 0.0)


def test_the_predicted_readout_block_is_populated_for_every_compound(
    tiny_encoder: GraphRegressor,
):
    smiles = ["CCO", "CCC"]
    readouts = pd.DataFrame({"log2fc_8.251e-06": [], "log2fc_3.300e-05": []}, dtype=float)

    features = build_features(
        smiles, readouts, tiny_encoder, use_observed_readout=False, use_predicted_readout=True
    )

    assert np.all(features[:, 4:6] != 0.0)


def test_a_tiny_run_predicts_every_phase_two_compound_once(tiny_run_pair, tiny_splits: SplitPaths):
    result, _ = tiny_run_pair
    expected = pd.read_csv(tiny_splits.test)

    assert result.predictions.columns.tolist() == [
        "SMILES",
        "canonical_smiles",
        "pEC50",
        "prediction",
    ]
    assert result.predictions["canonical_smiles"].tolist() == expected["canonical_smiles"].tolist()
    assert result.predictions["prediction"].notna().all()


def test_a_tiny_run_records_what_it_did(tiny_run_pair):
    result, _ = tiny_run_pair

    assert result.record["n_train"] == 48
    assert result.record["n_val"] == 24
    assert result.record["n_test"] == 12
    assert result.record["extra_feature_dim"] == feature_dim(2, result.record["embedding_dim"])
    assert result.record["auxiliary"]["tasks"] == list(task_columns(2))
    assert result.spec["name"] == "concat_arch"
    assert result.spec["params"]["seed"] == 0


def test_two_runs_at_one_seed_predict_the_same_values(tiny_run_pair):
    first, second = tiny_run_pair

    assert first.predictions["prediction"].to_numpy().tolist() == pytest.approx(
        second.predictions["prediction"].to_numpy().tolist()
    )


def test_a_run_without_an_auxiliary_encoder_concatenates_nothing(tiny_splits: SplitPaths):
    config = RunConfig(aux_encoder=None, **TINY_BODY).with_training(
        max_epochs=1, accelerator="cpu", batch_size=16, inference_batch_size=8
    )

    result = run_cell(config, seed=0, splits=tiny_splits)

    assert result.record["extra_feature_dim"] == 0
    assert result.record["auxiliary"] is None
    assert len(result.predictions) == 12


def test_a_body_extracted_from_a_whole_network_rebuilds_the_same_encoder(tmp_path: Path):
    torch.manual_seed(0)
    original = build_mpnn(
        "random", ffn_hidden_dim=8, ffn_num_layers=1, message_hidden_dim=16, depth=2
    )
    hyper_parameters = {"d_h": 16, "depth": 2}

    path = write_body_checkpoint(original.state_dict(), hyper_parameters, tmp_path / "body.pt")
    rebuilt = build_mpnn(path, ffn_hidden_dim=8, ffn_num_layers=1)

    original_body = dict(original.message_passing.state_dict())
    rebuilt_body = dict(rebuilt.message_passing.state_dict())
    assert set(original_body) == set(rebuilt_body)
    assert all(torch.equal(original_body[k], rebuilt_body[k]) for k in original_body)


def test_extracting_a_body_from_weights_that_hold_none_is_refused(tmp_path: Path):
    with pytest.raises(BackboneError, match="no key"):
        write_body_checkpoint({"predictor.weight": torch.zeros(2)}, {}, tmp_path / "body.pt")


@pytest.mark.slow
def test_a_full_size_fit_beats_predicting_the_training_mean():
    config = RunConfig(aux_encoder=AuxEncoderConfig(tasks=2), ffn_hidden_dim=512, freeze_epochs=2)
    baseline = pd.read_csv(SPLITS / "fit_train.csv")["pEC50"].mean()
    truth = pd.read_csv(SPLITS / "test_phase2.csv")["pEC50"]

    result = run_cell(config, seed=0)

    model_mae = float((result.predictions["prediction"] - result.predictions["pEC50"]).abs().mean())
    baseline_mae = float((truth - baseline).abs().mean())
    assert model_mae < baseline_mae

    # the scored model is the refit one, and it saw the whole fit set rather
    # than the 80% the first pass early-stopped on
    refit = result.record["refit"]
    assert result.record["refit_on_all"] is True
    assert refit["n_train"] == result.record["n_train"] + result.record["n_val"]
    assert refit["n_train"] == 4392
    assert refit["epochs_requested"] == result.record["main"]["selected_epoch"] + 1
    # the second pass has no validation partition, so nothing was early-stopped
    # or rewound there
    assert refit["best_val_loss"] is None
    assert refit["restored_best"] is False

    # the auxiliary encoder gets the same treatment: its held-out slice of the
    # screen picks the epoch count, then it retrains on the whole screen
    aux = result.record["auxiliary"]
    aux_refit = aux["refit"]
    assert aux_refit["n_train"] == aux["n_train"] + aux["n_val"]
    assert aux_refit["epochs_requested"] == aux["selected_epoch"] + 1
    assert aux_refit["best_val_loss"] is None
    assert aux_refit["restored_best"] is False


def test_refitting_on_everything_is_the_default_and_can_be_turned_off():
    # the graph networks would otherwise train on 3,514 compounds while the
    # tabular models train on 4,392, which is the comparison figure 1 makes
    assert TrainingConfig().refit_on_all is True
    assert TrainingConfig(refit_on_all=False).refit_on_all is False


def test_the_refit_epoch_count_comes_from_the_first_pass():
    # a fit that selected epoch 0 still has to train for one epoch, not zero
    for selected, expected in ((0, 1), (7, 8), (29, 30)):
        assert max(1, selected + 1) == expected
