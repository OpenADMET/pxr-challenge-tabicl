"""Contract tests for the trained-encoder blocks: leakage, caching, block shape.

The default run touches no encoder training. What it does cover is the part of
this module a mistake would be silent in: that no phase-2 compound reaches a
training label, that a seed names its own cached artifact, and that a block
comes back with the columns and index the stage-one contract requires. The one
test that actually trains is marked ``slow`` and skipped unless
``PXR_RUN_TRAINING`` is set in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import encoders
import features
import provenance
from data import CANONICAL_COL, RAW_DIR, SPLIT_DIR
from encoders import (
    LOG2FC_CONCENTRATION_COL,
    LOG2FC_CONCENTRATIONS,
    LOG2FC_VALUE_COL,
    TASK_COVERAGE_THRESHOLD,
    EncoderConfig,
    EncoderError,
    TrainingSet,
    _frame,
    _log2fc_targets,
    _safe_batch_size,
    _split_training_set,
    _training_set_for,
    embedding_columns,
    encoder_artifact,
    log2fc_training_set,
    pec50_training_set,
    phase2_molecules,
    register,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# molecules the fast tests train nothing on; four distinct, parsable structures
SMILES = ["CCO", "CCC", "c1ccccc1", "CCN"]

RUN_TRAINING = os.environ.get("PXR_RUN_TRAINING")


def _requires_raw_data() -> None:
    if not (RAW_DIR / "pxr-challenge_single_concentration_TRAIN.csv").exists():
        pytest.skip("raw downloads not present; run run/00_fetch_data.py first")
    if not (SPLIT_DIR / "test_phase2.csv").exists():
        pytest.skip("split resources not built; run run/01_build_split.py first")


@pytest.fixture(scope="session")
def log2fc_set() -> TrainingSet:
    _requires_raw_data()
    return log2fc_training_set(seed=0, val_fraction=0.2)


@pytest.fixture(scope="session")
def pec50_set() -> TrainingSet:
    _requires_raw_data()
    return pec50_training_set()


def _long_frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["SMILES", LOG2FC_CONCENTRATION_COL, LOG2FC_VALUE_COL])


def _training_set(n_train: int = 3, n_val: int = 1, value: float = 1.0) -> TrainingSet:
    return TrainingSet(
        task_names=["task"],
        train_smiles=SMILES[:n_train],
        train_targets=np.full((n_train, 1), value),
        val_smiles=SMILES[n_train : n_train + n_val],
        val_targets=np.full((n_val, 1), value),
    )


# ---------------------------------------------------------------------------
# leakage: no phase-2 compound may contribute a training label
# ---------------------------------------------------------------------------


def test_log2fc_training_set_is_disjoint_from_phase_two(log2fc_set):
    phase2 = phase2_molecules()

    labelled = set(log2fc_set.train_smiles) | set(log2fc_set.val_smiles)

    assert labelled & phase2 == set()


def test_pec50_training_set_is_disjoint_from_phase_two(pec50_set):
    phase2 = phase2_molecules()

    labelled = set(pec50_set.train_smiles) | set(pec50_set.val_smiles)

    assert labelled & phase2 == set()


def test_pec50_training_set_fits_on_the_train_partition_only(pec50_set):
    fit_train = set(pd.read_csv(SPLIT_DIR / "fit_train.csv")[CANONICAL_COL])
    fit_val = set(pd.read_csv(SPLIT_DIR / "fit_val.csv")[CANONICAL_COL])

    assert set(pec50_set.train_smiles) <= fit_train
    assert set(pec50_set.val_smiles) <= fit_val
    assert set(pec50_set.train_smiles) & set(pec50_set.val_smiles) == set()


def test_log2fc_training_set_drops_a_phase_two_compound_from_the_pool(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    splits = tmp_path / "splits"
    splits.mkdir()
    rows = []
    for smiles in ("CCO", "CCC", "CCN", "c1ccccc1"):
        rows += [(smiles, c, 1.0) for c in LOG2FC_CONCENTRATIONS]
    _long_frame(rows).to_csv(raw / encoders.LOG2FC_FILE, index=False)
    pd.DataFrame({CANONICAL_COL: ["CCO"]}).to_csv(splits / "test_phase2.csv", index=False)

    training = log2fc_training_set(seed=0, val_fraction=0.25, raw_dir=raw, split_dir=splits)

    assert "CCO" not in set(training.train_smiles) | set(training.val_smiles)
    assert len(training.train_smiles) + len(training.val_smiles) == 3


# ---------------------------------------------------------------------------
# the log2FC task columns, verified against the file rather than assumed
# ---------------------------------------------------------------------------


def test_declared_log2fc_tasks_are_the_only_well_covered_concentrations():
    _requires_raw_data()
    long = pd.read_csv(
        RAW_DIR / encoders.LOG2FC_FILE,
        usecols=["SMILES", LOG2FC_CONCENTRATION_COL, LOG2FC_VALUE_COL],
    )

    coverage = long.groupby(LOG2FC_CONCENTRATION_COL)["SMILES"].nunique() / long["SMILES"].nunique()

    well_covered = set(coverage.index[coverage >= TASK_COVERAGE_THRESHOLD])
    assert well_covered == set(LOG2FC_CONCENTRATIONS)


def test_log2fc_targets_names_one_column_per_task_concentration():
    rows = [("CCO", c, 0.5) for c in LOG2FC_CONCENTRATIONS]
    rows += [("CCC", c, 1.5) for c in LOG2FC_CONCENTRATIONS]

    frame, task_names = _log2fc_targets(_long_frame(rows))

    assert task_names == ["log2fc_8.251e-06", "log2fc_3.300e-05"]
    assert list(frame.columns) == task_names
    assert frame.index.name == CANONICAL_COL
    assert frame.shape == (2, 2)


def test_log2fc_targets_leaves_an_unscreened_concentration_missing():
    rows = [("CCO", c, 0.5) for c in LOG2FC_CONCENTRATIONS]
    rows += [("CCC", LOG2FC_CONCENTRATIONS[0], 1.5)]

    frame, _ = _log2fc_targets(_long_frame(rows))

    assert frame.notna().sum().tolist() == [2, 1]
    assert pd.isna(frame.loc["CCC", "log2fc_3.300e-05"])


def test_log2fc_targets_averages_readings_that_canonicalize_together():
    rows = [
        ("CCO", LOG2FC_CONCENTRATIONS[0], 1.0),
        ("OCC", LOG2FC_CONCENTRATIONS[0], 3.0),
        ("CCO", LOG2FC_CONCENTRATIONS[1], 2.0),
        ("CCC", LOG2FC_CONCENTRATIONS[0], 0.0),
        ("CCC", LOG2FC_CONCENTRATIONS[1], 0.0),
    ]

    frame, _ = _log2fc_targets(_long_frame(rows))

    assert frame.loc["CCO", "log2fc_8.251e-06"] == 2.0


def test_log2fc_targets_rejects_a_file_whose_coverage_disagrees_with_the_tasks():
    rows = []
    for smiles in ("CCO", "CCC"):
        rows += [(smiles, c, 1.0) for c in (*LOG2FC_CONCENTRATIONS, 1e-4)]

    with pytest.raises(ValueError, match="not the tasks"):
        _log2fc_targets(_long_frame(rows))


# ---------------------------------------------------------------------------
# caching: one seed, one artifact
# ---------------------------------------------------------------------------


def test_encoder_key_is_the_same_for_an_identical_configuration():
    training = _training_set()

    first = encoder_artifact("log2fc", EncoderConfig(seed=3), training)
    second = encoder_artifact("log2fc", EncoderConfig(seed=3), training)

    assert first.key == second.key
    assert first.path == second.path


def test_encoder_key_changes_with_the_seed():
    training = _training_set()

    keys = {encoder_artifact("log2fc", EncoderConfig(seed=seed), training).key for seed in range(5)}

    assert len(keys) == 5


def test_encoder_key_changes_when_a_training_label_changes():
    config = EncoderConfig(seed=0)

    unchanged = encoder_artifact("log2fc", config, _training_set(value=1.0))
    changed = encoder_artifact("log2fc", config, _training_set(value=2.0))

    assert unchanged.key != changed.key


def test_encoder_key_separates_the_foundation_initialisation():
    training = _training_set()

    scratch = encoder_artifact("log2fc", EncoderConfig(seed=0, from_foundation=None), training)
    foundation = encoder_artifact(
        "log2fc", EncoderConfig(seed=0, from_foundation="chemeleon"), training
    )

    assert scratch.key != foundation.key
    assert scratch.root != foundation.root


def test_embedding_and_readout_blocks_share_one_encoder():
    embedding = encoders.BLOCK_SPECS["chemprop_log2fc_embedding"]
    readout = encoders.BLOCK_SPECS["chemprop_log2fc_readout"]

    assert embedding.defaults == readout.defaults
    assert embedding.target == readout.target


@pytest.mark.parametrize("name", sorted(encoders.BLOCK_SPECS))
def test_every_block_is_seeded(name):
    assert "seed" in encoders.BLOCK_SPECS[name].defaults


@pytest.mark.parametrize("name", sorted(encoders.BLOCK_SPECS))
def test_block_defaults_construct_an_encoder_configuration(name):
    config = EncoderConfig(**encoders.BLOCK_SPECS[name].defaults)

    assert config.seed == 0


def test_register_installs_every_block_into_a_registry():
    registry: dict[str, features._Block] = {}

    register(registry)

    assert set(registry) == set(encoders.BLOCK_SPECS)
    assert all(isinstance(block, features._Block) for block in registry.values())


def test_registered_blocks_carry_their_defaults_and_version():
    registry: dict[str, features._Block] = {}

    register(registry)
    block = registry["chemeleon_pec50_embedding"]

    assert block.version == encoders.VERSION
    assert block.defaults == encoders.BLOCK_SPECS["chemeleon_pec50_embedding"].defaults


# ---------------------------------------------------------------------------
# block shape and column naming
# ---------------------------------------------------------------------------


def test_embedding_columns_are_zero_padded_and_prefixed():
    assert embedding_columns("chemprop_log2fc", 3) == [
        "chemprop_log2fc_0000",
        "chemprop_log2fc_0001",
        "chemprop_log2fc_0002",
    ]


def test_frame_is_indexed_by_canonical_smiles_in_the_order_asked_for():
    values = np.arange(6.0).reshape(3, 2)

    frame = _frame(SMILES[:3], values, ["a", "b"])

    assert list(frame.index) == SMILES[:3]
    assert frame.index.name == CANONICAL_COL
    assert list(frame.columns) == ["a", "b"]
    assert frame.loc["CCC", "b"] == 3.0


def test_frame_rejects_a_matrix_that_lost_a_molecule():
    values = np.zeros((2, 2))

    with pytest.raises(EncoderError, match="expected"):
        _frame(SMILES[:3], values, ["a", "b"])


@pytest.mark.parametrize("n_rows", [1, 2, 8, 65, 257, 4652])
def test_safe_batch_size_never_leaves_a_single_trailing_molecule(n_rows):
    batch = _safe_batch_size(n_rows, 256)

    assert batch >= 1
    assert n_rows % batch != 1 or batch == 1


# ---------------------------------------------------------------------------
# the seeded validation carve-out
# ---------------------------------------------------------------------------


def _targets_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"task": np.arange(float(n))},
        index=pd.Index([f"C{'C' * i}O" for i in range(n)], name=CANONICAL_COL),
    )


def test_split_training_set_partitions_every_row_exactly_once():
    targets = _targets_frame(20)

    training = _split_training_set(targets, ["task"], seed=1, val_fraction=0.25)

    assert len(training.train_smiles) == 15
    assert len(training.val_smiles) == 5
    assert set(training.train_smiles) & set(training.val_smiles) == set()
    assert set(training.train_smiles) | set(training.val_smiles) == set(targets.index)


def test_split_training_set_is_reproducible_and_seed_dependent():
    targets = _targets_frame(20)

    first = _split_training_set(targets, ["task"], seed=1, val_fraction=0.25)
    again = _split_training_set(targets, ["task"], seed=1, val_fraction=0.25)
    other = _split_training_set(targets, ["task"], seed=2, val_fraction=0.25)

    assert first.val_smiles == again.val_smiles
    assert first.val_smiles != other.val_smiles


def test_split_training_set_carries_the_targets_of_its_rows():
    targets = _targets_frame(10)

    training = _split_training_set(targets, ["task"], seed=0, val_fraction=0.2)

    expected = targets.loc[training.train_smiles, "task"].to_numpy()
    assert np.array_equal(training.train_targets[:, 0], expected)


def test_pec50_encoder_refuses_a_validation_carve_out():
    with pytest.raises(ValueError, match="val_fraction must be None"):
        _training_set_for("pec50", EncoderConfig(seed=0, val_fraction=0.2))


def test_log2fc_encoder_requires_a_validation_carve_out():
    with pytest.raises(ValueError, match="needs a val_fraction"):
        _training_set_for("log2fc", EncoderConfig(seed=0, val_fraction=None))


def test_unknown_encoder_target_is_rejected():
    with pytest.raises(KeyError):
        _training_set_for("solubility", EncoderConfig(seed=0))


# ---------------------------------------------------------------------------
# the one test that trains
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not RUN_TRAINING, reason="set PXR_RUN_TRAINING=1 to train an encoder")
def test_one_epoch_encoder_produces_an_embedding_and_a_readout_block(tmp_path, monkeypatch):
    _requires_raw_data()
    monkeypatch.setattr(encoders, "CHECKPOINT_DIR", tmp_path / "encoders")
    registry = dict(features.BLOCKS)
    register(registry)
    monkeypatch.setattr(features, "BLOCKS", registry)
    molecules = features.unique_molecules()[:8]
    params = {"max_epochs": 1, "freeze_epochs": 0, "patience": 1}

    embedding = features.load(
        features.build(
            "chemprop_log2fc_embedding", molecules, params=params, cache_dir=tmp_path / "features"
        )
    )
    readout = features.load(
        features.build(
            "chemprop_log2fc_readout", molecules, params=params, cache_dir=tmp_path / "features"
        )
    )

    assert embedding.shape == (8, 256)
    assert list(embedding.index) == molecules
    assert readout.shape == (8, 2)
    assert list(readout.columns) == ["log2fc_8.251e-06", "log2fc_3.300e-05"]
    # both blocks resolved to the one checkpoint rather than training twice
    assert len(list((tmp_path / "encoders" / "chemprop_log2fc").glob("*.pt"))) == 1


def test_every_encoder_block_is_marked_seeded():
    # each of these comes out of a training run, so a seed must reach it; an
    # unflagged block would silently share one artifact across all five seeds
    for name in encoders.BLOCK_SPECS:
        assert features.BLOCKS[name].seeded is True, name


def test_the_seed_reaches_an_encoder_block_through_the_sweep():
    import sweep

    for name in encoders.BLOCK_SPECS:
        assert sweep._seed_params(name, 3) == {"seed": 3}, name
    # the seed-independent blocks must not gain a seed they would ignore
    assert sweep._seed_params("mordred", 3) == {}


def test_different_seeds_are_different_artifacts(tmp_path):
    name = next(iter(encoders.BLOCK_SPECS))
    molecules = ["CCO", "CCC"]
    spec = features.BLOCKS[name]
    keys = set()
    for seed in (0, 1):
        merged = {**spec.defaults, "seed": seed}
        keys.add(
            provenance.spec_key(
                provenance.block_spec(
                    name,
                    spec.version,
                    params=merged,
                    molecules=provenance.spec_key({"smiles": molecules}),
                    n_molecules=len(molecules),
                )
            )
        )
    assert len(keys) == 2


def test_the_e4_bridge_extracts_a_loadable_message_passing_body(tmp_path, monkeypatch):
    import torch
    from chemprop.nn import BondMessagePassing

    from concat_arch.backbone import load_body_weights

    # a real body, wrapped in the key prefix a whole network carries, so the
    # extraction is exercised against chemprop's own shapes rather than a mock
    defaults = encoders.BLOCK_SPECS["chemprop_log2fc_embedding"].defaults
    # chemprop types this abstract though it instantiates, as backbone.py also notes
    body = BondMessagePassing(  # pyright: ignore[reportAbstractUsage]
        d_h=defaults["message_hidden_dim"], depth=defaults["depth"]
    )
    whole = {f"message_passing.{k}": v for k, v in body.state_dict().items()}
    whole["predictor.ffn.0.weight"] = torch.zeros(3, defaults["message_hidden_dim"])

    class _Stub:
        def __init__(self):
            self.estimator = self

        def state_dict(self):
            return whole

    monkeypatch.setattr(encoders, "_ensure_encoder", lambda *a, **k: _Stub())

    path = encoders.log2fc_body_checkpoint(seed=0, cache_dir=tmp_path)
    assert path.exists()

    saved = torch.load(path, weights_only=True, map_location="cpu")
    assert saved["hyper_parameters"] == {
        "d_h": defaults["message_hidden_dim"],
        "depth": defaults["depth"],
    }
    # the predictor head must not travel with the body
    assert not any(k.startswith("predictor") for k in saved["state_dict"])

    rebuilt = load_body_weights(path)
    assert rebuilt is not None


def test_the_e4_bridge_is_cached(tmp_path, monkeypatch):
    calls = []

    class _Stub:
        def __init__(self):
            self.estimator = self

        def state_dict(self):
            calls.append(1)
            return {"message_passing.W_i.weight": __import__("torch").zeros(4, 4)}

    monkeypatch.setattr(encoders, "_ensure_encoder", lambda *a, **k: _Stub())
    first = encoders.log2fc_body_checkpoint(seed=0, cache_dir=tmp_path)
    second = encoders.log2fc_body_checkpoint(seed=0, cache_dir=tmp_path)
    assert first == second
    # the second call must not reach the encoder at all
    assert len(calls) == 1


def test_the_vendored_architecture_finds_the_bridge():
    from concat_arch.run import ENCODERS_FACTORY, ENCODERS_MODULE

    module = __import__(ENCODERS_MODULE)
    # the E4 arm resolves this by name at call time, so the contract is the name
    assert callable(getattr(module, ENCODERS_FACTORY, None))


def test_encoders_refit_on_everything_by_default():
    # the validation rows exist to choose a stopping point; once chosen there is
    # no reason to leave them out of the encoder that gets used
    assert encoders.EncoderConfig(seed=0).refit_on_all is True
    assert encoders.EncoderConfig(seed=0, refit_on_all=False).refit_on_all is False


def test_the_refit_switch_changes_the_cache_key(log2fc_set):
    # a refitted encoder is a different encoder, so it must not be served from
    # the single-pass entry's cache
    base = encoders.EncoderConfig(seed=0, from_foundation=None)
    single = encoders.EncoderConfig(seed=0, from_foundation=None, refit_on_all=False)
    assert (
        encoders.encoder_artifact("log2fc", base, log2fc_set).key
        != encoders.encoder_artifact("log2fc", single, log2fc_set).key
    )


def test_the_encoder_version_moved_with_the_semantics():
    # bumping this is what stops artifacts trained under the old single-pass
    # behaviour from being served for the new one
    assert encoders.VERSION >= 2


def test_both_passes_use_the_same_scheduler():
    # a plateau schedule needs a validation loss and the refit pass has none;
    # running plateau then noam would choose an epoch count under one schedule
    # and spend it under another
    source = (REPO_ROOT / "src" / "encoders.py").read_text()

    assert 'scheduler="noam"' in source
    assert 'scheduler="plateau"' not in source
    assert "monitor_metric" not in source


def test_the_refit_keeps_the_first_pass_epoch_budget_for_the_schedule():
    # noam calibrates its decay against the trainer's max_epochs, so shortening
    # that to the chosen count would compress the whole schedule and train the
    # second pass under learning rates the first never saw
    source = (REPO_ROOT / "src" / "encoders.py").read_text()
    refit = source[source.index("def _refit_on_all") : source.index("def _ensure_encoder")]

    assert "max_epochs=config.max_epochs" in refit
    assert "_StopAfter(epochs)" in refit


def test_the_stop_callback_ends_training_at_the_chosen_epoch():
    stop = encoders._StopAfter(3)

    class _Trainer:
        def __init__(self, epoch):
            self.current_epoch = epoch
            self.should_stop = False

    early, last = _Trainer(1), _Trainer(2)
    # a stand-in for the trainer, which the callback only reads two fields of
    stop.on_train_epoch_end(cast("Any", early), None)
    stop.on_train_epoch_end(cast("Any", last), None)

    assert early.should_stop is False
    # epochs are zero-indexed, so finishing epoch 2 is the third epoch
    assert last.should_stop is True
