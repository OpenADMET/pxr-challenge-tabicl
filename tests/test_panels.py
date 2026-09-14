"""What a figure needs from a checkout that holds no feature caches."""

import json

import pandas as pd
import pytest

import manifest as manifest_module
import panels

RDKIT = ("descriptors", "rdkit")


@pytest.fixture(scope="module")
def spec():
    return manifest_module.load()


def _write_block(features_dir, block, n_columns):
    """Write a one-row feature block holding a key column and ``n_columns`` features."""
    directory = features_dir / block
    directory.mkdir(parents=True)
    columns = {"canonical_smiles": ["CCO"], **{f"f{i}": [0.0] for i in range(n_columns)}}
    pd.DataFrame(columns).to_parquet(directory / "part.parquet", index=False)


def test_a_block_without_features_takes_its_recorded_width(spec, tmp_path):
    record = tmp_path / "block_dims.json"
    record.write_text(json.dumps({"descriptors": {"rdkit": 217}}))

    dims = panels.block_dims(spec, features_dir=tmp_path / "features", record=record)

    assert dims[RDKIT] == 217


def test_a_built_block_is_measured_rather_than_recorded(spec, tmp_path):
    features = tmp_path / "features"
    _write_block(features, "rdkit", 3)
    record = tmp_path / "block_dims.json"
    record.write_text(json.dumps({"descriptors": {"rdkit": 217}}))

    dims = panels.block_dims(spec, features_dir=features, record=record)

    assert dims[RDKIT] == 3


def test_a_block_neither_built_nor_recorded_is_left_out(spec, tmp_path):
    missing = tmp_path / "block_dims.json"

    dims = panels.block_dims(spec, features_dir=tmp_path / "features", record=missing)

    assert RDKIT not in dims


def test_recording_keeps_widths_it_did_not_measure(tmp_path):
    record = tmp_path / "block_dims.json"
    record.write_text(json.dumps({"descriptors": {"mordred": 1613}}))

    panels.record_block_dims({RDKIT: 217}, record=record)

    assert json.loads(record.read_text()) == {"descriptors": {"mordred": 1613, "rdkit": 217}}
