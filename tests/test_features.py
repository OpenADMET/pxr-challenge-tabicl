import pandas as pd
import pytest

import features
from data import CANONICAL_COL


def test_split_molecules_are_every_compound_once():
    molecules = features.unique_molecules()
    assert len(molecules) == len(set(molecules))
    assert molecules == sorted(molecules)
    # the fit and test partitions together, deduplicated by canonical structure
    assert len(molecules) == 4652


def test_rdkit_block_covers_exactly_the_molecules_asked_for(tmp_path):
    molecules = features.unique_molecules()[:12]
    artifact = features.build("rdkit", molecules, cache_dir=tmp_path)
    frame = features.load(artifact)
    assert list(frame.index) == molecules
    assert frame.index.name == CANONICAL_COL
    assert frame.shape[1] > 100


def test_second_build_is_served_from_cache(tmp_path):
    molecules = features.unique_molecules()[:8]
    first = features.build("rdkit", molecules, cache_dir=tmp_path)
    mtime = first.path.stat().st_mtime_ns
    second = features.build("rdkit", molecules, cache_dir=tmp_path)
    assert second.key == first.key
    assert second.path.stat().st_mtime_ns == mtime


def test_changing_the_molecule_set_changes_the_key(tmp_path):
    molecules = features.unique_molecules()[:8]
    first = features.build("rdkit", molecules, cache_dir=tmp_path)
    second = features.build("rdkit", molecules[:7], cache_dir=tmp_path)
    assert first.key != second.key


def test_unknown_block_is_refused():
    with pytest.raises(KeyError, match="unknown feature block"):
        features.build("not_a_block", ["CCO"])


def test_a_block_missing_molecules_is_caught():
    frame = pd.DataFrame({"x": [1.0]}, index=pd.Index(["CCO"], name=CANONICAL_COL))
    with pytest.raises(features.FeatureError, match="does not match"):
        features._check_alignment("stub", frame, ["CCO", "CCC"])


def test_a_block_records_how_long_it_took(tmp_path):
    molecules = features.unique_molecules()[:8]
    artifact = features.build("rdkit", molecules, cache_dir=tmp_path)

    record = artifact.read_record()

    assert record["wall_clock_s"] >= 0.0
    # timing describes the production, not the artifact's identity
    assert "wall_clock_s" not in record["spec"]
