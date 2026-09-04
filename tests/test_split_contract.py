"""Contract tests for the challenge split: canonicalization, leakage, sizes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import (
    CANONICAL_COL,
    PHASE1_FILE,
    PHASE2_FILE,
    SPLIT_DIR,
    TARGET_COL,
    TRAIN_FILE,
    _assert_disjoint,
    _canonical_smiles,
    build_split,
)

# the challenge's official split sizes: TRAIN (4139) + phase 1 (253) fit, phase 2 test
EXPECTED_FIT_ROWS = 4392
EXPECTED_TEST_ROWS = 260


def test_canonical_smiles_strips_salt_to_parent():
    assert _canonical_smiles("CCO.Cl") == "CCO"


def test_canonical_smiles_is_input_order_independent():
    assert _canonical_smiles("OCC") == _canonical_smiles("CCO")


def test_canonical_smiles_returns_none_for_unparsable():
    assert _canonical_smiles("not_a_molecule_zz") is None


def test_assert_disjoint_raises_on_shared_compound():
    fit = pd.DataFrame({CANONICAL_COL: ["CCO", "CCC"]})
    test = pd.DataFrame({CANONICAL_COL: ["CCC", "CCCl"]})

    with pytest.raises(ValueError, match="leakage"):
        _assert_disjoint(fit, test)


def test_assert_disjoint_passes_on_disjoint_compounds():
    fit = pd.DataFrame({CANONICAL_COL: ["CCO", "CCC"]})
    test = pd.DataFrame({CANONICAL_COL: ["CCCl", "CCBr"]})

    _assert_disjoint(fit, test)


def _write_raw(raw_dir: Path, train: list[str], phase1: list[str], phase2: list[str]) -> None:
    """Write minimal train/phase1/phase2 CSVs into a synthetic raw directory."""
    for name, smiles in ((TRAIN_FILE, train), (PHASE1_FILE, phase1), (PHASE2_FILE, phase2)):
        frame = pd.DataFrame({"SMILES": smiles, TARGET_COL: [5.5] * len(smiles)})
        frame.to_csv(raw_dir / name, index=False)


def test_build_split_pools_train_and_phase1_and_partitions_fit(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "splits"
    _write_raw(
        raw,
        train=["CCO", "CCC", "CCCC", "CCCCC", "CCCCCC", "c1ccccc1"],
        phase1=["CCN", "CCCN"],
        phase2=["CCCl", "CCBr", "CCF"],
    )

    art = build_split(raw, out, val_fraction=0.5, seed=42)

    assert art.n_fit == 8
    assert art.n_test == 3
    assert art.n_fit_train == 4
    assert art.n_fit_val == 4
    assert art.n_fit_train + art.n_fit_val == art.n_fit


def test_build_split_writes_disjoint_fit_and_test(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "splits"
    _write_raw(raw, train=["CCO", "CCC"], phase1=["CCN"], phase2=["CCCl", "CCBr"])

    art = build_split(raw, out)
    fit = pd.read_csv(art.fit_all)
    test = pd.read_csv(art.test)

    assert set(fit[CANONICAL_COL]) & set(test[CANONICAL_COL]) == set()


def test_build_split_raises_when_test_compound_is_in_fit(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "splits"
    # "CCCl" appears in both the training file and phase 2
    _write_raw(raw, train=["CCO", "CCCl"], phase1=["CCN"], phase2=["CCCl", "CCBr"])

    with pytest.raises(ValueError, match="leakage"):
        build_split(raw, out)


def test_build_split_raises_on_implausible_pec50(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "splits"
    pd.DataFrame({"SMILES": ["CCO"], TARGET_COL: [99.0]}).to_csv(raw / TRAIN_FILE, index=False)
    pd.DataFrame({"SMILES": ["CCN"], TARGET_COL: [5.5]}).to_csv(raw / PHASE1_FILE, index=False)
    pd.DataFrame({"SMILES": ["CCCl"], TARGET_COL: [5.5]}).to_csv(raw / PHASE2_FILE, index=False)

    with pytest.raises(ValueError, match="outside"):
        build_split(raw, out)


def test_committed_split_matches_challenge_contract():
    fit_path = SPLIT_DIR / "fit_all.csv"
    test_path = SPLIT_DIR / "test_phase2.csv"
    if not fit_path.exists() or not test_path.exists():
        pytest.skip("split resources not built; run run/01_build_split.py first")

    fit = pd.read_csv(fit_path)
    test = pd.read_csv(test_path)

    assert len(fit) == EXPECTED_FIT_ROWS
    assert len(test) == EXPECTED_TEST_ROWS
    assert set(fit[CANONICAL_COL]) & set(test[CANONICAL_COL]) == set()
