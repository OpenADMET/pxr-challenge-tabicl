import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import features
import provenance
import reduce as reduction
from data import CANONICAL_COL, SPLIT_DIR

FIT = [f"FIT{i}" for i in range(60)]
HELD = [f"HELD{i}" for i in range(20)]


def _block(tmp_path, *, with_missing=False):
    """Write a synthetic raw block whose held-out rows differ sharply from the fit rows."""
    rng = np.random.default_rng(0)
    fit_values = rng.normal(0.0, 1.0, size=(len(FIT), 6))
    # held-out rows are shifted and scaled, so a PCA that saw them would differ
    held_values = rng.normal(50.0, 20.0, size=(len(HELD), 6))
    values = np.vstack([fit_values, held_values])
    if with_missing:
        values[0, 0] = np.nan
        values[len(FIT), 1] = np.inf
    frame = pd.DataFrame(
        values,
        index=pd.Index([*FIT, *HELD], name=CANONICAL_COL),
        columns=[f"c{i}" for i in range(6)],
    )
    artifact = provenance.Artifact(tmp_path / "synthetic", provenance.block_spec("synthetic", 1))
    artifact.root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(artifact.path)
    artifact.write_record()
    return artifact, frame


def test_pca_is_fitted_on_the_fit_rows_alone(tmp_path):
    block, frame = _block(tmp_path)
    artifact = reduction.build(block, width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")
    produced = reduction.load(artifact).to_numpy()

    # fit the same estimators by hand on the fit rows only, then apply to all
    values = frame.to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True).fit(values[: len(FIT)])
    filled = imputer.transform(values)
    pca = PCA(n_components=3, random_state=0).fit(filled[: len(FIT)])
    expected = pca.transform(filled)

    np.testing.assert_allclose(produced, expected, rtol=1e-9, atol=1e-9)


def test_fitting_on_everything_would_give_different_numbers(tmp_path):
    block, _ = _block(tmp_path)
    honest = reduction.build(block, width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")
    leaked = reduction.build(
        block, width=3, fit_smiles=[*FIT, *HELD], cache_dir=tmp_path / "reduced"
    )
    # the guard is only meaningful if the two differ, both in identity and in value
    assert honest.key != leaked.key
    assert not np.allclose(reduction.load(honest).to_numpy(), reduction.load(leaked).to_numpy())


def test_the_fit_partition_is_part_of_the_cache_key(tmp_path):
    block, _ = _block(tmp_path)
    a = reduction.build(block, width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")
    b = reduction.build(block, width=3, fit_smiles=FIT[:40], cache_dir=tmp_path / "reduced")
    assert a.key != b.key
    assert a.read_record()["spec"]["n_fit_molecules"] == len(FIT)


def test_missing_and_infinite_values_are_imputed_from_fit_rows(tmp_path):
    block, _ = _block(tmp_path, with_missing=True)
    artifact = reduction.build(block, width=None, fit_smiles=FIT, cache_dir=tmp_path / "reduced")
    produced = reduction.load(artifact).to_numpy()
    assert np.isfinite(produced).all()
    assert artifact.read_record()["n_missing_imputed"] == 2


def test_passthrough_keeps_every_column(tmp_path):
    block, frame = _block(tmp_path)
    artifact = reduction.build(block, width=None, fit_smiles=FIT, cache_dir=tmp_path / "reduced")
    assert reduction.load(artifact).shape == frame.shape
    assert artifact.read_record()["explained_variance_ratio"] is None


def test_a_fit_partition_the_block_cannot_cover_is_refused(tmp_path):
    block, _ = _block(tmp_path)
    with pytest.raises(reduction.LeakageError, match="missing"):
        reduction.build(block, width=3, fit_smiles=[*FIT, "ABSENT"], cache_dir=tmp_path / "reduced")


def test_reduction_is_cached(tmp_path):
    block, _ = _block(tmp_path)
    first = reduction.build(block, width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")
    mtime = first.path.stat().st_mtime_ns
    second = reduction.build(block, width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")
    assert second.path.stat().st_mtime_ns == mtime


def test_fit_molecules_are_the_fit_partition_and_exclude_phase_two():
    fit = set(reduction.fit_molecules())
    test = set(
        pd.read_csv(SPLIT_DIR / "test_phase2.csv", usecols=[CANONICAL_COL], dtype=str)[
            CANONICAL_COL
        ]
    )
    assert len(fit) == 4392
    assert len(test) == 260
    # the rows every reduction is fitted on must never include a scored compound
    assert fit.isdisjoint(test)
    assert fit | test == set(features.unique_molecules())
