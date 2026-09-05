import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import encoders
import features
import manifest
import provenance
import reduce as reduction
import sweep
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
    filled = np.asarray(imputer.transform(values), dtype=np.float64)
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


def test_several_blocks_are_reduced_together(tmp_path):
    first, frame = _block(tmp_path)
    second = provenance.Artifact(tmp_path / "other", provenance.block_spec("other", 1))
    second.root.mkdir(parents=True, exist_ok=True)
    frame.add_prefix("x").to_parquet(second.path)
    second.write_record()

    artifact = reduction.build(
        [first, second], width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced"
    )
    # one projection over the joined columns, not two projections glued together
    assert reduction.load(artifact).shape == (len(FIT) + len(HELD), 3)
    assert artifact.read_record()["spec"]["block"] == "synthetic+other"
    assert len(artifact.read_record()["spec"]["inputs"]) == 2


def test_joining_blocks_that_cover_different_molecules_is_refused(tmp_path):
    first, frame = _block(tmp_path)
    second = provenance.Artifact(tmp_path / "short", provenance.block_spec("short", 1))
    second.root.mkdir(parents=True, exist_ok=True)
    frame.iloc[:10].to_parquet(second.path)
    second.write_record()

    with pytest.raises(reduction.LeakageError, match="different molecules"):
        reduction.build([first, second], width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")


def test_a_reduction_needs_at_least_one_block(tmp_path):
    with pytest.raises(ValueError, match="at least one block"):
        reduction.build([], width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")


# the precompute script's plan, which has to ask for exactly what the sweep
# will later ask for; a mismatch here quietly builds the wrong cache
_PRECOMPUTE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run" / "03_reduce.py"))
planned_reductions = _PRECOMPUTE["planned_reductions"]


@pytest.fixture(scope="module")
def spec():
    encoders.register()
    return manifest.load()


def test_a_reducible_block_is_planned_at_every_width_its_axis_declares(spec):
    planned = planned_reductions(spec)
    widths = {width for names, width in planned if names == ["mordred"]}

    # Mordred is too wide to run unreduced, so it gets no unrotated pass
    assert widths == {32, 64, 128, 256}


def test_a_block_narrow_enough_to_keep_whole_is_planned_unrotated(spec):
    planned = planned_reductions(spec)
    widths = {width for names, width in planned if names == ["rdkit"]}

    # "native" is planned as an unrotated pass, which the reduction spells None
    assert None in widths
    assert widths == {None, 32, 64, 128}


def test_an_embedding_takes_its_widths_from_the_embedding_axis(spec):
    planned = planned_reductions(spec)
    widths = {width for names, width in planned if names == ["chemeleon"]}

    # the embedding axis runs wider than the descriptor one, and carries no
    # unrotated pass: 2,048 columns do not fit the tabular models here
    assert widths == {32, 64, 128, 256, 384, 512}
    assert None not in widths


def test_a_block_that_is_passed_through_is_planned_once_and_unrotated(spec):
    planned = planned_reductions(spec)
    readout = [(names, width) for names, width in planned if names == ["chemprop_log2fc_readout"]]

    # the two predicted log2FC columns mean something column by column, so a
    # rotation of them would not be the same feature
    assert readout == [(["chemprop_log2fc_readout"], None)]


def test_the_plan_matches_what_the_sweep_asks_for(spec):
    # the sweep resolves a width per configuration; every one of those has to
    # be in the precomputed plan, or the cache misses and the run pays twice
    planned = {(tuple(names), width) for names, width in planned_reductions(spec)}
    settled = {"descriptors": "mordred", "descriptor_pca": 128, "embedding_pca": 256}
    configs = spec.expand("descriptor_width") + spec.expand("ingredients", settled)

    for config in configs:
        for group in sweep.BLOCK_ORDER:
            level = getattr(config, group)
            if level == "none":
                continue
            axis = spec.axes[group][level]
            names = tuple(axis.get("blocks", [axis["block"]] if "block" in axis else []))
            width_axis = manifest.WIDTH_OF.get(group)
            width = getattr(config, width_axis) if width_axis and axis.get("reduce") else None
            if width in (manifest.NATIVE, manifest.NOT_REDUCED):
                width = None
            assert (names, width) in planned


def test_restricting_to_raw_blocks_leaves_out_the_encoder_reductions(spec):
    planned = planned_reductions(spec, blocks=["rdkit", "mordred", "chemeleon"])
    named = {name for names, _ in planned for name in names}

    # the two width probes run before any encoder is trained, so preparing them
    # must not drag encoder blocks in
    assert named == {"rdkit", "mordred", "chemeleon"}
    assert planned


def test_a_widening_reduction_is_refused_with_its_own_message(tmp_path):
    # the synthetic block has six columns, so six components is not a reduction
    block, _ = _block(tmp_path)

    with pytest.raises(reduction.ReductionError, match="not a reduction"):
        reduction.build(block, width=6, fit_smiles=FIT, cache_dir=tmp_path / "reduced", seed=0)


@pytest.mark.parametrize(
    ("axis", "level"),
    [
        ("descriptors", "rdkit"),
        ("descriptors", "mordred"),
        ("descriptors", "rdkit_mordred"),
        ("embedding", "chemeleon"),
    ],
)
def test_the_declared_column_count_matches_the_block_it_names(spec, axis, level, tmp_path):
    # the manifest declares each reducible block's own width so that a width
    # which cannot reduce it is never generated. Declaring it twice invites
    # drift, so the declaration is checked against the block itself
    declared = manifest.native_width(spec.axes, axis, level)
    names = spec.axes[axis][level].get("blocks") or [spec.axes[axis][level]["block"]]
    if not all((features.CACHE_DIR / name).exists() for name in names):
        pytest.skip("block not built; run run/02_featurize.py")

    columns = sum(
        features.load(features.build(name, cache_dir=features.CACHE_DIR)).shape[1] for name in names
    )

    assert declared == columns


def test_no_planned_reduction_would_widen_its_block(spec):
    for names, width in planned_reductions(spec):
        if width is None:
            continue
        declared = [
            manifest.native_width(spec.axes, axis, level)
            for axis in ("descriptors", "embedding")
            for level, entry in spec.axes[axis].items()
            if entry and (entry.get("blocks") or [entry.get("block")]) == names
        ]
        for columns in declared:
            assert columns is None or width < columns


def test_a_reduction_records_how_long_it_took(tmp_path):
    block, _ = _block(tmp_path)
    artifact = reduction.build(block, width=3, fit_smiles=FIT, cache_dir=tmp_path / "reduced")

    record = artifact.read_record()

    assert record["wall_clock_s"] >= 0.0
    assert "wall_clock_s" not in record["spec"]


def test_a_gated_block_is_planned_only_at_the_width_its_gate_settled(spec):
    # the width probe sweeps every level on the frozen block, so those are all
    # built; the fine-tuned embeddings are only ever read at the settled width,
    # and planning from the axes rather than the stages would build the rest
    settled = {"descriptors": "rdkit", "descriptor_pca": 128, "embedding_pca": 256}
    planned = planned_reductions(spec, resolved=settled)

    probe = {width for names, width in planned if names == ["chemeleon"]}
    finetuned = {width for names, width in planned if names == ["chemeleon_log2fc_embedding"]}

    assert probe == {32, 64, 128, 256, 384, 512}
    assert finetuned == {256}


def test_the_plan_is_a_function_of_the_gates_it_is_given(spec):
    # not of what results/gates happens to hold, so a reader can tell what a
    # given set of decisions builds without consulting the working tree
    base = {"descriptors": "rdkit", "descriptor_pca": 128}
    at_256 = planned_reductions(spec, resolved={**base, "embedding_pca": 256})
    at_64 = planned_reductions(spec, resolved={**base, "embedding_pca": 64})

    assert (["chemeleon_pec50_embedding"], 256) in at_256
    assert (["chemeleon_pec50_embedding"], 256) not in at_64
    assert (["chemeleon_pec50_embedding"], 64) in at_64


def test_before_any_gate_only_the_probes_are_planned(spec):
    # the two width probes need no trained encoder, which is what lets them run
    # first; nothing that depends on an encoder is planned until a gate exists
    planned = planned_reductions(spec, resolved={})
    named = {name for names, _ in planned for name in names}

    assert named == {"rdkit", "mordred", "chemeleon"}


def test_every_planned_reduction_is_one_some_stage_reads(spec):
    # the contract this script has to keep: running it with no arguments builds
    # what the sweep will look for and nothing besides, so no width has to be
    # named on the command line to get a correct cache
    settled = {"descriptors": "rdkit", "descriptor_pca": 128, "embedding_pca": 256}
    planned = {(tuple(names), width) for names, width in planned_reductions(spec, resolved=settled)}

    read: set[tuple[tuple[str, ...], int | None]] = set()
    for stage in spec.stages:
        if any(
            gate not in {"canonical_descriptors", "embedding_reduction"}
            for gate in stage.depends_on
        ):
            continue
        for config in spec.expand(stage.id, settled):
            for group in sweep.BLOCK_ORDER:
                level = getattr(config, group)
                if level == "none":
                    continue
                axis = spec.axes[group][level]
                names = tuple(axis.get("blocks", [axis["block"]] if "block" in axis else []))
                width_axis = manifest.WIDTH_OF.get(group)
                width = getattr(config, width_axis) if width_axis and axis.get("reduce") else None
                if width in (manifest.NATIVE, manifest.NOT_REDUCED):
                    width = None
                read.add((names, width))

    assert planned == read
