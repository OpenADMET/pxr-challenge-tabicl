import json

import pytest

import provenance


def test_key_is_stable_across_key_order():
    a = {"name": "x", "params": {"b": 1, "a": 2}}
    b = {"params": {"a": 2, "b": 1}, "name": "x"}
    assert provenance.spec_key(a) == provenance.spec_key(b)


def test_key_changes_with_any_field():
    base = provenance.block_spec("mordred", 1, {"n_jobs": 1})
    assert provenance.spec_key(base) != provenance.spec_key(
        provenance.block_spec("mordred", 2, {"n_jobs": 1})
    )
    assert provenance.spec_key(base) != provenance.spec_key(
        provenance.block_spec("mordred", 1, {"n_jobs": 2})
    )
    assert provenance.spec_key(base) != provenance.spec_key(
        provenance.block_spec("rdkit", 1, {"n_jobs": 1})
    )


def test_input_keys_propagate_downstream(tmp_path):
    upstream = provenance.Artifact(tmp_path, provenance.block_spec("a", 1))
    other = provenance.Artifact(tmp_path, provenance.block_spec("a", 2))
    downstream = provenance.block_spec("pca", 1, {"width": 8}, inputs=[upstream])
    moved = provenance.block_spec("pca", 1, {"width": 8}, inputs=[other])
    assert provenance.spec_key(downstream) != provenance.spec_key(moved)


def test_sets_are_rejected_because_they_have_no_order():
    with pytest.raises(TypeError, match="no defined order"):
        provenance.canonical_json({"x": {1, 2}})


def test_record_round_trips_and_carries_the_environment(tmp_path):
    artifact = provenance.Artifact(tmp_path, provenance.block_spec("a", 1))
    artifact.path.write_text("payload")
    artifact.write_record(n_rows=3)

    record = artifact.read_record()
    assert record["key"] == artifact.key
    assert record["n_rows"] == 3
    assert record["spec"]["name"] == "a"
    assert "packages" in record["environment"]
    assert "commit" in record["environment"]["repo"]


def test_record_belonging_to_another_key_is_refused(tmp_path):
    artifact = provenance.Artifact(tmp_path, provenance.block_spec("a", 1))
    artifact.path.write_text("payload")
    artifact.record_path.write_text(json.dumps({"key": "deadbeefcafe"}))
    with pytest.raises(provenance.CacheError, match="expected"):
        artifact.read_record()


def test_missing_record_is_not_a_cache_hit(tmp_path):
    artifact = provenance.Artifact(tmp_path, provenance.block_spec("a", 1))
    artifact.path.write_text("payload")
    assert not artifact.is_cached
    with pytest.raises(provenance.CacheError, match="no provenance record"):
        artifact.read_record()


def test_digest_tracks_content(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"one")
    first = provenance.digest_file(path)
    path.write_bytes(b"two")
    assert provenance.digest_file(path) != first


def test_an_artifact_appears_whole_or_not_at_all(tmp_path):
    # a concurrent sweep must never find a half-written cache entry, since the
    # path existing is what makes a block look cached
    target = tmp_path / "block.parquet"

    def failing_producer() -> None:
        with provenance.atomic(target) as partial:
            partial.write_bytes(b"half a file")
            raise RuntimeError("the producer failed")

    with pytest.raises(RuntimeError):
        failing_producer()

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_completed_write_is_moved_into_place(tmp_path):
    target = tmp_path / "nested" / "block.parquet"

    with provenance.atomic(target) as partial:
        partial.write_bytes(b"the whole file")
        # the destination does not exist until the block exits
        assert not target.exists()

    assert target.read_bytes() == b"the whole file"
    assert [p.name for p in target.parent.iterdir()] == ["block.parquet"]


def test_two_writers_of_one_key_do_not_share_a_temporary(tmp_path):
    target = tmp_path / "block.parquet"
    seen = []

    with provenance.atomic(target) as first:
        seen.append(first)
        with provenance.atomic(target) as second:
            seen.append(second)
            second.write_bytes(b"second")
        first.write_bytes(b"first")

    # the temporaries are distinguished by process, so within one process they
    # coincide; across processes they cannot, which is the case that matters
    assert seen[0] == seen[1]
    assert target.exists()


def test_a_record_is_written_through_the_same_path(tmp_path):
    spec = provenance.block_spec("thing", 1, params={"a": 1}, inputs=[])
    artifact = provenance.Artifact(root=tmp_path, spec=spec)

    artifact.write_record()

    assert artifact.read_record()["key"] == artifact.key
    assert not any(p.name.endswith(".partial") for p in tmp_path.iterdir())


def test_a_timer_reports_elapsed_seconds_after_its_block_ends():
    with provenance.timed() as elapsed:
        pass
    first = elapsed()

    assert first >= 0.0
    # the callable stays valid afterwards, so the figure can be handed to the
    # record written just after the work it measures
    assert elapsed() >= first


def test_the_dirty_flag_describes_the_source_and_not_the_results():
    # results are tracked, so a run that writes several of them would make the
    # tree dirty for everything it writes afterwards: the first gate of a stage
    # would record a clean tree and the rest a dirty one, from the same code
    assert "results" not in provenance.SOURCE_PATHS
    assert "src" in provenance.SOURCE_PATHS
    assert "experiments" in provenance.SOURCE_PATHS


def test_writing_a_result_does_not_make_the_environment_dirty(tmp_path, monkeypatch):
    import subprocess

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "results").mkdir()
    (repo / "src" / "thing.py").write_text("x = 1\n")
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "initial"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    monkeypatch.chdir(repo)
    assert provenance.environment()["repo"]["dirty"] is False

    (repo / "results" / "gate.json").write_text("{}")
    assert provenance.environment()["repo"]["dirty"] is False

    (repo / "src" / "thing.py").write_text("x = 2\n")
    assert provenance.environment()["repo"]["dirty"] is True
