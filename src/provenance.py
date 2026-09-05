"""Content-addressed cache keys and the provenance records that travel with them.

Every artifact in the pipeline, a raw feature block, a reduced feature block, a
fitted model's predictions, is named by a hash of the specification that
produces it. The specification includes the keys of the artifacts it consumes,
so a change anywhere upstream renames everything downstream: a stale artifact
cannot be mistaken for a fresh one, because it is a different file.

Beside each artifact sits a JSON record holding the specification in full,
along with the environment that produced it. Nothing has to be reconstructed
from a launch script or inferred from a directory name, and there is one
account of how an artifact came to exist rather than several to reconcile.

Every artifact is written through :func:`atomic`, which builds it under a
temporary name in the same directory and moves it into place in one step. A
cache entry therefore appears whole or not at all, so two processes sweeping
different stages at once cannot leave each other a half-written block that
still looks cached. Two processes that miss the same key both compute it and
one overwrites the other, which is wasted work rather than a corrupt file:
the key is a hash of the specification, so their output is the same.

Producer semantics are versioned by hand. A ``version`` field in a
specification is what separates "the same Mordred block" from "the Mordred
block after we changed how it handles a parse failure", since hashing the
producing code itself would invalidate a 118 MB cache on a comment change.
Bump it when the meaning of an artifact changes, never for a change that
cannot alter output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# characters of the hex digest used to name an artifact; 12 leaves a collision
# vanishingly unlikely across the few thousand artifacts a full sweep produces
KEY_LENGTH = 12


@contextmanager
def atomic(path: Path) -> Iterator[Path]:
    """Yield a temporary path that is moved onto ``path`` when the block exits.

    The temporary sits in the destination's own directory, so the move is a
    rename within one filesystem and cannot be observed half-done. A block that
    raises leaves the destination untouched and removes the temporary.

    Parameters
    ----------
    path : path-like
        Where the finished artifact belongs. Parent directories are created.

    Yields
    ------
    Path
        The path to write to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # the pid keeps two processes racing on one key from sharing a temporary
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        yield partial
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


# packages whose versions can change a numeric result, recorded with every artifact
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "rdkit",
    "mordredcommunity",
    "torch",
    "chemprop",
    "tabpfn",
    "tabicl",
    "tabfm",
    "lightgbm",
    "xgboost",
    "openadmet-models",
)

# checkouts whose commit is recorded alongside the package version, because an
# editable install's reported version does not track the working tree
TRACKED_CHECKOUTS = {"openadmet-models": Path("../openadmet-models")}


class CacheError(RuntimeError):
    """An artifact's record is missing, unreadable, or disagrees with its key."""


def canonical_json(spec: Mapping[str, Any]) -> str:
    """Render a specification as the canonical JSON its key is taken from.

    Parameters
    ----------
    spec : mapping
        The specification. Values may be JSON scalars, lists, tuples, or
        nested mappings; a set is rejected because its order is not defined.

    Returns
    -------
    str
        Compact JSON with sorted keys.

    Raises
    ------
    TypeError
        If the specification holds a value with no canonical rendering.
    """
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=_encode)


def spec_key(spec: Mapping[str, Any]) -> str:
    """Return the cache key of a specification."""
    digest = hashlib.sha256(canonical_json(spec).encode()).hexdigest()
    return digest[:KEY_LENGTH]


def digest_file(path: Path) -> str:
    """Return the content digest of a file, in the same form as a cache key."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:KEY_LENGTH]


def environment() -> dict[str, Any]:
    """Return the versions and commits a result depends on.

    Returns
    -------
    dict
        Package versions, tracked checkout commits, and this repository's
        commit and working-tree cleanliness.
    """
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    checkouts = {name: _git_commit(path) for name, path in TRACKED_CHECKOUTS.items()}
    return {
        "packages": packages,
        "checkouts": checkouts,
        "repo": {"commit": _git_commit(Path()), "dirty": _git_is_dirty(Path())},
    }


@dataclass(frozen=True)
class Artifact:
    """One cached file, named by the specification that produces it.

    Attributes
    ----------
    root : Path
        Directory the artifact and its record live in.
    spec : dict
        Everything that determines the artifact's content, including the keys
        of the artifacts it consumes.
    suffix : str
        File extension of the artifact itself, record excluded.
    """

    root: Path
    spec: dict[str, Any]
    suffix: str = ".parquet"

    @property
    def key(self) -> str:
        """The artifact's cache key."""
        return spec_key(self.spec)

    @property
    def path(self) -> Path:
        """Where the artifact itself lives."""
        return self.root / f"{self.key}{self.suffix}"

    @property
    def record_path(self) -> Path:
        """Where the artifact's provenance record lives."""
        return self.root / f"{self.key}.json"

    @property
    def is_cached(self) -> bool:
        """Whether both the artifact and its record are already on disk."""
        return self.path.exists() and self.record_path.exists()

    def write_record(self, **extra: Any) -> Path:
        """Write the provenance record beside the artifact.

        Parameters
        ----------
        **extra
            Additional fields to record, such as an artifact's row and column
            counts or a model's fitted metrics.

        Returns
        -------
        Path
            The record's path.
        """
        record = {
            "key": self.key,
            "artifact": self.path.name,
            "spec": self.spec,
            "environment": environment(),
            "written_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            **extra,
        }
        with atomic(self.record_path) as partial:
            partial.write_text(json.dumps(record, indent=2, default=_encode) + "\n")
        return self.record_path

    def read_record(self) -> dict[str, Any]:
        """Read the provenance record, checking it belongs to this artifact.

        Returns
        -------
        dict
            The recorded specification and environment.

        Raises
        ------
        CacheError
            If the record is absent or was written for a different key.
        """
        if not self.record_path.exists():
            raise CacheError(f"{self.path.name}: no provenance record beside the artifact")
        record = json.loads(self.record_path.read_text())
        if record.get("key") != self.key:
            raise CacheError(
                f"{self.record_path.name}: record names key {record.get('key')!r}, "
                f"expected {self.key!r}"
            )
        return record


def block_spec(
    name: str,
    version_: int,
    params: Mapping[str, Any] | None = None,
    inputs: Sequence[Artifact] = (),
    **extra: Any,
) -> dict[str, Any]:
    """Assemble a specification from a producer's identity, inputs and parameters.

    Parameters
    ----------
    name : str
        What the artifact is, for example ``mordred`` or ``pca``.
    version_ : int
        The producer's semantic version, bumped when its meaning changes.
    params : mapping, optional
        Parameters that alter the artifact's content.
    inputs : sequence of Artifact, optional
        Artifacts consumed, contributing their keys so an upstream change
        renames this artifact too.
    **extra
        Further specification fields.

    Returns
    -------
    dict
        The specification, ready to hash.
    """
    return {
        "name": name,
        "version": version_,
        "params": dict(params or {}),
        "inputs": [artifact.key for artifact in inputs],
        **extra,
    }


def _encode(value: Any) -> Any:
    """Render a value JSON can hold, rejecting anything without a stable order."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set | frozenset):
        raise TypeError("a set has no defined order and cannot appear in a cache key")
    # numpy scalars carry .item(); anything else is a programming error here
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"no canonical rendering for {type(value).__name__}")


def _git_commit(path: Path) -> str | None:
    """Return the commit a checkout is on, or None if it is not a repository."""
    return _git(path, "rev-parse", "HEAD")


def _git_is_dirty(path: Path) -> bool | None:
    """Return whether a checkout has uncommitted changes, or None if unknown."""
    status = _git(path, "status", "--porcelain")
    return None if status is None else bool(status)


def _git(path: Path, *args: str) -> str | None:
    """Run a git command in a checkout, returning None when it is unavailable."""
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()
