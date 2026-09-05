"""Stage one: compute each raw feature block once, keyed by canonical SMILES.

A block is a rectangular table of raw per-molecule features, one row per unique
compound across the fit and test partitions, indexed by the canonical SMILES the
split was built on. Computing every block over every compound at once, rather
than per partition or per run, buys three things.

It is the only way Mordred's column set stays fixed. Mordred emits the union of
descriptors that compute on the rows it is given, so featurizing partitions
separately can hand a downstream per-block reduction two different column
layouts for the same block. One table over all compounds has one layout by
construction.

It removes recomputation. Mordred over roughly 4,650 compounds is minutes of
work that would otherwise repeat for every featureset and regressor that wants
descriptors.

It separates what is expensive from what is seeded. RDKit, Mordred and the
frozen CheMeleon embedding do not depend on a seed and are computed once;
blocks that come from a trained encoder do, and are computed per seed.

Nothing here fits anything on the data. Every block is a per-molecule function
of structure alone, so a single table spanning fit and test carries no
information between them. Reductions that do fit, PCA and imputation, live in
stage two and see the fit partition only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import provenance
from data import CANONICAL_COL, SPLIT_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/features")

# partitions whose compounds a block must cover; the fit set and the test set
# together are every molecule the pipeline ever featurizes
PARTITION_FILES = ("fit_all.csv", "test_phase2.csv")

# molecules per forward pass for the neural featurizers
EMBEDDING_BATCH_SIZE = 256


class FeatureError(RuntimeError):
    """A block could not be computed for every molecule it was asked about."""


def unique_molecules(split_dir: Path = SPLIT_DIR) -> list[str]:
    """Return every canonical SMILES the pipeline features, sorted.

    Parameters
    ----------
    split_dir : path-like
        Directory holding the split resource CSVs.

    Returns
    -------
    list of str
        Sorted unique canonical SMILES across the fit and test partitions.
    """
    seen: set[str] = set()
    for name in PARTITION_FILES:
        frame = pd.read_csv(split_dir / name, usecols=[CANONICAL_COL], dtype=str)
        seen.update(frame[CANONICAL_COL])
    return sorted(seen)


def build(
    name: str,
    molecules: Sequence[str] | None = None,
    *,
    params: dict | None = None,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> provenance.Artifact:
    """Compute a feature block, or return the cached one.

    Parameters
    ----------
    name : str
        Block name, one of the keys of ``BLOCKS``.
    molecules : sequence of str, optional
        Canonical SMILES to featurize. Defaults to every molecule in the split.
    params : dict, optional
        Parameters for the block, merged over its defaults.
    cache_dir : path-like, optional
        Root of the feature cache.
    force : bool, optional
        Recompute even when a cached artifact exists.

    Returns
    -------
    Artifact
        The cached block, whose ``path`` holds a parquet indexed by canonical
        SMILES.

    Raises
    ------
    KeyError
        If the block name is not known.
    FeatureError
        If the block came back missing molecules or misaligned.
    """
    if name not in BLOCKS:
        raise KeyError(f"unknown feature block {name!r}; known: {sorted(BLOCKS)}")
    block = BLOCKS[name]

    molecules = list(molecules) if molecules is not None else unique_molecules()
    resolved = {**block.defaults, **(params or {})}

    # the molecule set is part of the artifact's identity: adding a compound
    # changes the block, so it must change the key
    spec = provenance.block_spec(
        name,
        block.version,
        params=resolved,
        molecules=provenance.spec_key({"smiles": molecules}),
        n_molecules=len(molecules),
    )
    artifact = provenance.Artifact(cache_dir / name, spec)

    if artifact.is_cached and not force:
        logger.info("%s: cached (%s)", name, artifact.key)
        return artifact

    logger.info("%s: computing over %d molecules", name, len(molecules))
    with provenance.timed() as elapsed:
        frame = block.compute(molecules, **resolved)
        _check_alignment(name, frame, molecules)

    with provenance.atomic(artifact.path) as partial:
        frame.to_parquet(partial)
    artifact.write_record(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        wall_clock_s=elapsed(),
    )
    logger.info("%s: wrote %d x %d -> %s", name, *frame.shape, artifact.path)
    return artifact


def load(artifact: provenance.Artifact) -> pd.DataFrame:
    """Read a cached block, indexed by canonical SMILES."""
    return pd.read_parquet(artifact.path)


def _check_alignment(name: str, frame: pd.DataFrame, molecules: Sequence[str]) -> None:
    """Fail loudly if a block does not cover exactly the molecules asked for."""
    if list(frame.index) != list(molecules):
        missing = set(molecules) - set(frame.index)
        extra = set(frame.index) - set(molecules)
        raise FeatureError(
            f"{name}: block does not match the molecules requested "
            f"({len(missing)} missing, {len(extra)} unexpected)"
        )
    if frame.columns.duplicated().any():
        raise FeatureError(f"{name}: block has duplicate column names")


def _compute_rdkit(molecules: Sequence[str]) -> pd.DataFrame:
    """Compute RDKit's 2D descriptor set for each molecule."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    rows = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise FeatureError(f"rdkit: {smiles!r} did not parse, though the split accepted it")
        rows.append(Descriptors.CalcMolDescriptors(mol))
    return pd.DataFrame(rows, index=pd.Index(molecules, name=CANONICAL_COL), dtype=float)


def _compute_mordred(molecules: Sequence[str], *, n_jobs: int = 1) -> pd.DataFrame:
    """Compute Mordred's 2D descriptor set for each molecule.

    Descriptors that fail on a molecule come back as Mordred error objects;
    they become NaN here and are imputed in stage two, on fit rows only.
    """
    from mordred import Calculator, descriptors
    from rdkit import Chem

    mols = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise FeatureError(f"mordred: {smiles!r} did not parse, though the split accepted it")
        mols.append(mol)

    calculator = Calculator(descriptors, ignore_3D=True)
    frame = calculator.pandas(mols, nproc=n_jobs, quiet=True)
    frame = frame.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    frame.index = pd.Index(molecules, name=CANONICAL_COL)
    return frame


def _compute_chemeleon(
    molecules: Sequence[str], *, batch_size: int = EMBEDDING_BATCH_SIZE, accelerator: str = "auto"
) -> pd.DataFrame:
    """Extract frozen CheMeleon MPNN embeddings, which no seed influences."""
    from openadmet.models.features.chemeleon_embedding import CheMeleonEmbeddingFeaturizer

    featurizer = CheMeleonEmbeddingFeaturizer(accelerator=accelerator, batch_size=batch_size)
    embeddings, indices = featurizer.featurize(list(molecules))
    embeddings = np.asarray(embeddings)
    if len(indices) != len(molecules):
        raise FeatureError(
            f"chemeleon: featurizer returned {len(indices)} of {len(molecules)} molecules"
        )
    columns = [f"chemeleon_{i:04d}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(
        embeddings, index=pd.Index(molecules, name=CANONICAL_COL), columns=columns, dtype=float
    )


class _Block:
    """A named feature block: what computes it, and what changes its meaning."""

    def __init__(
        self,
        version: int,
        compute: Callable[..., pd.DataFrame],
        defaults: dict,
        *,
        seeded: bool = False,
    ):
        self.version = version
        self.compute = compute
        self.defaults = defaults
        # a seeded block is computed once per replicate seed, because a trained
        # encoder produces different features each time it is trained
        self.seeded = seeded


# Bump a block's version when the meaning of its output changes; never for a
# change that cannot alter the numbers.
BLOCKS: dict[str, _Block] = {
    "rdkit": _Block(1, _compute_rdkit, {}),
    "mordred": _Block(1, _compute_mordred, {"n_jobs": 1}),
    "chemeleon": _Block(
        1, _compute_chemeleon, {"batch_size": EMBEDDING_BATCH_SIZE, "accelerator": "auto"}
    ),
}


# the seeded blocks live in encoders, registered here once BLOCKS exists; the
# import is late because encoders imports this module for the block contract
import encoders  # noqa: E402

encoders.register()
