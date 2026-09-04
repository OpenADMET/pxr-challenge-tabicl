"""Stage two: reduce a raw feature block, fitting on the fit partition alone.

This is the only stage that learns anything from the data, so it is the only
stage that can leak. Imputation means and PCA loadings are fitted on the rows
belonging to the fit partition and then applied to every molecule, test set
included. The fit partition is a required argument rather than a default, and
the rows it selects are the only rows the estimators ever see, so fitting on
everything is not something this module can be asked to do.

Reduction is cached like any other artifact, and the fit partition is part of
the cache key. Two reductions of the same block to the same width, fitted on
different partitions, are different artifacts with different names; neither can
be served from the other's cache entry.

A width of None passes the block through unreduced but still imputed, which is
what the narrow blocks want: two predicted log2FC columns mean something column
by column and have nothing to gain from a rotation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import features
import provenance
from data import CANONICAL_COL, SPLIT_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/reduced")

# bumped when the meaning of a reduction changes, never for a cosmetic edit
VERSION = 1

# the partition every estimator here is fitted on
FIT_PARTITION = "fit_all.csv"


class LeakageError(RuntimeError):
    """A reduction was asked to fit on rows outside the fit partition."""


def fit_molecules(split_dir: Path = SPLIT_DIR, partition: str = FIT_PARTITION) -> list[str]:
    """Return the canonical SMILES of the partition reductions are fitted on."""
    frame = pd.read_csv(split_dir / partition, usecols=[CANONICAL_COL], dtype=str)
    return sorted(set(frame[CANONICAL_COL]))


def build(
    block: provenance.Artifact,
    *,
    width: int | None,
    fit_smiles: Sequence[str],
    impute: str = "mean",
    seed: int = 0,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> provenance.Artifact:
    """Reduce a raw block, fitting on the fit partition and applying to all rows.

    Parameters
    ----------
    block : Artifact
        The stage-one block to reduce.
    width : int or None
        Number of principal components. None imputes but does not rotate.
    fit_smiles : sequence of str
        Canonical SMILES of the fit partition. The imputer and PCA see these
        rows and no others.
    impute : str, optional
        Imputation strategy for missing values, ``mean`` or ``median``.
    seed : int, optional
        Seed for PCA's randomized solver, which sklearn selects for wide
        matrices under a large reduction.
    cache_dir : path-like, optional
        Root of the reduction cache.
    force : bool, optional
        Recompute even when a cached artifact exists.

    Returns
    -------
    Artifact
        The reduced block, indexed by canonical SMILES.

    Raises
    ------
    LeakageError
        If the fit partition names molecules the block does not contain, which
        would otherwise silently narrow what is fitted on.
    """
    fit_smiles = sorted(set(fit_smiles))
    spec = provenance.block_spec(
        "pca" if width is not None else "impute_only",
        VERSION,
        params={"width": width, "impute": impute, "seed": seed},
        inputs=[block],
        block=block.spec["name"],
        fit_partition=provenance.spec_key({"smiles": fit_smiles}),
        n_fit_molecules=len(fit_smiles),
    )
    artifact = provenance.Artifact(cache_dir / block.spec["name"], spec)

    if artifact.is_cached and not force:
        logger.info("%s w=%s: cached (%s)", block.spec["name"], width, artifact.key)
        return artifact

    raw = features.load(block)
    fit_index = _fit_index(raw, fit_smiles, block.spec["name"])

    # infinities are as unusable to PCA as NaN, and RDKit emits them; both
    # become missing values for the imputer to fill from fit rows
    values = raw.to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan

    imputer = SimpleImputer(strategy=impute, keep_empty_features=True)
    imputer.fit(values[fit_index])
    filled = np.asarray(imputer.transform(values), dtype=np.float64)

    if width is None:
        reduced = filled
        columns = [f"{block.spec['name']}_{name}" for name in raw.columns]
        explained = None
    else:
        pca = PCA(n_components=width, random_state=seed)
        pca.fit(filled[fit_index])
        reduced = np.asarray(pca.transform(filled), dtype=np.float64)
        columns = [f"{block.spec['name']}_pc{i:03d}" for i in range(width)]
        explained = float(pca.explained_variance_ratio_.sum())

    frame = pd.DataFrame(reduced, index=raw.index.copy(), columns=columns)
    artifact.root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(artifact.path)
    artifact.write_record(
        n_rows=int(frame.shape[0]),
        n_columns=int(frame.shape[1]),
        n_fit_rows=int(fit_index.sum()),
        explained_variance_ratio=explained,
        n_missing_imputed=int(np.isnan(values).sum()),
    )
    logger.info(
        "%s w=%s: %d x %d, fitted on %d rows%s",
        block.spec["name"],
        width,
        *frame.shape,
        int(fit_index.sum()),
        "" if explained is None else f", explained variance {explained:.3f}",
    )
    return artifact


def load(artifact: provenance.Artifact) -> pd.DataFrame:
    """Read a cached reduction, indexed by canonical SMILES."""
    return pd.read_parquet(artifact.path)


def _fit_index(raw: pd.DataFrame, fit_smiles: Sequence[str], name: str) -> np.ndarray:
    """Return the boolean row mask selecting the fit partition, checking coverage."""
    missing = set(fit_smiles) - set(raw.index)
    if missing:
        raise LeakageError(
            f"{name}: the block is missing {len(missing)} fit molecules, so fitting on it "
            "would silently use a smaller partition than asked for"
        )
    mask = raw.index.isin(fit_smiles)
    if not mask.any():
        raise LeakageError(f"{name}: the fit partition selects no rows of this block")
    return mask
