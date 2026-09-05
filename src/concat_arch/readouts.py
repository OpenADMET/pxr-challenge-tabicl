"""Pivot the single-concentration screen into one log2FC column per concentration.

The raw file holds one row per compound per concentration per plate. The
auxiliary encoder wants one row per compound and one column per concentration,
so replicate measurements at the same concentration are pooled by median and
the result is indexed by the same canonical SMILES the splits are keyed on.

Four concentrations were screened and only two are used. Against 10,747 rows
at 8.251e-06 and 9,523 at 3.300e-05, the other two carry 27 rows at 9.803e-07
and 706 at 9.901e-05, which over 10,870 compounds is 0.2% and 6.5% coverage.
Columns that sparse contribute almost nothing through a masked loss, so they
are not built at all rather than being built and masked away. The pair that
remains is the one src/encoders.py pretrains on, so the auxiliary encoder and
the pretrained bodies predict the same two quantities.

A compound with no row at a used concentration has no value there, not a zero.
The missing entry is what the auxiliary encoder's masked loss is for.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from data import canonical_smiles

logger = logging.getLogger(__name__)

# the single-concentration screen, the log2FC training signal's only source
LOG2FC_PATH = Path("data/raw/pxr-challenge_single_concentration_TRAIN.csv")

# columns read from it
SMILES_COL = "SMILES"
CONCENTRATION_COL = "concentration_M"
VALUE_COL = "log2_fc_estimate"

# the screened concentrations that carry enough compounds to be a task, in
# molar, ordered low to high. The screen also ran 9.803e-07 and 9.901e-05,
# which the module docstring explains are not built
CONCENTRATIONS_M = (8.251e-06, 3.300e-05)

# the task column names, in the fixed order that decides the readout blocks'
# column order everywhere downstream
TASKS = tuple(f"log2fc_{concentration:.3e}" for concentration in CONCENTRATIONS_M)

# per-molecule parse failures are counted through None returns, so RDKit's
# per-row stderr chatter is redundant
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


def load_readouts(
    path: Path = LOG2FC_PATH,
    *,
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """Read the screen and pivot it to one row per compound.

    Parameters
    ----------
    path : path-like, optional
        The single-concentration CSV. Defaults to the raw file the split was
        built alongside.
    exclude : iterable of str, optional
        Canonical SMILES to drop, which is how phase-2 compounds are kept out
        of anything the auxiliary encoder trains on.

    Returns
    -------
    pandas.DataFrame
        Indexed by canonical SMILES, one float column per concentration in
        :data:`TASKS`, NaN where a compound was not screened at it. Rows with
        no value in any of them are dropped, since they carry no auxiliary
        signal.
    """
    columns = TASKS
    raw = pd.read_csv(path, usecols=[SMILES_COL, CONCENTRATION_COL, VALUE_COL])

    # canonicalize once per distinct input string, not once per row
    unique = pd.Series(raw[SMILES_COL].unique())
    mapping = pd.Series(unique.map(canonical_smiles).to_numpy(), index=unique.to_numpy())
    n_unparsable = int(mapping.isna().sum())
    if n_unparsable:
        logger.warning("log2FC screen: %d unparsable structures dropped", n_unparsable)

    keyed = raw.assign(canonical=raw[SMILES_COL].map(mapping)).dropna(subset=["canonical"])

    # replicate measurements at one concentration pool by median, matching the
    # prior pipeline's own pooling
    pooled = (
        keyed.groupby(["canonical", CONCENTRATION_COL])[VALUE_COL]
        .median()
        .unstack(CONCENTRATION_COL)
    )
    pooled.columns = [f"log2fc_{c:.3e}" for c in pooled.columns]

    # reindex onto the fixed task order so a concentration absent from the file
    # still appears, all-NaN, rather than shifting the block's column meaning
    table = pooled.reindex(columns=list(columns))
    table.index.name = "canonical_smiles"

    excluded = set(exclude)
    if excluded:
        keep = ~table.index.isin(excluded)
        n_dropped = int((~keep).sum())
        if n_dropped:
            logger.info("log2FC screen: %d excluded compounds dropped", n_dropped)
        table = table.loc[keep]

    informative = table.notna().any(axis=1)
    if not informative.all():
        logger.info(
            "log2FC screen: %d compounds have no value in the %d selected task(s)",
            int((~informative).sum()),
            len(columns),
        )
    return table.loc[informative].astype(np.float32)


def readout_blocks(smiles: list[str], table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align a readout table onto a compound list, returning values and their mask.

    Parameters
    ----------
    smiles : list of str
        Canonical SMILES to align onto, in the order the caller wants rows.
    table : pandas.DataFrame
        A :func:`load_readouts` result.

    Returns
    -------
    values : numpy.ndarray
        Shape ``(len(smiles), n_tasks)``, zero wherever nothing was observed.
    mask : numpy.ndarray
        Shape ``(len(smiles), n_tasks)``, 1.0 where a value was observed.

    Notes
    -----
    A missing value reads as zero rather than NaN because the block is fed
    straight into a linear layer. The mask, not the value, is what tells the
    model the difference between "measured as zero" and "not measured".
    """
    aligned = table.reindex(index=smiles)
    mask = aligned.notna().to_numpy(dtype=np.float32)
    values = np.nan_to_num(aligned.to_numpy(dtype=np.float32), nan=0.0)
    return values, mask
