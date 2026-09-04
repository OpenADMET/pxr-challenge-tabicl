"""The per-compound vector the auxiliary encoder contributes to the main model.

Five blocks, always in this order and always this wide: the observed log2FC
values, the mask saying which of them were measured, the encoder's predicted
log2FC values, the encoder's pooled structural embedding, and a single flag
marking whether the observed block was populated for this compound.

The width does not depend on the readout flags. Turning off the observed
readout zeroes its block rather than removing it, which costs a few hundred
dead input weights and buys a model whose input width is a function of the
task count alone. Two cells that differ only in a readout flag then produce
predictors of identical shape, and a feature matrix built under one setting
cannot be silently fed to a model fit under the other.

The embedding is the block that is always populated, for every compound, at
fit and at inference alike. That is deliberate: it means a compound with a
measured log2FC looks input-wise like any other compound, differing only in
whether its readout block also carries values, so the training and inference
input distributions have the same shape rather than the same shape most of the
time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .module import GraphRegressor
from .readouts import readout_blocks

logger = logging.getLogger(__name__)


def feature_dim(n_tasks: int, embedding_dim: int) -> int:
    """Return the width of the concatenated feature vector.

    Parameters
    ----------
    n_tasks : int
        Number of log2FC concentration columns the auxiliary encoder predicts.
    embedding_dim : int
        Width of the auxiliary encoder's pooled structural embedding.

    Returns
    -------
    int
        ``3 * n_tasks + embedding_dim + 1``: observed values, observed mask,
        predicted values, embedding, and the readout-used flag.

    Examples
    --------
    >>> feature_dim(2, 2048)
    2055
    """
    return 3 * n_tasks + embedding_dim + 1


def build_features(
    smiles: list[str],
    readouts: pd.DataFrame,
    encoder: GraphRegressor,
    *,
    use_observed_readout: bool,
    use_predicted_readout: bool,
    batch_size: int = 256,
) -> np.ndarray:
    """Build the concatenated feature matrix for a list of compounds.

    Parameters
    ----------
    smiles : list of str
        Canonical SMILES, one per row of the result.
    readouts : pandas.DataFrame
        A :func:`concat_arch.readouts.load_readouts` table, indexed by
        canonical SMILES. Compounds absent from it contribute an all-zero
        observed block and a zero flag.
    encoder : GraphRegressor
        The pretrained auxiliary encoder, supplying the embedding and, when
        asked for, the predicted readout.
    use_observed_readout : bool
        Whether measured log2FC values populate their block.
    use_predicted_readout : bool
        Whether the encoder's own predictions populate theirs.
    batch_size : int, optional
        Molecules per forward pass. Defaults to 256.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(smiles), feature_dim(n_tasks, embedding_dim))``, aligned
        with ``smiles``.
    """
    n_rows = len(smiles)
    n_tasks = readouts.shape[1]

    # observed values and their mask, zero for a compound that was never screened
    if use_observed_readout:
        observed, mask = readout_blocks(smiles, readouts)
        used = mask.any(axis=1, keepdims=True).astype(np.float32)
    else:
        observed = np.zeros((n_rows, n_tasks), dtype=np.float32)
        mask = np.zeros((n_rows, n_tasks), dtype=np.float32)
        used = np.zeros((n_rows, 1), dtype=np.float32)

    # the encoder's own predictions, which exist for every compound and so need
    # no mask or fallback branch
    predicted = (
        encoder.predict(smiles, batch_size=batch_size)
        if use_predicted_readout
        else np.zeros((n_rows, n_tasks), dtype=np.float32)
    )

    embeddings = encoder.embed(smiles, batch_size=batch_size)

    return np.concatenate([observed, mask, predicted, embeddings, used], axis=1).astype(np.float32)
