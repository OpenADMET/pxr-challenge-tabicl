"""One dataset shape serving both models in this package.

The auxiliary encoder and the main model differ in what they regress and in
whether a per-compound feature vector rides alongside the graph, but not in
the shape of a batch: a batched molecular graph, a target matrix, the mask
saying which target entries were observed, and a possibly zero-width extra
feature matrix. Carrying one dataset for both keeps the two training paths
from drifting apart in how they batch, shuffle, or move to the device.

The mask is what makes the two cases one. The auxiliary encoder's log2FC
targets are genuinely missing for most compound-concentration pairs, so its
loss has to skip them; the main model's pEC50 target is present on every row
it is given, so its mask is all ones and the same masked loss reduces to plain
mean squared error.

Splits are explicit here. A validation partition is a list of rows the caller
chose, never a fraction this module carves out, because the challenge's own
validation partition is a file on disk and inventing a second one alongside it
would early-stop on the wrong thing.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning
import numpy as np
import torch
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# a batch: graph, targets, observed mask, extra per-compound features
Batch = tuple[Any, Tensor, Tensor, Tensor]


class GraphDataset(Dataset):
    """Molecular graphs paired with masked targets and optional extra features.

    Parameters
    ----------
    smiles : list of str
        Canonical SMILES, one per row.
    targets : numpy.ndarray
        Shape ``(n, n_tasks)``. Entries where ``mask`` is 0 are ignored by the
        loss and may hold anything.
    mask : numpy.ndarray
        Shape ``(n, n_tasks)``, 1.0 where the target was observed.
    extra : numpy.ndarray or None
        Shape ``(n, d)`` per-compound features concatenated before the
        predictor, or None for a graph-only model.

    Raises
    ------
    ValueError
        If the arrays disagree on row count.
    """

    def __init__(
        self,
        smiles: list[str],
        targets: np.ndarray,
        mask: np.ndarray,
        extra: np.ndarray | None = None,
    ) -> None:
        n = len(smiles)
        if targets.shape[0] != n or mask.shape[0] != n:
            raise ValueError(
                f"smiles ({n}), targets ({targets.shape[0]}) and mask ({mask.shape[0]}) "
                "must agree on row count"
            )
        if extra is not None and extra.shape[0] != n:
            raise ValueError(f"extra features ({extra.shape[0]}) must have {n} rows")

        self.smiles = smiles
        self._graphs = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in smiles])  # pyright: ignore[reportArgumentType]
        self._targets = torch.as_tensor(targets, dtype=torch.float32)
        self._mask = torch.as_tensor(mask, dtype=torch.float32)
        width = 0 if extra is None else extra.shape[1]
        self._extra = (
            torch.zeros(n, 0, dtype=torch.float32)
            if extra is None
            else torch.as_tensor(extra, dtype=torch.float32)
        )
        self.extra_dim = width

    def __len__(self) -> int:
        """Return the number of compounds."""
        return len(self.smiles)

    def __getitem__(self, index: int) -> tuple[Any, Tensor, Tensor, Tensor]:
        """Return the datapoint, target row, mask row and extra-feature row at ``index``."""
        return self._graphs[index], self._targets[index], self._mask[index], self._extra[index]


def collate(items: list[tuple[Any, Tensor, Tensor, Tensor]]) -> Batch:
    """Batch a list of dataset items into graphs, targets, mask and extra features.

    Parameters
    ----------
    items : list
        Items as returned by :meth:`GraphDataset.__getitem__`.

    Returns
    -------
    tuple
        ``(BatchMolGraph, targets, mask, extra)``, the three tensors stacked
        along a new leading batch dimension.
    """
    datapoints, targets, masks, extras = zip(*items, strict=True)
    graph = BatchMolGraph([datapoint.mg for datapoint in datapoints])
    return graph, torch.stack(list(targets)), torch.stack(list(masks)), torch.stack(list(extras))


class GraphDataModule(lightning.LightningDataModule):
    """Two caller-chosen partitions, batched the same way.

    Parameters
    ----------
    train : GraphDataset
        Rows optimised on.
    val : GraphDataset or None
        Rows early stopping watches, or None for a fit with no validation.
    batch_size : int, optional
        Compounds per step. Defaults to 64.
    num_workers : int, optional
        DataLoader workers. Defaults to 0, the main process.
    seed : int, optional
        Seed for the training shuffle. Defaults to 0.
    """

    def __init__(
        self,
        train: GraphDataset,
        val: GraphDataset | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.train = train
        self.val = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def transfer_batch_to_device(
        self, batch: Batch, device: torch.device, dataloader_idx: int
    ) -> Batch:
        """Move a batch's graph and tensors to ``device``.

        Parameters
        ----------
        batch : tuple
            A ``(BatchMolGraph, targets, mask, extra)`` batch.
        device : torch.device
            Where to move it.
        dataloader_idx : int
            Which dataloader produced it; part of the Lightning interface.

        Returns
        -------
        tuple
            The same batch on ``device``.
        """
        graph, targets, mask, extra = batch
        graph = super().transfer_batch_to_device(graph, device, dataloader_idx)
        return graph, targets.to(device), mask.to(device), extra.to(device)

    def train_dataloader(self) -> DataLoader:
        """Return the shuffled training loader, its shuffle seeded per run."""
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation loader, empty when the fit has no validation partition.

        Lightning requires a real iterable from this hook, so a fit without a
        validation partition gets an empty loader rather than None.

        Returns
        -------
        DataLoader
            Unshuffled loader over the validation partition, or an empty one.
        """
        if self.val is None:
            return DataLoader([], batch_size=self.batch_size)  # pyright: ignore[reportArgumentType]
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
