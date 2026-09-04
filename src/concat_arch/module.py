"""The Lightning module both the auxiliary encoder and the main model are.

Two models, one class. The auxiliary encoder is this module wrapping a
multi-task D-MPNN over log2FC; the main model is the same module wrapping a
single-task D-MPNN over pEC50 whose predictor also takes a per-compound
feature vector. What differs between them is the network handed in and the
learning rates, not the training step.

The freeze schedule is the part worth stating plainly. The message-passing
body starts frozen and unfreezes at the epoch ``freeze_epochs`` names, at
which point the optimizer is rebuilt to pick up a second parameter group at
the body's own learning rate. Setting ``freeze_epochs`` at or above the epoch
budget therefore means the body never trains at all, which is E4's literal
recipe rather than an accident of a large number.

Inference chunks the input itself instead of going through chemprop's
dataloader, so a chunk's feature rows line up with its graphs by construction.
The model carries no batch normalisation over the pooled embedding, so the
chunk size cannot change what comes out.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import lightning
import numpy as np
import torch
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset
from chemprop.models import MPNN
from torch import Tensor
from torch.optim import Adam

from .backbone import body_parameters, head_parameters
from .datasets import Batch

logger = logging.getLogger(__name__)


def masked_mse_loss(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Return mean squared error over the observed target entries alone.

    Parameters
    ----------
    predictions : Tensor
        Shape ``(batch, n_tasks)``.
    targets : Tensor
        Shape ``(batch, n_tasks)``. Entries where ``mask`` is zero are ignored
        and may hold any placeholder.
    mask : Tensor
        Shape ``(batch, n_tasks)``, non-zero where a target was observed.

    Returns
    -------
    Tensor
        Scalar loss, differentiable with respect to ``predictions``. A batch
        with nothing observed returns a zero still attached to the graph, so
        the step stays well defined rather than dividing by zero.

    Examples
    --------
    >>> import torch
    >>> predictions = torch.tensor([[1.0, 5.0]])
    >>> targets = torch.tensor([[3.0, 0.0]])
    >>> mask = torch.tensor([[1.0, 0.0]])
    >>> float(masked_mse_loss(predictions, targets, mask))
    4.0
    """
    weights = mask.to(predictions.dtype)
    denominator = weights.sum()
    if denominator == 0:
        return predictions.sum() * 0.0
    return ((predictions - targets) ** 2 * weights).sum() / denominator


class GraphRegressor(lightning.LightningModule):
    """A D-MPNN trained by masked regression, with a body freeze schedule.

    Parameters
    ----------
    model : MPNN
        The network, from :func:`concat_arch.backbone.build_mpnn`.
    freeze_epochs : int
        Epochs the message-passing body stays frozen. At or above the trainer's
        epoch budget it never unfreezes.
    body_lr : float
        Learning rate applied to the body once it unfreezes.
    head_lr : float
        Learning rate applied to the aggregation and predictor from the start.
    weight_decay : float, optional
        Applied to both parameter groups. Defaults to 0.
    prefix : str, optional
        Prepended to the logged metric names, so an auxiliary fit and a main
        fit can be told apart in one log. Defaults to no prefix.
    """

    def __init__(
        self,
        model: MPNN,
        freeze_epochs: int,
        body_lr: float,
        head_lr: float,
        weight_decay: float = 0.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.freeze_epochs = freeze_epochs
        self.body_lr = body_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.prefix = prefix
        self.body_is_frozen = True
        self._freeze_body()

    @property
    def embedding_dim(self) -> int:
        """Width of the pooled structural embedding the body produces."""
        return cast(int, cast(Any, self.model).message_passing.output_dim)

    @property
    def train_metric(self) -> str:
        """Name this fit logs its training loss under."""
        return f"{self.prefix}train_loss"

    @property
    def val_metric(self) -> str:
        """Name this fit logs its validation loss under, and early stopping watches."""
        return f"{self.prefix}val_loss"

    def _freeze_body(self) -> None:
        """Hold the message-passing body's parameters out of the optimizer."""
        for parameter in body_parameters(self.model):
            parameter.requires_grad_(False)
        self.body_is_frozen = True

    def _unfreeze_body(self) -> None:
        """Return the message-passing body's parameters to the optimizer."""
        for parameter in body_parameters(self.model):
            parameter.requires_grad_(True)
        self.body_is_frozen = False

    def on_train_epoch_start(self) -> None:
        """Unfreeze the body once the warm-up is over, rebuilding the optimizer."""
        if self.body_is_frozen and self.current_epoch >= self.freeze_epochs:
            self._unfreeze_body()
            self.trainer.strategy.setup_optimizers(self.trainer)

    def forward(self, graph: Any, extra: Tensor | None = None) -> Tensor:
        """Return predictions for a batched graph.

        Parameters
        ----------
        graph : BatchMolGraph
            The batched molecular graph.
        extra : Tensor or None, optional
            Per-compound features of shape ``(batch, d)`` concatenated onto
            the pooled embedding. A zero-width tensor counts as absent.

        Returns
        -------
        Tensor
            Shape ``(batch, n_tasks)``.
        """
        features = None if extra is None or extra.shape[1] == 0 else extra
        return cast(Tensor, self.model(graph, X_d=features))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Compute, log and return the masked training loss for one batch."""
        graph, targets, mask, extra = batch
        loss = masked_mse_loss(self(graph, extra), targets, mask)
        self.log(self.train_metric, loss, prog_bar=True, batch_size=targets.shape[0])
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Compute and log the masked validation loss for one batch."""
        graph, targets, mask, extra = batch
        loss = masked_mse_loss(self(graph, extra), targets, mask)
        self.log(self.val_metric, loss, prog_bar=True, batch_size=targets.shape[0])

    def configure_optimizers(self) -> Adam:
        """Return an Adam over the head, joined by the body once it unfreezes.

        Returns
        -------
        Adam
            One parameter group while the body is frozen, two after. Lightning
            calls this again on unfreezing, which is how the body's group
            appears mid-run.
        """
        groups: list[dict[str, Any]] = [
            {
                "params": head_parameters(self.model),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            }
        ]
        if not self.body_is_frozen:
            groups.append(
                {
                    "params": body_parameters(self.model),
                    "lr": self.body_lr,
                    "weight_decay": self.weight_decay,
                }
            )
        return Adam(groups)

    @torch.inference_mode()
    def embed(self, smiles: list[str], batch_size: int = 256) -> np.ndarray:
        """Return pooled structural embeddings, stopping short of the predictor.

        Parameters
        ----------
        smiles : list of str
            Canonical, salt-stripped SMILES.
        batch_size : int, optional
            Molecules per forward pass. Defaults to 256.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(smiles), embedding_dim)``, aligned with the input.
        """
        return self._infer(smiles, batch_size, embed=True, extra=None)

    @torch.inference_mode()
    def predict(
        self, smiles: list[str], batch_size: int = 256, extra: np.ndarray | None = None
    ) -> np.ndarray:
        """Return predictions for a list of SMILES.

        Parameters
        ----------
        smiles : list of str
            Canonical, salt-stripped SMILES.
        batch_size : int, optional
            Molecules per forward pass. Defaults to 256.
        extra : numpy.ndarray or None, optional
            Per-compound features of shape ``(len(smiles), d)``, required when
            the network was built with a concatenated input.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(smiles), n_tasks)``, aligned with the input.
        """
        return self._infer(smiles, batch_size, embed=False, extra=extra)

    def _infer(
        self, smiles: list[str], batch_size: int, *, embed: bool, extra: np.ndarray | None
    ) -> np.ndarray:
        """Run chunked inference, each chunk's features sliced to match its graphs."""
        if extra is not None and len(extra) != len(smiles):
            raise ValueError(
                f"extra features ({len(extra)}) must have one row per SMILES ({len(smiles)})"
            )
        if not smiles:
            width = self.embedding_dim if embed else cast(Any, self.model).predictor.n_tasks
            return np.zeros((0, width), dtype=np.float32)

        was_training = self.training
        self.eval()
        outputs: list[np.ndarray] = []
        for start in range(0, len(smiles), batch_size):
            chunk = smiles[start : start + batch_size]
            dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in chunk])  # pyright: ignore[reportArgumentType]
            graph = BatchMolGraph([dataset[i].mg for i in range(len(dataset))])
            graph.to(self.device)
            if embed:
                result = cast(MPNN, self.model).fingerprint(graph)
            else:
                features = (
                    None
                    if extra is None
                    else torch.as_tensor(extra[start : start + len(chunk)], dtype=torch.float32).to(
                        self.device
                    )
                )
                result = self(graph, features)
            outputs.append(result.float().cpu().numpy())
        if was_training:
            self.train()
        return np.concatenate(outputs, axis=0).astype(np.float32)
