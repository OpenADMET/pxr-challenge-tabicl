"""Assemble the Chemprop D-MPNN both models in this package are built from.

The main pEC50 model and the auxiliary log2FC encoder share this constructor
rather than each rolling its own, so their pooled embeddings live in the same
representation space and the auxiliary encoder's output is something the main
model's predictor can meaningfully consume.

Three body initialisations are supported. The CheMeleon foundation checkpoint
is the figure's default and supplies its own architecture, so a checkpoint's
width and depth win over anything configured. A path to a checkpoint written
in the same shape covers the log2FC-pretrained body the E4 arm needs. A
randomly initialised body at a configured width covers cheap tests and
diagnostics.

Aggregation is always mean: CheMeleon pretrained with a mean readout, and
keeping the random-init branch on the same readout means the two paths differ
only in their weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from urllib.request import urlretrieve

import torch
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from torch import nn

import provenance

from .config import RANDOM_BODY

logger = logging.getLogger(__name__)

# where the CheMeleon checkpoint is cached, matching openadmet-models' own
# location so the two never download it twice
CHEMELEON_CACHE = Path.home() / ".chemprop" / "chemeleon_mp.pt"
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

# the name that asks for the foundation checkpoint rather than a file
CHEMELEON = "chemeleon"


class BackboneError(RuntimeError):
    """A body checkpoint is missing, or holds no weights to load."""


def chemeleon_checkpoint(cache: Path = CHEMELEON_CACHE) -> Path:
    """Return the local CheMeleon checkpoint, downloading it once if absent.

    Parameters
    ----------
    cache : path-like, optional
        Where the checkpoint lives. Defaults to ``~/.chemprop/chemeleon_mp.pt``.

    Returns
    -------
    Path
        The cached checkpoint.

    Notes
    -----
    Source: https://zenodo.org/records/15460715. Cite DOI
    10.48550/arXiv.2506.15792 when using CheMeleon in published work.
    """
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading the CheMeleon foundation checkpoint to %s", cache)
    urlretrieve(CHEMELEON_URL, cache)  # noqa: S310
    return cache


def load_body_weights(source: str | Path) -> dict[str, Any]:
    """Read a message-passing body checkpoint.

    Parameters
    ----------
    source : str or path-like
        ``"chemeleon"`` for the foundation checkpoint, otherwise a path to one
        written in the same shape: a mapping with ``hyper_parameters`` and
        ``state_dict`` keys for :class:`chemprop.nn.BondMessagePassing`.

    Returns
    -------
    dict
        The checkpoint's ``hyper_parameters`` and ``state_dict``.

    Raises
    ------
    BackboneError
        If the file is absent, or carries no weights, which would otherwise
        build a randomly initialised body under a pretrained body's name.
    """
    path = chemeleon_checkpoint() if source == CHEMELEON else Path(source)
    if not path.exists():
        raise BackboneError(f"no message-passing body checkpoint at {path}")

    checkpoint = cast(dict[str, Any], torch.load(path, weights_only=True, map_location="cpu"))
    if not checkpoint.get("state_dict"):
        raise BackboneError(
            f"{path} carries an empty state_dict; refusing to build a randomly "
            "initialised body under a pretrained body's name"
        )
    return checkpoint


def write_body_checkpoint(
    state_dict: dict[str, Any],
    hyper_parameters: dict[str, Any],
    path: Path,
    *,
    prefix: str = "message_passing.",
) -> Path:
    """Extract a message-passing body from a whole network's weights and save it.

    A body pretrained inside a full D-MPNN, an E4 log2FC pretraining run for
    instance, has to be handed to :func:`build_mpnn` in the shape a foundation
    checkpoint comes in: the body's weights alone, with the constructor
    arguments that rebuild it. This performs that conversion.

    Parameters
    ----------
    state_dict : dict
        The whole network's state dict, whose body keys carry ``prefix``.
    hyper_parameters : dict
        Constructor arguments for :class:`chemprop.nn.BondMessagePassing`,
        such as ``d_h`` and ``depth``.
    path : path-like
        Where to write the checkpoint.
    prefix : str, optional
        The body's key prefix in ``state_dict``. Defaults to
        ``message_passing.``, chemprop's own.

    Returns
    -------
    Path
        The written checkpoint, loadable by :func:`load_body_weights`.

    Raises
    ------
    BackboneError
        If no key carries the prefix, which would otherwise write an empty
        checkpoint that only fails much later, or if the declared width
        disagrees with the width of the weights.
    """
    body = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not body:
        raise BackboneError(f"no key in the state dict begins with {prefix!r}")

    # the width the caller declares has to be the width the weights carry. A
    # checkpoint that disagrees loads into a model built at the wrong size and
    # fails on the first shape it reaches, a long way from here: a foundation
    # init overrides the width a configuration asked for, so a caller passing
    # its own configuration's number is the way this goes wrong
    declared = hyper_parameters.get("d_h")
    carried = body["W_h.weight"].shape[0] if "W_h.weight" in body else None
    if declared is not None and carried is not None and int(declared) != int(carried):
        raise BackboneError(
            f"hyper_parameters say d_h={declared} and the weights are {carried} wide; "
            f"the checkpoint would build a body it cannot load"
        )

    with provenance.atomic(path) as partial:
        torch.save({"hyper_parameters": dict(hyper_parameters), "state_dict": body}, partial)
    return path


def build_mpnn(
    body: str | Path,
    *,
    ffn_hidden_dim: int,
    ffn_num_layers: int,
    message_hidden_dim: int = 300,
    depth: int = 3,
    n_tasks: int = 1,
    extra_input_dim: int = 0,
) -> MPNN:
    """Construct the D-MPNN, dispatching on how its body is initialised.

    Parameters
    ----------
    body : str or path-like
        ``"chemeleon"``, ``"random"``, or a path to a body checkpoint. A
        checkpoint supplies the body's own width and depth, so
        ``message_hidden_dim`` and ``depth`` apply to ``"random"`` alone.
    ffn_hidden_dim : int
        Predictor head width.
    ffn_num_layers : int
        Predictor head depth.
    message_hidden_dim : int, optional
        Body width for a random init. Defaults to 300.
    depth : int, optional
        Message-passing steps for a random init. Defaults to 3.
    n_tasks : int, optional
        Regression targets predicted per compound. Defaults to 1, the main
        model's single pEC50; the auxiliary encoder passes one per log2FC
        concentration.
    extra_input_dim : int, optional
        Width of a per-compound feature vector concatenated onto the pooled
        graph embedding before the predictor, through chemprop's own
        ``MPNN.forward(bmg, X_d=...)``. Defaults to 0, no concatenation.

    Returns
    -------
    MPNN
        The assembled network, mean-aggregated.
    """
    if body == RANDOM_BODY:
        logger.info(
            "building a random message-passing body (d_h=%d, depth=%d)", message_hidden_dim, depth
        )
        message_passing = BondMessagePassing(  # pyright: ignore[reportAbstractUsage]
            d_h=message_hidden_dim, depth=depth
        )
    else:
        checkpoint = load_body_weights(body)
        message_passing = BondMessagePassing(  # pyright: ignore[reportAbstractUsage]
            **checkpoint["hyper_parameters"]
        )
        message_passing.load_state_dict(checkpoint["state_dict"])

    predictor = RegressionFFN(  # pyright: ignore[reportAbstractUsage]
        n_tasks=n_tasks,
        input_dim=message_passing.output_dim + extra_input_dim,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers,
    )
    return MPNN(message_passing=message_passing, agg=MeanAggregation(), predictor=predictor)


def body_parameters(model: MPNN) -> list[nn.Parameter]:
    """Return the message-passing body's parameters, the ones a freeze schedule holds."""
    return list(cast(nn.Module, model.message_passing).parameters())


def head_parameters(model: MPNN) -> list[nn.Parameter]:
    """Return the aggregation and predictor parameters, which train from the first epoch."""
    return list(cast(nn.Module, model.agg).parameters()) + list(
        cast(nn.Module, model.predictor).parameters()
    )
