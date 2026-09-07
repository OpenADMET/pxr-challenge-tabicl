"""The resolved configuration a concatenation-architecture run is launched from.

The axis names here are the manifest's own (``experiments/manifest.yaml``,
figure 1): ``encoder_init``, ``finetune_target``, ``aux_encoder``,
``ffn_hidden_dim``, ``freeze_epochs``, ``gradient_clip_val``, ``calibration``.
A figure-1 cell's ``axes`` mapping is passed to :meth:`RunConfig.from_axes`
verbatim, so a cell wires through without a translation table in between.

Everything the axes do not name is a training detail rather than a design
choice under test: learning rates, epoch budget, early-stopping patience, the
auxiliary encoder's own width. Those carry the prior sweep's values as
defaults and stay fixed across cells.

Validation is total and happens at construction. A configuration object that
exists is one that can be run: the cross-axis rule that E4's log2FC-pretrained
body admits no auxiliary encoder, the freeze schedule's sign, and the gradient
clip's positivity are all settled here rather than surfacing as a shape error
forty minutes into a fit.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# how the main model's message-passing body is initialised; the manifest's
# vocabulary exactly, no more
ENCODER_INITS = frozenset({"chemeleon", "log2fc_checkpoint", "scratch"})

# what the main model is fine-tuned against
FINETUNE_TARGETS = frozenset({"pec50"})

# what the auxiliary encoder is pretrained against
AUX_TARGETS = frozenset({"log2fc"})

# the only regressor family this module builds
MODEL_KINDS = frozenset({"gnn"})

# post-hoc calibration is stage three's business, not this module's
CALIBRATIONS = frozenset({"none"})

# predictor widths the grid swept
FFN_HIDDEN_DIMS = frozenset({512, 1024})

# body checkpoint spelling that asks for a small randomly initialised encoder
# instead of a pretrained one; test and diagnostic use only
RANDOM_BODY = "random"


class ConfigError(ValueError):
    """A configuration names an axis value, or a combination of them, that cannot run."""


@dataclass(frozen=True)
class AuxEncoderConfig:
    """The auxiliary encoder arm, or the absence of it when a cell sets ``aux_encoder: null``.

    Attributes
    ----------
    target : str
        What the encoder is pretrained against. Only ``log2fc``.
    use_observed_readout : bool
        Whether a compound's own measured log2FC values (and their observed
        mask) are concatenated into the main model's predictor input.
    use_predicted_readout : bool
        Whether the encoder's predicted log2FC values are concatenated. These
        exist for every compound, so unlike the observed block they are
        populated identically at fit and at inference.
    use_embedding : bool
        Whether the encoder's pooled structural embedding is concatenated. It
        is 2,048 of the vector's columns against the readout blocks' handful,
        so turning it off is what asks whether the predicted readout carries
        anything on its own rather than only on top of the embedding.
    """

    target: str = "log2fc"
    use_observed_readout: bool = False
    use_predicted_readout: bool = False
    use_embedding: bool = True

    def __post_init__(self) -> None:
        """Reject an auxiliary arm whose target is unsupported or that carries nothing."""
        if self.target not in AUX_TARGETS:
            raise ConfigError(
                f"aux_encoder.target {self.target!r} is not one of {sorted(AUX_TARGETS)}"
            )

        # an arm supplying neither an embedding nor a readout still trains an
        # encoder and then concatenates nothing but zeros, which is a
        # configuration that costs a pretraining and cannot differ from
        # aux_encoder: null
        if not (self.use_embedding or self.use_observed_readout or self.use_predicted_readout):
            raise ConfigError(
                "aux_encoder supplies neither an embedding nor a readout, so it would "
                "train an encoder and concatenate nothing; set aux_encoder: null instead"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings held fixed across figure-1 cells.

    These are not axes. They carry the prior sweep's values, with the
    early-stopping patience the seed sweep actually ran at (5, not the
    configs' default of 10).

    Attributes
    ----------
    max_epochs : int
        Epoch budget for the main model, and the length the noam schedule
        calibrates its decay against. A ``freeze_epochs`` at or above this
        means the body never unfreezes.
    aux_max_epochs : int
        Epoch budget for the auxiliary encoder's pretraining, calibrating its
        schedule the same way.
    warmup_epochs, aux_warmup_epochs : int
        Epochs each noam schedule spends climbing to its peak learning rate.
    batch_size : int
        Molecules per optimisation step, both models.
    mpnn_lr, ffn_lr : float
        Peak learning rates for the message-passing body and the predictor
        head. They are the noam schedule's peaks rather than constant rates.
    aux_lr : float
        Single peak learning rate for the auxiliary encoder; the prior recipe
        drew no discriminative split there.
    aux_ffn_hidden_dim, aux_ffn_num_layers, aux_freeze_epochs : int
        The auxiliary encoder's own head width, head depth, and warm-up.
    aux_val_fraction : float
        Seeded held-out fraction of the log2FC pool used for the auxiliary
        encoder's early stopping.
    patience : int
        Early-stopping patience, in epochs, for both models.
    min_delta : float
        Improvement below which an epoch does not reset the patience counter.
    refit_on_all : bool
        After early stopping picks an epoch count on the training partition,
        reinitialise and retrain on the training and validation partitions
        together for that many epochs, and predict from that model. This is
        what makes the graph networks see the same 4,392 compounds the tabular
        models do, and it is what the challenge entry itself did: it retrained
        on the training set plus the released phase-1 compounds before
        submitting. Set false to predict from the first pass alone.
    restore_best : bool
        Whether the weights predictions are made from are the epoch with the
        best validation loss rather than the last epoch run. The prior sweep
        kept the last epoch; set this False to reproduce that.
    accelerator : str
        Lightning accelerator string.
    num_workers : int
        DataLoader worker count.
    inference_batch_size : int
        Molecules per forward pass at prediction time.
    """

    max_epochs: int = 30
    aux_max_epochs: int = 30
    batch_size: int = 64
    mpnn_lr: float = 1e-3
    ffn_lr: float = 1e-3
    aux_lr: float = 1e-3
    aux_ffn_hidden_dim: int = 512
    aux_ffn_num_layers: int = 2
    aux_freeze_epochs: int = 2
    warmup_epochs: int = 2
    aux_warmup_epochs: int = 2
    aux_val_fraction: float = 0.2
    patience: int = 5
    min_delta: float = 1e-3
    refit_on_all: bool = True
    restore_best: bool = True
    accelerator: str = "auto"
    num_workers: int = 0
    inference_batch_size: int = 256


@dataclass(frozen=True)
class RunConfig:
    """One figure-1 cell, resolved to everything a fit needs.

    Attributes
    ----------
    model_kind : str
        Always ``gnn``; the axis exists so a cell reads the same across
        figures that do vary it.
    encoder_init : str
        ``chemeleon`` initialises the body from the CheMeleon foundation
        checkpoint. ``log2fc_checkpoint`` initialises it from a body
        pretrained on log2FC, the E4 recipe, which replaces the foundation
        init rather than sitting beside it. ``scratch`` initialises it from
        nothing, which is the control the other two are worth measuring
        against: without it a figure can compare pretraining recipes and
        cannot say whether pretraining paid at all. It takes its width and
        depth from ``message_hidden_dim`` and ``depth``, since no checkpoint
        supplies them.
    finetune_target : str
        Always ``pec50``.
    aux_encoder : AuxEncoderConfig or None
        The auxiliary arm, or None for a plain fine-tune.
    ffn_hidden_dim : int
        Predictor head width, 512 or 1024.
    ffn_num_layers : int
        Predictor head depth. Not a figure-1 axis; the grid held it at 3.
    freeze_epochs : int
        Epochs the message-passing body stays frozen. A value at or above
        ``training.max_epochs`` means it never unfreezes.
    gradient_clip_val : float or None
        Gradient-norm clip, or None for no clipping.
    calibration : str
        Always ``none``. Post-hoc calibration is applied downstream.
    body_checkpoint : Path or str or None
        Where the message-passing body's weights come from, bypassing
        ``encoder_init``'s own resolution. Not an axis. For
        ``encoder_init="log2fc_checkpoint"`` this is the pretrained body, and
        when it is None the body is requested from ``encoders`` at run time.
        For ``encoder_init="chemeleon"`` it replaces the foundation
        checkpoint for the main body and the auxiliary encoder alike, which
        is how a test gets a cheap encoder; the literal ``"random"`` asks for
        a small randomly initialised body.
    message_hidden_dim, depth : int
        Body width and message-passing depth, used only when the body is
        randomly initialised. A checkpoint supplies its own.
    training : TrainingConfig
        Optimisation settings held fixed across cells.
    """

    model_kind: str = "gnn"
    encoder_init: str = "chemeleon"
    finetune_target: str = "pec50"
    aux_encoder: AuxEncoderConfig | None = None
    ffn_hidden_dim: int = 512
    ffn_num_layers: int = 3
    freeze_epochs: int = 2
    gradient_clip_val: float | None = None
    calibration: str = "none"
    body_checkpoint: Path | str | None = None
    message_hidden_dim: int = 300
    depth: int = 3
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        """Check every axis value and the one rule that spans two of them."""
        if self.model_kind not in MODEL_KINDS:
            raise ConfigError(f"model_kind {self.model_kind!r} is not one of {sorted(MODEL_KINDS)}")
        if self.encoder_init not in ENCODER_INITS:
            raise ConfigError(
                f"encoder_init {self.encoder_init!r} is not one of {sorted(ENCODER_INITS)}"
            )
        if self.finetune_target not in FINETUNE_TARGETS:
            raise ConfigError(
                f"finetune_target {self.finetune_target!r} is not one of {sorted(FINETUNE_TARGETS)}"
            )
        if self.calibration not in CALIBRATIONS:
            raise ConfigError(
                f"calibration {self.calibration!r} is not one of {sorted(CALIBRATIONS)}"
            )
        if self.ffn_hidden_dim not in FFN_HIDDEN_DIMS:
            raise ConfigError(
                f"ffn_hidden_dim {self.ffn_hidden_dim!r} is not one of {sorted(FFN_HIDDEN_DIMS)}"
            )
        if self.ffn_num_layers < 1:
            raise ConfigError(f"ffn_num_layers must be at least 1, got {self.ffn_num_layers!r}")

        # a negative warm-up has no meaning; zero (unfreeze immediately) does
        if self.freeze_epochs < 0:
            raise ConfigError(f"freeze_epochs must be non-negative, got {self.freeze_epochs!r}")

        # Lightning treats a non-positive clip as "no clipping" silently, which
        # would make a cell claiming a clip run without one
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ConfigError(
                f"gradient_clip_val must be positive or None (no clipping), "
                f"got {self.gradient_clip_val!r}"
            )

        # E4 pretrains the body on log2FC in place of the foundation init; the
        # auxiliary encoder is the alternative way of spending that same
        # signal, so the two together would use log2FC twice over
        if self.encoder_init == "log2fc_checkpoint" and self.aux_encoder is not None:
            raise ConfigError(
                "encoder_init='log2fc_checkpoint' replaces the foundation init with a "
                "log2FC-pretrained body and admits no auxiliary encoder; got "
                f"aux_encoder={self.aux_encoder!r}"
            )

        # scratch exists to say what pretraining bought, so an auxiliary
        # encoder beside it would be pretraining by another route and the cell
        # would answer neither question
        if self.encoder_init == "scratch" and self.aux_encoder is not None:
            raise ConfigError(
                "encoder_init='scratch' is the control for pretraining and admits no "
                f"auxiliary encoder; got aux_encoder={self.aux_encoder!r}"
            )

        if self.body_checkpoint is not None and self.body_checkpoint != RANDOM_BODY:
            path = Path(self.body_checkpoint)
            if not path.exists():
                raise ConfigError(f"body_checkpoint {str(path)!r} does not exist")

    @property
    def body_never_unfreezes(self) -> bool:
        """Whether the freeze schedule keeps the body frozen for the whole run."""
        return self.freeze_epochs >= self.training.max_epochs

    @property
    def uses_aux_encoder(self) -> bool:
        """Whether a pretrained auxiliary encoder feeds the main model's predictor."""
        return self.aux_encoder is not None

    @classmethod
    def from_axes(cls, axes: dict[str, Any], **overrides: Any) -> RunConfig:
        """Build a configuration from a manifest cell's ``axes`` mapping.

        Parameters
        ----------
        axes : dict
            A figure-1 cell's ``axes``, using the manifest's own names. The
            ``aux_encoder`` entry is either None or a mapping of
            :class:`AuxEncoderConfig` fields.
        **overrides
            Non-axis fields to set, such as ``body_checkpoint`` or
            ``training``.

        Returns
        -------
        RunConfig
            The validated configuration.

        Raises
        ------
        ConfigError
            If the mapping names a field this configuration does not have, or
            any value fails validation.

        Examples
        --------
        >>> config = RunConfig.from_axes(
        ...     {
        ...         "model_kind": "gnn",
        ...         "encoder_init": "chemeleon",
        ...         "finetune_target": "pec50",
        ...         "aux_encoder": {"target": "log2fc", "tasks": 2},
        ...         "ffn_hidden_dim": 512,
        ...         "freeze_epochs": 2,
        ...         "gradient_clip_val": None,
        ...         "calibration": "none",
        ...     }
        ... )
        >>> config.uses_aux_encoder, config.aux_encoder.target
        (True, 'log2fc')
        """
        fields = {f for f in cls.__dataclass_fields__}
        unknown = set(axes) - fields
        if unknown:
            raise ConfigError(
                f"axes name fields this configuration does not have: {sorted(unknown)}"
            )

        resolved = dict(axes)
        aux = resolved.get("aux_encoder")
        if isinstance(aux, dict):
            aux_unknown = set(aux) - set(AuxEncoderConfig.__dataclass_fields__)
            if aux_unknown:
                raise ConfigError(f"aux_encoder names unknown fields: {sorted(aux_unknown)}")
            resolved["aux_encoder"] = AuxEncoderConfig(**aux)
        return cls(**resolved, **overrides)

    def with_training(self, **changes: Any) -> RunConfig:
        """Return a copy with some of the fixed training settings changed."""
        return replace(self, training=replace(self.training, **changes))

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as plain JSON-compatible data.

        Returns
        -------
        dict
            Every field, nested dataclasses flattened into mappings and any
            path rendered as a string, ready to hash into a cache key.
        """
        payload = asdict(self)
        if payload["body_checkpoint"] is not None:
            payload["body_checkpoint"] = str(payload["body_checkpoint"])
        return payload
