"""Stage one, seeded: the feature blocks that come out of an encoder we train ourselves.

The blocks in ``features`` are functions of structure alone, so one table serves
every run. These four are not. Each is read off a Chemprop D-MPNN that this
module trains, so its numbers depend on the weight initialisation and on the
order the training batches arrived in. The seed that fixes both is part of every
block's specification, which makes each seed a separate cached artifact and a
five-seed sweep five encoders rather than one encoder used five times.

Two auxiliary supervisions are on offer. The single-concentration file holds
log2FC readouts for 10,870 compounds, more than twice the pool the dose-response
split covers, and a 2-task regression on the two well-populated concentrations
turns that into a representation. The pEC50 fine-tune instead trains on the
target itself, on the fit-train partition alone.

The from-scratch and CheMeleon-initialised log2FC encoders differ only in where
the message-passing weights start, so the pair is an ablation of what the
foundation checkpoint is worth once a few thousand in-domain readouts are
available.

Leakage
-------
**No phase-2 compound ever contributes a training label here.** The pEC50
encoder trains on ``fit_train.csv`` and uses ``fit_val.csv`` for early stopping
and nothing else. The log2FC encoders train on the single-concentration file
with every phase-2 compound removed by canonical structure before a target is
read; as of this writing that filter removes nothing, because the file and the
phase-2 set are already disjoint (0 of 260 phase-2 compounds appear in it), but
the filter runs unconditionally so a refreshed download cannot quietly
introduce a leak. Embeddings are afterwards *extracted* for all 4,652
compounds, which is inference and carries no label.

One caveat that is not leakage but reads like it. The pEC50 embedding is a
function of the fit-train labels, so those rows carry a feature that has seen
their own target. Against phase 2 that is sound, since phase 2 contributed
nothing to the encoder. Any cross-validation *within* the fit set is not: a
held-out fold's embedding was fitted on the folds around it and on the fold
itself, and its scores will read optimistically. Refit the encoder inside the
fold loop if that is ever the question being asked.

Wiring
------
The blocks are defined here rather than in ``features`` so the two modules can
be edited independently, and ``register`` inserts them into ``features.BLOCKS``
under the same ``_Block`` contract the seed-independent blocks use.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

# N812: "L" is Lightning 2.x's own documented alias for its namespace
import lightning as L  # noqa: N812
import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from lightning.pytorch.callbacks import EarlyStopping
from openadmet.models.architecture.chemprop import ChemPropModel

import provenance

# canonical_smiles is the canonicalizer the splits were built with; re-deriving
# it here would risk the log2FC pool and the split disagreeing about what one
# compound is
from data import (
    CANONICAL_COL,
    LOG2FC_FILE,
    RAW_DIR,
    SMILES_COL,
    SPLIT_DIR,
    TARGET_COL,
    canonical_smiles,
)

logger = logging.getLogger(__name__)

# where a trained encoder's weights live, so the embedding and readout blocks of
# one encoder train it once between them
CHECKPOINT_DIR = Path("data/encoders")

# bumped when the meaning of a trained encoder changes, never for a cosmetic edit
VERSION = 3

# columns of the single-concentration file this module reads
LOG2FC_VALUE_COL = "log2_fc_estimate"
LOG2FC_CONCENTRATION_COL = "concentration_M"

# the two concentrations the assay screened the library at, in molar; the file
# also holds a 9.901e-05 arm covering a few percent of compounds and a
# 9.803e-07 arm that is one positive control repeated, and neither carries
# enough coverage to be a task. _log2fc_targets verifies that split against the
# file rather than trusting it
LOG2FC_CONCENTRATIONS = (8.251e-06, 3.300e-05)

# fraction of the compound pool a concentration must cover to be a task; the
# populated arms cover 88% and 99%, the dropped ones 6.5% and 0.2%
TASK_COVERAGE_THRESHOLD = 0.5

# partition whose compounds may never contribute a training label
TEST_PARTITION = "test_phase2.csv"

# partitions the pEC50 encoder fits and early-stops on
PEC50_TRAIN_PARTITION = "fit_train.csv"
PEC50_VAL_PARTITION = "fit_val.csv"

# molecules per forward pass when extracting; batch norm is off in these models,
# so this changes throughput and nothing else
INFERENCE_BATCH_SIZE = 256


class EncoderError(RuntimeError):
    """An encoder could not be trained, or its output did not line up with its input."""


class LeakageError(RuntimeError):
    """A training set was about to include a compound the test partition holds."""


@dataclass(frozen=True)
class EncoderConfig:
    """Everything that determines a trained encoder's weights.

    Every field here lands in the encoder's cache key, so two encoders that
    differ in any of them are different artifacts. ``seed`` fixes the weight
    initialisation, the training-batch order, and the log2FC validation
    carve-out together.

    Attributes
    ----------
    seed : int
        Seed for weight initialisation, batch order, and the validation
        carve-out.
    from_foundation : str or None
        ``chemeleon`` to initialise message passing from the CheMeleon
        checkpoint, None to initialise it randomly. A foundation checkpoint
        supplies its own width and depth, so ``message_hidden_dim`` and
        ``depth`` apply to the random-init case only.
    refit_on_all : bool
        After early stopping picks an epoch count on the training rows,
        reinitialise and retrain on the training and validation rows together
        for that many epochs, and keep that model. The validation rows exist to
        choose a stopping point, and once chosen there is no reason to spend
        them. Set false to keep the first pass's model.
    val_fraction : float or None
        Fraction of the log2FC pool held out for early stopping. Must be None
        for the pEC50 encoder, whose validation partition is a split resource.
    max_epochs : int
        The training budget, and the length noam calibrates its decay against:
        the learning rate reaches ``final_lr`` at this epoch and not before.
        It is deliberately close to where early stopping is expected to land,
        so a fit that stops early has still travelled most of the schedule
        rather than halting near the peak learning rate.
    warmup_epochs : int
        Epochs the noam schedule spends ramping the learning rate to
        ``max_lr`` before it decays. Both passes run noam: a plateau schedule
        needs a validation loss and the second pass has none, and running one
        schedule in each pass would choose an epoch count under the first and
        spend it under the second. The decay is calibrated to ``max_epochs``
        in both passes, so the learning rate at a given epoch is the same
        whichever pass is running.
    """

    seed: int
    from_foundation: str | None = None
    refit_on_all: bool = True
    val_fraction: float | None = 0.2
    message_hidden_dim: int = 256
    depth: int = 3
    ffn_hidden_dim: int = 512
    ffn_num_layers: int = 2
    dropout: float = 0.0
    batch_size: int = 64
    max_lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 30
    freeze_epochs: int = 2
    patience: int = 10
    min_delta: float = 1e-3
    warmup_epochs: int = 2
    gradient_clip_val: float = 1.0
    accelerator: str = "auto"
    num_workers: int = 0


@dataclass(frozen=True)
class TrainingSet:
    """The molecules and multi-task targets one encoder is fitted on.

    Attributes
    ----------
    task_names : list of str
        Ordered target names, one per column of the target arrays.
    train_smiles, val_smiles : list of str
        Canonical SMILES of the training and validation rows.
    train_targets, val_targets : ndarray
        Target matrices of shape ``(n_rows, n_tasks)``, NaN where a compound
        has no observation for that task. Chemprop masks NaN out of the loss.
    """

    task_names: list[str]
    train_smiles: list[str]
    train_targets: np.ndarray
    val_smiles: list[str]
    val_targets: np.ndarray

    @property
    def key(self) -> str:
        """A cache key over the labels themselves, not merely their provenance."""
        return provenance.spec_key(
            {
                "tasks": self.task_names,
                "train_smiles": self.train_smiles,
                "train_targets": self.train_targets.tolist(),
                "val_smiles": self.val_smiles,
                "val_targets": self.val_targets.tolist(),
            }
        )


@dataclass(frozen=True)
class _BlockSpec:
    """A block's producer version, compute function name, and default parameters."""

    version: int
    target: str
    prefix: str
    readout: bool
    defaults: dict[str, Any] = field(default_factory=dict)


def phase2_molecules(split_dir: Path = SPLIT_DIR) -> set[str]:
    """Return the canonical SMILES no encoder here may train on."""
    frame = pd.read_csv(split_dir / TEST_PARTITION, usecols=[CANONICAL_COL], dtype=str)
    return set(frame[CANONICAL_COL])


def log2fc_targets(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Read the single-concentration file as one target row per compound.

    Parameters
    ----------
    raw_dir : path-like, optional
        Directory holding the raw challenge downloads.

    Returns
    -------
    frame : DataFrame
        Indexed by canonical SMILES, one column per task concentration, NaN
        where a compound was not screened at that concentration.
    task_names : list of str
        The column names, in order.
    """
    long = pd.read_csv(
        raw_dir / LOG2FC_FILE,
        usecols=[SMILES_COL, LOG2FC_CONCENTRATION_COL, LOG2FC_VALUE_COL],
        dtype={SMILES_COL: str},
    )
    return _log2fc_targets(long)


def log2fc_training_set(
    *,
    seed: int,
    val_fraction: float,
    raw_dir: Path = RAW_DIR,
    split_dir: Path = SPLIT_DIR,
) -> TrainingSet:
    """Assemble the log2FC training set, with every phase-2 compound removed.

    Parameters
    ----------
    seed : int
        Seed for the validation carve-out.
    val_fraction : float
        Fraction of the pool held out for early stopping.
    raw_dir, split_dir : path-like, optional
        Directories holding the raw downloads and the split resources.

    Returns
    -------
    TrainingSet
        Training and validation rows over the two log2FC tasks.

    Raises
    ------
    LeakageError
        If the phase-2 exclusion leaves nothing to train on.
    """
    targets, task_names = log2fc_targets(raw_dir)

    # the file covers far more compounds than the dose-response split, so it can
    # in principle reach a phase-2 structure; drop those before a label is read
    excluded = phase2_molecules(split_dir)
    keep = ~targets.index.isin(excluded)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.warning(
            "log2fc: dropping %d phase-2 compound(s) from the training pool; they are test "
            "compounds and may not contribute a training label",
            n_dropped,
        )
    targets = targets.loc[keep]
    if targets.empty:
        raise LeakageError("log2fc: the phase-2 exclusion left no compounds to train on")

    return _split_training_set(targets, task_names, seed=seed, val_fraction=val_fraction)


def pec50_training_set(split_dir: Path = SPLIT_DIR) -> TrainingSet:
    """Assemble the pEC50 training set from the fit-train and fit-val partitions.

    The validation partition exists for early stopping and is never fitted on;
    phase 2 appears in neither partition by construction of the split.

    Parameters
    ----------
    split_dir : path-like, optional
        Directory holding the split resource CSVs.

    Returns
    -------
    TrainingSet
        Training and validation rows over the single pEC50 task.
    """
    train = _pec50_targets(split_dir / PEC50_TRAIN_PARTITION)
    val = _pec50_targets(split_dir / PEC50_VAL_PARTITION)
    return TrainingSet(
        task_names=[TARGET_COL],
        train_smiles=list(train.index),
        train_targets=train.to_numpy(dtype=np.float64),
        val_smiles=list(val.index),
        val_targets=val.to_numpy(dtype=np.float64),
    )


def encoder_artifact(
    target: str,
    config: EncoderConfig,
    training: TrainingSet,
    *,
    cache_dir: Path = CHECKPOINT_DIR,
) -> provenance.Artifact:
    """Name the checkpoint a target, a configuration and a training set produce.

    Parameters
    ----------
    target : str
        What the encoder is trained on, ``log2fc`` or ``pec50``.
    config : EncoderConfig
        The configuration, every field of which enters the key.
    training : TrainingSet
        The labels, which enter the key by their own digest so a changed target
        renames the checkpoint.
    cache_dir : path-like, optional
        Root of the checkpoint cache.

    Returns
    -------
    Artifact
        The checkpoint, whose ``path`` holds a ``.pt`` state dict.
    """
    name = f"{'chemeleon' if config.from_foundation else 'chemprop'}_{target}"
    spec = provenance.block_spec(
        f"encoder_{name}",
        VERSION,
        params=asdict(config),
        target=target,
        tasks=training.task_names,
        training=training.key,
        n_train=len(training.train_smiles),
        n_val=len(training.val_smiles),
    )
    return provenance.Artifact(cache_dir / name, spec, suffix=".pt")


def register(registry: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
    """Add the trained-encoder blocks to a block registry.

    Parameters
    ----------
    registry : mutable mapping, optional
        Where to install the blocks. Defaults to ``features.BLOCKS``, which is
        what stage one dispatches on.

    Returns
    -------
    mutable mapping
        The registry, with this module's blocks installed.
    """
    # imported here rather than at module scope so features can register these
    # blocks from its own tail without the two modules importing in a cycle
    import features

    target = features.BLOCKS if registry is None else registry
    for name, spec in BLOCK_SPECS.items():
        # every block here comes out of a training run, so each seed is its own
        # artifact; stage three relies on this flag to thread the seed through
        target[name] = features._Block(
            spec.version,
            _compute_block(name),
            dict(spec.defaults),
            seeded=True,
            encoder=spec.prefix,
        )
    return target


def _log2fc_targets(long: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot the long single-concentration table to one row per compound.

    Raises
    ------
    ValueError
        If the concentrations that clear the coverage threshold are not the
        ones this module calls tasks, which would mean the file changed shape.
    """
    # canonicalize with the split's own canonicalizer, so a compound the split
    # knows and a compound this file knows are the same string; the file is long,
    # one row per compound and concentration, so each distinct input string is
    # canonicalized once rather than once per reading
    canonical = long[SMILES_COL].map(
        {smiles: canonical_smiles(smiles) for smiles in long[SMILES_COL].unique()}
    )
    n_bad = int(canonical.isna().sum())
    if n_bad:
        logger.warning("log2fc: dropping %d unparsable SMILES", n_bad)
    long = long.assign(**{CANONICAL_COL: canonical}).loc[canonical.notna()]

    # duplicate structures do occur, since distinct input SMILES can canonicalize
    # together; a compound screened twice at one concentration gets the mean
    wide = long.pivot_table(
        index=CANONICAL_COL,
        columns=LOG2FC_CONCENTRATION_COL,
        values=LOG2FC_VALUE_COL,
        aggfunc="mean",
    )

    # verify the task choice against the file: exactly the well-covered
    # concentrations become tasks, and they must be the ones we named
    coverage = wide.notna().mean()
    covered = tuple(sorted(coverage.index[coverage >= TASK_COVERAGE_THRESHOLD]))
    if covered != tuple(sorted(LOG2FC_CONCENTRATIONS)):
        raise ValueError(
            f"log2fc: concentrations covering at least {TASK_COVERAGE_THRESHOLD:.0%} of the "
            f"pool are {covered}, not the tasks this module declares "
            f"{tuple(sorted(LOG2FC_CONCENTRATIONS))}; coverage is "
            f"{coverage.round(3).to_dict()}"
        )

    task_names = [f"log2fc_{c:.3e}" for c in sorted(LOG2FC_CONCENTRATIONS)]
    frame = wide[list(covered)].astype(np.float64)
    frame.columns = pd.Index(task_names)
    frame.index.name = CANONICAL_COL

    # a compound with no reading at either task concentration carries no signal
    frame = frame.loc[frame.notna().any(axis=1)]
    return frame.sort_index(), task_names


def _pec50_targets(path: Path) -> pd.DataFrame:
    """Read one potency partition as a single-task target frame."""
    frame = pd.read_csv(path, usecols=[CANONICAL_COL, TARGET_COL])
    frame = frame.loc[frame[TARGET_COL].notna()]

    # pEC50 is already a log-scale quantity, so a compound measured twice
    # averages in the space it is reported in
    grouped = frame.groupby(CANONICAL_COL, sort=True)[[TARGET_COL]].mean()
    if len(grouped) != len(frame):
        logger.info("%s: %d rows collapsed to %d compounds", path.name, len(frame), len(grouped))
    return grouped


def _split_training_set(
    targets: pd.DataFrame, task_names: list[str], *, seed: int, val_fraction: float
) -> TrainingSet:
    """Carve a seeded validation partition out of a target frame."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(targets))
    n_val = max(1, int(round(len(targets) * val_fraction)))
    val_rows = targets.iloc[sorted(order[:n_val])]
    train_rows = targets.iloc[sorted(order[n_val:])]
    if train_rows.empty:
        raise ValueError("the validation carve-out left no training rows")

    return TrainingSet(
        task_names=list(task_names),
        train_smiles=list(train_rows.index),
        train_targets=train_rows.to_numpy(dtype=np.float64),
        val_smiles=list(val_rows.index),
        val_targets=val_rows.to_numpy(dtype=np.float64),
    )


class _StopAfter(L.Callback):
    """End training after a fixed number of epochs, leaving the schedule alone.

    Lightning's ``max_epochs`` is what the noam schedule calibrates its decay
    against, so it has to stay at the budget the first pass ran under. This
    stops the second pass at the epoch that pass chose without shortening the
    schedule underneath it.
    """

    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Ask the trainer to stop once the chosen count is reached."""
        if trainer.current_epoch + 1 >= self.epochs:
            trainer.should_stop = True


class _FreezeMessagePassing(L.Callback):
    """Hold the message-passing encoder fixed for the first few epochs.

    A foundation checkpoint arrives with a randomly initialised head in front
    of it, and the first gradients that head produces are noise. Freezing the
    encoder until the head has settled keeps that noise out of the pretrained
    weights. The same schedule runs for the random-init encoder, where it costs
    a couple of head-only epochs, so the two initialisations differ in nothing
    but where their weights start.
    """

    def __init__(self, freeze_epochs: int) -> None:
        self.freeze_epochs = freeze_epochs

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Set the encoder's requires_grad for the epoch about to start."""
        unfrozen = trainer.current_epoch >= self.freeze_epochs
        for parameter in cast(MPNN, pl_module).message_passing.parameters():
            parameter.requires_grad_(unfrozen)


class _KeepBestWeights(L.Callback):
    """Remember the weights of the best validation epoch, in memory.

    Early stopping leaves the model on the epoch training happened to stop at,
    which is by construction ``patience`` epochs past the best one. Holding a
    copy of the best state costs one model's worth of host memory and is what
    the embeddings are then read off.
    """

    def __init__(self, monitor: str = "val_loss") -> None:
        self.monitor = monitor
        self.best_score: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        # the epoch the best score came from, which is the epoch count a refit
        # on the combined rows then trains for
        self.best_epoch: int | None = None

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Copy the state dict when the monitored metric improves."""
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        score = float(metric)
        if self.best_score is not None and score >= self.best_score:
            return
        self.best_score = score
        self.best_epoch = trainer.current_epoch
        self.best_state = copy.deepcopy(
            {name: tensor.detach().cpu() for name, tensor in pl_module.state_dict().items()}
        )


def _build_model(config: EncoderConfig, n_tasks: int, scaler: Any) -> ChemPropModel:
    """Assemble an unfitted Chemprop model for a configuration."""
    model = ChemPropModel(
        n_tasks=n_tasks,
        from_foundation=config.from_foundation,
        message_hidden_dim=config.message_hidden_dim,
        depth=config.depth,
        ffn_hidden_dim=config.ffn_hidden_dim,
        ffn_num_layers=config.ffn_num_layers,
        dropout=config.dropout,
        # noam in both passes. A plateau schedule reacts to a validation loss,
        # and the second pass has none to react to, so it cannot run one; and a
        # first pass on plateau followed by a second on noam would pick an epoch
        # count under one schedule and spend it under another
        scheduler="noam",
        warmup_epochs=config.warmup_epochs,
        max_lr=config.max_lr,
        weight_decay=config.weight_decay,
    )
    model.build(scaler=scaler)
    return model


def _dataset(smiles: Sequence[str], targets: np.ndarray | None = None) -> MoleculeDataset:
    """Build a Chemprop dataset, with NaN marking an unobserved task."""
    if targets is None:
        points = [MoleculeDatapoint.from_smi(smi) for smi in smiles]
    else:
        points = [
            MoleculeDatapoint.from_smi(smi, y=row.astype(np.float32))
            for smi, row in zip(smiles, targets, strict=True)
        ]
    # from_smi is annotated as returning the mixin it is defined on, so chemprop's
    # own constructor rejects its own factory's output under a type checker
    return MoleculeDataset(points)  # pyright: ignore[reportArgumentType]


def _safe_batch_size(n_rows: int, batch_size: int) -> int:
    """Return a batch size Chemprop will not silently drop a trailing row from.

    ``build_dataloader`` drops a trailing batch of size one so batch norm cannot
    see a single molecule. During extraction that would lose a row and misalign
    the block, so the batch size is nudged down until the remainder is not one.
    """
    effective = min(batch_size, max(n_rows, 1))
    while effective > 1 and n_rows % effective == 1:
        effective -= 1
    return effective


def _train_encoder(target: str, config: EncoderConfig, training: TrainingSet) -> ChemPropModel:
    """Fit one encoder, returning it on its best validation epoch's weights."""
    L.seed_everything(config.seed, workers=True)

    train_dataset = _dataset(training.train_smiles, training.train_targets)
    val_dataset = _dataset(training.val_smiles, training.val_targets)

    # targets are standardized on the training rows alone and the model carries
    # the inverse transform, so predictions come back in the units of the assay
    scaler = train_dataset.normalize_targets()
    val_dataset.normalize_targets(scaler)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    model = _build_model(config, len(training.task_names), scaler)
    best = _KeepBestWeights()
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=config.gradient_clip_val,
        # the readout-bearing pools run to tens of batches an epoch, well under
        # Lightning's default logging interval, which would suppress every step
        log_every_n_steps=1,
        callbacks=[
            _FreezeMessagePassing(config.freeze_epochs),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.patience,
                min_delta=config.min_delta,
            ),
            best,
        ],
    )
    logger.info(
        "%s: training on %d compounds (%d validation), tasks=%s, seed=%d",
        target,
        len(training.train_smiles),
        len(training.val_smiles),
        training.task_names,
        config.seed,
    )
    trainer.fit(model.estimator, train_loader, val_loader)

    if best.best_state is None:
        raise EncoderError(f"{target}: training produced no validation score to select on")
    model.estimator.load_state_dict(best.best_state)
    logger.info(
        "%s: stopped after %d epoch(s), best val_loss %.4f",
        target,
        trainer.current_epoch,
        best.best_score if best.best_score is not None else float("nan"),
    )

    if not config.refit_on_all:
        return model
    return _refit_on_all(target, config, training, epochs=(best.best_epoch or 0) + 1)


def _refit_on_all(
    target: str, config: EncoderConfig, training: TrainingSet, *, epochs: int
) -> ChemPropModel:
    """Retrain from scratch on the training and validation rows together.

    The validation rows earned their keep by choosing the epoch count; spending
    them on that alone would leave the encoder fitted on a fraction of what is
    available. This pass reinitialises from the same seed, trains for exactly
    the epochs the first pass settled, and watches nothing, so there is no
    stopping decision to make and none to leak into.
    """
    L.seed_everything(config.seed, workers=True)

    smiles = [*training.train_smiles, *training.val_smiles]
    targets = np.vstack([training.train_targets, training.val_targets])
    dataset = _dataset(smiles, targets)

    # the scaler belongs to whatever this pass trains on, which is now both parts
    scaler = dataset.normalize_targets()
    loader = build_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        seed=config.seed,
    )

    model = _build_model(config, len(training.task_names), scaler)

    # noam calibrates its decay to the trainer's epoch budget, so the budget
    # here is the first pass's, not the count it settled on. Otherwise this pass
    # would compress the whole schedule into a handful of epochs and train under
    # learning rates the first pass never saw, which would make the count it
    # chose meaningless. The run is stopped at that count instead
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=1,
        callbacks=[_FreezeMessagePassing(config.freeze_epochs), _StopAfter(epochs)],
    )
    logger.info(
        "%s: refitting on %d compounds for %d epoch(s), no validation",
        target,
        len(smiles),
        epochs,
    )
    trainer.fit(model.estimator, loader)
    return model


def _ensure_encoder(
    target: str,
    config: EncoderConfig,
    training: TrainingSet,
    *,
    cache_dir: Path = CHECKPOINT_DIR,
) -> ChemPropModel:
    """Train the encoder for a configuration, or restore the cached one.

    The embedding and readout blocks of one encoder resolve to the same
    checkpoint, so the pair costs one training run between them.
    """
    artifact = encoder_artifact(target, config, training, cache_dir=cache_dir)
    n_tasks = len(training.task_names)

    if artifact.is_cached:
        logger.info("%s: restoring encoder from %s", target, artifact.path)
        state = torch.load(artifact.path, weights_only=True, map_location="cpu")
        if list(state["task_names"]) != training.task_names:
            raise EncoderError(
                f"{target}: checkpoint holds tasks {state['task_names']}, "
                f"expected {training.task_names}"
            )
        # the target scaler travels in the state dict as the output transform's
        # buffers, so the rebuilt model needs no scaler of its own
        model = _build_model(config, n_tasks, scaler=None)
        model.estimator.load_state_dict(state["state_dict"])
        return model

    with provenance.timed() as elapsed:
        model = _train_encoder(target, config, training)
    with provenance.atomic(artifact.path) as partial:
        torch.save(
            {
                "task_names": training.task_names,
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.estimator.state_dict().items()
                },
            },
            partial,
        )
    artifact.write_record(
        wall_clock_s=elapsed(),
        target=target,
        tasks=training.task_names,
        n_train=len(training.train_smiles),
        n_val=len(training.val_smiles),
    )
    logger.info("%s: wrote encoder checkpoint -> %s", target, artifact.path)
    return model


def _predict_readout(
    model: ChemPropModel, molecules: Sequence[str], config: EncoderConfig
) -> np.ndarray:
    """Run the full forward pass, returning predictions in the assay's own units."""
    dataset = _dataset(molecules)
    loader = build_dataloader(
        dataset,
        batch_size=_safe_batch_size(len(dataset), INFERENCE_BATCH_SIZE),
        num_workers=config.num_workers,
        shuffle=False,
    )
    trainer = L.Trainer(
        accelerator=config.accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    batches = trainer.predict(model.estimator, loader)
    if not batches:
        raise EncoderError("readout: prediction returned no batches")
    return torch.cat([torch.as_tensor(batch) for batch in batches]).cpu().numpy()


def _frame(molecules: Sequence[str], values: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Wrap an extracted matrix as a block, checking it lines up with its input."""
    if values.shape != (len(molecules), len(columns)):
        raise EncoderError(
            f"extraction returned {values.shape}, expected ({len(molecules)}, {len(columns)})"
        )
    return pd.DataFrame(
        np.asarray(values, dtype=np.float64),
        index=pd.Index(list(molecules), name=CANONICAL_COL),
        columns=pd.Index(list(columns)),
    )


def embedding_columns(prefix: str, width: int) -> list[str]:
    """Name the columns of an embedding block, one per encoder output unit."""
    return [f"{prefix}_{i:04d}" for i in range(width)]


def _training_set_for(target: str, config: EncoderConfig, **paths: Path) -> TrainingSet:
    """Assemble the training set a target names, checking the configuration fits it."""
    if target == "log2fc":
        if config.val_fraction is None:
            raise ValueError("the log2fc encoder needs a val_fraction to carve validation from")
        return log2fc_training_set(
            seed=config.seed,
            val_fraction=config.val_fraction,
            raw_dir=paths.get("raw_dir", RAW_DIR),
            split_dir=paths.get("split_dir", SPLIT_DIR),
        )
    if target == "pec50":
        if config.val_fraction is not None:
            raise ValueError(
                "the pec50 encoder validates on fit_val.csv, so val_fraction must be None"
            )
        return pec50_training_set(paths.get("split_dir", SPLIT_DIR))
    raise KeyError(f"unknown encoder target {target!r}")


def _compute_block(name: str) -> Callable[..., pd.DataFrame]:
    """Return the compute function for a block, closed over its specification."""
    spec = BLOCK_SPECS[name]

    def compute(molecules: Sequence[str], **params: Any) -> pd.DataFrame:
        """Train (or restore) the encoder and extract this block for the molecules."""
        config = EncoderConfig(**params)
        training = _training_set_for(spec.target, config)
        # read the module attribute at call time so a test can redirect the cache
        model = _ensure_encoder(spec.target, config, training, cache_dir=CHECKPOINT_DIR)

        molecules = list(molecules)
        if spec.readout:
            values = _predict_readout(model, molecules, config)
            return _frame(molecules, values, training.task_names)

        values = model.predict_embedding(
            molecules, batch_size=INFERENCE_BATCH_SIZE, accelerator=config.accelerator
        )
        return _frame(molecules, values, embedding_columns(spec.prefix, values.shape[1]))

    compute.__name__ = f"_compute_{name}"
    return compute


def _defaults(**overrides: Any) -> dict[str, Any]:
    """Render an EncoderConfig's fields as a block's default parameters."""
    return asdict(EncoderConfig(seed=0, **overrides))


# Bump a block's version when the meaning of its output changes; never for a
# change that cannot alter the numbers. A prefix's embedding and readout blocks
# carry identical defaults on purpose: that is what makes them one trained
# encoder, read two ways, rather than two encoders that happen to agree.
BLOCK_SPECS: dict[str, _BlockSpec] = {
    "chemprop_log2fc_embedding": _BlockSpec(
        version=VERSION,
        target="log2fc",
        prefix="chemprop_log2fc",
        readout=False,
        defaults=_defaults(from_foundation=None),
    ),
    "chemprop_log2fc_readout": _BlockSpec(
        version=VERSION,
        target="log2fc",
        prefix="chemprop_log2fc",
        readout=True,
        defaults=_defaults(from_foundation=None),
    ),
    "chemeleon_log2fc_embedding": _BlockSpec(
        version=VERSION,
        target="log2fc",
        prefix="chemeleon_log2fc",
        readout=False,
        defaults=_defaults(from_foundation="chemeleon"),
    ),
    # the readout twin of the block above, off the same trained encoder. Without
    # it the readout axis offers predictions from a randomly initialised
    # chemprop alone, so "adding the readout is worth X" in the tabular figures
    # and the same sentence about the graph networks' auxiliary arm, which is
    # CheMeleon-initialised, would be claims about different networks
    "chemeleon_log2fc_readout": _BlockSpec(
        version=VERSION,
        target="log2fc",
        prefix="chemeleon_log2fc",
        readout=True,
        defaults=_defaults(from_foundation="chemeleon"),
    ),
    "chemeleon_pec50_embedding": _BlockSpec(
        version=VERSION,
        target="pec50",
        prefix="chemeleon_pec50",
        readout=False,
        defaults=_defaults(from_foundation="chemeleon", val_fraction=None),
    ),
    # the fourth corner of what the previous generation explored: an encoder is
    # initialised from the foundation checkpoint or from nothing, and trained
    # on log2FC or on pEC50, and the other three combinations are above. The
    # prior sweep ran this one and it beat the CheMeleon-initialised pEC50
    # encoder that is here, so its absence was an omission rather than a
    # decision
    "chemprop_pec50_embedding": _BlockSpec(
        version=VERSION,
        target="pec50",
        prefix="chemprop_pec50",
        readout=False,
        defaults=_defaults(from_foundation=None, val_fraction=None),
    ),
}


def log2fc_body_checkpoint(
    seed: int,
    *,
    block: str = "chemprop_log2fc_embedding",
    cache_dir: Path = CHECKPOINT_DIR,
    **paths: Path,
) -> Path:
    """Return a message-passing body pretrained on log2FC, for the E4 arm.

    E4 replaces the foundation initialisation itself: rather than concatenating
    an auxiliary encoder's output, it starts the main model from a body already
    trained on log2FC and fine-tunes that on pEC50. The body it needs is the one
    inside the from-scratch log2FC encoder this module already trains and
    caches, so this extracts it rather than training a second one.

    Parameters
    ----------
    seed : int
        The replicate seed, so the pretrained body matches the run it feeds.
    block : str, optional
        Which log2FC encoder to take the body from. The default is the one
        trained from scratch, which is the E4 recipe. Naming the
        CheMeleon-initialised encoder instead gives a body that was pretrained
        twice, once on structures and once on the screen.
    cache_dir : path-like, optional
        Root of the checkpoint cache.
    **paths
        ``raw_dir`` and ``split_dir`` overrides, passed through to the training
        set.

    Returns
    -------
    Path
        A torch-saved mapping with ``hyper_parameters`` and ``state_dict`` keys
        for chemprop's ``BondMessagePassing``, in the shape the vendored
        architecture loads a foundation checkpoint in.
    """
    # the log2FC encoder, the same one the embedding and readout blocks share,
    # so an arm that starts from its body costs no additional training
    spec = BLOCK_SPECS[block]
    config = EncoderConfig(**{**spec.defaults, "seed": seed})
    training = _training_set_for(spec.target, config, **paths)
    artifact = encoder_artifact(spec.target, config, training, cache_dir=cache_dir)

    body_path = cache_dir / "log2fc_body" / f"{artifact.key}.pt"
    if body_path.exists():
        logger.info("log2fc body: cached (%s)", artifact.key)
        return body_path

    model = _ensure_encoder(spec.target, config, training, cache_dir=cache_dir)

    # imported here rather than at module scope: the vendored architecture
    # reaches back into this module for exactly this function, and only one of
    # the two directions may bind at import time
    from concat_arch import write_body_checkpoint

    # the trained body's own width and depth, not the configuration's: a
    # foundation initialisation overrides both, so an encoder built from
    # CheMeleon is 2,048 wide however narrow its configuration asked to be
    passing = model.estimator.message_passing
    hyper = {
        "d_h": int(passing.W_h.weight.shape[0]),
        "depth": int(getattr(passing, "depth", config.depth)),
    }

    body_path.parent.mkdir(parents=True, exist_ok=True)
    return write_body_checkpoint(model.estimator.state_dict(), hyper, body_path)
