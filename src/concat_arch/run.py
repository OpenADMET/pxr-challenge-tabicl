"""Fit one figure-1 cell and hand back its phase-2 predictions.

This is the module's whole public surface: a configuration, a seed, and the
split files go in; a table of per-compound phase-2 predictions and the fitted
model's own account of what it did come out. Nothing is written to disk. Where
those results live, under what cache key, and beside which provenance record
is stage three's decision, so this returns the specification stage three would
hash rather than hashing it into a directory itself.

The split is the challenge's own and is never resampled here. The model is fit
on ``fit_train.csv``, early-stopped on ``fit_val.csv``, and predicts
``test_phase2.csv``. The one seeded partition anywhere in this module is the
auxiliary encoder's own validation carve-out of the log2FC screen, which
touches no pEC50 label at all.

Phase 2 is checked out of every training pool before a fit starts, by
canonical structure, and again out of the log2FC screen the auxiliary encoder
pretrains on. The screen happens to contain no phase-2 compound, so the second
check currently removes nothing; it is there so that a change to the screen's
contents fails loudly instead of quietly leaking.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import lightning
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import provenance

from .backbone import CHEMELEON, build_mpnn
from .concat_features import build_features
from .config import RANDOM_BODY, RunConfig
from .datasets import GraphDataModule, GraphDataset
from .module import GraphRegressor
from .readouts import LOG2FC_PATH, TASKS, load_readouts

logger = logging.getLogger(__name__)

# bumped when the meaning of this module's output changes, so a cached result
# from an older definition cannot be mistaken for a current one
PRODUCER = "concat_arch"
PRODUCER_VERSION = 2

# where pretrained auxiliary encoders are cached, and the version of what one
# means. The auxiliary encoder is a function of the screen, the seed and its own
# hyperparameters, and of nothing about the main model it will be concatenated
# to, so most of figure 1's cells ask for the same one
AUX_CACHE_DIR = Path("data/encoders/aux")
AUX_VERSION = 1

# the training settings the auxiliary fit actually reads. Anything outside this
# list belongs to the main model and must not change the encoder's identity, or
# cells that could share one would each train their own
AUX_TRAINING_FIELDS = (
    "aux_max_epochs",
    "aux_lr",
    "aux_ffn_hidden_dim",
    "aux_ffn_num_layers",
    "aux_freeze_epochs",
    "aux_warmup_epochs",
    "aux_val_fraction",
    "batch_size",
    "num_workers",
    "patience",
    "min_delta",
    "refit_on_all",
    "restore_best",
    "accelerator",
)

# columns of the split CSVs this module reads
SMILES_COL = "SMILES"
CANONICAL_COL = "canonical_smiles"
TARGET_COL = "pEC50"

# what a log2FC-pretrained body is asked for, when a configuration does not
# name one outright: ``encoders.log2fc_body_checkpoint(seed=...)`` returning a
# path to a torch-saved mapping with ``hyper_parameters`` and ``state_dict``
# keys for chemprop's BondMessagePassing
ENCODERS_MODULE = "encoders"
ENCODERS_FACTORY = "log2fc_body_checkpoint"


class LeakageError(RuntimeError):
    """A compound in the phase-2 test set also appears in something being trained on."""


class MissingEncoderError(RuntimeError):
    """The log2FC-pretrained body a configuration asks for cannot be obtained."""


@dataclass(frozen=True)
class SplitPaths:
    """Where the challenge's fit and test partitions live.

    Attributes
    ----------
    fit_train : Path
        Rows optimised on.
    fit_val : Path
        Rows early stopping watches. Never optimised on, never predicted for.
    test : Path
        Phase 2, predicted once at the end.
    log2fc : Path
        The single-concentration screen the auxiliary encoder pretrains on.
    """

    fit_train: Path = Path("data/splits/fit_train.csv")
    fit_val: Path = Path("data/splits/fit_val.csv")
    test: Path = Path("data/splits/test_phase2.csv")
    log2fc: Path = LOG2FC_PATH

    def digests(self) -> dict[str, str]:
        """Return each split file's content digest, for the run's cache key."""
        return {
            "fit_train": provenance.digest_file(self.fit_train),
            "fit_val": provenance.digest_file(self.fit_val),
            "test": provenance.digest_file(self.test),
            "log2fc": provenance.digest_file(self.log2fc),
        }


@dataclass(frozen=True)
class RunResult:
    """What one fitted cell hands back.

    Attributes
    ----------
    predictions : pandas.DataFrame
        One row per phase-2 compound: ``SMILES``, ``canonical_smiles``, the
        held-out ``pEC50``, and the model's ``prediction``.
    spec : dict
        The specification stage three names the artifact by, in the shape
        :func:`provenance.block_spec` produces.
    record : dict
        The fit's own account of itself: partition sizes, epochs run, the
        validation loss selected on, the concatenated feature width, and what
        the auxiliary encoder did. Written beside the artifact as the
        provenance record's extra fields.
    """

    predictions: pd.DataFrame
    spec: dict[str, Any] = field(repr=False)
    record: dict[str, Any] = field(repr=False)


def log2fc_body_checkpoint(seed: int, block: str | None = None) -> Path:
    """Ask ``encoders`` for a message-passing body pretrained on log2FC.

    Parameters
    ----------
    seed : int
        The replicate seed, so the pretrained body matches the run it feeds.
    block : str, optional
        Which log2FC encoder to take the body from. None takes the module's
        own default, the encoder trained from scratch.

    Returns
    -------
    Path
        A torch-saved mapping with ``hyper_parameters`` and ``state_dict``
        keys for :class:`chemprop.nn.BondMessagePassing`.

    Raises
    ------
    MissingEncoderError
        If the module or the function is not available. The E4 arm depends on
        work that lands separately, so this path stays importable and fails
        only when it is actually taken.
    """
    try:
        module = __import__(ENCODERS_MODULE)
    except ImportError as err:
        raise MissingEncoderError(
            f"encoder_init='log2fc_checkpoint' needs {ENCODERS_MODULE}.{ENCODERS_FACTORY}, "
            f"which is not importable; pass RunConfig(body_checkpoint=...) to supply the "
            "pretrained body directly"
        ) from err

    factory = getattr(module, ENCODERS_FACTORY, None)
    if factory is None:
        raise MissingEncoderError(
            f"{ENCODERS_MODULE} defines no {ENCODERS_FACTORY}(seed) returning a path to a "
            "log2FC-pretrained BondMessagePassing checkpoint"
        )
    return Path(factory(seed=seed) if block is None else factory(seed=seed, block=block))


def run_cell(
    config: RunConfig,
    seed: int,
    splits: SplitPaths | None = None,
    *,
    aux_cache_dir: Path | None = None,
    force_aux: bool = False,
) -> RunResult:
    """Fit one cell at one seed and return its phase-2 predictions.

    Parameters
    ----------
    config : RunConfig
        The resolved cell, typically from ``RunConfig.from_axes(cell.axes)``.
    seed : int
        Replicate seed. Seeds weight initialisation, batch shuffling, and the
        auxiliary encoder's validation carve-out.
    splits : SplitPaths, optional
        Where the partitions live. Defaults to the repository's own.
    aux_cache_dir : path-like, optional
        Where pretrained auxiliary encoders are cached. Defaults to
        :data:`AUX_CACHE_DIR`.
    force_aux : bool, optional
        Pretrain the auxiliary encoder even when a matching one is cached.

    Returns
    -------
    RunResult
        Per-compound phase-2 predictions, the specification naming them, and
        the fit's record of itself.

    Raises
    ------
    LeakageError
        If any phase-2 compound appears in a training pool.
    MissingEncoderError
        If the configuration asks for a log2FC-pretrained body that cannot be
        obtained.
    """
    splits = splits or SplitPaths()
    started = time.perf_counter()
    lightning.seed_everything(seed, workers=True)

    fit_train = _read_split(splits.fit_train, require_target=True)
    fit_val = _read_split(splits.fit_val, require_target=True)
    test = _read_split(splits.test, require_target=False)
    test_compounds = set(test[CANONICAL_COL])

    # phase 2 must not appear in anything a label is taken from
    _assert_disjoint(fit_train, test_compounds, "fit_train")
    _assert_disjoint(fit_val, test_compounds, "fit_val")

    body = _resolve_body(config, seed)
    aux_body = body if config.encoder_init == CHEMELEON else CHEMELEON

    # the auxiliary arm, when the cell has one: pretrain on log2FC, then build
    # the per-compound vector its embedding and readouts contribute
    encoder: GraphRegressor | None = None
    aux_record: dict[str, Any] | None = None
    features: dict[str, np.ndarray] = {}
    if config.aux_encoder is not None:
        encoder, aux_record = _pretrain_encoder(
            config,
            seed,
            splits,
            aux_body,
            test_compounds,
            cache_dir=aux_cache_dir,
            force=force_aux,
        )
        readouts = load_readouts(splits.log2fc, exclude=test_compounds)
        for name, frame in (("train", fit_train), ("val", fit_val), ("test", test)):
            features[name] = build_features(
                frame[CANONICAL_COL].tolist(),
                readouts,
                encoder,
                use_observed_readout=config.aux_encoder.use_observed_readout,
                use_predicted_readout=config.aux_encoder.use_predicted_readout,
                use_embedding=config.aux_encoder.use_embedding,
                batch_size=config.training.inference_batch_size,
            )

    extra_dim = features["train"].shape[1] if features else 0

    def build_model() -> GraphRegressor:
        """Build a freshly initialised model, so both passes start the same way."""
        return GraphRegressor(
            build_mpnn(
                body,
                ffn_hidden_dim=config.ffn_hidden_dim,
                ffn_num_layers=config.ffn_num_layers,
                message_hidden_dim=config.message_hidden_dim,
                depth=config.depth,
                n_tasks=1,
                extra_input_dim=extra_dim,
            ),
            freeze_epochs=config.freeze_epochs,
            body_lr=config.training.mpnn_lr,
            head_lr=config.training.ffn_lr,
            warmup_epochs=config.training.warmup_epochs,
        )

    # first pass: train on the training partition and let early stopping choose
    # how many epochs this cell and seed want
    model = build_model()
    data = GraphDataModule(
        train=_dataset(fit_train, features.get("train")),
        val=_dataset(fit_val, features.get("val")),
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=seed,
    )
    fit_record = _fit(model, data, config, max_epochs=config.training.max_epochs)

    # second pass: rebuild from the same seed and retrain on both partitions for
    # the epoch count the first pass settled, so the model that is scored has
    # seen the whole fit set rather than 80% of it
    refit_record: dict[str, Any] | None = None
    if config.training.refit_on_all:
        epochs = max(1, int(fit_record["selected_epoch"]) + 1)
        fit_all = pd.concat([fit_train, fit_val], ignore_index=True)
        features_all = np.vstack([features["train"], features["val"]]) if features else None

        lightning.seed_everything(seed, workers=True)
        model = build_model()
        refit_data = GraphDataModule(
            train=_dataset(fit_all, features_all),
            val=None,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            seed=seed,
        )
        refit_record = _fit(
            model,
            refit_data,
            config,
            max_epochs=config.training.max_epochs,
            stop_after=epochs,
            validate=False,
        )
        refit_record["n_train"] = len(fit_all)
        refit_record["epochs_requested"] = epochs

    predictions = model.predict(
        test[CANONICAL_COL].tolist(),
        batch_size=config.training.inference_batch_size,
        extra=features.get("test"),
    )

    result = test.loc[:, [SMILES_COL, CANONICAL_COL, TARGET_COL]].assign(
        prediction=predictions[:, 0].astype(float)
    )

    spec = provenance.block_spec(
        PRODUCER,
        PRODUCER_VERSION,
        params={**config.as_dict(), "seed": seed},
        splits=splits.digests(),
    )
    record = {
        "refit_on_all": refit_record is not None,
        "refit": refit_record,
        "n_train": len(fit_train),
        "n_val": len(fit_val),
        "n_test": len(test),
        "extra_feature_dim": extra_dim,
        "embedding_dim": encoder.embedding_dim if encoder is not None else None,
        "body": str(body),
        "main": fit_record,
        "auxiliary": aux_record,
        "wall_clock_s": round(time.perf_counter() - started, 3),
    }
    return RunResult(predictions=result, spec=spec, record=record)


def _resolve_body(config: RunConfig, seed: int) -> str | Path:
    """Return what the main model's message-passing body is initialised from."""
    if config.body_checkpoint is not None:
        return config.body_checkpoint
    if config.encoder_init == "log2fc_checkpoint":
        return log2fc_body_checkpoint(seed)
    if config.encoder_init == "chemeleon_log2fc_checkpoint":
        return log2fc_body_checkpoint(seed, block="chemeleon_log2fc_embedding")
    if config.encoder_init == "scratch":
        # a body with no checkpoint behind it, built at the configured width
        return RANDOM_BODY
    return CHEMELEON


def _read_split(path: Path, *, require_target: bool) -> pd.DataFrame:
    """Read one split CSV, dropping label-less rows only where a label is needed."""
    frame = pd.read_csv(path)
    if not require_target:
        return frame.reset_index(drop=True)

    present = frame[TARGET_COL].notna()
    if not present.all():
        logger.info("%s: %d rows without a pEC50 dropped", path.name, int((~present).sum()))
    return frame.loc[present].reset_index(drop=True)


def _assert_disjoint(frame: pd.DataFrame, test_compounds: set[str], label: str) -> None:
    """Raise if a training partition shares any canonical structure with phase 2."""
    overlap = set(frame[CANONICAL_COL]) & test_compounds
    if overlap:
        raise LeakageError(
            f"{label}: {len(overlap)} compound(s) also in phase 2, e.g. {sorted(overlap)[:3]}"
        )


def _dataset(frame: pd.DataFrame, extra: np.ndarray | None) -> GraphDataset:
    """Wrap a split frame as a dataset, its single target observed on every row."""
    targets = frame[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    return GraphDataset(frame[CANONICAL_COL].tolist(), targets, np.ones_like(targets), extra=extra)


def aux_spec(
    config: RunConfig, seed: int, splits: SplitPaths, body: str | Path, tasks: Sequence[str]
) -> dict[str, Any]:
    """Return the specification identifying one pretrained auxiliary encoder.

    It names the screen and the test split it excludes, the seed, the task
    columns, the body it starts from, and the training settings the auxiliary
    fit reads. It deliberately names nothing about the main model, because the
    encoder does not depend on it: that is what lets the freeze and width cells
    of figure 1 share one.

    The gradient clip is the exception, and it is here because the auxiliary
    fit really does use the cell's clip value rather than one of its own.
    """
    return provenance.block_spec(
        "aux_encoder",
        AUX_VERSION,
        params={
            "seed": seed,
            "tasks": list(tasks),
            "body": str(body),
            "message_hidden_dim": config.message_hidden_dim,
            "depth": config.depth,
            "gradient_clip_val": config.gradient_clip_val,
            **{field: getattr(config.training, field) for field in AUX_TRAINING_FIELDS},
        },
        inputs=[],
        screen=provenance.digest_file(splits.log2fc),
        excluded_from=provenance.digest_file(splits.test),
    )


def _pretrain_encoder(
    config: RunConfig,
    seed: int,
    splits: SplitPaths,
    body: str | Path,
    exclude: set[str],
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> tuple[GraphRegressor, dict[str, Any]]:
    """Pretrain the auxiliary encoder on log2FC, holding out a seeded fraction.

    The result is cached by :func:`aux_spec`, so a cell whose auxiliary arm
    matches one already trained loads it instead of repeating the fit.
    """
    aux = config.aux_encoder
    assert aux is not None  # noqa: S101 - the caller checks; this narrows the type
    tasks = TASKS
    table = load_readouts(splits.log2fc, exclude=exclude)

    smiles = table.index.tolist()
    values = table.to_numpy(dtype=np.float32)
    mask = table.notna().to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0)

    # seeded carve-out of the screen; no pEC50 label is involved either side
    order = np.random.default_rng(seed).permutation(len(smiles))
    n_val = max(1, int(len(smiles) * config.training.aux_val_fraction))
    val_rows, train_rows = order[:n_val], order[n_val:]

    def build_encoder() -> GraphRegressor:
        """Build a freshly initialised auxiliary encoder."""
        return GraphRegressor(
            build_mpnn(
                body,
                ffn_hidden_dim=config.training.aux_ffn_hidden_dim,
                ffn_num_layers=config.training.aux_ffn_num_layers,
                message_hidden_dim=config.message_hidden_dim,
                depth=config.depth,
                n_tasks=len(tasks),
            ),
            freeze_epochs=config.training.aux_freeze_epochs,
            body_lr=config.training.aux_lr,
            head_lr=config.training.aux_lr,
            warmup_epochs=config.training.aux_warmup_epochs,
            prefix="aux_",
        )

    def partition(rows: np.ndarray) -> GraphDataset:
        return GraphDataset([smiles[i] for i in rows], values[rows], mask[rows])

    # a cached encoder is loaded into a freshly built one rather than refitted;
    # the record travels in the artifact's own provenance file
    artifact = provenance.Artifact(
        root=cache_dir if cache_dir is not None else AUX_CACHE_DIR,
        spec=aux_spec(config, seed, splits, body, tasks),
        suffix=".pt",
    )
    if artifact.is_cached and not force:
        logger.info("auxiliary encoder: cached (%s)", artifact.key)
        return _load_encoder(artifact, build_encoder)

    with provenance.timed() as elapsed:
        # first pass: hold out a slice of the screen to choose an epoch count
        encoder = build_encoder()
        data = GraphDataModule(
            train=partition(train_rows),
            val=partition(val_rows),
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            seed=seed,
        )
        fit_record = _fit(encoder, data, config, max_epochs=config.training.aux_max_epochs)

        # second pass: the held-out slice has done its job, so retrain on the whole
        # screen for the epochs it settled
        refit_record: dict[str, Any] | None = None
        if config.training.refit_on_all:
            epochs = max(1, int(fit_record["selected_epoch"]) + 1)
            lightning.seed_everything(seed, workers=True)
            encoder = build_encoder()
            refit_data = GraphDataModule(
                train=partition(order),
                val=None,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                seed=seed,
            )
            refit_record = _fit(
                encoder,
                refit_data,
                config,
                max_epochs=config.training.aux_max_epochs,
                stop_after=epochs,
                validate=False,
            )
            refit_record["n_train"] = len(order)
            refit_record["epochs_requested"] = epochs

    record = {
        "wall_clock_s": elapsed(),
        "tasks": list(tasks),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_excluded_as_test": len(exclude & set(smiles)),
        "refit": refit_record,
        **fit_record,
    }
    _save_encoder(artifact, encoder, record)
    return encoder, {**record, "cache_key": artifact.key, "loaded_from_cache": False}


def _save_encoder(
    artifact: provenance.Artifact, encoder: GraphRegressor, record: dict[str, Any]
) -> None:
    """Cache a pretrained encoder's weights, with its fit recorded beside them."""
    with provenance.atomic(artifact.path) as partial:
        torch.save(
            {name: tensor.detach().cpu() for name, tensor in encoder.state_dict().items()},
            partial,
        )
    artifact.write_record(
        auxiliary=record,
        body_is_frozen=encoder.body_is_frozen,
        wall_clock_s=record.get("wall_clock_s"),
    )


def _load_encoder(
    artifact: provenance.Artifact, build: Callable[[], GraphRegressor]
) -> tuple[GraphRegressor, dict[str, Any]]:
    """Rebuild a cached encoder and restore the state its fit left it in."""
    record = artifact.read_record()
    encoder = build()
    encoder.load_state_dict(torch.load(artifact.path, weights_only=True, map_location="cpu"))

    # freezing only gates which parameters take gradients, so it cannot change
    # the embeddings this encoder goes on to produce; it is restored anyway so a
    # loaded encoder is indistinguishable from a freshly trained one
    if not record.get("body_is_frozen", True):
        encoder._unfreeze_body()  # noqa: SLF001

    return encoder, {**record["auxiliary"], "cache_key": artifact.key, "loaded_from_cache": True}


class _StopAfter(lightning.Callback):
    """End training after a fixed number of epochs, leaving the schedule alone.

    ``max_epochs`` is what the noam schedule calibrates its decay against, so
    it has to stay at the budget the first pass ran under. This stops the
    second pass at the epoch that pass chose without shortening the schedule
    underneath it.
    """

    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: Any) -> None:
        """Ask the trainer to stop once the chosen count is reached."""
        if trainer.current_epoch + 1 >= self.epochs:
            trainer.should_stop = True


def _fit(
    model: GraphRegressor,
    data: GraphDataModule,
    config: RunConfig,
    *,
    max_epochs: int,
    stop_after: int | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Run one Lightning fit, optionally rewinding to the best validation epoch.

    With ``validate`` false there is no validation partition to watch, so
    neither early stopping nor checkpoint selection applies. That is the second
    pass of a refit, whose epoch count was already chosen by the first and is
    passed as ``stop_after``: ``max_epochs`` stays at the first pass's budget so
    the learning-rate schedule is the one that count was chosen under, and the
    run is stopped at the count instead of having the schedule compressed into
    it.
    """
    callbacks: list[Any] = []
    if stop_after is not None:
        callbacks.append(_StopAfter(stop_after))
    if validate:
        callbacks.append(
            EarlyStopping(
                monitor=model.val_metric,
                patience=config.training.patience,
                min_delta=config.training.min_delta,
                mode="min",
            )
        )

    with tempfile.TemporaryDirectory(prefix="concat_arch_") as scratch:
        checkpointer = (
            ModelCheckpoint(dirpath=scratch, monitor=model.val_metric, mode="min", save_top_k=1)
            if validate
            else None
        )
        if checkpointer is not None:
            callbacks.append(checkpointer)

        trainer = lightning.Trainer(
            max_epochs=max_epochs,
            accelerator=config.training.accelerator,
            devices=1,
            gradient_clip_val=config.gradient_clip_val,
            gradient_clip_algorithm="norm",
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=validate,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
            num_sanity_val_steps=0 if not validate else 2,
        )
        trainer.fit(model, datamodule=data)

        best_score = None if checkpointer is None else checkpointer.best_model_score
        best_epoch = trainer.current_epoch
        restored = False
        if (
            checkpointer is not None
            and config.training.restore_best
            and checkpointer.best_model_path
        ):
            state = cast(
                dict[str, Any],
                torch.load(checkpointer.best_model_path, weights_only=True, map_location="cpu"),
            )
            model.load_state_dict(state["state_dict"])
            best_epoch = int(state.get("epoch", best_epoch))
            restored = True

    return {
        "epochs_run": trainer.current_epoch,
        "selected_epoch": best_epoch,
        "best_val_loss": None if best_score is None else float(best_score),
        "restored_best": restored,
        "body_unfroze": not model.body_is_frozen,
    }
