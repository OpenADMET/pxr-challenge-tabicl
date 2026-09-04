"""The concatenation architecture: a pEC50 D-MPNN augmented by a log2FC encoder.

A main ChemProp D-MPNN predicts pEC50. Beside it, an auxiliary D-MPNN is
pretrained on the single-concentration log2FC screen, and what it learned is
concatenated into the main model's predictor input: always its pooled
structural embedding, and optionally a readout of log2FC values, either the
ones actually measured for that compound or the ones the encoder predicts for
it. The E4 arm spends the same log2FC signal differently, pretraining the main
model's own body on it in place of the CheMeleon foundation init, and so
carries no auxiliary encoder at all.

The entry point is :func:`run_cell`, which takes a resolved configuration, a
seed, and the split paths, fits, and returns phase-2 predictions with the
fit's record of itself. It writes nothing.

Examples
--------
>>> from concat_arch import RunConfig, run_cell  # doctest: +SKIP
>>> config = RunConfig.from_axes(cell.axes)  # doctest: +SKIP
>>> result = run_cell(config, seed=0)  # doctest: +SKIP
>>> result.predictions.columns.tolist()  # doctest: +SKIP
['SMILES', 'canonical_smiles', 'pEC50', 'prediction']
"""

from .backbone import BackboneError, build_mpnn, write_body_checkpoint
from .concat_features import build_features, feature_dim
from .config import AuxEncoderConfig, ConfigError, RunConfig, TrainingConfig
from .module import GraphRegressor, masked_mse_loss
from .readouts import load_readouts, task_columns
from .run import (
    PRODUCER,
    PRODUCER_VERSION,
    LeakageError,
    MissingEncoderError,
    RunResult,
    SplitPaths,
    run_cell,
)

__all__ = [
    "PRODUCER",
    "PRODUCER_VERSION",
    "AuxEncoderConfig",
    "BackboneError",
    "ConfigError",
    "GraphRegressor",
    "LeakageError",
    "MissingEncoderError",
    "RunConfig",
    "RunResult",
    "SplitPaths",
    "TrainingConfig",
    "build_features",
    "build_mpnn",
    "feature_dim",
    "load_readouts",
    "masked_mse_loss",
    "run_cell",
    "task_columns",
    "write_body_checkpoint",
]
