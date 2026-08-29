"""Build a canonical 80/20 train/validation split of the log2FC pretraining data.

`moal_plan_state.csv` has no train/val split of its own, and the E4 log2FC
pretraining recipe (e4_log2fc_pretrain/log2fc_pretrain.yaml) needs a held-out
val set distinct from train so early stopping has a real signal. This script
carves a fixed random 80/20 split out of moal_plan_state.csv, mirroring
make_train_val_split.py's approach for the pEC50 TRAIN set.

Run with `python make_log2fc_split.py` from the pxr-challenge repo root.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "moal_plan_state.csv"
TRAIN_OUTPUT_PATH = DATA_DIR / "moal_plan_state_split80.csv"
VAL_OUTPUT_PATH = DATA_DIR / "moal_plan_state_split20.csv"

RANDOM_SEED = 42
VAL_FRACTION = 0.2


def main() -> None:
    """Split moal_plan_state.csv 80/20 and write the two output files."""
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(INPUT_PATH)
    val_df = df.sample(frac=VAL_FRACTION, random_state=RANDOM_SEED)
    train_df = df.drop(val_df.index)

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    val_df.to_csv(VAL_OUTPUT_PATH, index=False)

    logger.info(
        "Split %d rows into %d train / %d val (seed=%d) -> %s, %s",
        len(df),
        len(train_df),
        len(val_df),
        RANDOM_SEED,
        TRAIN_OUTPUT_PATH,
        VAL_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
