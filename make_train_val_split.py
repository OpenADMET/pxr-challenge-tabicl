"""Build a canonical 80/20 train/validation split of the PXR TRAIN set.

`pxr-challenge_TRAIN.csv`'s own `Split` column is uninformative (every row is
`Train`), so baseline configs (e.g. `chemeleon_pxr_st.yaml`) have no fixed
validation split to select on, making runs hard to compare against each
other or against `moal`. This script carves a fixed random 80/20 split out of
`pxr-challenge_TRAIN.csv` and writes it as two new files so any config can
point at the same split.

Run with `python make_train_val_split.py` from the pxr-challenge repo root.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "pxr-challenge_TRAIN.csv"
TRAIN_OUTPUT_PATH = DATA_DIR / "pxr-challenge_TRAIN_split80.csv"
VAL_OUTPUT_PATH = DATA_DIR / "pxr-challenge_TRAIN_split20.csv"

RANDOM_SEED = 42
VAL_FRACTION = 0.2


def main() -> None:
    """Split `pxr-challenge_TRAIN.csv` 80/20 and write the two output files."""
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
