"""Build a DRC-only variant of the moal plan state CSV.

`moal plan`'s plain (non-auxiliary-model) path trains the main model on every
labeled record via the Tobit loss, mixing DRC ('==') and PS ('<'/'>=')
fidelities. To test training on DRC records alone (matching what the E3
concatenation architecture's main model already does, so the comparison is
apples to apples), this drops PS rows from moal_plan_state.csv while keeping
DRC rows and the blank/unqueried inference-target rows untouched.

Run with `python make_drc_only_state.py` from the pxr-challenge repo root.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "moal_plan_state.csv"
OUTPUT_PATH = DATA_DIR / "moal_plan_state_drc_only.csv"

PS_RELATIONS = {"<", ">="}


def main() -> None:
    """Drop PS-relation rows from moal_plan_state.csv and write the DRC-only file."""
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(INPUT_PATH)
    is_ps = df["relation"].isin(PS_RELATIONS)
    drc_only = df[~is_ps]
    drc_only.to_csv(OUTPUT_PATH, index=False)

    logger.info(
        "Dropped %d PS row(s) from %d total, keeping %d (DRC + unqueried) -> %s",
        int(is_ps.sum()),
        len(df),
        len(drc_only),
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
