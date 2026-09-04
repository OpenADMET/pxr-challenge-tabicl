"""Fetch the PXR challenge data and build the challenge's own evaluation split.

The challenge scores a model fit on the dose-response training set pooled with
phase 1 against phase 2 alone. This module pulls the raw CSVs from the public
Hugging Face dataset and writes the resource files the anvil recipes point at:
the pooled fit set, a seeded train/validation partition of it for early
stopping, and the phase-2 test set. A leakage guard asserts the fit and test
compound sets are disjoint by canonical structure before anything is written.

The train/test partition is the challenge's official split, reused verbatim so
results stay comparable to the leaderboard; only the validation carve-out of
the fit set is a seeded random split, and it exists solely for early stopping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# public Hugging Face dataset holding the challenge splits
HF_REPO_ID = "openadmet/pxr-challenge-train-test"
HF_REPO_TYPE = "dataset"

# raw file names within the dataset
TRAIN_FILE = "pxr-challenge_TRAIN.csv"
PHASE1_FILE = "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"
PHASE2_FILE = "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv"
LOG2FC_FILE = "pxr-challenge_single_concentration_TRAIN.csv"
RAW_FILES = (TRAIN_FILE, PHASE1_FILE, PHASE2_FILE, LOG2FC_FILE)

# on-disk locations, relative to the repo root
RAW_DIR = Path("data/raw")
SPLIT_DIR = Path("data/splits")

# columns shared across the dose-response CSVs, plus the two we add
SMILES_COL = "SMILES"
TARGET_COL = "pEC50"
SOURCE_COL = "source"
CANONICAL_COL = "canonical_smiles"

# seeded validation carve-out of the pooled fit set, for early stopping only
VAL_FRACTION = 0.2
SPLIT_SEED = 42

# physically plausible pEC50 window; a value outside it stops the pipeline
PEC50_MIN = 0.0
PEC50_MAX = 14.0

# per-molecule parse failures are counted in _prepare via None returns, so the
# per-row stderr spam is redundant once accounting is in place; DisableLog is a
# real RDKit call the bundled type stubs omit
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


@dataclass(frozen=True)
class SplitArtifacts:
    """Written resource-CSV paths and their row counts."""

    fit_all: Path
    fit_train: Path
    fit_val: Path
    test: Path
    n_fit: int
    n_fit_train: int
    n_fit_val: int
    n_test: int


def fetch_raw(dest_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Download the challenge CSVs from Hugging Face into a local directory.

    Parameters
    ----------
    dest_dir : path-like
        Directory to place the raw files in; created if absent. Defaults to
        ``data/raw``.

    Returns
    -------
    dict of str to Path
        Each raw file name mapped to its downloaded local path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in RAW_FILES:
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=name,
            repo_type=HF_REPO_TYPE,
            local_dir=str(dest_dir),
        )
        paths[name] = Path(local)
        logger.info("fetched %s", name)
    return paths


def build_split(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = SPLIT_DIR,
    *,
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> SplitArtifacts:
    """Write the challenge fit/test resource CSVs from the raw downloads.

    The fit set is the dose-response training file pooled with phase 1; the
    test set is phase 2. A seeded fraction of the fit set is held out as a
    validation partition for early stopping. Before anything is written, the
    fit and test compound sets are checked disjoint by canonical structure.

    Parameters
    ----------
    raw_dir : path-like
        Directory holding the files ``fetch_raw`` wrote. Defaults to
        ``data/raw``.
    out_dir : path-like
        Directory to write the resource CSVs into. Defaults to ``data/splits``.
    val_fraction : float, optional
        Fraction of the fit set held out for validation. Defaults to 0.2.
    seed : int, optional
        Seed for the validation partition. Defaults to 42.

    Returns
    -------
    SplitArtifacts
        The written paths and their row counts.

    Raises
    ------
    ValueError
        If any fit compound also appears in the phase-2 test set (leakage), or
        if a pEC50 value falls outside the plausible window.
    """
    # pool the training file with phase 1 for fitting; phase 2 is the test set
    train = _load_potency_csv(raw_dir / TRAIN_FILE, "train")
    phase1 = _load_potency_csv(raw_dir / PHASE1_FILE, "phase1")
    phase2 = _load_potency_csv(raw_dir / PHASE2_FILE, "phase2")

    fit = _prepare(pd.concat([train, phase1], ignore_index=True), "fit")
    test = _prepare(phase2, "test")

    # leakage guard: no fit compound may reappear in the test set
    _assert_disjoint(fit, test)

    # seeded validation carve-out of the fit set, for early stopping only
    val = fit.sample(frac=val_fraction, random_state=seed)
    fit_train = fit.drop(index=val.index).reset_index(drop=True)
    fit_val = val.reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = SplitArtifacts(
        fit_all=_write(fit, out_dir / "fit_all.csv"),
        fit_train=_write(fit_train, out_dir / "fit_train.csv"),
        fit_val=_write(fit_val, out_dir / "fit_val.csv"),
        test=_write(test, out_dir / "test_phase2.csv"),
        n_fit=len(fit),
        n_fit_train=len(fit_train),
        n_fit_val=len(fit_val),
        n_test=len(test),
    )
    logger.info(
        "fit=%d (train=%d, val=%d), test=%d",
        artifacts.n_fit,
        artifacts.n_fit_train,
        artifacts.n_fit_val,
        artifacts.n_test,
    )
    return artifacts


def canonical_smiles(smiles: str) -> str | None:
    """Return the canonical SMILES of the largest fragment, or None if unparsable.

    This is the identity key the split files are built on and joined by, so
    anything that needs to match a compound to a split row must come through
    here rather than canonicalize independently.

    Parameters
    ----------
    smiles : str
        A SMILES string, not necessarily canonical.

    Returns
    -------
    str or None
        The canonical parent SMILES, or None if RDKit cannot parse the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    parent = _LARGEST_FRAGMENT.choose(mol)
    return Chem.MolToSmiles(parent)


def _load_potency_csv(path: Path, source: str) -> pd.DataFrame:
    """Read SMILES and pEC50 from one challenge CSV, tagged with its source."""
    frame = pd.read_csv(
        path,
        usecols=[SMILES_COL, TARGET_COL],
        dtype={SMILES_COL: str, TARGET_COL: float},
    )
    return frame.assign(**{SOURCE_COL: source})


def _prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Add canonical SMILES, drop unparsable rows, and range-check pEC50."""
    # canonicalize; None marks a structure RDKit could not parse
    canonical = frame[SMILES_COL].map(canonical_smiles)
    n_bad = int(canonical.isna().sum())
    if n_bad:
        logger.warning("%s: dropping %d unparsable SMILES", label, n_bad)
    prepared = (
        frame.assign(**{CANONICAL_COL: canonical}).loc[canonical.notna()].reset_index(drop=True)
    )

    # rows without a pEC50 are kept; anvil drops them train-only at fit time
    present = prepared[TARGET_COL].notna()
    n_missing = int((~present).sum())
    if n_missing:
        logger.info("%s: %d rows have no pEC50 (kept; dropped at fit)", label, n_missing)

    # plausibility: a present pEC50 outside the physical window is a data defect
    values = prepared.loc[present, TARGET_COL]
    out_of_range = values[(values < PEC50_MIN) | (values > PEC50_MAX)]
    if not out_of_range.empty:
        raise ValueError(
            f"{label}: {len(out_of_range)} pEC50 values outside "
            f"[{PEC50_MIN}, {PEC50_MAX}], e.g. {out_of_range.head().tolist()}"
        )
    return prepared


def _assert_disjoint(fit: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any canonical structure appears in both the fit and test sets."""
    overlap = set(fit[CANONICAL_COL]) & set(test[CANONICAL_COL])
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"leakage: {len(overlap)} compounds in both fit and test, e.g. {examples}")


def _write(frame: pd.DataFrame, path: Path) -> Path:
    """Write a resource frame to CSV and return its path."""
    frame.to_csv(path, index=False)
    return path
