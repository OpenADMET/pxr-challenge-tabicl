"""Precompute RDKit and Mordred 2D descriptors once for every compound in the campaign.

Feature-concatenation experiments (TabPFN + CheMeleon + RDKit + Mordred) need
the same descriptor block on every run. RDKit's 217 descriptors are cheap,
but Mordred's ~1600 2D descriptors are not: recomputing them per-experiment
would dominate wall time. This script computes both once for every unique
canonical SMILES in `data/moal_plan_state.csv` (which already covers every
split: DRC training rows, PS-only rows, and the 513 blind targets) and caches
the result to a single parquet file keyed by canonical SMILES.

Run with:
    python precompute_descriptors.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors

from moal.preprocessing import SMILESPreprocessor

logger = logging.getLogger(__name__)

_INPUT_CSV = Path("data/moal_plan_state.csv")
_OUTPUT_PARQUET = Path("data/descriptor_cache.parquet")


def _unique_canonical_smiles(preprocessor: SMILESPreprocessor) -> list[str]:
    """Canonicalize every SMILES in the campaign state CSV and deduplicate."""
    raw_smiles = pd.read_csv(_INPUT_CSV)["smiles"].astype(str)
    canonical = raw_smiles.map(preprocessor.canonicalize)
    n_rejected = int(canonical.isna().sum())
    if n_rejected:
        logger.warning("Rejected %d SMILES that failed canonicalization.", n_rejected)
    return sorted(canonical.dropna().unique().tolist())


def main() -> None:
    """Compute and cache RDKit + Mordred 2D descriptors for every compound."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    preprocessor = SMILESPreprocessor()
    canonical_smiles = _unique_canonical_smiles(preprocessor)
    logger.info("Featurizing %d unique compound(s)", len(canonical_smiles))

    mols = [Chem.MolFromSmiles(smi) for smi in canonical_smiles]

    logger.info("Computing RDKit descriptors")
    rdkit_rows = [dict(Descriptors.CalcMolDescriptors(mol)) for mol in mols]
    rdkit_df = pd.DataFrame(rdkit_rows).add_prefix("rdkit_")

    logger.info("Computing Mordred descriptors")
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_df = calc.pandas(mols, nproc=os.cpu_count(), quiet=True).add_prefix("mordred_")
    mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")

    cache = pd.concat(
        [pd.DataFrame({"canonical_smiles": canonical_smiles}), rdkit_df, mordred_df],
        axis=1,
    )
    _OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(_OUTPUT_PARQUET, index=False)
    logger.info("Wrote %s (%d compounds, %d descriptor columns)", _OUTPUT_PARQUET, *cache.shape)


if __name__ == "__main__":
    main()
