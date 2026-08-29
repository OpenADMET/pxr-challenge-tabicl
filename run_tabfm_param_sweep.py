"""Sweep TabFM's n_estimators against the fixed max_num_rows=500 row cap.

Per `TabFMRegressor.__init__`, `n_estimators` (default 32, ensemble members
over row/feature subsamples) is the main lever for recovering accuracy lost
to the 500-row in-context cap (the full 4,139-row training set OOMs this
GPU's attention pass; 1000 and 2000 both OOM'd too in prior testing, so rows
stays fixed at 500 here). `max_num_features` (500 default) doesn't bind for
the embed+readout+Mordred-PCA128 recipe (386 features), so it is left alone.
`batch_size` is a memory/speed knob, not an accuracy one, so it's untouched.

Reuses the same seed-0..4 auxiliary-embedding caches `run_seed_sweep.sh`
already produced, so results sit on the same 5-seed replicate basis as every
other tabular recipe in the study.

Run with:
    python run_tabfm_param_sweep.py
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_SEEDS = [0, 1, 2, 3, 4]
_N_ESTIMATORS = [16, 32, 64]


def main() -> None:
    """Run tabfm at each n_estimators value against every seed's cache."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = Path("configs/tabpfn_small_embed.yaml")
    for n_estimators in _N_ESTIMATORS:
        for seed in _SEEDS:
            cache = Path(f"data/auxiliary_embedding_cache_seed{seed}.parquet")
            out_dir = Path(f"results/tabfm_n{n_estimators}_embed_readout_mordred_pca128_seed{seed}")
            if (out_dir / "pxr_tabpfn_predictions.csv").exists():
                logger.info("Skipping already-completed %s", out_dir)
                continue
            logger.info("tabfm n_estimators=%d seed=%d -> %s", n_estimators, seed, out_dir)
            subprocess.run(
                [
                    "python",
                    "tabpfn_concat_features.py",
                    str(config),
                    "--embedding-cache",
                    str(cache),
                    "--descriptor-source",
                    "mordred",
                    "--pca-components",
                    "128",
                    "--regressor",
                    "tabfm",
                    "--tabfm-n-estimators",
                    str(n_estimators),
                    "--output-dir",
                    str(out_dir),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
