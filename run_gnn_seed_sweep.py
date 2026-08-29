"""Rerun the CheMeleon non-tabular (fine-tuned GNN) config grid across 5 seeds.

The freeze*/e4 configs use `data.plan.*`, so they run through `moal plan`,
not `moal simulate` (the latter requires `data.simulate.*` and errors out
otherwise). `moal plan` already calls `L.seed_everything(cfg.seed)` before
building the model, so weight init and data splitting are fully controlled
by `seed` in the config; `split_seed` (consumed by `to_datamodule_kwargs()`)
governs the train/val split specifically. This script copies each freeze*/e4
config, overrides both to a replicate seed, and runs `moal plan` against the
copy, writing to `results/<config-stem>_seed{N}/`, so the fine-tuned-GNN
baseline gets the same 5-seed MAE distribution treatment as the
tabular-encoder recipes in `run_seed_sweep.sh`. Both `trainer` and
`auxiliary_trainer` early_stopping_patience are also overridden to 5
(from the configs' default of 10) to cut per-run wall-clock time.

Run with:
    python run_gnn_seed_sweep.py
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SEEDS = [0, 1, 2, 3, 4]

_CONFIG_NAMES = [
    f"freeze{f}_hd{hd}_clip{clip}.yaml"
    for f in (0, 1, 2)
    for hd in (512, 1024)
    for clip in ("1.0", "5.0", "off")
] + [
    "e4_finetune.yaml",
    "e4_finetune_drc_only.yaml",
    "e4_frozen.yaml",
    "e4_frozen_drc_only.yaml",
]


def _seed_variant_config(config_path: Path, seed: int, tmp_dir: Path) -> Path:
    """Write a copy of `config_path` with `seed` and every `split_seed` overridden."""
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = seed
    for block in ("trainer", "auxiliary_trainer"):
        if block in cfg and isinstance(cfg[block], dict) and "split_seed" in cfg[block]:
            cfg[block]["split_seed"] = seed
        if block in cfg and isinstance(cfg[block], dict) and "early_stopping_patience" in cfg[block]:
            cfg[block]["early_stopping_patience"] = 5
    if "data" in cfg and isinstance(cfg["data"], dict) and "output_dir" in cfg["data"]:
        cfg["data"]["output_dir"] = f"{cfg['data']['output_dir']}_seed{seed}"

    variant_path = tmp_dir / f"{config_path.stem}_seed{seed}.yaml"
    with variant_path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return variant_path


def main() -> None:
    """Generate seed-variant configs and run `moal simulate` against each."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    configs_dir = Path("configs")
    results_dir = Path("results")

    with tempfile.TemporaryDirectory(prefix="gnn_seed_sweep_") as tmp:
        tmp_dir = Path(tmp)
        for name in _CONFIG_NAMES:
            config_path = configs_dir / name
            if not config_path.exists():
                logger.warning("Skipping missing config %s", config_path)
                continue
            for seed in _SEEDS:
                out_dir = results_dir / f"{config_path.stem}_seed{seed}"
                if (out_dir / "pxr_aux_predictions.csv").exists():
                    logger.info("Skipping already-completed %s", out_dir)
                    continue
                variant = _seed_variant_config(config_path, seed, tmp_dir)
                logger.info("moal plan %s -> %s", variant, out_dir)
                subprocess.run(
                    ["moal", "plan", "--config", str(variant), "--output-dir", str(out_dir)],
                    check=True,
                )


if __name__ == "__main__":
    main()
