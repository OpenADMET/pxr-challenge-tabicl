"""Generate the moal plan config sweep grid over freeze_epochs, ffn_hidden_dim, and gradient_clip_val.

Reads moal_aux_plan_config.yaml as the base template and writes one config per
grid combination to configs/, named freeze{F}_hd{H}_clip{C}.yaml. Each config's
data.output_dir is set to results/<same-stem> so the sweep runner can tell
finished combinations apart by output directory alone.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

BASE_CONFIG = Path("moal_aux_plan_config.yaml")
CONFIGS_DIR = Path("configs")

FREEZE_EPOCHS = [0, 1, 2]
FFN_HIDDEN_DIMS = [512, 1024]
GRADIENT_CLIP_VALS = [None, 1.0, 5.0]


def clip_label(clip_val: float | None) -> str:
    """Render a gradient_clip_val as a filename-safe token."""
    return "off" if clip_val is None else str(clip_val)


def main() -> None:
    """Write one YAML config per (freeze_epochs, ffn_hidden_dim, gradient_clip_val) combination."""
    base = yaml.safe_load(BASE_CONFIG.read_text())
    CONFIGS_DIR.mkdir(exist_ok=True)

    for freeze_epochs in FREEZE_EPOCHS:
        for hidden_dim in FFN_HIDDEN_DIMS:
            for clip_val in GRADIENT_CLIP_VALS:
                stem = f"freeze{freeze_epochs}_hd{hidden_dim}_clip{clip_label(clip_val)}"
                cfg = copy.deepcopy(base)

                cfg["model"]["freeze_epochs"] = freeze_epochs
                cfg["model"]["ffn_hidden_dim"] = hidden_dim
                cfg["auxiliary_model"]["freeze_epochs"] = freeze_epochs
                cfg["auxiliary_model"]["ffn_hidden_dim"] = hidden_dim
                cfg["auxiliary_model"]["use_observed_readout"] = False

                cfg["trainer"]["gradient_clip_val"] = clip_val
                cfg["auxiliary_trainer"]["gradient_clip_val"] = clip_val
                cfg["trainer"]["early_stopping_patience"] = 10
                cfg["auxiliary_trainer"]["early_stopping_patience"] = 10

                cfg["data"]["output_dir"] = f"results/{stem}"

                out_path = CONFIGS_DIR / f"{stem}.yaml"
                out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    n = len(FREEZE_EPOCHS) * len(FFN_HIDDEN_DIMS) * len(GRADIENT_CLIP_VALS)
    print(f"Wrote {n} configs to {CONFIGS_DIR}/")


if __name__ == "__main__":
    main()
