"""Generate the log2FC feature-source sweep on top of the best E3 hyperparameters.

The 18-config grid (generate_sweep_configs.py) fixed auxiliary_model.use_observed_readout
to False throughout, so the concatenation architecture always fell back to
embedding-only. This sweeps the feature-source axis that grid never touched,
holding freeze_epochs=2, ffn_hidden_dim=512, gradient_clip_val=1.0 fixed at
freeze2_hd512_clip1.0's (the earlier grid's best config) values:

- embedding_only: unchanged from freeze2_hd512_clip1.0 (use_observed_readout: false,
  use_predicted_readout: false); not regenerated here, already have that result.
- observed_4task: raw observed log2FC, all 4 concentration columns (including
  the empty and near-empty ones), use_observed_readout: true. Already run.
- observed_2task: raw observed log2FC, trimmed to the two best-populated
  concentrations, matching e4_log2fc_pretrain/log2fc_pretrain.yaml's task set
  and the N283T report's own recipe. Already run.
- predicted_native_2task: the auxiliary encoder's own predicted readout
  (AuxiliaryEncoderModule.predict_smiles, moal/auxiliary_encoder.py),
  concatenated for every compound via the native use_predicted_readout flag
  rather than a precomputed CSV column. Not yet run.
- observed_and_predicted_2task: both use_observed_readout and
  use_predicted_readout enabled together, so a compound's real observation
  (when it has one) and the encoder's own prediction are both concatenated
  alongside the embedding. Not yet run.

Both predicted variants use the trimmed 2-task log2fc_columns (matching
observed_2task and the E4 pretraining recipe), not all 4 — the sparse
log2fc_9.901e-05 task and the entirely-empty log2fc_9.803e-07 task are
dropped for the same reason they were dropped from e4_log2fc_pretrain.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

BASE_CONFIG = Path("configs/freeze2_hd512_clip1.0.yaml")
CONFIGS_DIR = Path("configs")

ALL_4_LOG2FC_COLUMNS = [
    "log2fc_9.803e-07",
    "log2fc_8.251e-06",
    "log2fc_3.300e-05",
    "log2fc_9.901e-05",
]
TRIMMED_2_LOG2FC_COLUMNS = ["log2fc_8.251e-06", "log2fc_3.300e-05"]

VARIANTS = {
    "observed_4task": {
        "log2fc_columns": ALL_4_LOG2FC_COLUMNS,
        "use_observed_readout": True,
        "use_predicted_readout": False,
    },
    "observed_2task": {
        "log2fc_columns": TRIMMED_2_LOG2FC_COLUMNS,
        "use_observed_readout": True,
        "use_predicted_readout": False,
    },
    "predicted_native_2task": {
        "log2fc_columns": TRIMMED_2_LOG2FC_COLUMNS,
        "use_observed_readout": False,
        "use_predicted_readout": True,
    },
    "observed_and_predicted_2task": {
        "log2fc_columns": TRIMMED_2_LOG2FC_COLUMNS,
        "use_observed_readout": True,
        "use_predicted_readout": True,
    },
}


def main() -> None:
    """Write one YAML config per feature-source variant."""
    base = yaml.safe_load(BASE_CONFIG.read_text())

    for stem, overrides in VARIANTS.items():
        cfg = copy.deepcopy(base)

        cfg["data"]["output_dir"] = f"results/feat_{stem}"
        cfg["data"]["plan"]["input_csv"] = "data/moal_plan_state.csv"
        cfg["data"]["plan"]["output_csv"] = f"pxr_feat_{stem}_predictions.csv"
        cfg["data"]["plan"]["log2fc_columns"] = overrides["log2fc_columns"]
        cfg["auxiliary_model"]["use_observed_readout"] = overrides["use_observed_readout"]
        cfg["auxiliary_model"]["use_predicted_readout"] = overrides["use_predicted_readout"]

        out_path = CONFIGS_DIR / f"feat_{stem}.yaml"
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"Wrote {len(VARIANTS)} configs to {CONFIGS_DIR}/ (feat_*.yaml)")


if __name__ == "__main__":
    main()
