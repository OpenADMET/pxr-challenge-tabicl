#!/usr/bin/env bash
# Run every config in configs/ through `moal plan`, skipping combinations
# that already have a prediction CSV in their output dir so the sweep can be
# interrupted and resumed without rerunning finished work.
set -euo pipefail

cd "$(dirname "$0")"

for config in configs/*.yaml; do
    stem="$(basename "${config}" .yaml)"
    output_dir="results/${stem}"
    predictions_csv="${output_dir}/pxr_aux_predictions.csv"

    if [[ -f "${predictions_csv}" ]]; then
        echo "Skipping ${stem} (already have ${predictions_csv})"
        continue
    fi

    echo "Running ${stem}"
    moal plan --config "${config}" --output-dir "${output_dir}"
done
