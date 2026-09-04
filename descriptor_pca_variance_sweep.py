"""Sweep PCA width on cached Mordred descriptors and plot cumulative explained variance.

`reporting/blogpost.md`'s PCA-compression section claims a specific explained-variance
figure for the 128-component width used elsewhere in the sweeps; this script checks that
claim by fitting PCA at a range of widths from 2 to 256 components on the same
median-imputed, unstandardized Mordred descriptor block (`data/descriptor_cache.parquet`,
`mordred_*` columns) and reporting cumulative explained variance at each width.

Run with:
    python descriptor_pca_variance_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

_DESCRIPTOR_CACHE = Path("data/descriptor_cache.parquet")
_PCA_WIDTHS = [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
_SEED = 0


def _load_mordred_descriptors() -> np.ndarray:
    """Return the raw, median-imputed Mordred descriptor block."""
    cache = pd.read_parquet(_DESCRIPTOR_CACHE)
    mordred = cache[[c for c in cache.columns if c.startswith("mordred_")]]
    all_nan = mordred.columns[mordred.isna().all()]
    mordred = mordred.drop(columns=all_nan)
    values = mordred.to_numpy(dtype=np.float64)
    return SimpleImputer(strategy="median").fit_transform(values)


def main() -> None:
    """Fit PCA at each width in `_PCA_WIDTHS` and report/plot cumulative explained variance."""
    descriptors = _load_mordred_descriptors()
    print(f"Mordred descriptor block: {descriptors.shape[0]} compounds x {descriptors.shape[1]} columns")

    explained = []
    for width in _PCA_WIDTHS:
        pca = PCA(n_components=width, random_state=_SEED).fit(descriptors)
        ratio = float(pca.explained_variance_ratio_.sum())
        explained.append(ratio)
        print(f"  {width:>4d} components: {100 * ratio:9.5f}% cumulative explained variance")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(_PCA_WIDTHS, [100 * e for e in explained], marker="o")
    ax.axvline(128, color="gray", linestyle="--", linewidth=1, label="128 (used elsewhere)")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Mordred descriptor PCA: explained variance vs. width")
    ax.legend()
    out_path = Path("descriptor_pca_variance_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
