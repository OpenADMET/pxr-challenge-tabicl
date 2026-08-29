"""Frozen embedding + RDKit + Mordred + predicted-log2FC features fed through TabPFN.

Extends `tabpfn_frozen_features.py`'s frozen-embedding baseline with the
tabular-descriptor block the N283T report's winning ablation used (tabular
core with CheMeleon: 0.421 MAE), minus the Boltz-2/Pose-Jazzy structural
features that are out of scope here. Per compound: frozen auxiliary-encoder
embedding + predicted log2FC readout, concatenated with a PCA-compressed
block of RDKit descriptors (217) and Mordred 2D descriptors (~1613,
`data/descriptor_cache.parquet`, precomputed by `precompute_descriptors.py`).

The raw descriptor block (~1830 columns) OOMs TabPFN's between-items attention
even with `memory_saving_mode`/`fit_mode="low_memory"` on this GPU: that
attention pass scales with feature count as well as row count, not just row
count as the frozen-embedding baseline assumed. The same OOM reproduced with
CheMeleon's 2048-dim embedding alone (no descriptors at all), so this script
uses a from-scratch auxiliary encoder (`configs/tabpfn_small_embed.yaml`,
256-dim) and additionally median-imputes the descriptors (PCA cannot take
TabPFN's native NaN handling) and compresses them to `--pca-components`
components fit on the training split only, keeping total feature width well
under TabPFN's 500-feature untuned-limits threshold.

The embedding and predicted-readout block is read from
`data/auxiliary_embedding_cache.parquet` (`precompute_auxiliary_embeddings.py`),
a single from-scratch encoder trained once and reused by every regressor,
PCA width, and descriptor-family variant of this script. Retraining the
encoder per invocation, as this script did before, gave every run a
different random encoder init: MAE differences across runs then partly
reflected encoder variance rather than the feature or regressor choice
actually being ablated.

Run with:
    python tabpfn_concat_features.py configs/tabpfn_small_embed.yaml \
        --output-dir results/tabpfn_concat_features
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from evaluate_predictions import evaluate
from moal.auxiliary_encoder import AuxiliaryEncoderModule
from moal.config import PipelineConfig
from moal.planning import parse_campaign_state, training_records_for_refit
from moal.preprocessing import SMILESPreprocessor
from moal.types import QueryType

logger = logging.getLogger(__name__)

_DESCRIPTOR_CACHE = Path("data/descriptor_cache.parquet")
_EMBEDDING_CACHE = Path("data/auxiliary_embedding_cache.parquet")


def _load_descriptor_cache(source: str) -> pd.DataFrame:
    """Load the precomputed descriptor block, numeric columns only.

    Parameters
    ----------
    source : str
        Which descriptor family to keep: ``"all"`` for RDKit+Mordred,
        ``"rdkit"`` or ``"mordred"`` to ablate the other family out entirely.
    """
    cache = pd.read_parquet(_DESCRIPTOR_CACHE)
    numeric = cache.drop(columns=["canonical_smiles"]).select_dtypes(include=[np.number])
    if source in {"rdkit", "mordred"}:
        prefix = f"{source}_"
        numeric = numeric[[c for c in numeric.columns if c.startswith(prefix)]]
    all_nan = numeric.columns[numeric.isna().all()]
    if len(all_nan):
        logger.info("Dropping %d all-NaN descriptor column(s)", len(all_nan))
        numeric = numeric.drop(columns=all_nan)
    return pd.concat([cache[["canonical_smiles"]], numeric], axis=1).set_index("canonical_smiles")


def _load_embedding_cache(path: Path | None = None) -> pd.DataFrame:
    """Load a pre-trained auxiliary-encoder embedding + readout cache.

    Parameters
    ----------
    path : Path, optional
        Cache file to load. Defaults to the canonical `_EMBEDDING_CACHE`;
        pass a seed-specific cache (e.g.
        `data/auxiliary_embedding_cache_seed0.parquet`) to reuse one of the
        5-seed replicate embeddings from `precompute_auxiliary_embeddings.py
        --seed`, keeping the embedding fixed across a set of comparisons.

    Raises
    ------
    FileNotFoundError
        If the cache hasn't been minted yet; run
        `precompute_auxiliary_embeddings.py configs/tabpfn_small_embed.yaml` first.
    """
    cache_path = path if path is not None else _EMBEDDING_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found; run "
            "`python precompute_auxiliary_embeddings.py configs/tabpfn_small_embed.yaml` first"
        )
    return pd.read_parquet(cache_path).set_index("canonical_smiles")


def _embedding_features(
    embedding_cache: pd.DataFrame,
    canonical_smiles: list[str],
    include_embedding: bool,
    include_readout: bool,
) -> np.ndarray:
    """Return the cached embedding block, readout block, or their concatenation."""
    rows = embedding_cache.loc[canonical_smiles]
    blocks = []
    if include_embedding:
        blocks.append(rows[[c for c in rows.columns if c.startswith("embed_")]].to_numpy(dtype=np.float32))
    if include_readout:
        blocks.append(rows[[c for c in rows.columns if c.startswith("readout_")]].to_numpy(dtype=np.float32))
    return np.concatenate(blocks, axis=1)


def _live_embedding_features(
    aux_encoder: AuxiliaryEncoderModule,
    canonical_smiles: list[str],
    include_embedding: bool,
    include_readout: bool,
) -> np.ndarray:
    """Return the embedding block, the predicted-readout block, or their concatenation.

    Unlike `_embedding_features`, this runs the encoder directly instead of
    reading the canonical cache. Used only where retraining a fresh encoder
    per call is the point (`tabpfn_calibrate_predictions.py`'s per-fold,
    leakage-free OOF calibration), not for any run being compared against
    the cached-embedding results elsewhere.
    """
    blocks = []
    if include_embedding:
        blocks.append(aux_encoder.embed_smiles(canonical_smiles))
    if include_readout:
        blocks.append(aux_encoder.predict_smiles(canonical_smiles))
    return np.concatenate(blocks, axis=1)


def _raw_descriptors(canonical_smiles: list[str], descriptor_cache: pd.DataFrame) -> np.ndarray:
    """Look up the raw (uncompressed) descriptor block for a list of compounds."""
    return descriptor_cache.loc[canonical_smiles].to_numpy(dtype=np.float64)


def _build_regressor(
    name: str,
    seed: int,
    n_features: int,
    tabfm_n_estimators: int = 32,
    tabicl_batch_size: int | None = 2,
):
    """Build the final regressor fit on the assembled feature matrix.

    Parameters
    ----------
    name : str
        One of ``"tabpfn"``, ``"tabpfn-v2.6"``, ``"tabpfn-v3"``, ``"lgbm"``,
        ``"xgboost"``, ``"tabicl"``, ``"tabfm"``. The tabpfn variants pin
        `ModelVersion.V2_5`/`V2_6`/`V3` respectively (``"tabpfn"`` keeps the
        package default, v2.5, unpinned so it tracks whatever `TabPFNRegressor()`
        defaults to). tabicl/tabfm are additional tabular foundation models,
        included to ablate which tabular foundation model matters rather than
        just foundation-model-vs-classical-ML (already covered by lgbm/xgboost).
    seed : int
        Random seed to thread through the chosen regressor.
    n_features : int
        Width of the assembled feature matrix, used only to set TabPFN's
        `ignore_pretraining_limits` flag past its documented 2000-column cap.
    tabfm_n_estimators : int, optional
        Ensemble size passed to `TabFMRegressor` when `name == "tabfm"`;
        ignored otherwise. Default 32 matches the library default.
    tabicl_batch_size : int or None, optional
        `batch_size` passed to `TabICLRegressor` when `name == "tabicl"`;
        ignored otherwise. Default 2, tuned for the narrow (130-column)
        no-embedding featureset. `None` keeps the library default (8), which
        some wider featuresets need instead: batch_size doesn't reliably
        trade off against peak memory here, so a narrower value isn't a safe
        default for every featureset and should be checked per call site.
    """
    if name == "tabpfn":
        return TabPFNRegressor(
            random_state=seed,
            memory_saving_mode=True,
            fit_mode="low_memory",
            ignore_pretraining_limits=n_features > 2000,
        )
    if name in ("tabpfn-v2.6", "tabpfn-v3"):
        version = ModelVersion.V2_6 if name == "tabpfn-v2.6" else ModelVersion.V3
        return TabPFNRegressor.create_default_for_version(
            version,
            random_state=seed,
            memory_saving_mode=True,
            fit_mode="low_memory",
            ignore_pretraining_limits=n_features > 2000,
        )
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, verbose=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed)
    if name == "tabicl":
        from tabicl import TabICLRegressor

        # batch_size=2 (ensemble members processed in parallel through one
        # attention call) is what the narrow 130-column no-embedding
        # featureset needs to avoid OOM at the library default (8). This
        # isn't a universal fix, though: the CheMeleon-embedding featuresets
        # (258/386 columns) instead OOM harder *with* batch_size forced down
        # to 2 and only succeed at the library default, so batch_size doesn't
        # trade off against peak memory in one consistent direction across
        # featuresets here; callers on a new featureset should check both
        # before assuming this default transfers.
        kwargs = {} if tabicl_batch_size is None else {"batch_size": tabicl_batch_size}
        return TabICLRegressor(random_state=seed, **kwargs)
    from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch

    # TabFM ships pretrained weights under a separate non-commercial license
    # (`tabfm-non-commercial-v1.0`) distinct from the Apache-licensed package
    # code; auto-downloads from Hugging Face (google/tabfm-1.0.0-pytorch) on
    # first use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tabfm_v1_0_0_pytorch.load(model_type="regression", device=device)

    # the full 4139-row training context OOMs this GPU's attention pass (rows
    # enter the same between-items attention as TabPFN); capping the in-context
    # rows per ensemble member keeps memory bounded without dropping the block.
    # 2000 OOM'd (17.65 GiB allocated, tried to allocate 8.51 GiB more); 1000
    # still OOM'd on an otherwise-idle GPU (18.25 GiB allocated, tried to
    # allocate 5.12 GiB more), so cap tighter at 500
    return TabFMRegressor(
        model=model, random_state=seed, max_num_rows=500, n_estimators=tabfm_n_estimators
    )


def main() -> None:
    """Pretrain the frozen encoder, join descriptors, fit TabPFN, and score the blind set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="moal plan config supplying data/auxiliary_model settings")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tabpfn_concat_features"))
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Embedding cache to read (default: the canonical seed-42 cache); pass a "
        "seed-specific cache from precompute_auxiliary_embeddings.py --seed to hold "
        "the encoder fixed within a 5-seed replicate comparison",
    )
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Components to compress the descriptor block to before TabPFN",
    )
    parser.add_argument(
        "--descriptor-source",
        choices=["all", "rdkit", "mordred"],
        default="all",
        help="'all' for RDKit+Mordred, or 'rdkit'/'mordred' to keep only that family",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Drop the raw encoder embedding block (keep the predicted-readout block, if enabled)",
    )
    parser.add_argument(
        "--no-readout",
        action="store_true",
        help="Drop the predicted-log2FC-readout block (keep the raw embedding, if enabled)",
    )
    parser.add_argument(
        "--regressor",
        choices=["tabpfn", "tabpfn-v2.6", "tabpfn-v3", "lgbm", "xgboost", "tabicl", "tabfm"],
        default="tabpfn",
        help=(
            "Final regressor fit on the assembled feature matrix. tabpfn/tabpfn-v2.6/"
            "tabpfn-v3/tabicl/tabfm are tabular foundation models (in-context "
            "learners); lgbm/xgboost are classical gradient-boosted trees, kept as "
            "a non-foundation-model floor"
        ),
    )
    parser.add_argument(
        "--tabfm-n-estimators",
        type=int,
        default=32,
        help="Ensemble size for --regressor tabfm (ignored otherwise); library default is 32",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed (PCA and regressor random state) for a seed-sweep replicate",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    skip_encoder = args.no_embedding and args.no_readout
    if not skip_encoder and cfg.auxiliary_model is None:
        raise ValueError(f"{args.config} has no auxiliary_model block; TabPFN baseline needs one")

    descriptor_cache = _load_descriptor_cache(args.descriptor_source)

    preprocessor = SMILESPreprocessor()
    state_df = pd.read_csv(cfg.data.plan.input_csv)
    state = parse_campaign_state(
        state_df,
        cost_ps=cfg.oracle.cost_ps,
        cost_drc=cfg.oracle.cost_drc,
        upper_bound=cfg.oracle.upper_bound,
        preprocessor=preprocessor,
        smiles_column=cfg.data.plan.smiles_column,
        relation_column=cfg.data.plan.relation_column,
        value_column=cfg.data.plan.value_column,
        weight_column=cfg.data.plan.weight_column,
        log2fc_columns=cfg.data.plan.log2fc_columns,
        is_canonical=cfg.data.plan.is_canonical,
        expected_ps_threshold=cfg.oracle.ps_threshold,
    )
    fit_records = training_records_for_refit(state.training_records)
    drc_records = [r for r in fit_records if r.fidelity == QueryType.DOSE_RESPONSE]

    train_smiles = [r.canonical_smiles for r in drc_records]
    train_true = np.array([r.value for r in drc_records], dtype=np.float64)
    test_smiles = [smi for _, smi in state.unqueried_rows]

    if skip_encoder:
        logger.info("Skipping auxiliary encoder; fitting TabPFN on descriptors alone")
        train_embed = np.empty((len(train_smiles), 0), dtype=np.float32)
        test_embed = np.empty((len(test_smiles), 0), dtype=np.float32)
    else:
        embedding_cache = _load_embedding_cache(args.embedding_cache)
        train_embed = _embedding_features(
            embedding_cache, train_smiles, not args.no_embedding, not args.no_readout
        )
        test_embed = _embedding_features(
            embedding_cache, test_smiles, not args.no_embedding, not args.no_readout
        )

    train_descriptors_raw = _raw_descriptors(train_smiles, descriptor_cache)
    test_descriptors_raw = _raw_descriptors(test_smiles, descriptor_cache)

    imputer = SimpleImputer(strategy="median").fit(train_descriptors_raw)
    pca = PCA(n_components=args.pca_components, random_state=cfg.seed).fit(
        imputer.transform(train_descriptors_raw)
    )
    train_descriptors = pca.transform(imputer.transform(train_descriptors_raw))
    test_descriptors = pca.transform(imputer.transform(test_descriptors_raw))
    logger.info(
        "Compressed %d descriptor column(s) to %d PCA component(s) (%.1f%% variance explained)",
        train_descriptors_raw.shape[1],
        args.pca_components,
        100 * pca.explained_variance_ratio_.sum(),
    )

    train_features = np.concatenate([train_embed, train_descriptors], axis=1)
    test_features = np.concatenate([test_embed, test_descriptors], axis=1)
    logger.info("Extracted %s concatenated feature matrix for training", train_features.shape)
    logger.info("Extracted %s concatenated feature matrix for blind targets", test_features.shape)

    regressor = _build_regressor(
    args.regressor, cfg.seed, train_features.shape[1], tabfm_n_estimators=args.tabfm_n_estimators
)
    regressor.fit(train_features, train_true)
    test_predictions = regressor.predict(test_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = args.output_dir / "pxr_tabpfn_predictions.csv"
    pd.DataFrame({"smiles": test_smiles, "predicted_pec50": test_predictions}).to_csv(
        predictions_csv, index=False
    )
    logger.info("Wrote predictions to %s", predictions_csv)

    results = evaluate(predictions_csv, activity_threshold=args.activity_threshold)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results.to_string(index=False))
    results.to_csv(args.output_dir / "eval_out.csv", index=False)


if __name__ == "__main__":
    main()
