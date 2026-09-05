"""Stage four: read every run off disk into one table, summarize it, and check coverage.

A run directory is the unit of evidence, and this module is the only place that
turns a tree of them into something a figure can be drawn from. It reads each
run's own record rather than inferring anything from the directory name, so the
table carries the configuration, the metrics, the input block keys and the
commit that produced them, and a row can always be traced back to the run that
wrote it.

Three readings of the same runs are produced, and they are not interchangeable.
The per-seed rows are the raw evidence. The per-configuration seed summary is
the mean and spread across replicate seeds, which describes how much a result
moves when the training seed changes. The ensemble row averages the
per-compound predictions across a configuration's seeds and scores that average
once, which is what the challenge leaderboard entry was, so it is the only row
comparable to the anchor. Averaging seed-wise scores is not the same quantity
and is never reported as an ensemble, which is why the ensemble rows carry their
own ``aggregation`` label instead of being folded in with the seed means.

A run whose record is missing or unreadable raises rather than being skipped. A
silently dropped run reads downstream as a configuration that was never planned,
which is exactly how a coverage gap hides, so coverage is checked against the
manifest and a partial sweep is reported as partial.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate
from data import CANONICAL_COL
from manifest import TABULAR_AXES, Coverage, Manifest

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

# the name of the tidy table, written under the results root it was read from
TABLE_NAME = "results.parquet"

# the two families of run directory, each a subdirectory of the results root
KINDS = ("tabular", "gnn")

# metric columns every run carries, in the order evaluate reports them
METRIC_COLUMNS = ("n", *evaluate.METRIC_NAMES)

# how the summary labels a row's provenance: an average of seed-wise scores, or
# a single score of the seed-averaged predictions
SEED_MEAN = "seed_mean"
ENSEMBLE = "ensemble"

# columns of the tidy per-run table, in order
ROW_COLUMNS = (
    "kind",
    "config_id",
    "seed",
    *TABULAR_AXES,
    *METRIC_COLUMNS,
    "calibrated",
    "n_features",
    "n_fit",
    "n_test",
    "input_blocks",
    "input_keys",
    "commit",
    "run_dir",
)

# the columns that identify a configuration, which the summaries group on
GROUP_COLUMNS = ("kind", "config_id", *TABULAR_AXES, "calibrated")

# dtypes fixed at the I/O boundary, so a missing axis does not silently make an
# integer column float and a run count does not become a float either
COLUMN_DTYPES: dict[str, str] = {
    "kind": "string",
    "config_id": "string",
    "seed": "Int64",
    "embedding": "string",
    "readout": "string",
    "descriptors": "string",
    "descriptor_pca": "Int64",
    "regressor": "string",
    "calibration": "string",
    "calibrated": "boolean",
    "n_features": "Int64",
    "n_fit": "Int64",
    "n_test": "Int64",
    "input_blocks": "string",
    "input_keys": "string",
    "commit": "string",
    "run_dir": "string",
}

# how many paths a coverage report lists before it summarizes the remainder
REPORT_LIMIT = 20


class AggregateError(RuntimeError):
    """A run directory could not be read, or its parts disagree with each other."""


def discover_runs(results_root: Path = RESULTS_DIR) -> list[tuple[str, Path]]:
    """Find every run directory under a results root.

    A directory is a run because of where it sits, not because of what it
    contains, so an interrupted run with no record is still discovered and is
    then reported by :func:`read_run` rather than passed over.

    Parameters
    ----------
    results_root : path-like, optional
        Root holding the ``tabular`` and ``gnn`` subtrees.

    Returns
    -------
    list of (str, Path)
        The kind and directory of each run, sorted within each kind.
    """
    found: list[tuple[str, Path]] = []
    for kind in KINDS:
        root = results_root / kind
        if not root.is_dir():
            continue
        found.extend((kind, path) for path in sorted(root.glob("*/seed*")) if path.is_dir())
    return found


def read_run(run_dir: Path, kind: str) -> dict[str, Any]:
    """Read one run directory into a single tidy row.

    Parameters
    ----------
    run_dir : path-like
        A directory of the form ``<results root>/<kind>/<config id>/seed<n>``.
    kind : str
        Either ``tabular`` or ``gnn``.

    Returns
    -------
    dict
        One row, keyed by :data:`ROW_COLUMNS`. Axis columns a graph-network run
        has no equivalent for come back as None.

    Raises
    ------
    AggregateError
        If the record is absent or unreadable, if it holds no metrics, or if it
        names a configuration other than the directory it sits in.
    """
    record = _read_record(run_dir)
    config = record.get("config") or {}
    environment = record.get("environment") or {}
    inputs = record.get("inputs") or []

    config_id = run_dir.parent.name
    recorded_id = record.get("slug") or record.get("cell")
    if recorded_id is not None and recorded_id != config_id:
        raise AggregateError(
            f"{run_dir}: the record names configuration {recorded_id!r}, but the directory "
            f"is {config_id!r}, so the run has been moved or overwritten"
        )

    row: dict[str, Any] = {
        "kind": kind,
        "config_id": config_id,
        "seed": _seed_of(run_dir, record),
        **{axis: config.get(axis) for axis in TABULAR_AXES},
        **_read_metrics(run_dir, record),
        "calibrated": record.get("calibrated"),
        "n_features": record.get("n_features"),
        "n_fit": record.get("n_fit"),
        "n_test": record.get("n_test"),
        "input_blocks": ",".join(str(item.get("block")) for item in inputs),
        "input_keys": ",".join(str(item.get("key")) for item in inputs),
        "commit": (environment.get("repo") or {}).get("commit"),
        "run_dir": str(run_dir),
    }
    return row


def tidy(results_root: Path = RESULTS_DIR) -> pd.DataFrame:
    """Read every run under a results root into one row apiece.

    Parameters
    ----------
    results_root : path-like, optional
        Root holding the ``tabular`` and ``gnn`` subtrees.

    Returns
    -------
    DataFrame
        One row per run, with the columns of :data:`ROW_COLUMNS`. Empty, but
        still fully typed, when no run has been written yet.

    Raises
    ------
    AggregateError
        If any discovered run directory cannot be read.
    """
    rows = [read_run(run_dir, kind) for kind, run_dir in discover_runs(results_root)]
    frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    logger.info("read %d runs from %s", len(frame), results_root)
    return frame.astype(COLUMN_DTYPES)


def write_table(frame: pd.DataFrame, results_root: Path = RESULTS_DIR) -> Path:
    """Write the tidy table beside the runs it was read from.

    Parameters
    ----------
    frame : DataFrame
        The table from :func:`tidy`.
    results_root : path-like, optional
        Root the table is written into, as ``results.parquet``.

    Returns
    -------
    Path
        Where the table was written.
    """
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / TABLE_NAME
    frame.to_parquet(path, index=False)
    logger.info("wrote %d rows -> %s", len(frame), path)
    return path


def seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each configuration's metrics across its seeds.

    Parameters
    ----------
    frame : DataFrame
        The tidy per-run table.

    Returns
    -------
    DataFrame
        One row per configuration, carrying each metric's mean across seeds,
        its standard deviation in a ``<metric>_std`` column, the seed count,
        and ``aggregation`` set to ``seed_mean``. The standard deviation is NaN
        for a configuration with a single seed.
    """
    if frame.empty:
        return _empty_summary()

    aggregations: dict[str, tuple[str, str]] = {}
    for metric in METRIC_COLUMNS:
        aggregations[metric] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    aggregations["n_seeds"] = ("seed", "nunique")

    summary = (
        frame.groupby(list(GROUP_COLUMNS), dropna=False, observed=True)
        .agg(**aggregations)
        .reset_index()
        .assign(aggregation=SEED_MEAN)
    )
    return summary.reindex(columns=list(_summary_columns()))


def ensemble_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Score each configuration's seed-averaged predictions.

    The per-compound predictions are averaged across the configuration's seeds
    and scored once, which is the quantity the challenge leaderboard entry
    reports. It is not the mean of the seed-wise scores and is never merged
    with them.

    Parameters
    ----------
    frame : DataFrame
        The tidy per-run table.

    Returns
    -------
    DataFrame
        One row per configuration, in the same columns as
        :func:`seed_summary`, with ``aggregation`` set to ``ensemble`` and
        every ``<metric>_std`` column NaN, since one ensemble has no spread.

    Raises
    ------
    AggregateError
        If a run's predictions are missing, or if a configuration's seeds do
        not cover the same compounds.
    """
    if frame.empty:
        return _empty_summary()

    rows = []
    for key, group in frame.groupby(list(GROUP_COLUMNS), dropna=False, observed=True):
        stacked, observed = stack_predictions([Path(p) for p in group["run_dir"]])
        scores = evaluate.metrics(observed, evaluate.ensemble_mean(stacked))
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, key, strict=True)),
                **scores,
                "n_seeds": int(group["seed"].nunique()),
                "aggregation": ENSEMBLE,
            }
        )
    return pd.DataFrame(rows).reindex(columns=list(_summary_columns()))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Stack the seed summary and the ensemble rows into one labeled table.

    Parameters
    ----------
    frame : DataFrame
        The tidy per-run table.

    Returns
    -------
    DataFrame
        Two rows per configuration, distinguished by ``aggregation``.
    """
    seed_rows = seed_summary(frame)
    ensemble_rows = ensemble_summary(frame)
    return pd.concat([seed_rows, ensemble_rows], ignore_index=True)


def stage_coverage(
    manifest: Manifest,
    stage_id: str,
    resolved: dict[str, Any] | None = None,
    results_root: Path = RESULTS_DIR,
) -> Coverage:
    """Compare the tabular runs on disk against those one stage calls for.

    Parameters
    ----------
    manifest : Manifest
        The coverage spec.
    stage_id : str
        Which stage to check.
    resolved : dict, optional
        Axis values an earlier gate settled, required by a stage that names
        ``fixed_from``.
    results_root : path-like, optional
        Root holding the run directories.

    Returns
    -------
    Coverage
        The expected, missing and unplanned run directories. ``unplanned`` is
        relative to this stage alone, so runs another stage called for appear
        there; check each stage in turn to account for a whole sweep.
    """
    expected = manifest.planned_runs(stage_id, resolved, results_root)
    produced = [path for kind, path in discover_runs(results_root) if kind == "tabular"]
    return manifest.coverage(expected, produced)


def gnn_coverage(manifest: Manifest, results_root: Path = RESULTS_DIR) -> Coverage:
    """Compare the graph-network runs on disk against the cells the manifest names."""
    produced = [path for kind, path in discover_runs(results_root) if kind == "gnn"]
    return manifest.coverage(manifest.gnn_runs(results_root), produced)


def coverage_report(coverage: Coverage, label: str, limit: int = REPORT_LIMIT) -> str:
    """Render a coverage comparison as printable text.

    Parameters
    ----------
    coverage : Coverage
        The comparison to render.
    label : str
        What is being reported on, usually a stage identifier.
    limit : int, optional
        How many paths to list per category before summarizing the remainder.

    Returns
    -------
    str
        A headline count followed by the missing and unplanned paths.
    """
    present = len(coverage.expected) - len(coverage.missing)
    lines = [
        f"{label}: {present}/{len(coverage.expected)} planned runs present, "
        f"{len(coverage.missing)} missing, {len(coverage.unplanned)} unplanned"
    ]
    lines.extend(_listing("missing", coverage.missing, limit))
    lines.extend(_listing("unplanned", coverage.unplanned, limit))
    return "\n".join(lines)


def _read_record(run_dir: Path) -> dict[str, Any]:
    """Read a run's record, refusing to treat an unreadable run as absent."""
    path = run_dir / "run.json"
    if not path.exists():
        raise AggregateError(
            f"{run_dir}: no run.json, so this run cannot be read; delete the directory or "
            "rerun it rather than aggregating around it"
        )
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise AggregateError(f"{path}: run record is not valid JSON") from err
    if not isinstance(record, dict):
        raise AggregateError(f"{path}: run record is a {type(record).__name__}, expected an object")
    return record


def _read_metrics(run_dir: Path, record: dict[str, Any]) -> dict[str, float]:
    """Read a run's metrics, preferring the metrics file and falling back to the record."""
    path = run_dir / "metrics.json"
    scores: Any = record.get("metrics")
    if path.exists():
        try:
            scores = json.loads(path.read_text())
        except json.JSONDecodeError as err:
            raise AggregateError(f"{path}: metrics are not valid JSON") from err
    if not isinstance(scores, dict):
        raise AggregateError(f"{run_dir}: neither metrics.json nor the run record holds metrics")
    return {name: _as_float(scores.get(name)) for name in METRIC_COLUMNS}


def _seed_of(run_dir: Path, record: dict[str, Any]) -> int:
    """Return a run's seed, taken from the directory name and checked against the record."""
    suffix = run_dir.name.removeprefix("seed")
    if not suffix.isdigit():
        raise AggregateError(f"{run_dir}: directory name does not end in a seed number")
    seed = int(suffix)
    recorded = record.get("seed")
    if recorded is not None and int(recorded) != seed:
        raise AggregateError(
            f"{run_dir}: the record names seed {recorded}, but the directory names seed {seed}"
        )
    return seed


def stack_predictions(run_dirs: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Read a configuration's seed-wise predictions, aligned on compound.

    Returns the predictions as one row per seed and the observed values once,
    both ordered by canonical SMILES so the seeds are aligned by compound
    rather than by however each run happened to write its rows.
    """
    frames = []
    for run_dir in run_dirs:
        path = run_dir / "predictions.csv"
        if not path.exists():
            raise AggregateError(f"{run_dir}: no predictions.csv, so it cannot be ensembled")
        frame = pd.read_csv(path, dtype={CANONICAL_COL: str}).set_index(CANONICAL_COL).sort_index()
        frames.append((run_dir, frame))

    reference_dir, reference = frames[0]
    for run_dir, frame in frames[1:]:
        if not frame.index.equals(reference.index):
            raise AggregateError(
                f"{run_dir}: predictions cover different compounds than {reference_dir}, "
                "so averaging them would misalign the seeds"
            )

    observed = reference["observed"].to_numpy(dtype=np.float64)
    stacked = np.vstack([frame["predicted"].to_numpy(dtype=np.float64) for _, frame in frames])
    return stacked, observed


def _summary_columns() -> tuple[str, ...]:
    """Return the column order both summary tables share."""
    metrics: list[str] = []
    for metric in METRIC_COLUMNS:
        metrics.extend((metric, f"{metric}_std"))
    return (*GROUP_COLUMNS, "aggregation", "n_seeds", *metrics)


def _empty_summary() -> pd.DataFrame:
    """Return a summary table with no rows but every column."""
    return pd.DataFrame(columns=list(_summary_columns()))


def _listing(label: str, paths: Iterable[Path], limit: int) -> list[str]:
    """Render one category of a coverage report, truncated to a readable length."""
    paths = list(paths)
    if not paths:
        return []
    shown = [f"  {label}: {path}" for path in paths[:limit]]
    if len(paths) > limit:
        shown.append(f"  {label}: ... and {len(paths) - limit} more")
    return shown


def _as_float(value: Any) -> float:
    """Render a recorded metric as a float, with a missing metric becoming NaN."""
    return float("nan") if value is None else float(value)
