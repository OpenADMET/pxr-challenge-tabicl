"""Stage five: draw the manifest's seven figures as slices of one completed table.

The manifest declares figures, not stages. Each one is a selection over the
tidy per-run table plus, sometimes, a facet, so nothing here waits on anything
else and the seven can be drawn in any order once the runs exist. This module
is the only place that turns those declarations into images, and it reads the
selection off the manifest rather than restating it, so a figure whose ``select``
mapping changes there changes here without an edit.

Every panel shows the five replicate seeds as individual points, not a bar. The
runs are five-seed replicates and the seed spread is the only thing that makes a
difference between two configurations credible, so a mean drawn without it would
be the wrong summary. Two marks sit beside the seed points: the seed mean with a
standard-deviation bar, and the seed-ensemble score, which is a different
quantity and is the only one comparable to the leaderboard anchor.

The anchor is N283T's phase-2 leaderboard result, carried in the manifest and
read from there. It is drawn twice where it means something: the ensemble number
against the ensemble marks and the single-model number against the seed points.
It is a reference for where the leaderboard landed on these same 260 compounds,
never a target, and it is left off entirely when the slice was scored on a
different compound count or plotted on a metric it does not describe.

Figures 3 and 4 carry decisions, so they add a paired bootstrap over the 260
compounds beneath the spread panel: the leading configuration against its
nearest rivals, with the interval drawn on the difference. An interval spanning
zero is drawn as spanning zero rather than being read as a ranking.

Figure 7 is the exception to everything above. Predicted spread is per compound
and lives in each run's ``predictions.csv``, not in the tidy table, so that
figure reads the run directories directly. The spread also does not mean the
same thing across regressors: TabPFN reports an exact predictive standard
deviation and TabICL a normal-equivalent scale taken between its 10th and 90th
percentiles. Those are faceted apart and labelled with their kind rather than
pooled.

A figure whose slice is empty is skipped and reported. A blank chart with real
axes reads as a measured absence rather than a missing sweep, which is worse
than no chart at all.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import evaluate
import gates
import manifest as manifest_module
import regressors
from data import CANONICAL_COL
from evaluate import BootstrapResult
from manifest import Manifest

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")

# the three axes a featureset label is built from, in slug order
FEATURE_AXES = ("embedding", "readout", "descriptors")

# how each regressor's predicted_std is to be read. The kinds are not the same
# quantity and are never pooled: TabPFN's bar distribution gives a predictive
# standard deviation outright, TabICL only a percentile span rescaled as if the
# predictive distribution were normal. Regressors absent here report no spread.
UNCERTAINTY_KINDS: dict[str, str] = {
    **dict.fromkeys(regressors.TABPFN_NAMES, "exact predictive s.d."),
    "tabicl": "normal-equivalent s.d. from p10 to p90",
}

# metrics on which a smaller number is a better model; the rest are correlations
LOWER_IS_BETTER = frozenset({"mae", "rmse", "rae"})

# the metric every figure is drawn on unless a caller asks for another, chosen
# because it is the metric the anchor is quoted in
DEFAULT_METRIC = "mae"

# resamples behind a paired-bootstrap interval, and how many runners-up get one
DEFAULT_RESAMPLES = 2000
COMPARISON_LIMIT = 4

# output settings for the rendered files
DEFAULT_DPI = 200
FILE_SUFFIX = ".png"

# figure geometry, in inches; the height cap keeps a 39-featureset panel from
# growing into a file no viewer will open
PANEL_WIDTH = 3.8
LABEL_WIDTH = 3.4
ROW_HEIGHT = 0.30
COMPARISON_HEIGHT = 2.4
MAX_FIGURE_HEIGHT = 40.0
MIN_FIGURE_WIDTH = 9.0
MAX_FIGURE_WIDTH = 34.0

# under a normal predictive distribution, a compound whose scale is sigma has
# expected absolute error sigma * sqrt(2 / pi); this is the line a perfectly
# calibrated spread would sit on
CALIBRATED_SLOPE = float(np.sqrt(2.0 / np.pi))

# the placeholder hue used when a figure has no hue axis, so that the drawing
# code has one shape rather than two
NO_HUE = ""


class FigureError(RuntimeError):
    """A figure cannot be drawn from the table or run directories it was given."""


@dataclass(frozen=True)
class Comparison:
    """One paired-bootstrap comparison of a challenger against the leader.

    Attributes
    ----------
    leader : str
        Configuration identifier of the leading configuration.
    challenger : str
        Configuration identifier being compared against it.
    label : str
        Human-readable name for the challenger, as the figure axis shows it.
    result : BootstrapResult
        The interval on ``metric(challenger) - metric(leader)``, so a positive
        difference means the challenger is the worse model on an error metric.
    separates : bool
        Whether the interval lies wholly on one side of zero.
    """

    leader: str
    challenger: str
    label: str
    result: BootstrapResult
    separates: bool


@dataclass(frozen=True)
class Skipped:
    """A figure that was not drawn, and why."""

    figure_id: str
    reason: str


def figure_declaration(spec: Manifest, figure_id: str) -> dict[str, Any]:
    """Return one figure's declaration from the manifest.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    figure_id : str
        The figure identifier, for example ``fig4``.

    Returns
    -------
    dict
        The declaration, carrying ``id``, ``title``, ``question``, ``select``
        and optionally ``facet``.

    Raises
    ------
    KeyError
        If the manifest declares no such figure.
    """
    for declaration in spec.figures:
        if declaration["id"] == figure_id:
            return dict(declaration)
    raise KeyError(f"no figure {figure_id!r}; known: {[f['id'] for f in spec.figures]}")


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the manifest's selections and facets refer to.

    The manifest selects on ``n_blocks``, ``has_descriptors`` and
    ``has_uncertainty``, none of which the tidy table carries as columns. They
    are derived once here so that selection itself stays plain column equality
    and a figure cannot quietly disagree with the manifest about what a
    single-block configuration is.

    Parameters
    ----------
    frame : DataFrame
        The tidy per-run table from :func:`aggregate.tidy`.

    Returns
    -------
    DataFrame
        A copy carrying ``featureset``, ``n_blocks``, ``has_descriptors``,
        ``has_uncertainty`` and ``uncertainty_kind``. Rows with no tabular axes
        at all, such as a graph-network run, get NA for the first three.
    """
    labels, counts = _featureset_columns(frame)
    regressor = frame["regressor"] if "regressor" in frame else pd.Series(index=frame.index)
    descriptors = frame["descriptors"] if "descriptors" in frame else pd.Series(index=frame.index)
    embedding = frame["embedding"] if "embedding" in frame else pd.Series(index=frame.index)
    return frame.assign(
        featureset=labels,
        n_blocks=counts,
        has_descriptors=(descriptors.notna() & descriptors.ne("none")).astype("boolean"),
        has_embedding=(embedding.notna() & embedding.ne("none")).astype("boolean"),
        has_uncertainty=regressor.isin(UNCERTAINTY_KINDS).astype("boolean"),
        uncertainty_kind=regressor.map(UNCERTAINTY_KINDS).astype("string"),
    )


def select_rows(frame: pd.DataFrame, select: dict[str, Any]) -> pd.DataFrame:
    """Return the rows a figure's ``select`` mapping picks out.

    Parameters
    ----------
    frame : DataFrame
        An annotated table, as returned by :func:`annotate`.
    select : dict
        Column name mapped either to the single level it must equal or to a
        list of acceptable levels. A missing value never matches, so a
        graph-network row is not selected by a tabular figure's descriptor
        condition.

    Returns
    -------
    DataFrame
        The matching rows, as a copy.

    Raises
    ------
    FigureError
        If the mapping names a column the annotated table does not carry.
    """
    mask = pd.Series(True, index=frame.index)
    for column, value in select.items():
        if column not in frame.columns:
            raise FigureError(
                f"figure selection names {column!r}, which is not a column of the annotated "
                f"table; known: {sorted(frame.columns)}"
            )
        matches = frame[column].isin(value) if isinstance(value, list) else frame[column].eq(value)
        mask &= matches.fillna(False).astype(bool)
    return frame.loc[mask].copy()


def slice_for(spec: Manifest, figure_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate the tidy table and apply one figure's declared selection.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest, which owns the selection.
    figure_id : str
        Which figure to slice for.
    frame : DataFrame
        The tidy per-run table.

    Returns
    -------
    DataFrame
        The annotated rows this figure is drawn from, possibly empty.
    """
    declaration = figure_declaration(spec, figure_id)
    select = resolve_select(declaration.get("select") or {}, figure_id)
    return select_rows(annotate(frame), select)


def resolve_select(select: dict[str, Any], figure_id: str) -> dict[str, Any]:
    """Replace every ``@gate`` reference with the value the gates chose.

    A figure that pinned a winner in its own declaration would be a second
    place for that winner to live, and the two would drift the first time a
    gate landed somewhere else. So the reference is resolved against the
    recorded decisions instead.

    Raises
    ------
    FigureError
        If a reference names an axis no resolved gate has chosen.
    """
    chosen = gates.all_chosen()

    def resolve(axis: str, level: Any) -> Any:
        if level != manifest_module.GATE_REF:
            return level
        if axis not in chosen:
            raise FigureError(
                f"{figure_id}: selection says {manifest_module.GATE_REF!r} for {axis!r}, "
                f"but no resolved gate has chosen it; known: {sorted(chosen)}"
            )
        return chosen[axis]

    return {
        axis: [resolve(axis, level) for level in value]
        if isinstance(value, list)
        else resolve(axis, value)
        for axis, value in select.items()
    }


def anchor_lines(
    spec: Manifest, frame: pd.DataFrame, metric: str = DEFAULT_METRIC
) -> list[tuple[str, float]]:
    """Return the anchor reference lines that are meaningful for a slice.

    The anchor is read from the manifest, never restated here. It describes
    mean absolute error over the challenge's 260 phase-2 compounds, so it is
    withheld from a figure drawn on any other metric, and from one whose runs
    scored a different number of compounds, where the two numbers would not be
    on the same footing.

    One line, not two. The report's other widely quoted figure, 0.437, is an
    out-of-fold score over training compounds rather than a phase-2 score, so
    it has no place on an axis of phase-2 results; the manifest records it and
    the reason under ``anchor.not_the_anchor``.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest, whose ``anchor`` carries the values.
    frame : DataFrame
        The slice about to be drawn, read for its compound count.
    metric : str, optional
        The metric the figure is drawn on. Default is ``mae``.

    Returns
    -------
    list of (str, float)
        A label and a value per line, empty when the anchor does not apply.
    """
    anchor = spec.anchor
    if metric != "mae":
        logger.info("anchor withheld: it is quoted in mae, and this figure is drawn on %s", metric)
        return []

    counts = {int(value) for value in pd.to_numeric(frame["n"], errors="coerce").dropna().unique()}
    if counts != {int(anchor["n"])}:
        logger.info(
            "anchor withheld: it scores %d compounds, and this slice scores %s",
            int(anchor["n"]),
            sorted(counts) or "nothing",
        )
        return []

    name = anchor["name"]
    return [
        (f"{name}, ensemble", float(anchor["mae_ensemble"])),
    ]


def read_predictions(run_dir: Path) -> pd.DataFrame:
    """Read one run's per-compound predictions, indexed by canonical SMILES.

    Parameters
    ----------
    run_dir : path-like
        A run directory holding ``predictions.csv``.

    Returns
    -------
    DataFrame
        Indexed by canonical SMILES and sorted by it, carrying ``observed``,
        ``predicted`` and, where the regressor reported one, ``predicted_std``.

    Raises
    ------
    FigureError
        If the run wrote no predictions.
    """
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise FigureError(f"{run_dir}: no predictions.csv, so this run cannot be plotted")
    frame = pd.read_csv(path, dtype={CANONICAL_COL: str})
    return frame.set_index(CANONICAL_COL).sort_index()


def ensemble_predictions(run_dirs: Sequence[Path]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Average a configuration's seed-wise predictions, aligned on compound.

    Parameters
    ----------
    run_dirs : sequence of path-like
        Every seed of one configuration.

    Returns
    -------
    compounds : list of str
        The canonical SMILES the predictions are aligned on, sorted.
    observed : ndarray
        The ground truth, once.
    predicted : ndarray
        The per-compound mean across seeds, which is what the leaderboard
        entry was.

    Raises
    ------
    FigureError
        If no run directories were given, or the seeds cover different
        compounds and averaging them would misalign the seeds.
    """
    if not run_dirs:
        raise FigureError("cannot ensemble an empty set of runs")

    frames = [(run_dir, read_predictions(run_dir)) for run_dir in run_dirs]
    reference_dir, reference = frames[0]
    for run_dir, frame in frames[1:]:
        if not frame.index.equals(reference.index):
            raise FigureError(
                f"{run_dir}: predictions cover different compounds than {reference_dir}, "
                "so averaging them would misalign the seeds"
            )

    stacked = np.vstack([frame["predicted"].to_numpy(dtype=np.float64) for _, frame in frames])
    observed = reference["observed"].to_numpy(dtype=np.float64)
    return [str(value) for value in reference.index], observed, evaluate.ensemble_mean(stacked)


def paired_intervals(
    frame: pd.DataFrame,
    *,
    metric: str = DEFAULT_METRIC,
    group: str = "config_id",
    labels: dict[str, str] | None = None,
    limit: int = COMPARISON_LIMIT,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> list[Comparison]:
    """Compare the leading configuration against its nearest rivals.

    Ranking picks one configuration per `group`, its best, and orders those
    groups by their seed-ensemble score. The leader is then bootstrapped
    against each of the next `limit` groups over the compounds they share, so
    the interval describes which compounds happened to land in the evaluation
    set and nothing else. Seed spread is a different quantity and stays in the
    panel above.

    Parameters
    ----------
    frame : DataFrame
        An annotated slice, carrying ``config_id`` and ``run_dir``.
    metric : str, optional
        Which metric to resample. Default is ``mae``.
    group : str, optional
        Column whose levels are compared, one best configuration apiece.
        Default is ``config_id``, which compares configurations directly.
    labels : dict, optional
        Configuration identifier mapped to the name to show. Defaults to the
        identifier itself.
    limit : int, optional
        How many runners-up to compare against the leader.
    n_resamples : int, optional
        Compound resamples behind each interval.
    seed : int, optional
        Seed for the resampling.

    Returns
    -------
    list of Comparison
        Ordered best rival first, empty when the slice holds fewer than two
        comparable configurations.
    """
    predictions = _config_predictions(frame)
    if len(predictions) < 2:
        return []

    scores = {
        config_id: evaluate.metrics(observed, predicted)[metric]
        for config_id, (_, observed, predicted) in predictions.items()
    }
    ordered = _rank_groups(frame, predictions, scores, group=group, metric=metric)
    if len(ordered) < 2:
        return []

    leader, *challengers = ordered
    leader_compounds, leader_observed, leader_predicted = predictions[leader]
    names = labels or {}

    comparisons: list[Comparison] = []
    for challenger in challengers[:limit]:
        compounds, _, predicted = predictions[challenger]
        if compounds != leader_compounds:
            logger.warning(
                "%s covers different compounds than the leader %s, so it is not bootstrapped",
                challenger,
                leader,
            )
            continue
        result = evaluate.paired_bootstrap(
            leader_observed,
            predicted,
            leader_predicted,
            metric=metric,
            n_resamples=n_resamples,
            seed=seed,
        )
        comparisons.append(
            Comparison(
                leader=leader,
                challenger=challenger,
                label=names.get(challenger, challenger),
                result=result,
                separates=evaluate.excludes_zero(result),
            )
        )
    return comparisons


def figure_1(spec: Manifest, frame: pd.DataFrame, metric: str = DEFAULT_METRIC) -> Figure:
    """Draw the graph-network baselines, one row per enumerated cell.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest, which owns the selection and the anchor.
    frame : DataFrame
        The tidy per-run table.
    metric : str, optional
        Which metric to plot. Default is ``mae``.

    Returns
    -------
    Figure
        The rendered figure, which the caller saves and closes.

    Raises
    ------
    FigureError
        If the figure's slice is empty.
    """
    return _spread_figure(spec, "fig1", frame, category="config_id", hue=None, metric=metric)


def figure_2(spec: Manifest, frame: pd.DataFrame, metric: str = DEFAULT_METRIC) -> Figure:
    """Draw each single feature block on its own, coloured by regressor.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    frame : DataFrame
        The tidy per-run table.
    metric : str, optional
        Which metric to plot. Default is ``mae``.

    Returns
    -------
    Figure
        The rendered figure.

    Raises
    ------
    FigureError
        If the figure's slice is empty.
    """
    return _spread_figure(
        spec, "fig2", frame, category="featureset", hue="regressor", metric=metric
    )


def figure_3(
    spec: Manifest,
    frame: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Figure:
    """Draw every featureset combination, with a paired bootstrap under it.

    This figure chooses a featureset, so the panel beneath the spread compares
    the leading configuration against its nearest rivals over the evaluation
    compounds. Where an interval spans zero the two are not separated by this
    evaluation set, and the figure says so rather than implying an order.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    frame : DataFrame
        The tidy per-run table.
    metric : str, optional
        Which metric to plot. Default is ``mae``.
    n_resamples : int, optional
        Compound resamples behind each interval.

    Returns
    -------
    Figure
        The rendered figure.

    Raises
    ------
    FigureError
        If the figure's slice is empty.
    """
    return _spread_figure(
        spec,
        "fig3",
        frame,
        category="featureset",
        hue="regressor",
        metric=metric,
        compare="config_id",
        n_resamples=n_resamples,
    )


def figure_4(
    spec: Manifest,
    frame: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Figure:
    """Draw one panel per regressor, with a paired bootstrap between them.

    This figure chooses a regressor, so the panel beneath compares each
    regressor's best configuration against the leading one over the evaluation
    compounds rather than reading a ranking off means that may not separate.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest, whose ``facet`` names the regressor axis.
    frame : DataFrame
        The tidy per-run table.
    metric : str, optional
        Which metric to plot. Default is ``mae``.
    n_resamples : int, optional
        Compound resamples behind each interval.

    Returns
    -------
    Figure
        The rendered figure.

    Raises
    ------
    FigureError
        If the figure's slice is empty.
    """
    return _spread_figure(
        spec,
        "fig4",
        frame,
        category="featureset",
        hue=None,
        metric=metric,
        compare="regressor",
        n_resamples=n_resamples,
    )


def figure_5(spec: Manifest, frame: pd.DataFrame, metric: str = DEFAULT_METRIC) -> Figure:
    """Draw one panel per descriptor PCA width over descriptor-bearing rows.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    frame : DataFrame
        The tidy per-run table.
    metric : str, optional
        Which metric to plot. Default is ``mae``.

    Returns
    -------
    Figure
        The rendered figure.

    Raises
    ------
    FigureError
        If the figure's slice is empty, which is the state the manifest
        describes when the settled configuration carries no descriptor block.
    """
    return _spread_figure(
        spec, "fig5", frame, category="featureset", hue="regressor", metric=metric
    )


def figure_5b(spec: Manifest, frame: pd.DataFrame, metric: str = DEFAULT_METRIC) -> Figure:
    """Draw one panel per embedding PCA width over embedding-only rows.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    frame : DataFrame
        The tidy per-run table.
    metric : str, optional
        Which metric to plot. Default is ``mae``.

    Returns
    -------
    Figure
        The rendered figure.

    Raises
    ------
    FigureError
        If the figure's slice is empty.
    """
    return _spread_figure(
        spec, "fig5b", frame, category="featureset", hue="regressor", metric=metric
    )


def figure_6(spec: Manifest, frame: pd.DataFrame, metric: str = DEFAULT_METRIC) -> Figure:
    """Draw the calibrated and uncalibrated arms side by side.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    frame : DataFrame
        The tidy per-run table.
    metric : str, optional
        Which metric to plot. Default is ``mae``.

    Returns
    -------
    Figure
        The rendered figure.

    Raises
    ------
    FigureError
        If the figure's slice is empty.
    """
    return _spread_figure(
        spec, "fig6", frame, category="featureset", hue="regressor", metric=metric
    )


def figure_7(spec: Manifest, frame: pd.DataFrame) -> Figure:
    """Plot predicted spread against realized absolute error, per compound.

    Predicted spread is per compound and lives in each run's predictions, so
    this figure reads the run directories rather than the tidy table. One panel
    is drawn per regressor that reports a spread, each labelled with the kind
    of quantity that regressor reports, because an exact predictive standard
    deviation and a rescaled percentile span are not interchangeable and
    pooling them would invent a comparison. Within a panel the leading
    configuration is shown, one colour per seed, against the line a perfectly
    calibrated normal spread would sit on.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest, which owns the selection.
    frame : DataFrame
        The tidy per-run table.

    Returns
    -------
    Figure
        The rendered figure.

    Raises
    ------
    FigureError
        If the figure's slice is empty, or no selected run reported a spread.
    """
    declaration = figure_declaration(spec, "fig7")
    data = slice_for(spec, "fig7", frame)
    if data.empty:
        raise FigureError("fig7: the slice is empty")

    points = uncertainty_points(data)
    if points.empty:
        raise FigureError("fig7: no selected run wrote a predicted_std column")

    regressor_names = sorted(points["regressor"].unique())
    fig, axes = plt.subplots(
        1,
        len(regressor_names),
        figsize=_size(width=2.0 + 4.0 * len(regressor_names), height=4.6),
        squeeze=False,
        layout="constrained",
    )
    seeds = sorted(points["seed"].unique())
    colors = _palette(len(seeds))

    for column, name in enumerate(regressor_names):
        ax = axes[0, column]
        panel = points.loc[points["regressor"] == name]
        for index, seed in enumerate(seeds):
            per_seed = panel.loc[panel["seed"] == seed]
            if per_seed.empty:
                continue
            rho = evaluate.metrics(per_seed["predicted_std"], per_seed["abs_error"])["spearman_rho"]
            ax.scatter(
                per_seed["predicted_std"],
                per_seed["abs_error"],
                s=9,
                alpha=0.45,
                color=colors[index],
                label=f"seed {seed} (rho {rho:.2f})",
                zorder=3,
            )
        _draw_calibration_line(ax, panel["predicted_std"])
        kind = str(panel["uncertainty_kind"].iloc[0])
        ax.set(
            title=f"{name}\n{kind}",
            xlabel="predicted spread",
            ylabel="absolute error" if column == 0 else "",
        )
        ax.legend(fontsize="x-small", loc="upper left")

    _title(fig, declaration)
    return fig


def uncertainty_points(data: pd.DataFrame) -> pd.DataFrame:
    """Read per-compound spread and error from the runs a slice names.

    Only the leading configuration of each regressor is read, since the
    manifest's uncertainty stage is a diagnostic on a settled configuration
    rather than a sweep, and pooling several configurations of one regressor
    would blur the relationship the figure is about.

    Parameters
    ----------
    data : DataFrame
        An annotated slice carrying ``config_id``, ``regressor``,
        ``uncertainty_kind``, ``seed`` and ``run_dir``.

    Returns
    -------
    DataFrame
        One row per compound and seed, carrying ``config_id``, ``regressor``,
        ``uncertainty_kind``, ``seed``, ``predicted_std`` and ``abs_error``.
        Empty when no selected run wrote a spread.
    """
    leaders = _leading_configs(data, group="regressor", metric=DEFAULT_METRIC)
    chunks: list[pd.DataFrame] = []
    for _, config_id in sorted(leaders.items()):
        rows = data.loc[data["config_id"] == config_id]
        for run_dir, regressor, kind, seed in zip(
            rows["run_dir"],
            rows["regressor"],
            rows["uncertainty_kind"],
            rows["seed"],
            strict=True,
        ):
            frame = read_predictions(Path(str(run_dir)))
            if "predicted_std" not in frame.columns:
                logger.warning("%s wrote no predicted_std, so it is left out of fig7", run_dir)
                continue
            chunks.append(
                pd.DataFrame(
                    {
                        "config_id": config_id,
                        "regressor": str(regressor),
                        "uncertainty_kind": str(kind),
                        "seed": int(seed),
                        "predicted_std": frame["predicted_std"].to_numpy(dtype=np.float64),
                        "abs_error": np.abs(
                            frame["predicted"].to_numpy(dtype=np.float64)
                            - frame["observed"].to_numpy(dtype=np.float64)
                        ),
                    }
                )
            )
    if not chunks:
        return pd.DataFrame(
            columns=[
                "config_id",
                "regressor",
                "uncertainty_kind",
                "seed",
                "predicted_std",
                "abs_error",
            ]
        )
    return pd.concat(chunks, ignore_index=True)


def render(
    spec: Manifest,
    figure_id: str,
    frame: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
    *,
    metric: str = DEFAULT_METRIC,
    n_resamples: int = DEFAULT_RESAMPLES,
    dpi: int = DEFAULT_DPI,
) -> Path | None:
    """Draw one figure and write it, or skip it when its slice is empty.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    figure_id : str
        Which figure to draw.
    frame : DataFrame
        The tidy per-run table.
    out_dir : path-like, optional
        Directory the image is written into, created if absent.
    metric : str, optional
        Which metric to plot. Default is ``mae``.
    n_resamples : int, optional
        Compound resamples behind the bootstrap panels of figures 3 and 4.
    dpi : int, optional
        Output resolution.

    Returns
    -------
    Path or None
        Where the image was written, or None when the figure was skipped.

    Raises
    ------
    KeyError
        If the manifest declares no such figure.
    """
    data = slice_for(spec, figure_id, frame)
    if data.empty:
        logger.warning("%s: no run matches its selection, so it is not drawn", figure_id)
        return None

    if figure_id == "fig7":
        figure = figure_7(spec, frame)
    elif figure_id in ("fig3", "fig4"):
        builder = _WITH_BOOTSTRAP[figure_id]
        figure = builder(spec, frame, metric, n_resamples=n_resamples)
    else:
        figure = _PLAIN[figure_id](spec, frame, metric)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{figure_id}{FILE_SUFFIX}"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    logger.info("%s: %d runs -> %s", figure_id, len(data), path)
    return path


def render_all(
    spec: Manifest,
    frame: pd.DataFrame,
    out_dir: Path = FIGURES_DIR,
    figure_ids: Iterable[str] | None = None,
    *,
    metric: str = DEFAULT_METRIC,
    n_resamples: int = DEFAULT_RESAMPLES,
    dpi: int = DEFAULT_DPI,
) -> tuple[dict[str, Path], list[Skipped]]:
    """Draw every figure the manifest declares, skipping those with no runs.

    Parameters
    ----------
    spec : Manifest
        The parsed manifest.
    frame : DataFrame
        The tidy per-run table.
    out_dir : path-like, optional
        Directory the images are written into.
    figure_ids : iterable of str, optional
        Which figures to draw. Defaults to every figure the manifest declares.
    metric : str, optional
        Which metric to plot. Default is ``mae``.
    n_resamples : int, optional
        Compound resamples behind the bootstrap panels.
    dpi : int, optional
        Output resolution.

    Returns
    -------
    written : dict of str to Path
        Figure identifier mapped to where it was written.
    skipped : list of Skipped
        The figures that were not drawn, each with its reason.
    """
    wanted = list(figure_ids) if figure_ids is not None else [f["id"] for f in spec.figures]
    written: dict[str, Path] = {}
    skipped: list[Skipped] = []

    for figure_id in wanted:
        try:
            path = render(
                spec,
                figure_id,
                frame,
                out_dir,
                metric=metric,
                n_resamples=n_resamples,
                dpi=dpi,
            )
        except FigureError as err:
            # a figure that cannot be drawn is reported and the rest still run,
            # since one empty slice is not a reason to withhold six figures
            skipped.append(Skipped(figure_id, str(err)))
            continue
        if path is None:
            skipped.append(Skipped(figure_id, "no run matches its selection"))
            continue
        written[figure_id] = path

    return written, skipped


def _spread_figure(
    spec: Manifest,
    figure_id: str,
    frame: pd.DataFrame,
    *,
    category: str,
    hue: str | None,
    metric: str,
    compare: str | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Figure:
    """Draw one figure's spread panels, and its bootstrap panel where it has one."""
    declaration = figure_declaration(spec, figure_id)
    data = slice_for(spec, figure_id, frame)
    if data.empty:
        raise FigureError(f"{figure_id}: the slice is empty")

    facet = declaration.get("facet")
    points = _points(data, category=category, hue=hue, facet=facet, metric=metric)
    marks = _marks(data, points, metric=metric)
    categories = _category_order(marks, metric=metric)
    hues = sorted(points["hue"].unique())
    facets = sorted(points["facet"].unique(), key=str)
    anchors = anchor_lines(spec, data, metric)

    comparisons: list[Comparison] = []
    if compare is not None:
        labels = dict(zip(data["config_id"], _comparison_labels(data, compare), strict=True))
        comparisons = paired_intervals(
            data, metric=metric, group=compare, labels=labels, n_resamples=n_resamples
        )

    fig, panels, comparison_ax = _panels(
        n_facets=len(facets),
        n_categories=len(categories),
        n_hues=len(hues),
        with_comparison=compare is not None,
    )

    for index, facet_value in enumerate(facets):
        ax = panels[index]
        _draw_spread(
            ax,
            points.loc[points["facet"] == facet_value],
            marks.loc[marks["facet"] == facet_value],
            categories=categories,
            hues=hues,
            anchors=anchors,
        )
        ax.set(xlabel=metric, title="" if facet is None else f"{facet} = {facet_value}")
        if index == 0:
            ax.set_yticks(range(len(categories)), labels=categories, fontsize="small")
    if len(hues) > 1 or anchors:
        panels[0].legend(fontsize="x-small", loc="best")

    if comparison_ax is not None:
        _draw_comparisons(comparison_ax, comparisons, metric=metric, compare=compare or "")

    _title(fig, declaration)
    return fig


def _points(
    data: pd.DataFrame,
    *,
    category: str,
    hue: str | None,
    facet: str | None,
    metric: str,
) -> pd.DataFrame:
    """Reshape a slice into the per-seed points a spread panel draws."""
    return pd.DataFrame(
        {
            "facet": ["" for _ in range(len(data))] if facet is None else data[facet].astype(str),
            "category": data[category].astype(str),
            "hue": [NO_HUE for _ in range(len(data))] if hue is None else data[hue].astype(str),
            "config_id": data["config_id"].astype(str),
            "seed": data["seed"],
            "value": pd.to_numeric(data[metric], errors="coerce"),
        }
    ).reset_index(drop=True)


def _marks(data: pd.DataFrame, points: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Summarize the per-seed points into a seed mean, a spread and an ensemble score."""
    grouped = (
        points.groupby(["facet", "category", "hue", "config_id"], dropna=False, observed=True)
        .agg(seed_mean=("value", "mean"), seed_std=("value", "std"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    ensembles = _ensemble_scores(data, metric=metric)
    return grouped.assign(ensemble=grouped["config_id"].map(ensembles))


def _config_predictions(data: pd.DataFrame) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    """Read each configuration's seed-ensembled predictions, skipping unreadable ones."""
    predictions: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for config_id, group in data.groupby("config_id", dropna=False, observed=True):
        run_dirs = [Path(str(value)) for value in group["run_dir"]]
        try:
            predictions[str(config_id)] = ensemble_predictions(run_dirs)
        except FigureError as err:
            # a configuration whose predictions are unreadable loses its
            # ensemble mark and its bootstrap, not the whole figure
            logger.warning("%s: %s", config_id, err)
    return predictions


def _ensemble_scores(data: pd.DataFrame, *, metric: str) -> dict[str, float]:
    """Score each configuration's seed-averaged predictions once."""
    return {
        config_id: evaluate.metrics(observed, predicted)[metric]
        for config_id, (_, observed, predicted) in _config_predictions(data).items()
    }


def _leading_configs(data: pd.DataFrame, *, group: str, metric: str) -> dict[str, str]:
    """Return the best-scoring configuration of each level of a grouping column."""
    scores = _ensemble_scores(data, metric=metric)
    lookup = dict(zip(data["config_id"].astype(str), data[group].astype(str), strict=True))
    best: dict[str, str] = {}
    for config_id, value in scores.items():
        if np.isnan(value):
            continue
        key = lookup[config_id]
        if key not in best or _is_better(value, scores[best[key]], metric=metric):
            best[key] = config_id
    return best


def _rank_groups(
    data: pd.DataFrame,
    predictions: dict[str, tuple[list[str], np.ndarray, np.ndarray]],
    scores: dict[str, float],
    *,
    group: str,
    metric: str,
) -> list[str]:
    """Order one configuration per group, best first, on the ensemble score."""
    lookup = dict(zip(data["config_id"].astype(str), data[group].astype(str), strict=True))
    best: dict[str, str] = {}
    for config_id in predictions:
        if np.isnan(scores[config_id]):
            continue
        key = lookup[config_id]
        if key not in best or _is_better(scores[config_id], scores[best[key]], metric=metric):
            best[key] = config_id
    return sorted(best.values(), key=lambda c: scores[c], reverse=metric not in LOWER_IS_BETTER)


def _is_better(candidate: float, incumbent: float, *, metric: str) -> bool:
    """Return whether one score beats another, given the metric's direction."""
    if metric in LOWER_IS_BETTER:
        return candidate < incumbent
    return candidate > incumbent


def _category_order(marks: pd.DataFrame, *, metric: str) -> list[str]:
    """Order the categories by their best seed mean, so the leader sits at the top."""
    best = marks.groupby("category", observed=True)["seed_mean"].min()
    if metric not in LOWER_IS_BETTER:
        best = -marks.groupby("category", observed=True)["seed_mean"].max()
    return [str(name) for name in best.sort_values().index]


def _comparison_labels(data: pd.DataFrame, compare: str) -> list[str]:
    """Name each row's configuration the way the bootstrap panel shows it."""
    if compare == "config_id":
        return [f"{f} / {r}" for f, r in zip(data["featureset"], data["regressor"], strict=True)]
    return [str(value) for value in data[compare]]


def _panels(
    *, n_facets: int, n_categories: int, n_hues: int, with_comparison: bool
) -> tuple[Figure, list[Axes], Axes | None]:
    """Lay out the spread panels, and the bootstrap panel beneath them."""
    height = 1.8 + ROW_HEIGHT * max(n_categories * n_hues, n_categories)
    if with_comparison:
        height += COMPARISON_HEIGHT
    figure = plt.figure(
        figsize=_size(width=LABEL_WIDTH + PANEL_WIDTH * n_facets, height=height),
        layout="constrained",
    )

    rows = 2 if with_comparison else 1
    ratios = [max(height - COMPARISON_HEIGHT, 1.0), COMPARISON_HEIGHT] if with_comparison else [1.0]
    grid = figure.add_gridspec(rows, n_facets, height_ratios=ratios)

    # facets share both axes: they are the same categories on the same metric,
    # and a panel drawn on its own scale would flatter whichever facet is worst
    panels: list[Axes] = []
    for index in range(n_facets):
        shared = panels[0] if panels else None
        panels.append(figure.add_subplot(grid[0, index], sharey=shared, sharex=shared))
        if shared is not None:
            panels[index].tick_params(labelleft=False)
    panels[0].invert_yaxis()

    comparison_ax = figure.add_subplot(grid[1, :]) if with_comparison else None
    return figure, panels, comparison_ax


def _draw_spread(
    ax: Axes,
    points: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    categories: Sequence[str],
    hues: Sequence[str],
    anchors: Sequence[tuple[str, float]],
) -> None:
    """Draw one panel: every seed as a point, the seed mean, and the ensemble."""
    positions = {name: index for index, name in enumerate(categories)}
    offsets = _offsets(len(hues))
    colors = _palette(len(hues))

    for index, hue in enumerate(hues):
        subset = points.loc[points["hue"] == hue]
        ax.scatter(
            subset["value"],
            subset["category"].map(positions) + offsets[index],
            s=13,
            alpha=0.7,
            color=colors[index],
            label=None if hue == NO_HUE else hue,
            zorder=3,
        )

        row = marks.loc[marks["hue"] == hue]
        y = row["category"].map(positions) + offsets[index]
        ax.errorbar(
            row["seed_mean"],
            y,
            xerr=row["seed_std"].fillna(0.0),
            fmt="|",
            markersize=9,
            color=colors[index],
            elinewidth=1.1,
            capsize=2,
            zorder=4,
        )
        ax.scatter(
            row["ensemble"],
            y,
            marker="*",
            s=55,
            facecolor="none",
            edgecolor=colors[index],
            linewidths=0.9,
            label="seed ensemble" if index == 0 else None,
            zorder=5,
        )

    for style, (label, value) in zip(("--", ":"), anchors, strict=False):
        ax.axvline(value, linestyle=style, linewidth=1.0, color="0.35", label=label, zorder=1)

    ax.set_yticks(range(len(categories)), labels=list(categories), fontsize="small")
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax.set_ylim(len(categories) - 0.5, -0.5)


def _draw_comparisons(
    ax: Axes, comparisons: Sequence[Comparison], *, metric: str, compare: str
) -> None:
    """Draw the paired-bootstrap intervals against the leading configuration."""
    if not comparisons:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "fewer than two comparable configurations, so no paired bootstrap is drawn",
            ha="center",
            va="center",
            fontsize="small",
        )
        return

    y = np.arange(len(comparisons))
    observed = np.array([c.result.observed for c in comparisons])
    lower = observed - np.array([c.result.ci_low for c in comparisons])
    upper = np.array([c.result.ci_high for c in comparisons]) - observed
    colors = ["#b2182b" if c.separates else "0.45" for c in comparisons]

    for index in range(len(comparisons)):
        ax.errorbar(
            observed[index],
            y[index],
            xerr=[[lower[index]], [upper[index]]],
            fmt="o",
            markersize=5,
            color=colors[index],
            capsize=3,
            elinewidth=1.2,
        )

    leader = comparisons[0].leader
    level = comparisons[0].result.level
    ax.axvline(0.0, color="0.2", linewidth=1.0)
    ax.set_yticks(y, labels=[c.label for c in comparisons], fontsize="small")
    ax.set_ylim(len(comparisons) - 0.5, -0.5)
    ax.set(xlabel=f"{metric} minus the leader's, paired over compounds ({level:.0%} interval)")
    ax.set_title(
        f"paired bootstrap across {compare}; an interval spanning zero does not separate\n"
        f"leader: {leader}",
        fontsize="small",
    )
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)


def _draw_calibration_line(ax: Axes, spread: pd.Series) -> None:
    """Draw where a perfectly calibrated normal predictive spread would sit."""
    values = pd.to_numeric(spread, errors="coerce").dropna()
    if values.empty:
        return
    grid = np.linspace(0.0, float(values.max()), 2)
    ax.plot(
        grid,
        CALIBRATED_SLOPE * grid,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
        label="calibrated normal",
        zorder=2,
    )


def _title(fig: Figure, declaration: dict[str, Any]) -> None:
    """Put the manifest's own title and question on the figure."""
    question = str(declaration.get("question", "")).strip()
    fig.suptitle(f"{declaration['title']}\n{question}", fontsize="medium")


def _featureset_columns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Derive the featureset label and the block count from the three feature axes."""
    columns = [
        frame[axis] if axis in frame else pd.Series(index=frame.index) for axis in FEATURE_AXES
    ]
    labels: list[Any] = []
    counts: list[Any] = []
    for values in zip(*(column.tolist() for column in columns), strict=True):
        present = [str(v) for v in values if not pd.isna(v) and str(v) != "none"]
        if all(pd.isna(v) for v in values):
            labels.append(pd.NA)
            counts.append(pd.NA)
            continue
        labels.append("+".join(present) if present else "none")
        counts.append(len(present))
    return (
        pd.Series(labels, index=frame.index, dtype="string"),
        pd.Series(counts, index=frame.index, dtype="Int64"),
    )


def _offsets(n_hues: int) -> np.ndarray:
    """Spread a category's hues apart within its band, so points do not overlap."""
    if n_hues == 1:
        return np.zeros(1)
    return np.linspace(-0.32, 0.32, n_hues)


def _palette(n_colors: int) -> list[Any]:
    """Return distinguishable categorical colours, cycling past ten levels."""
    colormap = matplotlib.colormaps["tab10"]
    return [colormap(index % colormap.N) for index in range(max(n_colors, 1))]


def _size(*, width: float, height: float) -> tuple[float, float]:
    """Clamp a figure's size to something a viewer will actually open.

    The lower bound on width is what keeps a single-facet figure's axis
    labels, which carry configuration slugs, from running off the canvas.
    """
    return (
        min(max(width, MIN_FIGURE_WIDTH), MAX_FIGURE_WIDTH),
        min(height, MAX_FIGURE_HEIGHT),
    )


# figures whose only argument is the metric, and those that add a bootstrap panel
_PLAIN = {
    "fig1": figure_1,
    "fig2": figure_2,
    "fig5": figure_5,
    "fig5b": figure_5b,
    "fig6": figure_6,
}
_WITH_BOOTSTRAP = {
    "fig3": figure_3,
    "fig4": figure_4,
}
