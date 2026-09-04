"""Load and check the experiment manifest, the rebuild's coverage spec.

The manifest at ``experiments/manifest.yaml`` names every configuration each
figure needs and every configuration deliberately not run. It carries no copied
metrics: a cell names a configuration in the prior-sweep summary, and the
numbers are joined here at load time so the spec and its evidence cannot drift
apart.

Loading validates the manifest's internal structure. The pruning rule (a cell
dropped for ranking below a rival must be outside the margin, against a rival
run with the same regressor) is checked separately by ``prune_violations``, so
a design question reads as a failing test rather than an import error.

Run directories follow from cell identifiers: cell ``fig3/chemeleon_readout``
at seed 2 lives in ``results/fig3/chemeleon_readout/seed2``. ``coverage``
compares the directories a sweep produced against the ones the manifest calls
for, so a missing run and an unplanned one are both visible.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("experiments/manifest.yaml")
RESULTS_DIR = Path("results")

# a cell is either run at every seed or recorded as deliberately not run
STATUSES = frozenset({"run", "pruned"})

# why a cell is not run, where the reason is not a ranking one; a prune without
# one of these has to point at a rival it lost to
REASON_KINDS = frozenset({"structural", "unavailable", "no_prior_metric"})

# what a cell is for, which decides whether the margin rule applies to it
ROLES = frozenset({"shortlist", "baseline", "reference", "coverage", "diagnostic"})

# roles that compete at a figure's gate, and so must justify their status
# against the margin
COMPETING_ROLES = frozenset({"shortlist"})


class ManifestError(ValueError):
    """The manifest is internally inconsistent or disagrees with its evidence."""


@dataclass(frozen=True)
class Cell:
    """One configuration the manifest either calls for or excludes.

    Attributes
    ----------
    id : str
        Identifier, ``<figure>/<name>``, which is also the run directory path.
    figure : str
        Identifier of the figure the cell belongs to.
    status : str
        ``run`` or ``pruned``.
    role : str
        What the cell is for; only ``shortlist`` cells face the margin rule.
    reason : str
        Prose justification, required on every cell.
    axes : dict or None
        The configuration to run. None for a cell that shares another's runs
        or is pruned.
    prior_config : str or None
        Configuration in the prior summary this cell's evidence comes from.
    compared_to : str or None
        Configuration in the prior summary the margin is measured against.
    covers : tuple of str
        Further prior configurations this cell accounts for, so that a single
        pruned cell can retire a whole family.
    reason_kind : str or None
        Non-ranking justification, one of ``REASON_KINDS``.
    shares_runs_with : str or None
        Identifier of the cell whose runs serve this one.
    """

    id: str
    figure: str
    status: str
    role: str
    reason: str
    axes: dict | None = None
    prior_config: str | None = None
    compared_to: str | None = None
    covers: tuple[str, ...] = ()
    reason_kind: str | None = None
    shares_runs_with: str | None = None

    @property
    def owns_runs(self) -> bool:
        """Whether this cell's runs are its own rather than another cell's."""
        return self.status == "run" and self.shares_runs_with is None

    def run_dir(self, seed: int, results_dir: Path = RESULTS_DIR) -> Path:
        """Return the directory a given seed of this cell writes to."""
        return results_dir / self.id / f"seed{seed}"


@dataclass(frozen=True)
class Figure:
    """A figure and the cells that make it up."""

    id: str
    title: str
    question: str
    gate: str | None
    cells: tuple[Cell, ...]
    depends_on_gate: str | None = None
    provisional_featureset: str | None = None


@dataclass(frozen=True)
class Coverage:
    """The difference between the runs a manifest calls for and those on disk."""

    expected: tuple[Path, ...]
    missing: tuple[Path, ...]
    unplanned: tuple[Path, ...]

    @property
    def is_complete(self) -> bool:
        """Whether every expected run exists and nothing unplanned does."""
        return not self.missing and not self.unplanned


@dataclass(frozen=True)
class Manifest:
    """The parsed coverage spec, joined to the prior sweep's evidence."""

    seeds: tuple[int, ...]
    margin_rae: float
    anchor: dict
    split: dict
    regressors: dict
    figures: tuple[Figure, ...]
    prior: pd.DataFrame = field(repr=False)

    @property
    def cells(self) -> dict[str, Cell]:
        """Every cell, keyed by identifier."""
        return {cell.id: cell for figure in self.figures for cell in figure.cells}

    def prior_metric(self, cell: Cell, metric: str = "rae_mean") -> float | None:
        """Return a cell's prior value for a metric, or None if it has none.

        Parameters
        ----------
        cell : Cell
            The cell whose prior evidence is wanted.
        metric : str, optional
            Column of the prior summary to read. Defaults to mean RAE.

        Returns
        -------
        float or None
            The prior value, or None when the cell names no prior configuration
            or that configuration produced no metrics.
        """
        if cell.prior_config is None:
            return None
        value = self.prior.at[cell.prior_config, metric]
        if pd.isna(value):
            return None
        # pandas types a cell as Scalar, which nominally includes complex
        return float(value)  # pyright: ignore[reportArgumentType]

    def planned_runs(self, results_dir: Path = RESULTS_DIR) -> list[Path]:
        """Return every run directory the manifest calls for, in cell order."""
        return [
            cell.run_dir(seed, results_dir)
            for cell in self.cells.values()
            if cell.owns_runs
            for seed in self.seeds
        ]

    def coverage(self, produced: Iterable[Path], results_dir: Path = RESULTS_DIR) -> Coverage:
        """Compare produced run directories against the ones planned.

        Parameters
        ----------
        produced : iterable of Path
            Run directories a sweep actually wrote.
        results_dir : path-like, optional
            Root the planned directories hang off. Defaults to ``results``.

        Returns
        -------
        Coverage
            The expected set, those missing from it, and those produced without
            being planned.
        """
        expected = self.planned_runs(results_dir)
        expected_set = set(expected)
        produced_set = set(produced)
        return Coverage(
            expected=tuple(expected),
            missing=tuple(p for p in expected if p not in produced_set),
            unplanned=tuple(sorted(produced_set - expected_set)),
        )


def load(
    path: Path = MANIFEST_PATH,
    prior_summary: Path | None = None,
) -> Manifest:
    """Read the manifest and join it to the prior-sweep summary.

    Parameters
    ----------
    path : path-like, optional
        The manifest YAML. Defaults to ``experiments/manifest.yaml``.
    prior_summary : path-like, optional
        The prior evidence CSV. Defaults to the path the manifest names,
        resolved relative to the manifest's parent's parent.

    Returns
    -------
    Manifest
        The validated spec.

    Raises
    ------
    ManifestError
        If the manifest is internally inconsistent, or names a prior
        configuration the summary does not contain.
    """
    raw = yaml.safe_load(path.read_text())

    summary_path = prior_summary or (path.parent.parent / raw["prior"]["summary"])
    prior = pd.read_csv(summary_path).set_index("config")
    if not prior.index.is_unique:
        raise ManifestError(f"{summary_path}: duplicate configurations in the prior summary")

    figures = tuple(_parse_figure(entry) for entry in raw["figures"])
    manifest = Manifest(
        seeds=tuple(raw["seeds"]),
        margin_rae=float(raw["prior"]["shortlist_margin_rae"]),
        anchor=raw["anchor"],
        split=raw["split"],
        regressors=raw["regressors"],
        figures=figures,
        prior=prior,
    )
    _validate(manifest)
    return manifest


def prune_violations(manifest: Manifest) -> list[str]:
    """Return every place the manifest's pruning rule is broken.

    A shortlist cell that is run must be within the margin of the rival it
    names, and one that is pruned must be outside it, unless a non-ranking
    reason kind excuses it. Both sides of a comparison must have been run with
    the same regressor, since a cross-regressor gap confounds the axis under
    test.

    Parameters
    ----------
    manifest : Manifest
        The spec to check.

    Returns
    -------
    list of str
        One message per violation; empty when the manifest obeys its own rule.
    """
    violations: list[str] = []
    for cell in manifest.cells.values():
        if cell.role not in COMPETING_ROLES or cell.reason_kind in REASON_KINDS:
            continue

        if cell.compared_to is None:
            violations.append(f"{cell.id}: shortlist cell names no rival to compare against")
            continue

        rival = Cell(
            id=cell.compared_to,
            figure=cell.figure,
            status="run",
            role="shortlist",
            reason="",
            prior_config=cell.compared_to,
        )
        own = manifest.prior_metric(cell)
        against = manifest.prior_metric(rival)
        if own is None or against is None:
            violations.append(f"{cell.id}: comparison needs metrics on both sides")
            continue

        # a cross-regressor gap confounds the axis under test with the regressor
        own_group = _comparison_group(manifest, str(cell.prior_config))
        rival_group = _comparison_group(manifest, cell.compared_to)
        if own_group != rival_group:
            violations.append(
                f"{cell.id}: compares {own_group} against {rival_group}; "
                "a prune must be measured within one regressor"
            )
            continue

        margin = own - against
        if cell.status == "run" and margin > manifest.margin_rae:
            violations.append(
                f"{cell.id}: run as a shortlist cell but {margin:.4f} RAE behind "
                f"{cell.compared_to}, outside the {manifest.margin_rae} margin"
            )
        if cell.status == "pruned" and margin <= manifest.margin_rae:
            violations.append(
                f"{cell.id}: pruned but only {margin:.4f} RAE behind {cell.compared_to}, "
                f"inside the {manifest.margin_rae} margin"
            )
    return violations


def _comparison_group(manifest: Manifest, config: str) -> str:
    """Return the group a prior configuration may be compared within.

    Graph-network runs record no regressor, so a null reads as its own group
    rather than as unequal to itself.
    """
    kind = manifest.prior.at[config, "model_kind"]
    regressor = manifest.prior.at[config, "regressor"]
    return f"{kind}:{'none' if pd.isna(regressor) else regressor}"


def prior_configs_claimed(manifest: Manifest) -> set[str]:
    """Return every prior configuration some cell accounts for."""
    claimed: set[str] = set()
    for cell in manifest.cells.values():
        if cell.prior_config is not None:
            claimed.add(cell.prior_config)
        claimed.update(cell.covers)
    return claimed


def _parse_figure(entry: dict) -> Figure:
    """Build one figure and its cells from the manifest's YAML."""
    cells = tuple(_parse_cell(entry["id"], cell) for cell in entry["cells"])
    return Figure(
        id=entry["id"],
        title=entry["title"],
        question=entry["question"],
        gate=entry.get("gate"),
        cells=cells,
        depends_on_gate=entry.get("depends_on_gate"),
        provisional_featureset=entry.get("provisional_featureset"),
    )


def _parse_cell(figure_id: str, entry: dict) -> Cell:
    """Build one cell from the manifest's YAML."""
    prior = entry.get("prior") or {}
    return Cell(
        id=entry["id"],
        figure=figure_id,
        status=entry["status"],
        role=entry["role"],
        reason=entry["reason"],
        axes=entry.get("axes"),
        prior_config=prior.get("config"),
        compared_to=prior.get("compared_to"),
        covers=tuple(entry.get("covers", ())),
        reason_kind=entry.get("reason_kind"),
        shares_runs_with=entry.get("shares_runs_with"),
    )


def _validate(manifest: Manifest) -> None:
    """Check the manifest's structure, raising on the first problem found."""
    cells = manifest.cells
    for problem in _structural_problems(manifest, cells):
        raise ManifestError(problem)


def _structural_problems(manifest: Manifest, cells: dict[str, Cell]) -> Iterator[str]:
    """Yield every structural problem in the manifest, in cell order."""
    seen: set[str] = set()
    for figure in manifest.figures:
        if figure.provisional_featureset and figure.provisional_featureset not in cells:
            yield (
                f"{figure.id}: provisional featureset {figure.provisional_featureset} is not a cell"
            )

        for cell in figure.cells:
            if cell.id in seen:
                yield f"{cell.id}: duplicate cell identifier"
            seen.add(cell.id)

            if not cell.id.startswith(f"{figure.id}/"):
                yield f"{cell.id}: identifier does not sit under {figure.id}"
            if cell.status not in STATUSES:
                yield f"{cell.id}: unknown status {cell.status!r}"
            if cell.role not in ROLES:
                yield f"{cell.id}: unknown role {cell.role!r}"
            if cell.reason_kind is not None and cell.reason_kind not in REASON_KINDS:
                yield f"{cell.id}: unknown reason kind {cell.reason_kind!r}"
            if not cell.reason.strip():
                yield f"{cell.id}: no reason given"

            # a cell that owns its runs has to say what to run
            if cell.owns_runs and not cell.axes:
                yield f"{cell.id}: runs its own seeds but declares no axes"
            if cell.status == "pruned" and cell.axes:
                yield f"{cell.id}: pruned cells declare no axes"

            yield from _sharing_problems(cell, cells)
            yield from _evidence_problems(manifest, cell)


def _sharing_problems(cell: Cell, cells: dict[str, Cell]) -> Iterator[str]:
    """Yield problems with a cell's claim to reuse another cell's runs.

    Sharing chains: a figure 6 arm can point at figure 4's winner, which itself
    points at the figure 3 cell that owns the runs. The chain is followed to
    whichever cell owns runs, and a cycle is reported rather than looped on.
    """
    if cell.shares_runs_with is None:
        return

    seen = [cell.id]
    current = cell
    while current.shares_runs_with is not None:
        target = cells.get(current.shares_runs_with)
        if target is None:
            yield f"{cell.id}: shares runs with {current.shares_runs_with}, which is not a cell"
            return
        if target.id in seen:
            yield f"{cell.id}: sharing chain cycles through {' -> '.join([*seen, target.id])}"
            return
        seen.append(target.id)
        current = target

    if not current.owns_runs:
        yield f"{cell.id}: sharing chain ends at {current.id}, which owns no runs"


def _evidence_problems(manifest: Manifest, cell: Cell) -> Iterator[str]:
    """Yield problems with the prior configurations a cell names."""
    named = [c for c in (cell.prior_config, cell.compared_to, *cell.covers) if c is not None]
    for config in named:
        if config not in manifest.prior.index:
            yield f"{cell.id}: prior configuration {config!r} is not in the prior summary"
