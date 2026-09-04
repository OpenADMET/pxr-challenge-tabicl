"""Load the experiment manifest and expand its axes into configurations.

The manifest at ``experiments/manifest.yaml`` declares what the rebuild sweeps:
the tabular feature-block axes, the regressors, and the stages that sweep them.
It is a declaration rather than an enumeration because the first stage alone
crosses 39 featuresets with 7 regressors, and writing 273 configurations out by
hand would be a worse description than the cross product that generates them.

Two rules the expansion enforces, which are not the same rule. No level of any
axis is dropped on the prior sweep's evidence, because the prior scored a
different compound set and cannot rank anything on this split. But dimensions
are swept one stage at a time, each stage holding fixed what an earlier gate
already settled here. The first is about what counts as evidence; the second is
only about compute.

A configuration's identity is its slug, which is also its run directory, so a
run can be found by reading its configuration and a directory can be read back
into one. The prior summary is joined for context and to check that no
configuration family the previous generation explored has been forgotten; it
decides nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("experiments/manifest.yaml")
RESULTS_DIR = Path("results")

# the axes a tabular configuration is made of, in slug order
TABULAR_AXES = ("embedding", "readout", "descriptors", "descriptor_pca", "regressor", "calibration")


class ManifestError(ValueError):
    """The manifest is internally inconsistent or disagrees with its evidence."""


@dataclass(frozen=True)
class TabularConfig:
    """One tabular configuration: which blocks, reduced how, fitted by what."""

    embedding: str
    readout: str
    descriptors: str
    descriptor_pca: int
    regressor: str
    calibration: str

    @property
    def has_descriptors(self) -> bool:
        """Whether this configuration carries a descriptor block."""
        return self.descriptors != "none"

    @property
    def n_blocks(self) -> int:
        """How many feature blocks the configuration draws on."""
        present = [self.embedding != "none", self.readout != "none", self.has_descriptors]
        return sum(present)

    @property
    def slug(self) -> str:
        """A readable, deterministic identifier, used as the run directory name."""
        descriptors = (
            self.descriptors
            if not self.has_descriptors
            else (f"{self.descriptors}{self.descriptor_pca}")
        )
        return (
            f"emb-{self.embedding}__ro-{self.readout}__desc-{descriptors}"
            f"__reg-{self.regressor}__cal-{self.calibration}"
        )

    def run_dir(self, seed: int, results_dir: Path = RESULTS_DIR) -> Path:
        """Where this configuration writes a given seed's run."""
        return results_dir / "tabular" / self.slug / f"seed{seed}"

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as plain data, for a cache key or a record."""
        return {axis: getattr(self, axis) for axis in TABULAR_AXES}


@dataclass(frozen=True)
class GnnCell:
    """One graph-network configuration, enumerated rather than crossed."""

    id: str
    title: str
    axes: dict[str, Any]
    prior: str | None = None

    def run_dir(self, seed: int, results_dir: Path = RESULTS_DIR) -> Path:
        """Where this cell writes a given seed's run."""
        return results_dir / "gnn" / self.id / f"seed{seed}"


@dataclass(frozen=True)
class Gate:
    """A decision taken from a completed stage, fixing axes for later stages."""

    id: str
    chooses: tuple[str, ...]
    rule: str


@dataclass(frozen=True)
class Stage:
    """One sweep: the axes it varies, and what it holds fixed while doing so."""

    id: str
    figures: tuple[str, ...]
    sweeps: tuple[str, ...]
    fixed: dict[str, Any] = field(default_factory=dict)
    fixed_from: str | None = None
    gate: Gate | None = None
    reuses_runs: bool = False
    note: str | None = None


@dataclass(frozen=True)
class Coverage:
    """The difference between the runs a stage calls for and those on disk."""

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
    anchor: dict
    split: dict
    axes: dict
    stages: tuple[Stage, ...]
    gnn_cells: tuple[GnnCell, ...]
    figures: tuple[dict, ...]
    prior: pd.DataFrame = field(repr=False)

    def stage(self, stage_id: str) -> Stage:
        """Return one stage by identifier."""
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        raise KeyError(f"no stage {stage_id!r}; known: {[s.id for s in self.stages]}")

    def expand(self, stage_id: str, resolved: dict[str, Any] | None = None) -> list[TabularConfig]:
        """Return every configuration a stage runs.

        Parameters
        ----------
        stage_id : str
            Which stage to expand.
        resolved : dict, optional
            Axis values an earlier gate settled. Required for a stage that
            names ``fixed_from``, since it cannot know what to hold fixed
            until the gate before it has been decided.

        Returns
        -------
        list of TabularConfig
            The stage's configurations, in a deterministic order.

        Raises
        ------
        ManifestError
            If the stage depends on a gate whose choices were not supplied.
        """
        stage = self.stage(stage_id)
        if stage.reuses_runs:
            return []

        settled = dict(resolved or {})
        if stage.fixed_from is not None and not settled:
            raise ManifestError(
                f"{stage_id}: fixed_from {stage.fixed_from!r} means this stage needs the axes "
                "that gate chose, but none were supplied"
            )

        levels: dict[str, list[Any]] = {}
        for axis in TABULAR_AXES:
            if axis in stage.sweeps:
                levels[axis] = list(self._levels(axis))
            elif axis in stage.fixed:
                levels[axis] = [stage.fixed[axis]]
            elif axis in settled:
                levels[axis] = [settled[axis]]
            else:
                raise ManifestError(
                    f"{stage_id}: axis {axis!r} is neither swept, fixed, nor settled"
                )

        configs = [
            TabularConfig(**dict(zip(TABULAR_AXES, values, strict=True)))
            for values in product(*(levels[axis] for axis in TABULAR_AXES))
        ]
        # a width sweep over a featureset with no descriptors would repeat one
        # configuration under three names
        return [c for c in _deduplicate(configs) if c.n_blocks > 0]

    def planned_runs(
        self,
        stage_id: str,
        resolved: dict[str, Any] | None = None,
        results_dir: Path = RESULTS_DIR,
    ) -> list[Path]:
        """Return every run directory a stage calls for."""
        return [
            config.run_dir(seed, results_dir)
            for config in self.expand(stage_id, resolved)
            for seed in self.seeds
        ]

    def gnn_runs(self, results_dir: Path = RESULTS_DIR) -> list[Path]:
        """Return every run directory the graph-network cells call for."""
        return [cell.run_dir(seed, results_dir) for cell in self.gnn_cells for seed in self.seeds]

    def coverage(self, expected: Iterable[Path], produced: Iterable[Path]) -> Coverage:
        """Compare produced run directories against those expected."""
        expected = list(expected)
        expected_set = set(expected)
        produced_set = set(produced)
        return Coverage(
            expected=tuple(expected),
            missing=tuple(p for p in expected if p not in produced_set),
            unplanned=tuple(sorted(produced_set - expected_set)),
        )

    def _levels(self, axis: str) -> Iterator[Any]:
        """Yield every level of an axis, whether it is a mapping or a list."""
        values = self.axes[axis]
        yield from (values.keys() if isinstance(values, dict) else values)


def load(path: Path = MANIFEST_PATH, prior_summary: Path | None = None) -> Manifest:
    """Read the manifest and join it to the prior-sweep summary.

    Parameters
    ----------
    path : path-like, optional
        The manifest YAML.
    prior_summary : path-like, optional
        The prior evidence CSV. Defaults to the path the manifest names.

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

    tabular = raw["tabular"]
    axes = dict(tabular["axes"])
    stages = tuple(_parse_stage(entry) for entry in tabular["stages"])
    gnn_cells = tuple(_parse_gnn(raw["gnn"]))

    manifest = Manifest(
        seeds=tuple(raw["seeds"]),
        anchor=raw["anchor"],
        split=raw["split"],
        axes=axes,
        stages=stages,
        gnn_cells=gnn_cells,
        figures=tuple(raw["figures"]),
        prior=prior,
    )
    _validate(manifest)
    return manifest


def uncovered_prior_levels(manifest: Manifest) -> dict[str, set[str]]:
    """Return prior axis levels the manifest's vocabulary cannot express.

    The prior sweep decides nothing, but a family it explored that this grid
    cannot even name would be an oversight rather than a decision. This reports
    any such gap per axis, so an empty result means nothing was forgotten.

    Parameters
    ----------
    manifest : Manifest
        The spec to check.

    Returns
    -------
    dict of str to set of str
        Axis name mapped to the prior levels it cannot express. Empty when the
        vocabulary covers everything the prior ran.
    """
    prior = manifest.prior
    gaps: dict[str, set[str]] = {}

    # descriptor sources: the prior wrote a combined block as "all"
    prior_sources = {str(s) for s in prior["descriptor_sources"].dropna().unique() if s}
    descriptor_translation = {"all": "rdkit_mordred"}
    ours = {"mordred", "rdkit", "rdkit_mordred"}
    missing_sources = {descriptor_translation.get(s, s) for s in prior_sources} - ours
    if missing_sources:
        gaps["descriptors"] = missing_sources

    # regressors: the prior's bare "tabpfn" was the v2.5 checkpoint, and its
    # three TabFM ensemble sizes were all recorded under one name
    prior_regressors = {str(r) for r in prior["regressor"].dropna().unique() if r != "N/A"}
    regressor_translation = {"tabpfn": "tabpfn-v2.5"}
    declared = {str(r) for r in manifest.axes["regressor"]}
    missing_regressors = {
        name
        for name in (regressor_translation.get(r, r) for r in prior_regressors)
        if name not in declared and not name.startswith("tabfm")
    }
    if missing_regressors:
        gaps["regressor"] = missing_regressors

    return gaps


def _deduplicate(configs: Iterable[TabularConfig]) -> list[TabularConfig]:
    """Drop configurations whose slugs coincide, keeping the first of each."""
    seen: dict[str, TabularConfig] = {}
    for config in configs:
        seen.setdefault(config.slug, config)
    return list(seen.values())


def _parse_stage(entry: dict) -> Stage:
    """Build one stage from the manifest's YAML."""
    gate = entry.get("gate")
    return Stage(
        id=entry["id"],
        figures=tuple(entry.get("figures", ())),
        sweeps=tuple(entry.get("sweeps", ())),
        fixed=dict(entry.get("fixed") or {}),
        fixed_from=entry.get("fixed_from"),
        gate=None
        if not gate
        else Gate(id=gate["id"], chooses=tuple(gate["chooses"]), rule=gate["rule"]),
        reuses_runs=bool(entry.get("reuses_runs", False)),
        note=entry.get("note"),
    )


def _parse_gnn(entry: dict) -> Iterator[GnnCell]:
    """Yield every graph-network cell, expanding the declared sub-grids."""
    for grid in entry.get("grids", ()):
        axis_names = list(grid["axes"])
        short = grid.get("slug_names", {})
        for values in product(*(grid["axes"][name] for name in axis_names)):
            varied = dict(zip(axis_names, values, strict=True))
            slug = "_".join(
                f"{short.get(name, name)}{_slugify(value)}" for name, value in varied.items()
            )
            yield GnnCell(
                id=f"{grid['id']}_{slug}",
                title=grid["title"],
                axes={**grid["fixed"], **varied},
            )
    for cell in entry.get("cells", ()):
        yield GnnCell(
            id=cell["id"], title=cell["title"], axes=cell["axes"], prior=cell.get("prior")
        )


def _slugify(value: Any) -> str:
    """Render an axis value for a run directory name."""
    if value is None:
        return "off"
    return str(value).replace(".", "p")


def _validate(manifest: Manifest) -> None:
    """Check the manifest's structure, raising on the first problem found."""
    for problem in _problems(manifest):
        raise ManifestError(problem)


def _problems(manifest: Manifest) -> Iterator[str]:
    """Yield every structural problem in the manifest."""
    gates = {stage.gate.id for stage in manifest.stages if stage.gate}
    figures = {figure["id"] for figure in manifest.figures}

    for stage in manifest.stages:
        if stage.fixed_from is not None and stage.fixed_from not in gates:
            yield f"{stage.id}: fixed_from names {stage.fixed_from!r}, which no stage gates"
        for axis in (*stage.sweeps, *stage.fixed):
            if axis not in TABULAR_AXES:
                yield f"{stage.id}: unknown axis {axis!r}"
        for figure in stage.figures:
            if figure not in figures:
                yield f"{stage.id}: names figure {figure!r}, which the manifest does not declare"
        overlap = set(stage.sweeps) & set(stage.fixed)
        if overlap:
            yield f"{stage.id}: {sorted(overlap)} is both swept and fixed"

    seen: set[str] = set()
    for cell in manifest.gnn_cells:
        if cell.id in seen:
            yield f"{cell.id}: duplicate graph-network cell identifier"
        seen.add(cell.id)
        if cell.prior is not None and cell.prior not in manifest.prior.index:
            yield f"{cell.id}: prior configuration {cell.prior!r} is not in the prior summary"
