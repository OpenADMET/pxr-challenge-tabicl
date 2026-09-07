"""The comparison intervals the figures draw, and what they assume.

The contract these hold is that unblocked Tukey is statsmodels' Tukey and not
a reconstruction of it, that the blocked variant differs from it only in the
error term, and that the geometry keeps the property the figure is read under:
two intervals touch exactly when the procedure does not separate the pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multicomp import MultiComparison

import tukey

SEEDS = [0, 1, 2, 3, 4]


@pytest.fixture
def panel() -> dict[str, list[float]]:
    """Return a balanced panel with a clear leader and a spread of seed variances."""
    rng = np.random.default_rng(11)
    return {
        f"c{i}": list(rng.normal(0.45 + 0.02 * i, 0.004 + 0.006 * (i % 3), len(SEEDS)))
        for i in range(9)
    }


def _statsmodels(panel: dict[str, list[float]]) -> tuple[dict[str, bool], dict[str, float], float]:
    """Run the same panel through statsmodels, returning verdicts against the leader."""
    long = pd.DataFrame(
        [{"cfg": slug, "mae": value} for slug, values in panel.items() for value in values]
    )
    result = MultiComparison(long["mae"].to_numpy(), long["cfg"].to_numpy()).tukeyhsd(alpha=0.05)
    groups = list(result.groupsunique)
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1 :]]
    leader = long.groupby("cfg")["mae"].mean().idxmin()
    reject, p_values = {}, {}
    for (a, b), is_rejected, p_value in zip(pairs, result.reject, result.pvalues, strict=True):
        if leader in (a, b):
            other = b if a == leader else a
            reject[other], p_values[other] = bool(is_rejected), float(p_value)
    return reject, p_values, float(result.q_crit)


def test_unblocked_is_statsmodels(panel: dict[str, list[float]]) -> None:
    """Unblocked, every verdict and p-value is statsmodels' own to floating point."""
    measured = tukey.compare(panel, SEEDS, block_by_seed=False)
    reject, p_values, q_crit = _statsmodels(panel)

    assert measured["q_crit"] == pytest.approx(q_crit)
    for row in measured["against_leader"]:
        assert row["separated"] == reject[row["slug"]]
        assert row["p_value"] == pytest.approx(p_values[row["slug"]], abs=1e-12)


def test_blocking_changes_only_the_error_term(panel: dict[str, list[float]]) -> None:
    """The two variants share their ranking and differ in the critical distance."""
    blocked = tukey.compare(panel, SEEDS, block_by_seed=True)
    pooled = tukey.compare(panel, SEEDS, block_by_seed=False)

    assert blocked["leader_slug"] == pooled["leader_slug"]
    assert blocked["diagnostics"]["df_resid"] < pooled["diagnostics"]["df_resid"]
    assert blocked["hsd"] != pooled["hsd"]


def test_overlap_is_the_test(panel: dict[str, list[float]]) -> None:
    """A row's interval reaches the leader's exactly when the pair is not separated.

    This is the property the figure is read under, and the reason the panel has
    to be balanced: one critical distance for every pair is what makes it split
    evenly between the two intervals it joins.
    """
    measured = tukey.compare(panel, SEEDS, block_by_seed=True)
    means = {slug: float(np.mean(values)) for slug, values in panel.items()}
    leader = measured["leader_slug"]
    halfwidth = measured["halfwidths"][leader]

    for row in measured["against_leader"]:
        gap = means[row["slug"]] - means[leader]
        overlaps = gap <= halfwidth + measured["halfwidths"][row["slug"]]
        assert overlaps is not row["separated"]


def test_every_row_carries_a_halfwidth(panel: dict[str, list[float]]) -> None:
    """Including the leader, which the bootstrap has no difference to give one."""
    measured = tukey.compare(panel, SEEDS, block_by_seed=True)

    assert set(measured["halfwidths"]) == set(panel)
    assert len(set(measured["halfwidths"].values())) == 1


def test_unbalanced_panel_is_refused() -> None:
    """Tukey's constant critical distance is what a balanced panel buys."""
    with pytest.raises(tukey.TukeyError, match="balanced"):
        tukey.compare({"a": [0.5, 0.6], "b": [0.4]}, SEEDS[:2])


def test_single_configuration_is_refused() -> None:
    """There is nothing to compare, which is a mistake rather than an empty result."""
    with pytest.raises(tukey.TukeyError, match="at least two"):
        tukey.compare({"a": [0.5] * len(SEEDS)}, SEEDS)


def test_retest_refuses_a_record_without_per_seed() -> None:
    """A record predating the field cannot be redrawn from, and does not guess."""
    evidence = {"ranking": [{"slug": "a", "mae": 0.5}, {"slug": "b", "mae": 0.6}]}
    with pytest.raises(tukey.TukeyError, match="per-seed"):
        tukey.retest(evidence, SEEDS)


def test_retest_replaces_the_verdicts_and_keeps_the_ranking(
    panel: dict[str, list[float]],
) -> None:
    """The figure's verdicts change; what the record says was run does not."""
    ranking = [
        {"slug": slug, "mae": float(np.mean(values)), "per_seed": values, "n_blocks": 2}
        for slug, values in panel.items()
    ]
    evidence = {
        "ranking": ranking,
        "leader_slug": "c0",
        "indistinguishable_from_leader": ["c8"],
        "significance": {"comparison_intervals": {}},
    }
    measured = tukey.retest(evidence, SEEDS)

    assert measured["measured_by"] == tukey.METHOD
    assert measured["ranking"] == ranking
    assert measured["significance"]["method"] == tukey.METHOD
    tied = set(measured["indistinguishable_from_leader"])
    assert tied == set(panel) - {measured["leader_slug"]} - set(measured["separated_from_leader"])


def test_diagnostics_report_what_the_intervals_assume(panel: dict[str, list[float]]) -> None:
    """A panel carries its own assumption checks rather than leaving them implicit."""
    diagnostics = tukey.compare(panel, SEEDS, block_by_seed=True)["diagnostics"]

    assert diagnostics["groups"] == len(panel)
    assert diagnostics["blocks"] == len(SEEDS)
    assert diagnostics["variance_ratio"] > 1
    for name in ("block_p", "nonadditivity_p", "levene_p"):
        assert 0.0 <= diagnostics[name] <= 1.0
