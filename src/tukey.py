"""Tukey HSD comparison intervals for the figures, blocked on the seed.

The gates decide under Benjamini-Hochberg on a paired compound bootstrap, and
they keep deciding that way: this module does not touch them. What it supplies
is the geometry and the verdicts the panels draw, under the procedure the
comparison plot was designed around.

The two answer different questions and will not always agree. Tukey here treats
the five seeds as the replicates and the phase 2 test set as fixed, so it asks
whether a different training seed would reorder the table. The bootstrap treats
the fitted models as fixed and resamples compounds, so it asks whether a
different sample of molecules would. A row can be separated under one and tied
under the other without either being wrong, and a caption that shows one has to
say which.

Seed is a block rather than a nuisance to pool away. The seed sets the
validation carve-out for early stopping, and carve-out ``s`` is the same
partition for every configuration in a panel, which makes the design a balanced
randomized complete block. Analysing it as blocked is what the design licenses,
and dropping the block term after testing it would be a model selection that
invalidates the comparisons that follow, so ``block_by_seed`` is not chosen per
panel from its own diagnostics. It is a flag so that the unblocked analysis,
which is exactly what :func:`statsmodels.stats.multicomp.pairwise_tukeyhsd`
computes, stays one toggle away for comparison.

Blocking is not free. It assumes no configuration-by-seed interaction, which
the unblocked analysis never has to assume, and with one observation per cell
that assumption is only reachable through Tukey's one degree of freedom test
for nonadditivity. Both analyses assume the per-configuration seed variances
are equal, which in these panels they are not, by ratios from nine to two
hundred. Neither assumption is repaired here. Both are measured, reported on
:class:`Diagnostics`, and recorded with the figure, so what the intervals rest
on travels with them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from scipy.stats import studentized_range
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

logger = logging.getLogger(__name__)

# bumped when the meaning of an interval changes, never for a cosmetic edit
VERSION = 1

METHOD = "tukey-hsd"

# the familywise error rate the intervals and the verdicts are taken at
ALPHA = 0.05


class TukeyError(RuntimeError):
    """Raised when a panel cannot carry a Tukey comparison."""


@dataclass(frozen=True)
class Diagnostics:
    """What the intervals assume, measured on the panel that produced them.

    Attributes
    ----------
    groups, blocks : int
        Configurations compared, and seeds each was run at.
    mse, df_resid : float
        The error term the critical distance was taken from.
    hsd : float
        Tukey's critical distance. Two configurations are separated when their
        means differ by more than this.
    block_f, block_p : float
        The seed term's F test. A large p does not license dropping the block,
        which the design fixed; it says the panel is too small to resolve it.
    nonadditivity_p : float
        Tukey's one degree of freedom test. Small means a configuration by seed
        interaction the blocked model assumes away, and is a reason to read the
        unblocked figure beside this one.
    levene_p : float
        Equality of the per-configuration seed variances. Small means the
        pooled error term is too wide for the steady rows and too narrow for
        the noisy ones, which is inherited from Tukey rather than from blocking.
    variance_ratio : float
        Largest per-configuration seed variance over the smallest, which says
        how far the pooling has to stretch.
    """

    groups: int
    blocks: int
    mse: float
    df_resid: float
    hsd: float
    block_f: float
    block_p: float
    nonadditivity_p: float
    levene_p: float
    variance_ratio: float


def _long(per_seed: dict[str, list[float]], seeds: list[int]) -> pd.DataFrame:
    """Return the panel as one row per configuration and seed, checking balance."""
    widths = {slug: len(values) for slug, values in per_seed.items()}
    if not widths:
        raise TukeyError("no configurations to compare")
    if len(set(widths.values())) != 1:
        raise TukeyError(
            f"Tukey needs a balanced panel, got seed counts {sorted(set(widths.values()))}"
        )
    if len(seeds) != next(iter(widths.values())):
        raise TukeyError(f"{len(seeds)} seeds declared but {next(iter(widths.values()))} scored")

    return pd.DataFrame(
        [
            {"cfg": slug, "seed": str(seed), "mae": float(value)}
            for slug, values in per_seed.items()
            for seed, value in zip(seeds, values, strict=True)
        ]
    )


def _diagnostics(frame: pd.DataFrame, *, mse: float, df_resid: float, hsd: float) -> Diagnostics:
    """Measure what the intervals assume, on the panel that produced them."""
    blocked = smf.ols("mae ~ C(cfg) + C(seed)", data=frame).fit()
    table = anova_lm(blocked, typ=2)

    # Tukey's one degree of freedom test: a squared fitted value carries the
    # multiplicative interaction the additive model has no term for
    with_square = frame.assign(fitted_sq=blocked.fittedvalues**2)
    nonadditivity = smf.ols("mae ~ C(cfg) + C(seed) + fitted_sq", data=with_square).fit()

    groups = [group["mae"].to_numpy() for _, group in frame.groupby("cfg")]
    variances = np.array([values.var(ddof=1) for values in groups])
    positive = variances[variances > 0]

    return Diagnostics(
        groups=frame["cfg"].nunique(),
        blocks=frame["seed"].nunique(),
        mse=float(mse),
        df_resid=float(df_resid),
        hsd=float(hsd),
        block_f=float(table.loc["C(seed)", "F"]),
        block_p=float(table.loc["C(seed)", "PR(>F)"]),
        nonadditivity_p=float(nonadditivity.pvalues.get("fitted_sq", np.nan)),
        levene_p=float(st.levene(*groups, center="median").pvalue),
        variance_ratio=float(positive.max() / positive.min()) if positive.size else float("nan"),
    )


def compare(
    per_seed: dict[str, list[float]],
    seeds: list[int],
    *,
    block_by_seed: bool = True,
    alpha: float = ALPHA,
    label: str = "",
) -> dict[str, Any]:
    """Return Tukey comparison intervals and verdicts against the leader.

    The panel is balanced, so Tukey's critical distance is one number for every
    pair and splits evenly between the two intervals it joins. That is what
    makes overlap the test rather than an approximation of it, and it is the
    property the figure's geometry is read under. Each configuration is given
    half the critical distance on each side; two intervals touch exactly when
    the procedure does not separate the pair.

    Parameters
    ----------
    per_seed : dict
        Configuration slug mapped to its metric at each seed, in ``seeds``
        order. Lowest is best.
    seeds : list of int
        The seeds every configuration was run at, which are the blocks.
    block_by_seed : bool, optional
        Take the error term from ``mae ~ C(cfg) + C(seed)``, removing the
        shared seed effect. False pools it back in, which is what
        :func:`~statsmodels.stats.multicomp.pairwise_tukeyhsd` does, and that
        function is then what computes the result.
    alpha : float, optional
        Familywise error rate. Defaults to :data:`ALPHA`.
    label : str, optional
        Names the panel in the log, so a diagnostic warning says which figure
        it came from.

    Returns
    -------
    dict
        Shaped like the significance block a gate writes, so a figure draws it
        without knowing which procedure produced it: ``halfwidths`` per slug,
        ``difference_intervals`` against the leader, ``against_leader`` with a
        p-value and a verdict per row, and ``diagnostics``.

    Raises
    ------
    TukeyError
        If the panel is unbalanced, or has fewer than two configurations.
    """
    frame = _long(per_seed, seeds)
    means = frame.groupby("cfg")["mae"].mean().sort_values()
    groups, blocks = len(means), len(seeds)
    if groups < 2:
        raise TukeyError("a comparison needs at least two configurations")
    leader = str(means.index[0])

    # the error term, which is the only thing the flag changes. Unblocked, the
    # statsmodels routine is what computes it, so the toggle really does reach
    # their implementation rather than a reconstruction of it
    if block_by_seed:
        fitted = smf.ols("mae ~ C(cfg) + C(seed)", data=frame).fit()
        mse, df_resid = float(fitted.mse_resid), float(fitted.df_resid)
    else:
        result = pairwise_tukeyhsd(frame["mae"].to_numpy(), frame["cfg"].to_numpy(), alpha=alpha)
        mse, df_resid = float(result.variance), float(result.df_total)

    # the critical value and the p-values come from the exact studentized
    # range, which is what statsmodels 0.14 uses and reproduces its q_crit and
    # its p-values to full precision; the qsturng tabulation it kept for
    # backwards compatibility is interpolated and floors p at 0.001
    q_crit = float(studentized_range.ppf(1 - alpha, groups, df_resid))
    hsd = float(q_crit * np.sqrt(mse / blocks))
    halfwidth = hsd / 2

    # a difference from the leader, and the studentized range it sits at
    differences = {slug: float(means[slug] - means[leader]) for slug in means.index}
    against_leader = []
    for slug, difference in differences.items():
        if slug == leader:
            continue
        q = abs(difference) / np.sqrt(mse / blocks)
        p_value = float(studentized_range.sf(q, groups, df_resid))
        against_leader.append(
            {"slug": slug, "p_value": p_value, "separated": bool(abs(difference) > hsd)}
        )

    diagnostics = _diagnostics(frame, mse=mse, df_resid=df_resid, hsd=hsd)
    separated = sum(row["separated"] for row in against_leader)
    named = f"{label}: " if label else ""
    logger.info(
        "%s%d configurations, %s, HSD %.4f, %d of %d separated from %s",
        named,
        groups,
        "blocked on seed" if block_by_seed else "unblocked",
        hsd,
        separated,
        len(against_leader),
        leader,
    )
    if diagnostics.nonadditivity_p < alpha and block_by_seed:
        logger.warning(
            "%sconfiguration by seed interaction (nonadditivity p = %.3f); the blocked "
            "error term assumes there is none, so read the unblocked figure beside this one",
            named,
            diagnostics.nonadditivity_p,
        )
    if diagnostics.levene_p < alpha:
        logger.warning(
            "%sunequal seed variances across configurations (Levene p = %.3f, ratio %.0fx); "
            "the pooled error term is too wide for the steady rows and too narrow for the noisy",
            named,
            diagnostics.levene_p,
            diagnostics.variance_ratio,
        )

    return {
        "method": METHOD,
        "version": VERSION,
        "block_by_seed": block_by_seed,
        "alpha": alpha,
        "hsd": hsd,
        "q_crit": q_crit,
        "leader_slug": leader,
        # every row carries the same half width, which is what makes overlap
        # the test; the leader is not a special case here as it is under the
        # bootstrap, where it has no difference from itself to draw
        "leader_halfwidth": halfwidth,
        "halfwidths": {slug: halfwidth for slug in means.index},
        "difference_intervals": {
            slug: (difference - hsd, difference + hsd) for slug, difference in differences.items()
        },
        "against_leader": against_leader,
        "diagnostics": asdict(diagnostics),
    }


def retest(
    evidence: dict[str, Any],
    seeds: list[int],
    *,
    block_by_seed: bool = True,
    alpha: float = ALPHA,
    label: str = "",
) -> dict[str, Any]:
    """Return a copy of a gate's evidence with the figure's verdicts substituted.

    The gate's own decision is left where it is. What this replaces is the part
    a figure draws: the half widths, the verdicts against the leader, and the
    tie list a caption reads off. Everything else on the record, the ranking
    and its costs and its annotations, is carried through untouched, so a panel
    is drawn from one object whichever procedure measured it.

    The leader can change. Both procedures rank on the same seed means, so it
    does not change here, but the substituted evidence names its own leader
    rather than assuming the bootstrap's, since a figure that ranked one way
    and tested against another row would be the contradiction this is meant to
    remove.

    Parameters
    ----------
    evidence : dict
        A :func:`gates.measure` or :func:`gates.evidence` result. Its ranking
        rows must carry ``per_seed``, which records written before that field
        existed do not.
    seeds : list of int
        The seeds the family was run at, which are the blocks.
    block_by_seed : bool, optional
        Passed to :func:`compare`.
    alpha : float, optional
        Familywise error rate. Defaults to :data:`ALPHA`.
    label : str, optional
        Names the panel in the log.

    Returns
    -------
    dict
        The evidence, with ``significance``, ``indistinguishable_from_leader``
        and ``separated_from_leader`` replaced, and ``measured_by`` naming the
        procedure that replaced them.

    Raises
    ------
    TukeyError
        If any ranked row predates ``per_seed``.
    """
    ranking = evidence.get("ranking") or []
    missing = [row["slug"] for row in ranking if not row.get("per_seed")]
    if missing:
        raise TukeyError(
            f"{len(missing)} of {len(ranking)} rows carry no per-seed metric, first "
            f"{missing[0]}; re-measure rather than drawing an interval from the record"
        )

    significance = compare(
        {row["slug"]: list(row["per_seed"]) for row in ranking},
        seeds,
        block_by_seed=block_by_seed,
        alpha=alpha,
        label=label,
    )
    separated = {row["slug"] for row in significance["against_leader"] if row["separated"]}
    leader = significance["leader_slug"]
    return evidence | {
        "measured_by": METHOD,
        "leader_slug": leader,
        "indistinguishable_from_leader": [
            row["slug"] for row in ranking if row["slug"] != leader and row["slug"] not in separated
        ],
        "separated_from_leader": sorted(separated),
        "significance": significance,
    }
