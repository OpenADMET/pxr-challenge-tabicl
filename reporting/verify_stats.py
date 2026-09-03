"""Verify the comparative claims in ``blogpost.md`` against the result CSVs.

Every seed-swept experiment in ``manifest.py`` shares the same 5 seeds
(0-4) over the same fixed 513-compound blind set, so a comparison between
two experiments' per-seed MAE arrays is naturally paired by seed and
``scipy.stats.ttest_rel`` is the right instrument. The two N283T report
numbers (``n283t_ensemble``, ``n283t_target``) have no seed data of their
own; where the manuscript compares one of our 5-seed distributions against
one of those fixed external values, a one-sample ``ttest_1samp`` is used
instead.

Every number quoted in blogpost.md is asserted through :func:`check`,
:func:`check_below`, or :func:`check_atleast`, so this is a gate, not a
report: the script prints each comparison, then exits non-zero and names
any claim that has drifted past its tolerance. Run from the repo root::

    python reporting/verify_stats.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from manifest import EXPERIMENTS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEEDS = range(5)


# ── loading ──────────────────────────────────────────────────────────────────


def seed_mae(key: str, subset: str = "overall") -> np.ndarray:
    """Return one experiment's per-seed MAE on a given ``eval_out.csv`` subset.

    Ordered by seed (0-4), so two calls with different ``key``s line up
    for a paired test: seed ``i`` in one array was evaluated under the
    same conditions as seed ``i`` in the other.
    """
    exp = next(e for e in EXPERIMENTS if e.key == key)
    if exp.base_dir is None:
        raise ValueError(f"{key!r} has no seed data (fixed_mae={exp.fixed_mae}); use fixed_value")
    values: list[float] = []
    for seed in SEEDS:
        eval_csv = RESULTS_DIR / f"{exp.base_dir}_seed{seed}" / "eval_out.csv"
        if not eval_csv.exists():
            raise FileNotFoundError(f"Missing {eval_csv} for experiment {key!r}")
        row = pd.read_csv(eval_csv).query("subset == @subset")
        if len(row) != 1:
            raise ValueError(f"{eval_csv} has no unique {subset!r} row")
        values.append(float(row["mae"].iloc[0]))
    return np.array(values)


def fixed_value(key: str) -> float:
    """Return an external, non-reproducible fixed MAE (e.g. an N283T report number)."""
    exp = next(e for e in EXPERIMENTS if e.key == key)
    if exp.fixed_mae is None:
        raise ValueError(f"{key!r} has no fixed_mae; use seed_mae instead")
    return exp.fixed_mae


# ── reporting ────────────────────────────────────────────────────────────────


def report(
    label: str, a: np.ndarray, b: np.ndarray | float, *, paired: bool = True
) -> tuple[float, float, float]:
    """Print a t-test comparison and return ``(mean_a, mean_b, p_value)``.

    ``b`` as a bare float runs a one-sample test of ``a`` against that
    fixed reference value (used for the two N283T externals, which carry
    no seed distribution of their own); ``b`` as an array runs a paired
    (default) or independent two-sample test.
    """
    if isinstance(b, int | float):
        t, p = stats.ttest_1samp(a, b)
        b_mean, b_sd = float(b), 0.0
    else:
        t, p = stats.ttest_rel(a, b) if paired else stats.ttest_ind(a, b)
        b_mean, b_sd = float(b.mean()), float(b.std(ddof=1))
    sig = "SIGNIFICANT *" if p < 0.05 else "not significant"
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print(f"  A: mean={a.mean():.4f}  SD={a.std(ddof=1):.4f}  values={np.round(a, 4).tolist()}")
    print(f"  B: mean={b_mean:.4f}  SD={b_sd:.4f}")
    print(f"  diff(A-B)={a.mean() - b_mean:.4f}  t={t:.3f}  p={p:.4g}  -> {sig}")
    return float(a.mean()), b_mean, float(p)


# ── manuscript claim gate ────────────────────────────────────────────────────
#
# Every comparative number quoted in blogpost.md is asserted here, not just
# printed, so a re-run of the seed sweeps that shifts a mean fails this
# script loudly instead of leaving the prose silently stale. Point estimates
# use a tolerance that covers rounding and seed noise while still catching a
# real shift; p-values are checked against the bound the prose relies on (a
# significance direction, or an exact rounded value).

_CLAIMS: list[tuple[str, bool]] = []


def _record(label: str, ok: bool, detail: str) -> None:
    """Append a claim result and print its PASS/FAIL line."""
    _CLAIMS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")


def check(label: str, computed: float, claim: float, tol: float) -> None:
    """Assert a computed value matches its manuscript claim within ``tol``."""
    _record(
        label,
        abs(computed - claim) <= tol,
        f"computed={computed:.4f}  blog={claim:.4f}  tol=±{tol:.4f}",
    )


def check_below(label: str, computed: float, threshold: float) -> None:
    """Assert ``computed < threshold`` (e.g. a p-value below a significance cutoff)."""
    _record(label, computed < threshold, f"computed={computed:.4g}  must be < {threshold}")


def check_atleast(label: str, computed: float, floor: float) -> None:
    """Assert ``computed >= floor`` (e.g. a p-value at or above a non-significance bound)."""
    _record(label, computed >= floor, f"computed={computed:.4g}  must be >= {floor}")


# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run every labelled verification block."""
    print("=" * 64)
    print("PANEL 00 — graph-network baselines")
    print("=" * 64)

    # Concatenation architecture ("floor") vs the plain CheMeleon baseline:
    # text claims it "did surpass" the baseline.
    concat = seed_mae("concat_architecture")
    chemeleon_baseline = seed_mae("fine_tuned_pec50_direct")
    m_concat, m_chemeleon, p = report(
        "[00] concat_architecture vs fine_tuned_pec50_direct (CheMeleon baseline)",
        concat,
        chemeleon_baseline,
    )
    check("[00] concat_architecture mean", m_concat, 0.5176, 0.001)
    check("[00] CheMeleon baseline mean", m_chemeleon, 0.5215, 0.001)
    check_below("[00] concat nominally lower than CheMeleon baseline", m_concat - m_chemeleon, 0.0)
    check_atleast("[00] concat-vs-CheMeleon-baseline gap not significant", p, 0.05)

    # Frozen vs fine-tuned on primary-screen + dose-response data.
    frozen_full = seed_mae("frozen_log2fc_full")
    fine_tuned_full = seed_mae("fine_tuned_log2fc_full")
    m_frozen_full, m_fine_full, _ = report(
        "[00] frozen_log2fc_full vs fine_tuned_log2fc_full (primary + DRC)",
        frozen_full,
        fine_tuned_full,
    )
    check("[00] frozen (primary+DRC) mean", m_frozen_full, 0.5464, 0.001)
    check("[00] fine-tuned (primary+DRC) mean", m_fine_full, 0.5573, 0.001)

    # Frozen vs fine-tuned on dose-response-only data.
    frozen_drc = seed_mae("frozen_log2fc_drc_only")
    fine_tuned_drc = seed_mae("fine_tuned_log2fc_drc_only")
    m_frozen_drc, m_fine_drc, _ = report(
        "[00] frozen_log2fc_drc_only vs fine_tuned_log2fc_drc_only (DRC-only)",
        frozen_drc,
        fine_tuned_drc,
    )
    check("[00] frozen (DRC-only) mean", m_frozen_drc, 0.5849, 0.001)
    check("[00] fine-tuned (DRC-only) mean", m_fine_drc, 0.5339, 0.001)

    print("\n\n" + "=" * 64)
    print("PANEL 01 — individual component contributions")
    print("=" * 64)

    # RDKit descriptors "edged" the predicted log2FC readout.
    rdkit = seed_mae("rdkit_descriptors")
    predicted_readout = seed_mae("predicted_readout")
    m_rdkit, m_readout, _ = report(
        "[01] rdkit_descriptors vs predicted_readout", rdkit, predicted_readout
    )
    check("[01] RDKit descriptors mean", m_rdkit, 0.5312, 0.001)
    check("[01] predicted readout mean", m_readout, 0.5410, 0.001)
    check_below("[01] RDKit edges predicted readout", m_rdkit - m_readout, 0.0)

    # Mordred descriptors, worst single-ingredient row, "worse than the
    # graph-network floor".
    mordred = seed_mae("mordred_descriptors")
    m_mordred, m_floor, p = report(
        "[01] mordred_descriptors vs concat_architecture (GNN floor)", mordred, concat
    )
    check("[01] Mordred descriptors mean", m_mordred, 0.5784, 0.001)
    check_below("[01] Mordred worse than GNN floor", m_floor - m_mordred, 0.0)

    # CheMeleon embedding (no fine-tuning) "scored worse than" the
    # log2FC-trained Chemprop embedding.
    chemeleon_embed = seed_mae("chemeleon_embedding")
    log2fc_embed = seed_mae("log2fc_embedding")
    m_chemeleon_embed, m_log2fc_embed, _ = report(
        "[01] chemeleon_embedding vs log2fc_embedding", chemeleon_embed, log2fc_embed
    )
    check("[01] CheMeleon embedding mean", m_chemeleon_embed, 0.5002, 0.001)
    check("[01] log2FC-trained embedding mean", m_log2fc_embed, 0.4780, 0.001)
    check_below(
        "[01] CheMeleon embedding worse than log2FC embedding",
        m_log2fc_embed - m_chemeleon_embed,
        0.0,
    )

    print("\n\n" + "=" * 64)
    print("PANEL 02 — combining embedding, log2FC readout, and descriptors")
    print("=" * 64)

    top = seed_mae("chemeleon_readout_descriptors")
    embed_readout_desc = seed_mae("embed_readout_descriptors")
    readout_desc = seed_mae("readout_descriptors")
    embed_desc = seed_mae("embed_descriptors")
    chemeleon_readout = seed_mae("chemeleon_readout")
    chemeleon_desc = seed_mae("chemeleon_descriptors")

    m_top, m_erd, p_top_erd = report(
        "[02] chemeleon_readout_descriptors vs embed_readout_descriptors (top rows)",
        top,
        embed_readout_desc,
    )
    check("[02] top config mean", m_top, 0.4437, 0.001)
    check("[02] from-scratch three-ingredient mean", m_erd, 0.4574, 0.001)
    check_below("[02] top config significantly ahead of three-ingredient row", p_top_erd, 0.05)
    _, m_rd, p_top_rd = report(
        "[02] chemeleon_readout_descriptors vs readout_descriptors (top rows)", top, readout_desc
    )
    check("[02] readout + descriptors mean", m_rd, 0.4531, 0.001)
    check_below("[02] top config significantly ahead of readout+descriptors", p_top_rd, 0.05)

    # Once the log2FC-trained embedding is present, the readout is close
    # to redundant: "within seed variability" is a non-significance claim.
    m_erd2, m_ed, p_erd_ed = report(
        "[02] embed_readout_descriptors vs embed_descriptors (log2FC embedding, readout on/off)",
        embed_readout_desc,
        embed_desc,
    )
    check("[02] embedding + descriptors (no readout) mean", m_ed, 0.4610, 0.001)
    check_atleast("[02] log2FC-embedding readout redundancy within seed variability", p_erd_ed, 0.05)

    # For CheMeleon's off-the-shelf embedding, dropping the readout "cost
    # far more" than dropping descriptors: both a bigger gap and significant.
    m_cr, m_cd, p_cr_cd = report(
        "[02] chemeleon_readout vs chemeleon_descriptors (CheMeleon embedding, readout vs descriptors)",
        chemeleon_readout,
        chemeleon_desc,
    )
    check("[02] embedding + readout (CheMeleon) mean", m_cr, 0.4444, 0.001)
    check("[02] embedding + descriptors (CheMeleon) mean", m_cd, 0.5290, 0.001)
    check_below("[02] dropping readout costs far more than dropping descriptors", m_cr - m_cd, 0.0)
    check_below("[02] that gap is significant", p_cr_cd, 0.05)

    print("\n\n" + "=" * 64)
    print("PANEL 03 — evaluating tabular foundation models")
    print("=" * 64)

    tabicl = seed_mae("tabicl_v211")
    tabpfn_v3 = seed_mae("tabpfn_v3")
    our_best = seed_mae("our_best_overall")

    # TabPFN v3 vs TabPFN v2.5 (chemeleon_readout_descriptors): "0.0001 gap
    # well within seed noise".
    m_v3, m_v25, p_v3_v25 = report(
        "[03] tabpfn_v3 vs chemeleon_readout_descriptors (TabPFN v2.5)", tabpfn_v3, top
    )
    check("[03] TabPFN v3 mean", m_v3, 0.4436, 0.001)
    check("[03] TabPFN v2.5 mean", m_v25, 0.4437, 0.001)
    check_atleast("[03] TabPFN v3-vs-v2.5 gap within seed noise", p_v3_v25, 0.05)

    # TabICL on the 386-column sweep vs its own leaner-featureset best result.
    m_tabicl, m_ours, p_ours = report(
        "[03] tabicl_v211 (386-col) vs our_best_overall (258-col)", tabicl, our_best
    )
    check("[03] TabICL v2.1.1 (386-col) mean", m_tabicl, 0.4382, 0.001)
    check("[03] our best overall (258-col) mean", m_ours, 0.4356, 0.001)
    check_below("[03] leaner featureset nominally lower", m_ours - m_tabicl, 0.0)
    check_atleast("[03] leaner-featureset gap not significant", p_ours, 0.05)

    print("\n\n" + "=" * 64)
    print("PANEL 04 — descriptor PCA-width sweep")
    print("=" * 64)

    mordred_64 = seed_mae("mordred_pca64")
    mordred_128 = embed_readout_desc  # same row as panel 02's three-ingredient combo
    mordred_256 = seed_mae("mordred_pca256")
    rdkit_mordred_64 = seed_mae("rdkit_mordred_pca64")
    rdkit_mordred_128 = seed_mae("rdkit_mordred_pca128")
    rdkit_mordred_256 = seed_mae("rdkit_mordred_pca256")

    print("\nMordred-only, by PCA width:")
    for label, arr in (("64", mordred_64), ("128", mordred_128), ("256", mordred_256)):
        print(f"  {label:>3} components: mean={arr.mean():.4f}  SD={arr.std(ddof=1):.4f}")
    check("[04] Mordred-only @128 mean", mordred_128.mean(), 0.4574, 0.001)

    print("\nRDKit + Mordred, by PCA width:")
    for label, arr in (("64", rdkit_mordred_64), ("128", rdkit_mordred_128), ("256", rdkit_mordred_256)):
        print(f"  {label:>3} components: mean={arr.mean():.4f}  SD={arr.std(ddof=1):.4f}")
    check("[04] RDKit+Mordred @128 mean", rdkit_mordred_128.mean(), 0.4581, 0.001)

    # Mordred-only vs RDKit+Mordred at 128: "did not meaningfully improve
    # accuracy" is a non-significance claim on a tiny gap.
    m_m128, m_rm128, p_128 = report(
        "[04] mordred @128 vs rdkit+mordred @128", mordred_128, rdkit_mordred_128
    )
    check("[04] Mordred-vs-RDKit+Mordred @128 gap", m_rm128 - m_m128, 0.0007, 0.0005)
    check_atleast("[04] adding RDKit did not meaningfully improve accuracy", p_128, 0.05)

    print("\n\n" + "=" * 64)
    print("PANEL 05 — calibration")
    print("=" * 64)

    uncalibrated = seed_mae("calib_blind_uncalibrated")
    calibrated = seed_mae("calib_blind_calibrated")
    m_uncal, m_cal, p_cal = report(
        "[05] calib_blind_uncalibrated vs calib_blind_calibrated (overall)", uncalibrated, calibrated
    )
    check("[05] uncalibrated mean", m_uncal, 0.4356, 0.001)
    check("[05] calibrated mean", m_cal, 0.4507, 0.001)
    check_below("[05] calibration degrades performance", m_uncal - m_cal, 0.0)
    check_below("[05] that degradation is significant", p_cal, 0.05)

    uncalibrated_potent = seed_mae("calib_blind_uncalibrated", subset="potent (>= 6.0)")
    calibrated_potent = seed_mae("calib_blind_calibrated", subset="potent (>= 6.0)")
    m_uncal_p, m_cal_p, _ = report(
        "[05] calib_blind_uncalibrated vs calib_blind_calibrated (potent subset)",
        uncalibrated_potent,
        calibrated_potent,
    )
    check("[05] uncalibrated potent-subset mean", m_uncal_p, 0.6882, 0.001)
    check("[05] calibrated potent-subset mean", m_cal_p, 0.7402, 0.001)
    check_below("[05] potent subset also degrades", m_uncal_p - m_cal_p, 0.0)

    print("\n\n" + "=" * 64)
    print("PANEL 06 — predicted-uncertainty vs actual error")
    print("=" * 64)

    scored_csv = RESULTS_DIR / "tabpfn_uncertainty_mordred_pca128" / "pxr_tabpfn_distributions_scored.csv"
    scored = pd.read_csv(scored_csv)
    rho, p_rho = stats.spearmanr(scored["abs_residual"], scored["predicted_std"])
    print(f"\n  n={len(scored)}  Spearman rho={rho:.4f}  p={p_rho:.4g}")
    check("[06] Spearman rho(|residual|, predicted std)", float(rho), 0.2608, 0.0005)
    check("[06] n compounds", float(len(scored)), 513, 0)
    check_below("[06] Spearman correlation significant", float(p_rho), 1e-8)

    print("\n\n" + "=" * 64)
    print("LIMITATIONS — gap to the N283T single-pipeline target")
    print("=" * 64)

    # From-scratch/log2FC recipe's 5 seeds vs N283T's fixed 0.437 target.
    # No seed data backs the external value, so this is a one-sample test
    # of our own seed distribution against that fixed reference.
    n283t_target = fixed_value("n283t_target")
    m_recipe, m_target, p_gap = report(
        "[limitations] embed_readout_descriptors vs n283t_target (fixed)", embed_readout_desc, n283t_target
    )
    check("[limitations] N283T single-pipeline target", m_target, 0.437, 0.0005)
    check("[limitations] gap to N283T target", m_recipe - m_target, 0.0204, 0.001)
    check_below("[limitations] gap to N283T target significant", p_gap, 0.05)

    # Fail loudly if any locked claim drifted from the result CSVs.
    failed = [label for label, ok in _CLAIMS if not ok]
    print("\n\n" + "=" * 64)
    if failed:
        print(f"{len(failed)} of {len(_CLAIMS)} MANUSCRIPT CLAIMS DRIFTED FROM THE RESULT CSVs:")
        for label in failed:
            print(f"  - {label}")
        print("=" * 64)
        raise SystemExit(1)
    print(f"All {len(_CLAIMS)} locked manuscript claims match the result CSVs.")
    print("=" * 64)
    print("\nDone.")


if __name__ == "__main__":
    main()
