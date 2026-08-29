"""Build the long-format results dataframe backing panels 00/01/02, and render their HTML rows.

Reads `results/<base_dir>_seed{0..4}/eval_out.csv` for every experiment in
`manifest.py`, computes per-experiment mean/min/max overall MAE, checks that
each declared `role` ("winner", "floor", "single_best") still holds against the
freshly computed means, and writes each panel's `.row` HTML block between
marker comments in the artifact file. Re-running this after a new seed
sweep is the entire update path; no HTML is hand-edited.

Run with:
    python reporting/build_results_table.py --artifact PATH/TO/pxr_ablation_story_v2.html
    python reporting/build_results_table.py --dry-run  # print the table, write nothing
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from manifest import (
    EXPERIMENTS,
    GRID_TABLES,
    LABEL_OVERRIDE,
    MINI_CHARTS,
    PANEL_DIVIDER_AFTER,
    PANEL_ORDER,
    PANEL_SORT_FROM,
    ROLE_OVERRIDE,
    Experiment,
    GridTable,
    MiniChart,
)

SEEDS = range(5)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

AXIS_LOW, AXIS_HIGH = 0.30, 0.70


def pct(mae: float) -> float:
    """Map an MAE value to a percentage position on the panels' shared 0.30-0.70 axis."""
    return (mae - AXIS_LOW) / (AXIS_HIGH - AXIS_LOW) * 100


def load_long_table() -> pd.DataFrame:
    """Read every experiment's per-seed overall MAE into one long dataframe."""
    records: list[dict[str, object]] = []
    for exp in EXPERIMENTS:
        if exp.base_dir is None:
            records.append(
                {"experiment": exp.key, "seed": None, "mae": exp.fixed_mae, "category": exp.category}
            )
            continue
        if exp.single_run:
            eval_csv = RESULTS_DIR / exp.base_dir / "eval_out.csv"
            if not eval_csv.exists():
                raise FileNotFoundError(f"Missing {eval_csv} for single-run experiment {exp.key!r}")
            overall = pd.read_csv(eval_csv).query("subset == 'overall'")
            if len(overall) != 1:
                raise ValueError(f"{eval_csv} has no unique 'overall' row")
            records.append(
                {
                    "experiment": exp.key,
                    "seed": None,
                    "mae": float(overall["mae"].iloc[0]),
                    "category": exp.category,
                }
            )
            continue
        for seed in SEEDS:
            eval_csv = RESULTS_DIR / f"{exp.base_dir}_seed{seed}" / "eval_out.csv"
            if not eval_csv.exists():
                raise FileNotFoundError(
                    f"Missing {eval_csv} for experiment {exp.key!r}; run its seed sweep first"
                )
            overall = pd.read_csv(eval_csv).query("subset == 'overall'")
            if len(overall) != 1:
                raise ValueError(f"{eval_csv} has no unique 'overall' row")
            records.append(
                {
                    "experiment": exp.key,
                    "seed": seed,
                    "mae": float(overall["mae"].iloc[0]),
                    "category": exp.category,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long table to one row per experiment: mean/min/max/n_seeds."""
    return (
        long_df.groupby("experiment")["mae"]
        .agg(mean="mean", low="min", high="max", n_seeds="count")
        .reset_index()
    )



# Category each role's claim is scoped to: "floor" means best-of-gnn,
# "single_best" means best-of-single-ingredient. `None` means "beats every
# row that isn't an external reference," used for "winner" (our best overall).
ROLE_COMPARISON_CATEGORY: dict[str, str | None] = {
    "floor": "gnn",
    "single_best": "tabular-single",
    "winner": None,
    "sweepbest": "tabular-multiple",
}


def check_roles(summary: pd.DataFrame) -> None:
    """Verify each declared winner/floor/single_best role still has the lowest mean among its peers.

    Eligibility is derived from `category` (plus the panel a role is
    declared on), not from which rows happen to already carry that role in
    the manifest. A row can score lower than the declared role-holder
    without ever being tagged with that role itself; comparing only
    already-tagged rows against each other would silently miss exactly
    that case, which is what let the pEC50-direct row (category "gnn",
    role None) outscore the tagged "floor" row undetected.
    """
    by_key = summary.set_index("experiment")["mean"].to_dict()
    by_panel_category: dict[tuple[str, str], list[str]] = {}
    for exp in EXPERIMENTS:
        for panel in exp.panels:
            by_panel_category.setdefault((panel, exp.category), []).append(exp.key)

    declared: dict[tuple[str, str], list[str]] = {}
    for exp in EXPERIMENTS:
        if exp.role in ROLE_COMPARISON_CATEGORY:
            for panel in exp.panels:
                declared.setdefault((exp.role, panel), []).append(exp.key)

    for (role, panel), declared_keys in declared.items():
        comparison_category = ROLE_COMPARISON_CATEGORY[role]
        if comparison_category is None:
            peer_keys = [
                k
                for (p, cat), keys in by_panel_category.items()
                if p == panel and cat != "n283t"
                for k in keys
            ]
        else:
            peer_keys = by_panel_category.get((panel, comparison_category), [])
        eligible_keys = [k for k in peer_keys if by_key.get(k) is not None]
        if not eligible_keys:
            continue
        true_best = min(eligible_keys, key=lambda k: by_key[k])
        for declared_key in declared_keys:
            if by_key[declared_key] > by_key[true_best] + 1e-9:
                raise ValueError(
                    f"Role check failed: panel {panel} role={role!r} is assigned to "
                    f"{declared_key!r} (mean {by_key[declared_key]:.4f}) but {true_best!r} "
                    f"(mean {by_key[true_best]:.4f}) beats it among category-matched peers"
                )


def experiment_by_key(key: str) -> Experiment:
    return next(exp for exp in EXPERIMENTS if exp.key == key)


def render_row(
    exp: Experiment,
    stats: pd.Series,
    *,
    panel: str,
    divider: bool = False,
    reference_block: bool = False,
    role_override: str | None = None,
    label_override: tuple[str, str] | None = None,
) -> str:
    """Render one `.row` div, including a whisker only when real seed data backs it.

    Rows whose `native_panels` doesn't include `panel` are hatched (`.bar.reused`)
    rather than solid, since they're being shown for reference here, not because
    this panel's own sweep produced them. A row can be native to more than one
    panel (declared via `native_panels`) and stay solid in all of them.
    `reference_block` rows (the fixed anchor block above each panel's dashed
    divider: N283T rows, our best overall) never get the reused hatch even
    when not native here; they're already visually distinct and always shown
    for reference, so the hatch adds noise instead of information.
    """
    role = role_override or exp.role
    label, sub = label_override or (exp.label, exp.sub)
    classes = ["row"]
    bar_classes = ["bar"]
    style_attr = ""
    if role == "winner":
        classes.append("winner")
        bar_classes.append("win")
    if role == "floor":
        classes.append("floor")
        bar_classes.append("floor-bar")
    if role == "single_best":
        classes.append("single_best")
        style_attr = " background-color: var(--blue);"
    if role == "sweepbest":
        classes.append("sweepbest")
        style_attr = " background-color: var(--sage);"
    if role == "chemeleon":
        classes.append("chemeleon")
        style_attr = " background-color: var(--magenta);"
    if role == "context":
        classes.append("context")
        bar_classes.append("context-bar")
    if role == "target":
        classes.append("target")
        bar_classes.append("target-bar")
    if not reference_block and exp.native_panels and panel not in exp.native_panels:
        bar_classes.append("reused")
    if divider:
        classes.append("ref-divider")

    sub_html = f'<span class="sub">{sub}</span>' if sub else ""
    extra_sub_html = (
        f'<span class="sub" style="display:block;font-weight:400;font-size:0.7rem;'
        f'margin-top:0.2rem;">{exp.extra_sub}</span>'
        if exp.extra_sub
        else ""
    )

    if exp.fixed_mae is not None:
        bar_pct = pct(exp.fixed_mae)
        mae_text = f"~{exp.fixed_mae:.3f}" if exp.role == "context" else f"{exp.fixed_mae:.3f}"
        whisker_html = ""
    else:
        bar_pct = pct(stats["mean"])
        mae_text = f"{stats['mean']:.4f}"
        whisker_left = pct(stats["low"])
        whisker_width = pct(stats["high"]) - whisker_left
        whisker_html = (
            f'<div class="err-whisker" style="left: {whisker_left:.1f}%; '
            f'width: {whisker_width:.1f}%;"></div>'
            if stats["n_seeds"] > 1
            else ""
        )

    return (
        f'      <div class="{" ".join(classes)}">\n'
        f'        <div class="row-label">{label}{sub_html}{extra_sub_html}</div>\n'
        f'        <div class="track"><div class="{" ".join(bar_classes)}" '
        f'style="width: {bar_pct:.1f}%;{style_attr}"></div>{whisker_html}</div>\n'
        f'        <div class="mae-val">{mae_text}</div>\n'
        f"      </div>"
    )


def render_panel(panel: str, summary: pd.DataFrame) -> str:
    by_key = summary.set_index("experiment")
    keys = PANEL_ORDER[panel]
    sort_from = PANEL_SORT_FROM.get(panel)
    if sort_from is not None:
        head, tail = keys[:sort_from], keys[sort_from:]
        tail = sorted(tail, key=lambda k: by_key.loc[k, "mean"])
        keys = [*head, *tail]

    divider_key = PANEL_DIVIDER_AFTER.get(panel)
    rows = []
    in_reference_block = True
    for key in keys:
        exp = experiment_by_key(key)
        stats = by_key.loc[key] if key in by_key.index else None
        divider = divider_key == key
        role_override = ROLE_OVERRIDE.get((panel, key))
        label_override = LABEL_OVERRIDE.get((panel, key))
        rows.append(
            render_row(
                exp,
                stats,
                panel=panel,
                divider=divider,
                reference_block=in_reference_block,
                role_override=role_override,
                label_override=label_override,
            )
        )
        if divider:
            in_reference_block = False
    return "\n\n".join(rows)


def render_mini_chart(chart: MiniChart, summary: pd.DataFrame, index: int) -> str:
    """Render one `.mini-chart` block's title + rows (no wrapping div; that stays in the HTML)."""
    by_key = summary.set_index("experiment")
    means = {key: by_key.loc[key, "mean"] for key, _ in chart.rows if key in by_key.index}
    accent_key = min(means, key=means.get) if chart.accent == "min" else chart.accent

    rows = chart.rows
    if chart.sort_by_mean:
        rows = tuple(sorted(rows, key=lambda row: means[row[0]]))

    row_columns = f"{chart.label_width} 1fr 3.4rem"

    extra_sub_html = (
        f'<span class="sub" style="display:block;font-weight:400;font-size:0.7rem;'
        f'margin-top:0.2rem;">{chart.extra_sub}</span>'
        if chart.extra_sub
        else ""
    )
    lines = [f'    <div class="mini-chart">']
    lines.append(f'      <p class="mini-title">{chart.title}{extra_sub_html}</p>')
    lines.append(
        f'      <div class="axis-ref" style="grid-template-columns: {row_columns};">'
        f'<div></div><div class="axis-ticks"><span>0.30</span><span>0.50</span>'
        f"<span>0.70</span></div><div></div></div>"
    )
    for key, label in rows:
        stats = by_key.loc[key]
        bg = "var(--accent)" if key == accent_key else "var(--gray)"
        bar_pct = pct(stats["mean"])
        whisker_html = ""
        if chart.show_whiskers and stats["n_seeds"] > 1:
            whisker_left = pct(stats["low"])
            whisker_width = pct(stats["high"]) - whisker_left
            whisker_html = (
                f'<div class="err-whisker" style="left: {whisker_left:.1f}%; '
                f'width: {whisker_width:.1f}%;"></div>'
            )
        lines.append(
            f'      <div class="mini-row" style="grid-template-columns: {row_columns};">\n'
            f'        <span class="mini-label">{label}</span>\n'
            f'        <div class="mini-track"><div class="mini-bar" '
            f'style="width: {bar_pct:.1f}%; background: {bg};"></div>{whisker_html}</div>\n'
            f'        <span class="mini-val">{stats["mean"]:.4f}</span>\n'
            f"      </div>"
        )
    lines.append("    </div>")
    return "\n".join(lines)


def render_grid_table(table: GridTable, summary: pd.DataFrame) -> str:
    """Render one `.grid-table`, marking the single lowest-mean cell as `.best`."""
    by_key = summary.set_index("experiment")["mean"]
    flat_keys = [key for row in table.cells for key in row]
    best_key = min(flat_keys, key=lambda k: by_key.loc[k])

    lines = [
        '    <table class="grid-table">',
        f"      <caption>{table.caption}</caption>",
        "      <thead>",
        "        <tr><th></th>" + "".join(f"<th>{h}</th>" for h in table.col_headers) + "</tr>",
        "      </thead>",
        "      <tbody>",
    ]
    for row_header, row_keys in zip(table.row_headers, table.cells):
        cells_html = ""
        for key in row_keys:
            mae = by_key.loc[key]
            cls = ' class="best"' if key == best_key else ""
            cells_html += f"<td{cls}>{mae:.4f}</td>"
        lines.append(f"        <tr><th>{row_header}</th>{cells_html}</tr>")
    lines.append("      </tbody>")
    lines.append("    </table>")
    return "\n".join(lines)


def splice_into_artifact(
    artifact_path: Path,
    panel_html: dict[str, str],
    mini_chart_html: list[str],
    grid_table_html: list[str],
) -> None:
    """Replace each panel/mini-chart/grid-table block, delimited by its own marker comment pair."""
    text = artifact_path.read_text()

    def splice_one(name: str, html: str) -> None:
        nonlocal text
        pattern = re.compile(
            rf"(<!-- {name}:START -->\n)(.*?)(\n *<!-- {name}:END -->)",
            re.DOTALL,
        )
        new_text, n = pattern.subn(lambda m: m.group(1) + html + m.group(3), text)
        if n != 1:
            raise ValueError(f"Expected exactly one {name} marker pair in {artifact_path}, found {n}")
        text = new_text

    for panel, html in panel_html.items():
        splice_one(f"PANEL:{panel}", html)
    for index, html in enumerate(mini_chart_html):
        splice_one(f"MINICHART:{index}", html)
    for index, html in enumerate(grid_table_html):
        splice_one(f"GRIDTABLE:{index}", html)

    artifact_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    long_df = load_long_table()
    summary = summarize(long_df)
    check_roles(summary)

    if args.dry_run or args.artifact is None:
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print(summary.sort_values("mean").to_string(index=False))
        return

    panel_html = {panel: render_panel(panel, summary) for panel in PANEL_ORDER}
    mini_chart_html = [render_mini_chart(chart, summary, i) for i, chart in enumerate(MINI_CHARTS)]
    grid_table_html = [render_grid_table(table, summary) for table in GRID_TABLES]
    splice_into_artifact(args.artifact, panel_html, mini_chart_html, grid_table_html)
    print(
        f"Wrote panels {list(panel_html)}, {len(mini_chart_html)} mini-chart(s), "
        f"{len(grid_table_html)} grid table(s) into {args.artifact}"
    )


if __name__ == "__main__":
    main()
