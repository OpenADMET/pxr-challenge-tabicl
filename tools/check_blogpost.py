"""Check every number quoted in the blogpost against the runs behind it.

The post is written by hand from the figures, so its numbers are transcribed
rather than computed, and a transcription error looks exactly like a result.
This reads the post, pulls out every quoted score, p-value and count, and asks
whether each one appears anywhere in what the pipeline actually recorded.

What it proves and what it does not. A number that matches nothing is wrong,
and that is the error this exists to catch: a stale value from the previous
generation, a digit typed twice, a p-value from a test no longer run. A number
that matches something is only shown to exist; whether it was attached to the
right claim is a judgement, so every match is reported with what it matched so
a reader can check the attribution rather than trust it.

Text inside a bracketed flag is skipped. Those spans quote superseded numbers
on purpose, so validating them would report the very values the flag exists to
retire.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aggregate  # noqa: E402
import gates  # noqa: E402
import manifest as manifest_module  # noqa: E402
import panels  # noqa: E402
import regressors  # noqa: E402
import tukey  # noqa: E402

logger = logging.getLogger(__name__)

DESCRIPTION = "Check the blogpost's quoted numbers against the recorded runs."

POST = Path("blogpost.md")

# a bracketed flag quotes superseded numbers deliberately, so its span is cut
# out before anything is extracted
FLAG = re.compile(r"\[(?:STALE|NEEDS REWRITE|ORDERING|REMOVED|add section|add caption)\b[^\]]*\]")

# a score: three or four decimals, in the range these metrics occupy
SCORE = re.compile(r"(?<![\d.])(0\.\d{3,4})(?![\d])")

# a p-value as the post writes it, either decimal or scientific
P_VALUE = re.compile(r"\*p\*\s*=\s*([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)")

# how close a quoted score has to be to a recorded one to count as that value
SCORE_TOLERANCE = 5e-5

# Numbers the post cites from outside this repository. Nothing here can be
# checked against a run, so each is listed with where it comes from and is
# reported as cited rather than as unmatched. A number not on this list and
# not in the runs is an error.
EXTERNAL = {
    "0.437": "N283T report, Table 4, out-of-fold MAE over training compounds",
    "0.474": "N283T report, single-embedding comparison, GatedGCN",
    "0.475": "N283T report, single-embedding comparison, MoLFormer",
    "0.484": "N283T report, single-embedding comparison, AttentiveFP",
    "0.448": "N283T report, single-embedding comparison, KERMT",
    "0.408": "N283T report, their ensemble after calibration",
    "0.6755": "CYP challenge live leaderboard, MA-ST-RAE, not this assay or metric",
}


@dataclass
class Value:
    """One number the pipeline recorded, and where it came from."""

    number: float
    kind: str
    where: str


@dataclass
class Finding:
    """One number quoted in the post, and what it matched."""

    quoted: str
    kind: str
    line: int
    matches: list[str] = field(default_factory=list)

    @property
    def is_matched(self) -> bool:
        """Whether the pipeline recorded this number anywhere."""
        return bool(self.matches)


def recorded_values(
    spec: manifest_module.Manifest, *, results_dir: Path, block_by_seed: bool
) -> list[Value]:
    """Return every number the figures and gates can justify a claim with.

    Built from the panels rather than from the run table, so a score here is
    one a reader could have read off a figure, and a p-value is the one that
    figure's tooltip shows.
    """
    values: list[Value] = []
    for panel in panels.build(spec, results_dir=results_dir, block_by_seed=block_by_seed):
        significance = panel.evidence["significance"]
        label_of = {row["slug"]: row.get("label", row["slug"]) for row in panel.evidence["ranking"]}
        for row in panel.evidence["ranking"]:
            name = f"{panel.id} {label_of[row['slug']]}"
            values.append(Value(row[gates.RANK_METRIC], "score", f"{name}, seed mean"))
            values.append(Value(row["seed_spread"], "score", f"{name}, seed spread"))
            ensemble = (row.get("ensemble") or {}).get(gates.RANK_METRIC)
            if ensemble is not None:
                values.append(Value(ensemble, "score", f"{name}, seed ensemble"))
        values.append(Value(significance["hsd"], "score", f"{panel.id} Tukey HSD"))
        for row in significance["against_leader"]:
            where = f"{panel.id} {label_of.get(row['slug'], row['slug'])} against the leader"
            values.append(Value(row["p_value"], "p", where))

    # the anchor, which is published rather than measured here
    anchor = spec.anchor or {}
    published = anchor.get("mae_ensemble") if isinstance(anchor, dict) else None
    if published is not None:
        values.append(Value(float(published), "score", "the N283T leaderboard entry"))

    values += _ensemble_sweep(spec, results_dir=results_dir, block_by_seed=block_by_seed)
    values += _uncertainty(results_dir=results_dir)
    return values


def _ensemble_sweep(
    spec: manifest_module.Manifest, *, results_dir: Path, block_by_seed: bool
) -> list[Value]:
    """Return the ensemble-size sweep's scores and its verdicts against the leader."""
    try:
        frame = panels.ensemble_rows(spec, results_dir=results_dir)
    except (panels.PanelError, gates.GateError) as error:
        logger.info("ensemble sweep not scored: %s", error)
        return []

    values = []
    for _, row in frame.iterrows():
        name = f"fig7 {int(row['n_estimators'])} members"
        values.append(Value(float(row["mae"]), "score", f"{name}, seed mean"))
        values.append(Value(float(row["seed_spread"]), "score", f"{name}, seed spread"))
        values.append(Value(float(row["ensemble_mae"]), "score", f"{name}, seed ensemble"))
        values.append(
            Value(float(row["mae"] - row["ensemble_mae"]), "score", f"{name}, seed-ensemble gap")
        )
        for source in ("model", "ensemble"):
            column = f"spearman_{source}"
            if column in row:
                values.append(Value(float(row[column]), "score", f"{name}, {source} Spearman"))

    per_seed = {
        f"e{regressors.ENSEMBLE_SIZE_OF[config.regressor]}": list(
            frame.loc[frame["slug"] == config.slug, "mae"]
        )
        for config in spec.expand(panels.ENSEMBLE_STAGE, gates.settled(spec, panels.ENSEMBLE_STAGE))
    }
    del per_seed  # the sweep's own p-values come from the panel below
    measured = tukey.compare(
        {
            f"e{int(row['n_estimators'])}": _seed_metrics(spec, row["slug"], results_dir)
            for _, row in frame.iterrows()
        },
        list(spec.seeds),
        block_by_seed=block_by_seed,
    )
    values.append(Value(measured["hsd"], "score", "fig7 Tukey HSD"))
    for row in measured["against_leader"]:
        values.append(Value(row["p_value"], "p", f"fig7 {row['slug']} against the leader"))
    return values


def _seed_metrics(spec: manifest_module.Manifest, slug: str, results_dir: Path) -> list[float]:
    """Return one configuration's metric at each seed, in manifest order."""
    config = gates.find_config(spec, slug)
    run_dirs = [config.run_dir(seed, results_dir) for seed in spec.seeds]
    stacked, observed = aggregate.stack_predictions(run_dirs)
    return [gates.evaluate.metrics(observed, row)[gates.RANK_METRIC] for row in stacked]


def _uncertainty(*, results_dir: Path) -> list[Value]:
    """Return the uncertainty diagnostics, which no comparison panel carries."""
    values = []
    for record in sorted(Path(results_dir).glob("uncertainty/*/uncertainty.json")):
        payload = json.loads(record.read_text())
        for stage in ("diagnostics", "calibrated"):
            block = payload.get(stage)
            block = block.get("diagnostics", block) if isinstance(block, dict) else None
            if not isinstance(block, dict):
                continue
            for source, diagnostic in block.items():
                if not isinstance(diagnostic, dict):
                    continue
                for field_name, number in diagnostic.items():
                    if isinstance(number, (int, float)):
                        where = f"uncertainty {stage} {source} {field_name}"
                        values.append(Value(float(number), "score", where))
    return values


def strip_flags(text: str) -> str:
    """Blank out bracketed flags, keeping line numbers intact."""
    return FLAG.sub(lambda m: " " * len(m.group(0)), text)


def quoted_numbers(text: str) -> list[Finding]:
    """Return every score and p-value the post states as a result."""
    findings = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in P_VALUE.finditer(line):
            findings.append(Finding(match.group(1), "p", number))
        for match in SCORE.finditer(line):
            # a p-value's own digits are not a second claim
            if any(m.start() <= match.start() < m.end() for m in P_VALUE.finditer(line)):
                continue
            findings.append(Finding(match.group(1), "score", number))
    return findings


def resolve(findings: list[Finding], values: list[Value]) -> list[Finding]:
    """Attach every recorded value a quoted number could be."""
    for finding in findings:
        quoted = float(finding.quoted)
        for value in values:
            if value.kind != finding.kind:
                continue
            if finding.kind == "score":
                if abs(value.number - quoted) <= SCORE_TOLERANCE:
                    finding.matches.append(value.where)
            # a p-value is quoted rounded, so it matches at the precision given
            elif _same_p(quoted, value.number, finding.quoted):
                finding.matches.append(value.where)
    return findings


def _same_p(quoted: float, recorded: float, text: str) -> bool:
    """Whether a recorded p-value rounds to the one the post prints.

    Half a unit in the last place quoted, rather than Python's own rounding of
    both sides: 0.0315 is stored as a float just below its decimal value, so
    asking round() to agree rejects a p-value the author rounded up correctly.
    """
    if "e" in text:
        digits = len(text.split("e")[0].replace(".", "").lstrip("0")) or 1
        return f"{recorded:.{digits - 1}e}" == f"{quoted:.{digits - 1}e}"
    places = len(text.split(".")[1]) if "." in text else 0
    return abs(recorded - quoted) <= 0.5 * 10**-places + 1e-12


def report(findings: list[Finding]) -> int:
    """Print what matched and what did not, returning the number unmatched."""
    external = [f for f in findings if not f.is_matched and f.quoted in EXTERNAL]
    unmatched = [f for f in findings if not f.is_matched and f.quoted not in EXTERNAL]
    matched = [f for f in findings if f.is_matched]

    print(
        f"{len(findings)} numbers quoted outside flags: {len(matched)} matched, "
        f"{len(external)} cited from elsewhere, {len(unmatched)} unmatched\n"
    )
    if external:
        print("cited from outside this repository, not checkable here:")
        for finding in external:
            print(f"  L{finding.line:<4} {finding.quoted:<10s} {EXTERNAL[finding.quoted]}")
        print()
    if unmatched:
        print("UNMATCHED, these appear nowhere in the recorded runs:")
        for finding in unmatched:
            print(f"  L{finding.line:<4} {finding.kind:5s} {finding.quoted}")
        print()

    print("matched, with what each one could be (attribution is not checked):")
    for finding in matched:
        first = finding.matches[0]
        extra = f"  (+{len(finding.matches) - 1} more)" if len(finding.matches) > 1 else ""
        print(f"  L{finding.line:<4} {finding.kind:5s} {finding.quoted:<10s} {first}{extra}")
    return len(unmatched)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--post", type=Path, default=POST, help="the markdown file to check")
    parser.add_argument(
        "--results", type=Path, default=aggregate.RESULTS_DIR, help="root holding the runs"
    )
    parser.add_argument(
        "--no-block-by-seed",
        dest="block_by_seed",
        action="store_false",
        help="take the p-values from the unblocked Tukey, as the figures' toggle does",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Check the post and exit non-zero if any number matches nothing."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    spec = manifest_module.load()
    values = recorded_values(spec, results_dir=args.results, block_by_seed=args.block_by_seed)
    findings = resolve(quoted_numbers(strip_flags(args.post.read_text())), values)
    if report(findings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
