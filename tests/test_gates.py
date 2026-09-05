import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate
import gates
import manifest as manifest_module
from data import CANONICAL_COL

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "run"

# enough compounds that a paired bootstrap has something to resample
COMPOUNDS = [f"C{index:03d}" for index in range(60)]
OBSERVED = np.linspace(4.0, 8.0, len(COMPOUNDS))

SEEDS = (0, 1, 2, 3, 4)


@pytest.fixture(scope="module")
def spec():
    return manifest_module.load()


def write_runs(results_root, configs, offsets: dict[str, float | np.ndarray], jitter=0.0):
    """Write one synthetic run per configuration and seed, at a chosen error.

    Every seed of a configuration carries the same offset, so its ensemble
    carries that offset and the ranking is known in advance. ``jitter`` adds a
    per-seed draw on top, shared by every configuration at that seed, which
    gives each one a spread for the gate's second bar to measure while leaving
    the differences between configurations exactly their offsets.
    """
    rng = np.random.default_rng(1234)
    noise = {s: rng.normal(0.0, jitter, len(COMPOUNDS)) if jitter else 0.0 for s in SEEDS}
    for config in configs:
        offset = offsets[config.slug]
        for seed in SEEDS:
            run_dir = config.run_dir(seed, results_root)
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    CANONICAL_COL: COMPOUNDS,
                    "observed": OBSERVED,
                    "predicted": OBSERVED + offset + noise[seed],
                }
            ).to_csv(run_dir / "predictions.csv", index=False)
            (run_dir / "run.json").write_text(json.dumps({"config": config.as_dict()}))


def ranked_offsets(configs, best) -> dict[str, float | np.ndarray]:
    """Give the named configuration the smallest error and the rest more."""
    return {
        config.slug: (0.05 if config.slug == best else 0.4 + 0.01 * index)
        for index, config in enumerate(configs)
    }


def test_a_gate_chooses_the_configuration_with_the_lowest_ensemble_error(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    winner = configs[5]
    write_runs(tmp_path, configs, ranked_offsets(configs, winner.slug))

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    assert decision["chosen_slug"] == winner.slug
    assert decision["chosen"] == {
        "descriptors": winner.descriptors,
        "descriptor_pca": winner.descriptor_pca,
    }
    assert decision["gate"] == "canonical_descriptors"


def test_the_ranking_records_every_configuration_the_rule_saw(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug))

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    assert [row["slug"] for row in decision["ranking"]][0] == configs[0].slug
    assert len(decision["ranking"]) == len(configs)
    assert all(row["n_seeds"] == len(SEEDS) for row in decision["ranking"])


def test_only_a_saving_within_the_seed_spread_can_win_a_tie(spec, tmp_path):
    # the bootstrap at 260 compounds waves through differences many times the
    # run-to-run noise, so cheapness alone would trade accuracy for columns
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug), jitter=0.25)

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    by_slug = {row["slug"]: row for row in decision["ranking"]}
    leader = by_slug[decision["leader_slug"]]
    budget = decision["seed_spread_budget"]

    assert budget == pytest.approx(leader["seed_spread"])
    assert set(decision["within_seed_spread"]) <= set(decision["tied_with_leader"])
    for slug in decision["within_seed_spread"]:
        assert by_slug[slug]["mae"] - leader["mae"] <= budget
    for slug in set(decision["tied_with_leader"]) - set(decision["within_seed_spread"]):
        assert by_slug[slug]["mae"] - leader["mae"] > budget
    assert decision["chosen_slug"] in {decision["leader_slug"], *decision["within_seed_spread"]}


def test_without_seed_variation_no_saving_is_free(spec, tmp_path):
    # every seed identical means the metric does not move at all when the seed
    # changes, so there is no budget and the leader stands however cheap the
    # alternatives are
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[-1].slug))

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    assert decision["seed_spread_budget"] == 0.0
    assert decision["within_seed_spread"] == []
    assert decision["chosen_slug"] == decision["leader_slug"]
    assert decision["tie_broken_on_cost"] is False


def test_a_tie_is_broken_towards_the_cheaper_configuration(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    mordred = {c.descriptor_pca: c for c in configs if c.descriptors == "mordred"}
    narrow, wide = mordred[32], mordred[256]

    # two predictors of the same quality, the narrower one a hair behind: no
    # bootstrap over these compounds can separate them and the gap is far inside
    # the seed spread, so the leader and the affordable configuration are
    # different runs and cost has to decide between them
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.3, len(COMPOUNDS))
    offsets: dict[str, float | np.ndarray] = {config.slug: 0.9 for config in configs}
    offsets[wide.slug] = base
    offsets[narrow.slug] = base + 0.001
    write_runs(tmp_path, configs, offsets, jitter=0.25)

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    assert decision["leader_slug"] == wide.slug
    assert decision["chosen_slug"] == narrow.slug
    assert decision["tie_broken_on_cost"] is True
    assert narrow.slug in decision["tied_with_leader"]
    assert narrow.slug in decision["within_seed_spread"]


def test_a_clear_winner_is_not_recorded_as_a_tie_break(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[3].slug))

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    assert decision["tie_broken_on_cost"] is False
    assert decision["tied_with_leader"] == []


def test_a_partial_stage_cannot_be_decided(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs[:2], ranked_offsets(configs, configs[0].slug))

    with pytest.raises(gates.GateError, match="partial stage"):
        gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)


def test_a_stage_that_gates_nothing_cannot_be_resolved(spec, tmp_path):
    with pytest.raises(gates.GateError, match="no gate"):
        gates.resolve(spec, "uncertainty", results_dir=tmp_path)


def test_a_decision_round_trips_through_the_file_it_is_written_to(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path / "gates")
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[1].slug))

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)
    path = gates.write(decision)

    assert path.exists()
    assert gates.read("canonical_descriptors")["chosen"] == decision["chosen"]


def test_a_stage_reads_what_the_gates_before_it_chose(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path)
    (tmp_path / "canonical_descriptors.json").write_text(
        json.dumps(
            {
                "gate": "canonical_descriptors",
                "chosen": {"descriptors": "mordred", "descriptor_pca": 128},
            }
        )
    )
    (tmp_path / "embedding_reduction.json").write_text(
        json.dumps({"gate": "embedding_reduction", "chosen": {"embedding_pca": 64}})
    )

    settled = gates.settled(spec, "ingredients")

    assert settled == {"descriptors": "mordred", "descriptor_pca": 128, "embedding_pca": 64}
    assert gates.all_chosen(spec) == settled


def test_a_later_gate_supersedes_an_earlier_one_on_a_shared_axis(spec, tmp_path, monkeypatch):
    # the descriptor probe settles which block to carry; the featureset sweep
    # afterwards may find that carrying none of it wins, and that is a decision
    # rather than a conflict
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path)
    for gate_id, chosen in (
        ("canonical_descriptors", {"descriptors": "rdkit", "descriptor_pca": 128}),
        ("embedding_reduction", {"embedding_pca": 256}),
        ("best_featureset", {"embedding": "chemeleon", "readout": "none", "descriptors": "none"}),
    ):
        (tmp_path / f"{gate_id}.json").write_text(json.dumps({"gate": gate_id, "chosen": chosen}))

    settled = gates.settled(spec, "regressor")

    assert settled["descriptors"] == "none"
    assert settled["descriptor_pca"] == 128
    # and the figures read the same precedence, by stage order rather than by
    # whatever order the files happen to sort in
    assert gates.all_chosen(spec)["descriptors"] == "none"


def test_a_stage_whose_gate_is_undecided_says_so_rather_than_guessing(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path)

    with pytest.raises(gates.GateError, match="has not been resolved"):
        gates.settled(spec, "ingredients")


def test_the_leading_stages_inherit_nothing(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path)

    assert gates.settled(spec, "descriptor_width") == {}
    assert gates.settled(spec, "embedding_width") == {}


def test_cost_prefers_fewer_blocks_before_narrower_reductions(spec):
    configs = {c.slug: c for c in spec.expand("descriptor_width")}
    narrow = next(
        c for c in configs.values() if c.descriptors == "mordred" and c.descriptor_pca == 32
    )
    wide = next(
        c for c in configs.values() if c.descriptors == "mordred" and c.descriptor_pca == 256
    )

    assert gates.cost(narrow, spec.axes) < gates.cost(wide, spec.axes)


def test_a_block_kept_whole_costs_its_own_width_not_its_sentinel(spec):
    configs = {c.slug: c for c in spec.expand("descriptor_width")}
    whole = next(
        c
        for c in configs.values()
        if c.descriptors == "rdkit" and c.descriptor_pca == manifest_module.NATIVE
    )
    reduced = [c for c in configs.values() if c.descriptors == "rdkit" and c.descriptor_pca > 0]

    # RDKit's 217 columns kept whole are the widest option on that axis, not
    # the narrowest; ordering on the sentinel would make it the cheapest
    assert gates.cost(whole, spec.axes)[1] == 217
    assert all(gates.cost(c, spec.axes) < gates.cost(whole, spec.axes) for c in reduced)


def test_an_undeclared_block_kept_whole_is_treated_as_the_most_expensive(spec):
    whole = manifest_module.TabularConfig(
        "none",
        manifest_module.NOT_REDUCED,
        "none",
        "rdkit",
        manifest_module.NATIVE,
        "lgbm",
        "none",
    )

    # with no column count declared there is nothing to compare against, and
    # guessing cheap would hand a tie to the widest featureset
    assert gates.cost(whole, {"descriptors": {}, "embedding": {}})[1] == gates._WIDEST


def test_the_gate_ranks_on_the_ensemble_rather_than_on_a_mean_of_seed_scores(spec, tmp_path):
    # every seed carries the same offset here, so the ensemble score equals the
    # seed-wise score and the ranking is unambiguous either way; this pins the
    # quantity the record claims to hold
    configs = spec.expand("descriptor_width")[:1] + spec.expand("descriptor_width")[1:]
    offsets = ranked_offsets(configs, configs[2].slug)
    write_runs(tmp_path, configs, offsets)

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)
    leader = decision["ranking"][0]

    assert leader["slug"] == configs[2].slug
    assert leader["mae"] == pytest.approx(evaluate.metrics(OBSERVED, OBSERVED + 0.05)["mae"])


def test_the_sweep_reports_an_undecided_gate_without_a_traceback(tmp_path, monkeypatch):
    # an undecided gate is an ordinary state of a staged sweep, so the entry
    # point has to name the stage to run rather than dumping a stack
    import runpy
    import sys

    monkeypatch.setattr(gates, "GATES_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["04_sweep.py", "--stage", "ingredients", "--dry-run"])

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(RUN_DIR / "04_sweep.py"), run_name="__main__")

    assert "canonical_descriptors" in str(raised.value)
    assert "has not been resolved" in str(raised.value)


def test_no_stage_reads_the_gate_it_produces(spec):
    # a stage that inherited its own decision would plan against the answer it
    # exists to find: the ingredient stage would stop sweeping descriptors over
    # nothing-or-the-winner as soon as it had chosen, so the comparison that
    # ruled descriptors out could never be reproduced from scratch
    for stage in spec.stages:
        if stage.gate is None:
            continue
        assert stage.gate.id not in stage.depends_on, stage.id


def test_a_stage_inherits_only_the_gates_decided_before_it(spec):
    order = [s.id for s in spec.stages]
    produced_by = {s.gate.id: order.index(s.id) for s in spec.stages if s.gate}

    for position, stage in enumerate(spec.stages):
        for gate_id in stage.depends_on:
            assert produced_by[gate_id] < position, f"{stage.id} waits on {gate_id}"


def test_the_ingredient_stage_still_sweeps_both_descriptor_arms(spec, tmp_path, monkeypatch):
    # the featureset gate supersedes the descriptor gate on that axis, which is
    # correct for every stage after it and would be wrong for this one
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path / "gates")
    (tmp_path / "gates").mkdir(parents=True)
    for gate_id, chosen in (
        ("canonical_descriptors", {"descriptors": "rdkit", "descriptor_pca": 128}),
        ("embedding_reduction", {"embedding_pca": 256}),
        ("best_featureset", {"descriptors": "none", "embedding": "chemeleon_log2fc"}),
    ):
        (tmp_path / "gates" / f"{gate_id}.json").write_text(json.dumps({"chosen": chosen}))

    settled = gates.settled(spec, "ingredients")

    assert settled["descriptors"] == "rdkit"
    assert {c.descriptors for c in spec.expand("ingredients", settled)} == {"none", "rdkit"}
