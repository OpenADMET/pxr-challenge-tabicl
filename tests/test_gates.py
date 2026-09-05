import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate
import gates
import manifest as manifest_module
from data import CANONICAL_COL
from manifest import TabularConfig

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


def test_evidence_ranks_every_configuration_on_the_seed_mean(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug))

    measured = gates.evidence(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    assert measured["gate"] == "canonical_descriptors"
    assert len(measured["ranking"]) == len(configs)
    assert measured["ranking"][0]["slug"] == configs[0].slug
    assert all(row["n_seeds"] == len(SEEDS) for row in measured["ranking"])
    # it measures and does not decide
    assert "chosen" not in measured


def test_evidence_reports_cost_as_fact_rather_than_ordering_on_it(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug))

    measured = gates.evidence(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    for row in measured["ranking"]:
        assert row["n_blocks"] >= 1
        assert row["n_encoders_to_train"] >= 0
        assert row["n_columns"] >= 0
    # the leader is whatever scored best, never whatever was cheapest
    assert measured["leader_slug"] == min(measured["ranking"], key=lambda r: r["mae"])["slug"]


def test_the_seed_mean_is_ranked_on_and_the_ensemble_is_carried_alongside(spec, tmp_path):
    # a single model is the subject; seeds are replicates for power and a
    # variance estimate, not a way to build a better predictor
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug), jitter=0.25)

    measured = gates.evidence(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    for row in measured["ranking"]:
        assert "ensemble" in row
        assert row["ensemble"]["mae"] != row["mae"], "the two are different quantities"
    order = [r["mae"] for r in measured["ranking"]]
    assert order == sorted(order)


def test_a_decision_records_its_reason_and_who_made_it(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path / "gates")
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug))
    pick = configs[3]

    decision = gates.decide(
        spec,
        "descriptor_width",
        {"descriptors": pick.descriptors, "descriptor_pca": pick.descriptor_pca},
        "cheaper and nothing separates it from the leader",
        decided_by="tester",
        results_dir=tmp_path,
        n_resamples=200,
    )

    assert decision["chosen_slug"] == pick.slug
    assert decision["decided_by"] == "tester"
    assert "nothing separates it" in decision["reason"]
    # the evidence travels with the decision
    assert len(decision["ranking"]) == len(configs)
    assert decision["significance"]["fdr"] == gates.FDR


def test_a_decision_without_a_reason_is_refused(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug))

    with pytest.raises(gates.GateError, match="needs a reason"):
        gates.decide(
            spec,
            "descriptor_width",
            {"descriptors": "rdkit", "descriptor_pca": 128},
            "   ",
            decided_by="tester",
            results_dir=tmp_path,
            n_resamples=200,
        )


def test_a_decision_must_name_a_configuration_the_stage_ran(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug))

    with pytest.raises(gates.GateError, match="no configuration"):
        gates.decide(
            spec,
            "descriptor_width",
            {"descriptors": "rdkit", "descriptor_pca": 999},
            "a width nothing ran at",
            decided_by="tester",
            results_dir=tmp_path,
            n_resamples=200,
        )


def test_a_decision_must_cover_exactly_the_axes_the_gate_chooses(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[0].slug))

    with pytest.raises(gates.GateError, match="choose exactly"):
        gates.decide(
            spec,
            "descriptor_width",
            {"descriptors": "rdkit"},
            "missing the width",
            decided_by="tester",
            results_dir=tmp_path,
            n_resamples=200,
        )


def test_a_partial_stage_cannot_be_decided(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs[:2], ranked_offsets(configs, configs[0].slug))

    with pytest.raises(gates.GateError, match="partial stage"):
        gates.evidence(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)


def test_a_stage_that_gates_nothing_cannot_be_resolved(spec, tmp_path):
    with pytest.raises(gates.GateError, match="no gate"):
        gates.evidence(spec, "uncertainty", results_dir=tmp_path)


def test_a_decision_round_trips_through_the_file_it_is_written_to(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path / "gates")
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[1].slug))

    pick = configs[1]
    decision = gates.decide(
        spec,
        "descriptor_width",
        {"descriptors": pick.descriptors, "descriptor_pca": pick.descriptor_pca},
        "the one this test picked",
        decided_by="tester",
        results_dir=tmp_path,
        n_resamples=200,
    )
    path = gates.write(decision)

    assert path.exists()
    written = gates.read("canonical_descriptors")
    assert written["chosen"] == decision["chosen"]
    assert written["reason"] == decision["reason"]
    assert written["decided_by"] == "tester"


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
    assert gates.cost(whole, spec.axes)[2] == 217
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
    assert gates.cost(whole, {"descriptors": {}, "embedding": {}})[2] == gates._WIDEST


def test_the_gate_ranks_on_the_ensemble_rather_than_on_a_mean_of_seed_scores(spec, tmp_path):
    # every seed carries the same offset here, so the ensemble score equals the
    # seed-wise score and the ranking is unambiguous either way; this pins the
    # quantity the record claims to hold
    configs = spec.expand("descriptor_width")[:1] + spec.expand("descriptor_width")[1:]
    offsets = ranked_offsets(configs, configs[2].slug)
    write_runs(tmp_path, configs, offsets)

    decision = gates.evidence(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)
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


def test_benjamini_hochberg_steps_up_rather_than_down():
    # the largest rank clearing its own threshold carries everything below it,
    # including a p-value that does not clear its own. Holm would stop at the
    # first failure, which is the difference between the two procedures
    p = [0.001, 0.013, 0.021, 0.9]
    rejected = evaluate.benjamini_hochberg(p, fdr=0.05)

    assert list(rejected) == [True, True, True, False]


def test_benjamini_hochberg_rejects_nothing_when_no_rank_clears():
    assert not evaluate.benjamini_hochberg([0.2, 0.4, 0.9], fdr=0.05).any()


def test_a_bootstrap_p_value_cannot_be_exactly_zero():
    # a step-up procedure has to order the p-values, so the plus-one correction
    # floors them at 2 / (n + 1) rather than letting a tail count reach zero
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200)
    resampled = evaluate.bootstrap_family(
        truth, [truth, truth + 5.0], metric="mae", n_resamples=500
    )
    p = evaluate.difference_p_value(resampled, 0, 1)

    assert p == pytest.approx(2 / 501)
    assert p > 0


def test_the_family_bootstrap_pairs_every_predictor_on_one_draw():
    # drawing once is what makes any pair comparable, and what makes an
    # all-pairwise family cost one bootstrap per predictor rather than per pair
    rng = np.random.default_rng(1)
    truth = rng.normal(size=120)
    a, b = truth + rng.normal(scale=0.1, size=120), truth + rng.normal(scale=0.1, size=120)

    family = evaluate.bootstrap_family(truth, [a, b], n_resamples=300, seed=7)
    pairwise = evaluate.paired_bootstrap(truth, a, b, n_resamples=300, seed=7)

    assert family.shape == (2, 300)
    # the same seed and draw, so the family's difference reproduces the pair's
    assert float(np.mean(family[0] - family[1])) == pytest.approx(
        float((pairwise.ci_low + pairwise.ci_high) / 2), abs=0.02
    )


def test_a_gate_records_the_evidence_behind_every_verdict(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_DIR", tmp_path / "gates")
    configs = spec.expand("descriptor_width")
    write_runs(tmp_path, configs, ranked_offsets(configs, configs[1].slug))

    decision = gates.evidence(spec, "descriptor_width", results_dir=tmp_path, n_resamples=400)
    sig = decision["significance"]

    assert sig["fdr"] == gates.FDR
    # every pair, not only the comparisons against the leader
    assert sig["n_comparisons"] == len(configs) * (len(configs) - 1) // 2
    assert len(sig["against_leader"]) == len(configs) - 1
    separated = {r["slug"] for r in sig["against_leader"] if r["separated"]}
    assert separated.isdisjoint(decision["indistinguishable_from_leader"])
    for row in sig["against_leader"]:
        assert 0.0 < row["p_value"] <= 1.0
        assert 0.0 < row["bh_threshold"] <= gates.FDR


def test_two_blocks_off_one_encoder_are_charged_once(spec):
    # the embedding and readout of a prefix come off a single trained network,
    # so a configuration carrying both pays for one encoder and not two
    both = TabularConfig(
        embedding="chemprop_log2fc",
        readout="chemprop_log2fc",
        descriptors="none",
        embedding_pca=0,
        descriptor_pca=0,
        regressor="tabpfn-v3",
        calibration="none",
    )
    assert gates.trained_encoders(both, spec.axes) == 1


def test_a_block_needing_no_training_costs_no_encoder(spec):
    frozen = TabularConfig(
        embedding="chemeleon",
        readout="none",
        descriptors="rdkit",
        embedding_pca=256,
        descriptor_pca=128,
        regressor="tabpfn-v3",
        calibration="none",
    )
    assert gates.trained_encoders(frozen, spec.axes) == 0


def test_cost_prefers_the_featureset_with_fewer_encoders_to_train(spec):
    # both are two blocks at 256 columns; only the provenance differs, and only
    # one of them has to fit a network five times before it can run
    off_the_shelf = TabularConfig(
        embedding="chemeleon",
        readout="chemprop_log2fc",
        descriptors="none",
        embedding_pca=256,
        descriptor_pca=0,
        regressor="tabpfn-v3",
        calibration="none",
    )
    fine_tuned = TabularConfig(
        embedding="chemeleon_log2fc",
        readout="chemprop_log2fc",
        descriptors="none",
        embedding_pca=256,
        descriptor_pca=0,
        regressor="tabpfn-v3",
        calibration="none",
    )

    a, b = gates.cost(off_the_shelf, spec.axes), gates.cost(fine_tuned, spec.axes)

    # same shape, so only the encoder term may separate them
    assert a[0] == b[0]
    assert a[2] == b[2]
    assert a < b
