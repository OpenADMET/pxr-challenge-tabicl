import json

import numpy as np
import pandas as pd
import pytest

import evaluate
import gates
import manifest as manifest_module
from data import CANONICAL_COL

# enough compounds that a paired bootstrap has something to resample
COMPOUNDS = [f"C{index:03d}" for index in range(60)]
OBSERVED = np.linspace(4.0, 8.0, len(COMPOUNDS))

SEEDS = (0, 1, 2, 3, 4)


@pytest.fixture(scope="module")
def spec():
    return manifest_module.load()


def write_runs(results_root, configs, offsets: dict[str, float | np.ndarray]):
    """Write one synthetic run per configuration and seed, at a chosen error.

    Every seed of a configuration gets the same offset, so its ensemble carries
    that offset exactly and the ranking is known in advance.
    """
    for config in configs:
        offset = offsets[config.slug]
        for seed in SEEDS:
            run_dir = config.run_dir(seed, results_root)
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    CANONICAL_COL: COMPOUNDS,
                    "observed": OBSERVED,
                    "predicted": OBSERVED + offset,
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


def test_a_tie_is_broken_towards_the_cheaper_configuration(spec, tmp_path):
    configs = spec.expand("descriptor_width")
    mordred = {c.descriptor_pca: c for c in configs if c.descriptors == "mordred"}
    narrow, wide = mordred[32], mordred[256]

    # two noisy predictors of the same quality: the difference between them
    # moves compound by compound, so a bootstrap over these compounds cannot
    # separate them. The wider one is given the smaller error, so the leader and
    # the cheaper configuration are not the same run and cost has to decide
    rng = np.random.default_rng(0)
    residuals = [rng.normal(0.0, 0.3, len(COMPOUNDS)) for _ in range(2)]
    residuals.sort(key=lambda values: float(np.abs(values).mean()))
    offsets: dict[str, float | np.ndarray] = {config.slug: 0.9 for config in configs}
    offsets[wide.slug], offsets[narrow.slug] = residuals
    write_runs(tmp_path, configs, offsets)

    decision = gates.resolve(spec, "descriptor_width", results_dir=tmp_path, n_resamples=200)

    assert decision["leader_slug"] == wide.slug
    assert decision["chosen_slug"] == narrow.slug
    assert decision["tie_broken_on_cost"] is True
    assert narrow.slug in decision["tied_with_leader"]


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
        gates.resolve(spec, "calibration", results_dir=tmp_path)


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
    assert gates.all_chosen() == settled


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

    assert gates.cost(narrow) < gates.cost(wide)


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
