"""Tests for the scoring metrics, seed ensembling, and the paired bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import BootstrapResult, ensemble_mean, excludes_zero, metrics, paired_bootstrap

# a four-compound example small enough to score by hand:
# residuals 0.5, -0.5, 0.5, 1.0 against truth deviating -1.5, -0.5, 0.5, 1.5
HAND_TRUE = [1.0, 2.0, 3.0, 4.0]
HAND_PRED = [1.5, 1.5, 3.5, 5.0]


def test_metrics_match_hand_computed_values():
    scores = metrics(HAND_TRUE, HAND_PRED)

    assert scores["n"] == 4.0
    assert scores["mae"] == pytest.approx(0.625)
    assert scores["rmse"] == pytest.approx(np.sqrt(0.4375))
    assert scores["rae"] == pytest.approx(0.625)
    assert scores["r2"] == pytest.approx(0.65)
    # 5 concordant pairs of 6, one tied in the prediction: 5 / sqrt(6 * 5)
    assert scores["kendall_tau"] == pytest.approx(5.0 / np.sqrt(30.0))
    # Pearson correlation of the ranks 1,2,3,4 against 1.5,1.5,3,4
    assert scores["spearman_rho"] == pytest.approx(4.5 / np.sqrt(5.0 * 4.5))


def test_rae_matches_the_reference_denominator_convention():
    rng = np.random.default_rng(11)
    true = rng.normal(loc=6.0, scale=1.2, size=40)
    pred = true + rng.normal(scale=0.4, size=40)

    # transcribed from compute_metrics in the previous generation's
    # evaluate_predictions.py: the denominator is the truth's total absolute
    # deviation about its own mean, not about its median and not the truth's
    # own magnitude
    residual = pred - true
    true_deviation = true - np.mean(true)
    reference_rae = np.sum(np.abs(residual)) / np.sum(np.abs(true_deviation))

    assert metrics(true, pred)["rae"] == pytest.approx(reference_rae)


def test_rae_of_one_means_no_better_than_the_mean_predictor():
    true = np.array([1.0, 2.0, 3.0, 10.0])
    mean_predictor = np.full_like(true, true.mean())

    assert metrics(true, mean_predictor)["rae"] == pytest.approx(1.0)


def test_perfect_predictions_give_degenerate_values():
    scores = metrics(HAND_TRUE, HAND_TRUE)

    assert scores["mae"] == pytest.approx(0.0)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["rae"] == pytest.approx(0.0)
    assert scores["r2"] == pytest.approx(1.0)
    assert scores["kendall_tau"] == pytest.approx(1.0)
    assert scores["spearman_rho"] == pytest.approx(1.0)


def test_constant_truth_leaves_rae_and_r2_undefined():
    scores = metrics([3.0, 3.0, 3.0], [2.0, 3.0, 4.0])

    assert np.isnan(scores["rae"])
    assert np.isnan(scores["r2"])
    assert scores["mae"] == pytest.approx(2.0 / 3.0)


def test_metrics_raise_on_shape_mismatch_rather_than_broadcasting():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics([1.0, 2.0, 3.0], [1.0])


def test_ensemble_mean_averages_across_seeds():
    predictions = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

    np.testing.assert_allclose(ensemble_mean(predictions), [2.0, 2.0, 2.0])


def test_ensemble_mean_of_one_seed_is_that_seed():
    np.testing.assert_allclose(ensemble_mean([[0.5, 4.25]]), [0.5, 4.25])


def test_ensemble_mean_raises_on_ragged_seeds():
    with pytest.raises(ValueError, match="differing shapes"):
        ensemble_mean([[1.0, 2.0], [1.0]])


def test_ensemble_mean_raises_on_a_flat_prediction_vector():
    with pytest.raises(ValueError, match="n_seeds, n_compounds"):
        ensemble_mean([1.0, 2.0, 3.0])


def test_paired_bootstrap_interval_contains_zero_for_identical_predictors():
    rng = np.random.default_rng(3)
    true = rng.normal(loc=6.0, scale=1.0, size=60)
    pred = true + rng.normal(scale=0.5, size=60)

    result = paired_bootstrap(true, pred, pred, n_resamples=500, seed=0)

    assert result.observed == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)
    assert not excludes_zero(result)


def test_paired_bootstrap_interval_contains_zero_for_equally_noisy_predictors():
    rng = np.random.default_rng(5)
    true = rng.normal(loc=6.0, scale=1.0, size=60)
    good = true + rng.normal(scale=0.5, size=60)
    also_good = true + rng.normal(scale=0.5, size=60)

    result = paired_bootstrap(true, good, also_good, n_resamples=2000, seed=0)

    assert not excludes_zero(result)


def test_paired_bootstrap_interval_excludes_zero_when_one_predictor_is_better():
    rng = np.random.default_rng(7)
    true = rng.normal(loc=6.0, scale=1.0, size=60)
    good = true + rng.normal(scale=0.1, size=60)
    poor = true + rng.normal(scale=1.5, size=60)

    result = paired_bootstrap(true, good, poor, n_resamples=2000, seed=0)

    assert result.observed < 0.0
    assert result.ci_high < 0.0
    assert excludes_zero(result)


def test_paired_bootstrap_is_reproducible_under_a_fixed_seed():
    rng = np.random.default_rng(13)
    true = rng.normal(loc=6.0, scale=1.0, size=50)
    pred_a = true + rng.normal(scale=0.6, size=50)
    pred_b = true + rng.normal(scale=0.8, size=50)

    first = paired_bootstrap(true, pred_a, pred_b, n_resamples=400, seed=42)
    second = paired_bootstrap(true, pred_a, pred_b, n_resamples=400, seed=42)
    other = paired_bootstrap(true, pred_a, pred_b, n_resamples=400, seed=43)

    assert first == second
    assert isinstance(first, BootstrapResult)
    assert (other.ci_low, other.ci_high) != (first.ci_low, first.ci_high)


def test_paired_bootstrap_widens_the_interval_at_a_higher_level():
    rng = np.random.default_rng(17)
    true = rng.normal(loc=6.0, scale=1.0, size=60)
    pred_a = true + rng.normal(scale=0.5, size=60)
    pred_b = true + rng.normal(scale=0.7, size=60)

    narrow = paired_bootstrap(true, pred_a, pred_b, n_resamples=1000, level=0.80, seed=0)
    wide = paired_bootstrap(true, pred_a, pred_b, n_resamples=1000, level=0.99, seed=0)

    assert wide.ci_low < narrow.ci_low
    assert wide.ci_high > narrow.ci_high


def test_paired_bootstrap_defaults_to_a_95_percent_mae_interval():
    rng = np.random.default_rng(19)
    true = rng.normal(loc=6.0, scale=1.0, size=30)
    pred_a = true + rng.normal(scale=0.5, size=30)
    pred_b = true + rng.normal(scale=0.9, size=30)

    result = paired_bootstrap(true, pred_a, pred_b, n_resamples=200)

    assert result.level == 0.95
    assert result.metric == "mae"
    assert result.n_resamples == 200


def test_paired_bootstrap_observed_difference_matches_the_metric_difference():
    true = np.array(HAND_TRUE)
    pred_a = np.array(HAND_PRED)
    pred_b = true + 1.0

    result = paired_bootstrap(true, pred_a, pred_b, n_resamples=100, seed=0)

    assert result.observed == pytest.approx(0.625 - 1.0)


def test_paired_bootstrap_ranks_a_better_predictor_by_kendall_tau():
    rng = np.random.default_rng(23)
    true = rng.normal(loc=6.0, scale=1.0, size=60)
    good = true + rng.normal(scale=0.1, size=60)
    poor = rng.normal(loc=6.0, scale=1.0, size=60)

    result = paired_bootstrap(true, good, poor, metric="kendall_tau", n_resamples=300, seed=0)

    assert result.observed > 0.0
    assert excludes_zero(result)


def test_paired_bootstrap_raises_on_shape_mismatch_rather_than_broadcasting():
    with pytest.raises(ValueError, match="shape mismatch"):
        paired_bootstrap([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0])


def test_paired_bootstrap_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        paired_bootstrap([1.0, 2.0], [1.0, 2.0], [2.0, 1.0], metric="accuracy")


def test_paired_bootstrap_rejects_a_level_outside_the_unit_interval():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        paired_bootstrap([1.0, 2.0], [1.0, 2.0], [2.0, 1.0], level=1.0)


def test_excludes_zero_is_false_for_an_interval_touching_zero():
    touching = BootstrapResult("mae", -0.05, -0.1, 0.0, 0.95, 100)

    assert not excludes_zero(touching)
