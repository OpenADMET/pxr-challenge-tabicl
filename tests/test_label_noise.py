"""Tests for splitting a predicted spread into measurement noise and model error."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import label_noise


def test_spearman_rows_matches_scipy_per_row():
    rng = np.random.default_rng(0)
    x = rng.random(30)
    rows = rng.random((4, 30))

    expected = [stats.spearmanr(x, row).statistic for row in rows]

    np.testing.assert_allclose(label_noise.spearman_rows(x, rows), expected, atol=1e-12)


def test_measurement_noise_caps_curve_fit_se_at_sigma():
    sigma = np.array([0.5, 0.2])
    se = np.array([0.1, 0.4])

    np.testing.assert_allclose(label_noise.measurement_noise("curve_fit", sigma, se), [0.1, 0.2])


def test_measurement_noise_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown noise model"):
        label_noise.measurement_noise("replicate", np.ones(2), np.ones(2))


def test_share_model_assigns_the_given_fraction_of_variance():
    sigma = np.array([0.4, 1.0])

    noise = label_noise.measurement_noise("share_0.25", sigma, np.zeros(2))

    np.testing.assert_allclose(noise**2, [0.04, 0.25])


def test_pair_ranking_probability_is_a_coin_flip_for_equal_sigma():
    np.testing.assert_allclose(label_noise.pair_ranking_probability(np.array([1.0])), [0.5])


def test_pair_ranking_probability_matches_monte_carlo():
    draws = np.abs(np.random.default_rng(4).standard_normal((2, 400_000)))

    observed = np.mean(1.5 * draws[0] > draws[1])

    assert abs(label_noise.pair_ranking_probability(np.array([1.5]))[0] - observed) < 3e-3


def test_replicates_approach_perfect_ranking_when_all_of_sigma_is_noise():
    sigma = np.linspace(0.2, 1.0, 100)
    rng = np.random.default_rng(1)

    rho = label_noise.simulate_replicates(sigma, sigma, 400, 20, rng)

    assert rho.mean() > 0.95


def test_replicates_do_not_change_rho_without_measurement_noise():
    sigma = np.linspace(0.2, 1.0, 100)
    noise = np.zeros_like(sigma)

    one = label_noise.simulate_replicates(sigma, noise, 1, 50, np.random.default_rng(2))
    many = label_noise.simulate_replicates(sigma, noise, 30, 50, np.random.default_rng(2))

    np.testing.assert_allclose(one, many)


def test_folded_normal_mean_matches_monte_carlo():
    offset = np.array([-0.4, 0.0, 0.7])
    scale = np.array([0.3, 0.5, 0.2])
    draws = offset + np.random.default_rng(3).standard_normal((200_000, 3)) * scale

    np.testing.assert_allclose(
        label_noise.folded_normal_mean(offset, scale), np.abs(draws).mean(axis=0), atol=5e-3
    )


def test_folded_normal_mean_is_absolute_offset_at_zero_scale():
    offset = np.array([-0.4, 0.7])

    np.testing.assert_allclose(label_noise.folded_normal_mean(offset, np.zeros(2)), [0.4, 0.7])
