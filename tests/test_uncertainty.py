from __future__ import annotations

import numpy as np
import pytest

import uncertainty


def test_a_well_specified_gaussian_is_nearly_calibrated():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=2000)
    predicted = truth + rng.normal(scale=1.0, size=2000)

    d = uncertainty.diagnose(truth, predicted, np.ones(2000), source="model")

    # sampling noise keeps this off zero, but it must be far below what an
    # overconfident model scores
    assert d.miscalibration_area < 0.05
    assert abs(d.signed_gap) < 0.05


def test_an_overconfident_model_scores_worse_and_says_which_way():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=2000)
    predicted = truth + rng.normal(scale=1.0, size=2000)

    tight = uncertainty.diagnose(truth, predicted, np.full(2000, 0.3), source="model")
    loose = uncertainty.diagnose(truth, predicted, np.full(2000, 3.0), source="model")

    assert tight.miscalibration_area > 0.2
    assert loose.miscalibration_area > 0.2
    # too narrow means the truth falls outside the interval too often
    assert tight.signed_gap < 0
    assert loose.signed_gap > 0


def test_a_spread_that_tracks_the_error_correlates_with_it():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.2, 2.0, size=500)
    truth = rng.normal(size=500)
    predicted = truth + rng.normal(scale=sigma)

    d = uncertainty.diagnose(truth, predicted, sigma, source="model")

    assert d.spearman > 0.3


def test_a_constant_spread_reports_no_correlation_rather_than_zero():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=100)

    d = uncertainty.diagnose(truth, truth + rng.normal(size=100), np.ones(100), source="model")

    assert np.isnan(d.spearman)
    assert np.isnan(d.pearson)


def test_the_ensemble_spread_needs_more_than_one_seed():
    with pytest.raises(uncertainty.UncertaintyError, match="at least two seeds"):
        uncertainty.ensemble_spread(np.zeros((1, 10)))


def test_coverage_is_undefined_when_every_spread_is_zero():
    with pytest.raises(uncertainty.UncertaintyError, match="undefined"):
        uncertainty.coverage_curve(np.arange(5.0), np.arange(5.0), np.zeros(5))
