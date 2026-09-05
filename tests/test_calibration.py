from __future__ import annotations

import numpy as np
import pytest

import calibration
import uncertainty


def test_the_affine_map_recovers_a_known_linear_distortion():
    rng = np.random.default_rng(0)
    truth = rng.normal(5.0, 1.5, size=400)
    # a model that is systematically shrunk toward the mean and offset
    predicted = 0.6 * truth + 1.4

    slope, intercept = calibration.fit_affine(predicted, truth)

    assert slope == pytest.approx(1 / 0.6, rel=1e-6)
    assert intercept == pytest.approx(-1.4 / 0.6, rel=1e-6)


def test_weights_tilt_the_map_toward_the_compounds_that_carry_them():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    # the first pair lies on a slope of 1 and the second on a slope of 6, so
    # weighting the second up has to raise the fitted slope toward 6
    y = np.array([0.0, 1.0, 10.0, 16.0])

    unweighted, _ = calibration.fit_affine(x, y)
    tilted, _ = calibration.fit_affine(x, y, weights=np.array([0.01, 0.01, 1.0, 1.0]))

    assert tilted > unweighted


def test_the_density_ratio_separates_two_disjoint_chemotypes():
    aliphatic = ["CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CC(C)CC", "CCC(C)C"]
    aromatic = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "c1ccc2ccccc2c1"]

    weights, auc = calibration.density_ratio_weights(aliphatic, aromatic)

    # the sets are trivially separable, so the fit compounds are all unlike the
    # test set and the ratio pins them at the lower clip
    assert auc > 0.99
    assert weights.shape == (len(aliphatic),)
    assert np.all(weights >= calibration.WEIGHT_CLIP[0])
    assert np.all(weights <= calibration.WEIGHT_CLIP[1])


def test_weights_are_clipped_to_the_reported_bound():
    # the report calls the clip essential, so it is a property worth pinning
    # rather than an implementation detail
    fit = ["CCCC", "CCCCC", "c1ccccc1", "Cc1ccccc1"]
    test = ["c1ccccc1", "Cc1ccccc1"]

    weights, _ = calibration.density_ratio_weights(fit, test)

    low, high = calibration.WEIGHT_CLIP
    assert (low, high) == (pytest.approx(1 / 3), 3.0)
    assert np.all((weights >= low) & (weights <= high))


def test_a_positive_slope_cannot_change_a_ranking():
    rng = np.random.default_rng(1)
    predicted = rng.normal(size=100)
    fitted = calibration.AffineCalibration(
        slope=1.7, intercept=-0.4, n_fit=100, weight_summary={}, classifier_auc=0.5
    )

    assert np.array_equal(np.argsort(fitted.apply(predicted)), np.argsort(predicted))


def test_folds_cover_every_compound_exactly_once():
    folds = calibration.fold_indices(97, n_folds=5)
    held = np.concatenate([h for _, h in folds])

    assert len(folds) == 5
    assert sorted(held.tolist()) == list(range(97))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_folds": 1}, "at least 2 folds"),
        ({"n_folds": 200}, "over 97 compounds"),
    ],
)
def test_impossible_fold_counts_are_refused(kwargs, match):
    with pytest.raises(calibration.CalibrationError, match=match):
        calibration.fold_indices(97, **kwargs)


def test_a_constant_prediction_leaves_the_slope_undetermined():
    with pytest.raises(calibration.CalibrationError, match="constant"):
        calibration.fit_affine(np.ones(10), np.arange(10, dtype=float))


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


def test_calibration_rescales_the_spread_with_the_mean():
    # the affine map moves the whole predictive distribution, so leaving the
    # spread alone would report a distribution the calibration does not imply
    sigma = np.array([0.5, 1.0, 2.0])

    assert np.allclose(uncertainty.rescale_sigma(sigma, 2.0), [1.0, 2.0, 4.0])
    assert np.allclose(uncertainty.rescale_sigma(sigma, -2.0), [1.0, 2.0, 4.0])


def test_coverage_is_undefined_when_every_spread_is_zero():
    with pytest.raises(uncertainty.UncertaintyError, match="undefined"):
        uncertainty.coverage_curve(np.arange(5.0), np.arange(5.0), np.zeros(5))
