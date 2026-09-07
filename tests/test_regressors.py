"""Contract tests for the uniform regressor adapter."""

from __future__ import annotations

import dataclasses
import json
import logging

import numpy as np
import pytest

import regressors

# the two regressors that fit in milliseconds on CPU and so run on every pass
FAST_NAMES = ("lgbm", "xgboost")

# the GPU foundation models, deselected by default via the pytest addopts
SLOW_NAMES = ("tabpfn-v2.5", "tabpfn-v2.6", "tabpfn-v3", "tabicl", "tabfm")

# names whose predictions carry a predictive spread
SPREAD_NAMES = ("tabpfn-v2.5", "tabpfn-v2.6", "tabpfn-v3", "tabicl")


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a small linear regression problem with a train and a test half."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 5))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.1 * rng.normal(size=120)
    return x[:90], y[:90], x[90:], y[90:]


@pytest.mark.parametrize("name", FAST_NAMES)
def test_prediction_has_one_mean_per_test_row(name, data):
    x_train, y_train, x_test, _ = data

    prediction = regressors.fit_predict(name, x_train, y_train, x_test, seed=0)

    assert isinstance(prediction, regressors.Prediction)
    assert prediction.mean.shape == (x_test.shape[0],)


@pytest.mark.parametrize("name", FAST_NAMES)
def test_gradient_boosting_reports_no_predictive_spread(name, data):
    x_train, y_train, x_test, _ = data

    prediction = regressors.fit_predict(name, x_train, y_train, x_test, seed=0)

    assert prediction.std is None


@pytest.mark.parametrize("name", FAST_NAMES)
def test_prediction_tracks_the_signal_it_was_trained_on(name, data):
    x_train, y_train, x_test, y_test = data

    prediction = regressors.fit_predict(name, x_train, y_train, x_test, seed=0)

    # the target is a clean linear function of two columns, so any working
    # regressor beats predicting the training mean by a wide margin
    baseline = np.abs(y_test - y_train.mean()).mean()
    assert np.abs(y_test - prediction.mean).mean() < 0.5 * baseline


@pytest.mark.parametrize("name", FAST_NAMES)
def test_same_seed_gives_the_same_numbers(name, data):
    x_train, y_train, x_test, _ = data

    first = regressors.fit_predict(name, x_train, y_train, x_test, seed=7)
    second = regressors.fit_predict(name, x_train, y_train, x_test, seed=7)

    np.testing.assert_allclose(first.mean, second.mean, rtol=0, atol=0)


def test_mismatched_test_width_is_rejected(data):
    x_train, y_train, x_test, _ = data

    with pytest.raises(ValueError, match="columns"):
        regressors.fit_predict("lgbm", x_train, y_train, x_test[:, :3], seed=0)


@pytest.mark.parametrize("name", FAST_NAMES + SLOW_NAMES)
def test_resolved_params_are_json_serialisable(name):
    resolved = regressors.resolved_params(name, seed=3, n_features=130)

    assert json.loads(json.dumps(resolved, sort_keys=True)) == resolved


@pytest.mark.parametrize("name", FAST_NAMES + SLOW_NAMES)
def test_resolved_params_are_stable_across_calls(name):
    first = regressors.resolved_params(name, seed=3, n_features=130)
    second = regressors.resolved_params(name, seed=3, n_features=130)

    assert first == second


@pytest.mark.parametrize("name", FAST_NAMES + SLOW_NAMES)
def test_resolved_params_carry_the_seed(name):
    resolved = regressors.resolved_params(name, seed=11, n_features=130)

    assert resolved["random_state"] == 11


@pytest.mark.parametrize("name", FAST_NAMES + SLOW_NAMES)
def test_every_regressor_declares_a_semantic_version(name):
    assert regressors.REGRESSORS[name].version >= 1


def test_registry_holds_exactly_the_documented_names():
    # the ensemble sizes are v3 under a fixed member count, so they belong to
    # the registry without being separate models
    expected = FAST_NAMES + SLOW_NAMES + regressors.TABPFN_ENSEMBLE_NAMES
    assert sorted(regressors.REGRESSORS) == sorted(expected)


def test_every_ensemble_size_is_the_v3_checkpoint_at_a_fixed_member_count():
    for name, size in zip(
        regressors.TABPFN_ENSEMBLE_NAMES, regressors.TABPFN_ENSEMBLE_SIZES, strict=True
    ):
        resolved = regressors.resolved_params(name, seed=0, n_features=258)
        assert resolved["n_estimators"] == size
        # the memory settings are pinned for these exactly as for plain v3
        assert resolved["fit_mode"] == "low_memory"
        assert resolved["memory_saving_mode"] is True


def test_the_default_member_count_is_swept():
    # the library default is 8, which is what every other result was produced
    # under; without it in the sweep there is nothing to check the rest against
    assert 8 in regressors.TABPFN_ENSEMBLE_SIZES


def test_an_ensemble_size_reports_a_predictive_spread():
    # _predict reads the bar distribution off names in TABPFN_NAMES, so a
    # variant missing from it would silently write no spread at all
    for name in regressors.TABPFN_ENSEMBLE_NAMES:
        assert name in regressors.TABPFN_NAMES


def test_tabicl_runs_under_pinned_memory_settings():
    # the widths this sweep produces do not share a safe batch size at the
    # library default, so the smallest one is pinned and the whole sweep runs
    # under it rather than under a setting searched per featureset
    resolved = regressors.resolved_params("tabicl", seed=0, n_features=130)

    assert resolved["batch_size"] == regressors.TABICL_BATCH_SIZE == 1
    assert resolved["kv_cache"] is regressors.TABICL_KV_CACHE is False
    assert resolved["offload_mode"] == regressors.TABICL_OFFLOAD_MODE == "cpu"


def test_tabicl_memory_settings_can_still_be_overridden():
    resolved = regressors.resolved_params(
        "tabicl", seed=0, n_features=130, params={"batch_size": 8, "offload_mode": "auto"}
    )

    assert resolved["batch_size"] == 8
    assert resolved["offload_mode"] == "auto"


@pytest.mark.parametrize("name", regressors.TABPFN_NAMES)
def test_every_tabpfn_checkpoint_runs_under_pinned_memory_settings(name):
    resolved = regressors.resolved_params(name, seed=0, n_features=130)

    assert resolved["fit_mode"] == "low_memory"
    assert resolved["memory_saving_mode"] is True


@pytest.mark.parametrize(
    ("name", "setting"),
    [
        ("tabicl", "batch_size"),
        ("tabicl", "kv_cache"),
        ("tabicl", "offload_mode"),
        ("tabpfn-v3", "fit_mode"),
        ("tabpfn-v3", "memory_saving_mode"),
    ],
)
def test_a_memory_setting_is_a_real_constructor_argument(name, setting):
    # pinning a keyword the library does not take would fail only at fit time,
    # deep into a sweep, so the names are checked against the installed version
    import inspect

    estimator = regressors.build(name, seed=0, n_features=130)

    assert setting in inspect.signature(type(estimator).__init__).parameters


def test_tabfm_caps_its_in_context_rows():
    resolved = regressors.resolved_params("tabfm", seed=0, n_features=130)

    assert resolved["max_num_rows"] == 500


@pytest.mark.parametrize(
    ("n_features", "expected"),
    [(130, False), (2000, False), (2300, True)],
    ids=["narrow", "at-the-limit", "past-the-limit"],
)
def test_tabpfn_ignores_pretraining_limits_only_past_the_cap(n_features, expected):
    resolved = regressors.resolved_params("tabpfn-v3", seed=0, n_features=n_features)

    assert resolved["ignore_pretraining_limits"] is expected


def test_an_explicit_pretraining_limit_flag_wins_over_the_derived_one():
    resolved = regressors.resolved_params(
        "tabpfn-v3", seed=0, n_features=130, params={"ignore_pretraining_limits": True}
    )

    assert resolved["ignore_pretraining_limits"] is True


def test_overrides_do_not_mutate_the_registry_defaults():
    regressors.resolved_params("lgbm", seed=0, n_features=5, params={"verbose": 1})

    assert regressors.REGRESSORS["lgbm"].defaults["verbose"] == -1


@pytest.mark.parametrize("call", ["resolved_params", "build"])
def test_unknown_name_raises(call):
    with pytest.raises(KeyError, match="unknown regressor"):
        getattr(regressors, call)("catboost", seed=0, n_features=5)


def test_unknown_name_raises_before_a_fit_is_attempted(data):
    x_train, y_train, x_test, _ = data

    with pytest.raises(KeyError, match="unknown regressor"):
        regressors.fit_predict("catboost", x_train, y_train, x_test, seed=0)


@pytest.mark.parametrize("name", FAST_NAMES)
def test_build_returns_an_unfitted_estimator_with_the_sklearn_api(name):
    model = regressors.build(name, seed=0, n_features=5)

    assert callable(model.fit)
    assert callable(model.predict)


@pytest.mark.gpu
@pytest.mark.parametrize("name", SLOW_NAMES)
def test_foundation_model_predicts_every_test_row(name, data):
    x_train, y_train, x_test, _ = data

    prediction = regressors.fit_predict(name, x_train, y_train, x_test, seed=0)

    assert prediction.mean.shape == (x_test.shape[0],)


@pytest.mark.gpu
@pytest.mark.parametrize("name", SPREAD_NAMES)
def test_foundation_model_reports_a_positive_predictive_spread(name, data):
    x_train, y_train, x_test, _ = data

    prediction = regressors.fit_predict(name, x_train, y_train, x_test, seed=0)

    assert prediction.std is not None
    assert prediction.std.shape == (x_test.shape[0],)
    assert np.all(prediction.std > 0.0)


@pytest.mark.gpu
def test_tabfm_reports_no_predictive_spread(data):
    x_train, y_train, x_test, _ = data

    prediction = regressors.fit_predict("tabfm", x_train, y_train, x_test, seed=0)

    assert prediction.std is None


def test_a_library_warning_during_a_fit_reaches_the_prediction(monkeypatch):
    # TabPFN halves its row chunk when it runs out of memory and says so
    # through the logging module; a record that omits it describes a fit that
    # ran under parameters nobody pinned
    def noisy(params):
        class _Model:
            def fit(self, x, y):
                logging.getLogger("tabpfn.architectures.tabpfn_v3").warning(
                    "OOM: halving row_chunk_size to 1024"
                )

            def predict(self, x):
                return np.zeros(len(x))

        return _Model()

    monkeypatch.setitem(
        regressors.REGRESSORS,
        "lgbm",
        dataclasses.replace(regressors.REGRESSORS["lgbm"], construct=noisy),
    )

    prediction = regressors.fit_predict(
        "lgbm", np.zeros((4, 2)), np.zeros(4), np.zeros((2, 2)), seed=0
    )

    assert any("halving row_chunk_size" in note for note in prediction.notes)
    assert any(note.startswith("tabpfn.architectures") for note in prediction.notes)


def test_a_quiet_fit_records_no_notes():
    prediction = regressors.fit_predict(
        "lgbm", np.zeros((8, 2)), np.arange(8, dtype=float), np.zeros((2, 2)), seed=0
    )

    assert prediction.notes == ()
