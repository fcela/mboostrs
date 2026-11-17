"""Diagnostic observers preserve predictions and restore the original methods."""

import numpy as np
import pytest

from mboostrs import _rust
from benches.observe_calibration_scores import native_calls, control_predictions


def test_native_inputs_and_outputs_are_observed_without_changing_them():
    original = _rust.conformal_count_uniforms
    expected = original(5, 42, 0)
    events = []
    with native_calls(events):
        observed = _rust.conformal_count_uniforms(5, 42, 0)
    assert observed == expected
    assert _rust.conformal_count_uniforms is original
    assert events[0]["args"] == (5, 42, 0)
    assert events[0]["result"] == expected


def test_control_prediction_calls_keep_the_same_outputs_and_model_identity():
    pytest.importorskip("catboost")
    pytest.importorskip("lightgbm")
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    x = np.arange(40, dtype=float).reshape(20, 2)
    y = np.sin(x[:, 0])
    models = [
        CatBoostRegressor(iterations=2, thread_count=1, verbose=False, allow_writing_files=False),
        LGBMRegressor(n_estimators=2, n_jobs=1, verbosity=-1),
        GradientBoostingRegressor(n_estimators=2, random_state=0),
    ]
    for model in models:
        model.fit(x, y)
        expected = model.predict(x[:3])
        original = type(model).predict
        events = []
        with control_predictions(events):
            observed = model.predict(x[:3])
            model.predict(x[3:5])
        np.testing.assert_array_equal(observed, expected)
        assert events[0]["model"] == events[1]["model"] == 0
        assert [e["rows"] for e in events] == [3, 2]
        assert type(model).predict is original


def test_frozen_controls_expose_only_the_expected_calibration_and_query_calls():
    pytest.importorskip("catboost")
    pytest.importorskip("lightgbm")
    from benches.study_poisson_tuning_consistency import draw
    from benches.study_discovery_calibration import fit_band

    frame, _, _ = draw("original", 777)
    test, _, _ = draw("original", 100777)
    for method in ("catboost_split", "catboost_cqr", "lightgbm_split", "lightgbm_cqr", "sklearn_cqr"):
        events = []
        with control_predictions(events):
            fit_band(method, "poisson", 777, frame, test)
        expected = [500, 1000] if method.endswith("split") else [500, 500, 1000, 1000]
        assert [e["rows"] for e in events] == expected
