"""PropOdds truncated prediction and path-specific nuisance tests.

Validates that:
  1. predict(mstop=k, type="response") matches a separate mstop=k fit
  2. The invariant holds after pickle round trips
  3. Category-probability matrices sum to 1 per row
  4. CV works with 3 and 4 response levels
"""

import pickle

import numpy as np
import pytest
from mboostrs.cvrisk import cv_weights

import mboostrs as mboost


@pytest.fixture
def propodds_3class_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eta = 0.5 * x1 - 0.3 * x2
    thresholds = np.array([-0.8, 0.5])
    probs = np.column_stack([
        1.0 / (1.0 + np.exp(-(thresholds[0] - eta))),
        1.0 / (1.0 + np.exp(-(thresholds[1] - eta))) - 1.0 / (1.0 + np.exp(-(thresholds[0] - eta))),
        1.0 - 1.0 / (1.0 + np.exp(-(thresholds[1] - eta))),
    ])
    y = np.array([rng.choice([1, 2, 3], p=p) for p in probs], dtype=np.float64)
    return {"y": y, "x1": x1, "x2": x2, "n_levels": 3}


@pytest.fixture
def propodds_4class_data():
    rng = np.random.default_rng(123)
    n = 250
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eta = 0.4 * x1 + 0.2 * x2
    thresholds = np.array([-1.0, 0.0, 1.0])
    cum = [1.0 / (1.0 + np.exp(-(thresholds[j] - eta))) for j in range(3)]
    probs = np.column_stack([
        cum[0],
        cum[1] - cum[0],
        cum[2] - cum[1],
        1.0 - cum[2],
    ])
    y = np.array([rng.choice([1, 2, 3, 4], p=p) for p in probs], dtype=np.float64)
    return {"y": y, "x1": x1, "x2": x2, "n_levels": 4}


def _fit_model(data, n_levels, mstop, nu=0.1):
    family = mboost.PropOdds(n_levels=n_levels)
    ctrl = mboost.boost_control(mstop, nu)
    return mboost.glmboost(
        "y ~ x1 + x2",
        data={"y": data["y"], "x1": data["x1"], "x2": data["x2"]},
        family=family,
        control=ctrl,
    )


class TestPropOddsTruncatedPrediction:
    def test_glm_truncated_matches_separate_fit_3class(self, propodds_3class_data):
        d = propodds_3class_data
        full_model = _fit_model(d, 3, mstop=30)
        for k in [1, 5, 10, 15, 20, 30]:
            resp_full = full_model.predict(mstop=k, type="response")
            separate_model = _fit_model(d, 3, mstop=k)
            resp_separate = separate_model.predict(type="response")
            np.testing.assert_allclose(
                resp_full, resp_separate, rtol=1e-10, atol=1e-12,
                err_msg=f"Truncated predict(mstop={k}) disagrees with separate fit (3-class)",
            )

    def test_glm_truncated_matches_separate_fit_4class(self, propodds_4class_data):
        d = propodds_4class_data
        full_model = _fit_model(d, 4, mstop=30)
        for k in [1, 5, 10, 20, 30]:
            resp_full = full_model.predict(mstop=k, type="response")
            separate_model = _fit_model(d, 4, mstop=k)
            resp_separate = separate_model.predict(type="response")
            np.testing.assert_allclose(
                resp_full, resp_separate, rtol=1e-10, atol=1e-12,
                err_msg=f"Truncated predict(mstop={k}) disagrees with separate fit (4-class)",
            )

    def test_probabilities_sum_to_one(self, propodds_3class_data):
        d = propodds_3class_data
        model = _fit_model(d, 3, mstop=20)
        for k in [5, 10, 20]:
            resp = model.predict(mstop=k, type="response")
            row_sums = resp.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-10,
                err_msg=f"Probabilities don't sum to 1 at mstop={k}",
            )

    def test_pickle_roundtrip_preserves_nuisance_vec_path(self, propodds_3class_data):
        d = propodds_3class_data
        model = _fit_model(d, 3, mstop=20)
        pickled = pickle.dumps(model)
        model2 = pickle.loads(pickled)
        assert model2._nuisance_vec_path == model._nuisance_vec_path
        assert len(model2._nuisance_vec_path) == 21  # mstop+1
        # Full-mstop prediction should still match (uses stored _fitted)
        resp1 = model.predict(type="response")
        resp2 = model2.predict(type="response")
        np.testing.assert_allclose(resp1, resp2, rtol=1e-10)

    def test_nuisance_vec_path_length(self, propodds_3class_data):
        d = propodds_3class_data
        model = _fit_model(d, 3, mstop=25)
        # nuisance_vec_path should have mstop+1 entries (iteration 0 through mstop)
        assert len(model._nuisance_vec_path) == 26, (
            f"Expected 26 entries, got {len(model._nuisance_vec_path)}"
        )
        # Each entry should have n_levels-1 thresholds
        for i, entry in enumerate(model._nuisance_vec_path):
            assert len(entry) == 2, (
                f"Entry {i} has {len(entry)} thresholds, expected 2"
            )


class TestPropOddsCV:
    def test_cvrisk_3class(self, propodds_3class_data):
        d = propodds_3class_data
        model = _fit_model(d, 3, mstop=30)
        folds = cv_weights(n=len(d["y"]), type="kfold", B=5, seed=42)
        cv_result = model.cvrisk(folds=folds)
        assert cv_result.risk_matrix.shape[0] == 5
        assert cv_result.risk_matrix.shape[1] == 31  # mstop+1
        assert cv_result.optimal_mstop >= 1
        assert cv_result.optimal_mstop <= 30

    def test_cvrisk_4class(self, propodds_4class_data):
        d = propodds_4class_data
        model = _fit_model(d, 4, mstop=30)
        folds = cv_weights(n=len(d["y"]), type="kfold", B=5, seed=42)
        cv_result = model.cvrisk(folds=folds)
        assert cv_result.risk_matrix.shape[0] == 5
        assert cv_result.risk_matrix.shape[1] == 31
        assert cv_result.optimal_mstop >= 1
