"""Tests for prediction contribution decomposition (Milestone 7)."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture
def gaussian_glm():
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    return mb.glmboost(
        "y ~ x1 + x2",
        data={"y": y, "x1": x1, "x2": x2},
        family=mb.Gaussian(),
        control=mb.boost_control(30, 0.1),
    )


@pytest.fixture
def gaussian_gam():
    rng = np.random.default_rng(7)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.sin(x1) + 0.5 * x2 + rng.normal(scale=0.3, size=n)
    return mb.gamboost(
        "y ~ bbs(x1, df=4) + bols(x2)",
        data={"y": y, "x1": x1, "x2": x2},
        family=mb.Gaussian(),
        control=mb.boost_control(40, 0.1),
    )


# ─── GLMBoostModel ───────────────────────────────────────────────────────────


class TestGLMPredictionContributions:
    def test_predict_which_string(self, gaussian_glm):
        contrib_x1 = gaussian_glm.predict(which="x1")
        assert contrib_x1.shape == (80,)

    def test_predict_which_int(self, gaussian_glm):
        contrib_idx1 = gaussian_glm.predict(which=1)
        contrib_x1 = gaussian_glm.predict(which="x1")
        np.testing.assert_allclose(contrib_idx1, contrib_x1)

    def test_predict_which_list(self, gaussian_glm):
        both = gaussian_glm.predict(which=["x1", "x2"])
        x1_only = gaussian_glm.predict(which="x1")
        x2_only = gaussian_glm.predict(which="x2")
        np.testing.assert_allclose(both, x1_only + x2_only, atol=1e-14)

    def test_contributions_sum_to_full(self, gaussian_glm):
        full = gaussian_glm.predict()
        contrib_x1 = gaussian_glm.predict(which="x1")
        contrib_x2 = gaussian_glm.predict(which="x2")
        contrib_int = gaussian_glm.predict(which="(Intercept)")
        reconstruction = contrib_x1 + contrib_x2 + contrib_int + gaussian_glm.offset_
        np.testing.assert_allclose(reconstruction, full, atol=1e-14)

    def test_aggregate_none(self, gaussian_glm):
        steps = gaussian_glm.predict(aggregate="none")
        assert steps.shape == (30, 80)
        full = gaussian_glm.predict()
        np.testing.assert_allclose(steps.sum(axis=0) + gaussian_glm.offset_, full, atol=1e-12)

    def test_aggregate_cumsum(self, gaussian_glm):
        cumsum = gaussian_glm.predict(aggregate="cumsum")
        assert cumsum.shape == (30, 80)
        full = gaussian_glm.predict()
        np.testing.assert_allclose(cumsum[-1], full, atol=1e-12)

    def test_aggregate_none_with_which(self, gaussian_glm):
        steps = gaussian_glm.predict(which="x1", aggregate="none")
        assert steps.shape == (30, 80)
        x1_sum = gaussian_glm.predict(which="x1")
        np.testing.assert_allclose(steps.sum(axis=0), x1_sum, atol=1e-14)

    def test_aggregate_cumsum_with_which(self, gaussian_glm):
        cumsum = gaussian_glm.predict(which="x1", aggregate="cumsum")
        assert cumsum.shape == (30, 80)
        x1_sum = gaussian_glm.predict(which="x1")
        np.testing.assert_allclose(cumsum[-1], x1_sum, atol=1e-14)

    def test_prediction_contributions_method(self, gaussian_glm):
        pc = gaussian_glm.prediction_contributions(which="x1")
        via_predict = gaussian_glm.predict(which="x1")
        np.testing.assert_allclose(pc, via_predict, atol=1e-14)

    def test_aggregate_invalid(self, gaussian_glm):
        with pytest.raises(ValueError, match="aggregate"):
            gaussian_glm.predict(aggregate="bad")

    def test_type_invalid(self, gaussian_glm):
        with pytest.raises(ValueError, match="type"):
            gaussian_glm.predict(type="bad")

    def test_newdata(self, gaussian_glm):
        newdata = {"x1": np.ones(5), "x2": np.zeros(5)}
        contrib = gaussian_glm.predict(newdata=newdata, which="x1")
        assert contrib.shape == (5,)

    def test_mstop(self, gaussian_glm):
        half_mstop = gaussian_glm.predict(mstop=15, aggregate="none")
        assert half_mstop.shape[0] == 15

    def test_type_response(self, gaussian_glm):
        pred_resp = gaussian_glm.predict(type="response")
        contrib_x1_resp = gaussian_glm.predict(which="x1", type="response")
        assert pred_resp.shape == (80,)
        assert contrib_x1_resp.shape == (80,)

    def test_type_response_cumsum_full(self, gaussian_glm):
        cumsum_resp = gaussian_glm.predict(type="response", aggregate="cumsum")
        full_resp = gaussian_glm.predict(type="response")
        np.testing.assert_allclose(cumsum_resp[-1], full_resp, atol=1e-12)

    def test_type_response_cumsum_with_which_raises(self, gaussian_glm):
        with pytest.raises(ValueError, match="which=None"):
            gaussian_glm.predict(type="response", aggregate="cumsum", which="x1")

    def test_type_response_none_raises(self, gaussian_glm):
        with pytest.raises(ValueError, match="not additive"):
            gaussian_glm.predict(type="response", aggregate="none")


# ─── GAMBoostModel ───────────────────────────────────────────────────────────


class TestGAMPredictionContributions:
    def test_predict_which_string(self, gaussian_gam):
        contrib_x1 = gaussian_gam.predict(which="x1")
        assert contrib_x1.shape == (100,)

    def test_contributions_sum_to_full(self, gaussian_gam):
        full = gaussian_gam.predict()
        contrib_x1 = gaussian_gam.predict(which="x1")
        contrib_x2 = gaussian_gam.predict(which="x2")
        reconstruction = contrib_x1 + contrib_x2 + gaussian_gam.offset_
        np.testing.assert_allclose(reconstruction, full, atol=1e-12)

    def test_aggregate_none(self, gaussian_gam):
        steps = gaussian_gam.predict(aggregate="none")
        assert steps.shape == (40, 100)
        full = gaussian_gam.predict()
        np.testing.assert_allclose(steps.sum(axis=0) + gaussian_gam.offset_, full, atol=1e-10)

    def test_aggregate_cumsum(self, gaussian_gam):
        cumsum = gaussian_gam.predict(aggregate="cumsum")
        assert cumsum.shape == (40, 100)
        full = gaussian_gam.predict()
        np.testing.assert_allclose(cumsum[-1], full, atol=1e-10)

    def test_aggregate_none_with_which(self, gaussian_gam):
        steps = gaussian_gam.predict(which="x1", aggregate="none")
        assert steps.shape == (40, 100)
        x1_sum = gaussian_gam.predict(which="x1")
        np.testing.assert_allclose(steps.sum(axis=0), x1_sum, atol=1e-10)

    def test_prediction_contributions_method(self, gaussian_gam):
        pc = gaussian_gam.prediction_contributions(which="x1")
        via_predict = gaussian_gam.predict(which="x1")
        np.testing.assert_allclose(pc, via_predict, atol=1e-14)

    def test_aggregate_invalid(self, gaussian_gam):
        with pytest.raises(ValueError, match="aggregate"):
            gaussian_gam.predict(aggregate="bad")

    def test_type_response(self, gaussian_gam):
        pred_resp = gaussian_gam.predict(type="response")
        contrib_x1_resp = gaussian_gam.predict(which="x1", type="response")
        assert pred_resp.shape == (100,)
        assert contrib_x1_resp.shape == (100,)

    def test_type_response_cumsum_full(self, gaussian_gam):
        cumsum_resp = gaussian_gam.predict(type="response", aggregate="cumsum")
        full_resp = gaussian_gam.predict(type="response")
        np.testing.assert_allclose(cumsum_resp[-1], full_resp, atol=1e-10)

    def test_type_response_cumsum_with_which_raises(self, gaussian_gam):
        with pytest.raises(ValueError, match="which=None"):
            gaussian_gam.predict(type="response", aggregate="cumsum", which="x1")

    def test_type_response_none_raises(self, gaussian_gam):
        with pytest.raises(ValueError, match="not additive"):
            gaussian_gam.predict(type="response", aggregate="none")

    def test_mstop(self, gaussian_gam):
        steps = gaussian_gam.predict(mstop=20, aggregate="none")
        assert steps.shape[0] == 20

    def test_predict_sum_which_none_is_predict(self, gaussian_gam):
        """predict(aggregate='sum', which=None) returns same as predict()."""
        normal = gaussian_gam.predict()
        via_contrib = gaussian_gam.predict(aggregate="sum")
        np.testing.assert_allclose(via_contrib, normal, atol=1e-14)

    # ─── newdata + aggregate=none/cumsum (regression: was ignoring newdata) ──
    # Before the fix, GAMBoostModel._predict_single_step took the
    # LearnerStore fast path even for newdata, so these returned
    # training-length output (n=100) instead of the newdata row count.

    @pytest.fixture
    def _nd3(self):
        return {
            "x1": np.array([0.5, -0.5, 1.0]),
            "x2": np.array([0.2, -0.2, 0.4]),
        }

    def test_aggregate_none_newdata_shape_and_sum(self, gaussian_gam, _nd3):
        steps = gaussian_gam.predict(newdata=_nd3, aggregate="none")
        assert steps.shape == (40, 3)
        full = gaussian_gam.predict(newdata=_nd3, type="link")
        np.testing.assert_allclose(
            steps.sum(axis=0) + gaussian_gam.offset_, full, atol=1e-10
        )

    def test_aggregate_cumsum_newdata_shape_and_full(self, gaussian_gam, _nd3):
        cumsum = gaussian_gam.predict(newdata=_nd3, aggregate="cumsum")
        assert cumsum.shape == (40, 3)
        full = gaussian_gam.predict(newdata=_nd3, type="link")
        np.testing.assert_allclose(cumsum[-1], full, atol=1e-10)

    def test_aggregate_cumsum_newdata_with_which(self, gaussian_gam, _nd3):
        cumsum = gaussian_gam.predict(newdata=_nd3, which="x1", aggregate="cumsum")
        assert cumsum.shape == (40, 3)
        x1_sum = gaussian_gam.predict(newdata=_nd3, which="x1")
        np.testing.assert_allclose(cumsum[-1], x1_sum, atol=1e-10)
