"""Tests for Milestone 10: R functionality diff harness."""

import json

import numpy as np
import pytest
from mboostrs.compare import (
    CompareReport,
    CompareSpec,
    _compute_diffs,
    compare_to_r,
)


class TestCompareSpec:
    def test_creation(self):
        spec = CompareSpec(model_type="glmboost", formula="y ~ x1 + x2",
                          family="Gaussian", mstop=50, nu=0.1)
        assert spec.model_type == "glmboost"
        assert spec.mstop == 50

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            CompareSpec(model_type="bad")

    def test_to_dict(self):
        spec = CompareSpec()
        d = spec.to_dict()
        assert "model_type" in d
        assert "formula" in d

    def test_to_json(self):
        spec = CompareSpec()
        j = spec.to_json()
        parsed = json.loads(j)
        assert parsed["model_type"] == "glmboost"

    def test_from_dict(self):
        d = {"model_type": "gamboost", "formula": "y ~ x1", "family": "Binomial",
             "mstop": 20, "nu": 0.1, "n": 100, "seed": 42, "risk": "inbag"}
        spec = CompareSpec.from_dict(d)
        assert spec.model_type == "gamboost"
        assert spec.mstop == 20


class TestCompareReport:
    def test_to_json_serializable(self):
        spec = CompareSpec()
        report = CompareReport(spec)
        report.status = "r_unavailable"
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["status"] == "r_unavailable"

    def test_summary(self):
        spec = CompareSpec(family="Gaussian")
        report = CompareReport(spec)
        report.status = "r_unavailable"
        s = report.summary()
        assert "Gaussian" in s


class TestComputeDiffs:
    def test_lss_parameter_dicts_are_compared_numerically(self):
        py = {"fitted": {"mu": np.ones(3), "sigma": np.full(3, 2.0)}, "risk": [2, 1]}
        r = {"fitted": np.array([1, 1, 1, 2, 2, 2]), "risk": [2, 1]}
        diffs, maxima = _compute_diffs(py, r)
        assert maxima["fitted"] == 0.0
        assert not diffs["fitted"].get("mismatch", False)

    def test_missing_required_output_is_a_mismatch(self):
        diffs, maxima = _compute_diffs({"fitted": [1]}, {"fitted": [1], "risk": [1]})
        assert diffs["risk"]["mismatch"] is True
        assert np.isinf(maxima["risk"])

    def test_offset_type_mismatch_is_a_mismatch(self):
        diffs, maxima = _compute_diffs(
            {"fitted": [1], "risk": [1], "offset": {"mu": 0, "sigma": 1}},
            {"fitted": [1], "risk": [1], "offset": [0]},
        )
        assert diffs["offset"]["shape_mismatch"] is True
        assert np.isinf(maxima["offset"])


class TestCompareToR:
    def test_glmboost_python_always_works(self):
        """Python side should always produce results, even without R."""
        spec = CompareSpec(model_type="glmboost", formula="y ~ x1 + x2",
                          family="Gaussian", mstop=30, nu=0.1, n=100)
        report = compare_to_r(spec)
        assert report.status in ("ok", "r_unavailable")
        assert "fitted" in report.py_result
        assert len(report.py_result["fitted"]) == 100
        assert len(report.py_result["risk"]) == 31

    def test_gamboost_python_always_works(self):
        spec = CompareSpec(model_type="gamboost", formula="y ~ bbs(x1, df=4) + bols(x2)",
                          family="Gaussian", mstop=20)
        report = compare_to_r(spec)
        # Python side should always succeed even if R errors
        assert "fitted" in report.py_result
        assert len(report.py_result["fitted"]) == 200

    def test_glmboostLSS_python_always_works(self):
        spec = CompareSpec(model_type="glmboostLSS", formula="y ~ x1 + x2",
                          family="GaussianLSS", mstop=20)
        report = compare_to_r(spec)
        assert report.status in ("ok", "r_unavailable")
        assert len(report.py_result["fitted"]) == 2 * spec.n

    def test_gamboostLSS_python_always_works(self):
        spec = CompareSpec(
            model_type="gamboostLSS",
            formula="y ~ bbs(x1, df=4) + bbs(x2, df=4)",
            family="GaussianLSS",
            mstop=5,
            n=40,
        )
        report = compare_to_r(spec)
        assert report.status in ("ok", "r_unavailable")
        assert len(report.py_result["fitted"]) == 2 * spec.n

    def test_json_serializable_all_statuses(self):
        spec = CompareSpec(model_type="glmboost", mstop=10, n=50)
        report = compare_to_r(spec)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["status"] in ("ok", "r_unavailable", "error")

    @pytest.mark.r_parity
    def test_glmboost_r_parity(self):
        """When R is available, GLM fitted/risk/offset should match to ~1e-10."""
        spec = CompareSpec(model_type="glmboost", formula="y ~ x1 + x2",
                          family="Gaussian", mstop=30, nu=0.1, n=100)
        report = compare_to_r(spec)
        assert report.status == "ok"
        assert report.max_abs_diff["fitted"] < 1e-10
        assert report.max_abs_diff["risk"] < 1e-10

    @pytest.mark.r_parity
    def test_binomial_r_parity(self):
        """Binomial takes a 0/1 numeric response on the Python side; the R side
        converts it to the two-level factor that mboost's Binomial() requires."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        eta = 0.5 * x1 - 0.3 * x2
        p = 1 / (1 + np.exp(-eta))
        y = (rng.uniform(size=n) < p).astype(float)
        data = {"x1": x1, "x2": x2, "y": y}
        spec = CompareSpec(model_type="glmboost", formula="y ~ x1 + x2",
                          family="Binomial", mstop=20, nu=0.1, n=n)
        report = compare_to_r(spec, data=data)
        assert report.error is None, report.error
        assert report.status == "ok"
        assert report.max_abs_diff["fitted"] < 1e-8

    def test_custom_data(self):
        """Pass custom data instead of auto-generated."""
        rng = np.random.default_rng(99)
        n = 50
        data = {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
        spec = CompareSpec(model_type="glmboost", mstop=20)
        report = compare_to_r(spec, data=data)
        assert len(report.py_result["fitted"]) == 50

    def test_error_propagation(self):
        """Invalid family should produce an error report."""
        spec = CompareSpec(model_type="glmboost", family="Nonexistent")
        report = compare_to_r(spec)
        assert report.status == "error"
        assert "Python fit failed" in (report.error or "")

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            compare_to_r(CompareSpec(), backend="unknown")
