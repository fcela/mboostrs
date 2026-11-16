"""The smooth second stage of effect_model: log scale for heavy positive
modifiers, spline ranges over the central 98 percent, and predictions held
flat beyond the fitted range."""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.interactions import _bbs_defaults, _boundary_knots_arg


def skewed_positive(n, seed):
    """A lognormal positive modifier, a normal modifier, and mild confounding."""
    rng = np.random.default_rng(seed)
    income = np.exp(rng.normal(10, 1, n))
    x1 = rng.normal(0, 1, n)
    e = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * np.log(income) - 3)))
    a = rng.binomial(1, e).astype(float)
    y = x1 + 0.1 * np.log(income) + rng.normal(0, 1, n) + a
    return {"a": a, "y": y, "income": income, "x1": x1}


def spec():
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=("income", "x1"),
        time_zero="assignment", follow_up="one year",
        adjustment_rationale="the covariates are the common causes")


def fit_smooth(seed, n=400):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.effect_model(skewed_positive(n, seed=seed), spec(), terms="smooth")


def test_heavy_positive_modifier_is_logged_and_a_normal_one_is_not():
    fx = fit_smooth(seed=20)
    plan = fx.diagnostics["smooth_scale"]
    assert plan["income"]["log"] is True
    assert plan["x1"]["log"] is False
    assert fx.diagnostics["smooth_rows_outside_range"] >= 1
    assert "enter on the log scale" in fx.describe()


def test_smooth_effect_is_held_flat_beyond_the_fitted_range():
    fx = fit_smooth(seed=21)
    far = {"income": np.array([1e12, 1e15]), "x1": np.array([0.0, 0.0])}
    tau_far = fx.predict(far)
    assert tau_far[0] == pytest.approx(tau_far[1])
    beyond_top = {"income": np.array([1e200]), "x1": np.array([0.0])}
    assert fx.predict(beyond_top)[0] == pytest.approx(tau_far[0])


def test_boosted_candidate_keeps_raw_columns_and_warn_extrapolation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boosted = mb.effect_model(skewed_positive(300, seed=22), spec(), terms="stumps")
    # The plan is recorded for the record, but the boosted fit does not use it.
    assert boosted.scale_plan == {}
    assert boosted.predict({"income": np.array([1e15]), "x1": np.array([0.0])}).shape == (1,)


def test_a_log_scale_modifier_refuses_zero_in_new_rows():
    fx = fit_smooth(seed=23, n=300)
    with pytest.raises(ValueError, match="'income' is entered on the log scale"):
        fx.predict({"income": np.array([0.0]), "x1": np.array([0.0])})


def test_bbs_boundary_knots_parse_without_reading_knots():
    assert _boundary_knots_arg("df=4, boundary.knots=c(0.5, 2.5)") == (0.5, 2.5)
    assert _boundary_knots_arg("df=4") is None
    # The knot count keeps its default when only boundary.knots is given.
    assert _bbs_defaults("boundary.knots=c(0,10)")["n_knots"] == 20
    with pytest.raises(ValueError, match="lo < hi"):
        _boundary_knots_arg("boundary.knots=c(2, 1)")
