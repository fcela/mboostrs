"""Intercept removal (`- 1`, `+ 0`) is refused, with a ValueError, by every fitting
function except glmboost (issue #101, item 1).

Each refusal names glmboost as the only function that removes the intercept and
states what the refusing function does with the intercept instead. Before this
change, `gam` and `gamlss` silently fitted with an intercept for `+ 0`.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture
def data():
    rng = np.random.default_rng(101)
    n = 60
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 3.0 + 2.0 * x + rng.normal(size=n)
    return {"y": y, "x": x, "z": z}


def _gamboost(formula, data):
    return mb.gamboost(formula, data=data)


def _gam(formula, data):
    return mb.gam(formula, data=data)


def _gamlss(formula, data):
    return mb.gamlss({"mu": formula, "sigma": "y ~ bbs(z)"}, data=data)


def _gamboost_lss(formula, data):
    return mb.gamboostLSS({"mu": formula, "sigma": "y ~ bbs(z)"}, data=data)


def _glmboost_lss(formula, data):
    return mb.glmboostLSS({"mu": formula, "sigma": "y ~ z"}, data=data)


ENTRY_POINTS = {
    "gam": _gam,
    "gamlss": _gamlss,
    "gamboostLSS": _gamboost_lss,
    "glmboostLSS": _glmboost_lss,
}


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
@pytest.mark.parametrize("formula", ["y ~ bbs(x) - 1", "y ~ bbs(x) + 0"])
def test_smooth_formula_with_intercept_removal_is_refused(name, formula, data):
    with pytest.raises(ValueError, match="glmboost") as info:
        ENTRY_POINTS[name](formula, data)
    assert "intercept control" in str(info.value)
    assert isinstance(info.value, NotImplementedError)


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
@pytest.mark.parametrize("formula", ["y ~ x - 1", "y ~ x + 0", "y ~ 0 + x"])
def test_linear_formula_with_intercept_removal_is_refused(name, formula, data):
    # The entry points that route through the glmboost-style parser refuse a
    # bare `x` term too; gam and gamlss read it as a smooth or a factor.
    with pytest.raises(ValueError, match="glmboost"):
        ENTRY_POINTS[name](formula, data)


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_refusal_names_the_function_and_its_intercept_behaviour(name, data):
    with pytest.raises(ValueError) as info:
        ENTRY_POINTS[name]("y ~ bbs(x) + 0", data)
    message = str(info.value)
    assert f"{name} " in message
    assert "supported by glmboost only" in message
    if name in ("gam", "gamlss"):
        assert "To fit without an intercept, use glmboost()" in message


def test_plus_one_says_to_delete_the_term_not_to_use_glmboost(data):
    from mboostrs.formula import expand_formula

    with pytest.raises(ValueError, match="requests an intercept") as info:
        expand_formula("y ~ x + 1", data)
    assert "glmboost" not in str(info.value)


def test_gam_plus_zero_is_no_longer_silently_ignored(data):
    # Before #101 this fitted gam with an intercept and returned no error.
    with pytest.raises(ValueError, match="gam always fits an intercept"):
        mb.gam("y ~ bbs(x) + 0", data=data)


def test_gamlss_plus_zero_is_no_longer_silently_ignored(data):
    with pytest.raises(ValueError, match="gamlss fits an intercept"):
        mb.gamlss({"mu": "y ~ bbs(x) + 0", "sigma": "y ~ bbs(z)"}, data=data)


def test_a_column_named_zero_is_not_an_intercept_term(data):
    # A column called "0" is data, so `+ 0` names that column and the shared
    # check does not refuse it.
    from mboostrs.formula import refuse_intercept_control

    frame = dict(data)
    frame["0"] = data["x"]
    refuse_intercept_control("y ~ 0 + x", frame, "gam")
    with pytest.raises(ValueError):
        refuse_intercept_control("y ~ 0 + x", data, "gam")


def test_glmboost_still_accepts_minus_one_and_plus_zero(data):
    for formula in ("y ~ x - 1", "y ~ x + 0"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = mb.glmboost(formula, data=data, control=mb.boost_control(10, 0.1))
        assert "(Intercept)" not in fit.coef()
        assert list(fit.coef()) == ["x"]
