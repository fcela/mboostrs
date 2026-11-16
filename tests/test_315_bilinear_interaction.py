"""Regression tests for issue #315: the interaction screen misses a published
bilinear interaction because it only ever proposed the tensor and, for
non-Gaussian fits, screened raw response-scale residuals.

Two fixes are exercised here:

1. every continuous x continuous pair now carries three candidates — the
   tensor plus both 1-df varying-coefficient directions — from `pair_terms`
   and `pair_terms_for`, so `suggest_specifications` can offer the cheap
   check; and
2. `pearson_residual` puts a non-Gaussian fit's rows on a common variance
   scale, which is what the autospec screen now feeds `screen_interactions`.
"""
import numpy as np
import pytest

import mboostrs as mb
from mboostrs.explore import TermReport, pair_term_for, pair_terms_for, suggest_specifications
from mboostrs.terms import (
    ScreenedPair,
    pair_term,
    pair_terms,
    pearson_residual,
    screen_interactions,
)


# ─── pair_terms (plural) ─────────────────────────────────────────────────


def test_pair_terms_offers_tensor_and_both_varying_coefficients():
    levels = {"a": 10, "b": 10}
    out = pair_terms("a", "b", levels)
    assert out == [
        pair_term("a", "b", levels),
        "bols(a, by=b)",
        "bols(b, by=a)",
    ]
    assert out[0] == "bbs(a, df=2.5, knots=5) %X% bbs(b, df=2.5, knots=5)"


@pytest.mark.parametrize("levels,kind", [
    ({"a": 10, "b": 2}, "tensor"),   # continuous x binary
    ({"a": 2, "b": 2}, "tensor"),    # binary x binary
    ({"a": 10, "b": 10}, "tree"),    # tree learner
])
def test_pair_terms_delegates_for_every_other_case(levels, kind):
    out = pair_terms("a", "b", levels, kind=kind)
    assert out == [pair_term("a", "b", levels, kind=kind)]


# ─── pair_terms_for (plural) ─────────────────────────────────────────────


def test_pair_terms_for_offers_tensor_and_both_varying_coefficients():
    levels = {"a": 10, "b": 10}
    out = pair_terms_for("a", "b", levels=levels)
    assert out == [
        pair_term_for("a", "b", levels=levels),
        "bols(a, by=b)",
        "bols(b, by=a)",
    ]


@pytest.mark.parametrize("left,right,levels,groups", [
    ("a", "b", {"a": 10, "b": 2}, ()),          # continuous x binary
    ("a", "b", {"a": 2, "b": 2}, ()),           # binary x binary
    ("a", "b", {"a": 10, "b": 10}, {"b"}),      # continuous x group
])
def test_pair_terms_for_delegates_for_every_other_case(left, right, levels, groups):
    out = pair_terms_for(left, right, levels=levels, groups=groups)
    assert out == [pair_term_for(left, right, levels=levels, groups=groups)]


def test_pair_terms_for_keeps_the_transform_scale_in_every_candidate():
    out = pair_terms_for("a", "b", levels={"a": 10, "b": 10},
                         transforms={"a": "log"})
    assert out[0] == ("bbs(log(a), df=2.5, knots=5)"
                      " %X% bbs(b, df=2.5, knots=5)")
    assert out[1] == "bols(log(a), by=b)"
    assert out[2] == "bols(b, by=log(a))"


# ─── suggest_specifications ──────────────────────────────────────────────


def _report(levels):
    return TermReport(formula="y ~ bols(a) + bols(b)", response="y",
                      baseline=1.0, levels=levels)


def test_suggestions_offer_three_forms_for_a_continuous_pair():
    pair = ScreenedPair(left="a", right="b", removed=0.5)
    adds = [s for s in suggest_specifications(_report({"a": 10, "b": 10}),
                                              screened=[pair])
            if s.action == "add"]
    assert len(adds) == 3
    specs = [s.specification for s in adds]
    assert specs[0].endswith("bbs(a, df=2.5, knots=5) %X% bbs(b, df=2.5, knots=5)")
    assert specs[1].endswith("bols(a, by=b)")
    assert specs[2].endswith("bols(b, by=a)")
    assert "tensor-product spline surface" in adds[0].reason
    assert "varying coefficient" in adds[1].reason and "a varies" in adds[1].reason
    assert "varying coefficient" in adds[2].reason and "b varies" in adds[2].reason
    assert all(s.term == "a x b" and s.basis == "detected" for s in adds)


def test_a_binary_pair_still_yields_exactly_one_suggestion():
    pair = ScreenedPair(left="a", right="b", removed=0.5)
    adds = [s for s in suggest_specifications(_report({"a": 2, "b": 2}),
                                              screened=[pair])
            if s.action == "add"]
    assert len(adds) == 1
    assert adds[0].specification.endswith("bols(a, by=b)")


# ─── pearson_residual ────────────────────────────────────────────────────


def test_pearson_residual_matches_the_binomial_formula():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-(0.5 * x - 0.3)))
    y = (rng.uniform(size=n) < p_true).astype(float)
    data = {"x": x, "y": y}
    fit = mb.gamboost("y ~ bols(x)", data=data, family=mb.Binomial(),
                      control=mb.boost_control(60, 0.1))
    p = np.clip(np.asarray(fit.predict(type="response"),
                           dtype=float).ravel(), 1e-6, 1 - 1e-6)
    expected = (y - p) / np.sqrt(p * (1 - p))
    assert np.allclose(pearson_residual(fit, data, "y"), expected)


def test_pearson_residual_is_the_ordinary_residual_for_gaussian():
    rng = np.random.default_rng(4)
    n = 300
    data = {"x": rng.normal(size=n)}
    data["y"] = 1.0 + 2.0 * data["x"] + rng.normal(size=n)
    fit = mb.gamboost("y ~ bols(x)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    expected = data["y"] - np.asarray(fit.predict(), dtype=float).ravel()
    assert np.allclose(pearson_residual(fit, data, "y"), expected)


def test_pearson_residual_matches_the_poisson_formula():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    data = {"x": x}
    data["y"] = rng.poisson(np.exp(0.4 + 0.5 * x)).astype(float)
    fit = mb.gamboost("y ~ bols(x)", data=data, family=mb.Poisson(),
                      control=mb.boost_control(60, 0.1))
    mu = np.clip(np.asarray(fit.predict(type="response"),
                            dtype=float).ravel(), 1e-6, None)
    expected = (data["y"] - mu) / np.sqrt(mu)
    assert np.allclose(pearson_residual(fit, data, "y"), expected)


def test_pearson_residual_warns_and_falls_back_for_another_family():
    rng = np.random.default_rng(6)
    n = 300
    data = {"x": rng.normal(size=n)}
    data["y"] = np.exp(0.3 + 0.4 * data["x"] + 0.2 * rng.normal(size=n))
    fit = mb.gamboost("y ~ bols(x)", data=data, family=mb.GammaReg(),
                      control=mb.boost_control(50, 0.1))
    with pytest.warns(UserWarning, match="Pearson residuals are not defined"):
        residual = pearson_residual(fit, data, "y")
    expected = data["y"] - np.asarray(fit.predict(type="response"),
                                      dtype=float).ravel()
    assert np.allclose(residual, expected)


# ─── the screen ranks the true pair better on Pearson residuals ──────────


def _bilinear_binomial():
    """Binomial data with a known, modest bilinear interaction and a noise
    column that shadows the variance pattern of the raw residual."""
    rng = np.random.default_rng(2)
    n = 2000
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    data = {"x1": x1, "x2": x2}
    for i in range(5):
        data[f"z{i}"] = rng.uniform(-1, 1, n)
    # zp tracks x1, so cells in the high-variance region of a raw Binomial
    # residual can look structured without carrying any signal.
    data["zp"] = x1 + 0.05 * rng.normal(size=n)
    logit = -0.2 + 4.0 * x1 + 4.0 * x2 + 0.5 * x1 * x2
    p = 1 / (1 + np.exp(-logit))
    data["y"] = (rng.uniform(size=n) < p).astype(float)
    return data


def _rank_of_true_pair(data, residual):
    keys = [name for name in data if name != "y"]
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    screened = screen_interactions(data, residual, response="y", pairs=pairs,
                                   seed=0)
    order = [frozenset((row.left, row.right)) for row in screened]
    return order.index(frozenset(("x1", "x2"))), screened


def test_pearson_residuals_rank_the_true_pair_higher_than_raw():
    data = _bilinear_binomial()
    fit = mb.gamboost("y ~ bbs(x1, df=3) + bbs(x2, df=3)", data=data,
                      family=mb.Binomial(), control=mb.boost_control(200, 0.1))

    raw = data["y"] - np.asarray(fit.predict(type="response"),
                                 dtype=float).ravel()
    raw_rank, raw_screened = _rank_of_true_pair(data, raw)
    raw_true = next(row.removed for row in raw_screened
                    if {row.left, row.right} == {"x1", "x2"})

    pearson_rank, pearson_screened = _rank_of_true_pair(
        data, pearson_residual(fit, data, "y"))
    pearson_true = next(row.removed for row in pearson_screened
                        if {row.left, row.right} == {"x1", "x2"})

    # On raw residuals the true pair is negative — the screen finds nothing.
    assert raw_true < 0
    # On Pearson residuals it is positive and ranks strictly higher.
    assert pearson_true > 0
    assert pearson_rank < raw_rank
