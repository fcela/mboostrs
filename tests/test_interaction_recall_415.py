"""Interaction recall and the imputation of transformed predictors (issue #415).

Covers three changes made for the claims-discovery rerun:

* a transform of a predictor with missing values (``log(x)`` with NaN in ``x``)
  is fitted and predicted; before, the transform read the missing value and
  the fit raised ``log(x) produced n non-finite value(s)``;
* ``discover`` screens interactions with the mean of the screen scores over five
  row splits, and drafts each pair's smooths on the scale of the main terms;
* a pair is adopted at the multiple-comparison bar ``2 * sqrt(log(5))``.
"""

import importlib
import math
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs._control import boost_control
from mboostrs.terms import screen_interactions

discover_module = importlib.import_module("mboostrs.discover")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _with_missing(x, rng, share=0.05):
    out = x.copy()
    out[rng.choice(len(x), max(1, int(share * len(x))), replace=False)] = np.nan
    return out


@pytest.fixture(scope="module")
def logged_with_missing():
    """A positive predictor, skewed and with NaN, that enters on log(x)."""
    rng = np.random.default_rng(415)
    n = 400
    x = rng.lognormal(0.0, 1.0, n)
    z = rng.uniform(-1, 1, n)
    y = 1.2 * np.log(x) + np.sin(2 * z) + rng.normal(0, 0.3, n)
    frame = pl.DataFrame({"x": _with_missing(x, rng), "z": z, "y": y})
    return frame


def test_log_transform_of_a_predictor_with_missing_values_fits_and_predicts(logged_with_missing):
    model = _quiet(mb.gamboost, "y ~ bbs(log(x), df=4) + bols(z)", logged_with_missing,
                   control=boost_control(60, 0.1), na_action="impute")
    new = pl.DataFrame({"x": [None, 2.0], "z": [0.0, 0.5]})
    mean = np.asarray(model.predict(new), dtype=float).ravel()
    assert np.all(np.isfinite(mean))


def test_glmboost_log_term_with_missing_values_fits_and_predicts(logged_with_missing):
    model = _quiet(mb.glmboost, "y ~ bols(log(x)) + bols(z)", logged_with_missing,
                   control=boost_control(60, 0.1), na_action="impute")
    new = pl.DataFrame({"x": [None, 2.0], "z": [0.0, 0.5]})
    assert np.all(np.isfinite(np.asarray(model.predict(new), dtype=float).ravel()))


def test_distributional_log_term_with_missing_values_fits(logged_with_missing):
    lss = _quiet(mb.gamboostLSS, {"mu": "y ~ bbs(log(x), df=4)", "sigma": "y ~ bbs(z)"},
                 data=logged_with_missing, families=mb.GaussianLSS(),
                 control=boost_control(40, 0.1), na_action="impute")
    new = pl.DataFrame({"x": [None, 2.0], "z": [0.0, 0.5]})
    mu = lss.predict(newdata=new, parameter="mu", type="response")
    assert np.all(np.isfinite(mu))


def test_discover_on_a_skewed_predictor_with_missing_values_runs_every_stage(logged_with_missing):
    result = _quiet(mb.discover, logged_with_missing, "y", seed=0)
    assert [e for e in result.log if e["reason"].startswith("error")] == []
    assert "bbs(log(x), df=4.0)" in result.formula


def test_screen_scores_are_the_mean_over_five_splits():
    rng = np.random.default_rng(7)
    n = 600
    a = rng.uniform(-2, 2, n)
    b = rng.uniform(-2, 2, n)
    c = rng.normal(size=n)
    residual = np.sin(2 * a) * b + rng.normal(0, 0.5, n)
    frame = pl.DataFrame({"a": a, "b": b, "c": c})
    averaged = discover_module._screen_pairs(frame, residual, response=None, seed=0,
                                             na_action="error")
    singles = {}
    for offset in range(discover_module.SCREEN_SPLITS):
        for pair in screen_interactions(frame, residual, seed=offset, na_action="error"):
            key = frozenset((pair.left, pair.right))
            singles[key] = singles.get(key, 0.0) + pair.removed / discover_module.SCREEN_SPLITS
    assert {frozenset((p.left, p.right)) for p in averaged} == set(singles)
    for pair in averaged:
        assert pair.removed == pytest.approx(singles[frozenset((pair.left, pair.right))])
    scores = [p.removed for p in averaged]
    assert scores == sorted(scores, reverse=True)
    assert frozenset(("a", "b")) == frozenset((averaged[0].left, averaged[0].right))


def test_interaction_bar_is_the_multiple_comparison_bar_for_five_pairs():
    assert discover_module.MAX_INTERACTIONS == 5
    assert discover_module.INTERACTION_SE_RULE == pytest.approx(2.0 * math.sqrt(math.log(5)))
    assert discover_module.INTERACTION_SE_RULE > discover_module.SE_RULE


@pytest.fixture(scope="module")
def logged_interaction():
    """An interaction of z with a skewed positive predictor, which enters on log(x)."""
    rng = np.random.default_rng(3)
    n = 800
    x = rng.lognormal(0.0, 1.0, n)
    z = rng.uniform(-1, 1, n)
    junk = rng.normal(size=n)
    y = 1.5 * z * np.log(x) + 0.5 * junk + rng.normal(0, 0.2, n)
    return pl.DataFrame({"x": x, "z": z, "junk": junk, "y": y})


def test_adopted_interaction_is_drafted_on_the_log_scale_of_its_main_term(logged_interaction):
    result = _quiet(mb.discover, logged_interaction, "y", seed=0, spread=False)
    adopted = [i for i in result.interactions if set(i["variables"]) == {"x", "z"}]
    assert adopted, result.interactions
    assert "log(x)" in adopted[0]["term"]
    assert adopted[0]["change se"] > 0


def test_interaction_log_lines_state_the_bar(logged_interaction):
    result = _quiet(mb.discover, logged_interaction, "y", seed=0, spread=False)
    lines = [e["reason"] for e in result.log if e["stage"] == "interactions"
             and e["candidate"] is not None]
    assert lines, "the interaction stage evaluated no candidate"
    assert all("2.54 change se" in reason for reason in lines)
