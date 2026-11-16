"""Optional tree refits preserve predictions and never imply tree uncertainty."""

import importlib

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

MODULE = importlib.import_module("mboostrs.discover")
TREE = "btree(x1, maxdepth=2)"


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(671)
    x1, x2 = rng.uniform(-1, 1, (2, 180))
    return pl.DataFrame({"x1": x1, "x2": x2, "y": 5 * (x1 > 0) + 2 * x2 + rng.normal(0, .1, 180)})


def analysis(frame, terms):
    value = MODULE._Analysis.__new__(MODULE._Analysis)
    value.frame, value.response, value.family = frame, "y", mb.Gaussian()
    value.mean_terms, value.draft_terms = list(terms), list(terms)
    value.na_action, value.mstop = "omit", 100
    value.spread_term, value.refit, value.dropped_refit = None, None, None
    value.dropped_variables, value.log = set(), []
    value._admit = lambda *a, **kw: True
    value.model = mb.gamboost("y ~ " + " + ".join(terms), frame, control=mb.boost_control(100))
    return value


def test_mixed_refit_holds_tree_and_preserves_boosted_predictions(frame):
    value = analysis(frame, [TREE, "bols(x2)"])
    before = value.model.predict(frame)
    value._refit()
    assert value.refit is not None and value.refit.held == (TREE,)
    assert value.refit.terms == ("bols(x2)",)
    assert "conditional on the held" in value.log[-1]["reason"]
    np.testing.assert_array_equal(value.model.predict(frame), before)
    # The public held-contribution refit keeps the actual tree effect in its predictions.
    direct = mb.gam_refit(value.model, unsupported="hold")
    np.testing.assert_array_equal(value.refit.predict(frame), direct.predict(frame))


def test_all_tree_refit_is_explicitly_unavailable(frame):
    value = analysis(frame, [TREE])
    before = value.model.predict(frame)
    value._refit()
    assert value.refit is None
    assert value.log[-1]["reason"].startswith("not run: every selected term is held")
    np.testing.assert_array_equal(value.model.predict(frame), before)


def test_dropped_term_refit_is_conditioned_on_tree(frame):
    value = analysis(frame, [TREE])
    value.draft_terms = [TREE, "bols(x2)"]
    value._refit_draft()
    assert value.dropped_refit.held == (TREE,)
    assert value.dropped_variables == {"x2"}
    assert "conditional on the held" in value.log[-1]["reason"]


@pytest.mark.parametrize("variable", ["x1", "x2"])
def test_discovery_contrast_does_not_treat_held_tree_as_zero_or_certain(frame, variable):
    value = analysis(frame, [TREE, "bols(x2)"])
    value._refit()
    result = MODULE.DiscoveryResult.__new__(MODULE.DiscoveryResult)
    result.model, result.refit, result._frame = value.model, value.refit, frame
    result._variables, result.level, result.spread_formula = {"x1", "x2"}, .9, None
    contrast = result.contrast(variable, -.5, .5)
    direct = mb.prediction_contrast(value.model, frame, variable, -.5, .5, type="response")
    assert contrast.estimate == direct.estimate
    assert contrast.lower is contrast.upper is None
    assert "does not estimate their uncertainty" in contrast.note
    if variable == "x1":
        assert contrast.estimate > 1


def test_distributional_tree_refit_remains_explicitly_unavailable(frame):
    value = analysis(frame, [TREE])
    value.spread_term = "bols(x2)"
    value.model = mb.gamboostLSS({"mu": "y ~ " + TREE, "sigma": "y ~ bols(x2)"}, frame, mstop=100)
    before = value.model.predict_parameters(frame, type="response")
    value._refit()
    assert value.refit is None
    assert value.log[-1]["reason"].startswith("not run: a joint distributional REML")
    after = value.model.predict_parameters(frame, type="response")
    for parameter in before:
        np.testing.assert_array_equal(after[parameter], before[parameter])
