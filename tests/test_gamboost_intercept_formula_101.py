"""`y ~ x - 1` and `y ~ 0 + x` drop the global intercept candidate in glmboost and
gamboost (issue #101, item 1).

glmboost removes its intercept learner. gamboost has no global intercept
candidate (the offset is the family's offset and each learner carries its own
null space), so the term is removed from the formula and the learner list is
that of the formula without it. `+ 1` and `- 0` keep their current behaviour.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.formula import parse_gamboost_formula


@pytest.fixture
def data():
    rng = np.random.default_rng(101)
    n = 80
    x = rng.normal(size=n)
    return {"y": 3.0 + 2.0 * x + rng.normal(size=n), "x": x, "z": rng.normal(size=n)}


@pytest.mark.parametrize("formula", ["y ~ x - 1", "y ~ 0 + x", "y ~ x + 0"])
def test_parse_drops_intercept_terms(formula):
    response, learners = parse_gamboost_formula(formula)
    assert response == "y"
    assert [type(learner).__name__ for learner in learners] == ["Bbs"]


def test_parse_keeps_a_column_named_zero():
    frame = {"y": [1.0], "x": [1.0], "0": [1.0]}
    _, learners = parse_gamboost_formula("y ~ 0 + x", data=frame)
    assert len(learners) == 2


@pytest.mark.parametrize("formula", ["y ~ x - 1", "y ~ 0 + x"])
def test_gamboost_without_intercept_equals_plain_formula(formula, data):
    control = mb.boost_control(30, 0.1)
    plain = mb.gamboost("y ~ x", data=data, control=control)
    fit = mb.gamboost(formula, data=data, control=control)
    assert fit._global_intercept is False
    assert plain._global_intercept is True
    assert list(fit.selected) == list(plain.selected)
    np.testing.assert_allclose(fit.fitted(), plain.fitted(), rtol=1e-12)
    np.testing.assert_allclose(fit.predict(data), plain.predict(data), rtol=1e-12)


def test_gamboost_bols_without_intercept_fits(data):
    fit = mb.gamboost("y ~ bols(x) + bols(z) - 1", data=data,
                      control=mb.boost_control(20, 0.1))
    assert np.all(np.isfinite(fit.fitted()))


@pytest.mark.parametrize("formula", ["y ~ x + 1", "y ~ x - 0"])
def test_redundant_intercept_terms_are_still_refused(formula, data):
    with pytest.raises(ValueError, match="requests an intercept"):
        mb.gamboost(formula, data=data)


def test_glmboost_without_intercept_has_no_intercept_coefficient(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = mb.glmboost("y ~ x - 1", data=data, control=mb.boost_control(10, 0.1))
    assert list(fit.coef()) == ["x"]
    with_intercept = mb.glmboost("y ~ x + 1", data=data, control=mb.boost_control(10, 0.1))
    assert "(Intercept)" in with_intercept.coef()


def test_unsupported_syntax_raises_typed_error(data):
    with pytest.raises(ValueError):
        mb.gamboost("y ~ x * z - 1", data=data)
