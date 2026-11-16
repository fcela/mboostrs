"""Two learners on one column stay apart in coef, partial_effects, varimp and
confint.

`bols(x) + bbs(x, center=True, df=1)` is the standard split of a term into a
linear part and a deviation from it, and both learners are stored under the
column name "x". `coef()` was keyed by that name, so it kept one learner and
dropped the other's coefficients; `partial_effects()` and `confint()` resolved
"x" to the first learner only, so the effects no longer added up to the fit;
`varimp()` merged the two. They are now keyed by `base_learner_names()`:
"x (bols)" and "x (bbs)" for a shared name, the raw name otherwise.
"""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n = 200
    d = {"x": rng.uniform(-2, 2, n), "z": rng.normal(size=n)}
    d["y"] = np.sin(2 * d["x"]) + 2 * d["x"] + 0.5 * d["z"] + rng.normal(0, 0.3, n)
    fit = mb.gamboost("y ~ bols(x) + bbs(x, center=True, df=1) + bols(z)", data=d,
                      control=mb.boost_control(300))
    assert set(fit._selected) == {0, 1, 2}, "fixture must select both x learners"
    return fit, d


def test_coef_keeps_both_learners(model):
    fit, _ = model
    coefficients = fit.coef()
    assert set(coefficients) == {"x (bols)", "x (bbs)", "z", "(offset)"}
    assert np.size(coefficients["x (bbs)"]) > 1
    steps = fit.coef(aggregate="none")
    assert len(steps["x (bols)"]) + len(steps["x (bbs)"]) + len(steps["z"]) == 300
    assert set(fit.coef(which="x")) == {"x (bols)", "x (bbs)", "(offset)"}


def test_partial_effects_add_up_to_the_fit(model):
    fit, data = model
    effects = fit.partial_effects(newdata=data)
    assert set(effects) == {"x (bols)", "x (bbs)", "z"}
    total = sum(np.asarray(effect) for _, effect in effects.values())
    offset = float(np.asarray(fit.offset_).ravel()[0])
    np.testing.assert_allclose(offset + total, fit.fitted_, atol=1e-10)
    assert set(fit.partial_effects(which="x")) == {"x (bols)", "x (bbs)"}
    assert set(fit.partial_effects(which="x (bbs)")) == {"x (bbs)"}
    assert set(fit.partial_effects(which=1)) == {"x (bbs)"}


def test_varimp_has_one_entry_per_learner(model):
    fit, _ = model
    importance = fit.varimp()
    assert set(importance["reduction"]) == {"x (bols)", "x (bbs)", "z"}
    assert importance["variable_names"]["x (bbs)"] == "x"
    assert sum(importance["selfreq"].values()) == pytest.approx(1.0)


def test_confint_resolves_a_shared_name_to_both_learners(model):
    fit, _ = model
    both = fit.confint(parm="x", B=4, seed=1)
    assert set(both["lower"]) == {"x (bols)", "x (bbs)"}
    one = fit.confint(parm="x (bbs)", B=4, seed=1)
    assert set(one["lower"]) == {"x (bbs)"}
    np.testing.assert_allclose(one["lower"]["x (bbs)"], both["lower"]["x (bbs)"])
    with pytest.raises(ValueError, match="unknown base-learner"):
        fit.confint(parm="w", B=2)


def test_unshared_names_keep_their_keys():
    rng = np.random.default_rng(1)
    d = {"x": rng.normal(size=100), "z": rng.normal(size=100)}
    d["y"] = d["x"] + rng.normal(size=100)
    fit = mb.gamboost("y ~ bbs(x) + bols(z)", data=d, control=mb.boost_control(50))
    assert set(fit.coef()) <= {"x", "z", "(offset)"}
