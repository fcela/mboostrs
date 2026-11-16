"""`bols(f, ...)` for a categorical `f` in gamboost (issue #159).

A plain `bols(f)` was already fitted (as the equivalent level design). With
`intercept=False`, `df=` or `lambda=` it was refused. mboost builds such a term
from the treatment-contrast design `[1, d_2, ..., d_K]` (`[d_2, ..., d_K]`
without the intercept) with the ridge penalty `lambda * I`; gamboost now
builds the same design, so the fitted path matches mboost.
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

FORMULAS = [
    "y ~ bols(g, intercept=False) + bbs(x)",
    "y ~ bols(g, intercept=False) + bbs(x) + bols(w)",
    "y ~ bols(g, lambda=10) + bbs(x) + bols(w)",
    "y ~ bols(g, df=2) + bbs(x)",
    "y ~ bols(g, intercept=False, lambda=5) + bols(w)",
]


def _data(n=240, seed=4):
    rng = np.random.default_rng(seed)
    g = np.array(list("abcde"))[rng.integers(0, 5, n)]
    x = rng.uniform(-2, 2, n)
    w = rng.normal(size=n)
    y = (g == "b") * 1.0 - (g == "e") * 0.6 + np.sin(x) + 0.3 * w + rng.normal(0, 0.3, n)
    return {"y": y, "g": g, "x": x, "w": w}


@pytest.mark.parametrize("formula", FORMULAS)
def test_fits_and_replays_on_new_data(formula):
    data = _data()
    fit = mb.gamboost(formula, data=data, control=mb.boost_control(80, 0.1))
    np.testing.assert_allclose(fit.predict(newdata=data), fit.fitted_, atol=1e-10)


def test_without_intercept_the_first_level_is_the_zero_row():
    data = _data()
    fit = mb.gamboost("y ~ bols(g, intercept=False)", data=data,
                      control=mb.boost_control(200, 0.1))
    new = {"g": np.array(["a", "b"])}
    contribution = fit.predict(newdata=new) - fit.offset_
    assert contribution[0] == pytest.approx(0.0, abs=1e-12)
    assert contribution[1] > 0.5


def test_unseen_level_is_refused():
    data = _data()
    fit = mb.gamboost("y ~ bols(g, lambda=1) + bbs(x)", data=data,
                      control=mb.boost_control(30, 0.1))
    with pytest.raises(ValueError, match="not present"):
        fit.predict(newdata={"g": np.array(["z"]), "x": np.zeros(1), "w": np.zeros(1)})


def test_save_and_load_round_trip(tmp_path):
    data = _data()
    fit = mb.gamboost("y ~ bols(g, intercept=False) + bbs(x) + bols(w)", data=data,
                      control=mb.boost_control(60, 0.1))
    path = tmp_path / "model.json"
    mb.save_model(fit, path)
    loaded = mb.load_model(path)
    np.testing.assert_allclose(loaded.predict(newdata=data), fit.fitted_, atol=1e-10)


def test_glmboost_still_refuses():
    with pytest.raises(ValueError, match="categorical column"):
        mb.glmboost("y ~ bols(g, intercept=False) + x", data=_data())


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("formula", FORMULAS)
def test_matches_mboost(formula):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    data = _data()
    with localconverter(ro.default_converter + numpy2ri.converter):
        for key, value in data.items():
            ro.globalenv[key] = value
    r_formula = formula.replace("False", "FALSE")
    upstream = ro.r(
        "suppressMessages(library(mboost)); "
        "d <- data.frame(x = x, w = w, y = y, g = factor(g)); "
        f"m <- mboost::gamboost({r_formula}, data = d, "
        "control = mboost::boost_control(mstop = 100)); "
        "list(as.numeric(fitted(m)), as.numeric(m$risk()))")
    fit = mb.gamboost(formula, data=data, control=mb.boost_control(100))
    np.testing.assert_allclose(np.asarray(fit.fitted_), np.asarray(upstream[0]), atol=1e-9)
    np.testing.assert_allclose(np.asarray(fit.risk_), np.asarray(upstream[1]), rtol=1e-9)
