"""The unseen_levels= policy across learner kinds and stages (issue #123).

Each cell either predicts under the requested policy or refuses naming it.
"average" means the mean of the term's contributions over the fitted levels
for every term that reads the column, so a row with a level the fit never
saw predicts the mean of the predictions at the fitted levels. It used to
equal "zero_effect" for a factor by=, a %X% factor margin and a btree, and
gam_refit refused "average" and failed outright on a factor by=.
"""

import os
import tempfile

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

LEVELS = "abcde"
FORMULAS = [
    "y ~ bols(g) + bbs(x)",
    "y ~ brandom(g) + bbs(x)",
    "y ~ brandom(g, by=z) + bbs(x)",
    "y ~ bols(z, by=g) + bbs(x)",
    "y ~ bbs(x) %X% brandom(g)",
    "y ~ btree(g, x, maxdepth=3) + bbs(x)",
]
REFIT = FORMULAS[:5]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    g = rng.choice(list(LEVELS), n)
    x, z = rng.uniform(-1, 1, n), rng.normal(size=n)
    level = dict(a=0.0, b=1.0, c=-1.0, d=0.5, e=2.0)
    slope = dict(a=0.0, b=1.0, c=-0.5, d=0.3, e=0.8)
    y = (np.sin(2 * x) + np.array([level[v] for v in g])
         + np.array([slope[v] for v in g]) * z + rng.normal(0, 0.3, n))
    return pl.DataFrame({"g": g, "x": x, "z": z, "y": y})


def _row(level):
    return pl.DataFrame({"g": [level], "x": [0.2], "z": [1.3]})


def _check(predict):
    """'average' at an unseen level equals the mean over the fitted levels;
    'error' refuses naming the level; 'zero_effect' predicts."""
    per_level = [float(np.ravel(predict(_row(level), "error"))[0]) for level in LEVELS]
    average = float(np.ravel(predict(_row("zz"), "average"))[0])
    assert average == pytest.approx(np.mean(per_level), abs=1e-9)
    assert np.isfinite(np.ravel(predict(_row("zz"), "zero_effect"))[0])
    with pytest.raises(ValueError, match="zz"):
        predict(_row("zz"), "error")


@pytest.mark.parametrize("formula", FORMULAS)
def test_predict(formula, data):
    fit = mb.gamboost(formula, data=data, control=mb.boost_control(150))
    _check(lambda nd, policy: fit.predict(newdata=nd, unseen_levels=policy))


@pytest.mark.parametrize("formula", FORMULAS)
def test_saved_model_keeps_the_policy(formula, data):
    fit = mb.gamboost(formula, data=data, control=mb.boost_control(150))
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.mbr")
        mb.save_model(fit, path)
        loaded = mb.load_model(path)
    _check(lambda nd, policy: loaded.predict(newdata=nd, unseen_levels=policy))
    np.testing.assert_allclose(loaded.predict(newdata=_row("zz"), unseen_levels="average"),
                               fit.predict(newdata=_row("zz"), unseen_levels="average"))


@pytest.mark.parametrize("formula", REFIT)
def test_refit(formula, data):
    fit = mb.gamboost(formula, data=data, control=mb.boost_control(300))
    refit = mb.gam_refit(fit, unsupported="hold")
    _check(lambda nd, policy: refit.predict(nd, unseen_levels=policy))


def test_refit_holds_a_tree_under_the_policy(data):
    fit = mb.gamboost("y ~ btree(g, maxdepth=3) + bbs(x) + bols(z)", data=data,
                      control=mb.boost_control(150))
    refit = mb.gam_refit(fit, unsupported="hold")
    assert refit.held
    _check(lambda nd, policy: refit.predict(nd, unseen_levels=policy))


@pytest.mark.parametrize("formula", ["y ~ brandom(g) + bbs(x)", "y ~ bols(z, by=g) + bbs(x)",
                                     "y ~ btree(g, x, maxdepth=3) + bbs(x)"])
def test_lss(formula, data):
    fit = mb.gamboostLSS({"mu": formula, "sigma": "y ~ bols(x)"}, data=data,
                         families=mb.GaussianLSS(), control=mb.boost_control(100))
    _check(lambda nd, policy: fit.predict(newdata=nd, parameter="mu",
                                          unseen_levels=policy))


@pytest.mark.r_parity
def test_r_gamboostlss_trees_split_factor_levels_as_python_does(data):
    """A btree over a factor gives the same predictions in R and Python when
    both use the same split criterion.

    The two frontends default to different criteria. R's btree defaults to
    selection="ctree" (mboost's conditional-inference split, with factor
    levels coded as a set). Python's Btree defaults to selection="sse" (CART),
    a deliberate default for predictive accuracy (see Btree in
    python/mboostrs/learners.py). Comparing the two defaults with each other
    tests the criteria, not the factor handling: with the defaults the
    predictions differ by up to 0.53 over the 400 rows. Both calls therefore
    name selection="ctree" explicitly; with that, the predictions agree
    exactly (max abs difference 0.0 measured).
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    frame = data.to_pandas()
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(frame)
    upstream = np.asarray(ro.r(
        "d$g <- factor(d$g); m <- mboostrs::gamboostLSS(list(mu = y ~ btree(g, x, maxdepth = 3, "
        "selection = 'ctree') + bbs(x), sigma = y ~ bols(x)), data = d, "
        "families = mboostrs::GaussianLSS(), "
        "control = mboostrs::boost_control(mstop = 60)); predict(m, parameter = 'mu')"))
    fit = mb.gamboostLSS({"mu": "y ~ btree(g, x, maxdepth=3, selection=ctree) + bbs(x)",
                          "sigma": "y ~ bols(x)"},
                         data=frame, families=mb.GaussianLSS(), control=mb.boost_control(60))
    np.testing.assert_allclose(np.asarray(fit.predict(parameter="mu")), upstream, atol=1e-12)
