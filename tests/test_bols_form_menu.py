"""A centered bols needs every other learner to keep the residual mean at zero.

mboost's bols(x) has its own intercept column; mboostrs fits an unpenalized
bols as one centered column with the constant in the fixed offset. That is
equivalent only while every other learner's span contains the constant. A
`by = z` learner or a bols without intercept does not, the residual mean
drifts, and the centered column could not follow it: `bbs(x, by=z) +
bols(w)` sat 0.015 from mboost, `bols(x, intercept=False) + bols(w)` 0.013.
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

FORMULAS = [
    "y ~ bbs(x, by=z) + bols(w)",
    "y ~ bols(x, intercept=False) + bols(w)",
    "y ~ bols(x, by=z) + bols(w)",
    "y ~ bbs(x, by=z) + bbs(w) + bols(z)",
]


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("formula", FORMULAS)
def test_menu_matches_mboost(formula):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(6)
    n = 200
    data = {"x": rng.uniform(size=n), "z": rng.normal(size=n), "w": rng.uniform(size=n)}
    data["y"] = np.sin(3 * data["x"]) * data["z"] + data["w"] + rng.normal(0, 0.3, n)
    with localconverter(ro.default_converter + numpy2ri.converter):
        for key, value in data.items():
            ro.globalenv[key] = value
    r_formula = formula.replace("False", "FALSE")
    upstream = np.asarray(ro.r(
        "suppressMessages(library(mboost)); "
        "d <- data.frame(x = x, z = z, w = w, y = y); "
        f"as.numeric(fitted(mboost::gamboost({r_formula}, data = d, "
        "control = mboost::boost_control(mstop = 100))))"))
    fit = mb.gamboost(formula, data=data, control=mb.boost_control(100))
    np.testing.assert_allclose(np.asarray(fit.fitted_), upstream, atol=1e-9)


def test_by_learner_switches_the_menu_to_the_block_form():
    from mboostrs.gamboost import _bols_form_for
    from mboostrs.learners import Bbs, Bols

    assert _bols_form_for([Bbs("x"), Bols("w")]) == "centered"
    assert _bols_form_for([Bbs("x", by="z"), Bols("w")]) == "block"
    assert _bols_form_for([Bols("x", intercept=False), Bols("w")]) == "block"


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_user_design_without_a_constant_beside_bols_matches_mboost():
    """A `Buser` span need not contain the constant; the centered `bols(w)`
    beside `buser(cbind(x, x^2))` sat 0.22 from mboost."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(6)
    n = 200
    x, w = rng.uniform(size=n), rng.uniform(size=n)
    y = 2 * x + w + rng.normal(0, 0.3, n)
    X = np.column_stack([x, x ** 2])
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["X"], ro.globalenv["w"], ro.globalenv["y"] = X, w, y
    upstream = np.asarray(ro.r(
        "suppressMessages(library(mboost)); d <- data.frame(w = w, y = y); "
        "as.numeric(fitted(mboost::gamboost(y ~ buser(X, K = diag(2), lambda = 0) "
        "+ bols(w), data = d, control = mboost::boost_control(mstop = 100))))"))
    fit = mb.gamboost(
        "y ~ bols(w)", data={"y": y, "w": w, "u": np.zeros(n)},
        learners=[mb.Buser("u", X=X, K=np.eye(2), lambda_value=0.0), mb.Bols("w")],
        control=mb.boost_control(100))
    np.testing.assert_allclose(np.asarray(fit.fitted_), upstream, atol=1e-9)


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("family", ["Binomial", "Poisson", "Laplace", "Huber"])
def test_a_non_squared_loss_uses_the_block_form(family):
    """The centered bols relies on the residual mean staying at zero, which
    only a squared-error loss keeps. bbs(x) + bols(z) sat 6.4e-4 from mboost
    for Binomial, 6.8e-4 for Poisson and 0.053 for Laplace."""
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    d["b"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-d.x))).astype(float)
    d["c"] = rng.poisson(np.exp(0.5 + 0.3 * d.x)).astype(float)
    d["y"] = d.x + rng.standard_t(3, n)
    response, upstream_response = {"Binomial": ("b", "factor(b)"),
                                   "Poisson": ("c", "c")}.get(family, ("y", "y"))
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
    upstream = np.asarray(ro.r(
        f"suppressMessages(library(mboost)); as.numeric(fitted(mboost::gamboost("
        f"{upstream_response} ~ bbs(x) + bols(z), data = d, family = mboost::{family}(), "
        "control = mboost::boost_control(mstop = 60))))"))
    fit = mb.gamboost(f"{response} ~ bbs(x) + bols(z)", data=d,
                      family=getattr(mb, family)(), control=mb.boost_control(60))
    np.testing.assert_allclose(fit.fitted_, upstream, atol=1e-10)


def test_the_menu_follows_the_loss():
    from mboostrs.gamboost import _bols_form_for
    from mboostrs.learners import Bbs, Bols

    menu = [Bbs("x"), Bols("w")]
    assert _bols_form_for(menu, family=mb.Gaussian()) == "centered"
    assert _bols_form_for(menu, family=mb.Binomial()) == "block"
    assert _bols_form_for(menu, family=mb.Multinomial(n_classes=3), kronecker=True) == "centered"
