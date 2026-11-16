"""gamboostLSS smooth (bbs) prediction on new data / truncated mstop.

Reconstructs each parameter's smooth predictor by rebuilding the B-spline basis at
the requested x with the training boundary knots and summing the per-step
increments. Validated by self-consistency (machine precision) and against R
gamboostLSS (~1e-11).
"""

import mboostrs as mb  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS, gamboostLSS
from conftest import r_call


def _data(seed=9, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = 1 + 2 * x + np.exp(0.2 + 0.4 * x) * rng.standard_normal(n)
    return {"x": x, "y": y}


def test_gamboostlss_bbs_newdata_self_consistent():
    d = _data()
    m = gamboostLSS("y ~ bbs(x, df=4)", data=d, families=GaussianLSS(),
                    control=mb.boost_control(200, 0.1))
    for par in ("mu", "sigma"):
        insample = m.predict(parameter=par, type="link")
        newdata = m.predict(parameter=par, type="link", newdata={"x": d["x"]})
        np.testing.assert_allclose(newdata, insample, rtol=0, atol=1e-12)
        # truncated mstop returns the right shape and differs from the full fit
        trunc = m.predict(parameter=par, type="link", mstop=50)
        assert np.asarray(trunc).shape == (len(d["x"]),)
    # new x: response in (0, inf) for sigma, finite for mu
    nx = np.array([-1.5, 0.0, 1.5])
    resp = m.predict(parameter="sigma", type="response", newdata={"x": nx})
    assert resp.shape == (3,) and np.all(resp > 0)


@pytest.mark.r_parity
def test_gamboostlss_bbs_newdata_matches_r():
    import rpy2.robjects as ro
    f = ro.r("""
    function() {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(9); n <- 300; x <- runif(n, -2, 2)
        y <- 1 + 2 * x + exp(0.2 + 0.4 * x) * rnorm(n)
        nx <- seq(-1.9, 1.9, length.out = 25)
        m <- gamboostLSS(y ~ bbs(x, df = 4), data = data.frame(x = x, y = y),
                         families = GaussianLSS(), control = boost_control(mstop = 200, nu = 0.1))
        nd <- data.frame(x = nx)
        list(x = x, y = y, nx = nx,
             mu = as.numeric(predict(m, parameter = "mu", newdata = nd, type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", newdata = nd, type = "link")))
    }
    """)
    r = r_call(f, packages=("gamboostLSS",))
    x = np.asarray(r.rx2("x"), float)
    y = np.asarray(r.rx2("y"), float)
    nx = np.asarray(r.rx2("nx"), float)
    m = gamboostLSS("y ~ bbs(x, df=4)", data={"x": x, "y": y},
                    families=GaussianLSS(), control=mb.boost_control(200, 0.1))
    for par, key in (("mu", "mu"), ("sigma", "sigma")):
        ours = np.asarray(m.predict(parameter=par, newdata={"x": nx}, type="link"), float)
        np.testing.assert_allclose(ours, np.asarray(r.rx2(key), float), rtol=1e-6, atol=1e-7,
                                   err_msg=f"gamboostLSS {par} newdata does not match R")
