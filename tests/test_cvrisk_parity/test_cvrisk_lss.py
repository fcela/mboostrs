"""R-parity test for cvrisk on GAMLSS (glmboostLSS) models.

Verifies cross-validated risk over the 2-D per-parameter mstop grid against R's
``gamboostLSS`` ``cvrisk.mboostLSS``, value-for-value per fold and grid point.
Because cyclic boosting couples the parameters (freezing one changes the others'
later updates), each grid point is a separate per-parameter-mstop fit.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import GammaLSS, GaussianLSS, cvrisk_glmboostLSS
from conftest import r_call


def _r_cvrisk(rfamily, gen, mstop=30, nu=0.1, n=120, seed=1):
    import rpy2.robjects as ro
    f = ro.r(f"""
    function(mstop, nu, n, seed) {{
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        {gen}
        d <- data.frame(y = y, x1 = x1, x2 = x2)
        m <- glmboostLSS(y ~ x1 + x2, data = d, families = {rfamily}(),
                         control = boost_control(mstop = mstop, nu = nu))
        folds <- cv(model.weights(m), type = "kfold", B = 5)
        cvr <- cvrisk(m, folds = folds, trace = FALSE)
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
             folds = as.matrix(folds), risk = as.matrix(cvr),
             cn = colnames(cvr))
    }}
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    grid = [[int(a) for a in c.split(",")] for c in r.rx2("cn")]
    return {
        "x1": np.asarray(r.rx2("x1")), "x2": np.asarray(r.rx2("x2")),
        "y": np.asarray(r.rx2("y")), "folds": np.asarray(r.rx2("folds")),
        "risk": np.asarray(r.rx2("risk")), "grid": grid,
    }


@pytest.mark.r_parity
def test_cvrisk_glmboostlss_gaussian():
    gen = "x1 <- rnorm(n); x2 <- rnorm(n); y <- rnorm(n, 1 + 0.8 * x1, exp(0.3 * x2))"
    r = _r_cvrisk("GaussianLSS", gen)
    res = cvrisk_glmboostLSS(
        "y ~ x1 + x2", data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
        families=GaussianLSS(), control=mboost.boost_control(30, 0.1),
        folds=r["folds"], grid=r["grid"], response_scale=None,
    )
    np.testing.assert_allclose(res["risk"], r["risk"], rtol=1e-6, atol=1e-7,
                               err_msg="GaussianLSS cvrisk does not match R")


@pytest.mark.r_parity
def test_cvrisk_glmboostlss_gamma():
    gen = ("x1 <- rnorm(n); x2 <- rnorm(n); "
           "mu <- exp(0.5 + 0.3 * x1); sig <- exp(0.2 + 0.2 * x2); "
           "y <- rgamma(n, shape = sig, scale = mu / sig)")
    r = _r_cvrisk("GammaLSS", gen)
    res = cvrisk_glmboostLSS(
        "y ~ x1 + x2", data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
        families=GammaLSS(), control=mboost.boost_control(30, 0.1),
        folds=r["folds"], grid=r["grid"], response_scale=None,
    )
    np.testing.assert_allclose(res["risk"], r["risk"], rtol=1e-6, atol=1e-7,
                               err_msg="GammaLSS cvrisk does not match R")
