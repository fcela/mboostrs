"""Hat values and classical AIC for GLM families match mboost.

mboost forms the boosting hat matrix of a Binomial or Poisson fit with each
step weighted by the IRLS weight ``fW(f_m)`` at the fit after that step
(``bhatmat``); Gaussian uses the unweighted matrix. Both frontends used the
unweighted matrix for every family, which put Poisson gamboost AIC 1.04 and
Binomial glmboost AIC 0.16 away from mboost on the fixture below.
"""

import numpy as np
import pytest

import mboostrs as mb

# mboost's `predict(aggregate = "cumsum")` for a Binomial glmboost sits a
# constant 1.6e-4 away from its own `fitted()` on this fixture; the hat
# matrix inherits that through fW. Every other case agrees to 1e-11.
TOL = {("Binomial", "glm"): (1e-5, 1e-7)}

CASES = [(fam, kind) for fam in ("Gaussian", "Binomial", "Poisson") for kind in ("gam", "glm")]


@pytest.fixture(scope="module")
def data():
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r(
        """suppressMessages(library(mboost)); set.seed(42); n <- 300
        d <- data.frame(x = runif(n, -1, 1), z = rnorm(n), w = runif(n))
        eta <- sin(2 * d$x) + 0.5 * d$z
        d$yg <- eta + rnorm(n, sd = .4)
        d$yb <- factor(rbinom(n, 1, plogis(eta)))
        d$yp <- rpois(n, exp(0.3 * eta))
        d$yb01 <- as.integer(d$yb) - 1L"""
    )
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(ro.r("d"))


@pytest.mark.r_parity
@pytest.mark.parametrize("family,kind", CASES)
def test_hatvalues_and_aic_match_mboost(data, family, kind):
    import rpy2.robjects as ro

    resp = {"Gaussian": "yg", "Binomial": "yb", "Poisson": "yp"}[family]
    ours_resp = "yb01" if resp == "yb" else resp
    rhs = "bbs(x) + bols(z)" if kind == "gam" else "x + z + w"
    fit = "gamboost" if kind == "gam" else "glmboost"
    ro.r(f"m <- {fit}({resp} ~ {rhs}, data = d, family = {family}(), "
         "control = boost_control(mstop = 60))")
    model = getattr(mb, fit)(f"{ours_resp} ~ {rhs}", data=data,
                             family=getattr(mb, family)(), control=mb.boost_control(60))
    method = "corrected" if family == "Gaussian" else "classical"
    aic_tol, hat_tol = TOL.get((family, kind), (1e-10, 1e-11))

    upstream_aic = np.asarray(ro.r(f"attr(AIC(m, method = '{method}'), 'AIC')"))
    np.testing.assert_allclose(model.aic(method=method).aic, upstream_aic, rtol=0, atol=aic_tol)
    upstream_hat = np.asarray(ro.r("hatvalues(m)"))
    np.testing.assert_allclose(model.hatvalues(), upstream_hat, rtol=0, atol=hat_tol)
