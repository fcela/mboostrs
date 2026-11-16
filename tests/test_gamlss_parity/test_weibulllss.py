"""R-parity / correctness tests for the WeibullLSS survival family.

Weibull accelerated-failure-time model with a right-censored
``Surv(time, status)`` response: mu (identity link, log-location) and sigma
(log link, scale).

IMPORTANT: gamboostLSS computes each component's offset twice (once in
``mboostLSS_fit``'s offset loop, then again inside ``glmboost(offset="scalar")``).
By the second pass ``mu`` is already set in the family environment and
``WeibullLSS()``'s ``mu@offset`` returns ``log(mu)`` there — so gamboostLSS stores
``log`` of the optimized location even though mu's link is the identity. That is a
bug; mboostrs stores the correct (un-logged) location instead. See
``.remember/gamboostlss-survival-offset-bugs.md``.

Correctness is validated in two parts:
1. ``test_weibulllss_offset_minimizes_correct_risk``: our auto-offset is a local
   minimum of the correct Weibull negative log-likelihood.
2. ``test_weibulllss_boosting_parity_*``: fed R's own offset, our boosting matches
   R's gamboostLSS to machine precision — the gradients and machinery are
   identical; only the (corrected) offset differs.

The data scale is kept gentle (``sd ~ 0.5`` on the log-time scale): R's WeibullLSS
boosting gradient contains ``exp(eta)`` and overflows / aborts on heavy-tailed
draws, so a moderate response keeps the R reference fit finite.

Requires R packages mboost + gamboostLSS + survival; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import WeibullLSS, gamboostLSS, glmboostLSS
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-7
PARAMS = ("mu", "sigma")


def _nll(mu, eta_sigma, t, d):
    # Weibull standard log-density e - exp(e); log-survival -exp(e).
    e = (np.log(t) - mu) / np.exp(eta_sigma)
    logf = e - np.exp(e)
    logs = -np.exp(e)
    return float(np.sum(-d * (logf - eta_sigma) - (1.0 - d) * logs))


def _r_glm(mstop, nu, n=200, seed=7):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS)); suppressMessages(library(survival))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        t <- exp(1 + 0.3 * x1 + 0.5 * rnorm(n)); st <- rbinom(n, 1, 0.7)
        m <- glmboostLSS(Surv(t, st) ~ x1 + x2,
                         data = data.frame(t = t, st = st, x1 = x1, x2 = x2),
                         families = WeibullLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(t = as.numeric(t), st = as.numeric(st),
             x1 = as.numeric(x1), x2 = as.numeric(x2),
             omu = m$mu$offset, osigma = m$sigma$offset,
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    out = {k: np.asarray(r.rx2(k), dtype=np.float64)
           for k in ("t", "st", "x1", "x2", "mu", "sigma")}
    out["offset"] = [float(r.rx2("omu")[0]), float(r.rx2("osigma")[0])]
    return out


def _r_gam(mstop, nu, n=200, seed=9):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS)); suppressMessages(library(survival))
        set.seed(seed)
        x <- runif(n, -1, 1)
        t <- exp(1 + sin(2 * x) + 0.4 * rnorm(n)); st <- rbinom(n, 1, 0.7)
        m <- gamboostLSS(Surv(t, st) ~ bbs(x, knots = 20, df = 4),
                         data = data.frame(t = t, st = st, x = x),
                         families = WeibullLSS(),
                         control = boost_control(mstop = mstop, nu = nu))
        list(t = as.numeric(t), st = as.numeric(st), x = as.numeric(x),
             omu = m$mu$offset, osigma = m$sigma$offset,
             mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    out = {k: np.asarray(r.rx2(k), dtype=np.float64) for k in ("t", "st", "x", "mu", "sigma")}
    out["offset"] = [float(r.rx2("omu")[0]), float(r.rx2("osigma")[0])]
    return out


def test_weibulllss_offset_minimizes_correct_risk():
    # Our auto-offset must be a local min of the correct Weibull NLL — in
    # particular the mu offset is the un-logged location (gamboostLSS logs it).
    rng = np.random.default_rng(5)
    t = np.exp(0.8 + 0.5 * rng.standard_normal(150))
    d = (rng.random(150) < 0.7).astype(float)
    m = glmboostLSS("Surv(t, st) ~ x",
                    data={"t": t, "st": d, "x": rng.standard_normal(150)},
                    families=WeibullLSS(), control=mboost.boost_control(1, 0.1))
    mu_off = m._params[0]["offset"]
    sig_off = m._params[1]["offset"]
    h = 1e-3
    r0 = _nll(mu_off, 0.0, t, d)
    assert r0 <= _nll(mu_off + h, 0.0, t, d) and r0 <= _nll(mu_off - h, 0.0, t, d)
    s0 = _nll(mu_off, sig_off, t, d)
    assert s0 <= _nll(mu_off, sig_off + h, t, d) and s0 <= _nll(mu_off, sig_off - h, t, d)


@pytest.mark.r_parity
def test_weibulllss_boosting_parity_glm():
    mstop, nu = 100, 0.1
    r = _r_glm(mstop, nu)
    m = glmboostLSS("Surv(t, st) ~ x1 + x2",
                    data={"t": r["t"], "st": r["st"], "x1": r["x1"], "x2": r["x2"]},
                    families=WeibullLSS(), control=mboost.boost_control(mstop, nu),
                    offset=r["offset"])
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"WeibullLSS {p} (link) boosting does not match R at equal offset")


@pytest.mark.r_parity
def test_weibulllss_boosting_parity_gam_bbs():
    mstop, nu = 100, 0.1
    r = _r_gam(mstop, nu)
    m = gamboostLSS("Surv(t, st) ~ bbs(x, knots=20, df=4)",
                    data={"t": r["t"], "st": r["st"], "x": r["x"]},
                    families=WeibullLSS(), control=mboost.boost_control(mstop, nu),
                    offset=r["offset"])
    for p in PARAMS:
        np.testing.assert_allclose(
            m.predict(parameter=p, type="link"), r[p], rtol=RTOL, atol=ATOL,
            err_msg=f"WeibullLSS bbs {p} (link) boosting does not match R at equal offset")
