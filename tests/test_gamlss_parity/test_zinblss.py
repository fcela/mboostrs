"""Correctness tests for the ZINBLSS family (zero-inflated negative binomial I).

mu (log link, mean), sigma (log link, dispersion), nu (logit link, zero-inflation
probability), wrapping gamlss.dist's ZINBI.

IMPORTANT: mboostrs uses the mathematically correct ZINBI gradient. R's
gamboostLSS::ZINBLSS() does NOT: gamlss.dist's ``dNBI(0, mu, sigma)`` is called
with a *scalar* ``x = 0``, and its trailing ``ifelse(x < 0, 0, fy)`` collapses the
result to length 1 — so ZINBI's ``dldm``/``dldd``/``dldv`` recycle the *first
observation's* zero-inflation density across all rows. mboostrs computes it per
observation instead (see ``.remember/gamboostlss-survival-offset-bugs.md``).

Because that bug is in the per-step gradient, full-trajectory R parity is not
asserted. Correctness is established by:
1. ``test_zinblss_offsets_and_density_match_r``: the offsets (gamlss initials) and
   the negative log-likelihood at the offset match R to machine precision — the
   density/risk and offsets are correct (the bug is invisible at constant params).
2. The Rust core test ``zinb_and_zipo_ngradients_match_finite_diff`` proves the
   gradient is the exact derivative of that (correct) risk.
3. ``test_zinblss_reduces_risk``: cyclic fitting decreases the joint risk.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import ZINBLSS, glmboostLSS
from conftest import r_call


def _r_offsets_and_risk0(n=250, seed=4):
    import rpy2.robjects as ro
    f = ro.r("""
    function(n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        mu <- exp(0.7 + 0.3 * x1); pz <- plogis(-1.2 + 0.5 * x2)
        z <- rbinom(n, 1, pz); y <- ifelse(z == 1, 0, rnbinom(n, size = 2, mu = mu))
        m <- glmboostLSS(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                         families = ZINBLSS(), control = boost_control(mstop = 1, nu = 0.1))
        list(y = as.numeric(y), x1 = as.numeric(x1), x2 = as.numeric(x2),
             om = m$mu$offset, os = m$sigma$offset, ov = m$nu$offset,
             risk0 = risk(m, merge = TRUE)[1])
    }
    """)
    r = r_call(f, n, seed, packages=("gamboostLSS",))
    return {
        "y": np.asarray(r.rx2("y"), float),
        "x1": np.asarray(r.rx2("x1"), float),
        "x2": np.asarray(r.rx2("x2"), float),
        "offset": [float(r.rx2("om")[0]), float(r.rx2("os")[0]), float(r.rx2("ov")[0])],
        "risk0": float(r.rx2("risk0")[0]),
    }


@pytest.mark.r_parity
def test_zinblss_offsets_and_density_match_r():
    r = _r_offsets_and_risk0()
    m = glmboostLSS("y ~ x1 + x2",
                    data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
                    families=ZINBLSS(), control=mboost.boost_control(1, 0.1))
    got = [m._params[k]["offset"] for k in range(3)]
    np.testing.assert_allclose(got, r["offset"], rtol=0, atol=1e-9,
                               err_msg="ZINBLSS offsets do not match R gamlss initials")
    # risk[0] is the ZINBI NLL at the (constant) offset, where R's dNBI gradient
    # bug is invisible — so the density itself must match R exactly.
    np.testing.assert_allclose(m.risk_[0], r["risk0"], rtol=1e-10, atol=1e-8,
                               err_msg="ZINBLSS density (risk at offset) does not match R")


def test_zinblss_reduces_risk():
    rng = np.random.default_rng(11)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    mu = np.exp(0.7 + 0.3 * x1)
    pz = 1.0 / (1.0 + np.exp(-(-1.2 + 0.5 * x2)))
    z = rng.random(n) < pz
    y = np.where(z, 0, rng.poisson(mu)).astype(float)
    m = glmboostLSS("y ~ x1 + x2", data={"y": y, "x1": x1, "x2": x2},
                    families=ZINBLSS(), control=mboost.boost_control(100, 0.1))
    risk = m.risk_
    assert risk[-1] < risk[0] - 1.0, "ZINBLSS fitting should reduce the joint risk"
    assert np.all(np.diff(risk) <= 1e-9), "ZINBLSS risk path should be non-increasing"
