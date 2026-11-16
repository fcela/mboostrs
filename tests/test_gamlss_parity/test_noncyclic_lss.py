"""R-parity tests for non-cyclic glmboostLSS (R's method="noncyclic").

Non-cyclic boosting (Thomas et al. 2018) updates, at each step, only the single
distribution parameter whose best base-learner update most reduces the joint
risk. The selection rule (R's ``which.min`` over the per-parameter candidate
risks) and the per-parameter step counts match R exactly, and the final in-sample
risk matches to a tight relative tolerance.

NB: at near-tied steps (two parameters whose candidate risks differ by less than
the ~1e-8 scalar-bols fit precision), R and mboostrs may pick different but
equally-optimal parameters — the same flat-objective ambiguity that affects R's
``optimize``/Brent offsets. The two paths reconverge; predictions agree to a few
1e-3 and the optimized risk agrees to ~1e-5 relative. We therefore assert on the
selection counts and the risk value, not on exact predictions.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import mboostrs as mboost  # noqa: F401
import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS, glmboostLSS
from conftest import r_call


def _run_r(mstop, nu, n=150, seed=1):
    import rpy2.robjects as ro
    f = ro.r("""
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n); y <- rnorm(n, 1 + 0.8 * x1, exp(0.3 * x2))
        m <- glmboostLSS(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                         families = GaussianLSS(),
                         control = boost_control(mstop = mstop, nu = nu),
                         method = "noncyclic")
        cr <- attr(m, "combined_risk")()
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
             sel_mu = length(selected(m)$mu), sel_sigma = length(selected(m)$sigma),
             final_risk = as.numeric(tail(cr, 1)))
    }
    """)
    r = r_call(f, mstop, nu, n, seed, packages=("gamboostLSS",))
    return {
        "x1": np.asarray(r.rx2("x1")), "x2": np.asarray(r.rx2("x2")),
        "y": np.asarray(r.rx2("y")),
        "sel_mu": int(r.rx2("sel_mu")[0]), "sel_sigma": int(r.rx2("sel_sigma")[0]),
        "final_risk": float(r.rx2("final_risk")[0]),
    }


@pytest.mark.r_parity
def test_noncyclic_glmboostlss_counts_and_risk():
    mstop, nu = 50, 0.1
    r = _run_r(mstop, nu)
    m = glmboostLSS("y ~ x1 + x2", data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
                    families=GaussianLSS(), control=mboost.boost_control(mstop, nu),
                    method="noncyclic")
    n_mu = len(m._params[0]["selected"])
    n_sigma = len(m._params[1]["selected"])
    assert (n_mu, n_sigma) == (r["sel_mu"], r["sel_sigma"]), (
        f"non-cyclic step counts differ: mboostrs ({n_mu},{n_sigma}) "
        f"vs R ({r['sel_mu']},{r['sel_sigma']})")
    assert n_mu + n_sigma == mstop

    # Final in-sample risk (what non-cyclic optimizes) matches R closely.
    assert abs(float(m.risk_[-1]) - r["final_risk"]) / abs(r["final_risk"]) < 1e-4


@pytest.mark.r_parity
def test_noncyclic_matches_before_first_tie():
    # Up to the first near-tied step the paths are identical, so predictions
    # match to machine precision there.
    mstop, nu = 16, 0.1
    r = _run_r(mstop, nu)

    import rpy2.robjects as ro
    rp = ro.r("""
    function(x1, x2, y, mstop, nu) {
        suppressMessages(library(gamboostLSS))
        m <- glmboostLSS(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                         families = GaussianLSS(),
                         control = boost_control(mstop = mstop, nu = nu),
                         method = "noncyclic")
        list(mu = as.numeric(predict(m, parameter = "mu", type = "link")),
             sigma = as.numeric(predict(m, parameter = "sigma", type = "link")))
    }
    """)(ro.FloatVector(r["x1"]), ro.FloatVector(r["x2"]), ro.FloatVector(r["y"]),
         mstop, nu)
    r_mu = np.asarray(rp.rx2("mu"))
    r_sigma = np.asarray(rp.rx2("sigma"))

    m = glmboostLSS("y ~ x1 + x2", data={"y": r["y"], "x1": r["x1"], "x2": r["x2"]},
                    families=GaussianLSS(), control=mboost.boost_control(mstop, nu),
                    method="noncyclic")
    np.testing.assert_allclose(m.predict(parameter="mu", type="link"), r_mu,
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(m.predict(parameter="sigma", type="link"), r_sigma,
                               rtol=1e-6, atol=1e-7)
