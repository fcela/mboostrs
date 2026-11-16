"""R-parity test for gamboost Gaussian + bns.

R's `bns` base-learner was removed from mboost ("Consider 'bbs' instead").
mboostrs implements `bns` as a B-spline learner with an identity (ridge)
penalty, i.e. `bbs(..., differences = 0)`. So the parity reference is R's
`bbs(x, differences = 0)`.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

# bns uses an identity (differences=0) penalty, so the per-step linear solve is
# less regularized and more ill-conditioned than bbs; R and Rust diverge by
# ~1e-7 in fitted values (vs ~1e-12 for penalized bbs). Tolerances reflect that.
rtol = 1e-5
atol = 1e-6


def _run_r_gamboost_bns(mstop, nu, df, n=100, seed=42):
    """Fit R gamboost with bbs(differences=0) as the bns reference."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, df, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        y <- sin(2 * pi * x1) + rnorm(n, sd = 0.3)
        fit <- gamboost(
            y ~ bbs(x1, df = df, knots = 20, degree = 3, differences = 0),
            data = data.frame(x1 = x1, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        nx <- c(0.1, 0.3, 0.55, 0.9)
        list(
            x1 = as.numeric(x1),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit)),
            offset = as.numeric(fit$offset),
            nx = as.numeric(nx),
            newpred = as.numeric(stats::predict(fit, newdata = data.frame(x1 = nx),
                                                type = "link"))
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, df, n, seed)
    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "y": np.asarray(result[1], dtype=np.float64),
        "fitted": np.asarray(result[2], dtype=np.float64),
        "risk": np.asarray(result[3], dtype=np.float64),
        "offset": float(result[4][0]),
        "nx": np.asarray(result[5], dtype=np.float64),
        "newpred": np.asarray(result[6], dtype=np.float64),
    }


@pytest.mark.r_parity
def test_gamboost_bns_parity():
    mstop_val, nu_val, df_val = 50, 0.1, 4.0
    r = _run_r_gamboost_bns(mstop_val, nu_val, df_val)
    data = {"y": r["y"], "x1": r["x1"]}

    model = mboost.gamboost(
        "y ~ bns(x1, df=4, knots=20, degree=3)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="bns risk path does not match R bbs(differences=0)",
    )
    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="bns fitted values do not match R bbs(differences=0)",
    )
    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="bns offset does not match R",
    )

    newpred = np.asarray(model.predict(newdata={"x1": r["nx"]}, type="link"))
    np.testing.assert_allclose(
        newpred, r["newpred"], rtol=rtol, atol=atol,
        err_msg="bns newdata prediction does not match R bbs(differences=0)",
    )
