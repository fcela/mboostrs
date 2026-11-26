"""R-parity test for gamboostLSS by= (varying-coefficient) new-data prediction.

Mirrors tests/test_by_parity/test_by_r_parity.py::test_by_bols_predict_newdata_parity
(the ordinary gamboost oracle), but fits the same model with the R
``gamboostLSS`` package directly through rpy2 -- NOT the mboostrs R wrapper,
which rejects by= at parse time on purpose. Marked with @pytest.mark.r_parity
so it's skipped unless --run-r-parity is passed; also skips (rather than
fails) if the R gamboostLSS package itself is unavailable.
"""

import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS, gamboostLSS

import mboostrs as mb
from conftest import r_call

rtol = 1e-6
atol = 1e-7


@pytest.mark.r_parity
def test_by_bbs_gamboostlss_predict_newdata_parity():
    """bbs(x, by=z, df=4) + bols(w) under GaussianLSS(): newdata predictions
    for both mu and sigma must match R's gamboostLSS package."""
    import rpy2.robjects as ro

    mstop_val = 50
    nu_val = 0.1

    r_code = """
    function(mstop, nu, n, seed) {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(seed)
        x <- runif(n, 0, 10)
        z <- rbinom(n, 1, 0.5)
        w <- runif(n, -2, 2)
        y <- 3.0 + sin(x) * z + 0.5 * w + rnorm(n, sd = 0.5)
        fit <- gamboostLSS(
            y ~ bbs(x, by = z, df = 4) + bols(w),
            data = data.frame(x = x, z = z, w = w, y = y),
            families = GaussianLSS(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        nd_z0 <- data.frame(x = c(2.0, 4.0, 6.0, 8.0), z = c(0, 0, 0, 0), w = c(0.3, -0.3, 0.6, -0.6))
        nd_z1 <- data.frame(x = c(2.0, 4.0, 6.0, 8.0), z = c(1, 1, 1, 1), w = c(0.3, -0.3, 0.6, -0.6))
        list(
            x = as.numeric(x),
            z = as.numeric(z),
            w = as.numeric(w),
            y = as.numeric(y),
            mu_z0 = as.numeric(predict(fit, parameter = "mu", newdata = nd_z0, type = "link")),
            mu_z1 = as.numeric(predict(fit, parameter = "mu", newdata = nd_z1, type = "link")),
            sigma_z0 = as.numeric(predict(fit, parameter = "sigma", newdata = nd_z0, type = "link")),
            sigma_z1 = as.numeric(predict(fit, parameter = "sigma", newdata = nd_z1, type = "link"))
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_call(r_func, mstop_val, nu_val, 200, 42, packages=("gamboostLSS",))

    r = {
        "x": np.asarray(result.rx2("x"), dtype=np.float64),
        "z": np.asarray(result.rx2("z"), dtype=np.float64),
        "w": np.asarray(result.rx2("w"), dtype=np.float64),
        "y": np.asarray(result.rx2("y"), dtype=np.float64),
        "mu_z0": np.asarray(result.rx2("mu_z0"), dtype=np.float64),
        "mu_z1": np.asarray(result.rx2("mu_z1"), dtype=np.float64),
        "sigma_z0": np.asarray(result.rx2("sigma_z0"), dtype=np.float64),
        "sigma_z1": np.asarray(result.rx2("sigma_z1"), dtype=np.float64),
    }

    data = {"y": r["y"], "x": r["x"], "z": r["z"], "w": r["w"]}
    model = gamboostLSS(
        "y ~ bbs(x, df=4, by=z) + bols(w)",
        data=data,
        families=GaussianLSS(),
        control=mb.boost_control(mstop_val, nu_val),
        response_scale=None,
    )

    newdata_z0 = {
        "x": np.array([2.0, 4.0, 6.0, 8.0]),
        "z": np.array([0.0, 0.0, 0.0, 0.0]),
        "w": np.array([0.3, -0.3, 0.6, -0.6]),
    }
    newdata_z1 = {
        "x": np.array([2.0, 4.0, 6.0, 8.0]),
        "z": np.array([1.0, 1.0, 1.0, 1.0]),
        "w": np.array([0.3, -0.3, 0.6, -0.6]),
    }

    mu_z0 = np.asarray(model.predict(parameter="mu", newdata=newdata_z0, type="link"), dtype=np.float64)
    mu_z1 = np.asarray(model.predict(parameter="mu", newdata=newdata_z1, type="link"), dtype=np.float64)
    sigma_z0 = np.asarray(
        model.predict(parameter="sigma", newdata=newdata_z0, type="link"), dtype=np.float64
    )
    sigma_z1 = np.asarray(
        model.predict(parameter="sigma", newdata=newdata_z1, type="link"), dtype=np.float64
    )

    max_abs_diff = max(
        np.max(np.abs(mu_z0 - r["mu_z0"])),
        np.max(np.abs(mu_z1 - r["mu_z1"])),
        np.max(np.abs(sigma_z0 - r["sigma_z0"])),
        np.max(np.abs(sigma_z1 - r["sigma_z1"])),
    )
    print(f"\n  gamboostLSS by= newdata parity: max abs diff = {max_abs_diff:.3e}")

    np.testing.assert_allclose(
        mu_z0, r["mu_z0"], rtol=rtol, atol=atol,
        err_msg="gamboostLSS mu newdata (z=0) does not match R",
    )
    np.testing.assert_allclose(
        mu_z1, r["mu_z1"], rtol=rtol, atol=atol,
        err_msg="gamboostLSS mu newdata (z=1) does not match R",
    )
    np.testing.assert_allclose(
        sigma_z0, r["sigma_z0"], rtol=rtol, atol=atol,
        err_msg="gamboostLSS sigma newdata (z=0) does not match R",
    )
    np.testing.assert_allclose(
        sigma_z1, r["sigma_z1"], rtol=rtol, atol=atol,
        err_msg="gamboostLSS sigma newdata (z=1) does not match R",
    )
