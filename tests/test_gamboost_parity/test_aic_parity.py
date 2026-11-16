"""R-parity tests for AIC (corrected and gMDL) in gamboost."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-5
atol = 1e-6


def _run_r_aic(mstop, nu, n=50, seed=42, method="corrected"):
    """Generate data in R, fit gamboost, compute AIC, return results.

    R's AIC() returns a scalar (AIC at optimal mstop), but the full
    AIC vector is stored in attr(aic_obj, "AIC"). We extract both.
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, method) {
        library(mboost)
        set.seed(seed)
        x <- rnorm(n)
        z <- rnorm(n)
        y <- 1.5*x + sin(2*z) + rnorm(n, sd=0.3)
        data <- data.frame(x=x, z=z, y=y)
        fit <- gamboost(
            y ~ bols(x) + bbs(z, df=4),
            data=data,
            family=Gaussian(),
            control=boost_control(mstop=mstop, nu=nu)
        )
        aic_obj <- AIC(fit, method=method)
        list(
            x = as.numeric(x),
            z = as.numeric(z),
            y = as.numeric(y),
            aic_scalar = as.numeric(aic_obj),
            aic_all = as.numeric(attr(aic_obj, "AIC")),
            df = as.numeric(attr(aic_obj, "df")),
            mstop_opt = as.integer(attr(aic_obj, "mstop"))
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed, method)

    return {
        "x": np.asarray(result[0], dtype=np.float64),
        "z": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "aic_scalar": float(result[3][0]),
        "aic_all": np.asarray(result[4], dtype=np.float64),
        "df": np.asarray(result[5], dtype=np.float64),
        "mstop_opt": int(result[6][0]),
    }


@pytest.mark.r_parity
def test_aic_corrected_parity():
    """AIC corrected (Hurvich-Clifford) matches R."""
    mstop_val = 100
    nu_val = 0.1

    r = _run_r_aic(mstop_val, nu_val, method="corrected")

    data = {"y": r["y"], "x": r["x"], "z": r["z"]}
    model = mboost.gamboost(
        "y ~ bols(x) + bbs(z, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    aic_result = model.aic(method="corrected")

    # DF trace must match R
    np.testing.assert_allclose(
        aic_result.df, r["df"], rtol=rtol, atol=atol,
        err_msg="DF trace does not match R for corrected AIC",
    )

    # Full AIC vector must match R
    np.testing.assert_allclose(
        aic_result.aic, r["aic_all"], rtol=rtol, atol=atol,
        err_msg="AIC (corrected) values do not match R",
    )

    # AIC at optimal mstop must match R's scalar
    np.testing.assert_allclose(
        aic_result.aic[aic_result.mstop - 1], r["aic_scalar"], rtol=rtol, atol=atol,
        err_msg="AIC at optimal mstop does not match R",
    )

    # Optimal mstop must match
    assert aic_result.mstop == r["mstop_opt"], (
        f"Optimal mstop differs: Py={aic_result.mstop}, R={r['mstop_opt']}"
    )


@pytest.mark.r_parity
def test_aic_gmdl_parity():
    """AIC gMDL matches R."""
    mstop_val = 100
    nu_val = 0.1

    r = _run_r_aic(mstop_val, nu_val, method="gMDL")

    data = {"y": r["y"], "x": r["x"], "z": r["z"]}
    model = mboost.gamboost(
        "y ~ bols(x) + bbs(z, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    aic_result = model.aic(method="gmdl")

    # DF trace must match R
    np.testing.assert_allclose(
        aic_result.df, r["df"], rtol=rtol, atol=atol,
        err_msg="DF trace does not match R for gMDL AIC",
    )

    # Full AIC vector must match R
    np.testing.assert_allclose(
        aic_result.aic, r["aic_all"], rtol=rtol, atol=atol,
        err_msg="AIC (gMDL) values do not match R",
    )

    # AIC at optimal mstop must match R's scalar
    np.testing.assert_allclose(
        aic_result.aic[aic_result.mstop - 1], r["aic_scalar"], rtol=rtol, atol=atol,
        err_msg="AIC at optimal mstop (gMDL) does not match R",
    )

    # Optimal mstop must match
    assert aic_result.mstop == r["mstop_opt"], (
        f"Optimal mstop differs: Py={aic_result.mstop}, R={r['mstop_opt']}"
    )


@pytest.mark.r_parity
def test_aic_corrected_bbs_only():
    """AIC corrected with only bbs learners (no bols intercept issue)."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        z <- rnorm(n)
        y <- sin(2*z) + rnorm(n, sd=0.3)
        data <- data.frame(z=z, y=y)
        fit <- gamboost(
            y ~ bbs(z, df=4),
            data=data,
            family=Gaussian(),
            control=boost_control(mstop=mstop, nu=nu)
        )
        aic_obj <- AIC(fit, method="corrected")
        list(
            z = as.numeric(z),
            y = as.numeric(y),
            aic_all = as.numeric(attr(aic_obj, "AIC")),
            df = as.numeric(attr(aic_obj, "df")),
            mstop_opt = as.integer(attr(aic_obj, "mstop"))
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(100, 0.1, 50, 42)

    z = np.asarray(result[0], dtype=np.float64)
    y = np.asarray(result[1], dtype=np.float64)
    r_aic_all = np.asarray(result[2], dtype=np.float64)
    r_df = np.asarray(result[3], dtype=np.float64)
    r_mstop = int(result[4][0])

    data = {"y": y, "z": z}
    model = mboost.gamboost(
        "y ~ bbs(z, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )

    aic_result = model.aic(method="corrected")

    np.testing.assert_allclose(
        aic_result.df, r_df, rtol=rtol, atol=atol,
        err_msg="DF trace (bbs-only) does not match R",
    )
    np.testing.assert_allclose(
        aic_result.aic, r_aic_all, rtol=rtol, atol=atol,
        err_msg="AIC (bbs-only, corrected) does not match R",
    )
    assert aic_result.mstop == r_mstop, (
        f"Optimal mstop differs: Py={aic_result.mstop}, R={r_mstop}"
    )


@pytest.mark.r_parity
def test_aic_hat_matrix_trace():
    """Hat matrix traces match R basemodel hatvalues."""
    import rpy2.robjects as ro

    ro.r("library(mboost)")
    ro.r("set.seed(42)")
    ro.r("n <- 50; x <- rnorm(n); z <- rnorm(n)")
    ro.r("y <- 1.5*x + sin(2*z) + rnorm(n, sd=0.3)")
    ro.r("data <- data.frame(x=x, z=z, y=y)")
    ro.r("fit <- gamboost(y ~ bols(x) + bbs(z, df=4), data=data, control=boost_control(mstop=100, nu=0.1), family=Gaussian())")

    r_H0_trace = float(ro.r("sum(diag(fit$basemodel[[1]]$hatvalues()))")[0])
    r_Hnu0_trace = float(ro.r("sum(diag(fit$hatvalues()[[1]]))")[0])
    r_Hnu1_trace = float(ro.r("sum(diag(fit$hatvalues()[[2]]))")[0])

    x_py = np.asarray(ro.r("x"), dtype=np.float64)
    z_py = np.asarray(ro.r("z"), dtype=np.float64)
    y_py = np.asarray(ro.r("y"), dtype=np.float64)

    data = {"y": y_py, "x": x_py, "z": z_py}
    model = mboost.gamboost(
        "y ~ bols(x) + bbs(z, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )

    ht = model._learner_store.hat_traces()
    assert ht is not None, "Hat traces should not be None"

    py_H0_trace = ht[0]
    py_H1_trace = ht[1]

    np.testing.assert_allclose(
        py_H0_trace, r_Hnu0_trace, rtol=1e-10,
        err_msg=f"bols(x) hat matrix trace: Py={py_H0_trace}, R={r_Hnu0_trace}",
    )
    np.testing.assert_allclose(
        py_H1_trace, r_Hnu1_trace, rtol=1e-6,
        err_msg=f"bbs(z) hat matrix trace: Py={py_H1_trace}, R={r_Hnu1_trace}",
    )

    # Also check: unscaled trace should be 2 for bols(x, intercept=TRUE)
    nu = 0.1
    py_H0_unscaled_trace = py_H0_trace / nu
    np.testing.assert_allclose(
        py_H0_unscaled_trace, r_H0_trace, rtol=1e-10,
        err_msg=f"bols(x) unscaled hat trace: Py={py_H0_unscaled_trace}, R={r_H0_trace}",
    )


@pytest.mark.r_parity
def test_aic_df_trace_early_iterations():
    """DF trace at early iterations matches R (tests intercept fix)."""
    import rpy2.robjects as ro

    ro.r("library(mboost)")
    ro.r("set.seed(42)")
    ro.r("n <- 50; x <- rnorm(n); z <- rnorm(n)")
    ro.r("y <- 1.5*x + sin(2*z) + rnorm(n, sd=0.3)")
    ro.r("data <- data.frame(x=x, z=z, y=y)")
    ro.r("fit <- gamboost(y ~ bols(x) + bbs(z, df=4), data=data, control=boost_control(mstop=100, nu=0.1), family=Gaussian())")
    ro.r("aic_r <- AIC(fit, method='corrected')")
    ro.r("df_r <- attr(aic_r, 'df')")

    r_df = np.asarray(ro.r("df_r"), dtype=np.float64)

    x_py = np.asarray(ro.r("x"), dtype=np.float64)
    z_py = np.asarray(ro.r("z"), dtype=np.float64)
    y_py = np.asarray(ro.r("y"), dtype=np.float64)

    data = {"y": y_py, "x": x_py, "z": z_py}
    model = mboost.gamboost(
        "y ~ bols(x) + bbs(z, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )

    aic_result = model.aic(method="corrected")

    # Early iterations are the most sensitive to the intercept fix
    # Before the fix, df[0] was 0.1 (missing intercept), should be 0.2
    np.testing.assert_allclose(
        aic_result.df[:5], r_df[:5], rtol=1e-8,
        err_msg=f"Early DF trace mismatch: Py={aic_result.df[:5]}, R={r_df[:5]}",
    )

    # Full trace
    np.testing.assert_allclose(
        aic_result.df, r_df, rtol=1e-6, atol=1e-9,
        err_msg="Full DF trace does not match R",
    )
