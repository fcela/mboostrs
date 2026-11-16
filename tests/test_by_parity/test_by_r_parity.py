"""R-parity tests for by= varying-coefficient support vs R mboost.

These tests use rpy2 to run the same model in R and compare offset, risk, fitted, and selected.
Marked with @pytest.mark.r_parity so they're skipped unless --run-r-parity is passed.
"""

import time

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_gamboost_bols_by(mstop, nu, n=100, seed=42):
    """Run gamboost(y ~ bols(x1, by=z, intercept=FALSE)) in R, return data + results."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 10)
        z <- rbinom(n, 1, 0.5)
        y <- 3.0 + 2.0 * x1 * z + rnorm(n, sd = 0.5)
        t0 <- proc.time()
        fit <- gamboost(
            y ~ bols(x1, by = z, intercept = FALSE),
            data = data.frame(x1 = x1, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        t1 <- proc.time()
        list(
            x1 = as.numeric(x1),
            z = as.numeric(z),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit)),
            selected = as.integer(selected(fit)),
            offset = as.numeric(fit$offset),
            r_time = as.numeric((t1 - t0)["elapsed"])
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "z": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "fitted": np.asarray(result[3], dtype=np.float64),
        "risk": np.asarray(result[4], dtype=np.float64),
        "selected": np.asarray(result[5], dtype=np.int32).tolist(),
        "offset": float(result[6][0]),
        "r_time": float(result[7][0]),
    }


def _run_r_gamboost_bbs_by(mstop, nu, n=100, seed=42):
    """Run gamboost(y ~ bbs(x1, by=z, df=4)) in R, return data + results."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 10)
        z <- rbinom(n, 1, 0.5)
        y <- 3.0 + sin(x1) * z + rnorm(n, sd = 0.5)
        t0 <- proc.time()
        fit <- gamboost(
            y ~ bbs(x1, by = z, df = 4),
            data = data.frame(x1 = x1, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        t1 <- proc.time()
        list(
            x1 = as.numeric(x1),
            z = as.numeric(z),
            y = as.numeric(y),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit)),
            selected = as.integer(selected(fit)),
            offset = as.numeric(fit$offset),
            r_time = as.numeric((t1 - t0)["elapsed"])
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "z": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "fitted": np.asarray(result[3], dtype=np.float64),
        "risk": np.asarray(result[4], dtype=np.float64),
        "selected": np.asarray(result[5], dtype=np.int32).tolist(),
        "offset": float(result[6][0]),
        "r_time": float(result[7][0]),
    }


def _run_r_gamboost_bols_by_predict_newdata(mstop, nu, n=100, seed=42):
    """Run bols(x1, by=z) in R, then predict on newdata with different z values."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 10)
        z <- rbinom(n, 1, 0.5)
        y <- 3.0 + 2.0 * x1 * z + rnorm(n, sd = 0.5)
        fit <- gamboost(
            y ~ bols(x1, by = z, intercept = FALSE),
            data = data.frame(x1 = x1, z = z, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        # Predict on new data
        newdata_z0 <- data.frame(x1 = c(1.0, 5.0, 9.0), z = c(0, 0, 0))
        newdata_z1 <- data.frame(x1 = c(1.0, 5.0, 9.0), z = c(1, 1, 1))
        pred_z0 <- as.numeric(predict(fit, newdata = newdata_z0))
        pred_z1 <- as.numeric(predict(fit, newdata = newdata_z1))
        list(
            x1 = as.numeric(x1),
            z = as.numeric(z),
            y = as.numeric(y),
            offset = as.numeric(fit$offset),
            pred_z0 = pred_z0,
            pred_z1 = pred_z1
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "z": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "offset": float(result[3][0]),
        "pred_z0": np.asarray(result[4], dtype=np.float64),
        "pred_z1": np.asarray(result[5], dtype=np.float64),
    }


@pytest.mark.r_parity
def test_by_bols_gaussian_parity():
    """bols(x1, by=z, intercept=FALSE): offset, risk, fitted, selected must match R."""
    mstop_val = 20
    nu_val = 0.1

    r = _run_r_gamboost_bols_by(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "z": r["z"]}

    t0 = time.perf_counter()
    model = mboost.gamboost(
        "y ~ bols(x1, by=z, intercept=False)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )
    rust_time = time.perf_counter() - t0

    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="bols by= offset does not match R",
    )

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="bols by= risk path does not match R",
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="bols by= fitted values do not match R",
    )

    r_selected_0based = [s - 1 for s in r["selected"]]
    assert model._selected == r_selected_0based, (
        f"bols by= selected learners differ: Rust={model._selected}, R={r_selected_0based}"
    )

    r_time = r["r_time"]
    speedup = r_time / rust_time if rust_time > 0 else float("inf")
    print(f"\n  bols by= parity: Rust={rust_time:.4f}s, R={r_time:.4f}s, speedup={speedup:.1f}x")


@pytest.mark.r_parity
def test_by_bbs_gaussian_parity():
    """bbs(x1, by=z, df=4): offset, risk, fitted, selected must match R."""
    mstop_val = 20
    nu_val = 0.1

    r = _run_r_gamboost_bbs_by(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "z": r["z"]}

    t0 = time.perf_counter()
    model = mboost.gamboost(
        "y ~ bbs(x1, by=z, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )
    rust_time = time.perf_counter() - t0

    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="bbs by= offset does not match R",
    )

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="bbs by= risk path does not match R",
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="bbs by= fitted values do not match R",
    )

    r_selected_0based = [s - 1 for s in r["selected"]]
    assert model._selected == r_selected_0based, (
        f"bbs by= selected learners differ: Rust={model._selected}, R={r_selected_0based}"
    )

    r_time = r["r_time"]
    speedup = r_time / rust_time if rust_time > 0 else float("inf")
    print(f"\n  bbs by= parity: Rust={rust_time:.4f}s, R={r_time:.4f}s, speedup={speedup:.1f}x")


@pytest.mark.r_parity
def test_by_bols_predict_newdata_parity():
    """bols(x1, by=z): predict on new data must match R predictions."""
    mstop_val = 20
    nu_val = 0.1

    r = _run_r_gamboost_bols_by_predict_newdata(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"], "z": r["z"]}
    model = mboost.gamboost(
        "y ~ bols(x1, by=z, intercept=False)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="offset does not match R for newdata test",
    )

    # Predict with z=0
    newdata_z0 = {"x1": np.array([1.0, 5.0, 9.0]), "z": np.array([0.0, 0.0, 0.0])}
    pred_z0 = model.predict(newdata=newdata_z0)
    np.testing.assert_allclose(
        pred_z0, r["pred_z0"], rtol=rtol, atol=atol,
        err_msg="bols by= predict(newdata, z=0) does not match R",
    )

    # Predict with z=1
    newdata_z1 = {"x1": np.array([1.0, 5.0, 9.0]), "z": np.array([1.0, 1.0, 1.0])}
    pred_z1 = model.predict(newdata=newdata_z1)
    np.testing.assert_allclose(
        pred_z1, r["pred_z1"], rtol=rtol, atol=atol,
        err_msg="bols by= predict(newdata, z=1) does not match R",
    )


# --------------------------------------------------------------------------
# by= at the default intercept, and by= a factor
# --------------------------------------------------------------------------


def _run_r_by_variants(mstop, nu, n=200, seed=7):
    """Fit the by= spellings this file did not cover, in R.

    Every earlier test writes `intercept = FALSE`, which is the one spelling
    that never had to decide what the intercept column means under a `by`.
    """
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        z <- runif(n, -1, 1)
        g <- factor(sample(letters[1:4], n, TRUE))
        y <- z * (1 + 2 * x1) + as.numeric(g) * 0.3 * x1 + rnorm(n, sd = 0.3)
        d <- data.frame(x1 = x1, z = z, g = g, y = y)
        ctrl <- boost_control(mstop = mstop, nu = nu)
        fits <- list(
            bols_by_num    = gamboost(y ~ bols(x1, by = z), data = d,
                                      family = Gaussian(), control = ctrl),
            bols_by_num_df = gamboost(y ~ bols(x1, by = z, df = 1.5), data = d,
                                      family = Gaussian(), control = ctrl),
            bols_df        = gamboost(y ~ bols(x1, df = 1.5), data = d,
                                      family = Gaussian(), control = ctrl),
            bols_by_factor = gamboost(y ~ bols(x1, by = g), data = d,
                                      family = Gaussian(), control = ctrl),
            bbs_by_factor  = gamboost(y ~ bbs(x1, by = g, df = 10), data = d,
                                      family = Gaussian(), control = ctrl)
        )
        newdata <- data.frame(x1 = c(0.2, 0.5, 0.8), z = c(-0.5, 0.0, 0.5),
                              g = factor(c("a", "c", "d"), levels = levels(g)))
        list(
            x1 = as.numeric(x1), z = as.numeric(z), g = as.character(g),
            y = as.numeric(y),
            fitted = lapply(fits, function(f) as.numeric(fitted(f))),
            predicted = lapply(fits, function(f)
                as.numeric(stats::predict(f, newdata = newdata, type = "link")))
        )
    }
    """
    result = ro.r(r_code)(mstop, nu, n, seed)
    names = ["bols_by_num", "bols_by_num_df", "bols_df", "bols_by_factor",
             "bbs_by_factor"]
    return {
        "data": {
            "x1": np.asarray(result[0], dtype=np.float64),
            "z": np.asarray(result[1], dtype=np.float64),
            "g": np.asarray(list(result[2]), dtype=object),
            "y": np.asarray(result[3], dtype=np.float64),
        },
        "fitted": {
            name: np.asarray(result[4][i], dtype=np.float64)
            for i, name in enumerate(names)
        },
        "predicted": {
            name: np.asarray(result[5][i], dtype=np.float64)
            for i, name in enumerate(names)
        },
    }


_BY_VARIANT_FORMULAS = {
    "bols_by_num": "y ~ bols(x1, by=z)",
    "bols_by_num_df": "y ~ bols(x1, by=z, df=1.5)",
    "bols_df": "y ~ bols(x1, df=1.5)",
    "bols_by_factor": "y ~ bols(x1, by=g)",
    "bbs_by_factor": "y ~ bbs(x1, by=g, df=10)",
}


@pytest.mark.r_parity
@pytest.mark.parametrize("variant", sorted(_BY_VARIANT_FORMULAS))
def test_by_variants_match_r(variant):
    """Each by= and penalized spelling must fit what R fits.

    `bols(x, by=z)` at the default intercept fitted `{1, x*z}` where R fits
    `{z, x*z}` — R multiplies the whole design by `by`, so its learner carries
    the `by` main effect. A penalized `bols` had the same shape of error: R
    shrinks the learner's own intercept, and a constant coming from the model
    offset cannot be shrunk. A factor `by` was rejected outright.
    """
    mstop_val, nu_val = 100, 0.1
    r = _run_r_by_variants(mstop_val, nu_val)
    data = dict(r["data"])

    model = mboost.gamboost(
        _BY_VARIANT_FORMULAS[variant],
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )
    # `atol` is looser than this file's default: both implementations root-find
    # lambda to `sqrt(EPSILON)` independently, and on the stacked design of a
    # factor `by` that last digit accumulates over 100 steps to about 1e-8.
    np.testing.assert_allclose(
        model.fitted_, r["fitted"][variant], rtol=1e-6, atol=1e-7,
        err_msg=f"{variant}: fitted values do not match R",
    )

    newdata = {
        "x1": np.array([0.2, 0.5, 0.8]),
        "z": np.array([-0.5, 0.0, 0.5]),
        "g": np.array(["a", "c", "d"], dtype=object),
    }
    np.testing.assert_allclose(
        np.asarray(model.predict(newdata=newdata)).ravel(),
        r["predicted"][variant], rtol=1e-6, atol=1e-7,
        err_msg=f"{variant}: predictions on new data do not match R",
    )
