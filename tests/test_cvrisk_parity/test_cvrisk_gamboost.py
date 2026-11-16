"""R-parity tests for cvrisk() with gamboost Gaussian + bbs + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_cvrisk_gamboost(mstop_val, nu, n=200, seed=42, B=10):
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, B) {
    library(mboost)
    set.seed(seed)
    x1 <- runif(n, 0, 1)
    x2 <- rnorm(n)
    y <- sin(2 * pi * x1) + 0.5 * x2 + rnorm(n, sd = 0.3)
    data <- data.frame(x1 = x1, x2 = x2, y = y)
    fit <- gamboost(
        y ~ bbs(x1, df = 4, knots = 20) + bols(x2),
        data = data,
        family = Gaussian(),
        control = boost_control(mstop = mstop, nu = nu)
    )
    folds <- cv(model.weights(fit), type = "kfold", B = B)
    cvr <- cvrisk(fit, folds = folds)
    opt <- mstop(cvr)
    list(
        x1 = as.numeric(x1),
        x2 = as.numeric(x2),
        y = as.numeric(y),
        risk_matrix = as.matrix(cvr),
        mstop_opt = as.integer(opt),
        colmeans = as.numeric(colMeans(cvr, na.rm = TRUE)),
        folds = as.matrix(folds)
    )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop_val, nu, n, seed, B)

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "y": np.asarray(result[2], dtype=np.float64),
        "risk_matrix": np.asarray(result[3], dtype=np.float64),
        "mstop_opt": int(result[4][0]),
        "colmeans": np.asarray(result[5], dtype=np.float64),
        "folds": np.asarray(result[6], dtype=np.float64),
    }


@pytest.mark.r_parity
def test_cvrisk_kfold_gaussian_gamboost():
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_cvrisk_gamboost(mstop_val, nu_val, n=200, seed=42, B=10)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    result = model.cvrisk(folds=r["folds"])

    assert result.risk_matrix.shape == r["risk_matrix"].shape, (
        f"Shape mismatch: {result.risk_matrix.shape} vs {r['risk_matrix'].shape}"
    )

    np.testing.assert_allclose(
        result.risk_matrix, r["risk_matrix"],
        rtol=rtol, atol=atol,
        err_msg="OOB risk matrix does not match R",
    )

    np.testing.assert_allclose(
        result.aggregate, r["colmeans"],
        rtol=rtol, atol=atol,
        err_msg="Aggregate risk path does not match R",
    )

    assert result.optimal_mstop == r["mstop_opt"], (
        f"Optimal mstop: {result.optimal_mstop} vs R's {r['mstop_opt']}"
    )


@pytest.mark.r_parity
def test_cvrisk_subsampling_gaussian_gamboost():
    mstop_val = 100
    nu_val = 0.1

    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, B) {
    library(mboost)
    set.seed(seed)
    x1 <- runif(n, 0, 1)
    x2 <- rnorm(n)
    y <- sin(2 * pi * x1) + 0.5 * x2 + rnorm(n, sd = 0.3)
    data <- data.frame(x1 = x1, x2 = x2, y = y)
    fit <- gamboost(
        y ~ bbs(x1, df = 4, knots = 20) + bols(x2),
        data = data,
        family = Gaussian(),
        control = boost_control(mstop = mstop, nu = nu)
    )
    folds <- cv(model.weights(fit), type = "subsampling", B = B)
    cvr <- cvrisk(fit, folds = folds)
    list(
        x1 = as.numeric(x1),
        x2 = as.numeric(x2),
        y = as.numeric(y),
        risk_matrix = as.matrix(cvr),
        mstop_opt = as.integer(mstop(cvr)),
        folds = as.matrix(folds)
    )
    }
    """
    r_func = ro.r(r_code)
    result_r = r_func(mstop_val, nu_val, 100, 42, 25)

    r_x1 = np.asarray(result_r[0], dtype=np.float64)
    r_x2 = np.asarray(result_r[1], dtype=np.float64)
    r_y = np.asarray(result_r[2], dtype=np.float64)
    r_risk = np.asarray(result_r[3], dtype=np.float64)
    r_mstop = int(result_r[4][0])
    r_folds = np.asarray(result_r[5], dtype=np.float64)

    data = {"y": r_y, "x1": r_x1, "x2": r_x2}
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20) + bols(x2)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    result = model.cvrisk(folds=r_folds)

    np.testing.assert_allclose(
        result.risk_matrix, r_risk,
        rtol=rtol, atol=atol,
        err_msg="OOB risk matrix (subsampling) does not match R",
    )

    assert result.optimal_mstop == r_mstop, (
        f"Optimal mstop (subsampling): {result.optimal_mstop} vs R's {r_mstop}"
    )
