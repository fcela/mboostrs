"""R-parity tests for cvrisk() with glmboost Gaussian + bols."""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_cvrisk(mstop_val, nu, n=200, seed=42, B=10):
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, B) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.3)
        data <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data,
            family = mboost::Gaussian(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        folds <- mboost::cv(model.weights(fit), type = "kfold", B = B)
        cvr <- mboost::cvrisk(fit, folds = folds)
        opt <- mboost::mstop(cvr)
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            risk_matrix = as.matrix(cvr),
            mstop_opt = as.integer(opt),
            colmeans = as.numeric(colMeans(cvr, na.rm = TRUE))
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
    }


@pytest.mark.r_parity
def test_cvrisk_kfold_gaussian_glmboost():
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_cvrisk(mstop_val, nu_val, n=200, seed=42, B=10)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    # Generate same folds in Python (we use kfold with B=10, seed=42)
    # R uses set.seed(42) before cv(), our Python uses seed=42 in ChaCha8Rng
    # The RNGs are different, so we pass R's fold matrix directly
    # First, get R's folds and pass them to cvrisk
    import rpy2.robjects as ro

    r_folds_code = """
    function(mstop, nu, n, seed, B) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.3)
        data <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data,
            family = mboost::Gaussian(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        folds <- mboost::cv(model.weights(fit), type = "kfold", B = B)
        as.matrix(folds)
    }
    """
    r_folds_func = ro.r(r_folds_code)
    r_folds = np.asarray(r_folds_func(mstop_val, nu_val, 200, 42, 10), dtype=np.float64)

    result = model.cvrisk(folds=r_folds)

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
def test_cvrisk_subsampling_gaussian_glmboost():
    mstop_val = 100
    nu_val = 0.1

    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, B) {
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.3)
        data <- data.frame(x1 = x1, x2 = x2, y = y)
        fit <- mboost::glmboost(
            y ~ x1 + x2,
            data = data,
            family = mboost::Gaussian(),
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        folds <- mboost::cv(model.weights(fit), type = "subsampling", B = B)
        cvr <- mboost::cvrisk(fit, folds = folds)
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            y = as.numeric(y),
            risk_matrix = as.matrix(cvr),
            mstop_opt = as.integer(mboost::mstop(cvr)),
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
    model = mboost.glmboost(
        "y ~ x1 + x2",
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


@pytest.mark.r_parity
def test_cvrisk_bootstrap_binomial_glmboost():
    """cvrisk for Binomial — exercises the {0,1}->{-1,+1} response recoding.

    Regression test: the recoding was applied in the fit entry points but NOT
    on the cvrisk path, so cvrisk fed a raw {0,1} response into a loss expecting
    {-1,+1}; the OOB risk (and hence selected mstop) was wrong. The recoding now
    lives in the core (Family::encode_response), shared by fit and cvrisk.
    """
    import rpy2.robjects as ro
    from mboostrs.families import Binomial

    mstop_val, nu_val = 60, 0.1
    res = ro.r("""
    function(mstop, nu, n, seed, B) {
        suppressMessages(library(mboost))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        yn <- rbinom(n, 1, plogis(0.7 * x1)); yf <- factor(yn)
        fit <- glmboost(yf ~ x1 + x2, data = data.frame(yf = yf, x1 = x1, x2 = x2),
                        family = Binomial(), control = boost_control(mstop, nu))
        set.seed(99); folds <- cv(model.weights(fit), type = "bootstrap", B = B)
        cvr <- cvrisk(fit, folds = folds, papply = lapply)
        oob <- matrix(as.numeric(cvr), nrow = nrow(cvr))
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), yn = as.numeric(yn),
             folds = as.numeric(folds), nf = nrow(folds), Bf = ncol(folds),
             oob = as.numeric(oob), no = nrow(oob), Bo = ncol(oob),
             rms = as.integer(mstop(cvr)))
    }
    """)(mstop_val, nu_val, 130, 5, 12)
    def g(k):
        return np.asarray(res.rx2(k), dtype=np.float64)

    folds = g("folds").reshape((int(res.rx2("Bf")[0]), int(res.rx2("nf")[0]))).T
    oob = g("oob").reshape((int(res.rx2("Bo")[0]), int(res.rx2("no")[0]))).T

    data = {"y": g("yn"), "x1": g("x1"), "x2": g("x2")}
    model = mboost.glmboost("y ~ x1 + x2", data=data, family=Binomial(),
                            control=mboost.boost_control(mstop_val, nu_val))
    cvr = model.cvrisk(folds=folds, parallel=False)

    assert np.asarray(cvr.risk_matrix).shape == oob.shape
    np.testing.assert_allclose(
        np.asarray(cvr.risk_matrix), oob, rtol=1e-6, atol=1e-9,
        err_msg="Binomial cvrisk OOB risk matrix does not match R",
    )
    assert cvr.optimal_mstop == int(res.rx2("rms")[0]), (
        f"Binomial cvrisk mstop: py={cvr.optimal_mstop} R={int(res.rx2('rms')[0])}"
    )


@pytest.mark.r_parity
def test_cvrisk_bootstrap_nbinomial_glmboost():
    """cvrisk for NBinomial — exercises the per-step INBAG nuisance reuse.

    Regression test: cv previously RE-ESTIMATED the nuisance (sigma) on the OOB
    sample at each step, whereas R uses the inbag per-step sigma for the OOB
    risk. That gave the wrong OOB-risk curve and selected mstop (probe: py=29 vs
    R=8). The core now records the inbag per-step nuisance during the fit
    (BoostingPath::nuisance_path) and the OOB walk reuses it.
    """
    import rpy2.robjects as ro
    from mboostrs.families import NBinomial

    mstop_val, nu_val = 80, 0.1
    res = ro.r("""
    function(mstop, nu, n, seed, B) {
        suppressMessages(library(mboost))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        y <- rnbinom(n, mu = exp(0.4 * x1), size = 2)
        fit <- glmboost(y ~ x1 + x2, data = data.frame(y = y, x1 = x1, x2 = x2),
                        family = NBinomial(), control = boost_control(mstop, nu))
        set.seed(99); folds <- cv(model.weights(fit), type = "bootstrap", B = B)
        cvr <- cvrisk(fit, folds = folds, papply = lapply)
        oob <- matrix(as.numeric(cvr), nrow = nrow(cvr))
        list(x1 = as.numeric(x1), x2 = as.numeric(x2), y = as.numeric(y),
             folds = as.numeric(folds), nf = nrow(folds), Bf = ncol(folds),
             oob = as.numeric(oob), no = nrow(oob), Bo = ncol(oob),
             rms = as.integer(mstop(cvr)))
    }
    """)(mstop_val, nu_val, 130, 5, 12)
    def g(k):
        return np.asarray(res.rx2(k), dtype=np.float64)

    folds = g("folds").reshape((int(res.rx2("Bf")[0]), int(res.rx2("nf")[0]))).T
    oob = g("oob").reshape((int(res.rx2("Bo")[0]), int(res.rx2("no")[0]))).T

    data = {"y": g("y"), "x1": g("x1"), "x2": g("x2")}
    model = mboost.glmboost("y ~ x1 + x2", data=data, family=NBinomial(),
                            control=mboost.boost_control(mstop_val, nu_val))
    cvr = model.cvrisk(folds=folds, parallel=False)

    assert np.asarray(cvr.risk_matrix).shape == oob.shape
    # Residual is the per-step Brent sigma optimizer vs R's optimize (~1e-6).
    np.testing.assert_allclose(
        np.asarray(cvr.risk_matrix), oob, rtol=1e-4, atol=1e-5,
        err_msg="NBinomial cvrisk OOB risk matrix does not match R",
    )
    assert cvr.optimal_mstop == int(res.rx2("rms")[0]), (
        f"NBinomial cvrisk mstop: py={cvr.optimal_mstop} R={int(res.rx2('rms')[0])}"
    )
