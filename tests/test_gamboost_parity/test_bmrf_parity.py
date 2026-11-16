"""R-parity tests for bmrf (Markov Random Field base-learner).

Tests that our Rust/Python bmrf implementation matches R mboost's bmrf
for fitted values, risk path, selected learners, and predictions.

NOTE: R's bmrf requires the BayesX package. These tests are skipped if
BayesX is not available.

R's hyper_bmrf requires the penalty matrix K to be a Laplacian:
  - sum(K) == 0
  - K[i,j] = -1 for neighbors, K[i,i] = degree(i)
  - rownames/colnames must match factor levels
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.learners import Bmrf

rtol = 1e-6
atol = 1e-9


# BayesX detection: rpy2's requireNamespace returns None (logical NA)
# unless wrapped with as.logical(). Handle that explicitly.
def _check_bayesx():
    try:
        import rpy2.robjects as ro
        result = ro.r("as.logical(requireNamespace('BayesX', quietly=TRUE))")
        return bool(result[0])
    except Exception:
        return False


_has_bayesx = _check_bayesx()

requires_bayesx = pytest.mark.skipif(
    not _has_bayesx,
    reason="R package 'BayesX' not available",
)


def _make_laplacian_chain(n_levels):
    """Build a Laplacian matrix for a chain graph on n_levels nodes.

    R's hyper_bmrf requires: sum(K)==0, K[i,j]=-1 for neighbors,
    K[i,i]=degree, and rownames matching factor levels.
    """
    K = np.zeros((n_levels, n_levels))
    for i in range(n_levels - 1):
        K[i, i + 1] = -1
        K[i + 1, i] = -1
    # Diagonal = degree (number of neighbors)
    for i in range(n_levels):
        K[i, i] = int(np.sum(K[i] < 0))
    return K


def _run_r_bmrf(mstop, nu, n=200, n_levels=5, seed=42):
    """Fit gamboost with bmrf in R and return results."""
    import rpy2.robjects as ro

    # R code using proper Laplacian K with rownames
    r_code = """
    function(mstop, nu, n, n_levels, seed) {
        library(mboost)
        set.seed(seed)
        region <- sample(1:n_levels, n, replace = TRUE)
        y <- 3 * region + rnorm(n, sd = 1.0)

        # Build Laplacian K matrix (sum == 0) for a chain graph
        K <- matrix(0, nrow = n_levels, ncol = n_levels)
        for (i in 1:(n_levels - 1)) {
            K[i, i + 1] <- -1
            K[i + 1, i] <- -1
        }
        # Diagonal = degree
        for (i in 1:n_levels) {
            K[i, i] <- sum(K[i, ] < 0)
        }
        rownames(K) <- 1:n_levels
        colnames(K) <- 1:n_levels

        region_fac <- factor(region, levels = 1:n_levels)
        df <- data.frame(y = y, region = region_fac)

        fit <- gamboost(
            y ~ bmrf(region, bnd = K, df = 4),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )

        list(
            y = as.numeric(y),
            region = as.integer(region),
            K = as.matrix(K),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit)),
            selected = as.integer(selected(fit)),
            offset = as.numeric(fit$offset)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, n_levels, seed)
    return {
        "y": np.asarray(result[0], dtype=np.float64),
        "region": np.asarray(result[1], dtype=np.int32) - 1,
        "K": np.asarray(result[2], dtype=np.float64),
        "fitted": np.asarray(result[3], dtype=np.float64),
        "risk": np.asarray(result[4], dtype=np.float64),
        "selected": np.asarray(result[5], dtype=np.int32) - 1,
        "offset": float(result[6][0]),
    }


@requires_bayesx
@pytest.mark.r_parity
def test_bmrf_fit_parity():
    """bmrf fitted values and risk path must match R."""
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_bmrf(mstop_val, nu_val)
    data = {"y": r["y"], "region": r["region"].astype(np.float64)}
    K = r["K"]

    model = mboost.gamboost(
        "y ~ bmrf(region, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
        learners=[Bmrf("region", K=K, df=4)],
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="bmrf fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="bmrf risk path does not match R",
    )
    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="bmrf offset does not match R",
    )
    assert model._selected == r["selected"].tolist(), (
        f"bmrf selected learners differ: Python={model._selected}, R={r['selected'].tolist()}"
    )


@requires_bayesx
@pytest.mark.r_parity
def test_bmrf_predict_parity():
    """bmrf predict on new data must match R."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, n_levels, seed) {
        library(mboost)
        set.seed(seed)
        region <- sample(1:n_levels, n, replace = TRUE)
        y <- 3 * region + rnorm(n, sd = 1.0)

        # Laplacian K matrix for chain graph
        K <- matrix(0, nrow = n_levels, ncol = n_levels)
        for (i in 1:(n_levels - 1)) {
            K[i, i + 1] <- -1
            K[i + 1, i] <- -1
        }
        for (i in 1:n_levels) {
            K[i, i] <- sum(K[i, ] < 0)
        }
        rownames(K) <- 1:n_levels
        colnames(K) <- 1:n_levels

        region_fac <- factor(region, levels = 1:n_levels)
        df <- data.frame(y = y, region = region_fac)
        fit <- gamboost(
            y ~ bmrf(region, bnd = K, df = 4),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )

        # Predict on new data with same levels
        new_region <- sample(1:n_levels, 10, replace = TRUE)
        new_region_fac <- factor(new_region, levels = 1:n_levels)
        newdata <- data.frame(region = new_region_fac)
        pred_new <- as.numeric(stats::predict(fit, newdata = newdata, type = "link"))

        list(
            y = as.numeric(y),
            region = as.integer(region),
            K = as.matrix(K),
            new_region = as.integer(new_region),
            pred_new = pred_new
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(50, 0.1, 200, 5, 42)

    y = np.asarray(result[0], dtype=np.float64)
    region = np.asarray(result[1], dtype=np.int32) - 1
    K = np.asarray(result[2], dtype=np.float64)
    new_region = np.asarray(result[3], dtype=np.int32) - 1
    pred_new_r = np.asarray(result[4], dtype=np.float64)

    data = {"y": y, "region": region.astype(np.float64)}
    model = mboost.gamboost(
        "y ~ bmrf(region, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
        learners=[Bmrf("region", K=K, df=4)],
    )

    newdata = {"region": new_region.astype(np.float64)}
    py_pred = model.predict(newdata=newdata, type="link")

    np.testing.assert_allclose(
        py_pred, pred_new_r, rtol=rtol, atol=atol,
        err_msg="bmrf predict on new data does not match R",
    )


@requires_bayesx
@pytest.mark.r_parity
def test_bmrf_explicit_lambda_parity():
    """bmrf with explicit lambda must match R."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, lambda, n, n_levels, seed) {
        library(mboost)
        set.seed(seed)
        region <- sample(1:n_levels, n, replace = TRUE)
        y <- 3 * region + rnorm(n, sd = 1.0)

        # Laplacian K matrix for chain graph
        K <- matrix(0, nrow = n_levels, ncol = n_levels)
        for (i in 1:(n_levels - 1)) {
            K[i, i + 1] <- -1
            K[i + 1, i] <- -1
        }
        for (i in 1:n_levels) {
            K[i, i] <- sum(K[i, ] < 0)
        }
        rownames(K) <- 1:n_levels
        colnames(K) <- 1:n_levels

        region_fac <- factor(region, levels = 1:n_levels)
        df <- data.frame(y = y, region = region_fac)
        fit <- gamboost(
            y ~ bmrf(region, bnd = K, lambda = lambda, df = NULL),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        list(
            y = as.numeric(y),
            region = as.integer(region),
            K = as.matrix(K),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit))
        )
    }
    """
    r_func = ro.r(r_code)

    for lam in [0.1, 1.0]:
        result = r_func(50, 0.1, lam, 200, 5, 42)
        y = np.asarray(result[0], dtype=np.float64)
        region = np.asarray(result[1], dtype=np.int32) - 1
        K = np.asarray(result[2], dtype=np.float64)
        fitted_r = np.asarray(result[3], dtype=np.float64)
        risk_r = np.asarray(result[4], dtype=np.float64)

        data = {"y": y, "region": region.astype(np.float64)}
        model = mboost.gamboost(
            "y ~ bmrf(region)",
            data=data,
            family=Gaussian(),
            control=mboost.boost_control(50, 0.1),
            learners=[Bmrf("region", K=K, lambda_value=lam)],
        )

        np.testing.assert_allclose(
            model.fitted_, fitted_r, rtol=rtol, atol=atol,
            err_msg=f"bmrf lambda={lam} fitted values do not match R",
        )
        np.testing.assert_allclose(
            model.risk_, risk_r, rtol=rtol, atol=atol,
            err_msg=f"bmrf lambda={lam} risk path does not match R",
        )


@requires_bayesx
@pytest.mark.r_parity
def test_bmrf_center_parity():
    """bmrf with center=TRUE must match R."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, n_levels, seed) {
        library(mboost)
        set.seed(seed)
        region <- sample(1:n_levels, n, replace = TRUE)
        y <- 3 * region + rnorm(n, sd = 1.0)

        # Laplacian K matrix for chain graph
        K <- matrix(0, nrow = n_levels, ncol = n_levels)
        for (i in 1:(n_levels - 1)) {
            K[i, i + 1] <- -1
            K[i + 1, i] <- -1
        }
        for (i in 1:n_levels) {
            K[i, i] <- sum(K[i, ] < 0)
        }
        rownames(K) <- 1:n_levels
        colnames(K) <- 1:n_levels

        region_fac <- factor(region, levels = 1:n_levels)
        df <- data.frame(y = y, region = region_fac)
        fit <- gamboost(
            y ~ bmrf(region, bnd = K, df = 4, center = TRUE),
            data = df, family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        list(
            y = as.numeric(y),
            region = as.integer(region),
            K = as.matrix(K),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(risk(fit)),
            offset = as.numeric(fit$offset)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(50, 0.1, 200, 5, 42)

    y = np.asarray(result[0], dtype=np.float64)
    region = np.asarray(result[1], dtype=np.int32) - 1
    K = np.asarray(result[2], dtype=np.float64)
    fitted_r = np.asarray(result[3], dtype=np.float64)
    risk_r = np.asarray(result[4], dtype=np.float64)
    offset_r = float(result[5][0])

    data = {"y": y, "region": region.astype(np.float64)}
    model = mboost.gamboost(
        "y ~ bmrf(region, df=4, center=TRUE)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
        learners=[Bmrf("region", K=K, df=4, center=True)],
    )

    np.testing.assert_allclose(
        model.fitted_, fitted_r, rtol=rtol, atol=atol,
        err_msg="bmrf center=TRUE fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, risk_r, rtol=rtol, atol=atol,
        err_msg="bmrf center=TRUE risk path does not match R",
    )
    np.testing.assert_allclose(
        model.offset_, offset_r, rtol=rtol, atol=atol,
        err_msg="bmrf center=TRUE offset does not match R",
    )
