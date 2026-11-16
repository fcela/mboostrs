"""R-parity tests for brandom (random effects base-learner).

Tests that our Rust/Python brandom implementation matches R mboost's brandom
for fitted values, risk path, selected learners, and predictions.

R's brandom() is equivalent to bols() with contrasts.arg = "contr.dummy"
(i.e., indicator matrix without dropping a reference level), with a ridge
penalty (lambda * I) on the coefficients.

IMPORTANT: R's brandom(intercept=TRUE) is IDENTICAL to brandom() because:
- brandom() delegates to bols(intercept=TRUE, contrasts.arg="contr.dummy")
- bols defaults to intercept=TRUE, so brandom() always has intercept=TRUE
- With contr.dummy, X_ols uses fm="~ group -1" which removes the intercept
  column from the formula, resulting in an L-column indicator matrix
- The intercept=TRUE flag is only used to gate whether to strip the intercept
  column from model.matrix output (which doesn't happen because -1 already
  removed it)
- Therefore: brandom() == brandom(intercept=TRUE) in R

Our `center=False` (default) matches R's brandom() behavior exactly:
- L-column indicator matrix, identity penalty, no centering

Our `center=True` does spectral centering (subtracts weighted column means)
— this is our extension with NO R equivalent. R's brandom(intercept=FALSE)
ERRORS because "contr.dummy can only be used with intercept = TRUE".
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_brandom(mstop, nu, n=200, n_levels=5, seed=42):
    """Fit gamboost with brandom in R and return results."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, n_levels, seed) {
    library(mboost)
    set.seed(seed)
    group <- sample(1:n_levels, n, replace = TRUE)
    y <- 2 * group + rnorm(n, sd = 1.0)
    group_fac <- factor(group, levels = 1:n_levels)
    df <- data.frame(y = y, group = group_fac)

    fit <- gamboost(
        y ~ brandom(group, df = 4),
        data = df,
        family = Gaussian(),
        control = boost_control(mstop = mstop, nu = nu)
    )

    list(
        y = as.numeric(y),
        group = as.integer(group),
        fitted = as.numeric(stats::predict(fit, type = "link")),
        risk = as.numeric(risk(fit)),
        selected = as.integer(selected(fit)),
        offset = as.numeric(fit$offset),
        n_levels = as.integer(n_levels)
    )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, n_levels, seed)

    return {
        "y": np.asarray(result[0], dtype=np.float64),
        "group": np.asarray(result[1], dtype=np.int32),
        "fitted": np.asarray(result[2], dtype=np.float64),
        "risk": np.asarray(result[3], dtype=np.float64),
        "selected": np.asarray(result[4], dtype=np.int32).tolist(),
        "offset": float(result[5][0]),
        "n_levels": int(result[6][0]),
    }


@pytest.mark.r_parity
def test_brandom_parity():
    """Basic brandom parity: fitted, risk, offset, selected."""
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_brandom(mstop_val, nu_val)

    data = {"y": r["y"], "group": r["group"]}

    model = mboost.gamboost(
        "y ~ brandom(group, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="brandom risk path does not match R",
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="brandom fitted values do not match R",
    )

    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="brandom offset does not match R",
    )

    r_selected_0based = [s - 1 for s in r["selected"]]
    assert model._selected == r_selected_0based, (
        f"brandom selected learners differ: Rust={model._selected}, R={r_selected_0based}"
    )


@pytest.mark.r_parity
def test_brandom_parity_predict():
    """brandom predict parity: predict on same data matches R."""
    import rpy2.robjects as ro

    mstop_val = 50
    nu_val = 0.1
    n = 200
    n_levels = 5
    seed = 42

    r = _run_r_brandom(mstop_val, nu_val, n=n, n_levels=n_levels, seed=seed)

    # Fit in Python
    data = {"y": r["y"], "group": r["group"]}
    model = mboost.gamboost(
        "y ~ brandom(group, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    # Predict on same data in R
    r_code = """
    function(mstop, nu, n, n_levels, seed) {
    library(mboost)
    set.seed(seed)
    group <- sample(1:n_levels, n, replace = TRUE)
    y <- 2 * group + rnorm(n, sd = 1.0)
    group_fac <- factor(group, levels = 1:n_levels)
    df <- data.frame(y = y, group = group_fac)

    fit <- gamboost(
        y ~ brandom(group, df = 4),
        data = df,
        family = Gaussian(),
        control = boost_control(mstop = mstop, nu = nu)
    )

    # Predict on same data
    pred <- as.numeric(stats::predict(fit, newdata = df, type = "link"))

    list(pred = pred, group = as.integer(group))
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop_val, nu_val, n, n_levels, seed)

    r_pred = np.asarray(result[0], dtype=np.float64)

    # Predict on same data in Python
    py_pred = model.predict(newdata=data, type="link")

    np.testing.assert_allclose(
        py_pred, r_pred, rtol=rtol, atol=atol,
        err_msg="brandom predict does not match R",
    )


@pytest.mark.r_parity
def test_brandom_with_bbs_parity():
    """Mixed brandom + bbs: ensure brandom works alongside other learners."""
    import rpy2.robjects as ro

    mstop_val = 100
    nu_val = 0.1
    n = 200
    n_levels = 4
    seed = 42

    r_code = """
    function(mstop, nu, n, n_levels, seed) {
    library(mboost)
    set.seed(seed)
    x1 <- runif(n, 0, 1)
    group <- sample(1:n_levels, n, replace = TRUE)
    y <- sin(2 * pi * x1) + 2 * group + rnorm(n, sd = 1.0)
    group_fac <- factor(group, levels = 1:n_levels)
    df <- data.frame(y = y, x1 = x1, group = group_fac)

    fit <- gamboost(
        y ~ bbs(x1, df = 4) + brandom(group, df = 4),
        data = df,
        family = Gaussian(),
        control = boost_control(mstop = mstop, nu = nu)
    )

    list(
        x1 = as.numeric(x1),
        y = as.numeric(y),
        group = as.integer(group),
        fitted = as.numeric(stats::predict(fit, type = "link")),
        risk = as.numeric(risk(fit)),
        selected = as.integer(selected(fit)),
        offset = as.numeric(fit$offset)
    )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop_val, nu_val, n, n_levels, seed)

    r_x1 = np.asarray(result[0], dtype=np.float64)
    r_y = np.asarray(result[1], dtype=np.float64)
    r_group = np.asarray(result[2], dtype=np.int32)
    r_fitted = np.asarray(result[3], dtype=np.float64)
    r_risk = np.asarray(result[4], dtype=np.float64)
    r_selected = np.asarray(result[5], dtype=np.int32).tolist()
    r_offset = float(result[6][0])

    data = {"y": r_y, "x1": r_x1, "group": r_group}

    model = mboost.gamboost(
        "y ~ bbs(x1, df=4) + brandom(group, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.risk_, r_risk, rtol=rtol, atol=atol,
        err_msg="brandom+bbs risk path does not match R",
    )

    np.testing.assert_allclose(
        model.fitted_, r_fitted, rtol=rtol, atol=atol,
        err_msg="brandom+bbs fitted values do not match R",
    )

    np.testing.assert_allclose(
        model.offset_, r_offset, rtol=rtol, atol=atol,
        err_msg="brandom+bbs offset does not match R",
    )

    r_selected_0based = [s - 1 for s in r_selected]
    assert model._selected == r_selected_0based, (
        f"brandom+bbs selected learners differ: Rust={model._selected}, R={r_selected_0based}"
    )


@pytest.mark.r_parity
def test_brandom_intercept_true_parity():
    """brandom(intercept=TRUE) in R matches brandom() default in Python.

    R's brandom(intercept=TRUE) is IDENTICAL to brandom() because:
    - brandom() always delegates to bols(intercept=TRUE, contr.dummy)
    - With contr.dummy, X_ols uses fm='~ group -1' which already removes
      the intercept column, so intercept=TRUE has no additional effect
    - Both produce an L-column indicator matrix with identity penalty

    Our center=False (default) matches this behavior exactly.
    """
    import rpy2.robjects as ro

    mstop_val = 50
    nu_val = 0.1
    n = 200
    n_levels = 5
    seed = 42

    r_code = """
    function(mstop, nu, n, n_levels, seed) {
    library(mboost)
    set.seed(seed)
    group <- sample(1:n_levels, n, replace = TRUE)
    y <- 2 * group + rnorm(n, sd = 1.0)
    group_fac <- factor(group, levels = 1:n_levels)
    df <- data.frame(y = y, group = group_fac)

    fit <- gamboost(
        y ~ brandom(group, df = 4, intercept = TRUE),
        data = df,
        family = Gaussian(),
        control = boost_control(mstop = mstop, nu = nu)
    )

    # Predict on same data
    pred <- as.numeric(stats::predict(fit, newdata = df, type = "link"))

    list(
        y = as.numeric(y),
        group = as.integer(group),
        fitted = as.numeric(stats::predict(fit, type = "link")),
        risk = as.numeric(risk(fit)),
        selected = as.integer(selected(fit)),
        offset = as.numeric(fit$offset),
        n_levels = as.integer(n_levels),
        pred = pred
    )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop_val, nu_val, n, n_levels, seed)

    r_y = np.asarray(result[0], dtype=np.float64)
    r_group = np.asarray(result[1], dtype=np.int32)
    r_fitted = np.asarray(result[2], dtype=np.float64)
    r_risk = np.asarray(result[3], dtype=np.float64)
    r_selected = np.asarray(result[4], dtype=np.int32).tolist()
    r_offset = float(result[5][0])
    r_pred = np.asarray(result[7], dtype=np.float64)

    data = {"y": r_y, "group": r_group}

    # Use default center=False (no center arg) — matches R's brandom(intercept=TRUE)
    model = mboost.gamboost(
        "y ~ brandom(group, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.risk_, r_risk, rtol=rtol, atol=atol,
        err_msg="brandom() risk path does not match R brandom(intercept=TRUE)",
    )

    np.testing.assert_allclose(
        model.fitted_, r_fitted, rtol=rtol, atol=atol,
        err_msg="brandom() fitted values do not match R brandom(intercept=TRUE)",
    )

    np.testing.assert_allclose(
        model.offset_, r_offset, rtol=rtol, atol=atol,
        err_msg="brandom() offset does not match R brandom(intercept=TRUE)",
    )

    r_selected_0based = [s - 1 for s in r_selected]
    assert model._selected == r_selected_0based, (
        f"brandom() selected learners differ: Rust={model._selected}, R={r_selected_0based}"
    )

    py_pred = model.predict(newdata=data, type="link")
    np.testing.assert_allclose(
        py_pred, r_pred, rtol=rtol, atol=atol,
        err_msg="brandom() predict does not match R brandom(intercept=TRUE)",
    )


@pytest.mark.r_parity
def test_brandom_intercept_true_predict_newdata():
    """brandom(intercept=TRUE) predict on new data matches R.

    R's brandom(intercept=TRUE) == brandom() default. Test predict on
    new data to ensure the indicator matrix is built correctly for prediction.
    """
    import rpy2.robjects as ro

    mstop_val = 50
    nu_val = 0.1
    n = 200
    n_levels = 5
    seed = 42

    r_code = """
    function(mstop, nu, n, n_levels, seed) {
    library(mboost)
    set.seed(seed)
    group <- sample(1:n_levels, n, replace = TRUE)
    y <- 2 * group + rnorm(n, sd = 1.0)
    group_fac <- factor(group, levels = 1:n_levels)
    df <- data.frame(y = y, group = group_fac)

    fit <- gamboost(
        y ~ brandom(group, df = 4, intercept = TRUE),
        data = df,
        family = Gaussian(),
        control = boost_control(mstop = mstop, nu = nu)
    )

    # Predict on new data
    set.seed(seed + 1)
    new_group <- sample(1:n_levels, n, replace = TRUE)
    new_group_fac <- factor(new_group, levels = 1:n_levels)
    newdf <- data.frame(y = rep(0, n), group = new_group_fac)

    pred <- as.numeric(stats::predict(fit, newdata = newdf, type = "link"))

    list(
        y = as.numeric(y),
        group = as.integer(group),
        new_group = as.integer(new_group),
        fitted = as.numeric(stats::predict(fit, type = "link")),
        risk = as.numeric(risk(fit)),
        offset = as.numeric(fit$offset),
        n_levels = as.integer(n_levels),
        pred = pred
    )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop_val, nu_val, n, n_levels, seed)

    r_y = np.asarray(result[0], dtype=np.float64)
    r_group = np.asarray(result[1], dtype=np.int32)
    r_new_group = np.asarray(result[2], dtype=np.int32)
    r_fitted = np.asarray(result[3], dtype=np.float64)
    r_pred = np.asarray(result[7], dtype=np.float64)

    data = {"y": r_y, "group": r_group}
    newdata = {"y": np.zeros(n), "group": r_new_group}

    # Use default center=False — matches R's brandom(intercept=TRUE)
    model = mboost.gamboost(
        "y ~ brandom(group, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.fitted_, r_fitted, rtol=rtol, atol=atol,
        err_msg="brandom() fitted values do not match R brandom(intercept=TRUE)",
    )

    py_pred = model.predict(newdata=newdata, type="link")
    np.testing.assert_allclose(
        py_pred, r_pred, rtol=rtol, atol=atol,
        err_msg="brandom() predict(newdata) does not match R brandom(intercept=TRUE)",
    )


def test_brandom_center_true_consistency():
    """brandom(center=True) is our extension — no R equivalent.

    R's brandom(intercept=FALSE) errors because "contr.dummy can only be
    used with intercept = TRUE". Our center=True applies column-mean centering
    (subtracts weighted column means from the indicator matrix), which is
    not the same as R's intercept parameter.

    When df < n_levels, centering changes the effective df for a given lambda,
    so center=True should produce different fitted values than center=False.
    We use n_levels=8 with df=4 so lambda > 0 (penalized) in both cases.
    """
    np.random.seed(123)
    n = 200
    n_levels = 8
    group = np.random.choice(np.arange(1, n_levels + 1), size=n, replace=True)
    y = 2.0 * group + np.random.normal(scale=1.0, size=n)
    data = {"y": y, "group": group}

    model_centered = mboost.gamboost(
        "y ~ brandom(group, df=4, center=True)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    model_default = mboost.gamboost(
        "y ~ brandom(group, df=4)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    # center=True applies column-mean centering, center=False does not.
    # With df < n_levels, centering changes the effective df, so the models differ.
    max_diff = np.max(np.abs(model_centered.fitted_ - model_default.fitted_))
    assert max_diff > 0.1, (
        f"brandom center=True should differ from center=False; max_diff={max_diff}"
    )

    # Predict on training data should match fitted values
    py_pred = model_centered.predict(newdata=data, type="link")
    np.testing.assert_allclose(
        py_pred, model_centered.fitted_, rtol=1e-10, atol=1e-14,
        err_msg="brandom(center=True) predict on training data != fitted values",
    )

    # Predict on new data should work without errors
    new_group = np.random.choice(np.arange(1, n_levels + 1), size=n, replace=True)
    newdata = {"y": np.zeros(n), "group": new_group}
    py_pred_new = model_centered.predict(newdata=newdata, type="link")
    assert py_pred_new.shape == (n,), f"Predict shape mismatch: {py_pred_new.shape}"


def test_multi_brandom_center_predict():
    """Two brandom learners with different n_levels and center=True.

    This pins the centering metadata for each learner in the retained
    `LearnerStore`. The legacy array predictor concatenated all centering
    means and once used a uniform stride, which was wrong when learners had
    different numbers of levels; the compact store must preserve the same
    regression without depending on a shared flat offset.

    With group1 having 3 levels and group2 having 7 levels, the old code
    would compute start=2*3=6 for learner 2, but the correct offset is
    3 (the cumulative sum of group1's levels).
    """
    np.random.seed(99)
    n = 150
    n_levels_1 = 3
    n_levels_2 = 7
    group1 = np.random.choice(np.arange(1, n_levels_1 + 1), size=n, replace=True)
    group2 = np.random.choice(np.arange(1, n_levels_2 + 1), size=n, replace=True)
    y = 1.5 * group1 + 0.8 * group2 + np.random.normal(scale=0.5, size=n)
    data = {"y": y, "group1": group1, "group2": group2}

    model = mboost.gamboost(
        "y ~ brandom(group1, df=2, center=True) + brandom(group2, df=4, center=True)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(100, 0.1),
    )

    # predict on training data must match fitted values
    pred = model.predict(newdata=data, type="link")
    np.testing.assert_allclose(
        pred, model.fitted_, rtol=1e-10, atol=1e-14,
        err_msg="multi-brandom center=True predict != fitted values (cumulative-offset bug)",
    )

    # predict on new data must not crash and must be finite
    new_group1 = np.random.choice(np.arange(1, n_levels_1 + 1), size=n, replace=True)
    new_group2 = np.random.choice(np.arange(1, n_levels_2 + 1), size=n, replace=True)
    newdata = {"y": np.zeros(n), "group1": new_group1, "group2": new_group2}
    pred_new = model.predict(newdata=newdata, type="link")
    assert pred_new.shape == (n,), f"Predict shape mismatch: {pred_new.shape}"
    assert np.all(np.isfinite(pred_new)), "Predict on new data produced non-finite values"
