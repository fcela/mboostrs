"""R-parity tests for survfit() (R's mboost:::survFit.mboost) on CoxPH models.

Breslow baseline-survival estimator applied to a fitted boosting model's
linear predictor: a post-fit summary computation, not a new boosting family,
so machine-precision agreement with R is expected (no approximation).
"""

import numpy as np
import pytest
from mboostrs.families import CoxPH, Gaussian

import mboostrs as mboost


def _run_r_survfit(formula, mstop, nu, n=200, seed=11, gam=False):
    import rpy2.robjects as ro

    fit_call = "gamboost" if gam else "glmboost"
    r_code = f"""
    function(mstop, nu, n, seed) {{
        suppressMessages(library(mboost))
        suppressMessages(library(survival))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        time <- rexp(n, rate = exp(0.5 * x1 - 0.3 * x2))
        cens <- rexp(n, rate = 0.3)
        status <- as.integer(time <= cens)
        obs_time <- pmin(time, cens)
        dat <- data.frame(time = obs_time, status = status, x1 = x1, x2 = x2)
        m <- {fit_call}(
            {formula},
            data = dat, family = CoxPH(),
            control = boost_control(mstop = mstop, nu = nu))
        sf <- survFit(m)
        newdat <- data.frame(x1 = c(0.5, -0.5), x2 = c(0, 1))
        sf_new <- survFit(m, newdata = newdat)
        list(
            time = obs_time, status = as.double(status), x1 = x1, x2 = x2,
            sf_time = sf$time, sf_nevent = sf$n.event, sf_surv = as.numeric(sf$surv),
            sf_new_surv = as.numeric(sf_new$surv)
        )
    }}
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu, n, seed)
    return {k: np.asarray(result.rx2(k), dtype=np.float64)
            for k in ("time", "status", "x1", "x2", "sf_time", "sf_nevent", "sf_surv", "sf_new_surv")}


@pytest.mark.r_parity
def test_survfit_glmboost_matches_r():
    r = _run_r_survfit("Surv(time, status) ~ x1 + x2", 50, 0.1, seed=11)
    data = {"time": r["time"], "status": r["status"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, status) ~ x1 + x2", data=data, family=CoxPH(),
        control=mboost.boost_control(50, 0.1),
    )
    sf = model.survfit()
    assert sf["time"].shape == r["sf_time"].shape
    np.testing.assert_allclose(sf["time"], r["sf_time"], rtol=0, atol=0)
    np.testing.assert_allclose(sf["n_event"], r["sf_nevent"], rtol=0, atol=0)
    np.testing.assert_allclose(sf["surv"][:, 0], r["sf_surv"], rtol=1e-8, atol=1e-10)

    newdata = {"x1": np.array([0.5, -0.5]), "x2": np.array([0.0, 1.0])}
    sf_new = model.survfit(newdata=newdata)
    surv_r_new = r["sf_new_surv"].reshape(2, -1).T
    assert sf_new["surv"].shape == surv_r_new.shape
    np.testing.assert_allclose(sf_new["surv"], surv_r_new, rtol=1e-8, atol=1e-10)


@pytest.mark.r_parity
def test_survfit_gamboost_matches_r():
    r = _run_r_survfit(
        "Surv(time, status) ~ bbs(x1, df = 4) + bols(x2)", 60, 0.1, seed=22, gam=True,
    )
    data = {"time": r["time"], "status": r["status"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "Surv(time, status) ~ bbs(x1, df=4) + bols(x2)", data=data, family=CoxPH(),
        control=mboost.boost_control(60, 0.1),
    )
    sf = model.survfit()
    assert sf["time"].shape == r["sf_time"].shape
    np.testing.assert_allclose(sf["n_event"], r["sf_nevent"], rtol=0, atol=0)
    np.testing.assert_allclose(sf["surv"][:, 0], r["sf_surv"], rtol=1e-6, atol=1e-8)


def test_survfit_rejects_weighted_fit():
    rng = np.random.default_rng(3)
    n = 60
    x1 = rng.normal(size=n)
    time = rng.exponential(size=n)
    status = rng.binomial(1, 0.6, size=n).astype(float)
    data = {"time": time, "status": status, "x1": x1}
    weights = np.ones(n)
    weights[0] = 2.0
    model = mboost.glmboost(
        "Surv(time, status) ~ x1", data=data, family=CoxPH(),
        control=mboost.boost_control(30, 0.1), weights=weights,
    )
    with pytest.raises(NotImplementedError, match="weights"):
        model.survfit()


def test_survfit_rejects_non_coxph_family():
    rng = np.random.default_rng(4)
    n = 50
    data = {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    model = mboost.glmboost(
        "y ~ x1", data=data, family=Gaussian(), control=mboost.boost_control(20, 0.1),
    )
    with pytest.raises(NotImplementedError, match="CoxPH"):
        model.survfit()
