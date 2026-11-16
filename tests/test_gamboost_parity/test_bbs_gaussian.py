"""R-parity tests for gamboost Gaussian + bbs."""

import time

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_gamboost_bbs(mstop, nu, n=100, seed=42):
    """Generate data in R and fit gamboost with bbs, returning data + results + R runtime."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed) {
        library(mboost)
        set.seed(seed)
        x1 <- runif(n, 0, 1)
        y <- sin(2 * pi * x1) + rnorm(n, sd = 0.3)
        t0 <- proc.time()
        fit <- gamboost(
            y ~ bbs(x1, df = 4, knots = 20),
            data = data.frame(x1 = x1, y = y),
            family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        t1 <- proc.time()
        list(
            x1 = as.numeric(x1),
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
        "y": np.asarray(result[1], dtype=np.float64),
        "fitted": np.asarray(result[2], dtype=np.float64),
        "risk": np.asarray(result[3], dtype=np.float64),
        "selected": np.asarray(result[4], dtype=np.int32).tolist(),
        "offset": float(result[5][0]),
        "r_time": float(result[6][0]),
    }


@pytest.mark.r_parity
def test_gamboost_bbs_parity():
    mstop_val = 50
    nu_val = 0.1

    r = _run_r_gamboost_bbs(mstop_val, nu_val)

    data = {"y": r["y"], "x1": r["x1"]}

    t0 = time.perf_counter()
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )
    rust_time = time.perf_counter() - t0

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="risk path does not match R",
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="fitted values do not match R",
    )

    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="offset does not match R",
    )

    # R's selected indices are 1-based; ours are 0-based
    r_selected_0based = [s - 1 for s in r["selected"]]
    assert model._selected == r_selected_0based, (
        f"selected learners differ: Rust={model._selected}, R={r_selected_0based}"
    )

    r_time = r["r_time"]
    speedup = r_time / rust_time if rust_time > 0 else float("inf")
    # Informational only — wall-clock timing is not a correctness assertion.
    print(f"\n  bbs parity: Rust={rust_time:.4f}s, R={r_time:.4f}s, speedup={speedup:.1f}x")


@pytest.mark.r_parity
def test_gamboost_bbs_parity_large_n():
    """Test at n=5000 to check numerical stability and runtime scaling."""
    n_val = 5000
    mstop_val = 200
    nu_val = 0.1

    r = _run_r_gamboost_bbs(mstop_val, nu_val, n=n_val)

    data = {"y": r["y"], "x1": r["x1"]}

    t0 = time.perf_counter()
    model = mboost.gamboost(
        "y ~ bbs(x1, df=4, knots=20)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )
    rust_time = time.perf_counter() - t0

    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="risk path does not match R at n=5000",
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="fitted values do not match R at n=5000",
    )

    np.testing.assert_allclose(
        model.offset_, r["offset"], rtol=rtol, atol=atol,
        err_msg="offset does not match R at n=5000",
    )

    r_selected_0based = [s - 1 for s in r["selected"]]
    assert model._selected == r_selected_0based, (
        f"selected learners differ at n=5000: Rust={model._selected}, R={r_selected_0based}"
    )

    r_time = r["r_time"]
    speedup = r_time / rust_time if rust_time > 0 else float("inf")
    # Informational only — wall-clock timing is not a correctness assertion.
    print(f"\n  bbs n=5000: Rust={rust_time:.4f}s, R={r_time:.4f}s, speedup={speedup:.1f}x")
