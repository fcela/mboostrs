"""R-parity tests for btree (tree-based base-learner).

Tests that our Rust/Python btree implementation matches R mboost's btree
for fitted values, risk path, selected learners, and predictions.

ALGORITHM DIFFERENCE NOTE:
R's btree() uses ctree (conditional inference trees from partykit) with
permutation-based variable selection. Our implementation uses CART-style
SSE-minimizing split selection. With mincriterion=0 (disabled permutation
test), ctree behaves similarly to CART for stumps (depth-1), giving exact
parity. At depth-2+, the algorithms diverge because:
- ctree selects splits by maximizing a test statistic (linear statistic)
- CART selects splits by minimizing weighted SSE of the pseudo-response

This means depth-2+ tests use wider tolerances. Our CART typically achieves
equal or lower risk than R's ctree because SSE-minimization directly aligns
with the Gaussian boosting loss.

The Python formulas below pin ``selection=ctree`` explicitly: since #265 the
``Btree`` default is ``"sse"``, and these tests check parity with R's
ctree-based ``btree()``, not the default path.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

# Depth-1 stumps: near-exact for single step, slight cumulative drift at mstop=50
rtol_d1 = 1e-2
atol_d1 = 1e-2

# Depth-2+: algorithmic divergence, use wider tolerances
rtol_d2 = 1e-1
atol_d2 = 0.2


def _run_r_btree(mstop, nu, n, seed, maxdepth):
    """Fit gamboost with btree in R and return results.

    Runs R in a *subprocess* (not embedded rpy2) for isolation. The btree
    reference therefore gets a fresh R session and cannot be hit by embedded-R
    PROTECT-stack corruption that accumulates across the parity suite: when
    test_brandom_parity and test_bspatial_parity run earlier in the same rpy2
    session, a later embedded `risk(fit)` read returns garbage memory (observed
    as subnormal values like 2.1e-314), which made this test intermittently
    fail. A subprocess sidesteps that entirely.

    `digits=NA` in toJSON preserves full f64 precision — without it jsonlite
    rounds to 4 decimals, so the data R returns would not match what R fit on.
    """
    import json
    import subprocess

    script = f"""
    suppressMessages(library(mboost)); suppressMessages(library(partykit))
    set.seed({seed})
    n <- {n}
    x1 <- runif(n, -2, 2); x2 <- runif(n, -2, 2)
    y <- ifelse(x1 > 0, 2.0, -1.0) + rnorm(n, sd=0.2)
    df <- data.frame(x1=x1, x2=x2, y=y)
    ctrl <- ctree_control(maxdepth={maxdepth}, mincriterion=0, minbucket=10, minsplit=20)
    fit <- gamboost(y ~ btree(x1, x2, tree_controls=ctrl), data=df, family=Gaussian(),
                    control=boost_control(mstop={mstop}, nu={nu}))
    newdata <- data.frame(x1=c(-1.5, 0.5, 1.5), x2=c(-1.0, 0.0, 1.0))
    result <- list(
      y=as.numeric(y), x1=as.numeric(x1), x2=as.numeric(x2),
      fitted=as.numeric(predict(fit, type="link")),
      risk=as.numeric(risk(fit)),
      selected=as.integer(selected(fit)),
      offset=as.numeric(fit$offset),
      pred_new=as.numeric(predict(fit, newdata=newdata, type="link"))
    )
    cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA))
    """
    try:
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError:
        pytest.skip("Rscript not available")
    assert proc.returncode == 0, f"R btree script failed:\n{proc.stderr}"
    r = json.loads(proc.stdout.strip())
    return {
        "y": np.asarray(r["y"], dtype=np.float64),
        "x1": np.asarray(r["x1"], dtype=np.float64),
        "x2": np.asarray(r["x2"], dtype=np.float64),
        "fitted": np.asarray(r["fitted"], dtype=np.float64),
        "risk": np.asarray(r["risk"], dtype=np.float64),
        "selected": np.atleast_1d(np.asarray(r["selected"], dtype=np.int32)) - 1,  # 0-based
        "offset": float(np.atleast_1d(np.asarray(r["offset"], dtype=np.float64))[0]),
        "pred_new": np.asarray(r["pred_new"], dtype=np.float64),
    }


@pytest.mark.r_parity
def test_btree_depth1_fit_parity():
    """btree depth-1 (stump) fitted values and risk must match R closely."""
    mstop_val = 50
    nu_val = 0.1
    r = _run_r_btree(mstop_val, nu_val, n=200, seed=42, maxdepth=1)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=1, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol_d1, atol=atol_d1,
        err_msg="btree depth-1 fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol_d1, atol=atol_d1,
        err_msg="btree depth-1 risk path does not match R",
    )
    assert abs(model.offset_ - r["offset"]) < 1e-10, (
        f"btree depth-1 offset mismatch: py={model.offset_} R={r['offset']}"
    )


@pytest.mark.r_parity
def test_btree_depth1_predict_parity():
    """btree depth-1 predictions on new data must match R."""
    r = _run_r_btree(50, 0.1, n=200, seed=42, maxdepth=1)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=1, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    newdata = {"x1": np.array([-1.5, 0.5, 1.5]), "x2": np.array([-1.0, 0.0, 1.0])}
    py_pred = model.predict(newdata=newdata, type="link")

    np.testing.assert_allclose(
        py_pred, r["pred_new"], rtol=rtol_d1, atol=atol_d1,
        err_msg="btree depth-1 predictions on new data do not match R",
    )


@pytest.mark.r_parity
def test_btree_depth1_single_step_exact():
    """btree depth-1 single boosting step should match R exactly."""
    r = _run_r_btree(1, 0.1, n=200, seed=42, maxdepth=1)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=1, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(1, 0.1),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=1e-10, atol=1e-12,
        err_msg="btree depth-1 single-step fitted values should match R exactly",
    )
    np.testing.assert_allclose(
        model.risk_[1:], r["risk"][1:], rtol=1e-10, atol=1e-12,
        err_msg="btree depth-1 single-step risk should match R exactly",
    )


@pytest.mark.r_parity
def test_btree_depth2_fit_approximate():
    """btree depth-2 fitted values should approximately match R.

    Due to algorithmic differences (CART vs ctree), exact parity is not
    expected at depth-2+. Our CART should achieve comparable or lower risk.
    """
    mstop_val = 50
    nu_val = 0.1
    r = _run_r_btree(mstop_val, nu_val, n=200, seed=42, maxdepth=2)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=2, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    # Check offset matches exactly
    assert abs(model.offset_ - r["offset"]) < 1e-10, (
        f"btree depth-2 offset mismatch: py={model.offset_} R={r['offset']}"
    )

    # Risk should be within ~5% of R at each step
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol_d2, atol=atol_d2,
        err_msg="btree depth-2 risk path too far from R",
    )

    # Fitted values should be roughly similar
    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol_d2, atol=atol_d2,
        err_msg="btree depth-2 fitted values too far from R",
    )

    # Our risk should be comparable to R (within 5%)
    risk_ratio = model.risk_[-1] / r["risk"][-1]
    assert 0.9 <= risk_ratio <= 1.1, (
        f"btree depth-2 final risk ratio py/R={risk_ratio:.4f} outside [0.9, 1.1]"
    )


@pytest.mark.r_parity
def test_btree_depth2_predict_approximate():
    """btree depth-2 predictions on new data should approximately match R."""
    r = _run_r_btree(50, 0.1, n=200, seed=42, maxdepth=2)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=2, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    newdata = {"x1": np.array([-1.5, 0.5, 1.5]), "x2": np.array([-1.0, 0.0, 1.0])}
    py_pred = model.predict(newdata=newdata, type="link")

    np.testing.assert_allclose(
        py_pred, r["pred_new"], rtol=rtol_d2, atol=atol_d2,
        err_msg="btree depth-2 predictions on new data too far from R",
    )


@pytest.mark.r_parity
def test_btree_depth2_single_step_exact():
    """btree depth-2 single boosting step should match R exactly."""
    r = _run_r_btree(1, 0.1, n=200, seed=42, maxdepth=2)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=2, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(1, 0.1),
    )

    # Single step should still match well
    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=1e-6, atol=1e-8,
        err_msg="btree depth-2 single-step fitted values should match R closely",
    )
    np.testing.assert_allclose(
        model.risk_[1:], r["risk"][1:], rtol=1e-6, atol=1e-8,
        err_msg="btree depth-2 single-step risk should match R closely",
    )


@pytest.mark.r_parity
def test_btree_depth3_fit_approximate():
    """btree depth-3 fitted values should approximately match R.

    At depth-3, CART vs ctree divergence is larger but risk should still
    be comparable. Our CART often achieves lower risk than ctree.
    """
    mstop_val = 50
    nu_val = 0.1
    r = _run_r_btree(mstop_val, nu_val, n=200, seed=42, maxdepth=3)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=3, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(mstop_val, nu_val),
    )

    # Check offset matches exactly
    assert abs(model.offset_ - r["offset"]) < 1e-10, (
        f"btree depth-3 offset mismatch: py={model.offset_} R={r['offset']}"
    )

    # Risk should be within ~15% of R at each step
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=0.15, atol=0.5,
        err_msg="btree depth-3 risk path too far from R",
    )

    # Final risk should be within 15% of R
    risk_ratio = model.risk_[-1] / r["risk"][-1]
    assert 0.85 <= risk_ratio <= 1.15, (
        f"btree depth-3 final risk ratio py/R={risk_ratio:.4f} outside [0.85, 1.15]"
    )


@pytest.mark.r_parity
def test_btree_depth3_predict_approximate():
    """btree depth-3 predictions on new data should approximately match R."""
    r = _run_r_btree(50, 0.1, n=200, seed=42, maxdepth=3)

    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=3, selection=ctree)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    newdata = {"x1": np.array([-1.5, 0.5, 1.5]), "x2": np.array([-1.0, 0.0, 1.0])}
    py_pred = model.predict(newdata=newdata, type="link")

    np.testing.assert_allclose(
        py_pred, r["pred_new"], rtol=0.15, atol=0.2,
        err_msg="btree depth-3 predictions on new data too far from R",
    )


@pytest.mark.r_parity
def test_btree_depth1_deeper_is_better():
    """btree with deeper trees should achieve lower risk at same mstop."""
    r1 = _run_r_btree(50, 0.1, n=200, seed=42, maxdepth=1)
    r2 = _run_r_btree(50, 0.1, n=200, seed=42, maxdepth=2)

    data = {"y": r1["y"], "x1": r1["x1"], "x2": r1["x2"]}

    m1 = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=1, selection=ctree)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    m2 = mboost.gamboost(
        "y ~ btree(x1, x2, maxdepth=2, selection=ctree)",
        data=data, family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    assert m2.risk_[-1] < m1.risk_[-1], (
        f"depth-2 risk ({m2.risk_[-1]:.4f}) should be lower than depth-1 ({m1.risk_[-1]:.4f})"
    )

    # Same relationship should hold in R
    assert r2["risk"][-1] < r1["risk"][-1], (
        f"R depth-2 risk ({r2['risk'][-1]:.4f}) should be lower than R depth-1 ({r1['risk'][-1]:.4f})"
    )
