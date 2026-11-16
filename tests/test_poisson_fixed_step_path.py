"""Poisson gamboost risk path with the fixed step nu = 0.03 on large counts.

Data: 2,000 rows, y ~ Poisson(exp(3.9 + 0.8 sin(x1) + 0.5 x2)), so the
counts have a median near 50. The fit is gamboost(y ~ bbs(x1) + bbs(x2)) with
mstop = 100 and nu = 0.03, which is the step the Python layer warns about.

What the tests establish (issue #395):

- The mboostrs risk path agrees with R mboost 2.9.11 to about 1e-9 relative
  through iteration 20, and the first learner selections agree.
- From about iteration 26 the two paths drift apart by floating-point
  rounding that the boosting map amplifies. At iteration 100 they differ by
  about 2 percent on the same data, and R's own path moves by much more when
  its offset is changed by one unit in the last place. No change of
  arithmetic inside mboostrs can make iteration 100 agree with R to parity
  tolerance, so that comparison is recorded as an expected failure.
- The mboostrs values at iterations 1 and 20 are pinned in a test that does
  not need R. Iteration 100 is not pinned: a different BLAS, CPU or thread
  count moves it by a large fraction, for the same reason it differs from R.
"""

import numpy as np
import pytest

import mboostrs as mb

NU = 0.03
MSTOP = 100
FORMULA = "y ~ bbs(x1) + bbs(x2)"

# Tolerance for the R comparison through iteration 20. The observed
# relative difference at iteration 20 is about 5e-10.
RTOL_EARLY = 1e-8

# mboostrs values on this data (seed 0), printed by the same fit.
PINNED_RISK = {
    1: 77099.70470227547,
    20: 111228.57150682776,
}
RTOL_PINNED = 1e-9


def _large_counts(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    mu = np.exp(3.9 + 0.8 * np.sin(x1) + 0.5 * x2)
    return {"y": rng.poisson(mu), "x1": x1, "x2": x2}


def _fit_mboostrs(mstop=MSTOP, offset=None):
    data = _large_counts()
    kwargs = {}
    if offset is not None:
        kwargs["offset"] = offset
    return mb.gamboost(FORMULA, data=data, family=mb.Poisson(),
                       control=mb.boost_control(mstop, NU), **kwargs)


def test_risk_path_is_pinned_at_iterations_1_and_20():
    model = _fit_mboostrs()
    risk = np.asarray(model.risk_)
    assert risk.shape == (MSTOP + 1,)
    assert np.isfinite(risk).all()
    for iteration, value in PINNED_RISK.items():
        np.testing.assert_allclose(
            risk[iteration], value, rtol=RTOL_PINNED,
            err_msg=f"mboostrs risk at iteration {iteration} changed",
        )


def test_the_fit_steps_up_and_down_under_the_fixed_step():
    # The step is too large for this loss, so the training risk is not
    # monotone. The test records that the path is not monotone, so a future
    # change that makes it monotone is noticed.
    risk = np.asarray(_fit_mboostrs().risk_)
    assert np.any(np.diff(risk) > 0)


@pytest.mark.r_parity
def test_r_mboost_risk_path_agrees_through_iteration_20():
    import rpy2.robjects as ro

    data = _large_counts()
    ro.r("suppressMessages(library(mboost))")
    ro.globalenv["d"] = ro.DataFrame({
        "y": ro.FloatVector(data["y"].astype(float)),
        "x1": ro.FloatVector(data["x1"]),
        "x2": ro.FloatVector(data["x2"]),
    })
    ro.r(f"fitR <- mboost::gamboost({FORMULA}, data = d, "
         f"family = mboost::Poisson(), "
         f"control = mboost::boost_control(mstop = {MSTOP}, nu = {NU}))")
    r_risk = np.asarray(ro.r("as.numeric(mboost::risk(fitR))"))
    r_selected = np.asarray(ro.r("as.integer(mboost::selected(fitR))")) - 1

    model = _fit_mboostrs()
    np.testing.assert_allclose(
        np.asarray(model.risk_)[:21], r_risk[:21], rtol=RTOL_EARLY, atol=0,
        err_msg="risk path differs from R before iteration 21",
    )
    np.testing.assert_array_equal(
        np.asarray(model._selected)[:20], r_selected[:20],
        err_msg="learner selections differ from R in the first 20 steps",
    )


@pytest.mark.r_parity
@pytest.mark.xfail(
    strict=True,
    reason=("R and mboostrs paths drift apart by rounding amplified by the "
            "fixed step; see test_r_mboost_path_is_itself_sensitive_to_the_offset. "
            "Strict: this starts failing (XPASS) only if the paths come to agree."),
)
def test_r_mboost_risk_at_iteration_100_agrees_to_parity_tolerance():
    import rpy2.robjects as ro

    data = _large_counts()
    ro.r("suppressMessages(library(mboost))")
    ro.globalenv["d"] = ro.DataFrame({
        "y": ro.FloatVector(data["y"].astype(float)),
        "x1": ro.FloatVector(data["x1"]),
        "x2": ro.FloatVector(data["x2"]),
    })
    ro.r(f"fitR <- mboost::gamboost({FORMULA}, data = d, "
         f"family = mboost::Poisson(), "
         f"control = mboost::boost_control(mstop = {MSTOP}, nu = {NU}))")
    r_risk = np.asarray(ro.r("as.numeric(mboost::risk(fitR))"))

    model = _fit_mboostrs()
    np.testing.assert_allclose(
        np.asarray(model.risk_)[MSTOP], r_risk[MSTOP], rtol=1e-6,
    )


@pytest.mark.r_parity
def test_r_mboost_path_is_itself_sensitive_to_the_offset():
    """R's own risk at iteration 100 moves by a large fraction when the offset
    changes by one unit in the last place, so the iteration-100 mismatch with
    mboostrs is not a defect in the mboostrs arithmetic alone."""
    import rpy2.robjects as ro

    data = _large_counts()
    ro.r("suppressMessages(library(mboost))")
    ro.globalenv["d"] = ro.DataFrame({
        "y": ro.FloatVector(data["y"].astype(float)),
        "x1": ro.FloatVector(data["x1"]),
        "x2": ro.FloatVector(data["x2"]),
    })
    ro.r(f"fitA <- mboost::gamboost({FORMULA}, data = d, "
         f"family = mboost::Poisson(), "
         f"control = mboost::boost_control(mstop = {MSTOP}, nu = {NU}))")
    offset = float(ro.r("fitA$offset")[0])
    ro.globalenv["offp"] = ro.FloatVector([offset + np.spacing(offset)])
    ro.r(f"fitB <- mboost::gamboost({FORMULA}, data = d, "
         f"family = mboost::Poisson(), offset = offp, "
         f"control = mboost::boost_control(mstop = {MSTOP}, nu = {NU}))")
    risk_a = np.asarray(ro.r("as.numeric(mboost::risk(fitA))"))
    risk_b = np.asarray(ro.r("as.numeric(mboost::risk(fitB))"))

    relative_change = abs(risk_b[MSTOP] - risk_a[MSTOP]) / abs(risk_a[MSTOP])
    assert relative_change > 1e-3
