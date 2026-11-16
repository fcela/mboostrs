"""R-parity tests for gamboost with the monotone P-spline learner (bmono).

bmono was the one base-learner without a parity test. R's own default is
``type="quad.prog"`` (confirmed empirically: R's default and an explicit
``type="quad.prog"`` call give bit-identical output). mboostrs defaults to
``type="iterative"`` (the violation-penalty method, lambda2 = 1e6) instead --
a deliberate speed/correctness trade-off, not a copy of R's default -- and it
happens to track R's quad.prog output closely (~1e-5 here) for every
constraint tested, so the loose tolerances below are the iterative-solver
approximation gap, not evidence of a bug. Monotonicity of the prediction is
asserted exactly.

``type="quad.prog"`` solves the constrained QP exactly on both sides, so it
matches R to ~5e-10 rather than the iterative solver's ~1e-5; it gets the
tighter ``QP_RTOL``/``QP_ATOL`` below. (Until 2026-07-20 it did not: an
inverted Lagrange-multiplier sign stranded the active-set search on a
feasible but non-optimal face. See the ``BmonoSolver::QuadProg`` doc comment
and ``bmono_quadprog_matches_exhaustive_active_set_oracle`` in
``crates/mboostrs-core/src/learners.rs`` for the regression guard.)

Note: explicit ``type="iterative"`` in the installed mboost version raises a
parse error inside mboost's own R source (an upstream bug in this mboost
build's string-built ``type="iterative"`` fit function; unrelated to
mboostrs). Explicit ``type="quad.prog"`` works fine and is what R's default
uses.
"""

import numpy as np
import pytest
from mboostrs.families import Gaussian

import mboostrs as mboost

# Iterative solver: gap vs R is the violation-penalty approximation (~1e-5).
RTOL = 1e-3
ATOL = 1e-3
# quad.prog: both sides solve the QP exactly, so only float noise remains.
QP_RTOL = 1e-8
QP_ATOL = 1e-8


def _run_r_bmono(constraint, signal, mstop=100, nu=0.1, n=150, seed=5):
    import rpy2.robjects as ro

    r_func = ro.r(f"""
    function(mstop, nu, n, seed) {{
        suppressMessages(library(mboost))
        set.seed(seed)
        x <- runif(n, 0, 4)
        y <- {signal} + rnorm(n, sd = 0.3)
        m <- gamboost(
            y ~ bmono(x, constraint = "{constraint}"),
            data = data.frame(y = y, x = x), family = Gaussian(),
            control = boost_control(mstop = mstop, nu = nu))
        set.seed(9); nx <- sort(runif(40, 0, 4))
        list(
            y = as.numeric(y), x = as.numeric(x), nx = as.numeric(nx),
            fitted = as.numeric(stats::fitted(m)),
            pred = as.numeric(stats::predict(m, newdata = data.frame(x = nx), type = "link")),
            risk = as.numeric(mboost::risk(m))
        )
    }}
    """)
    res = r_func(mstop, nu, n, seed)
    return {k: np.asarray(res.rx2(k), dtype=np.float64)
            for k in ("y", "x", "nx", "fitted", "pred", "risk")}


@pytest.mark.r_parity
@pytest.mark.parametrize("constraint,signal", [
    ("increasing", "2 * log(x + 0.5)"),
    ("decreasing", "-1.5 * x"),
])
def test_bmono_parity(constraint, signal):
    r = _run_r_bmono(constraint, signal)
    data = {"y": r["y"], "x": r["x"]}
    model = mboost.gamboost(
        f'y ~ bmono(x, constraint="{constraint}")',
        data=data, family=Gaussian(), control=mboost.boost_control(100, 0.1),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=RTOL, atol=ATOL,
        err_msg=f"bmono({constraint}) fitted values do not match R",
    )

    pred_nd = np.asarray(model.predict(newdata={"x": r["nx"]}, type="link"))
    np.testing.assert_allclose(
        pred_nd, r["pred"], rtol=RTOL, atol=ATOL,
        err_msg=f"bmono({constraint}) newdata prediction does not match R",
    )

    # The constraint must hold exactly on the prediction (newdata is sorted).
    if constraint == "increasing":
        assert np.all(np.diff(pred_nd) >= -1e-9), "prediction not monotone increasing"
    else:
        assert np.all(np.diff(pred_nd) <= 1e-9), "prediction not monotone decreasing"


@pytest.mark.r_parity
@pytest.mark.parametrize("constraint,signal", [
    ("increasing", "2 * log(x + 0.5)"),
    ("decreasing", "-1.5 * x"),
    ("convex", "(x - 2) ** 2"),
    ("concave", "-((x - 2) ** 2)"),
])
def test_bmono_quadprog_matches_r(constraint, signal):
    """type="quad.prog" solves the same QP R does, so it should match closely.

    Both sides compute the exact constrained optimum (mboostrs' active-set QP
    vs R's ``quadprog::solve.QP`` via ``mboost:::solveLSEI``), so the residual
    here is pure floating point -- three orders of magnitude tighter than the
    iterative solver's penalty-approximation gap above.

    All four shapes are covered deliberately: the multiplier-sign bug fixed on
    2026-07-20 lived in shape-agnostic code and affected every constraint kind
    (measured: ~10% of binding cases for the difference shapes, ~23% for
    positive/negative). The convex fixture previously passed by luck, not
    because second-difference constraints were exempt.
    """
    r = _run_r_bmono(constraint, signal)
    data = {"y": r["y"], "x": r["x"]}
    model = mboost.gamboost(
        f'y ~ bmono(x, constraint="{constraint}", type="quad.prog")',
        data=data, family=Gaussian(), control=mboost.boost_control(100, 0.1),
    )
    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=QP_RTOL, atol=QP_ATOL,
        err_msg=f'bmono({constraint}, type="quad.prog") fitted values do not match R',
    )


def _fitted_curve(model, grid):
    return np.asarray(model.predict(newdata={"x": grid})).ravel()


def _total_fall(curve):
    """Total downward movement of a curve that is constrained not to fall."""
    steps = np.diff(curve)
    return float(steps[steps < 0].sum())


@pytest.mark.parametrize("mstop", [10, 300, 2000])
def test_default_solver_holds_the_constraint_at_every_iteration_count(mstop):
    """The default solver must not let an "increasing" fit fall.

    `type="iterative"` leaves a small violation at each boosting step whose
    sign is consistent, so boosting accumulates it: the fitted curve fell by
    0.0037 at mstop 10 and 0.675 at mstop 2000 while the default was
    `"iterative"`. The default is now `"quad.prog"`, which is exact and is
    also R's default.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 400)
    data = {"x": x, "y": -0.5 * x + rng.normal(0, 1, 400)}
    grid = np.linspace(0, 10, 50)

    model = mboost.gamboost(
        "y ~ bmono(x, df=4, constraint='increasing')",
        data=data, family=Gaussian(), control=mboost.boost_control(mstop, 0.1),
    )
    assert _total_fall(_fitted_curve(model, grid)) == pytest.approx(0.0, abs=1e-12)


def test_bbs_with_a_constraint_is_not_silently_unconstrained():
    """`bbs(x, constraint=...)` must fit the constrained learner.

    `learner_spec` decided the engine learner from the class alone, so a `Bbs`
    carrying a shape constraint lowered to a plain `bbs` and the constraint was
    dropped with no error: `constraint="increasing"` and
    `constraint="decreasing"` both returned the unconstrained fit.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 400)
    data = {"x": x, "y": -0.5 * x + rng.normal(0, 1, 400)}
    grid = np.linspace(0, 10, 50)
    control = mboost.boost_control(500, 0.1)

    constrained = mboost.gamboost(
        "y ~ bbs(x, df=4, constraint=increasing)",
        data=data, family=Gaussian(), control=control,
    )
    unconstrained = mboost.gamboost(
        "y ~ bbs(x, df=4)", data=data, family=Gaussian(), control=control,
    )
    monotone = mboost.gamboost(
        "y ~ bmono(x, df=4, constraint='increasing')",
        data=data, family=Gaussian(), control=control,
    )

    constrained_curve = _fitted_curve(constrained, grid)
    assert _total_fall(constrained_curve) == pytest.approx(0.0, abs=1e-12)
    # The unconstrained fit follows the data down, so the two cannot agree.
    assert _total_fall(_fitted_curve(unconstrained, grid)) < -1.0
    # And the constrained spelling must match the dedicated learner.
    np.testing.assert_allclose(
        constrained_curve, _fitted_curve(monotone, grid), rtol=1e-10, atol=1e-10
    )


def test_bbs_constraint_direction_is_carried_through():
    """Opposite constraints must give different fits."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 300)
    data = {"x": x, "y": 0.4 * x + rng.normal(0, 1, 300)}
    grid = np.linspace(0, 10, 50)
    control = mboost.boost_control(400, 0.1)

    up = _fitted_curve(mboost.gamboost(
        "y ~ bbs(x, df=4, constraint=increasing)",
        data=data, family=Gaussian(), control=control), grid)
    down = _fitted_curve(mboost.gamboost(
        "y ~ bbs(x, df=4, constraint=decreasing)",
        data=data, family=Gaussian(), control=control), grid)

    assert _total_fall(up) == pytest.approx(0.0, abs=1e-12)
    assert float(np.diff(down)[np.diff(down) > 0].sum()) == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(up, down)
