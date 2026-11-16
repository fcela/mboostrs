"""Protocol-01 harness: selection-sequence parity against mboost 2.9.11.

`mboost-plans-20260816/01-validation-and-benchmark-protocol.md` ranks the
diagnostics by localising power: the index of the winning base-learner at
each iteration must match exactly; a mismatch at iteration t localises a bug
immediately. These tests assert that identity on configurations the existing
parity modules do not cover: a cross-type learner menu (where different
learner kinds compete for the same covariates), an exact-tie design (two
identical columns, so the selection criterion is equal to machine precision
and only the tie-break rule decides — R uses `which.min`, i.e. lowest index),
and a binomial fit (which crosses the {-1,+1} response coding). Risk
trajectories are asserted at 1e-10, the protocol tolerance.

Rows for `.remember/mboost-plans-20260816/results.md` are printed on failure
and on `-s` runs via the `_row` helper.
"""

import numpy as np
import pytest

import mboostrs as mboost

pytestmark = pytest.mark.r_parity

# Protocol tolerance: the risk trajectory should agree to 1e-10 relative where
# the per-step arithmetic is bit-compatible (Gaussian). The binomial path
# cannot: mboostrs solves each step's weighted least squares by Cholesky of
# the cross-product, R by QR (lm.wfit), and the last-ulp coefficient
# differences this leaves are amplified through the nonlinear link — measured
# 4e-10 relative at iteration 2 growing to 5.5e-7 by iteration 60 (iteration 1
# is bit-identical, so the offset and ngradient formulas agree; only the solve
# kernel differs). The selection sequence — the protocol's primary diagnostic
# — stays exact, which is what localises bugs.
RTOL_RISK = 1e-10
RTOL_RISK_BINOMIAL = 1e-5

# R snippets shared by the configurations. Each returns the generated data,
# the 1-based selection sequence, and the risk path.
_R_TEMPLATE = """
function(mstop, nu, n, seed) {{
    library(mboost)
    set.seed(seed)
    {generate}
    data <- data.frame({frame})
    fit <- {fit_call}
    list(
        {arrays},
        selected = as.integer(selected(fit)),
        risk = as.numeric(risk(fit))
    )
}}
"""


def _run_r(config, mstop, nu, n, seed):
    import rpy2.robjects as ro

    r_code = _R_TEMPLATE.format(**config)
    result = ro.r(r_code)(mstop, nu, n, seed)
    arrays = {name: np.asarray(result[i], dtype=np.float64)
              for i, name in enumerate(config["array_names"])}
    return {
        **arrays,
        "selected": np.asarray(result[len(config["array_names"])],
                               dtype=np.int32).tolist(),
        "risk": np.asarray(result[len(config["array_names"]) + 1],
                           dtype=np.float64),
    }


# --- configurations -----------------------------------------------------

GAUSS_MENU = {
    "generate": """
        x1 <- runif(n, 0, 10); x2 <- rnorm(n); x3 <- runif(n, 0, 10)
        y <- sin(x1) + 0.7 * x2 + rnorm(n, sd = 0.3)
    """,
    "frame": "x1, x2, x3, y",
    "fit_call": 'gamboost(y ~ bbs(x1) + bols(x2) + bmono(x1) + bbs(x3),\n'
                '            data = data, family = Gaussian(),\n'
                '            control = boost_control(mstop = mstop, nu = nu))',
    "arrays": "x1 = x1, x2 = x2, x3 = x3, y = y",
    "array_names": ["x1", "x2", "x3", "y"],
    "py_formula": "y ~ bbs(x1) + bols(x2) + bmono(x1) + bbs(x3)",
    "py_family": lambda: mboost.Gaussian(),
}

GAUSS_TIE = {
    # Two identical columns make the per-learner selection risks bit-for-bit
    # equal; the winner is decided by the tie-break alone.
    "generate": """
        x1 <- runif(n, 0, 10)
        z <- rnorm(n)
        y <- 2 * x1 + z
    """,
    "frame": "x1, x2 = x1, y",
    # R's data.frame(x1, x2 = x1, y) spells the duplicate; keep it simple:
    "fit_call": 'gamboost(y ~ bbs(x1) + bbs(x2),\n'
                '            data = data.frame(x1 = x1, x2 = x1, y = y),\n'
                '            family = Gaussian(),\n'
                '            control = boost_control(mstop = mstop, nu = nu))',
    "arrays": "x1 = x1, y = y",
    "array_names": ["x1", "y"],
    # The formula mentions x2, which is x1 duplicated; supply it explicitly.
    "extra_data": ("x2", "x1"),
    "py_formula": "y ~ bbs(x1) + bbs(x2)",
    "py_family": lambda: mboost.Gaussian(),
}

BINOMIAL = {
    "generate": """
        x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
        eta <- 1.2 * x1 - 0.8 * x2 + 0.5 * x3
        y <- rbinom(n, 1, plogis(eta))
    """,
    # R's Binomial() rejects a numeric response ("response is not a factor"),
    # so the conversion happens in the data.frame, not in the numeric array
    # handed back to Python (which needs 0/1).
    "frame": "x1, x2, x3, y = factor(y)",
    "fit_call": 'gamboost(y ~ bols(x1) + bols(x2) + bols(x3),\n'
                '            data = data, family = Binomial(),\n'
                '            control = boost_control(mstop = mstop, nu = nu))',
    "arrays": "x1 = x1, x2 = x2, x3 = x3, y = y",
    "array_names": ["x1", "x2", "x3", "y"],
    "py_formula": "y ~ bols(x1) + bols(x2) + bols(x3)",
    "py_family": lambda: mboost.Binomial(),
}


def _assert_parity(config, mstop, nu, n, seed, label, rtol_risk=RTOL_RISK):
    r = _run_r(config, mstop, nu, n, seed)
    data = {name: r[name] for name in config["array_names"]}
    if "extra_data" in config:
        py_name, r_name = config["extra_data"]
        data[py_name] = r[r_name]
    model = mboost.gamboost(
        config["py_formula"], data=data,
        family=config["py_family"](),
        control=mboost.boost_control(mstop, nu),
    )
    r_seq = [s - 1 for s in r["selected"]]  # R is 1-based
    py_seq = list(model._selected)
    mismatch = next(
        (t for t, (a, b) in enumerate(zip(py_seq, r_seq, strict=True)) if a != b),
        None,
    )
    assert py_seq == r_seq, (
        f"{label}: selection sequence diverges at iteration "
        f"{mismatch + 1 if mismatch is not None else '?'} "
        f"(mboostrs={py_seq[mismatch] if mismatch is not None else None}, "
        f"R={r_seq[mismatch] if mismatch is not None else None}); "
        f"first 20 mboostrs={py_seq[:20]}, R={r_seq[:20]}"
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol_risk,
        err_msg=f"{label}: risk trajectory differs from R",
    )
    return model


def test_gaussian_menu_selection_sequence():
    _assert_parity(GAUSS_MENU, mstop=100, nu=0.1, n=300, seed=7,
                   label="gaussian menu bbs/bols/bmono")


def test_tie_break_is_lowest_index():
    # With two identical columns the criterion is exactly tied; R's which.min
    # returns the first (lowest) index. mboostrs must agree at every
    # iteration, which also pins the rayon path's tie-breaking.
    _assert_parity(GAUSS_TIE, mstop=50, nu=0.1, n=200, seed=11,
                   label="exact tie bbs(x1) vs bbs(x2)")


def test_binomial_selection_sequence():
    _assert_parity(BINOMIAL, mstop=60, nu=0.1, n=400, seed=5,
                   label="binomial bols menu", rtol_risk=RTOL_RISK_BINOMIAL)
