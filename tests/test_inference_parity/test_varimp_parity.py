"""R-parity tests for varimp() — variable importance."""

import json
import subprocess

import mboostrs as mboost
import numpy as np
import pytest


def _run_r(script):
    # Skip only when Rscript is genuinely absent; a non-zero exit or a hang is a
    # real failure (these are r_parity tests, run only when R is expected), so we
    # must not turn them into silent skips. JSON parse errors propagate too.
    try:
        result = subprocess.run(
            ["Rscript", "-e", script],
            capture_output=True, text=True, timeout=60
        )
    except FileNotFoundError:
        pytest.skip("Rscript not available")
    assert result.returncode == 0, f"R script failed (exit {result.returncode}):\n{result.stderr}"
    return json.loads(result.stdout.strip())


R_VARIMP_GLM_SCRIPT = r"""
library(mboost)
set.seed(42)
n <- 200
x1 <- runif(n)
x2 <- runif(n)
x3 <- runif(n)
y <- 2*x1 - 1*x2 + rnorm(n, sd=0.5)
data <- data.frame(y=y, x1=x1, x2=x2, x3=x3)

mod <- glmboost(y ~ x1 + x2 + x3, data=data, control=boost_control(mstop=100, nu=0.1), family=Gaussian())
vi <- varimp(mod)

result <- list(
    reduction = as.numeric(vi),
    names = names(vi),
    selfreqs = attr(vi, "selfreqs"),
    var_names = as.character(attr(vi, "variable_names")),
    # Return the data so Python fits on the SAME observations. R's runif and
    # numpy's uniform produce different sequences for the same seed, so without
    # this the two models would be fit on different data and could never match
    # selection frequencies exactly.
    x1 = x1, x2 = x2, x3 = x3, y = y
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA))  # digits=NA: full f64 precision so Python fits on identical data
"""


@pytest.mark.r_parity
def test_varimp_glmboost_selfreqs_r_parity():
    """Selection frequencies match R exactly when fit on the SAME data.

    Selection frequency = (times learner selected) / mstop. With identical data,
    identical learner selection (verified elsewhere) implies identical counts, so
    the frequencies match to machine precision -- hence atol=1e-12, not the old
    atol=0.05 (which was masking the fact that R and Python were fit on different
    runif/uniform sequences).
    """
    r_out = _run_r(R_VARIMP_GLM_SCRIPT)
    data = {
        "y": np.asarray(r_out["y"], float),
        "x1": np.asarray(r_out["x1"], float),
        "x2": np.asarray(r_out["x2"], float),
        "x3": np.asarray(r_out["x3"], float),
    }

    ctrl = mboost.boost_control(mstop=100, nu=0.1)
    mod = mboost.glmboost("y ~ x1 + x2 + x3", data=data, control=ctrl, family=mboost.Gaussian())
    vi = mod.varimp(percent=False)

    # R names are "(Intercept)", "x1", "x2", "x3"; selfreqs[i] is the frequency
    # for learner i.
    r_names = r_out["names"]
    r_selfreqs = np.array(r_out["selfreqs"])

    for i, name in enumerate(r_names):
        r_freq = r_selfreqs[i]
        py_freq = vi["selfreq"].get(name, 0.0)
        np.testing.assert_allclose(py_freq, r_freq, atol=1e-12,
                                   err_msg=f"Selection frequency mismatch for {name}")


@pytest.mark.r_parity
def test_varimp_glmboost_ordering_r_parity():
    """The ordering of variables by importance should match R (same data)."""
    # Fit on the same data R used (see R script note) so the comparison is real,
    # not a coincidence of two similar-but-different datasets.
    r_out = _run_r(R_VARIMP_GLM_SCRIPT)
    data = {
        "y": np.asarray(r_out["y"], float),
        "x1": np.asarray(r_out["x1"], float),
        "x2": np.asarray(r_out["x2"], float),
        "x3": np.asarray(r_out["x3"], float),
    }

    ctrl = mboost.boost_control(mstop=100, nu=0.1)
    mod = mboost.glmboost("y ~ x1 + x2 + x3", data=data, control=ctrl, family=mboost.Gaussian())
    vi = mod.varimp(percent=False)

    r_names = r_out["names"]
    r_reduction = np.array(r_out["reduction"])
    r_order = [r_names[i] for i in np.argsort(-r_reduction) if r_reduction[i] != 0.0]

    py_order = sorted(vi["reduction"].keys(), key=lambda k: -vi["reduction"][k])
    py_order = [k for k in py_order if vi["reduction"][k] != 0.0]

    assert py_order == r_order, f"Importance ordering mismatch: {py_order} vs {r_order}"


@pytest.mark.r_parity
def test_varimp_gamboost_ordering_r_parity():
    R_SCRIPT = r"""
    library(mboost)
    set.seed(42)
    n <- 200
    x1 <- runif(n)
    x2 <- runif(n)
    y <- 3*sin(2*pi*x1) + rnorm(n, sd=0.5)
    data <- data.frame(y=y, x1=x1, x2=x2)

    mod <- gamboost(y ~ bbs(x1) + bbs(x2), data=data, control=boost_control(mstop=50, nu=0.1), family=Gaussian())
    vi <- varimp(mod)

    result <- list(
        reduction = as.numeric(vi),
        names = names(vi),
        selfreqs = attr(vi, "selfreqs"),
        var_names = as.character(attr(vi, "variable_names")),
        # Share data with Python (R runif != numpy uniform for the same seed).
        x1 = x1, x2 = x2, y = y
    )
    cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA))  # digits=NA: full f64 precision so Python fits on identical data
    """
    r_out = _run_r(R_SCRIPT)

    data = {
        "y": np.asarray(r_out["y"], float),
        "x1": np.asarray(r_out["x1"], float),
        "x2": np.asarray(r_out["x2"], float),
    }

    ctrl = mboost.boost_control(mstop=50, nu=0.1)
    mod = mboost.gamboost("y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data, control=ctrl, family=mboost.Gaussian())
    vi = mod.varimp(percent=False)

    r_names = r_out["names"]
    r_reduction = np.array(r_out["reduction"])
    r_max_name = r_names[int(np.argmax(r_reduction))]

    py_max_name = max(vi["reduction"], key=vi["reduction"].get)
    assert "x1" in py_max_name, f"Expected x1 dominant, got {py_max_name}"
    assert "x1" in r_max_name, f"R expected x1 dominant, got {r_max_name}"


def test_varimp_percent_mode():
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(size=n)
    x2 = np.random.uniform(size=n)
    y = 2*x1 + 0.5*x2 + np.random.normal(size=n, scale=0.5)
    data = {"y": y, "x1": x1, "x2": x2}

    ctrl = mboost.boost_control(mstop=50, nu=0.1)
    mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())

    vi_abs = mod.varimp(percent=False)
    vi_pct = mod.varimp(percent=True)

    total_abs = sum(abs(v) for v in vi_abs["reduction"].values())
    for name in vi_pct["reduction"]:
        np.testing.assert_allclose(
            vi_pct["reduction"][name],
            vi_abs["reduction"][name] / total_abs if total_abs > 0 else 0.0,
            rtol=1e-10,
        )


def test_varimp_selfreq_sums_to_one():
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(size=n)
    x2 = np.random.uniform(size=n)
    y = x1 + x2 + np.random.normal(size=n, scale=0.5)
    data = {"y": y, "x1": x1, "x2": x2}

    ctrl = mboost.boost_control(mstop=50, nu=0.1)
    mod = mboost.glmboost("y ~ x1 + x2", data=data, control=ctrl, family=mboost.Gaussian())
    vi = mod.varimp()

    total_freq = sum(vi["selfreq"].values())
    np.testing.assert_allclose(total_freq, 1.0, rtol=1e-10)
