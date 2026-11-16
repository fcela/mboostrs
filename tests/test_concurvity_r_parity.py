"""Python and R `concurvity` call the same Rust core on the same fit.

On the same data and specification the two bindings must agree to
rounding: any difference is a binding rebuilding a different design or
handing the core a different boosting path. Term labels are binding-local
(the Python binding writes `bbs(x)` where the R binding writes `x`), so
they are compared after unwrapping the learner call.
"""

import re

import numpy as np
import pandas as pd
import pytest

import mboostrs as mb

pytestmark = pytest.mark.r_parity

rtol = 1e-9
atol = 1e-12


def _concurved_data(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.uniform(-1, 1, n)
    z = x + rng.normal(0, 0.05, n)
    w = rng.uniform(-1, 1, n)
    return pd.DataFrame({
        "x": x, "z": z, "w": w,
        "y": np.sin(2 * x) + w + rng.normal(0, 0.3, n),
    })


def _r_concurvity(frame, function, formula, pairwise):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # Namespace-qualified calls: attaching mboostrs would mask mboost for
    # every other R parity test sharing this R session.
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = frame
    suffix = "TRUE" if pairwise else "FALSE"
    out = ro.r(f"""
        fit <- mboostrs::{function}({formula}, data = d,
                                    family = mboostrs::Gaussian(),
                                    control = mboostrs::boost_control(mstop = 150, nu = 0.3))
        cc <- mboostrs::concurvity(fit, pairwise = {suffix})
        list(term = as.character(cc$term),
             against = if (is.null(cc$against)) character(0) else as.character(cc$against),
             worst = cc$worst, observed = cc$observed, estimate = cc$estimate)
    """)
    return {
        "term": list(out.rx2("term")),
        "against": list(out.rx2("against")),
        "worst": np.asarray(out.rx2("worst"), dtype=float),
        "observed": np.asarray(out.rx2("observed"), dtype=float),
        "estimate": np.asarray(out.rx2("estimate"), dtype=float),
    }


def _py_concurvity(frame, function, formula, pairwise):
    fit = getattr(mb, function)(formula, frame, family=mb.Gaussian(),
                                control=mb.boost_control(mstop=150, nu=0.3))
    rows = fit.concurvity(pairwise=pairwise, output="records")
    return {
        "term": [r["term"] for r in rows],
        "against": [r["against"] for r in rows] if pairwise else [],
        "worst": np.array([r["worst"] for r in rows], dtype=float),
        "observed": np.array([r["observed"] for r in rows], dtype=float),
        "estimate": np.array([r["estimate"] for r in rows], dtype=float),
    }


def _canonical(term):
    """`bbs(x)` and `x` name the same term in the two bindings."""
    m = re.match(r"^[a-z]+\((.*)\)$", term)
    return m.group(1) if m else term


def _assert_agree(r, py):
    assert [_canonical(t) for t in r["term"]] == [_canonical(t) for t in py["term"]]
    assert ([_canonical(t) for t in r["against"]] ==
            [_canonical(t) for t in py["against"]])
    for key in ("worst", "observed", "estimate"):
        r_vals, py_vals = r[key], py[key]
        assert r_vals.shape == py_vals.shape
        for rv, pv in zip(r_vals, py_vals, strict=True):
            if np.isnan(rv) or np.isnan(pv):
                assert np.isnan(rv) and np.isnan(pv)
            else:
                assert np.isclose(rv, pv, rtol=rtol, atol=atol), (key, rv, pv)


@pytest.mark.parametrize("pairwise", [False, True])
def test_gamboost_concurvity_agrees_between_python_and_r(pairwise):
    frame = _concurved_data()
    _assert_agree(
        _r_concurvity(frame, "gamboost", "y ~ bbs(x) + bbs(z) + bbs(w)", pairwise),
        _py_concurvity(frame, "gamboost", "y ~ bbs(x) + bbs(z) + bbs(w)", pairwise),
    )


@pytest.mark.parametrize("pairwise", [False, True])
def test_glmboost_concurvity_agrees_between_python_and_r(pairwise):
    frame = _concurved_data()
    _assert_agree(
        _r_concurvity(frame, "glmboost", "y ~ x + z + w", pairwise),
        _py_concurvity(frame, "glmboost", "y ~ x + z + w", pairwise),
    )
