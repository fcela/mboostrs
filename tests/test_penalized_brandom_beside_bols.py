"""A penalized brandom beside a bols selects the block bols form.

A ridge on level indicators shrinks the constant direction too, so the
residual mean drifts while `brandom(g)` is selected, and a centered `bols(w)`
(one weight-centered column, constant in the offset) cannot follow it:
`brandom(g) + bols(w)` sat 7e-4 from mboost. The df of `brandom(g, by=z)` is
calibrated on the design scaled by `z`; calibrating on the bare indicators
left it 5e-3 from mboost.
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

FORMULAS = [
    "y ~ brandom(g) + bols(w)",
    "y ~ brandom(g, by=z)",
    "y ~ brandom(g, by=z, df=2) + bbs(w)",
    "y ~ brandom(g, by=z) + bols(w)",
    "y ~ bols(g) + bols(w)",
]


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("formula", FORMULAS)
def test_matches_mboost(formula):
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(4)
    n = 300
    d = pd.DataFrame({"x": rng.uniform(size=n), "w": rng.uniform(size=n),
                      "z": rng.normal(size=n), "g": rng.choice(list("abcdef"), n)})
    d["y"] = np.sqrt(d.x) * d.z * (d.g == "b") + d.w + rng.normal(0, 1, n)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
    upstream = np.asarray(ro.r(
        "suppressMessages(library(mboost)); d$g <- factor(d$g); "
        f"as.numeric(fitted(mboost::gamboost({formula.replace('=', ' = ')}, data = d, "
        "control = mboost::boost_control(mstop = 50))))"))
    fit = mb.gamboost(formula, data=d, control=mb.boost_control(50))
    np.testing.assert_allclose(fit.fitted_, upstream, atol=1e-12)


def test_menu_form():
    from mboostrs.gamboost import _bols_form_for
    from mboostrs.learners import Bols, Brandom

    assert _bols_form_for([Brandom("g"), Bols("w")]) == "block"
    assert _bols_form_for([Brandom("g", lambda_value=0.0), Bols("w")]) == "centered"
