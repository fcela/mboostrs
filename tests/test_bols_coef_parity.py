"""coef() of a centered bols matches mboost's [intercept, slope]."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.mark.r_parity
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_centered_bols_reports_mboosts_intercept_and_slope(weighted):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r("""suppressMessages(library(mboost)); set.seed(7); n <- 250
        d <- data.frame(x = runif(n, -1, 1), z = rnorm(n))
        d$y <- sin(2 * d$x) + 0.5 * d$z + rnorm(n, sd = .4)
        w <- runif(n, 0.5, 2)""")
    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.rpy2py(ro.r("d"))
    weights = np.asarray(ro.r("w")) if weighted else None
    ro.r("m <- gamboost(y ~ bbs(x) + bols(z), data = d, "
         + ("weights = w, " if weighted else "")
         + "control = boost_control(mstop = 80))")
    model = mb.gamboost("y ~ bbs(x) + bols(z)", data=data, weights=weights,
                        control=mb.boost_control(80))
    np.testing.assert_allclose(model.coef()["z"], np.asarray(ro.r("coef(m)[[2]]")),
                               rtol=0, atol=1e-12)


def test_the_reported_intercept_is_the_centered_column_s_constant():
    rng = np.random.default_rng(2)
    n = 120
    data = {"z": rng.normal(1.0, 1.0, n)}
    data["y"] = 0.7 * data["z"] + rng.normal(size=n)
    model = mb.gamboost("y ~ bols(z)", data=data, control=mb.boost_control(50))
    intercept, slope = model.coef()["z"]
    assert intercept == pytest.approx(-slope * data["z"].mean())
    # The two numbers reproduce the fitted values with the offset.
    np.testing.assert_allclose(model.predict(),
                               model.coef()["(offset)"] + intercept + slope * data["z"])
