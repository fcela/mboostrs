"""Hurdle matches mboost::Hurdle(): fits, risk path, dispersion and cvrisk."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r(
        """suppressMessages(library(mboost)); set.seed(3); n <- 600
        d <- data.frame(x = runif(n, -1, 1), z = rnorm(n))
        d$y <- rnbinom(n, mu = exp(1 + sin(2*d$x) + 0.3*d$z), size = 1.5)
        d <- d[d$y > 0, ]"""
    )
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(ro.r("d"))


@pytest.mark.r_parity
@pytest.mark.parametrize("fit,rhs", [("glmboost", "x + z"), ("gamboost", "bbs(x) + bols(z)")],
                         ids=["glmboost", "gamboost"])
def test_hurdle_matches_mboost(data, fit, rhs):
    import rpy2.robjects as ro

    ro.r(f"m <- {fit}(y ~ {rhs}, data = d, family = Hurdle(), "
         "control = boost_control(mstop = 100))")
    model = getattr(mb, fit)(f"y ~ {rhs}", data=data, family=mb.Hurdle(),
                             control=mb.boost_control(100))
    np.testing.assert_allclose(model._fitted, np.asarray(ro.r("as.numeric(fitted(m))")),
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(model._risk, np.asarray(ro.r("as.numeric(m$risk())")),
                               rtol=1e-12, atol=1e-8)
    assert float(np.ravel(model.nuisance())[0]) == pytest.approx(
        float(ro.r("nuisance(m)")[0]), abs=1e-9)
    ro.r("set.seed(1); F <- cv(model.weights(m), type = 'kfold', B = 5)")
    folds = np.asarray(ro.r("F"))
    upstream = np.asarray(ro.r("colMeans(cvrisk(m, folds = F))"))
    np.testing.assert_allclose(model.cvrisk(folds=folds).aggregate, upstream, rtol=0, atol=1e-9)


def test_hurdle_refuses_a_non_integer_response():
    with pytest.raises(ValueError, match="whole counts"):
        mb.glmboost("y ~ x", {"x": np.arange(10.0), "y": np.arange(10.0) + 1.5},
                    family=mb.Hurdle())
