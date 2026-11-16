"""Per-parameter LSS risk paths match gamboostLSS's risk(merge = FALSE/TRUE).

The non-cyclic cases also pin the update sequence: gamboostLSS scores a
candidate parameter with the gradient its own model last computed, which
predates the previous update unless that update was to the same parameter,
and refits the chosen parameter with its current gradient. Following it
removed a 1e-3 difference in a glmboostLSS non-cyclic fit.
"""

import numpy as np
import pytest

import mboostrs as mb


@pytest.mark.r_parity
@pytest.mark.parametrize("kind", ["glm", "gam"])
@pytest.mark.parametrize("method", ["cyclic", "noncyclic"])
def test_parameter_risk_matches_gamboostlss(kind, method):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r("""suppressMessages(library(gamboostLSS)); set.seed(5); n <- 300
        d <- data.frame(x = runif(n, -1, 1), z = rnorm(n))
        d$y <- rnorm(n, mean = sin(2 * d$x) + 0.5 * d$z, sd = exp(0.2 + 0.4 * d$x))""")
    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.rpy2py(ro.r("d"))
    rhs = "x + z" if kind == "glm" else "bbs(x) + bols(z)"
    fit = "glmboostLSS" if kind == "glm" else "gamboostLSS"
    ro.r(f"m <- {fit}(y ~ {rhs}, data = d, families = GaussianLSS(), "
         f"control = boost_control(mstop = 150), method = '{method}')")
    model = getattr(mb, fit)(f"y ~ {rhs}", data=data, families=mb.GaussianLSS(),
                             control=mb.boost_control(150), method=method)
    paths = model.parameter_risk()
    for name in ("mu", "sigma"):
        np.testing.assert_allclose(paths[name], np.asarray(ro.r(f"risk(m)${name}")),
                                   rtol=1e-12, atol=1e-9)
    values, labels = model.parameter_risk(merge=True)
    np.testing.assert_allclose(values, np.asarray(ro.r("as.numeric(risk(m, merge = TRUE))")),
                               rtol=1e-12, atol=1e-9)
    assert labels == list(ro.r("names(risk(m, merge = TRUE))"))
    np.testing.assert_allclose(model.predict(parameter="mu"),
                               np.asarray(ro.r("as.numeric(fitted(m$mu))")), atol=1e-10)


def test_the_last_parameter_path_of_a_cyclic_fit_is_the_cycle_risk():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 200)
    data = {"x": x, "y": rng.normal(np.sin(2 * x), np.exp(0.3 * x))}
    model = mb.glmboostLSS("y ~ x", data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(40))
    paths = model.parameter_risk()
    np.testing.assert_allclose(paths["sigma"], model.risk)
    assert paths["mu"][0] == paths["sigma"][0] == model.risk[0]
    assert len(paths["mu"]) == len(paths["sigma"]) == 41
