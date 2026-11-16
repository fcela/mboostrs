"""Python and R `gam_refit` call the same Rust core (`mboostrs_core::gam`).

On the same data and terms they must agree to rounding: any difference is a
binding building a different design, penalty, response or weight.
"""

import numpy as np
import pytest

import mboostrs as mb

pytestmark = pytest.mark.r_parity


def _r_refit(frame, formula, family, which, newdata):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # Namespace-qualified calls: attaching mboostrs would mask mboost for
    # every other R parity test sharing this R session.
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = frame
        ro.globalenv["nd"] = newdata
    out = ro.r(f"""
        fit <- mboostrs::gamboost({formula}, data = d, family = mboostrs::{family}(),
                                  control = mboostrs::boost_control(mstop = 300, nu = 0.1))
        r <- mboostrs::gam_refit(fit, which = "{which}")
        list(edf = unname(r$edf), sigma = r$sigma, pred = predict(r, nd),
             lambda = unlist(r$smoothing))
    """)
    return {name: np.asarray(out.rx2(name)) for name in ("edf", "sigma", "pred", "lambda")}


@pytest.mark.parametrize("family, formula, which", [
    ("Gaussian", "y ~ bbs(x1) + bbs(x2) + bols(x3)", "all"),
    ("Poisson", "y ~ bbs(x1) + bbs(x2)", "all"),
    ("Gaussian", "y ~ bbs(x1) + brandom(g)", "all"),
    ("Gaussian", "y ~ bspatial(x1, x2, knots = 6)", "all"),
])
def test_python_and_r_refits_agree(family, formula, which):
    import pandas as pd

    rng = np.random.default_rng(21)
    n = 500
    x1, x2, x3 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.normal(size=n)
    g = rng.integers(1, 16, n)
    eta = np.sin(2 * x1) + 0.5 * x2 ** 2 + 0.3 * x3 + rng.normal(0, 0.5, 15)[g - 1]
    y = rng.poisson(np.exp(0.3 * eta)).astype(float) if family == "Poisson" \
        else eta + rng.normal(0, 0.4, n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "g": g})
    newdata = frame.iloc[:20].copy()

    fam = getattr(mb, family)()
    fit = mb.gamboost(formula, data=frame, family=fam, control=mb.boost_control(300, 0.1))
    py = mb.gam_refit(fit, which=which)
    r = _r_refit(frame, formula, family, which, newdata)

    np.testing.assert_allclose(list(py.edf.values()), r["edf"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(py.sigma, r["sigma"], rtol=1e-10)
    np.testing.assert_allclose(py.predict(newdata), r["pred"], rtol=1e-8, atol=1e-8)


def test_python_and_r_gamlss_refits_agree():
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(8)
    n = 400
    frame = pd.DataFrame({"x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n)})
    frame["y"] = np.sin(6 * frame.x1) + np.exp(-0.7 + frame.x2) * rng.normal(size=n)
    formulas = {"mu": "y ~ bbs(x1) + bbs(x2)", "sigma": "y ~ bbs(x2)"}
    fit = mb.gamboostLSS(formulas, data=frame, families=mb.GaussianLSS(),
                         control=mb.boost_control(300, 0.1))
    py = mb.gamlss_refit(fit, which="all")
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = frame
    out = ro.r("""
        fit <- mboostrs::gamboostLSS(list(mu = y ~ bbs(x1) + bbs(x2), sigma = y ~ bbs(x2)),
                                     data = d, families = mboostrs::GaussianLSS(),
                                     control = mboostrs::boost_control(mstop = 300, nu = 0.1))
        r <- mboostrs::gamlss_refit(fit, which = "all")
        p <- predict(r, d[1:20, ], type = "response")
        list(edf = c(r$edf$mu, r$edf$sigma), mu = p$mu, sigma = p$sigma)
    """)
    np.testing.assert_allclose([*py.edf["mu"].values(), *py.edf["sigma"].values()],
                               np.asarray(out.rx2("edf")), rtol=1e-6, atol=1e-6)
    predicted = py.predict(frame.iloc[:20], type="response")
    np.testing.assert_allclose(predicted["mu"], np.asarray(out.rx2("mu")), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(predicted["sigma"], np.asarray(out.rx2("sigma")),
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("basis", ["bols(x)", "bbs(x, knots=6)"])
def test_numeric_by_refit_space_agrees_across_python_and_r(basis):
    import pandas as pd

    rng = np.random.default_rng(619215)
    n = 350
    x = rng.uniform(-1, 1, n)
    g = rng.uniform(.5, 2, n)
    frame = pd.DataFrame(dict(x=x, g=g, y=.4+(2+3*x)*g+rng.normal(0, .02, n)))
    formula = "y ~ " + basis[:-1] + ", by=g)"
    fit = mb.gamboost(formula, data=frame, control=mb.boost_control(mstop=5))
    py = mb.gam_refit(fit, which="all")
    r = _r_refit(frame, formula, "Gaussian", "all", frame.iloc[:20])
    np.testing.assert_allclose(py.predict(frame.iloc[:20]), r["pred"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(list(py.edf.values()), r["edf"], atol=1e-5, rtol=1e-5)
    assert np.mean((py.predict(frame)-frame.y.to_numpy())**2) < .001
