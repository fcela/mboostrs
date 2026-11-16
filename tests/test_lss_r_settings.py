"""R and Python LSS fits agree with stabilization and per-parameter nu (#68).

R used to ignore both (they were Python-only); the R family constructors now
take `stabilization=` and `boost_control(nu = c(mu = ..., sigma = ...))` sets
per-parameter steps, as upstream gamboostLSS does.
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

pytestmark = [pytest.mark.r_parity,
              pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")]


def _setup():
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(0)
    n = 200
    x, z = rng.normal(size=n), rng.normal(size=n)
    y = rng.normal(np.sin(x), np.exp(0.3 * z))
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["x"], ro.globalenv["z"], ro.globalenv["y"] = x, z, y
    ro.r("d <- data.frame(x = x, z = z, y = y); "
         "nd <- d[1:20, ]")
    # Not attached: library(mboostrs) would mask mboost's gamboost, cvrisk and
    # families for every later parity test in this R session.
    return ro, {"x": x, "z": z, "y": y}


def _stack(pred):
    return np.concatenate([np.asarray(pred["mu"]), np.asarray(pred["sigma"])])


@pytest.mark.parametrize("stabilization", ["none", "MAD", "L2"])
@pytest.mark.parametrize("method", ["cyclic", "noncyclic"])
def test_gamboostlss(stabilization, method):
    ro, d = _setup()
    r = ro.r(
        f'm <- mboostrs::gamboostLSS(list(mu = y ~ bbs(x) + bols(z), sigma = y ~ bols(z)), data = d, '
        f'families = mboostrs::GaussianLSS(stabilization = "{stabilization}"), method = "{method}", '
        'control = mboostrs::boost_control(mstop = 40, nu = c(sigma = 0.05, mu = 0.1))); '
        'p <- predict(m, type = "link"); q <- predict(m, newdata = nd, type = "link"); '
        'list(c(p$mu, p$sigma), c(q$mu, q$sigma))')
    py = mb.gamboostLSS({"mu": "y ~ bbs(x) + bols(z)", "sigma": "y ~ bols(z)"}, data=d,
                        families=mb.GaussianLSS(), stabilization=stabilization,
                        nu={"mu": 0.1, "sigma": 0.05}, control=mb.boost_control(40),
                        method=method)
    np.testing.assert_allclose(np.asarray(r[0]), _stack(py.predict(type="link")), atol=1e-12)
    new = {k: v[:20] for k, v in d.items()}
    np.testing.assert_allclose(np.asarray(r[1]), _stack(py.predict(newdata=new, type="link")),
                               atol=1e-12)


@pytest.mark.parametrize("stabilization", ["none", "MAD"])
def test_glmboostlss(stabilization):
    ro, d = _setup()
    r = ro.r(
        f'm <- mboostrs::glmboostLSS(y ~ x + z, data = d, families = mboostrs::GaussianLSS(stabilization = '
        f'"{stabilization}"), control = mboostrs::boost_control(mstop = 40, nu = c(0.1, 0.05))); '
        'q <- predict(m, newdata = nd, type = "link"); c(q$mu, q$sigma)')
    py = mb.glmboostLSS({"mu": "y ~ x + z", "sigma": "y ~ x + z"}, data=d,
                        families=mb.GaussianLSS(), stabilization=stabilization,
                        nu={"mu": 0.1, "sigma": 0.05}, control=mb.boost_control(40))
    new = {k: v[:20] for k, v in d.items()}
    np.testing.assert_allclose(np.asarray(r), _stack(py.predict(newdata=new, type="link")),
                               atol=1e-12)
