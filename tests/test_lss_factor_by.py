"""by= in Python gamboostLSS (issue #176).

A factor by= was refused by the LSS frontends; it is now lowered per
parameter as gamboost lowers it (a buser term over the stacked per-level
design, whose recipe prediction replays). Numeric by= already worked. All four
models match gamboostLSS in sample and on new data, as the R frontend does
(mboostrs/tests/test_lss_by_parity.R).
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

CASES = [
    ("y ~ bbs(x, by=z) + bols(w)", "y ~ bols(w)"),
    ("y ~ bols(x, by=z) + bbs(w)", "y ~ bbs(w)"),
    ("y ~ bbs(x, by=g, df=6) + bols(w)", "y ~ bols(w)"),
    ("y ~ bols(x, by=g) + bols(w)", "y ~ bols(x, by=g, df=2)"),
]


def _data(n=400, seed=2):
    rng = np.random.default_rng(seed)
    d = {"x": rng.uniform(size=n), "z": rng.normal(size=n), "w": rng.uniform(size=n),
         "g": rng.choice(["a", "b", "c"], n)}
    d["y"] = rng.normal(np.sin(3 * d["x"]) * d["z"] + (d["g"] == "b"), np.exp(0.3 * d["w"]))
    return d


def _links(model, newdata=None):
    p = model.predict(newdata=newdata, type="link")
    return np.concatenate([np.asarray(p[k]) for k in ("mu", "sigma")])


@pytest.mark.parametrize("mu,sigma", CASES)
def test_fits_and_replays(mu, sigma):
    d = _data()
    fit = mb.gamboostLSS({"mu": mu, "sigma": sigma}, data=d, families=mb.GaussianLSS(),
                         control=mb.boost_control(60))
    np.testing.assert_allclose(_links(fit, d), _links(fit), atol=1e-10)


def test_both_parameters_can_carry_a_factor_by():
    d = _data()
    fit = mb.gamboostLSS({"mu": "y ~ bols(x, by=g)", "sigma": "y ~ bols(w, by=g)"},
                         data=d, families=mb.GaussianLSS(), control=mb.boost_control(60))
    np.testing.assert_allclose(_links(fit, d), _links(fit), atol=1e-10)


def test_cvrisk_and_stabsel_run_with_a_factor_by():
    d = _data(n=300)
    f = {"mu": "y ~ bbs(x, by=g, df=6) + bols(w)", "sigma": "y ~ bols(w)"}
    cv = mb.cvrisk_gamboostLSS(f, data=d, families=mb.GaussianLSS(),
                               control=mb.boost_control(40), folds=mb.cv_weights(300, "kfold", 3))
    assert np.all(np.isfinite(np.asarray(cv.risk_matrix, dtype=float)))


def test_glmboostlss_still_refuses_a_factor_by():
    with pytest.raises(NotImplementedError, match="factor by="):
        mb.glmboostLSS({"mu": "y ~ bols(x, by=g)", "sigma": "y ~ w"}, data=_data(),
                       families=mb.GaussianLSS(), control=mb.boost_control(10))


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("mu,sigma", CASES)
def test_matches_gamboostlss(mu, sigma):
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    d = pd.DataFrame(_data())
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
    upstream = ro.r(
        "suppressMessages(library(gamboostLSS)); d$g <- factor(d$g); "
        f"m <- gamboostLSS(list(mu = {mu}, sigma = {sigma}), data = d, "
        "families = GaussianLSS(), control = boost_control(mstop = 100)); "
        "nd <- d[c(201:211, 391:400), ]; nd$z <- 2 * nd$z; "
        "list(unlist(predict(m, type = 'link')), "
        "unlist(predict(m, newdata = nd, type = 'link')))")
    fit = mb.gamboostLSS({"mu": mu, "sigma": sigma}, data=d, families=mb.GaussianLSS(),
                         control=mb.boost_control(100))
    np.testing.assert_allclose(_links(fit), np.asarray(upstream[0]), atol=1e-10)
    new = d.iloc[list(range(200, 211)) + list(range(390, 400))].copy()
    new["z"] = 2 * new["z"]
    np.testing.assert_allclose(_links(fit, new), np.asarray(upstream[1]), atol=1e-10)
