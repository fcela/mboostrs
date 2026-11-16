"""glmboost(y ~ x - 1) against mboost (issue #101).

R's model.matrix drops the intercept column for `- 1`, so mboost offers no
constant learner while keeping the offset and centering the covariates. The
refusal this replaced suggested gamboost with bols() terms; that matches for a
Gaussian loss only (Binomial differed by 0.011).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("family", ["Gaussian", "Binomial"])
@pytest.mark.parametrize("rhs", ["x0 + x1 + x2 - 1", "0 + x0 + x1 + x2"])
def test_matches_mboost(family, rhs):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({f"x{j}": rng.normal(1, 1, n) for j in range(3)})
    d["y"] = 1.5 + 2 * d.x0 - d.x1 + rng.normal(0, 0.5, n)
    d["b"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(d.x0 - 1)))).astype(float)
    nd = pd.DataFrame({"x0": [0.0, 2.0], "x1": [1.0, -1.0], "x2": [0.0, 0.5]})
    folds = mb.cv_weights(n, "kfold", 5, seed=1)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"], ro.globalenv["nd"] = ro.conversion.py2rpy(d), ro.conversion.py2rpy(nd)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["f"] = folds
    response = "y" if family == "Gaussian" else "factor(b)"
    ro.r(f"m <- suppressWarnings(mboost::glmboost({response} ~ {rhs}, data = d, "
         f"family = mboost::{family}(), control = mboost::boost_control(mstop = 80)))")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.glmboost(f"{'y' if family == 'Gaussian' else 'b'} ~ {rhs}", data=d,
                          family=getattr(mb, family)(), control=mb.boost_control(80))
    np.testing.assert_allclose(fit.fitted_, np.asarray(ro.r("as.numeric(fitted(m))")), atol=1e-12)
    np.testing.assert_allclose(fit.predict(newdata=nd),
                               np.asarray(ro.r("as.numeric(predict(m, newdata = nd))")), atol=1e-12)
    upstream_coef = np.asarray(ro.r("suppressMessages(as.numeric(coef(m)))"))
    names = list(ro.r("names(suppressMessages(coef(m)))"))
    ours = fit.coef()
    assert "(Intercept)" not in ours
    np.testing.assert_allclose([ours[k] for k in names], upstream_coef, atol=1e-12)
    upstream_cv = np.asarray(ro.r("colMeans(mboost::cvrisk(m, folds = f, papply = lapply))"))
    np.testing.assert_allclose(np.asarray(fit.cvrisk(folds=folds).risk_matrix).mean(axis=0),
                               upstream_cv, atol=1e-12)


def test_lss_still_refuses():
    rng = np.random.default_rng(0)
    d = {"x": rng.normal(size=50), "y": rng.normal(size=50)}
    with pytest.raises(NotImplementedError, match="intercept control"):
        mb.glmboostLSS({"mu": "y ~ x - 1", "sigma": "y ~ x"}, data=d)


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("center", [True, False])
def test_cvrisk_keeps_the_fit_centering(center):
    """glmboost(center=False) fits the raw columns; cvrisk refitted every fold
    centered, 0.30 from mboost's cvrisk, and stability selection did too."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(3)
    n = 300
    d = pd.DataFrame({"x1": rng.normal(1, 1, n), "x2": rng.uniform(size=n), "x3": rng.normal(size=n)})
    d["y"] = 0.8 * d.x1 - 0.5 * d.x2 + rng.normal(size=n)
    folds = mb.cv_weights(n, "kfold", 5, seed=2)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["f"] = folds
    upstream = np.asarray(ro.r(
        f"m <- mboost::glmboost(y ~ x1 + x2 + x3, data = d, center = {str(center).upper()}, "
        "control = mboost::boost_control(mstop = 60)); "
        "colMeans(mboost::cvrisk(m, folds = f, papply = lapply))"))
    fit = mb.glmboost("y ~ x1 + x2 + x3", data=d, center=center, control=mb.boost_control(60))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = np.asarray(fit.cvrisk(folds=folds).risk_matrix).mean(axis=0)
    np.testing.assert_allclose(ours, upstream, atol=1e-12)
