"""R and Python agree on numeric by= for bns and bmono (issue #164); a tree's by= fits f(t) * z by least squares."""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.parametrize("formula", ["y ~ bns(x, by=z, df=4) + bbs(w)",
                                     "y ~ bmono(x, by=z) + bbs(w)"])
def test_r_matches_python(formula):
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(4)
    n = 300
    d = pd.DataFrame({"x": rng.uniform(size=n), "w": rng.uniform(size=n), "z": rng.normal(size=n)})
    d["y"] = np.sqrt(d.x) * d.z + np.sin(3 * d.w) + rng.normal(0, 0.2, n)
    nd = d.iloc[list(range(100, 111)) + list(range(280, 291))].copy()
    nd["z"] *= 2
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
        ro.globalenv["nd"] = ro.conversion.py2rpy(nd)
    r = ro.r(f"o <- mboostrs::gamboost({formula.replace('=', ' = ')}, data = d, "
             "control = mboostrs::boost_control(mstop = 100)); "
             "list(o$fitted, predict(o, newdata = nd))")
    fit = mb.gamboost(formula, data=d, control=mb.boost_control(100))
    np.testing.assert_allclose(np.asarray(r[0]), fit.fitted_, atol=1e-12)
    np.testing.assert_allclose(np.asarray(r[1]), fit.predict(newdata=nd), atol=1e-12)



def test_tree_by_fits_the_varying_coefficient_by_least_squares():
    """`btree(t, by=z)` fits f(t) * z: each leaf's value minimizes
    sum w (u - c z)^2, c = sum(w z u) / sum(w z^2). The tree was grown with
    weights w * z, negative wherever z is; on a one-split tree with a
    continuous slope the leaf values were not the least-squares ones."""
    rng = np.random.default_rng(1)
    n = 400
    t, z = rng.uniform(size=n), rng.normal(size=n)
    y = np.where(t > 0.5, 2.0, -1.0) * z + rng.normal(0, 0.1, n)
    fit = mb.gamboost("y ~ btree(t, by=z, maxdepth=1)", data={"y": y, "t": t, "z": z},
                      control=mb.boost_control(mstop=1, nu=1.0))
    u = y - fit.offset_
    contribution = fit.fitted_ - fit.offset_
    split = np.sort(np.unique(np.round(contribution / np.where(z == 0, 1, z), 10)))
    assert len(split) == 2
    leaf = np.isclose(contribution / z, split[1])
    for rows in (leaf, ~leaf):
        expected = (z[rows] * u[rows]).sum() / (z[rows] ** 2).sum()
        np.testing.assert_allclose(contribution[rows] / z[rows], expected, rtol=1e-10)
    assert abs(split[1] - 2.0) < 0.2 and abs(split[0] + 1.0) < 0.2


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
def test_r_tree_with_a_continuous_by_matches_python():
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(2)
    n = 300
    d = pd.DataFrame({"t": rng.uniform(size=n), "w": rng.uniform(size=n), "z": rng.normal(size=n)})
    d["y"] = np.where(d.t > 0.4, 1.5, -0.5) * d.z + np.sin(3 * d.w) + rng.normal(0, 0.2, n)
    nd = d.iloc[100:120].copy()
    nd["z"] *= 2
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["d"] = ro.conversion.py2rpy(d)
        ro.globalenv["nd"] = ro.conversion.py2rpy(nd)
    formula = "y ~ btree(t, by=z) + bbs(w)"
    r = ro.r(f"o <- mboostrs::gamboost({formula.replace('=', ' = ')}, data = d, "
             "control = mboostrs::boost_control(mstop = 80)); "
             "list(o$fitted, predict(o, newdata = nd))")
    fit = mb.gamboost(formula, data=d, control=mb.boost_control(80))
    np.testing.assert_allclose(np.asarray(r[0]), fit.fitted_, atol=1e-12)
    np.testing.assert_allclose(np.asarray(r[1]), fit.predict(newdata=nd), atol=1e-12)
