"""Non-Gaussian and LSS fits are unchanged by the BlockBols sparse products (#323).

The data are built so that the `bbs` learners (about four non-zeros per
row) and the three-level factor `bols(g)` (one non-zero per row) take the
sparse route, and the two-level factor `bols(h)` takes the dense route. The
reference values were produced by the build before the column-compressed
transpose product replaced the row-wise scatter; the fits must reproduce
them to 1e-10 relative. The selected learners must match exactly.
"""

import numpy as np
import pytest

import mboostrs as mb

RTOL = 1e-10
MSTOP = 40
RHS = "bbs(x, df=4) + bols(z) + bols(g) + bols(h)"


def _data(n=4000, seed=323):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    z = rng.uniform(-2.0, 2.0, n)
    g = rng.choice(["a", "b", "c", "d"], n)
    h = rng.choice(["u", "v"], n)
    eta = np.sin(x / 2.0) + 0.5 * z + 0.8 * (g == "b") - 0.6 * (g == "d") + 0.4 * (h == "v")
    y_bin = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    y_gauss = eta + rng.normal(0.0, 0.5 + 0.2 * np.abs(z), n)
    return {"x": x, "z": z, "g": g, "h": h, "y_bin": y_bin, "y_gauss": y_gauss}


@pytest.fixture(scope="module")
def data():
    return _data()


def test_binomial_gamboost_matches_the_pre_change_fit(data):
    binary = {k: v for k, v in data.items() if k != "y_gauss"}
    binary["y"] = binary.pop("y_bin")
    fit = mb.gamboost("y ~ " + RHS, binary, family=mb.Binomial(),
                      control=mb.boost_control(mstop=MSTOP))

    assert list(fit.selected)[:6] == ["x", "x", "x", "x", "bols(g)", "z"]
    assert list(fit.selected).count("bols(h)") == 3
    np.testing.assert_allclose(fit.risk_[0], 2694.0609772489224, rtol=RTOL)
    np.testing.assert_allclose(fit.risk_[-1], 2314.5264121943446, rtol=RTOL)
    np.testing.assert_allclose(
        np.sum(fit.predict(type="link")), 875.7306214834487, rtol=RTOL)


def test_gaussian_lss_matches_the_pre_change_fit(data):
    wage = {k: v for k, v in data.items() if k != "y_bin"}
    wage["log_wage"] = wage.pop("y_gauss")
    spec = {"mu": "log_wage ~ " + RHS, "sigma": "log_wage ~ " + RHS}
    fit = mb.gamboostLSS(spec, wage, family=mb.GaussianLSS(),
                         control=mb.boost_control(mstop=MSTOP))

    assert list(fit.selected["mu"])[:6] == ["x", "x", "x", "z", "x", "z"]
    assert list(fit.selected["sigma"])[:6] == ["x", "bols(g)", "bols(g)", "bols(g)", "x", "bols(g)"]
    np.testing.assert_allclose(fit.risk_[0], 6527.225716345903, rtol=RTOL)
    np.testing.assert_allclose(fit.risk_[-1], 4643.718809232899, rtol=RTOL)
    pred = fit.predict(newdata=wage, type="response")
    np.testing.assert_allclose(np.sum(pred["mu"]), 1637.9349032562322, rtol=RTOL)
    np.testing.assert_allclose(np.sum(pred["sigma"]), 3201.806462954775, rtol=RTOL)
