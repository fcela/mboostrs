"""Families added under route B, and one mislabelled family corrected.

Each family's risk was checked against `gamlss.dist` *before* the Rust was
written, and its gradient is pinned by a finite-difference test in
`lss.rs`. These tests cover the Python and R surfaces on top of that.
"""

import numpy as np
import pytest

import mboostrs as mb


def _fit(family, y, x, mstop=60):
    return mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"},
        data={"x": x, "y": y}, families=family,
        control=mb.boost_control(mstop, 0.1),
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    n = 400
    x = rng.normal(size=n)
    return {
        "x": x,
        "positive": rng.wald(np.exp(0.5 + 0.3 * x), 3.0),
        "real": 1 + 2 * x + rng.logistic(0, 0.6, n),
    }


class TestInverseGaussian:
    def test_fits_and_reduces_risk(self, data):
        fit = _fit(mb.InverseGaussianLSS(), data["positive"], data["x"])
        risk = np.asarray(fit.risk)
        assert risk[-1] < risk[0]
        assert np.all(np.isfinite(risk))

    def test_both_parameters_are_positive_on_the_response_scale(self, data):
        """Both links are log, so mu and sigma must come back strictly positive."""
        fit = _fit(mb.InverseGaussianLSS(), data["positive"], data["x"])
        for parameter in ("mu", "sigma"):
            values = np.asarray(fit.fitted(parameter=parameter, type="response"))
            assert np.all(values > 0), parameter

    def test_rejects_a_non_positive_response(self, data):
        y = data["positive"].copy()
        y[3] = 0.0
        with pytest.raises(Exception, match="positive|y must be"):
            _fit(mb.InverseGaussianLSS(), y, data["x"], mstop=5)

    def test_pdf_matches_the_closed_form(self):
        """Cross-checked against `gamlss.dist::dIG` when the family was written."""
        family = mb.InverseGaussianLSS()
        y = np.array([0.4, 1.3, 2.7, 0.9, 3.1])
        mu = np.array([1.0, 1.5, 2.0, 1.2, 2.5])
        sigma = np.array([0.8, 1.1, 0.9, 1.3, 0.7])
        density = family.pdf(y, {"mu": mu, "sigma": sigma})
        nll = -np.sum(np.log(density))
        assert nll == pytest.approx(7.0915263034, abs=1e-9)


class TestLogistic:
    def test_fits_and_reduces_risk(self, data):
        fit = _fit(mb.LogisticLSS(), data["real"], data["x"])
        risk = np.asarray(fit.risk)
        assert risk[-1] < risk[0]
        assert np.all(np.isfinite(risk))

    def test_mu_is_identity_and_sigma_is_log_linked(self, data):
        fit = _fit(mb.LogisticLSS(), data["real"], data["x"])
        mu_link = np.asarray(fit.fitted(parameter="mu", type="link"))
        mu_resp = np.asarray(fit.fitted(parameter="mu", type="response"))
        np.testing.assert_array_equal(mu_link, mu_resp)
        assert np.all(np.asarray(fit.fitted(parameter="sigma", type="response")) > 0)

    def test_accepts_an_unrestricted_response(self, data):
        y = data["real"].copy()
        y[0] = -1e3
        fit = _fit(mb.LogisticLSS(), y, data["x"], mstop=10)
        assert np.all(np.isfinite(np.asarray(fit.risk)))

    def test_pdf_matches_the_closed_form(self):
        """Cross-checked against `gamlss.dist::dLO`."""
        family = mb.LogisticLSS()
        y = np.array([0.4, -1.3, 2.7, 0.9, -3.1])
        mu = np.array([0.5, 0.2, 1.0, 0.7, -1.0])
        sigma = np.array([0.8, 1.1, 0.9, 1.3, 0.7])
        nll = -np.sum(np.log(family.pdf(y, {"mu": mu, "sigma": sigma})))
        assert nll == pytest.approx(9.5417107970, abs=1e-9)

    def test_heavier_tailed_than_gaussian_on_contaminated_data(self, data):
        """The reason to reach for it: a symmetric outlier should move the
        logistic location less than the Gaussian one."""
        from mboostrs.gamlss import GaussianLSS

        x = data["x"]
        y = data["real"].copy()
        clean_logistic = _fit(mb.LogisticLSS(), y, x)
        clean_gaussian = _fit(GaussianLSS(), y, x)
        y_dirty = y.copy()
        y_dirty[:5] += 40.0
        shift_logistic = np.abs(
            np.asarray(_fit(mb.LogisticLSS(), y_dirty, x).fitted(parameter="mu"))
            - np.asarray(clean_logistic.fitted(parameter="mu"))
        ).mean()
        shift_gaussian = np.abs(
            np.asarray(_fit(GaussianLSS(), y_dirty, x).fitted(parameter="mu"))
            - np.asarray(clean_gaussian.fitted(parameter="mu"))
        ).mean()
        assert shift_logistic < shift_gaussian


def test_gumbel_is_gamlss_rg_not_gu():
    """The Rust and Python docs both claimed `gamlss.dist::GU()` parity. The
    family is `RG()` — gamlss's naming inverts the common one, and its `RG` is
    the right-skewed form usually meant by "Gumbel". Measured:
    -sum(dRG(...)) = 9.042800226406863, -sum(dGU(...)) = 9.161943."""
    from mboostrs.gamlss import GumbelLSS

    assert "RG()" in GumbelLSS.__doc__
    assert "not" in GumbelLSS.__doc__.lower() and "GU()" in GumbelLSS.__doc__


@pytest.mark.parametrize("name", ["InverseGaussianLSS", "LogisticLSS"])
def test_the_new_families_are_exported_and_contracted(name):
    import json
    from pathlib import Path

    assert hasattr(mb, name) and name in mb.__all__
    manifest = json.loads(
        (Path(mb.__file__).parent / "capability_manifest.json").read_text()
    )
    row = next(r for r in manifest["tables"]["families"] if r["name"] == name)
    assert row["fit"] == "yes"
    assert name in manifest["public_exports"]["python"]
    assert name in manifest["public_exports"]["r"]


class TestZeroAdjustedGamma:
    """Semicontinuous data: exact zeros plus a right-skewed positive part."""

    @staticmethod
    def _semicontinuous(seed=9, n=500):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        p_zero = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * x)))
        is_zero = rng.random(n) < p_zero
        y = np.where(is_zero, 0.0, rng.gamma(3.0, np.exp(0.4 + 0.3 * x) / 3.0))
        return x, y, is_zero, p_zero

    def _fit(self, x, y, mstop=80):
        return mb.gamboostLSS(
            {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)",
             "nu": "y ~ bbs(x, df=4)"},
            data={"x": x, "y": y}, families=mb.ZeroAdjustedGammaLSS(),
            control=mb.boost_control(mstop, 0.1),
        )

    def test_nu_recovers_the_zero_mechanism(self):
        """The reason to use this family rather than a Gamma with the zeros
        dropped: `nu` is a fitted predictor of *whether* the outcome occurs."""
        x, y, is_zero, p_zero = self._semicontinuous()
        fit = self._fit(x, y)
        nu = np.asarray(fit.fitted(parameter="nu", type="response"))
        assert np.corrcoef(nu, p_zero)[0, 1] > 0.9
        assert nu.mean() == pytest.approx(is_zero.mean(), abs=0.05)

    def test_links_hold_on_the_response_scale(self):
        x, y, _, _ = self._semicontinuous()
        fit = self._fit(x, y, mstop=40)
        for parameter in ("mu", "sigma"):
            assert np.all(np.asarray(fit.fitted(parameter=parameter, type="response")) > 0)
        nu = np.asarray(fit.fitted(parameter="nu", type="response"))
        assert np.all((nu > 0.0) & (nu < 1.0))

    def test_risk_decreases(self):
        x, y, _, _ = self._semicontinuous()
        risk = np.asarray(self._fit(x, y).risk)
        assert risk[-1] < risk[0]
        assert np.all(np.isfinite(risk))

    def test_an_all_zero_response_is_refused(self):
        """The Gamma part would have no data to identify mu and sigma from."""
        x = np.random.default_rng(0).normal(size=60)
        with pytest.raises(Exception, match="all zeros"):
            self._fit(x, np.zeros(60), mstop=5)

    def test_a_negative_response_is_refused(self):
        x, y, _, _ = self._semicontinuous(n=60)
        y = y.copy()
        y[0] = -1.0
        with pytest.raises(Exception, match="non-negative|y must be"):
            self._fit(x, y, mstop=5)

    def test_pdf_matches_the_closed_form(self):
        """Cross-checked against `gamlss.dist::dZAGA` when the family was
        written: -sum(log f) = 10.6635430491 on this point."""
        pytest.importorskip("scipy")
        family = mb.ZeroAdjustedGammaLSS()
        y = np.array([0.0, 1.3, 0.0, 0.9, 3.1, 2.2])
        params = {
            "mu": np.array([1.0, 1.5, 2.0, 1.2, 2.5, 1.8]),
            "sigma": np.array([0.8, 1.1, 0.9, 1.3, 0.7, 1.0]),
            "nu": np.array([0.2, 0.3, 0.25, 0.4, 0.15, 0.35]),
        }
        nll = -np.sum(np.log(family.pdf(y, params)))
        assert nll == pytest.approx(10.6635430491, abs=1e-9)
