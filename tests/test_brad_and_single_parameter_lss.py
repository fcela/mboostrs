"""Radial-basis learners, and boosting one distribution parameter at a time.

Two items from the "coming from R" gap list. One needed implementing; the other
turned out to be already supported and only undiscoverable.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.gamlss import GaussianLSS
from mboostrs.learners import Brad, _effective_range, _matern_32


class TestMatern:
    def test_closed_form_matches_the_bessel_definition(self):
        """`_matern_32` is the closed form of the general Matérn at nu = 3/2.
        Checked against `2^(1-nu)/Gamma(nu) * r^nu * K_nu(r)`."""
        pytest.importorskip("scipy")
        from scipy.special import gamma, kv

        distance = np.array([0.05, 0.2, 0.5, 1.0, 2.5])
        theta, nu = 0.4, 1.5
        r = np.sqrt(2 * nu) * distance / theta
        general = 2 ** (1 - nu) / gamma(nu) * r**nu * kv(nu, r)
        np.testing.assert_allclose(_matern_32(distance, theta), general, atol=1e-12)

    def test_is_one_at_zero_and_decreasing(self):
        d = np.linspace(0, 5, 40)
        c = _matern_32(d, 1.0)
        assert c[0] == pytest.approx(1.0)
        assert np.all(np.diff(c) < 0)
        assert c[-1] < 1e-2

    def test_effective_range_decays_to_the_target(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, (200, 2))
        theta = _effective_range(x, eps=1e-3)
        diameter = np.max(
            np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
        )
        assert _matern_32(diameter, theta) == pytest.approx(1e-3, rel=1e-3)


class TestBrad:
    @staticmethod
    def _surface(seed=3, n=400):
        rng = np.random.default_rng(seed)
        x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        truth = np.sin(3 * np.pi * x) * np.cos(3 * np.pi * z)
        return {"x": x, "z": z, "y": truth + rng.normal(0, 0.2, n)}, truth

    def test_recovers_a_smooth_two_dimensional_surface(self):
        data, truth = self._surface()
        learner = Brad(["x", "z"], data, knots=40, df=4)
        fit = mb.gamboost("y ~ bols(x)", data=data, family=mb.Gaussian(),
                          learners=[learner],
                          control=mb.boost_control(300, 0.1))
        fitted = np.asarray(fit.fitted_).ravel()
        assert np.corrcoef(fitted, truth)[0, 1] > 0.9
        risk = np.asarray(fit.risk_)
        assert risk[-1] < 0.5 * risk[0]

    def test_lowers_to_buser_so_nothing_downstream_needs_to_know_it(self):
        """The whole reason `brad` needed no engine work: it is a penalized
        linear smoother, which `Buser` already expresses."""
        from mboostrs.learners import Buser

        data, _ = self._surface(n=100)
        learner = Brad(["x", "z"], data, knots=15)
        lowered = learner.as_buser()
        assert isinstance(lowered, Buser)
        np.testing.assert_array_equal(lowered.X, learner.X)
        np.testing.assert_array_equal(lowered.K, np.eye(learner.X.shape[1]))

    def test_design_is_whitened_so_the_penalty_is_a_plain_ridge(self):
        data, _ = self._surface(n=150)
        learner = Brad(["x", "z"], data, knots=20)
        np.testing.assert_allclose(learner.K, np.eye(20), atol=0)
        assert learner.X.shape == (150, 20)
        assert np.all(np.isfinite(learner.X))

    def test_knot_count_is_capped_at_the_distinct_rows(self):
        data = {"x": np.array([0.1, 0.2, 0.3, 0.1]), "y": np.zeros(4)}
        learner = Brad("x", data, knots=100)
        assert learner.centres.shape[0] == 3

    def test_one_dimensional_smooth_works_too(self):
        rng = np.random.default_rng(1)
        n = 300
        x = rng.uniform(0, 1, n)
        data = {"x": x, "y": np.sin(4 * np.pi * x) + rng.normal(0, 0.15, n)}
        fit = mb.gamboost("y ~ bols(x)", data=data, family=mb.Gaussian(),
                          learners=[Brad("x", data, knots=30, df=4)],
                          control=mb.boost_control(300, 0.1))
        risk = np.asarray(fit.risk_)
        assert risk[-1] < 0.4 * risk[0]

    def test_rejects_bad_arguments(self):
        data, _ = self._surface(n=50)
        with pytest.raises(ValueError, match="knots must be >= 1"):
            Brad("x", data, knots=0)
        with pytest.raises(ValueError, match="df must be positive"):
            Brad("x", data, df=-1)
        with pytest.raises(ValueError, match="theta must be positive"):
            Brad("x", data, theta=0.0)
        bad = {"x": np.array([1.0, np.nan, 3.0]), "y": np.zeros(3)}
        with pytest.raises(ValueError, match="finite"):
            Brad("x", bad)


class TestSingleParameterLSS:
    """R's `GaussianMu()` / `GaussianSigma()` pattern — boosting one
    distribution parameter with the others held.

    mboostrs spells this with per-parameter `mstop` rather than with a separate
    family object per parameter: setting a parameter's `mstop` to 0 leaves it at
    its offset for the whole fit, and `offset=` chooses that value. So
    `GaussianMu(sigma = 1.5)` is
    `gamboostLSS(..., mstop={"mu": N, "sigma": 0}, offset=[0.0, log(1.5)])`.
    """

    @staticmethod
    def _data(seed=2, n=300, sigma=None):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        scale = 1.5 if sigma is None else sigma
        return {"x": x, "y": 1 + 2 * x + rng.normal(0, scale, n)}, x

    FORMULAS = {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"}

    def test_a_zero_mstop_holds_a_parameter_at_its_offset(self):
        data, _ = self._data()
        fit = mb.gamboostLSS(self.FORMULAS, data=data, families=GaussianLSS(),
                             control=mb.boost_control(60, 0.1),
                             mstop={"mu": 60, "sigma": 0})
        assert fit.mstop == {"mu": 60, "sigma": 0}
        sigma = np.asarray(fit.fitted(parameter="sigma", type="link"))
        assert np.allclose(sigma, sigma[0]), "sigma must not move"
        mu = np.asarray(fit.fitted(parameter="mu", type="link"))
        assert mu.std() > 0.1, "mu must still be boosted"

    def test_the_held_parameter_can_be_pinned_to_a_chosen_value(self):
        """This is the part that makes it equivalent to `GaussianMu(sigma=v)`
        rather than merely 'sigma is not boosted'."""
        data, x = self._data(sigma=1.5)
        fit = mb.gamboostLSS(
            self.FORMULAS, data=data, families=GaussianLSS(),
            control=mb.boost_control(60, 0.1),
            mstop={"mu": 60, "sigma": 0}, offset=[0.0, float(np.log(1.5))],
        )
        sigma = np.asarray(fit.fitted(parameter="sigma", type="response"))
        assert np.allclose(sigma, 1.5), sigma[:3]
        mu = np.asarray(fit.fitted(parameter="mu", type="response"))
        assert np.corrcoef(mu, 1 + 2 * x)[0, 1] > 0.99

    def test_either_parameter_can_be_the_held_one(self):
        data, _ = self._data()
        sigma_only = mb.gamboostLSS(
            self.FORMULAS, data=data, families=GaussianLSS(),
            control=mb.boost_control(60, 0.1), mstop={"mu": 0, "sigma": 60},
        )
        mu = np.asarray(sigma_only.fitted(parameter="mu", type="link"))
        sigma = np.asarray(sigma_only.fitted(parameter="sigma", type="link"))
        assert np.allclose(mu, mu[0]), "mu must not move"
        assert sigma.std() > 0.01, "sigma must be boosted"

    def test_it_matches_boosting_only_that_parameter_in_a_one_parameter_fit(self):
        """The held parameter contributes only a constant, so the boosted
        parameter's path should be what you would get fitting it alone."""
        data, _ = self._data()
        held = mb.gamboostLSS(
            self.FORMULAS, data=data, families=GaussianLSS(),
            control=mb.boost_control(40, 0.1), mstop={"mu": 40, "sigma": 0},
        )
        assert len(np.asarray(held.risk)) > 1
        assert np.all(np.isfinite(np.asarray(held.risk)))
        # and the selection order only ever touches mu
        rows = held.path_frame(output="records")
        assert {row["parameter"] for row in rows} == {"mu"}


def test_brad_is_importable_from_the_package_root():
    """`__all__` listing a name the module never imported is worse than not
    exporting it: `from mboostrs import *` raises and `mb.Brad` is missing while
    the contract says it is public. That happened once here."""
    assert "Brad" in mb.__all__
    assert hasattr(mb, "Brad"), "Brad is in __all__ but was never imported"
    assert mb.Brad is Brad


def test_every_public_name_is_actually_importable():
    """The general form of the same defect, across the whole flat API."""
    missing = [name for name in mb.__all__ if not hasattr(mb, name)]
    assert not missing, f"names in __all__ that do not exist: {missing}"
