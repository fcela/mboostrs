"""Regression tests: ``predict(which=, mstop=m)`` truncates the path.

The ``which=`` term values were read from the full-length boosting path:
``predict(which=..., mstop=30)`` on a model fitted for 100 steps returned
the term's contribution after all 100 steps instead of after 30, differing
from a refit with ``mstop=30`` by about 0.37 (max abs) for the brandom
term and 0.65 for the bmrf term on the fixtures below. The acceptance
contract is that the truncated ``which=`` prediction equals the ``which=``
term of a model refit with ``mstop=m``.
"""

import numpy as np

import mboostrs as mb
from mboostrs.learners import Bbs, Bmrf

FULL = 100
TRUNC = 30
TOL = dict(atol=1e-10, rtol=1e-10)


def _brandom_data():
    rng = np.random.default_rng(848)
    n = 150
    g = rng.choice(np.arange(1, 6), size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    x = rng.uniform(size=n)
    y = np.sin(2 * np.pi * x) + np.array([-1, 0.5, 1, -0.3, 0.2])[g - 1] \
        + rng.normal(scale=0.3, size=n)
    return {"x": x, "g": g, "y": y}


def _brandom_fit(data, mstop):
    return mb.gamboost(
        "y ~ bbs(x, df=4) + brandom(g, df=3)",
        data=data, control=mb.boost_control(mstop),
    )


def _chain_laplacian(n_levels):
    k = np.zeros((n_levels, n_levels), dtype=np.float64)
    for i in range(n_levels - 1):
        k[i, i + 1] = -1.0
        k[i + 1, i] = -1.0
        k[i, i] += 1.0
        k[i + 1, i + 1] += 1.0
    return k


def _bmrf_data():
    rng = np.random.RandomState(42)
    n = 200
    n_levels = 5
    region = rng.randint(0, n_levels, n).astype(np.float64)
    x1 = rng.uniform(0, 1, n)
    y = np.array([3.0 * r for r in region]) + np.sin(2 * np.pi * x1) \
        + rng.normal(0, 0.5, n)
    return {"y": y, "x1": x1, "region": region}, n_levels


def _bmrf_fit(data, n_levels, mstop):
    return mb.gamboost(
        "y ~ x1 + region",
        data=data,
        family=mb.Gaussian(),
        control=mb.boost_control(mstop, 0.1),
        learners=[Bbs("x1", df=4), Bmrf("region", K=_chain_laplacian(n_levels), df=4)],
    )


class TestBrandomWhichMstop:
    def test_brandom_term_matches_refit(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        refit = _brandom_fit(data, TRUNC)
        np.testing.assert_allclose(
            full.predict(which="g", mstop=TRUNC), refit.predict(which="g"), **TOL)

    def test_bbs_term_matches_refit(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        refit = _brandom_fit(data, TRUNC)
        np.testing.assert_allclose(
            full.predict(which="x", mstop=TRUNC), refit.predict(which="x"), **TOL)

    def test_matches_refit_on_newdata(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        refit = _brandom_fit(data, TRUNC)
        np.testing.assert_allclose(
            full.predict(newdata=data, which="g", mstop=TRUNC),
            refit.predict(newdata=data, which="g"), **TOL)

    def test_partial_effects_mstop_matches_refit(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        refit = _brandom_fit(data, TRUNC)
        _, effect = full.partial_effects(which="g", mstop=TRUNC)["g"]
        _, want = refit.partial_effects(which="g")["g"]
        np.testing.assert_allclose(effect, want, **TOL)

    def test_prediction_contributions_mstop_matches_refit(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        refit = _brandom_fit(data, TRUNC)
        np.testing.assert_allclose(
            full.prediction_contributions(which="g", mstop=TRUNC),
            refit.predict(which="g"), **TOL)


class TestBmrfWhichMstop:
    def test_bmrf_term_matches_refit(self):
        data, n_levels = _bmrf_data()
        full = _bmrf_fit(data, n_levels, FULL)
        refit = _bmrf_fit(data, n_levels, TRUNC)
        np.testing.assert_allclose(
            full.predict(which="region", mstop=TRUNC),
            refit.predict(which="region"), **TOL)

    def test_bbs_term_matches_refit(self):
        data, n_levels = _bmrf_data()
        full = _bmrf_fit(data, n_levels, FULL)
        refit = _bmrf_fit(data, n_levels, TRUNC)
        np.testing.assert_allclose(
            full.predict(which="x1", mstop=TRUNC),
            refit.predict(which="x1"), **TOL)


class TestUnchangedBehaviour:
    def test_which_without_mstop_is_full_path_term(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        # The untruncated which= terms plus the offset still reconstruct
        # the full prediction.
        total = np.asarray(full.predict(which="g"), dtype=float) \
            + np.asarray(full.predict(which="x"), dtype=float) \
            + full.offset_
        np.testing.assert_allclose(total, full.predict(), **TOL)

    def test_predict_without_which_truncation_unchanged(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        refit = _brandom_fit(data, TRUNC)
        np.testing.assert_allclose(full.predict(mstop=TRUNC), refit.predict(), **TOL)
        np.testing.assert_allclose(full.predict(), full.fitted_, **TOL)

    def test_cumsum_route_still_matches_refit(self):
        data = _brandom_data()
        full = _brandom_fit(data, FULL)
        refit = _brandom_fit(data, TRUNC)
        cumsum = full.predict(which="g", mstop=TRUNC, aggregate="cumsum")
        np.testing.assert_allclose(cumsum[-1], refit.predict(which="g"), **TOL)
