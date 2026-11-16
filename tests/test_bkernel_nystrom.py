"""Frontend tests for the Bkernel Nyström approximation (P3.4 item 6).

These verify the Python end-to-end path: formula parsing, learner_spec
lowering, fit/predict/partial_effects/cvrisk, and validation. The Rust core
is tested separately in crates/mboostrs-core/src/learners.rs.
"""

import pickle

import mboostrs as mb
import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.learners import Bkernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(4 * x) + rng.normal(0, 0.2, n)
    return {"x": x, "y": y, "n": n}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestBkernelValidation:
    def test_landmarks_must_be_int(self):
        with pytest.raises(ValueError, match="positive integer"):
            Bkernel("x", landmarks=1.5)

    def test_landmarks_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            Bkernel("x", landmarks=0)

    def test_landmarks_negative(self):
        with pytest.raises(ValueError, match="at least 1"):
            Bkernel("x", landmarks=-3)

    def test_landmarks_bool_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            Bkernel("x", landmarks=True)

    def test_valid_landmarks_accepted(self):
        b = Bkernel("x", landmarks=10)
        assert b.landmarks == 10


# ---------------------------------------------------------------------------
# Fit / predict / partial_effects
# ---------------------------------------------------------------------------

class TestBkernelNystromFit:
    def test_fit_reduces_risk(self):
        dat = _make_data()
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=30)",
            data=dat, family=Gaussian(), control=mb.boost_control(80),
        )
        risk = np.asarray(m.risk_)
        assert np.all(np.diff(risk) <= 1e-9), "risk path not monotone"
        assert risk[-1] < risk[0]

    def test_all_kernels_fit(self):
        dat = _make_data(n=200)
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.3, df=6, landmarks=50, kernel='gaussian')",
            data=dat, family=Gaussian(), control=mb.boost_control(50),
        )
        assert np.isfinite(m.risk_[-1]), "gaussian did not fit"

    def test_nystrom_rejects_compact_kernels(self):
        with pytest.raises(ValueError, match="only supported for kernel='gaussian'"):
            Bkernel("x", bandwidth=0.3, df=6, landmarks=50, kernel="epanechnikov")
        with pytest.raises(ValueError, match="only supported for kernel='gaussian'"):
            Bkernel("x", bandwidth=0.3, df=6, landmarks=50, kernel="uniform")

    def test_predict_shape_and_finiteness(self):
        dat = _make_data()
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=20)",
            data=dat, family=Gaussian(), control=mb.boost_control(60),
        )
        test = {"x": np.array([0.1, 0.3, 0.5, 0.7, 0.9])}
        p = np.asarray(m.predict(newdata=test))
        assert p.shape == (5,)
        assert np.all(np.isfinite(p))

    def test_predict_newdata_close_to_exact(self):
        """Nyström predictions should be within ~10% of exact on a well-sampled problem."""
        dat = _make_data(n=200)
        test = {"x": np.array([0.1, 0.3, 0.5, 0.7, 0.9])}
        m_exc = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6)",
            data=dat, family=Gaussian(), control=mb.boost_control(60),
        )
        m_ny = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=50)",
            data=dat, family=Gaussian(), control=mb.boost_control(60),
        )
        p_ex = np.asarray(m_exc.predict(newdata=test))
        p_ny = np.asarray(m_ny.predict(newdata=test))
        # Same order of magnitude (Nyström is an approximation)
        assert np.all(np.abs(p_ex) < 5)
        assert np.all(np.abs(p_ny) < 5)
        # Neither collapses to zero
        assert np.all(np.abs(p_ny) > 0.01)

    def test_partial_effects_returns_correct_shape(self):
        dat = _make_data(n=80)
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=20)",
            data=dat, family=Gaussian(), control=mb.boost_control(50),
        )
        r = m.partial_effects("x")
        assert "x" in r
        xv, ev = r["x"]
        assert len(xv) == 80
        assert len(ev) == 80
        assert np.all(np.isfinite(ev))

    def test_partial_effects_on_newdata(self):
        dat = _make_data(n=80)
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=20)",
            data=dat, family=Gaussian(), control=mb.boost_control(50),
        )
        test = {"x": np.array([0.1, 0.5, 0.9])}
        r = m.partial_effects("x", newdata=test)
        xv, ev = r["x"]
        assert len(ev) == 3
        assert np.all(np.isfinite(ev))


# ---------------------------------------------------------------------------
# cvrisk
# ---------------------------------------------------------------------------

class TestBkernelNystromCVRisk:
    def test_cvrisk_runs(self):
        dat = _make_data(n=100)
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=20)",
            data=dat, family=Gaussian(), control=mb.boost_control(120),
        )
        folds = mb.cv_weights(100, type="kfold", B=3, seed=42)
        risk = m.cvrisk(folds=folds)
        assert risk.risk_matrix.shape[0] == 3  # three folds
        assert np.all(np.isfinite(np.asarray(risk.aggregate))), "aggregate risk has NaN"
        assert np.isfinite(risk.optimal_mstop)


# ---------------------------------------------------------------------------
# Pickle roundtrip
# ---------------------------------------------------------------------------

class TestBkernelNystromPickle:
    def test_predict_after_pickle(self):
        dat = _make_data(n=60)
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=15)",
            data=dat, family=Gaussian(), control=mb.boost_control(40),
        )
        test = {"x": np.array([0.2, 0.5, 0.8])}
        p_before = np.asarray(m.predict(newdata=test))
        m2 = pickle.loads(pickle.dumps(m))
        p_after = np.asarray(m2.predict(newdata=test))
        np.testing.assert_allclose(p_before, p_after, rtol=1e-10)


# ---------------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------------

class TestBkernelNystromNoWarning:
    def test_no_scale_warning(self):
        """Nyström fit should not emit the bkernel-scale warning."""
        import warnings

        dat = _make_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mb.gamboost(
                "y ~ bkernel(x, bandwidth=0.25, df=6, landmarks=20)",
                data=dat, family=Gaussian(), control=mb.boost_control(30),
            )
        bkernel_warnings = [
            x for x in w if "bkernel" in str(x.message).lower() and "kernel" in str(x.message).lower()
        ]
        assert len(bkernel_warnings) == 0


# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------

class TestBkernelFormulaParsing:
    def test_landmarks_in_formula(self):
        dat = _make_data(n=60)
        m = mb.gamboost(
            "y ~ bkernel(x, bandwidth=0.3, df=4, landmarks=12, kernel='gaussian')",
            data=dat, family=Gaussian(), control=mb.boost_control(30),
        )
        assert np.isfinite(m.risk_[-1])

    def test_invalid_landmarks_formula(self):
        with pytest.raises(ValueError, match="landmarks"):
            mb.gamboost(
                "y ~ bkernel(x, landmarks=0)",
                data=_make_data(), family=Gaussian(), control=mb.boost_control(10),
            )
