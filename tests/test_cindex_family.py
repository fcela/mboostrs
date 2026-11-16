"""Non-R smoke/contract tests for the Cindex (Uno's IPCW-weighted
concordance) survival family.

R-numerical parity lives in tests/test_glmboost_parity/test_cindex_bols.py
(Gate 1, requires --run-r-parity). The gradient-correctness safety net
(Gate 2) is a set of Rust-only unit tests: see
crates/mboostrs-core/src/families.rs,
cindex_ngradient_and_risk_match_r_anchor (a fixed R-verified anchor),
cindex_ngradient_matches_finite_difference_of_risk (finite-difference), and
cindex_min_time_censored_observation_gets_zero_gradient (structural).

Unlike Gehan, Cindex's offset is identically 0 in R (verified) -- no
centering deviation, and unlike Gehan/AUC, Cindex's weight domain is {0, 1}
(not unit-only), so subsampling-fold cvrisk() is supported.
"""

import numpy as np
import pytest
from mboostrs.families import Cindex

import mboostrs as mboost


def _survival_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 0.4 * x1 - 0.25 * x2
    time = rng.exponential(scale=np.exp(eta))
    event = (rng.uniform(size=n) < 0.7).astype(np.float64)
    return {"time": time, "event": event, "x1": x1, "x2": x2}


def test_cindex_importable():
    assert Cindex is not None
    assert "Cindex" in mboost.__all__


def test_cindex_default_sigma():
    fam = Cindex()
    assert fam.sigma == 0.1


def test_glmboost_cindex_fits_and_is_finite():
    data = _survival_data()
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        control=mboost.boost_control(60, 0.1),
    )
    assert model.offset_ == 0.0
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))
    # risk = -sum(approxLoss * W) with approxLoss in [0, 1] and W normalized
    # to sum 1, so risk is always in [-1, 0].
    assert np.all(model.risk_ >= -1.0 - 1e-9)
    assert np.all(model.risk_ <= 1e-9)


def test_glmboost_cindex_risk_is_nonincreasing_ish():
    data = _survival_data()
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        control=mboost.boost_control(100, 0.1),
    )
    assert model.risk_[-1] <= model.risk_[0] + 1e-8


def test_glmboost_cindex_rejects_nonpositive_time():
    data = _survival_data()
    data = dict(data)
    data["time"] = data["time"].copy()
    data["time"][0] = 0.0
    with pytest.raises(ValueError):
        mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=data,
            family=Cindex(),
            control=mboost.boost_control(10, 0.1),
        )


def test_glmboost_cindex_rejects_non_zero_one_weights():
    data = _survival_data()
    weights = np.ones(len(data["time"]))
    weights[::4] = 2.0
    with pytest.raises(ValueError):
        mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=data,
            family=Cindex(),
            weights=weights,
            control=mboost.boost_control(10, 0.1),
        )


def test_glmboost_cindex_accepts_all_ones_weights():
    data = _survival_data()
    weights = np.ones(len(data["time"]))
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        weights=weights,
        control=mboost.boost_control(10, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))


def test_glmboost_cindex_accepts_zero_one_subset_weights():
    # {0,1} weights are a *supported* domain for Cindex (unlike Gehan/AUC's
    # unit-weight-only restriction): the IPCW fit runs on the w != 0 subset,
    # scattering 0 into the excluded positions.
    rng = np.random.default_rng(11)
    data = _survival_data(n=100, seed=2)
    weights = (rng.uniform(size=100) < 0.75).astype(np.float64)
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        weights=weights,
        control=mboost.boost_control(20, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))


def test_glmboost_cindex_cvrisk_subsampling_runs():
    # Cindex's {0,1} weight domain is exactly the subsampling-fold format,
    # so (unlike Gehan) cvrisk with subsampling folds is supported.
    data = _survival_data(n=100, seed=5)
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        control=mboost.boost_control(20, 0.1),
    )
    result = model.cvrisk(type="subsampling", B=5, prob=0.7)
    assert result.risk_matrix.shape[0] == 5
    assert np.all(np.isfinite(result.risk_matrix))


def test_glmboost_cindex_cvrisk_bootstrap_is_rejected():
    # Bootstrap folds carry integer resample counts (values > 1), which
    # violate Cindex's {0,1} weight domain -- must be rejected with a clear
    # error rather than silently double-counting a resampled observation.
    data = _survival_data(n=100, seed=5)
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        control=mboost.boost_control(20, 0.1),
    )
    with pytest.raises(ValueError):
        model.cvrisk(type="bootstrap", B=5)


def test_glmboost_cindex_sigma_threads_through():
    data = _survival_data(n=100, seed=9)
    model_default = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        control=mboost.boost_control(30, 0.1),
    )
    model_sharp = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(sigma=0.01),
        control=mboost.boost_control(30, 0.1),
    )
    assert np.all(np.isfinite(model_sharp.fitted_))
    # Different sigma must produce a genuinely different risk path (not a
    # silently-ignored parameter).
    assert not np.allclose(model_default.risk_, model_sharp.risk_)


def test_glmboost_cindex_offset_is_zero_matching_r():
    # Unlike Gehan, Cindex's offset() == 0 IS R's own value (verified), not
    # an intentional deviation -- so an explicit offset override shifts fitted
    # values exactly as it would for any offset-invariant risk.
    data = _survival_data(n=100, seed=7)
    control = mboost.boost_control(30, 0.1)
    model_default = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2", data=data, family=Cindex(), control=control
    )
    model_shifted = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        control=control,
        offset=5.0,
    )
    assert model_default.offset_ == 0.0
    assert model_shifted.offset_ == 5.0
    np.testing.assert_allclose(
        model_shifted.risk_, model_default.risk_, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        model_shifted.fitted_ - model_default.fitted_,
        np.full_like(model_default.fitted_, 5.0),
        rtol=1e-10,
        atol=1e-10,
    )
