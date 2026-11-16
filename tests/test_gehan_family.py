"""Non-R smoke/contract tests for the Gehan rank-loss AFT family.

R-numerical parity lives in
tests/test_glmboost_parity/test_gehan_bols.py (Gate 1, requires
--run-r-parity). The gradient-correctness safety net (Gate 2) is a Rust-only
finite-difference unit test: see crates/mboostrs-core/src/families.rs,
gehan_ngradient_matches_finite_difference_of_risk and
gehan_ngradient_sign_matches_aft_ranking_direction.

The Gehan offset is unidentifiable (the loss depends only on pairwise
residual differences, so a constant shift of every fitted value cancels);
mboostrs defines offset() == 0.0 rather than R's arbitrary
optimize()-on-a-flat-function value. Several tests below exercise this
end-to-end: an explicit non-zero `offset=` override must reproduce an
identical risk path and a fitted path shifted by exactly that constant.

Non-unit weights: Gehan now implements a principled *weighted* gradient
(see the `Gehan` doc comment in families.rs) instead of rejecting non-unit
weights -- a deliberate mboostrs extension, not R parity (R's own weighted
Gehan ngradient is broken the same way AUC's is: a compacted vector of
length sum(w), not n). The R-independent correctness oracle lives in Rust
(`gehan_ngradient_matches_finite_difference_of_weighted_risk`,
`gehan_zero_one_weights_match_unit_weight_subset`); the tests below just
check that non-unit weights are accepted end-to-end and that cvrisk/stabsel
-- now unblocked -- run and produce sane output.
"""

import numpy as np
import pytest
from mboostrs.families import Gehan

import mboostrs as mboost


def _survival_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 0.4 * x1 - 0.25 * x2
    time = rng.exponential(scale=np.exp(eta))
    event = (rng.uniform(size=n) < 0.7).astype(np.float64)
    return {"time": time, "event": event, "x1": x1, "x2": x2}


def test_gehan_importable():
    assert Gehan is not None
    assert "Gehan" in mboost.__all__


def test_glmboost_gehan_fits_and_is_finite():
    data = _survival_data()
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
        control=mboost.boost_control(60, 0.1),
    )
    assert model.offset_ == 0.0
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))
    # The Gehan risk is a sum of non-negative pairwise terms.
    assert np.all(model.risk_ >= -1e-12)


def test_glmboost_gehan_risk_is_nonincreasing_ish():
    # Not monotone in general for boosting, but the final risk should be no
    # worse than the risk at f == offset == 0 (the null model).
    data = _survival_data()
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
        control=mboost.boost_control(100, 0.1),
    )
    assert model.risk_[-1] <= model.risk_[0] + 1e-8


def test_glmboost_gehan_rejects_nonpositive_time():
    data = _survival_data()
    data = dict(data)
    data["time"] = data["time"].copy()
    data["time"][0] = 0.0
    with pytest.raises(ValueError):
        mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=data,
            family=Gehan(),
            control=mboost.boost_control(10, 0.1),
        )


def test_glmboost_gehan_accepts_all_ones_weights():
    # A weights vector that is all 1.0 (not the default None) must still be
    # accepted -- requires_unit_weights checks values, not "was a weights
    # argument passed at all".
    data = _survival_data()
    weights = np.ones(len(data["time"]))
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
        weights=weights,
        control=mboost.boost_control(10, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))


def test_glmboost_gehan_accepts_non_unit_weights_and_is_finite():
    # Bootstrap-style integer case weights (not just {0,1}) are now accepted
    # and produce a finite fit -- previously rejected via
    # requires_unit_weights. See the module docstring for where the
    # correctness oracle for this actually lives (Rust, R-independent).
    data = _survival_data()
    rng = np.random.default_rng(4)
    weights = rng.integers(0, 4, size=len(data["time"])).astype(np.float64)
    weights[weights == 0] = 1.0
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
        weights=weights,
        control=mboost.boost_control(30, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))


def test_glmboost_gehan_cvrisk_subsampling_smoke():
    # Gate B: cvrisk previously raised (requires_unit_weights == True); the
    # weighted gradient unblocks it. Subsampling folds are {0,1} weights.
    # Not compared to R -- there is no R parity target for weighted Gehan.
    data = _survival_data(n=150, seed=5)
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
        control=mboost.boost_control(40, 0.1),
    )
    cvr = model.cvrisk(type="subsampling", B=8, seed=1, parallel=False)
    assert cvr.risk_matrix.shape[0] == 8
    assert np.all(np.isfinite(cvr.risk_matrix))
    assert np.isfinite(cvr.optimal_mstop)
    assert cvr.optimal_mstop >= 1


def test_glmboost_gehan_cvrisk_bootstrap_smoke():
    # Gate B: bootstrap folds are integer case-weight counts (not just
    # {0,1}); the weighted gradient accepts any non-negative weight, unlike
    # R's broken weighted Gehan ngradient.
    data = _survival_data(n=150, seed=6)
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
        control=mboost.boost_control(40, 0.1),
    )
    cvr = model.cvrisk(type="bootstrap", B=8, seed=1, parallel=False)
    assert cvr.risk_matrix.shape[0] == 8
    assert np.all(np.isfinite(cvr.risk_matrix))
    assert np.isfinite(cvr.optimal_mstop)
    assert cvr.optimal_mstop >= 1


def test_glmboost_gehan_stabsel_prefers_signal_over_noise():
    # Gate B sanity property: stabsel (built on subsampling cvrisk, so also
    # previously blocked) should select the true-signal predictor with
    # higher probability than a pure-noise predictor. Not an R-parity
    # assertion -- just a sanity check that the weighted pipeline finds
    # signal.
    rng = np.random.default_rng(13)
    n = 200
    x1 = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    eta = 0.9 * x1
    time = rng.exponential(scale=np.exp(eta))
    event = (rng.uniform(size=n) < 0.8).astype(np.float64)
    data = {"time": time, "event": event, "x1": x1, "x_noise": x_noise}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x_noise",
        data=data,
        family=Gehan(),
        control=mboost.boost_control(60, 0.1),
    )
    sel = mboost.stabsel(model, q=1, PFER=2, B=30)
    assert np.all(np.isfinite(sel.max_))
    idx = {name: i for i, name in enumerate(sel.learner_names)}
    assert sel.max_[idx["x1"]] > sel.max_[idx["x_noise"]]


def test_glmboost_gehan_offset_invariance_end_to_end():
    # The core claim: the Gehan risk is flat in a constant offset. An
    # end-to-end fit with an explicit non-zero offset must produce a risk
    # path identical to the default (offset == 0) fit, and a fitted path
    # shifted by exactly that constant at every boosting iteration -- because
    # the gradient/selected/coef path is itself invariant to a uniform shift
    # of every prediction.
    data = _survival_data(n=100, seed=7)
    control = mboost.boost_control(40, 0.1)

    model_default = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2", data=data, family=Gehan(), control=control
    )
    model_shifted = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Gehan(),
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
    # Centered fitted values must therefore match exactly too.
    np.testing.assert_allclose(
        model_shifted.fitted_ - model_shifted.fitted_.mean(),
        model_default.fitted_ - model_default.fitted_.mean(),
        rtol=1e-10,
        atol=1e-10,
    )
