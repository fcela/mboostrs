"""Non-R smoke/contract tests for the AUC ranking family.

R-numerical parity lives in tests/test_glmboost_parity/test_auc_bols.py
(Gate 1, requires --run-r-parity). The gradient-correctness safety net
(Gate 2) is a Rust-only finite-difference unit test: see
crates/mboostrs-core/src/families.rs,
auc_ngradient_matches_finite_difference_of_unstandardized_loss.

Non-unit weights: Auc now implements a principled *weighted* gradient (see
the `Auc` doc comment in families.rs) instead of rejecting non-unit weights
-- this is a deliberate mboostrs extension, not R parity (R's own weighted
AUC ngradient is broken: it returns a compacted vector of length sum(w), not
n). The R-independent correctness oracle for the weighted gradient lives in
Rust (`auc_ngradient_matches_finite_difference_of_weighted_unstandardized_loss`,
`auc_zero_one_weights_match_unit_weight_subset`); the tests below just check
that non-unit weights are accepted end-to-end and that cvrisk/stabsel -- now
unblocked by the weighted gradient -- run and produce sane output.
"""

import numpy as np
import pytest
from mboostrs.families import AUC

import mboostrs as mboost


def _binary_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 0.6 * x1 - 0.4 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return {"y": y, "x1": x1, "x2": x2}


def test_auc_importable():
    assert AUC is not None
    assert "AUC" in mboost.__all__


def test_glmboost_auc_fits_and_is_finite():
    data = _binary_data()
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=AUC(),
        control=mboost.boost_control(60, 0.1),
    )
    assert model.offset_ == 0.0
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))
    # 1 - AUC is bounded in [0, 1].
    assert np.all(model.risk_ >= -1e-12)
    assert np.all(model.risk_ <= 1.0 + 1e-12)


def test_glmboost_auc_encode_response_pm1_matches_01():
    # {0,1} and {-1,+1} codings of the same two classes must produce the
    # same fit (Auc::encode_response is a min/max threshold, idempotent on
    # an already {-1,+1}-coded column).
    data01 = _binary_data()
    data_pm1 = dict(data01)
    data_pm1["y"] = np.where(data01["y"] > 0.5, 1.0, -1.0)

    model01 = mboost.glmboost(
        "y ~ x1 + x2", data=data01, family=AUC(), control=mboost.boost_control(40, 0.1)
    )
    model_pm1 = mboost.glmboost(
        "y ~ x1 + x2", data=data_pm1, family=AUC(), control=mboost.boost_control(40, 0.1)
    )
    np.testing.assert_allclose(model01.fitted_, model_pm1.fitted_, atol=1e-12)
    np.testing.assert_allclose(model01.risk_, model_pm1.risk_, atol=1e-12)


def test_glmboost_auc_rejects_single_class():
    data = _binary_data()
    data["y"] = np.ones_like(data["y"])
    with pytest.raises(ValueError):
        mboost.glmboost(
            "y ~ x1 + x2",
            data=data,
            family=AUC(),
            control=mboost.boost_control(10, 0.1),
        )


def test_glmboost_auc_accepts_all_ones_weights():
    # A weights vector that is all 1.0 (not the default None) must still be
    # accepted -- requires_unit_weights checks values, not "was a weights
    # argument passed at all".
    data = _binary_data()
    weights = np.ones(len(data["y"]))
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=AUC(),
        weights=weights,
        control=mboost.boost_control(10, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))


def test_glmboost_auc_accepts_non_unit_weights_and_is_finite():
    # Bootstrap-style integer case weights (not just {0,1}) are now accepted
    # and produce a finite fit -- previously rejected via
    # requires_unit_weights. See the module docstring for where the
    # correctness oracle for this actually lives (Rust, R-independent).
    data = _binary_data()
    rng = np.random.default_rng(3)
    weights = rng.integers(0, 4, size=len(data["y"])).astype(np.float64)
    weights[weights == 0] = 1.0
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=AUC(),
        weights=weights,
        control=mboost.boost_control(30, 0.1),
    )
    assert np.all(np.isfinite(model.fitted_))
    assert np.all(np.isfinite(model.risk_))


def test_glmboost_auc_cvrisk_subsampling_smoke():
    # Gate B: cvrisk previously raised (requires_unit_weights == True); the
    # weighted gradient unblocks it. Subsampling folds are {0,1} weights.
    # Not compared to R -- there is no R parity target for weighted AUC.
    data = _binary_data(n=150, seed=5)
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=AUC(), control=mboost.boost_control(40, 0.1)
    )
    cvr = model.cvrisk(type="subsampling", B=8, seed=1, parallel=False)
    assert cvr.risk_matrix.shape[0] == 8
    assert np.all(np.isfinite(cvr.risk_matrix))
    assert np.isfinite(cvr.optimal_mstop)
    assert cvr.optimal_mstop >= 1


def test_glmboost_auc_cvrisk_bootstrap_smoke():
    # Gate B: bootstrap folds are integer case-weight counts (not just
    # {0,1}); the weighted gradient accepts any non-negative weight, unlike
    # R's broken weighted AUC ngradient.
    data = _binary_data(n=150, seed=6)
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=AUC(), control=mboost.boost_control(40, 0.1)
    )
    cvr = model.cvrisk(type="bootstrap", B=8, seed=1, parallel=False)
    assert cvr.risk_matrix.shape[0] == 8
    assert np.all(np.isfinite(cvr.risk_matrix))
    assert np.isfinite(cvr.optimal_mstop)
    assert cvr.optimal_mstop >= 1


def test_glmboost_auc_stabsel_prefers_signal_over_noise():
    # Gate B sanity property: stabsel (built on subsampling cvrisk, so also
    # previously blocked) should select the true-signal predictor with
    # higher probability than a pure-noise predictor. Not an R-parity
    # assertion -- just a sanity check that the weighted pipeline finds
    # signal.
    rng = np.random.default_rng(11)
    n = 200
    x1 = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    eta = 1.5 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    data = {"y": y, "x1": x1, "x_noise": x_noise}
    model = mboost.glmboost(
        "y ~ x1 + x_noise", data=data, family=AUC(), control=mboost.boost_control(60, 0.1)
    )
    sel = mboost.stabsel(model, q=1, PFER=2, B=30)
    assert np.all(np.isfinite(sel.max_))
    idx = {name: i for i, name in enumerate(sel.learner_names)}
    assert sel.max_[idx["x1"]] > sel.max_[idx["x_noise"]]
