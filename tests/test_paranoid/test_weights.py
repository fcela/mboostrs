"""Weight edge case tests: all-ones, zero weights, etc."""

import numpy as np
from numpy.testing import assert_allclose

import mboostrs as mboost

rng = np.random.default_rng(20250609)
n = 50
x1 = rng.normal(2.0, 3.0, n)
x2 = rng.normal(-1.0, 2.0, n)
y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
data = {"y": y, "x1": x1, "x2": x2}


def test_weights_all_ones_matches_none():
    """All-ones weights should give same result as no weights."""
    model_none = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    model_ones = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=np.ones(n),
        control=mboost.boost_control(50, 0.1),
    )
    c_none = model_none.coef()
    c_ones = model_ones.coef()
    for name in c_none:
        assert_allclose(c_ones[name], c_none[name], atol=1e-12,
                        err_msg=f"coef {name} differs with all-ones weights")
    assert_allclose(model_ones.fitted(), model_none.fitted(), atol=1e-12)
    assert_allclose(model_ones.risk_, model_none.risk_, atol=1e-12)


def test_rescaling_fractional_weights_changes_nothing():
    """Under the default policy, a global rescaling of analytic weights is inert.

    Fractional weights are read as analytic weights, where only the ratios
    carry meaning, so ``w`` and ``2 * w`` are rescaled to the same vector and
    give the same fitted values *and* the same risk. Upstream settles this in
    ``mboost_fit`` with ``rescale_weights``; the risk agreeing is the part
    that distinguishes the policy from passing the raw vector through, where
    the risk would scale with the weights.
    """
    w = rng.uniform(0.5, 3.0, n)
    model_w = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=w,
        control=mboost.boost_control(50, 0.1),
    )
    model_2w = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=2.0 * w,
        control=mboost.boost_control(50, 0.1),
    )
    assert_allclose(model_2w.fitted(), model_w.fitted(), atol=1e-12,
                    err_msg="rescaled analytic weights: fitted should match")
    assert_allclose(model_2w.risk_, model_w.risk_, atol=1e-10,
                    err_msg="rescaled analytic weights: risk should match too")
    assert_allclose(model_w._weights.sum(), float(n), atol=1e-10,
                    err_msg="rescaled weights sum to the positive-weight count")


def test_raw_policy_keeps_the_weight_scale():
    """``weight_policy="raw"`` passes the caller's scale through.

    The risk is a weighted sum, so doubling the weights doubles it, while the
    fitted values are unchanged because only the ratios enter the normal
    equations. This is what the default did before the upstream policy was
    adopted, and is kept as a named extension for callers whose weight scale
    is itself meaningful.
    """
    w = rng.uniform(0.5, 3.0, n)
    model_w = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=w, weight_policy="raw",
        control=mboost.boost_control(50, 0.1),
    )
    model_2w = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=2.0 * w, weight_policy="raw",
        control=mboost.boost_control(50, 0.1),
    )
    assert_allclose(model_2w.fitted(), model_w.fitted(), atol=1e-12,
                    err_msg="raw scaled weights: fitted should match")
    assert_allclose(model_2w.risk_, 2.0 * model_w.risk_, atol=1e-10,
                    err_msg="raw scaled weights: risk should scale by 2x")


def test_integer_weights_keep_their_scale():
    """Integer weights are frequencies, so their scale is not rescaled away.

    Weight 3 stands for three identical rows, and a fit of three copies has
    three times the risk of a fit of one. Doubling integer weights therefore
    doubles the risk under the default policy, where doubling fractional
    weights does nothing.
    """
    w = rng.integers(1, 4, n).astype(float)
    model_w = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=w, control=mboost.boost_control(50, 0.1),
    )
    model_2w = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=2.0 * w, control=mboost.boost_control(50, 0.1),
    )
    assert_allclose(model_w._weights, w, atol=1e-12,
                    err_msg="integer weights should reach the fit unchanged")
    assert_allclose(model_2w.fitted(), model_w.fitted(), atol=1e-12)
    assert_allclose(model_2w.risk_, 2.0 * model_w.risk_, atol=1e-10,
                    err_msg="integer frequency weights: risk scales with them")


def test_rescaling_fractional_weights_changes_nothing_gamboost():
    """Same for GAMBoost (bbs Cholesky path gives ~4e-5 drift in risk)."""
    model_w = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mboost.Gaussian(), weights=np.full(n, 1.5),
        control=mboost.boost_control(50, 0.1),
    )
    model_2w = mboost.gamboost(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        family=mboost.Gaussian(), weights=np.full(n, 3.5),
        control=mboost.boost_control(50, 0.1),
    )
    assert_allclose(model_2w.fitted(), model_w.fitted(), atol=1e-5,
                    err_msg="GAM rescaled weights: fitted should match (bbs numerics ~1e-6)")
    assert_allclose(model_2w.risk_, model_w.risk_, atol=5e-5,
                    err_msg="GAM rescaled weights: risk should match too")


def test_zero_weight_observation():
    """Zero-weight observations should not affect fit much."""
    rng2 = np.random.default_rng(20250610)
    n2 = 30
    x = rng2.normal(size=n2)
    y = 0.5 * x + rng2.normal(scale=0.3, size=n2)
    data2 = {"y": y, "x": x}
    w = np.ones(n2)
    w[5] = 0.0
    model_zw = mboost.glmboost(
        "y ~ x", data=data2, family=mboost.Gaussian(),
        weights=w,
        control=mboost.boost_control(50, 0.1),
    )
    f = model_zw.fitted()
    assert np.all(np.isfinite(f))
    c = model_zw.coef()
    assert all(np.isfinite(v) for v in c.values())


def test_weights_risk_decreasing():
    """With inbag risk, risk should be monotonically decreasing."""
    w = rng.uniform(0.5, 3.0, n)
    model = mboost.glmboost(
        "y ~ x1 + x2", data=data, family=mboost.Gaussian(),
        weights=w,
        control=mboost.boost_control(100, 0.1),
    )
    risk = np.asarray(model.risk_)
    assert np.all(np.diff(risk) <= 1e-12), "inbag risk not monotonically decreasing"
