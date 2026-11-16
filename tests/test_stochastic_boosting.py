"""Stochastic component (base-learner) subsampling.

At each iteration only a random fraction of candidate learners compete for
selection. This reduces per-iteration cost on wide-p models and perturbs
selection frequencies. It is off by default, seeded for reproducibility, and
reported. R mboost has no equivalent, so there is no parity oracle — the tests
are determinism, off-by-default equivalence, and the statistical effect.
"""

import numpy as np
import pytest

import mboostrs as mb


def _wide_linear(n=1000, p=40, seed=0):
    rng = np.random.default_rng(seed)
    d = {f"x{j}": rng.normal(size=n) for j in range(p)}
    # only x0, x1, x2 carry signal
    d["y"] = 1.5 * d["x0"] - 0.9 * d["x1"] + 0.6 * d["x2"] + rng.normal(scale=0.3, size=n)
    return d, " + ".join(f"x{j}" for j in range(p))


def _wide_smooth(n=1000, q=8, seed=1):
    rng = np.random.default_rng(seed)
    d = {f"z{j}": rng.uniform(-2, 2, n) for j in range(q)}
    d["y"] = np.sin(d["z0"]) + 0.5 * d["z1"] + rng.normal(scale=0.3, size=n)
    return d, " + ".join(f"bbs(z{j}, df=4)" for j in range(q))


def test_off_by_default():
    data, rhs = _wide_linear()
    fit = mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(200, 0.1))
    assert fit.execution_report().component_subsample_iterations == 0


def test_same_seed_is_deterministic():
    data, rhs = _wide_linear()
    kw = dict(family=mb.Gaussian(), control=mb.boost_control(300, 0.1),
              subsample=0.3, subsample_seed=42)
    a = mb.glmboost(f"y ~ {rhs}", data=data, **kw)
    b = mb.glmboost(f"y ~ {rhs}", data=data, **kw)
    np.testing.assert_array_equal(a.fitted_, b.fitted_)
    assert a.selected == b.selected


def test_different_seed_changes_the_path():
    data, rhs = _wide_linear()
    ctrl = mb.boost_control(300, 0.1)
    a = mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(), control=ctrl,
                    subsample=0.3, subsample_seed=1)
    b = mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(), control=ctrl,
                    subsample=0.3, subsample_seed=2)
    assert a.selected != b.selected


def test_subsampled_fit_still_recovers_the_signal():
    """Perturbing which candidates compete must not lose the real predictors."""
    data, rhs = _wide_linear()
    fit = mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(400, 0.1), subsample=0.3, subsample_seed=7)
    coefs = fit.coefficients_
    top3 = sorted(coefs, key=lambda k: -abs(coefs[k]))[:3]
    assert set(top3) == {"x0", "x1", "x2"}


def test_reported_iteration_count_matches_mstop():
    data, rhs = _wide_linear()
    fit = mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(150, 0.1), subsample=0.5, subsample_seed=3)
    assert fit.execution_report().component_subsample_iterations == 150


def test_gram_route_falls_back_to_general_under_subsampling():
    """The Gaussian Gram fast path scans a fixed candidate set, so subsampling
    must route through the general engine instead."""
    data, rhs = _wide_smooth()
    plain = mb.gamboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                        control=mb.boost_control(100, 0.1))
    assert plain.execution_report().route == "gaussian_gram"

    subsampled = mb.gamboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                             control=mb.boost_control(100, 0.1),
                             subsample=0.5, subsample_seed=1)
    assert subsampled.execution_report().route == "general"
    assert subsampled.execution_report().component_subsample_iterations == 100


def test_gamboost_subsampling_is_deterministic():
    data, rhs = _wide_smooth()
    kw = dict(family=mb.Gaussian(), control=mb.boost_control(120, 0.1),
              subsample=0.5, subsample_seed=9)
    a = mb.gamboost(f"y ~ {rhs}", data=data, **kw)
    b = mb.gamboost(f"y ~ {rhs}", data=data, **kw)
    np.testing.assert_array_equal(a.fitted_, b.fitted_)


def test_subsampled_model_predicts_on_new_data():
    data, rhs = _wide_linear(n=600)
    fit = mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(200, 0.1), subsample=0.4, subsample_seed=5)
    newdata = {k: v[:10] for k, v in data.items() if k != "y"}
    pred = fit.predict(newdata=newdata, type="response")
    assert np.asarray(pred).shape == (10,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2])
def test_invalid_fraction_is_rejected(bad):
    data, rhs = _wide_linear(n=200)
    with pytest.raises(ValueError, match="component_fraction"):
        mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(20, 0.1), subsample=bad)


def test_subsampling_runs_faster_than_full_on_wide_p():
    """The point of the feature: fewer candidate evaluations per iteration.

    Timing is noisy, so this only asserts the subsampled fit is not
    dramatically slower — a weak but non-vacuous check that it isn't doing
    extra work."""
    import time

    data, rhs = _wide_linear(n=2000, p=100)
    # 1,000 iterations: at 200 a fit took about 10 ms, and a few milliseconds
    # of scheduling delay under `pytest -n 8` decided the comparison (issue
    # #162, again on 2026-10-03: 17.2 ms against 10.3 ms).
    ctrl = mb.boost_control(1000, 0.1)

    def once(**extra):
        t = time.perf_counter()
        mb.glmboost(f"y ~ {rhs}", data=data, family=mb.Gaussian(),
                    control=ctrl, **extra)
        return time.perf_counter() - t

    # Interleaved, so both arms see the same machine load (another pytest
    # worker); timed in two separate stretches, a burst of load during one
    # decided the comparison (issue #162).
    full = sub = float("inf")
    for _ in range(3):
        full = min(full, once())
        sub = min(sub, once(subsample=0.2, subsample_seed=1))
    assert sub < full * 1.5
