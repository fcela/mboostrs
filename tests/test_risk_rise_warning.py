"""A fixed step too large for the loss's curvature is reported (issue #94).

On 300 Poisson rows with mean count about 3 and nu = 0.5, the training risk
rose on 56 of 100 steps, by up to 2.3 times, and the fit returned silently.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import _rust
from mboostrs.families import _family_params, _family_rust_name


@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=300)
    return {"x": x, "y": rng.poisson(3 * np.exp(x)).astype(float)}


def _messages(model):
    return [str(w.message) for w in model.fit_warnings]


def test_oscillating_poisson_path_warns(counts):
    with pytest.warns(RuntimeWarning, match="training risk rose"):
        fit = mb.gamboost("y ~ bbs(x)", data=counts, family=mb.Poisson(),
                          control=mb.boost_control(100, 0.5))
    assert any("step='line_search'" in m for m in _messages(fit))


@pytest.mark.parametrize("control", [
    mb.boost_control(100, 0.1),
    mb.boost_control(100, 0.5, step="line_search"),
])
def test_descending_paths_are_quiet(counts, control):
    fit = mb.gamboost("y ~ bbs(x)", data=counts, family=mb.Poisson(), control=control)
    assert not any("training risk rose" in m for m in _messages(fit))


def test_glmboost_checks_too(counts, monkeypatch):
    # A linear Poisson path goes from descending straight to a non-finite
    # risk (which already raises), so the check is exercised directly.
    monkeypatch.setattr(_rust, "risk_path_rises", lambda *args: (5, 0.5, 0.02, 10))
    with pytest.warns(RuntimeWarning, match="rose on 5 of 30"):
        mb.glmboost("y ~ x", data=counts, family=mb.Poisson(),
                    control=mb.boost_control(30, 0.1))


def test_adaptive_huber_is_not_judged():
    family = mb.Huber()
    assert _rust.risk_path_rises(_family_rust_name(family), _family_params(family),
                                 [1.0, 2.0, 3.0]) is None


@pytest.mark.parametrize("family", [
    mb.NBinomial(), mb.Hurdle(), mb.Weibull(), mb.Loglog(), mb.Lognormal(),
    mb.PropOdds(n_levels=3),
])
def test_profiled_risk_reports_small_rises_that_accumulate(family):
    args = (_family_rust_name(family), _family_params(family))
    count, largest, final_above_best, best = _rust.risk_path_rises(
        *args, [1000.0, 900.0, 900.5, 901.1])
    assert count == 2 and best == 1
    assert largest < 0.01 and final_above_best > 0.001
    assert _rust.risk_path_rises(*args, [1000.0, 900.0, 900.5]) is None
    assert _rust.risk_path_rises(*args, [1000.0, 900.0, 901.1, 890.0]) is None


@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_nbinomial_warning_keeps_the_fixed_fit_and_names_the_minimum(kind):
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    mu = 200 * np.exp(0.5 * x)
    data = {"x": x, "y": rng.negative_binomial(20, 20 / (20 + mu)).astype(float)}
    formula = "y ~ x" if kind == "glm" else "y ~ bols(x)"
    fit = mb.glmboost if kind == "glm" else mb.gamboost
    with pytest.warns(RuntimeWarning, match=r"training risk rose.*minimum at step"):
        unstable = fit(formula, data, family=mb.NBinomial(), control=mb.boost_control(400, .3))
    assert unstable.control.step == "fixed" and unstable.mstop == 400
    assert unstable.risk_[-1] > np.min(unstable.risk_) * 1.001
    messages = _messages(unstable)
    assert any(f"minimum at step {int(np.argmin(unstable.risk_))}" in message for message in messages)
    stable = fit(formula, data, family=mb.NBinomial(),
                 control=mb.boost_control(400, .3, step="line_search"))
    assert not any("training risk rose" in message for message in _messages(stable))
    assert np.max(np.diff(stable.risk_)) < 1e-7
