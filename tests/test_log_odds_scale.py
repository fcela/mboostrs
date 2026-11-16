"""`type="log_odds"` / `scale="log_odds"`: Binomial effects on the log-odds.

mboost's Binomial link is half the log-odds, so an effect read off
`predict(which=)` or `partial_effects` is half what another library reports.
"""

import numpy as np
import pytest

import mboostrs as mb


def _binary(n=2000, slope=1.5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-slope * x))
    return {"y": (rng.uniform(size=n) < p).astype(float), "x": x}


def test_glmboost_log_odds_recovers_the_logistic_slope():
    data = _binary()
    fit = mb.glmboost("y ~ x", data=data, family=mb.Binomial(),
                      control=mb.boost_control(1000, 0.1))
    x = np.array([0.0, 1.0])
    effect = np.asarray(fit.predict(newdata={"x": x}, type="log_odds", which="x"))
    assert effect.ravel()[1] - effect.ravel()[0] == pytest.approx(1.5, abs=0.2)
    assert np.allclose(fit.predict(type="log_odds"), 2.0 * fit.predict())


def test_gamboost_partial_effects_on_the_log_odds_scale():
    fit = mb.gamboost("y ~ bbs(x)", data=_binary(), family=mb.Binomial(),
                      control=mb.boost_control(300, 0.1))
    link = fit.partial_effects(which="x")["x"][1]
    log_odds = fit.partial_effects(which="x", scale="log_odds")["x"][1]
    assert np.allclose(log_odds, 2.0 * np.asarray(link))
    assert np.allclose(fit.predict(type="log_odds"), 2.0 * fit.predict())


def test_log_odds_is_refused_outside_binomial():
    fit = mb.gamboost("y ~ bbs(x)", data=_binary())
    with pytest.raises(ValueError, match="defined for Binomial"):
        fit.predict(type="log_odds")
    with pytest.raises(ValueError, match="defined for Binomial"):
        fit.partial_effects(scale="log_odds")
