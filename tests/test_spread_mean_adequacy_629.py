"""A spread win must survive an adequately fitted mean (issue #629).

The spread stage's screening comparison fits both specifications at the
screening fit's 200 steps, while the adopted model is fitted at its
tuned stopping. On a sharp interaction mean with constant scale, the
short mean fit leaves structure in the residuals that a log-scale term
absorbs, so the screening comparison adopts a varying scale that is not
there. Adoption therefore also requires the win in a confirmation that
tunes each specification's stopping inside the folds.
"""
import numpy as np
import polars as pl

import mboostrs as mb


def _sharp_draw():
    rng = np.random.default_rng(202610172)
    x = rng.uniform(-1, 1, (2800, 5))
    y = 3 * np.sign(x[:, 0]) * np.sign(x[:, 1]) + x[:, 2] + rng.normal(0, .5, 2800)
    data = pl.DataFrame({name: x[:, i] for i, name in enumerate(["x1", "x2", "x3", "n0", "n1"])})
    return data.with_columns(pl.Series("y", y)), y


def test_sharp_mean_counterexample_keeps_constant_scale():
    # The issue #629 counterexample: the true scale is the constant 0.5.
    # The screening comparison adopts bbs(x1, df=4) (change -0.452, se
    # 0.201); the confirmation must refuse it, and the mean recovery the
    # fixture exists for must be unchanged.
    data, y = _sharp_draw()
    result = mb.discover(data.head(800), "y", folds=5, seed=2, max_mstop=1600)
    assert result.spread_formula is None
    row = next(r for r in result.log
               if r["stage"] == "spread" and r["candidate"] == "sigma: bbs(x1, df=4)")
    assert not row["adopted"]
    # Refused at the screening comparison or at the confirmation; either way
    # the recorded change/se are no two-se win.
    assert "confirmation" in row["reason"] or "does not beat" in row["reason"]
    assert row["change"] > -2 * row["se"]
    assert any(e["variables"] == ["x3"] for e in result.mean_effects)
    prediction = result.model.predict(data.tail(2000).drop("y"), type="response",
                                      extrapolation="clip")
    mean = prediction["mu"] if isinstance(prediction, dict) else prediction
    assert np.mean((mean - y[800:]) ** 2) < .7


def test_genuine_varying_scale_survives_the_confirmation():
    # The #562 U-shaped scale: sd = 0.3 + 1.5*|x1| under a smooth mean.
    # The signal is real, so it clears the screening comparison and the
    # confirmation alike; the adopted row carries the confirmation's
    # change and se, which meet the two-se rule.
    rng = np.random.default_rng(0)
    x1, x2 = rng.uniform(-1, 1, (2, 1000))
    sd = .3 + 1.5 * np.abs(x1)
    y = np.sin(2 * np.pi * x1) + .5 * x2 + sd * rng.normal(size=1000)
    data = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    result = mb.discover(data, "y", seed=0)
    assert result.spread_formula == "bbs(x1, df=4)"
    selected = next(row for row in result.log
                    if row["stage"] == "spread" and row.get("adopted"))
    assert "confirmation" in selected["reason"]
    assert selected["change"] <= -2 * selected["se"]
