"""A weak main effect is reconsidered after a strong interaction is fitted."""
import numpy as np
import polars as pl

import mboostrs as mb


def test_dropped_main_effect_can_return_after_the_dominant_interaction():
    # Regression for axis_xor draw2 in the frozen mechanism study; this is
    # development evidence, not an independent policy confirmation draw.
    rng = np.random.default_rng(202610172)
    x = rng.uniform(-1, 1, (2800, 5))
    y = 3*np.sign(x[:, 0])*np.sign(x[:, 1])+x[:, 2]+rng.normal(0, .5, 2800)
    data = pl.DataFrame({name: x[:, i] for i, name in enumerate(["x1", "x2", "x3", "n0", "n1"])})
    data = data.with_columns(pl.Series("y", y))
    result = mb.discover(data.head(800), "y", folds=5, seed=2, max_mstop=1600)
    assert any(e["variables"] == ["x3"] for e in result.mean_effects)
    recovery = [r for r in result.log if r["stage"] == "mean recovery" and r["adopted"]]
    assert any("x3" in row["candidate"] and row["change"] < -2*row["se"] for row in recovery)
    prediction = result.model.predict(data.tail(2000).drop("y"), type="response", extrapolation="clip")
    # Discovery may adopt a location-scale model; this check targets its mean.
    mean = prediction["mu"] if isinstance(prediction, dict) else prediction
    assert np.mean((mean-y[800:])**2) < .7
