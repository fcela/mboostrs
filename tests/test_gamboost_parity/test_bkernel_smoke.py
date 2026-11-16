"""Smoke test for the bkernel base-learner.

NOTE: R mboost's `bkernel` lives in the kangar00 extension package and is a
genetic-pathway kernel ("lin"/"sia"/"net") with entirely different semantics
from mboostrs's `Bkernel` (a generic Gaussian/Epanechnikov/uniform kernel-ridge
smoother). There is therefore no R-mboost parity reference; this test only
verifies the learner fits, predicts, and reduces the in-bag risk monotonically.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov", "uniform"])
def test_bkernel_fits_and_reduces_risk(kernel):
    rng = np.random.default_rng(0)
    n = 80
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(n)
    data = {"y": y, "x": x}

    model = mboost.gamboost(
        f"y ~ bkernel(x, bandwidth=0.1, df=4, kernel='{kernel}')",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )

    risk = np.asarray(model.risk_)
    fitted = np.asarray(model.fitted_)

    assert fitted.shape == (n,)
    assert np.all(np.isfinite(fitted))
    # Risk is non-increasing along the boosting path.
    assert np.all(np.diff(risk) <= 1e-9), "bkernel risk path is not monotone"
    # The fit explains some structure (final risk well below the offset risk).
    assert risk[-1] < risk[0]

    # newdata prediction uses the Rust engine's kernel-ridge prediction.
    pred_new = np.asarray(model.predict(newdata={"x": np.array([0.2, 0.5])}))
    assert pred_new.shape == (2,)
    assert np.all(np.isfinite(pred_new))
