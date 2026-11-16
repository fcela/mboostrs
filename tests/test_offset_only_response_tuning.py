"""An intercept-only specification has a valid constant response-score path."""
import numpy as np
import pytest

import mboostrs as mb


@pytest.mark.parametrize("family,score", [(mb.Gaussian(), "mse"), (mb.Binomial(), "log_loss"),
                                         (mb.Poisson(), "mse")])
def test_offset_only_metric_tuning_preserves_family_and_weights(family, score):
    rng = np.random.default_rng(90421)
    n = 90
    y = (rng.binomial(1, .3, n).astype(float) if isinstance(family, mb.Binomial) else
         rng.poisson(3, n).astype(float) if isinstance(family, mb.Poisson) else rng.normal(3, 1, n))
    weights = rng.uniform(.4, 2, n)
    tuned = mb.tune_mstop("y ~ 1", {"y": y}, family=family, weights=weights,
                          folds=3, start=10, max_mstop=20, score=score, seed=7)
    assert tuned.mstop == 1 and tuned.largest == 10 and not tuned.at_boundary
    np.testing.assert_allclose(tuned.risk, tuned.risk[0], rtol=0, atol=0)
    expected = np.average(y, weights=weights)
    prediction = tuned.model.predict({"unused": np.zeros(12)}, type="response")
    np.testing.assert_allclose(prediction, expected, rtol=1e-10, atol=1e-10)
