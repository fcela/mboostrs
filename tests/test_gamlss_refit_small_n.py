"""gamlss_refit converges with a null smooth in sigma on small samples (#374).

The data: y = sin(3 x) + noise with sd exp(-1 + 0.8 z), a null covariate w,
mu ~ bbs(x) and sigma ~ bbs(z) + bbs(w). On these (n, seed) pairs the refit
used to stop with "the refit optimum cannot be re-evaluated" or "could not
evaluate its criterion at the start".
"""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb

# (n, seed) pairs that failed on main before the fix.
FAILING_CASES = [(80, 0), (80, 1), (80, 3), (80, 4), (80, 5), (80, 6), (80, 7), (150, 4)]


def _fit_and_refit(n, seed):
    rng = np.random.default_rng(seed)
    x, z, w = rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = np.sin(3 * x) + rng.normal(0, np.exp(-1 + 0.8 * z), n)
    df = pl.DataFrame({"x": x, "z": z, "w": w, "y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboostLSS(
            {"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(z) + bbs(w)"},
            data=df, families=mb.GaussianLSS(),
            control=mb.boost_control(mstop=200),
        )
        return mb.gamlss_refit(fit)


@pytest.mark.parametrize("n, seed", FAILING_CASES)
def test_refit_converges_and_shrinks_the_null_smooth(n, seed):
    ref = _fit_and_refit(n, seed)
    conv = ref.convergence
    assert conv["converged"] is True, conv
    assert conv["stop_reason"] in ("gradient", "stationary", "flat"), conv
    assert np.isfinite(ref.reml_score)
    # The null term keeps only a little beyond its unpenalized linear trend.
    # With a 20-knot basis, 2.5 df shows it was shrunk. The informative term is
    # linear in the truth (log sd is linear in z), so its edf is near 1 and is
    # not asserted.
    assert ref.edf["sigma"]["bbs(w)"] < 2.5, ref.edf["sigma"]
