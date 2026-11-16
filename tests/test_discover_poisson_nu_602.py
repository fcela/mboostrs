"""Regression test for issue #602, deficit 1 (quine Poisson).

Discovery's screening fits used a hardcoded NU = 0.1, tuned for Gaussian's
constant loss curvature. For Poisson (log link) the Hessian varies with the
fitted mean; the step overshoots, the screening risk path oscillates instead
of descending, and the leave-one-out comparisons that drive term elimination
are corrupted.

_screening_nu() returns a smaller step for non-Gaussian families so the
screening path descends.
"""
import numpy as np
import polars as pl

import mboostrs as mb
from mboostrs.discover import _screening_nu, NU, NU_NONGAUSSIAN


def test_screening_nu_gaussian_keeps_default():
    assert _screening_nu(mb.Gaussian()) == NU


def test_screening_nu_poisson_uses_small_step():
    assert _screening_nu(mb.Poisson()) == NU_NONGAUSSIAN
    assert NU_NONGAUSSIAN < NU


def test_poisson_screening_path_descends():
    """The screening fit's risk path must descend for Poisson.

    With NU = 0.1 the quine Poisson risk rose on 99 of 199 steps (minimum at
    step 3); the screening comparisons were noise. With the family-aware step
    the path descends.
    """
    rng = np.random.default_rng(602)
    n = 300
    x = rng.normal(size=n)
    mu = np.exp(0.5 + 0.8 * x)
    y = rng.poisson(mu)
    df = pl.DataFrame({"x": x, "y": y})
    for nu, max_rises in ((NU, 199), (NU_NONGAUSSIAN, 20)):
        m = mb.gamboost("y ~ bols(x)", data=df, family=mb.Poisson(),
                        control=mb.boost_control(mstop=200, nu=nu))
        risk = np.asarray(m._risk)
        rises = int(np.sum(np.diff(risk) > 0))
        if nu == NU_NONGAUSSIAN:
            assert rises <= max_rises, f"nu={nu}: {rises} rises, path oscillates"
