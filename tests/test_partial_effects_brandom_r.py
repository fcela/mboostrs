"""R and Python report the same brandom partial effect (issue #848).

R's `partial_effects` computed a grouped effect as `b * (1 - c)` and never
received the column centering the core applies for `brandom(center=TRUE)`,
so it returned the coefficient sum `b`. The fit's contribution of group g is
`b_g - sum_j c_j b_j`, which is what Python's `partial_effects` evaluates at
each training row. Both now take it from the core.
"""

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2

pytestmark = [pytest.mark.r_parity,
              pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")]


@pytest.mark.parametrize("center", [False, True])
def test_brandom_partial_effect_matches_python(center):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(848)
    n = 150
    g = rng.choice(np.arange(1, 6), size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    x = rng.uniform(size=n)
    y = np.sin(2 * np.pi * x) + np.array([-1, 0.5, 1, -0.3, 0.2])[g - 1] \
        + rng.normal(scale=0.3, size=n)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["x"], ro.globalenv["g"], ro.globalenv["y"] = x, g.astype(float), y
    flag = "TRUE" if center else "FALSE"
    # Not attached, so mboost's functions stay unmasked for other parity tests.
    r_effect = np.asarray(ro.r(
        "d <- data.frame(x = x, g = as.integer(g), y = y); "
        f"m <- mboostrs::gamboost(y ~ bbs(x, df = 4) + brandom(g, df = 3, center = {flag}), "
        "data = d, control = mboostrs::boost_control(mstop = 60)); "
        "mboostrs::partial_effects(m)$g$effect"))

    py = mb.gamboost(f"y ~ bbs(x, df=4) + brandom(g, df=3, center={center})",
                     data={"x": x, "g": g, "y": y}, control=mb.boost_control(60))
    groups, effect = py.partial_effects(which="g")["g"]
    groups = np.asarray(groups).astype(int)
    effect = np.asarray(effect, dtype=float)
    # Python reports one value per training row; every row of a group has
    # the group's effect.
    for level in range(1, 6):
        rows = effect[groups == level]
        np.testing.assert_allclose(rows, rows[0], atol=1e-12)
        np.testing.assert_allclose(r_effect[level - 1], rows[0], rtol=1e-8, atol=1e-12)
    if center:
        share = np.bincount(g, minlength=6)[1:] / n
        assert abs(share @ r_effect) < 1e-12
