"""The Laplace offset on an even number of rows equals mboost's (issue #161).

With an even row count the absolute-error risk is flat between the two middle
order statistics, and `optimize()` stops wherever its probe sequence ends in
that interval. R's `Brent_fmin` is C, which the compiler on aarch64 contracts
into fused multiply-adds; one ulp in a probe sent mboostrs to another point
of the flat interval (0.781342 against 0.782023). The core now evaluates the
same fused expressions there, so the offsets agree exactly.
"""

import platform

import numpy as np
import pytest

import mboostrs as mb
from conftest import HAS_R, HAS_RPY2


@pytest.mark.r_parity
@pytest.mark.skipif(not (HAS_R and HAS_RPY2), reason="requires R + rpy2")
@pytest.mark.skipif(platform.machine() not in ("arm64", "aarch64"),
                    reason="R's Brent_fmin fuses multiply-adds only on aarch64 builds")
@pytest.mark.parametrize("seed", range(6))
def test_even_length_offset_matches_mboost(seed):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    rng = np.random.default_rng(seed)
    n = 200
    x = rng.uniform(size=n)
    y = np.sin(3 * x) + rng.standard_t(3, n)
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["x"], ro.globalenv["y"] = x, y
    upstream = ro.r(
        "suppressMessages(library(mboost)); "
        "m <- mboost::gamboost(y ~ bbs(x), data = data.frame(x = x, y = y), "
        "family = Laplace(), control = boost_control(mstop = 50)); "
        "list(m$offset, as.numeric(fitted(m)))")
    fit = mb.gamboost("y ~ bbs(x)", data={"x": x, "y": y},
                      family=mb.Laplace(response_scale=None),
                      control=mb.boost_control(50))
    assert float(np.asarray(fit.offset_).ravel()[0]) == float(np.asarray(upstream[0])[0])
    np.testing.assert_allclose(np.asarray(fit.fitted_), np.asarray(upstream[1]),
                               atol=1e-10)
