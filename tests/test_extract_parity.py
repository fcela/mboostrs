"""extract() reports mboost's design, penalty, lambda and df, with weights."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.mark.r_parity
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_extract_matches_mboost(weighted):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r("""suppressMessages(library(mboost)); set.seed(7); n <- 200
        d <- data.frame(x = runif(n, -1, 1), z = rnorm(n), w = runif(n),
                        g = factor(sample(letters[1:4], n, TRUE)))
        d$y <- sin(2 * d$x) + 0.5 * d$z + as.integer(d$g) / 3 + rnorm(n, sd = .4)
        wt <- runif(n, 0.5, 2)""")
    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.rpy2py(ro.r("d"))
    data["g"] = np.asarray(ro.r("as.integer(d$g)"))
    weights = np.asarray(ro.r("wt")) if weighted else None
    ro.r("m <- gamboost(y ~ bbs(x) + bols(z) + bbs(w, df = 3) + brandom(g), data = d, "
         + ("weights = wt, " if weighted else "") + "control = boost_control(mstop = 120))")
    model = mb.gamboost("y ~ bbs(x) + bols(z) + bbs(w, df=3) + brandom(g)", data=data,
                        weights=weights, control=mb.boost_control(120))
    for what in ("design", "penalty", "lambda", "df"):
        ours = model.extract(what)
        for i, key in enumerate(ours):
            if what == "design" and key == "g":
                continue  # mboost reports a factor learner's design on its distinct rows
            upstream = np.asarray(ro.r(f"as.matrix(extract(m, what = '{what}')[[{i + 1}]])"))
            got = np.atleast_2d(np.asarray(ours[key], dtype=float)).reshape(upstream.shape)
            tol = 1e-7 * max(1.0, float(np.abs(upstream).max())) if what == "lambda" else 1e-9
            np.testing.assert_allclose(got, upstream, rtol=0, atol=tol, err_msg=f"{what} {key}")
