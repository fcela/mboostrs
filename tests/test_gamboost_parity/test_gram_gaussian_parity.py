"""R-parity tests for the Gaussian covariance/Gram-update fast path.

A Gaussian `gamboost` fit with one or more block learners (bbs/...) and the
default single-fit entry goes through `engine::try_fit_gaussian_gram`
(squared-error residual updates linearly, so per-learner projections are
maintained via cross-Grams instead of re-projecting each iteration). These
tests confirm that path reproduces R's `mboost::gamboost` fitted values and
risk path.

The Gram path is only taken when `total_p <= mstop`; for 2x bbs(df=4) the
centered designs have p=22 each (total 44), so the chosen mstop values
(>=50) keep it active. A model too wide for the crossover would silently
fall back to the (equally R-matching) general path, so these tests also
serve as plain gamboost-Gaussian parity coverage.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian

rtol = 1e-6
atol = 1e-9


def _run_r_gamboost(formula, mstop, nu, n, seed, n_terms):
    """Fit `formula` with R mboost; return data, fitted values and risk path."""
    import rpy2.robjects as ro

    r_func = ro.r(
        """
        function(formula, mstop, nu, n, seed, n_terms) {
            library(mboost)
            set.seed(seed)
            xs <- list()
            for (j in 1:n_terms) xs[[j]] <- runif(n, 0, 1)
            names(xs) <- paste0("x", 1:n_terms)
            eta <- rep(0, n)
            for (j in 1:n_terms) eta <- eta + sin(2 * pi * j * xs[[j]] / n_terms)
            y <- eta + rnorm(n, sd = 0.3)
            df <- as.data.frame(xs); df$y <- y
            fit <- gamboost(
                as.formula(formula), data = df, family = Gaussian(),
                control = boost_control(mstop = mstop, nu = nu)
            )
            out <- list(
                y = as.numeric(y),
                fitted = as.numeric(stats::predict(fit, type = "link")),
                risk = as.numeric(risk(fit))
            )
            for (j in 1:n_terms) out[[paste0("x", j)]] <- as.numeric(xs[[j]])
            out
        }
        """
    )
    res = r_func(formula, mstop, nu, n, seed, n_terms)
    d = {k: np.asarray(res.rx2(k), dtype=np.float64) for k in res.names}
    return d


@pytest.mark.r_parity
@pytest.mark.parametrize("mstop", [50, 200])
def test_gram_gaussian_2bbs_parity(mstop):
    nu = 0.1
    formula = "y ~ bbs(x1, df=4) + bbs(x2, df=4)"
    r = _run_r_gamboost(formula, mstop, nu, n=400, seed=11, n_terms=2)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"]}

    model = mboost.gamboost(
        formula, data=data, family=Gaussian(),
        control=mboost.boost_control(mstop, nu),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="Gram-path gamboost fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="Gram-path gamboost risk path does not match R",
    )


@pytest.mark.r_parity
def test_gram_gaussian_3bbs_parity():
    """3 block learners + higher mstop — exercises selection across more terms."""
    mstop, nu = 150, 0.1
    formula = "y ~ bbs(x1, df=4) + bbs(x2, df=4) + bbs(x3, df=4)"
    r = _run_r_gamboost(formula, mstop, nu, n=500, seed=23, n_terms=3)
    data = {"y": r["y"], "x1": r["x1"], "x2": r["x2"], "x3": r["x3"]}

    model = mboost.gamboost(
        formula, data=data, family=Gaussian(),
        control=mboost.boost_control(mstop, nu),
    )

    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=rtol, atol=atol,
        err_msg="Gram-path 3-bbs fitted values do not match R",
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=rtol, atol=atol,
        err_msg="Gram-path 3-bbs risk path does not match R",
    )
