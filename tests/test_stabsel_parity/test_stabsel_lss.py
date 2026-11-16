"""R-parity test for stability selection on a GAMLSS (glmboostLSS) model.

Verifies stabsel over the combined ``<predictor>.<parameter>`` learner set
against R's ``stabsel.mboostLSS`` (non-cyclic), using shared subsampling folds so
the comparison is deterministic.

The *selected set* matches R exactly. The per-learner maximum selection
frequencies match within a small tolerance: non-cyclic boosting has near-tied
steps where R and mboostrs pick different equally-optimal parameters (the
documented flat-objective ambiguity), so a couple of the subsamples' selection
paths can differ, shifting an individual frequency by 1-2 subsamples.

Requires R packages mboost + gamboostLSS; skipped otherwise.
"""

import numpy as np
import pytest
from mboostrs.gamlss import GaussianLSS, stabsel_glmboostLSS

import mboostrs as mboost  # noqa: F401
from conftest import r_call


@pytest.mark.r_parity
def test_stabsel_glmboostlss_gaussian():
    import rpy2.robjects as ro

    r = ro.r("""
    function() {
        if (!requireNamespace("gamboostLSS", quietly = TRUE)) stop("no gamboostLSS")
        suppressMessages(library(gamboostLSS))
        set.seed(1); n <- 100; p <- 8
        X <- matrix(rnorm(n * p), n, p); colnames(X) <- paste0("x", 1:p)
        y <- rnorm(n, 1 + 0.8 * X[, 1], exp(0.3 * X[, 2]))
        d <- data.frame(y = y, X)
        m <- glmboostLSS(y ~ ., data = d, families = GaussianLSS(),
                         control = boost_control(mstop = 50, nu = 0.1),
                         method = "noncyclic")
        set.seed(7); folds <- (cv(model.weights(m), type = "subsampling", B = 20) > 0) * 1
        s <- stabsel(m, folds = folds, q = 4, PFER = 2, sampling.type = "MB", B = 20)
        list(y = as.numeric(y), X = as.numeric(X), n = n, p = p,
             folds = as.numeric(folds),
             selected = names(s$selected), names = names(s$max),
             max = as.numeric(s$max),
             cutoff = as.numeric(s$cutoff), q = as.numeric(s$q))
    }
    """)
    res = r_call(r, packages=("gamboostLSS",))

    n, p = int(res.rx2("n")[0]), int(res.rx2("p")[0])
    X = np.asarray(res.rx2("X")).reshape(n, p, order="F")
    y = np.asarray(res.rx2("y"))
    folds = np.asarray(res.rx2("folds")).reshape(n, 20, order="F")
    r_selected = set(res.rx2("selected"))
    r_names = list(res.rx2("names"))
    r_max = np.asarray(res.rx2("max"))
    r_cutoff = float(res.rx2("cutoff")[0])
    r_q = int(res.rx2("q")[0])

    data = {"y": y}
    for j in range(p):
        data[f"x{j+1}"] = X[:, j]
    formula = "y ~ " + " + ".join(f"x{j+1}" for j in range(p))

    out = stabsel_glmboostLSS(formula, data=data, families=GaussianLSS(),
                              control=mboost.boost_control(50, 0.1),
                              q=4, PFER=2, folds=folds, sampling_type="MB")

    # Error-control parameters match R's stabs math exactly.
    assert out["cutoff"] == pytest.approx(r_cutoff)
    assert out["q"] == r_q

    # Selected set matches R exactly.
    assert set(out["selected"]) == r_selected, (
        f"stabsel selection differs: mboostrs {sorted(out['selected'])} "
        f"vs R {sorted(r_selected)}")

    # Per-learner max selection frequencies match within the near-tie tolerance.
    my_max = np.array([out["max"][nm] for nm in r_names])
    assert np.max(np.abs(my_max - r_max)) <= 0.15
