"""Parity tests for the cvrisk() example from R mboost's cvrisk.Rd.

Source: ``example("cvrisk", package = "mboost")`` — bodyfat data with
glmboost and 25 bootstrap cvrisk iterations.

Gated by ``@pytest.mark.r_parity``; requires R + rpy2 + mboost + TH.data.
"""

import mboostrs as mboost
import numpy as np
import pytest
from mboostrs.families import Gaussian
from conftest import r_call

RTOL = 1e-6
ATOL = 1e-8


def _run_r_cvrisk_bodyfat():
    """R's cvrisk example on bodyfat data (requires TH.data)."""
    import rpy2.robjects as ro

    r_code = """
    function() {
        suppressMessages(library(mboost))
        if (!requireNamespace("TH.data", quietly = TRUE))
            stop("TH.data not installed")
        data("bodyfat", package = "TH.data")

        model <- glmboost(DEXfat ~ ., data = bodyfat, center = TRUE)

        set.seed(290875)
        n <- nrow(bodyfat)
        bs25 <- rmultinom(25, n, rep(1, n) / n)
        cvm  <- cvrisk(model, folds = bs25, papply = lapply)

        X <- model.matrix(DEXfat ~ ., data = bodyfat)[, -1]
        folds_mat <- as.matrix(bs25)
        cv_flat <- as.numeric(unlist(lapply(cvm, function(x) as.numeric(x))))

        list(
            X = unname(X),
            y = as.numeric(bodyfat$DEXfat),
            fitted = as.numeric(predict(model, type = "link")),
            risk   = unname(as.numeric(risk(model))),
            offset = as.numeric(model$offset[1]),
            folds  = unname(folds_mat),
            cvrisk = cv_flat,
            cv_nrow = length(cvm),
            cv_ncol = length(cvm[[1]])
        )
    }
    """
    res = r_call(ro.r(r_code), packages=("mboost", "TH.data"))

    cv_flat = np.asarray(res[6], dtype=np.float64).ravel()
    n_folds = 25  # bs25 = rmultinom(25, ...)
    n_mstop = cv_flat.size // n_folds
    return {
        "X": np.asarray(res[0], dtype=np.float64),
        "y": np.asarray(res[1], dtype=np.float64),
        "fitted": np.asarray(res[2], dtype=np.float64),
        "risk": np.asarray(res[3], dtype=np.float64),
        "offset": float(res[4][0]),
        "folds": np.asarray(res[5], dtype=np.float64),
        "cvrisk": cv_flat.reshape(n_folds, n_mstop, order="F"),
    }


@pytest.mark.r_parity
class TestCvriskBodyfat:
    """``cvrisk(glmboost(DEXfat ~ ., bodyfat))`` parity against R."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.r = _run_r_cvrisk_bodyfat()
        col_names = [f"x{i}" for i in range(self.r["X"].shape[1])]
        self.data = {"y": self.r["y"]}
        for j, name in enumerate(col_names):
            self.data[name] = self.r["X"][:, j]
        self.formula = "y ~ " + " + ".join(col_names)

    def test_glmboost_fitted(self):
        m = mboost.glmboost(self.formula, data=self.data, family=Gaussian(),
                            control=mboost.boost_control(100, 0.1))
        np.testing.assert_allclose(m.fitted_, self.r["fitted"], rtol=RTOL, atol=ATOL,
                                   err_msg="bodyfat glmboost fitted mismatch")

    def test_glmboost_risk(self):
        m = mboost.glmboost(self.formula, data=self.data, family=Gaussian(),
                            control=mboost.boost_control(100, 0.1))
        np.testing.assert_allclose(m.risk_, self.r["risk"], rtol=RTOL, atol=ATOL,
                                   err_msg="bodyfat glmboost risk mismatch")

    def test_cvrisk_bootstrap(self):
        """Bootstrap cvrisk with R-generated folds."""
        m = mboost.glmboost(self.formula, data=self.data, family=Gaussian(),
                            control=mboost.boost_control(100, 0.1))
        folds = self.r["folds"].reshape(self.r["folds"].shape, order="F")
        cv_result = m.cvrisk(folds=folds)
        np.testing.assert_allclose(
            cv_result.risk_matrix, self.r["cvrisk"],
            rtol=RTOL, atol=ATOL,
            err_msg="cvrisk bootstrap risk matrix mismatch",
        )
