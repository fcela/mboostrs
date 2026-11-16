"""R-parity tests for glmboost Cindex + bols (Gate 1: R oracle).

See crates/mboostrs-core/src/families.rs
(cindex_ngradient_and_risk_match_r_anchor,
cindex_ngradient_matches_finite_difference_of_risk,
cindex_min_time_censored_observation_gets_zero_gradient) for Gate 2, the
R-independent finite-difference/anchor gradient safety net.

Unlike Gehan, Cindex's offset is identically 0 in R (verified: `Cindex()@offset`
is the constant function `0`, and the risk only ever sees pairwise differences
`f[i] - f[j]`, so it is offset-invariant anyway). mboostrs' `offset() == 0.0`
is therefore NOT an intentional deviation -- raw fitted values are compared
directly against R, with no centering needed.
"""

import numpy as np
import pytest
from mboostrs._rust import family_diagnostics
from mboostrs.families import Cindex

import mboostrs as mboost

RTOL_RISK = 1e-8
ATOL_RISK = 1e-8
RTOL_FITTED = 1e-6
ATOL_FITTED = 1e-8
RTOL_NGRAD = 1e-8
ATOL_NGRAD = 1e-8


def _run_r_cindex(mstop, nu, n=200, seed=42, sigma=0.1, weights=None):
    """Generate survival data in R and fit glmboost with the Cindex family."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, sigma, weights) {
        suppressMessages(library(survival))
        suppressMessages(library(mboost))
        set.seed(seed)
        x1 <- rnorm(n)
        x2 <- rnorm(n)
        eta <- 0.3 * x1 + 0.1 * x2
        time <- rexp(n, rate = exp(eta))
        event <- rbinom(n, 1, 0.7)
        df <- data.frame(time = time, event = event, x1 = x1, x2 = x2)
        if (is.null(weights)) {
            w <- rep(1, n)
        } else {
            w <- weights
        }
        fit <- mboost::glmboost(
            Surv(time, event) ~ x1 + x2,
            data = df,
            family = mboost::Cindex(sigma = sigma),
            weights = w,
            center = TRUE,
            control = mboost::boost_control(mstop = mstop, nu = nu)
        )
        list(
            x1 = as.numeric(x1),
            x2 = as.numeric(x2),
            time = as.numeric(time),
            event = as.numeric(event),
            fitted = as.numeric(stats::predict(fit, type = "link")),
            risk = as.numeric(mboost::risk(fit)),
            selected = as.integer(fit$xselect()),
            offset = as.numeric(fit$offset)
        )
    }
    """
    r_func = ro.r(r_code)
    weights_r = ro.NULL if weights is None else ro.FloatVector(weights)
    result = r_func(mstop, nu, n, seed, sigma, weights_r)

    return {
        "x1": np.asarray(result[0], dtype=np.float64),
        "x2": np.asarray(result[1], dtype=np.float64),
        "time": np.asarray(result[2], dtype=np.float64),
        "event": np.asarray(result[3], dtype=np.float64),
        "fitted": np.asarray(result[4], dtype=np.float64),
        "risk": np.asarray(result[5], dtype=np.float64),
        "selected": [int(s) - 1 for s in result[6]],
        "offset": float(result[7][0]),
    }


def _run_r_cindex_ngradient(n=80, data_seed=11, f_seed=99, sigma=0.1):
    """Pull R's Cindex(sigma)@ngradient(y, f, w) at a fixed random f,
    independent of any fit -- the strongest, most direct gradient oracle."""
    import rpy2.robjects as ro

    r_code = """
    function(n, data_seed, f_seed, sigma) {
        suppressMessages(library(survival))
        suppressMessages(library(mboost))
        set.seed(data_seed)
        time <- rexp(n, rate = 1)
        event <- rbinom(n, 1, 0.7)
        y <- Surv(time, event)
        set.seed(f_seed)
        f <- rnorm(n, sd = 1.5)
        w <- rep(1, n)
        fam <- mboost::Cindex(sigma = sigma)
        ng <- fam@ngradient(y, f, w)
        risk <- fam@risk(y, f, w)
        list(
            time = as.numeric(time),
            event = as.numeric(event),
            f = as.numeric(f),
            ng = as.numeric(ng),
            risk = as.numeric(risk)
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(n, data_seed, f_seed, sigma)
    return {
        "time": np.asarray(result[0], dtype=np.float64),
        "event": np.asarray(result[1], dtype=np.float64),
        "f": np.asarray(result[2], dtype=np.float64),
        "ng": np.asarray(result[3], dtype=np.float64),
        "risk": float(result[4][0]),
    }


@pytest.mark.r_parity
class TestCindexBols:
    """Cindex family fit parity against R mboost."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_cindex(mstop=50, nu=0.1, n=200, seed=42)
        self.data = {
            "time": self.r["time"],
            "event": self.r["event"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def _fit(self):
        return mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=self.data,
            family=Cindex(),
            control=mboost.boost_control(50, 0.1),
        )

    def test_risk_parity(self):
        model = self._fit()
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=RTOL_RISK,
            atol=ATOL_RISK,
            err_msg="Cindex risk path does not match R",
        )

    def test_selected_parity(self):
        model = self._fit()
        assert list(model._selected) == self.r["selected"]

    def test_raw_fitted_parity(self):
        # Unlike Gehan, Cindex's offset matches R exactly (both 0), so raw
        # fitted values -- not just centered ones -- are directly comparable.
        model = self._fit()
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=RTOL_FITTED,
            atol=ATOL_FITTED,
            err_msg="Cindex fitted values do not match R",
        )

    def test_offset_matches_r(self):
        model = self._fit()
        assert model.offset_ == 0.0
        assert self.r["offset"] == 0.0


@pytest.mark.r_parity
class TestCindexBolsShortMstop:
    """Cindex at a different mstop/seed, to catch fixture-specific coincidences."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.r = _run_r_cindex(mstop=20, nu=0.15, n=150, seed=123)
        self.data = {
            "time": self.r["time"],
            "event": self.r["event"],
            "x1": self.r["x1"],
            "x2": self.r["x2"],
        }

    def test_risk_parity_short(self):
        model = mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=self.data,
            family=Cindex(),
            control=mboost.boost_control(20, 0.15),
        )
        np.testing.assert_allclose(
            model.risk_,
            self.r["risk"],
            rtol=RTOL_RISK,
            atol=ATOL_RISK,
            err_msg="Cindex risk (short mstop) does not match R",
        )

    def test_raw_fitted_parity_short(self):
        model = mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=self.data,
            family=Cindex(),
            control=mboost.boost_control(20, 0.15),
        )
        np.testing.assert_allclose(
            model.fitted_,
            self.r["fitted"],
            rtol=RTOL_FITTED,
            atol=ATOL_FITTED,
            err_msg="Cindex fitted (short mstop) does not match R",
        )


@pytest.mark.r_parity
def test_cindex_sigma_not_default_end_to_end():
    """A non-default sigma (0.05, sharper smoothing) must thread through the
    Python -> Rust family_params boundary and match R end to end."""
    r = _run_r_cindex(mstop=40, nu=0.1, n=150, seed=17, sigma=0.05)
    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(sigma=0.05),
        control=mboost.boost_control(40, 0.1),
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=RTOL_RISK, atol=ATOL_RISK,
        err_msg="Cindex(sigma=0.05) risk path does not match R",
    )
    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=RTOL_FITTED, atol=ATOL_FITTED,
        err_msg="Cindex(sigma=0.05) fitted values do not match R",
    )
    assert list(model._selected) == r["selected"]


@pytest.mark.r_parity
def test_cindex_zero_one_weights_parity():
    """The IPCW-on-subset machinery is Cindex's genuinely new piece vs.
    Gehan/AUC: with some weights == 0, R computes IPCweights on the w != 0
    subset only, then scatters 0 back into the excluded positions. Fit with
    an explicit {0,1} weight vector end to end and compare against R."""
    rng = np.random.default_rng(3)
    n = 150
    weights = (rng.uniform(size=n) < 0.8).astype(np.float64)
    r = _run_r_cindex(mstop=40, nu=0.1, n=n, seed=55, weights=weights)
    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, event) ~ x1 + x2",
        data=data,
        family=Cindex(),
        weights=weights,
        control=mboost.boost_control(40, 0.1),
    )
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=RTOL_RISK, atol=ATOL_RISK,
        err_msg="Cindex risk with {0,1} weights does not match R",
    )
    np.testing.assert_allclose(
        model.fitted_, r["fitted"], rtol=RTOL_FITTED, atol=ATOL_FITTED,
        err_msg="Cindex fitted with {0,1} weights does not match R",
    )
    assert list(model._selected) == r["selected"]


@pytest.mark.r_parity
def test_cindex_ngradient_matches_r_element_wise():
    """Direct gradient oracle: mboostrs negative_gradient vs R's
    Cindex()@ngradient(y, f, w) at a fixed random f, independent of any fit.
    This is the make-or-break check for the IPCW + weight-matrix + gradient
    construction (see families.rs `cindex_ngradient_and_risk`).
    """
    r = _run_r_cindex_ngradient(n=80, data_seed=11, f_seed=99, sigma=0.1)
    y = np.concatenate([r["time"], r["event"]])
    w = np.ones_like(r["time"])

    _, grad, risk = family_diagnostics("cindex", {"sigma": 0.1}, y, r["f"], w)
    grad = np.asarray(grad)

    np.testing.assert_allclose(
        grad,
        r["ng"],
        rtol=RTOL_NGRAD,
        atol=ATOL_NGRAD,
        err_msg="Cindex negative_gradient does not match R's Cindex()@ngradient element-wise",
    )
    np.testing.assert_allclose(
        risk,
        r["risk"],
        rtol=RTOL_RISK,
        atol=ATOL_RISK,
        err_msg="Cindex risk does not match R's Cindex()@risk",
    )


@pytest.mark.r_parity
def test_cindex_ngradient_matches_r_element_wise_sigma_not_default():
    """Same direct oracle as above, but with sigma=0.05 to catch a
    hard-coded-default bug in the sigma threading."""
    r = _run_r_cindex_ngradient(n=60, data_seed=5, f_seed=13, sigma=0.05)
    y = np.concatenate([r["time"], r["event"]])
    w = np.ones_like(r["time"])

    _, grad, risk = family_diagnostics("cindex", {"sigma": 0.05}, y, r["f"], w)
    grad = np.asarray(grad)

    np.testing.assert_allclose(
        grad,
        r["ng"],
        rtol=RTOL_NGRAD,
        atol=ATOL_NGRAD,
        err_msg="Cindex(sigma=0.05) negative_gradient does not match R element-wise",
    )
    np.testing.assert_allclose(
        risk,
        r["risk"],
        rtol=RTOL_RISK,
        atol=ATOL_RISK,
        err_msg="Cindex(sigma=0.05) risk does not match R",
    )


@pytest.mark.r_parity
def test_cindex_rejects_non_zero_one_weights():
    """Non-{0,1} case weights are explicitly rejected (R: 'weights must be
    either 0 or 1 for family UnoC'); mboostrs should raise a clear error."""
    r = _run_r_cindex(mstop=10, nu=0.1, n=60, seed=7)
    data = {"time": r["time"], "event": r["event"], "x1": r["x1"], "x2": r["x2"]}
    weights = np.ones(60)
    weights[::3] = 2.0
    with pytest.raises(ValueError):
        mboost.glmboost(
            "Surv(time, event) ~ x1 + x2",
            data=data,
            family=Cindex(),
            weights=weights,
            control=mboost.boost_control(10, 0.1),
        )


def _run_r_cindex_cvrisk_subsampling(mstop, nu, n=100, seed=21, B=10, prob=0.7):
    """Fit Cindex glmboost in R and cross-validate with R-generated
    subsampling folds (Cindex's {0,1} weight domain supports subsampling,
    unlike bootstrap's integer counts)."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu, n, seed, B, prob) {
        suppressMessages(library(survival))
        suppressMessages(library(mboost))
        set.seed(seed)
        x1 <- rnorm(n); x2 <- rnorm(n)
        eta <- 0.4 * x1 - 0.2 * x2
        time <- rexp(n, exp(eta)); cens <- runif(n, 0, 3)
        status <- as.numeric(time <= cens); time <- pmin(time, cens)
        df <- data.frame(time = time, status = status, x1 = x1, x2 = x2)
        m <- mboost::glmboost(
            Surv(time, status) ~ x1 + x2, data = df,
            family = mboost::Cindex(),
            control = mboost::boost_control(mstop = mstop, nu = nu))
        set.seed(9)
        folds <- mboost::cv(stats::model.weights(m), type = "subsampling", B = B, prob = prob)
        cvr <- mboost::cvrisk(m, folds = folds, papply = lapply)
        oob <- matrix(as.numeric(cvr), nrow = nrow(cvr))
        list(time = as.numeric(time), status = as.numeric(status),
             x1 = as.numeric(x1), x2 = as.numeric(x2),
             folds = as.numeric(folds), nrow_f = nrow(folds), ncol_f = ncol(folds),
             oob = as.numeric(oob), nrow_o = nrow(oob), ncol_o = ncol(oob),
             rmstop = mboost::mstop(cvr))
    }
    """
    res = ro.r(r_code)(mstop, nu, n, seed, B, prob)
    d = {k: np.asarray(res.rx2(k), dtype=np.float64) for k in
         ("time", "status", "x1", "x2", "folds", "oob")}
    nrf, ncf = int(res.rx2("nrow_f")[0]), int(res.rx2("ncol_f")[0])
    nro, nco = int(res.rx2("nrow_o")[0]), int(res.rx2("ncol_o")[0])
    d["folds"] = d["folds"].reshape((ncf, nrf)).T
    d["oob"] = d["oob"].reshape((nco, nro)).T
    d["rmstop"] = int(res.rx2("rmstop")[0])
    return d


@pytest.mark.r_parity
def test_cindex_cvrisk_subsampling_matches_r():
    """Cindex's {0,1} weight domain is exactly the subsampling-fold format,
    so (unlike Gehan/AUC) cvrisk with subsampling folds is supported."""
    mstop, nu = 60, 0.1
    r = _run_r_cindex_cvrisk_subsampling(mstop, nu)
    data = {"time": r["time"], "status": r["status"], "x1": r["x1"], "x2": r["x2"]}
    model = mboost.glmboost(
        "Surv(time, status) ~ x1 + x2", data=data, family=Cindex(),
        control=mboost.boost_control(mstop, nu),
    )
    cvr = model.cvrisk(folds=r["folds"], parallel=False)
    py_oob = np.asarray(cvr.risk_matrix)

    assert py_oob.shape == r["oob"].shape
    np.testing.assert_allclose(
        py_oob, r["oob"], rtol=1e-6, atol=1e-8,
        err_msg="Cindex cvrisk OOB risk matrix does not match R",
    )
    assert cvr.optimal_mstop == r["rmstop"], (
        f"mstop mismatch: py={cvr.optimal_mstop} R={r['rmstop']}"
    )
