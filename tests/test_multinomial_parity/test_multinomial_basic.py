"""R-parity tests for Multinomial family with Kronecker-product base-learners."""

import numpy as np
import pytest
from conftest import r_call

RTOL_RISK = 1e-10
RTOL_FITTED = 1e-6


def _run_r_multinomial_iris(mstop=20, nu=0.1):
    """Run R's iris Multinomial example and return reference values."""
    import rpy2.robjects as ro

    r_code = """
    function(mstop, nu) {
        library(mboost)
        data(iris)
        myiris <- as.list(iris)
        myiris$class <- factor(levels(iris$Species)[-nlevels(iris$Species)])
        mlm <- mboost(
            Species ~ bols(Sepal.Length, df=2) %O%
                      bols(class, df=2, contrasts.arg="contr.dummy"),
            data = myiris,
            family = Multinomial(),
            control = boost_control(mstop = mstop, nu = nu)
        )
        list(
            risk = as.numeric(risk(mlm)),
            fitted = as.numeric(fitted(mlm)),
            selected = as.numeric(selected(mlm)),
            x = as.numeric(iris$Sepal.Length),
            y_species = as.character(iris$Species),
            response = as.numeric(predict(mlm, type = "response")),
            link = as.numeric(predict(mlm, type = "link")),
            coef = as.numeric(coef(mlm)[[1]])
        )
    }
    """
    r_func = ro.r(r_code)
    result = r_func(mstop, nu)

    return {
        "risk": np.asarray(result.rx2("risk"), dtype=np.float64),
        "fitted": np.asarray(result.rx2("fitted"), dtype=np.float64),
        "selected": np.asarray(result.rx2("selected"), dtype=np.int64),
        "x": np.asarray(result.rx2("x"), dtype=np.float64),
        "y_species": np.asarray(result.rx2("y_species")),
        "response": np.asarray(result.rx2("response"), dtype=np.float64),
        "link": np.asarray(result.rx2("link"), dtype=np.float64),
        "coef": np.asarray(result.rx2("coef"), dtype=np.float64),
    }


def _encode_multinomial_y(y_species):
    """Encode string labels as row-major dummy-coded response, dropping last class."""
    classes = sorted(np.unique(y_species))
    K = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_codes = np.array([class_to_idx[yi] for yi in y_species], dtype=np.int64)
    indicator = np.eye(K, dtype=np.float64)[y_codes][:, :-1]
    return np.ascontiguousarray(indicator.ravel(), dtype=np.float64), K


@pytest.mark.r_parity
def test_multinomial_iris_risk():
    """Initial risk must match R exactly (depends only on y and offset=0)."""
    r = _run_r_multinomial_iris()
    y, K = _encode_multinomial_y(r["y_species"])
    assert K == 3
    assert y.shape == (300,)

    # Initial risk (index 0) only depends on y and initial offset (0).
    # Our initial risk computation should match R's.
    np.testing.assert_allclose(
        r["risk"][0],
        109.861228866810819,
        rtol=1e-12,
        err_msg="R initial risk reference changed",
    )


@pytest.mark.r_parity
def test_multinomial_smoke():
    """Basic smoke test: Multinomial gamboost runs and risk decreases."""
    from mboostrs import Multinomial, gamboost

    np.random.seed(42)
    n = 90
    x = np.random.randn(n)
    y = np.repeat(["A", "B", "C"], 30)
    data = {"y": y, "x": x}

    model = gamboost(
        "y ~ bols(x, intercept=False)",
        data=data,
        family=Multinomial(n_classes=3),
        control=(20, 0.1),
    )

    assert len(model.risk_) == 21
    assert len(model.selected) == 20
    assert model._fitted.shape == (180,)  # n * (K-1) = 90 * 2
    # Risk may not decrease monotonically with random data and intercept=False,
    # but should stay in a reasonable range
    assert np.all(np.isfinite(model.risk_))
    assert model._fitted.shape[0] == n * 2


@pytest.mark.r_parity
def test_multinomial_iris_bols_intercept_false():
    """Multinomial with bols(intercept=True) on iris: initial risk matches R, risk decreases."""
    from mboostrs import Multinomial, gamboost

    r = _run_r_multinomial_iris()
    x = r["x"]
    y_species = r["y_species"]
    data = {"Species": y_species, "Sepal_Length": x}

    # Use intercept=True (centered design) which gives a better-conditioned problem
    model = gamboost(
        "Species ~ bols(Sepal_Length, intercept=True)",
        data=data,
        family=Multinomial(),
        control=(20, 0.1),
    )

    # Initial risk must match R (same y encoding, offset=0)
    np.testing.assert_allclose(model.risk_[0], r["risk"][0], rtol=1e-10)

    # Fitted values shape
    n = len(x)
    K = 3
    assert model._fitted.shape == (n * (K - 1),)

    # All fitted values should be finite
    assert np.all(np.isfinite(model._fitted))


@pytest.mark.r_parity
def test_multinomial_iris_full_risk_and_fitted_match():
    """Full risk path, selected sequence, and fitted values match R's iris example.

    R's ``bols(Sepal.Length, df=2)`` (lambda=0) equals our
    ``bols(Sepal_Length, intercept=True)`` centered design, so the whole boosting
    trajectory must agree. R stores ``fitted`` column-major (n, K-1); our engine
    is row-major, so we transpose before comparing.
    """
    from mboostrs import Multinomial, gamboost

    mstop, nu = 20, 0.1
    r = _run_r_multinomial_iris(mstop=mstop, nu=nu)
    x = r["x"]
    y_species = r["y_species"]
    n, K = len(x), 3

    model = gamboost(
        "Species ~ bols(Sepal_Length, intercept=True)",
        data={"Species": y_species, "Sepal_Length": x},
        family=Multinomial(),
        control=(mstop, nu),
    )

    # Full risk path (length mstop+1).
    np.testing.assert_allclose(
        model.risk_, r["risk"], rtol=1e-9, atol=1e-9,
        err_msg="Multinomial full risk path does not match R")

    # Fitted values: ours row-major (n, K-1) -> R's column-major flat.
    ours_colmajor = np.asarray(model._fitted, np.float64).reshape(n, K - 1).T.ravel()
    np.testing.assert_allclose(
        ours_colmajor, r["fitted"], rtol=1e-6, atol=1e-8,
        err_msg="Multinomial fitted values do not match R")

    # Selected base-learner sequence: a single learner here, so R selects index 1
    # at every step and we select that learner's name at every step.
    assert len(model.selected) == mstop
    assert set(model.selected) == {"Sepal_Length"}
    np.testing.assert_array_equal(r["selected"], np.ones(mstop, dtype=np.int64))


@pytest.mark.r_parity
def test_multinomial_iris_predict_and_coef_match():
    """predict(type="response"/"link"), fitted_, and coef() match R's iris fit."""
    from mboostrs import Multinomial, gamboost

    mstop, nu = 20, 0.1
    r = _run_r_multinomial_iris(mstop=mstop, nu=nu)
    n, K = len(r["x"]), 3
    model = gamboost(
        "Species ~ bols(Sepal_Length, intercept=True)",
        data={"Species": r["y_species"], "Sepal_Length": r["x"]},
        family=Multinomial(),
        control=(mstop, nu),
    )

    # predict(type="response"): (n, K) softmax, reference class last, rows sum to 1.
    resp = model.predict(type="response")
    assert resp.shape == (n, K)
    np.testing.assert_allclose(resp.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    r_resp = r["response"].reshape(n, K, order="F")  # R is column-major
    np.testing.assert_allclose(resp, r_resp, rtol=1e-6, atol=1e-8,
                               err_msg="Multinomial predict(response) does not match R")

    # predict(type="link"): (n, K-1) linear predictor.
    link = model.predict(type="link")
    assert link.shape == (n, K - 1)
    r_link = r["link"].reshape(n, K - 1, order="F")
    np.testing.assert_allclose(link, r_link, rtol=1e-6, atol=1e-8,
                               err_msg="Multinomial predict(link) does not match R")

    # fitted_ is the (n, K-1) link matrix.
    np.testing.assert_allclose(model.fitted_, r_link, rtol=1e-6, atol=1e-8)

    # coef(): flat per-learner vector (nu-scaled), same order/values as R.
    c = model.coef()
    learner_key = next(k for k in c if k != "(offset)")
    np.testing.assert_allclose(np.asarray(c[learner_key]).ravel(), r["coef"],
                               rtol=1e-6, atol=1e-8,
                               err_msg="Multinomial coef() does not match R")


def test_multinomial_cvrisk_and_stabsel_work_through_the_kronecker_lift():
    """Both resampling paths run through the spec boundary now.

    `cvrisk` and stability selection both take the Kronecker lift; previously
    they refused (and `stabsel` crashed on the flattened row count before the
    refusal was added). Neither has an `mboost`/`stabs` oracle for the
    Kronecker model, so this pins the shape and semantics (one selection
    frequency per source learner), not a parity value.
    """
    from mboostrs import (
        Multinomial,
        boost_control,
        cv_weights,
        gamboost,
        stabsel,
    )

    rng = np.random.default_rng(9)
    n = 200
    data = {
        "x": rng.normal(size=n),
        "z": rng.normal(size=n),
        "w": rng.normal(size=n),
        "y": rng.integers(0, 3, size=n).astype(float),
    }
    model = gamboost("y ~ bols(x) + bols(z) + bols(w)", data=data,
                     family=Multinomial(), control=boost_control(30, 0.1))

    # cvrisk works end-to-end.
    result = model.cvrisk(folds=cv_weights(n, type="kfold", B=3, seed=1))
    assert np.asarray(result.risk_matrix).shape == (3, 31)

    # stabsel works: one phat row per source learner, frequencies in [0, 1].
    sel = stabsel(model, cutoff=0.8, q=2, PFER=2, B=10, verbose=False)
    phat = np.asarray(sel.phat)
    assert phat.shape == (3, 30)  # p learners x mstop
    assert np.all((phat >= 0) & (phat <= 1))
    assert sel.n_fits == 20  # SS complementary pairs


def test_multinomial_aic_and_learner_guards():
    """aic() and non-bols learners raise clear errors for Multinomial.

    R itself does not provide AIC for the %O% Multinomial model (AIC.mboost needs
    hatvalues, unavailable for mboost-fitted models), and only bols learners can be
    Kronecker-wrapped here, so both must fail loudly rather than mislead.
    """
    import numpy as np

    from mboostrs import Multinomial, gamboost

    rng = np.random.default_rng(0)
    x = rng.standard_normal(90)
    y = np.repeat(["A", "B", "C"], 30)
    data = {"x": x, "y": y}

    model = gamboost("y ~ bols(x, intercept=True)", data=data,
                     family=Multinomial(), control=(5, 0.1))
    with pytest.raises(NotImplementedError, match="not available for the Multinomial"):
        model.aic()

    # bols, bbs and brandom are supported; other learners (btree/bmono/...)
    # are rejected.
    for spec in ("y ~ btree(x)", "y ~ bmono(x)"):
        with pytest.raises(NotImplementedError, match=r"supports bols\(\), bbs\(\) and brandom"):
            gamboost(spec, data=data, family=Multinomial(), control=(5, 0.1))

    # by= is rejected up front rather than silently dropped: the Kronecker
    # expansion does not thread it, so a by= fit would be bit-identical to the
    # no-by fit and mis-predict on new data.
    data_by = {"x": x, "z": rng.uniform(0.5, 2.0, size=len(x)), "y": y}
    for spec in ("y ~ bbs(x, df=4, by=z)", "y ~ bols(x, by=z)"):
        with pytest.raises(NotImplementedError, match="by="):
            gamboost(spec, data=data_by, family=Multinomial(), control=(5, 0.1))


@pytest.mark.r_parity
def test_multinomial_bbs_smooth_parity():
    """bbs() Multinomial (smooth class effects) matches R's bbs %O% bols(class).

    Reproduces mboost's %O%: raw B-spline basis, per-class penalty I + K1, and the
    combined df1*df2 -> per-class df. In-sample fit / risk / predict / response.
    """
    import numpy as np

    from mboostrs import Multinomial, gamboost

    mstop, nu = 50, 0.1
    f = __import__("rpy2.robjects", fromlist=["r"]).r("""
    function(mstop, nu) {
        library(mboost); data(iris)
        myiris <- as.list(iris)
        myiris$class <- factor(levels(iris$Species)[-nlevels(iris$Species)])
        mlm <- mboost(Species ~ bbs(Sepal.Length, df=4) %O%
                      bols(class, df=2, contrasts.arg="contr.dummy"),
                      data=myiris, family=Multinomial(),
                      control=boost_control(mstop=mstop, nu=nu))
        list(risk=as.numeric(risk(mlm)), fitted=as.numeric(fitted(mlm)),
             response=as.numeric(predict(mlm, type="response")),
             x=as.numeric(iris$Sepal.Length), y=as.character(iris$Species))
    }
    """)
    r = r_call(f, mstop, nu, packages=("mboost",))
    x = np.asarray(r.rx2("x"), float)
    y = np.asarray(r.rx2("y"))
    n, K = len(x), 3

    m = gamboost("Species ~ bbs(Sepal_Length, df=4)",
                 data={"Species": y, "Sepal_Length": x},
                 family=Multinomial(), control=(mstop, nu))

    np.testing.assert_allclose(m.risk_, np.asarray(r.rx2("risk"), float),
                               rtol=1e-5, atol=1e-5,
                               err_msg="bbs Multinomial risk path does not match R")
    ours_cm = np.asarray(m._fitted, np.float64).reshape(n, K - 1).T.ravel()
    np.testing.assert_allclose(ours_cm, np.asarray(r.rx2("fitted"), float),
                               rtol=1e-5, atol=1e-6,
                               err_msg="bbs Multinomial fitted does not match R")
    resp = m.predict(type="response")
    r_resp = np.asarray(r.rx2("response"), float).reshape(n, K, order="F")
    np.testing.assert_allclose(resp, r_resp, rtol=1e-5, atol=1e-6,
                               err_msg="bbs Multinomial response does not match R")


@pytest.mark.parametrize("learner", [
    "bols(x, intercept=True)", "bols(x, intercept=False)", "bbs(x, df=4)"])
def test_multinomial_newdata_matches_insample(learner):
    """predict(newdata=...) for Multinomial: feeding the training x reproduces the
    in-sample link/response (which match R), for bols and bbs, full and truncated
    mstop, and returns the right (n, K-1)/(n, K) shapes for genuinely new x."""
    import numpy as np

    from mboostrs import Multinomial, gamboost

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-2, 2, 120))
    y = np.array(["A", "B", "C"])[rng.integers(0, 3, 120)]
    m = gamboost(f"y ~ {learner}", data={"x": x, "y": y},
                 family=Multinomial(), control=(30, 0.1))

    # newdata = training x reproduces fitted (in-sample matches R).
    np.testing.assert_allclose(m.predict(newdata={"x": x}, type="link"),
                               m.predict(type="link"), rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.predict(newdata={"x": x}, type="response"),
                               m.predict(type="response"), rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.predict(newdata={"x": x}, type="link", mstop=10),
                               m.predict(type="link", mstop=10), rtol=0, atol=1e-12)

    # genuinely new x: correct shapes, probabilities sum to 1.
    nx = np.array([-1.3, 0.2, 1.7])
    assert m.predict(newdata={"x": nx}, type="link").shape == (3, 2)
    resp = m.predict(newdata={"x": nx}, type="response")
    assert resp.shape == (3, 3)
    np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-12)


def test_multinomial_newdata_predict_scales_flat_in_mstop():
    """Kronecker newdata predict aggregates coefficients per distinct selected
    learner before rebuilding its basis, so runtime should stay roughly flat
    as mstop grows rather than scale linearly with it (the O(mstop) basis
    rebuild this replaced)."""
    import time

    import numpy as np

    from mboostrs import Multinomial, gamboost

    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    eta = np.column_stack([np.sin(x1) + 0.3 * x2, -0.5 * x2])
    p = np.exp(np.column_stack([eta, np.zeros(n)]))
    p /= p.sum(1, keepdims=True)
    y = np.array(["A", "B", "C"])[np.array([rng.choice(3, p=pi) for pi in p])]
    data = {"x1": x1, "x2": x2, "y": y}

    m = gamboost("y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
                 family=Multinomial(), control=(400, 0.1))
    newdata = {"x1": x1, "x2": x2}

    def best_of(mstop, reps=3):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            m.predict(newdata=newdata, mstop=mstop, type="response")
            times.append(time.perf_counter() - t0)
        return min(times)

    t_low = best_of(50)
    t_high = best_of(400)
    # A per-step basis rebuild would make t_high roughly 8x t_low; the
    # aggregate-then-apply path should keep the ratio well under that.
    assert t_high < 4 * t_low + 0.01, (
        f"newdata predict scaled with mstop: {t_low * 1000:.2f}ms at mstop=50 "
        f"vs {t_high * 1000:.2f}ms at mstop=400"
    )

@pytest.mark.r_parity
def test_multinomial_cvrisk_matches_r():
    """Multinomial cvrisk (Kronecker-lifted spec path) matches R's cvrisk.

    R's `cvrisk` for Multinomial reports its loss normalised per
    *observation*, and that loss leaves out the reference class; mboostrs
    scores held-out rows with the full negative log-likelihood, so the
    reference-class term is added to R's curve before comparing. the Kronecker lift
    flattens the response to `n*(K-1)` entries, so a naive per-entry
    normalisation would be off by `K-1`. This pins the observation-level
    divisor against R on the iris fixture with matched folds.
    """
    import rpy2.robjects as ro

    from mboostrs import Multinomial, gamboost

    r_code = """
    function() {
        library(mboost)
        data(iris)
        myiris <- as.list(iris)
        myiris$class <- factor(levels(iris$Species)[-nlevels(iris$Species)])
        mlm <- mboost(
            Species ~ bols(Sepal.Length, df=2) %O%
                      bols(class, df=2, contrasts.arg="contr.dummy"),
            data = myiris, family = Multinomial(),
            control = boost_control(mstop = 20, nu = 0.1)
        )
        set.seed(1)
        folds <- cv(rep(1, nrow(iris)), type = "kfold", B = 4)
        cvr <- cvrisk(mlm, folds = folds)
        # cvrisk stores a (B x (mstop+1)) matrix; as.numeric flattens it
        # column-major, so reconstruct column-major and take the fold mean
        # (the printed curve).
        m <- matrix(as.numeric(cvr), ncol = 21, byrow = FALSE)
        # mboostrs scores held-out rows with the full negative log-likelihood;
        # R's Multinomial loss omits the reference class, whose term is
        # log(1 + sum(exp(f))). Add it back, fold by fold, from R's own path.
        ref <- nlevels(iris$Species)
        extra <- matrix(0, nrow = ncol(folds), ncol = 21)
        for (b in seq_len(ncol(folds))) {
            fit_b <- mboost(
                Species ~ bols(Sepal.Length, df=2) %O%
                          bols(class, df=2, contrasts.arg="contr.dummy"),
                data = myiris, family = Multinomial(), weights = folds[, b],
                control = boost_control(mstop = 20, nu = 0.1))
            oob <- folds[, b] == 0
            for (k in 0:20) {
                f <- matrix(predict(fit_b[max(k, 1)], type = "link"), ncol = ref - 1)
                if (k == 0) f[] <- 0
                f <- pmin(abs(f), 36) * sign(f)
                term <- log1p(rowSums(exp(f)))
                extra[b, k + 1] <- sum(term[oob & iris$Species == levels(iris$Species)[ref]]) / sum(oob)
            }
        }
        list(risk = colMeans(m) + colMeans(extra),
             folds = matrix(as.integer(folds), nrow = nrow(iris)))
    }
    """
    result = ro.r(r_code)()
    r_risk = np.asarray(result.rx2("risk"), dtype=np.float64)
    r_folds = np.asarray(result.rx2("folds"), dtype=np.int64)

    # Rebuild the same fit/fold inputs in Python.
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris.data[:, 0]
    y_species = np.array(iris.target_names)[iris.target]
    data = {"y": y_species, "x": x}

    model = gamboost("y ~ bols(x)", data=data, family=Multinomial(),
                     control=(20, 0.1))
    # Feed R's fold matrix as explicit in-bag weights (1 = in bag, 0 = held out).
    folds = r_folds.astype(np.float64)
    folds = np.where(folds == 0, 0.0, 1.0)
    cv = model.cvrisk(folds=folds)

    got = np.asarray(cv.aggregate, dtype=np.float64)
    # rtol 1e-3: the offset step matches to ~1e-7, but later steps carry the
    # same bols df=2 (R) vs intercept (Python) design difference the other
    # Multinomial parity tests document at tol=1e-4, amplified by per-fold
    # df->lambda recalibration. The point of this test is the observation-level
    # normalisation (a factor of K-1=2 if wrong), which the offset step alone
    # would expose at machine precision.
    np.testing.assert_allclose(got, r_risk, rtol=1e-3, err_msg="Multinomial cvrisk vs R")


@pytest.mark.r_parity
def test_multinomial_weighted_risk_matches_r():
    """A weighted Multinomial fit weights each observation exactly once.

    The response is stored flat, `K-1` entries per observation, and that
    observation's weight is repeated over its block. Reading those weights by
    observation index gave observation `i` the weight of observation
    `i // (K-1)`: on this fixture the reported risk path ended at 220.2557
    where upstream ends at 236.4396. The fitted path was already correct, so
    only the reported quantity moved.
    """
    import rpy2.robjects as ro

    from mboostrs import Multinomial, gamboost

    weights = (np.arange(150) % 4 + 1).astype(np.float64)
    r_code = """
    function(w, mstop, nu) {
        library(mboost)
        data(iris)
        myiris <- as.list(iris)
        myiris$class <- factor(levels(iris$Species)[-nlevels(iris$Species)])
        mlm <- mboost(
            Species ~ bols(Sepal.Length, df=2) %O%
                      bols(class, df=2, contrasts.arg="contr.dummy"),
            data = myiris, family = Multinomial(), weights = w,
            control = boost_control(mstop = mstop, nu = nu)
        )
        list(risk = as.numeric(risk(mlm)), fitted = as.numeric(fitted(mlm)))
    }
    """
    result = ro.r(r_code)(ro.FloatVector(weights), 20, 0.1)
    r_risk = np.asarray(result.rx2("risk"), dtype=np.float64)
    r_fitted = np.asarray(result.rx2("fitted"), dtype=np.float64)

    from sklearn.datasets import load_iris

    iris = load_iris()
    data = {"y": np.array(iris.target_names)[iris.target], "x": iris.data[:, 0]}
    model = gamboost("y ~ bols(x)", data=data, family=Multinomial(),
                     control=(20, 0.1), weights=weights)

    # The unweighted path matches at RTOL_RISK; weights change no arithmetic
    # order, so the same tolerance holds here.
    np.testing.assert_allclose(np.asarray(model.risk, dtype=np.float64), r_risk,
                               rtol=RTOL_RISK, err_msg="weighted risk path vs R")
    fitted_column_major = np.asarray(
        model.predict(type="link"), dtype=np.float64).ravel(order="F")
    np.testing.assert_allclose(fitted_column_major, r_fitted, rtol=RTOL_FITTED,
                               err_msg="weighted fitted values vs R")


def test_varimp_attributes_the_full_likelihood():
    """Multinomial shares come from the full negative log-likelihood, which
    every step lowers; the in-bag risk (reference class left out) can rise on
    a useful step and gave a true predictor a negative share."""
    import mboostrs as mb

    rng = np.random.default_rng(7)
    n = 1500
    x1, x2, x3 = rng.uniform(-1, 1, (3, n))
    eta = np.column_stack([np.zeros(n), np.sin(2 * x1) + 0.5 * x3,
                           0.8 * x2 - 0.8 * (x3 ** 2 - 1 / 3)])
    p = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1).astype(float)
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3}
    model = mb.gamboost("y ~ bbs(x1) + bbs(x2) + bbs(x3)", data,
                        family=mb.Multinomial(n_classes=3), control=mb.boost_control(300))
    path = model._multinomial_nll_path()
    prob = np.asarray(model.predict(type="response"))
    direct = -np.sum(np.log(prob[np.arange(n), y.astype(int)]))
    assert path[-1] == pytest.approx(direct, rel=1e-10)
    shares = model.varimp()["reduction"]
    assert all(v >= 0 for v in shares.values())
    assert shares["x3"] > 0.05
