"""scikit-learn compatibility tests for mboostrs.sklearn estimators."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from mboostrs.sklearn import (
    GAMBoostClassifier,
    GAMBoostRegressor,
    GLMBoostClassifier,
    GLMBoostRegressor,
    LSSRegressor,
    SurvivalBoost,
    TreeBoostClassifier,
    TreeBoostRegressor,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _reg_data(n=300, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.6, size=p)
    y = X @ beta + rng.normal(scale=0.3, size=n)
    return X, y


def _clf_data(n=300, p=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logit = X @ np.array([1.0, -0.8, 0.5, 0.0])[:p]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_glm_regressor_fit_predict():
    X, y = _reg_data()
    est = GLMBoostRegressor(mstop=100, nu=0.1).fit(X, y)
    check_is_fitted(est, "model_")
    pred = est.predict(X)
    assert pred.shape == (X.shape[0],)
    assert est.n_features_in_ == X.shape[1]
    # explains most of the (linear) signal
    assert est.score(X, y) > 0.8


def test_gam_regressor_nonlinear():
    rng = np.random.default_rng(3)
    n = 400
    X = rng.uniform(-2, 2, size=(n, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=0.2, size=n)
    est = GAMBoostRegressor(df=4, mstop=200, nu=0.1).fit(X, y)
    # the smooth model should beat a purely linear fit on this nonlinear target
    lin = GLMBoostRegressor(mstop=200, nu=0.1).fit(X, y)
    assert est.score(X, y) > lin.score(X, y)


def test_clf_fit_predict_proba():
    X, y = _clf_data()
    est = GLMBoostClassifier(mstop=100, nu=0.1).fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    pred = est.predict(X)
    assert set(np.unique(pred)).issubset(set(np.unique(y)))
    assert est.score(X, y) > 0.7


def test_gam_classifier_nonlinear():
    rng = np.random.default_rng(5)
    n = 400
    X = rng.uniform(-2, 2, size=(n, 2))
    # circular (nonlinear) decision boundary
    logit = 3.0 * (1.0 - (X[:, 0] ** 2 + X[:, 1] ** 2) / 2.0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    gam = GAMBoostClassifier(df=4, mstop=200, nu=0.1).fit(X, y)
    glm = GLMBoostClassifier(mstop=200, nu=0.1).fit(X, y)
    proba = gam.predict_proba(X)
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    # smooth model beats linear on a nonlinear boundary
    assert gam.score(X, y) > glm.score(X, y)


def test_clf_preserves_label_values():
    X, y = _clf_data()
    y_lab = np.where(y == 1, "pos", "neg")
    est = GLMBoostClassifier(mstop=50).fit(X, y_lab)
    assert set(est.classes_) == {"neg", "pos"}
    assert set(np.unique(est.predict(X))).issubset({"neg", "pos"})


def test_get_set_params_and_clone():
    est = GLMBoostRegressor(family="poisson", mstop=77, nu=0.05)
    params = est.get_params()
    assert params == {"family": "poisson", "mstop": 77, "nu": 0.05, "tau": None}
    est2 = clone(est)
    assert est2.get_params() == params
    est.set_params(mstop=123)
    assert est.mstop == 123


def test_pipeline_and_cross_val():
    X, y = _reg_data()
    pipe = make_pipeline(StandardScaler(), GLMBoostRegressor(mstop=80))
    scores = cross_val_score(pipe, X, y, cv=4, scoring="r2")
    assert scores.shape == (4,)
    assert np.mean(scores) > 0.7


def test_gridsearch_over_mstop():
    X, y = _reg_data()
    gs = GridSearchCV(
        GLMBoostRegressor(nu=0.1),
        {"mstop": [25, 100, 300]},
        cv=4,
        scoring="r2",
    ).fit(X, y)
    assert gs.best_params_["mstop"] in (25, 100, 300)
    assert hasattr(gs, "best_estimator_")


def test_formula_with_array_positional_names():
    # array input -> columns are f0, f1, ...; formula references them
    rng = np.random.default_rng(7)
    n = 400
    X = rng.uniform(-2, 2, size=(n, 2))
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(scale=0.2, size=n)
    est = GAMBoostRegressor(
        formula="y ~ bbs(f0, df=5) + bols(f1)", mstop=200, nu=0.1
    ).fit(X, y)
    assert est.score(X, y) > 0.5
    assert est.predict(X).shape == (n,)


def test_formula_with_dataframe_real_names():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(8)
    n = 400
    df = pd.DataFrame({
        "age": rng.uniform(-2, 2, n),
        "income": rng.uniform(-2, 2, n),
    })
    y = np.sin(2 * df["age"].to_numpy()) + 0.5 * df["income"].to_numpy() \
        + rng.normal(scale=0.2, size=n)
    est = GAMBoostRegressor(
        formula="y ~ bbs(age, df=5) + bols(income)", mstop=200
    ).fit(df, y)
    assert list(est.feature_names_in_) == ["age", "income"]
    assert est.score(df, y) > 0.5


def test_formula_is_a_tunable_param():
    # formula lives in the constructor → carried by clone / get_params / search
    est = GAMBoostRegressor(formula="y ~ bbs(f0, df=3)")
    assert est.get_params()["formula"] == "y ~ bbs(f0, df=3)"
    assert clone(est).get_params()["formula"] == "y ~ bbs(f0, df=3)"


def test_formula_classifier():
    rng = np.random.default_rng(9)
    n = 400
    X = rng.uniform(-2, 2, size=(n, 2))
    logit = 3.0 * (1.0 - (X[:, 0] ** 2 + X[:, 1] ** 2) / 2.0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    est = GAMBoostClassifier(
        formula="y ~ bbs(f0, df=4) + bbs(f1, df=4)", mstop=200
    ).fit(X, y)
    assert est.predict_proba(X).shape == (n, 2)
    assert est.score(X, y) > 0.7


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        GLMBoostRegressor(family="nonsense").fit(*_reg_data())


def test_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        GLMBoostRegressor().predict(np.zeros((3, 4)))


from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402


@parametrize_with_checks(
    [
        GLMBoostRegressor(),
        GAMBoostRegressor(),
        GLMBoostClassifier(),
        GAMBoostClassifier(),
        TreeBoostRegressor(),
        TreeBoostClassifier(),
        LSSRegressor(),
    ]
)
def test_sklearn_estimator_checks(estimator, check):
    """Run sklearn's full estimator-compliance battery (docs/book/usage/sklearn.ipynb
    demonstrates a subset of this interactively).

    ``GAMBoostClassifier``'s multiclass-underfit-at-default-mstop deviation
    (see ``__sklearn_tags__`` on that class) is expressed as
    ``classifier_tags.poor_score = True``, not a check skip here — that way
    sklearn still runs every other assertion in `check_classifiers_train`
    (fit/predict/predict_proba shapes, malformed-input errors, ...) and only
    the accuracy bar is waived, which is the honest scope of the deviation.

    ``SurvivalBoost`` is deliberately not in this list: its ``fit`` requires a
    structured-array ``y`` incompatible with the plain numeric/categorical
    targets this battery constructs (see its docstring and
    ``test_survival_boost_sklearn_check_failures_are_all_from_the_structured_target``
    below for what that looks like measured, not assumed).
    """
    name = check.func.__name__
    if name == "check_array_api_input":
        pytest.skip("SCIPY_ARRAY_API not set; not exercised in this environment")
    if name == "check_sample_weight_equivalence_on_dense_data" and type(estimator).__name__ in (
        "GLMBoostRegressor", "GAMBoostRegressor", "GAMBoostClassifier",
        "TreeBoostRegressor", "TreeBoostClassifier", "LSSRegressor",
    ):
        # GLMBoostClassifier is NOT skipped here: measured, it passes this check
        # as-is (verified 2026-07-24), so skipping it would hide real,
        # currently-passing coverage. LSSRegressor joined the skip set on
        # 2026-09-05: the LSS fit path now centers by the same unweighted column
        # mean as upstream gamboostLSS (previously it centered by the weighted
        # mean, which is what made the check pass — that behavior deviated from
        # R), so it now fails this check for the identical reason as the others.
        pytest.skip(
            "Deliberate deviation, verified against R mboost 2.9.11 and "
            "gamboostLSS 2.2-0: this check asserts that an integer sample_weight "
            "equals physically replicating the row. glmboost and glmboostLSS "
            "center each predictor by its UNWEIGHTED column mean, so the weighted "
            "frame and the replicated frame get different centering constants and "
            "therefore different paths. R mboost behaves identically (glmboost "
            "coef x1 0.9903 weighted vs 0.9989 replicated; gamboostLSS max "
            "|pw - pr| ~0.07 for non-constant weights, 0.117 with zero weights), "
            "so satisfying this check would mean breaking R parity. Without "
            "centering the equivalence is exact to 1.7e-16 — see "
            "test_case_weights_equal_replication_without_centering."
        )
    check(estimator)


# --- sample_weight ------------------------------------------------------------


def _weighted_setup(n=300, seed=0):
    """Two groups with different slopes, so weighting one group visibly moves
    the fit toward that group's slope."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    group = np.arange(n) % 2 == 0
    y = np.where(group, 3.0 * x, -1.0 * x) + rng.normal(scale=0.2, size=n)
    return x.reshape(-1, 1), y, group


@pytest.mark.parametrize(
    "est_cls", [GLMBoostRegressor, GAMBoostRegressor], ids=lambda c: c.__name__
)
def test_sample_weight_shifts_fit_toward_upweighted_group(est_cls):
    X, y, group = _weighted_setup()

    unweighted = est_cls(mstop=200, nu=0.1).fit(X, y)
    w = np.where(group, 20.0, 1.0)
    weighted = est_cls(mstop=200, nu=0.1).fit(X, y, sample_weight=w)

    # Group A has slope +3, group B slope -1. Up-weighting A must move
    # predictions toward A's steeper positive relationship.
    grid = np.linspace(-2, 2, 25).reshape(-1, 1)
    slope_un = np.polyfit(grid.ravel(), unweighted.predict(grid), 1)[0]
    slope_w = np.polyfit(grid.ravel(), weighted.predict(grid), 1)[0]
    assert slope_w > slope_un + 0.3, (slope_un, slope_w)


def test_case_weights_equal_replication_without_centering():
    """Case-weight semantics, isolated from centering.

    `weights=` are genuine case weights: with no predictor centering, weighting
    a row by k is bit-identical to including it k times. glmboost's default
    centering is what breaks the equivalence (companion test below), not the
    weighting itself.
    """
    import mboostrs as mb

    rng = np.random.default_rng(11)
    m = 60
    x1, x2 = rng.normal(size=m), rng.normal(size=m)
    y = x1 - 0.5 * x2 + rng.normal(scale=0.1, size=m)
    counts = rng.integers(1, 4, size=m)

    formula = "y ~ bols(x1, intercept=False) + bols(x2, intercept=False)"
    ctrl = mb.boost_control(100, 0.1)
    weighted = mb.gamboost(
        formula, data={"x1": x1, "x2": x2, "y": y}, family=mb.Gaussian(),
        control=ctrl, weights=counts.astype(float),
    )
    idx = np.repeat(np.arange(m), counts)
    replicated = mb.gamboost(
        formula, data={"x1": x1[idx], "x2": x2[idx], "y": y[idx]},
        family=mb.Gaussian(), control=ctrl,
    )

    a = np.concatenate([np.ravel(v) for v in weighted.coef().values()])
    b = np.concatenate([np.ravel(v) for v in replicated.coef().values()])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_zero_weight_excludes_an_observation():
    """A zero weight removes an observation's influence entirely.

    Checked without centering so the comparison is exact; with centering the
    zero-weight rows still shift the centering constant (see below).
    """
    import mboostrs as mb

    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(-2, 2, n)
    keep = np.arange(n) % 2 == 0
    y = np.where(keep, 3.0 * x, -1.0 * x) + rng.normal(scale=0.2, size=n)

    formula = "y ~ bols(x, intercept=False)"
    ctrl = mb.boost_control(120, 0.1)
    weighted = mb.gamboost(formula, data={"x": x, "y": y}, family=mb.Gaussian(),
                           control=ctrl, weights=keep.astype(float))
    subset = mb.gamboost(formula, data={"x": x[keep], "y": y[keep]},
                         family=mb.Gaussian(), control=ctrl)

    a = np.concatenate([np.ravel(v) for v in weighted.coef().values()])
    b = np.concatenate([np.ravel(v) for v in subset.coef().values()])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_centering_makes_case_weights_differ_from_replication():
    """Pin the known, R-matching deviation so it cannot change silently.

    glmboost centers each predictor by its *unweighted* column mean. The
    replicated frame has a different unweighted mean than the weighted one, so
    the two fits follow different paths. R mboost 2.9.11 does the same (coef
    x1 0.9903 weighted vs 0.9989 replicated), so removing this deviation would
    mean breaking R parity. Documented, not endorsed.
    """
    rng = np.random.default_rng(11)
    m = 60
    X = rng.normal(size=(m, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(scale=0.1, size=m)
    counts = rng.integers(1, 4, size=m)

    weighted = GLMBoostRegressor(mstop=100, nu=0.1).fit(X, y, sample_weight=counts)
    replicated = GLMBoostRegressor(mstop=100, nu=0.1).fit(
        np.repeat(X, counts, axis=0), np.repeat(y, counts)
    )

    grid = rng.normal(size=(20, 2))
    diff = float(np.max(np.abs(weighted.predict(grid) - replicated.predict(grid))))
    # Both fit the same signal, so they stay close — but not equal. The gap is
    # the centering constant, well above numerical noise.
    assert diff > 1e-6, "centering deviation vanished; re-check against R mboost"
    assert diff < 0.5, f"deviation unexpectedly large ({diff}); investigate"


def test_sample_weight_on_classifiers():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + rng.normal(scale=0.3, size=200) > 0).astype(int)

    for est_cls in (GLMBoostClassifier, GAMBoostClassifier):
        w = rng.uniform(0.5, 2.0, size=200)
        est = est_cls(mstop=60, nu=0.1).fit(X, y, sample_weight=w)
        proba = est.predict_proba(X)
        assert proba.shape == (200, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)


def test_uniform_sample_weight_matches_no_weight():
    X, y, _ = _weighted_setup(n=150, seed=7)
    a = GLMBoostRegressor(mstop=80, nu=0.1).fit(X, y)
    b = GLMBoostRegressor(mstop=80, nu=0.1).fit(X, y, sample_weight=np.ones(len(y)))
    grid = np.linspace(-2, 2, 15).reshape(-1, 1)
    np.testing.assert_allclose(a.predict(grid), b.predict(grid), rtol=1e-12, atol=1e-12)


def test_sample_weight_length_is_validated():
    X, y, _ = _weighted_setup(n=50, seed=2)
    with pytest.raises(ValueError):
        GLMBoostRegressor(mstop=20).fit(X, y, sample_weight=np.ones(10))


# --- multiclass ---------------------------------------------------------------


def _three_class(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    labels = np.array(["setosa", "versicolor", "virginica"])
    y = labels[(X[:, 0] > 0).astype(int) + (X[:, 1] > 0.6).astype(int)]
    return X, y


@pytest.mark.parametrize(
    "est_cls", [GLMBoostClassifier, GAMBoostClassifier], ids=lambda c: c.__name__
)
def test_multiclass_fit_predict_and_proba(est_cls):
    X, y = _three_class()
    est = est_cls(mstop=120, nu=0.1).fit(X, y)

    assert list(est.classes_) == ["setosa", "versicolor", "virginica"]
    proba = est.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
    assert np.all(proba >= 0)

    pred = est.predict(X)
    assert set(np.unique(pred)) <= set(est.classes_)
    assert np.mean(pred == y) > 0.6


@pytest.mark.parametrize(
    "est_cls", [GLMBoostClassifier, GAMBoostClassifier], ids=lambda c: c.__name__
)
def test_multiclass_proba_columns_align_with_classes(est_cls):
    """argmax(predict_proba) must agree with predict.

    This is the property that breaks if the Multinomial column order and
    `classes_` order disagree — the failure mode `_canonical_labels` exists to
    prevent.
    """
    X, y = _three_class(seed=4)
    est = est_cls(mstop=120, nu=0.1).fit(X, y)
    by_proba = est.classes_[np.argmax(est.predict_proba(X), axis=1)]
    np.testing.assert_array_equal(by_proba, est.predict(X))


@pytest.mark.parametrize(
    "labels",
    [
        np.array([1, 2, 10]),          # lexicographic order != numeric order
        np.array(["a", "b", "c"]),
        np.array([0, 1, 2]),
    ],
    ids=["int-unsorted-lexically", "str", "int"],
)
def test_multiclass_label_types_keep_their_ordering(labels):
    """Integer labels like [1, 2, 10] sort lexicographically as 1, 10, 2.

    If the response encoding used raw labels, the probability columns would be
    permuted relative to `classes_` and accuracy would collapse.
    """
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 2))
    idx = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.6).astype(int)
    y = labels[idx]

    est = GLMBoostClassifier(mstop=120, nu=0.1).fit(X, y)
    np.testing.assert_array_equal(est.classes_, np.unique(y))
    assert np.mean(est.predict(X) == y) > 0.6
    by_proba = est.classes_[np.argmax(est.predict_proba(X), axis=1)]
    np.testing.assert_array_equal(by_proba, est.predict(X))


def test_multiclass_gam_accuracy_improves_with_mstop():
    """Pins the reason check_classifiers_train is skipped for GAMBoostClassifier.

    The default mstop=100 underfits sklearn's 3-class blobs; the model is
    correct, it just needs more iterations because multiclass creates (K-1)*p
    base learners to distribute updates across.
    """
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, random_state=0)
    X = StandardScaler().fit_transform(X)

    accs = [
        float(np.mean(GAMBoostClassifier(mstop=m, nu=0.1).fit(X, y).predict(X) == y))
        for m in (100, 200, 400)
    ]
    assert accs[0] < accs[1] < accs[2], accs
    assert accs[1] > 0.83, f"mstop=200 should clear sklearn's bar, got {accs[1]}"


def test_multiclass_respects_sample_weight():
    X, y = _three_class(seed=9)
    w = np.where(y == "setosa", 5.0, 1.0)
    est = GLMBoostClassifier(mstop=100, nu=0.1).fit(X, y, sample_weight=w)
    proba = est.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)


def test_single_class_is_still_rejected():
    X, _ = _three_class(seed=3)
    with pytest.raises(ValueError, match="only one class"):
        GLMBoostClassifier(mstop=20).fit(X, np.zeros(X.shape[0]))


# --- quantile / expectile regression -------------------------------------------


def _heteroscedastic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = 2 * x + rng.normal(scale=0.5 + 0.4 * (x + 2), size=n)
    return x.reshape(-1, 1), y


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_quantile_regressor_is_calibrated(tau):
    """The fitted curve should have ~tau of the mass below it."""
    X, y = _heteroscedastic()
    est = GLMBoostRegressor(family="quantile", tau=tau, mstop=300, nu=0.1).fit(X, y)
    below = float(np.mean(y <= est.predict(X)))
    assert abs(below - tau) < 0.05, (tau, below)


def test_quantile_curves_are_ordered():
    X, y = _heteroscedastic(seed=2)
    preds = {
        tau: GLMBoostRegressor(
            family="quantile", tau=tau, mstop=300, nu=0.1
        ).fit(X, y).predict(X)
        for tau in (0.1, 0.5, 0.9)
    }
    # Quantile curves must not cross on average.
    assert float(np.mean(preds[0.1])) < float(np.mean(preds[0.5]))
    assert float(np.mean(preds[0.5])) < float(np.mean(preds[0.9]))


def test_expectile_regressor_runs_and_is_ordered():
    X, y = _heteroscedastic(seed=3)
    lo = GAMBoostRegressor(family="expectile", tau=0.1, mstop=200).fit(X, y).predict(X)
    hi = GAMBoostRegressor(family="expectile", tau=0.9, mstop=200).fit(X, y).predict(X)
    assert float(np.mean(lo)) < float(np.mean(hi))


def test_tau_is_a_tunable_param():
    est = GLMBoostRegressor(family="quantile", tau=0.25)
    assert est.get_params()["tau"] == 0.25
    assert clone(est).get_params()["tau"] == 0.25


def test_quantile_without_tau_is_rejected():
    X, y = _heteroscedastic(n=100)
    with pytest.raises(ValueError, match="requires tau"):
        GLMBoostRegressor(family="quantile").fit(X, y)


@pytest.mark.parametrize("bad_tau", [0.0, 1.0, -0.2, 1.5])
def test_invalid_tau_is_rejected(bad_tau):
    X, y = _heteroscedastic(n=100)
    with pytest.raises(ValueError, match="tau must be in"):
        GLMBoostRegressor(family="quantile", tau=bad_tau).fit(X, y)


def test_tau_on_a_mean_family_is_rejected_not_ignored():
    X, y = _heteroscedastic(n=100)
    with pytest.raises(ValueError, match="only meaningful"):
        GLMBoostRegressor(family="gaussian", tau=0.9).fit(X, y)


# --- distributional and survival estimators -----------------------------------


def _lss_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 1))
    y = 1.0 + 2.0 * X[:, 0] + rng.normal(scale=np.exp(0.3 * X[:, 0]), size=n)
    return X, y


def test_lss_regressor_predicts_the_location_parameter():
    X, y = _lss_data()
    est = LSSRegressor(mstop=150, nu=0.1).fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (X.shape[0],)
    # mu tracks the true 1 + 2x mean
    assert np.corrcoef(pred, 1.0 + 2.0 * X[:, 0])[0, 1] > 0.95


def test_lss_interval_width_adapts_and_covers():
    """The reason to reach for a distributional model at all."""
    X, y = _lss_data()
    est = LSSRegressor(mstop=200, nu=0.1).fit(X, y)
    lower, upper = est.predict_interval(X, level=0.9)

    width = upper - lower
    assert np.corrcoef(X[:, 0], width)[0, 1] > 0.8, "width did not track the scale"
    coverage = float(np.mean((y >= lower) & (y <= upper)))
    assert 0.84 < coverage < 0.96, coverage


def test_lss_exposes_every_distribution_parameter():
    X, y = _lss_data()
    est = LSSRegressor(mstop=100).fit(X, y)
    params = est.predict_params(X[:5])
    assert set(params) == {"mu", "sigma"}
    assert np.all(np.asarray(params["sigma"]) > 0)


def test_lss_smooth_variant_fits():
    rng = np.random.default_rng(4)
    n = 400
    X = rng.uniform(-2, 2, (n, 1))
    y = np.sin(2 * X[:, 0]) + rng.normal(scale=0.3, size=n)
    est = LSSRegressor(smooth=True, df=4, mstop=150).fit(X, y)
    assert np.corrcoef(est.predict(X), np.sin(2 * X[:, 0]))[0, 1] > 0.8


def test_lss_works_in_a_pipeline():
    X, y = _lss_data()
    pipe = make_pipeline(StandardScaler(), LSSRegressor(mstop=80))
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (X.shape[0],)


def test_lss_unknown_family_is_rejected():
    X, y = _lss_data(n=100)
    with pytest.raises(ValueError, match="unknown LSS family"):
        LSSRegressor(family="nonsense", mstop=20).fit(X, y)


@pytest.mark.parametrize("family", ["weibull", "lognormal", "loglog"])
def test_lss_regressor_routes_survival_families_to_structured_estimator(family):
    X, y = _lss_data(n=100)
    with pytest.raises(ValueError, match="censored survival.*SurvivalBoost"):
        LSSRegressor(family=family, mstop=20).fit(X, y)


def test_lss_distribution_log_score_matches_fitted_density():
    X, y = _lss_data(n=240)
    est = LSSRegressor(mstop=100).fit(X, y)
    score = est.distribution_log_score(X, y)
    params = est.predict_params(X)
    expected = np.mean(np.log(est.model_._family.pdf(y, params)))
    assert score == pytest.approx(expected)
    assert np.isfinite(score)
    # R-squared remains sklearn's location-only regressor convention.
    assert est.score(X, y) != pytest.approx(score)


def _surv_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 1))
    time = rng.exponential(scale=np.exp(-0.5 * X[:, 0]))
    event = rng.uniform(size=n) < 0.7
    y = np.array(
        list(zip(event, time, strict=True)),
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


def test_survival_risk_score_has_the_expected_orientation():
    """Higher x shortens survival, so risk must increase with x.

    A sign error here would silently invert every concordance index computed
    downstream, so the direction is the thing worth asserting.
    """
    X, y = _surv_data()
    est = SurvivalBoost(mstop=100, nu=0.1).fit(X, y)
    risk = est.predict(X)
    assert np.corrcoef(X[:, 0], risk)[0, 1] > 0.5, "risk score orientation is wrong"


def test_survival_accepts_the_structured_target_in_either_field_order():
    X, y = _surv_data()
    est = SurvivalBoost(mstop=50).fit(X, y)
    assert est.predict(X).shape == (X.shape[0],)
    reversed_y = np.empty(len(y), dtype=[("time", float), ("event", bool)])
    reversed_y["time"] = y["time"]
    reversed_y["event"] = y["event"]
    reversed_est = SurvivalBoost(mstop=50).fit(X, reversed_y)
    np.testing.assert_allclose(reversed_est.predict(X), est.predict(X))


def test_survival_rejects_a_plain_target():
    X, _ = _surv_data(n=100)
    with pytest.raises(ValueError, match="structured array"):
        SurvivalBoost(mstop=20).fit(X, np.arange(100.0))


def test_survival_rejects_negative_times():
    X, y = _surv_data(n=100)
    bad = y.copy()
    bad["time"][0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        SurvivalBoost(mstop=20).fit(X, bad)


@pytest.mark.parametrize("bad_time", [np.nan, np.inf])
def test_survival_rejects_nonfinite_times(bad_time):
    X, y = _surv_data(n=100)
    bad = y.copy()
    bad["time"][0] = bad_time
    with pytest.raises(ValueError, match="finite and non-negative"):
        SurvivalBoost(mstop=20).fit(X, bad)


def test_survival_formula_accepts_rhs_or_full_formula_spelling():
    X, y = _surv_data(n=160)
    rhs = SurvivalBoost(formula="bols(f0)", mstop=40).fit(X, y)
    full = SurvivalBoost(
        formula="Surv(time, event) ~ bols(f0)", mstop=40
    ).fit(X, y)
    np.testing.assert_allclose(rhs.predict(X), full.predict(X))


def test_survival_rejects_mismatched_row_counts():
    X, y = _surv_data(n=100)
    with pytest.raises(ValueError, match="rows"):
        SurvivalBoost(mstop=20).fit(X[:50], y)


def test_survival_smooth_variant_fits():
    X, y = _surv_data()
    est = SurvivalBoost(smooth=True, df=4, mstop=80).fit(X, y)
    assert np.all(np.isfinite(est.predict(X)))


def test_survival_unknown_family_is_rejected():
    X, y = _surv_data(n=100)
    with pytest.raises(ValueError, match="unknown survival family"):
        SurvivalBoost(family="nonsense", mstop=20).fit(X, y)


def test_lss_predict_before_fit_raises_not_fitted_error():
    """Regression test for a real bug caught by `check_estimator`.

    `predict` used to read `self.model_.params(newdata=self._frame(X))`,
    which Python evaluates left to right: the `self.model_` attribute lookup
    happened before `self._frame(X)` (which calls `check_is_fitted`) ever
    ran, so an unfitted call raised a bare `AttributeError` instead of
    sklearn's `NotFittedError`. `sklearn.utils.estimator_checks.
    check_estimators_unfitted` caught this immediately once `LSSRegressor`
    was run through the standard estimator checks.
    """
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        LSSRegressor().predict(np.zeros((3, 2)))


def test_adaexp_tree_exposes_a_margin_not_pseudo_probabilities():
    X, y = _clf_data(n=240)
    estimator = TreeBoostClassifier(
        family="adaexp", mstop=80, minsplit=10, minbucket=5
    ).fit(X, y)
    assert not hasattr(estimator, "predict_proba")
    margin = estimator.decision_function(X)
    expected = estimator.classes_[(margin >= 0).astype(int)]
    np.testing.assert_array_equal(estimator.predict(X), expected)
    assert hasattr(TreeBoostClassifier(family="binomial"), "predict_proba")


def test_survival_boost_sklearn_check_failures_are_all_from_the_structured_target():
    """`SurvivalBoost` is intentionally excluded from `parametrize_with_checks`
    above. This test measures why, instead of asserting it.

    `fit` requires a structured array `y` with event/time fields
    (`_split_target`); sklearn's generic check battery constructs plain
    numeric/categorical `y` for every check, so most of the ~90-check battery
    fails immediately inside `_split_target`, before estimator-specific logic
    ever runs. This is the same limitation scikit-survival's own estimators
    have against `check_estimator` (also target=structured array) — it is a
    contract mismatch with the generic battery, not a defect in this class.

    The test pins two things: (1) the actual API-level checks that do NOT
    depend on the y-contract (init/clone/get_params/repr/unfitted-predict)
    still pass, proving the class follows the sklearn estimator API where it
    is even possible to check; and (2) every failing check's exception is
    attributable to `_split_target`'s structured-array requirement, so a
    regression that broke something else (e.g. `get_params` or cloning) would
    surface as a new, unexplained failure rather than being silently folded
    into "expected to fail".
    """
    from sklearn.utils.estimator_checks import check_estimator

    results = check_estimator(SurvivalBoost(mstop=20), on_fail=None, on_skip=None)
    by_status = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    contract_independent_checks = {
        "check_estimator_cloneable",
        "check_estimator_tags_renamed",
        "check_valid_tag_types",
        "check_estimator_repr",
        "check_no_attributes_set_in_init",
        "check_estimators_unfitted",  # predict before fit -> NotFittedError
        "check_do_not_raise_errors_in_init_or_set_params",
        "check_mixin_order",
        "check_parameters_default_constructible",
        "check_get_params_invariance",
        "check_set_params",
    }
    passed_names = {r["check_name"] for r in by_status.get("passed", [])}
    missing = contract_independent_checks - passed_names
    assert not missing, f"contract-independent checks regressed: {missing}"

    assert len(by_status.get("failed", [])) > 20, (
        "expected the bulk of the generic battery to fail on the structured "
        "target contract; a much smaller failure count would mean the "
        "premise of this test (and the decision to exclude SurvivalBoost "
        "from parametrize_with_checks) needs re-checking"
    )

    # Root-cause proof, not inference from failure-message text: `fit` calls
    # `_split_target(y)` as its very first statement, before anything about
    # `X` is even looked at. So *every* generic check — which always builds a
    # plain, non-structured `y` — must fail at that exact line, regardless of
    # what X-specific behavior (sparse, NaN, negative values, dtype, weight
    # container type, ...) the check is nominally probing. Confirm that
    # directly: feed a battery of the X shapes/contents the checks above
    # actually exercise, paired with a plain y, and confirm every one hits
    # the identical structured-array error rather than getting further into
    # `fit` and failing for some X-specific reason.
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n = 20
    plain_y = rng.normal(size=n)  # exactly what the generic checks pass
    hostile_X = {
        "sparse_csr": sp.csr_matrix(rng.normal(size=(n, 2))),
        "negative_values": -np.abs(rng.normal(size=(n, 2))),
        "nan_values": np.where(
            np.arange(n * 2).reshape(n, 2) == 0, np.nan, rng.normal(size=(n, 2))
        ),
        "single_feature": rng.normal(size=(n, 1)),
        "single_sample": rng.normal(size=(1, 2)),
    }
    for label, X in hostile_X.items():
        with pytest.raises(ValueError, match="structured array") as exc_info:
            SurvivalBoost(mstop=20).fit(X, plain_y)
        assert "structured array" in str(exc_info.value), label


# --- set_output --------------------------------------------------------------


_ALL_ESTIMATOR_CLASSES = (
    GLMBoostRegressor, GAMBoostRegressor, GLMBoostClassifier, GAMBoostClassifier,
    LSSRegressor, SurvivalBoost,
)


@pytest.mark.parametrize("cls", _ALL_ESTIMATOR_CLASSES, ids=lambda c: c.__name__)
def test_no_estimator_exposes_set_output(cls):
    """`set_output(transform=...)` is sklearn's output-container API (gained
    `"polars"` in sklearn 1.4). It is wired through
    `sklearn.utils._set_output._SetOutputMixin`, which only `TransformerMixin`
    inherits, and it wraps `transform`/`fit_transform` specifically — see
    `_SetOutputMixin.__init_subclass__` and `_auto_wrap_is_configured`
    (`hasattr(estimator, "get_feature_names_out") and "transform" in
    auto_wrap_output_keys`).

    Every estimator in this module is a predictor: it has `predict` (and,
    for classifiers, `predict_proba`), never `transform`. There is no
    "transformed X" for `set_output` to format as polars/pandas — only a
    prediction — so the honest answer is that the API does not apply here,
    and this class does not define `set_output`, `transform`, or
    `fit_transform`. This test pins that absence so a future accidental
    `transform` method (e.g. copy-pasted from a transformer mixin) would be
    caught rather than silently gaining a half-implemented output API.
    """
    assert not hasattr(cls, "set_output")
    assert not hasattr(cls, "transform")
    assert not hasattr(cls, "fit_transform")
    assert not hasattr(cls, "get_feature_names_out")


# --- __sklearn_tags__ ----------------------------------------------------------


@pytest.mark.parametrize(
    "cls", [GLMBoostRegressor, GAMBoostRegressor, GLMBoostClassifier,
            GAMBoostClassifier, LSSRegressor, SurvivalBoost],
    ids=lambda c: c.__name__,
)
def test_tags_never_claim_sparse_or_nan_support(cls):
    """`_fit_boost` (and the equivalent `fit` on LSSRegressor/SurvivalBoost)
    calls `check_X_y`/`check_array` with the sklearn defaults
    (`accept_sparse=False`, `ensure_all_finite=True`), so a sparse `X` or a
    NaN/Inf value is rejected at the sklearn boundary before it would ever
    reach the Rust engine (which requires dense float64 regardless). The
    tags must say so, not claim support that would immediately raise.
    """
    from sklearn.utils import get_tags

    tags = get_tags(cls())
    assert tags.input_tags.sparse is False
    assert tags.input_tags.allow_nan is False


@pytest.mark.parametrize(
    "cls", [GLMBoostRegressor, GAMBoostRegressor, GLMBoostClassifier,
            GAMBoostClassifier, LSSRegressor, SurvivalBoost],
    ids=lambda c: c.__name__,
)
def test_sparse_input_is_actually_rejected(cls):
    """Behavior backing `input_tags.sparse = False`: fit on sparse X raises,
    rather than silently densifying somewhere downstream or (worse) being
    accepted and producing a wrong result."""
    from scipy import sparse

    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.normal(size=(30, 2)))
    y = rng.normal(size=30)
    if cls is SurvivalBoost:
        y = np.array(
            list(zip(rng.uniform(size=30) < 0.7, rng.exponential(size=30), strict=True)),
            dtype=[("event", bool), ("time", float)],
        )
    with pytest.raises((TypeError, ValueError)):
        cls(mstop=20).fit(X, y)


@pytest.mark.parametrize(
    "cls", [GLMBoostRegressor, GAMBoostRegressor, GLMBoostClassifier,
            GAMBoostClassifier, LSSRegressor, SurvivalBoost],
    ids=lambda c: c.__name__,
)
def test_nan_input_is_actually_rejected(cls):
    """Behavior backing `input_tags.allow_nan = False`. This is a real
    limitation worth pinning explicitly: the core engine's `na_action="omit"`
    (see `glmboost`/`gamboost`) is not reachable through these estimators —
    sklearn's own `check_array` rejects NaN first."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    X[0, 0] = np.nan
    y = rng.normal(size=30)
    if cls is SurvivalBoost:
        y = np.array(
            list(zip(rng.uniform(size=30) < 0.7, rng.exponential(size=30), strict=True)),
            dtype=[("event", bool), ("time", float)],
        )
    with pytest.raises(ValueError):
        cls(mstop=20).fit(X, y)


@pytest.mark.parametrize(
    "cls", [GLMBoostRegressor, GAMBoostRegressor, GLMBoostClassifier, GAMBoostClassifier,
            LSSRegressor],
    ids=lambda c: c.__name__,
)
def test_target_tags_state_single_output_and_required(cls):
    """`predict` always returns one 1-D column and `fit` always requires `y`
    (all five inherit `RegressorMixin`/`ClassifierMixin`, which already set
    `required=True`; this pins the multi_output side that this module adds)."""
    from sklearn.utils import get_tags

    tags = get_tags(cls())
    assert tags.target_tags.required is True
    assert tags.target_tags.multi_output is False
    assert tags.target_tags.single_output is True


def test_survival_boost_target_required_is_corrected_from_the_base_default():
    """`SurvivalBoost` inherits only `BaseEstimator` (deliberately: its
    `predict` output is a risk score, not an estimate of the `y` passed to
    `fit`, so it has no `score` method and `estimator_type` stays honestly
    `None`, unlike the other five). `BaseEstimator.__sklearn_tags__`
    defaults `target_tags.required` to `False`, which would be a lie here —
    `fit` always raises without a structured-array `y` — so `SurvivalBoost`
    overrides it explicitly."""
    from sklearn.utils import get_tags

    tags = get_tags(SurvivalBoost())
    assert tags.target_tags.required is True
    assert tags.estimator_type is None


def test_gam_classifier_poor_score_tag_matches_measured_accuracy():
    """Pins the specific numbers behind `GAMBoostClassifier`'s
    `classifier_tags.poor_score = True`: on sklearn's own
    `check_classifiers_train` benchmark (`make_blobs(n_samples=300,
    random_state=0)`), the binary subset clears the 0.83 bar comfortably but
    the 3-class subset does not, at the default `mstop=100`. `GLMBoostClassifier`
    clears both, so it does NOT set the tag — this test pins that asymmetry
    too, so the tag can't silently drift onto the wrong class."""
    from sklearn.datasets import make_blobs
    from sklearn.utils import get_tags, shuffle

    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    gam_multiclass_acc = (GAMBoostClassifier().fit(X_m, y_m).predict(X_m) == y_m).mean()
    gam_binary_acc = (GAMBoostClassifier().fit(X_b, y_b).predict(X_b) == y_b).mean()
    glm_multiclass_acc = (GLMBoostClassifier().fit(X_m, y_m).predict(X_m) == y_m).mean()

    assert gam_binary_acc > 0.83
    assert gam_multiclass_acc < 0.83
    assert glm_multiclass_acc > 0.83

    assert get_tags(GAMBoostClassifier()).classifier_tags.poor_score is True
    assert get_tags(GLMBoostClassifier()).classifier_tags.poor_score is False
