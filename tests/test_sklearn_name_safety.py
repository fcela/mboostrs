"""The estimators check column names, not positions.

`check_array` returns a bare ndarray, so a frame's column names were gone
before anything compared them with the ones the fit saw -- and this module
then attached the *training* names to the new columns by position. Measured on
a fit over `a` and `b`: `predict(X[["b", "a"]])` returned -2.119 where
`predict(X)` returned 1.039, having multiplied `b`'s values by `a`'s
coefficient. A renamed column was accepted outright: `predict(X.rename(
columns={"a": "z"}))` returned the original numbers as though nothing had
changed. Only a wrong *count* was caught.

`validate_data` does the comparison and raises, which is what every other
scikit-learn estimator does. Matching that matters more than being
accommodating: a pipeline's other steps behave that way, and an estimator that
quietly reorders would make the pipeline's behaviour depend on which step
saw the frame.

Two fitted-attribute conventions were also wrong. `feature_names_in_` was set
to `["f0", "f1"]` for a bare array, so a fitted estimator claimed feature names
it had never been given -- scikit-learn leaves the attribute absent. And a
regressor carried `n_classes_ = None`, which reads as "fitted with classes,
found none" rather than "does not apply".
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")

from mboostrs.sklearn import (  # noqa: E402
    GAMBoostClassifier,
    GAMBoostRegressor,
    GLMBoostClassifier,
    GLMBoostRegressor,
    LSSRegressor,
    SurvivalBoost,
    TreeBoostClassifier,
    TreeBoostRegressor,
)

N = 150


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
    y = 3.0 * X["a"].to_numpy() - 1.0 * X["b"].to_numpy() + rng.normal(size=N) * 0.1
    return X, y


@pytest.fixture(scope="module")
def fitted(frame):
    X, y = frame
    return GLMBoostRegressor(mstop=300, nu=0.1).fit(X, y)


# ─── names are compared, not positions ─────────────────────────────


def test_a_reordered_frame_is_refused(fitted, frame):
    X, _ = frame
    with pytest.raises(ValueError, match="same order as they were in fit"):
        fitted.predict(X[["b", "a"]])


def test_a_renamed_column_is_refused(fitted, frame):
    X, _ = frame
    with pytest.raises(ValueError, match="unseen at fit time"):
        fitted.predict(X.rename(columns={"a": "z"}))


def test_the_reordered_prediction_was_not_merely_different(frame):
    """It was wrong in a specific way: `b`'s values went through `a`'s
    coefficient. The check is that the two orderings would have disagreed, so
    the refusal is preventing an error rather than an inconvenience."""
    X, y = frame
    model = GLMBoostRegressor(mstop=300, nu=0.1).fit(X, y)
    in_order = model.predict(X)
    # What the old code computed: the same values, relabelled by position.
    # Passing an array is how to reach that path now, and scikit-learn warns
    # about the missing names, which is exactly the warning the old code had
    # no chance to emit.
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        positional = model.predict(X[["b", "a"]].to_numpy())
    assert not np.allclose(in_order, positional)
    assert np.max(np.abs(in_order - positional)) > 1.0


def test_the_same_frame_still_predicts(fitted, frame):
    X, _ = frame
    assert np.all(np.isfinite(fitted.predict(X)))
    assert np.all(np.isfinite(fitted.predict(X[["a", "b"]])))


def test_a_wrong_column_count_is_still_refused(fitted, frame):
    X, _ = frame
    with pytest.raises(ValueError):
        fitted.predict(X.assign(c=1.0))


def test_a_bare_array_is_accepted_by_position(frame):
    """Fitting on an array records no names, so predicting with one cannot be
    a name mismatch -- the count is all there is to check."""
    X, y = frame
    model = GLMBoostRegressor(mstop=50).fit(X.to_numpy(), y)
    assert np.all(np.isfinite(model.predict(X.to_numpy())))
    with pytest.raises(ValueError, match="features, but"):
        model.predict(np.column_stack([X.to_numpy(), np.ones(N)]))


def test_a_polars_frame_is_name_checked_too(frame):
    X, y = frame
    frame_pl = pl.DataFrame({"a": X["a"].to_numpy(), "b": X["b"].to_numpy()})
    model = GLMBoostRegressor(mstop=50).fit(frame_pl, y)
    assert np.all(np.isfinite(model.predict(frame_pl)))
    with pytest.raises(ValueError, match="same order|unseen at fit"):
        model.predict(frame_pl.select(["b", "a"]))


@pytest.mark.parametrize("estimator", [
    GLMBoostRegressor, GAMBoostRegressor, TreeBoostRegressor,
])
def test_every_regressor_checks_names(estimator, frame):
    X, y = frame
    model = estimator(mstop=40).fit(X, y)
    with pytest.raises(ValueError, match="same order as they were in fit"):
        model.predict(X[["b", "a"]])


@pytest.mark.parametrize("estimator", [
    GLMBoostClassifier, GAMBoostClassifier, TreeBoostClassifier,
])
def test_every_classifier_checks_names(estimator, frame):
    X, y = frame
    labels = (y > 0).astype(int)
    model = estimator(mstop=40).fit(X, labels)
    with pytest.raises(ValueError, match="same order as they were in fit"):
        model.predict(X[["b", "a"]])
    with pytest.raises(ValueError, match="same order as they were in fit"):
        model.predict_proba(X[["b", "a"]])


def test_the_distributional_regressor_checks_names(frame):
    X, y = frame
    model = LSSRegressor(mstop=40).fit(X, y)
    assert np.all(np.isfinite(model.predict(X)))
    with pytest.raises(ValueError, match="same order as they were in fit"):
        model.predict(X[["b", "a"]])


def test_the_survival_estimator_checks_names(frame):
    X, _ = frame
    rng = np.random.default_rng(1)
    time = np.exp(0.5 * X["a"].to_numpy() + rng.exponential(size=N))
    event = (rng.uniform(size=N) < 0.7).astype(float)
    target = np.array(list(zip(event.astype(bool), time, strict=True)),
                      dtype=[("event", bool), ("time", float)])
    model = SurvivalBoost(mstop=40).fit(X, target)
    assert np.all(np.isfinite(model.predict(X)))
    with pytest.raises(ValueError, match="same order as they were in fit"):
        model.predict(X[["b", "a"]])


# ─── fitted-attribute conventions ──────────────────────────────────


def test_feature_names_are_recorded_only_when_they_were_given(frame):
    X, y = frame
    named = GLMBoostRegressor(mstop=30).fit(X, y)
    np.testing.assert_array_equal(named.feature_names_in_, np.array(["a", "b"],
                                                                    dtype=object))
    bare = GLMBoostRegressor(mstop=30).fit(X.to_numpy(), y)
    assert not hasattr(bare, "feature_names_in_"), (
        "an estimator fitted on an array has no feature names to report"
    )
    assert bare.n_features_in_ == 2


def test_a_regressor_has_no_class_count(frame):
    X, y = frame
    model = GLMBoostRegressor(mstop=30).fit(X, y)
    assert not hasattr(model, "n_classes_")


def test_a_classifier_does(frame):
    X, y = frame
    model = GLMBoostClassifier(mstop=30).fit(X, (y > 0).astype(int))
    assert model.n_classes_ == 2
    assert len(model.classes_) == 2


# ─── the rest of the protocol still holds ──────────────────────────


def test_sparse_input_is_refused_with_the_fix_named(frame):
    """The engine takes a dense f64 design; densifying silently would turn a
    memory-shaped choice into a surprise."""
    sparse = pytest.importorskip("scipy.sparse")
    X, y = frame
    with pytest.raises(TypeError, match="toarray"):
        GLMBoostRegressor(mstop=10).fit(sparse.csr_matrix(X.to_numpy()), y)


def test_sample_weight_is_routable(frame):
    from sklearn import config_context

    X, y = frame
    with config_context(enable_metadata_routing=True):
        model = GLMBoostRegressor(mstop=20).set_fit_request(sample_weight=True)
        assert model.get_metadata_routing().fit.requests["sample_weight"] is True


def test_sample_weight_reaches_the_fit(frame):
    X, y = frame
    weights = np.ones(N)
    weights[N // 2:] = 0.0
    weighted = GLMBoostRegressor(mstop=100).fit(X, y, sample_weight=weights)
    plain = GLMBoostRegressor(mstop=100).fit(X, y)
    assert not np.allclose(weighted.predict(X), plain.predict(X))


def test_the_html_repr_works(frame):
    X, y = frame
    html = GLMBoostRegressor(mstop=20).fit(X, y)._repr_html_()
    assert "GLMBoostRegressor" in html


def test_the_validation_shim_covers_both_scikit_learn_apis():
    """`validate_data` was a `BaseEstimator` method up to 1.5, is a free
    function from 1.6, and the method is gone in 1.9. The package supports
    >= 1.3, so both are reached."""
    import inspect

    from mboostrs import sklearn as adapter

    source = inspect.getsource(adapter._validate)
    assert "from sklearn.utils.validation import validate_data" in source
    assert "est._validate_data(" in source
    assert "except ImportError" in source


def test_the_target_type_shim_covers_the_1_3_signature():
    """`type_of_target` gained `raise_unknown` after the 1.3 minimum. The
    shim falls back to the 1.3 signature when the keyword is unknown, which is
    how the classifier fit path stays green on the minimum. Under 1.6+ it
    forwards, so `unknown` targets raise instead of being returned."""
    import inspect as _inspect

    from sklearn.utils.multiclass import type_of_target as installed

    try:
        installed([1, 2, 3], input_name="y", raise_unknown=True)
        signature_has_raise_unknown = True
    except TypeError:
        signature_has_raise_unknown = False

    from mboostrs.sklearn import _type_of_target

    source = _inspect.getsource(_type_of_target)
    assert "raise_unknown=True" in source
    assert "except TypeError" in source
    # `multiclass` is a supported target in every release, so the shim agrees
    # with the installed API on that input regardless of the branch it takes.
    assert _type_of_target(np.array([0, 1, 2])) == "multiclass"
    assert signature_has_raise_unknown in (True, False)


def test_the_estimators_still_pass_a_pipeline_round_trip(frame):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = frame
    pipeline = make_pipeline(StandardScaler(), GLMBoostRegressor(mstop=50))
    pipeline.fit(X, y)
    assert np.all(np.isfinite(pipeline.predict(X)))
