"""The scikit-learn adapters as nuisance learners for EconML and DoubleML.

The recommended causal route fits the outcome and treatment models of EconML
or DoubleML with mboostrs estimators. Two groups of tests live here:

- Adapter tests (no causal library needed): the column names the generated
  formula uses. A feature named ``y`` used to be overwritten by the response in
  the data dictionary, so the fit silently used the response as a covariate. A
  column name with a space or a parenthesis made the formula parser fail.
  A non-numeric column failed with numpy's message, which names neither the
  column nor a remedy.
- Library tests, skipped unless ``econml`` or ``doubleml`` is installed: the
  estimators run with the adapters as nuisance models, clone and pickle, and
  recover a planted average effect.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from mboostrs.sklearn import (
    GAMBoostClassifier,
    GAMBoostRegressor,
    GLMBoostClassifier,
    GLMBoostRegressor,
)

# Cross-fitting predicts each held-out fold, and a held-out covariate outside
# the training range gives gamboost's ExtrapolationWarning on every fold. The
# library tests check the estimates, so the warning is silenced here only.
pytestmark = pytest.mark.filterwarnings(
    "ignore::mboostrs._extrapolation.ExtrapolationWarning"
)

N = 800
def _design(n=N, seed=1):
    """Confounded binary treatment with a heterogeneous effect 2 + X1.

    The treatment depends on X0, which also enters the outcome, so a naive
    difference in means is biased. The average effect over the sample is
    returned with the data.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1.0 / (1.0 + np.exp(-X[:, 0]))
    t = (rng.uniform(size=n) < p).astype(int)
    tau = 2.0 + X[:, 1]
    y = X[:, 0] + tau * t + np.sin(X[:, 1]) + rng.normal(size=n) * 0.5
    return X, t, y, float(np.mean(tau))


X, T, Y, ATE = _design()


# ─── adapter: formula column names ─────────────────────────────────────────


def test_a_feature_named_y_is_not_overwritten_by_the_response():
    # Before the fix the column "y" was replaced by the response in the data
    # dictionary, so this prediction differed from the NumPy fit by about 2.7.
    frame = pd.DataFrame(X, columns=["y", "x2", "x3"])
    via_frame = GAMBoostRegressor(mstop=100).fit(frame, Y).predict(frame)
    via_array = GAMBoostRegressor(mstop=100).fit(X, Y).predict(X)
    np.testing.assert_allclose(via_frame, via_array, rtol=0, atol=1e-10)


def test_names_with_spaces_and_parentheses_fit_like_plain_names():
    plain = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    awkward = pd.DataFrame(X, columns=["age years", "log(x)", "z-score"])
    a = GAMBoostRegressor(mstop=100).fit(plain, Y).predict(plain)
    b = GAMBoostRegressor(mstop=100).fit(awkward, Y).predict(awkward)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def test_integer_column_names_fit_like_a_bare_array():
    frame = pd.DataFrame(X, columns=[0, 1, 2])
    via_frame = GLMBoostRegressor(mstop=200).fit(frame, Y).predict(frame)
    via_array = GLMBoostRegressor(mstop=200).fit(X, Y).predict(X)
    np.testing.assert_allclose(via_frame, via_array, rtol=0, atol=1e-10)


def test_a_formula_refuses_a_column_name_it_cannot_read():
    frame = pd.DataFrame(X, columns=["age years", "x2", "x3"])
    est = GAMBoostRegressor(formula="y ~ bbs(x2, df=4) + bbs(x3, df=4)")
    with pytest.raises(ValueError, match="age_years"):
        est.fit(frame, Y)


def test_a_formula_refuses_a_column_named_like_the_response():
    frame = pd.DataFrame(X, columns=["resp", "x2", "x3"])
    est = GAMBoostRegressor(formula="resp ~ bbs(x2, df=4)")
    with pytest.raises(ValueError, match="same name as the response"):
        est.fit(frame, Y)


def test_a_formula_still_fits_when_names_are_plain():
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    est = GAMBoostRegressor(formula="y ~ bbs(x1, df=5) + bols(x2)").fit(frame, Y)
    assert est.predict(frame).shape == (N,)


def test_numeric_and_integer_coded_columns_fit_without_error():
    rng = np.random.default_rng(4)
    binary = (rng.uniform(size=N) < 0.4).astype(float)
    codes = rng.integers(0, 3, N).astype(float)
    design = np.column_stack([X[:, 0], binary, codes])
    est = GAMBoostRegressor(mstop=100).fit(design, Y)
    assert np.all(np.isfinite(est.predict(design)))


# ─── adapter: non-numeric columns ──────────────────────────────────────────


def test_a_string_column_is_refused_with_the_column_and_the_remedy():
    frame = pd.DataFrame({"g": ["a", "b", "c"] * (N // 3) + ["a", "b"], "x": X[:, 0]})
    with pytest.raises(ValueError, match="column 'g' is not numeric") as info:
        GLMBoostRegressor().fit(frame, Y)
    assert "pd.get_dummies" in str(info.value)


def test_a_category_column_is_refused_at_predict_time_too():
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    est = GLMBoostRegressor(mstop=50).fit(frame, Y)
    bad = frame.assign(x3=pd.Categorical(["u", "v"] * (N // 2)))
    with pytest.raises(ValueError, match="column 'x3' is not numeric"):
        est.predict(bad)


# ─── library tests ─────────────────────────────────────────────────────────


@pytest.fixture
def econml():
    return pytest.importorskip("econml")


@pytest.fixture
def doubleml():
    return pytest.importorskip("doubleml")


def _econml_lineardml(model_y, model_t):
    from econml.dml import LinearDML

    return LinearDML(
        model_y=model_y, model_t=model_t, discrete_treatment=True, cv=3, random_state=0
    )


def test_linear_dml_with_gam_nuisances_recovers_the_average_effect(econml):
    est = _econml_lineardml(GAMBoostRegressor(), GAMBoostClassifier())
    est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
    ate = float(np.mean(est.effect(X[:, :1])))
    assert abs(ate - ATE) < 0.25


def test_linear_dml_with_glm_nuisances_recovers_the_average_effect(econml):
    est = _econml_lineardml(GLMBoostRegressor(), GLMBoostClassifier())
    est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
    assert abs(float(np.mean(est.effect(X[:, :1]))) - ATE) < 0.25


def test_linear_dml_accepts_a_frame_and_sample_weight(econml):
    frame_x = pd.DataFrame(X[:, :1], columns=["x0"])
    frame_w = pd.DataFrame(X[:, 1:], columns=["x1", "x2"])
    est = _econml_lineardml(GAMBoostRegressor(), GAMBoostClassifier())
    est.fit(Y, T, X=frame_x, W=frame_w, sample_weight=np.ones(N))
    assert abs(float(np.mean(est.effect(frame_x))) - ATE) < 0.25


def test_a_three_level_treatment_gives_one_effect_per_level(econml):
    from econml.dml import LinearDML

    rng = np.random.default_rng(5)
    t3 = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    y3 = X[:, 0] + (t3 == 1) * 1.0 + (t3 == 2) * 2.0 + rng.normal(size=N) * 0.5
    est = LinearDML(
        model_y=GAMBoostRegressor(), model_t=GAMBoostClassifier(),
        discrete_treatment=True, cv=3, random_state=0,
    )
    est.fit(y3, t3, X=X[:, :1], W=X[:, 1:])
    # Each non-baseline level is compared with level 0 in turn.
    assert abs(float(est.effect(X[:, :1], T0=0, T1=1).mean()) - 1.0) < 0.25
    assert abs(float(est.effect(X[:, :1], T0=0, T1=2).mean()) - 2.0) < 0.25


def test_drlearner_with_gam_nuisances_recovers_the_average_effect(econml):
    from econml.dr import DRLearner

    est = DRLearner(
        model_regression=GAMBoostRegressor(), model_propensity=GAMBoostClassifier(),
        cv=3, random_state=0,
    )
    est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
    assert abs(float(np.mean(est.effect(X[:, :1]))) - ATE) < 0.25


def test_causal_forest_dml_with_gam_nuisances_recovers_the_average_effect(econml):
    from econml.dml import CausalForestDML

    est = CausalForestDML(
        model_y=GAMBoostRegressor(), model_t=GAMBoostClassifier(),
        discrete_treatment=True, cv=3, n_estimators=100, random_state=0,
    )
    est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
    assert abs(float(np.mean(est.effect(X[:, :1]))) - ATE) < 0.25


def test_a_fitted_linear_dml_pickles_and_predicts_the_same(econml):
    est = _econml_lineardml(GAMBoostRegressor(), GAMBoostClassifier())
    est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_allclose(
        est.effect(X[:20, :1]), restored.effect(X[:20, :1]), rtol=0, atol=1e-12
    )


def test_the_nuisance_estimators_clone_and_round_trip_their_parameters():
    from sklearn.base import clone

    model = GAMBoostClassifier(df=5.0, mstop=60, nu=0.05)
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    copy.set_params(mstop=30)
    assert copy.get_params()["mstop"] == 30 and model.get_params()["mstop"] == 60


def test_an_estimator_parameter_reaches_the_nuisance_model(econml):
    est = _econml_lineardml(GAMBoostRegressor(mstop=10), GAMBoostClassifier(mstop=10))
    est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
    fitted_y = est.models_y[0][0]
    assert fitted_y.mstop == 10


def test_doubleml_plr_with_gam_nuisances_recovers_the_effect(doubleml):
    import doubleml as dml

    data = dml.DoubleMLData.from_arrays(X, Y, T)
    obj = dml.DoubleMLPLR(
        data, ml_l=GAMBoostRegressor(), ml_m=GAMBoostClassifier(), n_folds=3
    )
    obj.fit()
    assert abs(float(obj.coef[0]) - ATE) < 0.25


def test_doubleml_irm_with_gam_and_glm_nuisances_recovers_the_effect(doubleml):
    import doubleml as dml

    data = dml.DoubleMLData.from_arrays(X, Y, T)
    for ml_g, ml_m in [
        (GAMBoostRegressor(), GAMBoostClassifier()),
        (GLMBoostRegressor(), GLMBoostClassifier()),
    ]:
        obj = dml.DoubleMLIRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=3)
        obj.fit()
        assert abs(float(obj.coef[0]) - ATE) < 0.25, type(ml_g).__name__


def test_doubleml_accepts_a_nuisance_parameter_change(doubleml):
    import doubleml as dml

    data = dml.DoubleMLData.from_arrays(X, Y, T)
    obj = dml.DoubleMLPLR(
        data, ml_l=GAMBoostRegressor(), ml_m=GAMBoostClassifier(), n_folds=3
    )
    obj.set_ml_nuisance_params("ml_m", "d", {"mstop": 30})
    obj.fit()
    assert np.isfinite(obj.coef[0])


def test_doubleml_ssm_with_gam_nuisances_runs_on_a_selected_sample(doubleml):
    import doubleml as dml

    rng = np.random.default_rng(6)
    selected = (rng.uniform(size=N) < 0.8).astype(float)
    # DoubleML rejects NaN in y, so an unobserved outcome holds a placeholder;
    # the selection indicator decides which rows the fit uses.
    y_obs = np.where(selected == 1, Y, 0.0)
    data = dml.DoubleMLSSMData.from_arrays(X, y_obs, T, s=selected)
    obj = dml.DoubleMLSSM(
        data,
        ml_g=GAMBoostRegressor(),
        ml_pi=GAMBoostClassifier(),
        ml_m=GAMBoostClassifier(),
        n_folds=3,
    )
    obj.fit()
    assert abs(float(obj.coef[0]) - ATE) < 0.4

