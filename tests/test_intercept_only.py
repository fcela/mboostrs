"""Intercept-only models: `y ~ 1` in gamboost() and glmboost().

`y ~ 1` fits no base learner. Every row is predicted by the family's offset,
and the boosting path has no selections. The calls a pipeline makes on such a
model must return a value rather than crash.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs.families import Binomial, Gaussian, Poisson

FAMILIES = {
    "Gaussian": ("y", Gaussian),
    "Binomial": ("b", Binomial),
    "Poisson": ("c", Poisson),
}
FITS = {"gamboost": mb.gamboost, "glmboost": mb.glmboost}
CASES = [(fit, fam) for fit in FITS for fam in FAMILIES]
CASE_IDS = [f"{fit}-{fam}" for fit, fam in CASES]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(357)
    n = 120
    x = rng.normal(size=n)
    eta = 0.4 + 0.5 * x
    return {
        "x": x,
        "z": rng.normal(size=n),
        "y": eta + rng.normal(size=n),
        "b": (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float),
        "c": rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float),
    }


def _fit(fit, family_name, data, rhs="1", **kwargs):
    response, family_cls = FAMILIES[family_name]
    return fit(f"{response} ~ {rhs}", data, family=family_cls(), **kwargs)


def _newdata(data, rows=5):
    return {"x": data["x"][:rows], "z": data["z"][:rows]}


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_fit_has_no_learner_and_no_boosting_step(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    assert list(model.selected) == []
    assert model.mstop == 0
    assert np.asarray(model.risk).shape == (1,)


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_link_prediction_is_the_family_offset_for_every_row(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    eta = np.asarray(model.predict())
    assert eta.shape == (len(data["x"]),)
    assert np.all(eta == eta[0])
    assert np.allclose(np.asarray(model.fitted()), eta)


@pytest.mark.parametrize("family_name", list(FAMILIES))
def test_offset_is_the_family_closed_form_constant(data, family_name):
    """The offset comes from the family implementation. Each family's constant
    has a closed form that the test checks against."""
    response, _ = FAMILIES[family_name]
    y = data[response]
    eta = float(np.asarray(_fit(mb.gamboost, family_name, data).predict())[0])
    if family_name == "Gaussian":
        expected = y.mean()
    elif family_name == "Binomial":
        expected = 0.5 * np.log(y.mean() / (1 - y.mean()))
    else:
        expected = np.log(y.mean())
    assert eta == pytest.approx(expected, rel=1e-9)


def test_gaussian_risk_is_the_squared_error_at_the_offset(data):
    model = _fit(mb.gamboost, "Gaussian", data)
    eta = float(np.asarray(model.predict())[0])
    assert np.asarray(model.risk)[0] == pytest.approx(np.sum((data["y"] - eta) ** 2), rel=1e-9)


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_newdata_prediction_returns_one_value_per_new_row(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    eta_new = np.asarray(model.predict(newdata=_newdata(data)))
    eta_train = np.asarray(model.predict())
    assert eta_new.shape == (5,)
    assert np.allclose(eta_new, eta_train[0])


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_response_prediction_is_the_inverse_link_of_the_offset(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    eta = np.asarray(model.predict())
    mu = np.asarray(model.predict(type="response"))
    assert mu.shape == eta.shape
    assert np.all(np.isfinite(mu))
    if family_name == "Gaussian":
        assert np.allclose(mu, eta)
    elif family_name == "Binomial":
        assert np.allclose(mu, 1 / (1 + np.exp(-2 * eta)))
    else:
        assert np.allclose(mu, np.exp(eta))


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_newdata_response_prediction(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    mu_new = np.asarray(model.predict(newdata=_newdata(data), type="response"))
    mu_train = np.asarray(model.predict(type="response"))
    assert mu_new.shape == (5,)
    assert np.allclose(mu_new, mu_train[0])


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_summary_varimp_and_coef_do_not_crash(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    assert model.summary() is not None
    assert isinstance(model.summary(as_dict=True), dict)
    varimp = model.varimp()
    assert varimp["reduction"] == {}
    assert isinstance(model.coef(), dict)


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_cvrisk_returns_a_result_whose_optimum_is_zero(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    result = model.cvrisk()
    assert type(result).__name__ == "CvRiskResult"
    assert result.optimal_mstop == 0
    assert isinstance(result.to_dict(), dict)


@pytest.mark.parametrize("family_name", list(FAMILIES))
def test_cv_risk_function_accepts_the_intercept_only_formula(data, family_name):
    response, family_cls = FAMILIES[family_name]
    risk = mb.cv_risk("1", data, response, family=family_cls(), folds=3, mstop=20)
    assert np.isfinite(risk)
    result = mb.cv_risk("1", data, response, family=family_cls(), folds=3, mstop=20,
                        return_result=True)
    assert isinstance(result.to_dict(), dict)


@pytest.mark.parametrize("family_name", list(FAMILIES))
def test_module_conformal_interval_accepts_the_intercept_only_model(data, family_name):
    response, family_cls = FAMILIES[family_name]
    model = mb.gamboost(f"{response} ~ 1", data, family=family_cls())
    interval = mb.conformal_interval(model, level=0.9)
    assert np.all(np.asarray(interval.lower) <= np.asarray(interval.upper))


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_save_and_load_round_trip(data, fit_name, family_name, tmp_path):
    model = _fit(FITS[fit_name], family_name, data)
    path = tmp_path / "intercept_only.mboost"
    mb.save_model(model, path)
    restored = mb.load_model(path)
    assert np.allclose(np.asarray(restored.predict()), np.asarray(model.predict()))
    assert list(restored.selected) == []


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_prediction_contrast_is_zero(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    contrast = mb.prediction_contrast(model, data, "x", 0.0, 1.0)
    assert contrast.estimate == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("fit_name", "family_name"), CASES, ids=CASE_IDS)
def test_conformal_interval_is_centred_on_the_offset(data, fit_name, family_name):
    model = _fit(FITS[fit_name], family_name, data)
    interval = model.conformal_interval(level=0.9)
    lower = np.asarray(interval.lower)
    upper = np.asarray(interval.upper)
    predicted = np.asarray(model.predict(type="link"))
    assert np.all(lower <= predicted) and np.all(predicted <= upper)
    assert np.all(upper >= lower)


@pytest.mark.parametrize("family_name", list(FAMILIES))
def test_gam_refit_names_the_reason_for_an_intercept_only_model(data, family_name):
    model = _fit(mb.gamboost, family_name, data)
    with pytest.raises(ValueError, match="intercept only"):
        mb.gam_refit(model)


def test_compare_specifications_accepts_one_as_a_right_hand_side(data):
    rows = mb.compare_specifications(
        ["1", "x"], data, "y", family=Gaussian(), folds=3, mstop=20,
    )
    specs = [row["specification"] for row in rows]
    assert specs == ["1", "x"]
    assert np.isfinite(rows[0]["out-of-fold risk"])


def test_requested_mstop_does_not_add_steps(data):
    control = mb.boost_control(mstop=50)
    model = _fit(mb.gamboost, "Gaussian", data, control=control)
    assert model.mstop == 0
    assert list(model.selected) == []


@pytest.mark.parametrize("fit", [mb.gamboost, mb.glmboost], ids=["gamboost", "glmboost"])
def test_gamboost_and_glmboost_agree_on_the_offset(data, fit):
    gam = mb.gamboost("y ~ 1", data, family=Gaussian())
    glm = mb.glmboost("y ~ 1", data, family=Gaussian())
    assert np.allclose(np.asarray(gam.predict()), np.asarray(glm.predict()))


def test_empty_formula_error_names_the_intercept_only_form(data):
    with pytest.raises(ValueError, match="y ~ 1"):
        mb.gamboost("y ~ ", data)


def test_gamboost_still_refuses_intercept_controls_with_other_terms(data):
    with pytest.raises(NotImplementedError, match="requests an intercept"):
        mb.gamboost("y ~ 1 + x", data)
