"""sensitivity() on distributional (gamboostLSS / glmboostLSS) models.

A distributional model has one predictor set per distribution parameter, so
sensitivity() measures one parameter at a time through ``parameter=``.
"""

import types

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture
def scale_data():
    """mu depends on xm; sigma depends on xs; xn is noise in both."""
    rng = np.random.default_rng(7)
    n = 500
    xm = rng.uniform(0, 10, n)
    xs = rng.uniform(0, 10, n)
    xn = rng.normal(0, 1, n)
    sigma = np.exp(0.8 * np.cos(xs) / 2)
    y = np.sin(xm) + sigma * rng.normal(0, 1, n)
    return {"xm": xm, "xs": xs, "xn": xn, "y": y}


@pytest.fixture
def mu_sigma_model(scale_data):
    return mb.gamboostLSS(
        {"mu": "y ~ bbs(xm, df=4)",
         "sigma": "y ~ bbs(xs, df=4) + bbs(xn, df=4)"},
        data=scale_data, family=mb.GaussianLSS(),
        control=mb.boost_control(150, 0.1))


def _predictors(data):
    return {k: v for k, v in data.items() if k != "y"}


def test_sigma_sensitivity_ranks_the_sigma_covariate_first(
        mu_sigma_model, scale_data):
    rows = mb.sensitivity(mu_sigma_model, _predictors(scale_data),
                          parameter="sigma")
    assert [row["variable"] for row in rows] == ["xs", "xn"]
    assert rows[0]["range of the effect"] > 2 * rows[1]["range of the effect"]


def test_each_row_names_the_parameter_it_was_measured_on(
        mu_sigma_model, scale_data):
    rows = mb.sensitivity(mu_sigma_model, _predictors(scale_data),
                          parameter="sigma")
    assert {row["parameter"] for row in rows} == {"sigma"}


def test_sigma_sensitivity_is_the_linear_predictor_on_its_link_scale(
        mu_sigma_model, scale_data):
    """The grid evaluates the model's own sigma predictor, so the effect
    matches a hand-built prediction of that predictor over the same grid."""
    rows = {row["variable"]: row for row in mb.sensitivity(
        mu_sigma_model, _predictors(scale_data), parameter="sigma",
        points=25)}
    xs = scale_data["xs"]
    grid = np.linspace(xs.min(), xs.max(), 25)
    held = {k: np.full(grid.size, float(np.mean(scale_data[k])))
            for k in ("xm", "xn")}
    held["xs"] = grid
    eta = np.asarray(mu_sigma_model.predict(
        newdata=held, parameter="sigma", type="link")).ravel()
    assert rows["xs"]["range of the effect"] == pytest.approx(
        float(eta.max() - eta.min()))


def test_default_parameter_is_the_first_one_of_the_family(
        mu_sigma_model, scale_data):
    data = _predictors(scale_data)
    default = mb.sensitivity(mu_sigma_model, data)
    explicit = mb.sensitivity(mu_sigma_model, data, parameter="mu")
    assert default == explicit
    assert [row["variable"] for row in default] == ["xm"]
    assert default[0]["parameter"] == "mu"


def test_each_parameter_lists_only_its_own_predictors(
        mu_sigma_model, scale_data):
    data = _predictors(scale_data)
    mu_vars = {row["variable"]
               for row in mb.sensitivity(mu_sigma_model, data, parameter="mu")}
    sigma_vars = {row["variable"] for row in
                  mb.sensitivity(mu_sigma_model, data, parameter="sigma")}
    assert mu_vars == {"xm"}
    assert sigma_vars == {"xs", "xn"}
    with pytest.raises(ValueError, match="not a predictor of parameter 'sigma'"):
        mb.sensitivity(mu_sigma_model, data, variables=["xm"], parameter="sigma")


def test_unknown_parameter_names_the_valid_ones(mu_sigma_model, scale_data):
    with pytest.raises(ValueError, match="expected one of"):
        mb.sensitivity(mu_sigma_model, _predictors(scale_data), parameter="nu")


def test_parameter_is_refused_for_a_gamboost_model(scale_data):
    model = mb.gamboost("y ~ bbs(xm, df=4)", data=scale_data,
                        family=mb.Gaussian(),
                        control=mb.boost_control(50, 0.1))
    with pytest.raises(ValueError, match="Drop parameter="):
        mb.sensitivity(model, _predictors(scale_data), parameter="mu")


def test_gamboost_rows_keep_their_original_keys(scale_data):
    model = mb.gamboost("y ~ bbs(xm, df=4) + bols(xn)", data=scale_data,
                        family=mb.Gaussian(),
                        control=mb.boost_control(50, 0.1))
    rows = mb.sensitivity(model, _predictors(scale_data))
    assert all("parameter" not in row for row in rows)
    assert set(rows[0]) == {"variable", "range of the effect",
                            "slope, 10% to 90%", "turns"}


def test_glm_lss_sensitivity_takes_parameter(scale_data):
    model = mb.glmboostLSS({"mu": "y ~ xm", "sigma": "y ~ xs + xn"},
                           data=scale_data, family=mb.GaussianLSS(),
                           control=mb.boost_control(100, 0.1))
    rows = mb.sensitivity(model, _predictors(scale_data), parameter="sigma")
    assert {row["variable"] for row in rows} == {"xs", "xn"}
    assert {row["parameter"] for row in rows} == {"sigma"}


def test_refusal_names_the_distributional_route(scale_data):
    """An object with neither learner specs nor a formula is still refused,
    and the message points at the distributional routes."""
    with pytest.raises(ValueError, match="sensitivity\\(model, data, parameter='sigma'\\)") as info:
        mb.sensitivity(types.SimpleNamespace(), _predictors(scale_data))
    assert "prediction_contrast" in str(info.value)
    assert "distributional_contrast" in str(info.value)


@pytest.fixture
def disjoint_data():
    """mu reads only a; sigma reads only b."""
    rng = np.random.default_rng(11)
    n = 400
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    y = np.sin(a) + np.exp(0.4 * np.cos(b)) * rng.normal(0, 1, n)
    return {"a": a, "b": b, "y": y}


@pytest.fixture
def disjoint_model(disjoint_data):
    return mb.gamboostLSS(
        {"mu": "y ~ bbs(a, df=4)", "sigma": "y ~ bbs(b, df=4)"},
        data=disjoint_data, family=mb.GaussianLSS(),
        control=mb.boost_control(150, 0.1))


def test_mu_sensitivity_holds_the_sigma_column_at_its_mean(disjoint_model, disjoint_data):
    """With sigma's column b in the data, mu is measured over a while the grid
    holds b at its mean, so the range is the mu predictor with b fixed."""
    rows = mb.sensitivity(disjoint_model, _predictors(disjoint_data), parameter="mu")
    assert [row["variable"] for row in rows] == ["a"]
    assert {row["parameter"] for row in rows} == {"mu"}
    a = disjoint_data["a"]
    grid = np.linspace(a.min(), a.max(), 25)
    held = {"a": grid, "b": np.full(grid.size, float(np.mean(disjoint_data["b"])))}
    eta = np.asarray(disjoint_model.predict(
        newdata=held, parameter="mu", type="link")).ravel()
    assert rows[0]["range of the effect"] == pytest.approx(float(eta.max() - eta.min()))


def test_sigma_sensitivity_holds_the_mu_column_at_its_mean(disjoint_model, disjoint_data):
    """With mu's column a in the data, sigma is measured over b while the grid
    holds a at its mean, so the range is the sigma predictor with a fixed."""
    rows = mb.sensitivity(disjoint_model, _predictors(disjoint_data), parameter="sigma")
    assert [row["variable"] for row in rows] == ["b"]
    assert {row["parameter"] for row in rows} == {"sigma"}
    b = disjoint_data["b"]
    grid = np.linspace(b.min(), b.max(), 25)
    held = {"a": np.full(grid.size, float(np.mean(disjoint_data["a"]))), "b": grid}
    eta = np.asarray(disjoint_model.predict(
        newdata=held, parameter="sigma", type="link")).ravel()
    assert rows[0]["range of the effect"] == pytest.approx(float(eta.max() - eta.min()))


@pytest.fixture
def transform_model(disjoint_data):
    """sigma reads log(b), so the model's transform recipe reads b for every parameter."""
    return mb.gamboostLSS(
        {"mu": "y ~ bbs(a, df=4)", "sigma": "y ~ bbs(log(b), df=4)"},
        data=disjoint_data, family=mb.GaussianLSS(),
        control=mb.boost_control(150, 0.1))


def test_mu_sensitivity_when_sigma_reads_a_transform_of_b(transform_model, disjoint_data):
    rows = mb.sensitivity(transform_model, _predictors(disjoint_data), parameter="mu")
    assert [row["variable"] for row in rows] == ["a"]


def test_sigma_sensitivity_names_the_source_column_of_a_transform(
        transform_model, disjoint_data):
    """The derived column log__b is reported under b, the column the caller holds."""
    rows = mb.sensitivity(transform_model, _predictors(disjoint_data), parameter="sigma")
    assert [row["variable"] for row in rows] == ["b"]


def test_mu_sensitivity_when_sigma_reads_a_factor(disjoint_data):
    """sigma reads a string column g that mu does not. The grid holds g at its
    most frequent level, and mu is measured on a alone."""
    rng = np.random.default_rng(12)
    data = {**disjoint_data, "g": rng.choice(["lo", "hi"], disjoint_data["a"].size)}
    model = mb.gamboostLSS(
        {"mu": "y ~ bbs(a, df=4)", "sigma": "y ~ bols(g)"},
        data=data, family=mb.GaussianLSS(),
        control=mb.boost_control(100, 0.1))
    rows = mb.sensitivity(model, _predictors(data), parameter="mu")
    assert [row["variable"] for row in rows] == ["a"]
    assert {row["parameter"] for row in rows} == {"mu"}
