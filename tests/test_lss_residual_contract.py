"""LSS residual kinds are the quantities their names refer to, or refused.

The standard-model fix (``test_residual_contract.py``) renamed working
gradients and refused response residuals for censored responses. The LSS
surface had the same two defects, one of them inverted:

- ``residuals(type="pearson")`` computed `(y - mu)/sigma` for a `GaussianLSS`
  and refused for every other family, silently even for the families whose
  response is censored (`LogNormalLSS`, `LogLogLSS`, `WeibullLSS`) or a
  composition vector (`DirichletLSS`), where ``y - mu`` is not a residual.
- A location response residual ``y - mu`` was claimed for every fit including
  survival and multivariate ones.

The kinds are declared by a table (:func:`mboostrs.gamlss.lss_residual_kinds`),
checked here in both directions: a kind a family offers must be computable, a
kind refused must not be offered. ``working`` -- the negative gradient from
the same core routine the fitting loop uses -- is the honest name for the
gradient vectors, and is available for *any* parameter of *every* LSS family,
survival and Dirichlet included.
"""

import numpy as np
import pytest
from mboostrs.gamlss import (
    _NO_RESPONSE_LSS,
    _PEARSON_LSS,
    _FAMILIES,
    lss_residual_kinds,
)

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 120
    x = rng.normal(size=n)
    return {
        "n": n,
        "x": x,
        "gaussian": 0.5 * x + rng.normal(size=n),
        "count": rng.poisson(np.exp(0.3 * x)).astype(float),
        "positive": np.exp(0.3 * x + rng.normal(scale=0.3, size=n)),
        "bounded": rng.beta(3.0, 3.0, size=n),
        "time": np.exp(0.2 * x + rng.normal(scale=0.4, size=n)),
        "status": rng.binomial(1, 0.8, size=n).astype(float),
    }


def _fit(data, family, y_key="gaussian", mstop=15, **kwargs):
    return mb.glmboostLSS(
        "y ~ x", data={"x": data["x"], "y": data[y_key]}, families=family,
        control=mb.boost_control(mstop, 0.1), **kwargs,
    )


# ─── the declaration table ─────────────────────────────────────────


def test_declared_kinds_match_the_tables():
    for name, family in _FAMILIES.items():
        kinds = lss_residual_kinds(family())
        assert kinds == tuple(dict.fromkeys(kinds)), "no duplicates"
        assert kinds[-1] == "working"
        assert set(kinds) <= {"response", "pearson", "working"}
        assert ("response" in kinds) == (name not in _NO_RESPONSE_LSS), name
        assert ("pearson" in kinds) == (name in _PEARSON_LSS), name


@pytest.mark.parametrize("name", ["GaussianLSS", "PoissonLSS"])
def test_pearson_only_where_a_variance_function_is_declared(data, name):
    family = _FAMILIES[name]()
    assert lss_residual_kinds(family) == ("response", "pearson", "working")


@pytest.mark.parametrize("name", ["GammaLSS", "StudentTLSS", "BetaLSS"])
def test_pearson_declined_without_a_variance_function(data, name):
    family = _FAMILIES[name]()
    assert lss_residual_kinds(family) == ("response", "working")


@pytest.mark.parametrize(
    "name", ["LogNormalLSS", "LogLogLSS", "WeibullLSS"]
)
def test_response_declined_for_survival(data, name):
    family = _FAMILIES[name]()
    assert lss_residual_kinds(family) == ("working",)


def test_response_declined_for_multivariate():
    assert lss_residual_kinds(mb.DirichletLSS(3)) == ("working",)


# ─── every declared kind is computable, every other refused ────────


@pytest.mark.parametrize(
    ("name", "y_key"),
    [
        ("GaussianLSS", "gaussian"),
        ("PoissonLSS", "count"),
        ("GammaLSS", "positive"),
        ("StudentTLSS", "gaussian"),
        ("NegativeBinomialLSS", "count"),
    ],
)
def test_every_declared_kind_is_computable_and_others_refuse(data, name, y_key):
    model = _fit(data, _FAMILIES[name](), y_key=y_key)
    assert model.residual_kinds() == lss_residual_kinds(model._family)
    for kind in model.residual_kinds():
        residual = model.residuals(type=kind)
        assert isinstance(residual, dict) and len(residual) == 1
        values = next(iter(residual.values()))
        assert np.asarray(values).size == data["n"]
        assert np.all(np.isfinite(values))
    for kind in ("response", "pearson", "working"):
        if kind in model.residual_kinds():
            continue
        with pytest.raises(NotImplementedError):
            model.residuals(type=kind)


def test_unknown_kind_fails_before_any_computation(data):
    model = _fit(data, mb.GaussianLSS())
    with pytest.raises(ValueError, match="type must be one of"):
        model.residuals(type="deviance")


# ─── the formulas the kinds name ───────────────────────────────────


def test_pearson_divides_by_the_fitted_sigma_for_gaussian(data):
    model = _fit(data, mb.GaussianLSS(), y_key="gaussian")
    y = np.asarray(data["gaussian"], dtype=np.float64)
    mu = np.asarray(model.predict(parameter="mu", type="response"))
    sigma = np.asarray(model.predict(parameter="sigma", type="response"))
    expected = (y - mu) / np.maximum(sigma, 1e-10)
    np.testing.assert_allclose(
        model.residuals(type="pearson")["mu"], expected, rtol=1e-10,
    )


def test_poisson_pearson_divides_by_sqrt_mu(data):
    model = _fit(data, mb.PoissonLSS(), y_key="count")
    y = np.asarray(data["count"], dtype=np.float64)
    mu = np.asarray(model.predict(parameter="mu", type="response"))
    expected = (y - mu) / np.sqrt(np.maximum(mu, 1e-10))
    np.testing.assert_allclose(
        model.residuals(type="pearson")["mu"], expected, rtol=1e-10,
    )


def test_response_is_y_minus_location_mu(data):
    model = _fit(data, mb.StudentTLSS(), y_key="gaussian")
    y = np.asarray(data["gaussian"], dtype=np.float64)
    mu = np.asarray(model.predict(parameter="mu", type="response"))
    np.testing.assert_allclose(
        model.residuals()["mu"], y - mu, rtol=1e-10,
    )
    frame = mb.residual_frame(model, kind="response", output="records")
    from_frame = np.asarray([row["residual"] for row in frame])
    np.testing.assert_allclose(from_frame, y - mu, rtol=1e-10)


def test_gaussian_working_is_the_negative_gradient(data):
    """The vector the next boosting step regresses on: (y - mu) / sigma^2."""
    model = _fit(data, mb.GaussianLSS(), y_key="gaussian")
    y = np.asarray(data["gaussian"], dtype=np.float64)
    mu = np.asarray(model.predict(parameter="mu", type="response"))
    sigma = np.asarray(model.predict(parameter="sigma", type="response"))
    expected = (y - mu) / (sigma * sigma)
    np.testing.assert_allclose(
        model.residuals(type="working")["mu"], expected, rtol=1e-8,
    )


@pytest.mark.parametrize("parameter", ["mu", "sigma"])
def test_working_is_available_for_every_parameter(data, parameter):
    model = _fit(data, mb.GaussianLSS(), y_key="gaussian")
    values = model.residuals(parameter=parameter, type="working")
    assert np.asarray(values).size == data["n"]
    assert np.all(np.isfinite(values))


# ─── refusal surfaces ──────────────────────────────────────────────


def test_non_location_response_residual_names_the_location(data):
    model = _fit(data, mb.GaussianLSS(), y_key="gaussian")
    with pytest.raises(NotImplementedError) as excinfo:
        model.residuals(parameter="sigma", type="response")
    assert "location parameter" in str(excinfo.value)
    assert "type='working'" in str(excinfo.value)


def test_survival_response_residuals_are_refused(data):
    model = mb.glmboostLSS(
        "Surv(time, status) ~ x",
        data={"x": data["x"], "time": data["time"], "status": data["status"]},
        families=mb.LogNormalLSS(),
        control=mb.boost_control(15, 0.1),
    )
    assert model.residual_kinds() == ("working",)
    with pytest.raises(NotImplementedError) as excinfo:
        model.residuals()
    assert "censored" in str(excinfo.value)
    with pytest.raises(NotImplementedError):
        mb.residual_frame(model, kind="response")
    working = model.residuals(type="working")["mu"]
    assert np.asarray(working).size == data["n"]
    assert np.all(np.isfinite(working))


def test_dirichlet_response_residuals_are_refused():
    rng = np.random.default_rng(5)
    n = 60
    x = rng.normal(size=n)
    composition = rng.dirichlet([2.0, 3.0, 4.0], size=n)
    model = mb.glmboostLSS(
        "cbind(y1, y2, y3) ~ x",
        data={"x": x, "y1": composition[:, 0], "y2": composition[:, 1],
              "y3": composition[:, 2]},
        families=mb.DirichletLSS(3),
        control=mb.boost_control(15, 0.1),
    )
    assert model.residual_kinds() == ("working",)
    with pytest.raises(NotImplementedError) as excinfo:
        model.residuals()
    assert "multivariate" in str(excinfo.value)
    for parameter in ("a1", "a2", "a3"):
        values = model.residuals(parameter=parameter, type="working")
        assert np.asarray(values).size == n
        assert np.all(np.isfinite(values))


def test_newdata_and_mstop_reach_the_residual(data):
    model = _fit(data, mb.GaussianLSS(), y_key="gaussian")
    new = {"x": data["x"][:10], "y": data["gaussian"][:10]}
    mstop = 7
    mu = np.asarray(
        model.predict(newdata=new, parameter="mu", type="response", mstop=mstop)
    )
    expected = np.asarray(data["gaussian"][:10]) - mu
    np.testing.assert_allclose(
        model.residuals(type="response", newdata=new, mstop=mstop)["mu"],
        expected, rtol=1e-10,
    )