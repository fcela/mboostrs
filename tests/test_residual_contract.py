"""Each residual kind is the quantity its name refers to, or it is refused.

A residual kind is a definition. Four were being conflated:

- ``type="pearson"`` on `Laplace`, `Quantile`, `Expectile` and `Huber`
  returned the *working gradient*. `Laplace`'s is ``sign(y - mu)``, which takes
  two distinct values across a whole sample and is not a Pearson residual
  under any definition.
- `CoxPH` returned ``time - exp(eta)``, subtracting an observed time from a
  relative risk. The AFT families returned the same shape of thing, with an
  ``eta`` that lives on the log-time scale and a censored response.
- The model method and ``residual_frame`` computed different Pearson residuals
  for the same Gaussian fit -- one divided by the residual RMS, the other did
  not -- and both used the name.

The available kinds come from the family likelihood table, so a kind offered
there must be computable and one refused here must not be offered there; the
last test in this file checks that in both directions.
"""

import numpy as np
import pytest
from mboostrs._residuals import RESIDUAL_KINDS, available_kinds

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    return {
        "n": n,
        "x": x,
        "gaussian": 2.0 * x + rng.normal(size=n),
        "binary": (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float),
        "count": rng.poisson(np.exp(0.5 * x)).astype(float),
        "time": np.exp(0.5 * x + rng.exponential(size=n)),
        "event": rng.binomial(1, 0.7, size=n).astype(float),
    }


def _fit(data, family, y_key="gaussian", mstop=20):
    return mb.glmboost(
        "y ~ x", data={"x": data["x"], "y": data[y_key]}, family=family,
        control=mb.boost_control(mstop, 0.1),
    )


def _survival_fit(data, family, mstop=20):
    return mb.glmboost(
        "Surv(time, event) ~ x",
        data={"x": data["x"], "time": data["time"], "event": data["event"]},
        family=family, control=mb.boost_control(mstop, 0.1),
    )


# ─── pearson is (y - mu) / sqrt(V(mu)), or nothing ─────────────────


@pytest.mark.parametrize("name", ["Laplace", "Quantile", "Expectile", "Huber"])
def test_a_robust_loss_has_no_pearson_residual(data, name):
    """These returned the working gradient under the Pearson name."""
    model = _fit(data, getattr(mb, name)())
    with pytest.raises(NotImplementedError) as excinfo:
        model.residuals(type="pearson")
    message = str(excinfo.value)
    assert "no variance function" in message
    assert "kind='working'" in message, "the refusal must name the alternative"


def test_laplace_working_residual_is_the_sign_it_used_to_call_pearson(data):
    """Available under its real name, where its two values make sense."""
    model = _fit(data, mb.Laplace())
    working = np.asarray(model.residuals(type="working"))
    assert set(np.unique(np.abs(working))) <= {1.0}
    assert len(np.unique(working)) <= 2


@pytest.mark.parametrize(
    ("name", "y_key", "variance"),
    [
        ("Gaussian", "gaussian", lambda mu: np.ones_like(mu)),
        ("Poisson", "count", lambda mu: mu),
        ("Binomial", "binary", lambda mu: mu * (1 - mu)),
    ],
)
def test_pearson_divides_by_the_declared_variance_function(data, name, y_key,
                                                           variance):
    model = _fit(data, getattr(mb, name)(), y_key=y_key)
    y = np.asarray(data[y_key], dtype=np.float64)
    mu = np.asarray(model.predict(type="response"), dtype=np.float64)
    expected = (y - mu) / np.sqrt(variance(mu))
    np.testing.assert_allclose(model.residuals(type="pearson"), expected,
                               rtol=1e-10)


def test_gaussian_pearson_is_the_raw_residual_not_a_standardised_one(data):
    """V(mu) = 1 for a Gaussian, so Pearson and response coincide.

    `residual_frame` used to divide by the residual RMS here while the method
    did not, and both called the result Pearson. Dividing by an estimated
    dispersion gives a *standardised* residual, which is a different quantity;
    R's `residuals.glm(type="pearson")` does not do it either.
    """
    model = _fit(data, mb.Gaussian())
    np.testing.assert_allclose(
        model.residuals(type="pearson"), model.residuals(type="response"),
        rtol=1e-12,
    )
    rms = float(np.sqrt(np.mean(np.asarray(model.residuals()) ** 2)))
    assert rms != pytest.approx(1.0, abs=1e-3), "the fixture must have rms != 1"


# ─── the method and the frame agree ────────────────────────────────


@pytest.mark.parametrize("kind", ["response", "pearson", "working", "deviance"])
def test_the_method_and_the_frame_compute_the_same_thing(data, kind):
    model = _fit(data, mb.Gaussian())
    method = np.asarray(model.residuals(type=kind))
    frame = np.asarray([
        row["residual"]
        for row in mb.residual_frame(model, kind=kind, output="records")
    ])
    np.testing.assert_allclose(method, frame, rtol=1e-12)


# ─── survival ──────────────────────────────────────────────────────


def test_cox_refuses_a_time_minus_risk_residual(data):
    model = _survival_fit(data, mb.CoxPH())
    with pytest.raises(NotImplementedError, match="relative risk, not a time"):
        model.residuals(type="response")


def test_cox_martingale_residuals_have_their_defining_properties(data):
    """Sum to zero, bounded above by 1, and negative where a case outlived
    its covariates."""
    model = _survival_fit(data, mb.CoxPH())
    martingale = np.asarray(model.residuals(type="martingale"))
    assert martingale.size == data["n"]
    assert np.all(np.isfinite(martingale))
    assert float(np.sum(martingale)) == pytest.approx(0.0, abs=1e-8)
    assert float(np.max(martingale)) <= 1.0 + 1e-12
    assert float(np.min(martingale)) < 0.0


def _breslow_martingale(time, event, eta, weights, at_time=None, at_event=None,
                        at_eta=None):
    event_times = np.unique(time[(event > 0) & (weights > 0)])
    increments = []
    for value in event_times:
        deaths = np.sum(weights[time == value] * event[time == value])
        denominator = np.sum(weights[time >= value] * np.exp(eta[time >= value]))
        increments.append(deaths / denominator)
    cumulative = np.cumsum(increments)
    query_time = time if at_time is None else at_time
    query_event = event if at_event is None else at_event
    query_eta = eta if at_eta is None else at_eta
    positions = np.searchsorted(event_times, query_time, side="right") - 1
    baseline = np.zeros(len(query_time))
    valid = positions >= 0
    baseline[valid] = cumulative[positions[valid]]
    return query_event - baseline * np.exp(query_eta)


def test_cox_ties_share_one_breslow_increment_and_weights_reach_it():
    time = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    x = np.array([-1.0, -0.4, 0.0, 0.3, 0.8, 1.2])
    weights = np.array([2.0, 1.0, 0.5, 1.5, 1.0, 1.0])
    model = mb.glmboost(
        "Surv(time, event) ~ x", data={"x": x, "time": time, "event": event},
        family=mb.CoxPH(), weights=weights,
        control=mb.boost_control(12, 0.1),
    )
    eta = np.asarray(model.predict(type="link"))
    expected = _breslow_martingale(time, event, eta, model.weights)
    got = np.asarray(model.residuals(type="martingale"))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert float(np.dot(model.weights, got)) == pytest.approx(0.0, abs=1e-10)


def test_cox_newdata_uses_the_training_baseline_hazard():
    time = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    x = np.linspace(-1.0, 1.0, time.size)
    model = mb.glmboost(
        "Surv(time, event) ~ x", data={"x": x, "time": time, "event": event},
        family=mb.CoxPH(), control=mb.boost_control(10, 0.1),
    )
    new = {
        "x": np.array([-0.7, 0.2, 0.9]),
        "time": np.array([1.5, 2.5, 6.0]),
        "event": np.array([0.0, 1.0, 0.0]),
    }
    train_eta = np.asarray(model.predict(type="link"))
    new_eta = np.asarray(model.predict(newdata=new, type="link"))
    expected = _breslow_martingale(
        time, event, train_eta, model.weights,
        at_time=new["time"], at_event=new["event"], at_eta=new_eta,
    )
    np.testing.assert_allclose(
        model.residuals(type="martingale", newdata=new), expected,
        rtol=1e-12, atol=1e-12,
    )


def test_cox_deviance_residuals_share_the_martingale_sign(data):
    model = _survival_fit(data, mb.CoxPH())
    martingale = np.asarray(model.residuals(type="martingale"))
    deviance = np.asarray(model.residuals(type="deviance"))
    assert np.all(np.isfinite(deviance))
    np.testing.assert_array_equal(np.sign(deviance), np.sign(martingale))
    # The transform exists to pull in the martingale residuals' long left
    # tail, so the deviance spread should be the narrower of the two.
    assert deviance.min() > martingale.min()


@pytest.mark.parametrize("name", ["Weibull", "Loglog", "Lognormal"])
def test_an_aft_family_refuses_a_response_residual(data, name):
    model = _survival_fit(data, getattr(mb, name)())
    with pytest.raises(NotImplementedError) as excinfo:
        model.residuals(type="response")
    message = str(excinfo.value)
    assert "log-time scale" in message
    assert "quantile_residuals" not in message


@pytest.mark.parametrize("name", ["Gehan", "Cindex"])
def test_a_ranking_loss_refuses_a_response_residual(data, name):
    model = _survival_fit(data, getattr(mb, name)())
    with pytest.raises(NotImplementedError, match="ranking score"):
        model.residuals(type="response")


def test_martingale_is_refused_where_there_is_no_hazard(data):
    model = _fit(data, mb.Gaussian())
    with pytest.raises(NotImplementedError, match="proportional-hazards"):
        model.residuals(type="martingale")


# ─── ordinal coding ────────────────────────────────────────────────


def test_propodds_documents_the_coding_it_accepts():
    """The class documented `0..K-1` while the fit required `1..K`.

    A caller following the docstring got "y must be integers in 1..3, got 0".
    The docstring now names the accepted coding, and the residual uses the
    same one, so all three agree.
    """
    rng = np.random.default_rng(21)
    n = 200
    x = rng.normal(size=n)
    category = np.digitize(x + rng.normal(scale=0.5, size=n), [-0.5, 0.5])

    with pytest.raises(ValueError, match=r"integers in 1\.\.3"):
        mb.glmboost("y ~ x", data={"x": x, "y": category.astype(float)},
                    family=mb.PropOdds(n_levels=3),
                    control=mb.boost_control(10, 0.1))

    model = mb.glmboost("y ~ x", data={"x": x, "y": (category + 1).astype(float)},
                        family=mb.PropOdds(n_levels=3),
                        control=mb.boost_control(20, 0.1))
    probabilities = np.asarray(model.predict(type="response"))
    assert probabilities.shape == (n, 3)

    # The response residual is against the expected category on the same
    # 1..K scale the fit accepted.
    expected = probabilities @ np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        model.residuals(type="response"), (category + 1) - expected, rtol=1e-10
    )


# ─── the declaration matches the implementation ────────────────────


ALL_FAMILIES = [
    ("Gaussian", "gaussian"), ("GaussReg", "gaussian"), ("Laplace", "gaussian"),
    ("Quantile", "gaussian"), ("Expectile", "gaussian"), ("Huber", "gaussian"),
    ("Binomial", "binary"), ("AdaExp", "binary"), ("Poisson", "count"),
    ("NBinomial", "count"),
]


@pytest.mark.parametrize(("name", "y_key"), ALL_FAMILIES)
def test_declared_kinds_are_computable_and_undeclared_ones_refuse(data, name,
                                                                  y_key):
    """Both directions, so the table and the code cannot drift apart."""
    model = _fit(data, getattr(mb, name)(), y_key=y_key)
    declared = set(available_kinds(model))

    for kind in declared:
        value = np.asarray(model.residuals(type=kind))
        assert value.size > 0, f"{name}: declared kind {kind} produced nothing"
        assert np.all(np.isfinite(value)), f"{name}: {kind} is not finite"

    for kind in set(RESIDUAL_KINDS) - declared:
        with pytest.raises(NotImplementedError):
            model.residuals(type=kind)


def test_working_is_available_for_every_family(data):
    """Every family has a negative gradient; that is what makes it fittable."""
    for name, y_key in ALL_FAMILIES:
        model = _fit(data, getattr(mb, name)(), y_key=y_key)
        assert "working" in available_kinds(model)
        assert np.all(np.isfinite(model.residuals(type="working")))


def test_an_unknown_kind_names_the_ones_that_exist(data):
    model = _fit(data, mb.Gaussian())
    with pytest.raises(ValueError, match="kind must be one of"):
        model.residuals(type="raw")


def test_residual_kinds_is_reachable_from_the_model(data):
    model = _fit(data, mb.Gaussian())
    assert model.residual_kinds() == ("response", "pearson", "deviance", "working")
    model = _fit(data, mb.Laplace())
    assert model.residual_kinds() == ("response", "working")
