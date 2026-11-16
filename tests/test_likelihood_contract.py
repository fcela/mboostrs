"""Every claim in the family likelihood table, recomputed.

``mboostrs._likelihood`` says, per family, whether the fitted risk is a
negative log-likelihood, whether it is the complete one, and what quantity the
density is over. Those are claims about arithmetic, so this file checks them by
fitting a model and comparing ``risk`` against a reference NLL built here from
``scipy`` — not by reading the loss definitions, which is how the claims got
out of step with the code in the first place.

A family whose row says ``has_normalized_loglik=True`` must agree with its
reference to floating-point precision. A family whose row says ``False`` must
*disagree*: a row that claims a defect the code does not have is as wrong as
one that misses a defect the code does have, and only the disagreement test
catches it.
"""

import numpy as np
import pytest
from mboostrs._likelihood import (
    FAMILY_LIKELIHOOD,
    comparable_measures,
    family_likelihood,
    loglik_refusal,
)
from scipy.special import gammaln
from scipy.stats import norm

import mboostrs as mb

CONTROL = mb.boost_control(30, 0.1)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    return {
        "n": n,
        "x": x,
        "gaussian": 2.0 * x + rng.normal(size=n),
        "binary": (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float),
        "count": rng.poisson(np.exp(0.5 * x)).astype(float),
        "overdispersed": rng.negative_binomial(
            4, 4 / (4 + np.exp(0.5 * x))
        ).astype(float),
        "time": np.exp(0.5 * x + rng.normal(scale=0.4, size=n)),
        "event": (rng.uniform(size=n) < 0.7).astype(float),
    }


def _fit(formula, data_map, family):
    return mb.glmboost(formula, data=data_map, family=family, control=CONTROL)


def _risk(model):
    return float(np.asarray(model._risk)[model.mstop])


# ─── families whose risk is the complete NLL ───────────────────────


def test_binomial_risk_is_the_exact_nll(data):
    y = data["binary"]
    model = _fit("y ~ x", {"x": data["x"], "y": y}, mb.Binomial())
    p = np.asarray(model.predict(type="response"))
    reference = -float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    assert _risk(model) == pytest.approx(reference, abs=1e-9)
    assert model.logLik() == pytest.approx(-reference, abs=1e-9)


def test_poisson_risk_includes_the_log_factorial(data):
    """The `log(y!)` term does not depend on the fit, but it is present."""
    y = data["count"]
    model = _fit("y ~ x", {"x": data["x"], "y": y}, mb.Poisson())
    mu = np.asarray(model.predict(type="response"))
    with_constant = float(np.sum(mu - y * np.log(mu) + gammaln(y + 1)))
    without = float(np.sum(mu - y * np.log(mu)))
    assert _risk(model) == pytest.approx(with_constant, abs=1e-9)
    assert _risk(model) != pytest.approx(without, abs=1e-3)


def test_nbinomial_risk_is_the_nll_at_the_profiled_dispersion(data):
    y = data["overdispersed"]
    model = _fit("y ~ x", {"x": data["x"], "y": y}, mb.NBinomial())
    mu = np.asarray(model.predict(type="response"))
    sigma = float(np.asarray(model._nuisance_vec_path)[model.mstop, 0])
    reference = -float(np.sum(
        gammaln(y + sigma) - gammaln(sigma) - gammaln(y + 1)
        + sigma * np.log(sigma / (sigma + mu))
        + y * np.log(mu / (sigma + mu))
    ))
    assert _risk(model) == pytest.approx(reference, abs=1e-8)


def test_coxph_risk_is_the_breslow_partial_likelihood(data):
    model = _fit(
        "Surv(time, event) ~ x",
        {"x": data["x"], "time": data["time"], "event": data["event"]},
        mb.CoxPH(),
    )
    eta = np.asarray(model.predict(type="link"))
    order = np.argsort(-data["time"])
    cumulative = np.logaddexp.accumulate(eta[order])
    events = data["event"][order] > 0
    reference = -float(np.sum((eta[order] - cumulative)[events]))
    assert _risk(model) == pytest.approx(reference, abs=1e-8)
    assert family_likelihood(mb.CoxPH()).loglik_measure == "partial"


AFT_CASES = {
    # family -> (log survival function, log density), both of the standardised
    # residual z = (log(t) - eta) / scale.
    "Lognormal": (norm.logsf, norm.logpdf),
    "Weibull": (lambda z: -np.exp(z), lambda z: z - np.exp(z)),
    "Loglog": (
        lambda z: -np.log1p(np.exp(z)),
        lambda z: z - 2 * np.log1p(np.exp(z)),
    ),
}


@pytest.mark.parametrize("name", sorted(AFT_CASES))
def test_aft_risk_is_the_nll_of_log_time_not_of_time(data, name):
    """The `1/t` Jacobian is absent, which is what `log_response` records.

    An AFT risk is a proper log-likelihood, but of `log(T)`. Comparing it with
    a Gaussian fit of the same `time` column by AIC would difference two
    densities over different quantities, so the measure is part of the
    contract rather than a footnote.
    """
    log_sf, log_pdf = AFT_CASES[name]
    family = getattr(mb, name)()
    model = _fit(
        "Surv(time, event) ~ x",
        {"x": data["x"], "time": data["time"], "event": data["event"]},
        family,
    )
    scale = float(np.asarray(model._nuisance_vec_path)[model.mstop, 0])
    eta = np.asarray(model.predict(type="link"))
    z = (np.log(data["time"]) - eta) / scale
    observed = data["event"] > 0

    on_log_scale = -float(np.sum(
        np.where(observed, log_pdf(z) - np.log(scale), log_sf(z))
    ))
    on_time_scale = on_log_scale + float(np.sum(
        np.log(data["time"])[observed]
    ))

    assert _risk(model) == pytest.approx(on_log_scale, abs=1e-8)
    assert _risk(model) != pytest.approx(on_time_scale, abs=1e-3)
    assert family_likelihood(family).loglik_measure == "log_response"


def test_propodds_risk_is_the_categorical_nll():
    rng = np.random.default_rng(21)
    n = 300
    x = rng.normal(size=n)
    u = rng.uniform(size=n)
    eta = 0.8 * x
    category = (
        1
        + (u > 1 / (1 + np.exp(0.5 + eta))).astype(int)
        + (u > 1 / (1 + np.exp(-0.7 + eta))).astype(int)
    )
    model = mb.glmboost(
        "y ~ x", data={"x": x, "y": category.astype(float)},
        family=mb.PropOdds(n_levels=3), control=mb.boost_control(40, 0.1),
    )
    probabilities = np.asarray(model.predict(type="response"))
    reference = -float(np.sum(np.log(
        np.maximum(probabilities[np.arange(n), category - 1], 1e-300)
    )))
    assert _risk(model) == pytest.approx(reference, abs=1e-8)


# ─── families whose risk is not the complete NLL ───────────────────


def test_gaussian_risk_is_rss_and_loglik_reconstructs_the_rest(data):
    """The risk is RSS, matching `mboost`'s `(y - f)^2` loss.

    Not the NLL, and not proportional to it either: the Gaussian NLL's
    quadratic term is `RSS / (2 sigma^2)`, so the risk is out by the variance
    and by a factor of two, on top of the missing `2*pi` constant.
    """
    y = data["gaussian"]
    model = _fit("y ~ x", {"x": data["x"], "y": y}, mb.Gaussian())
    fitted = np.asarray(model.fitted())
    rss = float(np.sum((y - fitted) ** 2))
    assert _risk(model) == pytest.approx(rss, abs=1e-9)

    n = len(y)
    sigma2 = rss / n
    reference = -n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    assert model.logLik() == pytest.approx(reference, abs=1e-9)
    # The two differ, which is the whole point of separating them.
    assert model.logLik() != pytest.approx(model.negative_risk(), abs=1e-3)


def test_multinomial_risk_omits_the_reference_class():
    """Measured, and matching `mboost::Multinomial`.

    The loss is summed over the K-1 non-reference columns, where an
    observation in the reference class is zero in every one, so it contributes
    nothing. The omitted `-log(p_reference)` moves with the fit, so this is a
    missing part of the objective rather than a constant that cancels.
    """
    rng = np.random.default_rng(21)
    n = 300
    x = rng.normal(size=n)
    y = rng.integers(0, 3, size=n).astype(float)
    model = mb.gamboost(
        "y ~ bols(x)", data={"x": x, "y": y}, family=mb.Multinomial(),
        control=mb.boost_control(40, 0.1),
    )
    probabilities = np.asarray(model.predict(type="response"))
    classes = y.astype(int)
    reference_class = probabilities.shape[1] - 1
    per_row = -np.log(np.maximum(
        probabilities[np.arange(n), classes], 1e-300
    ))

    risk = float(np.asarray(model._risk)[model.mstop])
    non_reference = float(per_row[classes != reference_class].sum())
    assert risk == pytest.approx(non_reference, abs=1e-8)
    # The omission is not negligible on this fixture, so the test is not
    # passing by accident on a dataset with no reference-class rows.
    assert int((classes == reference_class).sum()) > 0
    assert float(per_row[classes == reference_class].sum()) > 1.0

    with pytest.raises(NotImplementedError, match="reference class"):
        model.logLik()


def test_multinomial_negative_gradient_is_the_derivative_of_the_full_nll():
    """The fit descends the full categorical NLL, not the reported risk.

    The reported risk (`risk`, and therefore `negative_risk()`) omits every
    reference-class observation, but the negative gradient is `y_k - p_k`,
    which is the derivative of the *complete* cross-entropy including the
    reference-class term `-log(p_reference)`. Measured: central finite
    differences of the full NLL agree with the negated gradient to 1e-6, on
    rows drawn from every class.
    """
    from mboostrs._rust import family_diagnostics

    rng = np.random.default_rng(23)
    n_classes = 4
    n_columns = n_classes - 1
    n = 8
    labels = rng.integers(0, n_classes, size=n)
    labels[0] = n_classes - 1  # guarantee a reference-class row
    y = np.zeros((n, n_columns))
    for i in range(n):
        if labels[i] < n_columns:
            y[i, labels[i]] = 1.0
    y = y.ravel()
    f = rng.normal(0, 1.5, size=n * n_columns)
    # Give every reference-class row a large positive logit sum so the omitted
    # `-log(p_reference)` term is ~3.3 per row and cannot be dismissed as a
    # constant or a rounding artefact. (Values stay well inside the [-36, 36]
    # clip, so the reference softmax and the core's agree without clamping.)
    ref_indices = np.flatnonzero(labels == n_classes - 1)
    assert ref_indices.size > 0  # reference rows present
    for i in ref_indices:
        f[i * n_columns:(i + 1) * n_columns] = 2.2
    w = np.ones(n)

    _, gradient, risk = family_diagnostics(
        "multinomial", {"n_classes": float(n_classes)}, y, f, w
    )

    def probabilities(f_):
        logits = np.exp(np.clip(f_.reshape(n, n_columns), -36.0, 36.0))
        folded = np.hstack([logits, np.ones((n, 1))])
        return folded / folded.sum(axis=1, keepdims=True)

    def full_nll(f_):
        p = probabilities(f_)
        total = 0.0
        for i in range(n):
            for k in range(n_columns):
                if y[i * n_columns + k] > 0:
                    total -= np.log(p[i, k])
            if labels[i] == n_classes - 1:
                total -= np.log(p[i, -1])
        return total

    partial = sum(
        -np.log(probabilities(f)[i, labels[i]])
        for i in range(n)
        if labels[i] < n_columns
    )
    assert risk == pytest.approx(partial, abs=1e-8)
    # The omitted reference-class term is not negligible on this fixture.
    assert full_nll(f) - risk > 1.0

    eps = 1e-6
    for j in range(f.size):
        f_up = f.copy()
        f_up[j] += eps
        f_down = f.copy()
        f_down[j] -= eps
        finite = (full_nll(f_up) - full_nll(f_down)) / (2 * eps)
        assert finite == pytest.approx(-gradient[j], abs=1e-6)

    # On a reference-class row the risk contributes zero while the gradient
    # contributes `-p_k`; both halves of the contract must hold at once.
    ref_row = int(np.flatnonzero(labels == n_classes - 1)[0])
    p_ref = probabilities(f)[ref_row, :n_columns]
    assert gradient[ref_row * n_columns:(ref_row + 1) * n_columns] \
        == pytest.approx(-p_ref, abs=1e-12)


def test_multinomial_selectors_refuse_instead_of_consuming_the_partial_risk():
    """`cvrisk` scores the full log-likelihood; `aic`/`logLik` refuse.

    Both `mboost` and mboostrs report an in-bag risk that omits the reference
    class. `cvrisk` scores held-out rows with the full negative
    log-likelihood instead, a deliberate deviation from mboost, because
    minimizing the partial quantity stopped too early. `aic`/`logLik` refuse
    because a normalized log-likelihood cannot be reconstructed from the
    partial risk. `negative_risk()` stays the mboost-comparable reported value.
    """
    rng = np.random.default_rng(5)
    n = 90
    x = rng.normal(size=n)
    y = rng.integers(0, 3, size=n).astype(float)
    model = mb.gamboost(
        "y ~ bols(x)", data={"x": x, "y": y}, family=mb.Multinomial(),
        control=mb.boost_control(20, 0.1),
    )
    assert np.isfinite(model.negative_risk())
    result = model.cvrisk(folds=mb.cv_weights(n, type="kfold", B=4, seed=0))
    assert np.all(np.isfinite(np.asarray(result.risk_matrix)))
    with pytest.raises(NotImplementedError, match="Kronecker"):
        model.aic()


def test_gammareg_loglik_is_the_gamma_density_at_the_estimated_shape():
    """The shape is re-estimated at every step (the registry used to say it
    was fixed at 1 and logLik refused). The log-likelihood is the Gamma
    density at the fitted mean and the reported shape."""
    from scipy import stats

    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 400)
    y = rng.gamma(3.0, np.exp(0.5 * x) / 3.0)
    model = mb.glmboost("y ~ x", data={"x": x, "y": y}, family=mb.GammaReg())
    shape = float(model.nuisance()[0])
    mean = np.asarray(model.predict(type="response"), dtype=float)
    expected = stats.gamma(shape, scale=mean / shape).logpdf(y).sum()
    assert model.logLik() == pytest.approx(expected, rel=1e-8)


# ─── families that are not likelihoods ─────────────────────────────


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("Laplace", "conditional median"),
        ("Quantile", "conditional quantile"),
        ("Expectile", "conditional expectile"),
        ("Huber", "robustified squared-error"),
    ],
)
def test_a_robust_loss_refuses_and_names_the_target(data, name, fragment):
    """The refusal says what the fit estimates, not only what it is not."""
    family = getattr(mb, name)()
    model = _fit("y ~ x", {"x": data["x"], "y": data["gaussian"]}, family)
    with pytest.raises(NotImplementedError, match=fragment):
        model.logLik()
    with pytest.raises(NotImplementedError, match="cvrisk"):
        model.aic()
    # The quantity the fit minimised is still reachable, under its own name.
    assert model.negative_risk() == pytest.approx(-_risk(model))


def test_the_refusal_points_at_negative_risk(data):
    model = _fit("y ~ x", {"x": data["x"], "y": data["gaussian"]}, mb.Laplace())
    with pytest.raises(NotImplementedError, match="negative_risk"):
        model.logLik()


# ─── cross-family comparison ───────────────────────────────────────


def test_gaic_table_ranks_models_sharing_a_measure(data):
    a = _fit("y ~ x", {"x": data["x"], "y": data["gaussian"]}, mb.Gaussian())
    b = mb.glmboost("y ~ x", data={"x": data["x"], "y": data["gaussian"]},
                    family=mb.Gaussian(), control=mb.boost_control(60, 0.1))
    rows = mb.gaic_table(a, b)
    assert len(rows) == 2


def test_gaic_table_refuses_a_non_likelihood_and_names_it(data):
    likelihood = _fit("y ~ x", {"x": data["x"], "y": data["gaussian"]}, mb.Gaussian())
    loss = _fit("y ~ x", {"x": data["x"], "y": data["gaussian"]}, mb.Laplace())
    with pytest.raises(NotImplementedError, match="Laplace"):
        mb.gaic_table(likelihood, loss)


def test_gaic_table_refuses_likelihoods_over_different_quantities(data):
    """A Gaussian fit of `time` and an AFT fit of the same column.

    Both are proper log-likelihoods; neither is a defect. They still cannot be
    ranked against each other, because one is a density of `T` and the other
    of `log(T)`, so the difference between their AICs is not a likelihood
    ratio.
    """
    gaussian = mb.glmboost(
        "time ~ x", data={"x": data["x"], "time": data["time"]},
        family=mb.Gaussian(), control=CONTROL,
    )
    aft = _fit(
        "Surv(time, event) ~ x",
        {"x": data["x"], "time": data["time"], "event": data["event"]},
        mb.Lognormal(),
    )
    with pytest.raises(NotImplementedError, match="different quantities"):
        mb.gaic_table(gaussian, aft)


def test_comparable_measures_names_which_family_broke_it():
    ok, reason = comparable_measures([mb.Poisson(), mb.NBinomial()])
    assert ok and reason == ""

    # A Gaussian density and a Poisson mass of the same counts (issue #60).
    ok, reason = comparable_measures([mb.Gaussian(), mb.Poisson()])
    assert not ok
    assert "counting: Poisson" in reason and "lebesgue: Gaussian" in reason

    ok, reason = comparable_measures([mb.Gaussian(), mb.Cindex()])
    assert not ok and "Cindex" in reason

    ok, reason = comparable_measures([mb.Gaussian(), mb.Lognormal()])
    assert not ok
    assert "log_response" in reason and "response" in reason


# ─── the table itself ──────────────────────────────────────────────


def test_every_row_is_internally_consistent():
    """A row cannot promise a log-likelihood and also decline to say why not."""
    for name, contract in FAMILY_LIKELIHOOD.items():
        if contract.loglik_source is None:
            assert contract.unavailable_reason, (
                f"{name} declines to provide a log-likelihood without saying why"
            )
            assert not contract.has_normalized_loglik, name
        else:
            assert contract.is_likelihood, name
            assert contract.loglik_measure is not None, name
        if not contract.is_likelihood:
            assert contract.loglik_measure is None, name
            assert contract.variance_function is None, name


def test_pearson_is_offered_only_with_a_variance_function():
    for name, contract in FAMILY_LIKELIHOOD.items():
        if "pearson" in contract.residual_kinds:
            assert contract.variance_function is not None, (
                f"{name} offers Pearson residuals with no variance function"
            )


def test_an_undeclared_family_is_reported_as_undeclared_not_as_a_loss():
    """A missing row must not read as 'not a likelihood'.

    The two lead to different corrections: one is fixed by adding a row, the
    other is a fact about the loss.
    """
    assert family_likelihood("NotAFamily") is None
    message = loglik_refusal("NotAFamily")
    assert "no declared likelihood contract" in message
    assert "FAMILY_LIKELIHOOD" in message


def test_every_standard_family_mboostrs_exports_has_a_row():
    """A family reachable from the public API must declare its contract."""
    exported = [
        name for name in mb.__all__
        if isinstance(getattr(mb, name, None), type)
        and not name.endswith("LSS")
        and getattr(mb, name).__module__ == "mboostrs.families"
    ]
    missing = sorted(set(exported) - set(FAMILY_LIKELIHOOD))
    assert not missing, f"families with no likelihood contract: {missing}"


# ─── distributional families ───────────────────────────────────────


def test_gaussianlss_risk_is_the_joint_nll():
    rng = np.random.default_rng(31)
    n = 300
    x = rng.normal(size=n)
    y = rng.normal(1 + 0.8 * x, np.exp(-0.3 + 0.4 * x))
    model = mb.glmboostLSS("y ~ x", data={"x": x, "y": y},
                           families=mb.GaussianLSS(),
                           control=mb.boost_control(60, 0.1))
    predicted = model.predict(type="response")
    reference = -float(np.sum(norm.logpdf(
        y, loc=np.asarray(predicted["mu"]), scale=np.asarray(predicted["sigma"])
    )))
    assert float(np.asarray(model._risk)[-1]) == pytest.approx(reference, abs=1e-8)
    assert model.logLik() == pytest.approx(-reference, abs=1e-8)
    assert model.negative_risk() == pytest.approx(model.logLik(), abs=1e-12)


def test_nbinomiallss_risk_is_the_joint_nll_with_size_equal_to_sigma():
    rng = np.random.default_rng(41)
    n = 400
    x = rng.normal(size=n)
    y = rng.negative_binomial(3, 3 / (3 + np.exp(0.6 * x))).astype(float)
    model = mb.glmboostLSS("y ~ x", data={"x": x, "y": y},
                           families=mb.NBinomialLSS(),
                           control=mb.boost_control(50, 0.1))
    predicted = model.predict(type="response")
    mu = np.asarray(predicted["mu"])
    size = np.asarray(predicted["sigma"])
    reference = -float(np.sum(
        gammaln(y + size) - gammaln(size) - gammaln(y + 1)
        + size * np.log(size / (size + mu))
        + y * np.log(mu / (size + mu))
    ))
    assert float(np.asarray(model._risk)[-1]) == pytest.approx(reference, abs=1e-8)


def test_lognormallss_is_a_likelihood_of_log_time():
    rng = np.random.default_rng(51)
    n = 400
    x = rng.normal(size=n)
    time = np.exp(0.5 * x + rng.normal(scale=0.5, size=n))
    event = (rng.uniform(size=n) < 0.75).astype(float)
    model = mb.glmboostLSS(
        "Surv(time, event) ~ x",
        data={"x": x, "time": time, "event": event},
        families=mb.LogNormalLSS(), control=mb.boost_control(40, 0.1),
    )
    predicted = model.predict(type="response")
    z = (np.log(time) - np.asarray(predicted["mu"])) / np.asarray(predicted["sigma"])
    observed = event > 0
    on_log_scale = -float(np.sum(np.where(
        observed, norm.logpdf(z) - np.log(np.asarray(predicted["sigma"])), norm.logsf(z)
    )))
    assert float(np.asarray(model._risk)[-1]) == pytest.approx(on_log_scale, abs=1e-8)
    assert family_likelihood(mb.LogNormalLSS()).loglik_measure == "log_response"


def test_every_lss_family_declares_a_contract():
    """The LSS contract is derived from the family object, not enumerated.

    A newly added distributional family therefore gets one without an edit
    here; this checks that the derivation actually reaches all of them.
    """
    lss_names = [
        name for name in mb.__all__
        if name.endswith("LSS") and isinstance(getattr(mb, name, None), type)
    ]
    assert len(lss_names) > 10, "expected the LSS family set to be non-trivial"
    for name in lss_names:
        try:
            family = getattr(mb, name)()
        except TypeError:
            continue  # needs constructor arguments; covered by its own tests
        contract = family_likelihood(family)
        assert contract is not None, f"{name} has no likelihood contract"
        assert contract.loglik_source == "negative_risk", name
        expected = "log_response" if getattr(family, "_survival", False) else "response"
        assert contract.loglik_measure == expected, name
