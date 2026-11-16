"""Smoke tests for LSS families (no R required).

Verifies each family can fit via glmboostLSS, predictions are finite, and
basic invariants hold. Covers the 9 families that have Rust implementations
but no Python test coverage. PIGLSS is excluded (not yet implemented in Rust).
"""

import numpy as np
import pytest
from mboostrs.gamlss import (
    BCCGLSS,
    JSULSS,
    BinomialLSS,
    GaussianLSS,
    GeometricLSS,
    GumbelLSS,
    NegativeBinomialLSS,
    Pareto2LSS,
    PoissonLSS,
    ZeroAdjustedPoissonLSS,
    glmboostLSS,
)

import mboostrs as mboost

N = 100
MSTOP = 50
NU = 0.1
SEED = 42


def _check(m, n_params, param_names, n):
    """Common invariants after a glmboostLSS fit."""
    assert m.mstop == MSTOP
    assert len(m.risk_) == MSTOP + 1
    assert np.all(np.isfinite(m.risk_))
    assert m.risk_[-1] <= m.risk_[0] + 1e-9

    for k in range(n_params):
        name = param_names[k]
        pred = m.predict(parameter=name)
        assert pred.shape == (n,), f"{name} predict shape mismatch"
        assert np.all(np.isfinite(pred)), f"{name} predict has non-finite values"

        pred_resp = m.predict(parameter=name, type="response")
        assert pred_resp.shape == (n,), f"{name} response predict shape mismatch"
        assert np.all(np.isfinite(pred_resp)), f"{name} response predict has non-finite values"


# ── 1-parameter families ──────────────────────────────────────────


@pytest.mark.parametrize("family_cls, y_fn", [
    (PoissonLSS,
     lambda rng, x1, x2: rng.poisson(np.exp(1.0 + 0.3 * x1 + 0.2 * x2))),
    (GeometricLSS,
     lambda rng, x1, x2:
         rng.geometric(p=1.0 / (1.0 + np.exp(1.0 + 0.3 * x1)), size=N) - 1),
    (BinomialLSS,
     lambda rng, x1, x2:
         1.0 / (1.0 + np.exp(-(0.5 + 0.5 * x1 + 0.3 * x2)))),
])
def test_oneparam_lss_smoke(family_cls, y_fn):
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    y = y_fn(rng, x1, x2)
    data = {"y": y, "x1": x1, "x2": x2}
    family = family_cls()

    m = glmboostLSS(
        "y ~ x1 + x2", data=data, families=family,
        control=mboost.boost_control(MSTOP, NU))
    _check(m, family.n_params, family.param_names, N)

    nd = {"x1": rng.normal(size=5), "x2": rng.normal(size=5), "y": np.ones(5)}
    m.predict(parameter=family.param_names[0], newdata=nd)


# ── 2-parameter families ──────────────────────────────────────────


@pytest.mark.parametrize("family_cls, y_fn", [
    (NegativeBinomialLSS,
     lambda rng, x1, x2: rng.negative_binomial(
         n=5, p=1.0 / (1.0 + np.exp(0.5 * x1)), size=N)),
    (GumbelLSS,
     lambda rng, x1, x2:
         rng.gumbel(loc=1.0 + 0.3 * x1, scale=np.exp(0.2 * x2))),
    (Pareto2LSS,
     lambda rng, x1, x2: np.abs(
         1.0 + 0.3 * x1 + rng.standard_t(df=3, size=N) * np.exp(0.2 * x2))
     + 0.1),
    (ZeroAdjustedPoissonLSS,
     lambda rng, x1, x2:
         rng.poisson(np.exp(1.0 + 0.3 * x1)) * rng.binomial(1, 0.8, size=N)),
])
def test_twoparam_lss_smoke(family_cls, y_fn):
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    y = y_fn(rng, x1, x2)
    data = {"y": y, "x1": x1, "x2": x2}
    family = family_cls()

    m = glmboostLSS(
        "y ~ x1 + x2", data=data, families=family,
        control=mboost.boost_control(MSTOP, NU))
    _check(m, family.n_params, family.param_names, N)

    nd = {"x1": rng.normal(size=5), "x2": rng.normal(size=5), "y": np.ones(5)}
    for k in range(family.n_params):
        m.predict(parameter=family.param_names[k], newdata=nd)


# ── 3-parameter families ──────────────────────────────────────────


def test_bccglss_smoke():
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    x3 = rng.normal(size=N)
    mu_true = 5.0 + 0.5 * x1
    sigma_true = np.exp(0.2 * x2)
    nu_true = 0.5 + 0.1 * x3
    t = rng.standard_t(df=5, size=N)
    y = np.abs(mu_true + sigma_true * np.abs(t) ** nu_true) + 0.01
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3}

    m = glmboostLSS(
        "y ~ x1 + x2 + x3", data=data, families=BCCGLSS(),
        control=mboost.boost_control(MSTOP, NU))

    assert m.mstop == MSTOP
    assert np.all(np.isfinite(m.risk_))
    for name in ("mu", "sigma", "nu"):
        pred = m.predict(parameter=name)
        assert np.all(np.isfinite(pred))
        assert pred.shape == (N,)


# ── 4-parameter families ──────────────────────────────────────────


def test_jsulss_smoke():
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    x3 = rng.normal(size=N)
    x4 = rng.normal(size=N)
    mu_true = 1.0 + 0.3 * x1
    sigma_true = np.exp(0.2 * x2)
    nu_true = 0.5 + 0.1 * x3
    tau_true = np.exp(0.1 * x4)
    y = mu_true + sigma_true * np.sinh(
        (rng.normal(size=N) * tau_true) / nu_true + np.arcsinh(nu_true))
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4}

    m = glmboostLSS(
        "y ~ x1 + x2 + x3 + x4", data=data, families=JSULSS(),
        control=mboost.boost_control(MSTOP, NU))

    assert m.mstop == MSTOP
    assert np.all(np.isfinite(m.risk_))
    for name in ("mu", "sigma", "nu", "tau"):
        pred = m.predict(parameter=name)
        assert np.all(np.isfinite(pred))
        assert pred.shape == (N,)


# ── weighted LSS fits ─────────────────────────────────────────────


def test_glmboostlss_weights():
    """Weighted LSS fit changes risk vs unweighted, and weights are stored."""
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=N)
    data = {"y": y, "x1": x1, "x2": x2}

    m_unw = glmboostLSS(
        "y ~ x1 + x2", data=data, families=GaussianLSS(),
        control=mboost.boost_control(MSTOP, NU))

    w = np.ones(N)
    w[:50] = 2.0
    m_w = glmboostLSS(
        "y ~ x1 + x2", data=data, families=GaussianLSS(),
        control=mboost.boost_control(MSTOP, NU), weights=w)

    assert m_w.model_weights()[0] == 2.0
    assert m_unw.model_weights()[0] == 1.0
    assert not np.isclose(m_unw.risk_[-1], m_w.risk_[-1])


def test_gamboostlss_weights():
    """Weighted gamboostLSS fit stores weights and changes risk."""
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=N)
    data = {"y": y, "x1": x1, "x2": x2}

    w = np.ones(N)
    w[:50] = 2.0
    m = mboost.gamboostLSS(
        "y ~ bbs(x1, df=4) + bbs(x2, df=4)", data=data,
        families=GaussianLSS(),
        control=mboost.boost_control(MSTOP, NU), weights=w)

    assert m.model_weights()[0] == 2.0
    assert m.model_weights()[-1] == 1.0
    assert np.all(np.isfinite(m.risk_))


def test_lss_weight_validation():
    """Invalid weights raise ValueError, not PanicException."""
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    y = 0.5 * x1 + rng.normal(scale=0.3, size=N)
    data = {"y": y, "x1": x1}

    with pytest.raises(ValueError, match="weights length"):
        glmboostLSS(
            "y ~ x1", data=data, families=GaussianLSS(),
            weights=np.ones(N + 1))

    with pytest.raises(ValueError, match="non-negative"):
        glmboostLSS(
            "y ~ x1", data=data, families=GaussianLSS(),
            weights=-np.ones(N))


def test_lss_update_weights():
    """update(weights=...) refits with new weights."""
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=N)
    data = {"y": y, "x1": x1, "x2": x2}

    m = glmboostLSS(
        "y ~ x1 + x2", data=data, families=GaussianLSS(),
        control=mboost.boost_control(MSTOP, NU))

    w = np.ones(N)
    w[:50] = 3.0
    m2 = m.update(mstop=10, weights=w)

    assert m2.mstop == 10
    assert m2.model_weights()[0] == 3.0
    assert m.model_weights()[0] == 1.0
