"""Each family's negative gradient is the derivative of its reported risk (issue #69).

Checked by central finite differences of the risk `family_diagnostics` /
`lss_family_diagnostics` return, for every differentiable family. A wrong
gradient or a risk with the wrong sign (AGENTS gotchas 24, 27, 28) fails here
however good the fits look.

Three standard families follow mboost's scaling, which the test states rather
than hides: Gaussian and GaussReg report the risk (y - f)^2 with gradient
y - f, half its derivative; Binomial divides its gradient by log 2. Huber
recalibrates its delta from the current residuals and AUC is a rank loss, so
neither has a finite-difference derivative at fixed state.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._rust import family_diagnostics, lss_family_diagnostics
from mboostrs.families import _family_params, _family_rust_name

N = 12
H = 1e-6

# gradient = SCALE * (-dRisk/df)
STANDARD = {
    "Gaussian": 0.5, "GaussReg": 0.5, "Binomial": 1 / np.log(2.0),
    "AdaExp": 1.0, "HingeLoss": 1.0, "Laplace": 1.0, "Quantile": 1.0,
    "Expectile": 1.0, "Poisson": 1.0, "NBinomial": 1.0, "GammaReg": 1.0,
    "CoxPH": 1.0, "Weibull": 1.0, "Loglog": 1.0, "Lognormal": 1.0,
    "Gehan": 1.0, "Cindex": 1.0,
}
SURVIVAL = {"CoxPH", "Weibull", "Loglog", "Lognormal", "Gehan", "Cindex"}


def _standard_response(name, rng):
    if name in ("Binomial", "AdaExp", "HingeLoss"):
        return rng.integers(0, 2, N).astype(float)
    if name in ("Poisson", "NBinomial"):
        return rng.poisson(3, N).astype(float)
    if name == "GammaReg":
        return rng.gamma(2, 1, N)
    if name in SURVIVAL:
        return np.concatenate([rng.exponential(1, N), rng.integers(0, 2, N).astype(float)])
    return rng.normal(size=N)


@pytest.mark.parametrize("name", sorted(STANDARD))
def test_standard_family_gradient(name):
    rng = np.random.default_rng(1)
    family = getattr(mb, name)()
    args = (_family_rust_name(family), _family_params(family))
    y = _standard_response(name, rng)
    f = rng.normal(0, 0.3, N)
    w = np.ones(N)
    _, gradient, _ = family_diagnostics(*args, y, f, w)
    numeric = np.empty(N)
    for i in range(N):
        up, down = f.copy(), f.copy()
        up[i] += H
        down[i] -= H
        numeric[i] = -(family_diagnostics(*args, y, up, w)[2]
                       - family_diagnostics(*args, y, down, w)[2]) / (2 * H)
    np.testing.assert_allclose(np.asarray(gradient), STANDARD[name] * numeric,
                               rtol=1e-5, atol=1e-6)


LSS_DATA = {
    "GaussianLSS": "normal", "StudentTLSS": "normal", "LogisticLSS": "normal",
    "GumbelLSS": "normal", "JSULSS": "normal",
    "GammaLSS": "positive", "InverseGaussianLSS": "positive", "Pareto2LSS": "positive",
    "BCCGLSS": "positive", "BetaLSS": "unit", "BinomialLSS": "binary",
    "PoissonLSS": "count", "NBinomialLSS": "count", "NegativeBinomialLSS": "count",
    "GeometricLSS": "count", "ZIPoLSS": "count", "ZINBLSS": "count",
    "ZeroAdjustedPoissonLSS": "count", "PIGLSS": "count",
    "ZeroAdjustedGammaLSS": "zero_positive",
    "LogLogLSS": "survival", "LogNormalLSS": "survival", "WeibullLSS": "survival",
}


def _lss_response(kind, rng):
    if kind == "normal":
        return rng.normal(size=N)
    if kind == "positive":
        return rng.gamma(3, 1, N)
    if kind == "unit":
        return rng.uniform(0.1, 0.9, N)
    if kind == "binary":
        return rng.integers(0, 2, N).astype(float)
    if kind == "count":
        return rng.poisson(3, N).astype(float)
    if kind == "zero_positive":
        y = rng.gamma(3, 1, N)
        y[:3] = 0.0
        return y
    return np.concatenate([rng.exponential(2, N), rng.integers(0, 2, N).astype(float)])


def _lss_check(name, y, etas):
    family = getattr(mb, name)()
    args = (family._rust_name, family.n_params)
    w = np.ones(N)
    _, gradients, _ = lss_family_diagnostics(*args, y, etas, w)
    for k in range(family.n_params):
        numeric = np.empty(N)
        for i in range(N):
            up = [e.copy() for e in etas]
            down = [e.copy() for e in etas]
            up[k][i] += H
            down[k][i] -= H
            numeric[i] = -(lss_family_diagnostics(*args, y, up, w)[2]
                           - lss_family_diagnostics(*args, y, down, w)[2]) / (2 * H)
        np.testing.assert_allclose(np.asarray(gradients[k]), numeric,
                                   rtol=1e-4, atol=1e-6, err_msg=f"{name} parameter {k}")


@pytest.mark.parametrize("name", sorted(LSS_DATA))
def test_lss_family_gradient(name):
    rng = np.random.default_rng(2)
    family = getattr(mb, name)()
    y = _lss_response(LSS_DATA[name], rng)
    etas = [rng.normal(0, 0.2, N) for _ in range(family.n_params)]
    _lss_check(name, y, etas)


@pytest.mark.parametrize("name", ["BCTLSS", "BCPELSS"])
def test_box_cox_t_and_power_exponential_gradient(name):
    # mu has an identity link here, so it is evaluated near the data's scale.
    rng = np.random.default_rng(3)
    y = rng.gamma(3, 1, N)
    base = [3.0, np.log(0.3), 0.0, np.log(5.0)]
    etas = [b + rng.normal(0, 0.1, N) for b in base]
    _lss_check(name, y, etas)
