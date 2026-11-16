"""The core per-row LSS negative log-likelihood against the Python densities.

`cv_risk` scores the held-out rows of a distributional model with the core's
`lss_row_nll`, the per-observation terms the core LSS risk sums (issue #821).
Each family's Python `pdf` is an independent implementation of the same
density; the two must agree row by row, and the rows must sum to the core risk.
"""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import gamlss
from mboostrs._rust import lss_family_diagnostics, lss_row_nll

N = 7
REAL = np.array([-1.2, -0.3, 0.4, 1.1, 2.0, 0.9, -0.6])
POSITIVE = np.array([0.3, 0.8, 1.2, 2.5, 4.0, 1.7, 0.6])
COUNTS = np.array([0.0, 1.0, 2.0, 5.0, 0.0, 3.0, 1.0])

# Each family with a one-column response, and a response in its support.
FIXTURES = {
    "GaussianLSS": (gamlss.GaussianLSS(), REAL),
    "GammaLSS": (gamlss.GammaLSS(), POSITIVE),
    "StudentTLSS": (gamlss.StudentTLSS(), REAL),
    "BetaLSS": (gamlss.BetaLSS(), np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.6])),
    "NBinomialLSS": (gamlss.NBinomialLSS(), COUNTS),
    "ZIPoLSS": (gamlss.ZIPoLSS(), COUNTS),
    "ZINBLSS": (gamlss.ZINBLSS(), COUNTS),
    "PoissonLSS": (gamlss.PoissonLSS(), COUNTS),
    "GeometricLSS": (gamlss.GeometricLSS(), COUNTS),
    "BinomialLSS": (gamlss.BinomialLSS(), np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])),
    "NegativeBinomialLSS": (gamlss.NegativeBinomialLSS(), COUNTS),
    "ZeroAdjustedGammaLSS": (gamlss.ZeroAdjustedGammaLSS(),
                             np.array([0.0, 0.5, 1.5, 0.0, 3.0, 0.7, 2.2])),
    "InverseGaussianLSS": (gamlss.InverseGaussianLSS(), POSITIVE),
    "LogisticLSS": (gamlss.LogisticLSS(), REAL),
    "GumbelLSS": (gamlss.GumbelLSS(), REAL),
    "Pareto2LSS": (gamlss.Pareto2LSS(), POSITIVE),
    "ZeroAdjustedPoissonLSS": (gamlss.ZeroAdjustedPoissonLSS(), COUNTS),
    "BCCGLSS": (gamlss.BCCGLSS(), POSITIVE),
    "JSULSS": (gamlss.JSULSS(), REAL),
    "BCTLSS": (gamlss.BCTLSS(), POSITIVE),
    "BCPELSS": (gamlss.BCPELSS(), POSITIVE),
    "LogNormalLSS.uncensored": (gamlss.LogNormalLSS().uncensored(), POSITIVE),
    "LogLogLSS.uncensored": (gamlss.LogLogLSS().uncensored(), POSITIVE),
    "WeibullLSS.uncensored": (gamlss.WeibullLSS().uncensored(), POSITIVE),
}


def _etas(k):
    # The first parameter sits near 1, inside the domain of the families
    # (BCT, BCPE, BCCG) whose location has an identity link.
    rows = np.arange(N, dtype=np.float64)
    return [np.ascontiguousarray((0.8 if p == 0 else -0.2 + 0.05 * p) + 0.1 * rows)
            for p in range(k)]


def _check(family, y_core, y_pdf, pdf_args=(), shift=0.0):
    k = len(family.param_names)
    etas = _etas(k)
    loss, parameters = lss_row_nll(family._rust_name, k, np.ascontiguousarray(y_core), etas)
    loss = np.asarray(loss)
    params = {name: np.asarray(p) for name, p in zip(family.param_names, parameters, strict=True)}
    # The response-scale parameters are the Python inverse links of the same etas.
    for j, name in enumerate(family.param_names):
        np.testing.assert_allclose(params[name], family.response(j, etas[j]), rtol=1e-12)
    expected = -np.log(np.asarray(family.pdf(y_pdf, params, *pdf_args))) + shift
    np.testing.assert_allclose(loss, expected, rtol=1e-8, atol=0.0)
    # The rows sum, under any weights, to the risk the core boosts and
    # cross-validates.
    w = np.array([0.5, 1.0, 2.0, 0.0, 1.5, 1.0, 0.25])
    _offsets, _grads, risk = lss_family_diagnostics(
        family._rust_name, k, np.ascontiguousarray(y_core), etas, w)
    assert np.sum(w * loss) == pytest.approx(risk, rel=1e-12)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_core_row_nll_is_the_python_negative_log_density(name):
    family, y = FIXTURES[name]
    _check(family, y, y)


def test_known_variance_rows_read_the_variance_column():
    s2 = np.array([0.1, 0.2, 0.0, 0.1, 0.5, 0.3, 0.05])
    _check(gamlss.GaussianKnownVarLSS(), np.concatenate([REAL, s2]), REAL, (s2,))


def test_dirichlet_rows_read_every_component():
    y = np.array([[0.2, 0.3, 0.5], [0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4],
                  [0.6, 0.1, 0.3], [0.25, 0.25, 0.5], [0.4, 0.4, 0.2]])
    column_major = np.ascontiguousarray(y.T.ravel())
    _check(gamlss.DirichletLSS(3), column_major, y)


@pytest.mark.parametrize("family", [gamlss.LogNormalLSS(), gamlss.LogLogLSS(),
                                    gamlss.WeibullLSS()], ids=lambda f: type(f).__name__)
def test_censored_survival_event_rows_are_the_log_time_density(family):
    # With every status an event, the censored family's loss is the density of
    # log(T): -log f_T(t) - log(t).
    y = np.concatenate([POSITIVE, np.ones(N)])
    _check(family, y, POSITIVE, shift=-np.log(POSITIVE))


def test_a_response_outside_the_support_is_refused():
    with pytest.raises(ValueError, match="GammaLSS"):
        lss_row_nll("gammalss", 2, np.array([1.0, -1.0]), [np.zeros(2), np.zeros(2)])


def test_cv_risk_fold_score_is_the_core_held_out_risk():
    # The fold score cv_risk reports is the weighted mean core loss of the
    # fold's held-out rows at the fold fit's link-scale predictions.
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 120)
    y = np.exp(0.5 + x) * rng.gamma(4.0, 0.25, 120)
    data = {"x": x, "y": y}
    result = mb.cv_risk("bols(x)", data, "y", family=gamlss.GammaLSS(), folds=3, mstop=30,
                        seed=2, return_result=True)
    records = result.records
    folds = sorted({r["fold"] for r in records})
    for b in folds:
        rows = [r for r in records if r["fold"] == b]
        loss = np.array([r["loss"] for r in rows])
        weight = np.array([r["evaluation_weight"] for r in rows])
        score = next(r["score"] for r in result.folds if r["fold"] == b)
        assert score == pytest.approx(np.sum(weight * loss) / np.sum(weight), rel=1e-12)
        # Each record's loss is the core loss at its response-scale parameters.
        params = {"mu": np.array([r["prediction"][0] for r in rows]),
                  "sigma": np.array([r["prediction"][1] for r in rows])}
        observed = np.array([r["observed"] for r in rows])
        np.testing.assert_allclose(
            loss, -np.log(gamlss.GammaLSS().pdf(observed, params)), rtol=1e-8)
