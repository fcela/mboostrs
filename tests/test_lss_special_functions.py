"""The hand-rolled special functions behind the LSS CDFs, checked against SciPy.

`GammaLSS`, `StudentTLSS`, and `BetaLSS` need the regularized incomplete gamma
and beta functions, which neither `math` nor `statistics` provides. SciPy is not a
runtime dependency of this package — `gamlss.py` hand-rolls the normal CDF and the
negative-binomial CDF for the same reason — so they are implemented in
`gamlss.py` and validated here against SciPy, which is present in development and
CI environments but never imported by the package itself.

Every test skips rather than fails when SciPy is absent, so the suite still runs
on a bare install.
"""
from __future__ import annotations

import numpy as np
import pytest
from mboostrs.gamlss import (
    BetaLSS,
    GammaLSS,
    StudentTLSS,
    _betainc,
    _gammainc_p,
    _student_t_cdf_scalar,
)

scipy_special = pytest.importorskip("scipy.special")
scipy_stats = pytest.importorskip("scipy.stats")


def test_package_does_not_import_scipy():
    """The runtime path must stay SciPy-free; only these tests may use it."""
    from pathlib import Path

    import mboostrs.gamlss as g

    source = Path(g.__file__).read_text(encoding="utf-8")
    assert "scipy" not in source


def test_regularized_incomplete_gamma_matches_scipy():
    worst = 0.0
    for a in (0.05, 0.5, 1.0, 2.5, 10.0, 100.0, 500.0):
        for x in (0.0, 1e-6, 0.1, a, 2 * a + 1.0, 50.0, 1000.0):
            worst = max(worst, abs(_gammainc_p(a, x) - float(scipy_special.gammainc(a, x))))
    # Tail evaluations lose a few digits to the continued fraction's convergence
    # tolerance; 1e-13 is far tighter than any CDF consumer needs.
    assert worst < 1e-13, worst


def test_regularized_incomplete_beta_matches_scipy():
    worst, where = 0.0, None
    for a in (0.1, 0.5, 1.0, 3.0, 20.0, 200.0):
        for b in (0.1, 0.5, 1.0, 3.0, 20.0, 200.0):
            for x in (1e-9, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-9):
                err = abs(_betainc(a, b, x) - float(scipy_special.betainc(a, b, x)))
                if err > worst:
                    worst, where = err, (a, b, x)
    # Worst case is a = b = 0.5 at x = 1 - 1e-9, an extreme tail, at ~12
    # significant digits.
    assert worst < 1e-11, (worst, where)


def test_student_t_cdf_matches_scipy():
    worst = 0.0
    for df in (1.0, 1.5, 4.0, 10.0, 50.0, 500.0):
        for t in (-40.0, -3.0, -0.5, 0.0, 0.5, 3.0, 40.0):
            worst = max(worst, abs(_student_t_cdf_scalar(t, df) - float(scipy_stats.t.cdf(t, df=df))))
    assert worst < 1e-13, worst


def test_gamma_cdf_and_quantile_match_scipy():
    mu = np.array([0.5, 1.0, 3.0, 10.0])
    sigma = np.array([0.7, 1.0, 2.5, 8.0])
    fam = GammaLSS()
    y = np.array([0.1, 1.0, 2.0, 12.0])
    assert np.allclose(
        fam.cdf(y, {"mu": mu, "sigma": sigma}),
        scipy_stats.gamma.cdf(y, a=sigma, scale=mu / sigma),
        atol=1e-12,
    )
    p = np.array([0.05, 0.25, 0.5, 0.95])
    assert np.allclose(
        fam.quantile(p, {"mu": mu, "sigma": sigma}),
        scipy_stats.gamma.ppf(p, a=sigma, scale=mu / sigma),
        atol=1e-9,
    )


def test_student_t_cdf_and_quantile_match_scipy():
    mu = np.array([0.0, 1.0, -2.0, 5.0])
    sigma = np.array([1.0, 2.0, 0.5, 3.0])
    df = np.array([1.5, 4.0, 10.0, 50.0])
    params = {"mu": mu, "sigma": sigma, "df": df}
    fam = StudentTLSS()
    y = np.array([-1.0, 0.5, -3.0, 6.0])
    assert np.allclose(
        fam.cdf(y, params), scipy_stats.t.cdf((y - mu) / sigma, df=df), atol=1e-12
    )
    for p_scalar in (0.001, 0.025, 0.25, 0.5, 0.75, 0.975, 0.999):
        p = np.full(4, p_scalar)
        assert np.allclose(
            fam.quantile(p, params), mu + sigma * scipy_stats.t.ppf(p, df=df), atol=1e-9
        )


def test_student_t_median_is_exact_not_merely_close():
    """The CDF is numerically flat at 0.5, so the median is special-cased.

    `df/(df + t^2)` rounds to 1 for |t| below sqrt(df * eps), leaving a band in
    which no root finder can localize. Without the symmetry handling this
    returned about -3e-8 instead of the location.
    """
    fam = StudentTLSS()
    got = fam.quantile(
        np.array([0.5]),
        {"mu": np.array([3.0]), "sigma": np.array([2.0]), "df": np.array([7.0])},
    )
    assert got[0] == 3.0


def test_beta_cdf_and_quantile_match_scipy():
    mu = np.array([0.2, 0.5, 0.7, 0.9])
    phi = np.array([2.0, 5.0, 10.0, 30.0])
    fam = BetaLSS()
    y = np.array([0.1, 0.4, 0.65, 0.95])
    assert np.allclose(
        fam.cdf(y, {"mu": mu, "phi": phi}),
        scipy_stats.beta.cdf(y, a=mu * phi, b=(1 - mu) * phi),
        atol=1e-12,
    )
    p = np.array([0.05, 0.25, 0.5, 0.95])
    assert np.allclose(
        fam.quantile(p, {"mu": mu, "phi": phi}),
        scipy_stats.beta.ppf(p, a=mu * phi, b=(1 - mu) * phi),
        atol=1e-12,
    )


def test_cdf_and_quantile_round_trip():
    """quantile(cdf(y)) recovers y for the continuous families."""
    fam = GammaLSS()
    params = {"mu": np.array([2.0, 5.0]), "sigma": np.array([1.5, 3.0])}
    y = np.array([1.0, 4.0])
    assert np.allclose(fam.quantile(fam.cdf(y, params), params), y, atol=1e-8)

    fam = BetaLSS()
    params = {"mu": np.array([0.3, 0.6]), "phi": np.array([4.0, 12.0])}
    y = np.array([0.25, 0.7])
    assert np.allclose(fam.quantile(fam.cdf(y, params), params), y, atol=1e-10)


@pytest.mark.parametrize(
    ("family", "params", "bad"),
    [
        (GammaLSS(), {"mu": np.array([1.0]), "sigma": np.array([-1.0])}, np.array([1.0])),
        (BetaLSS(), {"mu": np.array([1.5]), "phi": np.array([2.0])}, np.array([0.5])),
        (
            StudentTLSS(),
            {"mu": np.array([0.0]), "sigma": np.array([1.0]), "df": np.array([0.0])},
            np.array([0.0]),
        ),
    ],
)
def test_invalid_parameters_raise(family, params, bad):
    with pytest.raises(ValueError):
        family.cdf(bad, params)
