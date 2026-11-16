"""Poisson-inverse Gaussian fits, and its numbers are the ones PIG claims.

P4.1 item 1: "Implement the exact PIG negative log-likelihood and gradients
with a tested modified-Bessel kernel; keep PIG fitting rejected until finite
differences, density normalization, simulation recovery, and R parity pass."

What was there approximated the mu gradient as `y - mu` — the Poisson one,
with a comment saying so — and computed the risk as a Poisson negative
log-likelihood outright. Both ran and produced numbers, so a fit would have
converged to the wrong model quietly; `check_y` refusing every response was
the only thing keeping that off the road.

The density is now exact. `K_{y-1/2}` has a closed form at half-integer
orders and an exact three-term recurrence, run in logs because it overflows a
double for a count of a few hundred at a small argument — which a PIG fit on
count data reaches. Both gradients follow `gamlss.dist`'s own algebra, which
writes them in terms of one quantity: `ty = (mu/(alpha sigma))
K_{y+1/2}(alpha)/K_{y-1/2}(alpha)`.

The kernel, the density and the gradients are checked in Rust, where they
live, against `gamlss.dist` 6.1-1 and against a finite difference of the risk
— see `mod pig_contract` in `crates/mboostrs-core/src/lss.rs`. This file
checks the Python frontend: that a fit runs, recovers what it was simulated
from, and lands near R's maximum likelihood.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb

N = 3000
TRUE_INTERCEPT = 0.5
TRUE_SLOPE = 0.6
TRUE_SIGMA = 0.8


@pytest.fixture(scope="module")
def simulated():
    """PIG is a Poisson mixed over an inverse Gaussian of mean 1, so it can be
    drawn without the density: mix, then draw counts."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=N)
    mu = np.exp(TRUE_INTERCEPT + TRUE_SLOPE * x)
    rate = rng.wald(mean=1.0, scale=1.0 / TRUE_SIGMA, size=N) * mu
    return {"x": x, "y": rng.poisson(rate).astype(float)}, mu


@pytest.fixture(scope="module")
def fitted(simulated):
    data, _ = simulated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.glmboostLSS("y ~ x", data, family=mb.PIGLSS(),
                              control=mb.boost_control(2000, 0.1))


def test_it_fits_at_all(fitted):
    """It did not: `check_y` refused every response with "PIGLSS is not yet
    implemented (requires Bessel K function)"."""
    assert set(fitted.fitted()) == {"mu", "sigma"}
    for values in fitted.fitted().values():
        assert np.all(np.isfinite(values))


def test_it_recovers_what_it_was_simulated_from(fitted, simulated):
    _, mu_true = simulated
    mu_hat = np.exp(fitted.fitted()["mu"])
    sigma_hat = np.exp(fitted.fitted()["sigma"])
    assert np.corrcoef(np.log(mu_hat), np.log(mu_true))[0, 1] > 0.99
    assert float(mu_hat.mean()) == pytest.approx(float(mu_true.mean()),
                                                 rel=0.05)
    assert float(sigma_hat.mean()) == pytest.approx(TRUE_SIGMA, rel=0.1)


def test_a_boosted_fit_lands_near_r_maximum_likelihood(fitted):
    """`gamlss(y ~ x, sigma.formula = ~1, family = PIG())` on the same 3,000
    rows gives mu = (0.5073, 0.5763) and sigma = 0.8163.

    Boosting and maximum likelihood do not agree exactly — shrinkage is the
    point of the former — so this is a tolerance, not a parity claim. The
    numerics are verified against `gamlss.dist` in Rust.
    """
    slope = fitted.coef()["mu"]["x"]
    assert slope == pytest.approx(0.5763, abs=0.05)
    sigma = float(np.exp(fitted.fitted()["sigma"]).mean())
    assert sigma == pytest.approx(0.8163, abs=0.05)


def test_it_is_not_the_poisson_it_used_to_approximate(simulated):
    """The old gradient was the Poisson one. On overdispersed counts the two
    families disagree about sigma by construction, so a fit that still
    matched Poisson would mean the approximation is back."""
    data, _ = simulated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poisson = mb.glmboostLSS("y ~ x", data, family=mb.PoissonLSS(),
                                 control=mb.boost_control(2000, 0.1))
        pig = mb.glmboostLSS("y ~ x", data, family=mb.PIGLSS(),
                             control=mb.boost_control(2000, 0.1))
    variance_ratio = float(data["y"].var() / data["y"].mean())
    assert variance_ratio > 2, "the fixture should be overdispersed"
    assert not np.allclose(np.exp(pig.fitted()["mu"]),
                           np.exp(poisson.fitted()["mu"]), rtol=1e-3)


def test_a_non_count_response_is_refused_by_name(simulated):
    data, _ = simulated
    broken = dict(data)
    broken["y"] = data["y"].copy()
    broken["y"][0] = 1.5
    with pytest.raises(ValueError, match="whole count"):
        mb.glmboostLSS("y ~ x", broken, family=mb.PIGLSS(),
                       control=mb.boost_control(20, 0.1))


def test_the_ledger_no_longer_says_the_gradient_is_approximate():
    row = next(r for r in mb.family_capabilities(as_frame=False)
               if r["name"] == "PIGLSS")
    assert row["fit"] == "yes"
    assert row["known_deviation"] == ""
    assert "Bessel" in row["notes"]
    assert row["python_fit_status"] == "smoke", (
        "the numerics are verified in Rust; the Python frontend's own "
        "evidence is a simulation recovery against R's MLE, which is a "
        "tolerance rather than a parity claim")


def test_cvrisk_runs_on_a_pig_fit(fitted):
    """It was `unsupported` in the ledger because fitting was."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fitted.cvrisk(B=3, seed=1)
    assert np.all(np.isfinite(np.asarray(result.risk_matrix)))
