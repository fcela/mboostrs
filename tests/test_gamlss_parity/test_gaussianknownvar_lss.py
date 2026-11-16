"""GaussianKnownVarLSS: fit, decomposition, and resampling-path tests.

The family has no R counterpart, so the oracles are analytical: a synthetic
population with known excess variance and known sampling variances must be
decomposed back to its components, the total-scale cdf/quantile must invert
each other, and the resampling paths (cvrisk, stabsel) must run with the
doubled cbind response kept aligned.
"""

import numpy as np
import pytest
from mboostrs.cvrisk import cv_weights
from mboostrs.gamlss import GaussianKnownVarLSS, GaussianLSS

import mboostrs as mb


@pytest.fixture(scope="module")
def synthetic():
    """y = mu(x) + sqrt(s2 + excess(x)^2) * noise, n = 400.

    Sampling variances span two orders of magnitude so the attenuation term
    varies strongly across observations.
    """
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(-1, 1, n)
    mu = 10.0 + 2.0 * x
    s2 = np.exp(rng.uniform(-4.0, 0.5, n))  # sampling variances
    excess_var = np.exp(0.3 + 0.8 * x)  # true excess variance
    y = mu + np.sqrt(s2 + excess_var) * rng.normal(size=n)
    data = {"x": x, "y": y, "s2": s2, "pop": np.exp(3.0 + x) + rng.uniform(size=n)}
    return data, mu, s2, excess_var


def _fit(data, families, mstop=150, method="noncyclic"):
    # Non-cyclic: the sigma-gradient is attenuated by exp(eta)/(s2+exp(eta))
    # where sampling variance dominates, so sigma needs many more updates than
    # mu and a shared cyclic budget underfits it (plan 03's own risk note).
    # sigma keeps an intercept-capable learner: with only center=True /
    # intercept=False terms the sigma level can never leave its offset, and
    # the moment-matched offset (marginal variance minus mean s2, which
    # still contains the between-observation mean structure) overstates the
    # excess. The intercept learner is what lets the boosted sigma correct
    # its own level.
    forms = {
        "mu": "cbind(y, s2) ~ bols(x, intercept=False) + bbs(x, center=True, df=1)",
        "sigma": "cbind(y, s2) ~ bols(x)",
    }
    return mb.gamboostLSS(forms, data=data, families=families, method=method,
                          control=mb.boost_control(mstop, 0.1))


def test_naive_sigma_absorbs_sampling_noise(synthetic):
    # The plan's predicted failure mode, as a control: plain GaussianLSS on
    # the same outcomes estimates TOTAL dispersion, inflated by s2.
    data, mu, s2, excess_var = synthetic
    forms = {"mu": "y ~ bbs(x)", "sigma": "y ~ bols(x)"}
    m = mb.gamboostLSS(forms, data=data, families=GaussianLSS(), method="noncyclic",
                       control=mb.boost_control(2000, 0.1))
    sigma_naive = m.predict(type="response")["sigma"]
    # naive sigma estimates the TOTAL dispersion, which includes the known
    # sampling noise the corrected family strips out.
    true_total = np.sqrt(s2 + excess_var)
    assert np.abs(sigma_naive.mean() - true_total.mean()) < 0.15
    assert sigma_naive.mean() > np.sqrt(excess_var).mean() + 0.05


def test_corrected_recovers_excess_variance(synthetic):
    data, mu, s2, excess_var = synthetic
    m = _fit(data, GaussianKnownVarLSS(), mstop=2000)
    pred = m.predict(type="response")
    excess_sd = pred["sigma"]
    true_excess_sd = np.sqrt(excess_var)
    assert np.abs(excess_sd.mean() - true_excess_sd.mean()) < 0.15
    # The total scale is the one that must cover the outcomes.
    total = np.sqrt(s2 + excess_sd**2)
    z = (data["y"] - pred["mu"]) / total
    assert np.abs(z.mean()) < 0.15
    assert 0.75 < z.std() < 1.25


def test_pdf_cdf_quantile_invert_on_total_scale(synthetic):
    data, *_ = synthetic
    fam = GaussianKnownVarLSS()
    m = _fit(data, GaussianKnownVarLSS(), mstop=80)
    pred = m.predict(type="response")
    params = {"mu": pred["mu"], "sigma": pred["sigma"]}
    p = np.linspace(0.05, 0.95, len(data["y"]))
    q = fam.quantile(p, params, s2=data["s2"])
    back = fam.cdf(q, params, s2=data["s2"])
    np.testing.assert_allclose(back, p, atol=2e-6)
    dens = fam.pdf(q, params, s2=data["s2"])
    assert np.all(dens > 0)


def test_cvrisk_and_stabsel_run_on_doubled_response(synthetic):
    data, *_ = synthetic
    forms = {
        "mu": "cbind(y, s2) ~ bols(x, intercept=False) + bols(pop, intercept=False)",
        "sigma": "cbind(y, s2) ~ bols(x, intercept=False) + bols(pop, intercept=False)",
    }
    m = mb.gamboostLSS(forms, data=data, families=GaussianKnownVarLSS(),
                       control=mb.boost_control(60, 0.1))
    folds = cv_weights(len(data["y"]), type="kfold", B=5, seed=1)
    cv = m.cvrisk(folds=folds, grid=[[30, 30], [60, 60]])
    assert len(cv.optimal_mstop) == 2
    stab = m.stabsel(q=2, PFER=1.0, B=10)
    freqs = np.asarray(stab.max_)
    assert freqs.shape == (4,)
    assert freqs.max() <= 1.0 and freqs.min() >= 0.0


def test_rejects_plain_response(synthetic):
    data, *_ = synthetic
    with pytest.raises(ValueError, match="plain response|cbind"):
        mb.gamboostLSS({"mu": "y ~ bbs(x)", "sigma": "y ~ bbs(x)"},
                       data=data, families=GaussianKnownVarLSS(),
                       control=mb.boost_control(10, 0.1))


def test_offset_initialization_is_moment_matched(synthetic):
    # offset(sigma) = log(total var - mean s2): excess variance at start.
    data, *_ = synthetic
    m = _fit(data, GaussianKnownVarLSS(), mstop=1)
    y, s2 = data["y"], data["s2"]
    expected = np.log(max(y.var(ddof=1) - s2.mean(), 1e-12))
    assert abs(m.offset["sigma"] - expected) < 1e-8
