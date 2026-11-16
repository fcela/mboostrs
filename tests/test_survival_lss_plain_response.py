"""Log-normal, Weibull and log-logistic LSS on a plain positive response (#333).

The three families were built for `Surv(time, status)`. A plain positive
response is now treated as uncensored (every status 1) and fitted on its own
scale, so its risk is the negative log-likelihood of the response under the
family's density, the same scale as `GammaLSS` and the other `realplus`
families. `scipy.stats` supplies the reference densities; nothing here reads
the family's own density code.
"""

import numpy as np
import pytest
from scipy import stats

import mboostrs as mb
from mboostrs.model_selection import FAMILY_SETS, scan_lss_families

SPEC = {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"}
CONTROL = mb.boost_control(60, 0.1)

# Reference density of the response T, with e = (log t - mu) / sigma, as scipy
# distributions: lognorm(s=sigma, scale=exp(mu)); weibull_min with shape 1/sigma
# and scale exp(mu), so P(T > t) = exp(-exp(e)); fisk (the log-logistic) with
# shape 1/sigma and scale exp(mu).
REFERENCE = {
    "LogNormalLSS": lambda y, mu, sigma: stats.lognorm(
        s=sigma, scale=np.exp(mu)).logpdf(y),
    "WeibullLSS": lambda y, mu, sigma: stats.weibull_min(
        c=1.0 / sigma, scale=np.exp(mu)).logpdf(y),
    "LogLogLSS": lambda y, mu, sigma: stats.fisk(
        c=1.0 / sigma, scale=np.exp(mu)).logpdf(y),
}
FAMILIES = sorted(REFERENCE)


def _positive_data(seed=21, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.exp(0.5 + 0.8 * np.sin(3 * x) + 0.4 * rng.standard_normal(n))
    return {"x": x, "y": y}


@pytest.mark.parametrize("name", FAMILIES)
def test_risk_is_the_exact_negative_log_likelihood_of_the_response(name):
    data = _positive_data()
    model = mb.gamboostLSS(SPEC, data, families=getattr(mb, name)(),
                           control=CONTROL)
    params = model.predict_parameters(type="response")
    mu = np.asarray(params["mu"], dtype=np.float64)
    sigma = np.asarray(params["sigma"], dtype=np.float64)
    exact = -float(np.sum(REFERENCE[name](data["y"], mu, sigma)))
    assert -model.negative_risk() == pytest.approx(exact, rel=1e-10)


@pytest.mark.parametrize("name", FAMILIES)
def test_plain_response_fit_is_the_survival_fit_with_every_event(name):
    """The censored code path with status 1 everywhere optimises the same
    objective up to the Jacobian `sum(log y)`, which does not depend on the
    parameters. The plain fit must therefore give the same parameters and a
    risk larger by exactly that sum."""
    data = _positive_data()
    with_status = dict(data, d=np.ones(len(data["y"])))
    plain = mb.gamboostLSS(SPEC, data, families=getattr(mb, name)(),
                           control=CONTROL)
    surv = mb.gamboostLSS({"mu": "Surv(y, d) ~ bbs(x, df=4)",
                           "sigma": "Surv(y, d) ~ bols(x)"},
                          with_status, families=getattr(mb, name)(),
                          control=CONTROL)
    jacobian = float(np.sum(np.log(data["y"])))
    # The Brent search for the offsets sees the two objectives differing only
    # by a constant, so its rounding can move the offset by a few ulps; the
    # tolerance is still far below the Jacobian (hundreds of nats here).
    assert -plain.negative_risk() == pytest.approx(
        -surv.negative_risk() + jacobian, rel=1e-8)
    # The Weibull `mu` offset is a Brent search over [0, max t] on a flat
    # objective, so it stops within the optimiser tolerance (about 2e-8 here).
    # The fitted parameters therefore agree to that tolerance, not to ulps.
    plain_params = plain.predict_parameters(type="response")
    surv_params = surv.predict_parameters(type="response")
    for key in ("mu", "sigma"):
        np.testing.assert_allclose(np.asarray(plain_params[key]),
                                   np.asarray(surv_params[key]), rtol=1e-6)


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("bad", [0.0, -1.5])
def test_non_positive_response_is_refused_with_the_remedy(name, bad):
    data = _positive_data()
    data["y"] = data["y"].copy()
    data["y"][7] = bad
    with pytest.raises(ValueError, match="positive response.*Surv\\(time, status\\)"):
        mb.gamboostLSS(SPEC, data, families=getattr(mb, name)(),
                       control=CONTROL)


def test_the_realplus_set_includes_the_survival_families():
    for name in FAMILIES:
        assert name in FAMILY_SETS["realplus"]


def test_scan_realplus_picks_the_log_normal_on_log_normal_data():
    # Log-normal data with a linear location and a constant scale. BCCGLSS has
    # a third parameter that nests the log-normal, so it is the close rival;
    # at this sample size the log-normal has the lower out-of-bag risk.
    rng = np.random.default_rng(31)
    n = 1000
    x = rng.normal(size=n)
    y = np.exp(0.3 + 0.6 * x + 0.5 * rng.standard_normal(n))
    ranked = scan_lss_families(
        {"mu": "y ~ x", "sigma": "y ~ 1"},
        data={"x": x, "y": y}, support="realplus", metric="cvrisk",
        control=mb.boost_control(100, 0.1),
    )
    names = {row["family"] for row in ranked}
    assert set(FAMILY_SETS["realplus"]) <= names
    assert all(row["status"] == "ok" for row in ranked
               if row["family"] in FAMILIES)
    assert ranked[0]["family"] == "LogNormalLSS"
