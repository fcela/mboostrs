"""Every family's density, and the three checks that make it trustworthy.

A family without a `pdf` cannot report a deviance contribution and cannot be
drawn against a histogram. Sixteen of twenty-four were in that state.

The checks are independent on purpose, because each catches a different
mistake:

* against `gamlss.dist::d<FAM>`, which catches a wrong formula;
* integrating or summing to one, which catches a wrong normalizing constant
  that a pointwise comparison at one parameterization can miss;
* against the family's own `risk`, implemented separately in Rust. `-sum log
  pdf` and `risk` are two implementations of one quantity in two languages, so
  their agreement is evidence rather than a restatement.

Two families are checked against something other than the obvious gamlss call,
and the mapping is the point: `NBinomialLSS.sigma` is the size, so it matches
`dNBI(y, mu, 1/sigma)`, while `NegativeBinomialLSS.sigma` is gamlss'
dispersion and matches `dNBI(y, mu, sigma)`. They are separate families for
that reason, not aliases, and an earlier draft of the density had both
computing the same thing.
"""

import numpy as np
import pytest
from mboostrs._rust import lss_family_diagnostics

import mboostrs as mb

Y_REAL = np.array([-2.3, -0.5, 0.0, 0.7, 2.9])
Y_POS = np.array([0.15, 0.8, 1.0, 2.6, 7.4])
Y_COUNT = np.array([0.0, 1.0, 2.0, 5.0, 11.0])


def _pars(values, size):
    return {k: np.full(size, v) for k, v in values.items()}


GAMLSS_DENSITY = {
    "GumbelLSS": (Y_REAL, dict(mu=0.4, sigma=1.7),
     [0.0215474921706161, 0.1828389228222849, 0.2100068795887662, 0.2132431379151763, 0.1074202291781847]),
    "JSULSS": (Y_REAL, dict(mu=0.4, sigma=1.2, nu=-0.3, tau=1.8),
     [0.02570388734206301, 0.2246116296119051, 0.3343560964063854, 0.3851692582208724, 0.02735808011813115]),
    "LogisticLSS": (Y_REAL, dict(mu=0.4, sigma=1.7),
     [0.08285717550915123, 0.1372174334503384, 0.145042042272067, 0.1459198182818043, 0.0893759383272468]),
    "BCCGLSS": (Y_POS, dict(mu=1.6, sigma=0.3, nu=0.5),
     [6.136402799279305e-05, 0.174686725647563, 0.3966661800602438, 0.1218110474715774, 6.468056094861822e-14]),
    "PoissonLSS": (Y_COUNT, dict(mu=2.4),
     [0.09071795328941251, 0.2177230878945901, 0.261267705473508, 0.06019607934109625, 3.458288211163907e-05]),
    "GeometricLSS": (Y_COUNT, dict(mu=2.4),
     [0.2941176470588235, 0.2076124567474048, 0.1465499694687564, 0.0515445445231042, 0.006376416168226252]),
    "NBinomialLSS": (Y_COUNT, dict(mu=2.4, sigma=1.3),
     [0.2567226084021889, 0.2164796049229269, 0.1614820836722373, 0.05524079400186415, 0.005110726788195609]),
    "NegativeBinomialLSS": (Y_COUNT, dict(mu=2.4, sigma=1.3),
     [0.3365132287234638, 0.1960271235282313, 0.1313191410043491, 0.04731612061211003, 0.007509667864059245]),
    "ZIPoLSS": (Y_COUNT, dict(mu=2.4, sigma=0.3),
     [0.3635025673025887, 0.152406161526213, 0.1828873938314556, 0.04213725553876738, 2.420801747814729e-05]),
    "ZINBLSS": (Y_COUNT, dict(mu=2.4, sigma=1.3, nu=0.3),
     [0.5355592601064246, 0.1372189864697619, 0.0919233987030444, 0.03312128442847703, 0.005256767504841474]),
    "ZeroAdjustedPoissonLSS": (Y_COUNT, dict(mu=2.4, sigma=0.3),
     [0.3, 0.1676115371215746, 0.2011338445458895, 0.04634123778337297, 2.662322165682486e-05]),
    "LogNormalLSS": (Y_POS, dict(mu=0.4, sigma=0.7),
     [0.01742947023932597, 0.4793364275107562, 0.4840684796525537, 0.159986504293179, 0.005623374354614884]),
    "WeibullLSS": (Y_POS, dict(mu=0.4, sigma=0.7),
     [0.3445992435206398, 0.4862861098487323, 0.4586482284810394, 0.1331119673024499, 9.998456732063544e-05]),
    "LogLogLSS": (Y_POS, dict(mu=0.4, sigma=0.7),
     [0.3323507905993726, 0.3684771684577428, 0.3295045832043745, 0.1178184594425338, 0.01614806484486446]),
    "Pareto2LSS": (Y_POS, dict(mu=1.6, sigma=2.3),
     [1.069493711864442, 0.3771435619592182, 0.2895954591308643, 0.05949509164144178, 0.004810647276451438]),
}


@pytest.mark.parametrize("name", sorted(GAMLSS_DENSITY))
def test_density_matches_gamlss(name):
    y, pars, expected = GAMLSS_DENSITY[name]
    density = getattr(mb, name)().pdf(y, _pars(pars, len(y)))
    np.testing.assert_allclose(np.asarray(density), expected, rtol=0, atol=1e-12)


def test_the_two_negative_binomials_are_different_distributions():
    """`NBinomialLSS.sigma` is the size; `NegativeBinomialLSS.sigma` is gamlss'
    dispersion. At the same nominal sigma they must disagree."""
    pars = _pars(dict(mu=2.4, sigma=1.3), len(Y_COUNT))
    size_form = np.asarray(mb.NBinomialLSS().pdf(Y_COUNT, pars))
    dispersion_form = np.asarray(mb.NegativeBinomialLSS().pdf(Y_COUNT, pars))
    assert not np.allclose(size_form, dispersion_form)


CONTINUOUS = {
    "GumbelLSS": (dict(mu=0.4, sigma=1.7), -40.0, 80.0),
    "LogisticLSS": (dict(mu=0.4, sigma=1.7), -80.0, 80.0),
    "JSULSS": (dict(mu=0.4, sigma=1.2, nu=-0.3, tau=1.8), -80.0, 80.0),
    "BCCGLSS": (dict(mu=1.6, sigma=0.3, nu=0.5), 1e-9, 12.0),
    "InverseGaussianLSS": (dict(mu=1.6, sigma=0.7), 1e-9, 400.0),
    "GammaLSS": (dict(mu=1.6, sigma=2.0), 1e-9, 60.0),
    "LogNormalLSS": (dict(mu=0.4, sigma=0.7), 1e-9, 600.0),
    "WeibullLSS": (dict(mu=0.4, sigma=0.7), 1e-9, 300.0),
}
DISCRETE = {
    "PoissonLSS": dict(mu=2.4),
    "GeometricLSS": dict(mu=2.4),
    "NBinomialLSS": dict(mu=2.4, sigma=1.3),
    "NegativeBinomialLSS": dict(mu=2.4, sigma=1.3),
    "ZIPoLSS": dict(mu=2.4, sigma=0.3),
    "ZINBLSS": dict(mu=2.4, sigma=1.3, nu=0.3),
    "ZeroAdjustedPoissonLSS": dict(mu=2.4, sigma=0.3),
    "BinomialLSS": dict(mu=0.4),
}


@pytest.mark.parametrize("name", sorted(CONTINUOUS))
def test_continuous_density_integrates_to_one(name):
    pars, lo, hi = CONTINUOUS[name]
    grid = np.linspace(lo, hi, 200001)
    total = np.trapezoid(
        np.asarray(getattr(mb, name)().pdf(grid, _pars(pars, len(grid)))), grid
    )
    assert total == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("name", sorted(DISCRETE))
def test_discrete_mass_sums_to_one(name):
    pars = DISCRETE[name]
    support = np.array([0.0, 1.0]) if name == "BinomialLSS" else np.arange(0.0, 3001.0)
    total = np.sum(
        np.asarray(getattr(mb, name)().pdf(support, _pars(pars, len(support))))
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_geometric_mass_survives_a_large_count():
    """Written as `mu**y / (1+mu)**(y+1)` both halves overflow long before the
    ratio becomes small, so the sum came out NaN."""
    counts = np.arange(0.0, 2001.0)
    mass = np.asarray(mb.GeometricLSS().pdf(counts, _pars(dict(mu=2.4), len(counts))))
    assert np.all(np.isfinite(mass))
    assert mass.sum() == pytest.approx(1.0, abs=1e-12)


RISK_CASES = [
    ("GumbelLSS", "gumbellss", Y_REAL, [(0.4, "id"), (1.7, "log")]),
    ("LogisticLSS", "logisticlss", Y_REAL, [(0.4, "id"), (1.7, "log")]),
    ("JSULSS", "jsulss", Y_REAL,
     [(0.4, "id"), (1.2, "log"), (-0.3, "id"), (1.8, "log")]),
    ("Pareto2LSS", "pareto2lss", Y_POS, [(1.6, "log"), (2.3, "log")]),
    ("InverseGaussianLSS", "inversegaussianlss", Y_POS, [(1.6, "log"), (0.7, "log")]),
    ("PoissonLSS", "poissonlss", Y_COUNT, [(2.4, "log")]),
    ("GeometricLSS", "geometriclss", Y_COUNT, [(2.4, "log")]),
    ("NBinomialLSS", "nbinomiallss", Y_COUNT, [(2.4, "log"), (1.3, "log")]),
    ("NegativeBinomialLSS", "negativebinomiallss", Y_COUNT,
     [(2.4, "log"), (1.3, "log")]),
    ("ZIPoLSS", "zipolss", Y_COUNT, [(2.4, "log"), (0.3, "logit")]),
    ("ZINBLSS", "zinblss", Y_COUNT, [(2.4, "log"), (1.3, "log"), (0.3, "logit")]),
    ("ZeroAdjustedPoissonLSS", "zeroadjustedpoissonlss", Y_COUNT,
     [(2.4, "log"), (0.3, "logit")]),
]


def _eta(value, link):
    if link == "id":
        return value
    if link == "log":
        return np.log(value)
    return np.log(value / (1 - value))


@pytest.mark.parametrize("label,rust_name,y,pars", RISK_CASES,
                         ids=[case[0] for case in RISK_CASES])
def test_density_agrees_with_the_rust_risk(label, rust_name, y, pars):
    """Two implementations of one quantity, in two languages."""
    family = getattr(mb, label)()
    params = {
        name: np.full(len(y), value)
        for (value, _), name in zip(pars, family.param_names, strict=True)
    }
    from_density = -float(np.sum(np.log(np.asarray(family.pdf(y, params)))))
    etas = [np.full(len(y), _eta(value, link)) for value, link in pars]
    risk = lss_family_diagnostics(rust_name, len(pars), y, etas, np.ones(len(y)))[2]
    assert from_density == pytest.approx(risk, abs=1e-9)


def test_every_family_with_a_density_reports_a_deviance():
    """`-2 log pdf` was copied into each family that defined it and simply
    absent from the rest; it is one implementation now."""
    families = sorted(
        name for name in dir(mb)
        if name.endswith("LSS") and isinstance(getattr(mb, name), type)
    )
    # PIGLSS: Bessel-kernel density still unimplemented. GaussianKnownVarLSS:
    # its density is defined on the total scale s2 + excess, so pdf takes a
    # required third argument and the two-argument deviance mixin does not
    # apply — it exposes total_sd/pdf/cdf/quantile with explicit s2 instead.
    with_density = [n for n in families
                    if hasattr(getattr(mb, n), "pdf") and n != "GaussianKnownVarLSS"]
    assert len(with_density) == len(families) - 2
    assert all(hasattr(getattr(mb, n), "deviance_increment") for n in with_density)


def test_deviance_sums_to_twice_the_risk():
    rng = np.random.default_rng(5)
    n = 250
    x = rng.uniform(0.1, 3, n)
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"},
        data={"x": x, "y": 0.4 + 1.2 * rng.gumbel(size=n)},
        families=mb.GumbelLSS(), control=mb.boost_control(40, 0.1),
    )
    deviance = np.asarray(fit.deviance_increment())
    assert float(np.sum(deviance)) == pytest.approx(2 * float(fit.risk_[-1]), rel=1e-9)


def test_centered_bbs_predicts_on_new_data():
    """`gamboostLSS` used to refuse `predict(newdata=)` and truncated-mstop
    prediction for `center=True` bbs terms, telling the user to refit with
    `center=False`. The restriction lived on a code path that no longer runs:
    every LSS parameter is now fitted through the learner store, whose replay
    recipe handles the spectral centering. Both settings work, and this pins
    that so the branch cannot come back."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0, 1, n)
    data = {"x": x, "y": np.sin(3 * x) + rng.normal(0, np.exp(-1 + 0.5 * x), n)}
    for centered in (True, False):
        fit = mb.gamboostLSS(
            {"mu": f"y ~ bbs(x, df=4, center={centered})", "sigma": "y ~ bols(x)"},
            data=data, families=mb.GaussianLSS(), control=mb.boost_control(40, 0.1),
        )
        newdata = {"x": rng.uniform(0, 1, 20)}
        assert np.asarray(fit.predict(newdata=newdata, parameter="mu")).shape == (20,)
        assert np.asarray(fit.predict(parameter="mu", mstop=15)).shape == (n,)


# ---------------------------------------------------------------------------
# BCCG's truncation normalizer
# ---------------------------------------------------------------------------
#
# BCCG is defined on y > 0, so its Box-Cox transform maps to a normal truncated
# below and the density carries a 1/Phi(1/(sigma|nu|)) normalizer. That factor
# depends on sigma and nu, so it belongs in their gradients as well as in the
# risk. It was omitted from both, documented as negligible "for practical
# parameter values" — true for small sigma*|nu| and false further out.

BCCG_NLL = {          # -sum(dBCCG(y, 1.6, sigma, nu, log = TRUE))
    (0.1, 0.5): 396.745696535, (0.1, 1.0): 732.589988472,
    (0.3, 0.5): 44.842707928, (0.3, 1.0): 82.829168042,
    (0.6, 0.5): 14.800418377, (0.6, 1.0): 24.622349196,
    (1.0, 0.5): 10.093706710, (1.0, 1.0): 13.452523948,
    (1.5, 0.5): 9.524158601, (1.5, 1.0): 10.793240081,
}


@pytest.mark.parametrize("sigma,nu", sorted(BCCG_NLL))
def test_bccg_risk_matches_the_density_across_the_parameter_range(sigma, nu):
    """Previously off by up to 1.46 over five points at sigma=1.5."""
    etas = [np.full(len(Y_POS), np.log(1.6)), np.full(len(Y_POS), np.log(sigma)),
            np.full(len(Y_POS), nu)]
    risk = lss_family_diagnostics("bccglss", 3, Y_POS, etas, np.ones(len(Y_POS)))[2]
    assert risk == pytest.approx(BCCG_NLL[(sigma, nu)], abs=1e-8)


@pytest.mark.parametrize("sigma,nu", sorted(BCCG_NLL))
def test_bccg_gradients_are_the_derivative_of_its_own_risk(sigma, nu):
    """The gradients carry the same sigma- and nu-dependent term the risk does,
    so they must remain its exact derivative at every parameter value."""
    weights = np.ones(len(Y_POS))

    def risk_at(etas):
        return lss_family_diagnostics("bccglss", 3, Y_POS, etas, weights)[2]

    base = [np.full(len(Y_POS), np.log(1.6)), np.full(len(Y_POS), np.log(sigma)),
            np.full(len(Y_POS), nu)]
    grads = lss_family_diagnostics("bccglss", 3, Y_POS, base, weights)[1]
    for k in range(3):
        step = 1e-6
        up = [e.copy() for e in base]
        down = [e.copy() for e in base]
        up[k] = up[k] + step
        down[k] = down[k] - step
        finite = -(risk_at(up) - risk_at(down)) / (2 * step)
        if abs(finite) > 1e-8:
            assert float(np.sum(grads[k])) == pytest.approx(finite, rel=1e-4)


def test_bccg_nu_gradient_at_zero_matches_the_density():
    """`gamlss.dist::BCCG$dldv` returns NaN at nu = 0; the limit is finite.

    An earlier version of this test asserted that mboostrs deliberately
    disagreed with gamlss' analytic derivatives, on the strength of a
    comparison that passed `nu` as a scalar. That triggers a length-1 `ifelse`
    inside gamlss, collapsing its standardized residual to the first
    observation. Called correctly, gamlss agrees with its own density and with
    this implementation; the claim was withdrawn. See
    `.remember/note-to-gamlss-authors.md`.

    What remains true is that gamlss does not define the nu = 0 case, and this
    implementation does.
    """
    y = np.array([0.15, 0.8, 1.0, 2.6, 7.4])
    mu, sigma = 1.6, 0.4
    etas = [np.full(len(y), np.log(mu)), np.full(len(y), np.log(sigma)),
            np.zeros(len(y))]
    weights = np.ones(len(y))
    analytic = float(np.sum(
        lss_family_diagnostics("bccglss", 3, y, etas, weights)[1][2]))

    step = 1e-6
    up = [e.copy() for e in etas]
    down = [e.copy() for e in etas]
    up[2] += step
    down[2] -= step
    finite = -(lss_family_diagnostics("bccglss", 3, y, up, weights)[2]
               - lss_family_diagnostics("bccglss", 3, y, down, weights)[2]) / (2 * step)

    assert np.isfinite(analytic)
    assert analytic == pytest.approx(finite, rel=1e-3)


# ---------------------------------------------------------------------------
# Two BCCG defects found while writing a growth-centile case study
# ---------------------------------------------------------------------------


def test_bccg_mu_is_log_linked_on_the_response_scale():
    """`BCCGLSS.response` treated mu as identity-linked while the engine fits
    it on the log scale, so `params(type="response")` reported log(mu) as mu.

    Latent until BCCGLSS gained a CDF: nothing had exercised the response
    scale, and once something did, every centile came out at 0% or 100%.
    """
    eta = np.array([-0.5, 0.0, 0.7, 1.5])
    family = mb.BCCGLSS()
    np.testing.assert_allclose(family.response(0, eta), np.exp(eta))   # mu
    np.testing.assert_allclose(family.response(1, eta), np.exp(eta))   # sigma
    np.testing.assert_allclose(family.response(2, eta), eta)           # nu


def test_bccg_nu_gradient_is_nonzero_at_nu_zero():
    """nu = 0 is a removable singularity in the gradient, not a fixed point.

    The code special-cased it to 0. Since nu's offset is also 0, that made it a
    fixed point: nu could never move, and every BCCG fit was silently its
    nu = 0 special case, the log-normal.

    Expanding z = (exp(nu*l) - 1)/(nu*sigma) about nu = 0 gives z -> l/sigma
    and dz/dnu -> l^2/(2*sigma), so dl/dnu -> l - l^3/(2*sigma^2).
    """
    y = np.array([0.6, 0.9, 1.0, 1.7, 3.2])
    weights = np.ones(len(y))

    for sigma in (0.2, 0.4, 0.8):
        etas = [np.full(len(y), np.log(1.2)), np.full(len(y), np.log(sigma)),
                np.zeros(len(y))]
        analytic = float(np.sum(
            lss_family_diagnostics("bccglss", 3, y, etas, weights)[1][2]))

        step = 1e-6
        up = [e.copy() for e in etas]
        down = [e.copy() for e in etas]
        up[2] += step
        down[2] -= step
        finite = -(lss_family_diagnostics("bccglss", 3, y, up, weights)[2]
                   - lss_family_diagnostics("bccglss", 3, y, down, weights)[2]) / (2 * step)

        assert analytic != 0.0
        assert analytic == pytest.approx(finite, rel=1e-3)

        # ... and the closed form the expansion gives.
        ell = np.log(y / 1.2)
        assert analytic == pytest.approx(
            float(np.sum(ell - ell**3 / (2 * sigma**2))), rel=1e-6)


def test_bccg_nu_can_move_away_from_zero_when_fitted():
    """The consequence of the above: a skewed response must pull nu off zero."""
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(0, 1, n)
    mu = np.exp(1.0 + 0.5 * x)
    z = rng.normal(size=n)
    y = mu * np.power(np.maximum(0.5 * 0.2 * z + 1.0, 1e-8), 1.0 / 0.5)

    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)", "nu": "y ~ bols(x)"},
        data={"x": x, "y": y}, families=mb.BCCGLSS(),
        control=mb.boost_control(200, 0.1),
    )
    nu = np.asarray(fit.params(type="response")["nu"])
    assert abs(float(nu.mean())) > 0.05


def test_bccg_cdf_is_uniform_at_the_true_parameters():
    """Separates the family from the fit: whatever the fitting does, the CDF
    itself must turn BCCG draws into a uniform."""
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.uniform(0, 1, n)
    mu = np.exp(1.0 + 0.5 * x)
    z = rng.normal(size=n)
    y = mu * np.power(np.maximum(0.5 * 0.2 * z + 1.0, 1e-8), 1.0 / 0.5)

    values = np.asarray(mb.BCCGLSS().cdf(
        y, {"mu": mu, "sigma": np.full(n, 0.2), "nu": np.full(n, 0.5)}))
    observed, _ = np.histogram(values, bins=10, range=(0, 1))
    chi_square = float(((observed - n / 10) ** 2 / (n / 10)).sum())
    assert chi_square < 16.9, chi_square


def test_bccg_mu_gradient_is_correct_at_nu_zero():
    """The mu gradient's nu = 0 branch subtracted a spurious 1.

    The general expression is `z/sig + nu*(z^2 - 1)`, whose limit at nu = 0 is
    simply `z/sig`; the branch returned `z/sig - 1`, wrong by exactly 1 per
    observation. Since nu's offset is 0, that is where every BCCG fit starts,
    so mu was biased from the first iteration onward.
    """
    y = np.array([0.6, 0.9, 1.0, 1.7, 3.2])
    weights = np.ones(len(y))
    for mu, sigma in ((1.2, 0.2), (2.5, 0.5), (0.8, 0.9)):
        etas = [np.full(len(y), np.log(mu)), np.full(len(y), np.log(sigma)),
                np.zeros(len(y))]
        analytic = float(np.sum(
            lss_family_diagnostics("bccglss", 3, y, etas, weights)[1][0]))
        expected = float(np.sum(np.log(y / mu) / sigma**2))
        assert analytic == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("nu", [0.0, 1e-8, 1e-4, 0.05, 0.5, 1.0, -0.4, -1.0])
@pytest.mark.parametrize("mu,sigma", [(1.2, 0.2), (2.5, 0.5), (0.8, 0.9)])
def test_bccg_gradients_match_finite_differences_everywhere(nu, mu, sigma):
    """Including near nu = 0, where the standardized residual used to lose all
    precision.

    `((y/mu)^nu - 1)/(nu*sigma)` cancels its leading term as nu shrinks and
    divides the remaining noise by nu. `expm1(nu * log(y/mu))` computes the
    same quantity without the cancellation, which is what makes the general
    branch usable for any nu that is not exactly zero.
    """
    y = np.array([0.6, 0.9, 1.0, 1.7, 3.2])
    weights = np.ones(len(y))
    base = [np.full(len(y), np.log(mu)), np.full(len(y), np.log(sigma)),
            np.full(len(y), nu)]
    grads = lss_family_diagnostics("bccglss", 3, y, base, weights)[1]

    for k in range(3):
        step = 1e-6
        up = [e.copy() for e in base]
        down = [e.copy() for e in base]
        up[k] += step
        down[k] -= step
        finite = -(lss_family_diagnostics("bccglss", 3, y, up, weights)[2]
                   - lss_family_diagnostics("bccglss", 3, y, down, weights)[2]) / (2 * step)
        if abs(finite) < 1e-8:
            continue
        assert float(np.sum(grads[k])) == pytest.approx(finite, rel=1e-3)


def test_bccg_fit_reproduces_the_distribution_it_was_drawn_from():
    """The end-to-end check the three gradient defects were caught by.

    Boosting shrinks parameters toward their offsets, so the point estimates
    are not expected to equal the truth. What must hold is that the fitted
    distribution describes the data: a uniform PIT.
    """
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.uniform(0, 1, n)
    mu = np.exp(1.0 + 0.5 * x)
    z = rng.normal(size=n)
    y = mu * np.power(np.maximum(0.5 * 0.2 * z + 1.0, 1e-8), 1.0 / 0.5)

    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)", "nu": "y ~ bols(x)"},
        data={"x": x, "y": y}, families=mb.BCCGLSS(),
        control=mb.boost_control(600, 0.1),
    )
    values = np.asarray(fit.pit(random_state=0))
    observed, _ = np.histogram(values, bins=10, range=(0, 1))
    chi_square = float(((observed - n / 10) ** 2 / (n / 10)).sum())
    assert chi_square < 16.9, chi_square

    parameters = fit.params(type="response")
    assert float(np.asarray(parameters["nu"]).mean()) > 0.2      # moved off 0
    assert float(np.asarray(parameters["mu"]).mean()) == pytest.approx(
        float(mu.mean()), rel=0.05)
