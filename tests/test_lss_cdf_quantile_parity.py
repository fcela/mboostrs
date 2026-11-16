"""CDF and quantile for the families that previously had neither.

Without a `cdf` and a `quantile` an LSS family fits perfectly well and then
refuses every distributional diagnostic: `pit()`, `quantile_residuals()`,
`centiles()` and `prediction_interval()` all check `hasattr(family, ...)` and
raise. Thirteen of twenty-four families were in that state, so the catalogue
was much narrower in practice than in the family list.

Reference values come from `gamlss.dist`, computed in R and recorded below.
Two families are deliberately checked against something else:

* `Pareto2LSS` is not `gamlss.dist::PARETO2` — gamlss parameterizes the shape
  as `1/sigma`. It is checked against its own closed form instead.
* `InverseGaussianLSS`'s quantile is checked by round trip rather than against
  `qIG`, because `qIG` is the less accurate of the two: at p = 0.025 it returns
  a point whose CDF is 0.024996554, while inverting our own CDF gives one whose
  CDF is 0.025000000.
"""

import numpy as np
import pytest

import mboostrs as mb

# --- recorded from gamlss.dist -------------------------------------------
Y_REAL = [-2.3, -0.5, 0.0, 0.7, 2.9]
Y_POS = [0.15, 0.8, 1.0, 2.6, 7.4]
Y_COUNT = [0.0, 1.0, 2.0, 5.0, 11.0]
PROBS = [0.025, 0.1, 0.5, 0.9, 0.975]

GAMLSS = {
    "LogisticLSS": dict(
        pars=dict(mu=0.4, sigma=1.7), y=Y_REAL,
        cdf=[0.1696323235629083, 0.3706540950225918, 0.4414463652922717, 0.5440035103251184, 0.8131467784341403],
        quantile=[-5.828054798420398, -3.335281781471573, 0.4, 4.135281781471573, 6.628054798420397]),
    "GumbelLSS": dict(
        pars=dict(mu=0.4, sigma=1.7), y=Y_REAL,
        cdf=[0.007483139282607375, 0.1830614792182606, 0.2821600388293647, 0.4324781680320068, 0.794700253797806],
        quantile=[-1.819048659637502, -1.017855156921525, 1.023071964988829, 4.225624456431158, 6.649620338522099]),
    "InverseGaussianLSS": dict(
        pars=dict(mu=1.6, sigma=0.7), y=Y_POS,
        cdf=[0.0007657221575029568, 0.3185719743025713, 0.4259703045707862, 0.8385569799166335, 0.991632361941463],
        quantile=None),
    "BCCGLSS": dict(
        pars=dict(mu=1.6, sigma=0.3, nu=0.5), y=Y_POS,
        cdf=[1.869140295782944e-06, 0.02543223324268415, 0.08132655134516535, 0.9665018752385259, 0.9999999999999915],
        quantile=[0.7975098050397315, 1.043980727510001, 1.600000000007872, 2.274270230411068, 2.67907523012769]),
    "JSULSS": dict(
        pars=dict(mu=0.4, sigma=1.2, nu=-0.3, tau=1.8), y=Y_REAL,
        cdf=[0.02110076924194265, 0.1935480443592921, 0.3336521841837237, 0.5975745502571839, 0.982766082334633],
        quantile=[-2.161313378998655, -1.047400885447284, 0.4505022942211339, 1.788397845706407, 2.666596626895487]),
    "ZeroAdjustedGammaLSS": dict(
        pars=dict(mu=1.6, sigma=0.8, nu=0.3), y=[0.0] + Y_POS,
        cdf=[0.3, 0.322910493467778, 0.5170739626796175, 0.5763476388356076, 0.8746248500533054, 0.9981318250392641],
        quantile=[0.0, 0.0, 0.7446192379759365, 2.874774628882777, 4.498490080393839]),
    "ZeroAdjustedPoissonLSS": dict(
        pars=dict(mu=2.4, sigma=0.3), y=Y_COUNT,
        cdf=[0.3, 0.4676115371215747, 0.6687453816674642, 0.9725379404695755, 0.9999934921631868],
        quantile=[0.0, 0.0, 2.0, 4.0, 6.0]),
}


def _broadcast(pars, size):
    return {k: np.full(size, v) for k, v in pars.items()}


@pytest.mark.parametrize("name", sorted(GAMLSS))
def test_cdf_matches_gamlss(name):
    case = GAMLSS[name]
    family = getattr(mb, name)()
    y = np.asarray(case["y"])
    got = np.asarray(family.cdf(y, _broadcast(case["pars"], len(y))))
    np.testing.assert_allclose(got, case["cdf"], rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", sorted(n for n in GAMLSS if GAMLSS[n]["quantile"]))
def test_quantile_matches_gamlss(name):
    case = GAMLSS[name]
    family = getattr(mb, name)()
    probs = np.asarray(PROBS)
    got = np.asarray(family.quantile(probs, _broadcast(case["pars"], len(probs))))
    np.testing.assert_allclose(got, case["quantile"], rtol=0, atol=1e-8)


def test_inverse_gaussian_quantile_is_more_accurate_than_r():
    """`qIG` inverts its CDF approximately; inverting ours lands on the
    requested probability to machine precision."""
    family = mb.InverseGaussianLSS()
    probs = np.asarray(PROBS)
    pars = _broadcast(dict(mu=1.6, sigma=0.7), len(probs))
    attained = np.asarray(family.cdf(np.asarray(family.quantile(probs, pars)), pars))
    np.testing.assert_allclose(attained, probs, rtol=0, atol=1e-12)

    r_quantiles = np.array([0.2870713159294049, 0.4436338088069559, 1.161085984350916, 3.284928196346636, 5.483885559012248])
    r_attained = np.asarray(family.cdf(r_quantiles, pars))
    assert np.max(np.abs(r_attained - probs)) > 1e-6  # R's own error


def test_pareto2_is_not_gamlss_pareto2():
    """gamlss uses `1/sigma` as the shape. The class documents this; the test
    pins both halves, so a future 'fix' toward gamlss cannot pass silently."""
    family = mb.Pareto2LSS()
    y = np.asarray(Y_POS)
    pars = _broadcast(dict(mu=1.6, sigma=2.3), len(y))
    got = np.asarray(family.cdf(y, pars))
    closed_form = 1.0 - (1.6 / (y + 1.6)) ** 2.3
    np.testing.assert_allclose(got, closed_form, rtol=0, atol=1e-14)

    # gamlss's pPARETO2(y, 1.6, 2.3), which is a different distribution.
    gamlss_at_same_nominal_sigma = [0.03821255909025989, 0.161624491002343, 0.1902990966962552, 0.3426905649636619, 0.5280893582305237]
    assert np.max(np.abs(got - gamlss_at_same_nominal_sigma)) > 0.1


@pytest.mark.parametrize(
    "name,pars,lo,hi",
    [
        ("LogisticLSS", dict(mu=0.4, sigma=1.7), -30.0, 30.0),
        ("GumbelLSS", dict(mu=0.4, sigma=1.7), -10.0, 40.0),
        ("InverseGaussianLSS", dict(mu=1.6, sigma=0.7), 1e-6, 60.0),
        ("Pareto2LSS", dict(mu=1.6, sigma=2.3), 1e-6, 200.0),
        ("BCCGLSS", dict(mu=1.6, sigma=0.3, nu=0.5), 1e-6, 10.0),
        ("JSULSS", dict(mu=0.4, sigma=1.2, nu=-0.3, tau=1.8), -20.0, 20.0),
        ("ZeroAdjustedGammaLSS", dict(mu=1.6, sigma=0.8, nu=0.3), 0.0, 40.0),
    ],
)
def test_cdf_is_a_distribution_function(name, pars, lo, hi):
    """Non-decreasing, in [0, 1], and approaching the right limits."""
    family = getattr(mb, name)()
    grid = np.linspace(lo, hi, 600)
    values = np.asarray(family.cdf(grid, _broadcast(pars, len(grid))))
    assert np.all(np.diff(values) >= -1e-12)
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert values[-1] > 0.97


@pytest.mark.parametrize(
    "name,pars",
    [
        ("LogisticLSS", dict(mu=0.4, sigma=1.7)),
        ("GumbelLSS", dict(mu=0.4, sigma=1.7)),
        ("InverseGaussianLSS", dict(mu=1.6, sigma=0.7)),
        ("Pareto2LSS", dict(mu=1.6, sigma=2.3)),
        ("BCCGLSS", dict(mu=1.6, sigma=0.3, nu=0.5)),
        ("JSULSS", dict(mu=0.4, sigma=1.2, nu=-0.3, tau=1.8)),
    ],
)
def test_quantile_inverts_the_cdf(name, pars):
    family = getattr(mb, name)()
    probs = np.array([0.02, 0.15, 0.5, 0.8, 0.97])
    P = _broadcast(pars, len(probs))
    attained = np.asarray(family.cdf(np.asarray(family.quantile(probs, P)), P))
    np.testing.assert_allclose(attained, probs, rtol=0, atol=1e-9)


def test_atom_families_put_the_quantile_at_zero_below_the_atom():
    """`ZeroAdjustedGammaLSS` has probability `nu` at zero, so every `p <= nu`
    has quantile zero and the round trip returns `nu`, not `p`. That is the
    definition of a quantile for a distribution with an atom, not an error."""
    family = mb.ZeroAdjustedGammaLSS()
    pars = dict(mu=1.6, sigma=0.8, nu=0.3)
    below = np.array([0.02, 0.15, 0.3 - 1e-9])
    np.testing.assert_array_equal(
        np.asarray(family.quantile(below, _broadcast(pars, len(below)))),
        np.zeros(3),
    )
    above = np.array([0.4, 0.6, 0.8, 0.97])
    P = _broadcast(pars, len(above))
    attained = np.asarray(family.cdf(np.asarray(family.quantile(above, P)), P))
    np.testing.assert_allclose(attained, above, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "name,pars,draw",
    [
        ("InverseGaussianLSS", dict(mu=2.0, sigma=0.6), lambda rng, n: rng.wald(mean=2.0, scale=1.0 / 0.6**2, size=n)),
        ("LogisticLSS", dict(mu=0.4, sigma=1.7), lambda rng, n: 0.4 + 1.7 * rng.logistic(size=n)),
        ("GumbelLSS", dict(mu=0.4, sigma=1.7), lambda rng, n: 0.4 + 1.7 * rng.gumbel(size=n)),
        ("Pareto2LSS", dict(mu=1.6, sigma=2.3), lambda rng, n: 1.6 * rng.pareto(2.3, size=n)),
    ],
)
def test_the_cdf_makes_a_uniform_pit(name, pars, draw):
    """The end the diagnostics actually depend on: applied to data the family
    generated, the CDF must come out uniform, or every PIT and worm plot built
    on it is misleading."""
    rng = np.random.default_rng(11)
    n = 4000
    family = getattr(mb, name)()
    u = np.asarray(family.cdf(draw(rng, n), _broadcast(pars, n)))
    ks = np.max(np.abs(np.sort(u) - (np.arange(1, n + 1) - 0.5) / n))
    assert ks < 1.36 / np.sqrt(n), f"KS {ks:.4f}"


def test_the_diagnostics_that_were_refusing_now_run():
    rng = np.random.default_rng(5)
    n = 300
    x = rng.uniform(0.1, 3, n)
    data = {"x": x, "y": rng.gamma(3.0, np.exp(0.3 * x) / 3.0)}
    fit = mb.gamboostLSS(
        {"mu": "y ~ bbs(x, df=4)", "sigma": "y ~ bols(x)"},
        data=data,
        families=mb.InverseGaussianLSS(),
        control=mb.boost_control(40, 0.1),
    )
    assert np.all(np.isfinite(np.asarray(fit.pit())))
    assert np.all(np.isfinite(np.asarray(fit.quantile_residuals())))
    assert np.asarray(fit.centiles(probs=[0.1, 0.9])).shape[0] == n
    assert np.asarray(fit.prediction_interval(level=0.9)).shape[0] == n


def test_bccg_risk_is_the_exact_log_likelihood():
    """`BCCGLSS.risk` omitted `log(2*pi)/2` per observation.

    Constant in the parameters, so no gradient and no selected path changed —
    but it made `risk()`, `logLik()` and AIC differ from `gamlss.dist::dBCCG`
    by `n * 0.9189385`, and made BCCG's risk incomparable with every other
    family's, which is what a family-ranking scan compares. Same defect class
    as the `log(y!)` that was missing from `NegativeBinomialLSS`.
    """
    from mboostrs._rust import lss_family_diagnostics

    y = np.array(Y_POS)
    mu, sigma, nu = 1.6, 0.3, 0.5
    etas = [np.full(len(y), np.log(mu)), np.full(len(y), np.log(sigma)),
            np.full(len(y), nu)]
    risk = lss_family_diagnostics("bccglss", 3, y, etas, np.ones(len(y)))[2]
    # -sum(dBCCG(y, 1.6, 0.3, 0.5, log = TRUE))
    assert risk == pytest.approx(44.842707928166, abs=1e-9)


def test_every_family_risk_is_the_exact_negative_log_likelihood():
    """Cross-family comparison is only meaningful if no family drops a term.

    Each reference is `-sum(d<FAM>(y, ..., log = TRUE))` from `gamlss.dist`, at
    the parameter values gamlss itself uses — note that three families are
    parameterized differently here and the mapping is part of the check:
    `GammaLSS.sigma` is the shape where gamlss' is the coefficient of
    variation, `BetaLSS.sigma` is the precision, and `NBinomialLSS.sigma` is
    the size.
    """
    from mboostrs._rust import lss_family_diagnostics

    y_real = np.array([-2.3, -0.5, 0.0, 0.7, 2.9])
    y_pos = np.array([0.15, 0.8, 1.0, 2.6, 7.4])
    y_count = np.array([0.0, 1.0, 2.0, 5.0, 11.0])
    y_prop = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_zero = np.array([0.0, 0.15, 0.8, 1.0, 2.6, 7.4])

    def eta(value, link):
        if link == "id":
            return value
        if link == "log":
            return np.log(value)
        return np.log(value / (1 - value))

    cases = [
        ("gaussianlss", 2, y_real, [(0.4, "id"), (1.7, "log")], 9.773785479254),
        ("logisticlss", 2, y_real, [(0.4, "id"), (1.7, "log")], 10.747158845344),
        ("gumbellss", 2, y_real, [(0.4, "id"), (1.7, "log")], 10.873589580306),
        ("studenttlss", 3, y_real,
         [(0.4, "id"), (1.7, "log"), (5.0, "log")], 10.015219899274),
        ("jsulss", 4, y_real,
         [(0.4, "id"), (1.2, "log"), (-0.3, "id"), (1.8, "log")], 10.802859965290),
        ("gammalss", 2, y_pos, [(1.6, "log"), (0.7, "log")], 9.676854611547),
        ("inversegaussianlss", 2, y_pos, [(1.6, "log"), (0.7, "log")], 12.081089336842),
        ("bccglss", 3, y_pos,
         [(1.6, "log"), (0.3, "log"), (0.5, "id")], 44.842707928166),
        ("betalss", 2, y_prop, [(0.4, "logit"), (0.3, "log")], 5.084355151100),
        ("poissonlss", 1, y_count, [(2.4, "log")], 18.349040764216),
        ("geometriclss", 1, y_count, [(2.4, "log")], 12.736704346659),
        ("nbinomiallss", 2, y_count, [(2.4, "log"), (1.3, "log")], 12.885846373900),
        ("negativebinomiallss", 2, y_count,
         [(2.4, "log"), (1.3, "log")], 12.691213048457),
        ("zipolss", 2, y_count, [(2.4, "log"), (0.3, "logit")], 18.387709450303),
        ("zinblss", 3, y_count,
         [(2.4, "log"), (1.3, "log"), (0.3, "logit")], 13.653238737216),
        ("zeroadjustedpoissonlss", 2, y_count,
         [(2.4, "log"), (0.3, "logit")], 18.199313537483),
        ("zeroadjustedgammalss", 3, y_zero,
         [(1.6, "log"), (0.8, "log"), (0.3, "logit")], 13.788351897477),
    ]
    # 1e-8 rather than exact: the log-gamma and incomplete-gamma routines here
    # are the package's own (SciPy is not a runtime dependency), so they differ
    # from R's in the last few digits. A dropped term shows up at 1e-2 or
    # larger, which is what this test is for.
    for name, k, y, pars, expected in cases:
        etas = [np.full(len(y), eta(v, link)) for v, link in pars]
        risk = lss_family_diagnostics(name, k, y, etas, np.ones(len(y)))[2]
        assert risk == pytest.approx(expected, abs=1e-8), name


# ---------------------------------------------------------------------------
# The three accelerated-failure-time families
# ---------------------------------------------------------------------------
#
# All three share ``e = (log t - mu) / sigma`` and differ only in the standard
# distribution of ``e``. The CDF is of the *event time*, which is what
# ``centiles()`` and ``prediction_interval()` need; a right-censored
# observation only tells you its event time exceeds the recorded one, so the
# PIT is not uniform and those methods still refuse.

SURVIVAL = {
    # pLOGNO(t, 0.4, 0.7)
    "LogNormalLSS": [0.00051609962250, 0.18667789051330, 0.28385458309868,
                     0.78628227521608, 0.98892624687194],
    # plogis((log t - 0.4) / 0.7)
    "LogLogLSS": [0.03620784074430, 0.29106750931542, 0.36090725483715,
                  0.68860117659916, 0.90786397745278],
    # pweibull(t, shape = 1/0.7, scale = exp(0.4))
    "WeibullLSS": [0.03687117411142, 0.33672894870102, 0.43147962244371,
                   0.89044361415420, 0.99994743805747],
}


@pytest.mark.parametrize("name", sorted(SURVIVAL))
def test_survival_time_cdf_matches_r(name):
    family = getattr(mb, name)()
    t = np.array(Y_POS)
    pars = _broadcast(dict(mu=0.4, sigma=0.7), len(t))
    np.testing.assert_allclose(
        np.asarray(family.cdf(t, pars)), SURVIVAL[name], rtol=0, atol=1e-13
    )


@pytest.mark.parametrize("name", sorted(SURVIVAL))
def test_survival_quantile_inverts_the_cdf(name):
    family = getattr(mb, name)()
    probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    pars = _broadcast(dict(mu=0.4, sigma=0.7), len(probs))
    q = np.asarray(family.quantile(probs, pars))
    assert np.all(q > 0) and np.all(np.diff(q) > 0)
    np.testing.assert_allclose(
        np.asarray(family.cdf(q, pars)), probs, rtol=0, atol=1e-12
    )


def test_lognormal_quantile_matches_r():
    family = mb.LogNormalLSS()
    probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    pars = _broadcast(dict(mu=0.4, sigma=0.7), len(probs))
    # qLOGNO(probs, 0.4, 0.7)
    np.testing.assert_allclose(
        np.asarray(family.quantile(probs, pars)),
        [0.47170686320280, 0.93039800209880, 1.49182469764127,
         2.39203107000669, 4.71805924845244],
        rtol=0, atol=1e-12,
    )


def test_survival_models_get_median_survival_time_but_not_a_pit():
    """What the CDF buys, and what it deliberately does not."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0, 2, n)
    event = np.exp(0.5 + 0.6 * x + 0.5 * rng.normal(size=n))
    censor = rng.exponential(scale=8.0, size=n)
    data = {"x": x, "time": np.minimum(event, censor),
            "status": (event <= censor).astype(float)}
    fit = mb.gamboostLSS(
        {"mu": "Surv(time, status) ~ bbs(x, df=4)",
         "sigma": "Surv(time, status) ~ bols(x)"},
        data=data, families=mb.LogNormalLSS(),
        control=mb.boost_control(60, 0.1),
    )
    median = np.asarray(fit.centiles(probs=[0.5]))[:, -1]
    assert np.all(median > 0)
    # The simulated median rises with x; the fit should track it.
    assert np.corrcoef(median, x)[0, 1] > 0.9
    assert np.asarray(fit.prediction_interval(level=0.8)).shape[0] == n

    for method in ("pit", "quantile_residuals"):
        with pytest.raises(NotImplementedError, match="survival"):
            getattr(fit, method)()
