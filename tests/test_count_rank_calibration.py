"""Independent likelihood/CDF and data-boundary checks for count ranks."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import mboostrs as mb
from mboostrs import _rust


def cdf(k, mu, pi):
    return 0. if k < 0 else pi+(1-pi)*poisson.cdf(k, mu/(1-pi))


def quantile(p, mu, pi):
    if p <= 0:
        return 0.
    if p >= 1:
        return np.inf
    k = 0. if p <= cdf(0, mu, pi) else poisson.ppf((p-pi)/(1-pi), mu/(1-pi))
    below, above = cdf(k-1, mu, pi), cdf(k, mu, pi)
    return k+(p-below)/(above-below)


def tail(mu, pi, level):
    if cdf(0, mu, pi) >= max(level, .5):
        return 0.
    grid = np.linspace(0, 1-level, 17)
    widths = [quantile(a+level, mu, pi)-quantile(a, mu, pi) for a in grid]
    return grid[int(np.argmin(widths))]


@pytest.mark.parametrize('pi', [0., .3])
def test_scores_and_inverse_bands_match_scipy(pi):
    means = np.array([0., .02, .2, 1.5, 10., 100.])
    y = np.array([0., 0., 2., 1., 18., 108.])
    uniforms = np.array([.95, .01, .4, .8, .2, .6])
    expected = []
    for value, mu, u in zip(y, means, uniforms, strict=True):
        rank = (1-u)*cdf(value-1, mu, pi)+u*cdf(value, mu, pi)
        a = tail(mu, pi, .9)
        expected.append(max(a-rank, rank-a-.9))
    scores = _rust.conformal_count_rank_scores(y, means, pi, uniforms, .9)
    np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=0)
    for q in (-.02, 0., .03):
        lo, hi = _rust.conformal_count_rank_band(means, pi, q, uniforms, .9)
        expected_lo, expected_hi = [], []
        for mu, u in zip(means, uniforms, strict=True):
            a = tail(mu, pi, .9)
            lower = max(0., quantile(a-q, mu, pi)-u)
            upper = quantile(a+.9+q, mu, pi)-u
            expected_lo.append(min(lower, upper))
            expected_hi.append(upper)
        np.testing.assert_allclose(lo, expected_lo, atol=1e-10, rtol=0)
        np.testing.assert_allclose(hi, expected_hi, atol=1e-10, rtol=0)


def test_reference_likelihood_and_weighted_diagnostic_match_independent_profile():
    rng = np.random.default_rng(66017)
    mu = np.exp(rng.uniform(0, 2, 600))
    y = ((rng.uniform(size=len(mu)) > .3)*rng.poisson(mu/.7)).astype(float)
    weights = rng.uniform(.5, 2., len(mu))

    def objective(pi):
        logp = np.log1p(-pi)+poisson.logpmf(y, mu/(1-pi))
        zeros = y == 0
        logp[zeros] = np.logaddexp(np.log(pi) if pi else -np.inf,
                                  np.log1p(-pi)-mu[zeros]/(1-pi))
        return -np.average(logp, weights=weights)

    optimum = minimize_scalar(objective, bounds=(0., .99), method='bounded', options={'xatol': 1e-12})
    pi, ratio, resolved = _rust.conformal_count_reference(y, mu, weights, True)
    assert pi == pytest.approx(optimum.x, abs=2e-7)
    variance = mu+pi/(1-pi)*mu**2
    assert ratio == pytest.approx(np.sum(weights*(y-mu)**2)/np.sum(weights*variance), abs=1e-12)
    assert resolved == ('poisson_normalized' if ratio > 2 else 'count_rank')
    plain, _, _ = _rust.conformal_count_reference(y, mu, weights, False)
    assert plain == 0


def test_inactive_rows_do_not_change_the_reference_fit():
    y = np.array([0.,0.,0.,2.,4.,5.])
    mu = np.full(len(y), 2.)
    w = np.ones(len(y))
    expected = _rust.conformal_count_reference(y, mu, w, True)
    observed = _rust.conformal_count_reference(np.r_[y, 8e15], np.r_[mu, 8e15], np.r_[w, 0.], True)
    assert observed == expected


def fixture():
    rng = np.random.default_rng(66018)
    x = rng.uniform(-1, 1, 300)
    return {'x': x, 'y': rng.poisson(np.exp(.3+.6*x)).astype(float)}


@pytest.mark.parametrize('frontend', [mb.glmboost, mb.gamboost])
def test_internal_and_external_calibration_agree_at_fixed_prefix(frontend):
    data = fixture()
    query = {'x': np.array([-.5, 0., .5])}
    formula = 'y ~ x' if frontend is mb.glmboost else 'y ~ bols(x)'
    perm = np.random.default_rng(4).permutation(300)
    proper = {k: v[perm[150:]] for k, v in data.items()}
    held = {k: v[perm[:150]] for k, v in data.items()}
    full = frontend(formula, data, family=mb.Poisson(), control=mb.boost_control(60, .02))
    fit = frontend(formula, proper, family=mb.Poisson(), control=mb.boost_control(60, .02))
    internal = mb.conformal_interval(full, query, seed=4, mstop=30)
    external = mb.conformal_interval(fit, query, calibration=held, seed=4, mstop=30)
    assert internal.score == external.score == 'count_rank'
    assert internal.requested_score == 'auto' and internal.support == 'nonnegative_integers'
    for name in ('lower', 'upper', 'predicted', 'count_size'):
        np.testing.assert_allclose(getattr(internal, name), getattr(external, name), atol=1e-10, rtol=0)
    assert internal.zero_probability == pytest.approx(external.zero_probability)
    assert internal.randomization_seed == external.randomization_seed == 4


def test_calibration_responses_cannot_change_reference_or_randomization():
    data = fixture()
    proper = {k: v[:150] for k, v in data.items()}
    held = {k: v[150:] for k, v in data.items()}
    fit = mb.gamboost('y ~ bols(x)', proper, family=mb.Poisson(), control=mb.boost_control(60, .02))
    a = mb.conformal_interval(fit, {'x': np.array([0.])}, calibration=held, seed=7)
    changed = {k: v.copy() for k, v in held.items()}
    changed['y'] += 20
    b = mb.conformal_interval(fit, {'x': np.array([0.])}, calibration=changed, seed=7)
    assert a.zero_probability == b.zero_probability
    assert a.reference_dispersion == b.reference_dispersion
    assert a.randomization_seed == b.randomization_seed
    np.testing.assert_array_equal(a.predicted, b.predicted)
    assert a.quantile != b.quantile


def test_auto_eligibility_does_not_read_internal_calibration_outcomes():
    data = fixture()
    held_row = np.random.default_rng(4).permutation(300)[0]
    data['y'][held_row] += .5
    model = mb.gamboost('y ~ bols(x)', data, family=mb.Poisson(), control=mb.boost_control(30, .02))
    with pytest.raises(ValueError, match='integer counts'):
        mb.conformal_interval(model, seed=4)


def test_randomization_chunks_and_empty_count_sets_are_explicit():
    seed, whole = _rust.conformal_count_uniforms(50, 42, 1)
    _, part = _rust.conformal_count_uniforms(15, seed, 1, 7)
    np.testing.assert_array_equal(whole[7:22], part)
    lo, hi = _rust.conformal_count_rank_band(np.array([0.,0.]), 0., 0., np.array([.3,.95]), .9)
    result = mb.ConformalInterval(lo, hi, np.zeros(2), .9, 'split', 500,
                                  score='count_rank', support='nonnegative_integers')
    np.testing.assert_array_equal(result.count_size, [1.,0.])
    np.testing.assert_array_equal(result.contains([0.,0.]), [True,False])
    assert result.to_frame(output='polars')['empty_count_set'].to_list() == [False,True]


def test_degenerate_counts_are_calibrated_over_auxiliary_randomness():
    coverage = []
    for seed in range(16):
        _, u = _rust.conformal_count_uniforms(500, seed, 0)
        scores = _rust.conformal_count_rank_scores(np.zeros(500), np.zeros(500), 0., np.array(u), .9)
        q = _rust.conformal_quantile(np.array(scores), .9)
        _, query = _rust.conformal_count_uniforms(4000, seed, 1)
        lo, hi = _rust.conformal_count_rank_band(np.zeros(4000), 0., q, np.array(query), .9)
        coverage.append(np.mean((np.array(lo) <= 0) & (np.array(hi) >= 0)))
    assert .88 < np.mean(coverage) < .92


@pytest.mark.parametrize('seed', [True, -1, 1.5, 2**53])
def test_invalid_auxiliary_seeds_refuse_cleanly(seed):
    data = fixture()
    model = mb.gamboost('y ~ bols(x)', data, family=mb.Poisson(), control=mb.boost_control(30, .02))
    with pytest.raises(ValueError, match='randomization seed'):
        mb.conformal_interval(model, {'x': np.array([0.])}, calibration=data,
                              score='count_rank', seed=seed)


def test_count_dictionary_includes_computed_cardinality_only_with_arrays():
    from mboostrs import ConformalInterval

    band = ConformalInterval(np.array([0., -1., 2.2]), np.array([np.inf, -1., 4.8]),
        np.array([1., 0., 3.]), .9, "split", 20, support="nonnegative_integers")
    assert "count_size" not in band.to_dict()
    record = band.to_dict(include_arrays=True)
    np.testing.assert_array_equal(record["count_size"], [np.inf, 0., 2.])
