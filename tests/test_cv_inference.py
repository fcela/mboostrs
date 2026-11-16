"""Reading a cross-validation curve as more than its minimum.

`cvrisk` reported the mean risk at each iteration and the iteration that
minimised it. That is the least a curve can say. It did not say how firmly the
minimum was located, whether the curve had turned at all before the path ran
out, or how to compare two of them.

The minimum is chosen by looking at the same curve whose noise it is being
chosen from, so it is drawn towards whichever point the fold split happened to
favour. The one-standard-error rule takes the simplest fit the folds cannot
distinguish from the best, which for boosting is the earliest stop.

Comparing two curves is the case where the paired structure matters most: the
fold-to-fold difficulty is usually larger than the difference between the
models, and pairing cancels it. The catch is that cross-validation folds share
training rows, so the across-fold standard error is too small. The
Nadeau-Bengio correction is applied and both values are reported, because the
naive one is what most software prints.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb

N = 300


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(2)
    x, z = rng.normal(size=N), rng.normal(size=N)
    return {"x": x, "z": z, "y": 2.0 * x + 0.5 * z + rng.normal(size=N)}


@pytest.fixture(scope="module")
def folds():
    return mb.cv_weights(N, type="kfold", B=10, seed=3)


@pytest.fixture(scope="module")
def curve(data, folds):
    model = mb.glmboost("y ~ x + z", data=data,
                        control=mb.boost_control(150, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return model.cvrisk(folds=folds)


# ─── the spread across folds ───────────────────────────────────────


def test_the_standard_error_is_the_across_fold_spread(curve):
    risk = np.asarray(curve.risk_matrix)
    expected = risk.std(axis=0, ddof=1) / np.sqrt(risk.shape[0])
    np.testing.assert_allclose(curve.standard_error, expected)
    assert curve.standard_error.shape == np.asarray(curve.aggregate).shape


def test_one_fold_has_no_spread_to_report(data):
    """Not an error: a single fold is a legitimate, if uninformative, run."""
    model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(20, 0.1))
    single = np.zeros((N, 1))
    single[: N // 2, 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = model.cvrisk(folds=single)
    assert np.all(result.standard_error == 0.0)


# ─── the one-standard-error rule ───────────────────────────────────


def test_the_one_se_stop_is_earlier_than_the_minimum(curve):
    assert curve.mstop_1se <= curve.optimal_mstop
    assert curve.mstop_1se in list(curve.grid)


def test_the_one_se_stop_is_within_one_standard_error(curve):
    aggregate = np.asarray(curve.aggregate)
    grid = list(curve.grid)
    best = int(np.argmin(aggregate))
    chosen = grid.index(curve.mstop_1se)
    assert aggregate[chosen] <= aggregate[best] + curve.standard_error[best]


def test_it_is_the_earliest_such_point(curve):
    aggregate = np.asarray(curve.aggregate)
    grid = list(curve.grid)
    best = int(np.argmin(aggregate))
    threshold = aggregate[best] + curve.standard_error[best]
    earliest = int(np.flatnonzero(aggregate <= threshold).min())
    assert curve.mstop_1se == grid[earliest]


def test_the_two_rules_differ_enough_to_matter(curve):
    """If they always agreed the rule would be decoration. On this fit the
    minimum is far past the point the folds can distinguish from it."""
    assert curve.mstop_1se < curve.optimal_mstop


# ─── the boundary ──────────────────────────────────────────────────


def test_a_curve_still_falling_at_the_last_iteration_warns(data, folds):
    model = mb.glmboost("y ~ x + z", data=data, control=mb.boost_control(5, 0.1))
    with pytest.warns(RuntimeWarning, match="still falling at mstop=5"):
        result = model.cvrisk(folds=folds)
    assert result.at_boundary()


def test_the_warning_says_what_to_run_instead(data, folds):
    model = mb.glmboost("y ~ x + z", data=data, control=mb.boost_control(5, 0.1))
    with pytest.warns(RuntimeWarning, match=r"continue_to\(10\)"):
        model.cvrisk(folds=folds)


def test_a_curve_that_has_turned_says_nothing(curve):
    assert not curve.at_boundary()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        from mboostrs.cvrisk import warn_if_at_boundary

        warn_if_at_boundary(curve)


def test_extending_the_path_moves_the_optimum_off_the_boundary(data, folds):
    """The warning's claim, checked: the reported optimum was the budget, and
    a longer path finds a different one."""
    short = mb.glmboost("y ~ x + z", data=data, control=mb.boost_control(5, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cramped = short.cvrisk(folds=folds)
        longer = mb.glmboost(
            "y ~ x + z", data=data, control=mb.boost_control(150, 0.1),
        ).cvrisk(folds=folds)
    assert cramped.optimal_mstop == 5
    assert longer.optimal_mstop > 5
    assert not longer.at_boundary()


# ─── paired comparison ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def pair(data, folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        without = mb.glmboost(
            "y ~ x", data=data, control=mb.boost_control(150, 0.1),
        ).cvrisk(folds=folds)
        with_z = mb.glmboost(
            "y ~ x + z", data=data, control=mb.boost_control(150, 0.1),
        ).cvrisk(folds=folds)
    return without, with_z


def test_the_comparison_is_paired_fold_by_fold(pair):
    without, with_z = pair
    comparison = without.compare_to(with_z)
    assert comparison.n_folds == 10
    assert comparison.differences.shape == (10,)
    column_a = int(np.argmin(np.asarray(without.aggregate)))
    column_b = int(np.argmin(np.asarray(with_z.aggregate)))
    np.testing.assert_allclose(
        comparison.differences,
        np.asarray(without.risk_matrix)[:, column_a]
        - np.asarray(with_z.risk_matrix)[:, column_b],
    )


def test_it_finds_the_predictor_that_is_really_there(pair):
    """`z` has a coefficient of 0.5 in the data-generating process, so
    dropping it should cost measurably. The comparison is at a fixed grid
    point: compared at each model's own optimum the two models are tuned
    separately, and that is descriptive (see
    ``test_own_optima_that_diverge_are_descriptive_not_inferential``)."""
    without, with_z = pair
    comparison = without.compare_to(with_z, mstop=100)
    assert comparison.mean_difference > 0
    assert comparison.inferential
    assert comparison.interval is not None
    assert comparison.interval[0] > 0, "the interval should exclude zero"


def test_own_optima_that_diverge_are_descriptive_not_inferential(pair):
    """Each model tuned to its own optimum on the same folds: the comparison
    selects two minima and the Nadeau-Bengio overlap correction does not
    cover that tuning bias, so no calibrated interval is reported."""
    without, with_z = pair
    comparison = without.compare_to(with_z)
    assert int(np.argmin(np.asarray(without.aggregate))) != \
        int(np.argmin(np.asarray(with_z.aggregate)))
    assert not comparison.inferential
    assert comparison.interval is None
    assert comparison.mean_difference > 0  # the descriptive number is still there
    text = comparison.describe()
    assert "descriptive" in text
    assert "No calibrated interval" in text


def test_the_correction_makes_the_standard_error_larger(pair):
    """Cross-validation folds share training rows, so the naive across-fold
    error is too small. For 10-fold the Nadeau-Bengio factor is
    sqrt((1/10 + (1/10)/(9/10)) / (1/10)) = sqrt(2.11) = 1.45."""
    comparison = pair[0].compare_to(pair[1])
    assert comparison.se_corrected > comparison.se_naive
    assert comparison.se_corrected / comparison.se_naive == pytest.approx(
        np.sqrt(1 + 10 * (1 / 9)), rel=1e-6
    )


def test_the_correction_uses_the_measured_held_out_share(data):
    """A bootstrap fold holds out about `1/e` of the sample -- the rows the
    resample happened to miss -- not `1/B`. Assuming `1/B` would leave the
    correction far too small."""
    model = mb.glmboost("y ~ x + z", data=data,
                        control=mb.boost_control(60, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        boot = model.cvrisk(folds=mb.cv_weights(N, type="bootstrap", B=25,
                                                seed=3))
    assert boot.held_out_fraction == pytest.approx(1 / np.e, abs=0.03)
    assert boot.held_out_fraction > 5 * (1 / 25)


def test_rolling_fold_risk_scores_only_the_judged_window(data):
    """A rolling-origin fold trains on the past and is scored on the horizon
    rows after it, not on every row it happened not to train on. Rows beyond
    the horizon are never judged."""
    n = len(data["y"])
    initial, horizon, B = 120, 15, 6
    folds = mb.rolling_folds(n, B=B, initial=initial, horizon=horizon)
    model = mb.glmboost("y ~ x + z", data=data,
                        control=mb.boost_control(150, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = model.cvrisk(folds=folds)
    matrix = np.asarray(res.risk_matrix)
    mstop = model.mstop

    # Reproduce fold 0 exactly as the engine does: fit on all n rows weighted
    # by the fold's in-bag weights, then take the mean squared residual over
    # the judged window rows [initial, initial + horizon).
    w = np.asarray(folds)[:, 0]
    m = mb.glmboost("y ~ x + z", data=data, weights=w,
                    control=mb.boost_control(150, 0.1))
    judged = data["y"][initial:initial + horizon]
    fitted = np.asarray(m.predict(
        newdata={"x": data["x"][initial:initial + horizon],
                 "z": data["z"][initial:initial + horizon]},
        type="link",
    )).ravel()
    expected = np.mean((judged - fitted) ** 2)
    assert matrix[0, mstop] == pytest.approx(expected, rel=1e-9)

    # The rows beyond the horizon would change the score measurably; scoring
    # them (the pre-fix behaviour) is refused by this assertion being tight.
    tail = data["y"][initial:]
    fitted_tail = np.asarray(m.predict(
        newdata={"x": data["x"][initial:], "z": data["z"][initial:]},
        type="link",
    )).ravel()
    buggy = np.mean((tail - fitted_tail) ** 2)
    assert not np.isclose(matrix[0, mstop], buggy, rtol=1e-6)

    # The reported held-out fraction is the judged share per fold, not the
    # share of rows weight-zero during training.
    assert res.held_out_fraction == pytest.approx(horizon / n, rel=1e-9)


def test_curves_from_different_folds_are_refused(data, folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = mb.glmboost("y ~ x", data=data,
                        control=mb.boost_control(60, 0.1)).cvrisk(folds=folds)
        b = mb.glmboost(
            "y ~ x + z", data=data, control=mb.boost_control(60, 0.1),
        ).cvrisk(folds=mb.cv_weights(N, type="kfold", B=10, seed=99))
    with pytest.raises(ValueError, match="computed on different folds"):
        a.compare_to(b)


def test_the_same_folds_give_the_same_fingerprint(data, folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = mb.glmboost("y ~ x", data=data,
                        control=mb.boost_control(60, 0.1)).cvrisk(folds=folds)
        b = mb.glmboost("y ~ z", data=data,
                        control=mb.boost_control(60, 0.1)).cvrisk(folds=folds)
    assert a.fold_fingerprint == b.fold_fingerprint
    assert len(a.fold_fingerprint) == 16


def test_describe_says_the_correction_is_applied(pair):
    text = pair[0].compare_to(pair[1], mstop=100).describe()
    assert "Nadeau-Bengio" in text
    assert "paired folds" in text


def _two_curves_on(data, folds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = mb.glmboost("y ~ x", data=data,
                        control=mb.boost_control(150, 0.1)).cvrisk(folds=folds)
        b = mb.glmboost("y ~ x + z", data=data,
                        control=mb.boost_control(150, 0.1)).cvrisk(folds=folds)
    return a, b


# ─── the interval refuses what it cannot claim ─────────────────────


@pytest.mark.parametrize("builder", [
    lambda n: mb.rolling_folds(n, B=5, initial=40, horizon=10),
    lambda n: mb.blocked_folds(n, B=5),
    lambda n: mb.repeated_kfold_folds(n, k=3, repeats=2, seed=1),
])
def test_an_interval_is_refused_for_ordered_and_repeated_folds(data, builder):
    """The Nadeau-Bengio interval assumes random splits of exchangeable
    rows; rolling, blocked, and repeated folds carry dependence it has no
    correction for, so the interval is refused instead of mislabeled."""
    folds = builder(N)
    a, b = _two_curves_on(data, folds)
    with pytest.raises(ValueError, match="random split"):
        a.compare_to(b)
    with pytest.raises(ValueError, match="random split"):
        b.compare_to(a)


def test_an_interval_is_refused_for_grouped_folds(data):
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 30, N)
    a, b = _two_curves_on(data, mb.grouped_folds(groups, B=5))
    with pytest.raises(ValueError, match="random split"):
        a.compare_to(b)


def test_an_interval_is_refused_for_bootstrap_draws(data):
    a, b = _two_curves_on(
        data, mb.cv_weights(N, type="bootstrap", B=25, seed=3))
    with pytest.raises(ValueError, match="random split"):
        a.compare_to(b)


def test_a_subsampling_comparison_is_allowed(data):
    """Sub-sampling is a random split of exchangeable rows, so an interval is
    available at a fixed grid point; at their own optima the two models land
    on different columns, so the default comparison is descriptive."""
    a, b = _two_curves_on(
        data, mb.cv_weights(N, type="subsampling", B=10, seed=3))
    comparison = a.compare_to(b)
    assert comparison.design == "10 supplied folds"
    assert not comparison.inferential
    fixed = a.compare_to(b, mstop=100)
    assert fixed.inferential
    assert fixed.interval is not None
    assert "assumes random folds" in fixed.describe()


# ─── the interval is a t interval at a fixed grid point ────────────


def test_the_critical_value_is_a_t_quantile_not_a_normal_one(data, folds):
    """The docstring promised a t interval on ``n_folds-1`` degrees of
    freedom but the code used the normal quantile (1.96 for 95%). On the
    handful of folds real comparisons run the two differ materially."""
    from mboostrs.cvrisk import _t_quantile

    a, b = _two_curves_on(data, folds)
    comparison = a.compare_to(b, mstop=100)
    assert comparison.n_folds == 10
    assert comparison.interval is not None
    half = (comparison.interval[1] - comparison.mean_difference) / \
        comparison.se_corrected
    assert half == pytest.approx(_t_quantile(0.975, 9), rel=1e-9)
    assert half > 1.96  # the normal quantile the code used to use


def test_a_fixed_point_on_the_grid_is_inferential_and_labels_itself(
        data, folds):
    a, b = _two_curves_on(data, folds)
    comparison = a.compare_to(b, mstop=150)
    assert comparison.inferential
    assert comparison.mstop_a == comparison.mstop_b == 150
    assert "mstop 150 vs 150" in comparison.describe()


def test_an_mstop_off_either_grid_is_refused(data, folds):
    a, b = _two_curves_on(data, folds)
    with pytest.raises(ValueError, match="on neither"):
        a.compare_to(b, mstop=200)


@pytest.mark.parametrize("t, dof, expected", [
    (0.975, 1, 12.7062047361747),
    (0.975, 4, 2.7764451051977987),
    (0.975, 9, 2.262157162798205),
    (0.9, 30, 1.3104150253913955),
])
def test_the_student_t_quantile_matches_known_values(t, dof, expected):
    """No scipy dependency: the critical values are computed analytically and
    pinned against reference values."""
    from mboostrs.cvrisk import _t_quantile

    assert _t_quantile(t, dof) == pytest.approx(expected, rel=1e-9)


# ─── the LSS grid ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lss(data):
    return mb.glmboostLSS("y ~ x + z", data=data, families=mb.GaussianLSS(),
                          control=mb.boost_control(60, 0.1))


def test_a_diagonal_grid_costs_one_path_per_fold(lss):
    grid = mb.diagonal_grid(60, 2, length_out=6)
    assert all(len(set(point)) == 1 for point in grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = lss.cvrisk(folds=mb.cv_weights(N, type="kfold", B=5, seed=1),
                            grid=grid)
    cost = result.grid_cost()
    assert cost["points"] == len(grid)
    assert cost["diagonal"] == len(grid)
    assert cost["fresh_fits"] == 0
    assert cost["paths_per_fold"] == 1


def test_an_off_diagonal_point_costs_its_own_fit(lss):
    """A cyclic LSS path is not prefix-consistent: freezing one parameter
    changes the later updates of the others, so the state after 20 mu-steps
    and 5 sigma-steps is not a prefix of any path that took 20 of each."""
    grid = [[5, 5], [20, 20], [20, 5], [5, 20]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = lss.cvrisk(folds=mb.cv_weights(N, type="kfold", B=5, seed=1),
                            grid=grid)
    cost = result.grid_cost()
    assert cost["diagonal"] == 2
    assert cost["fresh_fits"] == 2
    assert cost["paths_per_fold"] == 3
    assert cost["paths_total"] == 15


def test_an_lss_optimum_at_the_largest_budget_warns(lss):
    """The same failure as the standard boundary, on a grid: the reported
    optimum says where the search stopped."""
    with pytest.warns(RuntimeWarning, match="largest budget in the grid"):
        result = lss.cvrisk(
            folds=mb.cv_weights(N, type="kfold", B=5, seed=1),
            grid=[[2, 2], [4, 4], [8, 8]],
        )
    assert result.at_boundary()
    assert result.optimal_mstop == [8, 8]


def test_the_lss_curve_reports_its_spread_and_the_one_se_point(lss):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = lss.cvrisk(folds=mb.cv_weights(N, type="kfold", B=5, seed=1),
                            grid=mb.diagonal_grid(60, 2, length_out=6))
    assert result.standard_error.shape == np.asarray(result.aggregate).shape
    one_se = result.mstop_1se
    assert one_se in [list(point) for point in result.grid]
    assert sum(one_se) <= sum(result.optimal_mstop)
