"""The conformal numerics run in the Rust core and match the former Python.

The quantile rule, residual and CQR scores, band widening and midpoint, the
CV+ order statistics and the tuned step grid moved from `conformal.py` to
`mboostrs_core::conformal`. The numerics below are the Python implementation
they replaced, kept here as an oracle. Each oracle function is installed in
place of the Rust binding, so the orchestration (refits, permutations, fold
assignment, family quantiles) is the package's own and only the numerics
differ. Every interval field must agree to 1e-12.
"""

import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import conformal

RTOL = 1e-12
ATOL = 1e-12


def _near_integer(x):
    r = round(x)
    return r if abs(x - r) <= 1e-9 * max(abs(x), 1.0) else None


def _rank_floor(x):
    r = _near_integer(x)
    return int(np.floor(x)) if r is None else int(r)


def _rank_ceil(x):
    r = _near_integer(x)
    return int(np.ceil(x)) if r is None else int(r)


# --------------------------------------------------------------------------
# Oracle: the numerics as they were in python/mboostrs/conformal.py
# --------------------------------------------------------------------------


class OraclePythonNumerics:
    """Drop-in for `mboostrs._rust` with the former Python numerics."""

    def conformal_count_default_score(self, family, y):
        # This legacy oracle covers non-count fits; count-rank numerics have
        # independent likelihood/CDF references in test_count_rank_calibration.
        if family in ("poisson", "nbinomial"):
            raise NotImplementedError("count defaults use the separate count-rank oracle")
        return "absolute"

    def conformal_min_calibration_size(self, level):
        # Issue #817: ceil(level / (1 - level)), with a product within rounding
        # error of an integer taken as that integer (9 at 0.9, not 10).
        return _rank_ceil(level / (1.0 - level))

    def conformal_calibration_count(self, n, fraction):
        n_calib = int(round(n * fraction))
        if n_calib < 1 or n - n_calib < 1:
            raise ValueError(
                f"calib_fraction={fraction} leaves an empty split for n={n}"
            )
        return n_calib

    def conformal_quantile(self, scores, level):
        scores = np.sort(np.asarray(scores, dtype=np.float64).ravel())
        n = scores.size
        k = int(np.ceil((n + 1) * level))
        if k > n:
            return float("inf")
        return float(scores[k - 1])

    def conformal_absolute_residuals(self, y, predicted):
        return np.abs(np.asarray(y) - np.asarray(predicted))

    def conformal_symmetric_band(self, predicted, q):
        predicted = np.asarray(predicted)
        return predicted - q, predicted + q

    def conformal_ordered_pair(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.minimum(a, b), np.maximum(a, b)

    def conformal_quantile_scores(self, lower, upper, y):
        return np.maximum(np.asarray(lower) - y, np.asarray(y) - np.asarray(upper))

    def conformal_worst_covariate_bin(self, x, lower, upper, y, level, bins):
        x, lower, upper, y = (np.asarray(v, dtype=np.float64) for v in (x, lower, upper, y))
        n = x.size
        order = np.argsort(x, kind="stable")
        worst = None
        for b in range(bins):
            idx = order[b * n // bins:(b + 1) * n // bins]
            count = idx.size
            coverage = float(np.mean((lower[idx] <= y[idx]) & (y[idx] <= upper[idx])))
            z = (coverage - level) / np.sqrt(level * (1.0 - level) / count)
            if worst is None or z < worst[0]:
                worst = (float(z), b, coverage, count)
        return worst

    def conformal_scale_target(self, y, predicted):
        return np.log(np.maximum((np.asarray(y) - np.asarray(predicted)) ** 2, 1e-12))

    def conformal_rank_scale_columns(self, columns, target, keep):
        columns = [np.asarray(c, dtype=np.float64) for c in columns]
        target = np.asarray(target, dtype=np.float64)
        if len(columns) <= keep:
            return list(range(len(columns)))
        centred = target - target.mean()

        def strength(column):
            association = 0.0
            for view in (column, np.abs(column - np.median(column))):
                view = view - view.mean()
                denominator = np.sqrt(float(np.sum(view ** 2)) * float(np.sum(centred ** 2)))
                if denominator > 0:
                    association = max(association,
                                      abs(float(np.sum(view * centred)) / denominator))
            return association

        order = sorted(range(len(columns)), key=lambda i: strength(columns[i]), reverse=True)
        return order[:keep]

    def conformal_widen_band(self, lower, upper, q):
        lo, hi = np.asarray(lower) - q, np.asarray(upper) + q
        inverted = lo > hi
        if np.any(inverted):
            middle = 0.5 * (lo + hi)
            lo = np.where(inverted, middle, lo)
            hi = np.where(inverted, middle, hi)
        return lo, hi

    def conformal_band_midpoint(self, lower, upper):
        return 0.5 * (np.asarray(lower) + np.asarray(upper))

    def conformal_clip_to_band(self, values, lower, upper):
        return np.clip(np.asarray(values), np.asarray(lower), np.asarray(upper))

    def conformal_cv_plus_ranks(self, n, level):
        # Issue #817: whole-number products are exact, so n=9, level=0.9 gives (1, 9).
        alpha = 1.0 - level
        return _rank_floor(alpha * (n + 1)), _rank_ceil(level * (n + 1))

    def conformal_cv_plus_guaranteed_coverage(self, n, k, level):
        if n % k:
            return None
        alpha = 1.0 - level
        fold_size = n / k
        correction = min(
            2.0 * (1.0 - 1.0 / k) / (fold_size + 1.0),
            (1.0 - k / n) / (k + 1.0),
        )
        return max(0.0, 1.0 - 2.0 * alpha - correction)

    def conformal_cv_plus_bounds(self, residuals, fold_of, fold_predictions, level):
        residuals = np.asarray(residuals, dtype=np.float64)
        fold_predictions = np.asarray(fold_predictions, dtype=np.float64)
        n = residuals.size
        n_new = fold_predictions.shape[1]
        k_lo, k_hi = self.conformal_cv_plus_ranks(n, level)
        block = fold_predictions[np.asarray(fold_of, dtype=np.int64), :]
        lower = np.full(n_new, -np.inf)
        upper = np.full(n_new, np.inf)
        if k_lo >= 1:
            lower[:] = np.sort(block - residuals[:, None], axis=0)[k_lo - 1, :]
        if k_hi <= n:
            upper[:] = np.sort(block + residuals[:, None], axis=0)[k_hi - 1, :]
        return lower, upper

    def conformal_tuned_mstop_start(self, model_mstop):
        return max(int(model_mstop), 1600)

    def conformal_tuned_mstop_next(self, largest, cap):
        return min(2 * largest, cap)

    def conformal_tuned_mstop_choice(self, optimal):
        return max(int(optimal), 1)


@pytest.fixture
def oracle(monkeypatch):
    """Route every conformal numeric through the former Python code."""
    monkeypatch.setattr(conformal, "_rust", OraclePythonNumerics())


# --------------------------------------------------------------------------
# Data and models
# --------------------------------------------------------------------------


def _frame(seed=0, n=240):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.normal(size=n)
    y = np.sin(x1) + 0.5 * x2 + (0.2 + 0.4 * (x1 > 0)) * rng.standard_t(4, n)
    return {"x1": x1, "x2": x2, "y": y}


def _glm(data):
    return mb.glmboost("y ~ x1 + x2", data=data, control=mb.boost_control(40, 0.1))


def _gam(data):
    return mb.gamboost("y ~ bbs(x1, df=4) + x2", data=data,
                       control=mb.boost_control(40, 0.1))


def _lss(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboostLSS({"mu": "y ~ bbs(x1)", "sigma": "y ~ bbs(x1)"},
                              data={"x1": data["x1"], "y": data["y"]},
                              families=mb.GaussianLSS(),
                              control=mb.boost_control(60))


def _assert_same_interval(actual, expected):
    for field in ("lower", "upper", "predicted"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field),
                                   rtol=RTOL, atol=ATOL, err_msg=field)
    assert (actual.quantile is None) == (expected.quantile is None)
    if expected.quantile is not None:
        np.testing.assert_allclose(actual.quantile, expected.quantile, rtol=RTOL, atol=ATOL)
    assert actual.n_calibration == expected.n_calibration
    assert actual.guaranteed_coverage == expected.guaranteed_coverage
    assert actual.n_set_aside == expected.n_set_aside
    if expected.fitted_mean is not None:
        np.testing.assert_allclose(actual.fitted_mean, expected.fitted_mean,
                                   rtol=RTOL, atol=ATOL)


def _run_both(monkeypatch, make_interval):
    """The interval from the Rust core and from the oracle, same arguments."""
    rust = make_interval()
    monkeypatch.setattr(conformal, "_rust", OraclePythonNumerics())
    oracle_result = make_interval()
    monkeypatch.undo()
    return rust, oracle_result


# --------------------------------------------------------------------------
# Public intervals, method by method, against the oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["split", "cv+", "cqr"])
@pytest.mark.parametrize("level", [0.8, 0.9, 0.95])
def test_standard_model_intervals_match_the_python_numerics(monkeypatch, method, level):
    data = _frame(seed=1)
    model = _glm(data)
    rust, python = _run_both(monkeypatch, lambda: mb.conformal_interval(
        model, level=level, method=method, seed=3, n_folds=5, mstop=25))
    _assert_same_interval(rust, python)


@pytest.mark.parametrize("method", ["split", "cv+"])
def test_newdata_and_gam_intervals_match_the_python_numerics(monkeypatch, method):
    data = _frame(seed=2)
    model = _gam(data)
    new = {k: v[:37] for k, v in _frame(seed=9).items() if k != "y"}
    rust, python = _run_both(monkeypatch, lambda: mb.conformal_interval(
        model, newdata=new, level=0.9, method=method, seed=5, n_folds=4))
    _assert_same_interval(rust, python)


def test_cqr_tuned_quantile_fits_match_the_python_numerics(monkeypatch):
    """CQR tunes each quantile fit's step count by cross-validation, which
    walks the doubling grid through the core's tuned-step functions."""
    data = _frame(seed=4, n=160)
    model = _glm(data)
    rust, python = _run_both(monkeypatch, lambda: mb.conformal_interval(
        model, level=0.9, method="cqr", seed=2))
    _assert_same_interval(rust, python)


def test_held_out_calibration_matches_the_python_numerics(monkeypatch):
    data = _frame(seed=5)
    model = _glm(data)
    calibration = {k: v[:60] for k, v in _frame(seed=6).items()}
    rust, python = _run_both(monkeypatch, lambda: mb.conformal_interval(
        model, level=0.9, calibration=calibration))
    _assert_same_interval(rust, python)


def test_distributional_intervals_match_the_python_numerics(monkeypatch):
    data = _frame(seed=7, n=300)
    model = _lss(data)
    new = {"x1": data["x1"][:50]}
    rust, python = _run_both(monkeypatch, lambda: mb.conformal_interval(
        model, newdata=new, level=0.9, method="distributional", seed=8))
    _assert_same_interval(rust, python)


def test_inverted_cqr_bands_match_the_python_numerics(monkeypatch):
    """A negative correction can invert a band; the midpoint replacement is
    the part of the widening most likely to differ."""
    rng = np.random.default_rng(11)
    x = rng.uniform(-1, 1, 300)
    y = x + 0.02 * rng.normal(size=300)
    data = {"x": x, "y": y}
    model = mb.glmboost("y ~ x", data=data, control=mb.boost_control(30, 0.1))
    rust, python = _run_both(monkeypatch, lambda: mb.conformal_interval(
        model, level=0.5, method="cqr", seed=1, mstop=30))
    _assert_same_interval(rust, python)


# --------------------------------------------------------------------------
# Each binding against the oracle on random inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 9, 10, 57, 400])
@pytest.mark.parametrize("level", [0.05, 0.5, 0.9, 0.99])
def test_quantile_and_calibration_size_match_the_oracle(n, level):
    rng = np.random.default_rng(n)
    scores = rng.standard_normal(n)
    oracle = OraclePythonNumerics()
    got = conformal._conformal_quantile(scores, level)
    want = oracle.conformal_quantile(scores, level)
    assert got == want or (np.isinf(got) and np.isinf(want))
    assert conformal._min_calibration_size(level) == oracle.conformal_min_calibration_size(level)


@pytest.mark.parametrize("n,k", [(10, 5), (12, 4), (101, 5), (200, 10), (7, 7)])
@pytest.mark.parametrize("level", [0.6, 0.9, 0.95])
def test_guarantee_matches_the_oracle(n, k, level):
    oracle = OraclePythonNumerics()
    got = conformal._cv_plus_guaranteed_coverage(n, k, level)
    want = oracle.conformal_cv_plus_guaranteed_coverage(n, k, level)
    assert got == want or (got is not None and want is not None
                           and abs(got - want) <= RTOL)


@pytest.mark.parametrize("seed", range(6))
def test_cv_plus_bounds_match_the_oracle_on_random_folds(seed):
    from mboostrs._rust import conformal_cv_plus_bounds

    rng = np.random.default_rng(seed)
    n, k, m = 37, 4, 19
    residuals = np.abs(rng.standard_normal(n))
    fold_of = rng.integers(0, k, n).astype(np.int64)
    fold_predictions = rng.standard_normal((k, m))
    level = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999][seed]
    lo, hi = conformal_cv_plus_bounds(residuals, fold_of, fold_predictions, level)
    want_lo, want_hi = OraclePythonNumerics().conformal_cv_plus_bounds(
        residuals, fold_of, fold_predictions, level)
    np.testing.assert_allclose(np.asarray(lo), want_lo, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(hi), want_hi, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", range(4))
def test_widen_band_and_midpoint_match_the_oracle(seed):
    rng = np.random.default_rng(100 + seed)
    a = rng.standard_normal(50)
    b = a + np.abs(rng.standard_normal(50)) * 0.3
    oracle = OraclePythonNumerics()
    for q in (-0.4, -0.05, 0.0, 0.7, np.inf):
        lo, hi = conformal._widen(a, b, q)
        want_lo, want_hi = oracle.conformal_widen_band(a, b, q)
        np.testing.assert_allclose(lo, want_lo, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(hi, want_hi, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(conformal._midpoint(lo, hi),
                                   oracle.conformal_band_midpoint(lo, hi),
                                   rtol=RTOL, atol=ATOL)


def test_scores_residuals_and_clipping_match_the_oracle():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(30)
    lo = y - rng.uniform(0, 1, 30)
    hi = y + rng.uniform(-1, 1, 30)
    oracle = OraclePythonNumerics()
    np.testing.assert_allclose(conformal._quantile_scores(lo, hi, y),
                               oracle.conformal_quantile_scores(lo, hi, y), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(conformal._absolute_residuals(y, lo),
                               oracle.conformal_absolute_residuals(y, lo), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(conformal._clip(y, lo, hi),
                               oracle.conformal_clip_to_band(y, lo, hi), rtol=RTOL, atol=ATOL)
    a, b = conformal._ordered(lo, hi)
    want_a, want_b = oracle.conformal_ordered_pair(lo, hi)
    np.testing.assert_allclose(a, want_a, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b, want_b, rtol=RTOL, atol=ATOL)


def test_tuned_step_rules_match_the_oracle():
    from mboostrs._rust import (conformal_tuned_mstop_choice,
                                conformal_tuned_mstop_next, conformal_tuned_mstop_start)

    oracle = OraclePythonNumerics()
    for model_mstop in (1, 50, 200, 201, 900):
        assert conformal_tuned_mstop_start(model_mstop) == oracle.conformal_tuned_mstop_start(model_mstop)
    for largest in (200, 400, 6000, 9000, 10000):
        assert conformal_tuned_mstop_next(largest, 10_000) == oracle.conformal_tuned_mstop_next(largest, 10_000)
    for optimal in (0, 1, 37):
        assert conformal_tuned_mstop_choice(optimal) == oracle.conformal_tuned_mstop_choice(optimal)


# --------------------------------------------------------------------------
# Contract checks at the Python boundary
# --------------------------------------------------------------------------


def test_bad_level_is_refused_with_a_value_error():
    with pytest.raises(ValueError, match="level must be in"):
        mb.conformal_interval(_glm(_frame(seed=0)), level=1.2)


def test_a_non_finite_prediction_is_refused_before_a_quantile_is_taken():
    from mboostrs._rust import conformal_absolute_residuals

    with pytest.raises(ValueError, match="NaN"):
        conformal_absolute_residuals(np.array([1.0, 2.0]), np.array([np.nan, 2.0]))


def test_an_empty_calibration_split_is_refused():
    from mboostrs._rust import conformal_calibration_count

    with pytest.raises(ValueError, match="empty split"):
        conformal_calibration_count(1, 0.5)


def test_fold_index_must_be_within_the_fold_count():
    from mboostrs._rust import conformal_cv_plus_bounds

    with pytest.raises(ValueError, match="out of range"):
        conformal_cv_plus_bounds(np.array([1.0, 2.0]), np.array([0, 5], dtype=np.int64),
                                 np.zeros((2, 3)), 0.9)
