"""The #821 ports agree with independent NumPy/statistics calculations."""

import inspect
import subprocess
import sys
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pytest

from mboostrs import _rust
from mboostrs._autospec import _heteroscedasticity_screen
from mboostrs.causal import _candidate_gains
from mboostrs.split_refit import _draw_p, _two_sided_p, meinshausen_buhlmann_p


@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5, 0.9])
def test_split_p_matches_numpy(gamma):
    p = np.random.default_rng(821).uniform(0, 0.3, 24)
    expected = min(1.0, np.quantile(p / gamma, gamma))
    assert meinshausen_buhlmann_p(p, gamma) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("estimate", [-3.0, -0.1, 0.0, 0.1, 3.0])
@pytest.mark.parametrize("level", [0.5, 0.9, 0.95, 0.99])
def test_wald_p_matches_statistics(estimate, level):
    width = NormalDist().inv_cdf(0.5 + level / 2) * 0.7
    expected = 2 * NormalDist().cdf(-abs(estimate) / 0.7)
    assert _two_sided_p(estimate, estimate - width, estimate + width, level) == pytest.approx(
        expected, abs=2e-15)


def test_wald_tail_is_not_rounded_to_zero():
    width = NormalDist().inv_cdf(0.975)
    assert _two_sided_p(10.0, 10 - width, 10 + width, 0.95) == pytest.approx(
        1.523970604832105e-23, rel=1e-12, abs=0)


def test_draw_contrast_matches_numpy_on_noncontiguous_inputs():
    rng = np.random.default_rng(821)
    a = rng.normal(size=(40, 12))[:, ::2]
    b = rng.normal(size=(40, 12))[:, ::2]
    point_a, point_b = a[0], b[0]
    draws = (a - b).mean(axis=1)
    bounds = np.quantile(draws, [0.05, 0.95])
    p = min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
    expected = ((point_a - point_b).mean(), *bounds, p)
    np.testing.assert_allclose(_rust.draw_contrast(point_a, point_b, a, b, 0.9), expected,
                               atol=1e-14, rtol=1e-13)
    assert _draw_p(draws) == p
    assert _draw_p(np.zeros(4)) == 1.0


def test_candidate_gains_preserve_weighted_row_losses_and_flat_guard():
    rng = np.random.default_rng(821)
    weights = rng.uniform(0, 2, 30)
    weights[::4] = 0
    constant = weights * rng.normal(size=30) ** 2
    losses = {"smooth": weights * rng.normal(size=30) ** 2,
              "stumps": constant - weights}
    gain, se, flat, eligible = _candidate_gains(losses, constant, {"stumps": 1})
    for name, values in losses.items():
        diff = constant - values
        assert gain[name] == pytest.approx(diff.mean(), abs=1e-15)
        assert se[name] == pytest.approx(diff.std(ddof=1) / np.sqrt(len(diff)), abs=1e-15)
        assert eligible[name] == bool(gain[name] > se[name] and not flat[name])
    assert flat["stumps"] and not eligible["stumps"]
    assert np.isnan(_rust.paired_mean_se(np.ones(1), np.zeros(1))[1])


def _legacy_variance_screen(residual, design):
    """Pre-port equations, retained only as an independent test oracle."""
    x = design - np.nanmean(design, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sq = np.log(residual**2)
    keep = np.isfinite(log_sq) & np.isfinite(x).all(axis=1)
    x, y = x[keep], log_sq[keep]
    y = y - y.mean()
    inverse = np.linalg.inv(x.T @ x)
    slopes = inverse @ (x.T @ y)
    t_stats = slopes / np.sqrt(np.pi**2 / 2 * np.diag(inverse))
    r_squared = 1 - np.sum((y - x @ slopes)**2) / np.sum(y**2)
    return slopes, t_stats, r_squared, int(keep.sum())


@pytest.mark.parametrize("incomplete", [False, True])
def test_variance_screen_matches_numpy_and_column_mapping(incomplete):
    rng = np.random.default_rng(821)
    n = 100
    x = rng.normal(size=n)
    group = rng.integers(0, 3, n).astype(float)
    residual = np.exp(0.3 * x + 0.6 * (group == 2)) * rng.normal(size=n)
    if incomplete:
        residual[[2, 3, 4]] = [0, np.nan, np.inf]
        group[0] = np.nan
    columns = [SimpleNamespace(values=x, kind="numeric", name="x"),
               SimpleNamespace(values=group, kind="categorical", name="group")]
    design = np.column_stack([x, group == 1, group == 2])
    design[~np.isfinite(group), 1:] = np.nan
    expected = _legacy_variance_screen(residual, design)
    actual = _rust.log_variance_screen(residual, design)
    for got, want in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_allclose(got, want, atol=1e-13, rtol=1e-12)
    assert actual[2:] == pytest.approx(expected[2:], abs=1e-13)
    slopes, t_stats, owners, r_squared, n_used = _heteroscedasticity_screen(
        residual, columns, np.arange(n))
    assert owners == ["x", "group", "group"]
    np.testing.assert_allclose(slopes, expected[0], atol=1e-13)
    np.testing.assert_allclose(t_stats, expected[1], atol=1e-13)
    assert (r_squared, n_used) == pytest.approx(expected[2:])


def test_variance_screen_declines_degenerate_data():
    x = np.arange(20.0)
    assert _rust.log_variance_screen(np.zeros(20), x[:, None]) is None
    assert _rust.log_variance_screen(np.ones(20), np.ones((20, 2))) is None
    assert _rust.log_variance_screen(np.ones(5), x[:5, None]) is None
    with pytest.raises(ValueError, match="rows"):
        _rust.log_variance_screen(np.ones(5), x[:, None])


@pytest.mark.parametrize("seed", [0, 3, 4, 5])
def test_variance_screen_declines_duplicate_columns_despite_roundoff(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=40)
    assert _rust.log_variance_screen(rng.normal(size=40), np.column_stack([x, x])) is None


def test_variance_rank_check_is_relative_to_each_column_scale():
    rng = np.random.default_rng(821)
    x = rng.normal(size=(100, 2))
    residual = rng.normal(size=100)
    original = _rust.log_variance_screen(residual, x)
    x[:, 1] *= 1e-12
    scaled = _rust.log_variance_screen(residual, x)
    np.testing.assert_allclose(scaled[1], original[1], rtol=1e-12, atol=1e-13)
    assert scaled[0][1] * 1e-12 == pytest.approx(original[0][1], rel=1e-12)


@pytest.mark.parametrize("name", ["disc_scan", "disc_step_z", "iv_ratio_weighted"])
def test_former_fallback_kernels_are_native(name):
    assert inspect.isbuiltin(getattr(_rust, name))


def test_import_does_not_replace_missing_native_kernels():
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib
import mboostrs
from mboostrs import _rust
names = ('disc_scan', 'disc_step_z', 'iv_ratio_weighted')
for name in names:
    delattr(_rust, name)
importlib.reload(mboostrs)
assert not any(hasattr(_rust, name) for name in names)
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("p", [[], [np.nan], [np.inf], [-0.01], [1.01]])
def test_invalid_split_p_values_are_refused(p):
    with pytest.raises(ValueError, match="p_values"):
        meinshausen_buhlmann_p(p)


@pytest.mark.r_parity
def test_python_and_r_bindings_agree():
    import rpy2.robjects as ro

    call = ro.r('''function(name, ...) {
        loadNamespace("mboostrs")
        .Call(paste0("wrap__r_", name), ..., PACKAGE = "mboostrs")
    }''')

    def matrix(x):
        return ro.r.matrix(ro.FloatVector(x.ravel(order="F")), nrow=x.shape[0])

    rng = np.random.default_rng(821)
    x = rng.normal(size=(40, 3))
    residual = rng.normal(size=40) * np.exp(x[:, 0])
    py = _rust.log_variance_screen(residual, x)
    r = call("log_variance_screen", ro.FloatVector(residual), matrix(x))
    for name, expected in zip(("slopes", "t_stats", "r_squared", "n_used"), py, strict=True):
        np.testing.assert_allclose(np.asarray(r.rx2(name)), expected, atol=1e-13, rtol=1e-13)

    a, b = rng.normal(size=(40, 6)), rng.normal(size=(40, 6))
    py = _rust.draw_contrast(a[0], b[0], a, b, 0.95)
    r = call("draw_contrast", ro.FloatVector(a[0]), ro.FloatVector(b[0]),
             matrix(a), matrix(b), 0.95)
    np.testing.assert_allclose(np.asarray(ro.r.unlist(r)), py, atol=1e-13, rtol=1e-13)

    py = _rust.paired_mean_se(a[0], b[0])
    r = call("paired_mean_se", ro.FloatVector(a[0]), ro.FloatVector(b[0]))
    np.testing.assert_allclose(np.asarray(ro.r.unlist(r)), py, atol=1e-13, rtol=1e-13)
