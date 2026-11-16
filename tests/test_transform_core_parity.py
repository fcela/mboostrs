"""The Rust transform kernels equal the Python implementation they replaced.

`_old_*` are frozen copies of the NumPy code that `fit_recipe`, `_ecdf_params`,
`_evaluate`, `response_inverse`, `response_jacobian` and the median fills used
before the numerics moved to `mboostrs-core`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mboostrs import _rust
from mboostrs.transforms import (
    TransformRecipe,
    _ecdf_params,
    fit_recipe,
    extract_transforms,
    response_inverse,
    response_jacobian,
)


def _old_mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def _old_ecdf_params(values):
    finite = values[np.isfinite(values)]
    probabilities = np.linspace(0.0, 1.0, min(257, finite.size))
    quantiles = np.quantile(finite, probabilities)
    unique, inverse = np.unique(quantiles, return_inverse=True)
    averaged = np.bincount(inverse, weights=probabilities) / np.bincount(inverse)
    return (tuple(float(v) for v in unique), tuple(float(v) for v in averaged))


def _old_forward(kind, params, x):
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind in ("scale", "center"):
            return (x - params[0]) / params[1]
        if kind == "ecdf":
            return np.interp(x, params[0], params[1])
        return {"log": np.log, "log1p": np.log1p, "sqrt": np.sqrt, "exp": np.exp}[kind](x)


def _old_inverse(kind, params, v):
    if kind == "log":
        return np.exp(v)
    if kind == "log1p":
        return np.expm1(v)
    if kind == "sqrt":
        return v ** 2
    if kind == "exp":
        return np.log(v)
    if kind == "scale":
        return v * params[1] + params[0]
    return v + params[0]


def _old_jacobian(kind, params, y):
    if kind == "log":
        return 1.0 / y
    if kind == "log1p":
        return 1.0 / (1.0 + y)
    if kind == "sqrt":
        return 0.5 / np.sqrt(y)
    if kind == "exp":
        return np.exp(y)
    if kind == "scale":
        return np.full_like(y, 1.0 / params[1])
    return np.ones_like(y)


def _columns():
    rng = np.random.default_rng(7)
    return {
        "normal": rng.normal(3.0, 2.0, 500),
        "ties": rng.integers(0, 4, 300).astype(float),
        "heavy_ties": np.r_[np.zeros(200), rng.normal(size=20)],
        "n2": np.array([1.0, 4.0]),
        "n3": np.array([5.0, 1.0, 3.0]),
        "big": rng.normal(size=5000),
        "with_nan": np.r_[rng.normal(size=50), np.nan, np.inf],
        "large_offset": 1e9 + rng.normal(size=100),
    }


@pytest.mark.parametrize("name", ["normal", "ties", "heavy_ties", "n2", "n3", "big", "large_offset"])
def test_mean_sd_matches_numpy(name):
    x = _columns()[name]
    mean, sd = _rust.transform_mean_sd(x)
    omean, osd = _old_mean_sd(x)
    assert abs(mean - omean) <= 1e-12 * max(1.0, abs(omean))
    assert abs(sd - osd) <= 1e-12 * max(1.0, abs(osd))


def test_mean_sd_n1_and_nan():
    assert np.isnan(_rust.transform_mean_sd(np.array([2.0]))[1])
    assert _rust.transform_mean_sd(np.array([2.0]))[0] == 2.0
    assert np.isnan(_rust.transform_mean_sd(np.array([1.0, np.nan, 3.0]))[0])


@pytest.mark.parametrize("name", ["normal", "ties", "heavy_ties", "n2", "n3", "big", "with_nan"])
def test_ecdf_params_equal_numpy(name):
    x = _columns()[name]
    new = _ecdf_params(x, "x")
    old = _old_ecdf_params(x)
    assert new[0] == old[0]
    np.testing.assert_allclose(new[1], old[1], rtol=0, atol=1e-12)


def test_ecdf_params_degenerate_inputs_refused():
    for x in (np.array([1.0]), np.full(10, 2.0), np.array([np.nan, np.nan, 1.0])):
        with pytest.raises(ValueError, match=r"ecdf\(x\) needs at least two distinct"):
            _ecdf_params(x, "x")


@pytest.mark.parametrize("name", ["normal", "ties", "heavy_ties", "n2", "n3", "big"])
def test_ecdf_apply_equals_interp(name):
    x = _columns()[name]
    knots, probs = _old_ecdf_params(x)
    new_x = np.r_[x, x.min() - 1, x.max() + 1, np.linspace(x.min(), x.max(), 77), np.nan]
    got = np.asarray(_rust.transform_forward("ecdf", [*knots, *probs], new_x))
    want = np.interp(new_x, knots, probs)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 101, 1000])
def test_median_is_exact(n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    assert _rust.transform_median(x) == float(np.median(x))
    ints = rng.integers(0, 5, n).astype(float)
    assert _rust.transform_median(ints) == float(np.median(ints))


def test_median_nan_and_all_equal():
    assert np.isnan(_rust.transform_median(np.array([1.0, np.nan])))
    assert _rust.transform_median(np.full(7, 3.5)) == 3.5


def test_fit_recipe_scale_center_and_zero_variance():
    x = _columns()["normal"]
    data = {"x": x, "z": np.full(10, 2.0), "y": x}
    _, recipe = extract_transforms("y ~ scale(x) + center(x)", data)
    fitted = fit_recipe(recipe, data)
    omean, osd = _old_mean_sd(x)
    scale_step, center_step = fitted.steps
    assert scale_step.params == pytest.approx((omean, osd), rel=1e-12)
    assert center_step.params == (pytest.approx(omean, rel=1e-12), 1.0)
    data_z = {"z": data["z"], "y": data["z"]}
    _, bad = extract_transforms("y ~ scale(z)", data_z)
    with pytest.raises(ValueError, match="zero variance"):
        fit_recipe(bad, data_z)


@pytest.mark.parametrize("kind", ["log", "log1p", "sqrt", "exp"])
def test_stateless_kernels(kind):
    rng = np.random.default_rng(3)
    x = rng.uniform(0.01, 20.0, 400) if kind != "exp" else rng.uniform(-5, 5, 400)
    got = np.asarray(_rust.transform_forward(kind, [], x))
    np.testing.assert_allclose(got, _old_forward(kind, (), x), rtol=1e-12, atol=0)
    with np.errstate(invalid="ignore"):
        inv = np.asarray(_rust.transform_inverse(kind, [], x))
        np.testing.assert_allclose(inv, _old_inverse(kind, (), x), rtol=1e-12, atol=0, equal_nan=True)
        jac = np.asarray(_rust.transform_jacobian(kind, [], x))
        np.testing.assert_allclose(jac, _old_jacobian(kind, (), x), rtol=1e-12, atol=0, equal_nan=True)


@pytest.mark.parametrize("kind", ["scale", "center"])
def test_affine_kernels(kind):
    x = _columns()["normal"]
    params = (1.7, 2.3) if kind == "scale" else (1.7, 1.0)
    np.testing.assert_allclose(_rust.transform_forward(kind, list(params), x),
                               _old_forward(kind, params, x), rtol=1e-12, atol=0)
    np.testing.assert_allclose(_rust.transform_inverse(kind, list(params), x),
                               _old_inverse(kind, params, x), rtol=1e-12, atol=0)
    np.testing.assert_allclose(_rust.transform_jacobian(kind, list(params), x),
                               _old_jacobian(kind, params, x), rtol=1e-12, atol=0)


def test_response_helpers_use_the_kernels_and_refuse_noninvertible():
    y = np.abs(_columns()["normal"]) + 1.0
    data = {"y": y, "x": np.arange(500.0)}
    _, recipe = extract_transforms("log(y) ~ x", data)
    step = fit_recipe(recipe, data).steps[0]
    v = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(response_inverse(step, v), np.exp(v), rtol=1e-12)
    np.testing.assert_allclose(response_jacobian(step, np.exp(v)), 1.0 / np.exp(v), rtol=1e-12)
    assert response_inverse(step, v).shape == (2, 2)
    with pytest.raises(ValueError):
        _rust.transform_inverse("ecdf", [0.0, 1.0], np.array([0.5]))


def test_empty_recipe_untouched():
    assert not fit_recipe(TransformRecipe(), {})
