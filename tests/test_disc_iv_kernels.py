"""The compiled `disc_scan`, `disc_step_z` and `iv_ratio_weighted` kernels
return the values the NumPy copies in the removed `_rust_fallbacks.py`
returned on the same inputs (#821).

The fallbacks were installed into `mboostrs._rust` when a build did not
export these kernels. They were removed so that the Rust core is the only
implementation; the values below were computed by them before removal.
"""

import numpy as np
import pytest

from mboostrs import _rust


def _disc_inputs():
    rng = np.random.default_rng(11)
    n = 40
    u = np.round(rng.uniform(0, 1, n), 6)
    background = np.column_stack([np.ones(n), np.round(u - 0.5, 6)])
    target = np.round(0.5 * u + (u > 0.55) * 0.8 + rng.normal(scale=0.2, size=n), 6)
    fit = (np.arange(n) % 2).astype(np.uint8)
    return rng, u, background, target, fit


def test_disc_scan_and_step_z_match_the_former_numpy_values():
    _, u, background, target, fit = _disc_inputs()
    rows = _rust.disc_scan(u, background, target, fit, np.array([0.3, 0.55, 0.7]), 3)
    expected = [
        (0.3, 25, -0.4040907969465156, -2.4371466711063245, 0.08253685329095485,
         0.18800946801007465, 0.4390037063800002, -0.016779078288616397),
        (0.55, 16, 0.5111156114482753, 4.3859604725711305, 0.48847988010499915,
         0.17474734731490882, 2.7953493292502976, 0.025295518050619317),
        (0.7, 9, 0.22513640237802032, 1.3115230992081255, -0.3874026908346993,
         0.15639646120362774, -2.477055349291453, -0.019007668057144753),
    ]
    assert len(rows) == len(expected)
    for row, want in zip(rows, expected, strict=True):
        assert row[1] == want[1]
        np.testing.assert_allclose([row[0], *row[2:]], [want[0], *want[2:]], rtol=1e-8)
    indicator = (u > 0.55).astype(float)
    z = _rust.disc_step_z(background, indicator, target, fit)
    assert z == pytest.approx(4.385960472571109, rel=1e-8)


def test_iv_ratio_weighted_matches_the_former_numpy_values():
    rng, *_ = _disc_inputs()
    n = 40
    y_res = np.round(rng.normal(size=n), 6)
    d_res = np.round(rng.normal(size=n), 6)
    z_res = np.round(0.6 * d_res + rng.normal(size=n), 6)
    w = np.round(rng.uniform(0.5, 2.0, n), 6)
    w[3] = 0.0
    estimate, variance, influence, r2, first_stage_f, rows = _rust.iv_ratio_weighted(
        y_res, d_res, z_res, w)
    assert estimate == pytest.approx(0.24915238074070212, rel=1e-8)
    assert variance == pytest.approx(0.19533112323998048, rel=1e-8)
    assert r2 == pytest.approx(0.3206019008260111, rel=1e-8)
    assert first_stage_f == pytest.approx(18.403752008455918, rel=1e-8)
    assert rows == 40
    assert float(np.asarray(influence)[0]) == pytest.approx(-2.0716261198898733, rel=1e-8)


def test_the_package_installs_no_python_kernels():
    import mboostrs

    assert not hasattr(mboostrs, "_rust_fallbacks")
    for name in ("disc_scan", "disc_step_z", "iv_ratio_weighted"):
        assert type(getattr(_rust, name)).__name__ == "builtin_function_or_method"
