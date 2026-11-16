"""LSS gradient stabilization matches `gamboostLSS`.

The parameters of a GAMLSS live on different scales — a location in the
response's units, a log-scale in dimensionless ones — so their partial
gradients have different magnitudes and one `nu` takes very different-sized
steps on each. `gamboostLSS` addresses this by dividing each gradient by a
measure of its own spread before selection.

The two settings are `gamboostLSS:::stabilize_ngradient`'s, floors and ceiling
included:

- ``"MAD"``: divide by the weighted median absolute deviation from the
  weighted median, floored at 1e-4.
- ``"L2"``: divide by the root weighted mean square, clamped to [1e-4, 1e4].

Reference coefficients come from `gamboostLSS::glmboostLSS` with
``GaussianLSS(stabilization=...)`` on the fixture below. The unit-level
checks — the weighted median's tie rule, the floors, the ceiling — live in
`crates/mboostrs-core/src/lss.rs`, where the arithmetic is.
"""

import numpy as np
import pytest

import mboostrs as mb

#: `set.seed(3); n <- 300; x <- rnorm(n); y <- rnorm(n, 1 + 0.8*x,
#: exp(-0.3 + 0.4*x))`, written out so the Python side does not depend on
#: reproducing R's RNG.
FIXTURE = "tests/test_gamlss_parity/stabilization_fixture.csv"

#: From `gamboostLSS::glmboostLSS(y ~ x, families = GaussianLSS(
#: stabilization = ...), control = boost_control(mstop = 60, nu = 0.1))`.
GAMBOOSTLSS_MU_COEFFICIENTS = {
    "none": [-0.05234, 0.79607],
    "MAD": [-0.05332, 0.79500],
    "L2": [-0.05049, 0.79783],
}


@pytest.fixture(scope="module")
def data():
    import pathlib

    import polars as pl

    path = pathlib.Path(__file__).resolve().parents[2] / FIXTURE
    if not path.exists():
        pytest.skip(f"{FIXTURE} is missing")
    return pl.read_csv(path)


def _fit(data, stabilization):
    return mb.glmboostLSS(
        "y ~ x", data=data, families=mb.GaussianLSS(),
        control=mb.boost_control(60, 0.1), stabilization=stabilization,
    )


@pytest.mark.parametrize("stabilization", ["none", "MAD", "L2"])
def test_coefficients_match_gamboostlss(data, stabilization):
    model = _fit(data, stabilization)
    got = [float(value) for value in model.coef()["mu"].values()]
    expected = GAMBOOSTLSS_MU_COEFFICIENTS[stabilization]
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_the_three_settings_give_three_different_paths(data):
    """If they agreed, the reference values above would prove nothing."""
    paths = {
        name: [float(value) for value in _fit(data, name).coef()["mu"].values()]
        for name in ("none", "MAD", "L2")
    }
    assert paths["none"] != paths["MAD"]
    assert paths["none"] != paths["L2"]
    assert paths["MAD"] != paths["L2"]


def test_none_is_the_default(data):
    """`gamboostLSS`'s default too, so an unchanged call is unchanged."""
    np.testing.assert_allclose(
        [float(v) for v in _fit(data, "none").coef()["mu"].values()],
        [float(v) for v in mb.glmboostLSS(
            "y ~ x", data=data, families=mb.GaussianLSS(),
            control=mb.boost_control(60, 0.1),
        ).coef()["mu"].values()],
        rtol=1e-12,
    )


@pytest.mark.parametrize("spelling", ["MAD", "mad", "L2", "l2", "none"])
def test_the_spellings_both_frontends_use_are_accepted(data, spelling):
    model = _fit(data, spelling)
    assert np.all(np.isfinite(model._risk))


def test_an_unknown_setting_is_rejected_by_name(data):
    with pytest.raises(ValueError) as excinfo:
        _fit(data, "robust")
    message = str(excinfo.value)
    assert "stabilization must be" in message
    assert "robust" in message


@pytest.mark.parametrize("method", ["cyclic", "noncyclic"])
def test_both_cyclic_and_noncyclic_take_the_setting(data, method):
    """The three gradient sites in the engine are the two drivers plus the
    diagonal-checkpoint path, and all three had to be reached."""
    plain = mb.glmboostLSS("y ~ x", data=data, families=mb.GaussianLSS(),
                           control=mb.boost_control(40, 0.1), method=method)
    stabilized = mb.glmboostLSS("y ~ x", data=data, families=mb.GaussianLSS(),
                                control=mb.boost_control(40, 0.1), method=method,
                                stabilization="MAD")
    assert not np.allclose(plain._risk, stabilized._risk, atol=1e-9), (
        f"stabilization did not reach the {method} driver"
    )


def test_gamboostlss_takes_the_setting_too(data):
    plain = mb.gamboostLSS("y ~ bbs(x, df=3)", data=data,
                           families=mb.GaussianLSS(),
                           control=mb.boost_control(30, 0.1))
    stabilized = mb.gamboostLSS("y ~ bbs(x, df=3)", data=data,
                                families=mb.GaussianLSS(),
                                control=mb.boost_control(30, 0.1),
                                stabilization="MAD")
    assert not np.allclose(plain._risk, stabilized._risk, atol=1e-9)
