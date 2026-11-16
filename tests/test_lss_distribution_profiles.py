"""Distribution-profile tables and their plotting adapters."""

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def gaussian_lss():
    rng = np.random.default_rng(90)
    n = 100
    x = np.linspace(-1.5, 1.5, n)
    group = np.where(x < 0, "lower", "upper")
    y = 1.0 + 0.7 * x + rng.normal(scale=np.exp(-0.4 + 0.25 * x), size=n)
    data = {"y": y, "x": x, "group": group}
    model = mb.glmboostLSS(
        {"mu": "y ~ x", "sigma": "y ~ x"},
        data=data,
        families=mb.GaussianLSS(),
        control=mb.boost_control(12, 0.08),
    )
    return model, data


@pytest.fixture(scope="module")
def poisson_lss():
    rng = np.random.default_rng(91)
    n = 120
    x = np.linspace(-1, 1, n)
    y = rng.poisson(np.exp(0.4 + 0.5 * x)).astype(float)
    data = {"y": y, "x": x}
    model = mb.glmboostLSS(
        "y ~ x", data=data, families=mb.PoissonLSS(),
        control=mb.boost_control(10, 0.08),
    )
    return model, data


def _series(frame, name):
    return frame.filter(frame["series"] == name).sort("x")


def test_continuous_overlay_averages_row_specific_densities(gaussian_lss):
    model, _ = gaussian_lss
    grid = np.linspace(-5, 6, 301)
    frame = model.density_overlay_data(grid=grid, output="polars")
    fitted = _series(frame, "fitted")
    observed = _series(frame, "observed")
    params = model.params()

    for index in (0, 75, 150, 225, 300):
        expected = np.mean(model._family.pdf(np.full(100, grid[index]), params))
        assert fitted["value"][index] == pytest.approx(expected, abs=1e-14)
    assert np.trapezoid(fitted["value"].to_numpy(), grid) == pytest.approx(1, abs=2e-6)
    assert np.trapezoid(observed["value"].to_numpy(), grid) == pytest.approx(1, abs=2e-6)
    assert set(frame["kind"]) == {"density"}


def test_discrete_overlay_uses_empirical_and_predictive_mass(poisson_lss):
    model, data = poisson_lss
    grid = np.arange(0, 18, dtype=float)
    frame = mb.density_overlay_data(model, grid=grid, output="polars")
    fitted = _series(frame, "fitted")
    observed = _series(frame, "observed")
    expected_zero = np.mean(model._family.pmf(np.zeros(120), model.params()))

    assert fitted["value"][0] == pytest.approx(expected_zero, abs=1e-14)
    assert observed["value"][0] == pytest.approx(np.mean(data["y"] == 0), abs=1e-14)
    assert fitted["value"].sum() == pytest.approx(1, abs=2e-8)
    assert observed["value"].sum() == pytest.approx(1, abs=1e-14)
    assert set(frame["kind"]) == {"mass"}


def test_mixed_overlay_separates_atom_probability_from_positive_density():
    rng = np.random.default_rng(92)
    n = 180
    x = rng.normal(size=n)
    is_zero = rng.random(n) < 0.3
    y = np.where(is_zero, 0.0, rng.gamma(4.0, 0.5, size=n))
    model = mb.glmboostLSS(
        {"mu": "y ~ x", "sigma": "y ~ x", "nu": "y ~ x"},
        data={"y": y, "x": x}, families=mb.ZeroAdjustedGammaLSS(),
        control=mb.boost_control(10, 0.08),
    )
    grid = np.linspace(0.001, 12, 350)
    frame = model.density_overlay_data(grid=grid, output="polars")
    atom = frame.filter(frame["kind"] == "atom")
    curves = frame.filter(frame["kind"] == "density")
    observed_atom = atom.filter(atom["series"] == "observed")["value"][0]
    fitted_atom = atom.filter(atom["series"] == "fitted")["value"][0]

    assert observed_atom == pytest.approx(is_zero.mean(), abs=1e-14)
    assert fitted_atom == pytest.approx(model.params()["nu"].mean(), abs=1e-14)
    for series, target in (("observed", 1 - is_zero.mean()),
                           ("fitted", 1 - model.params()["nu"].mean())):
        curve = _series(curves, series)
        assert np.trapezoid(curve["value"].to_numpy(), curve["x"].to_numpy()) == pytest.approx(
            target, abs=4e-3
        )
    assert "hconcat" in model.plot_density(grid=grid).to_dict()


def test_centile_fan_retains_rows_order_and_split_labels(gaussian_lss):
    model, _ = gaussian_lss
    frame = model.centile_fan_data(
        "x", probs=[0.1, 0.5, 0.9], split_by="group", output="polars"
    )
    assert {"index", "x", "observed", "split", "xvar", "q10", "q50", "q90"} <= set(frame.columns)
    assert frame.height == 100
    assert set(frame["split"]) == {"lower", "upper"}
    assert np.all(frame["q10"].to_numpy() < frame["q50"].to_numpy())
    assert np.all(frame["q50"].to_numpy() < frame["q90"].to_numpy())
    for group in ("lower", "upper"):
        x = frame.filter(frame["split"] == group)["x"].to_numpy()
        assert np.all(np.diff(x) >= 0)


@pytest.mark.parametrize("probs", ([0.1, 0.9], [0.1, 0.5, 0.8]))
def test_centile_fan_refuses_an_ambiguous_band_definition(gaussian_lss, probs):
    model, _ = gaussian_lss
    with pytest.raises(ValueError, match="0.5|symmetric"):
        model.centile_fan_data("x", probs=probs)


def test_deviance_profile_reconstructs_weighted_joint_deviance(gaussian_lss):
    model, _ = gaussian_lss
    frame = mb.deviance_profile(model, xvar="x", bins=7, output="polars")
    assert frame["n"].sum() == 100
    assert frame["weight"].sum() == pytest.approx(100)
    assert frame["total_deviance"].sum() == pytest.approx(
        np.sum(model.deviance_increment()), rel=1e-13
    )
    offset_only = model.deviance_profile(xvar="x", bins=7, mstop=0, output="polars")
    assert not np.allclose(frame["mean_deviance"], offset_only["mean_deviance"])


def test_deviance_profile_bin_sizes_follow_the_documented_rule(gaussian_lss):
    """The first ``n % bins`` intervals take one row more than the rest.

    R's ``deviance_profile`` reproduces this allocation explicitly, so the
    two frontends bin the same rows together. ``cut(seq_len(n), breaks = k)``,
    which R used before, puts the remainder in different intervals.
    """
    model, _ = gaussian_lss
    n = 100
    for bins in (7, 9, 10, 13):
        frame = mb.deviance_profile(model, xvar="x", bins=bins, output="polars")
        sizes = list(frame["n"])
        base, remainder = divmod(n, bins)
        assert sizes == [base + 1] * remainder + [base] * (bins - remainder)
        assert sum(sizes) == n


def test_distribution_profile_plots_compile(gaussian_lss):
    model, _ = gaussian_lss
    density = model.plot_density(grid=np.linspace(-4, 5, 80)).to_dict()
    fan = model.plot_centile_fan(
        "x", probs=[0.1, 0.5, 0.9], split_by="group"
    ).to_dict()
    deviance = model.plot_deviance_profile(xvar="x", bins=6).to_dict()

    assert density["title"] == "Observed and average fitted distribution"
    assert "hconcat" in fan
    assert deviance["title"].startswith("Joint deviance profile")
