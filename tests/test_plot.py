"""Tests for plotting functions."""
import numpy as np
import pytest
from mboostrs.families import Gaussian
from mboostrs.plot import (
    plot_aic,
    plot_coef_path,
    plot_cvrisk,
    plot_diagnostics,
    plot_multinomial,
    plot_partial_effects,
    plot_reliability,
    plot_residuals,
    plot_risk,
    plot_selected,
    plot_stabsel,
    plot_varimp,
)

import mboostrs as mboost

pytestmark = pytest.mark.filterwarnings("ignore:Some 'x' values are beyond")


@pytest.fixture
def glm_model():
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 + 0.8 * x2 + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "x2": x2}
    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    return model, data


@pytest.fixture
def gam_model():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    y = 1.5 * x1 + np.sin(2 * z1) + rng.normal(scale=0.3, size=n)
    data = {"y": y, "x1": x1, "z1": z1}
    model = mboost.gamboost(
        "y ~ bbs(z1, df=4) + bols(x1)",
        data=data,
        family=Gaussian(),
        control=mboost.boost_control(50, 0.1),
    )
    return model, data


class TestPlotRisk:
    def test_plot_risk_glm(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_risk(model)
        spec = chart.to_dict()
        assert spec["encoding"]["x"]["title"] == "Iteration"
        assert spec["encoding"]["y"]["title"] == "Risk"

    def test_plot_risk_gam(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        chart = plot_risk(model)
        assert chart is not None

    def test_model_plot_risk(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = model.plot_risk()
        assert chart is not None


class TestPlotCoefPath:
    def test_coef_path_glm(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_coef_path(model)
        assert chart is not None

    def test_coef_path_which_str(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_coef_path(model, which="x1")
        assert chart is not None

    def test_coef_path_which_list(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_coef_path(model, which=["x1", "x2"])
        assert chart is not None

    def test_model_plot(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = model.plot()
        assert chart is not None

    def test_coef_path_method(self, glm_model):
        model, _ = glm_model
        paths = model.coef_path()
        assert isinstance(paths, dict)
        assert "x1" in paths or "(Intercept)" in paths
        mstop = model.mstop
        for path in paths.values():
            assert len(path) == mstop + 1


class TestPlotPartialEffects:
    def test_partial_effects_method(self, gam_model):
        model, _ = gam_model
        effects = model.partial_effects()
        assert isinstance(effects, dict)
        assert len(effects) > 0
        for x, y in effects.values():
            assert len(x) == len(y)
            assert np.all(np.isfinite(y))

    def test_partial_effects_which(self, gam_model):
        model, _ = gam_model
        selected_names = list(dict.fromkeys(model.selected))
        if len(selected_names) >= 1:
            effects = model.partial_effects(which=selected_names[0])
            assert len(effects) == 1

    def test_partial_effects_sum_matches_fitted(self, gam_model):
        model, _ = gam_model
        effects = model.partial_effects()
        total = np.zeros(model.fitted_.shape[0])
        for _x, y in effects.values():
            total = total + y
        assert total.shape == model.fitted_.shape
        np.testing.assert_allclose(total + model.offset_, model.fitted_, atol=1e-6)

    def test_partial_effects_n_points(self, gam_model):
        model, _ = gam_model
        effects = model.partial_effects(n_points=50)
        found_smooth = False
        for x, y in effects.values():
            if len(x) == 50:
                found_smooth = True
                assert np.all(np.isfinite(y))
                break
        if not found_smooth:
            pytest.skip("No smooth effects found")

    def test_partial_effects_with_string_column(self):
        rng = np.random.default_rng(7)
        n = 120
        x = rng.normal(size=n)
        y = 1.5 * x + rng.normal(scale=0.3, size=n)
        districts = np.array(["Medinipur", "Birbhum", "Murshidabad"] * 40)
        data = {"y": y, "x": x, "district": districts}
        model = mboost.gamboost(
            "y ~ bbs(x, df=4)",
            data=data,
            family=Gaussian(),
            control=mboost.boost_control(30, 0.1),
        )
        effects = model.partial_effects()
        assert "x" in effects
        for x_grid, y_effect in effects.values():
            assert len(x_grid) == len(y_effect)
            assert np.all(np.isfinite(y_effect))

    def test_plot_partial_effects(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        chart = plot_partial_effects(model)
        assert chart is not None

    def test_partial_effect_points_are_working_partial_residuals(self, gam_model):
        from mboostrs.plot import _partial_effects_with_points

        model, _ = gam_model
        effects = model.partial_effects(which="x1", n_points=25)
        _, points, _ = _partial_effects_with_points(model, effects, "x1", 25)["x1"]
        _, contribution = model.partial_effects(which="x1", n_points=None)["x1"]
        expected = contribution + model.residuals(type="working")
        np.testing.assert_allclose(points["y"].to_numpy(), expected)
        assert not np.allclose(points["y"].to_numpy(), contribution)

    def test_model_plot(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        chart = model.plot()
        assert chart is not None


class TestPlotCvrisk:
    def test_cvrisk_plot(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        cvr = model.cvrisk(B=3, seed=1)
        chart = plot_cvrisk(cvr)
        assert chart is not None

    def test_cvrisk_plot_method(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        cvr = model.cvrisk(B=3, seed=1)
        chart = cvr.plot()
        assert chart is not None


class TestPlotVarimp:
    def test_varimp_plot_glm(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_varimp(model)
        assert chart is not None

    def test_varimp_plot_gam(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        chart = plot_varimp(model)
        assert chart is not None


class TestPlotSelected:
    def test_selected_plot_glm(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_selected(model)
        assert chart is not None

    def test_selected_plot_gam(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        chart = plot_selected(model)
        assert chart is not None


class TestPlotAic:
    def test_aic_plot_glm(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        aic = model.aic()
        chart = plot_aic(aic)
        assert chart is not None

    def test_aic_plot_gam(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        aic = model.aic()
        chart = plot_aic(aic)
        assert chart is not None


class TestPlotResiduals:
    def test_residuals_glm_gaussian(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_residuals(model)
        assert chart is not None

    def test_residuals_gam_gaussian(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        chart = plot_residuals(model)
        assert chart is not None

    def test_residuals_binomial(self):
        pytest.importorskip("altair")
        rng = np.random.default_rng(1)
        n = 60
        x = rng.normal(size=n)
        y = (x > 0).astype(np.float64)
        model = mboost.glmboost("y ~ x", data={"y": y, "x": x},
                                 family=mboost.Binomial(),
                                 control=mboost.boost_control(30, 0.1))
        chart = plot_residuals(model)
        assert chart is not None

    def test_residuals_pearson(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_residuals(model, type="pearson")
        assert chart is not None


class TestPlotDiagnostics:
    def test_diagnostics_glm(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        chart = plot_diagnostics(model)
        assert chart is not None

    def test_diagnostics_gam(self, gam_model):
        pytest.importorskip("altair")
        model, _ = gam_model
        chart = plot_diagnostics(model)
        assert chart is not None


class TestPlotMultinomial:
    def test_multinomial_labels(self):
        pytest.importorskip("altair")
        model = mboost.gamboost(
            "Species ~ bbs(Petal.Length, df=3) + bbs(Petal.Width, df=3)",
            data=mboost.datasets.iris(),
            family=mboost.Multinomial(),
            control=mboost.boost_control(60, 0.1),
        )
        chart = plot_multinomial(model)
        assert chart is not None
        spec = str(chart.to_dict())
        for name in ("setosa", "versicolor", "virginica"):
            assert name in spec

    def test_multinomial_wrong_family(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        with pytest.raises(TypeError, match="Multinomial"):
            plot_multinomial(model)


class TestPlotReliability:
    def test_reliability_bins(self):
        pytest.importorskip("altair")
        rng = np.random.default_rng(0)
        n = 80
        predicted = rng.uniform(0.2, 0.9, size=n)
        observed = rng.binomial(1, predicted)
        chart = plot_reliability(predicted, observed, n_bins=5)
        datasets = chart.to_dict().get("datasets", {})
        binned = [v for v in datasets.values() if v and "predicted" in v[0]]
        assert len(binned) == 1
        assert len(binned[0]) == 5

    def test_reliability_title(self):
        pytest.importorskip("altair")
        rng = np.random.default_rng(1)
        predicted = rng.uniform(size=40)
        observed = rng.binomial(1, predicted)
        chart = plot_reliability(predicted, observed, title="Reference")
        assert chart.to_dict()["title"] == "Reference"

    def test_reliability_diagonal_reference(self):
        pytest.importorskip("altair")
        rng = np.random.default_rng(2)
        predicted = rng.uniform(0.1, 0.9, size=40)
        observed = rng.binomial(1, predicted)
        chart = plot_reliability(predicted, observed)
        layers = chart.to_dict()["layer"]
        assert len(layers) == 2
        assert layers[1]["mark"]["type"] == "line"


class TestStabselResult:
    def test_stabsel_plot(self):
        pytest.importorskip("altair")
        rng = np.random.default_rng(0)
        n = 100
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        y = 1.5 * x1 + 0.8 * x2 + rng.normal(scale=0.3, size=n)
        model = mboost.glmboost(
            "y ~ x1 + x2 + x3", data={"y": y, "x1": x1, "x2": x2, "x3": x3},
            family=Gaussian(), control=mboost.boost_control(100, 0.1),
        )
        ss = mboost.stabsel(model, cutoff=0.9, q=2, B=50, verbose=False)
        chart = plot_stabsel(ss)
        assert chart is not None


class TestThemeAndFonts:
    """The mboostrs Altair theme and the bundled DM Sans + CommitMono fonts."""

    def test_bundled_fonts_ship_with_package(self):
        from mboostrs.plot import _FONT_DIR

        expected = [
            "DMSans-400.ttf",
            "DMSans-700.ttf",
            "CommitMono-Regular.ttf",
            "CommitMono-Bold.ttf",
            "OFL-LICENSE.txt",
        ]
        for name in expected:
            p = _FONT_DIR / name
            assert p.is_file(), f"bundled font file missing: {p}"
            assert p.stat().st_size > 0

    def test_public_theme_and_palette_match_the_registered_style(self):
        import altair as alt
        from mboostrs.plot import FONT_MONO, FONT_SANS, PALETTE, theme

        # Importing mboostrs.plot must leave the theme active.
        active = getattr(getattr(alt, "theme", None), "active", None) or alt.themes.active
        assert active == "mboostrs"

        cfg = theme()["config"]
        assert cfg["font"] == FONT_SANS
        assert FONT_SANS.startswith("DM Sans")
        assert FONT_MONO.startswith("CommitMono")
        assert isinstance(PALETTE, tuple)
        assert len(PALETTE) == 10
        assert cfg["range"]["category"] == list(PALETTE)
        for axis in ("axisX", "axisY"):
            assert cfg[axis]["labelFont"] == FONT_MONO
            assert cfg[axis]["titleFont"] == FONT_SANS
        assert cfg["title"]["font"] == FONT_SANS

        # Each call returns an independent config a consumer can adapt before
        # registering it for a project-specific Altair theme.
        cfg["axisX"]["tickCount"] = 2
        assert theme()["config"]["axisX"]["tickCount"] == 8

    def test_png_export_renders_with_bundled_fonts(self, glm_model, tmp_path):
        # End-to-end: vl-convert static export with the registered bundled fonts.
        model, _ = glm_model
        chart = plot_risk(model).properties(title="Risk")
        out = tmp_path / "risk.png"
        chart.save(str(out))
        assert out.is_file()
        # A rendered 600x300 chart is far larger than an empty canvas.
        assert out.stat().st_size > 10_000

    def test_bars_are_square_not_rounded(self):
        """Rounded bar ends blur where a value falls; corners stay mitred."""
        from mboostrs.plot import theme

        bar = theme()["config"]["bar"]
        assert bar["cornerRadius"] == 0
        assert bar["strokeJoin"] == "miter"

    def test_no_chart_sets_a_bar_corner_radius(self):
        """The theme is only authoritative if no call site overrides it."""
        from pathlib import Path

        import mboostrs.plot as plot_module

        # Resolve from the imported module, not the working directory, so the
        # test does not depend on pytest being invoked from the repo root.
        source = Path(plot_module.__file__).read_text(encoding="utf-8")
        assert "cornerRadius" not in source.replace('"cornerRadius": 0', "")

    def test_rug_is_pinned_to_the_panel_bottom(self):
        """A rug with no y encoding gets centred vertically by Vega-Lite,
        which drew it across the middle of the panel instead of at the axis."""
        from mboostrs.plot import _rug

        spec = _rug(np.linspace(0.0, 1.0, 40), 300).to_dict()
        assert spec["encoding"]["y"] == {"value": 297}

    def test_rug_thins_large_samples_but_keeps_the_distribution(self):
        from mboostrs.plot import RUG_MAX_TICKS, _rug

        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        thinned = _rug(x, 300).data["x"].to_numpy()
        assert thinned.size == RUG_MAX_TICKS
        # Quantile-spaced thinning preserves the marginal distribution.
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            assert abs(np.quantile(x, q) - np.quantile(thinned, q)) < 0.05
        # Small samples are left alone, and non-finite values dropped.
        assert _rug(np.arange(50.0), 300).data["x"].to_numpy().size == 50
        assert _rug(np.array([1.0, np.nan, 2.0]), 300).data["x"].to_numpy().size == 2

    def test_varimp_axis_is_formatted_as_a_percentage(self, glm_model):
        """varimp() returns shares summing to 1, so the axis must not label a
        0-1 fraction as '(%)'."""
        from mboostrs.plot import plot_varimp

        model, _ = glm_model
        total = sum(model.varimp()["reduction"].values())
        assert total == pytest.approx(1.0)
        enc = plot_varimp(model).to_dict()["encoding"]["x"]
        assert enc["axis"]["format"] == ".0%"
        assert "%" not in enc["title"]


class TestConfintGridAlignment:
    """The effect and its interval must be evaluated on the same data.

    `_plot_confint_gam` used to take the effect from an `n_points` grid and
    the bounds from `confint`'s default (one value per training row), then
    index the bounds with the grid's sort order.
    """

    def test_confint_band_matches_the_effect_grid(self, gam_model):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_confint

        model, data = gam_model
        n_points = len(data["y"]) + 29  # deliberately unequal to the sample
        chart = plot_confint(model, parm="z1", n_points=n_points, B=8, seed=0)

        layers = chart.to_dict()["layer"]
        band = next(ly for ly in layers if ly["mark"]["type"] == "area")
        rows = band["data"]["values"] if "values" in band.get("data", {}) else None
        if rows is not None:
            assert len(rows) == n_points
            xs = [r["x"] for r in rows]
            assert xs == sorted(xs)
            assert all(r["lower"] <= r["upper"] for r in rows)

    def test_confint_accepts_newdata_of_a_different_length(self, gam_model):
        """Replicates that fail or skip a learner pad to the evaluation
        length; padding to the training length made the stack ragged."""
        from mboostrs.plot import _effect_grid_frame

        model, data = gam_model
        n_points = len(data["y"]) - 37
        grid = _effect_grid_frame(model, n_points)

        ci = model.confint(B=8, seed=0, newdata=grid)
        assert ci["lower"], "expected at least one selected learner"
        for name, lower in ci["lower"].items():
            assert len(lower) == n_points
            assert len(ci["upper"][name]) == n_points

    def test_confint_all_learners_renders(self, gam_model):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_confint

        model, _ = gam_model
        plot_confint(model, B=8, seed=0).to_dict()


class TestCoefPathRejectsMultiColumnLearners:
    def test_spline_learner_is_refused_with_an_actionable_message(self, gam_model):
        """A bbs learner has no single coefficient to trace; the numpy error
        this used to raise named neither the cause nor the alternative."""
        model, _ = gam_model
        with pytest.raises(ValueError, match="one coefficient per base learner"):
            plot_coef_path(model)

    def test_scalar_coefficient_model_still_plots(self, glm_model):
        pytest.importorskip("altair")
        model, _ = glm_model
        plot_coef_path(model).to_dict()


@pytest.fixture
def lss_model():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.uniform(-2, 2, size=n)
    y = 1.0 + 0.8 * x + rng.normal(scale=np.exp(0.2 * x), size=n)
    data = {"y": y, "x": x}
    model = mboost.gamboostLSS(
        "y ~ bols(x)",
        data=data,
        families=mboost.GaussianLSS(),
        control=mboost.boost_control(80, 0.1),
    )
    return model, data


class TestResidualsOnDistributionalModels:
    """`residuals()` returns one vector per distribution parameter, so the
    single-vector path built a 0-d object array and failed inside float()."""

    def test_lss_residuals_plot_defaults_to_the_first_parameter(self, lss_model):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_residuals

        model, _ = lss_model
        available = sorted(model.residuals())
        chart = plot_residuals(model).to_dict()
        assert chart["layer"][0]["encoding"]["y"]["title"] == "Residuals"

        explicit = plot_residuals(model, parameter=available[0]).to_dict()
        assert explicit["layer"][0]["encoding"] == chart["layer"][0]["encoding"]

    def test_unknown_parameter_lists_what_is_available(self, lss_model):
        from mboostrs.plot import plot_residuals

        model, _ = lss_model
        with pytest.raises(ValueError, match="no residuals for parameter"):
            plot_residuals(model, parameter="not_a_parameter")

    def test_parameter_is_refused_for_single_path_models(self, glm_model):
        from mboostrs.plot import plot_residuals

        model, _ = glm_model
        with pytest.raises(ValueError, match="distributional"):
            plot_residuals(model, parameter="mu")

    def test_diagnostics_dashboard_points_at_the_lss_equivalents(self, lss_model):
        from mboostrs.plot import plot_diagnostics

        model, _ = lss_model
        with pytest.raises(TypeError, match="plot_lss_selected"):
            plot_diagnostics(model)


class TestWormPlotThinning:
    """A worm plots one point per observation along a monotone curve, so a
    large sample renders as a solid band in a multi-megabyte SVG."""

    def test_thinning_preserves_the_endpoints_and_extremes(self):
        import polars as pl

        from mboostrs.plot import WORM_MAX_POINTS, _thin_worm_panel

        rng = np.random.default_rng(3)
        n = 5000
        expected = np.sort(rng.normal(size=n))
        deviation = np.sin(expected) + rng.normal(scale=0.01, size=n)
        panel = pl.DataFrame({"expected": expected, "deviation": deviation})

        thinned = _thin_worm_panel(panel)
        assert thinned.height <= WORM_MAX_POINTS + 4
        assert thinned["expected"].min() == panel["expected"].min()
        assert thinned["expected"].max() == panel["expected"].max()
        assert thinned["deviation"].min() == panel["deviation"].min()
        assert thinned["deviation"].max() == panel["deviation"].max()

    def test_small_panels_are_left_alone(self):
        import polars as pl

        from mboostrs.plot import _thin_worm_panel

        panel = pl.DataFrame({"expected": np.arange(50.0),
                              "deviation": np.arange(50.0)})
        assert _thin_worm_panel(panel).height == 50


class TestConfintPlotReusesAComputedInterval:
    """CF-03: a caller who computed `confint` once -- to inspect the
    replicate tally, or with settings the helper does not expose -- could
    not hand the interval to the plot without the helper bootstrapping the
    same replicates again."""

    def test_a_computed_interval_plots_without_a_second_bootstrap(
            self, gam_model, monkeypatch):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_confint

        model, _ = gam_model
        computed = model.confint(parm="z1", level=0.9, B=8, seed=0)
        # A second bootstrap would be the defect this test pins.
        monkeypatch.setattr(
            model, "confint",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(
                "plot_confint bootstrapped again although interval= was given")))
        chart = plot_confint(model, parm="z1", interval=computed)
        layers = chart.to_dict()["layer"]
        band = next(ly for ly in layers if ly["mark"]["type"] == "area")
        # Altair may hold the data at the chart level; inline is the small case.
        rows = band.get("data", {}).get("values")
        if rows is not None:
            assert len(rows) == len(computed["lower"]["z1"])
            assert all(r["lower"] <= r["upper"] for r in rows)
        # The band carries the level the interval was computed at.
        assert "90%" in str(chart.to_dict()["title"])

    def test_a_glm_computed_interval_plots_without_a_second_bootstrap(
            self, glm_model, monkeypatch):
        pytest.importorskip("altair")
        from mboostrs.plot import plot_confint

        model, _ = glm_model
        computed = model.confint(level=0.9, B=8, seed=0)
        monkeypatch.setattr(
            model, "confint",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(
                "plot_confint bootstrapped again although interval= was given")))
        chart = plot_confint(model, interval=computed)
        assert "90%" in str(chart.to_dict()["title"])

    def test_a_bad_interval_is_refused_with_the_expected_shape(self, gam_model):
        from mboostrs.plot import plot_confint

        model, _ = gam_model
        with pytest.raises(ValueError, match="model.confint"):
            plot_confint(model, parm="z1", interval=[0.1, 0.2])
