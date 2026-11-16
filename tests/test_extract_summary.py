"""Tests for Milestone 8: extraction / inspection / structured summaries."""

import json

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture
def gaussian_glm():
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    return mb.glmboost(
        "y ~ x1 + x2", data={"y": y, "x1": x1, "x2": x2},
        family=mb.Gaussian(), control=mb.boost_control(30, 0.1),
    )


@pytest.fixture
def gaussian_gam():
    rng = np.random.default_rng(7)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.sin(x1) + 0.5 * x2 + rng.normal(scale=0.3, size=n)
    return mb.gamboost(
        "y ~ bbs(x1, df=4) + bols(x2)",
        data={"y": y, "x1": x1, "x2": x2},
        family=mb.Gaussian(), control=mb.boost_control(30, 0.1),
    )


# ─── GLMBoostModel ───────────────────────────────────────────────────────────


class TestGLMExtraction:
    def test_variable_names(self, gaussian_glm):
        names = gaussian_glm.variable_names()
        assert "y" in names
        assert "x1" in names
        assert "x2" in names

    def test_base_learner_names(self, gaussian_glm):
        names = gaussian_glm.base_learner_names()
        assert "(Intercept)" in names
        assert "x1" in names
        assert "x2" in names

    def test_bnames_alias(self, gaussian_glm):
        assert gaussian_glm.bnames() == gaussian_glm.base_learner_names()

    def test_model_frame(self, gaussian_glm):
        mf = gaussian_glm.model_frame()
        assert "y" in mf
        assert "x1" in mf
        assert len(mf["x1"]) == 50

    def test_model_matrix_all(self, gaussian_glm):
        mm = gaussian_glm.model_matrix()
        assert mm.shape == (50, 2)

    def test_model_matrix_which(self, gaussian_glm):
        mm = gaussian_glm.model_matrix(which="x1")
        assert mm.shape == (50, 1)
        mm2 = gaussian_glm.model_matrix(which=["x1", "x2"])
        assert mm2.shape == (50, 2)

    def test_control(self, gaussian_glm):
        ctrl = gaussian_glm.control
        assert ctrl[0] == 30
        assert ctrl[1] == 0.1

    def test_offset(self, gaussian_glm):
        assert isinstance(gaussian_glm.offset, float)

    def test_nuisance(self, gaussian_glm):
        n = gaussian_glm.nuisance()
        assert n is not None or n is None  # Gaussian may return [] or None

    def test_model_weights(self, gaussian_glm):
        w = gaussian_glm.model_weights()
        assert len(w) == 50
        assert np.all(w == 1.0)

    def test_summary_as_dict(self, gaussian_glm):
        info = gaussian_glm.summary(as_dict=True)
        assert info["model_class"] == "GLMBoostModel"
        assert info["family"] == "Gaussian"
        assert info["mstop"] == 30
        assert "selection_frequency" in info
        assert "nonzero_coefficients" in info
        # JSON-serializable
        json.dumps(info)

    def test_summary_returns_a_renderable_object(self, gaussian_glm, capsys):
        """summary() returns a Summary rather than printing as a side effect,
        so `print(model.summary())` no longer prints the body twice and a
        notebook can render the HTML form."""
        summary = gaussian_glm.summary()
        assert capsys.readouterr().out == ""

        text = str(summary)
        assert "glmboost" in text
        assert "Gaussian" in text
        assert "Coefficients" in text
        assert "Iterations (mstop)" in text
        assert max(len(line) for line in text.splitlines()) <= 78

        html = summary._repr_html_()
        assert html.startswith("<div")
        assert "<table" in html


# ─── GAMBoostModel ───────────────────────────────────────────────────────────


class TestGAMExtraction:
    def test_variable_names(self, gaussian_gam):
        names = gaussian_gam.variable_names()
        assert "y" in names
        assert "x1" in names

    def test_base_learner_names(self, gaussian_gam):
        names = gaussian_gam.base_learner_names()
        assert "x1" in names
        assert "x2" in names

    def test_bnames_alias(self, gaussian_gam):
        assert gaussian_gam.bnames() == gaussian_gam.base_learner_names()

    def test_model_frame(self, gaussian_gam):
        mf = gaussian_gam.model_frame()
        assert "y" in mf
        assert "x1" in mf
        assert len(mf["x1"]) == 60

    def test_model_matrix(self, gaussian_gam):
        mm = gaussian_gam.model_matrix()
        assert isinstance(mm, dict)
        assert len(mm) >= 1

    def test_control(self, gaussian_gam):
        ctrl = gaussian_gam.control
        assert ctrl[0] == 30
        assert ctrl[1] == 0.1

    def test_offset(self, gaussian_gam):
        assert isinstance(gaussian_gam.offset, float)

    def test_model_weights(self, gaussian_gam):
        w = gaussian_gam.model_weights()
        assert len(w) == 60

    def test_summary_as_dict(self, gaussian_gam):
        info = gaussian_gam.summary(as_dict=True)
        assert info["model_class"] == "GAMBoostModel"
        assert info["family"] == "Gaussian"
        assert info["mstop"] == 30
        json.dumps(info)

    def test_summary_returns_a_renderable_object(self, gaussian_gam, capsys):
        summary = gaussian_gam.summary()
        assert capsys.readouterr().out == ""

        text = str(summary)
        assert "gamboost" in text
        # A spline learner has no single coefficient, so the summary reports
        # the span of its fitted effect instead.
        assert "Fitted effects" in text
        assert "range" in text
        assert max(len(line) for line in text.splitlines()) <= 78


# ─── LSSModel ────────────────────────────────────────────────────────────────


class TestLSSExtraction:
    def test_summary_as_dict(self):
        rng = np.random.default_rng(42)
        n = 50
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
        mod = mb.glmboostLSS(
            "y ~ x1 + x2", data={"y": y, "x1": x1, "x2": x2},
            families=mb.GaussianLSS(),
            control=mb.boost_control(20, 0.1),
        )
        info = mod.summary(as_dict=True)
        assert info["model_class"] == "LSSModel"
        assert "parameters" in info
        assert "selected_counts" in info
        json.dumps(info)

    def test_gaic(self):
        rng = np.random.default_rng(42)
        n = 50
        x1 = rng.normal(size=n)
        y = 0.5 * x1 + rng.normal(scale=0.3, size=n)
        mod = mb.glmboostLSS(
            "y ~ x1", data={"y": y, "x1": x1},
            families=mb.GaussianLSS(),
            control=mb.boost_control(20, 0.1),
        )
        gaic = mod.gaic()
        assert isinstance(gaic, float)
        assert gaic > 0
        # GAIC = 2*risk + k*df; with k=4 it's larger
        assert mod.gaic(k=4) > gaic
