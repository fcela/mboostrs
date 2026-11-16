"""Scoring-only artifacts and the LSS-aware portable reader (X04).

A full LSS checkpoint is a training record: it carries the response column,
the formula and control a refit needs, and enough state to unpick the model
and grow it. `save_model(..., scoring_only=True)` keeps what prediction
reads — fitted values, coefficients, learner definitions, recipes, tensor
designs, the response transform — and drops the rest: the response column
leaves the retained frame and the refit keys become `None`, so a loaded
scoring-only model predicts on new rows and on its stored fitted path, and
refuses `update`/`with_mstop`/`continue_to` with the retention error instead
of quietly holding a full checkpoint.

The R reader (`mboostrs/R/mboostrs_load.R`) loads distributional artifacts
for inspection: the per-parameter blocks decode, the `lss_family` tag names
its parameters, and a scoring-only artifact says so. Prediction stays with
the Python reader, which owns the spline and tensor machinery.
"""

import pathlib
import tempfile

import numpy as np
import pytest

import mboostrs as mb


def _rent_frame(seed=5, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    rent = np.exp(1.0 + 0.3 * x + 0.4 * rng.normal(size=n))
    return {"rent": rent, "x": x}, rent


def _save_both(fit, tmp):
    full = pathlib.Path(tmp) / "full.json"
    lean = pathlib.Path(tmp) / "scoring.json"
    mb.save_model(fit, full)
    mb.save_model(fit, lean, scoring_only=True)
    return mb.load_model(full), mb.load_model(lean)


def test_scoring_only_predicts_identically_on_new_rows():
    data, _ = _rent_frame()
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"}, data,
                         family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        full, lean = _save_both(fit, tmp)
        fresh = {"rent": np.full(3, 100.0), "x": np.array([0.5, 0.0, -1.0])}
        for name in ("mu", "sigma"):
            a = np.asarray(lean.predict(newdata=fresh, type="link")[name])
            b = np.asarray(full.predict(newdata=fresh, type="link")[name])
            np.testing.assert_allclose(a, b, rtol=1e-12)
        # The stored fitted path answers training-row prediction too, and the
        # interval inverts to the original scale exactly as the full artifact.
        np.testing.assert_allclose(np.asarray(lean.predict(type="link")["mu"]),
                                   np.asarray(fit.params(type="link")["mu"]),
                                   rtol=1e-12)
        lean_interval = lean.prediction_interval(newdata=fresh).to_dicts()
        full_interval = full.prediction_interval(newdata=fresh).to_dicts()
        np.testing.assert_allclose([r["median"] for r in lean_interval],
                                   [r["median"] for r in full_interval])


def test_scoring_only_retains_no_response_and_no_refit_state():
    data, _ = _rent_frame(n=120)
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"}, data,
                         family=mb.GaussianLSS(),
                         control=mb.boost_control(80, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        _full, lean = _save_both(fit, tmp)
        assert "rent" not in lean._data
        assert lean._formula is None and lean._control is None
        with pytest.raises(ValueError, match="stored formula/control"):
            lean.update(mstop=40)
        with pytest.raises(ValueError, match="stored formula/control"):
            lean.with_mstop([40, 40])
        # What prediction needs is all there.
        assert lean._response_transform is not None


def test_scoring_only_keeps_recipes_interactions_and_the_transform():
    """The features `predict` replays survive the strip: a `%X%` tensor and
    a `log(rent)` response transform both load and predict."""
    rng = np.random.default_rng(3)
    n = 150
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    data = {"rent": np.exp(0.3 * x + 0.2 * z + 0.2 * rng.normal(size=n)),
            "x": x, "z": z}
    fit = mb.gamboostLSS(
        {"mu": "log(rent) ~ bbs(x, df=4) %X% bbs(z, df=4)",
         "sigma": "log(rent) ~ 1"}, data, family=mb.GaussianLSS(),
        control=mb.boost_control(80, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        _full, lean = _save_both(fit, tmp)
        fresh = {"rent": np.full(2, 100.0), "x": np.array([0.5, -0.5]),
                 "z": np.array([1.0, -1.0])}
        a = np.asarray(lean.predict(newdata=fresh, type="link")["mu"])
        b = np.asarray(fit.predict(newdata=fresh, type="link")["mu"])
        np.testing.assert_allclose(a, b, rtol=1e-12)
        interval = lean.prediction_interval(newdata=fresh).to_dicts()
        # Original scale: the medians are exp of the fitted log-rent medians,
        # not the log-scale values.
        medians = [row["median"] for row in interval]
        log_medians = np.asarray(
            fit.params(newdata=fresh, type="link")["mu"])
        np.testing.assert_allclose(medians, np.exp(log_medians), rtol=1e-6)
        assert medians[0] != pytest.approx(log_medians[0], rel=0.5)


def test_a_standard_scoring_only_model_predicts_and_refuses_to_grow():
    data, _ = _rent_frame(n=120)
    fit = mb.gamboost("log(rent) ~ bbs(x, df=4)", data, family=mb.Gaussian(),
                      control=mb.boost_control(80, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        _full, lean = _save_both(fit, tmp)
        fresh = {"rent": np.full(3, 100.0), "x": np.array([0.5, 0.0, -1.0])}
        a = np.asarray(lean.predict(newdata=fresh, type="link"))
        b = np.asarray(fit.predict(newdata=fresh, type="link"))
        np.testing.assert_allclose(a, b, rtol=1e-12)
        assert np.all(np.isfinite(np.asarray(lean[:20].predict(type="link"))))
        with pytest.raises(ValueError, match="scoring-only artifact predicts"):
            # (message continues "but does not grow")
            lean.continue_to(120)


def test_scoring_only_is_smaller_than_the_full_checkpoint():
    data, _ = _rent_frame(n=300)
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"}, data,
                         family=mb.GaussianLSS(),
                         control=mb.boost_control(100, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        full, lean = _save_both(fit, tmp)
        full_path = pathlib.Path(tmp) / "full.json"
        lean_path = pathlib.Path(tmp) / "scoring.json"
        assert lean_path.stat().st_size < full_path.stat().st_size


@pytest.mark.r_parity
def test_the_r_reader_loads_distributional_parameter_blocks():
    """The R reader inspects LSS artifacts: parameter names from the
    `lss_family` tag, per-parameter mstop and offset, the learner names,
    and the scoring-only marker. It refuses to score them, pointing at the
    Python reader."""
    import subprocess

    data, _ = _rent_frame(n=120)
    fit = mb.gamboostLSS({"mu": "log(rent) ~ bbs(x, df=4)",
                          "sigma": "log(rent) ~ 1"}, data,
                         family=mb.GaussianLSS(),
                         control=mb.boost_control(60, 0.1))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "lss.json"
        mb.save_model(fit, path, scoring_only=True)
        script = (
            'suppressMessages(library(jsonlite));\n'
            'source("mboostrs/R/mboostrs_load.R");\n'
            f'm <- load_model("{path}");\n'
            'cat(family = m$family, "|", m$param_names, "|",\n'
            '    m$parameters$mu$mstop, "|", m$scoring_only, "\\n");\n'
            'p <- tryCatch(predict(m, newdata = list(x = c(0.5))),\n'
            '              error = function(e) "refused");\n'
            'cat("predict:", p, "\\n");\n'
        )
        result = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True,
            check=True)
        out = result.stdout
        assert "GaussianLSS | mu sigma | 60 | TRUE" in out
        assert "refused" in out
