"""A fixed mstop in conformal_interval must fit and predict that many steps."""

import numpy as np
import pytest

import mboostrs as mb
from mboostrs import conformal


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.standard_normal(n)
    return {"y": y, "x1": x1, "x2": x2}


def _fit(data):
    return mb.gamboost("y ~ bbs(x1, df=4) + bols(x2)", data, family=mb.Gaussian())


def test_refit_control_carries_the_requested_step_count():
    fit = _fit(_data())
    assert fit._control.mstop == 100
    assert conformal._refit_control(fit, 300).mstop == 300
    assert conformal._refit_control(fit, None) is None


@pytest.mark.parametrize("method", ["split", "cv+"])
def test_fixed_mstop_refits_to_that_many_steps(method, monkeypatch):
    fit = _fit(_data())
    fitted_steps = []
    original = conformal._refit

    def spy(model, data, *, family=None, control=None, rows=None):
        out = original(model, data, family=family, control=control, rows=rows)
        fitted_steps.append(int(out._control.mstop))
        return out

    monkeypatch.setattr(conformal, "_refit", spy)
    mb.conformal_interval(fit, level=0.9, seed=1, method=method, mstop=300)
    assert fitted_steps and set(fitted_steps) == {300}


def test_cqr_fixed_mstop_refits_quantile_models_to_that_many_steps(monkeypatch):
    fit = _fit(_data())
    fitted_steps = []
    original = conformal._refit

    def spy(model, data, *, family=None, control=None, rows=None):
        out = original(model, data, family=family, control=control, rows=rows)
        fitted_steps.append(int(out._control.mstop))
        return out

    monkeypatch.setattr(conformal, "_refit", spy)
    mb.conformal_interval(fit, level=0.9, seed=1, method="cqr", mstop=300)
    assert fitted_steps and set(fitted_steps) == {300}


def test_held_out_calibration_refuses_steps_beyond_the_fitted_path():
    data = _data()
    fit = _fit(data)
    calibration = {"y": data["y"][:200], "x1": data["x1"][:200], "x2": data["x2"][:200]}
    with pytest.raises(ValueError, match="beyond the model's fitted 100 steps"):
        mb.conformal_interval(fit, level=0.9, method="split", calibration=calibration, mstop=300)
