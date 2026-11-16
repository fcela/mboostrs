"""Regression tests for #226/#220: stopping diagnostics.

- ``cvrisk`` warns (attribute + UserWarning) when the optimum exhausts the grid.
- ``stopping_report()`` describes where the in-sample path stopped and flags
  the #220 signature (early best step, rising risk after) and a material
  same-family GLM shortfall.
"""
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs._model_api import _stopping_report_for


@pytest.fixture()
def linear_data():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.5, size=n)
    return pl.DataFrame({"x": x, "y": y})


def test_stopping_report_keys(linear_data):
    fit = mb.glmboost("y ~ bols(x)", data=linear_data, family=mb.Gaussian(),
                      control=mb.boost_control(mstop=100, nu=0.1))
    rep = fit.stopping_report()
    assert set(rep) >= {"mstop", "best_step", "risk_at_best", "risk_at_final",
                        "risk_rose_after_best", "flag", "glm_gap_nats",
                        "glm_gap_material"}
    assert rep["mstop"] == 100
    assert isinstance(rep["best_step"], int)
    assert rep["risk_at_best"] <= rep["risk_at_final"] + 1e-12


def test_stopping_report_220_signature_deterministic():
    """The #220 pattern — best step far behind the cap, risk rising after —
    flags deterministically via a stub risk path."""

    class Stub:
        mstop = 500
        _family = mb.Gaussian()

        @property
        def risk_(self):
            # falls to step 30, then climbs and never recovers
            down = np.linspace(10.0, 1.0, 31)
            up = np.linspace(1.0, 5.0, 470)
            return np.concatenate([down, up])

        def logLik(self):  # not needed; same_family_gap is bypassed below
            raise NotImplementedError

    rep = _stopping_report_for(Stub())
    assert rep["best_step"] == 30
    assert rep["risk_rose_after_best"] is True
    assert rep["flag"] is not None and "smaller nu" in rep["flag"]


def test_stopping_report_cap_flag():
    """Risk still falling at the cap gets the mild flag, not the #220 one."""

    class Stub:
        mstop = 100
        _family = mb.Gaussian()

        @property
        def risk_(self):
            return np.linspace(10.0, 1.0, 101)

        def logLik(self):
            raise NotImplementedError

    rep = _stopping_report_for(Stub())
    assert rep["best_step"] == 100
    assert rep["risk_rose_after_best"] is False
    assert "mstop cap" in rep["flag"]


def test_stopping_report_glm_gap_material():
    """Weighted GammaReg at nu=0.1 (#220): the same-family GLM gap is material."""
    rng = np.random.default_rng(2)
    n = 400
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    mu = 5000 * np.exp(0.15 * x0 - 0.1 * x1)
    y = rng.gamma(shape=20, scale=mu / 20)
    w = np.ones(n)
    w[:8] = rng.uniform(500, 3000, size=8)
    data = pl.DataFrame({"x0": x0, "x1": x1, "y": y})
    fit = mb.glmboost("y ~ bols(x0) + bols(x1)", data=data, family=mb.GammaReg(),
                      control=mb.boost_control(mstop=500, nu=0.1),
                      weights=w, weight_policy="raw")
    rep = fit.stopping_report()
    assert rep["glm_gap_material"] is True
    assert rep["glm_gap_nats"] > 1.0
    assert "short of" in rep["flag"]


def test_cvrisk_boundary_warning_fires(linear_data):
    """Optimum at the grid cap: attribute set, warning emitted, repr flagged."""
    fit = mb.glmboost("y ~ bols(x)", data=linear_data, family=mb.Gaussian(),
                      control=mb.boost_control(mstop=20, nu=0.1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cvr = fit.cvrisk(type="kfold", B=3, seed=1)
    assert cvr.optimal_mstop == 20  # risk still falling: the cap is "optimal"
    assert cvr.boundary_warning is not None
    assert "search range was exhausted" in cvr.boundary_warning
    assert any("search range was exhausted" in str(x.message) for x in w)
    assert "boundary" in repr(cvr)
    assert cvr.to_dict()["boundary_warning"] == cvr.boundary_warning


def test_cvrisk_no_boundary_warning_interior(linear_data):
    fit = mb.glmboost("y ~ bols(x)", data=linear_data, family=mb.Gaussian(),
                      control=mb.boost_control(mstop=200, nu=0.1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cvr = fit.cvrisk(type="kfold", B=3, seed=1)
    assert cvr.optimal_mstop < 0.95 * 200
    assert cvr.boundary_warning is None
    assert not any("search range was exhausted" in str(x.message) for x in w)
