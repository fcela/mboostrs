"""Fixture-based GLMBoost parity tests (no R/rpy2 required).

Loads pre-computed R reference data from tests/fixtures/glmboost_*.json
(weighted fixtures are in test_weighted_fixtures.py).
"""

import json
from pathlib import Path

import mboostrs as mboost
import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests/fixtures"

FAMILY_MAP = {
    "Gaussian": lambda d: mboost.Gaussian(),
    "GaussReg": lambda d: mboost.GaussReg(),
    "Binomial": lambda d: mboost.Binomial(),
    "AdaExp": lambda d: mboost.AdaExp(),
    "Poisson": lambda d: mboost.Poisson(),
    "Laplace": lambda d: mboost.Laplace(response_scale=None),
    "Huber": lambda d: mboost.Huber(),
    "GammaReg": lambda d: mboost.GammaReg(),
    "NBinomial": lambda d: mboost.NBinomial(),
    "QuantReg": lambda d: mboost.Quantile(tau=d["tau"], qoffset=0.5, response_scale=None),
    "ExpectReg": lambda d: mboost.Expectile(tau=d["tau"]),
    "Loglog": lambda d: mboost.Loglog(),
    "Weibull": lambda d: mboost.Weibull(),
    "CoxPH": lambda d: mboost.CoxPH(),
    "PropOdds": lambda d: mboost.PropOdds(n_levels=d.get("nlevels", 3)),
}

GLM_FIXTURES = sorted(
    p for p in FIXTURE_DIR.glob("glmboost_*.json")
    if "weighted" not in p.stem
)


def load_fixture(path):
    with open(path) as f:
        return json.load(f)


def build_data_and_model(d):
    data = {}
    family = FAMILY_MAP[d["family"]](d)
    rtol = d.get("tol_risk", 1e-9)
    ftol = d.get("tol_fitted", rtol)
    otol = d.get("tol_offset", 1e-12)
    atol_fitted = d.get("atol_fitted", 1e-12)

    if d.get("response_type") == "Surv":
        data = {
            "time": np.asarray(d["time"], dtype=np.float64),
            "event": np.asarray(d["event"], dtype=np.float64),
            "x1": np.asarray(d["x1"], dtype=np.float64),
            "x2": np.asarray(d["x2"], dtype=np.float64),
        }
    else:
        y = np.asarray(d["y"], dtype=np.float64)
        data["y"] = y
        data["x1"] = np.asarray(d["x1"], dtype=np.float64)
        data["x2"] = np.asarray(d["x2"], dtype=np.float64)

    kw = dict(
        data=data,
        family=family,
        control=mboost.boost_control(d["mstop"], d["nu"]),
    )
    if "w" in d:
        kw["weights"] = np.asarray(d["w"], dtype=np.float64)

    formula = d.get("formula", "y ~ x1 + x2")
    return data, family, kw, formula, rtol, ftol, otol, atol_fitted


@pytest.mark.parametrize("fixture_path", GLM_FIXTURES, ids=lambda p: p.stem)
def test_glmboost_default_fixtures(fixture_path):
    d = load_fixture(fixture_path)
    fam_name = d["family"]

    if fam_name == "CoxPH" and d["scenario"] == "coxph_cvrisk":
        _test_coxph_cvrisk_core(d)
        return

    _, _, kw, formula, rtol, ftol, otol, atol_fitted = build_data_and_model(d)
    model = mboost.glmboost(formula, **kw)

    skip_risk = d.get("skip_risk", False)
    skip_fitted = d.get("skip_fitted", False)

    if not skip_risk:
        np.testing.assert_allclose(
            model.risk_, d["risk"], rtol=rtol,
            err_msg=f"{fixture_path.stem}: risk mismatch",
        )
    if not skip_fitted:
        np.testing.assert_allclose(
            model.fitted_, d["fitted"], rtol=ftol, atol=atol_fitted,
            err_msg=f"{fixture_path.stem}: fitted mismatch",
        )
    if "offset" in d and d["offset"] is not None and fam_name != "PropOdds":
        np.testing.assert_allclose(
            model.offset_, d["offset"], rtol=otol, atol=otol,
            err_msg=f"{fixture_path.stem}: offset mismatch",
        )

    if d.get("has_selected"):
        assert list(model._selected) == d["selected"], (
            f"{fixture_path.stem}: selected mismatch"
        )

    if d.get("has_coef") and "coef" in d and fam_name != "PropOdds":
        coef_arr = [c for name in ("x1", "x2") for c in (list(model.coef().get(name, [])) if isinstance(model.coef().get(name), list) else [model.coef().get(name, 0)])]
        expected = d["coef"]
        if len(coef_arr) == len(expected):
            np.testing.assert_allclose(
                np.array(coef_arr), np.array(expected), rtol=rtol,
                err_msg=f"{fixture_path.stem}: coef mismatch",
            )

    if d.get("has_response") and "response" in d:
        resp = model.predict(type="response")
        rtol_r = d.get("tol_response", rtol)
        atol_r = d.get("atol_response", 1e-12)
        n_levels = d.get("nlevels", 3)
        expected_resp = np.asarray(d["response"], dtype=np.float64)
        if resp.ndim == 1:
            resp = resp.reshape(-1, 1)
        if expected_resp.ndim == 1:
            expected_resp = expected_resp.reshape(d["n"], n_levels)
        np.testing.assert_allclose(
            resp, expected_resp, rtol=rtol_r, atol=atol_r,
            err_msg=f"{fixture_path.stem}: response mismatch",
        )


def _test_coxph_cvrisk_core(d):
    data = {
        "time": np.asarray(d["time"], dtype=np.float64),
        "status": np.asarray(d["status"], dtype=np.float64),
        "x1": np.asarray(d["x1"], dtype=np.float64),
        "x2": np.asarray(d["x2"], dtype=np.float64),
    }
    model = mboost.glmboost(
        "Surv(time, status) ~ x1 + x2",
        data=data,
        family=mboost.CoxPH(),
        control=mboost.boost_control(d["mstop"], d["nu"]),
    )
    n, B = int(d["folds_nrow"]), int(d["folds_ncol"])
    folds_matrix = np.asarray(d["folds"], dtype=np.float64).reshape((n, B), order="F")
    cvr = model.cvrisk(folds=folds_matrix, parallel=False)
    py_oob = np.asarray(cvr.risk_matrix)
    expected_oob = np.asarray(d["oob"], dtype=np.float64).reshape(
        (int(d["oob_nrow"]), int(d["oob_ncol"])), order="F"
    )
    assert py_oob.shape == expected_oob.shape, (
        f"OOB shape mismatch: {py_oob.shape} vs {expected_oob.shape}"
    )
    np.testing.assert_allclose(
        py_oob, expected_oob, rtol=1e-6, atol=1e-8,
        err_msg="CoxPH cvrisk OOB risk matrix does not match R",
    )
    assert cvr.optimal_mstop == d["optimal_mstop"], (
        f"mstop mismatch: {cvr.optimal_mstop} vs {d['optimal_mstop']}"
    )


def test_gaussreg_matches_gaussian():
    """GaussReg should produce identical results to Gaussian (cross-check)."""
    d = load_fixture(
        FIXTURE_DIR / "glmboost_gaussian_default.json"
    )
    model_gr = mboost.glmboost(
        "y ~ x1 + x2",
        data={
            "y": np.asarray(d["y"], dtype=np.float64),
            "x1": np.asarray(d["x1"], dtype=np.float64),
            "x2": np.asarray(d["x2"], dtype=np.float64),
        },
        family=mboost.GaussReg(),
        control=mboost.boost_control(d["mstop"], d["nu"]),
    )
    model_g = mboost.glmboost(
        "y ~ x1 + x2",
        data={
            "y": np.asarray(d["y"], dtype=np.float64),
            "x1": np.asarray(d["x1"], dtype=np.float64),
            "x2": np.asarray(d["x2"], dtype=np.float64),
        },
        family=mboost.Gaussian(),
        control=mboost.boost_control(d["mstop"], d["nu"]),
    )
    np.testing.assert_allclose(
        model_gr.fitted_, model_g.fitted_, rtol=1e-12,
        err_msg="GaussReg and Gaussian fitted must be identical",
    )
    np.testing.assert_allclose(
        model_gr.risk_, model_g.risk_, rtol=1e-12,
        err_msg="GaussReg and Gaussian risk must be identical",
    )


def test_propodds_link_is_constant_shift():
    """PropOdds link differs from R by a constant (identifiability)."""
    d = load_fixture(FIXTURE_DIR / "glmboost_propodds_default.json")
    _, _, kw, formula, _, _, _, _ = build_data_and_model(d)
    model = mboost.glmboost(formula, **kw)
    diff = model.fitted_ - np.asarray(d["fitted"], dtype=np.float64)
    assert np.std(diff) < 1e-4, (
        f"PropOdds link should differ from R by a constant; std={np.std(diff)}"
    )


def test_propodds_risk_decreases():
    """PropOdds risk should be non-increasing (short scenario)."""
    d = load_fixture(FIXTURE_DIR / "glmboost_propodds_short.json")
    _, _, kw, formula, _, _, _, _ = build_data_and_model(d)
    model = mboost.glmboost(formula, **kw)
    diffs = np.diff(model.risk_)
    assert np.all(diffs <= 1e-10), "PropOdds risk should be non-increasing"


def test_propodds_risk_decreases_4class():
    """PropOdds risk should be non-increasing (4-class scenario)."""
    d = load_fixture(FIXTURE_DIR / "glmboost_propodds_4class.json")
    _, _, kw, formula, _, _, _, _ = build_data_and_model(d)
    model = mboost.glmboost(formula, **kw)
    diffs = np.diff(model.risk_)
    assert np.all(diffs <= 1e-10), "PropOdds risk (4 class) should be non-increasing"
