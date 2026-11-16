"""Fixture-based weighted GAMBoost parity tests (no R/rpy2 required).

Loads pre-computed R reference data from tests/fixtures/gamboost_*.json
and compares offset, risk, and fitted values for every family.
"""

import json
from pathlib import Path

import mboostrs as mboost
import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests/fixtures"

FAMILY_MAP = {
    "gaussian": lambda d: mboost.Gaussian(),
    "gaussreg": lambda d: mboost.GaussReg(),
    "poisson": lambda d: mboost.Poisson(),
    "binomial": lambda d: mboost.Binomial(),
    "adaexp": lambda d: mboost.AdaExp(),
    "laplace": lambda d: mboost.Laplace(),
    "gammareg": lambda d: mboost.GammaReg(),
    "huber": lambda d: mboost.Huber(),
    "expectreg": lambda d: mboost.Expectile(tau=d["tau"]),
    "quantreg": lambda d: mboost.Quantile(tau=d["tau"]),
}


def fixture_formula(family_lower):
    return "y ~ bbs(x1, df=4) + bbs(x2, df=4)"


GAM_FIXTURES = sorted(FIXTURE_DIR.glob("gamboost_*.json"))
# Exclude LSS fixtures (gamboostlss_*)
GAM_FIXTURES = [f for f in GAM_FIXTURES if "gamboostlss" not in f.stem]


def load_fixture(path):
    with open(path) as f:
        return json.load(f)


def build_data(d):
    return {
        "y": np.asarray(d["y"], dtype=np.float64),
        "x1": np.asarray(d["x1"], dtype=np.float64),
        "x2": np.asarray(d["x2"], dtype=np.float64),
    }


def build_family(d):
    fam = d["family"].lower()
    maker = FAMILY_MAP[fam]
    return maker(d)


@pytest.mark.parametrize("fixture_path", GAM_FIXTURES, ids=lambda p: p.stem)
def test_gamboost_weighted_parity(fixture_path):
    d = load_fixture(fixture_path)
    data = build_data(d)
    w = np.asarray(d["w"], dtype=np.float64)
    family = build_family(d)
    formula = fixture_formula(d["family"].lower())

    model = mboost.gamboost(
        formula,
        data=data,
        family=family,
        weights=w,
        control=mboost.boost_control(d["mstop"], d["nu"]),
    )

    # GAMBoost uses bbs which introduces very small per-iteration
    # differences that compound (typical drift 1e-7 to 1e-5).
    fam = d["family"].lower()
    if fam == "laplace":
        offset_atol = 1e-4
        fitted_atol = 1e-4
    else:
        offset_atol = 1e-12
        fitted_atol = 1e-12
    rtol = 5e-4
    atol = 1e-12

    np.testing.assert_allclose(
        model.offset_, d["offset"], atol=offset_atol,
        err_msg=f"{fixture_path.stem}: offset mismatch",
    )

    np.testing.assert_allclose(
        model.risk_, d["risk"], rtol=rtol, atol=atol,
        err_msg=f"{fixture_path.stem}: risk mismatch",
    )

    np.testing.assert_allclose(
        model.fitted_, d["fitted"], rtol=rtol, atol=fitted_atol,
        err_msg=f"{fixture_path.stem}: fitted mismatch",
    )
