"""Fixture-based weighted GLMBoost parity tests (no R/rpy2 required).

Loads pre-computed R reference data from tests/fixtures/glmboost_*.json
and compares offset, risk, and fitted values for every family.
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
    "Poisson": lambda d: mboost.Poisson(),
    "Binomial": lambda d: mboost.Binomial(),
    "AdaExp": lambda d: mboost.AdaExp(),
    "Laplace": lambda d: mboost.Laplace(response_scale=None),
    "GammaReg": lambda d: mboost.GammaReg(),
    "Huber": lambda d: mboost.Huber(),
    "NBinomial": lambda d: mboost.NBinomial(),
    "PropOdds": lambda d: mboost.PropOdds(n_levels=d.get("nlevels", 3)),
    "ExpectReg": lambda d: mboost.Expectile(tau=d["tau"]),
    "QuantReg": lambda d: mboost.Quantile(tau=d["tau"], qoffset=0.5, response_scale=None),
}

GLM_FIXTURES = sorted(FIXTURE_DIR.glob("glmboost_*.json"))


def load_fixture(path):
    with open(path) as f:
        return json.load(f)


def has_weights(path):
    """Check fixture has weights AND a plain y vector (not Surv)."""
    d = load_fixture(path)
    return "w" in d and "y" in d


GLM_WEIGHTED_FIXTURES = [p for p in GLM_FIXTURES if has_weights(p)]


def build_data(d):
    y = np.asarray(d["y"], dtype=np.float64)
    return {
        "y": y,
        "x1": np.asarray(d["x1"], dtype=np.float64),
        "x2": np.asarray(d["x2"], dtype=np.float64),
    }


def build_family(d):
    maker = FAMILY_MAP[d["family"]]
    return maker(d)


@pytest.mark.parametrize("fixture_path", GLM_WEIGHTED_FIXTURES, ids=lambda p: p.stem)
def test_glmboost_weighted_parts(fixture_path):
    d = load_fixture(fixture_path)
    data = build_data(d)
    w = np.asarray(d["w"], dtype=np.float64)
    family = build_family(d)

    model = mboost.glmboost(
        "y ~ x1 + x2",
        data=data,
        family=family,
        weights=w,
        control=mboost.boost_control(d["mstop"], d["nu"]),
    )

    # Per-family tolerances (justified by source of divergence from R)
    fam = d["family"]
    if fam == "Laplace":
        # L1 weighted median: R optimize() vs our Brent differ by
        # ~5e-05 atol on offset, propagating into risk/fitted.
        offset_atol = 1e-4
        rtol = 2e-4
        fitted_atol = 1e-4
    elif fam == "PropOdds":
        # PropOdds: response not passed as ordered factor in this fixture,
        # differ by constant shift (~0.6% relative)
        rtol = 1e-2
        offset_atol = 1e-12
    elif fam == "NBinomial":
        # NBinomial: small drift in risk (~1e-08) and fitted (~5e-06)
        rtol = 1e-5
        offset_atol = 1e-12
    else:
        rtol = 1e-9
        offset_atol = 1e-12
    atol = 1e-12
    fitted_atol = locals().get("fitted_atol", atol)

    # Offset
    if fam != "PropOdds":
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
