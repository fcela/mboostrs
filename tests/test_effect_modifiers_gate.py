"""effect_modifiers: the R-learner gate and covariates written as transforms."""

import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.causal import AIPWResult
from mboostrs.effect_modifiers import effect_modifiers


def _spec(covariates):
    return mb.CausalSpec(
        treatment="a", outcome="y", covariates=tuple(covariates), estimand="ate",
        target_population="simulated", unit="row", time_zero="assignment",
        follow_up="one period",
        adjustment_rationale="the covariates precede treatment and cause both")


def _draw(n, seed, modifier):
    """Five covariates, a confounded treatment, a nonlinear outcome and an effect
    that is constant (1) or rises with x3 by 1.5 per unit."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    e = 1 / (1 + np.exp(-(0.6 * x[:, 0] - 0.4 * x[:, 1])))
    a = rng.binomial(1, e).astype(float)
    tau = 1.0 + (1.5 * x[:, 2] if modifier else 0.0)
    y = 2 * x[:, 0] + x[:, 1] ** 2 + np.sin(x[:, 3]) + tau * a + rng.normal(size=n)
    data = {"a": a, "y": y}
    data.update({f"x{j + 1}": x[:, j] for j in range(5)})
    return data, _spec(f"x{j + 1}" for j in range(5))


# Seeds chosen from a 40-draw screen at n = 2000 with linear nuisances
# (smooth=False): the score regressions alone claim a modifier on these null draws.
NULL_SEEDS = [8, 28]


@pytest.mark.parametrize("seed", NULL_SEEDS)
def test_gate_removes_the_claim_on_a_no_effect_draw(seed):
    data, spec = _draw(2000, seed, modifier=False)
    result = mb.aipw(data, spec, seed=seed, smooth=False)
    ungated = effect_modifiers(result, data, gate=False)
    assert ungated.selected(), "the draw is chosen so the score tests alone claim a modifier"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gated = effect_modifiers(result, data)
    gate = gated.diagnostics["gate"]
    assert gate["run"] and not gate["passed"]
    assert gate["gain"] <= gate["gain_se"]
    assert gated.selected() == ()
    assert all(not r["decision"] for r in gated.records)
    assert any(r["test_decision"] for r in gated.records)
    assert "No modifier is claimed" in gated.message
    assert "does not beat a constant effect" in gated.message
    assert gated.message in gated.describe()


def test_gate_keeps_the_modifier_on_an_effect_modifier_draw():
    data, spec = _draw(2000, 0, modifier=True)
    result = mb.aipw(data, spec, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gated = effect_modifiers(result, data)
    gate = gated.diagnostics["gate"]
    assert gate["run"] and gate["passed"]
    assert gate["gain"] > gate["gain_se"] > 0
    assert "x3" in gated.selected()
    assert gated.message == ""
    assert [r["decision"] for r in gated.records] == [r["test_decision"] for r in gated.records]


def test_gate_false_runs_no_gate_and_records_that():
    data, spec = _draw(600, 1, modifier=True)
    result = mb.aipw(data, spec, seed=1, tune=False, mstop=60)
    out = effect_modifiers(result, data, gate=False)
    assert out.diagnostics["gate"] == {"run": False}
    assert out.message == ""
    assert "No R-learner gate was run" in out.describe()


def test_gate_names_covariates_the_effect_model_cannot_use():
    data, spec = _draw(600, 1, modifier=True)
    result = mb.aipw(data, spec, seed=1, tune=False, mstop=60)
    data = {**data, "extra": np.arange(600.0)}
    with pytest.raises(ValueError, match=r"gate=False"):
        effect_modifiers(result, data, covariates=["extra"])
    out = effect_modifiers(result, data, covariates=["extra"], gate=False)
    assert [r["covariate"] for r in out.records] == ["extra"]


# Covariates written as transforms.

def _positive_result(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    income = np.exp(rng.normal(10, 0.6, n))
    z = rng.normal(size=n)
    e = np.clip(1 / (1 + np.exp(-(0.4 * z))), 0.1, 0.9)
    tau = 800 * (np.log(income) - 10) + 300
    scores = tau + rng.normal(0, 3000, n)
    a = rng.binomial(1, e).astype(float)
    data = pl.DataFrame({"income": income, "z": z, "a": a})
    result = AIPWResult(
        estimate=float(scores.mean()), standard_error=None, interval=None, level=0.95,
        spec=_spec(("income", "z")), treatment=a, outcome=np.zeros(n), propensity=e,
        mu0=np.zeros(n), mu1=np.zeros(n), scores=scores,
        influence=scores - scores.mean(), fold_id=np.zeros(n, dtype=int),
        diagnostics={}, nuisance={})
    return result, data


def test_expression_covariate_is_named_as_written_and_computed_by_the_formula_transform():
    result, data = _positive_result()
    out = effect_modifiers(result, data, covariates=["log(income)", "z"], gate=False)
    assert [r["covariate"] for r in out.records] == ["log(income)", "z"]
    assert "log(income)" in out.describe()
    assert {row["covariate"] for row in out.joint} == {"log(income)", "z"}
    direct = effect_modifiers(
        result, data.with_columns(pl.col("income").log().alias("logincome")),
        covariates=["logincome", "z"], gate=False)
    a, b = out.records[0], direct.records[0]
    assert a["blp_slope"] == pytest.approx(b["blp_slope"], rel=1e-10)
    assert a["blp_p"] == pytest.approx(b["blp_p"], rel=1e-10)
    # The effect is linear in log income, so the transformed covariate is the stronger test.
    raw = effect_modifiers(result, data, covariates=["income", "z"], gate=False)
    assert a["blp_p"] < raw.records[0]["blp_p"]


def test_expression_projection_replays_the_fitted_transform_on_new_rows():
    result, data = _positive_result()
    out = effect_modifiers(result, data, covariates=["scale(income)", "log(income)"],
                           gate=False)
    proj = out.projection("all")
    new = pl.DataFrame({"income": [np.exp(9.0), np.exp(10.0), np.exp(11.0)]})
    values = proj.predict(new)
    assert values.shape == (3,)
    # scale(income) uses the fitting rows' mean and sd: one new row alone is not constant-scaled.
    assert np.all(np.isfinite(values))
    assert values[0] < values[2]


def test_unsupported_transform_names_the_supported_ones():
    result, data = _positive_result()
    with pytest.raises(ValueError) as error:
        effect_modifiers(result, data, covariates=["tanh(income)"], gate=False)
    message = str(error.value)
    assert "'tanh(income)'" in message
    for name in ("log", "log1p", "sqrt", "exp", "scale", "center", "ecdf", "pmin", "pmax"):
        assert name in message


def test_missing_column_error_names_the_argument():
    result, data = _positive_result()
    with pytest.raises(ValueError, match=r"covariates names 'nope'"):
        effect_modifiers(result, data, covariates=["nope"], gate=False)
    with pytest.raises(ValueError, match=r"'log\(nope\)'.*not in the data"):
        effect_modifiers(result, data, covariates=["log(nope)"], gate=False)


def test_gate_runs_on_an_expression_through_its_source_column():
    data, spec = _draw(800, 4, modifier=True)
    data["x3"] = np.exp(0.5 * data["x3"])
    result = mb.aipw(data, spec, seed=4, tune=False, mstop=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = effect_modifiers(result, data, covariates=["log(x3)", "x1"])
    gate = out.diagnostics["gate"]
    assert gate["run"] and gate["modifiers"] == ("x3", "x1")
    assert [r["covariate"] for r in out.records] == ["log(x3)", "x1"]
