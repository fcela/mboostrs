"""Stopping criteria keep their units, target sample and selected iteration."""

import numpy as np
import pytest

import mboostrs as mb


def fit(kind="glm", family=None, *, data=None, mstop=40, weights=None, risk="inbag"):
    if data is None:
        rng = np.random.default_rng(91)
        x = rng.normal(size=100)
        y = .3 * x + rng.normal(size=100)
        if isinstance(family, mb.Poisson):
            y = rng.poisson(np.exp(.2 * x)).astype(float)
        elif isinstance(family, mb.Binomial):
            y = (y > 0).astype(float)
        data = {"x": x, "y": y}
    fn = mb.glmboost if kind == "glm" else mb.gamboost
    formula = "y ~ x" if kind == "glm" else "y ~ bols(x)"
    return fn(formula, data=data, family=family, weights=weights,
              control=mb.boost_control(mstop, risk=risk))


@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_corrected_gaussian_formula_and_units(kind):
    model = fit(kind)
    result = model.aic()
    n = len(model.weights)
    expected = np.log(model.risk_[1:] / n) + (1 + result.df / n) / (1 - (result.df + 2) / n)
    np.testing.assert_allclose(result.aic, expected, rtol=1e-12)
    assert result.criterion == "mboost_corrected"
    assert result.normalization == "per_observation"
    assert result.penalty is None
    assert result.df_convention == "conditional_hat_trace"
    assert result.to_frame(output="records")[0]["criterion"] == result.criterion


@pytest.mark.parametrize("kind", ["glm", "gam"])
@pytest.mark.parametrize("family", [mb.Binomial(), mb.Poisson()])
def test_classical_likelihood_formula(kind, family):
    model = fit(kind, family)
    result = model.aic(method="classical", k=3)
    np.testing.assert_allclose(result.aic, 2 * model.risk_[1:] + 3 * result.df)
    assert result.normalization == "total"
    assert result.penalty == 3
    with pytest.raises(NotImplementedError, match="only implemented for Gaussian"):
        model.aic(method="corrected")


@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_classical_gaussian_and_ignored_penalties_are_refused(kind):
    model = fit(kind)
    with pytest.raises(NotImplementedError, match="classical AIC"):
        model.aic(method="classical")
    with pytest.raises(ValueError, match="fixed penalty"):
        model.aic(k=np.log(100))
    with pytest.raises(ValueError, match="aic method"):
        model.aic(method="typo")


def test_no_risk_fit_cannot_produce_an_information_criterion():
    with pytest.raises(ValueError, match="inbag"):
        fit(risk="none").aic()


@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_gmdl_uses_upstreams_uncentered_unweighted_signal_term(kind):
    rng = np.random.default_rng(8)
    data = {"x": rng.normal(size=100), "y": 10 + rng.normal(size=100)}
    weights = np.linspace(1, 3, 100)
    model = fit(kind, data=data, weights=weights)
    result = model.aic(method="gmdl")
    residual = model.risk_[1:]
    df = result.df
    sumw = model.weights.sum()
    sigma2 = residual / (sumw - df)
    expected = np.log(sigma2) + df / sumw * np.log((np.sum(data["y"] ** 2) - residual) / (df * sigma2))
    np.testing.assert_allclose(result.aic, expected, rtol=1e-12)


def test_gaussian_table_names_the_actual_criterion_and_evaluated_stop():
    model = fit(mstop=500)
    row = mb.gaic_table(model)[0]
    expected = model.aic()
    assert "GAIC(2)" not in row
    assert row["mboost_corrected"] == expected.aic[expected.mstop - 1]
    assert row["mstop"] == row["mboost_corrected_mstop"] == expected.mstop
    assert row["risk"] == model[:expected.mstop].risk_[-1]
    assert row["logLik"] == pytest.approx(model[:expected.mstop].logLik())
    with pytest.raises(ValueError, match="fixed penalty"):
        mb.gaic_table(model, k=(2, "logn"))


def test_lss_update_count_is_labeled_as_a_heuristic():
    x = np.linspace(-1, 1, 100)
    model = mb.glmboostLSS("y ~ x", data={"x": x, "y": x + np.sin(20 * x)},
                           control=mb.boost_control(20))
    row = mb.gaic_table(model)[0]
    assert row["criterion"] == "lss_update_count_heuristic"
    assert row["df_convention"] == "update_count_plus_one"
    assert row["heuristic_GAIC(2)"] == 2 * model.risk_[-1] + 2 * row["df"]
    assert model._logLik_df is None
    assert model._logLik_update_count == row["df"]


def test_mixed_criterion_definitions_cannot_be_ranked():
    rng = np.random.default_rng(91)
    data = {"x": rng.normal(size=100), "y": rng.normal(size=100)}
    standard = fit(data=data)
    lss = mb.glmboostLSS("y ~ x", data=data, control=mb.boost_control(40))
    with pytest.raises(ValueError, match="different criterion definitions"):
        mb.gaic_table(standard, lss)


def test_likelihood_profile_does_not_invert_the_corrected_aic_formula():
    model = fit()
    rows = mb.profile_mstop(model, grid=[1, 10, 40], metric="logLik")
    for row in rows:
        assert row["logLik"] == pytest.approx(model[:row["mstop"]].logLik())


def test_criterion_plot_uses_the_same_iteration_as_its_optimum():
    pytest.importorskip("altair")
    result = fit().aic()
    chart = result.plot().to_dict()
    values = next(rows for rows in chart["datasets"].values() if "aic" in rows[0])
    assert values[0]["iteration"] == 1
    assert values[-1]["iteration"] == len(result.aic)
    assert values[result.mstop - 1]["iteration"] == result.mstop
    assert "per_observation" in chart["title"]


def test_likelihood_penalties_really_change_the_criterion():
    model = fit(family=mb.Poisson())
    row = mb.gaic_table(model, k=(2, "logn"))[0]
    assert row["GAIC(2)"] != row["GAIC(logn)"]
    for penalty, key in [(2, "GAIC(2)"), (np.log(100), "GAIC(logn)")]:
        result = model.aic(method="classical", k=penalty)
        assert row[key] == result.aic[result.mstop - 1]
        assert row[key + "_mstop"] == result.mstop


def test_zero_and_nan_criteria_sort_after_negative_values(monkeypatch):
    from mboostrs import model_selection

    models = [fit() for _ in range(3)]
    values = dict(zip(map(id, models), [0., np.nan, -1.], strict=True))
    original = model_selection._criterion_row

    def criterion(model, k):
        row = original(model, k)
        return {**row, "value": values[id(model)]}

    monkeypatch.setattr(model_selection, "_criterion_row", criterion)
    table = mb.gaic_table(*models, labels=["zero", "nan", "negative"])
    assert [row["label"] for row in table] == ["negative", "zero", "nan"]


@pytest.mark.parametrize("change", ["length", "response", "weights", "order"])
def test_comparison_requires_matching_observations(change):
    rng = np.random.default_rng(7)
    data = {"x": rng.normal(size=100), "y": rng.normal(size=100)}
    a = fit(data=data)
    other = {k: v.copy() for k, v in data.items()}
    weights = None
    if change == "length":
        other = {k: v[:-1] for k, v in other.items()}
    elif change == "response":
        other["y"] += .1
    elif change == "weights":
        weights = np.linspace(1, 2, 100)
    else:
        other = {k: v[::-1] for k, v in other.items()}
    b = fit(data=other, weights=weights)
    with pytest.raises(ValueError, match="same observed responses"):
        mb.gaic_table(a, b)
