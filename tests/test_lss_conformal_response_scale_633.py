"""Original-response conformal units and proper-training preprocessing (#633)."""
import numpy as np
import pytest

import mboostrs as mb
from benches.audit_lss_response_scale import fit, inputs, reference, subset


@pytest.mark.parametrize("transform", ["log", "center", "scale"])
@pytest.mark.parametrize("mstop", [None, 30])
def test_internal_external_and_analytical_quantiles_agree(transform, mstop):
    data, query, train_idx, cal_idx = inputs()
    train, held = subset(data, train_idx), subset(data, cal_idx)
    full, proper = fit(data, transform), fit(train, transform)
    expected = reference(proper, train, held, query, transform, mstop)
    internal = mb.conformal_interval(full, query, seed=9, mstop=mstop)
    external = mb.conformal_interval(proper, query, calibration=held, mstop=mstop)
    for result in (internal, external):
        for name in ("lower", "predicted", "upper"):
            np.testing.assert_allclose(getattr(result, name), expected[name], atol=1e-10, rtol=0)
        assert result.quantile == pytest.approx(expected["quantile"], abs=1e-10)
        assert result.fitted_mean is None
        assert result.n_training == 150 and result.n_calibration == 150


@pytest.mark.parametrize("transform", ["log", "center", "scale"])
def test_linear_lss_uses_the_same_original_response_contract(transform):
    data, query, train_idx, cal_idx = inputs()
    train, held = subset(data, train_idx), subset(data, cal_idx)

    def linear(rows):
        formula = f"{transform}(y) ~ x"
        return mb.glmboostLSS({"mu": formula, "sigma": formula}, rows,
                             family=mb.GaussianLSS(), control=mb.boost_control(60))

    expected = reference(linear(train), train, held, query, transform, 30)
    interval = mb.conformal_interval(linear(data), query, seed=9, mstop=30)
    np.testing.assert_allclose(interval.lower, expected["lower"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(interval.upper, expected["upper"], atol=1e-10, rtol=0)
    assert interval.fitted_mean is None


@pytest.mark.parametrize("transform", ["center", "scale"])
def test_calibration_responses_cannot_change_refit_transforms_or_predictions(monkeypatch, transform):
    from mboostrs._refit import RefitSpec

    data, query, train_idx, cal_idx = inputs()
    changed = {name: column.copy() for name, column in data.items()}
    changed["y"][cal_idx] += 100
    captured = []
    build = RefitSpec.build

    def record(spec, **kwargs):
        model = build(spec, **kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(RefitSpec, "build", record)
    first = mb.conformal_interval(fit(data, transform), query, seed=9)
    second = mb.conformal_interval(fit(changed, transform), query, seed=9)
    assert len(captured) == 2
    assert captured[0]._response_transform.params == captured[1]._response_transform.params
    expected = fit(subset(data, train_idx), transform)
    assert captured[0]._response_transform.params == expected._response_transform.params
    for name, values in captured[0].params(query).items():
        np.testing.assert_array_equal(values, captured[1].params(query)[name])
    for parameter, coefficients in captured[0].coef().items():
        for term, value in coefficients.items():
            np.testing.assert_array_equal(value, captured[1].coef()[parameter][term])
    assert first.quantile != second.quantile


def test_training_targets_remain_in_original_row_order():
    data, _, train_idx, cal_idx = inputs()
    proper = fit(subset(data, train_idx), "log")
    expected = reference(proper, subset(data, train_idx), subset(data, cal_idx), data, "log", None)
    result = mb.conformal_interval(fit(data, "log"), seed=9)
    np.testing.assert_allclose(result.lower, expected["lower"], atol=1e-10, rtol=0)


def test_response_and_predictor_recipes_are_refitted_together(monkeypatch):
    from mboostrs._refit import RefitSpec

    data, _, train_idx, cal_idx = inputs()
    data["x"][::11] = np.nan
    changed = {name: column.copy() for name, column in data.items()}
    changed["x"][cal_idx] += 100
    changed["y"][cal_idx] *= 10
    query = {"x": np.array([-.5, np.nan, .5])}

    def fitted(rows):
        return mb.gamboostLSS({"mu": "log(y) ~ bols(scale(x))", "sigma": "log(y) ~ bols(x)"},
                              rows, family=mb.GaussianLSS(), control=mb.boost_control(60),
                              na_action="impute")

    captured, build = [], RefitSpec.build

    def record(spec, **kwargs):
        model = build(spec, **kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(RefitSpec, "build", record)
    internal = mb.conformal_interval(fitted(data), query, seed=9)
    mb.conformal_interval(fitted(changed), query, seed=9)
    proper = fitted(subset(data, train_idx))
    assert captured[0]._transform_recipe == captured[1]._transform_recipe == proper._transform_recipe
    expected = mb.conformal_interval(proper, query, calibration=subset(data, cal_idx))
    np.testing.assert_allclose(internal.lower, expected.lower, atol=1e-10, rtol=0)
    np.testing.assert_allclose(internal.upper, expected.upper, atol=1e-10, rtol=0)
    for name, values in captured[0].params(query).items():
        np.testing.assert_array_equal(values, captured[1].params(query)[name])


def test_omitted_training_rows_preserve_alignment():
    data, _, _, _ = inputs()
    data["x"][0] = np.nan
    model = mb.gamboostLSS({"mu": "log(y) ~ bols(x)", "sigma": "log(y) ~ bols(x)"},
                          data, family=mb.GaussianLSS(), control=mb.boost_control(30),
                          na_action="omit")
    # Compare equal path prefixes after omission, in retained training-row order.
    result = mb.conformal_interval(model, seed=9, mstop=30)
    expected = mb.conformal_interval(fit({name: values[1:] for name, values in data.items()}, "log"),
                                     seed=9, mstop=30)
    np.testing.assert_allclose(result.lower, expected.lower, atol=1e-10, rtol=0)
    assert len(result.lower) == 299


def test_misaligned_raw_inputs_refuse():
    data, _, _, _ = inputs()
    model = fit(data, "log")
    model._raw_inputs["data"] = {name: values[:-1] for name, values in data.items()}
    with pytest.raises(NotImplementedError, match="299 input rows, 300 training rows"):
        mb.conformal_interval(model, seed=9)


def test_missing_predictor_and_unretained_raw_inputs_refuse():
    data, _, _, _ = inputs()
    model = fit(data, "log")
    with pytest.raises((ValueError, KeyError), match="x"):
        mb.conformal_interval(model, {"other": np.ones(3)}, seed=9)
    model._raw_inputs = None
    with pytest.raises(NotImplementedError, match="inputs before those transforms"):
        mb.conformal_interval(model, seed=9)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_invalid_calibration_response_refuses(value):
    data, query, train_idx, cal_idx = inputs()
    held = subset(data, cal_idx)
    held["y"][0] = value
    with pytest.raises(ValueError, match="finite observations"):
        mb.conformal_interval(fit(subset(data, train_idx), "log"), query, calibration=held)
