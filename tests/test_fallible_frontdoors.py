"""Regression tests for typed, panic-free public validation paths."""

import numpy as np
import pytest
from mboostrs.cvrisk import cv_weights
from mboostrs.stabsel import _D_rc, _maxQ, _minD, _stabsel_parameters, _um_const

import mboostrs as mb


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 0}, "observation|positive"),
        ({"n": 5, "B": 0}, "fold|replicate|positive"),
        ({"n": 5, "type": "kfold", "B": 6}, "k.*n"),
        ({"n": 5, "type": "subsampling", "prob": float("nan")}, "finite"),
        ({"n": 5, "type": "subsampling", "prob": 1.0}, "prob"),
        ({"n": 5, "weights": [1, 1, -1, 1, 1]}, "non-negative"),
    ],
)
def test_cv_weights_invalid_inputs_raise_value_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cv_weights(**kwargs)


def test_cvrisk_rejects_fold_without_oob_observations():
    x = np.linspace(-1.0, 1.0, 20)
    y = 0.5 * x
    model = mb.glmboost(
        "y ~ x",
        data={"y": y, "x": x},
        control=mb.boost_control(5, 0.1),
    )
    with pytest.raises(ValueError, match="OOB"):
        model.cvrisk(folds=np.ones((len(y), 2)))


@pytest.mark.parametrize("method", ["cyclic", "noncyclic"])
def test_lss_invalid_offset_is_value_error_not_panic(method):
    x = np.linspace(-1.0, 1.0, 20)
    y = 0.5 * x + 1.0
    with pytest.raises(ValueError, match="offset.*finite|finite.*offset"):
        mb.glmboostLSS(
            "y ~ x",
            data={"y": y, "x": x},
            families=mb.GaussianLSS(),
            control=mb.boost_control(5, 0.1),
            offset=[float("nan"), 0.0],
            method=method,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: _maxQ(0, 50),
        lambda: _maxQ(10, 1),
        lambda: _um_const(float("nan"), 50, 0.1),
        lambda: _D_rc(2.0, 100, 10, -0.5),
        lambda: _minD(5, 5, 0.8, 2),
        lambda: _stabsel_parameters(p=0, cutoff=0.8, q=1),
        lambda: _stabsel_parameters(p=5, cutoff=0.4, q=1),
    ],
)
def test_stabsel_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


# --- non-finite input diagnostics -------------------------------------------
#
# Fitting rejects non-finite predictors, responses, and weights on every route.
# These tests pin the *diagnostic*: a modeler debugging a large frame needs the
# kind, the count, and a row to look at, not just "contains NaN or Inf".


def _finite_frame(n=40):
    x = np.linspace(-1.0, 1.0, n)
    return {"x": x, "z": np.linspace(0.0, 1.0, n), "y": 0.5 * x}


@pytest.mark.parametrize(
    ("formula", "family", "column"),
    [
        ("y ~ x + z", mb.Gaussian(), "x"),          # packed scalar
        ("y ~ x + z", mb.Gaussian(), "y"),
        ("y ~ bols(x) + bbs(z, df=4)", mb.Gaussian(), "z"),  # gaussian gram
        ("y ~ bols(x) + bbs(z, df=4)", mb.Gaussian(), "y"),
    ],
)
def test_nonfinite_column_message_names_kind_count_and_row(formula, family, column):
    data = _finite_frame()
    data[column] = data[column].copy()
    data[column][3] = np.nan
    data[column][9] = np.inf
    engine = mb.gamboost if "bbs" in formula else mb.glmboost

    with pytest.raises(ValueError) as excinfo:
        engine(formula, data=data, family=family,
               control=mb.boost_control(10, 0.1))

    message = str(excinfo.value)
    assert f"'{column}'" in message
    assert "1 NaN" in message
    assert "1 Inf" in message
    assert "row 3" in message          # first offending row, not the last


def test_nonfinite_weights_message_names_kind_count_and_row():
    data = _finite_frame()
    weights = np.ones(len(data["x"]))
    weights[5] = np.nan

    with pytest.raises(ValueError) as excinfo:
        mb.glmboost("y ~ x + z", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(10, 0.1), weights=weights)

    message = str(excinfo.value)
    assert "weights" in message
    assert "1 NaN" in message
    assert "row 5" in message


def test_nonfinite_newdata_is_rejected_at_prediction_time():
    """A clean fit must not silently produce NaN predictions."""
    data = _finite_frame()
    fit = mb.glmboost("y ~ x + z", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(10, 0.1))

    newdata = _finite_frame()
    newdata["x"] = newdata["x"].copy()
    newdata["x"][2] = np.nan

    with pytest.raises(ValueError, match="row 2"):
        fit.predict(newdata=newdata)
