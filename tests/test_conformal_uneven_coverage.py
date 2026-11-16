"""Split conformal warns when a constant-width band under-covers a region."""

import warnings

import numpy as np
import pytest

import mboostrs as mb


def _sample(seed, scale_fn, n=1000):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + scale_fn(x1) * rng.standard_normal(n)
    return {"y": y, "x1": x1, "x2": x2}


def _fit(data):
    return mb.gamboost("y ~ bbs(x1, df=4) + bols(x2)", data, family=mb.Gaussian())


def _uneven(record):
    return [str(w.message) for w in record if "uneven" in str(w.message)]


def test_split_warns_when_noise_grows_toward_both_tails():
    data = _sample(0, lambda x1: 0.3 + 1.5 * np.abs(x1))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(_fit(data), level=0.9, seed=1, method="split")
    messages = _uneven(record)
    assert len(messages) == 1
    assert "|x1 - median|" in messages[0]
    assert "method='cqr'" in messages[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_is_silent_under_constant_noise(seed):
    data = _sample(seed, lambda x1: np.ones_like(x1))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(_fit(data), level=0.9, seed=1, method="split")
    assert _uneven(record) == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_is_silent_under_heavy_tailed_noise(seed):
    # The calibration study's third draw: t(3) scaled to unit variance. Its
    # tails cost marginal width everywhere, so no scale third under-covers
    # and the fitted-scale view must not fire either.
    rng = np.random.default_rng(seed)
    n = 1000
    x1, x2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    y = (np.sin(2 * np.pi * x1) + 0.5 * x2
         + rng.standard_t(3, n) * np.sqrt(1 / 3))
    data = {"y": y, "x1": x1, "x2": x2}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(_fit(data), level=0.9, seed=1, method="split")
    assert _uneven(record) == []


def _views(message):
    """The flagged views of one uneven-coverage warning, in order."""
    return [part.split(" (")[0].split(" third")[0] for part in
            message.split("against a nominal 0.9: ", 1)[-1]
            .split(". The band")[0].split("; ")]


def test_split_reports_the_scale_model_view_on_heteroscedastic_noise():
    # The measured harm of #543: overall coverage looks nominal while the
    # highest true-scale third covers about 0.78. The fitted log-residual
    # scale is the proxy for that scale on the calibration rows, so the
    # warning names it -- not only a covariate third that happens to align.
    data = _sample(0, lambda x1: 0.3 + 1.5 * np.abs(x1))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(_fit(data), level=0.9, seed=1, method="split")
    messages = _uneven(record)
    assert len(messages) == 1
    assert "fitted scale" in _views(messages[0])


def test_scale_model_sees_pooled_heteroscedasticity_no_covariate_third_sees():
    # Multiplicative noise: the scale is the fitted mean, a combination of
    # the covariates, so every single covariate's thirds see only a diluted
    # share of it and stay silent, while the pooled scale model fires. This
    # is the coverage the covariate-third checks of PR #550 miss (#543).
    rng = np.random.default_rng(3)
    n = 1000
    x1, x2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    y = (np.sin(2 * np.pi * x1) + 0.5 * x2 + 1.5) * (1.0 + 0.7 * rng.standard_normal(n))
    data = {"y": y, "x1": x1, "x2": x2}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(_fit(data), level=0.9, seed=1, method="split")
    messages = _uneven(record)
    assert _views(messages[0]) == ["fitted scale"]


def test_scale_model_needs_enough_calibration_rows():
    # Below the fitted scale model's minimum rows the check skips itself; the
    # interval is still computed and nothing about the call changes.
    data = _sample(0, lambda x1: 0.3 + 1.5 * np.abs(x1), n=100)
    interval = mb.conformal_interval(_fit(data), level=0.9, seed=1, method="split")
    assert interval.lower.shape == interval.upper.shape == (100,)


def test_cqr_is_not_checked_for_uneven_coverage():
    data = _sample(0, lambda x1: 0.3 + 1.5 * np.abs(x1), n=400)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mb.conformal_interval(_fit(data), level=0.9, seed=1, method="cqr")
    assert _uneven(record) == []
