"""`risk` works as an attribute and as a call (#357).

`fit.risk` is the risk path as a read-only array. `fit.risk()` used to raise
numpy's "'numpy.ndarray' object is not callable", which names neither the
attribute nor the fix. The call now returns the same values as a fresh
writable ndarray, and the attribute keeps behaving as the array it was.
"""

import pickle
import warnings

import numpy as np
import pytest

import mboostrs as mb

N = 120


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    x, z = rng.normal(size=N), rng.normal(size=N)
    return {"x": x, "z": z, "y": 2.0 * x - 0.5 * z + rng.normal(size=N)}


@pytest.fixture(scope="module")
def control():
    return mb.boost_control(25, 0.1)


@pytest.fixture(scope="module")
def fits(data, control):
    return {
        "gamboost": mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data,
                                control=control),
        "glmboost": mb.glmboost("y ~ x + z", data, control=control),
        "gamboostLSS": mb.gamboostLSS("y ~ bbs(x, df=4)", data,
                                      families=mb.GaussianLSS(),
                                      control=control),
    }


CLASSES = ["gamboost", "glmboost", "gamboostLSS"]


@pytest.mark.parametrize("kind", CLASSES)
def test_risk_as_an_attribute_is_the_risk_path(fits, kind):
    fit = fits[kind]
    risk = fit.risk
    assert isinstance(risk, np.ndarray)
    assert risk.ndim == 1
    assert np.array_equal(np.asarray(risk), np.asarray(fit.risk_))
    assert risk[0] == fit.risk_[0]
    assert len(risk) == len(fit.risk_)
    assert float(np.min(risk)) == float(np.min(fit.risk_))


@pytest.mark.parametrize("kind", CLASSES)
def test_risk_as_a_call_returns_the_same_values(fits, kind):
    fit = fits[kind]
    called = fit.risk()
    assert isinstance(called, np.ndarray)
    assert np.array_equal(called, np.asarray(fit.risk))
    assert np.array_equal(called, np.asarray(fit.risk_))


@pytest.mark.parametrize("kind", CLASSES)
def test_the_attribute_and_the_call_compare_equal(fits, kind):
    fit = fits[kind]
    assert np.array_equal(fit.risk, fit.risk())
    assert bool(np.all(fit.risk == fit.risk()))


@pytest.mark.parametrize("kind", CLASSES)
def test_pickling_the_attribute_round_trips_to_a_plain_array(fits, kind):
    fit = fits[kind]
    restored = pickle.loads(pickle.dumps(fit.risk))
    assert type(restored) is np.ndarray
    assert np.array_equal(restored, np.asarray(fit.risk_))
    assert restored.flags.writeable


@pytest.mark.parametrize("kind", CLASSES)
def test_asarray_of_the_attribute_is_a_plain_read_only_array(fits, kind):
    fit = fits[kind]
    plain = np.asarray(fit.risk)
    assert type(plain) is np.ndarray
    assert not plain.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        plain[0] = 0.0


@pytest.mark.parametrize("kind", CLASSES)
def test_the_attribute_is_still_read_only(fits, kind):
    with pytest.raises(ValueError, match="read-only"):
        fits[kind].risk[0] = 0.0


@pytest.mark.parametrize("kind", CLASSES)
def test_the_call_returns_a_writable_copy(fits, kind):
    fit = fits[kind]
    called = fit.risk()
    assert called.flags.writeable
    called[0] = 1.0e9
    assert fit.risk[0] != 1.0e9
    assert np.array_equal(fit.risk, fit.risk_)


@pytest.mark.parametrize("kind", CLASSES)
def test_arithmetic_indexing_and_repr_are_plain_arrays(fits, kind):
    fit = fits[kind]
    assert type(fit.risk * 2) is np.ndarray
    assert type(fit.risk[:3]) is np.ndarray
    assert type(np.sum(fit.risk)) is np.float64
    assert type(fit.risk[0]) is np.float64
    assert repr(fit.risk) == repr(np.asarray(fit.risk_))
    assert str(fit.risk) == str(np.asarray(fit.risk_))


@pytest.mark.parametrize("kind", ["gamboost", "glmboost"])
def test_the_call_takes_no_arguments(fits, kind):
    with pytest.raises(TypeError, match="takes no arguments"):
        fits[kind].risk(0)


def test_the_lss_call_is_silent_and_parameter_risk_still_works(fits):
    fit = fits["gamboostLSS"]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        called = fit.risk()
    assert not [w for w in seen if issubclass(w.category, FutureWarning)]
    assert np.array_equal(called, np.asarray(fit.risk_))

    per_parameter = fit.parameter_risk()
    assert set(per_parameter) == set(fit.param_names)
    with pytest.raises(NotImplementedError, match="parameter_risk"):
        fit.risk(parameter=fit.param_names[0])
