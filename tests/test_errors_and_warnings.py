"""A5: typed errors and fit-warnings surfaced on the model.

Rust-originated errors surface as ``MboostrsError`` subclasses that also
inherit the matching builtin, carrying ``code``/``subject``/``details``/
``rust_variant`` attributes so a package wrapper can dispatch on the failure
kind without parsing message text. Formula typos suggest the intended
parameter. Warnings a fit raises are recorded on the model under
``fit_warnings``, visible to ``pytest.warns`` and re-emittable later,
without changing what the caller inherits.
"""

import importlib
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._errors import (
    MboostrsError,
    MboostrsKeyError,
    MboostrsRuntimeError,
    MboostrsValueError,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n = 60
    x = np.linspace(0, 1, n)
    return {"x": x, "y": 1.0 + 2.0 * x + rng.normal(scale=0.2, size=n)}


# ── exception hierarchy ────────────────────────────────────────────


def test_exception_hierarchy_matches_builtins():
    assert issubclass(MboostrsError, Exception)
    assert issubclass(MboostrsValueError, MboostrsError)
    assert issubclass(MboostrsValueError, ValueError)
    assert issubclass(MboostrsRuntimeError, MboostrsError)
    assert issubclass(MboostrsRuntimeError, RuntimeError)
    assert issubclass(MboostrsKeyError, MboostrsError)
    assert issubclass(MboostrsKeyError, KeyError)


def test_default_attributes_are_none():
    err = MboostrsError("message")
    assert err.code is None
    assert err.subject is None
    assert err.details is None
    assert err.rust_variant is None


def test_constructor_takes_keyword_attributes():
    err = MboostrsValueError(
        "boom", code="control", subject="nu", details="1.5", rust_variant="Control"
    )
    assert err.code == "control"
    assert err.subject == "nu"
    assert err.details == "1.5"
    assert err.rust_variant == "Control"


def test_errors_are_exported():
    for name in ("MboostrsError", "MboostrsValueError", "MboostrsRuntimeError", "MboostrsKeyError"):
        assert getattr(mb, name) is not None


# ── Rust error variants keep their machine-readable fields ─────────


def test_bad_response_is_typed(data):
    y_neg = data["y"] - 2.0
    with pytest.raises(MboostrsValueError) as excinfo:
        mb.gamboost(
            "y ~ bbs(x)", data={"x": data["x"], "y": y_neg}, family=mb.Poisson()
        )
    err = excinfo.value
    assert err.code == "bad_response"
    assert err.subject == "Poisson"
    assert err.rust_variant == "BadResponse"
    assert "family 'Poisson'" in str(err)
    assert isinstance(err, ValueError)


def test_control_variant_is_typed(data):
    with pytest.raises(MboostrsValueError) as excinfo:
        mb.glmboost(
            "y ~ x", data=data, control=mb.boost_control(mstop=10, nu=2.0)
        )
    err = excinfo.value
    assert err.code == "control"
    assert err.rust_variant == "Control"


def test_family_diagnostics_response_check_is_typed():
    from mboostrs._rust import family_diagnostics

    with pytest.raises(MboostrsValueError) as excinfo:
        family_diagnostics(
            "poisson", {}, np.array([-1.0, 2.0]), np.zeros(2), np.ones(2))
    assert excinfo.value.code == "bad_response"
    assert excinfo.value.rust_variant == "BadResponse"


# ── formula typo suggestions ───────────────────────────────────────


def test_unknown_parameter_suggests_close_one(data):
    with pytest.raises(mb.MboostrsValueError) as excinfo:
        mb.gamboost("y ~ bbs(x, dff=4)", data=data)
    err = excinfo.value
    assert err.code == "unknown_parameter"
    assert err.subject == "bbs"
    assert err.details == "dff"
    assert "did you mean 'df'?" in str(err)


def test_unknown_parameter_without_match_plain(data):
    with pytest.raises(mb.MboostrsValueError) as excinfo:
        mb.gamboost("y ~ bbs(x, bogus=1)", data=data)
    err = excinfo.value
    assert err.code == "unknown_parameter"
    assert err.subject == "bbs"
    assert err.details == "bogus"
    assert "did you mean" not in str(err)


# ── missing columns raise KeyError-shaped errors ───────────────────


def test_missing_column_is_key_error(data):
    with pytest.raises(mb.MboostrsKeyError) as excinfo:
        mb.glmboost("y ~ zzz", data=data)
    err = excinfo.value
    assert err.code == "missing_column"
    assert err.subject == "zzz"
    assert isinstance(err, KeyError)


# ── fit warnings are recorded on the model ─────────────────────────


def test_fit_warnings_empty_for_quiet_fit(data):
    fit = mb.gamboost("y ~ bbs(x)", data=data)
    assert fit.fit_warnings == ()


def test_bkernel_warning_captured_and_reemitted(data, monkeypatch):
    gm = importlib.import_module("mboostrs.gamboost")
    monkeypatch.setattr(gm, "_BKERNEL_WARN_CELLS", 1)
    with pytest.warns(ResourceWarning) as recorded:
        fit = mb.gamboost("y ~ bkernel(x)", data=data)
    assert len(recorded) == 1
    assert len(fit.fit_warnings) == 1
    warning = fit.fit_warnings[0]
    assert isinstance(warning, warnings.WarningMessage)
    assert warning.category is ResourceWarning
    assert "dense" in str(warning.message)
    # the stored copy is the same warning the caller saw
    assert str(warning.message) == str(recorded[0].message)


def test_bkernel_warning_reemits_to_later_client(data, monkeypatch):
    gm = importlib.import_module("mboostrs.gamboost")
    monkeypatch.setattr(gm, "_BKERNEL_WARN_CELLS", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        fit = mb.gamboost("y ~ bkernel(x)", data=data)
    with pytest.warns(ResourceWarning):
        for warning in fit.fit_warnings:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno,
                source=warning.line,
            )


def _warning_fit():
    """A decorated fit that raises one warning from a fixed line."""
    from mboostrs._model_api import capture_fit_warnings

    class Model:
        pass

    @capture_fit_warnings
    def fit():
        warnings.warn("fit-time probe", UserWarning, stacklevel=1)
        return Model()

    return fit


def test_fit_warnings_honour_module_filters():
    fit = _warning_fit()
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", module=__name__)
        model = fit()
    assert shown == []
    assert model._fit_warnings == ()


def test_fit_warnings_keep_once_per_location_deduplication():
    fit = _warning_fit()
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("default")
        models = [fit() for _ in range(3)]
    assert len(shown) == 1
    assert [len(m._fit_warnings) for m in models] == [1, 0, 0]


def test_fit_warnings_error_filter_raises_inside_the_fit():
    reached = []
    from mboostrs._model_api import capture_fit_warnings

    @capture_fit_warnings
    def fit():
        warnings.warn("fit-time probe", UserWarning, stacklevel=1)
        reached.append(True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            fit()
    assert reached == []


def test_fit_restores_the_previous_showwarning():
    fit = _warning_fit()
    before = warnings.showwarning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fit()
    assert warnings.showwarning is before


def test_lss_fit_warnings_attribute(data):
    fit = mb.glmboostLSS("y ~ x", data=data, family=mb.GaussianLSS())
    assert fit.fit_warnings == ()


# ── fit warnings are not persisted through the artifact ────────────


def test_fit_warnings_not_saved_in_artifact(data, monkeypatch, tmp_path):
    gm = importlib.import_module("mboostrs.gamboost")
    monkeypatch.setattr(gm, "_BKERNEL_WARN_CELLS", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        fit = mb.gamboost("y ~ bkernel(x)", data=data)
    assert len(fit.fit_warnings) == 1
    path = tmp_path / "warned.json"
    mb.save_model(fit, str(path))
    loaded = mb.load_model(str(path))
    assert loaded.fit_warnings == ()