"""A missing optional extra must not break attribute introspection (#308)."""

import builtins
import inspect
import sys

import pytest

import mboostrs


@pytest.fixture
def no_altair(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "altair" or name.startswith("altair."):
            raise ImportError("No module named 'altair'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "mboostrs.plot", raising=False)
    monkeypatch.delitem(mboostrs.__dict__, "plot", raising=False)
    yield


def test_hasattr_is_false_without_altair(no_altair):
    assert hasattr(mboostrs, "plot") is False


def test_getmembers_works_without_altair(no_altair):
    names = [n for n, _ in inspect.getmembers(mboostrs, inspect.isfunction)]
    assert "gamboost" in names


def test_from_import_still_raises_import_error_with_hint(no_altair):
    with pytest.raises(ImportError, match="mboostrs\\[plot\\]"):
        from mboostrs import plot  # noqa: F401


def test_attribute_access_keeps_the_install_hint(no_altair):
    with pytest.raises(AttributeError, match="mboostrs\\[plot\\]"):
        mboostrs.plot  # noqa: B018
