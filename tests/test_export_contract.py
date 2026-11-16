"""The export convention every public result record follows.

`to_dict(*, include_arrays=False)`: the flag is keyword-only and off by
default, so a positional `to_dict(True)` is a TypeError rather than a silent
change of payload, and a bare `to_dict()` is always accepted by `json.dumps`.
"""

import importlib
import inspect
import json
import pkgutil

import numpy as np
import pytest

import mboostrs as mb


def _classes_with_to_dict():
    seen = {}
    for info in pkgutil.iter_modules(mb.__path__):
        if info.name.startswith("_rust"):
            continue
        try:
            module = importlib.import_module(f"mboostrs.{info.name}")
        except ImportError:        # optional extras such as `plot`
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or name.startswith("_"):
                continue
            if "to_dict" in cls.__dict__:
                seen[f"{module.__name__}.{name}"] = cls
    return seen


CLASSES = _classes_with_to_dict()


def test_the_scan_finds_the_result_classes():
    assert len(CLASSES) >= 20
    assert "mboostrs.selection.Deselection" in CLASSES


@pytest.mark.parametrize("qualname", sorted(CLASSES))
def test_to_dict_takes_include_arrays_as_a_keyword_only_flag(qualname):
    parameters = inspect.signature(CLASSES[qualname].to_dict).parameters
    assert "include_arrays" in parameters, qualname
    flag = parameters["include_arrays"]
    assert flag.kind is inspect.Parameter.KEYWORD_ONLY, qualname
    assert flag.default is False, qualname


def _records():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (3, 200))
    data = {"y": np.sin(2 * x[0]) + rng.normal(scale=0.3, size=200),
            "x0": x[0], "x1": x[1], "x2": x[2]}
    fit = mb.gamboost("y ~ bbs(x0) + bbs(x1) + bbs(x2)", data=data,
                      control=mb.boost_control(100, 0.1))
    folds = mb.cv_weights(200, "kfold", 3, seed=1)
    return {
        "cvrisk": fit.cvrisk(folds=folds),
        "execution_report": fit.execution_report(),
        "gam_refit": mb.gam_refit(fit),
        "deselection": mb.deselect(fit, folds=folds),
    }


def test_default_records_are_json_safe_and_arrays_come_only_on_request():
    for name, record in _records().items():
        text = json.dumps(record.to_dict())
        assert text, name
        with pytest.raises(TypeError):
            record.to_dict(True)
        json.dumps(record.to_dict(include_arrays=True), default=float)
