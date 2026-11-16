"""Production cross-validation uses the compact learner-spec FFI boundary.

P3.1: "Route all non-Multinomial fit/CV/stability paths through compact
learner specs, then teach the spec path the Multinomial Kronecker lift. Delete
the roughly 75-argument `fit_gamboost`/`cvrisk_gamboost` surface..."

Measured, that surface is 89 and 90 arguments. The *fit* path already routed
everything but Multinomial through the compact specs; the *CV* path routed
only binned models, so every ordinary cross-validation went through the
90-argument boundary — 90 parallel arrays grouped by learner kind, where the
spec boundary passes one self-describing object per learner.

The Multinomial Kronecker lift is now part of `cvrisk_gamboost_specs`, so every
family crosses the same production boundary. The old native symbol remains a
compatibility surface until the scheduled 0.2 removal, but no Python model
imports or calls it. These tests pin production reachability and the remaining
native signature separately.
"""

import importlib
import warnings

import numpy as np
import pytest

import mboostrs as mb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    n = 200
    frame = {"x": rng.normal(size=n), "z": rng.normal(size=n),
             "g": rng.integers(0, 4, n).astype(float)}
    frame["y"] = frame["x"] + np.sin(2 * frame["z"]) + rng.normal(scale=0.3,
                                                                 size=n)
    return frame


@pytest.fixture(scope="module")
def folds(data):
    return mb.cv_weights(len(data["y"]), type="kfold", B=4, seed=3)


def fit(formula, data, family=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mb.gamboost(formula, data, family=family or mb.Gaussian(),
                           control=mb.boost_control(40, 0.1))


@pytest.mark.parametrize("formula", [
    "y ~ bols(x) + bbs(z, df=4)",
    "y ~ bbs(x, df=4) + bbs(z, df=4)",
    "y ~ bols(x) + brandom(g, df=3)",
    "y ~ btree(x) + bbs(z, df=4)",
])
def test_production_cvrisk_uses_the_spec_boundary(formula, data, folds,
                                                  monkeypatch):
    """Every supported learner menu reaches the one compact CV boundary."""
    gamboost_module = importlib.import_module("mboostrs.gamboost")
    native = gamboost_module._cvrisk_gamboost_specs
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0])
        return native(*args, **kwargs)

    monkeypatch.setattr(gamboost_module, "_cvrisk_gamboost_specs", observed)
    model = fit(formula, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.cvrisk(folds=folds)
    assert calls and len(calls[0]) == len(model._learners_spec)
    assert np.all(np.isfinite(np.asarray(result.risk_matrix)))


def test_the_branch_sends_only_multinomial_to_the_arrays():
    """Every family crosses the spec boundary now.

    Multinomial joined once the Kronecker lift was taught to the spec CV path
    (`cvrisk_gamboost_specs` takes `multinomial_n_classes`), so the
    90-argument array CV boundary is unreachable and its deletion is P3.1's
    remaining work.
    """
    from mboostrs.gamboost import uses_spec_cv_boundary

    assert uses_spec_cv_boundary(mb.Gaussian()) is True
    assert uses_spec_cv_boundary(mb.Huber(delta=1.0)) is True
    assert uses_spec_cv_boundary(mb.Multinomial(n_classes=3)) is True


def test_production_cvrisk_has_no_array_boundary_fallback():
    """Compatibility symbols may remain until 0.2; production has one path."""
    import inspect

    gamboost_module = importlib.import_module("mboostrs.gamboost")
    source = inspect.getsource(gamboost_module.GAMBoostModel.cvrisk)
    assert "_cvrisk_gamboost_specs" in source
    assert "_cvrisk_gamboost(" not in source


def test_production_fit_has_no_array_boundary_fallback():
    """The public fit factory also reaches only the compact spec boundary."""
    import inspect

    gamboost_module = importlib.import_module("mboostrs.gamboost")
    source = inspect.getsource(gamboost_module.gamboost)
    assert "_fit_gamboost_specs" in source
    assert "_fit_gamboost(" not in source


def test_a_binned_model_still_uses_the_spec_boundary(data, folds):
    """Binning was the one case already routed through the specs, because the
    array boundary has no field for it. It must not regress."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mb.gamboost("y ~ bbs(x, df=4) + bbs(z, df=4)", data,
                            control=mb.boost_control(40, 0.1), bins=32)
        result = model.cvrisk(folds=folds)
    assert np.all(np.isfinite(np.asarray(result.risk_matrix)))


def test_the_array_surface_is_as_wide_as_recorded():
    """91 and 92 arguments, against P3.1's "roughly 75". The number is the
    argument for retiring it, so it is worth pinning: a boundary that grows
    is going the wrong way. (Raised from 89/90 once, deliberately: the E1
    step rule added `step` and `step_iterations` to every fit entry.)"""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "crates" / "mboostrs-py"
              / "src" / "lib.rs").read_text()
    widths = {}
    for name in ("fit_gamboost", "cvrisk_gamboost"):
        match = re.search(rf"fn {name}\((.*?)\n\) ->", source, re.S)
        widths[name] = sum(1 for line in match.group(1).split("\n")
                           if ":" in line)
    assert widths["fit_gamboost"] <= 91, widths
    assert widths["cvrisk_gamboost"] <= 92, widths
