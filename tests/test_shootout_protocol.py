"""The study's partition, score and cache rules are enforceable."""
import numpy as np
import polars as pl
import pytest

from benches._jobs import _local_modules
from benches.shootout_data import partitions
from benches.shootout_metrics import interval_radius
from benches.shootout_models import clean, fit_model, loss


@pytest.mark.parametrize("design", ["iid", "spatial", "temporal"])
def test_partitions_are_disjoint_and_respect_dependency_units(design):
    groups = np.repeat(np.arange(100), 10)
    cuts = partitions(1000, 0, "regression", design, groups=groups)
    for i, a in enumerate(cuts):
        assert len(a) > 0
        for b in cuts[i + 1:]:
            assert not set(a) & set(b)
            if design != "iid":
                assert not set(groups[a]) & set(groups[b])
    if design == "temporal":
        assert all(a.max() < b.min() for a, b in zip(cuts, cuts[1:], strict=False))


def test_imputation_never_reads_validation_or_test_statistics():
    fit = pl.DataFrame({"x": [1., 3., None], "y": [0., 1., 0.]})
    val = pl.DataFrame({"x": [1000., None], "y": [1., 0.]})
    a, b = clean(fit, val, cats=[])
    assert a["x"].to_list() == [1., 3., 2.]
    assert b["x"].to_list() == [1000., 2.]


def test_native_path_selection_matches_an_independent_train_only_fit():
    import mboostrs as mb

    rng = np.random.default_rng(91)
    x = rng.uniform(-1, 1, 180)
    frame = pl.DataFrame({"x": x, "y": np.sin(2 * x) + rng.normal(0, .2, len(x))})
    train, val = frame.head(120), frame.tail(60)
    fitted = fit_model("mboost_additive", {"df": 4, "nu": .1}, train, [], "regression", 0,
                       validation=val)
    stop = fitted.config["mstop"]
    independent = mb.gamboost(fitted.config["formula"], train,
                              control=mb.boost_control(stop, nu=.1))
    expected = loss(val["y"].to_numpy(), independent.predict(val, extrapolation="allow"), "regression")
    assert fitted.config["validation_loss"] == pytest.approx(expected, rel=1e-9)


def test_conformal_radius_uses_a_finite_sample_order_statistic():
    assert interval_radius(np.arange(10.)) == 9
    with pytest.raises(ValueError):
        interval_radius(np.array([np.nan]))


def test_cache_identity_covers_package_qualified_local_helpers():
    from pathlib import Path

    modules = _local_modules("bench_competitive_shootout", Path(__file__).resolve().parents[1] / "benches")
    assert {"shootout_models", "shootout_data", "shootout_metrics", "_jobs"} <= set(modules)


def test_causal_evidence_serializes_as_strict_json():
    import json

    from benches.bench_shootout_causal import run_job

    import mboostrs as mb

    row = run_job(("smooth_effect", 0, "unadjusted"))
    decoded = json.loads(json.dumps(row, allow_nan=False))
    # The record names the profile of the extension that ran, whatever it is:
    # a debug build used for a test run must not be reported as release.
    assert decoded["native_build"]["profile"] == mb.build_info()["profile"]
