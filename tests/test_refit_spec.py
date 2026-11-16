"""A refit reproduces the fit it came from, or refuses and says why.

Several operations refit a model, and each used to rebuild the call from
whatever attributes it knew about. `conformal.py`'s version passed the
formula, the data, the family and the control, and dropped the weights,
`center`, the row and component subsampling settings, explicitly supplied
learners, `env` and `bins` — so a conformal interval on a weighted, centred,
binned fit was calibrated against an unweighted, uncentred, exact one, and
reported a coverage guarantee for a model the caller had not fitted.

These tests assert the properties that failure violated: the rebuilt model
agrees with the original where it should, a row subset keeps every row-aligned
object aligned, and a fit that cannot be validly subset is refused by name
rather than silently misaligned.
"""

import warnings

import numpy as np
import pytest
from mboostrs._refit import RefitSpec, refit_spec, require_row_subsettable

import mboostrs as mb


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    n = 160
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return {
        "n": n,
        "x1": x1,
        "x2": x2,
        "y": 2.0 * x1 - x2 + rng.normal(size=n),
        "weights": rng.uniform(0.5, 2.0, n),
    }


def _frame(data):
    return {"x1": data["x1"], "x2": data["x2"], "y": data["y"]}


# ─── the spec carries the whole call ───────────────────────────────


def test_glmboost_spec_carries_every_argument_the_fit_took(data):
    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), weights=data["weights"],
        center=False, weight_policy="raw",
        control=mb.boost_control(20, 0.1),
    )
    spec = refit_spec(model)
    assert spec.kind == "glmboost"
    assert spec.arguments["center"] is False
    assert spec.arguments["weight_policy"] == "raw"
    np.testing.assert_array_equal(spec.arguments["weights"], data["weights"])


def test_gamboost_spec_carries_bins_and_learners(data):
    model = mb.gamboost(
        "y ~ bbs(x1, df=3) + bbs(x2, df=3)", data=_frame(data),
        weights=data["weights"], bins=20,
        control=mb.boost_control(20, 0.1),
    )
    spec = refit_spec(model)
    assert spec.kind == "gamboost"
    assert spec.arguments["bins"] == 20
    assert len(spec.arguments["learners"]) == 2


def test_rebuilding_the_full_spec_reproduces_the_fit(data):
    """The property the dropped arguments broke: same call, same model."""
    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), weights=data["weights"],
        center=False, control=mb.boost_control(25, 0.1),
    )
    rebuilt = refit_spec(model).build()
    np.testing.assert_allclose(rebuilt.fitted(), model.fitted(), rtol=1e-12)
    np.testing.assert_allclose(rebuilt._risk, model._risk, rtol=1e-12)
    assert rebuilt.mstop == model.mstop


def test_a_spec_that_dropped_the_weights_would_not_reproduce_the_fit(data):
    """The failure this guards against, made visible.

    A refit built without the weights fits a different model. If this ever
    stops holding, the fixture has stopped exercising weighting and the tests
    above prove nothing.
    """
    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), weights=data["weights"],
        control=mb.boost_control(25, 0.1),
    )
    unweighted = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), control=mb.boost_control(25, 0.1),
    )
    assert not np.allclose(unweighted.fitted(), model.fitted(), atol=1e-6)


def test_a_spec_that_dropped_center_would_not_reproduce_the_fit(data):
    model = mb.glmboost("y ~ x1 + x2", data=_frame(data), center=False,
                        control=mb.boost_control(8, 0.1))
    centred = mb.glmboost("y ~ x1 + x2", data=_frame(data), center=True,
                          control=mb.boost_control(8, 0.1))
    assert not np.allclose(centred.fitted(), model.fitted(), atol=1e-6)


# ─── subsetting keeps row-aligned objects together ─────────────────


def test_subset_takes_the_weights_with_the_data(data):
    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), weights=data["weights"],
        control=mb.boost_control(20, 0.1),
    )
    rows = np.arange(0, data["n"], 2)
    subset = refit_spec(model).subset(rows)

    assert len(subset.data["y"]) == rows.size
    assert len(subset.arguments["weights"]) == rows.size
    np.testing.assert_array_equal(subset.arguments["weights"],
                                  data["weights"][rows])
    np.testing.assert_array_equal(subset.data["y"], data["y"][rows])


def test_a_subset_refit_uses_the_subset_weights(data):
    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), weights=data["weights"],
        control=mb.boost_control(20, 0.1),
    )
    rows = np.arange(0, data["n"], 2)
    rebuilt = refit_spec(model).subset(rows).build()

    direct = mb.glmboost(
        "y ~ x1 + x2",
        data={key: value[rows] for key, value in _frame(data).items()},
        weights=data["weights"][rows], control=mb.boost_control(20, 0.1),
    )
    np.testing.assert_allclose(rebuilt.fitted(), direct.fitted(), rtol=1e-12)


@pytest.mark.parametrize("kind", ["glm", "gam"])
def test_training_subset_refits_stateful_transforms(kind):
    x = np.r_[np.arange(90.), np.arange(10.) + 1000]
    frame = {"x": x, "y": np.sin(x)}
    if kind == "glm":
        model = mb.glmboost("y ~ scale(x)", frame, control=mb.boost_control(10))
    else:
        model = mb.gamboost("y ~ bbs(scale(x))", frame, control=mb.boost_control(10))
    rows = np.arange(90)
    fixed = refit_spec(model).subset(rows).build()
    trained = refit_spec(model).subset(rows, refit_preprocessing=True).build()
    fixed_mean = next(step["params"][0] for step in fixed._transform_recipe.describe()
                      if step["kind"] == "scale")
    trained_mean = next(step["params"][0] for step in trained._transform_recipe.describe()
                        if step["kind"] == "scale")
    assert fixed_mean == pytest.approx(140.5)
    assert trained_mean == pytest.approx(44.5)


def test_training_subset_rebuilds_formula_interactions(data):
    frame = _frame(data)
    model = mb.gamboost(
        "y ~ bols(x1) + bbs(x1) %X% bols(x2)", frame,
        control=mb.boost_control(10),
    )
    rows = np.arange(100)
    spec = refit_spec(model)
    assert spec.row_subset_blockers(refit_preprocessing=True) == []
    rebuilt = spec.subset(rows, refit_preprocessing=True).build()
    direct = mb.gamboost(
        "y ~ bols(x1) + bbs(x1) %X% bols(x2)",
        {name: value[rows] for name, value in frame.items()},
        control=mb.boost_control(10),
    )
    np.testing.assert_allclose(rebuilt.fitted(), direct.fitted(), atol=1e-12)


def test_conformal_can_rebuild_formula_interactions(data):
    frame = _frame(data)
    model = mb.gamboost(
        "y ~ bols(x1) + bbs(x1, df=3) %X% bols(x2)", frame,
        control=mb.boost_control(10),
    )
    interval = model.conformal_interval(newdata=frame, seed=4)
    assert interval.lower.shape == (data["n"],)
    assert np.all(np.isfinite(interval.lower))


def test_subset_leaves_hyperparameters_alone(data):
    model = mb.gamboost(
        "y ~ bbs(x1, df=3)", data=_frame(data), bins=20,
        control=mb.boost_control(20, 0.1),
    )
    subset = refit_spec(model).subset(np.arange(50))
    assert subset.arguments["bins"] == 20
    assert subset.arguments["control"] == model._control


def test_subset_works_on_a_polars_frame(data):
    polars = pytest.importorskip("polars")
    frame = polars.DataFrame(_frame(data))
    model = mb.glmboost("y ~ x1 + x2", data=frame,
                        control=mb.boost_control(15, 0.1))
    subset = refit_spec(model).subset(np.arange(40))
    assert subset.data.height == 40
    assert subset.build().mstop == 15


# ─── what cannot be subset is named ────────────────────────────────


def test_a_buser_design_blocks_a_row_subset(data):
    """Its matrix has one row per observation and cannot follow the subset."""
    design = np.column_stack([data["x1"], data["x1"] ** 2])
    model = mb.gamboost(
        "y ~ .", data=_frame(data),
        learners=[mb.Buser("u", X=design, K=np.eye(2))],
        control=mb.boost_control(15, 0.1),
    )
    blockers = refit_spec(model).row_subset_blockers()
    assert any("one row per observation" in reason for reason in blockers)

    with pytest.raises(NotImplementedError) as excinfo:
        require_row_subsettable(model, "conformal calibration")
    assert "conformal calibration" in str(excinfo.value)
    assert "Buser" in str(excinfo.value)


def test_row_subsampling_blocks_a_row_subset(data):
    """A subsampled fit is already a fit of a random subset."""
    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), subsample=0.7, subsample_seed=1,
        control=mb.boost_control(15, 0.1),
    )
    blockers = refit_spec(model).row_subset_blockers()
    assert any("row subsampling" in reason for reason in blockers)


def test_an_ordinary_fit_blocks_nothing(data):
    model = mb.gamboost("y ~ bbs(x1, df=3) + bols(x2)", data=_frame(data),
                        control=mb.boost_control(15, 0.1))
    assert refit_spec(model).row_subset_blockers() == []
    assert isinstance(require_row_subsettable(model, "anything"), RefitSpec)


def test_a_model_without_a_bundle_says_so(data):
    model = mb.glmboost("y ~ x1", data=_frame(data),
                        control=mb.boost_control(10, 0.1))
    del model._refit_bundle
    with pytest.raises(ValueError, match="did not retain the inputs"):
        refit_spec(model)


# ─── conformal uses it ─────────────────────────────────────────────


def test_conformal_refits_with_the_weights_the_fit_used(data):
    """The regression: the calibration model differed from the fitted one."""
    from mboostrs import conformal

    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), weights=data["weights"],
        center=False, control=mb.boost_control(20, 0.1),
    )
    rows = np.arange(0, data["n"], 2)
    refitted = conformal._refit(model, None, rows=rows)

    assert refitted.input_weights() is not None
    np.testing.assert_array_equal(refitted.input_weights(),
                                  data["weights"][rows])


def test_conformal_refits_preprocessing_inside_training_rows():
    x = np.r_[np.arange(90.), np.arange(10.) + 1000]
    model = mb.glmboost("y ~ scale(x)", {"x": x, "y": np.sin(x)},
                        control=mb.boost_control(10))
    from mboostrs import conformal

    refitted = conformal._refit(model, None, rows=np.arange(90))
    mean = next(step["params"][0] for step in refitted._transform_recipe.describe()
                if step["kind"] == "scale")
    assert mean == pytest.approx(44.5)


def test_conformal_warns_that_a_weighted_guarantee_is_not_implemented(data):
    """Split conformal's coverage rests on exchangeability, which weights break.

    The interval is still computed; what changes is that the caller is told
    the level is nominal rather than guaranteed.
    """
    model = mb.glmboost(
        "y ~ x1 + x2", data=_frame(data), weights=data["weights"],
        control=mb.boost_control(20, 0.1),
    )
    with pytest.warns(UserWarning, match="case weights"):
        interval = model.conformal_interval(level=0.9)
    assert np.all(np.isfinite(interval.lower))


def test_conformal_does_not_warn_on_an_unweighted_fit(data):
    model = mb.glmboost("y ~ x1 + x2", data=_frame(data),
                        control=mb.boost_control(20, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.conformal_interval(level=0.9)


def test_conformal_refuses_a_fit_it_cannot_subset(data):
    design = np.column_stack([data["x1"], data["x1"] ** 2])
    model = mb.gamboost(
        "y ~ .", data=_frame(data),
        learners=[mb.Buser("u", X=design, K=np.eye(2))],
        control=mb.boost_control(15, 0.1),
    )
    with pytest.raises(NotImplementedError, match="Buser"):
        model.conformal_interval(level=0.9)


# ─── CV+ ───────────────────────────────────────────────────────────


def test_cv_plus_covers_at_the_nominal_level(data):
    model = mb.glmboost("y ~ x1 + x2", data=_frame(data),
                        control=mb.boost_control(20, 0.1))
    interval = model.conformal_interval(method="cv+", level=0.9, n_folds=5)
    covered = np.mean((data["y"] >= interval.lower) & (data["y"] <= interval.upper))
    assert covered >= 0.85


def test_cv_plus_chunking_does_not_change_the_answer(data):
    """The endpoints are order statistics per target row, so the target rows
    are independent and can be taken in blocks. A pool of `n x n_new` is
    800 MB at n = 10,000, which the default `newdata=None` asks for."""
    from mboostrs import conformal

    model = mb.glmboost("y ~ x1 + x2", data=_frame(data),
                        control=mb.boost_control(20, 0.1))
    whole = model.conformal_interval(method="cv+", level=0.9, n_folds=4, seed=1)

    original = conformal._cv_plus_chunk
    conformal._cv_plus_chunk = lambda n, n_new: 7
    try:
        chunked = model.conformal_interval(method="cv+", level=0.9, n_folds=4,
                                           seed=1)
    finally:
        conformal._cv_plus_chunk = original

    np.testing.assert_allclose(chunked.lower, whole.lower, rtol=1e-12)
    np.testing.assert_allclose(chunked.upper, whole.upper, rtol=1e-12)


def test_cv_plus_does_not_refit_the_full_model(data, monkeypatch):
    """It was refitting the whole path twice for a model already in hand."""
    from mboostrs import conformal

    calls = []
    original = conformal._refit

    def counted(model, frame, **kwargs):
        calls.append(kwargs.get("rows"))
        return original(model, frame, **kwargs)

    monkeypatch.setattr(conformal, "_refit", counted)
    model = mb.glmboost("y ~ x1 + x2", data=_frame(data),
                        control=mb.boost_control(15, 0.1))
    model.conformal_interval(method="cv+", level=0.9, n_folds=4)

    assert len(calls) == 4, "one refit per fold, and no full-data refit"
    assert all(rows is not None for rows in calls)


def test_a_sliced_model_refits_at_its_own_mstop():
    """`fit[:63]` keeps the control it was fitted with (400 steps). Refits --
    conformal calibration, confint replicates, stability selection -- used
    that control and so refitted a different model from the one passed in."""
    import warnings

    import mboostrs as mb
    from mboostrs._refit import refit_spec

    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200)
    data = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 200)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mb.gamboost("y ~ bbs(x)", data=data, control=mb.boost_control(400, 0.1))
        sliced = fit[:63]
        assert refit_spec(sliced).build().mstop == 63
        assert refit_spec(sliced).subset(np.arange(150)).build().mstop == 63
        assert refit_spec(sliced).subset(
            np.arange(150), refit_preprocessing=True).build().mstop == 63
        ci = sliced.confint(B=3, B_mstop=0, seed=1)
    assert ci["n_used"] == 3
