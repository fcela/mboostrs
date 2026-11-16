"""Inline formula transforms and recipe replay.

The property that matters is that a *fitted* transform uses its training
parameters at prediction time. Recomputing `scale()` on new data is the silent
error this feature exists to remove, so most of these tests are about that.
"""

import numpy as np
import polars as pl
import pytest
from mboostrs.transforms import apply_recipe, extract_transforms, fit_recipe

import mboostrs as mb

CTRL = mb.boost_control(200, 0.1)


def _log_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 10.0, n)
    z = rng.normal(size=n)
    y = 2.0 * np.log(x) + 0.5 * z + rng.normal(scale=0.2, size=n)
    return {"x": x, "z": z, "y": y}


def _scale_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=50.0, scale=10.0, size=n)
    std = (x - x.mean()) / x.std(ddof=1)
    return {"x": x, "y": 2.0 * std + rng.normal(scale=0.2, size=n)}


# --------------------------------------------------------------------------
# rewriting
# --------------------------------------------------------------------------


def test_transform_is_rewritten_to_a_derived_column():
    data = {"x": np.array([1.0, 2.0]), "y": np.array([0.0, 1.0])}
    formula, recipe = extract_transforms("y ~ log(x)", data)
    assert formula == "y ~ log__x"
    assert recipe.derived_names == ["log__x"]
    assert recipe.source_names == ["x"]


def test_transform_inside_a_learner_call_is_rewritten():
    data = {"x": np.array([1.0, 2.0]), "y": np.array([0.0, 1.0])}
    formula, recipe = extract_transforms("y ~ bbs(scale(x), df=4)", data)
    assert formula == "y ~ bbs(scale__x, df=4)"
    assert recipe.derived_names == ["scale__x"]


def test_repeated_transform_creates_one_derived_column():
    data = {"x": np.array([1.0, 2.0]), "y": np.array([0.0, 1.0])}
    formula, recipe = extract_transforms("y ~ log(x) + bbs(log(x), df=4)", data)
    assert formula == "y ~ log__x + bbs(log__x, df=4)"
    assert len(recipe.steps) == 1


def test_formula_without_transforms_is_unchanged_with_an_empty_recipe():
    data = {"x": np.array([1.0]), "y": np.array([0.0])}
    formula, recipe = extract_transforms("y ~ x + bbs(x, df=4)", data)
    assert formula == "y ~ x + bbs(x, df=4)"
    assert not recipe


def test_transform_of_an_unknown_column_is_rejected():
    data = {"x": np.array([1.0]), "y": np.array([0.0])}
    with pytest.raises(ValueError, match="not in the data"):
        extract_transforms("y ~ log(nope)", data)


def test_a_derived_name_colliding_with_a_real_column_is_rejected():
    # log(x) derives "log__x"; a data column of that exact name would
    # otherwise be silently replaced by the derived values, and the
    # formula's two distinct terms would collapse into one.
    data = {
        "x": np.array([1.0, 2.0]),
        "log__x": np.array([100.0, 100.0]),
        "y": np.array([0.0, 1.0]),
    }
    with pytest.raises(ValueError, match="log__x.*already exists"):
        extract_transforms("y ~ log(x) + log__x", data)


# --------------------------------------------------------------------------
# the recipe replays training parameters
# --------------------------------------------------------------------------


def test_scale_records_training_mean_and_sd():
    data = _scale_data()
    fit = mb.glmboost("y ~ scale(x)", data=data, family=mb.Gaussian(), control=CTRL)
    (step,) = fit._transform_recipe.describe()
    mean, sd = step["params"]
    assert mean == pytest.approx(float(np.mean(data["x"])))
    assert sd == pytest.approx(float(np.std(data["x"], ddof=1)))


def test_scale_uses_training_parameters_on_new_data():
    """The core guarantee.

    If `scale` were recomputed on `newdata`, predictions would depend on which
    other rows happened to be in the prediction batch. They must not.
    """
    data = _scale_data()
    fit = mb.glmboost("y ~ scale(x)", data=data, family=mb.Gaussian(), control=CTRL)

    small = {"x": np.array([50.0, 60.0])}
    # Same two rows, but embedded among wildly different values. Recomputing
    # the scaling would change the first two predictions; replay must not.
    embedded = {"x": np.array([50.0, 60.0, 500.0, -400.0])}

    np.testing.assert_allclose(
        np.asarray(fit.predict(newdata=small, type="response")),
        np.asarray(fit.predict(newdata=embedded, type="response"))[:2],
        rtol=0,
        atol=1e-12,
    )


def test_single_row_prediction_works_for_scale():
    """A one-row batch has undefined sample sd, so recomputation would fail."""
    data = _scale_data()
    fit = mb.glmboost("y ~ scale(x)", data=data, family=mb.Gaussian(), control=CTRL)
    pred = fit.predict(newdata={"x": np.array([55.0])}, type="response")
    assert np.asarray(pred).shape == (1,)
    assert np.all(np.isfinite(pred))


def test_transform_matches_a_manually_derived_column():
    data = _log_data()
    manual = {**data, "logx": np.log(data["x"])}
    a = mb.glmboost("y ~ log(x) + z", data=data, family=mb.Gaussian(), control=CTRL)
    b = mb.glmboost("y ~ logx + z", data=manual, family=mb.Gaussian(), control=CTRL)
    np.testing.assert_allclose(a.fitted_, b.fitted_, rtol=0, atol=1e-12)


def test_recipe_survives_slicing():
    data = _scale_data()
    fit = mb.glmboost("y ~ scale(x)", data=data, family=mb.Gaussian(), control=CTRL)
    newdata = {"x": np.array([50.0, 60.0])}
    sliced = fit[:50]
    assert sliced._transform_recipe is not None
    pred = sliced.predict(newdata=newdata, type="response")
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("engine", ["glmboost", "gamboost"])
def test_both_engines_support_transforms(engine):
    data = _log_data()
    formula = "y ~ log(x) + z" if engine == "glmboost" else "y ~ bbs(log(x), df=4) + z"
    fit = getattr(mb, engine)(
        formula, data=data, family=mb.Gaussian(), control=mb.boost_control(150, 0.1)
    )
    pred = fit.predict(newdata={"x": np.array([2.0, 8.0]), "z": np.zeros(2)},
                       type="response")
    assert np.all(np.isfinite(pred))
    # log(x) is the real relationship, so the fit should be good
    assert np.corrcoef(fit.fitted_, data["y"])[0, 1] > 0.9


def test_transforms_work_on_polars_frames():
    data = pl.DataFrame(_log_data())
    fit = mb.glmboost("y ~ log(x) + z", data=data, family=mb.Gaussian(), control=CTRL)
    newdata = pl.DataFrame({"x": [2.0, 8.0], "z": [0.0, 0.0]})
    assert np.all(np.isfinite(fit.predict(newdata=newdata, type="response")))


def test_identity_expression_is_supported():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(-2, 2, n)
    data = {"x": x, "y": 1.5 * x**2 + rng.normal(scale=0.2, size=n)}
    fit = mb.glmboost("y ~ I(x**2)", data=data, family=mb.Gaussian(), control=CTRL)
    assert np.corrcoef(fit.fitted_, data["y"])[0, 1] > 0.95
    pred = fit.predict(newdata={"x": np.array([0.0, 1.0])}, type="response")
    assert np.all(np.isfinite(pred))


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_log_of_a_nonpositive_column_reports_the_expression():
    data = {"x": np.array([1.0, -1.0, 2.0]), "y": np.array([0.0, 1.0, 2.0])}
    with pytest.raises(ValueError, match=r"log\(x\).*non-finite"):
        mb.glmboost("y ~ log(x)", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(10, 0.1))


def test_scale_of_a_constant_column_is_rejected():
    data = {"x": np.full(20, 3.0), "y": np.arange(20.0)}
    with pytest.raises(ValueError, match="zero variance"):
        mb.glmboost("y ~ scale(x)", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(10, 0.1))


def test_prediction_data_missing_the_source_column_is_named():
    data = _log_data()
    fit = mb.glmboost("y ~ log(x) + z", data=data, family=mb.Gaussian(), control=CTRL)
    with pytest.raises(ValueError, match="missing column 'x'"):
        fit.predict(newdata={"z": np.zeros(3)}, type="response")


def test_identity_may_combine_columns_as_it_does_in_r():
    data = {"a": np.array([1.0, 3.0]), "b": np.array([2.0, 4.0]),
            "y": np.array([0.0, 1.0])}
    rewritten, recipe = extract_transforms("y ~ I(a * b)", data)
    (step,) = recipe.steps
    assert step.sources == ("a", "b")
    assert step.expression == "I(a * b)"
    assert rewritten == f"y ~ {step.name}"
    np.testing.assert_allclose(
        apply_recipe(recipe, data)[step.name], [2.0, 12.0]
    )


def test_identity_with_an_unknown_name_is_rejected():
    data = {"a": np.array([1.0]), "y": np.array([0.0])}
    with pytest.raises(ValueError, match="unknown name"):
        extract_transforms("y ~ I(a + qq)", data)


def test_recipe_application_does_not_mutate_the_input():
    data = {"x": np.array([1.0, 2.0]), "y": np.array([0.0, 1.0])}
    _, recipe = extract_transforms("y ~ log(x)", data)
    recipe = fit_recipe(recipe, data)
    out = apply_recipe(recipe, data)
    assert "log__x" in out
    assert "log__x" not in data


# --------------------------------------------------------------------------
# categorical predictors
# --------------------------------------------------------------------------


def _factor_data(n=300, seed=0, levels=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    g = np.array(levels)[rng.integers(0, len(levels), n)]
    x = rng.normal(size=n)
    effect = {"a": 1.0, "b": -1.0, "c": 2.0}
    y = np.array([effect[v] for v in g]) + 0.5 * x + rng.normal(scale=0.2, size=n)
    return {"g": g, "x": x, "y": y}


def test_brandom_accepts_a_string_factor_and_recovers_level_effects():
    data = _factor_data()
    fit = mb.gamboost(
        "y ~ brandom(g) + bols(x)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(300, 0.1),
    )
    assert np.size(fit.coef()["brandom(g)"]) == 3

    # Predict at x = 0 for each level; the fitted level effects should be close
    # to the truth (shrunk toward the mean by the ridge penalty).
    pred = np.asarray(
        fit.predict(
            newdata={"g": np.array(["a", "b", "c"]), "x": np.zeros(3)},
            type="response",
        )
    )
    assert pred[2] > pred[0] > pred[1], pred


def test_factor_levels_are_recorded_in_the_recipe():
    data = _factor_data()
    fit = mb.gamboost("y ~ brandom(g)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    (step,) = [s for s in fit._transform_recipe.describe() if s["kind"] == "factor"]
    assert step["params"] == [("a", "b", "c")]


def test_unseen_level_at_prediction_time_is_named():
    """Item K: a level the model never saw has no estimated effect."""
    data = _factor_data()
    fit = mb.gamboost("y ~ brandom(g) + bols(x)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    with pytest.raises(ValueError, match=r"level\(s\) \['zzz'\] not present"):
        fit.predict(newdata={"g": np.array(["a", "zzz"]), "x": np.zeros(2)})


def test_level_order_is_stable_regardless_of_row_order():
    """Encoding must not depend on which level happens to appear first."""
    data = _factor_data(seed=1)
    shuffled = {k: v[::-1] for k, v in data.items()}
    a = mb.gamboost("y ~ brandom(g)", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(50, 0.1))
    b = mb.gamboost("y ~ brandom(g)", data=shuffled, family=mb.Gaussian(),
                    control=mb.boost_control(50, 0.1))
    (sa,) = [s for s in a._transform_recipe.describe() if s["kind"] == "factor"]
    (sb,) = [s for s in b._transform_recipe.describe() if s["kind"] == "factor"]
    assert sa["params"] == sb["params"]


@pytest.mark.parametrize("term", ["bbs(g, df=4)", "bmono(g)"])
def test_noncategorical_learner_on_a_factor_is_refused_with_a_reason(term):
    """Encoding silently would make the learner read levels as an ordering.

    That is a statistical error with no visible symptom, so it must fail.
    """
    data = _factor_data()
    with pytest.raises(ValueError, match="categorical column"):
        mb.gamboost(f"y ~ {term} + bols(x)", data=data, family=mb.Gaussian(),
                    control=mb.boost_control(10, 0.1))


@pytest.mark.parametrize("term", ["bols(g)", "g"])
def test_plain_bols_on_a_factor_fits_one_effect_per_level(term):
    """R's `bols(factor)` is least squares on the dummy design; a plain
    `bols(g)` or a bare `g` fits the same column space, one effect per level,
    never the integer codes."""
    data = _factor_data()
    control = mb.boost_control(50, 0.1)
    fit = mb.gamboost(f"y ~ {term} + bols(x)", data=data, family=mb.Gaussian(),
                      control=control)
    levels = mb.gamboost("y ~ brandom(g, lambda=0) + bols(x)", data=data,
                         family=mb.Gaussian(), control=control)
    np.testing.assert_allclose(fit.fitted(), levels.fitted(), atol=1e-12)


def test_factor_works_from_a_polars_frame():
    data = pl.DataFrame(_factor_data())
    fit = mb.gamboost("y ~ brandom(g) + bols(x)", data=data, family=mb.Gaussian(),
                      control=mb.boost_control(100, 0.1))
    newdata = pl.DataFrame({"g": ["a", "c"], "x": [0.0, 0.0]})
    assert np.all(np.isfinite(fit.predict(newdata=newdata, type="response")))


def test_numeric_columns_are_untouched_by_factor_encoding():
    data = _factor_data()
    numeric_only = {"x": data["x"], "y": data["y"]}
    fit = mb.glmboost("y ~ x", data=numeric_only, family=mb.Gaussian(),
                      control=mb.boost_control(50, 0.1))
    assert fit._transform_recipe is None


# ---------------------------------------------------------------------------
# Parity with R's formula transforms
# ---------------------------------------------------------------------------
#
# R hands the formula to `model.matrix`, so it accepts anything R can evaluate.
# Python whitelists transforms instead, which keeps a formula safe to accept
# from a config file and keeps every transform replayable at prediction time.
# The whitelist is therefore narrower on purpose — but where both frontends
# accept a formula, they must agree on the numbers *and* on the term label.
#
# Reference values below were produced by R mboostrs on the identical data
# (seed 11, n=300), printed to ten digits.

R_REFERENCE = {
    "y ~ log(x)": {"log(x)": 2.0482178143},
    "y ~ log1p(x)": {"log1p(x)": 2.8219773717},
    "y ~ sqrt(x)": {"sqrt(x)": 2.4171632506},
    "y ~ exp(z)": {"exp(z)": 0.2821210327},
    "y ~ pmin(x, 3)": {"pmin(x, 3)": 1.4616715119},
    "y ~ pmax(x, 3)": {"pmax(x, 3)": 0.8685792076},
    "y ~ I(x^2)": {"I(x^2)": 0.0908829683},
    "y ~ I(x^2) + I(x^3)": {"I(x^2)": 0.1092903447, "I(x^3)": -0.0035541595},
    "y ~ I(x*z)": {"I(x*z)": 0.1345426046},
    "y ~ I(x * 2 + 1)": {"I(x * 2 + 1)": 0.3365544743},
}


@pytest.fixture(scope="module")
def r_parity_data():
    """The draw R used, recorded rather than regenerated.

    R's RNG stream is not numpy's, so the columns are read from the CSV that
    the R reference run wrote (set.seed(11), n=300).
    """
    import csv
    import pathlib

    fixture = pathlib.Path(__file__).parent / "fixtures" / "formula_transform_parity.csv"
    if not fixture.exists():
        pytest.skip(f"missing R fixture {fixture}")
    rows = list(csv.DictReader(fixture.open()))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("x", "z", "y")}


@pytest.mark.parametrize("formula", sorted(R_REFERENCE))
def test_transform_coefficients_match_r(formula, r_parity_data):
    fit = mb.glmboost(
        formula, data=r_parity_data, family=mb.Gaussian(),
        control=mb.boost_control(200, 0.1),
    )
    coefficients = fit.coef()
    for term, expected in R_REFERENCE[formula].items():
        # Indexing by the R term label is half the assertion: a differently
        # spelled key raises KeyError here.
        assert coefficients[term] == pytest.approx(expected, abs=1e-9)


def test_two_polynomial_terms_stay_two_terms():
    """`I(x**2)` and `I(x**3)` both used to derive the name `I__x`. A repeated
    derived name is treated as the same column, so the second term was dropped
    and `y ~ I(x**2) + I(x**3)` silently fitted a one-term model — no error, no
    warning, the cubic simply absent."""
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(1, 4, n)
    data = {"x": x, "y": 3 * x**2 - 2 * x**3 + rng.normal(0, 0.5, n)}

    rewritten, recipe = extract_transforms("y ~ I(x**2) + I(x**3)", data)
    assert len(recipe.steps) == 2
    assert len({step.name for step in recipe.steps}) == 2

    fit = mb.glmboost(
        "y ~ I(x**2) + I(x**3)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(200, 0.1),
    )
    assert {"I(x**2)", "I(x**3)"} <= set(fit.coef())


def test_expressions_that_sanitize_alike_get_distinct_columns():
    """`I(x**2)` and `I(x*2)` both sanitize to `I__x_2`; they are different
    columns and must stay so."""
    rng = np.random.default_rng(4)
    n = 200
    x = rng.uniform(1, 4, n)
    data = {"x": x, "y": x**2 + 2 * x + rng.normal(0, 0.2, n)}
    _, recipe = extract_transforms("y ~ I(x**2) + I(x*2)", data)
    assert len({step.name for step in recipe.steps}) == 2

    built = apply_recipe(fit_recipe(recipe, data), data)
    squared, doubled = (built[step.name] for step in recipe.steps)
    np.testing.assert_allclose(squared, x**2)
    np.testing.assert_allclose(doubled, x * 2)


@pytest.mark.parametrize(
    "formula, expected",
    [("y ~ pmin(x, 3)", lambda x: np.minimum(x, 3.0)),
     ("y ~ pmax(x, 3)", lambda x: np.maximum(x, 3.0))],
)
def test_clamps_build_the_column_r_would(formula, expected):
    rng = np.random.default_rng(6)
    n = 100
    x = rng.uniform(1, 6, n)
    data = {"x": x, "y": rng.normal(size=n)}
    _, recipe = extract_transforms(formula, data)
    (step,) = recipe.steps
    built = apply_recipe(fit_recipe(recipe, data), data)
    np.testing.assert_allclose(built[step.name], expected(x))


def test_caret_and_double_star_mean_the_same_thing():
    """R writes `I(x^2)`, Python `I(x**2)`. Both are accepted, and the label
    reported back is the one the user wrote."""
    rng = np.random.default_rng(8)
    n = 120
    x = rng.uniform(1, 4, n)
    data = {"x": x, "y": x**2 + rng.normal(0, 0.2, n)}

    _, caret = extract_transforms("y ~ I(x^2)", data)
    _, star = extract_transforms("y ~ I(x**2)", data)
    np.testing.assert_allclose(
        apply_recipe(caret, data)[caret.steps[0].name],
        apply_recipe(star, data)[star.steps[0].name],
    )
    assert caret.steps[0].expression == "I(x^2)"
    assert star.steps[0].expression == "I(x**2)"


def test_a_multi_column_identity_replays_on_new_data():
    """Both source columns have to travel into prediction."""
    rng = np.random.default_rng(9)
    n = 200
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    data["y"] = 1.5 * data["x"] * data["z"] + rng.normal(0, 0.2, n)
    fit = mb.glmboost(
        "y ~ I(x*z)", data=data, family=mb.Gaussian(),
        control=mb.boost_control(200, 0.1),
    )
    newdata = {"x": rng.normal(size=50), "z": rng.normal(size=50)}
    assert np.all(np.isfinite(np.asarray(fit.predict(newdata=newdata))))

    with pytest.raises(ValueError, match="missing column 'z'"):
        fit.predict(newdata={"x": newdata["x"]})


def test_transformed_terms_report_the_expression_not_the_derived_column():
    """The derived column is `log__x` so the rewritten formula stays parseable.
    Users see `log(x)`, as they do in R."""
    rng = np.random.default_rng(12)
    n = 200
    x = rng.uniform(1, 10, n)
    data = {"x": x, "z": rng.normal(size=n)}
    data["y"] = 2 * np.log(x) + 0.5 * data["z"] + rng.normal(0, 0.2, n)
    fit = mb.glmboost(
        "y ~ log(x) + z", data=data, family=mb.Gaussian(),
        control=mb.boost_control(100, 0.1),
    )
    assert "log(x)" in fit.coef()
    assert "log__x" not in fit.coef()
    assert "log(x)" in set(fit.selected)
    assert "log(x)" in fit.base_learner_names()
    # The column itself keeps the formula-safe name.
    assert "log__x" in fit._predictor_names


def _skewed_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(sigma=1.5, size=n)
    return {"x": x, "y": np.log(x) + rng.normal(scale=0.3, size=n)}


def test_ecdf_maps_to_training_quantiles_and_replays_them():
    data = _skewed_data()
    formula, recipe = extract_transforms("y ~ bbs(ecdf(x))", data)
    recipe = fit_recipe(recipe, data)
    (step,) = recipe.steps
    derived = apply_recipe(recipe, data)[step.name]
    ranks = (np.argsort(np.argsort(data["x"])) + 0.5) / len(data["x"])
    assert np.max(np.abs(derived - ranks)) < 0.01
    # Values beyond the training range are held at the extremes.
    far = apply_recipe(recipe, {"x": np.array([-5.0, 1e6])})[step.name]
    assert far.tolist() == [0.0, 1.0]


def test_ecdf_spline_predicts_flat_beyond_the_training_range(tmp_path):
    data = _skewed_data()
    fit = mb.gamboost("y ~ bbs(ecdf(x), df=4)", data=data, control=CTRL)
    top = float(np.max(data["x"]))
    new = {"x": np.array([top, 10 * top, 1000 * top])}
    prediction = fit.predict(newdata=new)
    assert np.ptp(prediction) == pytest.approx(0.0, abs=1e-12)

    path = tmp_path / "ecdf.json"
    mb.save_model(fit, path)
    assert np.allclose(mb.load_model(path).predict(newdata=new), prediction)


def test_ecdf_refuses_a_constant_column():
    data = {"x": np.ones(20), "y": np.arange(20.0)}
    with pytest.raises(ValueError, match="two distinct"):
        mb.gamboost("y ~ bbs(ecdf(x))", data=data)


def test_a_log_term_holds_new_values_below_the_training_minimum():
    """A random split can put a zero in the test rows of a column whose
    training rows are all positive; log(0) used to make prediction fail. The
    value is held at the training minimum, with a warning."""
    import warnings

    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 5, 300)
    y = np.log(x) + rng.normal(0, 0.2, 300)
    model = mb.gamboost("y ~ bbs(log(x))", {"x": x, "y": y})
    at_min = np.asarray(model.predict(newdata={"x": np.array([x.min()])})).ravel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        low = np.asarray(model.predict(newdata={"x": np.array([0.0, -1.0])})).ravel()
    np.testing.assert_allclose(low, at_min[0])
    assert any("smallest training value" in str(w.message) for w in caught)
