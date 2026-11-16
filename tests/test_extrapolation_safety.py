"""Extrapolation is visible and safe at prediction time (issue #357, item 3).

Three things were wrong. The `ExtrapolationWarning` subclassed `UserWarning`,
so a blanket `filterwarnings("ignore", category=UserWarning)` removed it. The
`clip` policy clamped the spline's own covariate but not the `by=` variable of
a varying-coefficient term, so a `by=` value of 1,243 against a training
maximum near 600 still scaled the effect by 1,243. And nothing reported, as a
table, which terms and covariates left their training range.
"""

import inspect
import warnings

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs._extrapolation import REPORT_COLUMNS, ExtrapolationWarning
from mboostrs.gamboost import GAMBoostModel
from mboostrs.glmboost import GLMBoostModel


@pytest.fixture(scope="module")
def housing():
    """Age and occupancy, with an interaction: the effect of age depends on
    occupancy, which is the `by=` shape the benchmark used."""
    rng = np.random.default_rng(1)
    n = 400
    age = rng.uniform(0.0, 50.0, n)
    occ = rng.uniform(1.0, 6.0, n)
    y = np.sin(age / 8.0) * occ + rng.normal(size=n) * 0.2
    return {"age": age, "occ": occ, "y": y}


@pytest.fixture(scope="module")
def varying(housing):
    return mb.gamboost("y ~ bbs(age, df=4, by=occ)", data=housing,
                       control=mb.boost_control(100, 0.1))


@pytest.fixture(scope="module")
def far_occupancy(housing):
    """Two rows; the second has an occupancy far above the training maximum."""
    return {"age": np.array([10.0, 10.0]), "occ": np.array([3.0, 60.0])}


@pytest.fixture(scope="module")
def at_the_boundary(housing):
    """The same age, with occupancy at the training maximum."""
    return {"age": np.array([10.0]), "occ": np.array([housing["occ"].max()])}


def _messages(caught):
    """Texts of the extrapolation warnings among `caught`, ignoring others."""
    return [str(w.message) for w in caught
            if issubclass(w.category, ExtrapolationWarning)]


# ─── 1. the warning is not a UserWarning ───────────────────────────


def test_the_extrapolation_warning_is_a_warning_but_not_a_user_warning():
    assert issubclass(ExtrapolationWarning, Warning)
    assert not issubclass(ExtrapolationWarning, UserWarning)


def test_a_blanket_user_warning_filter_does_not_hide_extrapolation(varying, far_occupancy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=UserWarning)
        varying.predict(newdata=far_occupancy)
    assert [w.category for w in caught] == [ExtrapolationWarning]


def test_the_default_policy_is_still_warn():
    """R mboost parity depends on the default; the change is in the class of
    the warning, not in what predict does by default."""
    for method in (GAMBoostModel.predict, GAMBoostModel.predict_over_mstop):
        default = inspect.signature(method).parameters["extrapolation"].default
        assert default == "warn", method.__name__


# ─── 2. clip clamps the by= variable ───────────────────────────────


def test_the_warning_names_the_by_covariate(varying, far_occupancy):
    with pytest.warns(ExtrapolationWarning) as caught:
        varying.predict(newdata=far_occupancy)
    (message,) = _messages(caught)
    assert "bbs(age), by=occ" in message
    assert "covariate occ" in message


def test_clip_clamps_an_out_of_range_by_variable_to_the_boundary(
        varying, far_occupancy, at_the_boundary):
    """The row with occupancy 60 must predict what occupancy at the training
    maximum predicts; without the by= clamp it is scaled by 60."""
    with pytest.warns(ExtrapolationWarning):
        clipped = np.asarray(varying.predict(newdata=far_occupancy,
                                             extrapolation="clip")).ravel()
    edge = np.asarray(varying.predict(newdata=at_the_boundary)).ravel()
    assert clipped[1] == pytest.approx(edge[0], abs=1e-12)


def test_the_unclipped_prediction_moves_with_the_by_variable(varying, far_occupancy):
    """The reason clipping matters: linear continuation carries the effect on
    at the out-of-range occupancy, so the allowed and clipped predictions
    differ materially."""
    with pytest.warns(ExtrapolationWarning):
        clipped = np.asarray(varying.predict(newdata=far_occupancy,
                                             extrapolation="clip")).ravel()
    allowed = np.asarray(varying.predict(newdata=far_occupancy,
                                         extrapolation="allow")).ravel()
    assert abs(allowed[1] - clipped[1]) > 1.0
    assert allowed[0] == pytest.approx(clipped[0], abs=1e-12)


def test_clip_clamps_the_by_variable_in_contributions(varying, far_occupancy, at_the_boundary):
    """`predict(which=)` rebuilds the basis itself; the clamp must reach it."""
    with pytest.warns(ExtrapolationWarning):
        clipped = np.asarray(varying.predict(newdata=far_occupancy, which=0,
                                             extrapolation="clip")).ravel()
    edge = np.asarray(varying.predict(newdata=at_the_boundary, which=0)).ravel()
    assert clipped[1] == pytest.approx(edge[0], abs=1e-12)


def test_clip_clamps_the_by_variable_in_predict_over_mstop(
        varying, far_occupancy, at_the_boundary):
    with pytest.warns(ExtrapolationWarning):
        grid = varying.predict_over_mstop([50, 100], newdata=far_occupancy,
                                          extrapolation="clip", output="numpy")
    edge = varying.predict_over_mstop([50, 100], newdata=at_the_boundary,
                                      output="numpy")
    np.testing.assert_allclose(grid[:, 1], edge[:, 0], atol=1e-12)


def test_clip_leaves_in_range_by_values_alone(varying, housing):
    inside = {"age": housing["age"][:20], "occ": housing["occ"][:20]}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        clipped = varying.predict(newdata=inside, extrapolation="clip")
    np.testing.assert_array_equal(clipped, varying.predict(newdata=inside, extrapolation="allow"))


def test_clip_clamps_the_by_variable_for_gamboostlss(housing, at_the_boundary, far_occupancy):
    lss = mb.gamboostLSS({"mu": "y ~ bbs(age, df=4, by=occ)", "sigma": "y ~ bbs(age, df=4)"},
                         data=housing, family=mb.GaussianLSS(), mstop=60)
    with pytest.warns(ExtrapolationWarning) as caught:
        clipped = np.asarray(lss.predict(newdata=far_occupancy, parameter="mu",
                                         type="link", extrapolation="clip")).ravel()
    assert any("bbs(age), by=occ [mu]" in text for text in _messages(caught))
    edge = np.asarray(lss.predict(newdata=at_the_boundary, parameter="mu",
                                  type="link")).ravel()
    assert clipped[1] == pytest.approx(edge[0], abs=1e-12)


# ─── 3. extrapolation_report ───────────────────────────────────────


def test_the_report_has_one_row_for_the_by_covariate(varying, far_occupancy, housing):
    report = varying.extrapolation_report(far_occupancy, output="records")
    assert len(report) == 1
    row = report[0]
    assert row["term"] == "bbs(age), by=occ"
    assert row["covariate"] == "occ"
    assert row["kind"] == "spline basis"
    assert row["n_below"] == 0 and row["n_above"] == 1 and row["n_outside"] == 1
    assert row["train_low"] == pytest.approx(housing["occ"].min())
    assert row["train_high"] == pytest.approx(housing["occ"].max())
    assert row["new_high"] == pytest.approx(60.0)
    expected = (60.0 - housing["occ"].max()) / (housing["occ"].max() - housing["occ"].min())
    assert row["worst_distance"] == pytest.approx(expected, rel=1e-9)


def test_the_report_counts_rows_below_the_range(varying, housing):
    low = {"age": np.array([10.0, 10.0, 10.0]),
           "occ": np.array([0.2, 0.5, 3.0])}
    report = varying.extrapolation_report(low, output="records")
    row = next(r for r in report if r["covariate"] == "occ")
    assert row["n_below"] == 2 and row["n_above"] == 0


def test_the_report_lists_the_spline_covariate_too(varying):
    newdata = {"age": np.array([80.0]), "occ": np.array([3.0])}
    report = varying.extrapolation_report(newdata, output="records")
    assert [(r["term"], r["covariate"]) for r in report] == [("bbs(age), by=occ", "age")]


def test_the_report_is_empty_with_the_columns_when_nothing_is_outside(varying, housing):
    inside = {"age": housing["age"][:10], "occ": housing["occ"][:10]}
    frame = varying.extrapolation_report(inside)
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 0
    assert frame.columns == list(REPORT_COLUMNS)


def test_the_report_returns_a_polars_frame_by_default(varying, far_occupancy):
    frame = varying.extrapolation_report(far_occupancy)
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == list(REPORT_COLUMNS)
    assert frame["covariate"].to_list() == ["occ"]


def test_the_report_agrees_with_range_report(varying, far_occupancy):
    """It is a filter of the same computation, not a second one."""
    full = {(r["learner"], r["variable"]): r
            for r in varying.range_report(far_occupancy, output="records")}
    report = varying.extrapolation_report(far_occupancy, output="records")
    for row in report:
        source = full[(row["term"], row["covariate"])]
        assert row["worst_distance"] == source["worst_distance"]
        assert row["n_outside"] == source["n_outside"]


def test_the_report_rejects_an_unknown_output(varying, far_occupancy):
    with pytest.raises(ValueError, match="output must be"):
        varying.extrapolation_report(far_occupancy, output="excel")


def test_the_lss_report_names_the_parameter_and_skips_linear_parameters(housing):
    lss = mb.gamboostLSS({"mu": "y ~ bbs(age, df=4, by=occ)", "sigma": "y ~ bols(occ)"},
                         data=housing, family=mb.GaussianLSS(), mstop=40)
    far = {"age": np.array([10.0]), "occ": np.array([60.0])}
    report = lss.extrapolation_report(far, output="records")
    assert [(r["parameter"], r["term"], r["covariate"]) for r in report] == [
        ("mu", "bbs(age), by=occ", "occ"),
    ]
    assert report[0]["n_above"] == 1


def test_the_lss_report_is_empty_inside_the_range(housing):
    lss = mb.gamboostLSS({"mu": "y ~ bbs(age, df=4)", "sigma": "y ~ bbs(age, df=4)"},
                         data=housing, family=mb.GaussianLSS(), mstop=20)
    frame = lss.extrapolation_report({"age": np.array([20.0])})
    assert frame.height == 0 and frame.columns == list(REPORT_COLUMNS)


def test_the_glm_report_lists_linear_predictors_outside_their_range():
    rng = np.random.default_rng(0)
    n = 200
    w = rng.normal(size=n)
    x = rng.uniform(0.0, 1.0, n)
    data = {"w": w, "x": x, "y": w + x + rng.normal(size=n) * 0.3}
    glm = mb.glmboost("y ~ bols(w) + bols(x)", data=data, control=mb.boost_control(50))
    span = w.max() - w.min()
    newdata = {"w": np.array([w.max() + span]), "x": np.array([0.5])}
    report = glm.extrapolation_report(newdata, output="records")
    assert [(r["term"], r["covariate"], r["kind"]) for r in report] == [
        ("bols(w)", "w", "linear"),
    ]
    assert report[0]["worst_distance"] == pytest.approx(1.0, rel=1e-9)


def test_the_glm_report_needs_the_stored_training_data():
    rng = np.random.default_rng(0)
    data = {"w": rng.normal(size=50), "y": rng.normal(size=50)}
    glm = mb.glmboost("y ~ bols(w)", data=data, control=mb.boost_control(10))
    assert isinstance(glm, GLMBoostModel)
    glm._data = None
    with pytest.raises(ValueError, match="stored training data"):
        glm.extrapolation_report({"w": np.array([0.0])})


# ─── the warning and the report share one computation ─────────────


def test_the_warning_rows_are_the_report_rows(varying, far_occupancy):
    """The warning and the report are built from the same rows."""
    with pytest.warns(ExtrapolationWarning) as caught:
        varying.predict(newdata=far_occupancy)
    (warning,) = [w.message for w in caught
                  if issubclass(w.category, ExtrapolationWarning)]
    warned = [(r.learner, r.variable, r.n_outside) for r in warning.rows]
    reported = [(r["term"], r["covariate"], r["n_outside"])
                for r in varying.extrapolation_report(far_occupancy, output="records")]
    assert warned == reported
