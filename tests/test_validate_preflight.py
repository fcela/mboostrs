"""``validate(model, newdata)`` lists the problems predict would meet (#293).

Each issue kind has a test that builds the condition on a gamboost model and on a
gamboostLSS model. Where predict refuses or warns about the same condition, the
test also runs predict and checks that it does what the record says. The second
half covers the range-policy coverage that #293 asked for: contribution
predictions, Multinomial predictions, the two coordinates of ``bspatial`` and
the margins of ``%X%`` terms.
"""

import json
import warnings

import numpy as np
import pytest

import mboostrs as mb
from mboostrs._extrapolation import ExtrapolationWarning
from mboostrs._preflight import ISSUE_KINDS, RECORD_COLUMNS


def _issues(model, newdata, **kwargs):
    return mb.validate(model, newdata, output="records", **kwargs)


def _kinds(issues):
    return [issue["kind"] for issue in issues]


def _one(issues, kind, **match):
    """The single issue of ``kind`` whose fields include ``match``."""
    found = [issue for issue in issues
             if issue["kind"] == kind
             and all(issue.get(key) == value for key, value in match.items())]
    assert len(found) == 1, f"expected one {kind} issue {match}, got {issues}"
    return found[0]


def _extrapolation_warnings(caught):
    return [w for w in caught if issubclass(w.category, ExtrapolationWarning)]


@pytest.fixture(scope="module")
def rows():
    rng = np.random.default_rng(293)
    n = 300
    x = rng.uniform(0.0, 10.0, n)
    z = rng.uniform(0.0, 5.0, n)
    g = rng.choice(["a", "b", "c"], n)
    y = np.sin(x) + 0.3 * z + (g == "a") * 0.5 + rng.normal(size=n) * 0.1
    return {"x": x, "z": z, "g": g, "y": y, "junk": rng.normal(size=n)}


@pytest.fixture(scope="module")
def gam(rows):
    """A gamboost over a spline, a random effect and a linear term."""
    data = {k: rows[k] for k in ("x", "z", "g", "y")}
    return mb.gamboost("y ~ bbs(x, df=4) + brandom(g) + bols(z)", data=data,
                       control=mb.boost_control(60, 0.1))


@pytest.fixture(scope="module")
def lss(rows):
    """A gamboostLSS: a spline on mu, a bols and a random effect on sigma."""
    data = {k: rows[k] for k in ("x", "z", "g", "y")}
    return mb.gamboostLSS({"mu": "y ~ bbs(x, df=3)",
                           "sigma": "y ~ bols(z) + brandom(g)"},
                          data=data, family=mb.GaussianLSS(), mstop=30)


def _clean(rows, n=5):
    return {"x": rows["x"][:n], "z": rows["z"][:n], "g": rows["g"][:n]}


# ─── the record and the return type ───────────────────────────────────


def test_a_clean_frame_has_no_issues(gam, lss, rows):
    assert _issues(gam, _clean(rows)) == []
    assert _issues(lss, _clean(rows)) == []


def test_every_record_has_the_documented_keys_and_plain_types(gam, rows):
    newdata = dict(_clean(rows), x=np.array([np.nan, 50.0, 2.0, 3.0, 4.0]))
    issues = _issues(gam, newdata)
    assert issues
    for issue in issues:
        assert set(issue) == set(RECORD_COLUMNS)
        assert issue["kind"] in ISSUE_KINDS
        assert issue["severity"] in ("error", "warning")
        assert isinstance(issue["suggested_policy"], str) and issue["suggested_policy"]
    json.dumps(issues)  # every value is a plain type


def test_the_default_output_is_a_frame_with_one_row_per_issue(gam, rows):
    newdata = dict(_clean(rows), g=np.array(["a", "q", "b", "a", "a"]))
    frame = mb.validate(gam, newdata)
    assert list(frame.columns) == list(RECORD_COLUMNS)
    assert frame.shape[0] == len(_issues(gam, newdata))


def test_an_empty_result_is_an_empty_frame_with_the_columns(gam, rows):
    frame = mb.validate(gam, _clean(rows))
    assert frame.shape[0] == 0
    assert list(frame.columns) == list(RECORD_COLUMNS)


def test_the_model_and_newdata_are_checked_before_anything_else(gam):
    with pytest.raises(TypeError, match="newdata"):
        mb.validate(gam, None)
    with pytest.raises(TypeError, match="gamboost, glmboost or gamboostLSS"):
        mb.validate(object(), {"x": [1.0]})


# ─── missing and extra columns ────────────────────────────────────────


def test_a_missing_column_is_an_error_for_gamboost_and_gamboostlss(gam, lss, rows):
    newdata = {"x": rows["x"][:3], "g": rows["g"][:3]}
    for model in (gam, lss):
        issues = _issues(model, newdata)
        missing = _one(issues, "missing_column", column="z")
        assert missing["severity"] == "error"
        assert "bols(z)" in missing["term"] or "z" in missing["message"]
        assert "add column 'z'" in missing["suggested_policy"]
    with pytest.raises(Exception, match="z"):
        gam.predict(newdata=newdata)


def test_missing_columns_skip_the_extrapolation_check_and_say_so(gam, rows):
    issues = _issues(gam, {"x": np.array([50.0])})
    skipped = _one(issues, "check_skipped")
    assert skipped["severity"] == "warning"
    assert "missing from newdata" in skipped["message"]
    assert "extrapolation" not in _kinds(issues)


def test_an_extra_column_is_a_warning_and_the_response_is_not_listed(gam, rows):
    newdata = dict(_clean(rows), junk=rows["junk"][:5], y=rows["y"][:5])
    issues = _issues(gam, newdata)
    extra = _one(issues, "extra_column", column="junk")
    assert extra["severity"] == "warning"
    assert all(issue.get("column") != "y" for issue in issues)
    gam.predict(newdata=newdata)  # predict accepts it


def test_an_extra_column_is_listed_for_gamboostlss(lss, rows):
    newdata = dict(_clean(rows), junk=rows["junk"][:5])
    _one(_issues(lss, newdata), "extra_column", column="junk")


# ─── missing values ───────────────────────────────────────────────────


def test_missing_values_are_an_error_predict_refuses(gam, rows):
    newdata = dict(_clean(rows), x=np.array([np.nan, 1.0, 2.0, 3.0, 4.0]))
    issues = _issues(gam, newdata)
    na = _one(issues, "na", column="x")
    assert na["severity"] == "error"
    assert na["count"] == 1 and na["n_rows"] == 5
    skipped = _one(issues, "check_skipped")
    assert "missing values in 'x'" in skipped["message"]
    assert "extrapolation" not in _kinds(issues)
    with pytest.raises(ValueError, match="finite"):
        gam.predict(newdata=newdata)


def test_glmboost_is_checked_for_missing_and_linear_excursions(rows):
    """A glmboost has only linear terms: its excursions are reported, never clipped."""
    data = {"x": rows["x"], "z": rows["z"], "y": rows["y"]}
    glm = mb.glmboost("y ~ bols(x) + bols(z)", data=data, control=mb.boost_control(40, 0.1))
    assert _issues(glm, {"x": rows["x"][:3], "z": rows["z"][:3]}) == []
    _one(_issues(glm, {"x": rows["x"][:3]}), "missing_column", column="z")
    ext = _one(_issues(glm, {"x": np.array([50.0]), "z": np.array([1.0])}),
               "extrapolation", column="x")
    assert ext["term"] == "bols(x)"
    assert "no extrapolation= policy applies" in ext["suggested_policy"]


def test_missing_values_are_an_error_for_gamboostlss(lss, rows):
    newdata = dict(_clean(rows), z=np.array([np.nan, 1.0, 2.0, 3.0, 4.0]))
    na = _one(_issues(lss, newdata), "na", column="z")
    assert na["severity"] == "error"


def test_an_imputed_column_is_a_warning_because_predict_fills_it(rows):
    data = {k: rows[k] for k in ("x", "z", "g", "y")}
    data["x"] = data["x"].copy()
    data["x"][:10] = np.nan
    model = mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=data, na_action="impute",
                        control=mb.boost_control(40, 0.1))
    newdata = {"x": np.array([np.nan, 4.0]), "z": np.array([1.0, 2.0])}
    na = _one(_issues(model, newdata), "na", column="x")
    assert na["severity"] == "warning"
    assert "imputes" in na["message"]
    model.predict(newdata=newdata)  # predict fills the row, so it runs


# ─── unseen levels ────────────────────────────────────────────────────


def test_an_unseen_level_is_an_error_that_names_the_level_and_the_count(gam, rows):
    newdata = dict(_clean(rows), g=np.array(["a", "zz", "zz", "b", "a"]))
    unseen = _one(_issues(gam, newdata), "unseen_level", column="g")
    assert unseen["severity"] == "error"
    assert unseen["levels"] == ["zz"]
    assert unseen["count"] == 2
    assert "unseen_levels='average'" in unseen["suggested_policy"]
    with pytest.raises(ValueError, match="never saw"):
        gam.predict(newdata=newdata)
    gam.predict(newdata=newdata, unseen_levels="average")  # the policy it names


def test_an_unseen_level_on_a_factor_bols_names_the_average_policy_that_works(rows):
    """A factor bols is fitted as level indicators, which the average policy
    predicts; the record names that policy and predict accepts it."""
    data = {k: rows[k] for k in ("x", "g", "y")}
    model = mb.gamboost("y ~ bbs(x, df=4) + bols(g)", data=data,
                        control=mb.boost_control(40, 0.1))
    newdata = {"x": np.array([1.0]), "g": np.array(["zz"])}
    unseen = _one(_issues(model, newdata), "unseen_level", column="g")
    assert "unseen_levels='average'" in unseen["suggested_policy"]
    model.predict(newdata=newdata, unseen_levels="average")


def test_an_unseen_level_is_listed_for_gamboostlss_random_effects(lss, rows):
    newdata = dict(_clean(rows), g=np.array(["a", "b", "q", "a", "b"]))
    unseen = _one(_issues(lss, newdata), "unseen_level", column="g")
    assert unseen["levels"] == ["q"] and unseen["count"] == 1
    lss.predict(newdata=newdata, unseen_levels="average")


def test_a_dtype_change_is_an_error_with_no_policy_offered(rows):
    data = {"x": rows["x"], "g": np.array([str(i % 3) for i in range(300)]), "y": rows["y"]}
    model = mb.gamboost("y ~ bbs(x, df=4) + brandom(g)", data=data,
                        control=mb.boost_control(40, 0.1))
    newdata = {"x": np.array([1.0, 2.0]), "g": np.array([0, 1])}
    issues = _issues(model, newdata)
    mismatch = _one(issues, "unseen_level", column="g")
    assert mismatch["severity"] == "error"
    assert "dtype" in mismatch["suggested_policy"] or "type" in mismatch["suggested_policy"]


# ─── row count and offset ─────────────────────────────────────────────


def test_no_rows_and_unequal_columns_are_row_count_errors(gam):
    empty = _issues(gam, {"x": np.array([]), "z": np.array([]), "g": np.array([], dtype=object)})
    assert _one(empty, "row_count", count=0)["severity"] == "error"
    ragged = _issues(gam, {"x": np.array([1.0, 2.0]), "z": np.array([1.0]),
                           "g": np.array(["a", "b"])})
    assert _one(ragged, "row_count")["severity"] == "error"
    assert "different lengths" in _one(ragged, "row_count")["message"]


def test_a_per_observation_offset_must_be_supplied_and_match_the_rows(rows):
    off = np.linspace(0.5, 1.5, 300)
    model = mb.gamboost("y ~ bbs(x, df=4)", data={"x": rows["x"], "y": rows["y"]},
                        offset=off, control=mb.boost_control(30, 0.1))
    newdata = {"x": rows["x"][:3]}
    missing = _one(_issues(model, newdata), "offset")
    assert missing["severity"] == "error"
    with pytest.raises(ValueError, match="offset"):
        model.predict(newdata=newdata)
    assert _issues(model, newdata, offset=off[:3]) == []
    wrong = _one(_issues(model, newdata, offset=off[:2]), "offset")
    assert "2 entries" in wrong["message"]


def test_an_offset_given_to_gamboostlss_is_an_error(lss, rows):
    issues = _issues(lss, _clean(rows), offset=0.0)
    off = _one(issues, "offset")
    assert off["severity"] == "error"
    assert "takes no offset" in off["message"]


# ─── extrapolation ────────────────────────────────────────────────────


def test_a_spline_excursion_is_a_warning_with_the_distance_and_the_clip_policy(gam, rows):
    newdata = dict(_clean(rows), x=np.array([1.0, 1.0, 1.0, 1.0, 40.0]))
    ext = _one(_issues(gam, newdata), "extrapolation", column="x")
    assert ext["severity"] == "warning"
    assert ext["count"] == 1 and ext["n_rows"] == 5
    assert ext["term"] == "bbs(x)" and ext["parameter"] == "f"
    assert ext["worst_distance"] > 1.0
    assert "clip" in ext["suggested_policy"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam.predict(newdata=newdata)
    assert _extrapolation_warnings(caught)


def test_a_clipped_excursion_is_predicted_at_the_training_edge(gam, rows):
    far = dict(_clean(rows, 1), x=np.array([40.0]))
    edge = dict(_clean(rows, 1), x=np.array([float(rows["x"].max())]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        clipped = gam.predict(newdata=far, extrapolation="clip")
    np.testing.assert_allclose(clipped, gam.predict(newdata=edge), rtol=1e-12)


def test_a_gamboostlss_excursion_names_the_parameter(lss, rows):
    newdata = dict(_clean(rows), x=np.array([1.0, 1.0, 1.0, 1.0, 40.0]))
    ext = _one(_issues(lss, newdata), "extrapolation", column="x")
    assert ext["parameter"] == "mu"
    assert ext["term"] == "bbs(x)"


def test_a_linear_excursion_is_reported_with_no_policy_to_apply(gam, rows):
    newdata = dict(_clean(rows), z=np.array([0.5, 0.5, 0.5, 0.5, 50.0]))
    ext = _one(_issues(gam, newdata), "extrapolation", column="z")
    assert "no extrapolation= policy applies" in ext["suggested_policy"]


def test_a_tensor_margin_excursion_is_reported_by_the_term_as_written(rows):
    data = {"x": rows["x"], "z": rows["z"], "y": rows["y"]}
    model = mb.gamboost("y ~ bbs(x, df=3) %X% bbs(z, df=3)", data=data,
                        control=mb.boost_control(40, 0.1))
    newdata = {"x": np.array([5.0, 5.0]), "z": np.array([2.0, 30.0])}
    ext = _one(_issues(model, newdata), "extrapolation", column="z")
    assert ext["term"] == "bbs(x, df=3) %X% bbs(z, df=3)"


def test_a_bspatial_coordinate_excursion_is_reported(rows):
    data = {"x": rows["x"], "z": rows["z"], "y": rows["y"]}
    model = mb.gamboost("y ~ bspatial(x, z, df=6)", data=data, control=mb.boost_control(40, 0.1))
    newdata = {"x": np.array([5.0, 50.0]), "z": np.array([2.0, 2.0])}
    ext = _one(_issues(model, newdata), "extrapolation", column="x")
    assert ext["term"] == "bspatial(x,z)"


# ─── terms with no range check ────────────────────────────────────────


def test_a_btree_term_is_listed_as_not_range_checked(rows):
    data = {"x": rows["x"], "z": rows["z"], "y": rows["y"]}
    model = mb.gamboost("y ~ btree(x, z, maxdepth=2)", data=data, control=mb.boost_control(30, 0.1))
    listed = _one(_issues(model, {"x": rows["x"][:3], "z": rows["z"][:3]}), "not_range_checked")
    assert listed["severity"] == "warning"
    assert "no extrapolation= policy applies" in listed["suggested_policy"]


def test_a_btree_term_of_gamboostlss_is_listed_under_its_parameter(rows):
    data = {"x": rows["x"], "z": rows["z"], "y": rows["y"]}
    model = mb.gamboostLSS({"mu": "y ~ bbs(x, df=3)", "sigma": "y ~ btree(z, maxdepth=2)"},
                           data=data, family=mb.GaussianLSS(), mstop=20)
    listed = _one(_issues(model, {"x": rows["x"][:3], "z": rows["z"][:3]}), "not_range_checked")
    assert listed["parameter"] == "sigma"


# ─── the range policy now covers the paths #293 listed ────────────────


def test_contribution_predictions_warn_and_clip_like_predict(gam, rows):
    far = dict(_clean(rows, 2), x=np.array([5.0, 40.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam.prediction_contributions(newdata=far, which=0)
    assert _extrapolation_warnings(caught)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam.prediction_contributions(newdata=far, which=0, extrapolation="allow")
    assert not _extrapolation_warnings(caught)
    clipped = gam.prediction_contributions(newdata=far, which=0, extrapolation="clip")
    edge = dict(_clean(rows, 1), x=np.array([float(rows["x"].max())]))
    at_edge = gam.prediction_contributions(newdata=edge, which=0)
    np.testing.assert_allclose(clipped[1], at_edge[0], rtol=1e-12)


@pytest.fixture(scope="module")
def multinomial(rows):
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    labels = np.array(["p", "q", "r"])[rng.integers(0, 3, n)]
    data = {"x": x, "z": z, "y": labels}
    return mb.gamboost("y ~ bbs(x) + bols(z)", data=data, family=mb.Multinomial(),
                       control=mb.boost_control(60, 0.1)), data


def test_multinomial_predictions_warn_on_an_excursion(multinomial):
    model, _data = multinomial
    far = {"x": np.array([0.0, 40.0]), "z": np.array([0.0, 0.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(newdata=far)
    assert _extrapolation_warnings(caught)
    issues = _issues(model, far)
    assert _one(issues, "extrapolation", column="x")["count"] == 1


def test_multinomial_clip_predicts_at_the_training_edge(multinomial):
    model, data = multinomial
    far = {"x": np.array([40.0]), "z": np.array([0.0])}
    edge = {"x": np.array([float(data["x"].max())]), "z": np.array([0.0])}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        clipped = model.predict(newdata=far, extrapolation="clip")
    np.testing.assert_allclose(clipped, model.predict(newdata=edge), rtol=1e-10)


@pytest.fixture(scope="module")
def surface(rows):
    data = {"x": rows["x"], "z": rows["z"], "y": rows["y"]}
    return mb.gamboost("y ~ bspatial(x, z, df=6)", data=data, control=mb.boost_control(40, 0.1))


def test_bspatial_predict_warns_and_clips_each_coordinate(surface, rows):
    far = {"x": np.array([5.0, 50.0]), "z": np.array([2.0, 2.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        surface.predict(newdata=far)
    assert _extrapolation_warnings(caught)
    clipped = surface.predict(newdata=far, extrapolation="clip")
    edge = {"x": np.array([float(rows["x"].max())]), "z": np.array([2.0])}
    np.testing.assert_allclose(clipped[1], surface.predict(newdata=edge)[0], rtol=1e-12)


def test_bspatial_is_in_the_extrapolation_report(surface):
    far = {"x": np.array([5.0, 5.0]), "z": np.array([2.0, 50.0])}
    report = surface.extrapolation_report(far, output="records")
    assert [row["covariate"] for row in report] == ["z"]
    assert report[0]["term"] == "bspatial(x,z)"


@pytest.fixture(scope="module")
def tensor(rows):
    data = {"x": rows["x"], "z": rows["z"], "y": rows["y"]}
    return mb.gamboost("y ~ bbs(x, df=3) %X% bbs(z, df=3)", data=data,
                       control=mb.boost_control(40, 0.1))


def test_tensor_margin_predict_warns_and_clip_holds_the_margin_flat(tensor, rows):
    far = {"x": np.array([5.0, 5.0]), "z": np.array([2.0, 30.0])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensor.predict(newdata=far)
    assert _extrapolation_warnings(caught)
    clipped = tensor.predict(newdata=far, extrapolation="clip")
    edge = {"x": np.array([5.0]), "z": np.array([float(rows["z"].max())])}
    np.testing.assert_allclose(clipped[1], tensor.predict(newdata=edge)[0], rtol=1e-12)


def test_tensor_margin_inside_its_range_is_quiet_on_a_log_margin(rows):
    """A margin on log(a) is measured on the log scale its knots are on."""
    rng = np.random.default_rng(11)
    a = rng.uniform(1.0, 50.0, 300)
    b = rng.uniform(0.0, 1.0, 300)
    data = {"a": a, "b": b, "y": np.log(a) * b + rng.normal(scale=0.2, size=300)}
    model = mb.gamboost("y ~ bbs(log(a), df=2.5, knots=5) %X% bbs(b, df=2.5, knots=5)",
                        data=data, control=mb.boost_control(40, 0.1))
    inside = {"a": np.array([2.0, 30.0]), "b": np.array([0.2, 0.9])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(newdata=inside)
    assert not _extrapolation_warnings(caught)
    assert _issues(model, inside) == []


def test_lss_partial_effects_allow_by_default_and_warn_on_request(lss, rows):
    far = {"x": np.array([5.0, 40.0]), "z": np.array([1.0, 1.0]), "g": np.array(["a", "a"])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lss.partial_effects("mu", which=0, newdata=far)
    assert not _extrapolation_warnings(caught)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lss.partial_effects("mu", which=0, newdata=far, extrapolation="warn")
    assert _extrapolation_warnings(caught)
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        lss.partial_effects("mu", which=0, newdata=far, extrapolation="nope")


def test_validate_and_predict_agree_on_a_refused_row(gam, rows):
    """A row validate calls an error is one predict refuses; a clean frame
    is one predict accepts."""
    refused = dict(_clean(rows), g=np.array(["a", "zz", "b", "c", "a"]))
    assert any(issue["severity"] == "error" for issue in _issues(gam, refused))
    with pytest.raises(ValueError):
        gam.predict(newdata=refused)
    clean = _clean(rows)
    assert _issues(gam, clean) == []
    gam.predict(newdata=clean)


# ─── unused columns: a column read through a derived column is not unused ───


def test_a_source_read_through_a_recorded_transform_is_not_listed_as_unused():
    """The source of ``log(households)`` is read through the recipe's ``log__households``."""
    rng = np.random.default_rng(7)
    n = 200
    households = rng.gamma(2.0, 100.0, n)
    data = {"y": np.log(households) + rng.normal(0, 0.2, n), "households": households,
            "x": rng.normal(size=n), "junk": rng.normal(size=n)}
    model = mb.gamboost("y ~ bbs(x, df=3) + log(households)", data=data,
                        control=mb.boost_control(30, 0.1))
    newdata = {k: v[:10] for k, v in data.items() if k != "y"}
    issues = _issues(model, newdata)

    through = _one(issues, "read_through_transform")
    assert through["message"] == "read through derived columns: households (log__households)"
    extra = _one(issues, "extra_column", column="junk")
    assert "households" not in extra["message"]


def test_a_source_read_through_a_hand_built_derived_column_is_not_listed_as_unused():
    """No recipe entry exists for ``households__log``, so the name links it to ``households``."""
    rng = np.random.default_rng(8)
    n = 200
    households = rng.gamma(2.0, 100.0, n)
    data = {"y": rng.normal(size=n), "households": households,
            "households__log": np.log(households), "x": rng.normal(size=n),
            "junk": rng.normal(size=n)}
    model = mb.gamboost("y ~ bbs(x, df=3) + bols(households__log)", data=data,
                        control=mb.boost_control(30, 0.1))
    newdata = {k: v[:10] for k, v in data.items() if k != "y"}
    issues = _issues(model, newdata)

    through = _one(issues, "read_through_transform")
    assert through["message"] == "read through derived columns: households (households__log)"
    extra = _one(issues, "extra_column", column="junk")
    assert extra["count"] == 1
    assert "households" not in extra["message"]


def test_a_column_no_term_reads_and_no_derived_column_is_still_listed(rows):
    model = _gam_for_unused(rows)
    newdata = {"x": rows["x"][:5], "z": rows["z"][:5], "junk": rows["junk"][:5]}
    issues = _issues(model, newdata)

    extra = _one(issues, "extra_column", column="junk")
    assert extra["message"].endswith("predict ignores it: 'junk'")
    assert not [i for i in issues if i["kind"] == "read_through_transform"]


def _gam_for_unused(rows):
    data = {k: rows[k] for k in ("x", "z", "g", "y")}
    return mb.gamboost("y ~ bbs(x, df=4) + bols(z)", data=data,
                       control=mb.boost_control(30, 0.1))
