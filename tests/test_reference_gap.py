"""reference_gap: the held-out gap to a btree reference, and where it comes from.

The synthetic problems have a known source of structure, so each test checks
that the gap and the candidate ranking find it, and that a null response finds
nothing. Fits are short (a few hundred steps on a few hundred rows) so the file
runs in seconds.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import mboostrs as mb
from mboostrs.families import Gaussian
from mboostrs.folds import FoldScheme, rolling_folds
from mboostrs.reference_gap import (
    ReferenceGapResult,
    _interaction_term,
    _lag_and_rolling_features,
    _make_folds,
    reference_gap,
)

MODEL = "y ~ bbs(x1, df=4) + bbs(x2, df=4) + bols(g)"


def _times(n):
    start = datetime(2020, 1, 1)
    return [start + timedelta(hours=int(i)) for i in range(n)]


def _interaction_frame(n=900, seed=3):
    """A response with an x1-by-x2 interaction the additive model cannot express."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    g = rng.choice(["a", "b", "c"], n)
    y = np.sin(2 * x1) + 1.5 * x1 * x2 + 0.9 * x2 * (g == "a") + rng.normal(0, 0.3, n)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def _lag_frame(n=900, seed=4):
    """A response with an x1-by-x2 interaction, and the covariate `temp` three
    hours back. The formula names x1 and x2 but not temp, so only a lag
    candidate can carry the `temp` structure."""
    rng = np.random.default_rng(seed)
    temp = np.cumsum(rng.normal(0, 0.3, n))
    lag3 = np.concatenate([np.zeros(3), temp[:-3]])
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    y = np.sin(2 * x1) + 1.5 * x1 * x2 + 1.0 * lag3 + rng.normal(0, 0.3, n)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "temp": temp, "ts": _times(n)})


def _null_frame(n=600, seed=5):
    """A response that is independent of every covariate."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "y": rng.normal(0, 1, n),
        "x1": rng.uniform(-2, 2, n),
        "x2": rng.uniform(-2, 2, n),
        "g": rng.choice(["a", "b", "c"], n),
    })


# ---------------------------------------------------------------------------
# The result and the gap


def test_returns_the_documented_result_fields():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=3,
                        max_mstop=200, top=0)
    assert isinstance(res, ReferenceGapResult)
    for name in ("gap", "gap_se", "reference_loss", "model_loss", "candidates",
                 "closed_share", "material", "relative_gain", "folds", "family",
                 "model_formula", "reference_formula", "notes", "data"):
        assert hasattr(res, name), name
    assert isinstance(res.candidates, pl.DataFrame)
    assert res.folds == 3
    assert res.reference_formula.startswith("y ~ btree(x1, x2, g, maxdepth=4")
    assert np.isfinite(res.gap) and np.isfinite(res.gap_se) and res.gap_se > 0


def test_gap_is_the_paired_fold_difference_and_its_sign_says_which_is_better():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=4,
                        max_mstop=300, top=0)
    # Losses are risks: a positive gap means the reference's loss is lower.
    assert res.gap == pytest.approx(res.model_loss - res.reference_loss, abs=0.5)
    assert res.gap > 0


def test_structured_response_gives_a_material_gap_and_the_interaction_first():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=4,
                        max_mstop=400, top=3)
    assert res.material
    assert res.gap > 2 * res.gap_se
    assert res.candidates.height >= 1
    assert res.candidates["term"][0] == "bbs(x1) %X% bbs(x2)"
    assert res.candidates["kind"][0] == "interaction"
    assert res.candidates["measured_gain"][0] > 2 * res.candidates["gain_se"][0]
    assert res.closed_share is not None and res.closed_share > 0.5
    assert res.combined_formula.startswith(MODEL)


def test_null_response_finds_no_material_gap_and_no_clear_candidate():
    res = reference_gap(MODEL, _null_frame(), family=Gaussian(), folds=4,
                        max_mstop=300, top=5)
    assert not res.material
    assert res.closed_share is None
    # Candidates are measured whatever the gap; none of them gains clearly.
    clear = res.candidates.filter(pl.col("measured_gain") > 2 * pl.col("gain_se"))
    assert clear.height == 0
    assert "not above 2 standard errors" in res.describe()


def test_named_candidates_are_measured_even_when_the_gap_is_not_material():
    res = reference_gap(MODEL, _null_frame(), family=Gaussian(), folds=4,
                        max_mstop=150, candidates=["bbs(x1) %X% bbs(x2)"], top=5)
    assert not res.material
    assert res.candidates["term"].to_list() == ["bbs(x1) %X% bbs(x2)"]
    assert res.closed_share is None
    assert "Add a candidate only if its measured gain is above 2 standard" in res.describe()


def test_candidates_frame_has_its_schema_when_nothing_is_measured():
    res = reference_gap(MODEL, _null_frame(), family=Gaussian(), folds=3,
                        max_mstop=150, top=0)
    assert res.candidates.height == 0
    assert set(res.candidates.columns) == {
        "rank", "kind", "term", "source", "screen_share", "measured_gain", "gain_se",
        "relative_gain", "closed_share", "mstop", "at_cap"}


def test_top_limits_the_number_of_refits():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=4,
                        max_mstop=300, top=1)
    assert res.candidates.height <= 1


def test_model_input_supplies_its_formula_and_family():
    frame = _interaction_frame()
    fit = mb.gamboost(MODEL, data=frame, family=Gaussian(),
                      control=mb.boost_control(mstop=50, nu=0.1))
    res = reference_gap(fit, frame, folds=3, max_mstop=150, top=0)
    assert res.model_formula == MODEL
    assert res.family == "gaussian"


def test_candidate_terms_can_be_supplied_by_the_caller():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=4,
                        max_mstop=300, candidates=["bbs(x1) %X% bbs(x2)"], top=5)
    assert res.material
    assert res.candidates["term"].to_list() == ["bbs(x1) %X% bbs(x2)"]
    assert res.candidates["kind"][0] == "user"


def test_describe_and_to_dict_are_plain_and_json_ready():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=4,
                        max_mstop=300, top=2)
    text = res.describe()
    assert "Held-out loss" in text and "Reference overfit check" in text
    assert "Next:" in text
    record = res.to_dict()
    assert "data" not in record and "fold_losses" not in record
    json.dumps(record)
    with_arrays = res.to_dict(include_arrays=True)
    assert len(with_arrays["fold_losses"]["model"]) == 4
    assert len(with_arrays["fold_losses"]["reference"]) == 4
    json.dumps(with_arrays)


# ---------------------------------------------------------------------------
# Folds


def test_fewer_than_three_folds_is_noted():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=2,
                        max_mstop=150, top=0)
    assert any("fewer than 3" in note for note in res.notes)


def test_at_cap_step_counts_are_reported_as_unfinished_tuning():
    res = reference_gap(MODEL, _interaction_frame(), family=Gaussian(), folds=3,
                        max_mstop=100, top=0)
    assert res.model_at_cap
    assert any("specification's best step count" in note for note in res.notes)


def test_time_folds_train_only_on_earlier_rows():
    times = np.array(_times(600), dtype="datetime64[s]")
    folds = _make_folds(600, 4, None, times, seed=0)
    assert isinstance(folds, FoldScheme)
    assert folds.ordered
    weights = np.asarray(folds.weights)
    held = np.asarray(folds.held_out)
    for b in range(weights.shape[1]):
        trained = times[weights[:, b] > 0]
        judged = times[held[:, b]]
        assert trained.max() < judged.min()


def test_random_folds_hold_out_every_row_once():
    folds = _make_folds(300, 5, None, None, seed=1)
    held = np.asarray(folds.held_out)
    assert held.shape == (300, 5)
    assert (held.sum(axis=1) == 1).all()


def test_an_explicit_fold_scheme_is_used():
    frame = _interaction_frame(n=400)
    scheme = rolling_folds(400, B=3, time=np.arange(400.0))
    res = reference_gap(MODEL, frame, family=Gaussian(), fold_scheme=scheme,
                        max_mstop=150, top=0)
    assert res.folds == 3


def test_fold_scheme_row_count_must_match_the_data():
    scheme = rolling_folds(300, B=3)
    with pytest.raises(ValueError, match="fold_scheme has 300 rows"):
        reference_gap(MODEL, _interaction_frame(n=400), family=Gaussian(),
                      fold_scheme=scheme, max_mstop=150)


def test_fold_scheme_must_be_a_scheme():
    with pytest.raises(TypeError, match="FoldScheme"):
        reference_gap(MODEL, _interaction_frame(n=200), family=Gaussian(),
                      fold_scheme=np.ones((200, 3)), max_mstop=150)


# ---------------------------------------------------------------------------
# Time-ordered candidates


def test_lag_candidate_is_nominated_for_time_ordered_data():
    res = reference_gap("y ~ bbs(x1, df=4) + bbs(x2, df=4)", _lag_frame(), family=Gaussian(),
                        time="ts", folds=4, max_mstop=400, top=4, lags=(1, 3, 24))
    assert res.material
    lag_rows = res.candidates.filter(pl.col("kind") == "lag")
    assert lag_rows.height >= 1
    assert lag_rows["term"][0].startswith("bbs(temp_lag")
    assert lag_rows["measured_gain"][0] > 0
    assert "temp_lag3" in res.data.columns


def test_time_ordered_analysis_rows_are_complete_for_every_lag():
    res = reference_gap("y ~ bbs(x1, df=4)", _lag_frame(n=300), family=Gaussian(),
                        time="ts", folds=3, max_mstop=120, top=0, lags=(1, 3, 24))
    lag_columns = [c for c in res.data.columns if "_lag" in c or "_roll" in c]
    assert lag_columns
    assert sum(res.data.select(lag_columns).null_count().row(0)) == 0
    assert any("no full lag or rolling window" in note for note in res.notes)


def test_lag_features_follow_time_order_not_row_order():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    times = np.array([4, 2, 0, 3, 1])  # row i happened at time times[i]
    frame = pl.DataFrame({"temp": values})
    feats = _lag_and_rolling_features(frame, times, ["temp"], (1, 2))
    # In time order the values are 30, 50, 20, 40, 10 (times 0 to 4).
    expected_lag1 = {0: np.nan, 1: 30.0, 2: 50.0, 3: 20.0, 4: 40.0}
    for row in range(5):
        want = expected_lag1[int(times[row])]
        got = feats["temp_lag1"][row]
        assert (np.isnan(want) and np.isnan(got)) or got == want
    # A window of two rows is missing only at the first row in time order.
    assert np.isnan(feats["temp_roll2"]).sum() == 1
    assert feats["temp_roll2"][int(np.flatnonzero(times == 3)[0])] == 30.0


def test_integer_typed_and_binary_columns_are_not_lagged():
    frame = _lag_frame(n=200).with_columns(hour=pl.Series(np.arange(200) % 24),
                                           flag=pl.Series((np.arange(200) % 2).astype(float)))
    res = reference_gap("y ~ bbs(x1, df=4)", frame, family=Gaussian(), time="ts",
                        folds=3, max_mstop=120, top=0)
    assert not any(c.startswith("hour_lag") for c in res.data.columns)
    assert not any(c.startswith("flag_lag") for c in res.data.columns)
    assert any(c.startswith("temp_lag") for c in res.data.columns)


def test_a_measurement_stored_as_whole_numbers_is_lagged():
    # A temperature recorded in whole degrees is a float column whose values
    # are all integers; it is a measurement and is lagged.
    frame = _lag_frame(n=200).with_columns(
        temp=pl.col("temp").round(0), ts=pl.Series(_times(200)))
    res = reference_gap("y ~ bbs(x1, df=4)", frame, family=Gaussian(), time="ts",
                        folds=3, max_mstop=120, top=0)
    assert "temp_lag1" in res.data.columns


# ---------------------------------------------------------------------------
# Candidate terms


def test_numeric_pair_becomes_a_tensor_term():
    assert _interaction_term("a", "b", set()) == "bbs(a) %X% bbs(b)"


def test_numeric_by_factor_becomes_a_by_term():
    assert _interaction_term("x", "f", {"f"}) == "bbs(x, by=f, df=6)"
    assert _interaction_term("f", "x", {"f"}) == "bbs(x, by=f, df=6)"


def test_two_factors_have_no_term():
    assert _interaction_term("f", "h", {"f", "h"}) is None


# ---------------------------------------------------------------------------
# Input validation


def test_unknown_reference_is_rejected_with_the_choice():
    with pytest.raises(ValueError, match="reference must be one of"):
        reference_gap(MODEL, _interaction_frame(n=200), reference="forest")


def test_unknown_extrapolation_is_rejected():
    with pytest.raises(ValueError, match="extrapolation must be one of"):
        reference_gap(MODEL, _interaction_frame(n=200), extrapolation="guess")


def test_formula_without_a_tilde_is_rejected():
    with pytest.raises(ValueError, match="has no '~'"):
        reference_gap("bbs(x1)", _interaction_frame(n=200))


def test_absent_formula_column_is_named():
    with pytest.raises(ValueError, match="not columns of data"):
        reference_gap("y ~ bbs(zz, df=4)", _interaction_frame(n=200), family=Gaussian())


def test_missing_time_column_is_named():
    with pytest.raises(ValueError, match="time column 'when' is not a column"):
        reference_gap(MODEL, _interaction_frame(n=200), time="when")


def test_maxdepth_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="maxdepth must be"):
        reference_gap(MODEL, _interaction_frame(n=200), maxdepth=0)


def test_max_mstop_below_the_minimum_is_rejected():
    with pytest.raises(ValueError, match="max_mstop must be at least 100"):
        reference_gap(MODEL, _interaction_frame(n=200), max_mstop=50)


def test_folds_below_two_are_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        reference_gap(MODEL, _interaction_frame(n=200), folds=1)


def test_explicit_scheme_with_lags_is_restricted_to_the_analysis_rows():
    # The scheme is built on every row; the first 24 rows have no full lag window
    # and are removed from the analysis, so the scheme must be restricted to the rest.
    frame = _lag_frame(n=300)
    scheme = rolling_folds(300, B=3, time=np.array(_times(300), dtype="datetime64[s]"))
    res = reference_gap("y ~ bbs(x1, df=4)", frame, family=Gaussian(), time="ts",
                        fold_scheme=scheme, lags=(1, 3, 24), max_mstop=120, top=0)
    assert res.folds == 3
    # A 24-step lag is missing on the first 24 rows in time order.
    assert res.data.height == 300 - 24


# ---------------------------------------------------------------------------
# What the reference sees, candidates without the gap gate, family-loss screen


def _omitted_column_frame(n=900, seed=8):
    """The signal is in `z`, a column the formula leaves out."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    y = np.sin(2 * x1) + 0.5 * x2 + 1.5 * np.sin(2 * z) + rng.normal(0, 0.3, n)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z})


OMITTED_MODEL = "y ~ bbs(x1, df=4) + bbs(x2, df=4)"


def test_reference_sees_a_column_the_formula_omits_and_names_it():
    frame = _omitted_column_frame()
    old = reference_gap(OMITTED_MODEL, frame, family=Gaussian(), folds=4, max_mstop=400,
                        top=0, reference_columns="formula")
    new = reference_gap(OMITTED_MODEL, frame, family=Gaussian(), folds=4, max_mstop=400,
                        top=3)
    assert old.reference_extra_columns == []
    assert old.reference_formula.startswith("y ~ btree(x1, x2, maxdepth=4")
    assert new.material
    assert new.gap > old.gap
    assert new.reference_columns == ["x1", "x2", "z"]
    assert new.reference_extra_columns == ["z"]
    assert new.reference_formula.startswith("y ~ btree(x1, x2, z, maxdepth=4")
    assert new.candidates["term"][0] == "bbs(z, df=4)"
    assert new.candidates["kind"][0] == "column"
    assert new.candidates["measured_gain"][0] > 2 * new.candidates["gain_se"][0]
    assert "z" in new.describe() and "does not use" in new.describe()
    assert new.to_dict()["reference_extra_columns"] == ["z"]


def test_old_reference_reports_no_material_gap_for_an_omitted_column():
    old = reference_gap(OMITTED_MODEL, _omitted_column_frame(), family=Gaussian(), folds=4,
                        max_mstop=400, top=0, reference_columns="formula")
    assert not old.material


def test_exclude_columns_are_not_given_to_the_reference():
    res = reference_gap(OMITTED_MODEL, _omitted_column_frame(), family=Gaussian(), folds=3,
                        max_mstop=150, top=0, exclude_columns=["z"])
    assert res.reference_extra_columns == []
    with pytest.raises(ValueError, match="exclude_columns"):
        reference_gap(OMITTED_MODEL, _omitted_column_frame(n=200), exclude_columns=["nope"])


def test_reference_columns_accepts_a_list_and_rejects_unknown_names():
    frame = _omitted_column_frame().with_columns(w=pl.Series(np.arange(900.0)))
    res = reference_gap(OMITTED_MODEL, frame, family=Gaussian(), folds=3, max_mstop=150,
                        top=0, reference_columns=["z"])
    assert res.reference_extra_columns == ["z"]
    with pytest.raises(ValueError, match="reference_columns"):
        reference_gap(OMITTED_MODEL, frame, reference_columns="some")
    with pytest.raises(ValueError, match="not columns of data"):
        reference_gap(OMITTED_MODEL, frame, reference_columns=["nope"])


def test_unusable_columns_are_skipped_with_a_note():
    frame = _omitted_column_frame(n=400).with_columns(
        const=pl.lit(1.0),
        gappy=pl.Series([None if i % 7 == 0 else 1.0 * i for i in range(400)]))
    res = reference_gap(OMITTED_MODEL, frame, family=Gaussian(), folds=3, max_mstop=150, top=0)
    assert res.reference_extra_columns == ["z"]
    assert any("const" in note and "gappy" in note for note in res.notes)


def test_top_candidates_are_measured_when_the_gap_is_not_material():
    # With the reference limited to the formula's columns the gap is not material,
    # and the omitted column is still screened and measured.
    res = reference_gap(OMITTED_MODEL, _omitted_column_frame(), family=Gaussian(), folds=4,
                        max_mstop=400, top=2, reference_columns="formula")
    assert not res.material
    assert res.candidates.height >= 1
    assert res.candidates["term"][0] == "bbs(z, df=4)"
    assert res.candidates["measured_gain"][0] > 0
    assert res.closed_share is None


def test_time_ordered_reference_sees_the_lag_features():
    formula = "y ~ bbs(x1, df=4) + bbs(x2, df=4)"
    res = reference_gap(formula, _lag_frame(), family=Gaussian(), time="ts", folds=4,
                        max_mstop=200, top=0, lags=(1, 3, 24))
    assert "temp_lag3" in res.reference_columns
    assert "temp_lag3" in res.reference_extra_columns
    assert "temp" in res.reference_extra_columns
    assert "ts" not in res.reference_columns
    assert "temp_lag3" in res.reference_formula
    old = reference_gap(formula, _lag_frame(), family=Gaussian(), time="ts", folds=4,
                        max_mstop=200, top=0, lags=(1, 3, 24), reference_columns="formula")
    assert "temp_lag3" not in old.reference_formula
    assert res.gap > old.gap


def test_gaussian_screen_residual_is_the_ordinary_residual():
    from mboostrs.reference_gap import _working_residual

    frame = _omitted_column_frame(n=300)
    fit = mb.gamboost(OMITTED_MODEL, data=frame, family=Gaussian(),
                      control=mb.boost_control(mstop=60, nu=0.1))
    y = frame["y"].to_numpy()
    eta = np.asarray(fit.predict(type="link")).ravel()
    mu = np.asarray(fit.predict(type="response")).ravel()
    resid, kind = _working_residual(fit, Gaussian(), y, eta, mu)
    assert kind == "response"
    assert np.array_equal(resid, y - mu)
    res = reference_gap(OMITTED_MODEL, frame, family=Gaussian(), folds=3, max_mstop=150, top=1)
    assert res.screen_residual == "response"


def _poisson_frame(n=3000, seed=11):
    """Two omitted columns. `u` shifts the log rate by 0.9 where the mean is
    about 0.4; `v` shifts it by 0.1 where the mean is about 10. On the count
    scale v's shift is the larger one; on the family-loss scale u's is."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    eta = np.where(x1 < 0, -1.0 + 0.9 * u, 2.3 + 0.1 * v) + 0.2 * x1
    return pl.DataFrame({"y": rng.poisson(np.exp(eta)).astype(float), "x1": x1, "u": u, "v": v})


def test_poisson_screen_ranks_on_family_loss_not_count_scale_squared_error():
    from mboostrs.families import Poisson
    from mboostrs.reference_gap import _out_of_fold_residuals, _screen_candidates

    frame = _poisson_frame()
    formula = "y ~ bbs(x1, df=4)"
    scheme = _make_folds(frame.height, 4, None, None, 0)
    family_resid, kind = _out_of_fold_residuals(frame, "y", formula, Poisson(), scheme, 300, 0.1)
    assert kind == "family"
    fit = mb.gamboost(formula, data=frame, family=Poisson(),
                      control=mb.boost_control(mstop=300, nu=0.1))
    y = frame["y"].to_numpy()
    mu = np.asarray(fit.predict(type="response")).ravel()
    # The family residual is the Pearson residual for the Poisson log-link loss.
    seen = np.isfinite(family_resid)
    assert np.corrcoef(family_resid[seen], ((y - mu) / np.sqrt(mu))[seen])[0, 1] > 0.99
    count_resid = np.where(seen, y - mu, np.nan)

    def ranking(residual):
        found = _screen_candidates(frame, "y", ["x1"], residual, set(), ["u", "v"], 0, [])
        return [c["source"] for c in found]

    assert ranking(family_resid)[0] == "u"
    assert ranking(count_resid)[0] == "v"
    res = reference_gap(formula, frame, family=Poisson(), folds=4, max_mstop=300, top=2)
    assert res.screen_residual == "family"
    assert res.candidates["source"][0] == "u"
