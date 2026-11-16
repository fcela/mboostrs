"""The specification assistants, on the frames people actually have.

Every one of `classify_columns`, `suggest_terms`, `term_report`, `cv_risk` and
`compare_specifications` cast each column with `np.asarray(v, dtype=float)`.
A frame with one string column raised ``could not convert string to float:
'b'`` -- so a frame with a city, an arm, a flag or a date could not be
classified, scored or reported on, even for the numeric columns beside it. Two
of the five also assumed a mapping and raised ``'DataFrame' object has no
attribute 'items'`` on a Polars frame.

Three refusals in the fit path had the same shape. A **boolean** column was
read as an unordered factor, so `bols(flag)` was refused with "encoding it as
numbers would make bols treat the levels as an ordered measurement" -- which
is exactly what one wants for a boolean, since `False < True` is the coding
rather than an arbitrary label order. A **date** was read the same way, and
was refused under a mapping and pandas while fitting under Polars, whose dtype
name happened to fall outside the categorical set: the same data, two answers.
And a **two-level** factor was refused for `bols`, though with two levels every
coding gives the same fit up to the sign of one coefficient.

The scoring had a separate problem: `cv_risk` computed squared error of the
response for every family. For a `Binomial` fit the response is 0/1 and the
prediction is a log-odds, so it compared a probability-scale outcome with an
unbounded number and ranked specifications by something the model is not
fitted to minimise.
"""

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

import mboostrs as mb

N = 200


@pytest.fixture(scope="module")
def mixed():
    """One frame with every column type the assistants have to read."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    return {
        "y": 2.0 * x + rng.normal(size=N),
        "x": x,
        "z": rng.normal(size=N),
        "city": rng.choice(list("abcdefgh"), size=N),
        "arm": rng.choice(["ctrl", "treat"], size=N),
        "flag": rng.choice([True, False], size=N),
        "when": [dt.date(2020, 1, 1) + dt.timedelta(days=int(i))
                 for i in range(N)],
        "rooms": rng.integers(1, 6, size=N),
    }


@pytest.fixture(scope="module")
def backends(mixed):
    return {"mapping": mixed, "polars": pl.DataFrame(mixed),
            "pandas": pd.DataFrame(mixed)}


# ─── every assistant reads every backend ───────────────────────────


def test_classify_columns_reads_every_backend(backends):
    answers = {}
    for label, data in backends.items():
        answers[label] = {c.name: c.term
                          for c in mb.classify_columns(data, response="y")}
    assert answers["mapping"] == answers["polars"] == answers["pandas"], (
        "the same data must classify the same way whatever holds it"
    )
    assert set(answers["mapping"]) == {"x", "z", "city", "arm", "flag",
                                       "when", "rooms"}


def test_term_report_and_cv_risk_read_every_backend(backends):
    risks, reports = {}, {}
    for label, data in backends.items():
        risks[label] = mb.cv_risk("bols(x) + brandom(city, df=4)", data, "y",
                                  folds=3, mstop=50)
        reports[label] = mb.term_report(
            "y ~ bols(x) + brandom(city, df=4)", data, "y", folds=3, mstop=50,
        ).baseline
    assert len(set(round(v, 10) for v in risks.values())) == 1, risks
    assert len(set(round(v, 10) for v in reports.values())) == 1, reports


def test_compare_specifications_reads_every_backend(backends):
    for data in backends.values():
        rows = mb.compare_specifications(["bols(x)", "bbs(x, df=4)"], data,
                                         "y", folds=3, mstop=50)
        assert len(rows) == 2
        assert rows[0]["change"] == 0.0


# ─── what each column type is read as ──────────────────────────────


def test_a_string_column_becomes_levels_and_gets_a_random_effect(mixed):
    kinds = {c.name: c for c in mb.classify_columns(mixed, response="y")}
    city = kinds["city"]
    assert city.kind == "categorical"
    assert city.dtype == "categorical"
    assert city.levels == 8
    assert city.level_names == tuple("abcdefgh")
    assert not city.ordered
    assert city.term == "brandom(city, df=4.0)"


def test_a_boolean_column_is_an_indicator_not_a_factor(mixed):
    kinds = {c.name: c for c in mb.classify_columns(mixed, response="y")}
    flag = kinds["flag"]
    assert flag.kind == "binary"
    assert flag.dtype == "boolean"
    assert flag.ordered, "False < True is the coding, not a label order"
    assert flag.term == "bols(flag)"


def test_a_boolean_column_fits_where_it_used_to_be_refused(mixed):
    """The refusal said `bols` would "treat the levels as an ordered
    measurement". For a boolean that is the correct reading."""
    for data in (mixed, pl.DataFrame(mixed), pd.DataFrame(mixed)):
        model = mb.gamboost("y ~ bols(flag)", data=data,
                            control=mb.boost_control(20, 0.1))
        assert np.all(np.isfinite(model.fitted()))


def test_a_boolean_fit_equals_the_same_column_as_zero_and_one():
    rng = np.random.default_rng(4)
    flag = rng.choice([True, False], size=N)
    y = 3.0 * flag + rng.normal(size=N)
    boolean = mb.glmboost("y ~ flag", data={"y": y, "flag": flag},
                          control=mb.boost_control(60, 0.1))
    numeric = mb.glmboost("y ~ flag",
                          data={"y": y, "flag": flag.astype(float)},
                          control=mb.boost_control(60, 0.1))
    np.testing.assert_allclose(boolean.fitted(), numeric.fitted(), atol=1e-12)


def test_a_date_column_is_a_continuous_axis(mixed):
    kinds = {c.name: c for c in mb.classify_columns(mixed, response="y")}
    when = kinds["when"]
    assert when.kind == "datetime"
    assert when.unit == "days"
    assert when.ordered
    assert when.term == "bbs(when, df=4.0)"


def test_a_date_fits_the_same_way_in_every_backend(mixed):
    """It used to fit under Polars -- whose dtype name fell outside the
    categorical set -- and be refused under a mapping and pandas."""
    fits = []
    for data in (mixed, pl.DataFrame(mixed), pd.DataFrame(mixed)):
        model = mb.gamboost("y ~ bbs(when, df=4)", data=data,
                            control=mb.boost_control(20, 0.1))
        fits.append(model.fitted())
    np.testing.assert_allclose(fits[0], fits[1], atol=1e-12)
    np.testing.assert_allclose(fits[0], fits[2], atol=1e-12)


def test_a_two_level_factor_fits_as_an_indicator(mixed):
    """Two levels have no ordering to get wrong: every coding gives the same
    fit up to the sign of one coefficient."""
    model = mb.gamboost("y ~ bols(arm)", data=mixed,
                        control=mb.boost_control(20, 0.1))
    assert np.all(np.isfinite(model.fitted()))


def test_a_many_level_factor_under_bols_gets_one_effect_per_level(mixed):
    """`bols(city)` is R's dummy-coded learner: eight level effects, not a
    slope on the integer codes."""
    model = mb.gamboost("y ~ bols(city)", data=mixed,
                        control=mb.boost_control(10, 0.1))
    assert len(model.coef()["bols(city)"]) == 8


def test_a_declared_order_is_kept_not_re_sorted():
    """Declaring the order is the user saying it matters; sorting the labels
    would turn low < mid < high into high < low < mid."""
    rng = np.random.default_rng(5)
    size = pd.Categorical(rng.choice(["low", "mid", "high"], size=N),
                          categories=["low", "mid", "high"], ordered=True)
    frame = pd.DataFrame({"y": rng.normal(size=N), "size": size})
    kind = {c.name: c for c in mb.classify_columns(frame, response="y")}["size"]
    assert kind.level_names == ("low", "mid", "high")
    assert kind.ordered
    assert kind.term == "brandom(size, df=4.0)"
    assert "does not use their declared order" in kind.reason
    assert any("size_score" in term for term, _ in kind.alternatives)


def test_a_polars_enum_keeps_its_declared_order():
    rng = np.random.default_rng(5)
    frame = pl.DataFrame({
        "y": rng.normal(size=N),
        "size": rng.choice(["low", "mid", "high"], size=N),
    }).with_columns(pl.col("size").cast(pl.Enum(["low", "mid", "high"])))
    kind = {c.name: c for c in mb.classify_columns(frame, response="y")}["size"]
    assert kind.level_names == ("low", "mid", "high")
    assert kind.ordered


def test_an_unordered_polars_categorical_is_not_treated_as_ordered():
    rng = np.random.default_rng(5)
    frame = pl.DataFrame({
        "y": rng.normal(size=N),
        "city": rng.choice(list("abcd"), size=N),
    }).with_columns(pl.col("city").cast(pl.Categorical))
    kind = {c.name: c for c in mb.classify_columns(frame, response="y")}["city"]
    assert not kind.ordered
    assert kind.level_names == ("a", "b", "c", "d")


def test_missing_values_are_counted_not_fatal():
    rng = np.random.default_rng(6)
    x = rng.normal(size=N)
    x[:5] = np.nan
    labels = np.array(rng.choice(list("abc"), size=N), dtype=object)
    labels[:3] = None
    data = {"y": rng.normal(size=N), "x": x, "label": labels}
    kinds = {c.name: c for c in mb.classify_columns(data, response="y")}
    assert kinds["x"].n_missing == 5
    assert kinds["label"].n_missing == 3
    assert kinds["label"].level_names == ("a", "b", "c")


# ─── the suggested formula is one that fits ────────────────────────


def test_the_suggested_right_hand_side_actually_fits(mixed):
    """A recommendation that raises when passed to `gamboost` is worse than
    none."""
    rhs = mb.suggest_terms(mixed, response="y")
    model = mb.gamboost(f"y ~ {rhs}", data=mixed,
                        control=mb.boost_control(20, 0.1))
    assert np.all(np.isfinite(model.fitted()))


def test_the_shape_search_only_tries_shapes_that_fit(mixed):
    """An unordered factor has one shape: a slope, a spline or a threshold
    over its codes fits a curve through an alphabetical accident, and the fit
    path refuses them for that reason."""
    report = mb.term_report("y ~ bols(x) + brandom(city, df=4)", mixed, "y",
                            folds=3, mstop=50)
    tried = {row["term"] for row in report.shapes}
    assert "bols(x)" in tried
    assert "brandom(city, df=4)" not in tried


# ─── recommendations say what they assume ──────────────────────────


def test_every_recommendation_states_a_reason(mixed):
    for kind in mb.classify_columns(mixed, response="y"):
        assert len(kind.reason) > 25, kind.name


def test_a_continuous_column_names_the_shapes_it_did_not_choose(mixed):
    kinds = {c.name: c for c in mb.classify_columns(mixed, response="y")}
    alternatives = dict(kinds["x"].alternatives)
    assert "bols(x)" in alternatives
    assert "btree(x)" in alternatives
    assert "straight" in alternatives["bols(x)"]
    assert "threshold" in alternatives["btree(x)"]


def test_a_high_cardinality_factor_explains_fixed_against_random(mixed):
    kinds = {c.name: c for c in mb.classify_columns(mixed, response="y")}
    city = kinds["city"]
    assert "shrink" in city.reason
    assert city.term == "brandom(city, df=4.0)"
    assert not city.alternatives, (
        "the current fit path has no unshrunk multi-level factor learner; "
        "bols(city) is not a valid group-means alternative"
    )


def test_explain_prints_the_recommendation_and_the_alternatives(mixed):
    kinds = {c.name: c for c in mb.classify_columns(mixed, response="y")}
    text = kinds["city"].explain()
    assert "suggested: brandom(city, df=4.0)" in text
    assert "shrink" in text
    assert "or bols(city)" not in text
    assert "'a', 'b'" in text


# ─── scoring ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def binomial_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=N)
    p = 1.0 / (1.0 + np.exp(-2.0 * x))
    return {"x": x, "z": rng.normal(size=N),
            "y": (rng.uniform(size=N) < p).astype(float)}


def test_the_family_score_is_the_loss_the_fit_minimises(binomial_data):
    """Squared error of a 0/1 response against a log-odds compares two
    different scales."""
    family = mb.cv_risk("bols(x)", binomial_data, "y", family=mb.Binomial(),
                        folds=4, mstop=60)
    mse = mb.cv_risk("bols(x)", binomial_data, "y", family=mb.Binomial(),
                     folds=4, mstop=60, score="mse")
    assert family != mse
    assert family > 0


def test_the_two_scores_can_rank_specifications_differently(binomial_data):
    """Not a certainty on every data set, but the reason the default changed:
    the ranking under squared error is not the ranking the model optimises."""
    specifications = ["bols(x)", "bols(x) + bols(z)", "bbs(x, df=4)"]
    by_family = [mb.cv_risk(s, binomial_data, "y", family=mb.Binomial(),
                            folds=4, mstop=60) for s in specifications]
    by_mse = [mb.cv_risk(s, binomial_data, "y", family=mb.Binomial(),
                         folds=4, mstop=60, score="mse") for s in specifications]
    assert by_family != by_mse


def test_an_unknown_score_is_refused(mixed):
    with pytest.raises(ValueError, match="score must be"):
        mb.cv_risk("bols(x)", mixed, "y", folds=3, mstop=20, score="rmse")


# ─── weights, folds, missing values, tuned mstop ───────────────────


def test_weights_reach_both_the_fit_and_the_score(mixed):
    plain = mb.cv_risk("bols(x)", mixed, "y", folds=3, mstop=50)
    doubled = mb.cv_risk("bols(x)", mixed, "y", folds=3, mstop=50,
                         weights=np.full(N, 2.0))
    # A constant weight rescales nothing under the upstream policy.
    assert doubled == pytest.approx(plain, rel=1e-9)
    rng = np.random.default_rng(8)
    uneven = mb.cv_risk("bols(x)", mixed, "y", folds=3, mstop=50,
                        weights=rng.uniform(0.5, 2.0, size=N))
    assert uneven != pytest.approx(plain, rel=1e-6)


def test_a_wrong_length_weight_vector_is_refused(mixed):
    with pytest.raises(ValueError, match="expected \\(200,\\)"):
        mb.cv_risk("bols(x)", mixed, "y", folds=3, weights=np.ones(5))


def test_a_fold_scheme_is_accepted_where_a_fold_count_was(mixed):
    """This is how grouped and time-ordered validation reach the assistants: a
    specification scored on a random split of clustered rows is scored
    optimistically, and there was no way to say otherwise."""
    groups = np.repeat(np.arange(40), 5)
    grouped = mb.cv_risk("bols(x)", mixed, "y", folds=mb.grouped_folds(groups, B=4),
                         mstop=50)
    random_split = mb.cv_risk("bols(x)", mixed, "y", folds=4, mstop=50)
    assert np.isfinite(grouped) and np.isfinite(random_split)


def test_a_raw_fold_matrix_is_accepted(mixed):
    matrix = mb.cv_weights(N, type="kfold", B=4, seed=1)
    assert np.isfinite(mb.cv_risk("bols(x)", mixed, "y", folds=matrix, mstop=50))


def test_a_bootstrap_fold_matrix_keeps_its_multiplicities(mixed):
    """A bootstrap column carries each row's multiplicity in the resample, so
    a row drawn twice should train twice. Flattening the column to a ``> 0``
    mask (the pre-fix behaviour) evaluates a different scheme than the one
    supplied."""
    matrix = mb.cv_weights(N, type="bootstrap", B=3, seed=2)
    # Each column resamples N rows with replacement; the multiplicities are
    # what distinguishes it from the set of distinct rows it revisits.
    assert all(np.asarray(matrix).sum(axis=0) == N)
    assert np.asarray(matrix).max() > 1
    masked = (np.asarray(matrix) > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        weighted = mb.cv_risk("bols(x)", mixed, "y", folds=matrix, mstop=60)
        flattened = mb.cv_risk("bols(x)", mixed, "y", folds=masked, mstop=60)
    assert not np.isclose(weighted, flattened, rtol=1e-6)


def test_a_fold_matrix_of_the_wrong_height_is_refused(mixed):
    with pytest.raises(ValueError, match="in-bag weight matrix"):
        mb.cv_risk("bols(x)", mixed, "y", folds=np.ones((7, 3)), mstop=20)


def test_na_action_omit_drops_the_row_from_both_sides():
    """Dropping it only from the fit left `predict` refusing the held-out
    fold, which is where the missing row still was."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=N)
    x[:8] = np.nan
    data = {"x": x, "y": rng.normal(size=N)}
    assert np.isfinite(
        mb.cv_risk("bols(x)", data, "y", folds=3, mstop=40, na_action="omit")
    )
    with pytest.raises(ValueError, match="NaN"):
        mb.cv_risk("bols(x)", data, "y", folds=3, mstop=40)


def test_tuning_mstop_inside_the_fold_gives_a_finite_score(mixed):
    """A fixed budget compares specifications at one number of iterations,
    which favours whichever suits it. `mstop="cv"` gives each its own."""
    tuned = mb.cv_risk("bols(x) + bols(z)", mixed, "y", folds=3, mstop="cv")
    assert np.isfinite(tuned)

    grouped = mb.cv_risk("bols(x) + bols(z)", mixed, "y",
                         folds=mb.grouped_folds(np.repeat(np.arange(40), 5),
                                                B=4, seed=1),
                         mstop="cv")
    assert np.isfinite(grouped)
    again = mb.cv_risk("bols(x) + bols(z)", mixed, "y",
                       folds=mb.grouped_folds(np.repeat(np.arange(40), 5),
                                              B=4, seed=1),
                       mstop="cv")
    assert grouped == again


def test_inner_tuning_folds_keep_the_outer_scheme_structure():
    """`mstop="cv"` retunes on random inner folds, which re-admits the leak a
    grouped or ordered outer scheme was introduced to prevent: the tuning
    would pick the `mstop` that wins by seeing the rows the outer scheme
    holds apart. The inner folds must reuse the outer structure."""
    from mboostrs.explore import _inner_folds

    n = 120
    groups = np.repeat(np.arange(24), 5)
    grouped = mb.grouped_folds(groups, B=4, seed=1)
    inner = _inner_folds(grouped, np.ones(n, dtype=bool), seed=3)
    assert inner is not None
    assert inner.check_leakage()["leaks"] == []

    # A plain fold count has no structure for random inner folds to violate,
    # so the ordinary tuning is kept.
    assert _inner_folds(4, np.ones(n, dtype=bool), seed=3) is None

    ordered = mb.rolling_folds(n, B=3, initial=60, horizon=10)
    inner = _inner_folds(ordered, np.ones(n, dtype=bool), seed=3)
    assert inner is not None
    assert inner.check_leakage()["leaks"] == []


def test_negative_controls_for_family_and_resampling_contracts():
    """Correctly specified models must not be flagged by the diagnostics.

    The audit's proximate false positive was a correctly specified Poisson
    reported as strong heteroscedasticity (`t=15.83`, `spread_ratio=4.09`)
    because the residual spread grows with the mean even when the model is
    right. The Gaussian log-chi-squared test is only valid under the
    normal-errors contract, so it now refuses a Poisson fit outright. Under
    that same contract, iid Gaussian errors must not read as a pattern.
    """
    rng = np.random.default_rng(7)
    n = 800
    x = rng.normal(size=n)

    poisson = mb.gamboost("y ~ bols(x)",
                          data={"x": x, "y": rng.poisson(np.exp(0.3 * x))},
                          family=mb.Poisson())
    with pytest.raises(ValueError, match="Poisson"):
        mb.heteroscedasticity(poisson)

    gaussian = mb.gamboost("y ~ bols(x)",
                           data={"x": x, "y": rng.normal(size=n)},
                           family=mb.Gaussian())
    stats = mb.heteroscedasticity(gaussian)
    assert stats["t"] < 3.0            # the slope is noise, not a signal
    assert stats["spread_ratio"] < 2.0

    # A correctly specified Gaussian scores sanely under every resampling
    # scheme the assistants accept, and mstop="cv" stays in that range.
    data = {"x": x, "y": rng.normal(0.5 * x, size=n)}
    for folds, name in [
        (3, "k-fold count"),
        (mb.grouped_folds(np.repeat(np.arange(50), 16), B=4, seed=2),
         "grouped"),
        (mb.rolling_folds(n, B=3, initial=n // 2, horizon=n // 10), "rolling"),
    ]:
        risk = mb.cv_risk("bols(x)", data, "y", folds=folds, mstop=50)
        assert np.isfinite(risk) and risk > 0, name
    tuned = mb.cv_risk("bols(x)", data, "y", folds=3, mstop="cv")
    assert np.isfinite(tuned) and tuned > 0


# ─── the output is editable, and carries its evidence ──────────────


def test_a_comparison_row_carries_the_specification_and_the_settings(mixed):
    rows = mb.compare_specifications({"linear": "bols(x)"}, mixed, "y",
                                     folds=3, mstop=50)
    row = rows[0]
    assert row["rhs"] == "bols(x)"
    assert row["score"] == "family"
    assert row["folds"] == "3-fold random split"
    assert row["mstop"] == 50
    assert row["weighted"] is False


def test_a_comparison_records_the_scheme_it_used(mixed):
    groups = np.repeat(np.arange(40), 5)
    rows = mb.compare_specifications(["bols(x)"], mixed, "y", mstop=50,
                                     folds=mb.grouped_folds(groups, B=4))
    assert rows[0]["folds"] == "grouped 4-fold"


def test_a_list_of_specifications_is_accepted(mixed):
    """It used to raise ``'list' object has no attribute 'items'``."""
    rows = mb.compare_specifications(["bols(x)", "bbs(x, df=4)"], mixed, "y",
                                     folds=3, mstop=50)
    assert [row["specification"] for row in rows] == ["bols(x)", "bbs(x, df=4)"]


@pytest.mark.parametrize("output", ["polars", "pandas"])
def test_a_comparison_can_come_back_as_a_frame(mixed, output):
    frame = mb.compare_specifications(["bols(x)"], mixed, "y", folds=3,
                                      mstop=50, output=output)
    assert type(frame).__module__.split(".")[0] == output


def test_identical_specifications_have_zero_optimism_and_share_the_set(mixed):
    rows = mb.compare_specifications(["bols(x)", "bols(x)"], mixed, "y",
                                     folds=3, mstop=50)
    assert rows[0]["out-of-fold risk"] == rows[1]["out-of-fold risk"]
    # The winner won every fold, and nothing is distinguishable from it.
    assert rows[0]["search optimism"] == 0.0
    # A repeated candidate lists its name once: a name is a specification.
    assert rows[0]["competitive set"] == ["bols(x)"]


def test_the_search_optimism_matches_the_hand_computed_estimate():
    """The Tibshirani & Tibshirani estimate, recomputed from the fold scores."""
    rng = np.random.default_rng(7)
    n = 150
    x, z = rng.normal(size=n), rng.normal(size=n)
    data = {"y": 0.4 * x + rng.normal(0, 0.5, n), "x": x, "z": z}
    candidates = {"x smooth": "bbs(x, df=4) + bols(z)",
                  "z smooth": "bbs(z, df=4) + bols(x)"}
    rows = mb.compare_specifications(candidates, data, "y", folds=5,
                                     mstop=80, seed=1,
                                     include_evaluations=True)
    scores = np.vstack([row["evaluation"].fold_scores for row in rows])
    winner = int(np.argmin([row["out-of-fold risk"] for row in rows]))
    hand = float(np.mean(scores[winner] - scores.min(axis=0)))
    assert hand > 0.0  # the winner lost at least one fold: the case to estimate
    assert rows[winner]["search optimism"] == pytest.approx(hand)
    assert rows[0]["search optimism"] == rows[1]["search optimism"]


def test_the_competitive_set_holds_candidates_within_one_paired_se():
    rng = np.random.default_rng(7)
    n = 150
    x, z = rng.normal(size=n), rng.normal(size=n)
    data = {"y": 0.4 * x + rng.normal(0, 0.5, n), "x": x, "z": z}
    rows = mb.compare_specifications(
        {"x smooth": "bbs(x, df=6) + bols(z)",
         "z smooth": "bbs(z, df=6) + bols(x)",
         "no signal": "bols(z)"},
        data, "y", folds=5, mstop=80, seed=1, include_evaluations=True)
    from mboostrs.explore import _paired_se

    winner = min(rows, key=lambda row: row["out-of-fold risk"])
    expected = [row["specification"] for row in rows
                if row is winner or row["out-of-fold risk"] - winner["out-of-fold risk"]
                <= _paired_se(row["evaluation"].fold_scores,
                              winner["evaluation"].fold_scores)]
    assert rows[0]["competitive set"] == expected
    assert winner["specification"] in rows[0]["competitive set"]
    assert "no signal" not in rows[0]["competitive set"]


def test_the_search_optimism_is_partition_averaged_on_repeated_folds():
    rng = np.random.default_rng(7)
    n = 150
    x, z = rng.normal(size=n), rng.normal(size=n)
    data = {"y": 0.4 * x + rng.normal(0, 0.5, n), "x": x, "z": z}
    scheme = mb.repeated_kfold_folds(n, k=5, repeats=3, seed=1)
    rows = mb.compare_specifications(
        {"x smooth": "bbs(x, df=4) + bols(z)",
         "z smooth": "bbs(z, df=4) + bols(x)"},
        data, "y", folds=scheme, mstop=80, include_evaluations=True)
    scores = np.vstack([row["evaluation"].fold_scores
                        for row in rows]).reshape(len(rows), 3, 5)
    winner = int(np.argmin([row["out-of-fold risk"] for row in rows]))
    within = scores[winner] - scores.min(axis=0)
    hand = float(np.mean(within.mean(axis=1)))
    assert rows[winner]["search optimism"] == pytest.approx(hand)


def test_a_group_term_on_folds_that_split_its_groups_warns():
    rng = np.random.default_rng(1)
    n = 120
    groups = rng.integers(0, 12, n)
    data = {"y": rng.normal(groups, 1, n), "x": rng.normal(size=n),
            "g": groups}
    with pytest.warns(UserWarning, match=r"brandom\(g\).*split its groups"):
        mb.cv_risk("bbs(x, df=3) + brandom(g, df=4)", data, "y",
                   folds=3, mstop=60)


def test_a_group_term_on_folds_that_keep_groups_together_is_silent():
    rng = np.random.default_rng(1)
    n = 120
    groups = rng.integers(0, 12, n)
    data = {"y": rng.normal(groups, 1, n), "x": rng.normal(size=n),
            "g": groups}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        mb.cv_risk("bbs(x, df=3) + brandom(g, df=4)", data, "y",
                   folds=mb.grouped_folds(groups, B=3), mstop=60)
    assert not any("split its groups" in str(w.message) for w in seen)


def test_a_comparison_warns_when_folds_split_the_groups_of_a_candidate():
    rng = np.random.default_rng(1)
    n = 120
    groups = rng.integers(0, 12, n)
    data = {"y": rng.normal(groups, 1, n), "x": rng.normal(size=n),
            "g": groups}
    with pytest.warns(UserWarning, match=r"brandom\(g\)"):
        mb.compare_specifications(
            {"no group": "bbs(x, df=3)", "grouped": "bbs(x, df=3) + brandom(g, df=4)"},
            data, "y", folds=3, mstop=60)


def test_a_specification_without_a_group_term_never_warns_about_splits(mixed):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        mb.cv_risk("bbs(x, df=3)", mixed, "y", folds=3, mstop=50)
    assert not any("split its groups" in str(w.message) for w in seen)


def test_history_extends_the_search_across_calls():
    rng = np.random.default_rng(7)
    n = 150
    x, z = rng.normal(size=n), rng.normal(size=n)
    data = {"y": 0.4 * x + rng.normal(0, 0.5, n), "x": x, "z": z}
    first = mb.compare_specifications(
        {"x smooth": "bbs(x, df=4) + bols(z)",
         "z smooth": "bbs(z, df=4) + bols(x)"},
        data, "y", folds=5, mstop=60, seed=1, include_evaluations=True)
    second = mb.compare_specifications(
        {"both lines": "bols(x) + bols(z)"},
        data, "y", folds=5, mstop=60, seed=1, history=first,
        include_evaluations=True)
    assert second[0]["search candidates"] == 3
    assert second[0]["search fits"] == (
        sum(row["fit count"] for row in first) + second[0]["cumulative fit count"])
    # The estimate covers all three candidates, in the order run.
    all_scores = np.vstack([row["evaluation"].fold_scores
                            for row in first + second])
    winner = int(np.argmin([row["out-of-fold risk"] for row in first + second]))
    hand = float(np.mean(all_scores[winner] - all_scores.min(axis=0)))
    assert second[0]["search optimism"] == pytest.approx(hand)
    assert "x smooth" in second[0]["competitive set"]  # history names appear
    # Without history the same fields describe this call alone.
    assert first[0]["search candidates"] == 2
    assert first[0]["search fits"] == first[0]["cumulative fit count"]


def test_history_rows_without_evaluations_are_refused():
    rng = np.random.default_rng(7)
    n = 100
    x = rng.normal(size=n)
    data = {"y": x + rng.normal(0, 0.3, n), "x": x}
    plain = mb.compare_specifications(["bols(x)"], data, "y", folds=3, mstop=30)
    assert "evaluation" not in plain[0]
    with pytest.raises(ValueError, match="include_evaluations=True"):
        mb.compare_specifications(["bbs(x, df=3)"], data, "y", folds=3,
                                  mstop=30, history=plain)


def test_history_from_different_folds_is_refused():
    rng = np.random.default_rng(7)
    n = 100
    x = rng.normal(size=n)
    data = {"y": x + rng.normal(0, 0.3, n), "x": x}
    first = mb.compare_specifications(["bols(x)"], data, "y", folds=3,
                                      mstop=30, seed=1,
                                      include_evaluations=True)
    with pytest.raises(ValueError, match="not comparable"):
        mb.compare_specifications(["bbs(x, df=3)"], data, "y", folds=3,
                                  mstop=30, seed=2, history=first)


def test_holdout_splits_rows_groups_and_time():
    rng = np.random.default_rng(5)
    n = 200
    x, z = rng.normal(size=n), rng.normal(size=n)
    groups = rng.integers(0, 10, n)
    data = {"y": 0.5 * x + rng.normal(0, 0.4, n), "x": x, "z": z, "g": groups}

    search, test = mb.holdout(data, fraction=0.2, seed=1)
    assert len(test["y"]) + len(search["y"]) == n
    assert 0 < len(test["y"]) < n

    search, test = mb.holdout(data, fraction=0.2, groups="g", seed=1)
    trained = set(np.unique(search["g"]).tolist())
    held = set(np.unique(test["g"]).tolist())
    assert held and trained and held.isdisjoint(trained)

    search, test = mb.holdout(data, fraction=0.2, time="x")
    assert test["x"].max() > search["x"].max()  # the latest rows are the test

    frame = mb.holdout(pl.DataFrame(data), fraction=0.25)[0]
    assert isinstance(frame, pl.DataFrame) and frame.height == 150


def test_holdout_refuses_groups_combined_with_time():
    rng = np.random.default_rng(0)
    data = {"y": rng.normal(size=50), "x": rng.normal(size=50),
            "t": np.arange(50.0)}
    with pytest.raises(ValueError, match="groups= and time="):
        mb.holdout(data, fraction=0.2, groups="x", time="t")


def test_the_test_part_warns_when_scored_twice():
    rng = np.random.default_rng(9)
    n = 120
    x = rng.normal(size=n)
    data = {"y": x + rng.normal(0, 0.3, n), "x": x}
    _search, test = mb.holdout(data, fraction=0.25, seed=3)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        mb.evaluate_specification("bbs(x, df=3)", test, "y", folds=2, mstop=30)
        assert not any("holdout()" in str(w.message) for w in seen)
        mb.evaluate_specification("bols(x)", test, "y", folds=2, mstop=30)
        scored = [w for w in seen if "holdout()" in str(w.message)]
    assert len(scored) == 1
    assert "2 specifications" in str(scored[0].message)


def test_a_suggestion_carries_the_specification_it_produces():
    rng = np.random.default_rng(10)
    n = 300
    x, z, w = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    data = {"x": x, "z": z, "w": w,
            "y": np.sin(1.5 * x) * 2 + 0.3 * z + rng.normal(size=n)}
    report = mb.term_report("y ~ bols(x) + bols(z) + bols(w)", data, "y",
                            folds=4, mstop=100)
    suggestions = mb.suggest_specifications(report)
    assert suggestions
    for suggestion in suggestions:
        assert suggestion.specification
        assert suggestion.evidence
    best = suggestions[0]
    assert "bbs(x" in best.specification
    # The specification is runnable as it stands, which is the point of
    # returning it rather than describing the change.
    assert np.isfinite(mb.cv_risk(best.specification, data, "y", folds=3,
                                  mstop=100))


def test_a_shape_suggestion_says_what_the_new_shape_assumes():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.normal(size=n)
    data = {"x": x, "y": np.sin(1.5 * x) * 2 + rng.normal(size=n)}
    report = mb.term_report("y ~ bols(x)", data, "y", folds=4, mstop=100)
    suggestion = mb.suggest_specifications(report)[0]
    assert "may bend" in suggestion.reason


# ─── mean against distributional response ──────────────────────────


@pytest.fixture(scope="module")
def unequal_spread():
    rng = np.random.default_rng(1)
    n = 400
    x, z = rng.normal(size=n), rng.normal(size=n)
    return {"x": x, "z": z,
            "y": 2 * x + 0.3 * z + rng.normal(size=n) * np.exp(0.5 * x)}


def test_the_spread_test_separates_a_real_pattern_from_none(unequal_spread):
    """`log(residual^2)` has variance `pi^2/2` under constant variance
    whatever the scale of the data, which gives the slope a standard error
    without estimating one."""
    model = mb.gamboost("y ~ bols(x) + bols(z)", data=unequal_spread,
                        control=mb.boost_control(200, 0.1))
    measured = mb.heteroscedasticity(model)
    assert measured["t"] > 5
    assert measured["spread_ratio"] > 2

    rng = np.random.default_rng(2)
    n = len(unequal_spread["x"])
    flat = {"x": unequal_spread["x"], "z": unequal_spread["z"],
            "y": 2 * unequal_spread["x"] + rng.normal(size=n)}
    control = mb.gamboost("y ~ bols(x) + bols(z)", data=flat,
                          control=mb.boost_control(200, 0.1))
    assert abs(mb.heteroscedasticity(control)["t"]) < 3


def test_a_distributional_model_is_suggested_when_the_spread_moves(
    unequal_spread,
):
    model = mb.gamboost("y ~ bols(x) + bols(z)", data=unequal_spread,
                        control=mb.boost_control(200, 0.1))
    report = mb.term_report("y ~ bols(x) + bols(z)", unequal_spread, "y",
                            folds=3, mstop=100)
    suggestions = mb.suggest_specifications(report, model=model)
    distributional = [s for s in suggestions if s.action == "model"]
    assert distributional, "expected a distributional-response suggestion"
    suggestion = distributional[0]
    assert "interval" in suggestion.reason
    assert "GaussianLSS" in suggestion.specification
    assert suggestion.basis == "detected"


def test_no_distributional_suggestion_when_the_spread_is_flat():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    data = {"x": x, "z": rng.normal(size=n), "y": 2 * x + rng.normal(size=n)}
    model = mb.gamboost("y ~ bols(x) + bols(z)", data=data,
                        control=mb.boost_control(200, 0.1))
    report = mb.term_report("y ~ bols(x) + bols(z)", data, "y", folds=3,
                            mstop=100)
    assert not [s for s in mb.suggest_specifications(report, model=model)
                if s.action == "model"]


def test_heteroscedasticity_refuses_non_gaussian_families():
    """`log(residual^2)` is only `log(chi^2_1)` under the normal-errors
    contract. A correctly specified Poisson naturally grows its spread with
    the mean, so the test would fire on a perfect fit; it refuses by name
    rather than return a number that reads as evidence."""
    rng = np.random.default_rng(4)
    n = 400
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.8 * x)).astype(float)
    model = mb.gamboost("y ~ bols(x)", data={"x": x, "y": y},
                        family=mb.Poisson(),
                        control=mb.boost_control(200, 0.1))
    with pytest.raises(ValueError, match="fitted with Poisson"):
        mb.heteroscedasticity(model)

    p = 1 / (1 + np.exp(-(0.3 + 0.8 * x)))
    binary = rng.binomial(1, p).astype(float)
    bmodel = mb.gamboost("y ~ bols(x)", data={"x": x, "y": binary},
                         family=mb.Binomial(),
                         control=mb.boost_control(200, 0.1))
    with pytest.raises(ValueError, match="fitted with Binomial"):
        mb.heteroscedasticity(bmodel)


def test_no_distributional_suggestion_for_a_non_gaussian_fit():
    """A moment pattern in the raw residual spread of a Poisson fit is the
    family's ordinary variance relationship, not evidence for a
    distributional Gaussian fit."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.8 * x)).astype(float)
    data = {"x": x, "y": y}
    model = mb.gamboost("y ~ bols(x)", data=data, family=mb.Poisson(),
                        control=mb.boost_control(200, 0.1))
    report = mb.term_report("y ~ bols(x)", data, "y", folds=3, mstop=100)
    assert not [s for s in mb.suggest_specifications(report, model=model)
                if s.action == "model"]


def test_an_interaction_suggestion_names_the_trade_off():
    from mboostrs.terms import ScreenedPair

    rng = np.random.default_rng(12)
    n = 200
    data = {"x": rng.normal(size=n), "z": rng.normal(size=n)}
    data["y"] = data["x"] * data["z"] + rng.normal(size=n)
    report = mb.term_report("y ~ bols(x) + bols(z)", data, "y", folds=3,
                            mstop=60)
    pair = ScreenedPair(left="x", right="z", removed=0.5)
    suggestion = [s for s in mb.suggest_specifications(report, screened=[pair])
                  if s.action == "add"][0]
    assert "btree" in suggestion.reason
    assert "%X%" in suggestion.specification
    assert suggestion.basis == "detected"
