"""``autospec`` returns a spec that names each column and fits as-is.

The whole contract of "I'm feeling lucky" is that ``autospec(data, y)`` hands
back ``(formula, learners, family, mstop)`` and ``gamboost`` accepts all four
without another argument. These tests pin that: every returned term must be one
the fit path can actually build (``classify_columns`` alone is not enough —
it suggests ``bols`` for an ordered categorical, which ``encode_factors``
refuses), and family inference must pick the loss the response calls for.
"""

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pytest

import mboostrs as mb

N = 200


@pytest.fixture(scope="module")
def mixed():
    """One frame with every column type autospec has to name a learner for."""
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


# ─── round trip: every returned spec fits with no edits ────────────


def test_autospec_round_trip_every_backend(backends):
    for _label, data in backends.items():
        formula, learners, family, mstop = mb.autospec(data, "y")
        assert isinstance(family, mb.Gaussian)
        assert mstop == 100
        # The formula names the response and every other column once.
        for column in data.columns if hasattr(data, "columns") else data:
            if column == "y":
                continue
            assert f"({column}" in formula
        fit = mb.gamboost(
            formula, data, learners=learners, family=family,
            control=mb.boost_control(mstop=30),
        )
        assert fit.fitted_.shape == (N,)


def test_round_trip_without_learners_argument(backends):
    """The formula alone must parse to the same fit — learners= is optional."""
    data = backends["polars"]
    formula, _, family, _ = mb.autospec(data, "y")
    with_learners = mb.gamboost(
        formula, data, learners=mb.autospec(data, "y")[1], family=family,
        control=mb.boost_control(mstop=30),
    )
    without = mb.gamboost(
        formula, data, family=family, control=mb.boost_control(mstop=30),
    )
    np.testing.assert_allclose(with_learners.fitted_, without.fitted_)


# ─── learner choices per column type ───────────────────────────────


def test_continuous_predictor_is_bbs():
    formula, _, _, _ = mb.autospec({"y": np.arange(20.0), "x": np.arange(20.0)}, "y")
    assert "bbs(x, df=4.0)" in formula


def test_boolean_predictor_is_bols(mixed):
    formula, _, _, _ = mb.autospec(mixed, "y")
    assert "bols(flag)" in formula


def test_two_level_factor_is_bols(mixed):
    formula, _, _, _ = mb.autospec(mixed, "y")
    assert "bols(arm)" in formula


def test_multi_level_factor_is_brandom(mixed):
    formula, _, _, _ = mb.autospec(mixed, "y")
    assert "brandom(city, df=4.0)" in formula


def test_ordered_categorical_warns_that_brandom_drops_the_order():
    rng = np.random.default_rng(5)
    size = pd.Categorical(rng.choice(["low", "mid", "high"], size=N),
                          categories=["low", "mid", "high"], ordered=True)
    frame = pd.DataFrame({"y": rng.normal(size=N), "size": size})
    with pytest.warns(mb.OrderedCategoryWarning, match="does not use their declared order"):
        formula, _, _, _ = mb.autospec(frame, "y")
    assert "brandom(size, df=4.0)" in formula
    mb.gamboost(formula, frame, control=mb.boost_control(mstop=20))

    decision = mb.classify_columns(frame, response="y")[0]
    assert decision.term == "brandom(size, df=4.0)"
    assert "does not use their declared order" in decision.reason
    assert any("size_score" in term for term, _ in decision.alternatives)


def test_numeric_binary_predictor_is_bols():
    formula, _, _, _ = mb.autospec(
        {"y": np.arange(20.0), "flag": np.tile([0.0, 1.0], 10)}, "y")
    assert "bols(flag)" in formula


def test_whole_number_predictor_matches_classify_columns():
    """A numeric column of few whole-number label values is levels, not a
    measurement -- the same rule classify_columns applies, so the two
    functions agree on the same frame. A column that reads as a count
    (#313) is the one whole-number column that keeps a measurement term:
    `visits` 0..5 and `stage` 0..3 both extend to a value the fit never
    saw, which `brandom` cannot score."""
    rng = np.random.default_rng(11)
    n = 200
    data = {
        "y": rng.normal(size=n),
        "visits": rng.integers(0, 6, n).astype(float),   # a count 0..5 -> bbs
        "stage": rng.integers(0, 4, n).astype(float),    # a count 0..3 -> bols
    }
    formula, _, _, _ = mb.autospec(data, "y")
    assert "bbs(visits, df=4.0)" in formula
    assert "bols(stage)" in formula
    for c in mb.classify_columns(data, response="y"):
        assert f"({c.name}" in formula


def test_three_numeric_levels_fit_group_effects_not_one_code_slope():
    """Three label codes ({10, 27, 53}) are levels: a slope across them
    would order the groups by an arbitrary number. Codes that read as a
    count (0, 1, 2 spanning their whole range) are a measurement instead —
    #313, covered in tests/test_count_typing.py."""
    rng = np.random.default_rng(12)
    codes = np.array([10.0, 27.0, 53.0])
    level = np.tile(codes, 100)
    targets = np.array([-2.0, 3.0, 0.0])
    y = targets[np.tile(np.arange(3), 100)] + rng.normal(scale=0.05, size=level.size)
    data = {"y": y, "level": level}
    formula, learners, family, _ = mb.autospec(data, "y")

    assert formula == "y ~ brandom(level, df=4.0)"
    assert isinstance(learners[0], mb.Brandom)
    fit = mb.gamboost(
        formula, data, learners=learners, family=family,
        control=mb.boost_control(200, 0.1),
    )
    coefficients = np.asarray(fit.coef(aggregate="sum")["level"])
    fitted_means = np.array([fit.predict()[level == value].mean() for value in codes])

    assert coefficients.shape == (3,)
    assert abs(coefficients[1] - 0.5 * (coefficients[0] + coefficients[2])) > 2.0
    np.testing.assert_allclose(fitted_means, targets, atol=0.08)


def test_datetime_predictor_is_bbs(mixed):
    formula, _, _, _ = mb.autospec(mixed, "y")
    assert "bbs(when, df=4.0)" in formula


def test_btree_smooth():
    rng = np.random.default_rng(1)
    data = {"y": rng.normal(size=N), "x": rng.normal(size=N)}
    formula, from_terms, family, _ = mb.autospec(data, "y", smooth="btree")
    assert formula == "y ~ btree(x)"
    assert len(from_terms) == 1


# ─── family inference from the response ────────────────────────────


def test_family_from_boolean_response():
    rng = np.random.default_rng(2)
    data = {"y": rng.choice([True, False], size=N), "x": rng.normal(size=N)}
    _, _, family, _ = mb.autospec(data, "y")
    assert isinstance(family, mb.Binomial)


def test_family_from_numeric_zero_one_response():
    rng = np.random.default_rng(3)
    data = {"y": rng.integers(0, 2, size=N).astype(float), "x": rng.normal(size=N)}
    _, _, family, _ = mb.autospec(data, "y")
    assert isinstance(family, mb.Binomial)


def test_family_from_count_response():
    rng = np.random.default_rng(4)
    data = {"y": rng.poisson(3.0, size=N), "x": rng.normal(size=N)}
    _, _, family, _ = mb.autospec(data, "y")
    assert isinstance(family, mb.Poisson)


def test_float_whole_number_response_warns_that_count_meaning_is_ambiguous():
    rng = np.random.default_rng(44)
    data = {"y": rng.poisson(3.0, size=N).astype(float),
            "x": rng.normal(size=N)}
    with pytest.warns(UserWarning, match="whole-number values stored as floating"):
        rec = mb.autospec(data, "y")
    assert isinstance(rec.family, mb.Poisson)


def test_large_whole_number_response_is_read_as_a_measurement():
    """House values stored as whole numbers (median near 2e5) are not counts:
    Poisson's gradient is on the response's scale and diverges in the first
    step, and the family probe compares Gamma with Gaussian instead."""
    rng = np.random.default_rng(45)
    x = rng.normal(size=N)
    data = {"y": np.round(2e5 * np.exp(0.3 * x + 0.3 * rng.normal(size=N))), "x": x}
    with pytest.warns(UserWarning, match="indistinguishable"):
        rec = mb.autospec(data, "y")
    assert isinstance(rec.family, mb.GammaReg)
    with pytest.warns(UserWarning, match="indistinguishable"):
        rec = mb.autospec(data, "y", budget=60)
    assert type(rec.family).__name__ in ("GammaReg", "Gaussian")
    assert np.all(np.isfinite(np.asarray(rec.fit(data).predict(), dtype=float)))


def test_family_from_positive_continuous_response():
    rng = np.random.default_rng(5)
    data = {"y": np.exp(rng.normal(size=N)), "x": rng.normal(size=N)}
    _, _, family, _ = mb.autospec(data, "y")
    assert isinstance(family, mb.GammaReg)


def test_a_proportion_in_the_unit_interval_is_not_gamma():
    """GammaReg's support continues past 1. A response that never leaves
    (0, 1) is a proportion: without a budget autospec keeps Gaussian and
    says so; with one, the family probe scores BetaLSS against Gaussian on
    held-out log score and adopts it when it clearly wins (#228) — beta
    data is beta data at any precision."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=N)
    mu = 1 / (1 + np.exp(-(0.2 + 0.4 * x)))
    y = rng.beta(mu * 8, (1 - mu) * 8)
    data = {"y": y, "x": x}
    assert np.all((y > 0) & (y < 1))
    with pytest.warns(UserWarning, match=r"lies in \(0, 1\)"):
        rec = mb.autospec(data, "y")
    assert isinstance(rec.family, mb.Gaussian)
    measured = mb.autospec(data, "y", budget=40, mstop=40, seed=0)
    assert isinstance(measured.family, mb.BetaLSS)
    evidence = measured.evidence["family"]
    assert evidence.lss_score < evidence.mean_log_score
    assert "nats/row" in evidence.sentence
    assert "BetaLSS" in evidence.sentence


def test_family_from_multiclass_label_response():
    rng = np.random.default_rng(6)
    data = {"y": rng.choice(list("abcde"), size=N), "x": rng.normal(size=N)}
    _, _, family, _ = mb.autospec(data, "y")
    assert isinstance(family, mb.Multinomial)
    assert family.n_classes == 5


def test_two_level_label_response_is_refused_with_reason():
    rng = np.random.default_rng(7)
    data = {"y": rng.choice(["a", "b"], size=N), "x": rng.normal(size=N)}
    with pytest.raises(ValueError, match="two levels"):
        mb.autospec(data, "y")


def test_family_override_skips_inference():
    rng = np.random.default_rng(8)
    data = {"y": rng.choice(["a", "b"], size=N), "x": rng.normal(size=N)}
    _, _, family, _ = mb.autospec(data, "y", family=mb.Binomial())
    assert isinstance(family, mb.Binomial)


def test_response_not_in_data_is_refused():
    with pytest.raises(ValueError, match="not in the data"):
        mb.autospec({"y": np.arange(5.0)}, "z")


def test_mstop_override():
    rng = np.random.default_rng(9)
    data = {"y": rng.normal(size=N), "x": rng.normal(size=N)}
    _, _, _, mstop = mb.autospec(data, "y", mstop=250)
    assert mstop == 250


# --------------------------------------------------------------------------
# the recommendation object
# --------------------------------------------------------------------------


def test_the_recommendation_still_unpacks_as_the_old_tuple():
    """Every caller written against the four-tuple keeps working."""
    data = {"x": np.linspace(0, 1, 40), "y": np.linspace(0, 1, 40) ** 2}
    rec = mb.autospec(data, "y")
    formula, learners, family, mstop = rec
    assert formula == rec.formula and learners == rec.learners
    assert family is rec.family and mstop == rec.mstop
    assert len(rec) == 4 and rec[0] == formula


def test_the_recommendation_fits_what_it_recommends():
    rng = np.random.default_rng(0)
    n = 120
    data = {"x": rng.normal(size=n), "g": rng.integers(0, 3, size=n).astype(float)}
    data["y"] = np.sin(data["x"]) + rng.normal(scale=0.3, size=n)
    rec = mb.autospec(data, "y", mstop=30)
    model = rec.fit(data)
    assert model.mstop == 30
    assert np.all(np.isfinite(model.fitted_))


# W07, correctness priority 0: the replay used to rebuild the control with
# the default step length, so a caller's nu=0.03 silently became nu=0.1 and
# the fitted values differed from the explicit rate-0.03 fit (by up to 0.94
# on the 80-row probe that found this). The recommendation now carries the
# control it was asked for.
def test_the_replay_uses_the_requested_step_length():
    rng = np.random.default_rng(4)
    n = 80
    x = rng.uniform(-3, 3, n)
    data = {"x": x, "y": x + rng.normal(size=n)}
    rec = mb.autospec(data, "y", nu=0.03, mstop=15)
    replayed = rec.fit(data)
    explicit = mb.gamboost(rec.formula, data, family=rec.family,
                          control=mb.boost_control(15, 0.03))
    assert np.asarray(replayed.fitted()) == pytest.approx(
        np.asarray(explicit.fitted()), abs=0, rel=0)


def test_the_replay_uses_the_requested_step_length_with_a_probe_budget():
    """A budget can move the step count; the step length still replays."""
    rng = np.random.default_rng(4)
    n = 80
    x = rng.uniform(-3, 3, n)
    data = {"x": x, "y": x + rng.normal(size=n)}
    rec = mb.autospec(data, "y", nu=0.03, mstop=15, budget=6)
    replayed = rec.fit(data)
    assert rec.control == mb.boost_control(rec.mstop, 0.03)
    explicit = mb.gamboost(rec.formula, data, family=rec.family,
                           control=mb.boost_control(rec.mstop, 0.03))
    assert np.asarray(replayed.fitted()) == pytest.approx(
        np.asarray(explicit.fitted()), abs=0, rel=0)


def test_an_explicit_control_still_overrides_the_replay():
    rng = np.random.default_rng(4)
    n = 80
    x = rng.uniform(-3, 3, n)
    data = {"x": x, "y": x + rng.normal(size=n)}
    rec = mb.autospec(data, "y", nu=0.03, mstop=15)
    replayed = rec.fit(data, control=mb.boost_control(15, 0.2))
    explicit = mb.gamboost(rec.formula, data, family=rec.family,
                           control=mb.boost_control(15, 0.2))
    assert np.asarray(replayed.fitted()) == pytest.approx(
        np.asarray(explicit.fitted()), abs=0, rel=0)


def test_the_default_step_length_is_unchanged_without_a_request():
    data = {"x": np.linspace(0, 1, 40), "y": np.linspace(0, 1, 40) ** 2}
    rec = mb.autospec(data, "y", mstop=25)
    assert rec.control == mb.boost_control(25, 0.3)
    assert "step length" not in rec.report()


def test_a_constant_column_is_reported_and_left_out():
    """A learner on a constant column fits the intercept a second time."""
    rng = np.random.default_rng(1)
    n = 60
    data = {"x": rng.normal(size=n), "flat": np.full(n, 2.5),
            "y": rng.normal(size=n)}
    rec = mb.autospec(data, "y")
    assert "flat" not in rec.formula
    kinds = {f.column: f.kind for f in rec.hygiene}
    assert kinds.get("flat") == "constant"


def test_heavy_missingness_is_reported_with_what_it_costs():
    rng = np.random.default_rng(2)
    n = 80
    sparse = rng.normal(size=n)
    sparse[: int(0.7 * n)] = np.nan
    data = {"x": rng.normal(size=n), "sparse": sparse, "y": rng.normal(size=n)}
    rec = mb.autospec(data, "y")
    finding = next(f for f in rec.hygiene if f.column == "sparse")
    assert finding.kind == "missing"
    assert "70%" in finding.detail


def test_the_report_says_what_was_measured_and_what_was_assumed():
    data = {"x": np.linspace(0, 1, 40), "y": np.linspace(0, 1, 40) ** 2}
    text = mb.autospec(data, "y").report()
    assert "data hygiene" in text
    assert "not measured" in text          # no probe has run yet
    assert "model fits spent: 0" in text


# --------------------------------------------------------------------------
# the family probe: what the held-out spread says
# --------------------------------------------------------------------------


def _count_data(seed, overdispersed):
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.normal(size=n)
    mu = np.exp(0.6 + 0.5 * x)
    if overdispersed:
        y = rng.negative_binomial(1.5, 1.5 / (1.5 + mu)).astype(float)
    else:
        y = rng.poisson(mu).astype(float)
    return {"x": x, "y": y}


def test_an_overdispersed_count_is_not_left_on_poisson():
    """The dtype rule cannot see overdispersion: a negative-binomial count is
    a non-negative integer column like any other. The probe measures the
    held-out Pearson dispersion, which is what separates them."""
    data = _count_data(11, overdispersed=True)
    assert type(mb.autospec(data, "y").family).__name__ == "Poisson"
    rec = mb.autospec(data, "y", budget=40, mstop=60)
    assert type(rec.family).__name__ == "NBinomial"
    assert rec.evidence["family"].dispersion > 1.5
    assert "family" in rec.measured


def test_a_genuine_poisson_is_left_alone():
    data = _count_data(11, overdispersed=False)
    rec = mb.autospec(data, "y", budget=40, mstop=60)
    assert type(rec.family).__name__ == "Poisson"
    assert rec.evidence["family"].dispersion < 1.5


def test_a_constant_variance_positive_response_is_moved_off_gamma():
    """Every positive response is inferred as GammaReg, whose spread grows
    with the mean. When it does not grow, Gaussian is the family that fits."""
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    data = {"x": x, "y": 5.0 + 0.8 * x + rng.normal(scale=1.0, size=n)}
    assert type(mb.autospec(data, "y").family).__name__ == "GammaReg"
    rec = mb.autospec(data, "y", budget=40, mstop=60)
    assert type(rec.family).__name__ == "Gaussian"


def test_a_gamma_shaped_response_keeps_gamma():
    """Constant-shape gamma data keeps a gamma family. With a budget the
    probe may go one further and adopt GammaLSS (#228): its stabilized
    gradient routinely beats a boosted GammaReg whose fixed step
    overshoots, so the measured log score — not the generating model —
    decides between the two gamma objects."""
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    mu = np.exp(0.6 + 0.5 * x)
    rec = mb.autospec({"x": x, "y": rng.gamma(2.0, mu / 2.0)}, "y",
                      budget=40, mstop=60)
    assert type(rec.family).__name__ in ("GammaReg", "GammaLSS")
    assert rec.evidence["family"].spread_tracks_mean > 0.25


def test_zero_budget_spends_no_fits_and_measures_nothing():
    """The default must cost what it always cost: a dtype scan."""
    data = _count_data(3, overdispersed=True)
    rec = mb.autospec(data, "y")
    assert rec.fits == 0
    assert rec.measured == ()
    assert "family" in rec.defaulted
    assert type(rec.family).__name__ == "Poisson"


def test_a_budget_below_one_fold_sweep_runs_no_probe():
    """A probe that cannot complete is not started: the report says the
    answer was defaulted rather than implying a partial measurement."""
    data = _count_data(3, overdispersed=True)
    rec = mb.autospec(data, "y", budget=2)
    assert rec.fits == 0
    assert "family" in rec.defaulted


@pytest.mark.parametrize("budget", [0, 40])
def test_a_supplied_family_is_never_overridden_by_the_probe(budget):
    """A stated family is the modeller's decision; the probe does not revisit
    it. Other probes still run -- the budget is theirs to spend."""
    data = _count_data(11, overdispersed=True)
    rec = mb.autospec(data, "y", family=mb.Poisson(), budget=budget)
    assert type(rec.family).__name__ == "Poisson"
    assert "family" not in rec.defaulted
    assert "family" not in rec.measured
    assert "supplied by the caller; no family comparison" in rec.report()
    assert "not measured: family" not in rec.report()


# --------------------------------------------------------------------------
# the shape probe: does the learner match the column
# --------------------------------------------------------------------------


def _shaped_data(kind, seed=5, n=400):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    signal = {
        "threshold": np.where(x > 0, 2.0, -2.0),
        "linear": 1.5 * x,
        "curve": np.sin(1.5 * x) * 2,
    }[kind]
    return {"x": x, "y": signal + rng.normal(scale=0.4, size=n)}


def test_a_threshold_effect_moves_off_the_spline():
    """`bbs` is the dtype default for a float column. A step is what a tree
    fits and a smooth curve cannot."""
    rec = mb.autospec(_shaped_data("threshold"), "y", budget=200, mstop=60)
    row = next(r for r in rec.evidence["shapes"] if r.column == "x")
    assert row.incumbent == "bbs"
    assert row.chosen == "btree"
    assert "btree" in rec.formula
    assert "shapes" in rec.measured


def test_a_smooth_effect_keeps_the_spline():
    rec = mb.autospec(_shaped_data("curve"), "y", budget=200, mstop=60)
    row = next(r for r in rec.evidence["shapes"] if r.column == "x")
    assert row.chosen == "bbs"
    assert "bbs(x" in rec.formula


@pytest.mark.parametrize("preexploration", ["off", "native"])
def test_a_shape_that_wins_by_a_hair_does_not_displace_the_default(preexploration):
    """A straight line is fitted about equally by bols and bbs. Churning the
    specification for a fraction of a percent would be noise, so the default
    stands and the report carries both numbers."""
    rec = mb.autospec(_shaped_data("linear"), "y", budget=200, mstop=60,
                      preexploration=preexploration)
    row = next(r for r in rec.evidence["shapes"] if r.column == "x")
    assert row.chosen == "bbs"
    expected = {"bbs", "bols", "btree"} if preexploration == "off" else {"bbs", "bols"}
    assert {score.shape for score in row.scores} == expected
    assert "stands" in row.sentence


def test_a_categorical_column_has_no_shape_alternative():
    rng = np.random.default_rng(2)
    n = 200
    codes = np.array([10.0, 27.0, 53.0, 98.0])
    g = codes[rng.integers(0, 4, size=n)]
    data = {"g": g, "y": g + rng.normal(scale=0.5, size=n)}
    rec = mb.autospec(data, "y", budget=200, mstop=40)
    assert all(row.column != "g" for row in rec.evidence.get("shapes", ()))


# --------------------------------------------------------------------------
# the mstop sweep
# --------------------------------------------------------------------------


def test_a_boundary_optimum_is_reported_as_a_floor():
    """`mstop` at the end of the grid is the largest value tried, not an
    optimum. Saying otherwise invites a reader to stop where the sweep did."""
    rng = np.random.default_rng(4)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": np.sin(1.5 * x) * 3 + rng.normal(scale=0.2, size=n)}
    # An explicit step length is used as given, so the sweep can end at the
    # cap with the risk still falling.
    rec = mb.autospec(data, "y", budget=400, mstop=100, nu=0.1)
    row = rec.evidence["mstop"]
    assert row.at_boundary
    assert "floor" in row.sentence
    assert rec.mstop == row.chosen
    # Left to itself, autospec lengthens the steps until the risk turns.
    auto = mb.autospec(data, "y", budget=400, mstop=100)
    # autospec's own step (0.3) reaches the optimum inside the cap here; a
    # path still falling at the cap would switch to 0.5.
    assert auto.control.nu in (0.3, 0.5) and not auto.evidence["mstop"].at_boundary
    if auto.control.nu == 0.5:
        assert "step length: nu=0.5" in auto.report()


def test_a_response_with_no_signal_stops_almost_immediately():
    rng = np.random.default_rng(4)
    n = 300
    data = {"x": rng.uniform(-3, 3, size=n), "y": rng.normal(size=n)}
    rec = mb.autospec(data, "y", budget=400, mstop=100)
    row = rec.evidence["mstop"]
    assert not row.at_boundary
    assert rec.mstop < 20


# --------------------------------------------------------------------------
# the interaction screen
# --------------------------------------------------------------------------


def test_a_real_pair_is_reported_without_being_added():
    """The screen reads a training residual, not a held-out score, so it is a
    lead. The recommendation stays additive and hands over the term text."""
    rng = np.random.default_rng(0)
    n = 400
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    data = {"x1": x1, "x2": x2, "x3": x3,
            "y": x1 * x2 + 0.5 * x3 + rng.normal(scale=0.3, size=n)}
    rec = mb.autospec(data, "y", budget=500, mstop=80)
    leads = rec.evidence["interactions"]
    assert leads, "the generating interaction was not found"
    top = leads[0]
    assert {top.left, top.right} == {"x1", "x2"}
    assert top.kind == "tensor" and "%X%" in top.term
    assert "%X%" not in rec.formula


def test_an_additive_response_yields_no_pair():
    rng = np.random.default_rng(0)
    n = 400
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    data = {"x1": x1, "x2": x2, "x3": x3,
            "y": x1 + x2 + 0.5 * x3 + rng.normal(scale=0.3, size=n)}
    rec = mb.autospec(data, "y", budget=500, mstop=80)
    assert rec.evidence["interactions"] == ()


def test_a_group_column_is_paired_as_a_varying_coefficient():
    """A group coded as integers is still a group: `%X%` would grid its
    labels, and `bbs(x1, by=g)` would multiply the smooth by the code's
    numeric value. The lead is a random slope, one shrunken slope on x1 per
    group. The classification follows the term the recommendation chose;
    the codes are labels ({10, 27, 53}), not 0-based counts, which #313
    reads as a measurement."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    index = rng.integers(0, 3, size=n)
    g = np.array([10.0, 27.0, 53.0])[index]
    data = {"x1": x1, "g": g,
            "y": x1 * (index - 1.0) + rng.normal(scale=0.3, size=n)}
    rec = mb.autospec(data, "y", budget=500, mstop=80)
    top = rec.evidence["interactions"][0]
    assert top.kind == "varying"
    assert top.term == "brandom(g, by=x1)"


# --------------------------------------------------------------------------
# the modeller's prior, and the report
# --------------------------------------------------------------------------


def _threshold_with_second_column(seed=6, n=400):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    z = rng.normal(size=n)
    y = np.where(x > 0, 2.0, -2.0) + 0.5 * z + rng.normal(scale=0.4, size=n)
    return {"x": x, "z": z, "y": y}


def test_a_prior_is_the_baseline_and_only_a_measured_gain_displaces_it():
    """The modeller's specification is the anchor: each term it names is the
    incumbent, and a probe replaces one only by measuring better."""
    data = _threshold_with_second_column()
    rec = mb.autospec(data, "y", prior="y ~ bbs(x, df=4.0) + bbs(z, df=4.0)",
                      budget=400, mstop=80)
    by_column = {row.column: row for row in rec.evidence["shapes"]}
    assert by_column["x"].incumbent == "bbs" and by_column["x"].chosen == "btree"
    assert by_column["z"].chosen == "bbs"
    assert "btree(x)" in rec.formula and "bbs(z" in rec.formula


def test_a_prior_naming_some_columns_keeps_only_those():
    data = _threshold_with_second_column()
    rec = mb.autospec(data, "y", prior="y ~ bols(z)", budget=0)
    assert rec.formula == "y ~ bols(z)"
    assert "x" not in rec.formula


def test_a_prior_states_its_family_and_mstop():
    data = _threshold_with_second_column()
    rec = mb.autospec(data, "y", prior={"formula": "y ~ bbs(x)",
                                        "family": mb.Huber(), "mstop": 42})
    assert type(rec.family).__name__ == "Huber"
    assert rec.mstop == 42
    assert "family" not in rec.defaulted  # a stated family is not inferred or probed
    assert "supplied by the caller; no family comparison" in rec.report()


@pytest.mark.parametrize("bad,message", [
    ("bbs(x)", "must name the response"),
    ("q ~ bbs(x)", "models 'q' but the response is 'y'"),
    ({"family": None}, "mapping with a 'formula' key"),
])
def test_a_malformed_prior_is_refused_by_name(bad, message):
    data = _threshold_with_second_column()
    with pytest.raises(ValueError, match=message):
        mb.autospec(data, "y", prior=bad)


def test_the_report_orders_findings_by_what_they_cost_to_get_wrong():
    """A flat list of probe results makes the modeller rank them. The report
    ranks: data first, then the family every later score is measured under,
    then structure no amount of tuning recovers."""
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    g = rng.integers(0, 4, size=n).astype(float)
    mu = np.exp(0.4 + 0.6 * x + 0.2 * g)
    data = {"x": x, "g": g, "flat": np.full(n, 1.0),
            "y": rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)}
    rec = mb.autospec(data, "y", budget=600, mstop=80)
    text = rec.report()

    assert rec.priority[0].startswith("data:")          # the constant column
    assert any(line.startswith("family:") for line in rec.priority)
    assert "priority (most consequential first)" in text
    assert "model fits spent:" in text
    assert "flat" not in rec.formula
    # A dropped column cannot appear as half of an interaction lead.
    assert all("flat" not in (lead.left, lead.right)
               for lead in rec.evidence.get("interactions", ()))


# ─── declaration keywords: mechanism knowledge enters as arguments ──────


def test_monotone_declaration_writes_bmono_and_fits():
    rng = np.random.default_rng(5)
    n = 200
    spend = rng.uniform(-3, 3, size=n)
    data = {"spend": spend,
            "y": np.exp(0.4 + 0.9 * spend) + rng.normal(scale=0.1, size=n)}
    rec = mb.autospec(data, "y", monotone={"spend": "increasing"}, mstop=50)
    assert rec.formula == "y ~ bmono(spend, constraint=increasing, df=4.0)"
    assert rec.declared == ("spend",)
    learner = rec.learners[0]
    assert type(learner).__name__ == "Bmono"
    assert learner.constraint == "increasing"
    fit = rec.fit(data)
    assert sorted(set(fit.selected)) == ["spend"]


def test_declared_monotone_is_priced_not_replaced():
    """A wrong-direction constraint is reported as costly, never swapped out."""
    rng = np.random.default_rng(6)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": -x + rng.normal(scale=0.1, size=n)}
    rec = mb.autospec(data, "y", monotone={"x": "increasing"},
                      budget=200, mstop=40)
    # The declared term survives measurement even though it is the wrong shape.
    assert "bmono(x, constraint=increasing" in rec.formula
    row = next(r for r in rec.evidence["shapes"] if r.column == "x")
    assert row.chosen == row.incumbent
    assert "stands (declared)" in row.sentence
    assert "costs" in row.sentence


def test_declared_constraint_that_fits_best_costs_nothing():
    rng = np.random.default_rng(5)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": x + rng.normal(scale=0.1, size=n)}
    rec = mb.autospec(data, "y", monotone={"x": "increasing"},
                      budget=200, mstop=40)
    row = next(r for r in rec.evidence["shapes"] if r.column == "x")
    assert "stands (declared)" in row.sentence
    # Either it literally fits best, or nothing beats it by the margin; in
    # both cases the constraint is free at this data.
    assert ("costs nothing measurable" in row.sentence
            or "no alternative beats it" in row.sentence)


def test_coordinates_declaration_merges_two_columns():
    rng = np.random.default_rng(7)
    n = 200
    data = {"lon": rng.normal(size=n), "lat": rng.normal(size=n)}
    data["y"] = (np.sin(1.5 * data["lon"]) * np.cos(1.5 * data["lat"])
                 + rng.normal(scale=0.1, size=n))
    rec = mb.autospec(data, "y", coordinates=("lon", "lat"), mstop=60)
    assert "bspatial(lon, lat, df=6.0)" in rec.formula
    assert "bbs(lon" not in rec.formula and "bbs(lat" not in rec.formula
    assert set(rec.declared) == {"lon", "lat"}
    # df=4 would be exactly the tensor's 4-dimensional null space; the
    # declaration must not trigger the null-space warning.
    assert type(rec.learners[0]).__name__ == "Bspatial"
    rec.fit(data)


def test_adjacency_declaration_attaches_K():
    rng = np.random.default_rng(8)
    n = 240
    region = rng.integers(0, 9, size=n).astype(float)   # 9 levels, 0-based
    level = {i: float(i) for i in range(0, 9)}
    data = {"region": region,
            "y": np.array([level[r] for r in region])
                 + rng.normal(scale=0.3, size=n)}
    A = np.zeros((9, 9))
    for i in range(9):
        A[i, (i + 1) % 9] = A[(i + 1) % 9, i] = 1.0
    K = np.diag(A.sum(axis=1)) - A
    rec = mb.autospec(data, "y", adjacency={"region": K}, mstop=50)
    assert "bmrf(region, K=" in rec.formula
    learner = rec.learners[0]
    assert type(learner).__name__ == "Bmrf"
    assert np.allclose(np.asarray(learner.K), K)
    assert rec.declared == ("region",)
    rec.fit(data)


def test_basis_declaration_builds_buser():
    rng = np.random.default_rng(9)
    n = 200
    x = rng.uniform(-2, 2, size=n)
    X = np.column_stack([np.ones(n), x, x ** 2])
    K = np.diag([0.0, 0.0, 1.0])
    data = {"x": x, "y": X @ np.array([1.0, -2.0, 1.5])
                 + rng.normal(scale=0.1, size=n)}
    rec = mb.autospec(data, "y", basis={"x": (X, K)}, mstop=50)
    assert "buser(x, X=" in rec.formula
    learner = rec.learners[0]
    assert type(learner).__name__ == "Buser"
    assert np.allclose(np.asarray(learner.X), X)
    assert np.allclose(np.asarray(learner.K), K)
    rec.fit(data)


def test_prior_and_declaration_are_mutually_exclusive():
    rng = np.random.default_rng(5)
    x = rng.normal(size=100)
    data = {"x": x, "y": x + rng.normal(size=100)}
    with pytest.raises(ValueError, match="not both"):
        mb.autospec(data, "y", prior="y ~ bbs(x)",
                    monotone={"x": "increasing"})


def test_declarations_are_validated():
    rng = np.random.default_rng(5)
    x = rng.normal(size=100)
    data = {"x": x, "y": x + rng.normal(size=100)}
    with pytest.raises(ValueError, match="not a predictor"):
        mb.autospec(data, "y", monotone={"nope": "increasing"})
    with pytest.raises(ValueError, match="must be one of"):
        mb.autospec(data, "y", monotone={"x": "sideways"})
    with pytest.raises(ValueError, match="two distinct columns"):
        mb.autospec(data, "y", coordinates=("x",))
    with pytest.raises(ValueError, match="square matrix"):
        mb.autospec(data, "y", adjacency={"x": np.ones((3, 4))})


def test_report_distinguishes_declared_terms_when_unmeasured():
    rng = np.random.default_rng(5)
    n = 200
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": x + rng.normal(scale=0.1, size=n)}
    rec = mb.autospec(data, "y", monotone={"x": "increasing"})
    assert "stand as written" in rec.report()
    assert "dtype implies" not in rec.report()


def test_prior_non_catalogue_term_is_a_fixed_incumbent():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-3, 3, size=n)
    data = {"x": x, "y": x + rng.normal(scale=0.1, size=n)}
    rec = mb.autospec(data, "y", prior="y ~ bmono(x, constraint=increasing)",
                      budget=200, mstop=40)
    assert "bmono" in rec.formula
    row = next(r for r in rec.evidence["shapes"] if r.column == "x")
    assert row.chosen == row.incumbent
    assert "declared" in row.sentence


# --------------------------------------------------------------------------
# the family probe declines when the measurement is inside its own noise
# --------------------------------------------------------------------------


def test_a_count_on_the_decision_line_is_not_chosen_either_way():
    """A dispersion of 1.49 with a standard error of 0.07 does not
    distinguish Poisson from NBinomial: the same data could have measured
    1.43 or 1.55. The probe declines, keeps the dtype-inferred family, and
    says what would separate them, rather than reporting a verdict the
    noise could have flipped."""
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    mu = np.exp(0.6 + 0.5 * x)
    data = {"x": x,
            "y": rng.negative_binomial(5.5, 5.5 / (5.5 + mu)).astype(float)}
    rec = mb.autospec(data, "y", budget=40, mstop=60)
    row = rec.evidence["family"]
    assert abs(row.dispersion - 1.5) < 2 * (2.0 / row.n_scored) ** 0.5
    assert row.declined is True
    assert type(rec.family).__name__ == "Poisson"
    assert "declines to choose" in row.sentence
    assert "cvrisk" in row.sentence


def _mild_heteroskedasticity(seed=11, n=400, b=0.05):
    """A positive response whose spread grows with the level only slightly:
    the correlation lands between the constant-variance and Gamma lines."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu = 4.0 + 1.5 * x
    scale = 0.4 + b * (mu - mu.min())
    return {"x": x, "y": mu + scale * rng.normal(size=n)}


def test_a_spread_measurement_inside_noise_declines_to_choose():
    data = _mild_heteroskedasticity()
    rec = mb.autospec(data, "y", budget=40, mstop=60)
    row = rec.evidence["family"]
    assert row.declined is True
    assert "declines to choose" in row.sentence
    # The spread probe declines, so the dtype-inferred GammaReg is kept only
    # if the squared-error check against Gaussian does not reject it; either
    # verdict is stated in the sentence.
    assert type(rec.family).__name__ in ("GammaReg", "Gaussian")
    assert "Gaussian" in row.sentence


# --------------------------------------------------------------------------
# the folds follow the frame's own structure
# --------------------------------------------------------------------------


def test_a_date_column_orders_the_folds_without_being_asked():
    """Rows with a time direction must not be split at random: a random
    fold trains on the future to judge the past. A date column names the
    order on its own."""
    import datetime as dt

    rng = np.random.default_rng(3)
    n = 240
    data = {"x": rng.normal(size=n),
            "when": [dt.date(2020, 1, 1) + dt.timedelta(days=i)
                     for i in range(n)],
            "y": rng.normal(size=n) + 0.4 * np.linspace(-2, 2, n)}
    rec = mb.autospec(data, "y", budget=120, mstop=60)
    assert "rolling-origin" in rec.folds_description
    assert "when" in rec.folds_description
    assert "when" in rec.formula          # modelled, and ordering the folds
    assert rec.fits > 0


def test_shuffled_dates_still_order_the_folds():
    """The rows need not arrive in time order: the folds are built on the
    date column's order and mapped back to the rows as they sit in the
    frame. Only rows after the first half are judged, whatever order the
    frame stores them in."""
    import datetime as dt

    rng = np.random.default_rng(4)
    n = 400
    days = rng.permutation(n)
    x = rng.normal(size=n)
    data = {"x": x,
            "when": [dt.date(2020, 1, 1) + dt.timedelta(days=int(d))
                     for d in days],
            "y": np.exp(0.5 + 0.4 * x) * np.exp(0.1 * rng.normal(size=n))}
    rec = mb.autospec(data, "y", budget=60, mstop=60)
    assert "rolling-origin" in rec.folds_description
    row = rec.evidence["family"]
    # The judge set is the second half of the rows, not all of them.
    assert row.n_scored < n
    assert row.n_scored >= n // 2 - 1


def test_supplied_folds_override_the_detected_structure():
    import datetime as dt

    rng = np.random.default_rng(3)
    n = 240
    data = {"x": rng.normal(size=n),
            "when": [dt.date(2020, 1, 1) + dt.timedelta(days=i)
                     for i in range(n)],
            "y": rng.normal(size=n)}
    rec = mb.autospec(data, "y", folds=mb.cv_weights(n, type="kfold", B=4,
                                                     seed=1))
    assert "supplied by the caller" in rec.folds_description
    assert "rolling" not in rec.folds_description


def test_a_group_column_is_detected_even_with_many_levels():
    """A grouping code above the dtype cutoff is read as labels when the
    response says so, and the folds hold its groups out without a
    groups= declaration. The codes are sparse over their range (3 apart,
    not 0..29), so they do not read as a count (#313)."""
    rng = np.random.default_rng(0)
    n = 400
    index = rng.integers(0, 30, n)
    clinic = (index * 3).astype(float)
    x = rng.normal(size=n)
    data = {"x": x, "clinic": clinic,
            "y": rng.normal(size=30)[index] + 0.5 * x
                 + rng.normal(scale=0.5, size=n)}
    rec = mb.autospec(data, "y", budget=60, mstop=60)
    assert "brandom(clinic" in rec.formula
    assert rec.groups == ()
    assert "grouped folds" in rec.folds_description
    assert "clinic" in rec.folds_description


# --------------------------------------------------------------------------
# fits two probes need in common are made once
# --------------------------------------------------------------------------


def test_the_shape_probe_reuses_the_family_probes_incumbent_fits():
    """When the family probe leaves the inferred family standing, the
    specification it fitted on the shared folds is the shape probe's
    incumbent variant: those fold fits are made once and counted once."""
    data = _count_data(11, overdispersed=False)
    rec = mb.autospec(data, "y", budget=200, mstop=60)
    assert "family" in rec.measured and "shapes" in rec.measured
    assert type(rec.family).__name__ == "Poisson"
    assert rec.reused >= 5          # one fold fit per shared column
    assert "reused from an earlier probe" in rec.report()


def test_the_interaction_screen_reuses_the_mstop_sweeps_main_effect_fit():
    """The screen reads the residual of the main-effect path the mstop
    sweep already fitted, instead of fitting the same model again."""
    rng = np.random.default_rng(0)
    n = 400
    x1, x2 = (rng.normal(size=n) for _ in range(2))
    data = {"x1": x1, "x2": x2,
            "y": x1 * x2 + 0.5 * x1 + rng.normal(scale=0.3, size=n)}
    rec = mb.autospec(data, "y", budget=500, mstop=80)
    assert "mstop" in rec.measured and "interactions" in rec.measured
    assert rec.reused >= 1
    # The reuse is visible as work not spent: the screen cost no fit.
    assert "reused from an earlier probe" in rec.report()


# --------------------------------------------------------------------------
# a declared treatment is a different question
# --------------------------------------------------------------------------


def test_a_declared_treatment_is_left_out_and_points_at_the_causal_workflow():
    """Scoring a treatment by predictive association answers a different
    question from estimating its effect. The battery leaves it out and
    names the workflow that estimates effects."""
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    trt = rng.integers(0, 2, n).astype(float)
    data = {"x": x, "trt": trt, "y": 0.5 * x + 0.4 * trt
                               + rng.normal(scale=0.3, size=n)}
    rec = mb.autospec(data, "y", treatment="trt")
    assert "trt" not in rec.formula
    assert "bols(x)" not in rec.formula          # trt is not recoded as x
    assert rec.treatment == ("trt",)
    finding = next(f for f in rec.hygiene if f.column == "trt")
    assert finding.kind == "treatment"
    assert "aipw" in finding.consequence
    assert rec.priority[0].startswith("treatment:")
    rec.fit(data)


def test_a_causal_spec_brings_its_post_treatment_columns_with_it():
    from mboostrs.causal import CausalSpec

    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=n)
    data = {"x": x, "trt": rng.integers(0, 2, n).astype(float),
            "mediator": rng.normal(size=n),
            "y": 0.5 * x + rng.normal(scale=0.3, size=n)}
    spec = CausalSpec(treatment="trt", outcome="y", covariates=("x",),
                      post_treatment=("mediator",),
                      time_zero="enrolment", follow_up="12 months",
                      adjustment_rationale="x is measured at baseline and "
                      "screens the treatment assignment")
    rec = mb.autospec(data, "y", treatment=spec)
    assert "trt" not in rec.formula
    assert "mediator" not in rec.formula
    assert set(rec.treatment) == {"trt", "mediator"}
    rec.fit(data)


@pytest.mark.parametrize("kwargs,message", [
    ({"treatment": "nope"}, "not a column in the data"),
    ({"treatment": "y"}, "must not name the response"),
])
def test_a_bad_treatment_declaration_is_refused_by_name(kwargs, message):
    rng = np.random.default_rng(2)
    data = {"x": rng.normal(size=100), "trt": rng.integers(0, 2, 100),
            "y": rng.normal(size=100)}
    with pytest.raises(ValueError, match=message):
        mb.autospec(data, "y", **kwargs)


def test_a_causal_spec_for_another_outcome_is_refused():
    from mboostrs.causal import CausalSpec

    rng = np.random.default_rng(3)
    data = {"x": rng.normal(size=100), "trt": rng.integers(0, 2, 100),
            "y": rng.normal(size=100), "other": rng.normal(size=100)}
    spec = CausalSpec(treatment="trt", outcome="other", covariates=("x",),
                      time_zero="enrolment", follow_up="12 months",
                      adjustment_rationale="x is measured at baseline")
    with pytest.raises(ValueError, match="outcome 'other'"):
        mb.autospec(data, "y", treatment=spec)


def test_several_group_terms_fold_by_the_finest_one():
    """A frame can fit a unit grouping (children) and phase groupings
    (exam round, season) as brandom terms at once. Combining them into one
    fold label makes every child-round pair its own group and quietly
    turns the grouped folds into random ones; the detection holds out the
    grouping with the most levels and names the rest as split. The codes
    are labels, not 0-based counts, which #313 would read as a count."""
    rng = np.random.default_rng(6)
    n = 320
    child_index = rng.integers(0, 20, n)
    child = (child_index * 7 + 3).astype(float)       # the unit: 20 levels
    season_index = rng.integers(0, 4, n)
    season = np.array([10.0, 27.0, 53.0, 98.0])[season_index]
    x = rng.normal(size=n)
    data = {"child": child, "season": season, "x": x,
            "y": rng.normal(size=20)[child_index] + x
                 + rng.normal(size=4)[season_index]
                 + rng.normal(scale=0.5, size=n)}
    rec = mb.autospec(data, "y", budget=60, mstop=60)
    assert "brandom(child" in rec.formula and "brandom(season" in rec.formula
    assert "groups of child out" in rec.folds_description
    assert "season" in rec.folds_description
    assert "split" in rec.folds_description


def test_a_binary_response_does_not_raise_continuous_family_findings():
    """A 0/1 float response is the Binomial family's own coding. The
    zero-mass and few-valued findings used to fire on it and warn about a
    family choice the tool itself had already made correctly."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    data = {"x": x, "y": rng.integers(0, 2, n).astype(float)}
    rec = mb.autospec(data, "y")
    assert type(rec.family).__name__ == "Binomial"
    kinds = {finding.kind for finding in rec.hygiene if finding.column == "y"}
    assert "zero-inflated" not in kinds
    assert "few-valued" not in kinds


def test_a_supplied_family_on_a_binary_response_still_gets_the_findings():
    """The suppression follows the inference: override the family on the
    same response and the coding findings return, because the fit the
    caller chose is the one they would warn about."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    data = {"x": x, "y": rng.integers(0, 2, n).astype(float)}
    rec = mb.autospec(data, "y", family=mb.Gaussian())
    kinds = {finding.kind for finding in rec.hygiene if finding.column == "y"}
    assert "few-valued" in kinds


# --------------------------------------------------------------------------
# declared measurements, judged step counts, duplicate-axis leads
# --------------------------------------------------------------------------


def test_declared_measurements_override_the_label_reading():
    """A few-valued column the dtype rule reads as labels can be declared
    an ordered quantity: one slope instead of group effects, the shape
    probe prices the alternatives, and the column leaves the fold
    grouping."""
    rng = np.random.default_rng(8)
    n = 300
    cosine = rng.choice([-1.0, 0.0, 1.0], size=n)   # whole-number coordinates
    data = {"cosine": cosine, "x": rng.normal(size=n),
            "y": 0.8 * cosine + 0.4 * rng.normal(size=n)}
    bare = mb.autospec(data, "y")
    assert "brandom(cosine, df=4.0)" in bare.formula

    rec = mb.autospec(data, "y", measurements="cosine", budget=150, mstop=60)
    assert "bols(cosine)" in rec.formula
    finding = next(f for f in rec.hygiene if f.column == "cosine")
    assert "declared as measurements" in finding.consequence
    if rec.evidence.get("shapes"):
        row = next(r for r in rec.evidence["shapes"] if r.column == "cosine")
        assert row.incumbent == "bols"
    # A declared measurement is not a grouping for the folds.
    assert "cosine" not in rec.folds_description


def test_measurement_declarations_are_validated():
    rng = np.random.default_rng(8)
    data = {"x": rng.normal(size=100), "g": rng.integers(0, 4, 100),
            "y": rng.normal(size=100)}
    with pytest.raises(ValueError, match="not a predictor"):
        mb.autospec(data, "y", measurements="nope")
    with pytest.raises(ValueError, match="not a predictor"):
        mb.autospec(data, "y", measurements="y")
    with pytest.raises(ValueError, match="not both"):
        mb.autospec(data, "y", measurements="g", groups="g")
    with pytest.raises(ValueError, match="not both"):
        mb.autospec(data, "y", prior="y ~ bbs(x)", measurements="x")


def test_a_duplicate_axis_pair_is_reported_not_proposed(monkeypatch):
    """Year and date are one axis measured twice; a lead on the pair is
    arithmetic the main effects already carry, and the screen says so
    instead of proposing a term. The screen itself is replaced here: with
    response-scale residuals the polio fit leaves no lead on the pair, so the
    classification is checked on a lead fed to it directly."""
    import mboostrs
    from mboostrs.datasets import polio

    def screen(data, residual, **kwargs):
        return [mboostrs.ScreenedPair("year", "date", 0.2)]

    monkeypatch.setattr(mboostrs, "screen_interactions", screen)
    rec = mb.autospec(polio(), "cases", budget=400, mstop=100, interactions="adopt")
    kinds = {(lead.left, lead.right): lead for lead in rec.evidence["interactions"]}
    duplicate = kinds[("year", "date")]
    assert duplicate.kind == "duplicate-axis"
    assert duplicate.term == ""
    assert "one axis twice" in str(duplicate)
    # A duplicate is neither adopted nor proposed as the structure to act on.
    assert not rec.evidence.get("adopted_interactions")
    assert "%X%" not in rec.formula
    assert all(line.startswith("structure:") is False
               or "duplicate" not in line for line in rec.priority)


def test_adopted_interactions_enter_the_formula_and_the_fit():
    """`interactions="adopt"` adds the screened pair term, re-tunes mstop, and
    `rec.fit` expands the tensor; `"report"` leaves the formula additive."""
    rng = np.random.default_rng(0)
    n = 600
    data = {f"x{i}": rng.uniform(-2, 2, n) for i in range(1, 5)}
    data["y"] = (np.sin(data["x1"]) + 0.8 * data["x1"] * data["x2"]
                 + 0.5 * data["x3"] + rng.normal(0, 0.4, n))

    reported = mb.autospec(data, "y", budget=300, mstop=150)
    adopted = mb.autospec(data, "y", budget=300, mstop=150, interactions="adopt")

    assert "%X%" not in reported.formula
    assert "%X%" in adopted.formula
    assert adopted.evidence["adopted_interactions"]
    fit = adopted.fit(data)
    assert any("%X%" in name for name in fit.base_learner_names())
    with pytest.raises(ValueError, match="report"):
        mb.autospec(data, "y", interactions="always")


def test_the_interaction_screen_finds_no_pair_in_an_additive_logit():
    """The screen used `observed - fitted_`, which for a Binomial fit mixes
    0/1 outcomes with the half-log-odds predictor; an additive logit then
    showed pair "leads" between every two signal columns."""
    rng = np.random.default_rng(0)
    n = 1000
    data = {f"x{i}": rng.uniform(-2, 2, n) for i in range(1, 6)}
    logit = 1.5 * (0.5 * data["x1"] + np.sin(1.5 * data["x2"]) + 0.4 * data["x3"] ** 2 - 0.5)
    data["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    rec = mb.autospec(data, "y", budget=300, mstop=150, interactions="adopt")
    assert type(rec.family).__name__ == "Binomial"
    assert not rec.evidence.get("adopted_interactions")


def test_a_heavy_tailed_predictor_is_proposed_on_the_log_scale():
    """On the raw scale one cross-validation fold holding out the extreme rows
    can decide mstop; autospec proposes bbs(log(x)) and the fit replays it."""
    rng = np.random.default_rng(0)
    n = 500
    occupancy = np.exp(rng.normal(1.0, 0.8, n))
    occupancy[:3] = [300.0, 450.0, 600.0]
    data = {"occupancy": occupancy, "x": rng.normal(size=n)}
    data["y"] = np.log(occupancy) + data["x"] + rng.normal(0, 0.3, n)
    rec = mb.autospec(data, "y", family=mb.Gaussian())
    assert "bbs(log(occupancy)" in rec.formula
    fit = rec.fit(data)
    new = {"occupancy": np.array([2.0, 50.0]), "x": np.array([0.0, 0.0])}
    assert np.all(np.isfinite(np.asarray(fit.predict(newdata=new)).ravel()))


def test_a_large_frame_is_pointed_at_the_binned_route():
    rng = np.random.default_rng(0)
    n = 120_000
    data = {"x": rng.uniform(size=n)}
    data["y"] = np.sin(3 * data["x"]) + rng.normal(size=n)
    rec = mb.autospec(data, "y")
    finding = next(f for f in rec.hygiene if f.kind == "large frame")
    assert "bins=256" in finding.consequence
    small = mb.autospec({k: v[:2000] for k, v in data.items()}, "y")
    assert not any(f.kind == "large frame" for f in small.hygiene)


def test_the_probes_judge_candidates_at_the_depth_the_data_need():
    """A signal that needs thousands of steps: at the default 100 steps every
    candidate is still close to the intercept-only risk, so the probes used to
    compare unfitted models and the final sweep stopped after three
    doublings. The recommendation now reaches the step count where the
    cross-validated risk turns, and the shape evidence is on that scale."""
    rng = np.random.default_rng(0)
    n = 600
    data = {f"x{i}": rng.uniform(-1, 1, n) for i in range(1, 5)}
    data["y"] = (np.sin(3 * np.pi * data["x1"]) + 1.0 * (data["x2"] > 0.2)
                 + rng.normal(0, 0.3, n))
    rec = mb.autospec(data, "y", budget=200, seed=0, nu=0.1)
    assert rec.mstop > 1600
    assert not rec.evidence["mstop"].at_boundary
    shapes = {row.column: row for row in rec.evidence["shapes"]}
    # The step at 0.2 is a threshold; the three-period sine is a smooth.
    assert shapes["x2"].chosen == "btree"
    assert shapes["x1"].chosen == "bbs"
    best_risk = min(s.risk for s in shapes["x1"].scores)
    assert best_risk < 0.5 * np.var(data["y"])


def test_noise_columns_are_deselected_after_tuning():
    """A cross-validated boosting fit keeps every term selected even once;
    the recommendation drops terms under 1% of the tuned fit's risk
    reduction and says which, unless deselect=False."""
    rng = np.random.default_rng(5)
    n = 500
    data = {f"x{i}": rng.uniform(-1, 1, n) for i in range(1, 11)}
    data["y"] = np.sin(np.pi * data["x1"]) + data["x2"] + rng.normal(0, 0.5, n)
    rec = mb.autospec(data, "y", budget=200, seed=0)
    kept = {v for term in rec.formula.split("~")[1].split("+")
            for v in mb.formula.term_variables(term)}
    assert kept == {"x1", "x2"}
    dropped = {v for term, _ in rec.evidence["deselect"].dropped
               for v in mb.formula.term_variables(term)}
    assert dropped == {f"x{i}" for i in range(3, 11)}
    assert "risk reduction" in rec.report()
    kept_all = mb.autospec(data, "y", budget=200, seed=0, deselect=False)
    assert "deselect" not in kept_all.evidence
    assert len(kept_all.formula.split("+")) == 10


def test_a_declared_column_is_never_deselected():
    rng = np.random.default_rng(6)
    n = 400
    data = {"x1": rng.uniform(-1, 1, n), "noise": rng.uniform(-1, 1, n)}
    data["y"] = data["x1"] + rng.normal(0, 0.5, n)
    rec = mb.autospec(data, "y", budget=200, seed=0, monotone={"noise": "increasing"})
    assert "noise" in rec.formula


def test_autospec_retries_a_diverging_count_fit_with_line_search():
    """The recommendation and its final fit use the same safe step rule."""
    rng = np.random.default_rng(5)
    n = 160
    x = rng.normal(size=n)
    y = np.maximum(np.rint(150 + 25 * x + rng.normal(scale=7, size=n)), 1.0).astype(int)
    data = {"y": y, "x": x}

    rec = mb.autospec(data, "y", budget=30, mstop=20, seed=0)
    fit = rec.fit(data)

    # The dtype guess (Poisson) is judged before it is fitted, so its fixed
    # steps never overflow the search; whatever step rule the recommendation
    # carries, its final fit replays it and stays finite.
    assert fit.control.step == rec.control.step
    if type(rec.family).__name__ in ("Poisson", "NBinomial", "GammaReg"):
        assert rec.control.step == "line_search" or "step" not in rec.evidence
    assert np.isfinite(fit.risk_).all()


def test_a_noise_column_changes_learner_only_on_the_margin():
    """The paired-fold rule for changing a learner applies to columns with
    signal; across many noise columns two standard errors are passed by
    chance, and a tree on a noise column then fits chance splits."""
    rng = np.random.default_rng(1)
    n = 800
    data = {f"z{i}": rng.uniform(-1, 1, n) for i in range(20)}
    data["x"] = rng.uniform(-1, 1, n)
    eta = 2 * np.sin(np.pi * data["x"])
    data["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    rec = mb.autospec(data, "y", family=mb.Binomial(), budget=200, seed=1)
    # A noise column may be simplified to a line; it is never given a tree
    # (which fits chance splits) without the 2% margin.
    for row in rec.evidence["shapes"]:
        if row.column.startswith("z") and row.chosen == "btree":
            incumbent = next(s for s in row.scores if s.shape == row.incumbent)
            chosen = next(s for s in row.scores if s.shape == row.chosen)
            assert (incumbent.risk - chosen.risk) / incumbent.risk >= 0.02


def test_adopted_pairs_must_lower_held_out_risk():
    """The screen reads the training residual, so with many columns and few
    rows it proposes chance pairs. An adopted pair stays only when removing
    it costs 2% of the held-out risk or two paired fold standard errors; the
    others are listed as removed."""
    rng = np.random.default_rng(2)
    n = 400
    data = {f"x{i}": rng.uniform(-1, 1, n) for i in range(12)}
    data["y"] = (np.sin(np.pi * data["x0"]) + data["x1"] + 1.5 * data["x2"] * data["x3"]
                 + rng.normal(0, 0.5, n))
    rec = mb.autospec(data, "y", budget=300, seed=0, interactions="adopt")
    adopted = rec.evidence.get("adopted_interactions", ())
    assert any("x2" in t and "x3" in t for t in adopted)
    for _term, rel, z in rec.evidence.get("rejected_interactions", ()):
        assert rel < 0.02 and z < 2.0
    if rec.evidence.get("rejected_interactions"):
        assert "removed because held-out risk" in rec.report()


def test_measurement_and_group_declarations_are_never_deselected():
    rng = np.random.default_rng(7)
    n = 400
    data = {"x": rng.uniform(-1, 1, n), "code": rng.integers(0, 3, n).astype(float),
            "clinic": rng.integers(0, 20, n).astype(float)}
    data["y"] = np.sin(np.pi * data["x"]) + rng.normal(0, 0.5, n)
    rec = mb.autospec(data, "y", budget=200, seed=0, measurements="code", groups="clinic")
    assert "code" in rec.formula and "clinic" in rec.formula


def test_deselection_keeps_small_terms_that_carry_held_out_signal():
    """Fifteen real effects of 0.1% of the risk reduction each and three noise
    columns next to one dominant effect: every small term is under the 1%
    share, but dropping the real ones together multiplies the held-out risk,
    so only what can go without raising it more than 1% is dropped."""
    rng = np.random.default_rng(46)
    n = 1500
    data = {f"x{j}": rng.uniform(-1, 1, n) for j in range(19)}
    y = 10 * data["x0"] + sum(0.3 * data[f"x{j}"] for j in range(1, 16))
    data["y"] = y + rng.normal(scale=0.1, size=n)
    rec = mb.autospec(data, "y", budget=200, seed=0)
    evidence = rec.evidence["deselect"]
    from mboostrs.formula import term_variables

    dropped = {v for term, _ in evidence.dropped for v in term_variables(term)}
    assert not dropped & {f"x{j}" for j in range(1, 16)}
    assert evidence.held_out_loss is not None and evidence.held_out_loss <= 0.01
    assert "kept" in evidence.sentence


def test_deselect_accepts_a_share():
    rng = np.random.default_rng(47)
    n = 600
    data = {f"x{j}": rng.uniform(-1, 1, n) for j in range(6)}
    data["y"] = np.sin(3 * data["x0"]) + data["x1"] + rng.normal(scale=0.5, size=n)
    rec = mb.autospec(data, "y", budget=120, deselect=0.025)
    assert rec.evidence["deselect"].threshold == 0.025
    with pytest.raises(ValueError, match="deselect must be"):
        mb.autospec(data, "y", deselect=0.7)


def test_multinomial_with_a_categorical_predictor():
    """A Multinomial model takes a factor through its level indicators, and
    autospec neither offers a tree nor screens pairs for it."""
    rng = np.random.default_rng(48)
    n = 1200
    g = rng.choice(["a", "b", "c", "d"], n)
    x = rng.uniform(-1, 1, n)
    effect = {"a": 1.2, "b": 0.0, "c": -0.8, "d": 0.4}
    logits = np.column_stack([np.zeros(n), np.array([effect[v] for v in g]) + np.sin(2 * x),
                              0.8 * x])
    prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(prob, axis=1)).sum(axis=1).astype(float)
    data = {"g": g, "x": x, "y": y}
    rec = mb.autospec(data, "y", family=mb.Multinomial(n_classes=3), budget=120)
    assert "btree" not in rec.formula
    assert "not screened" in rec.report()
    model = rec.fit(data)
    p = np.asarray(model.predict(newdata={"g": np.array(["a", "c", "z"]),
                                          "x": np.zeros(3)},
                                 type="response", unseen_levels="zero_effect"))
    assert p[0, 1] > p[1, 1]
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_label_family_response_draws_no_coding_warnings():
    """Integer class codes are the family's coding, not a zero mass or a
    few-valued float response, when the family models labels."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-1, 1, n)
    y = rng.integers(0, 3, n).astype(float)
    rec = mb.autospec({"y": y, "x": x}, "y", family=mb.Multinomial(n_classes=3), budget=0)
    report = rec.report()
    assert "exactly zero" not in report
    assert "distinct values in a float response" not in report


def test_residual_tree_is_adopted_for_higher_order_structure_only():
    """A tree over the numeric columns is added when held-out risk says the
    main effects and pairs leave structure unexplained, and declined on
    additive data."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.uniform(-1, 1, (n, 4))
    data = {f"x{i}": X[:, i] for i in range(4)}
    threeway = X[:, 0] + np.sin(3 * X[:, 1] * X[:, 2] * X[:, 3]) + rng.normal(0, .3, n)
    rec = mb.autospec(data | {"y": threeway}, "y", budget=200, residual_tree=True)
    assert rec.evidence["residual_tree"]["adopted"]
    assert "btree(x0, x1, x2, x3, maxdepth=5, selection=sse)" in rec.formula
    assert "residual tree:" in rec.report()
    additive = X[:, 0] + X[:, 1] ** 2 + rng.normal(0, .3, n)
    rec = mb.autospec(data | {"y": additive}, "y", budget=200, residual_tree=True)
    assert not rec.evidence["residual_tree"]["adopted"]
    assert "btree" not in rec.formula
    model = mb.autospec(data | {"y": threeway}, "y", budget=200, residual_tree=True).fit(
        data | {"y": threeway})
    assert np.isfinite(np.asarray(model.predict(newdata=data))).all()


def test_btree_splits_a_factor_into_level_sets():
    """A tree term takes a categorical column: it splits the levels into two
    sets, and an unseen level follows the heavier child."""
    rng = np.random.default_rng(0)
    n = 1200
    g = rng.choice(["a", "b", "c", "d", "e"], n)
    x = rng.uniform(-1, 1, n)
    y = np.where(np.isin(g, ["a", "c"]), 2 * x, -x) + rng.normal(0, .2, n)
    data = {"g": g, "x": x, "y": y}
    model = mb.tune_mstop("y ~ btree(g, x, maxdepth=3)", data, seed=0, max_mstop=5000).model
    new = {"g": np.array(["a", "b", "z"]), "x": np.array([0.8, 0.8, 0.8])}
    pred = np.asarray(model.predict(newdata=new, unseen_levels="zero_effect"))
    assert pred[0] > 1.2 and pred[1] < -0.5
    # b, d, e hold three fifths of the rows, so the unseen label goes with
    # them (deeper splits among the three may route it to one or another).
    assert pred[2] < -0.5
    with pytest.raises(ValueError, match="not present"):
        model.predict(newdata=new)


def test_log_link_family_is_fitted_with_the_better_step_rule():
    """A Gamma response in the hundreds of thousands stops after a few fixed
    steps; autospec also runs the line search and keeps the recommendation
    with the lower cross-validated risk."""
    rng = np.random.default_rng(5)
    n = 800
    x, z = rng.uniform(-1, 1, (2, n))
    mean = np.exp(12 + np.sin(2 * x) + 0.5 * z)
    y = rng.gamma(shape=4.0, scale=mean / 4.0)
    data = {"x": x, "z": z, "y": y}
    rec = mb.autospec(data, "y", family=mb.GammaReg(), budget=60)
    fixed_only = mb.autospec(data, "y", family=mb.GammaReg(), budget=60, step="fixed")
    # Both searches ran; the fits of both are counted.
    assert rec.fits > fixed_only.fits
    step = rec.evidence.get("step", "")
    if step.startswith("line_search"):
        assert "lower cross-validated risk" in step


def test_nonnegative_response_with_zeros_tries_a_log_link():
    """A non-negative measurement with zeros and multiplicative effects is
    fitted with quasi-Poisson when it predicts clearly better than Gaussian,
    and stays Gaussian on additive data."""
    rng = np.random.default_rng(2)
    n = 900
    x, z = rng.uniform(-1, 1, (2, n))
    mean = np.exp(1.5 * x + 1.5 * z)
    y = np.where(rng.uniform(size=n) < 0.2, 0.0, rng.gamma(2.0, mean / 2.0))
    rec = mb.autospec({"x": x, "z": z, "y": y}, "y", budget=200)
    assert type(rec.family).__name__ == "Poisson"
    assert "log link (quasi-Poisson)" in rec.report()
    additive = np.maximum(0.0, 2 + x + z + rng.normal(0, .5, n)) * (rng.uniform(size=n) > 0.1)
    rec = mb.autospec({"x": x, "z": z, "y": additive}, "y", budget=200)
    assert type(rec.family).__name__ == "Gaussian"


def test_ctree_pvalue_selection_does_not_favour_many_level_factors():
    """Among noise columns, ctree's statistic favours a factor with many
    levels at the first split (its statistic has L - 1 degrees of freedom);
    the p-value rule does not."""
    rng = np.random.default_rng(0)
    n = 600
    picks = {"ctree": 0, "ctree_pvalue": 0}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        data = {"g": rng.choice([f"l{i}" for i in range(30)], n),
                "x": rng.uniform(-1, 1, n), "y": rng.normal(size=n)}
        for rule in picks:
            model = mb.gamboost(f"y ~ btree(g, x, selection={rule})", data,
                                control=mb.boost_control(1))
            tree = np.asarray(model.predict(newdata={"g": np.array(["l0"] * 3),
                                                     "x": np.array([-0.9, 0.0, 0.9])}))
            # A split on x changes the prediction across x; a split on g does not.
            picks[rule] += int(np.ptp(tree) == 0)
    assert picks["ctree"] > picks["ctree_pvalue"]


def test_a_tree_surrogate_proposes_pairs_the_residual_screen_needs_help_with():
    """Two interactions without main effects: the surrogate's H statistic
    proposes both pairs and held-out risk adopts them."""
    rng = np.random.default_rng(3)
    n = 1500
    X = rng.uniform(-1, 1, (n, 6))
    data = {f"x{i}": X[:, i] for i in range(6)}
    data["y"] = X[:, 0] + np.sin(3 * X[:, 1] * X[:, 2]) + X[:, 3] * X[:, 4] + rng.normal(0, .3, n)
    rec = mb.autospec(data, "y", budget=300, interactions="adopt")
    adopted = rec.evidence.get("adopted_interactions", ())
    assert any("x1" in t and "x2" in t for t in adopted)
    assert any("x3" in t and "x4" in t for t in adopted)


def test_column_screen_names_each_columns_marginal_shape():
    rng = np.random.default_rng(0)
    n = 1000
    data = {name: rng.uniform(-1, 1, n) for name in ("line", "curve", "step", "noise")}
    data["g"] = rng.choice(list("abcd"), n)
    data["y"] = (data["line"] + 2 * data["curve"] ** 2 + (data["step"] > 0.3)
                 + (data["g"] == "a") + rng.normal(0, .3, n))
    hints = {row["column"]: row["hint"] for row in mb.column_screen(data, "y", output="records")}
    assert hints == {"line": "line", "curve": "curve", "step": "step",
                     "g": "levels", "noise": "none"}
