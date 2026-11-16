"""Comparisons and attachments refuse rows that do not share data (issue #60).

compare_specifications under na_action='omit' scored each candidate on the
rows complete for its own columns, then refused the pair with "record-fold-
weight pairing"; it now scores every candidate on the rows complete for all
of them and reports the excluded rows. llm_explain_prompt(cv=...) accepted a
cross-validation result fitted to another response; results now carry a
digest of the response and weights, checked against the model.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import mboostrs as mb


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 2, n)
    z = rng.normal(size=n)
    z[:30] = np.nan
    return {"x": x, "y": np.exp(0.5 * x + rng.normal(0, 0.2, n)), "z": z}, rng


@pytest.mark.parametrize("frame", [dict, pd.DataFrame])
def test_candidates_share_one_sample(frame):
    data, _ = _data()
    with pytest.warns(UserWarning, match="30 row"):
        rows = mb.compare_specifications(["bbs(x)", "bbs(x) + bols(z)"], frame(data), "y",
                                         na_action="omit")
    assert [r["n evaluations"] for r in rows] == [170, 170]
    assert [r["common sample excluded rows"] for r in rows] == [30, 30]
    # The reference is scored on the common rows too: the same as passing them.
    complete = {k: v[30:] for k, v in data.items()}
    alone = mb.compare_specifications(["bbs(x)"], complete, "y")
    assert rows[0]["out-of-fold risk"] == pytest.approx(alone[0]["out-of-fold risk"])


def test_a_fold_scheme_is_subset_with_the_sample():
    data, _ = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = mb.compare_specifications(["bbs(x)", "bbs(x) + bols(z)"], data, "y",
                                         na_action="omit",
                                         folds=mb.grouped_folds(np.arange(200) % 10, B=5))
    assert [r["n evaluations"] for r in rows] == [170, 170]


@pytest.mark.parametrize("kind", ["gamboost", "glmboost"])
def test_explanation_refuses_a_cv_result_from_other_data(kind):
    data, rng = _data()
    data = {"x": data["x"], "y": data["y"]}
    formula = "y ~ bbs(x)" if kind == "gamboost" else "y ~ x"
    fit = getattr(mb, kind)(formula, data=data, control=mb.boost_control(40))
    other = getattr(mb, kind)(formula, data={"x": data["x"], "y": rng.permutation(data["y"])},
                              control=mb.boost_control(40))
    folds = mb.cv_weights(200, "kfold", 5)
    assert fit.cvrisk(folds=folds).data_fingerprint
    fit.llm_explain_prompt(cv=fit.cvrisk(folds=folds))
    with pytest.raises(ValueError, match="another response"):
        fit.llm_explain_prompt(cv=other.cvrisk(folds=folds))


@pytest.mark.parametrize("kind", ["gamboost", "glmboost"])
def test_explanation_refuses_a_stabsel_result_from_other_data(kind):
    """Same formula, same learner count, permuted response: the learner-count
    check passes, the data digest does not."""
    data, rng = _data()
    data = {"x": data["x"], "z": rng.normal(size=200), "y": data["y"]}
    formula = "y ~ bbs(x) + bbs(z)" if kind == "gamboost" else "y ~ x + z"
    fit = getattr(mb, kind)(formula, data=data, control=mb.boost_control(40))
    other = getattr(mb, kind)(
        formula, data={"x": data["x"], "z": data["z"], "y": rng.permutation(data["y"])},
        control=mb.boost_control(40))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = fit.stabsel(q=1, PFER=1, B=10, verbose=False)
        foreign = other.stabsel(q=1, PFER=1, B=10, verbose=False)
    assert own.data_fingerprint and own.data_fingerprint != foreign.data_fingerprint
    fit.llm_explain_prompt(stabsel=own)
    with pytest.raises(ValueError, match="stabsel= result was computed on another response"):
        fit.llm_explain_prompt(stabsel=foreign)


# ── comparison identity across the ranking helpers ──────────────────────────
#
# Each helper is exercised for the three outcomes: rank (same rows, response,
# effective weights and response scale), refuse (a permuted response, a
# different kept sample, different effective weights, retained evaluations
# scored under different settings), and segregate (same rows on a different
# response scale, or a different weight policy), shown with a comparison reason.

CONTROL = mb.boost_control(60, 0.1)


def _sample_data(seed=1, n=120):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.exp(0.5 + 0.8 * x + rng.normal(0, 0.3, n))
    return {"y": y, "x": x, "z": z}


def _glm(formula, data, **kwargs):
    return mb.glmboost(formula, data=data, family=mb.Gaussian(), control=CONTROL, **kwargs)


def _permuted(data, seed=5):
    """The same response values in another row order, with the same length."""
    rng = np.random.default_rng(seed)
    out = dict(data)
    out["y"] = data["y"][rng.permutation(len(data["y"]))]
    return out


def _gaic(*models):
    return mb.gaic_table(*models, output="records")


def _compare(*models):
    return mb.compare_models(*models, output="records")


MODEL_HELPERS = pytest.mark.parametrize("helper", [_gaic, _compare], ids=["gaic_table", "compare_models"])


# ── identity records ────────────────────────────────────────────────────────

def test_glmboost_fit_records_identity_fields():
    fit = _glm("y ~ bols(x)", _sample_data())
    identity = fit._comparison_identity
    assert identity["response"]["columns"] == ["y"]
    assert identity["response"]["n_rows"] == 120
    assert identity["sample"]["n_input"] == identity["sample"]["n_rows"] == 120
    assert identity["weights"]["policy"] == "upstream"
    assert identity["response_transform"] == "identity"
    assert identity["likelihood_measure"] == "response"
    assert len(identity["response"]["sha256"]) == 64


def test_identity_survives_truncation_and_pickling():
    fit = _glm("y ~ bols(x)", _sample_data())
    assert fit[:30]._comparison_identity == fit._comparison_identity
    restored = pickle.loads(pickle.dumps(fit))
    assert restored._comparison_identity == fit._comparison_identity


def test_gamboost_and_lss_fits_record_identity():
    data = _sample_data()
    gam = mb.gamboost("y ~ bbs(x, df=3)", data=data, family=mb.Gaussian(), control=CONTROL)
    lss = mb.glmboostLSS("y ~ bols(x)", data, families=mb.GaussianLSS(), control=CONTROL)
    assert gam._comparison_identity["sample"]["n_input"] == 120
    assert lss._comparison_identity["response"]["columns"] == ["y"]
    assert lss._comparison_identity["response_transform"] == "identity"


def test_missing_identity_falls_back_to_retained_data_check():
    data = _sample_data()
    first = _glm("y ~ bols(x)", data)
    second = _glm("y ~ bols(x) + bols(z)", data)
    first._comparison_identity = None
    rows = _gaic(first, second)
    assert len(rows) == 2
    assert all("comparison" not in row for row in rows)


# ── rank: legitimate same-sample comparisons ────────────────────────────────

@MODEL_HELPERS
def test_same_sample_models_rank_without_a_comparison_column(helper):
    data = _sample_data()
    rows = helper(_glm("y ~ bols(x)", data), _glm("y ~ bols(x) + bols(z)", data))
    assert len(rows) == 2
    assert all("comparison" not in row for row in rows)


def test_gaic_table_one_model_needs_no_check():
    rows = _gaic(_glm("y ~ bols(x)", _sample_data()))
    assert len(rows) == 1 and "comparison" not in rows[0]


def test_scan_lss_families_ranks_one_sample_without_a_comparison_column():
    rows = mb.scan_lss_families(
        "y ~ bols(x)", _sample_data(), families=[mb.GaussianLSS(), mb.GammaLSS()],
        metric="GAIC", control=CONTROL)
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert all("comparison" not in row for row in rows)


def test_compare_specifications_same_sample_ranks():
    rows = mb.compare_specifications(
        {"x only": "bols(x)", "x and z": "bols(x) + bols(z)"}, _sample_data(), "y",
        folds=3, mstop=30)
    assert [row["specification"] for row in rows] == ["x only", "x and z"]


def test_compare_specifications_common_sample_reports_one_exclusion_count_for_all_rows():
    data = _sample_data()
    data["z"] = data["z"].copy()
    data["z"][7] = np.nan
    with pytest.warns(UserWarning, match="excluded from every score"):
        rows = mb.compare_specifications(
            {"x only": "bols(x)", "z": "bols(z)"}, data, "y", folds=3, mstop=30,
            na_action="omit")
    counts = {row["common sample excluded rows"] for row in rows}
    assert counts == {1}


# ── refuse: different data ──────────────────────────────────────────────────

@MODEL_HELPERS
def test_permuted_response_of_the_same_length_is_refused(helper):
    data = _sample_data()
    with pytest.raises(ValueError, match="different response values or row order"):
        helper(_glm("y ~ bols(x)", data), _glm("y ~ bols(x)", _permuted(data)))


@MODEL_HELPERS
def test_different_kept_rows_of_the_same_count_are_refused(helper):
    # Each candidate loses one row, but a different one: the x-model drops
    # row 3 (x missing) and the z-model drops row 5 (z missing).
    data = _sample_data()
    x_missing = dict(data, x=data["x"].copy())
    x_missing["x"][3] = np.nan
    z_missing = dict(data, z=data["z"].copy())
    z_missing["z"][5] = np.nan
    first = _glm("y ~ bols(x)", x_missing, na_action="omit")
    second = _glm("y ~ bols(z)", z_missing, na_action="omit")
    with pytest.raises(ValueError, match="different sample"):
        helper(first, second)


@MODEL_HELPERS
def test_a_formula_specific_omission_is_refused_as_a_different_sample(helper):
    data = _sample_data()
    data["z"] = data["z"].copy()
    data["z"][5] = np.nan
    full = _glm("y ~ bols(x)", data)
    omitted = _glm("y ~ bols(x) + bols(z)", data, na_action="omit")
    with pytest.raises(ValueError, match="kept 119 of 120 rows"):
        helper(full, omitted)


@MODEL_HELPERS
def test_different_effective_weights_are_refused(helper):
    data = _sample_data()
    weights = 1.0 + np.random.default_rng(3).uniform(size=120)
    with pytest.raises(ValueError, match="different effective weights"):
        helper(_glm("y ~ bols(x)", data), _glm("y ~ bols(x)", data, weights=weights))


def test_scan_lss_families_refuses_a_candidate_fitted_on_another_sample(monkeypatch):
    real = mb.glmboostLSS
    calls = []

    def fit(spec, data, families, control):
        calls.append(families.__class__.__name__)
        if len(calls) == 2:
            shifted = dict(data)
            shifted["x"] = data["x"].copy()
            shifted["x"][3] = np.nan
            return real(spec, shifted, families=families, control=control, na_action="omit")
        return real(spec, data, families=families, control=control)

    monkeypatch.setattr(mb, "glmboostLSS", fit)
    with pytest.raises(ValueError, match="different sample"):
        mb.scan_lss_families("y ~ bols(x)", _sample_data(),
                             families=[mb.GaussianLSS(), mb.GaussianLSS()], control=CONTROL)


def _history_row(evaluation, specification):
    return {"specification": specification, "evaluation": evaluation,
            "fit count": evaluation.fit_count}


def test_compare_specifications_refuses_a_history_with_a_permuted_response():
    data = _sample_data()
    first = mb.compare_specifications(["bols(x)"], data, "y", folds=3, mstop=30,
                                      include_evaluations=True)
    with pytest.raises(ValueError, match="not comparable"):
        mb.compare_specifications(["bols(z)"], _permuted(data), "y", folds=3, mstop=30,
                                  history=first)


def test_compare_specifications_refuses_a_history_from_another_sample():
    data = _sample_data()
    first = mb.compare_specifications(["bols(x)"], data, "y", folds=3, mstop=30,
                                      include_evaluations=True)
    smaller = {key: value[:100] for key, value in data.items()}
    with pytest.raises(ValueError, match="not comparable"):
        mb.compare_specifications(["bols(z)"], smaller, "y", folds=3, mstop=30,
                                  history=first)


def test_compare_specifications_refuses_a_history_scored_under_other_weights():
    data = _sample_data()
    weights = 1.0 + np.random.default_rng(4).uniform(size=120)
    weighted = mb.evaluate_specification("bols(x)", data, "y", folds=3, mstop=30,
                                         seed=0, weights=weights)
    with pytest.raises(ValueError, match="not comparable"):
        mb.compare_specifications(["bols(x)"], data, "y", folds=3, mstop=30,
                                  history=[_history_row(weighted, "bols(x)")])


def test_compare_specifications_accepts_a_log_fit_scored_on_the_response_scale():
    # A log-response fit scored by squared error on the back-transformed
    # response is the same target as an identity fit, so the history pairs.
    data = _sample_data()
    logged = mb.evaluate_specification("bols(x)", data, "y", folds=3, mstop=30,
                                       seed=0, score="mse", response_transform="log")
    rows = mb.compare_specifications(["bols(x)"], data, "y", folds=3, mstop=30,
                                     score="mse", history=[_history_row(logged, "bols(x)")])
    assert len(rows) == 1


# ── segregate: same rows, different response scale or weight policy ─────────

@MODEL_HELPERS
def test_transformed_and_untransformed_responses_are_segregated(helper):
    data = _sample_data()
    rows = helper(_glm("y ~ bols(x)", data), _glm("log(y) ~ bols(x)", data))
    assert len(rows) == 2
    assert all("response transform" in row["comparison"] for row in rows)
    assert all(row["comparison"].startswith("not ranked:") for row in rows)


def test_gaic_table_keeps_segregated_rows_in_the_order_given():
    data = _sample_data()
    rows = mb.gaic_table(_glm("log(y) ~ bols(x)", data), _glm("y ~ bols(x)", data),
                         labels=["logged", "raw"], output="records")
    assert [row["label"] for row in rows] == ["logged", "raw"]
    assert all("comparison" in row for row in rows)


def test_compare_models_refuses_to_sort_segregated_rows():
    data = _sample_data()
    with pytest.raises(ValueError, match="cannot sort by 'GAIC'"):
        mb.compare_models(_glm("y ~ bols(x)", data), _glm("log(y) ~ bols(x)", data),
                          sort="GAIC", output="records")


@MODEL_HELPERS
def test_different_weight_policies_are_segregated(helper):
    data = _sample_data()
    weights = np.random.default_rng(6).integers(1, 4, 120).astype(float)
    rows = helper(_glm("y ~ bols(x)", data, weights=weights),
                  _glm("y ~ bols(x)", data, weights=weights, weight_policy="raw"))
    assert all("weight policy" in row["comparison"] for row in rows)


def test_scan_lss_families_segregates_a_candidate_on_another_response_scale(monkeypatch):
    real = mb.glmboostLSS
    calls = []

    def fit(spec, data, families, control):
        calls.append(1)
        if len(calls) == 2:
            logged = spec.replace("y ~", "log(y) ~", 1) if isinstance(spec, str) else {
                key: value.replace("y ~", "log(y) ~", 1) for key, value in spec.items()}
            return real(logged, data, families=families, control=control)
        return real(spec, data, families=families, control=control)

    monkeypatch.setattr(mb, "glmboostLSS", fit)
    rows = mb.scan_lss_families("y ~ bols(x)", _sample_data(),
                                families=[mb.GaussianLSS(), mb.GaussianLSS()], control=CONTROL)
    assert len(rows) == 2
    assert all(row["comparison"].startswith("not ranked:") for row in rows)
    assert all("response transform" in row["comparison"] for row in rows)


# ── summaries, cross-validation comparisons and common_sample= ──────────────
#
# compare_summaries, CvRiskResult.compare_to and compare_cv read the same
# identity records as the model helpers. gaic_table and compare_models accept
# common_sample=True: every candidate is refitted on the rows all of them can
# use, and each row reports what that removed.

COMMON_FIELDS = ("common sample rows", "common sample excluded rows", "common sample excluded by")


def _gapped(seed=1, n=120):
    """Two candidates with different own samples: x misses row 3, z misses row 5."""
    data = _sample_data(seed=seed, n=n)
    data["x"] = data["x"].copy()
    data["x"][3] = np.nan
    data["z"] = data["z"].copy()
    data["z"][5] = np.nan
    return data


def _without_rows(data, rows):
    """The data with the given rows removed, as a fit on those rows alone would read it."""
    keep = np.setdiff1d(np.arange(len(data["y"])), rows)
    return {key: np.asarray(value)[keep] for key, value in data.items()}


def _folds(n, seed=0):
    return mb.cv_weights(n, type="kfold", B=5, seed=seed)


# ── compare_summaries ───────────────────────────────────────────────────────

def test_compare_summaries_on_one_sample_has_no_ranking_note():
    data = _sample_data()
    summary = mb.compare_summaries(_glm("y ~ bols(x)", data), _glm("y ~ bols(x) + bols(z)", data))
    assert not any("Not ranked" in note for note in summary.notes)


def test_compare_summaries_refuses_a_permuted_response():
    data = _sample_data()
    with pytest.raises(ValueError, match="different response values or row order"):
        mb.compare_summaries(_glm("y ~ bols(x)", data), _glm("y ~ bols(x)", _permuted(data)))


def test_compare_summaries_notes_a_response_transform_as_not_ranked():
    data = _sample_data()
    summary = mb.compare_summaries(_glm("y ~ bols(x)", data), _glm("log(y) ~ bols(x)", data))
    notes = [note for note in summary.notes if note.startswith("Not ranked against each other")]
    assert len(notes) == 1 and "response transform" in notes[0]


def test_compare_summaries_refuses_cross_validation_on_different_folds():
    data = _sample_data()
    model = _glm("y ~ bols(x)", data)
    a = model.cvrisk(folds=_folds(120, seed=0))
    b = model.cvrisk(folds=_folds(120, seed=1))
    with pytest.raises(ValueError, match="same folds"):
        mb.compare_summaries(model, model, cv=[a, b])


# ── cross-validation comparisons ────────────────────────────────────────────

def test_cv_curves_on_the_same_data_pair_at_an_explicit_mstop():
    data = _sample_data()
    a = _glm("y ~ bols(x)", data).cvrisk(folds=_folds(120))
    b = _glm("y ~ bols(x) + bols(z)", data).cvrisk(folds=_folds(120))
    result = a.compare_to(b, mstop=20)
    assert result.comparison == ""
    assert result.inferential and result.interval is not None


def test_cv_curves_of_a_log_and_a_raw_response_are_not_ranked():
    data = _sample_data()
    raw = _glm("y ~ bols(x)", data).cvrisk(folds=_folds(120))
    logged = _glm("log(y) ~ bols(x)", data).cvrisk(folds=_folds(120))
    result = raw.compare_to(logged, mstop=20)
    assert "response transform" in result.comparison
    assert np.isnan(result.mean_difference)
    assert result.interval is None and not result.inferential
    assert "not ranked" in result.describe()


def test_cv_curves_on_a_permuted_response_are_refused():
    data = _sample_data()
    raw = _glm("y ~ bols(x)", data).cvrisk(folds=_folds(120))
    permuted = _glm("y ~ bols(x)", _permuted(data)).cvrisk(folds=_folds(120))
    with pytest.raises(ValueError, match="cannot pair these curves"):
        raw.compare_to(permuted, mstop=20)


def test_cv_curves_without_an_identity_fall_back_to_the_fold_check():
    data = _sample_data()
    a = _glm("y ~ bols(x)", data).cvrisk(folds=_folds(120))
    b = _glm("y ~ bols(x) + bols(z)", data).cvrisk(folds=_folds(120))
    a.comparison_identity = None
    result = a.compare_to(b, mstop=20)
    assert result.comparison == ""
    assert result.inferential


# ── common_sample= ──────────────────────────────────────────────────────────

def test_gaic_table_common_sample_scores_the_rows_every_candidate_keeps():
    data = _gapped()
    x_model = _glm("y ~ bols(x)", data, na_action="omit")
    z_model = _glm("y ~ bols(z)", data, na_action="omit")
    with pytest.warns(UserWarning, match="common_sample=True"):
        rows = mb.gaic_table(x_model, z_model, labels=["x", "z"], output="records",
                             common_sample=True)
    reference = _without_rows(data, [3, 5])
    expected = mb.gaic_table(_glm("y ~ bols(x)", reference), _glm("y ~ bols(z)", reference),
                             labels=["x", "z"], output="records")
    for got, want in zip(rows, expected, strict=True):
        assert got["mboost_corrected"] == pytest.approx(want["mboost_corrected"], rel=1e-12)
        assert got["mboost_corrected_mstop"] == want["mboost_corrected_mstop"]
        assert got["n"] == want["n"] == 118


def test_common_sample_reports_each_excluded_row_and_the_candidate_that_dropped_it():
    data = _gapped()
    rows = mb.gaic_table(_glm("y ~ bols(x)", data, na_action="omit"),
                         _glm("y ~ bols(z)", data, na_action="omit"),
                         labels=["x", "z"], output="records", common_sample=True)
    assert [row["common sample rows"] for row in rows] == [118, 118]
    assert [row["common sample excluded rows"] for row in rows] == [1, 1]
    assert rows[0]["common sample excluded by"] == "z (1 row)"
    assert rows[1]["common sample excluded by"] == "x (1 row)"


def test_compare_models_common_sample_scores_risk_and_mstop_on_the_common_rows():
    data = _gapped()
    x_model = _glm("y ~ bols(x)", data, na_action="omit")
    z_model = _glm("y ~ bols(z)", data, na_action="omit")
    rows = mb.compare_models(x_model, z_model, labels=["x", "z"], output="records",
                             common_sample=True)
    reference = _without_rows(data, [3, 5])
    expected = mb.compare_models(_glm("y ~ bols(x)", reference), _glm("y ~ bols(z)", reference),
                                 labels=["x", "z"], output="records")
    for got, want in zip(rows, expected, strict=True):
        assert got["risk"] == pytest.approx(want["risk"], rel=1e-12)
        assert got["mstop"] == want["mstop"]
        assert got["common sample excluded rows"] == 1


def test_common_sample_refits_gam_preprocessing_on_the_common_rows():
    data = _gapped()
    x_model = mb.gamboost("y ~ bbs(x, df=4)", data=data, family=mb.Gaussian(),
                          control=CONTROL, na_action="omit")
    z_model = mb.gamboost("y ~ bbs(z, df=4)", data=data, family=mb.Gaussian(),
                          control=CONTROL, na_action="omit")
    with pytest.warns(UserWarning, match="common_sample=True"):
        rows = mb.gaic_table(x_model, z_model, labels=["x", "z"], output="records",
                             common_sample=True)
    reference = _without_rows(data, [3, 5])
    expected = mb.gaic_table(
        mb.gamboost("y ~ bbs(x, df=4)", data=reference, family=mb.Gaussian(), control=CONTROL),
        mb.gamboost("y ~ bbs(z, df=4)", data=reference, family=mb.Gaussian(), control=CONTROL),
        labels=["x", "z"], output="records")
    for got, want in zip(rows, expected, strict=True):
        assert got["mboost_corrected"] == pytest.approx(want["mboost_corrected"], rel=1e-10)


def test_common_sample_refits_a_distributional_candidate_on_the_common_rows():
    data = _gapped()
    x_model = mb.glmboostLSS("y ~ bols(x)", data, families=mb.GaussianLSS(), control=CONTROL,
                             na_action="omit")
    z_model = mb.glmboostLSS("y ~ bols(z)", data, families=mb.GaussianLSS(), control=CONTROL,
                             na_action="omit")
    rows = mb.gaic_table(x_model, z_model, labels=["x", "z"], output="records",
                         common_sample=True)
    reference = _without_rows(data, [3, 5])
    expected = mb.gaic_table(
        mb.glmboostLSS("y ~ bols(x)", reference, families=mb.GaussianLSS(), control=CONTROL),
        mb.glmboostLSS("y ~ bols(z)", reference, families=mb.GaussianLSS(), control=CONTROL),
        labels=["x", "z"], output="records")
    for got, want in zip(rows, expected, strict=True):
        assert got["heuristic_GAIC(2)"] == pytest.approx(want["heuristic_GAIC(2)"], rel=1e-10)
        assert got["common sample excluded rows"] == 1


def test_common_sample_with_nothing_excluded_changes_no_score_and_no_model():
    data = _sample_data()
    a = _glm("y ~ bols(x)", data)
    b = _glm("y ~ bols(x) + bols(z)", data)
    plain = mb.gaic_table(a, b, output="records")
    common = mb.gaic_table(a, b, output="records", common_sample=True)
    assert not any(key in row for row in plain for key in COMMON_FIELDS)
    for p, c in zip(plain, common, strict=True):
        assert {k: v for k, v in c.items() if k not in COMMON_FIELDS} == p
        assert c["common sample rows"] == 120
        assert c["common sample excluded rows"] == 0
        assert c["common sample excluded by"] == ""


def test_default_refuses_different_kept_rows_and_common_sample_does_not_refuse_them():
    data = _gapped()
    x_model = _glm("y ~ bols(x)", data, na_action="omit")
    z_model = _glm("y ~ bols(z)", data, na_action="omit")
    with pytest.raises(ValueError, match="different sample"):
        mb.gaic_table(x_model, z_model, output="records")
    rows = mb.gaic_table(x_model, z_model, output="records", common_sample=True)
    assert all("comparison" not in row for row in rows)


@pytest.mark.parametrize("call", [mb.gaic_table, mb.compare_models],
                         ids=["gaic_table", "compare_models"])
def test_common_sample_still_refuses_a_permuted_response(call):
    data = _gapped()
    with pytest.raises(ValueError, match="different response values or row order"):
        call(_glm("y ~ bols(x)", data, na_action="omit"),
             _glm("y ~ bols(z)", _permuted(data), na_action="omit"),
             common_sample=True, output="records")


def test_common_sample_keeps_a_response_transform_segregated():
    data = _gapped()
    rows = mb.gaic_table(_glm("log(y) ~ bols(x)", data, na_action="omit"),
                         _glm("y ~ bols(z)", data, na_action="omit"),
                         output="records", common_sample=True)
    assert all("response transform" in row["comparison"] for row in rows)
    assert all(row["common sample excluded rows"] == 1 for row in rows)


def test_common_sample_needs_identity_records():
    data = _gapped()
    x_model = _glm("y ~ bols(x)", data, na_action="omit")
    z_model = _glm("y ~ bols(z)", data, na_action="omit")
    x_model._comparison_identity = None
    with pytest.raises(ValueError, match="comparison identity"):
        mb.gaic_table(x_model, z_model, output="records", common_sample=True)


def test_compare_models_common_sample_refuses_cv_results_when_rows_are_excluded():
    data = _gapped()
    x_model = _glm("y ~ bols(x)", data, na_action="omit")
    z_model = _glm("y ~ bols(z)", data, na_action="omit")
    cv = x_model.cvrisk(folds=_folds(119))
    with pytest.raises(ValueError, match="common_sample=True refits"):
        mb.compare_models(x_model, z_model, cv=[cv, None], output="records",
                          common_sample=True)


def test_common_sample_reads_a_truncated_candidate_with_dropped_rows():
    # fit[:k] keeps the rows the fit dropped, so its exclusions are still reported.
    data = _gapped()
    x_model = _glm("y ~ bols(x)", data, na_action="omit")
    z_model = _glm("y ~ bols(z)", data, na_action="omit")
    rows = mb.gaic_table(x_model[:30], z_model, output="records", common_sample=True)
    assert [row["common sample excluded rows"] for row in rows] == [1, 1]
    assert all(row["common sample rows"] == 118 for row in rows)


# ── reference measure and scoring rule ──────────────────────────────────────
#
# A count family's likelihood is a probability mass and a continuous family's
# is a density; the two log-likelihoods of the same counts differ by a term
# that depends on the units, so neither an information criterion nor a mean
# negative log-likelihood ranks them. Risks under different scoring rules
# (squared error against absolute error) have no paired difference.


def _count_data(seed=3, n=150):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = rng.poisson(np.exp(1.0 + 0.4 * x)).astype(float)
    return {"y": y, "x": x, "z": z}


def _lss(family, data, formula="y ~ bols(x)", **kwargs):
    return mb.glmboostLSS(formula, data=data, families=family, control=CONTROL, **kwargs)


def test_reference_measure_classifies_count_continuous_and_mixed_families():
    from mboostrs._likelihood import reference_measure

    assert reference_measure(mb.Poisson()) == "counting"
    assert reference_measure(mb.PoissonLSS()) == "counting"
    assert reference_measure(mb.NBinomialLSS()) == "counting"
    assert reference_measure(mb.GaussianLSS()) == "lebesgue"
    assert reference_measure(mb.Gaussian()) == "lebesgue"
    assert reference_measure(mb.ZeroAdjustedGammaLSS()) == "zero_atom_lebesgue"
    assert reference_measure(mb.CoxPH()) == "partial"
    assert reference_measure(mb.Laplace()) is None


def test_identity_records_reference_measure_and_risk_scale():
    record = _lss(mb.PoissonLSS(), _count_data())._comparison_identity
    assert record["reference_measure"] == "counting"
    assert record["risk_scale"] == "lss_nll:response:counting"
    glm = _glm("y ~ bols(x)", _sample_data())._comparison_identity
    assert glm["reference_measure"] == "lebesgue"
    assert glm["risk_scale"] == glm["scoring_rule"]


def test_gaic_table_refuses_a_mass_and_a_density_of_the_same_counts():
    data = _count_data()
    with pytest.raises(NotImplementedError, match="different reference measures"):
        mb.gaic_table(_lss(mb.PoissonLSS(), data), _lss(mb.GaussianLSS(), data),
                      output="records")


def test_gaic_table_ranks_two_count_families():
    data = _count_data()
    rows = mb.gaic_table(_lss(mb.PoissonLSS(), data), _lss(mb.NBinomialLSS(), data),
                         output="records")
    assert all("comparison" not in row for row in rows)


def test_compare_models_does_not_sort_a_mass_against_a_density():
    data = _count_data()
    models = (_lss(mb.PoissonLSS(), data), _lss(mb.GaussianLSS(), data))
    rows = mb.compare_models(*models, metrics=("logLik",), output="records")
    assert all("'counting'" in row["comparison"] for row in rows)
    with pytest.raises(ValueError, match="cannot sort by 'logLik'"):
        mb.compare_models(*models, metrics=("logLik",), sort="logLik", output="records")


def test_scan_lss_families_segregates_count_and_continuous_families():
    rows = mb.scan_lss_families("y ~ bols(x)", _count_data(),
                                families=[mb.GaussianLSS(), mb.PoissonLSS()],
                                metric="logLik", control=CONTROL)
    assert [row["family"] for row in rows] == ["GaussianLSS", "PoissonLSS"]
    assert all("'lebesgue'" in row["comparison"] and "'counting'" in row["comparison"]
               for row in rows)


def test_cv_curves_under_different_scoring_rules_are_refused():
    data = _sample_data()
    squared = _glm("y ~ bols(x)", data).cvrisk(folds=_folds(120))
    absolute = mb.glmboost("y ~ bols(x)", data=data, family=mb.Laplace(),
                           control=CONTROL).cvrisk(folds=_folds(120))
    with pytest.raises(ValueError, match="different scoring rules"):
        squared.compare_to(absolute, mstop=20)


def test_cv_curves_of_two_quantile_levels_are_refused():
    data = _sample_data()
    low, high = (mb.glmboost("y ~ bols(x)", data=data, family=mb.Quantile(tau=tau),
                             control=CONTROL).cvrisk(folds=_folds(120))
                 for tau in (0.2, 0.8))
    with pytest.raises(ValueError, match="different scoring rules"):
        low.compare_to(high, mstop=20)


def test_cv_curves_of_records_without_a_risk_scale_are_checked_on_data_alone():
    data = _sample_data()
    squared = _glm("y ~ bols(x)", data).cvrisk(folds=_folds(120))
    absolute = mb.glmboost("y ~ bols(x)", data=data, family=mb.Laplace(),
                           control=CONTROL).cvrisk(folds=_folds(120))
    for curve in (squared, absolute):
        curve.comparison_identity = {key: value for key, value in
                                     curve.comparison_identity.items()
                                     if key != "risk_scale"}
    assert squared.compare_to(absolute, mstop=20).comparison == ""


# ── scan_lss_families on one common sample ──────────────────────────────────


def _gapped_counts():
    data = _count_data()
    data["z"] = data["z"].copy()
    data["z"][:12] = np.nan
    return data


SCAN_SPEC = {"mu": "y ~ bols(x)", "sigma": "y ~ bols(z)"}


def test_scan_omit_scores_every_family_on_the_rows_all_formulas_keep():
    data = _gapped_counts()
    with pytest.warns(UserWarning, match="12 rows with a missing value"):
        rows = mb.scan_lss_families(SCAN_SPEC, data,
                                    families=[mb.PoissonLSS(), mb.NBinomialLSS()],
                                    metric="logLik", control=CONTROL, na_action="omit")
    by_family = {row["family"]: row for row in rows}
    assert {row["status"] for row in rows} == {"ok"}
    assert all(row["common sample rows"] == 138 for row in rows)
    assert all(row["n"] == 138 for row in rows)
    # PoissonLSS reads only the mu formula, so the common sample removes 12
    # rows it would have kept; NBinomialLSS reads z and would drop them itself.
    assert by_family["PoissonLSS"]["common sample excluded rows"] == 12
    assert by_family["NBinomialLSS"]["common sample excluded rows"] == 0
    complete = {key: value[12:] for key, value in data.items()}
    alone = _lss(mb.PoissonLSS(), complete, formula=SCAN_SPEC["mu"])
    assert by_family["PoissonLSS"]["logLik"] == pytest.approx(float(alone.logLik()))


def test_scan_omit_restricts_a_fold_matrix_to_the_kept_rows():
    data = _gapped_counts()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rows = mb.scan_lss_families(SCAN_SPEC, data,
                                    families=[mb.PoissonLSS(), mb.NBinomialLSS()],
                                    metric="cvrisk", control=CONTROL, folds=_folds(150),
                                    na_action="omit")
    assert {row["status"] for row in rows} == {"ok"}
    assert all(row["cvrisk"] is not None for row in rows)


def test_scan_omit_on_complete_data_changes_no_score():
    data = _count_data()
    families = [mb.PoissonLSS(), mb.NBinomialLSS()]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        omitted = mb.scan_lss_families(SCAN_SPEC, data, families=families,
                                       metric="logLik", control=CONTROL, na_action="omit")
    default = mb.scan_lss_families(SCAN_SPEC, data, families=families,
                                   metric="logLik", control=CONTROL)
    assert [row["logLik"] for row in omitted] == [row["logLik"] for row in default]
    assert all(row["common sample excluded rows"] == 0 for row in omitted)
    assert all("common sample rows" not in row for row in default)


def test_scan_rejects_an_unknown_na_action():
    with pytest.raises(ValueError, match="na_action must be 'error' or 'omit'"):
        mb.scan_lss_families("y ~ bols(x)", _count_data(), families=[mb.PoissonLSS()],
                             na_action="impute")
