"""Fixed-horizon survival predictions and IPCW evaluation for CoxPH models.

Each computation has a reference that does not use the Rust core:

* the Breslow baseline is checked against the Python implementation it
  replaced, kept here as ``_breslow_oracle``;
* S(t | x), the IPCW Brier score and the IPCW AUC are checked against
  small hand-computed examples and against brute-force numpy sums;
* the scikit-survival and lifelines comparisons run only when those packages
  are installed in the environment.
"""

import warnings

import numpy as np
import pytest
from mboostrs._rust import ipcw_auc_at, ipcw_brier_at
from mboostrs.families import CoxPH, Gaussian
from mboostrs.glmboost import _breslow_survfit

import mboostrs as mb


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _breslow_oracle(time, event, weights, linpred_train, linpred_new=None):
    """The Python Breslow estimator that ``_breslow_survfit`` used before the port."""
    if weights is not None and not np.allclose(weights, 1.0):
        raise NotImplementedError("survfit() cannot (yet) deal with weights")
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    linpred_train = np.asarray(linpred_train, dtype=np.float64)

    order = np.argsort(time, kind="stable")
    time_sorted = time[order]
    event_sorted = event[order]
    centered = linpred_train[order] - np.mean(linpred_train)

    unique_times, inverse = np.unique(time_sorted, return_inverse=True)
    n_event = np.zeros(len(unique_times), dtype=np.float64)
    np.add.at(n_event, inverse, event_sorted)

    suffix_exp_sum = np.cumsum(np.exp(centered)[::-1])[::-1]
    first_index = np.searchsorted(time_sorted, unique_times, side="left")
    risk_set = suffix_exp_sum[first_index]
    risk_set = np.where(risk_set < 1e-6, np.inf, risk_set)

    baseline_hazard = np.cumsum(n_event / risk_set)
    has_event = n_event > 0

    if linpred_new is not None:
        linpred_new = np.asarray(linpred_new, dtype=np.float64)
        relative_risk = np.exp(linpred_new - np.mean(linpred_train))
        surv = np.exp(-np.outer(baseline_hazard, relative_risk))
    else:
        surv = np.exp(-baseline_hazard)[:, None]

    return surv[has_event, :], unique_times[has_event], n_event[has_event]


def _km_value(time, indicator, weights, t, *, left=False):
    """Weighted Kaplan-Meier value at ``t``, or ``S(t-)`` when ``left``."""
    value = 1.0
    for u in np.unique(time):
        if (u >= t) if left else (u > t):
            break
        at_risk = weights[time >= u].sum()
        deaths = weights[(time == u) & (indicator == 1)].sum()
        if deaths > 0:
            value *= 1.0 - deaths / at_risk
    return value


def _brier_oracle(time, event, weights, survival, t):
    """Graf et al. (1999) IPCW Brier score, computed by brute force."""
    censored = 1.0 - event
    g_t = _km_value(time, censored, weights, t)
    total = 0.0
    for i in range(len(time)):
        if time[i] <= t and event[i] == 1:
            g_ti = _km_value(time, censored, weights, time[i], left=True)
            total += weights[i] * survival[i] ** 2 / g_ti
        elif time[i] > t:
            total += weights[i] * (1 - survival[i]) ** 2 / g_t
    return total / weights.sum()


def _auc_oracle(time, event, weights, score, t):
    """Heagerty-Zheng cumulative/dynamic IPCW AUC, by brute-force pairs."""
    censored = 1.0 - event
    numerator = 0.0
    denominator = 0.0
    for i in range(len(time)):
        if not (time[i] <= t and event[i] == 1):
            continue
        case_weight = weights[i] / _km_value(time, censored, weights, time[i], left=True)
        for j in range(len(time)):
            if time[j] <= t:
                continue
            pair = case_weight * weights[j]
            denominator += pair
            if score[i] > score[j]:
                numerator += pair
            elif score[i] == score[j]:
                numerator += 0.5 * pair
    return numerator / denominator


def _step_baseline(model_baseline, t):
    """H0(t) read from a stored baseline as a right-continuous step."""
    event_times = np.asarray(model_baseline["event_times"])
    cumhaz = np.asarray(model_baseline["cumhaz"])
    passed = np.searchsorted(event_times, t, side="right")
    return 0.0 if passed == 0 else cumhaz[passed - 1]


def _survival_frame_data(rng, n, *, censor_rate=0.4):
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    time = rng.exponential(np.exp(-0.5 * x1 + 0.3 * x2))
    censor = rng.exponential(1.0 / censor_rate, size=n)
    return {
        "time": np.minimum(time, censor),
        "status": (time <= censor).astype(float),
        "x1": x1,
        "x2": x2,
    }


# ---------------------------------------------------------------------------
# Breslow baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_breslow_port_matches_python_oracle_to_1e12(seed):
    rng = np.random.default_rng(seed)
    n = 120
    # Rounding the times creates ties, which exercise the tied-time blocks.
    time = np.round(rng.exponential(size=n), 1)
    event = rng.binomial(1, 0.6, size=n).astype(float)
    linpred = rng.normal(scale=0.7, size=n)
    linpred_new = rng.normal(size=5)

    for new in (None, linpred_new):
        expected = _breslow_oracle(time, event, None, linpred, new)
        actual = _breslow_survfit(time, event, None, linpred, new)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_breslow_hand_computed_with_nonzero_predictor():
    # Three subjects, events at t=1 and t=2, linpred (0, ln 2, 0).
    # center = ln(2)/3, so the relative risks are 2^(-1/3), 2^(2/3), 2^(-1/3).
    # Risk set at t=1 is 2^(-1/3) + 2^(2/3) + 2^(-1/3) = 2^(5/3).
    # Risk set at t=2 is 2^(2/3) + 2^(-1/3) = 3 * 2^(-1/3).
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 0.0])
    linpred = np.array([0.0, np.log(2.0), 0.0])
    hazard_1 = 1.0 / 2 ** (5 / 3)
    hazard_2 = 1.0 / (3 * 2 ** (-1 / 3))
    from mboostrs._rust import breslow_baseline

    event_times, n_event, cumhaz, center = breslow_baseline(time, event, linpred)
    np.testing.assert_allclose(event_times, [1.0, 2.0], rtol=0, atol=0)
    np.testing.assert_allclose(n_event, [1.0, 1.0], rtol=0, atol=0)
    np.testing.assert_allclose(cumhaz, [hazard_1, hazard_1 + hazard_2], rtol=1e-14)
    np.testing.assert_allclose(center, np.log(2.0) / 3, rtol=1e-14)

    # A subject with linpred 0 has relative risk 2^(-1/3) against the centered
    # baseline, so S(2.5) = exp(-H0(2) * 2^(-1/3)).
    surv, _, _ = _breslow_survfit(time, event, None, linpred, np.array([0.0]))
    expected = np.exp(-(hazard_1 + hazard_2) * 2 ** (-1 / 3))
    np.testing.assert_allclose(surv[-1, 0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Fixed-horizon survival S(t | x)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cox_fit():
    rng = np.random.default_rng(7)
    data = _survival_frame_data(rng, 160)
    model = mb.glmboost(
        "Surv(time, status) ~ x1 + x2", data=data, family=CoxPH(),
        control=mb.boost_control(40, 0.1),
    )
    return model, data


def test_survival_at_matches_step_formula(cox_fit):
    model, data = cox_fit
    newdata = {"x1": np.array([0.0, 1.0, -1.0]), "x2": np.array([0.0, -0.5, 2.0])}
    horizons = np.array([0.0, 0.2, 0.7, 1.5, 3.0])
    surv = model.survival_at(horizons, newdata=newdata)

    baseline = model._breslow_baseline
    eta = np.asarray(model.predict(newdata=newdata, type="link"), dtype=float)
    relative = np.exp(eta - baseline["center"])
    expected = np.exp(
        -np.outer(relative, [_step_baseline(baseline, t) for t in horizons]))
    assert surv.shape == (3, 5)
    np.testing.assert_allclose(surv, expected, rtol=1e-12, atol=1e-14)


def test_survival_at_is_non_increasing_and_one_before_first_event(cox_fit):
    model, _ = cox_fit
    horizons = np.linspace(0.0, 2.0, 41)
    surv = model.survival_at(horizons)
    assert np.all(np.diff(surv, axis=1) <= 1e-15)
    assert np.all((surv >= 0) & (surv <= 1))
    first_event = model._breslow_baseline["event_times"][0]
    np.testing.assert_array_equal(
        model.survival_at([0.5 * first_event])[:, 0], np.ones(surv.shape[0]))


def test_survival_at_is_right_continuous_at_event_times(cox_fit):
    model, _ = cox_fit
    event_time = model._breslow_baseline["event_times"][3]
    at, just_after = model.survival_at([event_time, event_time + 1e-9])[0]
    assert at == pytest.approx(just_after, rel=1e-8)
    before = model.survival_at([event_time - 1e-9])[0, 0]
    assert before > at


def test_survival_at_works_after_save_and_load_without_training_data(cox_fit, tmp_path):
    model, _ = cox_fit
    newdata = {"x1": np.array([0.3, -0.8]), "x2": np.array([1.0, 0.1])}
    path = tmp_path / "cox.json"
    mb.save_model(model, path)
    restored = mb.load_model(path)
    assert restored._data is None
    np.testing.assert_allclose(
        restored.survival_at([0.4, 1.2], newdata=newdata),
        model.survival_at([0.4, 1.2], newdata=newdata), rtol=0, atol=0)
    np.testing.assert_allclose(
        restored.survfit()["surv"], model.survfit()["surv"], rtol=0, atol=0)


def test_gamboost_survival_after_save_load_and_continue_to(tmp_path):
    rng = np.random.default_rng(9)
    data = _survival_frame_data(rng, 150)
    formula = "Surv(time, status) ~ bbs(x1, df=4) + bols(x2)"
    model = mb.gamboost(formula, data=data, family=CoxPH(),
                        control=mb.boost_control(30, 0.1))
    newdata = {"x1": np.array([0.0, 1.5]), "x2": np.array([0.0, -1.0])}
    path = tmp_path / "gam.json"
    mb.save_model(model, path)
    restored = mb.load_model(path)
    np.testing.assert_allclose(
        restored.survival_at([0.5, 1.5], newdata=newdata),
        model.survival_at([0.5, 1.5], newdata=newdata), rtol=0, atol=0)
    grown = model.continue_to(50)
    refit = mb.gamboost(formula, data=data, family=CoxPH(),
                        control=mb.boost_control(50, 0.1))
    np.testing.assert_allclose(
        grown.survival_at([0.5, 1.5], newdata=newdata),
        refit.survival_at([0.5, 1.5], newdata=newdata), rtol=1e-10, atol=1e-12)


def test_survival_at_slice_recomputes_baseline_at_its_mstop(cox_fit):
    model, data = cox_fit
    early = model[:5]
    refit = mb.glmboost(
        "Surv(time, status) ~ x1 + x2", data=data, family=CoxPH(),
        control=mb.boost_control(5, 0.1),
    )
    np.testing.assert_allclose(
        early.survival_at([0.5, 1.0]), refit.survival_at([0.5, 1.0]),
        rtol=1e-10, atol=1e-12)


def test_survival_at_rejects_non_coxph_and_bad_times(cox_fit):
    model, data = cox_fit
    gaussian = mb.glmboost(
        "x1 ~ x2", data={"x1": data["x1"], "x2": data["x2"]},
        family=Gaussian(), control=mb.boost_control(10, 0.1))
    with pytest.raises(NotImplementedError, match="CoxPH"):
        gaussian.survival_at([1.0])
    with pytest.raises(ValueError, match="non-negative"):
        model.survival_at([-1.0])
    with pytest.raises(ValueError, match="finite"):
        model.survival_at([np.nan])


# ---------------------------------------------------------------------------
# IPCW scores: hand-computed examples
# ---------------------------------------------------------------------------

# Times 1, 2, 3, 4. Row 3 is censored at 3, so G(t) = 1 before 3 and 1/2 from 3.


def test_ipcw_brier_hand_computed_before_censoring():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 0.0, 1.0])
    weights = np.ones(4)
    survival = np.array([0.5, 0.8, 0.9, 0.6])
    # At t = 2.5: row 1 has an event (0.5^2 / 1), row 2 has an event
    # (0.8^2 / 1), row 3 is still at risk ((1 - 0.9)^2 / G(2.5) = 0.01),
    # row 4 is still at risk ((1 - 0.6)^2 / 1 = 0.16).
    score = ipcw_brier_at(time, event, weights, survival, 2.5, 0.05)
    assert score[0] == pytest.approx((0.25 + 0.64 + 0.01 + 0.16) / 4, rel=1e-14)
    assert score[1] == pytest.approx(1.0)
    assert score[2] is False


def test_ipcw_brier_hand_computed_after_censoring():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 0.0, 1.0])
    weights = np.ones(4)
    survival = np.array([0.4, 0.7, 0.8, 0.5])
    # At t = 3.5, G(3.5) = 1/2. Row 3 was censored by t and contributes 0.
    # Row 4 is at risk: (1 - 0.5)^2 / (1/2) = 0.5.
    score = ipcw_brier_at(time, event, weights, survival, 3.5, 0.05)
    assert score[0] == pytest.approx((0.16 + 0.49 + 0.0 + 0.5) / 4, rel=1e-14)
    assert score[1] == pytest.approx(0.5)
    assert score[2] is False


def test_ipcw_brier_flags_censoring_below_floor():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 0.0, 1.0])
    survival = np.array([0.4, 0.7, 0.8, 0.5])
    flagged = ipcw_brier_at(time, event, np.ones(4), survival, 3.5, 0.6)
    assert flagged[2] is True
    assert flagged[1] == pytest.approx(0.5)


def test_ipcw_auc_hand_computed_no_ties():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 0.0, 1.0])
    score = np.array([0.2, 0.9, 0.5, 0.1])
    # t = 2.5. Cases: rows 1 and 2. Controls: rows 3 and 4 (weights 1).
    # Pairs won by the case: (0.9 > 0.5), (0.9 > 0.1), (0.2 > 0.1) = 3 of 4.
    result = ipcw_auc_at(time, event, np.ones(4), score, 2.5, 0.05)
    assert result[0] == pytest.approx(0.75, rel=1e-14)


def test_ipcw_auc_hand_computed_with_tie_and_weighted_case():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 0.0, 1.0])
    score = np.array([0.2, 0.9, 0.5, 0.9])
    # Row 4 ties with case 2 (half credit). Case 1 wins no pair.
    # Numerator 0.5 + 1 = 1.5 over denominator 4.
    result = ipcw_auc_at(time, event, np.ones(4), score, 2.5, 0.05)
    assert result[0] == pytest.approx(0.375, rel=1e-14)


def test_ipcw_auc_upweights_case_observed_after_censoring():
    # Row 2 is censored at 2, so G(2) = 2/3 and a case at t=3 gets weight 1.5.
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 0.0, 1.0, 1.0])
    score = np.array([0.8, 0.3, 0.6, 0.7])
    # At t = 3.5: cases rows 1 (weight 1) and 3 (weight 1.5); control row 4.
    # Case 1 beats 0.7; case 3 loses to 0.7. AUC = 1 / (1 + 1.5) = 0.4.
    result = ipcw_auc_at(time, event, np.ones(4), score, 3.5, 0.05)
    assert result[0] == pytest.approx(0.4, rel=1e-14)
    assert result[3] == pytest.approx(2.5)
    assert result[4] == pytest.approx(1.0)


def test_ipcw_refuses_zero_censoring_at_horizon():
    # Row 2 is the only row at risk at t=2 and it is censored there, so
    # G(2) = 1 - 1/1 = 0 and the weights 1/G(t) are infinite.
    time = np.array([1.0, 2.0])
    event = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="censoring survival"):
        ipcw_brier_at(time, event, np.ones(2), np.array([0.5, 0.5]), 2.0, 0.05)


# ---------------------------------------------------------------------------
# IPCW scores against brute-force numpy references on fitted data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 4])
def test_ipcw_scores_match_brute_force_on_model(seed, cox_fit):
    model, _ = cox_fit
    rng = np.random.default_rng(seed)
    data = _survival_frame_data(rng, 90, censor_rate=0.25)
    horizons = [0.4, 0.9]
    result = mb.evaluate_survival(model, data, horizons=horizons, data_role="external",
                                  censoring_floor=0.0)
    survival = model.survival_at(horizons, newdata=data)
    eta = np.asarray(model.predict(newdata=data, type="link"), dtype=float)
    for position, t in enumerate(horizons):
        expected_brier = _brier_oracle(
            data["time"], data["status"], np.ones(90), survival[:, position], t)
        expected_auc = _auc_oracle(data["time"], data["status"], np.ones(90), eta, t)
        row = result.horizon_rows[position]
        assert row["brier"] == pytest.approx(expected_brier, rel=1e-12)
        assert row["auc"] == pytest.approx(expected_auc, rel=1e-12)


# ---------------------------------------------------------------------------
# evaluate_survival: result contract and positivity policy
# ---------------------------------------------------------------------------


def _late_censoring_data():
    # Three events at 1, 2, 3 and seven censorings at 4..10. The censoring
    # survival is G(8) = 2/7 and G(9) = 1/7; G(10) = 0.
    time = np.arange(1.0, 11.0)
    status = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    x1 = np.linspace(-1, 1, 10)
    return {"time": time, "status": status, "x1": x1}


def test_evaluate_refuses_horizon_below_censoring_floor():
    data = _late_censoring_data()
    model = mb.glmboost(
        "Surv(time, status) ~ x1", data=data, family=CoxPH(),
        control=mb.boost_control(10, 0.1))
    with pytest.raises(ValueError, match="censoring_floor=0.2"):
        mb.evaluate_survival(model, data, horizons=[9.0], data_role="external",
                             censoring_floor=0.2)


def test_evaluate_warns_and_flags_below_floor_when_asked():
    data = _late_censoring_data()
    model = mb.glmboost(
        "Surv(time, status) ~ x1", data=data, family=CoxPH(),
        control=mb.boost_control(10, 0.1))
    with pytest.warns(RuntimeWarning, match="censoring_floor"):
        result = mb.evaluate_survival(
            model, data, horizons=[8.0, 9.0], data_role="external",
            censoring_floor=0.2, on_low_censoring="warn")
    flags = [row["below_censoring_floor"] for row in result.horizon_rows]
    assert flags == [False, True]
    assert result.horizon_rows[1]["censoring_survival"] == pytest.approx(1 / 7, rel=1e-12)
    assert result.horizon_rows[0]["censoring_survival"] == pytest.approx(2 / 7, rel=1e-12)


def test_evaluate_refuses_horizon_where_censoring_is_zero_even_when_warning():
    data = _late_censoring_data()
    model = mb.glmboost(
        "Surv(time, status) ~ x1", data=data, family=CoxPH(),
        control=mb.boost_control(10, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="zero"):
            mb.evaluate_survival(model, data, horizons=[10.0], data_role="external",
                                 on_low_censoring="warn")


def test_evaluate_result_fields_and_frames(cox_fit):
    model, data = cox_fit
    result = mb.evaluate_survival(model, data, horizons=[0.5, 1.0], data_role="external")
    assert result.family == "CoxPH"
    assert result.response == ("time", "status")
    assert result.n_observations == len(data["time"])
    assert result.survival.shape == (len(data["time"]), 2)
    frame = result.horizon_frame("records")
    assert [row["horizon"] for row in frame] == [0.5, 1.0]
    assert set(frame[0]) >= {"brier", "auc", "censoring_survival", "below_censoring_floor",
                             "events_by_horizon", "at_risk_at_horizon"}
    long = result.survival_frame("records")
    assert len(long) == 2 * len(data["time"])
    record = result.to_dict()
    assert record["n"] == len(data["time"])
    assert "external" in result.describe()
    assert any("not used to fit" in text for text in result.assumptions)


def test_evaluate_rejects_bad_arguments(cox_fit):
    model, data = cox_fit
    with pytest.raises(ValueError, match="data_role"):
        mb.evaluate_survival(model, data, horizons=[1.0], data_role="holdout")
    with pytest.raises(ValueError, match="censoring_floor"):
        mb.evaluate_survival(model, data, horizons=[1.0], data_role="external",
                             censoring_floor=1.0)
    with pytest.raises(ValueError, match="on_low_censoring"):
        mb.evaluate_survival(model, data, horizons=[1.0], data_role="external",
                             on_low_censoring="ignore")
    bad = dict(data, status=data["status"] * 2)
    with pytest.raises(ValueError, match="coded 0"):
        mb.evaluate_survival(model, bad, horizons=[1.0], data_role="external")
    with pytest.raises(ValueError, match="weights"):
        mb.evaluate_survival(model, data, horizons=[1.0], data_role="external",
                             weights=np.ones(3))


def test_evaluate_training_role_is_recorded_as_fit_diagnostic(cox_fit):
    model, data = cox_fit
    result = mb.evaluate_survival(model, data, horizons=[0.5], data_role="training")
    assert result.data_role == "training"
    assert any("fit" in text and "new subjects" in text for text in result.assumptions)


# ---------------------------------------------------------------------------
# Optional comparisons with scikit-survival and lifelines
# ---------------------------------------------------------------------------


def _structured(time, status):
    dtype = [("event", bool), ("time", float)]
    return np.array(list(zip(status.astype(bool), time, strict=True)), dtype=dtype)


def test_matches_scikit_survival_brier_and_dynamic_auc():
    sksurv_metrics = pytest.importorskip("sksurv.metrics")
    rng = np.random.default_rng(21)
    data = _survival_frame_data(rng, 120, censor_rate=0.3)
    model = mb.glmboost(
        "Surv(time, status) ~ x1 + x2", data=data, family=CoxPH(),
        control=mb.boost_control(40, 0.1))
    horizons = [0.5, 1.0]
    result = mb.evaluate_survival(model, data, horizons=horizons, data_role="external",
                                  censoring_floor=0.0)
    survival = model.survival_at(horizons, newdata=data)
    eta = np.asarray(model.predict(newdata=data, type="link"), dtype=float)
    y = _structured(data["time"], data["status"])
    times = np.array(horizons)
    _, brier = sksurv_metrics.brier_score(y, y, survival, times)
    aucs, _ = sksurv_metrics.cumulative_dynamic_auc(
        y, y, np.column_stack([eta, eta]), times)
    for position in range(len(horizons)):
        assert result.horizon_rows[position]["brier"] == pytest.approx(
            brier[position], rel=1e-8)
        assert result.horizon_rows[position]["auc"] == pytest.approx(
            aucs[position], rel=1e-8)


def test_censoring_survival_matches_lifelines_kaplan_meier():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(5)
    data = _survival_frame_data(rng, 100, censor_rate=0.5)
    model = mb.glmboost(
        "Surv(time, status) ~ x1", data=data, family=CoxPH(),
        control=mb.boost_control(10, 0.1))
    horizons = [0.3, 0.8]
    result = mb.evaluate_survival(model, data, horizons=horizons, data_role="external",
                                  censoring_floor=0.0)
    kmf = lifelines.KaplanMeierFitter().fit(
        data["time"], event_observed=1 - data["status"])
    for position, t in enumerate(horizons):
        expected = float(kmf.survival_function_at_times(t).iloc[0])
        assert result.horizon_rows[position]["censoring_survival"] == pytest.approx(
            expected, rel=1e-10)
