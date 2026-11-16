"""Identity, scoring, and row-provenance contracts for held-out evaluation."""

import numpy as np
import pytest

import mboostrs as mb


def data(n=90):
    rng = np.random.default_rng(12)
    x = rng.normal(size=n)
    y = 1.4 * x + rng.normal(scale=.4, size=n)
    return {"x": x, "y": y}


def test_evaluation_result_reconstructs_equal_fold_mse():
    frame = data()
    ids = np.array([f"person-{i:03d}" for i in range(90)])
    result = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, seed=5, mstop=30,
        score="mse", observation_ids=ids,
    )
    assert isinstance(result, mb.EvaluationResult)
    assert result.prediction_scale == "response"
    assert result.score_parameter is None
    assert result.aggregation == "equal_fold_mean"
    assert result.n_observations == result.n_evaluations == 90
    assert result.fit_count == 3
    assert result.excluded_record_ids == ()
    assert result.evaluation_task == "new_exchangeable_observation"
    assert result.independent_unit == "record"
    assert result.n_independent_units == 90
    assert len(result.records) == 90

    rebuilt = []
    for fold in range(3):
        rows = [row for row in result.records if row["fold"] == fold]
        weights = np.array([row["evaluation_weight"] for row in rows])
        losses = np.array([row["loss"] for row in rows])
        rebuilt.append(np.average(losses, weights=weights))
    np.testing.assert_allclose(result.fold_scores, rebuilt)
    assert result.score == pytest.approx(np.mean(rebuilt))
    assert float(result) == result.score
    assert {row["record_id"] for row in result.records} == set(ids)
    assert repr(result).startswith("EvaluationResult(mse=")
    assert "array(" not in repr(result)


def test_rolling_evaluation_records_rows_that_were_never_judged():
    frame = data(50)
    scheme = mb.rolling_folds(50, B=3, initial=20, horizon=5)
    result = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=scheme, mstop=20, score="mse",
    )
    assert result.fold_scheme.startswith("rolling-origin")
    assert result.evaluation_task == "future_horizon"
    assert result.n_evaluations == 15
    assert {row["row"] for row in result.records} == set(range(20, 35))
    assert set(result.excluded_record_ids) == set(range(20)) | set(range(35, 50))
    assert [row["n_evaluation"] for row in result.folds] == [5, 5, 5]


def test_grouped_result_names_new_group_prediction_as_its_target():
    frame = data(60)
    groups = np.repeat(np.arange(15), 4)
    result = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=mb.grouped_folds(groups, B=3),
        mstop=20, score="mse",
    )
    assert result.evaluation_task == "new_group"
    assert result.independent_unit == "group"
    assert result.n_independent_units == 15


def test_different_prediction_tasks_are_not_comparable():
    frame = data(60)
    random = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, mstop=20, score="mse",
    )
    grouped = mb.evaluate_specification(
        "bols(x)", frame, "y",
        folds=mb.grouped_folds(np.repeat(np.arange(15), 4), B=3),
        mstop=20, score="mse",
    )
    with pytest.raises(ValueError, match="evaluation task"):
        random.assert_comparable(grouped, paired=False)


def test_compact_result_keeps_scores_but_omits_row_arrays():
    result = mb.evaluate_specification(
        "bols(x)", data(60), "y", folds=3, mstop=20, score="mse",
        retain_arrays=False,
    )
    assert result.records == ()
    assert len(result.folds) == 3
    assert all(row["training_rows"] is None and row["evaluation_rows"] is None
               for row in result.folds)
    assert np.isfinite(result.score)


def test_data_fingerprint_changes_with_row_order_or_values():
    frame = data(60)
    first = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, mstop=10, retain_arrays=False,
    )
    permutation = np.arange(59, -1, -1)
    reordered = {name: values[permutation] for name, values in frame.items()}
    second = mb.evaluate_specification(
        "bols(x)", reordered, "y", folds=3, mstop=10, retain_arrays=False,
    )
    changed = {name: values.copy() for name, values in frame.items()}
    changed["x"][0] += .01
    third = mb.evaluate_specification(
        "bols(x)", changed, "y", folds=3, mstop=10, retain_arrays=False,
    )
    assert len(first.data_fingerprint) == 64
    assert len({first.data_fingerprint, second.data_fingerprint,
                third.data_fingerprint}) == 3


def test_cv_risk_float_contract_and_result_score_agree():
    frame = data(60)
    scalar = mb.cv_risk("bols(x)", frame, "y", folds=3, seed=2,
                        mstop=20, score="mse")
    result = mb.cv_risk("bols(x)", frame, "y", folds=3, seed=2,
                        mstop=20, score="mse", return_result=True)
    assert isinstance(scalar, float)
    assert scalar == result.score


def test_comparison_requires_the_same_target_and_exact_pairing():
    frame = data(60)
    linear = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, seed=2, mstop=20, score="mse",
    )
    smooth = mb.evaluate_specification(
        "bbs(x, df=3)", frame, "y", folds=3, seed=2, mstop=20, score="mse",
    )
    assert linear.comparison_issues(smooth) == ()
    linear.assert_comparable(smooth)

    other_folds = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, seed=3, mstop=20, score="mse",
    )
    with pytest.raises(ValueError, match="record-fold-weight pairing"):
        linear.assert_comparable(other_folds)

    family_loss = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, seed=2, mstop=20, score="family",
    )
    with pytest.raises(ValueError, match="score definition|prediction scale|loss units"):
        linear.assert_comparable(family_loss)


def test_compact_results_cannot_claim_pairing_after_membership_is_discarded():
    frame = data(60)
    first = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, seed=2, mstop=20,
        retain_arrays=False,
    )
    second = mb.evaluate_specification(
        "bbs(x, df=3)", frame, "y", folds=3, seed=2, mstop=20,
        retain_arrays=False,
    )
    assert first.comparison_issues(second, paired=False) == ()
    with pytest.raises(ValueError, match="retained record-fold membership"):
        first.assert_comparable(second)


def test_row_reordering_is_not_comparable_even_when_values_are_the_same_set():
    frame = data(60)
    first = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, seed=2, mstop=20,
    )
    order = np.arange(59, -1, -1)
    reordered = {name: values[order] for name, values in frame.items()}
    second = mb.evaluate_specification(
        "bols(x)", reordered, "y", folds=3, seed=2, mstop=20,
    )
    with pytest.raises(ValueError, match="data fingerprint"):
        first.assert_comparable(second)


@pytest.mark.parametrize("score, parameter", [("pinball", .8), ("expectile", .8)])
def test_asymmetric_scores_are_reconstructible(score, parameter):
    result = mb.evaluate_specification(
        "bols(x)", data(60), "y", folds=3, seed=2, mstop=20,
        score=score, score_parameter=parameter,
    )
    y = np.array([row["observed"] for row in result.records])
    prediction = np.array([row["prediction"] for row in result.records])
    residual = y - prediction
    if score == "pinball":
        expected = np.where(residual >= 0, parameter * residual,
                            (parameter - 1) * residual)
    else:
        expected = np.where(residual >= 0, parameter, 1 - parameter) * residual**2
    np.testing.assert_allclose([row["loss"] for row in result.records], expected)
    assert result.score_parameter == parameter


def test_binomial_log_loss_uses_response_probabilities_and_natural_logs():
    rng = np.random.default_rng(3)
    x = rng.normal(size=120)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    result = mb.evaluate_specification(
        "bols(x)", {"x": x, "y": y}, "y", family=mb.Binomial(),
        folds=4, seed=2, mstop=30, score="log_loss",
    )
    observed = np.array([row["observed"] for row in result.records])
    probability = np.array([row["prediction"] for row in result.records])
    expected = -(observed * np.log(probability)
                 + (1 - observed) * np.log1p(-probability))
    np.testing.assert_allclose([row["loss"] for row in result.records], expected)
    assert np.all((probability >= 0) & (probability <= 1))
    assert result.loss_units == "nats_per_observation"
    residuals = result.residual_frame(output="records")
    np.testing.assert_allclose(
        [row["residual"] for row in residuals], observed - probability
    )
    calibration = result.calibration_frame(n_bins=6, output="records")
    assert sum(row["n"] for row in calibration) == 120
    assert all(0 <= row["predicted_mean"] <= 1 for row in calibration)
    assert all(0 <= row["observed_rate"] <= 1 for row in calibration)
    assert all(row["uncertainty"] == "iid_kish_effective_n_wilson"
               for row in calibration)
    assert all(row["lower"] <= row["observed_rate"] <= row["upper"]
               for row in calibration)
    brier = mb.evaluate_specification(
        "bols(x)", {"x": x, "y": y}, "y", family=mb.Binomial(),
        folds=4, seed=2, mstop=30, score="brier",
    )
    np.testing.assert_allclose(
        [row["loss"] for row in brier.records],
        [(row["observed"] - row["prediction"]) ** 2 for row in brier.records],
    )
    assert brier.loss_units == "squared_probability_units"


def test_grouped_calibration_does_not_report_row_iid_intervals():
    rng = np.random.default_rng(9)
    x = rng.normal(size=120)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    groups = np.repeat(np.arange(30), 4)
    result = mb.evaluate_specification(
        "bols(x)", {"x": x, "y": y}, "y", family=mb.Binomial(),
        folds=mb.grouped_folds(groups, B=5), mstop=30, score="log_loss",
    )
    rows = result.calibration_frame(n_bins=6, output="records")
    assert all(row["uncertainty"] == "unavailable_for_declared_dependence"
               for row in rows)
    assert all(row["lower"] is None and row["upper"] is None for row in rows)


def test_response_diagnostics_require_retained_response_predictions():
    compact = mb.evaluate_specification(
        "bols(x)", data(60), "y", folds=3, mstop=10, score="mse",
        retain_arrays=False,
    )
    with pytest.raises(ValueError, match="retain_arrays"):
        compact.residual_frame()
    family_loss = mb.evaluate_specification(
        "bols(x)", data(60), "y", folds=3, mstop=10, score="family",
    )
    with pytest.raises(ValueError, match="response-scale"):
        family_loss.residual_frame()
    with pytest.raises(ValueError, match="response-scale"):
        family_loss.calibration_frame()


def test_evaluation_diagnostic_plots_use_the_retained_tables():
    pytest.importorskip("altair")
    rng = np.random.default_rng(31)
    x = rng.normal(size=120)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    result = mb.evaluate_specification(
        "bols(x)", {"x": x, "y": y}, "y", family=mb.Binomial(),
        folds=4, mstop=30, score="log_loss",
    )
    residuals = mb.plot.plot_evaluation_residuals(result).to_dict()
    calibration = mb.plot.plot_evaluation_calibration(result, n_bins=6).to_dict()
    assert residuals["title"].startswith("Held-out residuals")
    assert calibration["title"] == "Held-out probability calibration"


@pytest.mark.parametrize("score, parameter, message", [
    ("pinball", None, "required"),
    ("expectile", 1.0, "must be in"),
    ("mse", .5, "only used"),
    ("unknown", None, "one of"),
])
def test_score_protocol_refuses_missing_or_ignored_semantics(score, parameter, message):
    with pytest.raises(ValueError, match=message):
        mb.evaluate_specification(
            "bols(x)", data(60), "y", folds=3, mstop=10,
            score=score, score_parameter=parameter,
        )


@pytest.mark.parametrize("ids", [np.arange(59), np.zeros(60)])
def test_observation_ids_must_uniquely_match_rows(ids):
    with pytest.raises(ValueError, match="observation_ids"):
        mb.evaluate_specification(
            "bols(x)", data(60), "y", folds=3, mstop=10,
            observation_ids=ids,
        )


def test_specification_comparison_carries_and_checks_evaluation_identity():
    frame = data(90)
    ids = np.array([f"case-{i}" for i in range(90)])
    rows = mb.compare_specifications(
        {"linear": "bols(x)", "smooth": "bbs(x, df=3)"},
        frame, "y", folds=3, seed=4, mstop=20, score="pinball",
        score_parameter=.8, observation_ids=ids, include_evaluations=True,
    )
    assert len(rows) == 2
    assert rows[0]["score parameter"] == .8
    assert rows[0]["prediction scale"] == "response"
    assert rows[0]["loss units"] == "response_units"
    assert rows[0]["data fingerprint"] == rows[1]["data fingerprint"]
    assert rows[0]["evaluation task"] == "new_exchangeable_observation"
    first, second = rows[0]["evaluation"], rows[1]["evaluation"]
    first.assert_comparable(second)
    assert {row["record_id"] for row in first.records} == set(ids)


def test_evaluation_objects_are_not_hidden_inside_frame_output():
    with pytest.raises(ValueError, match="requires output='records'"):
        mb.compare_specifications(
            ["bols(x)", "bbs(x, df=3)"], data(60), "y", folds=3,
            mstop=20, include_evaluations=True, output="polars",
        )


def test_specification_comparison_preflights_and_reports_fit_budget():
    frame = data(60)
    with pytest.raises(ValueError, match="needs 6 fits.*fit_budget=5"):
        mb.compare_specifications(
            ["bols(x)", "bbs(x, df=3)"], frame, "y", folds=3,
            mstop=20, fit_budget=5,
        )
    rows = mb.compare_specifications(
        ["bols(x)", "bbs(x, df=3)"], frame, "y", folds=3,
        mstop=20, fit_budget=6,
    )
    assert [row["fit count"] for row in rows] == [3, 3]
    assert [row["cumulative fit count"] for row in rows] == [3, 6]
    assert all(row["planned total fits"] == 6 for row in rows)
    assert all(row["fit budget"] == 6 for row in rows)


def test_fit_count_includes_inner_tuning_refits():
    frame = data(60)
    with pytest.raises(ValueError, match="needs 12 fits.*fit_budget=11"):
        mb.compare_specifications(
            ["bols(x)"], frame, "y", folds=2, mstop="cv", fit_budget=11,
        )
    rows = mb.compare_specifications(
        ["bols(x)"], frame, "y", folds=2, mstop="cv", fit_budget=12,
    )
    assert rows[0]["planned fit count"] == 12
    assert rows[0]["fit count"] == 12


def known_classification_result():
    observed = np.array([1.0, 0.0, 1.0, 0.0])
    probability = np.array([.9, .8, .4, .1])
    weights = np.array([1.0, 2.0, 1.0, 1.0])
    records = tuple({
        "record_id": f"case-{i}", "row": i, "fold": 0,
        "observed": observed[i], "prediction": probability[i],
        "evaluation_weight": weights[i], "loss": np.nan,
    } for i in range(4))
    return mb.EvaluationResult(
        score=.3, fold_scores=np.array([.3]), formula="y ~ score",
        response="y", family="Binomial", score_name="log_loss",
        score_parameter=None, prediction_scale="response",
        loss_units="nats_per_observation", aggregation="equal_fold_mean",
        fold_scheme="supplied", evaluation_task="new_exchangeable_observation",
        independent_unit="record", n_independent_units=4, seed=1,
        tuning="fixed_mstop=1", data_fingerprint="0" * 64,
        fingerprint_scope="all_columns", n_observations=4, n_evaluations=4,
        fit_count=1,
        excluded_record_ids=(), records=records,
        folds=({"fold": 0, "score": .3},),
    )


def test_weighted_classification_metrics_match_a_hand_calculation():
    result = known_classification_result()
    metrics = result.classification_metrics(threshold=.5)
    assert metrics["true_positive_weight"] == 1
    assert metrics["false_positive_weight"] == 2
    assert metrics["false_negative_weight"] == 1
    assert metrics["true_negative_weight"] == 1
    assert metrics["accuracy"] == pytest.approx(2 / 5)
    assert metrics["sensitivity"] == pytest.approx(.5)
    assert metrics["specificity"] == pytest.approx(1 / 3)
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(.4)
    expected_brier = (.1**2 + 2 * .8**2 + .6**2 + .1**2) / 5
    assert metrics["brier"] == pytest.approx(expected_brier)
    expected_log = -(
        np.log(.9) + 2 * np.log(.2) + np.log(.4) + np.log(.9)
    ) / 5
    assert metrics["log_loss"] == pytest.approx(expected_log)
    bins = result.calibration_frame(n_bins=4, output="records")
    zero_bin = next(row for row in bins if row["observed_rate"] == 0)
    one_bin = next(row for row in bins if row["observed_rate"] == 1)
    assert zero_bin["upper"] > 0
    assert one_bin["lower"] < 1
    assert zero_bin["uncertainty"] == "iid_kish_effective_n_wilson"
    at_zero = result.classification_metrics(threshold=0)
    at_one = result.classification_metrics(threshold=1)
    assert at_zero["true_negative_weight"] == 0
    assert at_zero["false_negative_weight"] == 0
    assert at_one["true_positive_weight"] == 0
    assert at_one["false_positive_weight"] == 0


def test_weighted_roc_auc_matches_pairwise_probability_oracle():
    rows = known_classification_result().roc_frame(output="records")
    assert rows[0]["false_positive_rate"] == 0
    assert rows[0]["sensitivity"] == 0
    assert rows[-1]["false_positive_rate"] == 1
    assert rows[-1]["sensitivity"] == 1
    assert all(row["auc"] == pytest.approx(4 / 6) for row in rows)


def test_weighted_average_precision_matches_step_integral():
    rows = known_classification_result().precision_recall_frame(output="records")
    assert rows[0]["recall"] == 0 and rows[0]["precision"] == 1
    assert rows[-1]["recall"] == 1
    assert all(row["average_precision"] == pytest.approx(.75) for row in rows)


def test_threshold_utility_uses_declared_confusion_payoffs():
    result = known_classification_result()
    rows = result.threshold_frame([.5, .85], output="records")
    at_half = rows[0]
    assert at_half["average_utility"] == pytest.approx(-3 / 5)
    custom = result.threshold_frame(
        [.5, .85], utility=(1, -2, -4, 3), output="records"
    )[0]
    assert custom["average_utility"] == pytest.approx((1 - 4 - 4 + 3) / 5)


def test_classification_dashboard_uses_held_out_curves_and_declared_utility():
    pytest.importorskip("altair")
    chart = mb.plot.plot_classification_evaluation(
        known_classification_result(), thresholds=[.25, .5, .75],
        utility=(1, -2, -4, 3),
    ).to_dict()
    assert len(chart["hconcat"]) == 3
    assert chart["hconcat"][0]["title"].startswith("ROC · AUC")
    assert chart["hconcat"][1]["title"].startswith("Precision–recall · AP")
    assert chart["hconcat"][2]["title"] == "Threshold utility"


def test_classification_tools_refuse_nonprobability_or_single_class_evidence():
    continuous = mb.evaluate_specification(
        "bols(x)", data(60), "y", folds=3, mstop=10, score="mse"
    )
    with pytest.raises(ValueError, match="coded 0/1"):
        continuous.classification_metrics()
    result = known_classification_result()
    with pytest.raises(ValueError, match="thresholds"):
        result.threshold_frame([.5])
    with pytest.raises(ValueError, match="utility"):
        result.threshold_frame([.4, .6], utility=(0, -1, -1))


def known_count_result(prediction):
    observed = np.array([0.0, 2.0, 1.0])
    weights = np.array([1.0, 2.0, 1.0])
    records = tuple({
        "record_id": f"count-{i}", "row": i, "fold": 0,
        "observed": observed[i], "prediction": prediction[i],
        "evaluation_weight": weights[i], "loss": np.nan,
    } for i in range(3))
    return mb.EvaluationResult(
        score=.4, fold_scores=np.array([.4]), formula="y ~ x",
        response="y", family="Poisson", score_name="mse",
        score_parameter=None, prediction_scale="response",
        loss_units="squared_response_units", aggregation="equal_fold_mean",
        fold_scheme="supplied", evaluation_task="new_exchangeable_observation",
        independent_unit="record", n_independent_units=3, seed=1,
        tuning="fixed_mstop=1", data_fingerprint="1" * 64,
        fingerprint_scope="all_columns", n_observations=3, n_evaluations=3,
        fit_count=1, excluded_record_ids=(), records=records,
        folds=({"fold": 0, "score": .4},),
    )


def test_exposure_aware_count_metrics_match_a_hand_calculation():
    exposure = np.array([1.0, 2.0, .5])
    predicted_rate = np.array([1.0, .8, 1.5])
    expected_count = predicted_rate * exposure
    result = known_count_result(predicted_rate)
    metrics = result.count_metrics(
        exposure=exposure, prediction_quantity="rate"
    )
    observed = np.array([0.0, 2.0, 1.0])
    weights = np.array([1.0, 2.0, 1.0])
    deviance = 2 * np.array([
        expected_count[0],
        observed[1] * np.log(observed[1] / expected_count[1])
        - (observed[1] - expected_count[1]),
        observed[2] * np.log(observed[2] / expected_count[2])
        - (observed[2] - expected_count[2]),
    ])
    assert metrics["predicted_count_total"] == pytest.approx(
        np.sum(weights * expected_count)
    )
    assert metrics["observed_rate"] == pytest.approx(
        np.sum(weights * observed) / np.sum(weights * exposure)
    )
    assert metrics["predicted_rate"] == pytest.approx(
        np.sum(weights * expected_count) / np.sum(weights * exposure)
    )
    assert metrics["mean_poisson_deviance"] == pytest.approx(
        np.average(deviance, weights=weights)
    )


def test_count_and_rate_prediction_contracts_agree_after_exposure_conversion():
    exposure = np.array([1.0, 2.0, .5])
    rate = np.array([1.0, .8, 1.5])
    rate_metrics = known_count_result(rate).count_metrics(
        exposure=exposure, prediction_quantity="rate"
    )
    count_metrics = known_count_result(rate * exposure).count_metrics(
        exposure=exposure, prediction_quantity="count"
    )
    for name in (
        "predicted_count_total", "predicted_rate", "mean_absolute_count_error",
        "root_mean_squared_count_error", "mean_absolute_rate_error",
        "mean_poisson_deviance", "mean_poisson_log_score",
    ):
        assert rate_metrics[name] == pytest.approx(count_metrics[name])


@pytest.mark.parametrize("exposure, quantity, message", [
    (np.ones(2), "count", "shape"),
    (np.array([1.0, 0.0, 1.0]), "count", "positive"),
    (None, "rate", "requires exposure"),
    (np.ones(3), "mean", "prediction_quantity"),
])
def test_count_metrics_validate_exposure_and_prediction_semantics(
        exposure, quantity, message):
    with pytest.raises(ValueError, match=message):
        known_count_result(np.ones(3)).count_metrics(
            exposure=exposure, prediction_quantity=quantity
        )


def known_multiclass_result():
    observed = np.array([0, 1, 2, 1])
    probability = np.array([
        [.7, .2, .1],
        [.3, .4, .3],
        [.2, .5, .3],
        [.1, .2, .7],
    ])
    weights = np.array([1.0, 2.0, 1.0, 1.0])
    records = tuple({
        "record_id": f"multi-{i}", "row": i, "fold": 0,
        "observed": int(observed[i]), "prediction": probability[i].tolist(),
        "evaluation_weight": weights[i], "loss": np.nan,
    } for i in range(4))
    return mb.EvaluationResult(
        score=.7, fold_scores=np.array([.7]), formula="y ~ x",
        response="y", family="Multinomial", score_name="log_loss",
        score_parameter=None, prediction_scale="response",
        loss_units="nats_per_observation", aggregation="equal_fold_mean",
        fold_scheme="supplied", evaluation_task="new_exchangeable_observation",
        independent_unit="record", n_independent_units=4, seed=1,
        tuning="fixed_mstop=1", data_fingerprint="2" * 64,
        fingerprint_scope="all_columns", n_observations=4, n_evaluations=4,
        fit_count=1, excluded_record_ids=(), records=records,
        folds=({"fold": 0, "score": .7},), class_labels=("a", "b", "c"),
    )


def test_multiclass_metrics_match_a_weighted_hand_calculation():
    result = known_multiclass_result()
    metrics = result.multiclass_metrics()
    probability = np.array([row["prediction"] for row in result.records])
    observed = np.array([row["observed"] for row in result.records])
    weights = np.array([row["evaluation_weight"] for row in result.records])
    one_hot = np.eye(3)[observed]
    assert metrics["n_classes"] == 3
    assert metrics["top_class_accuracy"] == pytest.approx(3 / 5)
    assert metrics["categorical_log_loss"] == pytest.approx(np.average(
        -np.log(probability[np.arange(4), observed]), weights=weights
    ))
    assert metrics["multiclass_brier"] == pytest.approx(np.average(
        np.sum((one_hot - probability) ** 2, axis=1), weights=weights
    ))
    assert metrics["macro_recall"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert [row["class_label"] for row in metrics["per_class"]] == ["a", "b", "c"]


def test_multiclass_confusion_and_calibration_preserve_every_class():
    result = known_multiclass_result()
    confusion = result.multiclass_confusion_frame(output="records")
    calibration = result.multiclass_calibration_frame(n_bins=2, output="records")
    assert len(confusion) == 9
    assert sum(row["weight"] for row in confusion) == 5
    assert {row["observed_class"] for row in confusion} == {0, 1, 2}
    assert {row["observed_label"] for row in confusion} == {"a", "b", "c"}
    assert len(calibration) == 6
    assert {row["class"] for row in calibration} == {0, 1, 2}
    assert all(row["uncertainty"] == "not_implemented_for_multiclass_bins"
               for row in calibration)


def test_multiclass_dashboard_uses_weighted_confusion_and_classwise_calibration():
    pytest.importorskip("altair")
    chart = mb.plot.plot_multiclass_evaluation(
        known_multiclass_result(), n_bins=2
    ).to_dict()
    assert len(chart["hconcat"]) == 2
    assert chart["hconcat"][0]["layer"][0]["title"] == "Weighted confusion matrix"
    assert chart["hconcat"][1]["title"] == "One-vs-rest calibration"


def test_binary_and_multiclass_probability_contracts_do_not_cross_dispatch():
    result = known_multiclass_result()
    with pytest.raises(ValueError, match="use multiclass_metrics"):
        result.classification_metrics()
    with pytest.raises(ValueError, match="use multiclass_metrics"):
        result.roc_frame()
    with pytest.raises(ValueError, match="n by K"):
        known_classification_result().multiclass_metrics()


def test_multinomial_specification_evaluation_retains_full_probability_rows():
    rng = np.random.default_rng(101)
    n = 180
    x = rng.normal(size=n)
    logits = np.column_stack([.8 * x, -.5 * x, np.zeros(n)])
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    codes = np.array([rng.choice(3, p=row) for row in probabilities])
    labels = np.array(["alpha", "beta", "gamma"])[codes]
    data_multi = {"x": x, "class": labels}
    folds = mb.stratified_folds(labels, B=3, seed=3)
    result = mb.evaluate_specification(
        "bols(x)", data_multi, "class", family=mb.Multinomial(n_classes=3),
        folds=folds, mstop=40, score="log_loss",
    )
    assert result.class_labels == ("alpha", "beta", "gamma")
    retained = np.asarray([row["prediction"] for row in result.records])
    observed = np.asarray([row["observed"] for row in result.records], dtype=int)
    np.testing.assert_allclose(retained.sum(axis=1), 1)
    expected = -np.log(retained[np.arange(n), observed])
    np.testing.assert_allclose([row["loss"] for row in result.records], expected)
    assert result.score == pytest.approx(np.mean(result.fold_scores))
    assert result.multiclass_metrics()["n_classes"] == 3

    brier = mb.evaluate_specification(
        "bols(x)", data_multi, "class", family=mb.Multinomial(),
        folds=folds, mstop=40, score="brier",
    )
    one_hot = np.eye(3)[np.asarray(
        [row["observed"] for row in brier.records], dtype=int
    )]
    predicted = np.asarray([row["prediction"] for row in brier.records])
    np.testing.assert_allclose(
        [row["loss"] for row in brier.records],
        np.sum((one_hot - predicted) ** 2, axis=1),
    )


def test_multinomial_evaluation_refuses_a_training_fold_missing_a_class():
    data_multi = {
        "x": np.arange(6, dtype=float),
        "class": np.array(["rare", "a", "a", "b", "b", "b"]),
    }
    weights = np.ones((6, 2))
    weights[0, 0] = 0
    weights[3:, 1] = 0
    with pytest.raises(ValueError, match="omit classes.*stratified"):
        mb.evaluate_specification(
            "bols(x)", data_multi, "class", family=mb.Multinomial(),
            folds=weights, mstop=10, score="log_loss",
        )


def test_multinomial_numeric_labels_follow_fit_order_not_lexical_repr():
    from mboostrs.explore import _multinomial_response

    codes, labels = _multinomial_response(
        {"class": np.array([10, 2, 1, 10])}, "class"
    )
    assert labels == (1, 2, 10)
    np.testing.assert_array_equal(codes, [2, 1, 0, 2])


def survival_data(n=120, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    latent = np.exp(0.5 * x + rng.normal(scale=0.5, size=n))
    censor = np.exp(rng.normal(0.3, 0.5, n))
    return {
        "x": x,
        "time": np.minimum(latent, censor),
        "status": (latent <= censor).astype(float),
    }


@pytest.mark.parametrize("family", [mb.Weibull(), mb.CoxPH()])
def test_survival_result_scores_the_held_out_family_loss(family):
    result = mb.evaluate_specification(
        "bols(x)", survival_data(), ("time", "status"), family=family,
        folds=3, mstop=30, seed=1,
    )
    assert result.response == "Surv(time, status)"
    assert result.prediction_scale == "link"
    assert result.loss_units == "family_loss"
    assert result.n_evaluations == 120
    assert all("event" in row for row in result.records)
    assert {row["event"] for row in result.records} <= {0.0, 1.0}
    assert np.isfinite(result.score)


def test_weibull_row_losses_reconstruct_each_survival_fold_score():
    result = mb.evaluate_specification(
        "bols(x)", survival_data(), "Surv(time, status)", family=mb.Weibull(),
        folds=3, mstop=30, seed=1,
    )
    for fold, score in enumerate(result.fold_scores):
        rows = [row for row in result.records if row["fold"] == fold]
        weights = np.array([row["evaluation_weight"] for row in rows])
        losses = np.array([row["loss"] for row in rows])
        assert np.average(losses, weights=weights) == pytest.approx(score)


def test_survival_family_refuses_a_response_it_cannot_score():
    data = survival_data()
    with pytest.raises(ValueError, match="requires a Surv"):
        mb.evaluate_specification("bols(x)", data, "time", family=mb.CoxPH(),
                                  folds=3, mstop=10)
    with pytest.raises(ValueError, match="needs a survival family"):
        mb.evaluate_specification("bols(x)", data, "Surv(time, status)",
                                  family=mb.Gaussian(), folds=3, mstop=10)
    with pytest.raises(ValueError, match="score='family'"):
        mb.evaluate_specification("bols(x)", data, ("time", "status"),
                                  family=mb.CoxPH(), folds=3, mstop=10, score="mse")


def test_lss_result_names_the_joint_density_score_and_its_records():
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 120)
    y = 1 + x + np.exp(0.3 * x) * rng.normal(size=120)
    result = mb.evaluate_specification(
        {"mu": "bbs(x, df=3)", "sigma": "bols(x)"}, {"x": x, "y": y}, "y",
        family=mb.GaussianLSS(), folds=3, mstop=30, seed=1,
    )
    assert result.score_name == "family"
    assert result.loss_units == "nats_per_observation"
    assert result.prediction_scale == "distributional parameters, response scale"
    assert len(result.records[0]["prediction"]) == 2
    for fold, score in enumerate(result.fold_scores):
        rows = [row for row in result.records if row["fold"] == fold]
        weights = np.array([row["evaluation_weight"] for row in rows])
        losses = np.array([row["loss"] for row in rows])
        assert np.average(losses, weights=weights) == pytest.approx(score)
    assert "scale scale" not in result.describe()


def test_known_group_folds_hold_out_rows_whose_group_trains_the_fold():
    groups = np.repeat(np.arange(15), 4)
    scheme = mb.known_group_folds(groups, B=3, seed=4)
    for b in range(scheme.n_folds):
        for row in scheme.fold(b):
            mates = np.flatnonzero(groups == groups[row])
            assert np.any(scheme.weights[mates, b] > 0)
    result = mb.evaluate_specification(
        "bols(x)", data(60), "y", folds=scheme, mstop=20, score="mse",
    )
    assert result.evaluation_task == "known_group"
    assert result.independent_unit == "record"
    assert "known group" in result.describe()
    grouped = mb.evaluate_specification(
        "bols(x)", data(60), "y", folds=mb.grouped_folds(groups, B=3),
        mstop=20, score="mse",
    )
    with pytest.raises(ValueError, match="evaluation task"):
        result.assert_comparable(grouped, paired=False)


def test_known_group_folds_refuse_a_group_with_one_row():
    groups = np.array([0, 0, 1, 1, 2])
    with pytest.raises(ValueError, match="group 2 has one"):
        mb.known_group_folds(groups, B=2)


def test_rolling_folds_order_rows_by_timestamp_and_keep_an_embargo_gap():
    rng = np.random.default_rng(7)
    stamps = rng.permutation(np.arange(50, dtype=float))
    scheme = mb.rolling_folds(50, B=3, initial=20, horizon=5, time=stamps,
                              embargo=2.0)
    assert scheme.check_leakage(raise_on_leak=False)["clean"]
    for b in range(3):
        trained = np.flatnonzero(scheme.weights[:, b] > 0)
        judged = scheme.fold(b)
        assert stamps[trained].max() <= stamps[judged].min() - 2.0
        assert not np.any(scheme.held_out[trained, b])
    first_judged = stamps[scheme.fold(0)].min()
    rows_in_gap = np.flatnonzero((stamps > first_judged - 2.0) & (stamps < first_judged))
    assert rows_in_gap.size and np.all(scheme.weights[rows_in_gap, 0] == 0)
    assert np.all(~scheme.held_out[rows_in_gap, 0])


def test_embargo_violation_is_reported_as_a_leak():
    stamps = np.arange(50, dtype=float)
    scheme = mb.rolling_folds(50, B=2, initial=20, horizon=5, time=stamps)
    tight = mb.FoldScheme(
        weights=np.asarray(scheme.weights), held_out=np.asarray(scheme.held_out),
        scheme="tight", ordered=True, times=stamps, embargo=3.0,
    )
    report = tight.check_leakage(raise_on_leak=False)
    assert [row["kind"] for row in report["leaks"]] == ["embargo", "embargo"]
    with pytest.raises(ValueError, match="inside the embargo"):
        tight.check_leakage()


def test_embargo_needs_timestamps_and_a_matching_unit():
    with pytest.raises(ValueError, match="needs time="):
        mb.rolling_folds(20, B=2, initial=10, horizon=3, embargo=1.0)
    days = np.arange(20).astype("datetime64[D]")
    with pytest.raises(ValueError, match="timedelta64"):
        mb.rolling_folds(20, B=2, initial=10, horizon=3, time=days, embargo=1.0)
    with pytest.raises(ValueError, match="same units"):
        mb.rolling_folds(20, B=2, initial=10, horizon=3, time=np.arange(20.0),
                         embargo=np.timedelta64(1, "D"))
    days = np.arange(20).astype("datetime64[D]")
    scheme = mb.rolling_folds(20, B=2, initial=10, horizon=3, time=days,
                              embargo=np.timedelta64(2, "D"))
    assert scheme.check_leakage(raise_on_leak=False)["clean"]


def test_failed_fold_stays_listed_and_makes_the_score_undefined():
    rng = np.random.default_rng(9)
    x = rng.normal(size=60)
    y = x + rng.normal(scale=.3, size=60)
    x[:20] = np.nan
    result = mb.evaluate_specification(
        "bols(x)", {"x": x, "y": y}, "y", folds=mb.blocked_folds(60, B=3),
        mstop=20, score="mse", na_action="omit",
    )
    assert [row["status"] for row in result.folds] == ["failed", "ok", "ok"]
    failed = result.failed_folds
    assert len(failed) == 1 and failed[0]["fold"] == 0
    assert "na_action='omit'" in failed[0]["error"]
    assert np.isnan(result.score) and np.isnan(result.fold_scores[0])
    assert np.all(np.isfinite(result.fold_scores[1:]))
    assert len(result.fold_frame(output="records")) == 3
    assert "is undefined" in result.describe() and "1 of 3 folds failed" in result.describe()
    healthy = mb.evaluate_specification(
        "bols(x)", {"x": np.nan_to_num(x), "y": y}, "y",
        folds=mb.blocked_folds(60, B=3), mstop=20, score="mse",
    )
    assert "failed folds" in result.comparison_issues(healthy, paired=False)


def test_the_float_contract_still_raises_on_a_failed_fold():
    rng = np.random.default_rng(9)
    x = rng.normal(size=60)
    y = x + rng.normal(scale=.3, size=60)
    x[:20] = np.nan
    with pytest.raises(ValueError, match="evaluation with na_action='omit'"):
        mb.cv_risk("bols(x)", {"x": x, "y": y}, "y", folds=mb.blocked_folds(60, B=3),
                   mstop=20, score="mse", na_action="omit")


def test_one_workflow_evaluates_standard_lss_classification_and_grouped_models():
    rng = np.random.default_rng(11)
    n = 120
    subject = np.repeat(np.arange(30), 4)
    x = rng.normal(size=n)
    y = x + rng.normal(scale=.5, size=n)
    labels = (x + rng.normal(size=n) > 0).astype(float)
    visit = np.tile(np.arange(4), 30)
    ids = np.array([f"subject-{s:02d}-visit-{v}" for s, v in zip(subject, visit, strict=True)])
    frame = {"x": x, "y": y, "label": labels, "subject": subject.astype(float)}
    standard = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=3, mstop=20, score="mse", observation_ids=ids,
    )
    lss = mb.evaluate_specification(
        {"mu": "bols(x)", "sigma": "bols(x)"}, frame, "y", family=mb.GaussianLSS(),
        folds=3, mstop=20, observation_ids=ids,
    )
    classification = mb.evaluate_specification(
        "bols(x)", frame, "label", family=mb.Binomial(), folds=3, mstop=20,
        score="log_loss", observation_ids=ids,
    )
    grouped = mb.evaluate_specification(
        "bols(x)", frame, "y", folds=mb.grouped_folds(subject, B=3), mstop=20,
        score="mse", observation_ids=ids,
    )
    results = (standard, lss, classification, grouped)
    for result in results:
        assert {row["record_id"] for row in result.records} == set(ids)
        assert result.n_evaluations == n
    assert [result.evaluation_task for result in results] == [
        "new_exchangeable_observation", "new_exchangeable_observation",
        "new_exchangeable_observation", "new_group",
    ]


def test_cvrisk_and_stabsel_gamboostlss_drop_missing_rows_under_omit():
    rng = np.random.default_rng(5)
    n = 120
    x = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    x[[3, 40]] = np.nan
    y = 1 + z + np.exp(0.3 * z) * rng.normal(size=n)
    frame = {"x": x, "z": z, "y": y}
    formula = {"mu": "y ~ bbs(x, df=3) %X% bbs(z, df=3)", "sigma": "y ~ bols(z)"}
    folds = mb.grouped_folds(np.arange(n) // 4, B=3)
    control = mb.boost_control(20, 0.1)
    with pytest.raises(Exception,
                      match=r"missing value\(s\).*na_action='impute'.*na_action='omit'"):
        mb.cvrisk_gamboostLSS(formula, frame, family=mb.GaussianLSS(), folds=folds,
                              control=control)
    cv = mb.cvrisk_gamboostLSS(formula, frame, family=mb.GaussianLSS(), folds=folds,
                               control=control, na_action="omit")
    risk = np.asarray(cv.risk_matrix)
    assert risk.shape[0] == 3
    assert np.all(np.isfinite(risk))
    selection = mb.stabsel_gamboostLSS(formula, frame, family=mb.GaussianLSS(), B=6,
                                       q=1, PFER=2.0, seed=2, control=control,
                                       na_action="omit")
    assert type(selection).__name__ == "LssStabSelResult"


def test_multi_column_lss_responses_are_refused_by_name_not_read_as_a_column():
    rng = np.random.default_rng(4)
    data = {"x": rng.uniform(-1, 1, 40), "a": rng.poisson(3, 40).astype(float),
            "b": rng.poisson(2, 40).astype(float)}
    with pytest.raises(ValueError, match="DirichletLSS.*cvrisk"):
        mb.evaluate_specification({"alpha": "bols(x)", "beta": "bols(x)"}, data,
                                  "cbind(a, b)", family=mb.DirichletLSS(K=2),
                                  folds=3, mstop=10)
