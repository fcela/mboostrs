"""Identity-aware results for held-out model evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from mboostrs._frame import column_names, read_column
from mboostrs._output import export_result, output_frame

__all__ = ["EvaluationResult", "data_fingerprint"]


def data_fingerprint(data) -> str:
    """Hash column identity, order, values, levels and missingness.

    The digest covers every supplied column because a generic frame has no
    authoritative formula dependency graph. It records identity without
    retaining private row data. Adding an unused column therefore changes the
    digest; callers comparing results can see that the input artifacts differed
    rather than treating a partial hash as proof that they were identical.
    """
    digest = hashlib.sha256()
    for name in column_names(data):
        column = read_column(data, name)
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(column.kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(column.levels).encode("utf-8"))
        values = np.ascontiguousarray(column.values, dtype=np.float64)
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


#: Columns of ``EvaluationResult.to_frame``. An evaluation run without
#: ``retain_arrays`` has no records, and its frame keeps these names.
RECORD_COLUMNS = (
    "record_id", "row", "fold", "observed", "prediction", "evaluation_weight",
    "loss",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Held-out predictions, scoring semantics and provenance for one specification."""

    score: float
    fold_scores: np.ndarray
    formula: str
    response: str
    family: str
    score_name: str
    score_parameter: float | None
    prediction_scale: str
    loss_units: str
    aggregation: str
    fold_scheme: str
    evaluation_task: str
    independent_unit: str
    n_independent_units: int
    seed: int
    tuning: str
    data_fingerprint: str
    fingerprint_scope: str
    n_observations: int
    n_evaluations: int
    fit_count: int
    excluded_record_ids: tuple
    records: tuple[dict, ...]
    folds: tuple[dict, ...]
    warnings: tuple[str, ...] = ()
    class_labels: tuple | None = None
    #: `build_info()["build_id"]` of the native extension that produced the
    #: result: the compiled code, its compiler and its target features.
    build_id: str = ""
    #: The response transform the fits used: ``"identity"``, or ``"log"`` /
    #: ``"log1p"`` when `cv_risk` fitted the transformed response. Scores are
    #: on the response scale for the external scores (predictions are carried
    #: back through the inverse), so a log fit and an identity fit compare on
    #: those; the family loss is not offered with a transform.
    response_transform: str = "identity"
    #: Digest of the evaluation weights, the same whichever container holds
    #: them. Empty when the result was built without a weight record.
    weights_fingerprint: str = ""
    #: SHA-256 digest of the observation IDs in row order, the observed
    #: response and the evaluation weights
    #: (`mboostrs._comparison_identity.data_identity`). Empty for a result
    #: built without one; two non-empty digests that differ are not compared.
    data_identity: str = ""

    def __post_init__(self):
        scores = np.asarray(self.fold_scores, dtype=np.float64).copy()
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("fold_scores must be a non-empty one-dimensional array")
        failed = np.array([row.get("status") == "failed" for row in self.folds], dtype=bool)
        if failed.size and failed.size != scores.size:
            raise ValueError("folds and fold_scores must describe the same folds")
        if np.any(failed):
            # A failed fold keeps its record and a NaN score, and the result has
            # no score: a mean over the other folds would describe fewer folds.
            if not np.all(np.isnan(scores[failed])) or not np.all(np.isfinite(scores[~failed])):
                raise ValueError("a failed fold must carry NaN and every other fold a finite score")
            if not np.isnan(self.score):
                raise ValueError("score must be NaN when a fold failed")
        else:
            if not np.all(np.isfinite(scores)):
                raise ValueError("fold_scores must be finite when no fold failed")
            if not np.isfinite(self.score):
                raise ValueError("score must be finite")
        if self.aggregation != "equal_fold_mean":
            raise ValueError("aggregation must be 'equal_fold_mean'")
        scores.setflags(write=False)
        object.__setattr__(self, "fold_scores", scores)
        object.__setattr__(self, "records", tuple(dict(row) for row in self.records))
        object.__setattr__(self, "folds", tuple(dict(row) for row in self.folds))
        object.__setattr__(self, "excluded_record_ids", tuple(self.excluded_record_ids))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.class_labels is not None:
            object.__setattr__(self, "class_labels", tuple(self.class_labels))

    @property
    def aggregate(self) -> float:
        """The equal-fold mean score, the number `cv_risk` returns without
        ``return_result=True``. The same value as :attr:`score` and ``float(result)``."""
        return self.score

    def __float__(self) -> float:
        return self.score

    def __repr__(self) -> str:
        return (
            f"EvaluationResult({self.score_name}={self.score:.6g}, "
            f"task={self.evaluation_task!r}, folds={len(self.fold_scores)}, "
            f"evaluations={self.n_evaluations}, fits={self.fit_count})"
        )

    def to_dict(self, *, include_arrays=False):
        """A JSON-safe record; fold scores and row evidence only on request."""
        return export_result(
            self,
            include_arrays=include_arrays,
            arrays=("fold_scores", "records", "folds"),
        )

    def to_frame(self, output="auto"):
        """One row per held-out prediction, or an empty frame if arrays were omitted."""
        return output_frame(list(self.records), output, columns=RECORD_COLUMNS)

    def fold_frame(self, output="auto"):
        """One row per fold with its score, sample sizes, weights and stopping point."""
        return output_frame(list(self.folds), output)

    def residual_frame(self, output="auto"):
        """Held-out response residuals aligned with record and fold identity."""
        if self.prediction_scale != "response":
            raise ValueError(
                "residual_frame requires response-scale predictions; evaluate with "
                "mse, log_loss, pinball, or expectile rather than family loss"
            )
        if not self.records:
            raise ValueError("row records were omitted; rerun with retain_arrays=True")
        if np.asarray(self.records[0]["prediction"]).ndim != 0:
            raise ValueError(
                "residual_frame requires one response prediction per row; use "
                "multiclass_metrics or multiclass_calibration_frame for a probability matrix"
            )
        rows = []
        for record in self.records:
            row = dict(record)
            row["residual"] = record["observed"] - record["prediction"]
            rows.append(row)
        return output_frame(rows, output)

    def calibration_frame(self, *, n_bins=10, level=.95, output="auto"):
        """Equal-count reliability bins from held-out probability predictions.

        The interval describes sampling variation of the observed binary rate
        within a bin using a Wilson score interval with Kish effective sample
        size. Treating that effective size as binomial is an IID approximation,
        not uncertainty in the fitted probabilities. Grouped and temporal
        results retain counts and rates but return no row-IID interval.
        """
        from mboostrs._validate import check_int_at_least, check_unit_interval

        if self.prediction_scale != "response":
            raise ValueError("calibration_frame requires response-scale probabilities")
        if not self.records:
            raise ValueError("row records were omitted; rerun with retain_arrays=True")
        n_bins = check_int_at_least("n_bins", n_bins, 2, allow_none=False)
        level = check_unit_interval("level", level)
        observed = np.asarray([row["observed"] for row in self.records])
        predicted = np.asarray([row["prediction"] for row in self.records])
        weights = np.asarray([row["evaluation_weight"] for row in self.records])
        if predicted.ndim != 1:
            raise ValueError(
                "binary calibration requires one probability per row; use "
                "multiclass_calibration_frame for a probability matrix"
            )
        if np.any((observed != 0) & (observed != 1)):
            raise ValueError("calibration_frame requires binary observed outcomes coded 0/1")
        if np.any(~np.isfinite(predicted)) or np.any((predicted < 0) | (predicted > 1)):
            raise ValueError("calibration_frame requires finite probabilities in [0, 1]")
        if n_bins > len(observed):
            raise ValueError(
                f"n_bins={n_bins} exceeds the {len(observed)} held-out predictions"
            )
        order = np.argsort(predicted, kind="stable")
        critical = NormalDist().inv_cdf(.5 + level / 2)
        rows = []
        for index, positions in enumerate(np.array_split(order, n_bins)):
            weight = weights[positions]
            total = float(weight.sum())
            predicted_mean = float(np.sum(weight * predicted[positions]) / total)
            observed_rate = float(np.sum(weight * observed[positions]) / total)
            effective_n = float(total**2 / np.sum(weight**2))
            if self.evaluation_task == "new_exchangeable_observation":
                standard_error = float(np.sqrt(observed_rate * (1 - observed_rate) / effective_n))
                z2_over_n = critical**2 / effective_n
                center = (observed_rate + z2_over_n / 2) / (1 + z2_over_n)
                half_width = critical / (1 + z2_over_n) * np.sqrt(
                    observed_rate * (1 - observed_rate) / effective_n
                    + critical**2 / (4 * effective_n**2)
                )
                lower = max(0.0, float(center - half_width))
                upper = min(1.0, float(center + half_width))
                uncertainty = "iid_kish_effective_n_wilson"
            else:
                standard_error = lower = upper = None
                uncertainty = "unavailable_for_declared_dependence"
            rows.append({
                "bin": index, "n": int(len(positions)),
                "weight_total": total, "effective_n": effective_n,
                "predicted_mean": predicted_mean, "observed_rate": observed_rate,
                "standard_error": standard_error, "lower": lower, "upper": upper,
                "level": level, "uncertainty": uncertainty,
            })
        return output_frame(rows, output)

    def _classification_arrays(self):
        if self.prediction_scale != "response":
            raise ValueError("classification evaluation requires response-scale probabilities")
        if not self.records:
            raise ValueError("row records were omitted; rerun with retain_arrays=True")
        observed = np.asarray([row["observed"] for row in self.records], dtype=np.float64)
        probability = np.asarray([row["prediction"] for row in self.records], dtype=np.float64)
        weights = np.asarray([row["evaluation_weight"] for row in self.records], dtype=np.float64)
        if probability.ndim != 1:
            raise ValueError(
                "binary classification evaluation requires one probability per row; "
                "use multiclass_metrics for a probability matrix"
            )
        if np.any((observed != 0) & (observed != 1)):
            raise ValueError("classification evaluation requires outcomes coded 0/1")
        if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
            raise ValueError("classification evaluation requires finite probabilities in [0, 1]")
        if np.sum(weights[observed == 1]) <= 0 or np.sum(weights[observed == 0]) <= 0:
            raise ValueError("classification evaluation requires positive weight in both classes")
        return observed, probability, weights

    def classification_metrics(self, threshold=.5) -> dict:
        """Weighted held-out probability and threshold metrics.

        Brier and natural-log loss evaluate probabilities. Accuracy,
        sensitivity, specificity, precision and F1 additionally use
        ``probability >= threshold``. Weighted confusion totals retain the
        scale of the evaluation weights; rates divide by their relevant total.
        """
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite value in [0, 1]")
        observed, probability, weights = self._classification_arrays()
        predicted = probability >= threshold
        positive = observed == 1
        negative = ~positive
        tp = float(np.sum(weights[predicted & positive]))
        fp = float(np.sum(weights[predicted & negative]))
        fn = float(np.sum(weights[~predicted & positive]))
        tn = float(np.sum(weights[~predicted & negative]))

        def ratio(numerator, denominator):
            return numerator / denominator if denominator > 0 else None

        sensitivity = ratio(tp, tp + fn)
        specificity = ratio(tn, tn + fp)
        precision = ratio(tp, tp + fp)
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if precision is not None and sensitivity is not None
              and precision + sensitivity > 0 else None)
        tiny = np.finfo(np.float64).tiny
        p = np.clip(probability, tiny, 1 - np.finfo(np.float64).eps)
        total = float(weights.sum())
        return {
            "threshold": threshold,
            "true_positive_weight": tp, "false_positive_weight": fp,
            "false_negative_weight": fn, "true_negative_weight": tn,
            "prevalence": float(np.sum(weights * observed) / total),
            "accuracy": (tp + tn) / total,
            "sensitivity": sensitivity, "specificity": specificity,
            "precision": precision, "f1": f1,
            "brier": float(np.sum(weights * (observed - probability) ** 2) / total),
            "log_loss": float(np.sum(
                weights * -(observed * np.log(p) + (1 - observed) * np.log1p(-p))
            ) / total),
        }

    def roc_frame(self, output="auto"):
        """Weighted empirical ROC curve and trapezoidal AUC.

        Ties enter at one threshold together. The AUC is the weighted
        probability that a randomly drawn positive has a higher score than a
        randomly drawn negative, with half credit for ties.
        """
        observed, probability, weights = self._classification_arrays()
        thresholds = np.r_[np.inf, np.unique(probability)[::-1]]
        positive_total = float(np.sum(weights[observed == 1]))
        negative_total = float(np.sum(weights[observed == 0]))
        rows = []
        for threshold in thresholds:
            predicted = probability >= threshold
            tp = float(np.sum(weights[predicted & (observed == 1)]))
            fp = float(np.sum(weights[predicted & (observed == 0)]))
            rows.append({
                "threshold": float(threshold), "true_positive_weight": tp,
                "false_positive_weight": fp, "sensitivity": tp / positive_total,
                "false_positive_rate": fp / negative_total,
            })
        x = np.asarray([row["false_positive_rate"] for row in rows])
        y = np.asarray([row["sensitivity"] for row in rows])
        auc = float(np.trapezoid(y, x))
        for row in rows:
            row["auc"] = auc
        return output_frame(rows, output)

    def precision_recall_frame(self, output="auto"):
        """Weighted precision–recall curve and step-integrated average precision."""
        observed, probability, weights = self._classification_arrays()
        thresholds = np.r_[np.inf, np.unique(probability)[::-1]]
        positive_total = float(np.sum(weights[observed == 1]))
        rows = []
        for threshold in thresholds:
            predicted = probability >= threshold
            tp = float(np.sum(weights[predicted & (observed == 1)]))
            fp = float(np.sum(weights[predicted & (observed == 0)]))
            rows.append({
                "threshold": float(threshold),
                "recall": tp / positive_total,
                "precision": tp / (tp + fp) if tp + fp > 0 else 1.0,
                "predicted_positive_weight": tp + fp,
            })
        recall = np.asarray([row["recall"] for row in rows])
        precision = np.asarray([row["precision"] for row in rows])
        average_precision = float(np.sum(np.diff(recall) * precision[1:]))
        for row in rows:
            row["average_precision"] = average_precision
        return output_frame(rows, output)

    def _auc_inputs(self):
        observed, probability, weights = self._classification_arrays()
        if self.independent_unit != "record":
            raise ValueError(
                "AUC uncertainty treats each held-out record as an independent unit, "
                f"but this result's independent unit is {self.independent_unit!r}; "
                "report the AUC without an interval, or evaluate at the record level"
            )
        return observed, probability, weights

    def auc_interval(self, level=.95) -> dict:
        """Weighted AUC with a DeLong standard error and a normal interval.

        The variance is the DeLong placement estimator with the evaluation
        weights read as frequency weights. The interval is
        ``AUC +/- z * SE`` and is clipped to [0, 1]. Records from different
        folds come from different fits, so the pooled placements treat them as
        one independent sample; the interval is an approximation for K-fold
        results and does not add fold-to-fold model variance.
        """
        from mboostrs._rust import auc_delong
        from mboostrs._validate import check_unit_interval

        level = check_unit_interval("AUC interval level", level)
        observed, probability, weights = self._auc_inputs()
        auc, covariance = auc_delong(observed, probability.reshape(-1, 1), weights)
        estimate = float(auc[0])
        standard_error = math.sqrt(max(float(covariance[0][0]), 0.0))
        z = NormalDist().inv_cdf(1 - (1 - level) / 2)
        return {
            "auc": estimate,
            "standard_error": standard_error,
            "lower": max(0.0, estimate - z * standard_error),
            "upper": min(1.0, estimate + z * standard_error),
            "level": level,
            "positive_weight": float(np.sum(weights[observed == 1])),
            "negative_weight": float(np.sum(weights[observed == 0])),
        }

    def auc_difference(self, other, level=.95) -> dict:
        """Paired AUC difference ``self - other`` with a DeLong interval.

        Both results must hold the same held-out records with the same weights
        (``comparison_issues(paired=True)`` is empty). The two AUCs share the
        records, so their covariance enters the variance of the difference.
        """
        from mboostrs._rust import auc_delong
        from mboostrs._validate import check_unit_interval

        if not isinstance(other, EvaluationResult):
            raise ValueError("other must be an EvaluationResult")
        issues = self.comparison_issues(other, paired=True)
        if issues:
            raise ValueError(
                "AUC difference needs paired results: " + ", ".join(issues)
            )
        level = check_unit_interval("AUC interval level", level)
        observed_a, probability_a, weights_a = self._auc_inputs()
        observed_b, probability_b, _weights_b = other._auc_inputs()

        def sort_key(row):
            return (repr(row["record_id"]), row["fold"])

        order_a = sorted(range(len(self.records)), key=lambda i: sort_key(self.records[i]))
        order_b = sorted(range(len(other.records)), key=lambda i: sort_key(other.records[i]))
        observed = observed_a[order_a]
        if not np.array_equal(observed, observed_b[order_b]):
            raise ValueError("paired AUC results disagree on the observed outcome of a record")
        scores = np.column_stack([probability_a[order_a], probability_b[order_b]])
        weights = weights_a[order_a]
        auc, covariance = auc_delong(observed, scores, weights)
        difference = float(auc[0] - auc[1])
        variance = float(covariance[0][0] + covariance[1][1] - 2 * covariance[0][1])
        standard_error = math.sqrt(max(variance, 0.0))
        z = NormalDist().inv_cdf(1 - (1 - level) / 2)
        return {
            "difference": difference,
            "standard_error": standard_error,
            "lower": max(-1.0, difference - z * standard_error),
            "upper": min(1.0, difference + z * standard_error),
            "level": level,
            "auc_self": float(auc[0]),
            "auc_other": float(auc[1]),
        }

    def threshold_frame(self, thresholds=101, *, utility=(0.0, -1.0, -1.0, 0.0),
                        output="auto"):
        """Weighted threshold metrics and average decision utility.

        ``utility`` contains values for ``(true negative, false positive,
        false negative, true positive)`` in one declared decision unit. The
        default charges one unit for either error and zero for either correct
        decision; it is illustrative, not a universal cost ratio.
        """
        observed, probability, weights = self._classification_arrays()
        if isinstance(thresholds, (int, np.integer)):
            if int(thresholds) < 2:
                raise ValueError("thresholds must contain at least two values")
            values = np.linspace(0, 1, int(thresholds))
        else:
            values = np.asarray(thresholds, dtype=np.float64)
            if values.ndim != 1 or values.size < 2:
                raise ValueError("thresholds must be a one-dimensional sequence of length >= 2")
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("thresholds must be finite values in [0, 1]")
        payoff = np.asarray(utility, dtype=np.float64)
        if payoff.shape != (4,) or not np.all(np.isfinite(payoff)):
            raise ValueError("utility must contain four finite values: TN, FP, FN, TP")
        total = float(weights.sum())
        rows = []
        for threshold in values:
            predicted = probability >= threshold
            positive = observed == 1
            tp = float(np.sum(weights[predicted & positive]))
            fp = float(np.sum(weights[predicted & ~positive]))
            fn = float(np.sum(weights[~predicted & positive]))
            tn = float(np.sum(weights[~predicted & ~positive]))
            rows.append({
                "threshold": float(threshold), "true_positive_weight": tp,
                "false_positive_weight": fp, "false_negative_weight": fn,
                "true_negative_weight": tn,
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "precision": tp / (tp + fp) if tp + fp > 0 else None,
                "average_utility": float(
                    (tn * payoff[0] + fp * payoff[1] + fn * payoff[2]
                     + tp * payoff[3]) / total
                ),
                "utility_tn": float(payoff[0]), "utility_fp": float(payoff[1]),
                "utility_fn": float(payoff[2]), "utility_tp": float(payoff[3]),
            })
        return output_frame(rows, output)

    def count_metrics(self, *, exposure=None, prediction_quantity="count") -> dict:
        """Weighted held-out count, rate, Poisson deviance, and log-score metrics.

        ``prediction_quantity="count"`` reads each retained prediction as the
        expected count for that row's supplied exposure and derives its rate by
        division. ``"rate"`` reads the prediction as an expected rate and
        multiplies by exposure before evaluating the count likelihood. The
        choice is explicit because a numerical prediction alone cannot reveal
        whether an exposure offset was present during fitting.
        """
        if self.prediction_scale != "response":
            raise ValueError("count_metrics requires response-scale predictions")
        if not self.records:
            raise ValueError("row records were omitted; rerun with retain_arrays=True")
        if prediction_quantity not in ("count", "rate"):
            raise ValueError("prediction_quantity must be 'count' or 'rate'")
        observed = np.asarray([row["observed"] for row in self.records], dtype=np.float64)
        predicted = np.asarray([row["prediction"] for row in self.records], dtype=np.float64)
        weights = np.asarray([row["evaluation_weight"] for row in self.records], dtype=np.float64)
        row_index = np.asarray([row["row"] for row in self.records], dtype=np.intp)
        if np.any(~np.isfinite(observed)) or np.any(observed < 0) or np.any(observed != np.floor(observed)):
            raise ValueError("count_metrics requires finite non-negative integer outcomes")
        if np.any(~np.isfinite(predicted)) or np.any(predicted < 0):
            raise ValueError("count_metrics requires finite non-negative response predictions")
        if exposure is None:
            if prediction_quantity == "rate":
                raise ValueError("prediction_quantity='rate' requires exposure")
            exposure_rows = np.ones(len(self.records), dtype=np.float64)
        else:
            exposure = np.asarray(exposure, dtype=np.float64)
            if exposure.shape != (self.n_observations,):
                raise ValueError(
                    f"exposure has shape {exposure.shape}, expected ({self.n_observations},)"
                )
            if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
                raise ValueError("exposure must contain finite positive values")
            exposure_rows = exposure[row_index]
        if prediction_quantity == "count":
            expected_count = predicted
            predicted_rate = predicted / exposure_rows
        else:
            predicted_rate = predicted
            expected_count = predicted * exposure_rows
        if np.any(expected_count <= 0):
            raise ValueError(
                "Poisson log score requires strictly positive expected counts; "
                "a zero expected count assigns infinite loss to a possible event"
            )
        observed_rate = observed / exposure_rows
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(observed == 0, 0.0,
                                 observed * np.log(observed / expected_count))
        deviance = 2 * (log_ratio - (observed - expected_count))
        log_factorial = np.asarray([math.lgamma(value + 1) for value in observed])
        negative_log_score = expected_count - observed * np.log(expected_count) + log_factorial
        total = float(weights.sum())
        exposure_total = float(np.sum(weights * exposure_rows))
        return {
            "prediction_quantity": prediction_quantity,
            "exposure_supplied": exposure is not None,
            "observed_count_total": float(np.sum(weights * observed)),
            "predicted_count_total": float(np.sum(weights * expected_count)),
            "observed_rate": float(np.sum(weights * observed) / exposure_total),
            "predicted_rate": float(np.sum(weights * expected_count) / exposure_total),
            "mean_absolute_count_error": float(
                np.sum(weights * np.abs(observed - expected_count)) / total
            ),
            "root_mean_squared_count_error": float(np.sqrt(
                np.sum(weights * (observed - expected_count) ** 2) / total
            )),
            "mean_absolute_rate_error": float(
                np.sum(weights * np.abs(observed_rate - predicted_rate)) / total
            ),
            "mean_poisson_deviance": float(np.sum(weights * deviance) / total),
            "mean_poisson_log_score": float(
                np.sum(weights * negative_log_score) / total
            ),
            "count_unit": "events per row exposure",
            "rate_unit": "events per exposure unit",
        }

    def _multiclass_arrays(self):
        if self.prediction_scale != "response":
            raise ValueError("multiclass evaluation requires response-scale probabilities")
        if not self.records:
            raise ValueError("row records were omitted; rerun with retain_arrays=True")
        observed = np.asarray([row["observed"] for row in self.records], dtype=np.float64)
        probability = np.asarray([row["prediction"] for row in self.records], dtype=np.float64)
        weights = np.asarray([row["evaluation_weight"] for row in self.records], dtype=np.float64)
        if probability.ndim != 2 or probability.shape[1] < 3:
            raise ValueError("multiclass evaluation requires an n by K probability matrix with K>=3")
        if self.class_labels is not None and len(self.class_labels) != probability.shape[1]:
            raise ValueError("class_labels length must match the multiclass probability columns")
        if np.any(~np.isfinite(probability)) or np.any(probability < 0):
            raise ValueError("multiclass probabilities must be finite and non-negative")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0):
            raise ValueError("each multiclass probability row must sum to one")
        if (np.any(~np.isfinite(observed)) or np.any(observed != np.floor(observed))
                or np.any(observed < 0) or np.any(observed >= probability.shape[1])):
            raise ValueError(
                f"multiclass outcomes must be integer codes 0 through {probability.shape[1] - 1}"
            )
        return observed.astype(np.intp), probability, weights

    def multiclass_metrics(self) -> dict:
        """Weighted held-out categorical log loss, Brier score, and class summaries."""
        observed, probability, weights = self._multiclass_arrays()
        n_classes = probability.shape[1]
        predicted = np.argmax(probability, axis=1)
        total = float(weights.sum())
        chosen = np.maximum(
            probability[np.arange(len(observed)), observed],
            np.finfo(np.float64).tiny,
        )
        one_hot = np.eye(n_classes)[observed]
        per_class = []
        for label in range(n_classes):
            actual = observed == label
            called = predicted == label
            tp = float(np.sum(weights[actual & called]))
            support = float(np.sum(weights[actual]))
            called_weight = float(np.sum(weights[called]))
            per_class.append({
                "class": label,
                "class_label": (self.class_labels[label]
                                if self.class_labels is not None else label),
                "support_weight": support,
                "predicted_weight": called_weight,
                "recall": tp / support if support > 0 else None,
                "precision": tp / called_weight if called_weight > 0 else None,
            })
        recalls = [row["recall"] for row in per_class if row["recall"] is not None]
        return {
            "n_classes": n_classes,
            "top_class_accuracy": float(np.sum(weights * (predicted == observed)) / total),
            "categorical_log_loss": float(np.sum(weights * -np.log(chosen)) / total),
            "multiclass_brier": float(
                np.sum(weights * np.sum((one_hot - probability) ** 2, axis=1)) / total
            ),
            "macro_recall": float(np.mean(recalls)),
            "per_class": tuple(per_class),
        }

    def multiclass_confusion_frame(self, output="auto"):
        """Weighted observed-by-predicted confusion cells for every class pair."""
        observed, probability, weights = self._multiclass_arrays()
        predicted = np.argmax(probability, axis=1)
        n_classes = probability.shape[1]
        rows = []
        for actual in range(n_classes):
            for called in range(n_classes):
                rows.append({
                    "observed_class": actual, "predicted_class": called,
                    "observed_label": (self.class_labels[actual]
                                       if self.class_labels is not None else actual),
                    "predicted_label": (self.class_labels[called]
                                        if self.class_labels is not None else called),
                    "weight": float(np.sum(
                        weights[(observed == actual) & (predicted == called)]
                    )),
                })
        return output_frame(rows, output)

    def multiclass_calibration_frame(self, *, n_bins=10, output="auto"):
        """One-vs-rest reliability bins for every retained probability column.

        The table is descriptive and reports counts and weight totals. It does
        not attach row-IID intervals to classwise bins because simultaneous
        multiclass and dependence-aware uncertainty require a separate design.
        """
        from mboostrs._validate import check_int_at_least

        observed, probability, weights = self._multiclass_arrays()
        n_bins = check_int_at_least("n_bins", n_bins, 2, allow_none=False)
        if n_bins > len(observed):
            raise ValueError(
                f"n_bins={n_bins} exceeds the {len(observed)} held-out predictions"
            )
        rows = []
        for label in range(probability.shape[1]):
            order = np.argsort(probability[:, label], kind="stable")
            for index, positions in enumerate(np.array_split(order, n_bins)):
                weight = weights[positions]
                total = float(weight.sum())
                rows.append({
                    "class": label,
                    "class_label": (self.class_labels[label]
                                    if self.class_labels is not None else label),
                    "bin": index, "n": int(len(positions)),
                    "weight_total": total,
                    "predicted_mean": float(np.sum(
                        weight * probability[positions, label]
                    ) / total),
                    "observed_rate": float(np.sum(
                        weight * (observed[positions] == label)
                    ) / total),
                    "uncertainty": "not_implemented_for_multiclass_bins",
                })
        return output_frame(rows, output)

    @property
    def observation_weighted_score(self) -> float:
        """The held-out loss averaged over observations rather than folds.

        `score` is the equal-fold mean: every fold counts once, whatever its
        size. This weights each fold's mean by its evaluation weight total,
        which is the mean loss per held-out observation (per unit of weight).
        The two agree when folds carry equal weight; they differ for grouped
        folds of unequal size and for rolling-origin designs, where the
        equal-fold mean gives a small fold's score the same influence as a
        large one's.
        """
        totals = np.array([row["evaluation_weight_total"] for row in self.folds], dtype=float)
        return float(np.sum(self.fold_scores * totals) / np.sum(totals))

    @property
    def failed_folds(self) -> tuple[dict, ...]:
        """The fold records of folds that could not be fitted or scored.

        Each carries ``status == "failed"``, the ``error`` that stopped it and
        its sizes. A result with any failed fold has a NaN ``score``.
        """
        return tuple(row for row in self.folds if row.get("status") == "failed")

    def describe(self) -> str:
        """A compact statement of the evaluated quantity and population."""
        fingerprint = f"Data fingerprint: {self.data_fingerprint[:12]}…"
        if self.failed_folds:
            listed = "; ".join(f"fold {row['fold']}: {row['error']}"
                               for row in self.failed_folds)
            return (
                f"{self.score_name} is undefined: {len(self.failed_folds)} of "
                f"{len(self.fold_scores)} folds failed ({listed}). The failed folds "
                f"are listed by fold_frame(). {fingerprint}"
            )
        exclusions = (
            f", {len(self.excluded_record_ids)} records excluded"
            if self.excluded_record_ids else ""
        )
        pooled = self.observation_weighted_score
        pooled_note = (
            "" if abs(pooled - self.score) <= 1e-12 * max(1.0, abs(self.score))
            else f" (observation-weighted mean {pooled:.6g})")
        scale = (self.prediction_scale if self.prediction_scale.endswith("scale")
                 else f"{self.prediction_scale} scale")
        return (
            f"{self.score_name}={self.score:.6g} on the {scale}; "
            f"{len(self.fold_scores)} folds aggregated by equal-fold mean{pooled_note}, "
            f"{self.n_evaluations} held-out predictions for "
            f"{self.evaluation_task.replace('_', ' ')} from {self.fit_count} fits"
            f"{exclusions}. "
            f"{fingerprint}"
        )

    def comparison_issues(self, other, *, paired=True) -> tuple[str, ...]:
        """Reasons the two results do not estimate a common scoring target.

        Formula and family may differ when both results use the same external
        score, such as response-scale MSE. Family-loss comparisons additionally
        require the same family because their losses need not share units.
        ``paired=True`` also requires exact record, fold, and evaluation-weight
        identity. Seed or fold count alone cannot establish that pairing.
        """
        if not isinstance(other, EvaluationResult):
            return ("other is not an EvaluationResult",)
        issues = []
        for field, label in (
            ("data_fingerprint", "data fingerprint"),
            ("response", "response"),
            ("score_name", "score definition"),
            ("score_parameter", "score parameter"),
            ("prediction_scale", "prediction scale"),
            ("loss_units", "loss units"),
            ("aggregation", "fold aggregation"),
            ("evaluation_task", "evaluation task"),
            ("class_labels", "class labels"),
        ):
            if getattr(self, field) != getattr(other, field):
                issues.append(label)
        if self.failed_folds or other.failed_folds:
            issues.append("failed folds")
        if self.score_name == other.score_name == "family" and self.family != other.family:
            issues.append("family-loss identity")
        if (self.weights_fingerprint and other.weights_fingerprint
                and self.weights_fingerprint != other.weights_fingerprint):
            issues.append("evaluation weights")
        if (self.data_identity and other.data_identity
                and self.data_identity != other.data_identity):
            issues.append("data identity (observation IDs, response values or weights)")
        if paired:
            if not self.records or not other.records:
                issues.append("retained record-fold membership")
            else:
                left = sorted(
                    (repr(row["record_id"]), row["fold"], row["evaluation_weight"])
                    for row in self.records
                )
                right = sorted(
                    (repr(row["record_id"]), row["fold"], row["evaluation_weight"])
                    for row in other.records
                )
                if left != right:
                    issues.append("record-fold-weight pairing")
        return tuple(issues)

    def assert_comparable(self, other, *, paired=True) -> None:
        """Raise when a score comparison would change its target or pairing."""
        issues = self.comparison_issues(other, paired=paired)
        if issues:
            raise ValueError(
                "evaluation results are not comparable: " + ", ".join(issues)
            )
