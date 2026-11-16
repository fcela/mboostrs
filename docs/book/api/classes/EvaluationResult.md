# `EvaluationResult`

Defined in `mboostrs.evaluation`; exported as `mboostrs.EvaluationResult`.


*Class*

```python
EvaluationResult(
    score: float,
    fold_scores: numpy.ndarray,
    formula: str,
    response: str,
    family: str,
    score_name: str,
    score_parameter: float | None,
    prediction_scale: str,
    loss_units: str,
    aggregation: str,
    fold_scheme: str,
    evaluation_task: str,
    independent_unit: str,
    n_independent_units: int,
    seed: int,
    tuning: str,
    data_fingerprint: str,
    fingerprint_scope: str,
    n_observations: int,
    n_evaluations: int,
    fit_count: int,
    excluded_record_ids: tuple,
    records: tuple[dict, ...],
    folds: tuple[dict, ...],
    warnings: tuple[str, ...] = (),
    class_labels: tuple | None = None,
    build_id: str = '',
    response_transform: str = 'identity',
    weights_fingerprint: str = '',
    data_identity: str = ''
) -> None
```

Held-out predictions, scoring semantics and provenance for one specification.

Source: [`python/mboostrs/evaluation.py` lines 49-807](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L49-L807)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`aggregate`](#evaluationresult-aggregate) | Property | The equal-fold mean score, the number `cv_risk` returns without |
| [`to_dict`](#evaluationresult-to_dict) | Method | A JSON-safe record; fold scores and row evidence only on request. |
| [`to_frame`](#evaluationresult-to_frame) | Method | One row per held-out prediction, or an empty frame if arrays were omitted. |
| [`fold_frame`](#evaluationresult-fold_frame) | Method | One row per fold with its score, sample sizes, weights and stopping point. |
| [`residual_frame`](#evaluationresult-residual_frame) | Method | Held-out response residuals aligned with record and fold identity. |
| [`calibration_frame`](#evaluationresult-calibration_frame) | Method | Equal-count reliability bins from held-out probability predictions. |
| [`classification_metrics`](#evaluationresult-classification_metrics) | Method | Weighted held-out probability and threshold metrics. |
| [`roc_frame`](#evaluationresult-roc_frame) | Method | Weighted empirical ROC curve and trapezoidal AUC. |
| [`precision_recall_frame`](#evaluationresult-precision_recall_frame) | Method | Weighted precision–recall curve and step-integrated average precision. |
| [`auc_interval`](#evaluationresult-auc_interval) | Method | Weighted AUC with a DeLong standard error and a normal interval. |
| [`auc_difference`](#evaluationresult-auc_difference) | Method | Paired AUC difference ``self - other`` with a DeLong interval. |
| [`threshold_frame`](#evaluationresult-threshold_frame) | Method | Weighted threshold metrics and average decision utility. |
| [`count_metrics`](#evaluationresult-count_metrics) | Method | Weighted held-out count, rate, Poisson deviance, and log-score metrics. |
| [`multiclass_metrics`](#evaluationresult-multiclass_metrics) | Method | Weighted held-out categorical log loss, Brier score, and class summaries. |
| [`multiclass_confusion_frame`](#evaluationresult-multiclass_confusion_frame) | Method | Weighted observed-by-predicted confusion cells for every class pair. |
| [`multiclass_calibration_frame`](#evaluationresult-multiclass_calibration_frame) | Method | One-vs-rest reliability bins for every retained probability column. |
| [`observation_weighted_score`](#evaluationresult-observation_weighted_score) | Property | The held-out loss averaged over observations rather than folds. |
| [`failed_folds`](#evaluationresult-failed_folds) | Property | The fold records of folds that could not be fitted or scored. |
| [`describe`](#evaluationresult-describe) | Method | A compact statement of the evaluated quantity and population. |
| [`comparison_issues`](#evaluationresult-comparison_issues) | Method | Reasons the two results do not estimate a common scoring target. |
| [`assert_comparable`](#evaluationresult-assert_comparable) | Method | Raise when a score comparison would change its target or pairing. |

(evaluationresult-aggregate)=
## `EvaluationResult.aggregate`

*Property* of `EvaluationResult`

```python
EvaluationResult.aggregate  # -> float
```

The equal-fold mean score, the number `cv_risk` returns without
`return_result=True`. The same value as `score` and `float(result)`.

Source: [`python/mboostrs/evaluation.py` lines 127-131](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L127-L131)

(evaluationresult-to_dict)=
## `EvaluationResult.to_dict`

*Method* of `EvaluationResult`

```python
EvaluationResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; fold scores and row evidence only on request.

Source: [`python/mboostrs/evaluation.py` lines 143-149](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L143-L149)

(evaluationresult-to_frame)=
## `EvaluationResult.to_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.to_frame(output='auto')
```

One row per held-out prediction, or an empty frame if arrays were omitted.

Source: [`python/mboostrs/evaluation.py` lines 151-153](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L151-L153)

(evaluationresult-fold_frame)=
## `EvaluationResult.fold_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.fold_frame(output='auto')
```

One row per fold with its score, sample sizes, weights and stopping point.

Source: [`python/mboostrs/evaluation.py` lines 155-157](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L155-L157)

(evaluationresult-residual_frame)=
## `EvaluationResult.residual_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.residual_frame(output='auto')
```

Held-out response residuals aligned with record and fold identity.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 159-178](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L159-L178)

(evaluationresult-calibration_frame)=
## `EvaluationResult.calibration_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.calibration_frame(*, n_bins=10, level=0.95, output='auto')
```

Equal-count reliability bins from held-out probability predictions.

The interval describes sampling variation of the observed binary rate
within a bin using a Wilson score interval with Kish effective sample
size. Treating that effective size as binomial is an IID approximation,
not uncertainty in the fitted probabilities. Grouped and temporal
results retain counts and rates but return no row-IID interval.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 180-243](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L180-L243)

(evaluationresult-classification_metrics)=
## `EvaluationResult.classification_metrics`

*Method* of `EvaluationResult`

```python
EvaluationResult.classification_metrics(threshold=0.5) -> dict
```

Weighted held-out probability and threshold metrics.

Brier and natural-log loss evaluate probabilities. Accuracy,
sensitivity, specificity, precision and F1 additionally use
`probability >= threshold`. Weighted confusion totals retain the
scale of the evaluation weights; rates divide by their relevant total.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 266-310](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L266-L310)

(evaluationresult-roc_frame)=
## `EvaluationResult.roc_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.roc_frame(output='auto')
```

Weighted empirical ROC curve and trapezoidal AUC.

Ties enter at one threshold together. The AUC is the weighted
probability that a randomly drawn positive has a higher score than a
randomly drawn negative, with half credit for ties.

Source: [`python/mboostrs/evaluation.py` lines 312-338](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L312-L338)

(evaluationresult-precision_recall_frame)=
## `EvaluationResult.precision_recall_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.precision_recall_frame(output='auto')
```

Weighted precision–recall curve and step-integrated average precision.

Source: [`python/mboostrs/evaluation.py` lines 340-361](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L340-L361)

(evaluationresult-auc_interval)=
## `EvaluationResult.auc_interval`

*Method* of `EvaluationResult`

```python
EvaluationResult.auc_interval(level=0.95) -> dict
```

Weighted AUC with a DeLong standard error and a normal interval.

The variance is the DeLong placement estimator with the evaluation
weights read as frequency weights. The interval is
`AUC +/- z * SE` and is clipped to [0, 1]. Records from different
folds come from different fits, so the pooled placements treat them as
one independent sample; the interval is an approximation for K-fold
results and does not add fold-to-fold model variance.

Source: [`python/mboostrs/evaluation.py` lines 373-400](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L373-L400)

(evaluationresult-auc_difference)=
## `EvaluationResult.auc_difference`

*Method* of `EvaluationResult`

```python
EvaluationResult.auc_difference(other, level=0.95) -> dict
```

Paired AUC difference `self - other` with a DeLong interval.

Both results must hold the same held-out records with the same weights
(`comparison_issues(paired=True)` is empty). The two AUCs share the
records, so their covariance enters the variance of the difference.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 402-446](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L402-L446)

(evaluationresult-threshold_frame)=
## `EvaluationResult.threshold_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.threshold_frame(
    thresholds=101,
    utility=(0.0, -1.0, -1.0, 0.0),
    output='auto'
)
```

Weighted threshold metrics and average decision utility.

`utility` contains values for ``(true negative, false positive,
false negative, true positive)`` in one declared decision unit. The
default charges one unit for either error and zero for either correct
decision; it is illustrative, not a universal cost ratio.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 448-494](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L448-L494)

(evaluationresult-count_metrics)=
## `EvaluationResult.count_metrics`

*Method* of `EvaluationResult`

```python
EvaluationResult.count_metrics(*, exposure=None, prediction_quantity='count') -> dict
```

Weighted held-out count, rate, Poisson deviance, and log-score metrics.

`prediction_quantity="count"` reads each retained prediction as the
expected count for that row's supplied exposure and derives its rate by
division. `"rate"` reads the prediction as an expected rate and
multiplies by exposure before evaluating the count likelihood. The
choice is explicit because a numerical prediction alone cannot reveal
whether an exposure offset was present during fitting.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 496-575](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L496-L575)

(evaluationresult-multiclass_metrics)=
## `EvaluationResult.multiclass_metrics`

*Method* of `EvaluationResult`

```python
EvaluationResult.multiclass_metrics() -> dict
```

Weighted held-out categorical log loss, Brier score, and class summaries.

Source: [`python/mboostrs/evaluation.py` lines 600-637](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L600-L637)

(evaluationresult-multiclass_confusion_frame)=
## `EvaluationResult.multiclass_confusion_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.multiclass_confusion_frame(output='auto')
```

Weighted observed-by-predicted confusion cells for every class pair.

Source: [`python/mboostrs/evaluation.py` lines 639-657](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L639-L657)

(evaluationresult-multiclass_calibration_frame)=
## `EvaluationResult.multiclass_calibration_frame`

*Method* of `EvaluationResult`

```python
EvaluationResult.multiclass_calibration_frame(*, n_bins=10, output='auto')
```

One-vs-rest reliability bins for every retained probability column.

The table is descriptive and reports counts and weight totals. It does
not attach row-IID intervals to classwise bins because simultaneous
multiclass and dependence-aware uncertainty require a separate design.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 659-694](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L659-L694)

(evaluationresult-observation_weighted_score)=
## `EvaluationResult.observation_weighted_score`

*Property* of `EvaluationResult`

```python
EvaluationResult.observation_weighted_score  # -> float
```

The held-out loss averaged over observations rather than folds.

`score` is the equal-fold mean: every fold counts once, whatever its
size. This weights each fold's mean by its evaluation weight total,
which is the mean loss per held-out observation (per unit of weight).
The two agree when folds carry equal weight; they differ for grouped
folds of unequal size and for rolling-origin designs, where the
equal-fold mean gives a small fold's score the same influence as a
large one's.

Source: [`python/mboostrs/evaluation.py` lines 696-709](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L696-L709)

(evaluationresult-failed_folds)=
## `EvaluationResult.failed_folds`

*Property* of `EvaluationResult`

```python
EvaluationResult.failed_folds  # -> tuple[dict, ...]
```

The fold records of folds that could not be fitted or scored.

Each carries `status == "failed"`, the `error` that stopped it and
its sizes. A result with any failed fold has a NaN `score`.

Source: [`python/mboostrs/evaluation.py` lines 711-718](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L711-L718)

(evaluationresult-describe)=
## `EvaluationResult.describe`

*Method* of `EvaluationResult`

```python
EvaluationResult.describe() -> str
```

A compact statement of the evaluated quantity and population.

Source: [`python/mboostrs/evaluation.py` lines 720-748](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L720-L748)

(evaluationresult-comparison_issues)=
## `EvaluationResult.comparison_issues`

*Method* of `EvaluationResult`

```python
EvaluationResult.comparison_issues(other, *, paired=True) -> tuple[str, ...]
```

Reasons the two results do not estimate a common scoring target.

Formula and family may differ when both results use the same external
score, such as response-scale MSE. Family-loss comparisons additionally
require the same family because their losses need not share units.
`paired=True` also requires exact record, fold, and evaluation-weight
identity. Seed or fold count alone cannot establish that pairing.

Source: [`python/mboostrs/evaluation.py` lines 750-799](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L750-L799)

(evaluationresult-assert_comparable)=
## `EvaluationResult.assert_comparable`

*Method* of `EvaluationResult`

```python
EvaluationResult.assert_comparable(other, *, paired=True) -> None
```

Raise when a score comparison would change its target or pairing.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/evaluation.py` lines 801-807](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/evaluation.py#L801-L807)
