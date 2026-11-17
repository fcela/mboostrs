# `SurvivalEvaluationResult`

Defined in `mboostrs.survival_evaluation`; exported as `mboostrs.SurvivalEvaluationResult`.


*Class*

```python
SurvivalEvaluationResult(
    family: str,
    response: tuple,
    data_role: str,
    data_fingerprint: str,
    horizons: tuple,
    n_observations: int,
    total_weight: float,
    censoring_floor: float,
    on_low_censoring: str,
    time: numpy.ndarray,
    event: numpy.ndarray,
    weights: numpy.ndarray,
    linear_predictor: numpy.ndarray,
    survival: numpy.ndarray,
    horizon_rows: tuple[dict, ...],
    assumptions: tuple[str, ...]
) -> None
```

Fixed-horizon Brier score and time-dependent AUC of a CoxPH model.

One row of `horizon_frame` per horizon. The Brier score measures
absolute predicted survival `S(t | x)`; the AUC measures how well the
linear predictor ranks subjects who have an event by `t` above subjects
still event-free at `t`. Both use inverse-probability-of-censoring
weights, so a row censored before `t` is down-weighted rather than
dropped.

Source: [`python/mboostrs/survival_evaluation.py` lines 26-120](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/survival_evaluation.py#L26-L120)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | A JSON-safe record; per-subject arrays only on request. |
| `horizon_frame` | Method | One row per horizon: IPCW Brier score, AUC, censoring survival and counts. |
| `survival_frame` | Method | One row per subject and horizon with the predicted ``S(t \| x)``. |
| `describe` | Method | State the horizons, scores, censoring floor and data role. |

(survivalevaluationresult-to_dict)=
## `SurvivalEvaluationResult.to_dict`

*Method* of `SurvivalEvaluationResult`

```python
SurvivalEvaluationResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; per-subject arrays only on request.

Source: [`python/mboostrs/survival_evaluation.py` lines 74-81](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/survival_evaluation.py#L74-L81)

(survivalevaluationresult-horizon_frame)=
## `SurvivalEvaluationResult.horizon_frame`

*Method* of `SurvivalEvaluationResult`

```python
SurvivalEvaluationResult.horizon_frame(output='auto')
```

One row per horizon: IPCW Brier score, AUC, censoring survival and counts.

Columns: `brier` (IPCW Brier score, lower is better), `auc`
(cumulative/dynamic IPCW AUC, 0.5 is chance), `censoring_survival`
(Kaplan-Meier `G(t)` on these rows), `below_censoring_floor`,
`case_weight` and `control_weight` (the weight totals behind the
AUC), `events_by_horizon` and `at_risk_at_horizon` (weighted counts),
and `predicted_survival_mean` (weighted mean of `S(t | x)`).

Source: [`python/mboostrs/survival_evaluation.py` lines 83-93](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/survival_evaluation.py#L83-L93)

(survivalevaluationresult-survival_frame)=
## `SurvivalEvaluationResult.survival_frame`

*Method* of `SurvivalEvaluationResult`

```python
SurvivalEvaluationResult.survival_frame(output='auto')
```

One row per subject and horizon with the predicted `S(t | x)`.

Source: [`python/mboostrs/survival_evaluation.py` lines 95-107](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/survival_evaluation.py#L95-L107)

(survivalevaluationresult-describe)=
## `SurvivalEvaluationResult.describe`

*Method* of `SurvivalEvaluationResult`

```python
SurvivalEvaluationResult.describe() -> str
```

State the horizons, scores, censoring floor and data role.

Source: [`python/mboostrs/survival_evaluation.py` lines 109-120](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/survival_evaluation.py#L109-L120)
