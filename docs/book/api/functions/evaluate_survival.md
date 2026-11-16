# `evaluate_survival`

Defined in `mboostrs.survival_evaluation`; exported as `mboostrs.evaluate_survival`.


*Function*

```python
evaluate_survival(
    model,
    data,
    horizons,
    data_role,
    weights=None,
    censoring_floor=0.05,
    on_low_censoring='error'
)
```

Brier score and time-dependent AUC of a CoxPH model at fixed horizons.

`data` supplies the predictors and the `Surv(time, status)` columns the
model was fitted with. The predicted survival `S(t | x)` comes from the
Breslow baseline stored on `model` and the linear predictor of `data`.
The censoring survival `G(t)` is a Kaplan-Meier estimate on the rows of
`data`.

`data_role` must be `"external"` or `"training"`. External means the
outcomes were not used to fit, tune or choose the model. This is a caller
declaration recorded in the result, not something the fitted object can
check. Training results describe fit and do not estimate prediction on new
subjects.

`censoring_floor` is the smallest `G(t)` at which the IPCW weights are
considered stable. Below it, `on_low_censoring="error"` (default) raises
and `"warn"` issues a warning and records `below_censoring_floor` in
the result. Rows with `G(t) = 0` raise in either mode, because their
weights are infinite.

**Parameters**

- `model` (GLMBoostModel or GAMBoostModel): A fit with `family=CoxPH()` and a `Surv(time, status)` response.
- `data` (dict or DataFrame): Rows to evaluate. Event status must be coded 0 (censored) or 1 (event).
- `horizons` (number or array-like): Non-negative times on the response scale.
- `data_role` ({"external", "training"})
- `weights` (array-like, optional): Non-negative evaluation weights, one per row of `data`.
- `censoring_floor` (float, default 0.05): Threshold on `G(t)` in `[0, 1)`.
- `on_low_censoring` ({"error", "warn"}, default "error")

**Returns**

SurvivalEvaluationResult

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/survival_evaluation.py` lines 143-288](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/survival_evaluation.py#L143-L288)
