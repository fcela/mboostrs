# `IVResult`

Defined in `mboostrs.causal`; exported as `mboostrs.IVResult`.


*Class*

```python
IVResult(
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: tuple,
    estimate: float,
    standard_error: float,
    level: float,
    interval: tuple,
    first_stage_r2: float,
    first_stage_f: float,
    weak_threshold: float,
    weak_instrument: bool,
    n: int,
    folds: int,
    seed: int,
    nuisance: dict,
    influence: numpy.ndarray,
    outcome_residual: numpy.ndarray,
    treatment_residual: numpy.ndarray,
    instrument_residual: numpy.ndarray,
    fold: numpy.ndarray,
    assumptions: tuple = ('relevance: the instrument moves the probability of treatment after the covariates', 'independence: the instrument shares no unmeasured cause with the outcome, given the covariates', 'exclusion: the instrument affects the outcome only through the treatment, given the covariates', 'monotonicity: no unit is moved out of treatment by the instrument, so the units it moves all move the same way', 'consistency: each observed outcome equals the potential outcome under the observed treatment and instrument value')
) -> None
```

A cross-fitted instrumental-variable estimate, its interval and its first-stage diagnostics.

Source: [`python/mboostrs/causal.py` lines 5832-5937](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5832-L5937)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | A JSON-safe record; the per-row residuals and influence values only on request. |
| `to_frame` | Method | One row per observation: its fold, its three residuals and its influence value. |
| `describe` | Method | The estimand, the estimate with its interval, the first stage and the assumptions. |

(ivresult-to_dict)=
## `IVResult.to_dict`

*Method* of `IVResult`

```python
IVResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the per-row residuals and influence values only on request.

Source: [`python/mboostrs/causal.py` lines 5876-5884](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5876-L5884)

(ivresult-to_frame)=
## `IVResult.to_frame`

*Method* of `IVResult`

```python
IVResult.to_frame(output='auto')
```

One row per observation: its fold, its three residuals and its influence value.

Source: [`python/mboostrs/causal.py` lines 5886-5898](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5886-L5898)

(ivresult-describe)=
## `IVResult.describe`

*Method* of `IVResult`

```python
IVResult.describe() -> str
```

The estimand, the estimate with its interval, the first stage and the assumptions.

Source: [`python/mboostrs/causal.py` lines 5900-5937](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5900-L5937)
