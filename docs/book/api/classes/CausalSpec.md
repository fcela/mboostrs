# `CausalSpec`

Defined in `mboostrs.causal`; exported as `mboostrs.CausalSpec`.


*Class*

```python
CausalSpec(
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    estimand: str = 'ate',
    outcome_type: str = 'continuous',
    target_population: str = 'sample population',
    unit: str = 'observation',
    time_zero: str | None = None,
    follow_up: str | None = None,
    treatment_contrast: tuple[object, object] = (0, 1),
    post_treatment: tuple[str, ...] = (),
    adjustment_rationale: str = '',
    assumptions: tuple[str, ...] = ('consistency: each observed outcome equals the potential outcome under the observed treatment', 'conditional exchangeability: potential outcomes are independent of treatment given the baseline covariates', 'positivity: both treatment levels have positive probability across the target covariate support', "no interference: one unit's treatment does not change another unit's outcome"),
    propensity_clip: tuple[float, float] | None = (0.01, 0.99),
    outcome_missing: str = 'error',
    overlap_bounds: tuple[float, float] = (0.02, 0.98)
) -> None
```

A binary-treatment causal question and its adjustment design.

Source: [`python/mboostrs/causal.py` lines 348-438](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L348-L438)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#causalspec-to_dict) | Method | A JSON-safe record; every field is a small scalar or tuple. |
| [`describe`](#causalspec-describe) | Method | The intervention and identification claim in plain text. |

(causalspec-to_dict)=
## `CausalSpec.to_dict`

*Method* of `CausalSpec`

```python
CausalSpec.to_dict(*, include_arrays=False)
```

A JSON-safe record; every field is a small scalar or tuple.

Source: [`python/mboostrs/causal.py` lines 417-423](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L417-L423)

(causalspec-describe)=
## `CausalSpec.describe`

*Method* of `CausalSpec`

```python
CausalSpec.describe() -> str
```

The intervention and identification claim in plain text.

Source: [`python/mboostrs/causal.py` lines 425-438](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L425-L438)
