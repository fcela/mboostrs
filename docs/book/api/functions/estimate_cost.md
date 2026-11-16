# `estimate_cost`

Defined in `mboostrs.cost`; exported as `mboostrs.estimate_cost`.


*Function*

```python
estimate_cost(call: str, *args, **kwargs) -> CostEstimate
```

Count the fits of a call and time one fit, without running the call.

**Parameters**

- `call` (str): One of `"gamboost"`, `"gamboostLSS"`, `"tune_mstop"`,
  `"compare_specifications"`, `"term_report"`, `"stabsel"`,
  `"conformal_interval"`, `"aipw"`, `"effect_model"`,
  `"ci_test"`, `"iv_effect"`, `"confounding_benchmark"` or
  `"discover"`.
- `*args, **kwargs`: The arguments the call would take, exactly as they would be passed to
  it. For `"stabsel"` and `"conformal_interval"` the first argument
  is the fitted model.

**Returns**

CostEstimate
    The fit count, the time of one fit, the product of the two, and a
    one-line explanation of the count. `stages` holds the count of each
    stage of a staged call (`discover`).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

`"gamboost"` and `"gamboostLSS"` are one fit, so the estimate runs the
call once and reports its time. Every other call is timed on one fit of its
own specification: the same formula, family and step control the call
would use. A call that tunes its step count (`mstop="cv"`) cannot know
that count before it runs, so the timed fit uses 100 steps; a tuning that
settles above 100 steps makes the estimate low.

The causal calls (`aipw`, `effect_model`, `ci_test`, `iv_effect`,
`confounding_benchmark`) count a tuned nuisance model as the fits of
`mboostrs.explore.tune_mstop` (its inner folds plus one full-data
fit) and an untuned one as a single fit. The counts follow the arguments
as the call binds them, with defaults read from the call's signature.
Two counts are upper bounds because the call decides after fitting:
`aipw(outcome_model="auto")` counts separate outcome models for the two
arms, and `effect_model(terms="auto")` counts the final smooth fit.
With `outcome_model="cv"` every candidate outcome model is counted.

`"discover"` counts the fits each stage makes on its first pass. A stage
that repeats or branches on the data (rounds of backward elimination,
interaction candidates, spread tests, the tuning of a spread model) adds
more. `CostEstimate.stages` gives the first-pass count of each stage, and
`DiscoveryResult.stage_seconds` the measured seconds after a run.

**Examples**

>>> import numpy as np, mboostrs as mb
>>> rng = np.random.default_rng(0)
>>> x = rng.uniform(-2, 2, 300)
>>> data = {"x": x, "y": np.sin(x) + rng.normal(scale=0.3, size=300)}
>>> cost = mb.estimate_cost("tune_mstop", "y ~ bbs(x, df=4)", data, folds=5)
>>> cost.fits
6

Source: [`python/mboostrs/cost.py` lines 45-112](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cost.py#L45-L112)
