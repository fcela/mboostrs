# `confounding_sensitivity`

Defined in `mboostrs.causal`; exported as `mboostrs.confounding_sensitivity`.


*Function*

```python
confounding_sensitivity(
    result: mboostrs.causal.AIPWResult | None = None,
    outcome_effects,
    imbalances,
    estimate=None,
    se=None,
    scores=None,
    level=0.95,
    output='auto'
)
```

Bias grid for one omitted baseline confounder under an additive model.

`outcome_effects` gives the change in the outcome caused by a one-unit
increase in the omitted confounder, conditional on measured covariates.
`imbalances` gives its adjusted treated-minus-control mean difference in
the same confounder units. Their product is the assumed omitted-variable
bias, so `adjusted_estimate = estimate - outcome_effect * imbalance`.

The estimate comes from one of two inputs:

* `result`, an `AIPWResult`. Its estimate and, when available,
  its interval (the design-aware one, including cluster-robust intervals)
  are shifted by the bias.
* `estimate`, a plain number from any estimator, with `se`, its
  standard error. The interval is `estimate +/- z * se` at `level`
  with a normal critical value. Without `se` the rows carry no interval.
* `scores`, a vector of per-row scores whose mean is the estimate. The
  standard error is the IID one, `sd(scores, ddof=1) / sqrt(n)`, unless
  `se` is also given. When `estimate` is given with `scores`, the
  estimate is used as given.

Pass exactly one of `result` and `estimate`/`se`/`scores`.

This is a transparent bias function, not identification or a calibrated
bound. Choose ranges from subject-matter knowledge or benchmark them to
measured covariates, and report the units. The confidence interval, when
available, is shifted by the assumed bias and does not include uncertainty
in the sensitivity parameters.

`output` chooses the return type, as for the other frames of the
package: `"auto"` (the default) is a Polars frame when Polars is
installed and a pandas frame otherwise, `"polars"` and `"pandas"` force
one, and `"records"` returns a list of dicts holding plain Python
numbers, booleans and `None`, which `json.dumps` accepts. The rows
are one per (outcome effect, imbalance) pair.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `TypeError`

Source: [`python/mboostrs/causal.py` lines 5843-5938](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L5843-L5938)
