# `causal_trap_report`

Defined in `mboostrs.causal`; exported as `mboostrs.causal_trap_report`.


*Function*

```python
causal_trap_report(
    frame,
    spec,
    min_per_arm=5,
    min_arm_share=0.01,
    max_missing_share=0.2,
    outcome_missing=from_spec
)
```

Check a declared causal design against the data before estimating it.

A `CausalSpec` fixes the intervention, the timing and the
adjustment set in words. This function reports where the frame
contradicts that account, and where the frame cannot check it at all --
causes are not in the data, so a spec whose timing cannot be verified
says so rather than passing silently. Run it before `aipw`: a
blocking finding means the estimand is not identified on these rows,
not that the estimator would have returned a wider interval.

**Parameters**

- `frame` (mapping of column name to array)
- `spec` (CausalSpec): The design to check.
- `min_per_arm` (int): Rows the smaller arm must contain.
- `min_arm_share` (float): Share of rows the smaller arm must reach.
- `max_missing_share` (float): Missing share above which an adjustment column changes the
  population the adjustment set represents.
- `outcome_missing` ({"error", "ipw"}, optional): How the outcome's missing values are handled by `aipw`. Defaults
  to the spec's `outcome_missing`. With `"ipw"` the estimate targets
  all rows, and the report says so in place of the subpopulation note
  it gives for the default.

**Returns**

TrapReport

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/causal.py` lines 157-330](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L157-L330)
