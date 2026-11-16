# `same_family_gap`

Defined in `mboostrs._same_family`; exported as `mboostrs.same_family_gap`.


*Function*

```python
same_family_gap(model, cv=None, *, comparison_loglik=None, tol=1.0)
```

Compare a linear boost at its cross-validated step with the same-family GLM.

Reports the chosen `mstop`, the boost `logLik` there, the comparison
log-likelihood, and the gap. When the gap is larger than `tol` (nats,
total, default 1), the message says cross-validation stopped while the
likelihood was still short of the GLM. The fit is not extended toward
the maximum, and the step cross-validation chose is left where it is.

**Parameters**

- `model`: A fitted `glmboost` or linear `glmboostLSS` model. The path must
  already reach the step being judged; this function does not call
  `continue_to`.
- `cv` (cvrisk result, optional): The result of `model.cvrisk(...)`. The step judged is
  `cv.optimal_mstop`. Without it, the model's own `mstop` is used
  and the message does not attribute the step to cross-validation.
- `comparison_loglik` (float, optional): A same-family log-likelihood the caller already computed, on the same
  rows and with the same constants as `logLik`. When given, no GLM
  is fit.
- `tol` (float): A gap above this, in nats, is material. A boost that has reached the
  GLM sits well below it; the default is 1.

**Returns**

SameFamilyGap
    `available` is false when this family has no likelihood, or when no
    cheap same-family GLM can be fit and none was supplied. `material`
    is then false too.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/_same_family.py` lines 119-213](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_same_family.py#L119-L213)
