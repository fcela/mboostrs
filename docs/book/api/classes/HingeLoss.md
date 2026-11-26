# `HingeLoss`

Defined in `mboostrs.families`; exported as `mboostrs.HingeLoss`.


*Class*

```python
HingeLoss(alpha_cost=0.5)
```

Hinge loss, `max(0, 1 - y f)` — the support vector loss.

Classification with a two-class response. Observations already classified
with margin contribute nothing to the gradient, so the fit is driven by the
points near the boundary.

Its `offset()` returns the AdaBoost log-odds rather than the minimiser of
its own risk. The risk is piecewise linear in a constant `c` with its
minimum at the boundary `c = ±1`, so no interior offset can minimise it;
R's internal `HingeLoss` returns the identical value.

**Parameters**

- `alpha_cost` (float): Asymmetric misclassification cost. `0.5`, the default, is the
  symmetric hinge loss.

Source: [`python/mboostrs/families.py` lines 352-374](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L352-L374)
