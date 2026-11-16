# `PropOdds`

Defined in `mboostrs.families`; exported as `mboostrs.PropOdds`.


*Class*

```python
PropOdds(n_levels=None, levels=None)
```

Proportional-odds cumulative model for an ordered categorical response.

A single linear predictor `f` shifts a set of ordered thresholds, giving
cumulative probabilities `P(y <= k) = (1 + exp(-(theta_k - f)))**-1`. The
thresholds are nuisance parameters estimated alongside the boosting steps.

The response is given in one of three ways:

* an **ordered categorical column** (a pandas `Categorical` with
  `ordered=True` or a polars `Enum`), whose declared order is used;
* any column with `levels=`, the categories in order, lowest first:
  `PropOdds(levels=["low", "mid", "high"])` or, for a zero-based integer
  code, `PropOdds(levels=[0, 1, 2])`;
* integers `{1, ..., K}` with `n_levels=K`, R's 1-based factor codes.

The order has to be declared rather than read from the values: a subset
of the rows (a cross-validation fold, a conformal split) can lack the
lowest class, and a coding inferred from the values it holds would shift
every class by one. `predict(type="response")` returns an `(n, K)`
probability matrix whose column `k` is the probability of the `k`-th
level.

**Parameters**

- `n_levels` (int, optional): Number of ordered categories, at least 3. Taken from `levels` or
  from an ordered categorical column when those give it. (Two ordered
  categories is a binary response, which `Binomial` fits.)
- `levels` (sequence, optional): The categories in order, lowest first.

Source: [`python/mboostrs/families.py` lines 201-246](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L201-L246)
