# `make_grid`

Defined in `mboostrs.cvrisk`; exported as `mboostrs.make_grid`.


*Function*

```python
make_grid(max, length_out=10, min=None, log=True, dense_mu_grid=True)
```

Build a per-parameter `mstop` grid for distributional cross-validation.

The counterpart of `gamboostLSS::make.grid`. Tuning an LSS model means
choosing one `mstop` per distribution parameter, so the search space is a
product grid rather than a line, and it has to be built deliberately.

**Parameters**

- `max` (int or mapping): The largest `mstop` to consider. An integer returns a single vector;
  a mapping of parameter name to integer returns the product grid over
  those parameters, in the mapping's order.
- `length_out` (int or mapping): How many points per parameter.
- `min` (int or mapping, optional): The smallest `mstop`. Defaults to 1.
- `log` (bool): Space the points logarithmically. Boosting improves quickly at first
  and slowly later, so a log grid puts its resolution where the risk
  actually changes. Points are rounded to integers, and duplicates
  produced by rounding are dropped.
- `dense_mu_grid` (bool): Fill in every integer along the first parameter above the largest value
  of the others. Cross-validation can extend a fitted path along that
  direction without refitting, so those points are nearly free; the
  remaining grid points each cost a fit.

**Returns**

list[int] or list[dict]
    A list of `mstop` values for a scalar `max`; otherwise a list of
    grid points, each a mapping of parameter name to `mstop`, ready to
    pass as `cvrisk(grid=...)`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Examples**

```python
>>> make_grid(100, length_out=6)
[1, 3, 6, 16, 40, 100]
```

Source: [`python/mboostrs/cvrisk.py` lines 1012-1105](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L1012-L1105)
