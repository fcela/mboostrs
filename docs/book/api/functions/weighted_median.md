# `weighted_median`

Defined in `mboostrs.cvrisk`; exported as `mboostrs.weighted_median`.


*Function*

```python
weighted_median(x, w=1.0, na_rm=False)
```

The weighted median, matching `gamboostLSS:::weighted.median`.

Observations with zero weight are dropped, the rest are sorted, and the
value where the cumulative weight first passes half the total is returned.
When the weights are whole numbers summing to an even total the two central
values are averaged, matching the unweighted median's convention; otherwise
the larger of the two is returned, which is what the R implementation does.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/cvrisk.py` lines 1108-1144](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L1108-L1144)
