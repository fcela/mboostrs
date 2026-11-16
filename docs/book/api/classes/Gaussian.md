# `Gaussian`

Defined in `mboostrs.families`; exported as `mboostrs.Gaussian`.


*Class*

```python
Gaussian()
```

Squared error loss, `0.5 * (y - f)**2`, with an identity link.

The default family. The negative gradient is the raw residual `y - f`,
and the offset is the weighted mean of the response. Use it for a
continuous response whose conditional distribution is roughly symmetric
with no heavy tails; `Huber` or `Quantile(0.5)` are the robust
alternatives when it is not.

Source: [`python/mboostrs/families.py` lines 13-21](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L13-L21)
