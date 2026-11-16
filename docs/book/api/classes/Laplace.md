# `Laplace`

Defined in `mboostrs.families`; exported as `mboostrs.Laplace`.


*Class*

```python
Laplace()
```

Absolute error loss, `|y - f|`, with an identity link.

The negative gradient is `sign(y - f)`, so every observation contributes
the same magnitude regardless of how far off it is, and the fit targets the
conditional **median** rather than the mean. The offset is the weighted
median. Use it when outliers should not pull the fit.

Source: [`python/mboostrs/families.py` lines 45-52](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L45-L52)
