# `Expectile`

Defined in `mboostrs.families`; exported as `mboostrs.Expectile`.


*Class*

```python
Expectile(tau=0.5)
```

Asymmetric squared error loss for expectile regression.

The expectile analogue of `Quantile`: squared rather than absolute
deviations, weighted `tau` above the fit and `1 - tau` below. Expectiles
are smoother in the data than quantiles and cheaper to fit, at the cost of
a less direct interpretation — the `tau`-expectile is not the
`tau`-quantile.

**Parameters**

- `tau` (float): The expectile to fit, in (0, 1). `0.5` recovers the mean.

Source: [`python/mboostrs/families.py` lines 114-130](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L114-L130)
