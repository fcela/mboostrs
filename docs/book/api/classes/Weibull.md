# `Weibull`

Defined in `mboostrs.families`; exported as `mboostrs.Weibull`.


*Class*

```python
Weibull()
```

Weibull accelerated-failure-time model for right-censored data.

Takes a `Surv(time, status)` response and models log survival time
directly, so the fit is a statement about time rather than about a hazard
ratio. The scale parameter is a nuisance estimated alongside the path.

Source: [`python/mboostrs/families.py` lines 260-266](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L260-L266)
