# `Loglog`

Defined in `mboostrs.families`; exported as `mboostrs.Loglog`.


*Class*

```python
Loglog()
```

Log-logistic accelerated-failure-time model for right-censored data.

Takes a `Surv(time, status)` response. Like `Lognormal` it allows
a non-monotone hazard, with heavier tails, and its survival function has a
closed form.

Source: [`python/mboostrs/families.py` lines 269-275](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L269-L275)
