# `Lognormal`

Defined in `mboostrs.families`; exported as `mboostrs.Lognormal`.


*Class*

```python
Lognormal()
```

Log-normal accelerated-failure-time model for right-censored data.

Takes a `Surv(time, status)` response and models log survival time as
normal, which gives a hazard that rises and then falls — appropriate when
risk peaks some time after the origin rather than decaying monotonically.

Source: [`python/mboostrs/families.py` lines 343-349](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L343-L349)
