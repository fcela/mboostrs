# `CoxPH`

Defined in `mboostrs.families`; exported as `mboostrs.CoxPH`.


*Class*

```python
CoxPH()
```

Cox partial likelihood for right-censored survival data.

Takes a `Surv(time, status)` response, with `status` 1 for an observed
event and 0 for a censored observation. Fits the log hazard ratio, leaving
the baseline hazard unspecified, so predictions order patients by risk
rather than giving survival times. For an absolute time scale, use one of
the accelerated-failure-time families instead.

Source: [`python/mboostrs/families.py` lines 314-322](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L314-L322)
