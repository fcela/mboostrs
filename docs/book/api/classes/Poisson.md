# `Poisson`

Defined in `mboostrs.families`; exported as `mboostrs.Poisson`.


*Class*

```python
Poisson()
```

Poisson log-likelihood with a log link, for counts.

`mu = exp(f)` and the negative gradient is the response-scale residual
`y - mu`. The offset is the log of the weighted mean rate. Poisson fixes
the variance equal to the mean; when the data is overdispersed, use
`NBinomial`, and check with a PIT histogram rather than assuming.

Source: [`python/mboostrs/families.py` lines 35-42](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L35-L42)
