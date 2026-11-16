# `Binomial`

Defined in `mboostrs.families`; exported as `mboostrs.Binomial`.


*Class*

```python
Binomial()
```

Binomial log-likelihood with a logit link, for a two-class response.

The response is coded `{0, 1}`. `predict(type="response")` returns
`P(y = 1) = (1 + exp(-2f))**-1` — note the factor of two, which follows
`mboost`'s internal `±1` coding, so the linear predictor is half the
usual log-odds. The offset is half the log-odds of the weighted class
frequency.

Source: [`python/mboostrs/families.py` lines 24-32](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L24-L32)
