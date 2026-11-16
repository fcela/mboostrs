# `Hurdle`

Defined in `mboostrs.families`; exported as `mboostrs.Hurdle`.


*Class*

```python
Hurdle()
```

The count part of a hurdle model: a zero-truncated negative binomial.

A negative binomial with a log link, conditioned on the count being
positive, as `mboost::Hurdle()`. Fit it to the positive counts only;
whether a count is positive is a separate binary model (`Binomial` on
`y > 0`), and the two together describe counts with more or fewer zeros
than one negative binomial allows. The dispersion `sigma` is
re-estimated at every step by a Brent search on (0, 100), matching R's
`optimize`. `sigma` is the gamlss dispersion, so the variance is
`mu + sigma * mu**2` and the size is `1 / sigma`. That is the
reciprocal of `NBinomial`, whose `sigma` is the size
(`mu + mu**2 / sigma`). `predict(type="response")` returns `exp(f)`,
the mean of the untruncated negative binomial, as mboost does; the mean of
the truncated count is that divided by `1 - (1 + exp(f) * sigma) ** (-1 / sigma)`.

Source: [`python/mboostrs/families.py` lines 183-198](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L183-L198)
