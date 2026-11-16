# `q_stats`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.q_stats`.


*Function*

```python
q_stats(
    model,
    xvar=None,
    n_intervals=10,
    newdata=None,
    mstop=None,
    randomized=True,
    random_state=None,
    output='auto'
)
```

Binned quantile-residual moment summaries.

One row per bin: the interval number, the binning variable's name and
range, the count, and the first four moments of the (randomized) quantile
residuals in that bin. Under a correctly specified model the residuals are
standard normal, so a bin's mean should sit near 0, its variance near 1,
and its skewness and excess kurtosis near 0; a column that drifts with
`xvar` points at the part of the predictor space the distribution is
getting wrong.

`xvar=None` bins by observation order, which detects a departure that
tracks the data's own ordering and nothing else. Pass the variable you
suspect.

Descriptive moments only. GAMLSS's `Q.stats` also reports Z-statistics
and p-values against them; those need the exact reference definition and
its assumptions reproduced before they would mean anything, which is a
separate piece of work.

Source: [`python/mboostrs/diagnostics.py` lines 345-385](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L345-L385)
