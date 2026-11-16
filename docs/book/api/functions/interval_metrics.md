# `interval_metrics`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.interval_metrics`.


*Function*

```python
interval_metrics(
    observed,
    lower,
    upper,
    level=0.95,
    weights=None,
    retain_scores=False
)
```

Weighted coverage, width, and proper central-interval score.

For nominal coverage `level` and `alpha = 1-level`, the observation's
interval score is

`(upper-lower) + 2/alpha*(lower-y)*I(y<lower) + 2/alpha*(y-upper)*I(y>upper)`.

Width rewards sharp intervals while the miss penalties preserve calibration;
comparing width alone favors intervals that cover nothing. The empirical
coverage and lower/upper miss rates describe these observations and are not
a future-coverage guarantee. `weights` define the evaluation population.

Source: [`python/mboostrs/diagnostics.py` lines 847-897](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L847-L897)
