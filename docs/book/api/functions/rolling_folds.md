# `rolling_folds`

Defined in `mboostrs.folds`; exported as `mboostrs.rolling_folds`.


*Function*

```python
rolling_folds(
    n,
    B=5,
    horizon=None,
    initial=None,
    expanding=True,
    weights=None,
    time=None,
    embargo=None
) -> FoldScheme
```

Rolling-origin splits: train on the past, judge on what comes next.

Fold `b` trains on rows up to an origin and is judged on the `horizon`
rows after it, so no fold ever sees a row later than the ones it predicts.
This is the only scheme here that measures forecasting.

`expanding=True` grows the training window from the first row each time;
`expanding=False` slides a window of fixed length, which is what to use
when the relationship itself drifts and old rows should stop counting.

`horizon` defaults to an even division of the rows after `initial`;
`initial` defaults to half the rows. Both count rows in time order.

`time` gives each row a timestamp, numeric or `datetime64`. The rows
are then ordered by it (ties keep their array order), so the data need not
be sorted and "before" means "earlier in time", not "earlier in the
array". Without `time` the array order is the time order.

`embargo` is a minimum gap between the last training timestamp and the
first judged timestamp of each fold. Training rows inside the gap are
dropped from that fold and are neither trained on nor judged. It needs
`time`: a gap is measured in timestamps, not in row positions. Numeric
timestamps take a number in their own units; `datetime64` timestamps
take a `numpy.timedelta64`, such as `np.timedelta64(7, "D")`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/folds.py` lines 772-878](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/folds.py#L772-L878)
