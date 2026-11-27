# `screen_interactions`

Defined in `mboostrs.terms`; exported as `mboostrs.screen_interactions`.


*Function*

```python
screen_interactions(
    data,
    residual,
    response=None,
    bins=4,
    seed=0,
    pairs=None,
    groups=(),
    na_action='error'
)
```

Rank candidate pairs by the residual variation they explain out of sample.

Both columns of a pair are binned into `bins` quantile cells on half the
rows, the cell means of `residual` are taken there, and the mean squared
error those means remove is measured on the other half. A pair that fits
only noise removes nothing out of sample, so its expected score is below
zero (about minus the cell count over the fit rows) and the ranking is not
a count of flexibility. A single score can exceed zero by chance. No model
is fitted.

The half-split is a seeded permutation of the rows. A fixed seed gives
the same scores across runs and thread counts.

`residual` is what the additive fit left behind, on the response
scale: `y - model.predict(type="response")`. For a Gaussian fit that is
`y - model.predict()`. For a family with a non-identity link (Binomial,
Poisson, Gamma) the default `predict()` is the link-scale predictor, and
subtracting it from the response leaves curvature that every pair of
signal columns appears to explain.

`removed` is a mean squared error, per held-out observation, so it is on
the same scale as the difference between two out-of-fold risks and can be
ranked alongside them.

A column named in `groups` is not binned: each of its levels is its own
cell. Quantile bins of a grouping code put unrelated groups in one cell, so
a slope that differs from group to group averages out and a random slope
goes undetected.

Missing values (NaN, or infinity) are handled by `na_action`, with the
meanings the fit functions give it:

* `"error"` (default): raise, naming the column, the number of missing
  values, the first row that has one, and the remedy.
* `"impute"`: a numeric column's missing values become its median. A
  grouped column's missing level is a group of its own, as the fit
  functions give a categorical predictor's missing level.
* `"omit"`: rows with a missing value in any screened column, or a
  non-finite `residual`, are dropped before the screen. The residual
  must then be the one for the same rows as `data`.

`residual` must be finite under `"error"` and `"impute"`.

**Returns**

list of ScreenedPair
    Sorted by `removed`, descending; equal scores tie by the pair's
    column order. Callers choose the cutoff; a score at or below zero is
    evidence against the pair, not merely a weak ranking.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/terms.py` lines 869-962](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L869-L962)
