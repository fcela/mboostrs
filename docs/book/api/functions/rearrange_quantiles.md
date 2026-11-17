# `rearrange_quantiles`

Defined in `mboostrs.rearrange`; exported as `mboostrs.rearrange_quantiles`.


*Function*

```python
rearrange_quantiles(models, newdata=None, *, levels=None, output='auto')
```

Predict several quantile or expectile fits and sort each row across levels.

Each model estimates one level `tau` of the conditional distribution, so
the separately fitted curves can cross: at some rows the predicted
`tau = 0.9` value lies below the predicted `tau = 0.5` value. This
function predicts every model on `newdata` and, for each row, sorts the
predictions into increasing order of `tau` (the rearrangement of
Chernozhukov, Fernandez-Val and Galichon, 2010). Rows that did not cross
are returned unchanged. The result also reports, from the unsorted
predictions, the share of rows with a crossing, the number of crossings
per pair of adjacent levels and the largest crossing.

Use it when two or more `Quantile` fits, or two or more `Expectile`
fits, are read together as a conditional distribution or as the edges of
a prediction band. The identity link makes the predictions of these
families response-scale values. Rows without a crossing are unchanged.
Chernozhukov et al. show that, when the true curve is increasing in
`tau`, the sorted estimate is no farther from it than the unsorted
estimate in every `L^p` norm. The fitted models are not modified and
nothing is refit.

**Parameters**

- `models` (sequence of fitted models, or dict): Fitted `glmboost`/`gamboost` models, all with `Quantile`
  families or all with `Expectile` families, at distinct `tau`.
  A dict is read as `{tau: model}` and its keys must equal each
  model's family `tau`.
- `newdata` (dict or DataFrame, optional): Rows to predict, with the predictor columns every model needs.
  `None` predicts the training rows; all models must then share the
  same training rows.
- `levels` (sequence of float, optional): If given, the `tau` of each model in `models` order, checked
  against the `tau` read from each model's family.
- `output` ({"auto", "polars", "pandas", "records"}): Frame type of the returned tables.

**Returns**

RearrangedQuantiles
    Sorted and unsorted predictions with one column per level, ordered by
    increasing `tau`, and the crossing report.

**Raises**

- `MboostrsValueError`: If fewer than two models are given; if a model is not a Quantile or
Expectile fit; if Quantile and Expectile fits are mixed; if two
models share a `tau`; if `levels` disagrees with the models; or if
the predictions differ in length or contain NaN.

**Examples**

```python
>>> fits = [mb.glmboost("y ~ bols(x)", data=d, family=mb.Quantile(tau=t))
...         for t in (0.1, 0.5, 0.9)]                      # doctest: +SKIP
>>> r = mb.rearrange_quantiles(fits, d)                    # doctest: +SKIP
>>> r.crossing_share, r.predictions                        # doctest: +SKIP
```

Source: [`python/mboostrs/rearrange.py` lines 157-297](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/rearrange.py#L157-L297)
