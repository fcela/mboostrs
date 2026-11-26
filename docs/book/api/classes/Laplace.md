# `Laplace`

Defined in `mboostrs.families`; exported as `mboostrs.Laplace`.


*Class*

```python
Laplace(response_scale='auto')
```

Absolute error loss, `|y - f|`, with an identity link.

The negative gradient is `sign(y - f)`, so every observation contributes
the same magnitude regardless of how far off it is, and the fit targets the
conditional **median** rather than the mean. The offset is the weighted
median. Use it when outliers should not pull the fit.

**Parameters**

- `response_scale` ({"auto", "none"} or float, default "auto"): The scale the response is divided by before fitting. The negative
  gradient of this loss does not grow with the response, so each
  boosting step moves the fit by about `nu` times a constant in
  response units: without a scale the same data in dollars and in
  thousands of dollars give different fits at the same `mstop`, and a
  response in large units barely leaves its offset. `"auto"` fits the
  response divided by its weighted SD (the SD of the rows each fit sees)
  and maps the model back exactly, so predictions, coefficients and the
  risk are in the original units. `"none"` (or `None`) is R mboost's
  raw-units path; a number fixes the scale.

Source: [`python/mboostrs/families.py` lines 71-95](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L71-L95)
