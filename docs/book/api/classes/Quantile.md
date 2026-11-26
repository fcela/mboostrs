# `Quantile`

Defined in `mboostrs.families`; exported as `mboostrs.Quantile`.


*Class*

```python
Quantile(tau=0.5, qoffset=None, response_scale='auto')
```

Check (pinball) loss for quantile regression.

Targets the conditional `tau`-quantile: residuals above the fit are
weighted `tau` and those below `1 - tau`, so `tau=0.5` gives median
regression. The offset is the weighted `qoffset` quantile of the
response, by default the quantile being fitted.

Fitting several `tau` values gives a picture of the whole conditional
distribution without assuming its shape, which is the alternative to the
distributional families when only a few quantiles are needed.

Three defaults differ from R mboost's `QuantReg`, chosen on 18
quantile fits (six OpenML-CTR23 data sets, `tau` 0.1, 0.5, 0.9): the
offset starts at `tau` rather than the median; a fit given no
`control` (and `tune_mstop` given no `nu`) takes steps of
`nu = 0.3` rather than 0.1, which cut the median time 2.7-fold with
the pinball loss 1.6% lower on average; and the response is scaled (see
`response_scale`). `Quantile(tau, qoffset=0.5, response_scale=None)`
with `boost_control(..., nu=0.1)` is R's `QuantReg`.

**Parameters**

- `tau` (float): The quantile to fit, in (0, 1).
- `qoffset` (float, optional): Quantile of the response used for the starting offset. `None`
  (default) is `tau`; R's `QuantReg` uses 0.5 whatever `tau`.
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

Source: [`python/mboostrs/families.py` lines 129-176](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L129-L176)
