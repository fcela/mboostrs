# `Quantile`

Defined in `mboostrs.families`; exported as `mboostrs.Quantile`.


*Class*

```python
Quantile(tau=0.5, qoffset=0.5)
```

Check (pinball) loss for quantile regression.

Targets the conditional `tau`-quantile: residuals above the fit are
weighted `tau` and those below `1 - tau`, so `tau=0.5` gives median
regression. The offset is the weighted `qoffset` quantile of the
response, not `tau`. `qoffset` defaults to 0.5 even when `tau` is
not, matching R, so `Quantile(tau=0.9)` starts at the median.

Fitting several `tau` values gives a picture of the whole conditional
distribution without assuming its shape, which is the alternative to the
distributional families when only a few quantiles are needed.

**Parameters**

- `tau` (float): The quantile to fit, in (0, 1).
- `qoffset` (float): Quantile of the response used for the starting offset. Default 0.5,
  independent of `tau`. Pass `qoffset=tau` to start at the quantile
  being fitted.

Source: [`python/mboostrs/families.py` lines 86-111](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L86-L111)
