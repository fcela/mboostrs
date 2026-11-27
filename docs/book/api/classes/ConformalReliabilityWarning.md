# `ConformalReliabilityWarning`

Defined in `mboostrs.conformal`; exported as `mboostrs.ConformalReliabilityWarning`.


*Class*

```python
ConformalReliabilityWarning(...)
```

Warning emitted when conformal reliability diagnostics detect risk to coverage.

Triggered when query rows extrapolate outside the training predictor support,
when covariate mean shift is substantial, or when the calibration split is
small enough that finite-sample coverage variance is wide.

Source: [`python/mboostrs/conformal.py` lines 62-68](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/conformal.py#L62-L68)
