# `calibration_frame`

Defined in `mboostrs.diagnostics`; exported as `mboostrs.calibration_frame`.


*Function*

```python
calibration_frame(model, xvar=None, bins=10, newdata=None, mstop=None, output='auto')
```

Compare observed and fitted means within bins.

Bins the observations by `xvar` (or by observation order when it is
`None`) and reports the mean response and the mean fitted value in each.
A well-calibrated fit puts the two columns on top of each other across
every bin; a systematic gap that moves with `xvar` is the fit getting
that part of the predictor space wrong.

On a distributional model, this computes the family's conditional response
mean from all fitted parameters. It refuses families without a declared
finite mean. For example, zero-adjusted Gamma uses `(1-nu)*mu` and Gumbel
uses `mu + EulerGamma*sigma`; the first parameter alone is not either
mean. The result column is `fitted_mean` on both standard and LSS models.

A distribution-free calibration check that does not depend on which
parameter means what is `pit` or `worm_data`: both ask whether
the fitted distribution is right as a whole, rather than whether one of
its parameters tracks the response.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`
- `NotImplementedError`

Source: [`python/mboostrs/diagnostics.py` lines 710-794](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/diagnostics.py#L710-L794)
