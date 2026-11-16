# `GammaReg`

Defined in `mboostrs.families`; exported as `mboostrs.GammaReg`.


*Class*

```python
GammaReg()
```

Gamma log-likelihood with a log link, for a positive continuous response.

`mu = exp(f)`, so the response is strictly positive and its variance
grows with the square of the mean — constant coefficient of variation.
`nuisance()` is the shape in `Var(Y) = mu**2 / shape`. A Gamma GLM
reports the dispersion, which is the reciprocal of that shape.
Being right-skewed does not by itself make a response Gamma; check the fit
with a PIT histogram before committing to it.

Source: [`python/mboostrs/families.py` lines 55-64](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L55-L64)
