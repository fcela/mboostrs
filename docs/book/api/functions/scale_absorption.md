# `scale_absorption`

Defined in `mboostrs.absorption`; exported as `mboostrs.scale_absorption`.


*Function*

```python
scale_absorption(model, *, folds=5, seed=0, flexible=None)
```

Test whether a fitted LSS scale effect survives a more flexible mean.

`model` is a cyclic `gamboostLSS` fit with a continuous family that
has a mean `mu` and a scale `sigma` (`GaussianLSS` and the other
location-scale families). The function refits the model's specification
three ways on the same `folds`-fold split, all with the given model's
control:

- `given`: the model as it was fitted. Its in-sample scale and its
  out-of-fold score are read from the fit itself.
- `flexible_mean`: the mean formula with each `bbs` and `bspatial`
  smooth at twice its degrees of freedom, and each continuous predictor
  that enters the mean only linearly (`x` or `bols(x)`) given a
  `bbs` smooth at the default df. `flexible=` replaces this formula
  with a mean formula the caller writes (a full `"y ~ ..."` string).
- `slower_scale`: the same formulas with learning rates
  `{mu: nu, sigma: nu / 4}`, where `nu` is the given model's rate for
  each parameter.

For each variant the function reports the spread of fitted log-sigma
over the training rows, the range of each sigma term, the out-of-fold
joint score, and the residual structure left in the mean.

Verdict rule: `"scale effect absorbs mean misspecification"` when the
flexible mean removes at least 25% of the given fit's log-sigma spread
and its out-of-fold gain has `t >= 2`; `"scale effect robust"` when
it removes less than 10%; `"inconclusive"` otherwise. A given fit whose
log-sigma spread is below 0.1 has no scale effect to absorb, and the
verdict is inconclusive. The slower-scale variant is reported and does
not enter the verdict: a slower rate with the same `mstop` moves sigma
less by construction, so its shrinkage is not evidence about the mean.

**Parameters**

- `model` (LSSModel): A fit from `mboostrs.gamboostLSS` with a continuous
  location-scale family. Non-cyclic and `glmboostLSS` fits are
  refused.
- `folds` (int, default 5): Number of folds for the out-of-fold scores.
- `seed` (int, default 0): Seed of the fold assignment.
- `flexible` (str, optional): Mean formula for the flexible variant, for example
  `"y ~ bbs(x, df=8)"`. Its response must match the model's.

**Returns**

ScaleAbsorption
    The verdict, the measures behind it and `to_dict()`.

**Raises**

- `NotImplementedError`: If the family is not a continuous location-scale family, or the fit
is non-cyclic or from `glmboostLSS`.
- `ValueError`: If no flexible mean can be built from the given mean formula, and
`flexible=` was not supplied.

**Examples**

>>> fit = mb.gamboostLSS({"mu": "y ~ bols(x)", "sigma": "y ~ bbs(x)"},
...                      data=data, families=mb.GaussianLSS())
>>> result = mb.scale_absorption(fit, folds=5, seed=0)
>>> result.verdict  # doctest: +SKIP
'scale effect absorbs mean misspecification'

Source: [`python/mboostrs/absorption.py` lines 125-283](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/absorption.py#L125-L283)
