# `GamlssRefit`

Defined in `mboostrs.selection`; exported as `mboostrs.GamlssRefit`.


*Class*

```python
GamlssRefit(
    family: str,
    terms: dict,
    edf: dict,
    smoothing: dict,
    reml_score: float,
    convergence: dict,
    conditional_on_selection: bool,
    _blocks: dict = <factory>,
    _core: object = None,
    _replay: object = None,
    _training_data: object = None,
    _ranges: dict = <factory>,
    intercepts: dict = <factory>,
    intercept_only: tuple = ()
) -> None
```

A penalized refit of a distributional model's terms, all parameters
estimated jointly, with REML smoothing parameters.
Usage:

```python
import mboostrs as mb
refit = mb.gamlss_refit(model)  # model: a gamboostLSS/glmboostLSS fit
refit.effects(newdata, parameter="mu")
refit.predict(newdata)
```

Produced by `gamlss_refit`. `edf` and `smoothing` map each
distribution parameter to `{term: value}`. `effects` returns each
term's centred effect with a pointwise interval; `predict` returns
each parameter on the link or the natural scale.

Source: [`python/mboostrs/selection.py` lines 1967-2256](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1967-L2256)

## Members

| Member | Kind | Summary |
|---|---|---|
| `parameters` | Property | The distribution parameters, in family order. |
| `vcov` | Method | Covariance of the joint coefficients (each parameter's intercept, |
| `effects` | Method | ``{parameter: {term: (estimate, lower, upper)}}`` at the rows of |
| `predict` | Method | Each parameter's linear predictor (``type="link"``) or its value on |
| `contrast` | Method | The average change in each parameter's linear predictor when covariates move. |
| `draws` | Method | Draws of every distribution parameter, on its natural scale, at the |
| `summary` | Method | Each distribution parameter's terms, with their effective degrees of freedom. |
| `to_dict` | Method | A JSON-safe record of the terms, edf and smoothing parameters per |

(gamlssrefit-parameters)=
## `GamlssRefit.parameters`

*Property* of `GamlssRefit`

```python
GamlssRefit.parameters
```

The distribution parameters, in family order.

Source: [`python/mboostrs/selection.py` lines 2021-2024](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2021-L2024)

(gamlssrefit-vcov)=
## `GamlssRefit.vcov`

*Method* of `GamlssRefit`

```python
GamlssRefit.vcov(type='bayesian')
```

Covariance of the joint coefficients (each parameter's intercept,
then its terms, parameter by parameter); `type="corrected"` adds the
first-order uncertainty of the smoothing parameters.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 2031-2039](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2031-L2039)

(gamlssrefit-effects)=
## `GamlssRefit.effects`

*Method* of `GamlssRefit`

```python
GamlssRefit.effects(
    newdata,
    parameter=None,
    level=0.9,
    unseen_levels='error',
    corrected=False
)
```

`{parameter: {term: (estimate, lower, upper)}}` at the rows of
`newdata`, on each parameter's link scale; with `parameter`, only that
parameter's `{term: ...}`. Each effect is centred over the fitting
rows. `unseen_levels` and `corrected` are as in
`GamRefit.effects`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 2041-2052](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2041-L2052)

(gamlssrefit-predict)=
## `GamlssRefit.predict`

*Method* of `GamlssRefit`

```python
GamlssRefit.predict(
    newdata,
    parameter=None,
    type='link',
    unseen_levels='error',
    interval=False,
    level=0.9,
    corrected=False,
    output='auto',
    extrapolation='warn'
)
```

Each parameter's linear predictor (`type="link"`) or its value on
the natural scale through the family's inverse link
(`type="response"`), as `{parameter: array}`; with `parameter`,
that parameter's array.

`interval=True` returns, for each parameter, a frame with
`estimate`, `se`, `lower` and `upper` per row (a frame for
`parameter` alone, else a dict of frames). The estimate is the
parameter's intercept plus its terms, and `se` is
`sqrt(x' V x)` over that parameter's own coefficients (its
intercept and terms), with the covariance between them. `se` is on
the link scale; with `type="response"` the bounds are mapped through
the inverse link. `outside_range` is true for a row where a spline
covariate of that parameter's terms lies outside its training range.
`level`, `corrected`, `output` and `extrapolation` are as in
`GamRefit.predict`; the policy is applied to each covariate
once, across all parameters that read it.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 2077-2119](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2077-L2119)

(gamlssrefit-contrast)=
## `GamlssRefit.contrast`

*Method* of `GamlssRefit`

```python
GamlssRefit.contrast(variable, from_value, to_value, *, data, parameter=None, level=0.9)
```

The average change in each parameter's linear predictor when covariates move.

Each row of `data` is taken with the covariates at `from_value` in one
copy and at `to_value` in another, every other column at the row's own
value, and the mean of the difference of each parameter's link-scale
predictor is returned with its standard error and `level` interval.
Arguments are as for `GamRefit.contrast`. The result is a dict
`{parameter: EffectContrast}`, or one `EffectContrast` when
`parameter` is given. Each parameter's change is on its own link scale:
the identity for `mu` and the log for a scale parameter.

Source: [`python/mboostrs/selection.py` lines 2121-2145](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2121-L2145)

(gamlssrefit-draws)=
## `GamlssRefit.draws`

*Method* of `GamlssRefit`

```python
GamlssRefit.draws(
    newdata,
    n_draws=1000,
    seed=0,
    corrected=False,
    unseen_levels='error'
)
```

Draws of every distribution parameter, on its natural scale, at the
rows of `newdata`, from the approximate posterior of the joint
coefficients: `{parameter: array of shape (n_draws, n_rows)}`.

A quantity that combines parameters, such as a centile of the
response distribution, is computed draw by draw from these arrays,
and its interval taken from the quantiles over the draws; the draws
keep the correlation between the parameters' estimates.
`corrected=True` draws from the covariance corrected for
smoothing-parameter uncertainty.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 2147-2165](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2147-L2165)

(gamlssrefit-summary)=
## `GamlssRefit.summary`

*Method* of `GamlssRefit`

```python
GamlssRefit.summary()
```

Each distribution parameter's terms, with their effective degrees of freedom.

One row per parameter and term: `edf` is the effective degrees of
freedom the penalized refit gives the term and `smoothing` its REML
smoothing parameter. The refit carries no term tests, so the table
reports the fitted quantities only. `effects` gives a term's
shape with a pointwise interval.

Source: [`python/mboostrs/selection.py` lines 2229-2248](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2229-L2248)

(gamlssrefit-to_dict)=
## `GamlssRefit.to_dict`

*Method* of `GamlssRefit`

```python
GamlssRefit.to_dict(*, include_arrays=False)
```

A JSON-safe record of the terms, edf and smoothing parameters per
distribution parameter; `include_arrays=True` adds the coefficients.

Source: [`python/mboostrs/selection.py` lines 2250-2256](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L2250-L2256)
