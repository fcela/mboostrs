# `GamRefit`

Defined in `mboostrs.selection`; exported as `mboostrs.GamRefit`.


*Class*

```python
GamRefit(
    terms: tuple,
    edf: dict,
    smoothing: dict,
    sigma: float,
    conditional_on_selection: bool,
    link: str = 'identity',
    reml_score: float = nan,
    convergence: dict = <factory>,
    theta: float | None = None,
    held: tuple = (),
    _blocks: tuple = (),
    _core: object = None,
    _held_model: object = None,
    _held_learners: tuple = (),
    _held_scale: float = 1.0,
    _impute: object = None,
    _training_data: object = None,
    _ranges: dict = <factory>,
    _weights: object = None
) -> None
```

A jointly penalized refit of a set of terms, with Bayesian intervals.

Produced by `gam_refit`. `edf` and `smoothing` give each term's
effective degrees of freedom and REML smoothing parameter, `sigma` the
residual standard deviation. `effects` returns each term's
effect at new rows with a pointwise interval.

Source: [`python/mboostrs/selection.py` lines 750-1312](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L750-L1312)

## Members

| Member | Kind | Summary |
|---|---|---|
| `tests` | Property | ``{term: {"edf", "ref_df", "statistic", "p_value"}}``: each term's |
| `summary` | Method | A table of the terms, as mgcv's ``summary.gam`` gives it. |
| `concurvity` | Method | mgcv's concurvity of each term against the rest of the model. |
| `loglik` | Property | Log-likelihood of the refit, as mgcv's ``logLik.gam`` computes it: |
| `df` | Property | Effective degrees of freedom of the fit, plus one for an estimated |
| `aic` | Method | Conditional AIC, ``-2 loglik + 2 df``. mgcv's ``AIC(gam)`` counts |
| `vcov` | Method | Covariance matrix of the coefficients, as a square NumPy array. |
| `coef` | Method | The refit's coefficients as a table, with pointwise intervals. |
| `term_coefficients` | Method | The rows of :meth:`coef` that belong to one formula term. |
| `effects` | Method | ``{term: (estimate, lower, upper)}`` at the rows of `newdata`. |
| `contrast` | Method | The average change in the link-scale prediction when covariates move. |
| `predict` | Method | The refit's prediction at the rows of `newdata`, on the link scale |
| `bootstrap` | Method | Refit this Gaussian refit on each bootstrap draw of its rows. |
| `to_dict` | Method | A JSON-safe record of the terms, their edf, smoothing parameters, |

(gamrefit-tests)=
## `GamRefit.tests`

*Property* of `GamRefit`

```python
GamRefit.tests
```

`{term: {"edf", "ref_df", "statistic", "p_value"}}`: each term's
approximate test of `f = 0` (Wood 2013), the numbers mgcv's
`summary.gam` prints. The statistic is a chi-square for a known
scale (Binomial, Poisson, negative binomial, Cox) and an F statistic
(`statistic / ref_df`) when the scale is estimated. See
`summary` for how to read the p-values.

Source: [`python/mboostrs/selection.py` lines 814-835](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L814-L835)

(gamrefit-summary)=
## `GamRefit.summary`

*Method* of `GamRefit`

```python
GamRefit.summary()
```

A table of the terms, as mgcv's `summary.gam` gives it.

Each row gives the term's effective degrees of freedom (`edf`), the
rank its test uses (`ref_df`), the test statistic (F when the scale
is estimated, chi-square otherwise) and an approximate p-value for
the hypothesis that the term's effect is zero everywhere (Wood 2013).
The p-values treat the smoothing parameters as known, as mgcv's do.
When the terms were selected on the same data
(`conditional_on_selection`), they are smaller than the p-values
of a model fixed in advance, and a small p-value says the kept term's
shape is well determined rather than that selection was justified.

Source: [`python/mboostrs/selection.py` lines 837-872](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L837-L872)

(gamrefit-concurvity)=
## `GamRefit.concurvity`

*Method* of `GamRefit`

```python
GamRefit.concurvity(output='auto')
```

mgcv's concurvity of each term against the rest of the model.

Concurvity is the additive-model form of collinearity: how far a
term's fitted function could be reproduced by the other terms. When it
is high the partial effects are not separately identified -- the
same fit can be redistributed between the terms -- so their shapes,
intervals and the selection that chose them are unstable. Each row
gives mgcv's three measures (`concurvity(full = TRUE)`; Wood,
*Generalized Additive Models*, 2nd ed., section 5.6.2), each between
0 (the term is orthogonal to the others) and 1 (it lies in their
span): `worst` over every coefficient vector of the term,
`observed` at the fitted coefficients, and `estimate`, an average
over coefficient directions. They are computed in the core from the
refit's model matrix, the intercept included among the other terms.

Source: [`python/mboostrs/selection.py` lines 874-900](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L874-L900)

(gamrefit-loglik)=
## `GamRefit.loglik`

*Property* of `GamRefit`

```python
GamRefit.loglik
```

Log-likelihood of the refit, as mgcv's `logLik.gam` computes it:
Gaussian with the maximum-likelihood variance RSS/n, Cox the partial
likelihood. Its degrees of freedom are `df`.

Source: [`python/mboostrs/selection.py` lines 923-928](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L923-L928)

(gamrefit-df)=
## `GamRefit.df`

*Property* of `GamRefit`

```python
GamRefit.df
```

Effective degrees of freedom of the fit, plus one for an estimated
scale, negative binomial theta or gamma shape.

Source: [`python/mboostrs/selection.py` lines 930-934](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L930-L934)

(gamrefit-aic)=
## `GamRefit.aic`

*Method* of `GamRefit`

```python
GamRefit.aic()
```

Conditional AIC, `-2 loglik + 2 df`. mgcv's `AIC(gam)` counts
the degrees of freedom corrected for smoothing-parameter uncertainty
(its `edf2`), which are larger, so mgcv's AIC is larger by about
twice that difference (0.9 on the body-fat model). It compares
refits of the same response, not boosted models.

Source: [`python/mboostrs/selection.py` lines 936-942](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L936-L942)

(gamrefit-vcov)=
## `GamRefit.vcov`

*Method* of `GamRefit`

```python
GamRefit.vcov(type='bayesian')
```

Covariance matrix of the coefficients, as a square NumPy array.

Row and column `i` refer to coefficient `i` of the vector that
`coef` lists, in the same order. Index 0 is the intercept. After
it come the terms in the order of `terms`, each with its own
block of consecutive indices, one per column of the term's constrained
basis: a factor `bols` has one fewer than its number of levels, a
numeric `bols` one, and a `bbs` term one fewer than its basis
columns. The last index is therefore the last column of the last
term, which is not a treatment effect unless the treatment is that
term, and even then it is a constrained-basis weight, not a contrast.
Use `coef` or `term_coefficients` to find an index by
name, and `effects` for an effect.
`type="bayesian"` is conditional on the smoothing parameters;
`type="corrected"` adds the first-order correction for their
uncertainty (Wood, Pya and Saefken 2016), mgcv's
`vcov(unconditional=TRUE)` without its second-order term.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 944-966](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L944-L966)

(gamrefit-coef)=
## `GamRefit.coef`

*Method* of `GamRefit`

```python
GamRefit.coef(level=0.95, *, type='bayesian')
```

The refit's coefficients as a table, with pointwise intervals.

Returns a Polars DataFrame with one row per coefficient, in the order
of `vcov`, and the columns:

`term`
    The formula term the coefficient belongs to; `"(Intercept)"` for
    the intercept.
`coefficient`
    The term's formula text with the 1-based position of the column in
    its constrained basis, such as `bbs(x, df=4)[3]` or
    `bols(g)[2]`. The basis is the refit's sum-to-zero
    reparametrization of the term's design, so one coefficient is not
    a level difference or a slope: a factor's coefficients do not
    compare the levels one by one. Read the effect at each row with
    `effects`, and its interval there.
`estimate`
    The coefficient.
`se`
    Its standard error, the square root of the diagonal of
    `vcov(type)`.
`lower`, `upper`
    The interval `estimate -/+ z * se` at `level`, with `z` the
    standard normal quantile (a normal approximation).

`type` selects the covariance as in `vcov`. The intervals are
for the coefficients, not for an effect: an effect at a row combines
several coefficients, so `effects` gives that interval.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 990-1037](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L990-L1037)

(gamrefit-term_coefficients)=
## `GamRefit.term_coefficients`

*Method* of `GamRefit`

```python
GamRefit.term_coefficients(term, level=0.95, *, type='bayesian')
```

The rows of `coef` that belong to one formula term.

`term` is a term as written in `terms`, such as
`"bols(ethnicity)"` or `"bbs(x, df=4)"`. Raises a ValueError that
lists the terms when the name is not one of them. Use `"(Intercept)"`
for the intercept.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 1039-1053](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1039-L1053)

(gamrefit-effects)=
## `GamRefit.effects`

*Method* of `GamRefit`

```python
GamRefit.effects(
    newdata,
    level=0.9,
    unseen_levels='error',
    corrected=False,
    centre=False
)
```

`{term: (estimate, lower, upper)}` at the rows of `newdata`.

`unseen_levels="zero_effect"` gives a `brandom` term zero effect
(with zero standard error) at a group the refit never saw, the
average over groups, which is the prediction for a new group that
lme4 (`allow.new.levels`) and mgcv (`exclude=`) make. Its interval
is that of the average; a new group's own effect varies around it by
the random-effect standard deviation, which the interval leaves out.
`unseen_levels="average"` gives every term that reads the column
(a `brandom`, a `bols` over a factor, a factor `by=`, a
`%X%` factor margin) the mean of its level effects at such a row,
as `GAMBoostModel.predict` does.

Ordinary smooths are centred over the weighted fitting rows. A
nonconstant numeric `by=` term retains its constant-in-x direction,
which becomes the effect of the by variable. Its product with the
by variable need not have mean zero. The interval is the Bayesian
pointwise interval `estimate +/- z * se` from the coefficient
covariance (Wahba 1983; Nychka 1988), which averages close to nominal
coverage over the effect's range when the smoothing parameter is
estimated by REML. `corrected=True` widens it by the uncertainty of
the estimated smoothing parameters (see `vcov`).

`centre=True` centres each effect over the rows of `newdata`
instead: the estimate is `f(x_i)` minus its mean over those rows,
and the interval is that contrast's, from the covariance of the
centred design (the design row minus the column means over
`newdata`), not the uncentred interval shifted. Use it to compare an
effect with a curve centred over the same rows, such as a known truth
or another package's estimate.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 1055-1097](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1055-L1097)

(gamrefit-contrast)=
## `GamRefit.contrast`

*Method* of `GamRefit`

```python
GamRefit.contrast(variable, from_value, to_value, *, data, level=0.9)
```

The average change in the link-scale prediction when covariates move.

Every row of `data` is taken with `variable` set to `from_value` in one
copy and to `to_value` in another, all other columns at the row's own
values. The result is the mean over the rows of the difference of the
two link-scale predictions, with its standard error and the
`level` interval, returned as an `EffectContrast`. The
standard error uses the full covariance, so the terms a covariate
enters through (a smooth and a tensor, say) are combined with their
covariance rather than summed as separate intervals.

`variable` is one column name, with a scalar `from_value` and
`to_value` (a factor's level labels work too), or a sequence of names
moved together, such as `("latitude", "longitude")`, with sequences of
values. A covariate that no term reads has a change of exactly zero with
zero standard error. The spline covariates are checked against their
training ranges as `predict` checks them, with a warning.

Source: [`python/mboostrs/selection.py` lines 1099-1130](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1099-L1130)

(gamrefit-predict)=
## `GamRefit.predict`

*Method* of `GamRefit`

```python
GamRefit.predict(
    newdata,
    type='link',
    unseen_levels='error',
    offset=None,
    interval=False,
    level=0.9,
    corrected=False,
    output='auto',
    extrapolation='warn'
)
```

The refit's prediction at the rows of `newdata`, on the link scale
(`type="link"`) or the response scale (`type="response"`).
`unseen_levels` is as in `effects`. `offset` is the new
rows' offset (log exposure) when the model was fitted with one; it is
added to the linear predictor.

`extrapolation` is as for `GAMBoostModel.predict`, applied to
the covariates of the spline terms (`bbs`, `bspatial` and the
B-spline margins of `%X%`) against their training ranges. A basis
continues linearly beyond its boundary knots, so the default
`"warn"` predicts and issues one warning naming the covariate and
the rows outside; `"error"` refuses; `"allow"` predicts silently;
`"clip"` holds each covariate at its nearest training value.

`interval=True` returns a frame with `estimate`, `se`, `lower`
and `upper` for each row instead of the bare array, and
`outside_range`, true where a range-checked covariate of that row
lies outside its training range (computed on the values as given,
before any clipping). The estimate is the sum of the intercept and
every term's effect, and `se` is
`sqrt(x' V x)` from the full covariance `V` of the intercept and
all terms, so it includes the covariance between terms and the
intercept's uncertainty; a term's own interval (`effects`) omits
the intercept. `se` is always on the link scale. The bounds are
`estimate +/- z * se` on the link scale, and mapped through the
inverse link when `type="response"`. `level` is the coverage;
`corrected=True` uses the covariance corrected for the uncertainty of
the smoothing parameters; `output` is as for other frames. The
interval is conditional on the held terms (`held`) and on the
chosen terms, as `effects` is.

**Returns**

numpy.ndarray or polars.DataFrame
    Without `interval=True`, a NumPy array with one value per row of
    `newdata`, on the scale named by `type`. With `interval=True`, a
    Polars frame with one row per row of `newdata` and the columns
    `estimate`, `se`, `lower`, `upper` and `outside_range`
    (`output="auto"`, the default, gives Polars when it is installed;
    `output="records"` gives a list of dicts with the same keys).

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/selection.py` lines 1132-1189](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1132-L1189)

(gamrefit-bootstrap)=
## `GamRefit.bootstrap`

*Method* of `GamRefit`

```python
GamRefit.bootstrap(counts, response, weights=None)
```

Refit this Gaussian refit on each bootstrap draw of its rows.

`counts` is a `B x n` array of row multiplicities, one row per draw
(a multinomial resample gives each row a count and the counts of a draw
sum to `n`). `response` is the response of the `n` fitting rows, in
the order the refit used, and `weights` their prior weights; the
weights default to those the refit was fitted with. Draw `b` is fitted
with prior weights `weights * counts[b]`. The designs, penalties and
spline knots are the full sample's, and each draw re-estimates its
smoothing parameters by REML. Returns a `GamRefitBootstrap`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/selection.py` lines 1264-1302](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1264-L1302)

(gamrefit-to_dict)=
## `GamRefit.to_dict`

*Method* of `GamRefit`

```python
GamRefit.to_dict(*, include_arrays=False)
```

A JSON-safe record of the terms, their edf, smoothing parameters,
sigma and term tests (`tests`); `include_arrays=True` adds the
refit's coefficient vector.

Source: [`python/mboostrs/selection.py` lines 1304-1312](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1304-L1312)
