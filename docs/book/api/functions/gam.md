# `gam`

Defined in `mboostrs.direct_gam`; exported as `mboostrs.gam`.


*Function*

```python
gam(
    formula,
    data,
    family='gaussian',
    weights=None,
    offset=None,
    na_action='error',
    method='REML'
)
```

Fit a penalized GAM by REML from a formula, with no boosting.

Workflow, pitfalls and worked analyses: `mboostrs.agent_guide()`, or
`python -m mboostrs guide index` from a shell. After a fit, run
`mboostrs.reference_gap` to measure what a flexible model still gains.

Every term of the formula is in the fit, and each term's smoothing
parameter is chosen by REML: the criterion of mgcv's
`gam(method="REML")` (Wood 2011), maximized by Newton's method in the
Rust core. The terms are built as `gam_refit` builds them, so the
fit uses the same designs and penalties as a refit of a boosted model whose
selected terms are the same.

**Parameters**

- `formula` (str): `"y ~ terms"`. Supported terms: `bbs(x, ...)`, a P-spline smooth (a
  bare numeric name means `bbs(name)`); `bols(x)`, a linear term (a
  bare categorical name means `bols(name)`, an unpenalized factor with
  one coefficient per level after the first); `brandom(g)`, a penalized
  random effect; `bspatial(x, y)`, a smooth surface; and `a %X% b`, a
  tensor product of two smooths. A numeric `by=` column makes a
  varying coefficient. A transform such as `log(x)` may be written
  inside a term.
- `data` (polars DataFrame, pandas DataFrame or dict of columns)
- `family` ({"gaussian", "binomial", "poisson", "gamma"} or a family object): `mb.Gaussian()`, `mb.Binomial()`, `mb.Poisson()` or `mb.GammaReg()`.
  The link is the canonical one: identity, logit, log and log.
- `weights` (array, optional): Prior weights, one per row.
- `offset` (array, optional): A per-row offset on the link scale, such as log exposure. It is added
  to the linear predictor and not estimated.
- `na_action` ({"error", "omit", "impute"}): `"error"` refuses a formula column with a missing value. `"omit"`
  drops the rows that have one. `"impute"` fills a numeric column with
  its training median and adds an indicator term for the rows that were
  missing; the fill is replayed on new rows by `GamRefit.predict`.
- `method` ({"REML"}): Only REML is available.

**Returns**

GamRefit
    With `predict` (`interval=True` and `extrapolation=`),
    `effects`, `contrast`, `coef`, `vcov`, `summary`, and the
    per-term `edf` and `smoothing`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

**Notes**

The intervals are Bayesian pointwise intervals conditional on the
smoothing parameters, as `GamRefit.effects` describes. The
smoothing parameters are estimated from the same rows the intervals use.

Source: [`python/mboostrs/direct_gam.py` lines 197-295](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/direct_gam.py#L197-L295)
