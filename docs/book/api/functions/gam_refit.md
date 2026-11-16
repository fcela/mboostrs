# `gam_refit`

Defined in `mboostrs.selection`; exported as `mboostrs.gam_refit`.


*Function*

```python
gam_refit(model, terms=None, *, which='selected', unsupported='error')
```

Refit a boosted model's terms as one penalized GAM with REML smoothing.

Boosting shrinks every effect and couples selection with smoothing, so a
bootstrap of the boosted fit reproduces the shrinkage and its bands
under-cover (62% for nominal 90% on `benches/bench_insights.py`'s additive
task). This refits the chosen terms jointly on the same designs and
penalties the learners use: a penalized least-squares fit with one
smoothing parameter per term estimated by REML, the construction of
mgcv's `gam(method="REML")` (Wood 2011), and pointwise Bayesian intervals
from the coefficient covariance.

**Parameters**

- `model` (GAMBoostModel): A `gamboost` fit with the Gaussian, Binomial, Poisson, NBinomial,
  GammaReg or CoxPH family. Effects are on the canonical link scale:
  log-odds for Binomial (mboostrs' own Binomial predictor is half the
  log-odds), the log of the mean for Poisson, NBinomial and GammaReg,
  and the log hazard ratio for CoxPH (fitted through the Poisson form of
  the Cox likelihood, alternating with a Breslow baseline hazard).
- `terms` (sequence of str, optional): Formula terms to refit. By default the terms of the model's formula
  whose learners were selected (`which="selected"`) or all of them
  (`which="all"`); pass the output of `deselect(...).kept` to refit a
  deselected model.
- `unsupported` ({"error", "hold"}): What to do with a term the refit cannot represent (`btree`,
  `bmono`, `bkernel` ...). `"error"` refuses it by name;
  `"hold"` keeps it at its boosted contribution, entered as a known
  offset, and refits the other terms around it. Held terms are listed in
  `GamRefit.held`, included in `predict` and absent from
  `effects`, and the intervals are conditional on them.

**Returns**

GamRefit

Supported terms are `bbs(x, ...)`, `bols(x)` and `brandom(g, ...)`
over one column, each optionally with a numeric `by=` column (a
varying coefficient), and `bspatial`/`%X%` tensor terms; any other
term is refused by name. A centred smooth `bbs(x, center=TRUE)` is
refitted as `bbs(x)`, which absorbs a `bols(x)` term over the same
column: the refit's penalty leaves the linear part free. Non-Gaussian families are fitted by penalized
iteratively reweighted least squares, and their smoothing parameters
minimize the Laplace-approximate REML criterion, as mgcv does; the
negative binomial theta and the gamma shape are estimated by the same
criterion. The scale is fixed at 1 for Binomial, Poisson and Cox. The intervals are
conditional on the chosen terms: when the terms were selected on the same
data, they do not account for that selection.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `TypeError`
- `NotImplementedError`
- `ValueError`

Source: [`python/mboostrs/selection.py` lines 1664-1847](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/selection.py#L1664-L1847)
