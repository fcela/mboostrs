# `term_report`

Defined in `mboostrs.explore`; exported as `mboostrs.term_report`.


*Function*

```python
term_report(
    formula,
    data,
    response=None,
    family=None,
    folds=5,
    mstop=500,
    nu=0.1,
    seed=0,
    shapes=True,
    df=4.0,
    weights=None,
    score='family',
    na_action='error',
    fit_budget=None,
    extrapolation='warn'
)
```

Refit a specification without each term, and each term in other shapes.

The comparison uses shared folds for all candidate specifications.

**Term contribution.** The model is refitted without each term. The
change in out-of-fold error estimates the term's predictive contribution
under this fold design; a negative removal difference favors excluding it.

**Alternative shapes.** Each term on a column with more than two
distinct values is refitted as `bols`, `bbs` and `btree` in turn, the rest
of the specification held fixed.

The cost is one cross-validation per candidate specification: the full
specification, each specification with one term removed, and each term in
each other shape. A cross-validation costs `folds` fits (more with
`mstop="cv"`), so the call costs `folds x (1 + n_terms + n_shapes)`
fits, where `n_shapes` counts the shapes tried for every term that has
one. `fit_budget`, when given, refuses the call before any fit if that
count is larger; `estimate_cost` returns the same count. The call is a
tool for a specification of tens of terms, not thousands.

Every gain carries `se`, the standard error of the paired difference in
fold scores between the two specifications. Both are scored on the same
folds, so the fold-to-fold variation they share cancels; a gain within two
standard errors of zero is not distinguishable from noise on these folds.

A group term (`brandom`) on folds that hold its groups out is reported
with `measurable=False`. Every held-out row then belongs to a group the
fold's fit never saw, so the term predicts the average group for it, and
none of the score is the term's own prediction. What the row reports is
still what removing the term costs: fitted without it, the other terms
take up the variation it was absorbing, which changes what they predict
for a new group. What these folds cannot say is whether the groups differ,
which shows on folds that keep groups together.

`folds`, `weights`, `score`, `na_action` and `mstop="cv"` are
passed to `cv_risk` and mean what they mean there: pass a
`mboostrs.folds` scheme to score on grouped or time-ordered validation
rather than a random split.

`extrapolation` is passed to the held-out predictions of each fold, as
in `cv_risk`: `"warn"` (default) reports the fold rows outside their
training range in one warning, `"clip"` holds the spline flat at the
range edge, `"error"` refuses, and `"allow"` predicts silently.

Source: [`python/mboostrs/explore.py` lines 2267-2392](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L2267-L2392)
