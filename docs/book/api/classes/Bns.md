# `Bns`

Defined in `mboostrs.learners`; exported as `mboostrs.Bns`.


*Class*

```python
Bns(name, *, knots=20, degree=3, df=None, center=False, by=None)
```

Ridge-penalized B-spline base-learner (mboostrs extension).

Constructs a B-spline basis from the predictor and fits it with an
identity (ridge) penalty. Because the basis is not penalized by a
difference operator, bns uses more degrees of freedom than bbs and
produces a less smooth fit.

Current upstream `mboost::bns()` reports that the learner was removed
and recommends `bbs`. This class retains the historical name for
compatibility but is not an upstream-equivalent natural spline.

Args:
    name: predictor name in the data frame.
    knots: number of inner knots (default 20).
    degree: B-spline degree (default 3, cubic).
    df: degrees of freedom (default = ncol(basis), i.e. effectively
        unpenalized). Reducing df adds ridge regularization.
    center: centre the design by spectral projection (default False).
    by: optional varying-coefficient variable name.

Source: [`python/mboostrs/learners.py` lines 733-767](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L733-L767)
