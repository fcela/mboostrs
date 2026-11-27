# `Brandom`

Defined in `mboostrs.learners`; exported as `mboostrs.Brandom`.


*Class*

```python
Brandom(name, *, df=4, lambda_value=None, center=False, by=None, bins=None)
```

Random-effects base-learner (R `brandom`).

Fits a factor as random effects: expands the factor into an
indicator design matrix (one column per level, no reference
dropped — equivalent to R's `contrasts.arg = "contr.dummy"`),
then applies a ridge penalty. This is identical to `bols` on
the indicator matrix, so internally it maps to the BlockBols path.

Args:
    name: predictor name in the data frame (must be a factor or
        integer column whose unique values are the levels).
    df: degrees of freedom (default 4). Used to compute lambda
        when lambda is not supplied.
    lambda_value: optional smoothing parameter lambda. Takes
        precedence over df when supplied.
    center: centre the design by removing the weighted column
        means (default False). When True, the constant effect
        is absorbed into the intercept.
    by: optional varying-coefficient variable name.

Source: [`python/mboostrs/learners.py` lines 580-617](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L580-L617)
