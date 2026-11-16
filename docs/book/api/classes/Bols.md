# `Bols`

Defined in `mboostrs.learners`; exported as `mboostrs.Bols`.


*Class*

```python
Bols(name, *, intercept=True, df=None, lambda_value=None, by=None, bins=None)
```

Ordinary least-squares base-learner (R `bols`).

Fits the predictor linearly: the design is the column itself (plus an
intercept column unless `intercept=False`), and each boosting step solves
an unpenalized least-squares problem on it. A categorical column expands to
one indicator per level. This is the only learner `glmboost` uses, and a
bare column name in a `glmboost` formula expands to `bols(name)`.

Args:
    name: predictor name in the data frame.
    intercept: include a per-learner intercept column (default True).
    df: degrees of freedom. Supplying it adds a ridge penalty calibrated
        so the learner's hat matrix has this trace; omitted (the default),
        the learner is unpenalized.
    lambda_value: ridge penalty, used directly instead of calibrating from
        `df`. Takes precedence over `df` when supplied.
    by: optional varying-coefficient variable name. `bols(x, by=z)`
        multiplies the whole design by `z`, so the learner fits
        `z * (b0 + b1 * x)` and carries the `z` main effect with it. A
        factor `by` has no single column to multiply by: the design is
        stacked once per non-reference level, so the term fits each
        level's difference from the first. `gamboost` builds that;
        `glmboost` and the LSS frontends refuse it.
    bins: optional quantile-bin count for compact training storage.
        Currently requires `intercept=False` and no `by=` term.

Source: [`python/mboostrs/learners.py` lines 245-288](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L245-L288)
