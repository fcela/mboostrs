# `Buser`

Defined in `mboostrs.learners`; exported as `mboostrs.Buser`.


*Class*

```python
Buser(name, *, X, K, df=4, lambda_value=None, center=False, by=None)
```

User-supplied base-learner (R `buser`).

Takes an arbitrary design matrix X (n × p) and penalty matrix K (p × p),
then wraps them as a penalized linear base-learner. This is the most
flexible base-learner — users can define custom learners without writing
Rust code.

Args:
    name: base-learner name.
    X: design matrix, shape (n_obs, n_params).
    K: penalty matrix, shape (n_params, n_params).
    df: degrees of freedom (default 4). Used to compute lambda when
        lambda_value is not supplied.
    lambda_value: optional smoothing parameter lambda. Takes precedence
        over df when supplied.
    center: centre the design by removing weighted column means
        (default False).
    by: optional varying-coefficient variable name.

Source: [`python/mboostrs/learners.py` lines 869-909](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L869-L909)
