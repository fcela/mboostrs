# `Bconstrained`

Defined in `mboostrs.learners`; exported as `mboostrs.Bconstrained`.


*Class*

```python
Bconstrained(
    name,
    C,
    knots=20,
    degree=3,
    differences=2,
    df=4,
    lambda_value=None,
    type='quad.prog',
    lambda2=1000000.0,
    niter=10,
    by=None
)
```

Constrained P-spline base-learner (R `bconstrained`).

generalizes bmono to arbitrary linear constraints Cβ ≥ 0. The
constraint matrix C is supplied by the caller. For monotone
constraints, C = D (first-difference matrix); for convexity, C = D².

Each step solves the constrained least-squares problem exactly with the
primal active-set QP bmono uses by default (`type="quad.prog"`), so every
step's coefficients satisfy `Cβ >= 0`. `type="iterative"` selects the
violation-penalty fixed point instead, which is cheaper but leaves a
small same-signed violation each step that boosting accumulates (3e-6
below zero after one step on `C = I`).

Args:
    name: predictor name in the data frame.
    C: constraint matrix (m x p) such that Cβ ≥ 0.
    knots: number of inner knots (default 20).
    degree: B-spline degree (default 3).
    differences: order of the difference penalty (default 2).
    df: degrees of freedom (default 4).
    lambda_value: optional smoothing parameter. Takes precedence
        over df when supplied.
    type: "quad.prog" (default) or "iterative"; alias "quadprog".
    lambda2: violation-penalty coefficient (default 1e6). Used by
        type="iterative" only.
    niter: max inner fixed-point iterations (default 10). Used by
        type="iterative" only.
    by: optional varying-coefficient variable name.

Source: [`python/mboostrs/learners.py` lines 876-936](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L876-L936)
