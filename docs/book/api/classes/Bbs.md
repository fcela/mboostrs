# `Bbs`

Defined in `mboostrs.learners`; exported as `mboostrs.Bbs`.


*Class*

```python
Bbs(
    name,
    knots=20,
    degree=3,
    differences=2,
    df=4,
    lambda_value=None,
    center=False,
    by=None,
    cyclic=False,
    boundary_knots=None,
    constraint=None,
    deriv=0,
    bins=None,
    knot_placement='equidistant'
)
```

Penalized B-spline (P-spline) base-learner (R `bbs`).

Expands the predictor into a B-spline basis over `knots` equally spaced
inner knots and penalizes the `differences`-order differences between
adjacent coefficients. The penalty weight is calibrated so the learner's
hat matrix has trace `df`. This is the default expansion of a bare column
name in a `gamboost` formula.

Args:
    name: predictor name in the data frame.
    knots: number of inner knots (default 20). The basis has
        `knots + degree + 1` columns.
    degree: B-spline degree (default 3, cubic).
    differences: order of the difference penalty (default 2). Its null
        space — the polynomial part the penalty cannot shrink — has
        dimension `differences`, which is the floor on `df`.
    df: degrees of freedom (default 4). Must be at least `differences`
        unless `center=True`.
    lambda_value: penalty weight, used directly instead of calibrating
        from `df`. Takes precedence over `df` when supplied.
    center: project the penalty's null space out of the design (default
        False), absorbing the polynomial part into the model intercept.
    by: optional varying-coefficient variable name. Multiplies the basis
        by that column, fitting a curve whose height varies with it. A
        factor `by` stacks the basis once per non-reference level instead,
        so the term fits each level's difference from the first, and needs
        `df` at least `differences * (levels - 1)`. `gamboost` builds that;
        `glmboost` and the LSS frontends refuse it.
    cyclic: wrap the basis at the boundary knots so the fitted curve and
        its derivatives match at both ends (default False).
    boundary_knots: `(lo, hi)` pair bounding the basis. Defaults to the
        observed range of the predictor.
    constraint: optional shape constraint, one of `VALID_CONSTRAINTS`.
    deriv: order of derivative to evaluate the basis at (default 0, the
        function itself).
    bins: optional quantile-bin count. At least one bin per distinct value
        reproduces the exact design; smaller values approximate it.
    knot_placement: how inner knots are placed, `"equidistant"` (default,
        matching R `mboost::bbs` with a finite-difference penalty) or
        `"quantile"` (an O-spline with deduplicated empirical-quantile
        inner knots and an exact integrated second-derivative curvature
        penalty `integral B''(x) B''(x)^T dx` that preserves raw-scale
        linearity in the penalty null space).

Source: [`python/mboostrs/learners.py` lines 291-399](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L291-L399)
