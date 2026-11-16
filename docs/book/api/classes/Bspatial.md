# `Bspatial`

Defined in `mboostrs.learners`; exported as `mboostrs.Bspatial`.


*Class*

```python
Bspatial(
    name1,
    name2,
    knots=20,
    degree=3,
    differences=2,
    df=6,
    lambda_value=None,
    center=False,
    by=None,
    bins=None
)
```

Tensor-product P-spline base-learner (R `bspatial`).

Fits a smooth 2-D surface via the Kronecker product of two marginal
B-spline bases. The penalty is the additive Kronecker penalty
K1 ⊗ I2 + I1 ⊗ K2.

In R, `bspatial(x1, x2, ...)` is a thin wrapper around `bbs` that
detects two positional arguments and switches to tensor-product mode.
Here we make the two-coordinate interface explicit.

Args:
    name1: name of the first coordinate column in the data frame.
    name2: name of the second coordinate column in the data frame.
    knots: number of inner knots for each marginal (default 20).
    degree: B-spline degree for each marginal (default 3).
    differences: order of the difference penalty for each marginal
                 (default 2).
    df: degrees of freedom (default 6, matching R `bspatial`).
    lambda_value: optional smoothing parameter lambda. Takes precedence
                  over df when supplied.
    center: centre the design by spectral projection (default False).
    by: optional varying-coefficient variable name.
    bins: enable exact compact factor storage. Must be at least the number
        of observed levels; factor levels are never merged.

Source: [`python/mboostrs/learners.py` lines 496-553](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L496-L553)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`name`](#bspatial-name) | Property |  |

(bspatial-name)=
## `Bspatial.name`

*Property* of `Bspatial`

```python
Bspatial.name
```

*No description in the docstring.*

Source: [`python/mboostrs/learners.py` lines 551-553](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L551-L553)
