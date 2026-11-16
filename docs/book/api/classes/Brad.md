# `Brad`

Defined in `mboostrs.learners`; exported as `mboostrs.Brad`.


*Class*

```python
Brad(
    names,
    data=None,
    knots=100,
    df=4,
    lambda_value=None,
    theta=None,
    center=False,
    by=None
)
```

Radial-basis base-learner (R `brad`).

Places `knots` centres over the predictor space, evaluates a Matérn
covariance from every observation to every centre, and whitens the result
by the centres' own covariance so a plain ridge penalty applies:

    X = C(x, knots) @ C(knots, knots)^(-1/2),    K = I

That is the construction `mboost::brad` uses. Being a penalized linear
smoother, it lowers to `Buser` — no new engine variant — so it works
everywhere `buser` does, including cross-validation and stability
selection. The centres, range and whitening matrix are retained for
`design(newdata)` and automatic model prediction at new locations.

Use it for smooth surfaces over two or more coordinates where a
tensor-product spline (`bspatial`) is awkward: scattered spatial data,
or three-plus dimensional smooths where the tensor grid would explode.

Args:
    names: predictor column name, or a list of them for a multi-dimensional
        smooth.
    data: mapping or frame the columns are read from. `brad` has to build
        its design from the fitting data. Omit data for a deferred formula learner.
    knots: number of basis centres (default 100, capped at the number of
        distinct rows).
    df: degrees of freedom (default 4).
    lambda_value: penalty weight, used instead of calibrating from `df`.
    theta: Matérn range. Defaults to the range making correlation at the
        training data diameter equal to 1e-3.
    center: centre the design (default False).
    by: optional varying-coefficient variable name.

Note this will not reproduce R's `brad` to machine precision: R selects
centres with `fields::cover.design` and this uses greedy farthest-point
selection. Both are space-filling; neither choice is canonical.

Source: [`python/mboostrs/learners.py` lines 948-1047](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L948-L1047)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`design`](#brad-design) | Method | Whitened design at new locations, using the retained training recipe. |
| [`name`](#brad-name) | Property |  |
| [`as_buser`](#brad-as_buser) | Method | Equivalent user design with a retained radial prediction recipe. |

(brad-design)=
## `Brad.design`

*Method* of `Brad`

```python
Brad.design(newdata)
```

Whitened design at new locations, using the retained training recipe.

Returns an `(n_new, n_centres)` NumPy matrix. Fitting applies `by`
and weighted centering after this construction, as for a user design.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/learners.py` lines 1022-1031](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L1022-L1031)

(brad-name)=
## `Brad.name`

*Property* of `Brad`

```python
Brad.name
```

*No description in the docstring.*

Source: [`python/mboostrs/learners.py` lines 1033-1035](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L1033-L1035)

(brad-as_buser)=
## `Brad.as_buser`

*Method* of `Brad`

```python
Brad.as_buser(data=None)
```

Equivalent user design with a retained radial prediction recipe.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/learners.py` lines 1037-1047](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L1037-L1047)
