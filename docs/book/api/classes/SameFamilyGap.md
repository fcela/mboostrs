# `SameFamilyGap`

Defined in `mboostrs._same_family`; exported as `mboostrs.SameFamilyGap`.


*Class*

```python
SameFamilyGap(
    available: bool,
    family: str,
    mstop: object,
    mstop_source: str,
    boost_loglik: float | None,
    comparison_loglik: float | None,
    gap: float | None,
    material: bool,
    comparison: str,
    message: str,
    tol: float = 1.0,
    n_obs: int | None = None
) -> None
```

Log-likelihood gap between a boost and the same-family GLM.

`gap` is `comparison_loglik - boost_loglik`, in nats, on the same
rows. Positive means the boost is short of the comparison. `material`
is true only when that gap exceeds `tol` and the comparison is a
completed maximum of the same likelihood `logLik` reports. The selected
step is never changed.

`mstop` is the cross-validated step when `cv` was passed, otherwise
the fit's own step. `mstop_source` says which.

Source: [`python/mboostrs/_same_family.py` lines 39-116](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_same_family.py#L39-L116)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method |  |
| `to_frame` | Method | One row. ``mstop`` is written as text when it is not a single step. |

(samefamilygap-to_dict)=
## `SameFamilyGap.to_dict`

*Method* of `SameFamilyGap`

```python
SameFamilyGap.to_dict(*, include_arrays: bool = False) -> dict
```

*No description in the docstring.*

Source: [`python/mboostrs/_same_family.py` lines 82-107](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_same_family.py#L82-L107)

(samefamilygap-to_frame)=
## `SameFamilyGap.to_frame`

*Method* of `SameFamilyGap`

```python
SameFamilyGap.to_frame(output='auto')
```

One row. `mstop` is written as text when it is not a single step.

Source: [`python/mboostrs/_same_family.py` lines 109-116](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_same_family.py#L109-L116)
