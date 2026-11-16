# `CITest`

Defined in `mboostrs.causal`; exported as `mboostrs.CITest`.


*Class*

```python
CITest(
    x: str,
    y: str,
    given: tuple[str, ...],
    statistic: float,
    p_value: float,
    residual_correlation: float,
    n: int,
    folds: int,
    functions: int = 1,
    dropped: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
    skipped: bool = False,
    status: str = 'tested',
    outcome_missing: str = 'error',
    n_missing_outcome: int = 0
) -> None
```

A test of `x` independent of `y` given the columns in `given`.

`status` is `"tested"` when the test ran, and `"dependent"` when x or y
is an exact linear combination of the conditioning set. In that case
`skipped` is True, nothing was tested, and `statistic`, `p_value` and
`residual_correlation` are NaN. `notes` names the column to drop to break
the identity, the one `ci_test` suggests. Read a skipped result as
undetermined: no p-value exists for it.

Source: [`python/mboostrs/causal.py` lines 4871-4935](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4871-L4935)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#citest-to_dict) | Method |  |
| [`describe`](#citest-describe) | Method |  |

(citest-to_dict)=
## `CITest.to_dict`

*Method* of `CITest`

```python
CITest.to_dict(*, include_arrays=False)
```

*No description in the docstring.*

Source: [`python/mboostrs/causal.py` lines 4911-4912](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4911-L4912)

(citest-describe)=
## `CITest.describe`

*Method* of `CITest`

```python
CITest.describe() -> str
```

*No description in the docstring.*

Source: [`python/mboostrs/causal.py` lines 4919-4935](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L4919-L4935)
