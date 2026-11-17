# `AicResult`

Defined in `mboostrs._results`; exported as `mboostrs.AicResult`.


*Class*

```python
AicResult(
    aic: object,
    mstop: int,
    df: object,
    provenance: mboostrs._results.Provenance | None = None,
    criterion: str = 'unspecified',
    normalization: str = 'unspecified',
    df_convention: str = 'unspecified',
    penalty: float | None = None
) -> None
```

An information-criterion path: the value at each iteration, the
iteration that minimises it, and the degrees of freedom behind it.

Source: [`python/mboostrs/_results.py` lines 182-236](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_results.py#L182-L236)

## Members

| Member | Kind | Summary |
|---|---|---|
| `to_dict` | Method | A JSON-safe record; the criterion and df paths only on request. |
| `plot` | Method |  |
| `to_frame` | Method | One row per iteration: the criterion and its degrees of freedom. |

(aicresult-to_dict)=
## `AicResult.to_dict`

*Method* of `AicResult`

```python
AicResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the criterion and df paths only on request.

Source: [`python/mboostrs/_results.py` lines 203-210](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_results.py#L203-L210)

(aicresult-plot)=
## `AicResult.plot`

*Method* of `AicResult`

```python
AicResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/_results.py` lines 215-217](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_results.py#L215-L217)

(aicresult-to_frame)=
## `AicResult.to_frame`

*Method* of `AicResult`

```python
AicResult.to_frame(output='auto')
```

One row per iteration: the criterion and its degrees of freedom.

Source: [`python/mboostrs/_results.py` lines 219-236](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_results.py#L219-L236)
