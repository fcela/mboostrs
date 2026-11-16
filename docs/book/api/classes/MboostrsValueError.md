# `MboostrsValueError`

Defined in `mboostrs._errors`; exported as `mboostrs.MboostrsValueError`.


*Class*

```python
MboostrsValueError(
    message: str = '',
    code: str | None = None,
    subject: str | None = None,
    details: str | None = None,
    rust_variant: str | None = None
)
```

A failed validation: a bad argument, shape, parameter, or response.

Source: [`python/mboostrs/_errors.py` lines 61-62](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_errors.py#L61-L62)
