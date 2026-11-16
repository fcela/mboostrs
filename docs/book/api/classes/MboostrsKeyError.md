# `MboostrsKeyError`

Defined in `mboostrs._errors`; exported as `mboostrs.MboostrsKeyError`.


*Class*

```python
MboostrsKeyError(
    message: str = '',
    code: str | None = None,
    subject: str | None = None,
    details: str | None = None,
    rust_variant: str | None = None
)
```

A missing map key, such as a column the model needs but the data lacks.

Source: [`python/mboostrs/_errors.py` lines 69-70](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_errors.py#L69-L70)
