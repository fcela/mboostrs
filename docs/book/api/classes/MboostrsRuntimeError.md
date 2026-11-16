# `MboostrsRuntimeError`

Defined in `mboostrs._errors`; exported as `mboostrs.MboostrsRuntimeError`.


*Class*

```python
MboostrsRuntimeError(
    message: str = '',
    code: str | None = None,
    subject: str | None = None,
    details: str | None = None,
    rust_variant: str | None = None
)
```

A numerical or solver failure in the Rust engine.

Source: [`python/mboostrs/_errors.py` lines 65-66](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_errors.py#L65-L66)
