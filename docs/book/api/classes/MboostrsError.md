# `MboostrsError`

Defined in `mboostrs._errors`; exported as `mboostrs.MboostrsError`.


*Class*

```python
MboostrsError(
    message: str = '',
    code: str | None = None,
    subject: str | None = None,
    details: str | None = None,
    rust_variant: str | None = None
)
```

Base class for every mboostrs error, carrying error attributes.

Subclasses add the built-in exception they replace as one of their bases.
`MboostrsError` alone is never raised; use a subclass so ``except
ValueError`` and friends keep matching.

Source: [`python/mboostrs/_errors.py` lines 28-58](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_errors.py#L28-L58)
