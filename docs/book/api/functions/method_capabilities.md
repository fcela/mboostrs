# `method_capabilities`

Defined in `mboostrs.capabilities`; exported as `mboostrs.method_capabilities`.


*Function*

```python
method_capabilities(
    frontend: str = 'python',
    as_frame: bool = True,
    output: str | None = None
) -> Any
```

Return public method support for `frontend='python'` or `'r'`.

**Exceptions raised in the body**

Found by reading the function's `raise` statements; conditions are in the source.

- `ValueError`

Source: [`python/mboostrs/capabilities.py` lines 117-122](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/capabilities.py#L117-L122)
