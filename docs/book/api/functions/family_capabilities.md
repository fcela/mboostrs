# `family_capabilities`

Defined in `mboostrs.capabilities`; exported as `mboostrs.family_capabilities`.


*Function*

```python
family_capabilities(*, as_frame: bool = True, output: str | None = None) -> Any
```

Return family support and evidence rows for both frontends.

The legacy yes/no columns remain for compatibility. The
`python_*_status` and `r_*_status` columns use the checked vocabulary:
`verified`, `smoke`, `intentional-deviation`, `unsupported`, or
`unverified`.

Source: [`python/mboostrs/capabilities.py` lines 99-108](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/capabilities.py#L99-L108)
