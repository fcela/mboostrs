# `api_card`

Defined in `mboostrs.api_card`; exported as `mboostrs.api_card`.


*Function*

```python
api_card(obj, *, width: int = 100) -> str
```

A bounded card for one function, method or class.

Prints the signature, a one-line return type and a minimal call in at most
`MAX_LINES` lines. `obj` is the object itself (`mb.tune_mstop`) or
its name (`"tune_mstop"`, `"GAMBoostModel.predict"`):

```python
import mboostrs as mb
print(mb.api_card(mb.tune_mstop))
```

From a shell:

```python
python -m mboostrs sig tune_mstop
```

Raises `KeyError` when a name does not resolve.

Source: [`python/mboostrs/api_card.py` lines 201-231](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/api_card.py#L201-L231)
