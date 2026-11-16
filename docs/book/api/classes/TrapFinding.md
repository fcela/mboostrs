# `TrapFinding`

Defined in `mboostrs.causal`; exported as `mboostrs.TrapFinding`.


*Class*

```python
TrapFinding(name: str, severity: str, message: str) -> None
```

One design problem a causal claim would rest on.

`severity` is `"blocking"` when the declared estimand is not
identified on this data (a column is absent, an arm is empty, one arm
of a level never appears), and `"advisory"` when the estimate is
computable but describes a smaller or different population than the
spec's wording suggests.

Source: [`python/mboostrs/causal.py` lines 86-102](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L86-L102)
