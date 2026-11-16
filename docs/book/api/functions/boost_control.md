# `boost_control`

Defined in `mboostrs._control`; exported as `mboostrs.boost_control`.


*Function*

```python
boost_control(
    mstop: int = 100,
    nu: float = 0.1,
    risk: str = 'inbag',
    trace: bool = False,
    step: str = 'fixed',
    step_iterations: int = 4
) -> BoostControl
```

The control record a fitting function takes.

`boost_control(100, 0.1)` is the default: a hundred iterations at a
step length of a tenth. See `BoostControl` for what each field
means and how to choose it.

`mstop` must be at least 1, `nu` in `(0, 1]`, and `risk` one of
`"inbag"`, `"oobag"`, `"none"`. `step` must be `"fixed"` or
`"line_search"`, and `step_iterations` at least 1.

**Parameters**

- `mstop` (int, default 100): Number of boosting iterations; at least 1.
- `nu` (float, default 0.1): Step length in `(0, 1]`.
- `risk` ({"inbag", "oobag", "none"}, default "inbag"): Which risk the recorded path holds.
- `trace` (bool, default False): Print the risk at each iteration to stderr.
- `step` ({"fixed", "line_search"}, default "fixed"): Step-length rule; see `BoostControl`.
- `step_iterations` (int, default 4): Golden-section steps per iteration under `step="line_search"`;
  at least 1.

**Returns**

BoostControl
    Named tuple with the fields above.

**Raises**

- `ValueError`: If any argument is outside the ranges above (`mstop`, `nu` and
`risk` are checked by the Rust constructor).

Source: [`python/mboostrs/_control.py` lines 81-131](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_control.py#L81-L131)
