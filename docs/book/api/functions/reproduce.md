# `reproduce`

Defined in `mboostrs._procedure._reproduce`; exported as `mboostrs.reproduce`.


*Function*

```python
reproduce(
    trail: mboostrs._procedure._fit.Fit | mboostrs._procedure._trail.DecisionTrail | dict[str, typing.Any] | str | pathlib.Path,
    data: Any,
    report_format: str = 'markdown',
    verify: bool = True,
    **overrides: Any
) -> ReproductionResult
```

Reproduce a recorded `fit` and its discovery report in one call.

**Parameters**

- `trail` (Fit, DecisionTrail, dict, str, or Path): The recorded decision trail (or a `Fit` holding one, or a path to a
  saved `trail.json` file).
- `data` (DataFrame or dict): The dataset to fit on.
- `report_format` ({"markdown", "text", "json"}, default "markdown"): Format for the reproduced discovery report stored in `result.report`.
- `verify` (bool, default True): When True, compare the reproduced final specification against the
  recorded trail and populate `matched` and `differences`.
- `**overrides`: Optional keyword arguments forwarded to `fit` (overriding values
  read from the trail).

Source: [`python/mboostrs/_procedure/_reproduce.py` lines 179-238](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_procedure/_reproduce.py#L179-L238)
