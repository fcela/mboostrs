# `SplitRefitResult`

Defined in `mboostrs.split_refit`; exported as `mboostrs.SplitRefitResult`.


*Class*

```python
SplitRefitResult(
    formula: object,
    family: str,
    select: str,
    repeats: int,
    fraction: float,
    level: float,
    seed: int,
    scale: str,
    terms: tuple,
    contrasts: tuple,
    splits: tuple = ()
) -> None
```

The output of `split_refit`: the aggregate, and the per-repeat records.

`terms` lists every term chosen in at least one repeat. `contrasts`
holds one `ContrastSummary` per requested contrast. `splits`
holds one `SplitFit` per repeat.

Source: [`python/mboostrs/split_refit.py` lines 117-189](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/split_refit.py#L117-L189)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`to_dict`](#splitrefitresult-to_dict) | Method | A JSON-safe record. ``include_arrays=True`` adds one entry per repeat |
| [`summary`](#splitrefitresult-summary) | Method | Selection frequencies and contrast intervals, one line each. |

(splitrefitresult-to_dict)=
## `SplitRefitResult.to_dict`

*Method* of `SplitRefitResult`

```python
SplitRefitResult.to_dict(*, include_arrays=False)
```

A JSON-safe record. `include_arrays=True` adds one entry per repeat
under `"splits"` (the rows used, the terms chosen, the term p-values and
each contrast's per-split interval); by default only the aggregate is
returned.

Source: [`python/mboostrs/split_refit.py` lines 140-163](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/split_refit.py#L140-L163)

(splitrefitresult-summary)=
## `SplitRefitResult.summary`

*Method* of `SplitRefitResult`

```python
SplitRefitResult.summary()
```

Selection frequencies and contrast intervals, one line each.

Source: [`python/mboostrs/split_refit.py` lines 165-185](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/split_refit.py#L165-L185)
