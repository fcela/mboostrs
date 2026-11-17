# `MstopTuning`

Defined in `mboostrs.explore`; exported as `mboostrs.MstopTuning`.


*Class*

```python
MstopTuning(
    mstop: int,
    risk: tuple,
    largest: int,
    first_largest: int,
    at_boundary: bool,
    fits: int,
    _fit: object = None,
    _fit_at_mstop: bool = False,
    score: str = 'family',
    score_parameter: float | None = None,
    full_data_fits: int = 2,
    fold_scores: tuple = (),
    grid_ends: tuple = (),
    patience: int | None = None
) -> None
```

A cross-validated step count, and whether the grid grew to find it.

Source: [`python/mboostrs/explore.py` lines 1296-1415](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1296-L1415)

## Members

| Member | Kind | Summary |
|---|---|---|
| `total_fits` | Property | Fold fits plus the preparation/final fits performed by the tuner. |
| `to_dict` | Method | A JSON-safe record. The full risk path is exported only on request. |
| `extended` | Property |  |
| `model` | Property | The full-data fit at the chosen step count, ready to predict. |
| `sentence` | Property |  |

(mstoptuning-total_fits)=
## `MstopTuning.total_fits`

*Property* of `MstopTuning`

```python
MstopTuning.total_fits
```

Fold fits plus the preparation/final fits performed by the tuner.

Source: [`python/mboostrs/explore.py` lines 1338-1341](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1338-L1341)

(mstoptuning-to_dict)=
## `MstopTuning.to_dict`

*Method* of `MstopTuning`

```python
MstopTuning.to_dict(*, include_arrays=False)
```

A JSON-safe record. The full risk path is exported only on request.

Source: [`python/mboostrs/explore.py` lines 1350-1357](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1350-L1357)

(mstoptuning-extended)=
## `MstopTuning.extended`

*Property* of `MstopTuning`

```python
MstopTuning.extended  # -> bool
```

*No description in the docstring.*

Source: [`python/mboostrs/explore.py` lines 1359-1361](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1359-L1361)

(mstoptuning-model)=
## `MstopTuning.model`

*Property* of `MstopTuning`

```python
MstopTuning.model
```

The full-data fit at the chosen step count, ready to predict.

For a standard specification the sweep already fitted the model on
all rows to `largest` iterations; this is that fit cut to `mstop`, so
`tune_mstop(...)` both tunes and fits in one call. For a
distributional specification the model is refitted at `mstop` (every
parameter moves by the same step count), and that refit is returned.
An offset-only fit is returned directly: its constant score path has
a positive selected grid index but the model has zero actual updates.

Source: [`python/mboostrs/explore.py` lines 1363-1381](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1363-L1381)

(mstoptuning-sentence)=
## `MstopTuning.sentence`

*Property* of `MstopTuning`

```python
MstopTuning.sentence  # -> str
```

*No description in the docstring.*

Source: [`python/mboostrs/explore.py` lines 1383-1415](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/explore.py#L1383-L1415)
