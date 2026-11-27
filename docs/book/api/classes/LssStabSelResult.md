# `LssStabSelResult`

Defined in `mboostrs.gamlss`; exported as `mboostrs.LssStabSelResult`.


*Class*

```python
LssStabSelResult(
    phat: numpy.ndarray,
    selected: list[str],
    max_: numpy.ndarray,
    cutoff: float,
    q: int,
    PFER: float,
    learner_names: list[str],
    sampling_type: str = 'SS',
    assumption: str = 'unimodal',
    B: int = 0,
    n_fits: int = 0,
    provenance: object = None
) -> None
```

Stability-selection result for combined LSS learner/parameter effects.

Source: [`python/mboostrs/gamlss.py` lines 5905-5997](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5905-L5997)

## Members

| Member | Kind | Summary |
|---|---|---|
| `max` | Property |  |
| `names` | Property |  |
| `to_frame` | Method |  |
| `plot` | Method |  |
| `to_dict` | Method | A JSON-safe record; the selection-probability paths only on request. |

(lssstabselresult-max)=
## `LssStabSelResult.max`

*Property* of `LssStabSelResult`

```python
LssStabSelResult.max
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5947-5949](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5947-L5949)

(lssstabselresult-names)=
## `LssStabSelResult.names`

*Property* of `LssStabSelResult`

```python
LssStabSelResult.names
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5951-5953](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5951-L5953)

(lssstabselresult-to_frame)=
## `LssStabSelResult.to_frame`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_frame(output='auto')
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5955-5975](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5955-L5975)

(lssstabselresult-plot)=
## `LssStabSelResult.plot`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5977-5980](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5977-L5980)

(lssstabselresult-to_dict)=
## `LssStabSelResult.to_dict`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the selection-probability paths only on request.

Source: [`python/mboostrs/gamlss.py` lines 5990-5997](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5990-L5997)
