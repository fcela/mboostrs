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

Source: [`python/mboostrs/gamlss.py` lines 5869-5961](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5869-L5961)

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

Source: [`python/mboostrs/gamlss.py` lines 5911-5913](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5911-L5913)

(lssstabselresult-names)=
## `LssStabSelResult.names`

*Property* of `LssStabSelResult`

```python
LssStabSelResult.names
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5915-5917](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5915-L5917)

(lssstabselresult-to_frame)=
## `LssStabSelResult.to_frame`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_frame(output='auto')
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5919-5939](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5919-L5939)

(lssstabselresult-plot)=
## `LssStabSelResult.plot`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5941-5944](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5941-L5944)

(lssstabselresult-to_dict)=
## `LssStabSelResult.to_dict`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the selection-probability paths only on request.

Source: [`python/mboostrs/gamlss.py` lines 5954-5961](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5954-L5961)
