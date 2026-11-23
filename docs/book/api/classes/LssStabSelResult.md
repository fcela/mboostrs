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

Source: [`python/mboostrs/gamlss.py` lines 5792-5884](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5792-L5884)

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

Source: [`python/mboostrs/gamlss.py` lines 5834-5836](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5834-L5836)

(lssstabselresult-names)=
## `LssStabSelResult.names`

*Property* of `LssStabSelResult`

```python
LssStabSelResult.names
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5838-5840](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5838-L5840)

(lssstabselresult-to_frame)=
## `LssStabSelResult.to_frame`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_frame(output='auto')
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5842-5862](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5842-L5862)

(lssstabselresult-plot)=
## `LssStabSelResult.plot`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5864-5867](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5864-L5867)

(lssstabselresult-to_dict)=
## `LssStabSelResult.to_dict`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the selection-probability paths only on request.

Source: [`python/mboostrs/gamlss.py` lines 5877-5884](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5877-L5884)
