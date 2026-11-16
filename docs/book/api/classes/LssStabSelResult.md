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

Source: [`python/mboostrs/gamlss.py` lines 5777-5869](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5777-L5869)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`max`](#lssstabselresult-max) | Property |  |
| [`names`](#lssstabselresult-names) | Property |  |
| [`to_frame`](#lssstabselresult-to_frame) | Method |  |
| [`plot`](#lssstabselresult-plot) | Method |  |
| [`to_dict`](#lssstabselresult-to_dict) | Method | A JSON-safe record; the selection-probability paths only on request. |

(lssstabselresult-max)=
## `LssStabSelResult.max`

*Property* of `LssStabSelResult`

```python
LssStabSelResult.max
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5819-5821](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5819-L5821)

(lssstabselresult-names)=
## `LssStabSelResult.names`

*Property* of `LssStabSelResult`

```python
LssStabSelResult.names
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5823-5825](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5823-L5825)

(lssstabselresult-to_frame)=
## `LssStabSelResult.to_frame`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_frame(output='auto')
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5827-5847](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5827-L5847)

(lssstabselresult-plot)=
## `LssStabSelResult.plot`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5849-5852](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5849-L5852)

(lssstabselresult-to_dict)=
## `LssStabSelResult.to_dict`

*Method* of `LssStabSelResult`

```python
LssStabSelResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the selection-probability paths only on request.

Source: [`python/mboostrs/gamlss.py` lines 5862-5869](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5862-L5869)
