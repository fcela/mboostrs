# `StabSelResult`

Defined in `mboostrs.stabsel`; exported as `mboostrs.StabSelResult`.


*Class*

```python
StabSelResult(
    phat: numpy.ndarray,
    selected: list[int],
    max_: numpy.ndarray,
    cutoff: float,
    q: int,
    PFER: float,
    p: int,
    B: int,
    sampling_type: str,
    assumption: str,
    learner_names: list[str],
    n_fits: int = 0,
    provenance: object = None,
    data_fingerprint: str = '',
    fit_selected: tuple = ()
) -> None
```

Stability-selection frequencies, selected learners, and error bound.

`B` is the number of initial subsamples (or complementary pairs under
`sampling_type="SS"`); `n_fits` is the number of models actually fit.
`PFER` is an upper bound on the expected number of falsely selected
learners, not a probability attached to any one learner.

Source: [`python/mboostrs/stabsel.py` lines 175-290](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L175-L290)

## Members

| Member | Kind | Summary |
|---|---|---|
| `max` | Property | Each learner's largest ``phat``, keyed by name. |
| `names` | Property |  |
| `selected_names` | Property | The selected learners by name (``selected`` holds their indices). |
| `variables` | Property | The data columns the selected learners read, in learner order. |
| `to_dict` | Method | A JSON-safe record; the selection-probability paths only on request. |
| `plot` | Method |  |
| `to_frame` | Method | One row per candidate learner: name, max selection probability |

(stabselresult-max)=
## `StabSelResult.max`

*Property* of `StabSelResult`

```python
StabSelResult.max
```

Each learner's largest `phat`, keyed by name.

Source: [`python/mboostrs/stabsel.py` lines 230-235](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L230-L235)

(stabselresult-names)=
## `StabSelResult.names`

*Property* of `StabSelResult`

```python
StabSelResult.names
```

*No description in the docstring.*

Source: [`python/mboostrs/stabsel.py` lines 237-239](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L237-L239)

(stabselresult-selected_names)=
## `StabSelResult.selected_names`

*Property* of `StabSelResult`

```python
StabSelResult.selected_names
```

The selected learners by name (`selected` holds their indices).

Source: [`python/mboostrs/stabsel.py` lines 241-244](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L241-L244)

(stabselresult-variables)=
## `StabSelResult.variables`

*Property* of `StabSelResult`

```python
StabSelResult.variables
```

The data columns the selected learners read, in learner order.

Source: [`python/mboostrs/stabsel.py` lines 246-251](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L246-L251)

(stabselresult-to_dict)=
## `StabSelResult.to_dict`

*Method* of `StabSelResult`

```python
StabSelResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the selection-probability paths only on request.

Source: [`python/mboostrs/stabsel.py` lines 260-268](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L260-L268)

(stabselresult-plot)=
## `StabSelResult.plot`

*Method* of `StabSelResult`

```python
StabSelResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/stabsel.py` lines 270-272](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L270-L272)

(stabselresult-to_frame)=
## `StabSelResult.to_frame`

*Method* of `StabSelResult`

```python
StabSelResult.to_frame(output='auto')
```

One row per candidate learner: name, max selection probability
over the subsampling path, and whether it was selected.

Source: [`python/mboostrs/stabsel.py` lines 274-290](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/stabsel.py#L274-L290)
