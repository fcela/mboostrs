# `LssCvRiskResult`

Defined in `mboostrs.gamlss`; exported as `mboostrs.LssCvRiskResult`.


*Class*

```python
LssCvRiskResult(
    risk_matrix: numpy.ndarray,
    aggregate: numpy.ndarray,
    grid: list[list[int]],
    optimal_mstop: list[int],
    fold_type: str = 'custom',
    provenance: object = None,
    param_names: tuple = (),
    data_identity: str = ''
) -> None
```

Cross-validated LSS risk over a per-parameter mstop grid.

Source: [`python/mboostrs/gamlss.py` lines 5614-5774](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5614-L5774)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`standard_error`](#lsscvriskresult-standard_error) | Property | Standard error of the mean risk at each grid point. |
| [`mstop_1se`](#lsscvriskresult-mstop_1se) | Property | The one-standard-error rule over the grid. |
| [`report`](#lsscvriskresult-report) | Method | Fold design, grid, optimum, held-out score and boundary flag as a |
| [`at_boundary`](#lsscvriskresult-at_boundary) | Method | Whether the best grid point is the largest budget searched. |
| [`grid_cost`](#lsscvriskresult-grid_cost) | Method | How many fits this grid actually cost, per fold. |
| [`to_frame`](#lsscvriskresult-to_frame) | Method |  |
| [`plot`](#lsscvriskresult-plot) | Method |  |
| [`to_dict`](#lsscvriskresult-to_dict) | Method | A JSON-safe record; the raw fold-risk matrix only on request. |

(lsscvriskresult-standard_error)=
## `LssCvRiskResult.standard_error`

*Property* of `LssCvRiskResult`

```python
LssCvRiskResult.standard_error  # -> np.ndarray
```

Standard error of the mean risk at each grid point.

The across-fold standard deviation divided by the square root of the
fold count. It describes variation across these folds; because the
folds share training rows, it is not a confidence bound for variation
across datasets.

Source: [`python/mboostrs/gamlss.py` lines 5644-5656](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5644-L5656)

(lsscvriskresult-mstop_1se)=
## `LssCvRiskResult.mstop_1se`

*Property* of `LssCvRiskResult`

```python
LssCvRiskResult.mstop_1se  # -> list[int]
```

The one-standard-error rule over the grid.

Adds the standard error at the minimum to the minimum risk, then
returns the grid point with the smallest total budget whose risk is
within that threshold.

Source: [`python/mboostrs/gamlss.py` lines 5658-5674](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5658-L5674)

(lsscvriskresult-report)=
## `LssCvRiskResult.report`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.report()
```

Fold design, grid, optimum, held-out score and boundary flag as a
`mboostrs.cvrisk.CvReport`, the shape a standard result reports
too; the optimum is `{parameter: steps}`.

Source: [`python/mboostrs/gamlss.py` lines 5676-5692](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5676-L5692)

(lsscvriskresult-at_boundary)=
## `LssCvRiskResult.at_boundary`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.at_boundary() -> bool
```

Whether the best grid point is the largest budget searched.

The minimum lies at a grid point with the largest total budget.
Extend the grid to check whether risk declines further.

Source: [`python/mboostrs/gamlss.py` lines 5694-5704](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5694-L5704)

(lsscvriskresult-grid_cost)=
## `LssCvRiskResult.grid_cost`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.grid_cost() -> dict
```

How many fits this grid actually cost, per fold.

Grid points where every parameter shares one budget are **diagonal**,
and the engine reads them off a single path fitted to the largest of
them: one path serves all of them. Every other point needs its own
fit, because a cyclic LSS path is not prefix-consistent -- freezing
one parameter changes the later updates of the others, so the state
after 20 mu-steps and 5 sigma-steps is not any prefix of a path that
took 20 of each.

So a grid of ten diagonal points costs one path per fold, and a
10 x 10 product grid costs a hundred. That is the difference this
reports.

Source: [`python/mboostrs/gamlss.py` lines 5706-5731](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5706-L5731)

(lsscvriskresult-to_frame)=
## `LssCvRiskResult.to_frame`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.to_frame(output='auto')
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5733-5750](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5733-L5750)

(lsscvriskresult-plot)=
## `LssCvRiskResult.plot`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5753-5756](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5753-L5756)

(lsscvriskresult-to_dict)=
## `LssCvRiskResult.to_dict`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the raw fold-risk matrix only on request.

Source: [`python/mboostrs/gamlss.py` lines 5764-5774](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5764-L5774)
