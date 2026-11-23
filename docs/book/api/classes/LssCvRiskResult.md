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

Source: [`python/mboostrs/gamlss.py` lines 5629-5789](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5629-L5789)

## Members

| Member | Kind | Summary |
|---|---|---|
| `standard_error` | Property | Standard error of the mean risk at each grid point. |
| `mstop_1se` | Property | The one-standard-error rule over the grid. |
| `report` | Method | Fold design, grid, optimum, held-out score and boundary flag as a |
| `at_boundary` | Method | Whether the best grid point is the largest budget searched. |
| `grid_cost` | Method | How many fits this grid actually cost, per fold. |
| `to_frame` | Method |  |
| `plot` | Method |  |
| `to_dict` | Method | A JSON-safe record; the raw fold-risk matrix only on request. |

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

Source: [`python/mboostrs/gamlss.py` lines 5659-5671](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5659-L5671)

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

Source: [`python/mboostrs/gamlss.py` lines 5673-5689](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5673-L5689)

(lsscvriskresult-report)=
## `LssCvRiskResult.report`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.report()
```

Fold design, grid, optimum, held-out score and boundary flag as a
`mboostrs.cvrisk.CvReport`, the shape a standard result reports
too; the optimum is `{parameter: steps}`.

Source: [`python/mboostrs/gamlss.py` lines 5691-5707](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5691-L5707)

(lsscvriskresult-at_boundary)=
## `LssCvRiskResult.at_boundary`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.at_boundary() -> bool
```

Whether the best grid point is the largest budget searched.

The minimum lies at a grid point with the largest total budget.
Extend the grid to check whether risk declines further.

Source: [`python/mboostrs/gamlss.py` lines 5709-5719](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5709-L5719)

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

Source: [`python/mboostrs/gamlss.py` lines 5721-5746](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5721-L5746)

(lsscvriskresult-to_frame)=
## `LssCvRiskResult.to_frame`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.to_frame(output='auto')
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5748-5765](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5748-L5765)

(lsscvriskresult-plot)=
## `LssCvRiskResult.plot`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.plot()
```

*No description in the docstring.*

Source: [`python/mboostrs/gamlss.py` lines 5768-5771](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5768-L5771)

(lsscvriskresult-to_dict)=
## `LssCvRiskResult.to_dict`

*Method* of `LssCvRiskResult`

```python
LssCvRiskResult.to_dict(*, include_arrays=False)
```

A JSON-safe record; the raw fold-risk matrix only on request.

Source: [`python/mboostrs/gamlss.py` lines 5779-5789](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L5779-L5789)
