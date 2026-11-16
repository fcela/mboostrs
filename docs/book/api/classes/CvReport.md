# `CvReport`

Defined in `mboostrs.cvrisk`; exported as `mboostrs.CvReport`.


*Class*

```python
CvReport(
    folds: str,
    n_folds: int,
    grid: tuple,
    optimum: object,
    risk_at_optimum: float,
    score: str,
    at_boundary: bool,
    preprocessing: str = 'fixed'
) -> None
```

Summary of a cross-validation run for standard and distributional models.

`folds` describes the fold design in words and `n_folds` counts its
columns. `grid` lists the budgets evaluated: step counts for a standard
model, one tuple per grid point (a budget per parameter) for a
distributional one. `optimum` is the chosen budget, an int or
`{parameter: steps}`; `risk_at_optimum` is the mean held-out risk
there, in the units `score` names. `at_boundary` is true when the optimum
is the largest budget searched, so the search stopped at the grid's edge.

Source: [`python/mboostrs/cvrisk.py` lines 68-88](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/cvrisk.py#L68-L88)
