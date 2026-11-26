# `diagonal_grid`

Defined in `mboostrs.gamlss`; exported as `mboostrs.diagonal_grid`.


*Function*

```python
diagonal_grid(max_mstop, n_params, length_out=10, log=True)
```

A grid whose points give every distribution parameter the same budget.

Cross-validating an LSS fit means searching over one `mstop` per
distribution parameter, and the full Cartesian product of `length_out`
marginal values costs `length_out ** n_params` fits per fold -- 100 for
two parameters, 1000 for three.

The diagonal is the subset where all the budgets agree, and it is cheap
for a mechanical reason: the engine fits **one** path per fold to the
largest budget and reads every diagonal point off it as a checkpoint. Ten
diagonal points cost one path per fold rather than ten.

Nothing off the diagonal can be read off that path. A cyclic LSS fit
updates the parameters in turn, so freezing one changes what the others do
next: the state after 20 mu-steps and 5 sigma-steps is not a prefix of any
path that took 20 of each, and it needs its own fit.

Start on the diagonal to find the scale, then search a small product grid
around the winner if the parameters plainly want different budgets --
which shows up as the diagonal optimum sitting at one end of its range.

Source: [`python/mboostrs/gamlss.py` lines 6789-6812](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/gamlss.py#L6789-L6812)
