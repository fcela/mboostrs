# `Bmono`

Defined in `mboostrs.learners`; exported as `mboostrs.Bmono`.


*Class*

```python
Bmono(
    name,
    constraint='increasing',
    knots=20,
    degree=3,
    differences=2,
    df=4,
    type='quad.prog',
    lambda2=1000000.0,
    niter=10,
    by=None,
    center=False
)
```

Monotone-constrained P-spline learner (R `bmono`).

At each boosting step solves the constrained QP
`min ½β'Aβ - b'β  s.t.  Dβ ≥ 0` via one of two solvers:

- `type="quad.prog"` (default, and R's default): primal active-set QP.
  Returns the exact minimiser, so the fitted coefficients satisfy `Dβ ≥ 0`
  at every boosting step and the cumulative fit satisfies the constraint
  at every iteration count. Matches R to ~5e-10 on the parity fixtures for
  every constraint shape.
- `type="iterative"`: R's `type = "iterative"` — violation-penalty fixed
  point with `λ₂ = 1e6`. About 35x faster per step (measured: 8 ms against
  282 ms for 1000 iterations on n=600), and feasible only approximately.

The per-step violation left by `"iterative"` is small but carries a
consistent sign, and boosting sums the steps, so the violation in the
cumulative fit grows with the iteration count instead of averaging out.
Fitting `constraint="increasing"` to a falling relation, the fitted curve
drops 0.0037 in total by iteration 10 and 0.675 by iteration 2000, where
the constraint permits no drop at all. `"quad.prog"` holds at exactly zero
for every iteration count, which is why it is the default. Choose
`"iterative"` only where the speed matters more than exact feasibility.

Args:
    name: predictor name in the data frame.
    constraint: one of "increasing", "decreasing", "convex", "concave",
        "positive", "negative".
    knots, degree, differences, df: same as `Bbs`.
    type: "quad.prog" (default) or "iterative". Aliases for quad.prog:
        "quadprog".
    lambda2: iterative-only; violation-penalty coefficient (default
        1e6 — matches R). Ignored when type="quad.prog".
    niter: iterative-only; max inner fixed-point iterations (default 10).
        Ignored when type="quad.prog".

Source: [`python/mboostrs/learners.py` lines 402-474](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/learners.py#L402-L474)
