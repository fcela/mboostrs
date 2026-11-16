# Fitting and tuning the stopping iteration (steps 3 and 4)

## Step 3 — Fit: `gamboost`

What it does: component-wise gradient boosting of the formula. One learner
is selected and updated per iteration, so the fit stays additive and each
term's contribution can be read separately. Call it as
`gamboost(formula, data, family=..., control=boost_control(mstop, nu))`.

What you decide: the family and the initial `mstop`. The family comes from
the response's support: continuous unbounded → `Gaussian`; 0/1 → `Binomial`;
multi-class → `Multinomial`; counts → `Poisson` (and over-dispersion is
step 8's job, not an excuse to guess `NBinomial` now); positive continuous
→ `GammaReg`; positive with mass at zero or one → say so and let `autospec`'s
family probe or `scan_lss_families` (step 8) propose the exotic ones. The
default `mstop=100` is a placeholder to be tuned away in step 4 — never
report a fit whose `mstop` you did not choose deliberately. `nu=0.1` is the
conservative default; count families with large responses may need
`step="line_search"` (see step 4).

Branches: a bare numeric name in the formula is a smooth (`bbs` at df=4),
not a line — write `bols(x)` when you mean the line. A bare categorical name
is coded as `brandom(name, lambda=0)`: one unpenalized effect per level, the
same fit `bols(name)` gives. `gamboost` warns once, naming the column; write
`bols(name)` to state that coding in the formula, or `brandom(name, df=...)`
for a ridge-penalized random effect. Group terms (`brandom`) in the
specification change what every later fold-based number means; see the
pitfalls topic (`python -m mboostrs guide pitfalls`) before fitting one.

## Step 4 — Tune the stopping iteration: `tune_mstop`

What it does: cross-validates the step count. The model is fitted once per
fold and the out-of-fold risk is read at every iteration; if the minimum is at
the edge of the grid, the grid doubles and the sweep repeats, to `max_mstop`.

What the output looks like: an `MstopTuning` with `mstop` (the selected
count), the risk curve, and `at_boundary` — true when the risk was still
falling when the grid stopped.

What you decide: refit at the returned `mstop`, or distrust it.
`at_boundary=True` means the number is where the grid stopped, not where the
data turned: raise `max_mstop`, or give `extensions` room, or pass `patience`
to stop early once the curve is flat. A fit reported at a boundary `mstop`
is unfinished, and every later number (term earnings, screens, contrasts) is
computed at an arbitrary depth.

Branches: with grouped data (a `brandom` term on district, say), pass
`folds=grouped_folds(district, B=...)` — random folds split groups across
the train/held-out line and the held-out score partly rewards memorized
level effects; the library warns when it sees this, but still computes the requested score. A grouped tune measures prediction in new groups,
which is usually the claim worth making; a row-random tune of a
group-carrying model measures mostly memory. Also choose the `score`:
`"family"` (default, negative log-likelihood on the link scale) or a
response-scale score (`"mse"`, `"brier"`, `"pinball"` with
`score_parameter=`, ...). Response scores need `unseen_levels=` set for
grouped folds (see the pitfalls topic). Distributional (mapping) specifications are
tuned with `score="family"` only.
