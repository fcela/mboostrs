# Uncertainty after selection: terms chosen on the same data

## Terms selected on fitting data

The intervals in `uncertainty` are conditional on the chosen terms. When the same rows
chose the terms, they ignore that choice and are too narrow for the effects
boosting kept. `mb.split_refit` removes the dependence: a random half of the
rows chooses the terms, and the other half refits only those terms by REML
(`gam_refit` for a `gamboost` formula, `gamlss_refit` for a mapping of
distribution parameters). The refit's intervals and tests are then computed
on rows that played no part in the choice.

- `select="deselect"` (the default) keeps the terms `mb.deselect` keeps;
  `select="stabsel"` keeps the terms `mb.stabsel` selects; a callable
  `select(data)` returns the terms to keep. When the question is which
  inputs carry signal, prefer `select="stabsel"`: it is the only selector
  of the three with an error-rate guarantee (PFER bounds the expected
  number of false learners), while `deselect` and `probe` carry none.
  `deselect` stays the default because it is far cheaper — a single fit
  against `stabsel`'s 100.
- `repeats=` draws that many splits. Medians of the per-split estimates and
  bounds are reported with the selection frequency of each term, and the
  p-values are aggregated by the Meinshausen, Meier and Bühlmann (2009) rule.
- Each refit sees only the inference half (`fraction=0.5` by default), so a
  split's interval is wider than one from all rows. A refit that cannot
  represent a selected term raises and names the term.
- Selective inference, which conditions on the whole boosting path, is not
  implemented.

Two learners that carry the same information can split the selections between
them. Each then falls below the `stabsel` cutoff while one of them is selected
in nearly every subsample. `mb.substitution_sets(stab)` reports such groups
with the group's frequency next to each member's frequency, and
`mb.equivalence_sets(model)` lists the learners whose design spaces overlap at
0.9 or more. `mb.selection_report(model, deselection=..., stability=...)`
combines both with the terms `deselect` dropped. These are predictive
statements about the rows the fit used: a group does not identify which member
is causal.

```python
import numpy as np
import mboostrs as mb

rng = np.random.default_rng(0)
n = 400
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
data = {"y": 2 * x + rng.normal(0, 1, n), "x": x, "z": z}

# Each split chooses terms on 200 rows and refits them on the other 200.
result = mb.split_refit("y ~ bbs(x) + bbs(z)", data, repeats=5, seed=0,
                        contrasts=[("x", 0.0, 1.0)])
print(result.summary())
contrast = result.contrasts[0]
print(contrast.estimate, contrast.lower, contrast.upper, contrast.selection_frequency)
```

Read the contrast as the average, over the inference rows, of the change in
the predictor when `x` moves from 0 to 1. Its interval is the 90% interval
for that average. A variable no kept term reads has a contrast of exactly
zero, and its selection frequency says how often that happened.
