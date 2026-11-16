# Quick reference: the workflow in one runnable block

Learners (`bbs`, `bols`, `brandom`, `btree`, ...) are written inside formula
strings, not imported. The field names below are the ones agents most often
guess wrong: a `ColumnKind`'s column is `.name`; an `autospec` result's
formula is `.formula`.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 400
df = pl.DataFrame({
    "x1": rng.uniform(-2, 2, n),
    "x2": rng.uniform(0, 1, n),
    "g": rng.choice(["a", "b", "c"], n),
})
df = df.with_columns(y=np.sin(df["x1"].to_numpy()) + 2 * df["x2"].to_numpy() + rng.normal(0, 0.3, n))

screen = mb.column_screen(df, "y")          # polars frame, one row per column
print(screen.columns)                        # 'column', 'curve_r2', 'linear_r2', 'hint', ...
kinds = mb.classify_columns(df, "y")        # list[ColumnKind]
for k in kinds:
    print(k.name, k.kind, k.term, k.reason)  # the column name is .name
rhs = mb.suggest_terms(df, "y")              # e.g. "bbs(x1, df=4.0) + bbs(x2, df=4.0) + bols(g)"
rec = mb.autospec(df, "y")                   # SpecRecommendation
print(rec.formula, rec.family)               # rec.report() explains measured vs defaulted
formula = f"y ~ {rhs}"
tuned = mb.tune_mstop(formula, df)           # .mstop = CV-chosen stopping iteration
model = mb.gamboost(formula, df, control=mb.boost_control(mstop=tuned.mstop))
report = mb.term_report(formula, df, response="y", mstop=tuned.mstop)
for c in report.contributions:               # what each term earns on held-out folds
    print(c)
stab = mb.stabsel(model, q=2, PFER=1)        # which inputs carry signal (recommended; the only selector with an error-rate guarantee; q must stay below the number of effects)
print(stab.selected_names)                   # the stable set
print(mb.heteroscedasticity(model))          # dict: slope, t, spread_ratio (Gaussian fits only)
for f in mb.insights(df, "y"):              # ranked candidate findings
    print(f.finding, f.effect_size, f.uncertainty, f.basis, f.next_action)
print(mb.family_set("realplus"))             # candidate LSS families for a positive response (step 8)
```

Steps 5-7 are a loop: compare, adopt at most a few changes, refit, and re-run
the screen before believing the new model is done. Steps 8-9 branch on what
the earlier steps found. Step 10 always runs last.

Signatures on demand: `mb.api_card(mb.tune_mstop)` (or
`python -m mboostrs sig tune_mstop`) prints a ≤20-line card: signature, return
type, and a minimal example.
