# First analysis workflow

The first analysis uses `fit()` to fit a model and `report()` to summarize
its specification, criterion and decision trail. Open a follow-up topic when
the report identifies a question that needs further analysis.

The example uses `fit()` and `report()` when they are available. Otherwise it
uses `tune_mstop`, `gamboost`, `term_report` and `reference_gap` to fit and
inspect the model.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 300
df = pl.DataFrame({"x1": rng.uniform(-2, 2, n),
                   "g": rng.choice(["a", "b", "c"], n)})
df = df.with_columns(y=np.sin(df["x1"].to_numpy()) + rng.normal(0, 0.3, n))

formula = "y ~ bbs(x1, df=4.0) + bols(g)"    # or mb.autospec(df, "y").formula

if hasattr(mb, "fit"):
    fit = mb.fit(df, "y")                     # data, then the response column
    print(mb.report(fit))                     # specification, criterion, decision trail
else:                                         # same path, current primitives
    tuned = mb.tune_mstop(formula, df)
    fit = mb.gamboost(formula, df, control=mb.boost_control(mstop=tuned.mstop))
    report = mb.term_report(formula, df, response="y", mstop=tuned.mstop)
    print(mb.reference_gap(formula, df))      # does the fit still leave structure?
    for c in report.contributions:            # what each term earns, with its SE
        print(c)
```

Each follow-up call addresses a question raised by the report.

## The three common follow-ups

1. **The fit may leave structure.** `reference_gap` compares the tuned formula
   with a flexible tree booster on the same held-out rows. A gap above about
   two SE means the formula misses structure: open `modeling-tactics`. If a
   term's contribution is flat, or two variables plausibly interact, open
   `interactions-and-terms`.

2. **You must state an interval or test an effect.** A `term_report` gain
   inside two SE of zero is noise on these folds. To report an interval or a
   test, open `uncertainty`; if the same rows chose the terms, open
   `uncertainty-selection` first so the intervals account for that search.

3. **A residual pattern names a cause.** Clusters of residuals point at a
   grouping (`brandom`); a trend in time points at a smooth or lag; one or two
   rows dominating point at those rows. Open `diagnostics` for the
   pattern-to-revision guide, and `pitfalls` if a fit failed or a level was
   unseen.

## Topics by trigger

- `before-the-first-fit`: only before the first fit, to fix the target, the
  evaluation rows, the variable roles and the missing-data policy.
- `screen-and-draft`: only if `formula` was written by hand; `autospec` and
  `suggest_terms` draft it from the screen.
- `fit-and-tune`: only if `tune_mstop` is flagged `at_boundary` (unfinished).
- `modeling-tactics`: only if `reference_gap` is above about two SE.
- `interactions-and-terms`: only if a term is flat or a pair is plausible.
- `distributional-branch`: only if the spread depends on covariates
  (`heteroscedasticity` flags it).
- `sensitivity-and-contrasts`: only if the report must compare scenarios or
  rows as well as reporting effects.
- `uncertainty` / `uncertainty-selection`: only if an interval or test is
  required; use the selection-aware one when the terms were chosen.
- `diagnostics` / `reading-a-fit`: only if a residual pattern needs a revision,
  or someone must read the effects.
- `closing-the-analysis`: only when writing the final report.
- `causal`: only if the question is a treatment effect from observational data;
  do not follow this path then.

Do not read the other topics up front. The report says which question is live.
