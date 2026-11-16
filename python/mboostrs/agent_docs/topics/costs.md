# Costs and budgets

Most calls in the workflow are a number of ordinary fits. `estimate_cost`
reports the fit count of a call and the time of one fit, so the product is the
call's cost before it runs. It takes the call's name and the arguments you
would pass to it, as in `mb.estimate_cost("term_report", "y ~ bbs(x0) +
bols(x1)", df, folds=5)`. It returns a record with `fits`,
`seconds_per_fit`, `estimated_seconds`, a one-line `explanation` and, for a
call that runs in stages, `stages` (the fit count of each stage). `call` is one
of `"gamboost"`, `"gamboostLSS"`, `"tune_mstop"`, `"compare_specifications"`,
`"term_report"`, `"stabsel"`, `"conformal_interval"`, `"aipw"`,
`"effect_model"`, `"ci_test"`, `"iv_effect"`, `"confounding_benchmark"` or
`"discover"`. The time is measured on your data, because cost
depends on rows, terms, basis sizes, `mstop` and the family, and a
distributional fit costs roughly one mean fit per parameter.

The fit counts are:

| Call | Fits |
|---|---|
| `gamboost`, `gamboostLSS` | 1 |
| `tune_mstop(..., folds=k)` | k + 1: one fit per fold, then one fit on the full data. Each grid's doubling resumes each fold's path where the previous grid stopped, so it adds no fits, except that a response-score fold on the Gaussian Gram route (for example a `bbs` term under `score="mse"`) is refitted from zero at each grid, adding one fit per grid per fold. `fit_budget` is checked against the worst case: k + 1 for a family-score or distributional search, and k times the grid count + 1 for a response-score search |
| `compare_specifications(specs, ..., folds=k)` | k × len(specs); with `mstop="cv"` also k inner folds per outer fit, and under a score other than `"family"` the inner folds are refitted at each grid up to `max_mstop` (default 500) |
| `term_report(..., folds=k)` | k × (1 + removals + shape candidates): the full model, one fit set per term removed, and one per shape tried for each term that has another shape to compare |
| `stabsel(model, B=b)` | 2b under `sampling_type="SS"` (each complementary pair is two fits), b under `"MB"` |
| `conformal_interval` | `"split"`: 1; `"cv+"`: `n_folds`; `"cqr"`: 2. Each tuned refit (`mstop="tune"`, or CQR without an integer `mstop`) adds 5 inner fits, and its grid doubles while the risk falls |
| `aipw(data, spec, folds=k, repeats=r)` | Nuisance models: r × k × (propensity candidates + response model when outcomes are missing under `outcome_missing="ipw"` + outcome models), plus one all-rows propensity model. Outcome models are 2 for `"separate"` and `"two_part"`, 1 for `"pooled"`, 2 for `"auto"` (an upper bound: `"auto"` may choose the pooled model after the sparse-arm check) and every candidate for `"cv"`. A model with `tune=True` costs `inner_folds` + 1 fits, an untuned one 1. Defaults are read from the call's signature |
| `effect_model(data, spec, folds=k)` | Nuisances: k × (propensity candidates + 1 outcome + response model when outcomes are missing) tuned models of 6 fits each, none when `nuisance=` is an `AIPWResult`. Second stage: k refits for each candidate in `terms`, plus k + 1 tuning fits for `"stumps"` and `"joint"`; `"smooth"` or `terms="auto"` adds the final smooth fit (an upper bound under `"auto"`). One propensity model. `interval_method="repeats"` adds `interval_repeats` − 1 and `"bootstrap"` adds `bootstrap_draws` full smooth refits of the pipeline |
| `ci_test(data, x, y, given, folds=k)` | One residual regression for the response, one for each residual function of `x` (1, 4 with the quartile hinges, or one per level of a factor) and one for the response model under `outcome_missing="ipw"` with missing outcomes; each regression is k folds of 6 tuning fits. No regression when `given` is empty. The count comes from running the test with the regressions replaced by a recorder, so it follows the reduction of the conditioning set |
| `iv_effect(data, spec, instrument, folds=k)` | k × (3 nuisance models: outcome, treatment, instrument, plus the response model under `outcome_missing="ipw"` with missing outcomes), each `inner_folds` + 1 fits when `tune=True`, 1 otherwise |
| `confounding_benchmark(data, result)` | One `aipw` refit per benchmarked covariate (`variables=`), each counted as for `aipw` with the options recorded in `result` |
| `discover(data, response)` | First pass of each stage, reported in `stages`: mean effects (terms + 1) × folds; interactions 1; spread 1 (Gaussian); tune folds + 1 per grid; final model 1; count-model forms 2 × folds. Rounds of backward elimination, interaction candidates, spread tests, the spread model's tuning and each further doubling of the step grid add fits that depend on the data. After a run, `result.stage_seconds` holds the measured seconds of each stage |
| `gam_refit`, `gamlss_refit` | 1 REML fit; its cost grows with the cube of the total number of coefficients, so a `bspatial` or `%X%` surface dominates |
| `confint(method="bootstrap", B=b)` | b, plus `B_mstop` tuning fits each |

Ways to cut a run down:

- `bins=` on the fit (or per term) quantile-bins `bols`/`bbs`/`bspatial`/
  `brandom` designs. On large data it cuts the cost of every fit several
  fold with the same selected terms; on a few thousand rows it buys little.
- Compare at a fixed `mstop` first, and tune only the candidates that
  survive. `mstop="cv"` inside `compare_specifications` multiplies the cost
  by the inner fold count.
- Pass `fit_budget=` to `compare_specifications`, `tune_mstop`, `term_report`,
  `stabsel`, `stabsel_glmboostLSS` or `stabsel_gamboostLSS`. Each one counts
  its fits from its arguments and refuses before any fit runs when the count
  is larger than the budget. The refusal names the count and the change that
  makes the call fit. `estimate_cost` returns the same count.
- Use `nu=0.1` (the default) or larger while exploring. A smaller step needs
  proportionally more iterations for the same fit.
- Thread count: the engine parallelizes the learner race; set
  `RAYON_NUM_THREADS` to share a machine.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 2000
df = pl.DataFrame({f"x{j}": rng.uniform(0, 1, n) for j in range(6)})
df = df.with_columns(y=np.sin(3 * df["x0"].to_numpy()) + rng.normal(0, 0.3, n))
rhs = " + ".join(f"bbs(x{j}, df=4)" for j in range(6))
specs = [rhs, "bbs(x0, df=4)"]

# The estimate times one fit, then counts the comparison's fits from the same
# plan compare_specifications uses.
cost = mb.estimate_cost("compare_specifications", specs, df, "y",
                        folds=5, mstop=200)
print(f"{cost.fits} fits; one fit takes {cost.seconds_per_fit:.2f} s, "
      f"so the comparison takes about {cost.estimated_seconds:.1f} s")
print(cost.explanation)

# A budget below the count refuses before any fit runs.
try:
    mb.compare_specifications(specs, df, "y", folds=5, mstop=200,
                              fit_budget=cost.fits - 1)
except ValueError as error:
    print(error)
```
