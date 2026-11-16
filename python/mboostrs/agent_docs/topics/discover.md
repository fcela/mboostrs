# `discover`: the rule-driven analysis

`mboostrs.discover(data, response)` runs one fixed sequence of fits and
comparisons, makes each keep-or-drop decision by a stated rule, and returns a
`DiscoveryResult`. The result holds the final model, the adopted terms with
their evidence, the interactions and log-scale drivers it adopted, the
prediction intervals for `newdata`, every warning the run raised, and a log
with one entry per decision. Use it when the question is which columns matter
and how, and the analysis has to run unattended on new data. For a hand-built
analysis, use the steps in the main workflow.

```python
import numpy as np, polars as pl
import mboostrs as mb

rng = np.random.default_rng(0)
n = 300
x = rng.uniform(-2, 2, n)
z = rng.uniform(-2, 2, n)
g = rng.integers(0, 2, n).astype(float)
junk = rng.normal(size=n)
y = np.sin(x) + 1.5 * z * g + rng.normal(0, 0.5, n) * np.exp(0.5 * x)
df = pl.DataFrame({"x": x, "z": z, "g": g, "junk": junk, "y": y})
new = pl.DataFrame({"x": [0.0, 3.0], "z": [1.0, 1.0], "g": [1.0, 1.0], "junk": [0.0, 0.0]})

result = mb.discover(df, "y", newdata=new, seed=0)
print(result.formula, "|", result.spread_formula)
for effect in result.mean_effects:
    print(effect["variables"], effect["shape"], round(effect["change"], 3))
print(result.interactions and result.interactions[0]["variables"])
print(result.spread_drivers and result.spread_drivers[0]["direction"])
print(result.predictions)
print(result.contrast("z", 0.0, 1.0).estimate)
```

## The response's meaning

The family is inferred from the response unless `family=` supplies it.
Whole numbers are treated as counts; pass `family=Gaussian()` for a rounded
continuous measurement, or `family=PropOdds()` for ordered classes coded
`1..K`. For exposure offsets and a recommendation with `report()` and `fit()`,
use `autospec`; `discover` returns a `DiscoveryResult`.

## The stages and their rules

Each stage is one call to an existing function, and each adoption is a
comparison on the same folds (`compare_specifications`). A term or pair is
adopted only if its change in out-of-fold risk clears two `change se`
(`2 * sqrt(log(5))`, about 2.54, for an interaction; see step 3). When a
candidate uses a factor, the folds keep every level of that factor in each
training fold, so a small sample does not lose a level from a training fold. A
candidate that cannot be scored is listed with its `error` and the other
candidates are still compared; the comparison raises only when none can be
scored, and the message names the cause of each.

1. Draft: a numeric column that is a copy or monotone transform of the
   response (rank correlation at least 0.999, so transforms count too) is
   dropped and named in `result.warnings`; a whole-number column with one
   value per row is warned about as a possible identifier. Then
   `classify_columns` gives one term per column. Each continuous
   column is a P-spline at `df=4`, so every smooth competes at the same
   flexibility. Two rules change that. An unordered categorical column with at
   most ten levels is a `bols` term, one coefficient per level. A positive
   column with sample skewness above 1 gets its P-spline on `log(x)`, written
   into the formula, so the log is replayed on `newdata`. Both are logged, and
   the effect records carry `transform: "log"`. Two coordinate columns stay
   separate here; `autospec` with a budget compares `btree(x, y, maxdepth=3)`
   with `bspatial(x, y)` on the pair.
2. Mean effects: each round compares the current terms with each term removed
   in turn. The least useful term whose removal does not raise the risk by more
   than two `change se` is dropped, and rounds repeat until no term qualifies.
   When every term is dropped the mean model is `y ~ 1`, and `result.warnings`
   says the mean model is intercept-only and names the family.
3. Interactions: `screen_interactions` ranks residual column pairs by their
   mean score over five row splits. The top five positive pairs are tried with
   `pair_term_for`, on each main term's scale (`log(x)` when drafted there).
   Two factors with at most ten levels each use `bols(a, by=b)`, one coefficient
   per level pair; more levels use `brandom`, with its grouped-folds caveat.
   Continuous-pair references use `bbs(x, df=2.5, knots=5)` for both main
   effects, matching the tensor margins for a matched comparison. Adoption
   requires a risk decrease exceeding `2*sqrt(log(5))`, about 2.54, `change se`.
   Unselected pairs with change and SE both zero supply no evidence.
   Intercept-only models still screen the centred response, permitting pure
   interactions such as `sin(3*x1*x2)`.
   Continuous pairs also compare a depth-three tree with the incumbent.
   It replaces the pair's tensor only above the same bar. After interactions,
   discarded draft main terms are reconsidered in draft order: a gain above
   two `change se` recovers a main effect masked earlier by the interaction.
   Refusal or failure retains the model. Tree terms have no smooth REML effect
   intervals; refit refusal is logged, and their contrasts use the boosted
   model without an interval.
4. Spread (Gaussian response only): `heteroscedasticity` gives a t statistic
   for the residual spread against every numeric predictor of the data, whether
   or not the mean model uses it. A predictor that changes the spread without
   changing the mean (a column that only sets the noise scale) is found this way.
   Each predictor is tested on the scale it is drafted on, so a right-skewed
   column is tested on `log(x)`. Predictors with `|t| >= 2` are nominated, and a
   log-scale spline for the best of them is adopted only if it beats a constant
   scale by two `change se` under the joint score of `GaussianLSS`. The log's
   spread entry lists the columns tested. Its `heteroscedasticity` result also
   has `tested_columns` and `untested_columns`, with the reason for each column
   that was not tested (a categorical column, for example).
5. Stopping iteration: `tune_mstop` for a mean model, on a grid of 100 steps.
   A grid whose risk is still falling at its end is doubled, up to `max_mstop`
   (default 400 steps for the mean model; the distributional grid starts at 400
   and goes to 1600 by default), and each doubling is admitted by the budget. A
   distributional model uses its cross-validated risk (`LSSModel.cvrisk`) with
   the same doubling, and one value for both parameters. If the risk still has
   not turned, the log records the stage with a reason beginning `boundary`,
   `result.warnings` says the same and names `discover(max_mstop=...)`, and the
   full adopted term set is kept at the stopping iteration found. A boundary
   never removes a term. Pass `max_mstop=` a larger value to search further.
6. Refit: `gam_refit` or `gamlss_refit` gives REML effect intervals and term
   tests for the adopted terms.
7. `newdata`: `extrapolation_report` lists rows outside the training range,
   predictions use `extrapolation="clip"`, and intervals come from
   `conformal_interval`.

The rules are fixed. The interaction reference and the five-split screen were
chosen on simulated draws of one benchmark generator (20 full-size and 4 null
draws), not on the data at hand; the other thresholds were not tuned. The `change se` is known to
understate the variation between data sets (see `compare_specifications`).

## The result

- `formula` is the right-hand side of the adopted mean model, and
  `spread_formula` the log-scale one, or `None`.
- `mean_effects`: one record per adopted main term, with `variables`, `shape`
  (`increasing`, `decreasing`, `u_shaped`, `inverted_u`,
  `threshold_increase`, `threshold_decrease`, `nonmonotone`, `categorical`,
  `spatial`), `range` and `slope` (from `sensitivity`), `transform` (`"log"`
  for a smooth drafted on `log(x)`, else `None`), and `change` and
  `change se`, the evidence for keeping the term. `range` and `slope` are on
  the original scale of `x`.
- `interactions` and `spread_drivers`: the same evidence. A spread driver has
  `direction`, which is `"increasing"` when the log scale rises across the
  variable's range.
- `predictions`: one row per `newdata` row with `row`, `mean`, `lower` and
  `upper`. With a spread model the frame also has `sigma`, the fitted scale
  of each row. Split intervals have the same width at every row, on the
  response scale, so a binary response can have a lower bound below 0.
- `log`: every decision, with `stage`, `candidate`, `score` (out-of-fold risk),
  `change`, `se`, `adopted` and `reason`.
- `warnings`: each distinct warning the run raised, as text.
- `to_dict()` returns JSON-ready data, with a `provenance` record that holds the
  library build, the environment and a fingerprint of the training data.
