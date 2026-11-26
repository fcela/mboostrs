---
name: analyzing-data-with-mboostrs
description: Fit, tune, diagnose, interpret, and save mboostrs models in Python. Use for additive boosting, REML GAMs, distributional regression, and the package's causal estimators.
---

# Analyzing data with mboostrs

Use the API installed in the working environment. This guide ships inside the
wheel; no repository checkout or internet access is needed. Start with
[the runnable first analysis](topics/golden-path.md) (`"golden-path"`). It
reserves test rows, tunes a boosted model, predicts, and checks the result.
Adapt its response, predictors, family, and split to the user's task.

## Find the exact call before using it

```sh
python -m mboostrs guide golden-path
python -m mboostrs sig tune_mstop
python -m mboostrs sig GAMBoostModel.predict
python -m mboostrs guide index
```

In Python, `mb.agent_guide("golden-path")` returns the same text;
`mb.api_card("tune_mstop")` gives the signature, return type, and example.
Use `help(mb.tune_mstop)` for details omitted by the short card. Inspect the
returned object's type and documented fields instead of guessing sklearn
methods. Fitting takes a formula and data together; prediction is
`model.predict(newdata=rows, type="response")`.

## Choose the fitting route

| User's task | Start with | Read next |
|---|---|---|
| Additive prediction, shrinkage, or learner selection | `mb.gamboost` + `mb.tune_mstop`; `mb.glmboost` for linear learners | [first analysis](topics/golden-path.md) |
| Penalized smooth regression with REML and effect intervals, without a boosting path | `mb.gam` | [direct REML](topics/direct-reml-gam.md) |
| Model location and scale or other distribution parameters | `mb.gamboostLSS` (boosting) or `mb.gamlss` (REML) | [distributional branch](topics/distributional-branch.md), [direct REML](topics/direct-reml-gam.md) |
| Automated exploratory search requested by the user | `mb.discover` | [discover](topics/discover.md), [limits](topics/discover-limits.md) |
| Observational treatment effect | The estimators and assumptions in the causal route | [causal](topics/causal.md) |

A supplied formula is a starting specification; do not replace it with an
automatic search unless the task calls for that. `autospec` can draft a
formula when one is needed (pass `search="fast"` for the single-pass radar).
Its recommendation has `.formula`, `.family`, and `.report()`; it is not
evidence that every proposed term matters.

## Limits of the convenience wrapper

`mb.fit(data, "y")` and `mb.report(result)` wrap `autospec` (defaulting to
`na_action="impute"`, `interactions="adopt"`, `residual_tree="auto"`, and
`unseen_levels="average"` on the returned model) with a decision trail.
`budget="quick"/"standard"/"thorough"` gives 16, 64 or 512 fits (a stage in
progress can exceed this) and 60, 120 or 1200 seconds (`"standard"` is the
default). Above 3,000 rows without explicit `folds`/`groups`, search runs on
a 3,000-row subsample preserving response extremes and refits on all rows.
`result.trail.final["formula"]` keeps terms carrying at least 1% of risk
reduction (or LSS effect variance), or intercept-only on a flat CV curve.
`result.trail.budget` records limits, fits spent, and any `degradations`.
`time=` and `uncertainty="stability"/"bootstrap"` raise
`NotImplementedError`. Use the explicit tuning path below when you need to
control validation and stopping, and the dedicated uncertainty or time-fold
APIs for those tasks. Read `result.trail` and `mb.report(result)` as a record
of what ran, not evidence that omitted stages were completed. The underlying
fitted model is `result.model`.

## Decisions that affect correctness

- Reserve evaluation rows before screening, imputation, tuning, or selecting
  terms. Split by the deployment unit: independent rows, new groups, or
  future times. Use the same kind of split inside tuning. See
  [before the first fit](topics/before-the-first-fit.md).
- State the response support and use a matching family. Binary responses
  are 0/1. Use `type="response"` for probabilities or means; the link-scale
  predictor is different for non-identity links.
- Terms are formula syntax, not imported Python constructors:
  `"y ~ bols(x) + bbs(z, df=4) + brandom(site, df=4)"`. Exclude IDs and
  post-outcome columns unless their role is deliberate. Set a missing-data
  policy and keep category dtypes consistent at prediction.
- For boosting, inspect `.mstop` and `.at_boundary` from `tune_mstop` and
  use `.model`, which reuses its fitted path. A boundary optimum needs more
  search or an explicitly reported budget limit. For REML, inspect
  `.convergence` and warnings; there is no boosting stop count to tune.
- Predictive term gains, selection frequency, coefficient intervals, and
  causal effects answer different questions. A CV standard error is not an
  effect interval. Read the uncertainty route before making inferential claims.

## Open only the references needed for this task

Each quoted name also works with `python -m mboostrs guide NAME`. These
references are bundled beside this file and work after skill installation.

| Need | Topics |
|---|---|
| First working model and common call/return conventions | "golden-path", "quick-reference" |
| Data roles, missing values, categorical levels | "before-the-first-fit", "missing-data", "categoricals" |
| Draft or inspect a formula | "screen-and-draft", "interactions-and-terms" |
| Tune boosting and define evaluation folds | "fit-and-tune", "pitfalls" |
| REML models and uncertainty after term search | "direct-reml-gam", "uncertainty", "uncertainty-selection" |
| Residual patterns or a weak held-out score | "diagnostics", "modeling-tactics", "specification-search" |
| Distribution choice, spread, or scenario comparisons | "distributional-branch", "sensitivity-and-contrasts" |
| Interpret and communicate an existing fit | "reading-a-fit", "interpretation", "explain", "closing-the-analysis" |
| Automated exploration and compute budgets | "insights", "discover", "discover-limits", "costs", "unattended-pipelines" |
| Find Python/R/Rust functions or save a fitted model | "task-map", "task-map-interpret-export" |
| Causal question, adjustment set, average effect | "causal", "causal-roles", "causal-average", "causal-outcome-models" |
| Heterogeneous effects and their checks | "causal-heterogeneity", "causal-effect-model-checks", "causal-effect-modifiers" |
| Overlap, confounding sensitivity, instruments | "causal-overlap-sensitivity", "causal-instruments" |

Worked analyses: "example:rent-analysis", "example:claims-distributional",
"example:readmissions-pitfalls", "example:pm25-reference-gap", and
"example:disc-screening". Their measured results illustrate those datasets;
do not copy their thresholds or conclusions onto a new analysis.

## Finish at the requested scope

A first fit does not require every screen, stability selection, or a reference
booster. Run a diagnostic when it answers a live question; budget expensive
searches with the "costs" topic. A `reference_gap` or two-SE comparison is a
predictive diagnostic, not proof of a missing mechanism or a formal effect test.
Keep final test rows out of revisions: once their score informs a new model,
they are validation data, and a new independent assessment is needed.

Report the formula, family, split, tuning or convergence status, measured
performance, warnings, and limitations relevant to the user's question.
Save a reusable model with `mb.save_model(model, path)`; verify a reload's
predictions and retain the version, preprocessing, and evaluation record.
Creating an analysis does not authorize posting it to an issue or other service.

Optional skill installation: `python -m mboostrs skill install` copies this
folder to `.claude/skills/`; pass `--dest .agents/skills` for a project agent
skills directory. Installation is unnecessary to read the bundled guide.
