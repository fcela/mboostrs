---
name: analyzing-data-with-mboostrs
description: Run an end-to-end data analysis with mboostrs, from a raw frame to a tuned, diagnosed, and explained model, deciding each next step from measured evidence rather than a fixed recipe.
---

# Analyzing data with mboostrs

mboostrs is a Python library (Rust core, Python bindings) for component-wise
gradient boosting. A model is a sum of terms written in a formula
(`bbs` smooth, `bols` linear, `brandom` random effect, `btree` tree). Each
boosting step updates one term, so every term's contribution can be read on
its own. All entry points are top-level: `mboostrs.column_screen`,
`mboostrs.gamboost`, `mboostrs.term_report`, and so on.

## Start here

Read the `golden-path` topic first. It is the whole workflow on one page --
data, `fit()`, `report()`, then the three common follow-ups -- and it says
which topic to open only when its evidence appears. The five steps below are
that same path with every measured step spelled out; read them when you need
the detail.

A fit reports associations. A causal question needs the `causal` topic and its `causal-*`
sub-topics, not this workflow.

Before any fit, settle the prediction target, the rows reserved for
evaluation, each column's role (a group ID is a grouping, not a measurement)
and the missing-data policy. Read `python -m mboostrs guide before-the-first-fit`.

## The first five steps

1. **Screen the columns** with `column_screen(df, "y")`. It fits no model; it
   reports how much of the response each column explains as a line, a curve or
   a step. Topic: `screen-and-draft`.
2. **Draft the specification** with `classify_columns`, `suggest_terms` or
   `autospec`. Each column gets a term and a reason. Write `df=` on every
   term, because selection between terms is fair only at equal df. Topic:
   `screen-and-draft`.
3. **Fit** with `gamboost(formula, data, control=boost_control(mstop=...))`.
   The default `mstop=100` is a placeholder. Topic: `fit-and-tune`.
4. **Tune the stopping iteration** with `tune_mstop(formula, data)`, then refit
   at `.mstop`. A value flagged `at_boundary` is unfinished. Topic:
   `fit-and-tune`.
5. **Measure what the fit left.** Run `reference_gap(formula, data)`: it
   compares the tuned formula with a flexible tree booster on the same
   held-out rows. A gap above two SE means the formula misses structure;
   then ask *where*. Run `screen_discontinuities` on the tuned fit's
   response-scale residual: a `typed_jump` row names a boundary no smooth
   term can represent -- add `btree(pair)` or the boundary's step indicator
   to the step-7 candidates; a flagged row with a large `type_ratio` is a steep
   smooth crossing -- prefer a smooth or tensor term; and a screen that does
   not flag finds no screened evidence for a jump in these columns. Add the candidates
   `reference_gap` ranks (interactions, lags, log scales) and run both
   again. Then read what each term earns with `term_report`. Do not stop
   after step 4. Topics: `modeling-tactics`, `interactions-and-terms`.

Steps 6 to 10 (ranked changes, scoring candidates, the distributional branch,
sensitivity and contrasts, explanation) are in the topics below. Steps 5 to 7
form a loop: compare, adopt a few changes, refit, screen again.

Read the closing topic, `closing-the-analysis`, before writing the report.

## Topics

Read one with `python -m mboostrs guide <topic>` (or
`mboostrs.agent_guide("<topic>")` from Python). List them all with
`python -m mboostrs guide index`. Four worked analyses, with their wrong
turns, are `example:rent-analysis`, `example:claims-distributional`,
`example:readmissions-pitfalls`, `example:pm25-reference-gap` and
`example:disc-screening`; read one
before a similar analysis. Read each topic before the step it covers,
and quote its numbers and refusals in the report rather than from memory.

- `"golden-path"`: start here; data, `fit()`, `report()`, then the three follow-ups and when to open each topic.
- `"quick-reference"`: one runnable block that runs the workflow on simulated data.
- `"insights"`: `insights()`, an optional screen, fit, diagnose and propose pass whose findings are hypotheses.
- `"before-the-first-fit"`: prediction target, evaluation rows, variable roles, missing-data policy.
- `"screen-and-draft"`: steps 1 and 2, `column_screen`, `classify_columns`, `suggest_terms`, `autospec`.
- `"fit-and-tune"`: steps 3 and 4, `gamboost`, `tune_mstop`, grouped folds, scores.
- `"interactions-and-terms"`: step 5, `screen_interactions`, `term_report`, the binary-response caveat.
- `"modeling-tactics"`: step 5, `reference_gap`, and the changes that closed gaps in benchmarks: interactions, lags, log scales, imputation, basis size, null and small-data checks.
- `"specification-search"`: steps 6 and 7, `suggest_specifications`, `compare_specifications`, the two-SE rule.
- `"distributional-branch"`: step 8, `heteroscedasticity`, `scan_lss_families`, `scale_absorption`.
- `"sensitivity-and-contrasts"`: step 9, `sensitivity`, `prediction_contrast`, `distributional_contrast`.
- `"explain"`: step 10, `llm_explain_prompt`.
- `"pitfalls"`: the tensor df floor, screening residual scale, group folds, unseen levels, unequal df.
- `"interpretation"`: thresholds that make an effect, a spread or an exceedance worth reporting.
- `"reading-a-fit"`: reading effects, what the fit cannot tell, what a first reading misses.
- `"closing-the-analysis"`: what the final report must state.
- `"missing-data"`: `na_action="omit"` or `"impute"` on every fitting function.
- `"uncertainty"`: effect intervals and term tests (REML refits), contrast intervals, prediction intervals, stability selection.
- `"direct-reml-gam"`: a penalized GAM by REML without boosting (`mb.gam`, `mb.gamlss`) fitted from a formula, with contrasts and intervals.
- `"uncertainty-selection"`: intervals and tests after the same rows chose the terms (`mb.split_refit`).
- `"categoricals"`: `bols` against `brandom` for labels, unseen levels, labels in `predict` and refits.
- `"costs"`: what each call costs in fits and seconds, and how to budget a run.
- `"unattended-pipelines"`: scripts that decide by rule on new data.
- `"discover"`: `mb.discover(data, response)`, the whole rule-driven analysis in one call.
- `"discover-limits"`: constraints, contrasts, null and small-data behaviour, budgets, limits of `discover`.
- `"task-map"`: which function performs each task in Python, R and the Rust core.
- `"task-map-interpret-export"`: the functions that interpret a fit, predict with uncertainty and export it.
- `"diagnostics"`: `revision_checks` for clusters, serial dependence and influential rows.
- `"causal"`: treatment effects from observational data, the adjustment set, overlap, sensitivity.
- `"causal-roles"`: the causal question and its effect scale, `causal_trap_report`, the adjustment set, `ci_test`.
- `"causal-average"`: `aipw` for the average effect, missing outcomes, repeats, nuisance formulas.
- `"causal-outcome-models"`: the outcome model choice, nuisance step counts, warnings, the sparse-arm check.
- `"causal-heterogeneity"`: `effect_model` with per-row intervals, `rank_by_effect`.
- `"causal-effect-model-checks"`: the candidate choice in `effect_model`, the comparison with a constant effect.
- `"causal-effect-modifiers"`: `effect_modifiers`, the BLP and GATES tests.
- `"causal-overlap-sensitivity"`: overlap, the overlap-weighted target, confounding sensitivity.
- `"causal-instruments"`: `iv_effect` and its assumptions.

`python -m mboostrs skill install` copies this skill and its topics into
`.claude/skills/` of the current project.
