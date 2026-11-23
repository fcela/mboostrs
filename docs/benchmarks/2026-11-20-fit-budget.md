# Fit budget mapping

`mb.fit` chooses a specification by calling `autospec`, whose `budget`
argument is the number of model fits its probes may spend. Each fit is one
held-out or cross-validated evaluation of a candidate: a family, a learner for
one column, an interaction, or a stopping depth. Until this change `fit`
passed no budget, so `autospec` ran with budget 0: it measured nothing and
returned a P-spline (`bbs`) for each continuous column and a dtype-chosen
learner for each other column. `fit`'s own `budget="quick"/"standard"/"thorough"`
argument was validated and then ignored.

This experiment measured held-out loss and time at budgets 0, 8, 16, 32, 128
and 512 to choose the fit counts behind the three settings. The result is
`quick` = 16, `standard` = 32 (the default), `thorough` = 512.

## Data and fitting

Ten panels from the frozen competitive study
([protocol](competitive_shootout_protocol.md)): eight real datasets
(bike iid, bike future, cancer, concrete, credit, diamonds, housing iid,
protein) and two generated ones (smooth sparse, joint boundary), with seeds 0,
1 and 2. Superconductivity was left out because every autospec workflow on it
reached the study's 240-second ceiling. Splits come from
`benches/shootout_data.py`: autospec receives the fit and validation rows (at
most 4,000) and its default folds, which is what `fit` does, and the chosen
model is scored on the test rows (at most 2,000). The loss is test MSE for
regression panels and log loss for the two binary panels (cancer, credit).

Each process ran at one thread; five processes ran at the same time on one
Apple M4, so the times include some contention. The runner is
`benches/bench_fit_budget.py`; every run, with its formula and fit count, is in
[the results file](2026-11-20-fit-budget-results.json).

## Results

Mean test loss over three seeds, as a percentage above the best budget on
that panel. Lower is better; 0.0 marks the best budget.

| Panel | 0 | 8 | 16 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| bike future | failed | 15.3 | failed | 0.0 | 1.3 | 1.0 |
| bike iid | 26.3 | 13.4 | 6.7 | 0.5 | 0.5 | 0.0 |
| cancer | 6.0 | 0.8 | 0.0 | 0.7 | 15.8 | 15.8 |
| concrete | 28.7 | 24.5 | 16.1 | 0.0 | 2.6 | 2.6 |
| credit | 0.0 | 0.4 | 0.2 | 0.2 | 0.1 | 0.0 |
| diamonds | 34.8 | 18.6 | 6.1 | 8.4 | 0.0 | 0.0 |
| housing iid | 63.0 | 52.8 | 44.6 | 2.7 | 0.0 | 0.0 |
| joint boundary | 0.6 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| protein | 7.8 | 5.0 | 3.1 | 0.0 | 0.0 | 0.0 |
| smooth sparse | 0.6 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| **median** | 7.8 | 9.2 | 3.1 | 0.1 | 0.1 | 0.0 |
| **worst** | 63.0 | 52.8 | 44.6 | 8.4 | 15.8 | 15.8 |
| **median seconds** | 0.03 | 0.26 | 0.45 | 4.5 | 23.8 | 24.0 |

Medians and worst cases are over the panels that completed. "Failed" means
prediction stopped on two of the three seeds: bike future splits by time, so
some seasons occur only in the test rows, and at budgets 0 and 16 the chosen
specification kept the `season` factor. `predict` refuses levels the model
never saw unless `unseen_levels=` says how to treat them. At the other budgets
the search dropped `season`, so prediction completed; this is a property of
those specifications, not a guard.

Three things decide the mapping:

1. **32 fits is where the curve flattens.** Median excess loss falls from
   3.1% at 16 fits to 0.1% at 32 and does not improve further. The worst
   case falls from 44.6% (housing iid) to 8.4% (diamonds).
2. **More fits cost time and do not help in the median.** Budget 128 takes
   about five times as long as 32 and is 15.8% worse on cancer. That loss
   comes from search over-fitting the validation rows on 569 observations,
   recorded in fcela/mboostrs-dev#541.
3. **autospec stops on its own.** It spent at most 161 fits at budget 512 and
   chose the same formula as at 128 on all three seeds of eight panels (two of
   three on bike future, none on bike iid). A larger budget only
   matters on wider tables, where more columns and pairs need probes.

`quick` = 16 runs in under a second and is for a first look; its worst case
here is 45% above the best budget. `standard` = 32 is the default.
`thorough` = 512 removes the budget as a constraint on these panels.

## Limits

- Three overlapping splits per real panel describe these datasets. They are
  not independent replications of a population.
- `autospec` was called with its own defaults, as `fit` calls it. In
  particular `interactions="report"` never adopts a pair, so joint boundary
  stays near 9.17 MSE at every budget, against 0.676 for the competitive
  study's hand-built hybrid. That gap is item 2 of fcela/mboostrs-dev#913
  and fcela/mboostrs-dev#911, not a budget question.
- The mapping is in fits, not seconds. Time per fit grows with rows and
  columns, so a wall-clock limit (fcela/mboostrs-dev#89) is still needed.
