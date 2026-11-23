# `fit` on held-out datasets

`mb.fit(data, "y")` is the one-call entry point an agent uses. Before this
change it called autospec with no budget, so no candidate was measured and
every column got its default learner. The change gives the search a fit budget
(16, 32 or 512 fits; [budget mapping](2026-11-20-fit-budget.md)) and a
wall-clock limit (60, 120 or 1200 s), and fixes an autospec step rule that
let an unchecked fixed step through. This study tests the change on data that
played no part in choosing it, against criteria written before each run.

On the final code, held-out loss is lower on 38 of 62 paired datasets and
higher on 20 (Wilcoxon p = 8e-5). The typical gain is small: the median loss
ratio is 0.990. A few datasets with deterministic or near-deterministic
structure improve by orders of magnitude, which is why the geometric mean
ratio (0.15) is far from the median. Published effect directions are
recovered in 98.7% of cases against 94.8% before. On the 13 well-studied
datasets alone the prediction gain is not significant (8 better, 5 worse,
p = 0.24). A run takes a median 5.9 s against 0.11 s before.

## Design

**Arms.** *old*: `python/mboostrs/_procedure/_fit.py` at commit 12c87a3,
extracted verbatim. *new*: the working tree, default budget `"standard"`.
*quick*: the working tree with `budget="quick"`. No family is passed; each arm
infers it, as a bare agent call does.

**Data.** 95 datasets. 36 come from the OpenML datasets in the local
scikit-learn cache, excluding those used to choose the budget mapping and two
duplicates. 56 are every UCI dataset importable with `ucimlrepo` that is
tabular, has one target, 300 to 500,000 rows and 2 to 300 features, and whose
target is binary or numeric regression, minus duplicates. 3 are the
`mboostrs.datasets` tables Munich rent, India stunting and bodyfat. The
manifests list each dataset with its source columns.

**Well-studied tier.** 18 datasets whose effects are reported in a cited
paper, for example the returns to education in the CPS wage data (Berndt
1991), ambient temperature in the combined-cycle power plant (Tüfekci 2014),
and alcohol and volatile acidity in wine quality (Cortez et al. 2009).
[known_effects.json](fit_heldout/known_effects.json) lists 38 effect
directions with their citations. Every direction agrees with the marginal
Spearman sign on the full data, which rules out a target coded backwards.

**Data scan.** Before any run, [onceover](https://pypi.org/project/onceover/)
0.1.7 profiled every dataset with the target declared. It found an exact
affine identity between the target and a column in 17 datasets (for example
`y = c3 + 1.18` in synchronous_machine) and leakage findings in 5. The flags
stratify the report and remove nothing.

**Splits and scores.** Per seed: a random 80/20 split (stratified for binary
targets), at most 4,000 training and 2,000 test rows, test MSE or test log
loss on the response scale (the mean for distributional models), one thread,
600 s per run. A known effect is *recovered* when moving that column from its
training 10th to 90th percentile on every test row changes the mean
prediction with the published sign. A model that does not use the column
gives a change of exactly zero and does not recover it.

**Criteria.** A: across all datasets where both arms complete every seed, the
mean log loss ratio new/old is below zero with the 95% bootstrap interval below
zero, a two-sided Wilcoxon test gives p < 0.05, and new fails no more runs than
old. B: the first two parts of A on the well-studied tier. C: new recovers
known effects at a rate no more than 5 points below old.

## Iterations

Each study found a defect, which was fixed, and the next study used new
seeds. The pre-registrations are in [fit_heldout/](fit_heldout/).

| Study | Seeds | Code | A | B | C | Failed runs new / old |
|---|---|---|---|---|---|---|
| v1 | 1000–1002 | budget only; 36 OpenML datasets | fails (3 timeouts) | — | — | 19 / 17 |
| v2 | 2000–2002 | + time limit; 95 datasets | holds | fails (6 / 6, p = 0.27) | holds (98.7% / 92.2%) | 78 / 90 |
| v3 | 3000–3002 | + step rule, same family | holds | holds (9 / 3, p = 0.034) | holds (100% / 96%) | 82 / 88 |
| v4 | 4000–4002 | + step rule, any family | holds | fails (8 / 5, p = 0.24) | holds (98.7% / 94.8%) | 79 / 88 |

- **v1** failed on a 1,025-column table (QSAR_TID_11), where every new-arm
  run exceeded 600 s. That led to the wall-clock limit; the table now returns
  the default specification after 121 s.
- **v2** found a Poisson fit on student_performance_por whose fitted mean was
  0.18 against a target mean of 12 (test MSE 138.9 against 7.2). The budget
  had run out before the stopping depth was measured, and autospec kept a
  fixed step of 0.3 that no fit had checked.
- **v3** found the same cause choosing the family. On another split, the
  fixed-step run's family probe read a Pearson dispersion of 4631 off a
  diverged Poisson probe and adopted NBinomial (MSE 20.0 against 7.7).

## Final results (v4)

| Analysis | Datasets | New better / worse | Median ratio | Geometric mean ratio | Wilcoxon p |
|---|---|---|---|---|---|
| A, all datasets | 62 | 38 / 20 | 0.990 | 0.154 | 8.3e-5 |
| UCI only | 30 | 17 / 12 | 0.994 | 0.706 | 0.050 |
| OpenML only | 29 | 19 / 7 | 0.948 | 0.029 | 0.0012 |
| no onceover flag | 55 | 34 / 18 | 0.993 | 0.536 | 0.0002 |
| onceover flag | 7 | 4 / 2 | 0.988 | ≈ 0 | 0.22 |
| B, well-studied tier | 13 | 8 / 5 | 0.998 | 0.620 | 0.24 |

The ratio is new loss over old loss; below 1 is better. Seven datasets are
more than 2% worse with the new arm: Ionosphere (1.118), Differentiated
Thyroid Cancer Recurrence (1.060), white wine (1.051), Taiwanese Bankruptcy
(1.044), Thoracic Surgery (1.038), Beijing PM2.5 (1.037) and Glioma Grading
(1.036).

Known effects: new recovers 76 of 77 (effect, seed) pairs, old 73 of 77. The
new arm misses the effect of the number of convenience stores on Taipei house
prices on one seed, where the search dropped the column; old misses it on all
three seeds and reverses the model-year effect in Auto MPG on one.

The new arm failed no run that the old arm completed; the old arm failed nine
that the new arm completed. Median run time is 5.9 s against 0.11 s, and the
95th percentile is 82 s.

## Limits

- **Failures from the data.** Both arms fail on about 28% of runs: 79 of 285
  for new, 88 for old. Most come from factor levels in the prediction rows
  that the training rows lack, from missing values, and from tables where
  every row has a missing value somewhere. These are tracked in
  fcela/mboostrs-dev#915. Analyses A to C cover only datasets where both arms
  completed every seed, so five of the 18 well-studied datasets (Adult,
  Auto MPG on two seeds, Breast Cancer Wisconsin Original, Mammographic Mass,
  Default of Credit Card Clients) drop out of B.
- **Prediction on the well-studied tier is not established.** B held in one of
  three studies; the tier has 12 to 13 paired datasets.
- **The effect test is coarse.** It checks the sign of a 10th-to-90th
  percentile change, not shape or size. The old arm keeps every column by
  construction, so the comparison mainly measures whether the search drops or
  reverses documented effects.
- **Family choice.** On cpu_activity the family stage swaps Poisson for
  Gaussian on a dispersion rule and loses 43% in MSE (fcela/mboostrs-dev#914).
- **The studies reuse the 95 datasets.** Each new seed gives new splits of
  the same tables, so v3 and v4 are not independent of v2 at the dataset
  level.
- **Time limits were chosen from v1 run times**, on one Apple M4 with eight
  concurrent single-thread runs.

## Artifacts

[fit_heldout/](fit_heldout/) holds the pre-registrations and their SHA-256
hashes, the deviations note for v1, the known-effects key, the onceover scan,
the manifests, every run (`results_v1.jsonl` to `results_v4.jsonl`) and every
analysis. The runner and the data preparation are in
[benches/fit_heldout](../../benches/fit_heldout/README.md).
