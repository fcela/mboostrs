# Whole-model check on the frozen batteries: study record

## Study objective

Issue #804: every adoption inside the automatic specification search
is gated locally — each candidate term, shape, pair form or family
must beat the specification as it stands at that moment, on held-out
evidence. The reverse comparison was missing: the finished discovered
specification was never compared, as a whole, against prespecified
simple finished models fitted independently on the same held-out
evidence, so a search could finish behind a simple model without any
step in the procedure registering the fact. On the frozen cancer
battery the discovered specifications average 0.1179 outer log loss
over draws 0–2, against 0.0965 for the library's recorded additive
track and 0.0989 for an all-linear specification fitted through the
library's own machinery (records under
`docs/book/benchmarks/competitive_refresh/` and
`docs/book/benchmarks/single_holdout_adoption_gates.md`).

A whole-model check at the end of the search was designed and
shipped in response: it scores the specification the search settled
on, as a whole, against two prespecified simple finished models on
shared folds — "linear", the starting specification with every
one-column smooth written as `bols`, and "additive", the starting
specification itself. It shipped without its frozen-battery
evaluation ever being run. This study is that evaluation, under bars
fixed in advance. It is record-only: no library code changes, and no
default changes either way.

## The step under test

The check runs at the end of the search whenever the fit budget is
positive, with no opt-out. A rival replaces the searched
specification only when its held-out risk is lower by more than 1
standard error of the paired fold difference, and never on one fold.
The outcome is recorded in `evidence["whole_model"]`: whether the
comparison ran, each rival's fold risk and its difference from the
chosen specification's fold risk, the paired standard error where
one exists, the winner, and whether a replacement occurred.

One design fact shapes the reading of every result below. The frozen
batteries judge the search on a single supplied holdout: the refresh
driver's fold matrix has one column. On one fold the paired standard
error is undefined, so under the shipped rule the step can score the
rivals on these batteries but can never replace the searched
specification. The bars are evaluated exactly as fixed; the record
states this mechanism with the per-rival evidence rather than
treating it as a reason to change the protocol.

## Frozen protocol

Fixed in writing before any panel run in this study; the bars did
not move after results were seen.

- Code under test: main at 7ad96634, both arms.
  `python/mboostrs/autospec.py` carries `_whole_model_step`,
  `_linear_rival` and `_WHOLE_MODEL_SE = 1.0`, called
  unconditionally at the end of the search whenever budget > 0.
- ON arm: the whole-model step as shipped.
- OFF arm: the same search with the whole-model step not applied.
  The shipped code has no switch, so the OFF arm is produced by a
  test-only harness toggle that is not shipped: the harness replaces
  the module's `_whole_model_step` with a function that records that
  the step was disabled by the harness and returns its inputs
  unchanged, spending no fits. Nothing else is patched.
- Cancer battery: task "cancer", draws (seeds) 0, 1, 2, the refresh
  driver's own path (`benches/shootout_data.payload`,
  `benches/shootout_models.search` with method mboost_auto, tier
  "small": autospec budget 32, nu 0.1, mstop cap 200, single
  supplied holdout, interactions="adopt", residual_tree="auto"),
  refit on the development rows, test rows scored after
  development-fit cleaning, loss = log loss on the test rows.
- Panels protein, bike_iid, diamonds: same driver path, draws 0, 1,
  2, loss = mean squared error on the test rows, exactly as the
  adoption-gate studies ran them.
- Both arms on identical draws, folds and budgets, in fresh
  processes, one thread. Partition digests recorded per run.
- Recorded comparators (labelled as recorded, not re-measured):
  discovered cancer mean 0.1179; additive track 0.0965; all-linear
  0.0989. The OFF arm is also the current-code baseline: the
  recorded 0.1179 may not reproduce exactly because the code has
  changed since it was recorded; any gap is reported plainly and a
  reproduction gap alone is not a bar failure.
- Every number below comes from a run executed in this study, with
  seeds fixed; no number is copied from an older record except as a
  clearly labelled recorded comparator.

### The bars

- Bar A (the check runs): on every cancer draw where the budget
  covers the comparison, the step's evidence records
  compared=True; every non-comparison carries its recorded reason.
  Whether the budget covers the comparison is read from the
  evidence itself; the fit need is also recomputed from the
  recorded rivals and the chosen formula — n_folds × (1 + distinct
  rivals differing from the chosen formula), with n_folds = 1 on
  this harness.
- Bar B (cancer benefit): the ON arm's mean outer log loss over
  cancer draws 0–2 is at most 0.108, AND the ON mean is better than
  the OFF mean on the same draws and code (paired by draw).
- Bar C (do no harm): on each of protein, bike_iid and diamonds,
  the ON arm's mean score is no more than +1% relative worse than
  the OFF arm's mean on the same draws.
- Bar D (rule integrity): every replacement that occurs satisfies
  the shipped rule as recorded in the evidence (rival risk lower by
  more than 1 paired standard error; never decided on one fold).

Verdict rule: if all four bars pass, the verdict is that the shipped
check earns its place as the default, with the measured margins
stated. If any bar fails, the record states which bar failed, by
how much, and the recommended follow-up. This study ships no
library code change and no default change either way.

## Results

In all 24 runs the two arms returned the same formula on every draw,
and the ON and OFF test losses are identical to the last digit on
every draw of every panel: with one fold, no replacement can occur,
so the check cannot change any outcome these batteries measure.
Partition digests match between arms on every (task, draw) pair.
The step's cost is visible in the fit counts: the comparison spends
2–3 fits per run.

The OFF arm reproduces the recorded shipped-arm draws: on cancer
exactly (largest absolute difference 0.0 against the recorded draws
0.124439 / 0.068763 / 0.160425), and on protein, bike_iid and
diamonds within 4.8e-07 per draw, the rounding of the older records.
There is no reproduction gap to explain.

### Cancer battery (outer log loss; lower is better)

| Draw | ON | OFF | Fits ON | Fits OFF |
|---|---|---|---|---|
| 0 | 0.124438638 | 0.124438638 | 29 | 26 |
| 1 | 0.068762678 | 0.068762678 | 26 | 23 |
| 2 | 0.160425228 | 0.160425228 | 32 | 29 |
| Mean | 0.117875515 | 0.117875515 | | |

Whole-model evidence, ON arm (fold risks on the single supplied
holdout; differences are rival risk minus chosen risk, so a positive
difference means the searched specification had the lower fold
risk):

| Draw | Chosen | Linear rival | Additive rival | Replaced |
|---|---|---|---|---|
| 0 | 0.070684 | 0.083389 (+0.012705) | 0.089280 (+0.018596) | No |
| 1 | 0.057203 | 0.103628 (+0.046424) | 0.093855 (+0.036652) | No |
| 2 | 0.027628 | 0.049249 (+0.021621) | 0.032734 (+0.005106) | No |

On all three draws the comparison ran (compared=True) and spent
exactly 3 fits, the recomputed need for one fold with two distinct
rivals. On the fold itself, the searched specification's risk was
lower than both rivals' on every draw. Each rival's recorded reason
is the same: one fold gives no standard error of the fold
differences, so no replacement is possible.

### Protein (test MSE; lower is better)

| Draw | ON | OFF | Fits ON | Fits OFF |
|---|---|---|---|---|
| 0 | 24.647187118 | 24.647187118 | 20 | 17 |
| 1 | 25.765588630 | 25.765588630 | 16 | 14 |
| 2 | 25.336264134 | 25.336264134 | 19 | 16 |
| Mean | 25.249679961 | 25.249679961 | | |

Relative ON−OFF difference: +0.0000%. The comparison ran on all
three draws and no replacement occurred. On draw 1 the additive
rival is the searched specification itself (the search settled on
the starting specification), so the step spent 2 fits, the
recomputed need with one distinct rival. On draw 2 the additive
rival's fold risk was lower than the chosen specification's
(26.284794 against 26.407088, difference −0.122293); the linear
rival's was higher on every draw. The lower additive fold risk
could not act: one fold, no standard error.

### Bike IID (test MSE; lower is better)

| Draw | ON | OFF | Fits ON | Fits OFF |
|---|---|---|---|---|
| 0 | 0.134896729 | 0.134896729 | 32 | 29 |
| 1 | 0.128918476 | 0.128918476 | 32 | 29 |
| 2 | 0.139613022 | 0.139613022 | 32 | 29 |
| Mean | 0.134476076 | 0.134476076 | | |

Relative ON−OFF difference: +0.0000%. The comparison ran on all
three draws and no replacement occurred. Both rivals' fold risks
were far above the chosen specification's on every draw (linear
0.931534–1.034296, additive 0.338491–0.363202, chosen
0.121746–0.143699).

### Diamonds (test MSE; lower is better)

| Draw | ON | OFF | Fits ON | Fits OFF |
|---|---|---|---|---|
| 0 | 0.013715041 | 0.013715041 | 21 | 18 |
| 1 | 0.012658569 | 0.012658569 | 25 | 22 |
| 2 | 0.013929948 | 0.013929948 | 26 | 23 |
| Mean | 0.013434519 | 0.013434519 | | |

Relative ON−OFF difference: +0.0000%. The comparison ran on all
three draws and no replacement occurred. On draw 0 the additive
rival's fold risk was lower than the chosen specification's
(0.035824 against 0.038918, difference −0.003093); the linear
rival's was higher on every draw. As on protein draw 2, the lower
fold risk could not act on one fold.

Across the 12 ON runs, a rival posted a lower fold risk than the
chosen specification in exactly 2 of the 24 rival comparisons
(protein draw 2 and diamonds draw 0, both the additive rival). In
the remaining 22 the searched specification had the lower fold
risk, or the rival was the searched specification itself.

## Verdict against the bars

- Bar A (the check runs): **met.** On all three cancer draws the
  evidence records compared=True, and the fits the step spent (3
  per draw) equal the recomputed need. No cancer draw went
  uncompared, so there is no missing-reason case. Across the study
  the comparison ran on all 12 ON runs.
- Bar B (cancer benefit): **not met, on both clauses.** The ON
  mean is 0.117875515, above the 0.108 bar by 0.009876. The ON mean
  is not better than the OFF mean: the two arms are identical on
  every draw (paired differences exactly 0.0), because on a single
  fold the shipped rule can never replace the searched
  specification. The recorded gap the check was built to close —
  0.1179 discovered against 0.0989 for the all-linear specification
  — is unchanged by the check on this battery. On the fold evidence
  itself the searched specification also scored below both rivals
  on all three cancer draws, so the fold comparison did not point
  to a replacement either; the bar is judged on outer test loss,
  and there the check changed nothing.
- Bar C (do no harm): **met** on all three panels. The relative
  ON−OFF difference is +0.0000% on protein, bike_iid and diamonds,
  against a bar of +1%. The check's only cost on these panels is
  its 2–3 fits per run.
- Bar D (rule integrity): **met.** No replacement occurred in any
  of the 12 ON runs, and the evidence shows the rule applied as
  shipped: in the two comparisons where a rival's fold risk was
  lower, the rival was not adopted, and the recorded reason is the
  rule's own — a standard error of the fold differences needs at
  least two folds. beats_chosen is false in all 24 rival
  comparisons, each with its reason recorded.

## Interpretation

On the frozen batteries, the whole-model check is a measurement,
not a safeguard. It runs, it scores both rivals on every draw, and
it records the comparison in the evidence — that part of the design
works as shipped, at a cost of 2–3 fits per search. But the
batteries judge the search on a single holdout, and the shipped
rule never replaces on one fold, so on these batteries the check
cannot close the gap it was built to close, and Bar B's failure is
a property of the evaluation setup as much as of the step: the
frozen batteries cannot distinguish a check that would replace
well from one that would replace badly, because neither can
replace at all.

Two things follow. First, the 0.1179 cancer mean stands as the
search's measured result with the check on: nothing in this study
moves it, and the gap to the recorded simple-model results
(0.0965 additive track, 0.0989 all-linear) remains open. Second,
evaluating the check as a safeguard requires an evaluation where
its rule can act — a multi-fold comparison, where the paired
standard error exists — or a change to the rule itself for the
single-holdout case. Either is a separate step; this study ships
no code change and no default change, and the verdict on the
check's value as a default stays open until an evaluation under a
scheme where replacements can occur has been run.
