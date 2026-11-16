# Joint-tree reservation and the coordinate-surface step: arbitration record

## Study objective

Issue #864: the joint-tree reservation evaluation
(`docs/book/benchmarks/joint_tree_reservation_eval.md`) measured a cost
on coordinate data that nothing arbitrated. The autospec search takes
a fit reservation for the final joint-tree step (21 fits at the
coordinate fixture's five folds) once the family is settled, without
regard to the coordinate-surface step's claim on the same fits. On the
coordinate fixture of `tests/test_joint_tree_autospec.py` at autospec
budgets 60 and 70, taking the reservation leaves the
coordinate-surface step unpaid: it skips with "insufficient budget for
five candidate sweeps and the final sweep", while the joint tree is
adopted. The held-out cost is wide — mean squared error 0.477539 with
the reservation against 0.112311 without it on a fresh sample (seed
31, n = 1000), and 0.434847 against 0.070196 against the noiseless
signal. The fixture's truth is a lon/lat interaction plus a separate
additive z effect; the surface step's winning specification (a tree
over the pair plus a smooth z term) matches that structure, while the
joint tree must represent z inside the same tree and does so worse at
this size. The trade is bounded from the other side: at budget 40,
with no reservation neither step is fit and the search ends at an
additive specification (held-out signal MSE 9.074783), while with the
reservation the joint tree is adopted at the same scores as at
budgets 60 and 70.

The issue names three options: (a) ordering — let the surface step
run before the reservation is taken; (b) arbitration — take the
reservation only when the surface step has been scored or cannot run;
(c) pricing — state the surface step's cost in the reservation's
entry check.

## The choice, stated before any result of this study

Option (b), arbitration, applied at the coordinate-surface step — the
point where both claims on the fits are known. The reservation is
taken as before, early, so it still protects the joint-tree step
against the shape, mstop, interaction and deselection spending that
starved it in the first place. When the surface step is reached, the
reservation is released to it if three conditions hold:

1. the surface step cannot be paid from the unreserved remainder
   (`budget − spent` below its cost);
2. it can be paid once the reservation is released
   (`budget + reserved − spent` at least its cost);
3. the budget outside the reservation is itself at least the surface
   step's full cost.

The surface step's cost is stated by one helper,
`_coordinate_surface_cost(n_folds, audit_reserve)` = `6 * n_folds + 2`
plus the single-holdout audit reserve when the fold design calls for
it, and the step's own budget check now reads that helper, so the
arbitration and the step price the step the same way.

Condition 3 is the arbitration rule. It separates the two measured
regimes without naming a budget: at budgets 60 and 70 on the fixture
the unreserved budget is 39 and 49 against a surface cost of 32 — the
surface step was affordable outside the reservation, and only the
earlier steps' spending against the reduced budget, together with the
reservation, left it unpaid. At budget 40 the unreserved budget is 19,
below the surface step's cost of 32: the surface step was never
affordable outside the reservation, so the reservation stands for the
final step it was taken for, which is the only route to the fixture's
interaction at that budget.

Option (a) was not chosen because the surface step's inputs — the
formula, modelled columns and terms after the interaction screen and
deselection — do not exist when the reservation is taken, so the step
cannot literally be moved ahead of it without restructuring the
search. Option (c) was not chosen because pricing the surface step
into the reservation's entry check, at the point where almost nothing
has been spent yet, either refuses the reservation at every budget
where the two costs together exceed the budget — losing the budget-40
joint-tree adoption — or admits it at budgets 60 and 70, where the
starvation was measured.

## The change

One file, `python/mboostrs/autospec.py`:

- At the coordinate-surface call site, the arbitration above: a
  `joint_reserve` that meets the three conditions is added back to
  `budget` for the surface step and cleared. If the step then stands
  without spending (no coordinate pair, a declared pair, a
  constraint), the reservation is restored in full for the joint-tree
  step. If the step spends, the release is recorded as
  `evidence["coordinate_surface"]["joint_reserve_released"]`, and the
  joint-tree step keeps its own budget check on what remains.
- `_coordinate_surface_cost` states the surface step's cost; the
  step's budget check uses it. The check's value is unchanged.
- The reservation's comment records that the surface step arbitrates
  against it at its call site.

Nothing else changes: the reservation's size and entry checks, the
surface step's candidates and adoption rules, the joint-tree step's
cost, candidates and adoption bar, and both steps' evidence shapes
apart from the one recorded release field.

## Frozen protocol

Fixed in advance by the dispatch of this study; the bars did not move
after results were seen.

- Bar 1 (mechanism): at budgets 60 and 70 on the coordinate fixture,
  the coordinate-surface step no longer skips for budget as a direct
  result of the reservation — either the surface step is scored, or
  the reservation is not taken when the surface step is both runnable
  and unpaid.
- Bar 2 (fixture quality): at budgets 60 and 70, held-out
  fresh-sample MSE (seed 31, n = 1000, the evaluation's own scoring)
  at most 0.20, and noiseless-signal MSE at most 0.15. Baselines:
  0.477539 / 0.434847 with the reservation; 0.112311 / 0.070196
  without.
- Bar 3 (budget 40 holds): at budget 40 the joint tree is still
  adopted and its noiseless-signal MSE stays at most 1.0.
- Bar 4 (no harm on the frozen panels): on the frozen panels
  (protein, bike_iid, diamonds; seeds 0–2; shipped arm) plus the
  cancer battery and smooth_sparse controls, outcomes match the
  reservation evaluation record — bit-identical in the shipped arm,
  or any difference measured and explained; and the reservation's
  panel benefit in the evaluation's diagnostic arm is preserved
  (bike_iid draws whose entry checks pass still score both candidate
  depths with the reservation on, and starve with it off).

The fixture harness is the evaluation's own: the coordinate fixture
of `tests/test_joint_tree_autospec.py` (600 rows, seed 7, mstop 80,
preexploration off, deselect off, five folds), held-out scoring on a
fresh sample from the same generator (seed 31, n = 1000). The panel
harness is the evaluation's: the refresh driver's path
(`benches/shootout_data.payload`, `benches/shootout_models.search`,
tier "small", autospec budget 32), refit on the development rows,
test rows scored after development-fit cleaning.

A regression test covers the budget-60/70 starvation:
`test_reservation_yields_to_the_coordinate_surface_step` in
`tests/test_joint_tree_autospec.py`, parametrized at budgets 60 and
70. It fails on the code before this change (the surface step's
candidate list is empty) and passes with it.

## Results

### Coordinate fixture

| budget | coordinate-surface step | joint-tree step | held-out MSE (response) | held-out MSE (signal) |
|---|---|---|---|---|
| 40 | skipped: "insufficient budget for five candidate sweeps and the final sweep" | adopted, depth 6 over lon, lat, z; reserved 21 | 0.477539 | 0.434847 |
| 60 | adopted: btree(lon, lat, maxdepth=3), chosen gain +94.94%; reservation of 21 released to it | skipped: "insufficient budget for the incumbent and two joint-tree depths" | 0.112311 | 0.070196 |
| 70 | adopted: btree(lon, lat, maxdepth=3), chosen gain +94.94%; reservation of 21 released to it | skipped: "insufficient budget for the incumbent and two joint-tree depths" | 0.112311 | 0.070196 |

At budget 40 the outcome is the reservation arm's, unchanged to the
last digit: the arbitration's third condition does not hold (the
unreserved budget is 19, the surface step's cost is 32), the
reservation stands, and the joint tree is adopted at its recorded
scores. At budgets 60 and 70 the outcome is the no-reservation arm's,
also unchanged to the last digit: the surface step is scored, adopts
`y ~ bbs(z, df=4.0) + btree(lon, lat, maxdepth=3)`, and the joint-tree
step then fails its own budget check on the fits that remain. Total
fits are 26 at budget 40, 60 at budget 60 and 70 at budget 70, each
within its budget; at budget 60 the search spends fits the
no-reservation run left unspent (its total was 55), on the steps that
follow the surface adoption, and the adopted specification and both
held-out scores are the same.

### Frozen panels, shipped arm

The shipped arm was run on the changed code for protein, bike_iid
and diamonds (seeds 0–2), with smooth_sparse and the cancer battery
as controls:

| panel | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| protein | 24.647187 | 25.765589 | 25.336264 | 25.249680 |
| bike_iid | 0.134897 | 0.128918 | 0.139613 | 0.134476 |
| diamonds | 0.013715 | 0.012659 | 0.013930 | 0.013435 |
| smooth_sparse | 0.258837 | 0.242381 | 0.259147 | 0.253455 |
| cancer (log loss) | 0.124439 | 0.068763 | 0.160425 | 0.117876 |

Every cell is bit-identical to the reservation evaluation's
shipped-arm run records — loss, mstop, fit count, formula and both
steps' evidence agree to the last digit on all fifteen draws. The
cancer battery reproduces its recorded values bit-exactly
(0.12443863785108761, 0.06876267784646255, 0.1604252279147264).
This is the expected result, verified rather than assumed: on
these panels the shipped entry checks never reach the joint-tree
step (no draw carries joint-tree evidence), so no reservation is
taken and the arbitration has nothing to arbitrate.

### Diagnostic arm: bike_iid

The diagnostic configuration is the reservation evaluation's
Arm G — its gate edits applied on top of the changed code — run on
bike_iid, the reservation's home panel, with the reservation on
and off:

| seed | OFF | ON | joint-tree outcome (ON) |
|---|---|---|---|
| 0 | 0.134897 | 0.121441 | adopted, depth 3, held-out gain +64.24%, reserved 9 |
| 1 | 0.128918 | 0.130195 | adopted, depth 6, held-out gain +75.09%, reserved 9 |
| 2 | 0.139613 | 0.111769 | adopted, depth 3, held-out gain +63.07%, reserved 9 |
| mean | 0.134476 | 0.121135 | |

With the reservation on, both candidate depths are scored on every
draw and the candidate is adopted on all three. With it off, the
step starves on every draw: "insufficient budget for the incumbent
and two joint-tree depths". Every cell is bit-identical to the
evaluation's diagnostic-arm run records, so the reservation's
panel benefit — mean test loss 0.121135 with the reservation
against 0.134476 without it, the 9.9% reduction the evaluation
measured — is preserved to the last digit. The coordinate-surface
step does not arise on this panel (no draw in either configuration
carries coordinate-surface evidence), so the arbitration has no
occasion to engage here; its effect is confined to data where both
steps claim the same fits.

## Verdict against the frozen bars

- Bar 1 (mechanism): **met.** At budgets 60 and 70 the
  coordinate-surface step is scored and adopts; it no longer skips
  for budget as a result of the reservation, and the release of
  the 21 reserved fits is recorded in its evidence
  (`joint_reserve_released`).
- Bar 2 (fixture quality): **met.** At budgets 60 and 70 the
  held-out fresh-sample MSE is 0.112311 (bar: at most 0.20) and
  the noiseless-signal MSE is 0.070196 (bar: at most 0.15) — the
  no-reservation arm's scores, reproduced to the last digit.
- Bar 3 (budget 40 holds): **met.** At budget 40 the joint tree
  is still adopted and its noiseless-signal MSE is 0.434847
  (bar: at most 1.0) — the reservation arm's outcome, unchanged.
- Bar 4 (no harm on the frozen panels): **met.** The shipped arm
  is bit-identical to the reservation evaluation's records on
  every panel and control draw, and the diagnostic arm preserves
  the reservation's bike_iid benefit exactly: both candidate
  depths scored with the reservation on, starved with it off.

The arbitration preserves both measured outcomes. At budgets 60 and 70,
the surface step fits its stronger specification because its cost is
within the unreserved budget; at budget 40, the joint tree remains
available because the surface step cannot be paid for without the
reservation. Shipped-panel results and the diagnostic reservation benefit
remain bit-identical to the earlier evaluation.

## Provenance

- Code under test: the claim branch for this issue, from main at
  92fa88d566bf2a7031aceac87c170485e7439acc. Before this change,
  `python/mboostrs/autospec.py` is blob
  4eb20ea548c3e638606807240ada3745af6c32d1, the blob the reservation
  evaluation records for its shipped arm.
- Baselines are quoted from the reservation evaluation record
  (`docs/book/benchmarks/joint_tree_reservation_eval.md`), opened in
  the same session as this study's runs.
- Every loss in this record comes from this study's own run records,
  written per run as it completed.
