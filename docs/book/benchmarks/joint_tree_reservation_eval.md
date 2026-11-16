# Joint-tree reservation: study record

## Study objective

Issue #847: the joint-tree gate study
(`docs/book/benchmarks/joint_tree_gate.md`) found that on the frozen
panels the final joint-tree step, once its coordinate-name gate was
removed, was still never scored on bike_iid: the search had spent more
than 23 of its 32 small-tier fits when the step was reached, and the
step reserves 3 × (folds + 2) = 9 fits to score the incumbent and the
two candidate depths. The fit reservation merged as the fix for that
starvation sets 3 × (folds + 2) fits aside for the step once the
family is settled, if the step's entry checks can pass, and restores
them just before the step runs. It was tested on a synthetic lon/lat
fixture only. This study asks: does the reservation do its job on the
frozen panels, does fit quality reach the gate study's bars, and does
it do harm — including the open trade at budgets 60–70 on coordinate
data, where the reservation makes the coordinate-surface step skip
for budget while the joint tree is adopted?

## The configurations under test

One file carries every difference: `python/mboostrs/autospec.py`.

- Arm S (shipped): main as it stands, with the reservation in place.
- Arm G (diagnostic): Arm S plus the gate study's change (items C1–C3
  of its protocol: the call site no longer requires a coordinate
  pair; the step's early return becomes an availability check; the
  column filters admit boolean and categorical columns alongside
  numeric ones), plus the matching change to the reservation's own
  entry check, so `_joint_tree_may_run`'s final return is the same
  availability check and the reservation is taken exactly when the
  ungated step could run. Its other checks (family, heredity,
  forbidden pairs, protected columns) are unchanged. Arm G ships
  nothing; it is a measurement configuration only. It exists because
  on the shipped code the step is gated on coordinate column names
  and none of protein, bike_iid or diamonds carries a recognised pair
  among its modelled predictors (protein: F1–F9; bike_iid: no
  lon/lat-type names; diamonds: the predictors include x and z, and y
  is the response, so the {x, y} pair is not among the modelled
  predictors). Without Arm G the reservation's effect on the frozen
  panels is unobservable.

Within each arm, two modes. ON: the reservation as coded. OFF: the
reservation disabled exactly as `tests/test_joint_tree_autospec.py`
disables it — the module's `_joint_tree_may_run` replaced by a
function returning False. The step itself, its cost check and the
adoption bar are untouched in both modes.

## Frozen protocol

Fixed in writing before any panel run in this study; the bars below
did not move after results were seen.

- Panels and harness identical to the gate study: the refresh
  driver's own path (`benches/shootout_data.payload`,
  `benches/shootout_models.search` with method mboost_auto, tier
  "small": autospec budget 32, nu 0.1, mstop cap 200, single supplied
  holdout, folds matrix with one column, so the joint-tree cost is
  3 × (1 + 2) = 9 fits; interactions="adopt", residual_tree="auto"),
  refit on the development rows, test rows scored after
  development-fit cleaning, loss = mean squared error. Panels
  protein, bike_iid, diamonds; outer draws (seeds) 0, 1, 2.
  Partition digests recorded per run.
- Controls: (a) the cancer battery, seeds 0–2, same harness (task
  "cancer", Binomial family, log loss) — the joint-tree call site is
  Gaussian-only, so the reservation cannot engage; recorded post-fix
  test log losses 0.12443863785108761, 0.06876267784646255,
  0.1604252279147264 (mean 0.1179); run in Arm S ON and Arm G ON.
  (b) smooth_sparse, seeds 0–2, both arms, both modes — autospec is
  the best recorded method there (mean 0.25196) and the truth is
  additive; harm is a systematic worsening across the paired draws.
  (c) the coordinate fixture of `tests/test_joint_tree_autospec.py`
  (`_geography_with_main_effect`: 600 rows, seed 7, lon/lat/z uniform
  on (−1, 1), signal 3·sign(lon·lat) + z, noise sd 0.2), Arm S, both
  modes, autospec budget 60 and 70 (the recorded trade range), mstop
  80, preexploration off, deselect off, five folds (the test harness
  shape; joint-tree cost 21); held-out score on a fresh sample from
  the same generator (seed 31, n = 1000), mean squared error against
  the observed response and against the noiseless signal; the
  coordinate-surface evidence and the joint-tree evidence recorded on
  both modes. A budget-40 pair was run as well, outside the protocol's
  range, and is reported as a supplementary measurement where it
  appears.
- Bar A (the reservation does its job): with the reservation ON, the
  joint-tree step is reached and scored (both candidate depths
  recorded) on every draw where its entry checks pass; reserved
  amounts and skip reasons are recorded from the evidence. In Arm S
  the entry checks are those of the shipped code, including the
  coordinate-name gate in `_joint_tree_may_run`; in Arm G they are
  the ungated ones, and a budget skip in Arm G ON on a draw whose
  entry checks pass is a Bar A failure on that draw.
- Bar B (fit quality): mean excess over the best recorded competitor
  within 5 points of the counterfactual's excess on at least 2 of
  protein / bike_iid / diamonds — protein ≤ +13.94% (mean loss
  ≤ 20.339), bike_iid ≤ +12.19% (≤ 0.104771), diamonds ≤ +4.34%
  (≤ 0.011718). Bar B is evaluated on Arm G ON, the configuration in
  which the step can run on these panels, and also reported for
  Arm S ON; on Arm S, any panel the name gate blocks is reported
  with the gate named.
- Bar C (do no harm): (a) the cancer battery reproduces its recorded
  values within noise in both arms ON; (b) smooth_sparse shows no
  systematic worsening ON vs OFF in either arm; (c) on the
  coordinate fixture at budgets 60 and 70, the held-out scores ON vs
  OFF are stated with numbers, and the direction of the trade is
  stated plainly. Bar C(c) is a measurement, not a pass/fail
  threshold: the study's job is to decide the trade with numbers.

A refit that fails to reproduce a recorded score exactly does not
block comparison: the gap is reported and the paired ON/OFF
comparisons (identical code apart from the mode) carry the verdict.
Nothing was tuned against the panels' test outcomes.

## Baselines

From the gate study, re-derived there from the refresh records
(`docs/book/benchmarks/competitive_refresh/`):

| panel | autospec (recorded) | best recorded non-mboostrs | counterfactual | counterfactual excess |
|---|---|---|---|---|
| protein | 25.24968 | 17.85089 (CatBoost, tuned) | 19.44626 | +8.94% |
| bike_iid | 0.134476 | 0.093386 (CatBoost, small) | 0.100106 | +7.19% |
| diamonds | 0.013533 | 0.011231 (CatBoost, small) | 0.011157 | −0.66% |

## Results

### Arm S — shipped code

On the shipped code the final joint-tree call site requires
`_has_coordinate_pair(joint_modelled)`, and on no frozen panel is
that check met. The consequence for this study: the evidence of
every Arm S run — all three panels, smooth_sparse, both modes —
carries no joint-tree entry at all. The step is not reached, so the
reservation is never consulted, nothing is reserved, and nothing is
skipped. ON and OFF are bit-identical on every draw:

| panel | seed 0 | seed 1 | seed 2 | mean | excess |
|---|---|---|---|---|---|
| protein | 24.647187 | 25.765589 | 25.336264 | 25.249680 | +41.45% |
| bike_iid | 0.134897 | 0.128918 | 0.139613 | 0.134476 | +44.00% |
| diamonds | 0.013715 | 0.012659 | 0.013930 | 0.013435 | +19.62% |

These are the gate study's "before" numbers (protein and bike_iid
means equal the recorded autospec means; the diamonds mean is the
measured 0.013435 against the recorded 0.013533, the known seed-0
drift the gate study records). For bike_iid specifically, the panel
the follow-up issue names: the reservation exists in the shipped
code, but it guards a step this panel cannot enter, so the panel's
outcome under the shipped reservation is exactly its recorded
outcome, +44.00% against a Bar B threshold of +12.19% it cannot be
evaluated against in this arm. The gate is the binding constraint
in Arm S, and it is a name check, not a budget check.

### Arm G protein — all draws scored, bar met

| seed | OFF | ON | joint-tree outcome (ON) |
|---|---|---|---|
| 0 | 18.983026 | 18.983026 | adopted, depth 6, held-out gain +18.27%, reserved 9 |
| 1 | 20.521879 | 20.521879 | adopted, depth 6, held-out gain +13.81%, reserved 9 |
| 2 | 18.952984 | 18.952984 | adopted, depth 6, held-out gain +23.80%, reserved 9 |
| mean | 19.485963 | 19.485963 | |

Both candidate depths are scored on every draw, the adopted
specification on each draw is a single tree over the panel's nine
predictors (F1–F9), and ON and OFF agree to the last digit: on this
panel the search still had the step's fits unspent when it arrived,
so the reservation is not what makes the difference — the gate
study's result (+41.45% before, +9.16% after) is reproduced exactly.
Mean excess +9.16% against the counterfactual's +8.94% and the bar's
+13.94%. This panel meets its bar.

### Arm G bike_iid — step reached, bar not met

| seed | OFF | ON | joint-tree outcome (ON) |
|---|---|---|---|
| 0 | 0.134897 | 0.121441 | adopted, depth 3, held-out gain +64.24%, reserved 9 |
| 1 | 0.128918 | 0.130195 | adopted, depth 6, held-out gain +75.09%, reserved 9 |
| 2 | 0.139613 | 0.111769 | adopted, depth 3, held-out gain +63.07%, reserved 9 |
| mean | 0.134476 | 0.121135 | |

OFF is the gate study's outcome restated: on all three draws the
step is skipped, "insufficient budget for the incumbent and two
joint-tree depths", and the losses are the recorded ones. ON is the
reservation doing exactly the job it was built for: the 9 fits are
set aside before the earlier stages can spend them, both candidate
depths are scored on all three draws, and the candidate is adopted
on all three — the adopted tree sits alongside the group terms the
replacement does not cover (brandom(season), brandom(weathersit),
and brandom(weekday) on seed 1). The mean test loss falls from
0.134476 to 0.121135, a 9.9% reduction, and the mean excess from
+44.00% to +29.71%.

It does not reach the bar (+12.19%, mean loss 0.104771), and the
draws show why: the held-out gains the adoption bar sees are +63% to
+75%, but the test rows return far less — seed 0 improves 10.0%,
seed 2 improves 19.9%, and seed 1 regresses slightly (0.128918 to
0.130195) behind a +75.09% held-out gain. The single-holdout gain
overstates the test return on this panel, the same overstatement
the gate study measured on diamonds. With the reservation in place,
reach is no longer this panel's problem; adoption quality is.

### Arm G diamonds — scored, unchanged from the gate study

| seed | OFF | ON | joint-tree outcome (ON) |
|---|---|---|---|
| 0 | 0.012660 | 0.012660 | adopted, depth 3, held-out gain +64.73%, reserved 9 |
| 1 | 0.012659 | 0.012659 | scored, refused by the 2% bar, reserved 9 |
| 2 | 0.013509 | 0.013509 | adopted, depth 3, held-out gain +2.13%, reserved 9 |
| mean | 0.012942 | 0.012942 | |

ON and OFF agree on every draw, and the panel reproduces the gate
study's after numbers exactly: adopted on seeds 0 and 2 (the tree
covers carat, depth, table, x and z; bols(clarity, by=color) is kept
on both adopting draws, and brandom(cut) on seed 2), refused on
seed 1. Mean excess +15.24% against the counterfactual's −0.66% and
the bar's +4.34%. This panel fails its bar, as it did in the gate
study; the reservation changes nothing here because the step was
already reachable without it.

### Controls

(a) Cancer battery, both arms ON: seeds 0–2 reproduce the recorded
post-fix test log losses bit-exactly — 0.12443863785108761,
0.06876267784646255, 0.1604252279147264 (mean 0.117876, recorded as
0.1179) — in Arm S and in Arm G alike. The evidence records no
joint-tree entry on any draw; the call site is Gaussian-only and the
reservation never engages.

(b) smooth_sparse (additive truth; autospec the best recorded
method): ON vs OFF is bit-identical in both arms on all three draws
— 0.258837, 0.242381, 0.259147. In Arm G the step scores both
candidate depths on every draw (reserved 9) and adopts on none:
"neither joint tree lowers the shared-fold risk by at least 2%"
each time, with candidate gains of −8.6% to −14.4%. In Arm S the
step is not reached. No draw worsens; the worsening is not merely
unsystematic, it is exactly zero.

(c) Coordinate fixture, Arm S, budgets 60 and 70 — the outcome is
the same at both budgets, in both modes:

| mode | coordinate-surface step | joint-tree step | held-out MSE (response) | held-out MSE (signal) |
|---|---|---|---|---|
| ON | skipped: "insufficient budget for five candidate sweeps and the final sweep" | adopted, depth 6 over lon, lat, z; reserved 21 | 0.477539 | 0.434847 |
| OFF | adopted: btree(lon, lat, maxdepth=3), chosen gain +94.94% | skipped: "insufficient budget for the incumbent and two joint-tree depths" | 0.112311 | 0.070196 |

ON ends at `y ~ btree(lon, lat, z, maxdepth=6)`; OFF ends at
`y ~ bbs(z, df=4.0) + btree(lon, lat, maxdepth=3)`. The trade
hurts, at both budgets, by a wide margin: ON's held-out MSE is
4.25× OFF's against the observed response (0.477539 against
0.112311) and 6.19× against the noiseless signal (0.434847 against
0.070196). The mechanism is visible in the two specifications. The
fixture's truth is additive across the split the surface step
exploits — a lon/lat interaction plus a separate z effect — and
the surface step's winning specification matches that structure:
a tree over the pair, a smooth term for z. The joint tree must
represent z inside the same tree that represents the interaction,
and at the fixture's size it does so worse. Reserving 21 of the
60–70 fits for the joint tree leaves the surface step unpaid, so
at these budgets the reservation buys the weaker of the two
candidate specifications and prices out the stronger one.

The supplementary budget-40 pair bounds the trade from the other
side. At budget 40, OFF fits neither step — both skip for budget —
and ends at the additive `y ~ bols(lon) + bols(lat) + bols(z)`,
held-out MSE 9.072826 against the response and 9.074783 against
the signal: the interaction is not fit at all. ON adopts the same
joint tree as at budgets 60 and 70, at the same held-out scores
(0.477539 / 0.434847); ON's outcome is in fact identical at
budgets 40, 60 and 70, because once the fits are reserved the
result no longer depends on the budget in this range. Below the
trade range, the reservation is the difference between fitting
the interaction and not fitting it.

## Verdict against the frozen bars

- Bar A (the reservation does its job): **met where the step can
  be entered.** In Arm G ON, all twelve panel draws and all three
  smooth_sparse draws score both candidate depths with reserved = 9
  recorded from the evidence, and no draw whose entry checks pass
  skips for budget; the paired OFF runs show the same step starved
  on every bike_iid draw. In Arm S the question does not arise on
  the frozen panels: the shipped entry checks never pass, the step
  is not reached, and the reservation's effect as shipped is
  exactly zero there.
- Bar B (fit quality): **not met** — 1 of 3 panels. Protein meets
  its bar at +9.16% (≤ +13.94%). bike_iid improves from +44.00% to
  +29.71% with the reservation and still fails +12.19%. Diamonds
  is unchanged at +15.24% against +4.34%.
- Bar C (do no harm): **(a) met** — the cancer battery reproduces
  its recorded values bit-exactly in both arms. **(b) met** —
  smooth_sparse is bit-identical ON vs OFF in both arms.
  **(c) measured** — on the coordinate fixture at budgets 60 and
  70 the trade hurts: the reservation is paid for by the
  coordinate-surface step, and the held-out cost is 4.25× against
  the observed response and 6.19× against the noiseless signal;
  at budget 40 the same reservation is the only reason the
  fixture's interaction is fit at all.

What the study establishes. First, the reservation works as built:
where the joint-tree step can be entered, the fits are there when
the step runs — bike_iid under the ungated configuration is the
proof case, starved on all three draws without the reservation and
scored on all three with it, a 9.9% mean test-loss reduction from
the reservation alone. Second, as shipped the reservation is inert
on the frozen panels, because the call site's coordinate-name gate
decides entry before the reservation is consulted; any future
change to that gate now has its numbers — including the matching
`_joint_tree_may_run` change, which Arm G exercised on every draw
without a spurious reservation. Third, reaching the step is not
reaching the counterfactual: bike_iid's remaining gap is adoption
quality on a single holdout, not reach. Fourth, the reservation is
not free on coordinate data: at budgets 60–70 it is bought with
the coordinate-surface step, and on the fixture that is a clear
loss, while at budget 40 it is the only route to the interaction.
Any change to the reservation's size, timing or arbitration
against the surface step has to price both sides of that trade.

Nothing in this study changes library code: the reservation stays
as shipped, and the ungated configuration was a measurement
configuration only. This record is the study's only product.

## Provenance

- Code under test: main at 92fa88d566bf2a7031aceac87c170485e7439acc;
  `python/mboostrs/autospec.py` blob
  4eb20ea548c3e638606807240ada3745af6c32d1 (Arm S). Arm G's file is
  blob b6739c75e35d9ec2e3bf5ca370720e70ba9abf72, produced from the
  Arm S file by the five edits of the protocol's D1–D4 and nothing
  else; the two study trees differ in no other source file (bytecode
  caches apart).
- Harness: the refresh driver's job path (payload, search, clean,
  loss) called per task and seed, with the reservation mode as the
  only switch; partition digests of every run match the refresh
  records and the gate study (for example protein seed 0
  sha256:22c84a1d8…, bike_iid seed 0 sha256:8da11e03a…, diamonds
  seed 0 sha256:6ea85f888…).
- Every loss in this record comes from the study's own run records,
  written per run as it completed; baselines are quoted from the
  gate study record and the refresh records it cites, opened in the
  same session.
