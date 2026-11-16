# Joint-tree gate: study record

## Study objective

Issue #539: the automatic specification search considers replacing its
model with one joint tree over all predictors only when the data
contains a coordinate pair. The final joint-tree step in
`python/mboostrs/autospec.py` was gated on `_has_coordinate_pair(...)`,
a check on column names, and its replacement helper admitted only
numeric columns. On three measured non-coordinate panels of the
competitive refresh (`docs/book/benchmarks/competitive_refresh/`),
autospec's mean test losses sat far above the best recorded competitor,
while a diagnostic counterfactual — a joint tree over all predictors,
depth chosen from (3, 6) on the same training/validation split —
reached near-parity. The study measures whether considering the joint-tree candidate on
non-coordinate panels, under the existing adoption bars, reduces their
excess losses toward the counterfactual without harming controls.

## The change under test

One file, `python/mboostrs/autospec.py`:

- The call site no longer requires `_has_coordinate_pair(joint_modelled)`
  for the final joint-tree step to run.
- Inside the step, the coordinate-pair early return is replaced by an
  availability check (at least two free main-effect columns), and the
  column filter admits boolean and categorical columns alongside
  numeric ones, matching the kinds the residual-tree step already
  admits for its tree.
- The retained-column construction feeding the step admits the same
  three kinds. Its term-shape condition is unchanged: a column counts
  only while its current term is a single-column `bbs`, `bols`,
  `btree` or `bns` term, so a column carrying a `brandom` group term
  is not a replacement input and its term is kept.
- Two docstring passages that described the comparison as
  coordinate-triggered were updated to match.

Everything else is unchanged: the protected-term closure, the
heredity/forbidden-pair refusal, the budget reservation of
3 × (folds + 2) fits, the candidate depths (3, 6), the tree step
length, the adoption bar (the better candidate must lower the
shared-fold risk by at least 2% relative to the incumbent), and the
evidence shape. The coordinate-surface comparison and the
residual-tree offer keep their existing coordinate logic.

## Frozen protocol

Fixed before any study run.

- Development panels: protein, bike_iid, diamonds — frozen shootout
  splits, outer draws (seeds) 0, 1, 2 each; the refresh driver's own
  path (autospec budget 32, nu 0.1, mstop cap 200, single supplied
  holdout, refit on the development rows, test scored after
  development-fit cleaning; loss is MSE on the test rows). Before =
  the code as it stands on main; after = the change above; identical
  harness both sides.
- Controls: (a) the cancer battery, seeds 0–2, after the change — the
  step's call site is restricted to Gaussian families, so the battery
  should reproduce its recorded post-fix test log losses exactly;
  (b) smooth_sparse, seeds 0–2, before and after — chosen from the
  refresh records because autospec is the best recorded method there
  and its truth is additive; (c) null draws for the joint step: the
  judging-layer constructive test's secondary construction
  (`docs/book/benchmarks/judging_constructive_test.md`, census point
  C15) with the coordinate recognition removed — four U(0,1) columns
  x0–x3, response 1.5·sin(π·x2) + x3 + 0.7·ε (additive truth; the joint
  tree has nothing real to adopt), n = 364, seeds 7914000–7914039, the
  first 91 permuted rows judged, autospec budget 40 in the shipped
  single-holdout harness shape. That construction, with two noise
  columns renamed to a recognised coordinate pair, measured this step
  adopting in 1 of 40 draws.
- Success bars: (1) on at least 2 of the 3 development panels, the
  mean excess over the best recorded non-mboostrs method falls to
  within 5 points of the counterfactual's excess — protein ≤ +13.94%
  (mean loss ≤ 20.339), bike_iid ≤ +12.19% (≤ 0.104771), diamonds
  ≤ +4.34% (≤ 0.011718); (2) no development panel ends worse than it
  started; (3) every control stays within its paired noise; (4) the
  null-adoption rate is at most 2 of 40.

## Baselines

Re-derived from the refresh's per-draw records
(`docs/book/benchmarks/competitive_refresh/small.json`) and the
counterfactual probe
(`docs/book/benchmarks/competitive_refresh/joint_replacement_probe.json`);
partition digests agree across the two records and with this study's
runs.

| panel | autospec (recorded) | best recorded non-mboostrs | counterfactual | counterfactual excess |
|---|---|---|---|---|
| protein | 25.24968 | 17.85089 (CatBoost, tuned) | 19.44626 | +8.94% |
| bike_iid | 0.134476 | 0.093386 (CatBoost, small) | 0.100106 | +7.19% |
| diamonds | 0.013533 | 0.011231 (CatBoost, small) | 0.011157 | −0.66% |

The harness reproduces the recorded autospec draws before any change:
protein and bike_iid match the records to at least nine significant
digits on all three draws, as do diamonds seeds 1–2 and all three
smooth_sparse draws. Diamonds seed 0 does not reproduce: the current
search reaches a smaller specification (four terms against the
recorded seven) and a lower loss, 0.013715 against the recorded
0.014010. Both autospec source files differ between the refresh
revision and current main (blob hashes in the provenance), and the
intervening changes include the single-holdout adoption bars recorded
in `docs/book/benchmarks/single_holdout_adoption_gates.md`. The
before/after comparisons below are paired on identical code apart
from the change under test, so the drift does not enter them; where
an excess is quoted against the recorded baseline it is labelled.

## Results

### protein

| seed | before | after | joint-tree outcome |
|---|---|---|---|
| 0 | 24.647187 | 18.983026 | adopted, depth 6, held-out gain +18.27% |
| 1 | 25.765589 | 20.521879 | adopted, depth 6, held-out gain +13.81% |
| 2 | 25.336264 | 18.952984 | adopted, depth 6, held-out gain +23.80% |
| mean | 25.249680 | 19.485963 | |

The adopted specification on each draw is a single tree over the
panel's nine predictors (F1–F9). Mean excess over the best recorded
competitor: +41.45% before, +9.16% after, against the counterfactual's
+8.94% and the bar's +13.94%. This panel meets its bar.

### bike_iid — candidate not scored

| seed | before | after | joint-tree outcome |
|---|---|---|---|
| 0 | 0.134897 | 0.134897 | skipped: insufficient budget |
| 1 | 0.128918 | 0.128918 | skipped: insufficient budget |
| 2 | 0.139613 | 0.139613 | skipped: insufficient budget |
| mean | 0.134476 | 0.134476 | |

The step now runs its entry checks on this panel, but on every draw
the search had spent more than 23 of its 32 fits when the step was
reached, and the step reserves 3 × (folds + 2) = 9 fits to score the
incumbent and the two candidate depths. The recorded skip reason is
"insufficient budget for the incumbent and two joint-tree depths" on
all three draws. Removing the name gate does not reach this panel at
the small-tier budget; the excess stays at +44.00% against a bar of
+12.19%. This panel fails its bar because the candidate does not have enough
reserved fits to be scored.

### diamonds

| seed | before | after | joint-tree outcome |
|---|---|---|---|
| 0 | 0.013715 | 0.012660 | adopted, depth 3, held-out gain +64.49% |
| 1 | 0.012659 | 0.012659 | scored, refused by the 2% bar |
| 2 | 0.013930 | 0.013509 | adopted, depth 3, held-out gain +2.13% |
| mean | 0.013435 | 0.012942 | |

On the adopting draws the tree covers carat, depth, table, x and z.
The adopted specifications keep the terms the replacement does not
cover: bols(clarity, by=color) on both draws, and brandom(cut) on
seed 2, where deselection had kept it — the categorical predictors
carry no single-column main-effect term in these specifications, so
the replacement neither covers nor removes them. On seed 2 the
depth-6 candidate scored −16.11% while depth 3 cleared the bar at
+2.13%. The seed-0 held-out gain (+64% at the path minima on the
single holdout) did not carry to the test rows at the same size: the
test loss fell 7.7%. Mean excess over the best recorded competitor:
+19.62% before (measured; +20.50% on the recorded baseline), +15.24%
after, against the counterfactual's −0.66% and the bar's +4.34%.
This panel fails its bar.

### Controls

(a) Cancer battery, after the change: seeds 0–2 reproduce the
recorded post-fix test log losses exactly — 0.12443863785108761,
0.06876267784646255, 0.1604252279147264. The step never runs for
a Binomial search (its call site is Gaussian-only), and the evidence
records no joint-tree entry on any draw.

(b) smooth_sparse (additive truth; autospec the best recorded
method): the step scored both candidate depths on all three draws
after the change and adopted on none — "neither joint tree lowers
the shared-fold risk by at least 2%" each time. Test losses are
bit-identical before and after: 0.258837, 0.242381, 0.259147.

(c) Null draws: the step scored both candidate depths in 40 of 40
draws and adopted in 1 — seed 7914008, at a held-out gain of
+3.69%, the same draw and the same gain as the single adoption the
coordinate-named construction produced. The better depth's gain was
usually negative (median −7.96%); the second-highest draw reached
+1.01%. Removing the name gate does not change the step's behaviour
on noise: the 2% bar refuses it at the background rate.

## Verdict against the frozen bars

1. At least 2 of 3 development panels within 5 points of the
   counterfactual: **not met** — 1 of 3 (protein). bike_iid never
   scored the candidate; diamonds fell from +19.62% to +15.24%,
   short of +4.34%.
2. No development panel worse than it started: **met** — protein
   and diamonds improve on mean and on no draw regresses; bike_iid
   is unchanged to the last digit.
3. Controls within paired noise: **met** — the cancer battery is
   bit-identical to its recorded values and smooth_sparse is
   bit-identical before and after.
4. Null adoptions at most 2 of 40: **met** — 1 of 40, the same draw
   and gain as the coordinate-named background measurement.

The change does not ship. `python/mboostrs/autospec.py` is restored
bit-exactly (blob hash in the provenance); this record is the study's
only result. On protein, removing the name gate lets the candidate be
considered and adopted on every draw, bringing mean excess from +41.45%
to +9.16%, close to the +8.94% counterfactual. On bike_iid, the step
cannot be scored within the 32-fit budget because earlier stages leave
fewer than the nine fits it requires. On diamonds, the tree is considered
and adopted twice, but mean excess remains +15.24% against the
counterfactual's −0.66%; the held-out gains do not carry through to the test
rows. The study passes its no-regression and control bars, but not the
required improvement on at least two development panels.

## Provenance

- Code under test: main at 8ee9759150b484c55e9aa83498fa3a952fabc151;
  `python/mboostrs/autospec.py` blob fffdcbd09bd38d93855395315ecc9d7eeb8204fa
  before the change, e7877a2a8a7bd76960cc0332d0484e009548d445 with it,
  restored to fffdcbd09bd38d93855395315ecc9d7eeb8204fa after the verdict.
- Refresh revision 6c4c194: `python/mboostrs/autospec.py` blob
  20b59b266bbf19c5e5c9310c9c773a01f4f64e4d, `python/mboostrs/_autospec.py`
  blob 757473585b3cfbd97ef15198e9ca65a39a898d7d; at current main the
  latter is 825a3ee9b983eb863016d3a32c853d28e782dfe2.
- Harness: the refresh driver's job path (payload, search, clean,
  loss) called per task and seed; partition digests of every run
  match the refresh records (for example protein seed 0
  sha256:22c84a1d8…, bike_iid seed 0 sha256:8da11e03a…, diamonds
  seed 0 sha256:6ea85f888…).
- Every loss in this record comes from the study's own run records,
  written per run as it completed; baselines are quoted from the two
  refresh record files named above, opened in the same session.
