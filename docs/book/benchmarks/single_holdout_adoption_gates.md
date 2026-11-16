# Single-holdout adoption gates: the residual tree, the shape probe, and the pair-form picker (#541)

## Provenance

The two fixes this record covers shipped with their measurements in
their pull requests only — PR #776 (the residual-tree step) and
PR #787 (the shape probe and the pair-form picker) — and no record
under `docs/book/benchmarks/`. This record was written after the
fact from those reported measurements, cross-checked against the
studies' retained run records. The run records agree with the
reported numbers at every printed digit, with one rounding
difference in a narrative example: the retained shape audit gives
draw 2's x24 replacement gain as 0.1885, printed as 18.9% in
PR #787; this record uses the retained value. The battery state the
two fixes leave behind is the starting point of
`single_holdout_stopping.md`, which evaluates stopping after those
adoption gates.

## The defect class

Every stage of the specification search reads the same supplied
fold. On a single fold — one supplied holdout — every paired fold
route in the decision stack is inert: the paired fold standard
error is undefined, the fold z in each adoption rule is identically
0, and a bare 2% margin on the aggregate held-out curve decides.
The curve is read off the same rows that judge it, so its minima
carry a winner's-curse bias the margin does not price. The remedy,
applied route by route, is the standard the fold route uses when
folds exist: on a single supplied holdout, adoption must also clear
the paired per-row bar, z ≥ 2.0, computed from per-row held-out
losses on the judging rows (fold fits refit on the training rows,
each side read at its own chosen step).

## The battery

The frozen small-tier battery of #541: three supplied outer splits
of the breast-cancer panel; each draw fits on 273 rows, validates
on a single supplied holdout of 91 rows, and is judged on a
disjoint calibration partition of 91 rows and a test partition of
114 rows. The autospec call is fixed throughout: budget 32,
nu = 0.1, initial mstop 200, interactions adopt, residual tree
auto, one supplied fold per draw. The baseline on the code before
either fix, reproduced exactly at the start of the first study:

| draw | fits | mstop | test log loss | calibration log loss |
|------|------|-------|---------------|----------------------|
| 0 | 24 | 9990 | 0.1020 | 0.0819 |
| 1 | 22 | 9993 | 0.0907 | 0.0531 |
| 2 | 22 | 2269 | 0.4219 | 0.5580 |
| mean | | | 0.2049 | |

Draw 2 dominates: it validates at 0.0322 and tests at 0.4219, while
a prespecified all-linear specification tests at 0.1179 on the same
draw. The issue's target is a battery mean within 5% of the best
recorded comparator (about 0.0901); the all-linear counterfactual
on this battery is 0.0989.

## Fix 1: the residual-tree step (PR #776)

### Audit

Auditing draw 2's adoptions with per-row losses on the same 91
validation rows separates the stages:

- The residual tree is the failure. It was adopted on a 34.7%
  relative gain of path minima (recorded relative gain 0.347371);
  its per-row paired z is 1.47 at the procedure's own chosen steps
  and 0.86 in a train-only path audit at grid steps. On the
  disjoint calibration rows its effect is −0.21 per row
  (z = −1.88). At the adopted step count the tree term carries a
  contribution share of 1.0: the adopted model is effectively the
  depth-5 all-column tree alone, and it scores 0.558 log loss on
  the calibration partition, worse than on test.
- The two adopted interaction pairs are not the failure: their
  per-row evidence is weakly positive (z 0.8–1.7), and a refit
  counterfactual with the pairs kept and the tree removed tests at
  0.1212 on draw 2.
- Stopping amplifies the tree but does not cause the failure: the
  selected formula's validation curve is still falling at the
  10,000-step cap while the calibration and test curves bottom out
  at steps 50–120, yet the same long stop on the tree-free formula
  tests at 0.1212.
- Ruled out: capacity (the flexible models record lower validation
  minima on all three draws and test worse), calibration as the
  primary fault (AUC stays 0.96; the loss is extremeness from the
  adopted term), budget overrun (22 of 32 fits spent), and scoring
  (settled by the earlier prediction-scale audit).

### The change

In `_residual_tree_step`, when the judging design is a single
supplied holdout (one fold column, no fold scheme), adoption
additionally requires the paired per-row z of the held-out losses —
incumbent against candidate, each at its own chosen step — to clear
2.0. The evidence entry then reports the per-row gain, its standard
error and its z in place of the undefined fold ones. Two fold fits
pay for the audit and are charged to the budget, reserved up front
in the step's budget check. Families whose loss couples rows
(CoxPH, AUC, Cindex, Gehan) have no per-row losses; their
single-holdout decisions keep the margin rule, as do all multi-fold
and scheme designs, which are untouched.

Two adjacent changes were implemented, measured, and rejected:

- A hard per-row bar for interaction pairs: it also rejects a pair
  on draw 0 whose per-row z is 1.41, and that draw then regresses
  from 0.1020 to 0.1513. Pairs keep the margin rule.
- A one-standard-error stopping step for single holdouts, in two
  forms. Level form (earliest step within one level standard error
  of the minimum): draws 0.1047 / 0.1057 / 0.1312, mean 0.1139 —
  draw 1 regresses from 0.0907, because the loss level's spread
  stops a steadily improving path at step 237 of 9993. Paired form
  (within one paired standard error): draws 0.1513 / 0.0705 /
  0.1295, mean 0.1171. Neither dominates the shipped change, so
  neither is shipped.

### Battery after the change

| draw | fits | mstop | test log loss | AUC | Brier | calibration log loss |
|------|------|-------|---------------|-----|-------|----------------------|
| 0 | 24 → 24 | 9990 → 9990 | 0.1020 → 0.1020 | 0.9917 | 0.0324 | 0.0819 |
| 1 | 22 → 22 | 9993 → 9993 | 0.0907 → 0.0907 | 0.9970 | — | 0.0531 |
| 2 | 22 → 30 | 2269 → 972 | 0.4219 → 0.1306 | 0.9595 → 0.9858 | 0.0670 → 0.0341 | 0.5580 → 0.0890 |
| mean | | | 0.2049 → 0.1078 | | | |

Draws 0 and 1 are byte-identical to the baseline: their tree steps
never adopted, so the gate never engages. On draw 2 the tree step
records relative gain 0.347371, held-out gain 0.017182, held-out
standard error 0.011726, z 1.47, adopted False, and the search
proceeds without the tree.

Synthetic controls under the same configuration (364-row panels,
single 273/91 holdout, fresh-sample outer loss, four draws each —
correlated-linear, nonlinear with one true product term, and
null): every control draw is identical to the baseline — same outer
loss, stopping point, adopted pairs, and no tree adoption under
either version; the true product pair is adopted wherever the
baseline adopts it.

Regression tests: `tests/test_single_holdout_rules.py` was created
with four tests (the scoping predicate; the per-row audit
certifying a genuine tree, sign-product truth, z 5.15; the audit
not certifying a valueless tree on a linear-truth panel; a scope
guard that multi-fold stopping still returns the plain argmin).
The autospec-related suites pass on the final tree, 155 tests.

What remains: the mean 0.1078 is close to the all-linear
counterfactual 0.0989 but still above the best recorded comparator
(about 0.0901), so the 5%-excess target is not met. The residual
gap is the flexible-versus-linear difference on draws where nothing
was mis-adopted. The fix's report flags the shape probe — which
makes the same margin-only decisions on a single holdout — as the
next place to look.

## Fix 2: the shape probe and the pair-form picker (PR #787)

### Audit

Both sibling routes flagged by the first fix carry the same
defect, measured on the battery before any change:

- **Shape probe.** Across the three draws the probe made 20 shape
  replacements (draw 0: 4, draw 1: 5, draw 2: 11; mostly bbs→btree;
  aggregate gains 2.2%–22.5%). Audited with per-row losses on the
  same holdout rows, the two full formulas at the steps the probe
  judged, every one of the 20 has a paired per-row z under 2.0
  (range −0.84 to 1.66; four are ≤ 0 — draw 2's x22 at a 10.5%
  margin has z −0.84, x24 at an 18.8% margin has z −0.01). Under
  the one-term statistic the production audit uses (each term's own
  one-term path at its chosen step), the checked cases agree:
  x22 z 1.12, x24 z 0.98.
- **Pair-form picker.** Draw 0 adopted a tensor form at a 3.5%
  margin (per-row z 0.50); draw 2 adopted a tensor form at a 4.0%
  margin (per-row z 0.65; the production evidence recorded paired
  z 0.0, the inert fold route). Draw 1's picker adopted neither
  form.
- **Synthetic panels.** On additive-truth panels the picker adopted
  a valueless joint-tree form on a single holdout (margin 4.2%,
  per-row z 1.53), and never adopted on the same panels under five
  folds.

### The change

Both routes apply the first fix's remedy, scoped identically: when
the judging design is a single supplied holdout (one fold column,
no fold scheme, no judge set), the decision must also clear the
paired per-row z ≥ 2.0 read from the comparison that produced the
margin. Multi-fold designs, fold schemes, judged designs and
row-coupled families keep the existing rules exactly.

- Shape probe (`probe_shapes`): a replacement that passes on the
  aggregate margin alone is audited before it is made — with a fold
  backfit, from each term's one-term path at its own chosen step
  (`_FoldBackfit.risk_rows`, `_shape_row_z`); without one, from the
  two full formulas at their path-chosen steps
  (`_held_out_pair_z`). The evidence sentence reports the outcome
  either way, and the audit's evaluations are charged to the probe
  budget at the per-variant rate.
- Pair-form picker (`autospec`): when the winning form's margin
  warrants adoption, the audit refits the additive and winning
  forms at their own selected steps on the same holdout and
  adoption requires its z ≥ 2.0. Two fold fits are reserved up
  front in the picker's budget check. The evidence records the
  audit's z as `paired_z` — the z the decision used — in place of
  the inert fold 0.0.

Regression tests: four added to `tests/test_single_holdout_rules.py`
(a valueless shape replacement blocked, one-term per-row z 1.7
against a 6.3% margin; a genuine step-truth replacement kept at
z 3.6; a valueless picker adoption vetoed by the audit at z −0.9
with the margin verdict forced, because with the shape gate in
place the additive panels no longer present the picker a live
margin case end-to-end; a planted tensor interaction adopted at
z 4.7). The file runs 8 tests; the autospec-related suites run 171
tests, all passing.

### Battery after the change

| draw | test log loss, fix 1 merged | test log loss, this change | what the gates did |
|------|-----------------------------|-----------------------------|--------------------|
| 0 | 0.1020 | 0.1244 | 4 shape replacements blocked; picker adoption blocked by the audit (pair x23×x27, margin 10.6%, per-row z 1.21 recorded as `paired_z`) |
| 1 | 0.0907 | 0.0688 | 4 of 5 replacements blocked; x17's btree replacement kept (its one-term per-row z clears the bar; the full-formula reading of the same swap is 1.54 — the audit reads the comparison that produced the margin, the one-term paths); picker rejected on margin, audit not run |
| 2 | 0.1306 | 0.1604 | all 11 replacements blocked; picker degenerate (the pair was already in via the pair route), audit not run; the residual tree is rejected on the aggregate margin (−10.5%) against the corrected incumbent, before its own audit |
| mean | 0.1078 | 0.1179 | |

Chosen steps after the change: 3261 / 3894 / 9989. The tree gate
itself is untouched by this change and, run on the pre-change tree,
reproduced its fix-1 draw-2 record exactly (relative gain 0.347371,
per-row z 1.47, not adopted).

The mean is worse, and the mechanism is measurable. A
pre-implementation simulation (replacements forced to stand, picker
adoptions blocked, everything else untouched) predicted 0.1244 /
0.0750 / 0.1604 (mean 0.1199); the production run lands on 0.1244 /
0.0688 / 0.1604, slightly better because the gate kept draw 1's
genuine replacement instead of blocking it too. Train-only refits
of draw 2's baseline and blocked formulas at matched steps are
close (step 972: test 0.1239 against 0.1253; step 500: 0.1229
against 0.1277; the blocked formula is much better on the disjoint
calibration rows, 0.0575 against 0.0805 at step 972). What separates
them is where each stops: the blocked formula's validation curve is
still falling at its chosen stop (9989, the cap) while its test
curve has risen since about step 1000 (0.1253 at step 972 to 0.1765
at 9989 in the refit). That is the single-holdout stopping rule —
argmin on the reused holdout — re-tuning on the surviving formula:
the mechanism the first fix documented and deliberately left
unchanged. The gates remove adoptions whose per-row evidence does
not certify them; they cannot fix where the surviving formula
stops, and on this battery that untouched rule, not the blocked
terms, carries the difference. The follow-on study of that rule is
`single_holdout_stopping.md`.

### Power cost, measured

The bar is a per-row test on 91–200 holdout rows, and it costs
power exactly where the fold route's bar does. On a step-truth
panel (n = 364, 91 holdout rows), 4 of 8 genuine btree
replacements have a one-term per-row z under 2.0 and are blocked;
on synthetic interaction panels the genuine picker adoptions clear
the bar in 5 of 8 draws. The synthetic controls (C1/C2/C3, four
draws each, scored on 20,000 fresh rows, against the first fix's
run of the same apparatus): 8 of 12 draws are identical; C1 draws
1–3 improve (fresh log loss 0.7132→0.6914, 0.4702→0.4410,
0.5170→0.5122); C2 draw 1 worsens (0.5328→0.6225) — there the true
x2×x6 product's picker form carries an 8.4% margin but a per-row z
of 1.08 on 91 rows, so the audit vetoes a genuine form (the same
run's noise-column shape replacement is blocked at z 1.4). Mean
fresh log loss across the 12 draws: 0.5878 → 0.5906, carried by
that one draw. This is the trade the merged tree gate already
makes; the alternative measured on this battery is 20 of 20 shape
replacements and 2 of 2 picker adoptions carrying per-row evidence
under the bar.

## State after both fixes

Every adoption route in the search that can fire on a single
supplied holdout — residual tree, shape replacements, pair-form
picker — now requires the paired per-row bar there. The battery
mean stands at 0.1179, worse than after the first fix alone
(0.1078) for the stopping reason recorded above, and the #541
target (within 5% of the best comparator, about 0.0901) is not
met. What the two fixes establish is narrower: no adoption on a
single holdout is taken on evidence the procedure's own standard,
applied on every other route, would not certify. The stopping
question the battery then exposes is studied in
`single_holdout_stopping.md`; its verdict is that no candidate
rule clears the bar fixed for it, and the argmin behaviour stands.
