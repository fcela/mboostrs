# Coordinate-surface adoption on a single holdout: audit and fix (census C14)

## Evidence summary

The judging-layer census (`docs/design/judging-layer.md`) classifies
the coordinate-surface step (`_coordinate_surface_step`, census point
C14) as exposed: its adoption is decided by `coordinate_beats` — the
2% margin or the paired fold z — with no single-holdout audit. At one
fold the paired route is arithmetically inert (the fold-difference
vector has one element, so the recorded z is 0.0 and cannot fire),
leaving the bare margin to approve the adoption: the configuration
the sibling routes were repaired for in PRs #776 and #787.

The constructive test (`judging_constructive_test.md`) ran its frozen
protocol at C13 and C15 and returned kill criterion K2. Outside the
protocol it recorded one observation: in the secondary leg (Gaussian
additive panels, n = 364, a single supplied holdout of 91 judged
rows, seeds 7914000–7914039), the surface step adopted a term over
the coordinate pair in 4 of 40 draws, each time on the margin route
with the recorded fold z 0.0. The two coordinate columns of those
panels are pure noise, so every one of those adoptions is valueless
by construction. This study audits the four adoptions on fresh data
and, the defect confirming, applies the sibling fix.

## Audit method

- The four adopting draws were reproduced exactly, with the
  constructive test's generator and autospec call (budget 40,
  nu = 0.1, `interactions="adopt"`). Final formulas, margins and the
  inert fold z match the study's records to all printed digits.
- For each adoption, the incumbent at the surface step (the additive
  pair formula) and the adopted formula were judged the way the
  shipped audits judge: each side at its own selected step, refitted
  on the original 273 training rows, per-row losses paired through
  the production primitive (`_held_out_pair_z`). Two judgements are
  reported: on the original 91 judged rows (what a same-holdout audit
  reads) and on 20,000 fresh draws from the same process, on a stream
  the selection never touched (what the adoption is actually worth).

## Audit results

| seed | adopted term | margin | fold z | per-row z, own holdout | per-row z, 20,000 fresh rows |
|------|--------------|--------|--------|------------------------|------------------------------|
| 7914010 | `bspatial(x, y, df=6.0)` | +6.16% | 0.0 | 2.35 | −3.61 (mean −0.00290, se 0.00080) |
| 7914015 | `btree(x, y)` | +2.14% | 0.0 | 1.06 | −2.68 (mean −0.00172, se 0.00064) |
| 7914024 | `bspatial(x, y, df=6.0)` | +2.98% | 0.0 | 1.13 | +1.30 (mean +0.00103, se 0.00079) |
| 7914025 | `btree(x, y)` | +3.78% | 0.0 | 0.95 | −10.74 (mean −0.01759, se 0.00164) |

Selected steps (incumbent / adopted): 46/109, 199/187, 76/67, 41/76.

All four adoptions fail the 2.0 bar on fresh data; three of the four
lose significantly out of sample — the adopted stump at seed 7914025
costs 0.0176 of squared error per fresh row against a standard error
of 0.0016. The defect is confirmed: the adoptions are real
(reproduced exactly from the study's records) and repeatable (four
of four recorded cases fail the fresh audit, on both surface forms
the step can adopt).

## The fix

`_coordinate_surface_step` now follows the sibling pattern of
PRs #776/#787, scoped the same way:

- The single-holdout predicate is the shipped one
  (`_single_fold_design`: one supplied fold column, no scheme, not
  the distributional view). When it holds, the step's budget gate
  reserves the audit's two fold fits up front.
- When the shape rule would adopt, the per-row paired z of the
  adoption comparison — the additive pair against the chosen
  candidate, each at its own selected step — must also clear
  `_SHAPE_SE` = 2.0. The report's `chosen_z` then records the per-row
  z in place of the inert fold z, and a refusal states that the
  margin was cleared but the per-row bar was not.
- Families whose loss couples rows have no per-row losses; the audit
  returns no z there and the margin rule stands, as at the sibling
  sites. Multi-fold behaviour is untouched, as is the z-only "both"
  sub-decision (already proof-only, and inert rather than permissive
  at one fold).

Regression tests in `tests/test_single_holdout_rules.py`: the
recorded seed-7914025 adoption is refused with its audit z recorded,
and a planted product surface over the pair is still adopted with
its audit z (3.8) recorded. Both tests fail on the unfixed code.

## Evaluation

**Construction panels, fixed code.** The full 40-draw secondary leg
was rerun: surface adoptions fall from 4 to 1. The three refused
draws (7914015, 7914024, 7914025) end with the additive pair
standing, and the other 37 draws' final formulas are byte-identical
to the unfixed code — including the joint-tree adoption at seed
7914008 (census C15), which is not this fix's scope.

**The surviving adoption, stated plainly.** Seed 7914010's surface
clears the per-row bar on its own holdout (z 2.35) and is therefore
still adopted; the fresh audit condemns it (z −3.61). So the count
of adoptions failing the fresh audit falls from 4 to 1, not to 0.
That residue is the measured limit of any audit that reads the
supplied rows: at 91 judged rows the holdout can genuinely favour a
term that fresh rows reverse (the stopping study's finding, in
`single_holdout_stopping.md`). What the fix removes is approval by
the bare margin alone: every adoption on this route now carries a
per-row proof on the caller's own evidence, recorded in place of an
inert 0.0, and three of the four recorded failures are refused by it.

**Frozen cancer battery (#541).** Reproduced on current main before
the change (chosen steps 3261 / 3894 / 9989; judged test log losses
0.1244 / 0.0688 / 0.1604; mean 0.1179) and rerun after: every draw's
formula, chosen step and judged loss are bit-identical. The
battery's data carries no recognised coordinate pair, so the surface
step never engages there (no `coordinate_surface` evidence on any
draw); the battery stands as a no-regression check on the wider
autospec run.

**Tests.** `tests/test_single_holdout_rules.py` and
`tests/test_spatial_single_holdout.py`: 14 passed, including the two
new tests. The autospec suite (test_autospec*.py,
test_joint_tree_autospec.py, test_spatial_tree_choice.py,
test_bmrf_bspatial_smoke.py, test_compare_specifications_lss.py):
223 passed.

## Reproduction

Harness and records are kept with this study's working files at
`~/workspace/mboostrs-dev-wc14/` (`audit_c14.py` — reproduction and
both audits of the four draws; `leg40.py` — the full secondary leg
on the fixed code; `battery541.py` — the frozen battery in
before/after form; records in `audit_before.json`,
`audit_after.json`, `leg40_after.json`, `battery_before.json`,
`battery_after.json`). The code under audit is identical at the
constructive test's base (`ac8141bf`) and at this study's base
(`b59cd7b1`): under `python/` only `discover.py` changed between
them, and the compiled extension corresponds to Rust sources
unchanged across both.
