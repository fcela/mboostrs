# Judging layer: constructive test — study record

## Study objective

The design note `docs/design/judging-layer.md` censused 28 decision
points in the automatic discovery procedures and classified 9 as
exposed: a decision that can be approved on rows already used to
choose it, with the stronger proof switched off in the configuration
the caller supplied. The note froze a gate for building the proposed
judging layer (its section 4): before anything is built, the current
code must be shown actually failing at an unpatched site, under a
fixed construction. If neither construction site produces the failure
at the stated rate, kill criterion K2 applies — do not build; the
census stands as documentation. This study runs the frozen
construction exactly as written, on current main
(`ac8141bfdd7c7fb75bf3967a53d25c88b185758d`).

## Method

Both sites, as frozen by the note:

- **Primary, C13 (`_held_out_pairs`).** Gaussian panels, n = 364,
  additive truth, no interactions; `autospec` with
  `interactions="adopt"` on a single supplied holdout (273 training
  rows, 91 judged rows). 40 draws, seeds 7913000–7913039. The
  construction succeeds if the current code keeps at least one
  valueless pair on a bare ≥2% removal margin in at least 4 of 40
  draws. At one fold the keep rule's paired route is arithmetically
  inert (the fold-difference vector has one element, so its z is
  recorded as 0.0 and cannot fire), so any keep at this site is a
  bare-margin keep by construction.
- **Secondary, C15 (`_joint_tree_step`).** The same panel family with
  two columns the step recognises as a coordinate pair, `joint_tree`
  at its default. 40 draws, seeds 7914000–7914039. The construction
  succeeds if the joint tree is adopted (≥2% at the path minima; no
  other proof exists at this site) in at least 4 of 40 draws.
- **Control.** The identical primary panels under five folds, as the
  note specifies: the five-fold kept set is the reference.

The note pins the panel family, sizes, seeds, and bars; it does not
pin every knob. The readings chosen, each the closest available
anchor, are:

- Panel form: the PR #787 picker measurement's additive panel,
  verbatim — four U(0,1) columns,
  `y = 1.5·sin(π·x2) + x3 + 0.7·ε` (two main effects; x0, x1 are pure
  noise columns, so all six column pairs are valueless as
  interactions), generated at the note's n = 364 with the note's
  273/91 split.
- Budget 40 and step length nu = 0.1, the picker measurement's call;
  `autospec`'s seed set to the draw seed, as in the shipped
  single-holdout harnesses. Both sites were verified live at this
  budget: the keep step ran in 40/40 primary draws and the joint-tree
  step scored both depths in 40/40 secondary draws.
- Split: one random stream per draw (panel, then permutation); the
  first 91 permuted rows are the judged rows, marked 0.0 in a
  one-column fold-weight matrix — the construction of the shipped
  single-holdout tests.
- Secondary panel: the recognition is by column name
  (`_COORDINATE_PAIRS`), so the two noise columns are renamed `x`
  and `y`; the signal columns keep their names, and the response is
  renamed to avoid colliding with the predictor `y`. The joint tree,
  when adopted, replaces the free main effects of all four numeric
  columns, as the step is written.
- For every pair the keep step retained, and for the one joint-tree
  adoption, the per-row paired z the layer would read was computed
  with the production audit primitives (`_path_held_out_evidence`,
  `_held_out_pair_z`), each side at its own selected step — the
  shipped audits' method. Verdicts were recorded through a wrapper
  that observes `_held_out_pairs` without altering it; both
  exhibiting primary draws were re-run in a fresh process and
  reproduced to all printed digits.

## Results

### Primary site: 2 of 40 draws (bar: 4)

The keep step judged 115 pairs across the 40 draws. Removal margins
ran from −0.0896 to +0.0355 (median 0.0; 90th percentile +0.0090).
Two draws kept a valueless pair on the bare margin:

| seed | kept pair | removal margin | per-row audit z |
|------|-----------|----------------|-----------------|
| 7913021 | `bbs(x0) %X% bbs(x1)` (both columns noise) | +0.0294 | 1.28 |
| 7913039 | `bbs(x3) %X% bbs(x1)` | +0.0355 | 0.70 |

Both pairs survived every later stage and stand in the shipped
formula of their run. Both audit z values are under the 2.0 bar the
layer would apply: the failure the note describes — a keep approved
by the margin the selection itself produced, refused by the per-row
proof — occurred, twice. The patched sibling route behaved as the
note expects: the pair-form picker adopted nothing in 40 draws,
refusing margins of up to +0.117 whose audit z stayed under its bar.

### Secondary site: 1 of 40 draws (bar: 4)

The joint-tree step scored both candidate depths in all 40 draws.
The better depth's gain was usually negative (median −0.0849); the
second-highest draw reached only +0.0101. One draw adopted:

| seed | adoption | gain at path minima | per-row audit z |
|------|----------|---------------------|-----------------|
| 7914008 | `btree(x, y, x2, x3, maxdepth=6)` replaces the whole additive specification | +0.0369 (risk 0.6607 → 0.6363) | 0.14 |

### Control: the five-fold reference

At the protocol's budget of 40, the five-fold keep step never
executes: its budget gate (5 folds × (1 + pairs) validation fits) is
unpaid in 40/40 draws, the adopted pairs are restored away as
unvalidated, and the C13 kept set is empty in 40/40 draws. The kept
sets agree with the single-holdout runs in 38/40 draws; the two
exceptions are exactly the exhibiting draws above, whose pairs do
not stand under five folds.

Because a kept set produced by a budget gate is a weak reference, the
control was also run at budget 150, where the step executes (a
supplement, labelled as such; the frozen construction itself ran at
40). The step then judges the same 115 pairs and keeps 2 — on two
draws where the single holdout kept none (7913018, 7913032), the
multi-fold route's own verdicts with its z route live. Both
exhibiting pairs are refused: on seed 7913021's panel the x0–x1 pair
reads margin +0.0120 with fold z 1.53 against its 2.1 bar; on seed
7913039's panel the x3–x1 pair reads margin −0.0007 with fold z
−0.26.

### Outside the frozen protocol

One observation, recorded because the census lists the site as
exposed (C14, `_coordinate_surface_step`); it is not a construction
site under the note and plays no part in the verdict. In the
secondary runs the surface step adopted a term over the coordinate
pair in 4 of 40 draws, each time on the margin route with the
recorded fold z 0.0 (inert at one fold) — e.g. seed 7914025, where a
stump `btree(x, y)` was adopted at +0.0378. If a future construction
is frozen, this site produced failures at the rate the two frozen
sites were asked for.

## Verdict

Kill criterion K2, as written: the construction fails at both sites
at the stated rates — 2 of 40 at the pair keep rule and 1 of 40 at
the joint-tree step, against a bar of 4 of 40 each. **Do not build
the judging layer. The census stands as documentation.**

The two sites failed three times in 80 site-draws. Each failure was an
approval based on the bare ≥2% margin, while the per-row audit z values
(1.28, 0.70, 0.14) fell below the bars used at the patched sites. At n = 364,
the screen and tuned fit seldom produce margins that can be approved; the
removal margins cluster near zero because the joint sweep makes little use
of the candidate pairs. These results support the note's alternative reading:
the sites are reachable in principle but rarely reached at representative
sample sizes. The layer would therefore consolidate four patches without
changing their decisions, which does not justify the migration risk.

## Reproduction

Harness and raw records (per-draw evidence for all four legs:
primary, secondary, and the five-fold control at budgets 40 and 150)
are kept with this study's working files at
`~/workspace/mboostrs-dev-wjct/` (`run_construction.py`; records in
`records/`). The study ran against a blob-verified reconstruction of
main at `ac8141bfdd7c7fb75bf3967a53d25c88b185758d` (every file under
the code directories byte-identical to the remote tree; the compiled
extension corresponds to that tree's Rust sources, which are
unchanged from the sibling studies' builds). No library code was
modified.
