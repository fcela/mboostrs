# Paired estimation–discovery protocol

**Instrument:** `paired-estimation-discovery`, version `paired-ab/v1.1`.
**Status:** v1.1 ports measurement-validity pieces from the v2 design
(branch `feat/paired-estimation-discovery-511`, PR #527, not merged) into
the published v1.0 battery. An earlier unpushed draft
(`paired-ab/v1.1`/`v1.2`, laptop-local, never landed on the remote) is
not an ancestor of this instrument; the published versioning restarts at
v1.0, and this v1.1 continues that published lineage.

**Change log (published lineage):**
- `paired-ab/v1.1`: (1) A-arm stopping grows by doubling
  (`continue_to`) until the validation minimum is interior, ceiling
  3200 — the v1.0 fixed-cap-1000 argmin truncated 3/8 interaction
  replicates and contaminated estimation excess with a tuning
  limitation; `at_boundary`, `final_cap` and `growth_steps` are
  recorded per replicate. The B_discovery procedure is untouched.
  (2) Four-way train/selection/calibration/test partitions (60/15/15/10,
  n=1000 per panel): selection rows choose stopping iterations;
  calibration rows are held out disjoint from selection, reserved for
  future interval calibration (this battery does not calibrate
  intervals yet); test rows are touched once. All arms share the
  identical partition per replicate (paired). (3) Evaluator-only
  supplied-basis floors per panel: noiseless fits of the supplied and
  partial formulas, separating an inadequate basis from estimation
  error within the basis. (4) New task `threshold_jump`: a
  discontinuous threshold the default smooth basis cannot follow — the
  first task where discovery is expected to struggle, so the instrument
  demonstrates it can detect a gap. Deliberately not ported: v2's
  Task-dataclass redesign; `linear_sparse`/`mixed_factor_proxy`
  (redundant with `linear_factors`); v2's frozen baseline artifacts (a
  fresh frozen run follows after this lands).
- `paired-ab/v1.0`: initial published instrument (4 panels, paired A/B
  arms, budget-32 B arm, fixed-cap-1000 A-arm argmin).

**Design authority:** issue #511 (evaluation redesign: separate estimation
with a supplied specification from DGP discovery). Scorecard 2.1 maps:
Track A → D1a (recovery under a prescribed specification), Track B → D1b
(discovery from data alone); D11 (predictive validity) scores prediction
under matched budgets with no cross-credit to structure.

## Information contracts

*Frozen since v1.0 — do not change without a version bump and a design
review on the issue.*

| Contract | Information supplied | Work left to the method |
|---|---|---|
| A: supplied specification | Response family/link, relevant predictors, term types, required interactions; an adequate representable function space | Estimate effects and nuisance parameters; select smoothing/regularization and stopping using training/selection data; quantify uncertainty |
| A_partial: partial knowledge | Candidate predictors and plausible smoothness (all candidates as smooth terms), unknown interactions and scale effects | Same as A, from a weaker starting specification; scored as its own arm |
| B: discovery | Observed data, response semantics, column metadata, sampling design, a frozen resource budget | Choose predictors, representations and interactions, then fit, tune and report — via a frozen discovery procedure |

Coefficients, true curves, exact signal strengths, latent variables and
evaluation outcomes stay hidden from fitting/tuning in every arm. Track A
is not permission to compute predictions from the generator.

## Paired decomposition

For synthetic tasks, arms A and B run on the same samples. Let `R_A` and
`R_B` be independent-test risks, `R*` the risk of the optimal predictor
(Bayes risk, known for the synthetic panels), and `s > 0` a prespecified
task scale frozen per panel (here: `Var(y)` from a 50,000-draw pilot —
50 replicates of 1000 rows — at a fixed seed, recorded below):

- Estimation excess: `(R_A − R*) / s`.
- Discovery gap: `(R_B − R_A) / s`.
- Partial-knowledge gap: `(R_P − R_A) / s`.
- End-to-end excess: `(R_B − R*) / s`.

The first two sum to the third as an accounting identity, not as proof of
a causal mechanism. Signed gaps are preserved; paired uncertainty is
reported. Original loss units stay in the tables.

For real data the DGP is unknown: an expert formula is labeled
**prespecified**, never oracle truth.

## Precision contract

Each reported gap carries its paired CI half-width (t_{0.975,7} ×
SE over the 8 replicate gaps): the smallest absolute gap this design
resolves at the frozen replicate count. A gap of interest narrower than
that needs more replicates, not weaker claims. As a rule of thumb,
resolving a gap δ needs roughly 8 × (hw/δ)² replicates where hw is the
measured half-width — the report prints hw per gap so the arithmetic is
explicit, not implicit.

## Panels (v1.1 battery)

All panels are Gaussian with known noise variance σ² (so `R* = σ²`).
Per replicate: 600 train / 150 selection / 150 calibration / 100 test rows
from one RNG stream (n=1000); all three arms share the identical
partition (paired). The calibration split is held out, disjoint from
selection, reserved for future interval calibration. 8 replicates per
panel, seeds `511000 + replicate`.

| Panel | Truth | Driver groups (equivalence) | σ | s = Var(y) |
|---|---|---|---|---|
| `linear_factors` | y = 2·x0 − 1.5·x1 + x2 + 1.2·[g=b] − 0.8·[g=c] + N(0, 0.5²); x3 = x0 + N(0, 0.1²) (correlated substitute); x4..x9 null; g 3-level factor; x0..x9 ∼ N(0,1) | {x0,x3}, {x1}, {x2}, {g} | 0.5 | 8.1029 |
| `smooth_additive` | y = 2·sin(π·x0) + 1.5·x1² − x1 + 0.8·x2 + N(0, 0.5²); x0..x7 ∼ U(−2,2); x3 = x0 + N(0, 0.05²) (substitute); x4..x7 null | {x0,x3}, {x1}, {x2} | 0.5 | 7.6756 |
| `kinked` | y = 3·max(0, x0−0.3) − 2·max(0, 0.7−x1) + 2·sin(π·x2) + N(0, 0.5²); x0..x6 ∼ U(0,1); x3..x6 null | {x0}, {x1}, {x2} | 0.5 | 1.3379 |
| `interaction` | y = 3·(x0−0.5)(x1−0.5) + 1.5·sin(π·x2) + x3 + N(0, 0.7²); x0..x5 ∼ U(0,1); x4,x5 null. The product term has no linear marginal signal in x0 or x1 (R² < 0.01 on 20k draws) — the hard discovery case by construction | {x0,x1} (joint), {x2}, {x3} | 0.7 | 0.8511 |
| `threshold_jump` | y = 1.8·sin(1.5·x0) + 1.2·[x1 > 0.2] + N(0, 0.5²); x0..x9 ∼ U(−1,1); x2..x9 null. The smooth basis cannot follow the jump (partial floor ≈ 0.11 vs supplied ≈ 0.0025 on noiseless draws) — the first task where discovery is expected to struggle | {x0}, {x1} | 0.5 | 2.1375 |

A_supplied formulas (adequate representable spaces):
- `linear_factors`: `y ~ bols(x0) + bols(x1) + bols(x2) + brandom(g)`
- `smooth_additive`: `y ~ bbs(x0) + bbs(x1) + bols(x2)`
- `kinked`: `y ~ hinge(x0) + hinge(x1) + bbs(x2)`
- `interaction`: `y ~ bbs(x0) %X% bbs(x1) + bbs(x2) + bols(x3)`
- `threshold_jump`: `y ~ bbs(x0) + btree(x1, maxdepth=2, selection=sse)` — x1 declared threshold-like, expressed as a shallow tree (exact for a step); the location is NOT supplied

A_partial formula: all numeric candidates as `bbs`, factors as `brandom`,
no interactions.

B_discovery: frozen `autospec` — `budget=32, nu=0.1,
interactions="adopt", residual_tree="auto"`, fixed seed per replicate.

## Stopping rule

mstop is chosen on train/selection only; the test set is touched once,
after every arm has chosen. No oracle minima anywhere: the test path is
never scanned for a minimum.

- **A arms** (v1.1): fit the path on train (cap 1000, ν=0.1), score on
  the selection split, and grow by doubling (`continue_to`) until the
  selection minimum is interior (below 95% of the cap) or the ceiling
  3200 is reached. The ceiling (3.2× the v1.0 cap) bounds the cost to at
  most two doublings; it follows the #509 stopping-ablation evidence
  that boundary growth is never materially worse. `at_boundary`,
  `final_cap` and `growth_steps` are recorded per replicate; a nonzero
  boundary rate means the estimation excess on that panel mixes in a
  tuning limitation.
- **B arm** (frozen since v1.0): fixed-cap-1000 selection argmin, no
  growth. The discovery procedure is untouched by v1.1.

## Evaluator-only floors

Per panel, the supplied formula and the partial arm's formula are fit to
large noiseless draws from the DGP (y = truth exactly); the RMSE against
the truth is the floor the basis itself imposes. This separates an
inadequate basis from estimation error within the basis *a priori*:
when the partial floor sits far above the supplied floor, a
partial-knowledge gap is predicted before any arm runs. Floors are
evaluator-only — arms never see the noiseless draws and floors never
enter arm scoring. Each floor records its settings (rows, mstop, nu);
the reported RMSE includes a small optimization shortfall from the
finite-mstop boosting fit, not pure basis error.

## Structural accounting (the #509 audit rules)

*Unchanged from v1.0.*

Per fitted model, per term, record:

- `term`: the learner's disambiguated name.
- `is_tree`: **True iff the term appears in `btree_splits()`** —
  membership, never name parsing. A `btree(x, z)` named plain `x`
  (no `(btree)` marker) is still a tree.
- `candidate_columns`: every column the term may draw on —
  `spec.names` for trees (via `_learner_columns`), the learner's columns
  otherwise.
- `used_columns`: columns the term actually used — for trees, the union
  of numeric `splits[*]["variable"]` **and** `level_split_variables`
  (categorical split variables live there, not in `splits`); for other
  learners, the learner's columns when selected at least once, else empty.
- `numeric_cutpoint_count`: `len(splits)` for trees (a count of numeric
  cutpoint records — not a count of split nodes, boosting updates, or
  categorical splits); null for non-trees.

Split columns are **structure, never effect proof**: a recorded split can
have zero predictive effect (`build_tree_recursive` accepts any valid best
split without comparing to the unsplit parent), and entering every input
does not prove its signal was adequately fitted. Structural recall/FDP
are reported separately from predictive gaps; tree-term partial effects
are excluded from per-column effect recovery.

Structural scores (B arm; A_partial for reference):
- Driver-group recall: fraction of true driver groups with ≥1 used column
  in the group (equivalence groups: recovering any member counts; the
  interaction pair {x0,x1} requires both members used).
- False discovery proportion: fraction of used-column groups outside all
  true driver groups. Averaged over replicates; reported alongside recall
  so returning no findings cannot win.

## Failure accounting

Every arm fit is wrapped; failures (non-convergence, errors, timeouts)
are recorded with the replicate, arm, panel and traceback, and the run
continues. Failed replicates are reported, never silently dropped.

## What v1.1 does not measure

Interval coverage and uncertainty calibration (wanted by #511; the
calibration split is now reserved and disjoint for exactly this —
scheduled, not in this battery). Real-data prespecified-specification
runs. Spatial and location–scale panels (follow the #511 experiment
design order).

## Provenance

Every run records: instrument version, git commit, package versions,
per-replicate seeds, partition digests, frozen `s` values, wall time per
arm, A-arm stopping records (`at_boundary`, `final_cap`,
`growth_steps`), and raw per-replicate records (predictions digested,
never the test outcomes beyond the scored loss). Results:
`docs/benchmarks/paired_ab/`. The v1.0 battery numbers are not
comparable to v1.1 (different n, partitions, stopping rule and task
set); v1.0 results remain in the repository history.
