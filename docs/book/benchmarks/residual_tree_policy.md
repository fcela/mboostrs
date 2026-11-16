# Residual-tree adoption policy: study for #244

## Observed failure

Issue #244 records the default search finishing 39–48% behind the best
tree library on California housing (MSE ratios 1.388 and 1.475 on seeds
0–1 of the bench_realdata protocol), with the residual tree — a
component the library already has — closing nearly the whole gap when
a user switches it on (1.057 / 1.055), and no policy ever adopting one.
This study froze the obvious candidate policy — the default route runs
with the existing training-only offer rule, `residual_tree="auto"` —
and ran it against four bars fixed in advance. The policy does not
ship: on the current tree the premise has moved, and the policy's
marginal effect on California is negative.

Two findings carry the result. First, the default route no longer
shows the recorded gap: on the current tree its mean ratio over seeds
0–4 is 1.057, because the coordinate-surface step now adopts
`btree(longitude, latitude, maxdepth=3)` and does the geography work
the issue assigned to the residual tree. Second, with the policy on,
the residual tree is offered on all five California seeds and adopted
on all five — the held-out judge credits it with +11% to +18% — yet
the test score worsens on four of the five seeds. The adoption judge
(spatial-cell grouped folds) and the bench's random-row test split
measure different generalization, and on this panel they disagree.

## The frozen rule

The policy under test is exactly "the default route runs with
`residual_tree="auto"`"; no part of the rule changed during the study.
As implemented in `python/mboostrs/autospec.py`, the step sits after
the interaction step of the search, before deselection, and runs when
the family is not Multinomial.

- Detection (training rows only). `leftover_share`: the share of the
  response variance the incumbent fit does not explain. `tree_r2`: the
  in-sample R² of a depth-3 regression tree fitted on the incumbent's
  training residual over the numeric columns.
- Offer (training-only). Offer a residual tree when (b) the modelled
  predictors include a coordinates-like pair (column names matching
  {latitude, longitude}, {lat, lon}, {lat, lng}, {lat, long}, {x, y},
  or {easting, northing}), or (a) pairs were adopted by the interaction
  step and `tree_r2 >= 0.10`.
- Adoption (the held-out incremental value, the statistic #124 asks
  for). The candidate adds `btree(<all numeric/boolean/categorical
  columns>, maxdepth=5, selection=sse)` at step length min(nu, 0.1).
  Adoption is decided on held-out folds — spatial-cell grouped folds
  when coordinates are present, the shared folds otherwise; on a
  single supplied holdout, the per-row paired z audit. Adopt iff the
  relative held-out risk gain is at least 2%, or the gain is positive
  with paired z >= 2.0 (on a single holdout: gain >= 2% and per-row
  z >= 2.0). The step records `held_out_gain`, `held_out_se`, `z`, the
  validation scheme, and the tree's share of the adopted prediction.

Arms: policy OFF passes `residual_tree=False` explicitly (the default
at the study's base); policy ON passes `residual_tree="auto"`.
Everything else in the call is identical.

## Baseline re-derivation

The recorded bench (`docs/book/cases/bench_realdata.json`, mboostrs
0.1.3) shows the issue's gap. On the current tree the bench's own
`fit_mboostrs` default variant reproduces this study's seed-0
policy-off MSE to the last digit (2,339,722,013.78; ratio 1.0098
against the recorded denominator), and the chosen formula carries the
coordinate-surface term. The bars below therefore measure the
residual-tree policy's marginal effect on top of the current default
route, with ratios still taken against the recorded per-seed best tree
library (seed 0 CatBoost 2.317039e9, seed 1 CatBoost 2.488087e9,
seed 2 CatBoost 2.308928e9, seed 3 CatBoost 2.335957e9, seed 4 XGBoost
2.238453e9). The issue's ratios stand as the history that motivated
the study.

## Panels and bars

- California housing: bench_realdata protocol exactly (OpenML 44977,
  the bench's own load/split, seeds 0–4, `autospec(budget=200,
  seed=seed, interactions="report", residual_tree=<arm>)`,
  `rec.fit(train, na_action="impute")`, test MSE).
  Bar 1: policy-on mean ratio at most 1.10, and no seed worse than
  policy-off on that seed.
- Controls: every development-suite dataset whose recorded default
  mean ratio is at most 1.00, excluding California — brazilian_houses
  (0.491), socmob (0.888), Moneyball (0.958), forest_fires (0.970),
  abalone (0.996); same protocol and seeds. Bar 2: no panel's mean
  worsens beyond its paired noise (paired per-seed log-ratios,
  mean/SE > 2 with mean > 0), and at most one panel may have offers
  without improving. Of the five, only forest_fires has a
  coordinates-like pair (columns `X`, `Y`).
- Cancer battery: the shootout harness unchanged (method
  `mboost_auto`, tier small, single supplied holdout,
  `interactions="adopt"`, `residual_tree="auto"` — the harness already
  passes it), seeds 0–2, recorded post-fix draws 0.12443863785108761,
  0.06876267784646255, 0.1604252279147264. Bar 3: exact reproduction;
  the flip under test does not touch this call, so any nonzero
  deviation would mean the change leaked into a pinned path.
- Bar 4: every run records whether the tree was offered and why,
  whether it was adopted, and its held-out incremental value.

One protocol amendment, made during the grid and before the affected
arm ran: forest_fires seed 0 chooses a distributional family
(GaussianLSS) on the current tree, whose prediction is a dict of
parameters; the bench's own scoring call cannot score such a fit. MSE
is a score on the mean, so the runner takes the `mu` component when a
fit predicts a dict, and records the family and the prediction form
in every run file. The amendment is symmetric across arms and panels;
the bars are unchanged.

## Results

### California (bar 1: FAIL)

| seed | off MSE | off ratio | on MSE | on ratio | test change | held-out gain | z | adopted |
|------|---------|-----------|--------|----------|-------------|---------------|---|---------|
| 0 | 2,339,722,013.78 | 1.0098 | 2,469,550,657.68 | 1.0658 | +5.55% | +15.80% | 6.69 | yes |
| 1 | 2,628,664,320.74 | 1.0565 | 2,642,801,122.27 | 1.0622 | +0.54% | +16.96% | 3.43 | yes |
| 2 | 2,462,489,361.35 | 1.0665 | 2,475,063,165.81 | 1.0720 | +0.51% | +18.43% | 5.93 | yes |
| 3 | 2,510,018,737.34 | 1.0745 | 2,443,623,743.40 | 1.0461 | −2.65% | +17.77% | 7.88 | yes |
| 4 | 2,415,539,771.72 | 1.0791 | 2,486,270,083.07 | 1.1107 | +2.93% | +10.58% | 3.12 | yes |

Mean ratio: off 1.0573, on 1.0714. The mean clause passes (1.0714 ≤
1.10); the per-seed clause fails on seeds 0, 1, 2 and 4. Seeds 0 and 4
were re-run after the grid and reproduced their MSEs to the last
digit. The offer fired on the coordinates trigger on every seed; the
detection statistic was small throughout (`tree_r2` 0.009–0.062,
`leftover_share` 0.195–0.248), so trigger (a) played no part.

### Controls (bar 2: PASS)

All 25 paired draws are bit-identical between arms (paired log-ratio
0.00000 on every panel; off means: brazilian_houses 4.1486510e7,
socmob 92.886, Moneyball 562.685, forest_fires 417.472, abalone
4.635). No panel worsens. Offers occur only on forest_fires, on four
of five draws, always on the coordinates trigger, and the spatial-cell
judge rejects the tree every time (relative gains −0.83%, −0.33%,
+0.83%, +0.41% — none reaches the 2% margin), so nothing is adopted
and no score moves. forest_fires is the one panel with offers and no
improvement, which the bar allows (the limit is two). On forest_fires
seed 0 both arms take the GaussianLSS route, which records no offer;
the mu-scored MSE is 813.92178319457 in both arms.

### Cancer battery (bar 3: PASS)

All three draws reproduce the recorded post-fix losses exactly
(0.12443863785108761, 0.06876267784646255, 0.1604252279147264; mean
0.11788). The rule offers the tree on all three draws — via trigger
(a): adopted pairs leave residual structure, with detection `tree_r2`
0.374, 0.102 and 0.391 — and the held-out statistic rejects it on all
three (relative gains −38.7%, −49.6%, −10.5%). The battery's recorded
result is unchanged. No policy ships, so there is no after state to
compare; the before state is the shipped state.

### Attribution (bar 4: PASS)

Every policy-on run (30 realdata draws, 3 cancer draws) records the
offer and its reason, the adoption decision, `held_out_gain`,
`held_out_se`, `z`, the validation scheme, the detection values, the
chosen formula and step count, and the test score. The tables above
are tabulated from those run files.

## Mechanism

Three observations explain the California anatomy.

1. **The tree replaces the surface; it does not add to it.** In every
   policy-off run the measured steps include `coordinate_surface`; in
   every policy-on run they do not — the residual-tree step runs
   earlier in the sequence and the surface step never runs. The
   comparison the policy makes on California is therefore not
   "surface plus tree" against "surface", but "tree" against
   "surface", and the surface is the better answer on four of five
   random-row splits.
2. **The adoption judge measures a different generalization than the
   bench.** With coordinates present, adoption is judged on
   spatial-cell grouped folds: whole grid cells are held out, so the
   judge rewards a term that predicts in unseen regions. The bench
   test split is random rows, where neighbouring rows of a test point
   are usually in training and a smooth surface is hard to beat. The
   judge credited the tree with +10.6% to +18.4% on every seed; the
   random-row test returned −2.65% to +5.55% (mean +1.38% worse).
   Neither judge is wrong about what it measures; they answer
   different questions, and the bar is written on the bench's
   question.
3. **An adopted "residual" tree carries the whole fit.** On every
   California seed the adopted tree's covariance share of the final
   prediction is 1.0 to printed precision: after adoption and the
   re-tuned step count (199–373, against 263–475 policy-off), the
   smooth terms contribute nothing the share can see. The step is
   labelled residual, but as adopted here it is a replacement model,
   which is why its cost or benefit is the size of a model swap rather
   than of a correction term.

## Conclusion and implementation status

Bar 1 fails, bars 2–4 pass. Under the frozen gates the adoption
policy does not ship: `autospec`'s default stays `residual_tree=False`,
and no library code changes. The study ran entirely through explicit
arguments, so there is nothing to revert.

What ships instead is the record and the page correction. The book
page `library_comparison_interactions.ipynb` gains a section fitted in
the notebook on its own splits (the same capped rows, the same three
75/25 splits, predictions held to the training range): the autospec
default route scores 0.7589 (the all-pairs configuration's recorded
0.7587; additive 0.7187), and the same route with `residual_tree=True`
scores 0.7619 — the tree adopted on two of three splits, +0.022 on
repeat 0 and −0.012 on repeat 2 — against the recorded best other
score of 0.8278. The conclusion is rewritten to those executed
numbers, and states that the two benches build the data differently
and are not interchangeable. The page's earlier statement — that the
surviving 0.069 of the R² gap is a property of the pairs-only
configuration — now stands next to the search route's own numbers
rather than a citation of a different bench.

Avenues the study points at, not pursued here: judging the residual
tree on the same random-row folds as the other steps when the surface
step has already claimed the coordinates. The spatial judge evaluates
predictions in regions without training observations (observation 2);
using that judge is a protocol choice. Another option is to run the
surface step before the tree step, so the tree is judged on the residual
structure left by the surface.

## Environment

Tree at b70404db (the study branch's base), reconstructed and
blob-hash verified; the Rust extension is the build from blob-identical
Rust sources. Python: study venv, CPython 3.12, numpy 2.5.4, polars
2.1.0, pandas 3.1.1, with scikit-learn shadowed to 1.9.0 — the version
recorded in the bench's provenance — because 1.9.1's ARFF parser
rejects two of the panels outright; the shadow carries a one-line
patch in its ARFF pandas parser (pandas 3 returns integer categories
for integer-looking nominal values, and the parser's quote-stripping
helper assumed strings; the patch returns the label's string form,
which is what the recorded environment produced). Study infrastructure
only; the library under study is untouched. All six panels' frames
were verified against the recorded bench's invariants (row counts,
column counts, categorical counts, per-seed train/test sizes) before
the protocol froze. The shootout cache is byte-identical to the cache
the recorded cancer battery used. Current-tree off-arm means differ
from the recorded 0.1.3 means on some panels (Moneyball seed 0:
560.760 against a recorded 463.562); the bars are paired on/off
comparisons on the current tree precisely so that drift cannot enter
the verdict.

One further observation, outside this study's scope: the bench
harness (`benches/bench_realdata.py`, `fit_mboostrs`) scores
`np.asarray(model.predict(...), float)` and so cannot score a
distributional (LSS) fit at all — on the current tree that is the
route forest_fires seed 0 takes, so the bench as shipped would stop
there rather than record a score.
