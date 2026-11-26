# Distributional prediction shootout, version 2: protocol

Written 2026-11-24 before the version-2 run, after the library changes that
version 1 prompted. Everything not stated here is as in
[PROTOCOL.md](PROTOCOL.md) (version 1) and its
[DEVIATIONS.md](DEVIATIONS.md) analysis rules.

## What changed in mboostrs between the runs

1. Distributional fits divide the response by a scale and map the fitted
   model back exactly (`response_scale="auto"`, the default): the weighted SD
   of the response each fit sees. Counts, proportions and compositions are
   not scaled.
2. Quantile and Laplace fits do the same (the check and absolute losses have
   a gradient that does not grow with the response). `Quantile` starts at the
   `tau`-quantile and, given no control, takes steps of `nu = 0.3`.
3. `tune_mstop` tunes non-cyclic distributional specifications (#4).
4. `mb.fit` models predict an unseen group code as the average group.
5. `autospec`'s bar for adopting a distributional family is read on the
   response divided by its SD, so it does not depend on the units.
6. When the linear spread screen finds no covariate, `autospec` can offer
   GaussianLSS with a joint depth-3 tree over the numeric predictors in its
   scale equation, decided by the held-out log score.
7. `autospec(lss_trees=...)` governs that candidate and a joint tree (depths
   3 and 6) offered to every parameter of an adopted distributional model.
   `True` runs both; `"auto"` (the default) runs each only when the budget
   left pays for it without taking the fits the mean route's stopping search
   needs, which at `mb.fit`'s standard budget (32 fits) means never. A first
   held-out run with the scale candidate always on (2026-11-24) showed the
   cost: on OpenML wave_energy the stopping search reached 400 steps instead
   of 3,200 and the test MSE was 43 times larger.

## Design

- **Same tasks and splits as version 1**: the 48 planted-truth instances and
  the 35 OpenML-CTR23 datasets × seeds 0, 1, 2, with the same development and
  test rows.
- **Raw responses for every arm.** Version 1 divided every response by its
  development SD for every arm, which hid the mboostrs units defect. Version
  2 passes the response as stored.
- **The same competitor arms and settings** as version 1. Their libraries are
  unchanged, so their version-2 scores differ from version 1 only through the
  raw response (hessboost, NGBoost, CatBoost and LightGBM are not
  scale-equivariant; the forests are).
- **mboostrs arms**: the six version-1 arms at the new library defaults, plus

  | Arm | What it fits |
  |---|---|
  | `mboostrs.fit_lss_trees` | `mb.fit(dev, "y", lss_trees=True)` |
  | `mboostrs.LSS_select_noncyclic` | `mboostrs.LSS_select` with `method="noncyclic"` (total step count tuned by `tune_mstop`) |
  | `mboostrs.LSS_select_tree` | `mboostrs.LSS_select` with `btree(<numeric predictors>, maxdepth=6)` added to every parameter |

- Code at commit 81e8b12 plus the uncommitted working tree on 2026-11-24;
  the run records library versions as version 1 did.

## Analysis

The version-1 analysis, run on the version-2 results, plus the two checks
below. The book page and the report present version-2 results only. Each
arm's version-2 CRPS is also paired with its version-1 CRPS on the same task
instances (geometric-mean ratio with a bootstrap 95% interval, better, tied
and worse at a 1% band) and written to `v1_vs_v2.md` for the maintainers.

1. **The `lss_trees` default**: `mboostrs.fit_lss_trees` against
   `mboostrs.fit` on the same tasks. The decision rule, fixed now: the default
   becomes `True` if the geometric-mean CRPS ratio on real data is below 0.99
   with its 95% interval below 1, the planted-task ratio is not above 1.01,
   and the median run time rises by less than a factor of 2, and a held-out
   run of `mb.fit(..., lss_trees=True)` on the study below also passes its
   mean-accuracy rule. Otherwise `"auto"` stays.
2. **Mean accuracy**: the held-out study of `mb.fit`
   (`benches/fit_heldout`, 95 datasets, seeds 6000–6002) with the version-1
   library (commit 81e8b12, built in a separate worktree) and the
   version-2 library, scored by test MSE (regression) and log loss (binary),
   paired per dataset. The version-2 library must not be worse on mean
   accuracy: geometric-mean loss ratio not above 1.01 with its 95% interval
   reaching below 1.01.

## Exposure

The library changes were developed with the version-1 results in view, on
seed-0 data of S1, S3, S7, superconductivity, California housing, kin8nm,
concrete, CPU activity, abalone, Miami housing and video transcoding. The
version-2 run uses the same seeds; its scores are therefore not an
independent test of the changes, and the report says so.
