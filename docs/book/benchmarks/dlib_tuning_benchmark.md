# dlib global optimization vs the autospec tuning route

## Study objective

Issue #898: the automated search tunes exactly one numeric parameter
by measurement — `mstop`, read off the cross-validated risk curve of
one boosting path per fold — and never searches the step length
`nu`, which is fixed at 0.3. dlib's global optimizer
(`find_min_global`) is parameter-free and treats its budget as a
maximum number of function calls, which makes it a natural oracle for
the question: at the same total fit count, does a black-box search
over `(nu, mstop)` beat the path read-off? The benchmark runs the
existing Python binding as an oracle only (dlib is Boost Software
License; it is not a dependency). A Rust reimplementation was to be
proposed only if dlib won a majority of the panels under the win rule
below.

## Protocol in brief

Seven frozen shootout panels (the stopping study's default list),
seed 0; data is each payload's fit + validation partitions, response
`y`. Arm 1 is the shipped route: `autospec` at budget 200 on a shared
5-fold matrix; its result is the selected specification at
`(0.3, route mstop)`, and its budget B is the fits the run spent. Arm 2
is dlib over `(nu, mstop)` on that specification, same folds, same
cross-validated risk, one evaluation costing 5 fits (one per fold), so
at most `B // 5` evaluations; `nu` in [0.02, 1.0], `mstop` up to the
end of the grid the route's own probe evaluated. dlib wins a panel
when its best risk beats the route's by more than the paired standard
error of the per-fold risk difference at the two chosen
configurations. The full protocol, the anytime curves and the
per-evaluation traces are in the dated record,
`docs/benchmarks/2026-11-14-dlib-tuning-benchmark.md`.

## Results

| panel | budget (fits) | route risk (mstop) | dlib risk (nu, mstop) | dlib wins |
| --- | ---: | ---: | ---: | :---: |
| cancer | 52 | 0.0907211 (625) | 0.0904787 (0.443, 479) | no |
| bike_iid | 136 | 0.353397 (9993) | 0.353353 (0.532, 9844) | no |
| bike_future | 106 | 0.283388 (9994) | 0.282810 (0.990, 9915) | yes |
| concrete | 78 | 29.9035 (9941) | 28.8451 (0.609, 9868) | yes |
| joint_boundary | 36 | 9.35373 (9) | 9.36527 (0.163, 57) | no |
| smooth_sparse | 45 | 0.250478 (145) | 0.250489 (0.445, 101) | no |
| housing_iid | 132 | 0.0608359 (195) | 0.0590529 (0.258, 475) | yes |

dlib's best risk is lower on five panels, but on cancer and bike_iid
the margin is within the paired standard error. It wins three.

## Reading

The budget is the binding constraint on the black-box arm: 36–136
route fits are 7–27 evaluations. Inside that allowance the optimizer
localized the optimum region quickly — its first evaluation was
already within 0.1% of its final best on two panels — and it was
never far behind anywhere (worst relative margin −0.12%). All three
wins come through the step length the route never searches (winning
`nu` 0.990, 0.609, 0.258 against the fixed 0.3); where 0.3 is already
about right, the black box only matches the read-off. On the three
panels where the route's chosen `mstop` sits in the boundary band of
the 10,000-step cap, dlib landed near the cap too: both searches are
grid-bound there, and `nu` decides.

## Verdict

Three of seven is not a majority, so the pre-specified rule does not
propose a Rust reimplementation. The comparison shows that a
parameter-free black-box search at an equal fit budget can match the
path read-off; its three wins occur when it changes the step length,
which the route keeps fixed.
