# A bounded step-length search for the autospec route

## Study objective

The autospec route, the library's automatic formula search, fixes the step length `nu` (the fraction of each selected term's fit added per step) at 0.3 for its probes. It reads only `mstop` (the number of boosting steps) off the cross-validated risk curve, which is held-out risk at each step count, estimated over resampled folds. The dlib tuning benchmark used dlib, a C++ optimization library, as a black-box search over `(nu, mstop)`, meaning a search that uses only the objective's values. That search beat the autospec route on three of seven panels, and all three wins came through `nu`, which the route never searches. This study tests a bounded search over `nu`: the route re-reads its finished specification at five further step lengths (0.1, 0.2, 0.5, 0.7, 1.0) inside the same fit accounting, where each probed step length costs one path per fold, charged to the run's fit total. The question is whether this grid recovers the gains from the black-box search without worsening the four panels the route already holds.

## Protocol in brief

The same seven frozen shootout panels, seed 0, the same shared
5-fold matrix, and the same 200-fit autospec budget as the dlib
benchmark. The autospec route's and dlib's results are taken from that
benchmark's record and are not recomputed here. Each probed step length costs one
path per fold, charged to the run's fit total; a candidate replaces the incumbent (the current specification) only when its cross-validated risk is lower by more
than a relative margin of 1e-4, so a flat-curve tie at a compensating
step count changes nothing. Comparisons use the paired standard
error of the per-fold risk difference on the shared folds. The full protocol and per-candidate readings are in
`docs/benchmarks/2026-11-14-nu-search-benchmark.md`.

## Results

| panel | autospec fits | autospec risk (mstop @ nu=0.3) | nu-route fits (search) | nu-route risk (nu, mstop) | dlib risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| cancer | 52 | 0.0907211 (625) | 77 (25) | 0.0890924 (1, 170) | 0.0904787 |
| bike_iid | 136 | 0.353397 (9993) | 161 (25) | 0.353397 (0.3, 9993) | 0.353353 |
| bike_future | 106 | 0.283388 (9994) | 131 (25) | 0.282807 (1, 9999) | 0.28281 |
| concrete | 78 | 29.9035 (9941) | 103 (25) | 28.9739 (0.5, 9987) | 28.8451 |
| joint_boundary | 36 | 9.35373 (9) | 61 (25) | 9.34848 (1, 1) | 9.36527 |
| smooth_sparse | 45 | 0.250478 (145) | 70 (25) | 0.250478 (0.3, 145) | 0.250489 |
| housing_iid | 132 | 0.0608359 (195) | 152 (20) | 0.0586226 (0.1, 1679) | 0.0590529 |

The selected specification is unchanged on every panel; only the
tuning moved. On the three panels dlib won, the searching route
recovers 100% of dlib's margin on bike_future, 88% on concrete, and
124% on housing_iid, where it ends below dlib's best by more than one
paired standard error. On the four panels the route already held, it
returns the autospec configuration unchanged on two (bike_iid,
smooth_sparse, where no probed value cleared the adoption margin, the relative margin of 1e-4)
and improves within noise on the other two.

## Reading

The grid selects step lengths close to those that the continuous search found: the adopted step
lengths (1.0, 0.5, 0.1) sit next to dlib's winners (0.990, 0.609,
0.258), because once the step count is re-read at each step length
the risk is smooth in `nu` and five values are enough to find the
region. The cost is visible in the fits columns: the search spends
20–25 charged fits per panel that the autospec route did not spend,
so its totals (61–161) exceed the autospec route's per-panel spending
(36–136) while staying inside the 200-fit budget both routes were
given. On the panels where 0.3 performed well, the probe fits did not find a
step length that improved cross-validated risk enough to replace it.

## Verdict

The grid recovers at least 88% of dlib's margin on each of the three
panels it won, and it does not worsen the four panels the route already
holds. The comparison supports a five-value step-length search within
the existing fit budget; it does not require a black-box optimizer.
