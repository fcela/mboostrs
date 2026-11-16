# Bounded step-length search for the autospec route (issue #900)

The dlib benchmark record (`2026-11-14-dlib-tuning-benchmark.md`) found
that a black-box search over `(nu, mstop)` at an equal fit budget beat
the shipped autospec route on three of seven frozen panels, and that
all three wins came through the step length `nu` — the one numeric
parameter the route never searches. The route fixed `nu = 0.3` for its
probes and read only `mstop` off the cross-validated risk curve. This
record measures the route with a bounded step-length search added,
on the same panels, folds, and budget, against the same win rule.

## The change

When the route has selected its specification and tuned it at the
incumbent step length, it now re-reads that finished specification at
five further step lengths — 0.1, 0.2, 0.5, 0.7, 1.0, bracketing the
incumbent on both sides (`probe_nu_search` in
`python/mboostrs/_autospec.py`, wired into the final tuning stage of
`autospec`). Each probed value costs one path per fold, read at the
incumbent sweep's grid end with no grid extensions, and is charged to
the run's existing fit accounting like every other probe; candidates
are skipped once the remaining budget cannot pay for another sweep.
A candidate replaces the incumbent only when its cross-validated risk
is lower by more than a relative margin of 1e-4: along a flat risk
curve the step count compensates for the step length almost exactly,
so differences below that margin are the same path read at another
step count, not a better step length. The run's evidence records the
grid, each value's chosen step count and risk, and whether the search
moved the step length (`evidence["nu_search"]`). Nothing else in the
route changes: the specification search, the families, and the
`mstop` read-off are as shipped.

## Protocol

Panels: the same seven frozen shootout panels as the dlib benchmark,
seed 0; data per panel is the shootout payload's fit and validation
partitions concatenated, response `y`:

| panel | kind | development rows |
| --- | --- | --- |
| cancer | binary | 364 |
| bike_iid | regression | 4000 |
| bike_future | regression | 4000 |
| concrete | regression | 659 |
| joint_boundary | regression (synthetic) | 3200 |
| smooth_sparse | regression (synthetic) | 3200 |
| housing_iid | regression | 4000 |

Arms, per panel:

1. The shipped route (fixed step length 0.3) and dlib's best, taken
   from the dlib benchmark's results file: risks, chosen
   configurations, and the per-fold judge risks at those
   configurations. Neither arm is re-run; both are embedded in this
   record's results file.
2. The route with the step-length search: `autospec(data, "y",
   budget=200, seed=0)` on the same shared 5-fold matrix
   (`cv_weights(n, type="kfold", B=5, seed=0)`), in the library as
   changed. Its result is the selected specification at the searched
   `(nu, mstop)` with its cross-validated family risk; its budget is
   the run's own fit accounting, which charges each probed step
   length one path per fold.

Comparisons use the dlib benchmark's paired rule: per-fold risks at
two configurations on the shared folds, differenced fold by fold. The
nu route's per-fold risks were recomputed once with `cvrisk` for this
comparison; those judge fits measure the outcome and were not charged
to any arm.

Bars, fixed in advance:

- Bar A (recovery): on bike_future, concrete and housing_iid — the
  panels dlib won — the searching route improves on the shipped route
  on all three, and its risk is within one paired standard error of
  dlib's best on at least two of the three.
- Bar B (do no harm): on cancer, bike_iid, joint_boundary and
  smooth_sparse the searching route is not worse than the shipped
  route by more than one paired standard error on any panel.
- Bar C (budget): the search stays inside the existing fit
  accounting — every probed step length is charged in the run's fit
  total, and the route's total fits are reported against the fits the
  shipped route spent on the panel and against the 200-fit budget
  both routes were given.

## Environment

mboostrs 0.2.0 built from the tree at main commit
`1f7f37513dfbd41b48612ee8b1f019b51a5d382e` with the step-length search
applied; Python 3.12.3. Harness:
`benches/bench_nu_search_900.py`; aggregation:
`benches/analyze_nu_search_900.py`; full per-panel records, including
each candidate's reading and the per-fold judge risks:
`docs/benchmarks/2026-11-14-nu-search-results.json`.

## Results

| panel | shipped fits | shipped risk (mstop @ nu=0.3) | nu-route fits (search) | nu-route risk (nu, mstop) | dlib risk (nu, mstop) | improvement vs shipped (paired SE) | distance to dlib (paired SE) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cancer | 52 | 0.0907211 (625) | 77 (25) | 0.0890924 (1, 170) | 0.0904787 (0.443, 479) | +0.00162868 (0.00226) | -0.00138636 (0.00228) |
| bike_iid | 136 | 0.353397 (9993) | 161 (25) | 0.353397 (0.3, 9993) | 0.353353 (0.532, 9844) | +0 (0) | +4.36382e-05 (0.000473) |
| bike_future | 106 | 0.283388 (9994) | 131 (25) | 0.282807 (1, 9999) | 0.28281 (0.99, 9915) | +0.000580948 (0.000265) | -3.39755e-06 (1.23e-05) |
| concrete | 78 | 29.9035 (9941) | 103 (25) | 28.9739 (0.5, 9987) | 28.8451 (0.609, 9868) | +0.929546 (0.412) | +0.128807 (0.138) |
| joint_boundary | 36 | 9.35373 (9) | 61 (25) | 9.34848 (1, 1) | 9.36527 (0.163, 57) | +0.00525008 (0.00946) | -0.0167903 (0.0146) |
| smooth_sparse | 45 | 0.250478 (145) | 70 (25) | 0.250478 (0.3, 145) | 0.250489 (0.445, 101) | +0 (0) | -1.07919e-05 (2.26e-05) |
| housing_iid | 132 | 0.0608359 (195) | 152 (20) | 0.0586226 (0.1, 1679) | 0.0590529 (0.258, 475) | +0.00221331 (0.000823) | -0.000430329 (0.000318) |

On every panel the searching route selected the same specification
(same family, same formula) as the shipped route; only the tuning
changed. The search moved the step length on five panels (cancer to
1.0, bike_future to 1.0, concrete to 0.5, joint_boundary to 1.0,
housing_iid to 0.1) and kept the incumbent on bike_iid and
smooth_sparse, where no probed value cleared the adoption margin. On
housing_iid the finished specification's incumbent step length was
0.5 rather than 0.3 — its final probe came from a stage that runs at
0.5 — so the grid bracketed 0.5 and the search cost 20 fits there
instead of 25.

Against dlib's margins from the earlier record — the amount by which
dlib's best beat the shipped route — the searching route recovers:

- bike_future: dlib's margin was +0.000578; the route recovers
  +0.000581, the whole margin. Its risk sits 3.4e-06 below dlib's
  best, a tie at the paired standard error of 1.23e-05.
- concrete: dlib's margin was +1.058; the route recovers +0.930,
  88% of it. The remaining distance to dlib's best is +0.128807,
  inside one paired standard error (0.138).
- housing_iid: dlib's margin was +0.001783; the route recovers
  +0.002213, more than the whole margin. Its risk is below dlib's
  best by 0.000430, more than one paired standard error (0.000318).

Bar A passes: the route improves on the shipped route on all three
panels — on each by more than two paired standard errors — and is
within one paired standard error of dlib's best on all three.

Bar B passes: no held panel is worse. bike_iid and smooth_sparse
return the shipped configuration unchanged; cancer and joint_boundary
improve by amounts inside one paired standard error (+0.00163 against
0.00226; +0.00525 against 0.00946).

Bar C passes: the search spends 20–25
fits per panel that the shipped route did not spend, so the searching
route's totals (61–161 fits) are higher than the shipped route's
per-panel spending (36–136 fits) on every panel. Every probed step
length is charged in those totals, and every total stays inside the
200-fit budget both routes were given. The search uses fits that the
shipped route leaves unused.

## Observations

1. The grid finds the same region dlib's continuous search found,
   from five fixed values: the adopted step lengths (1.0, 0.5, 0.1)
   sit next to dlib's winners (0.990, 0.609, 0.258) on the three
   recovered panels, and the risks land within noise of dlib's best
   on two of them and below it on the third. A coarse grid crossed
   with the path read-off captures what the black-box search bought,
   because the risk is smooth in the step length once the step count
   is re-read at each value.
2. The adoption margin is what keeps the held panels held. On
   bike_iid and smooth_sparse the probed values read within the
   margin of the incumbent — the flat-curve case, where a different
   step length at a compensating step count is the same fit — and
   the route returns the shipped configuration exactly, at the cost
   of the 25 probe fits.
3. The boundary band is unchanged. On bike_iid, bike_future and
   concrete the chosen step count still sits at the end of the
   10,000-step grid (9993–9999); the step length decides those panels
   because the grid end binds both searches, as the dlib record
   observed. The search does not extend the grid, by design: it reads
   each candidate at the incumbent sweep's grid end.

## Verdict

Against the success measure set for this work — recover most of
dlib's margin on bike_future, concrete and housing_iid without giving
back cancer, bike_iid, joint_boundary or smooth_sparse, judged by the
same paired-standard-error rule — the measure is met. The route
recovers 100%, 88% and 124% of dlib's three margins, ends within one
paired standard error of dlib's best on all three panels (ahead of it
on housing_iid), and returns the shipped result or better on all four
held panels. The price is 20–25 charged fits per panel out of the
200-fit budget. No black-box optimizer is needed for this gain: the
fixed step length was the gap, and a five-value grid inside the
existing accounting closes it.

## Reproduction

```
python benches/bench_nu_search_900.py \
    --reference docs/benchmarks/2026-11-14-dlib-tuning-results.json
python benches/analyze_nu_search_900.py  # aggregates the record tables
```

The harness takes the shipped and dlib arms from the dlib benchmark's
results file; the searching route runs live. The shootout payloads
prepare on first use, as in that record.
