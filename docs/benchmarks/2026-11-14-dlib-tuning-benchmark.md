# dlib global optimization vs the autospec tuning route (issue #898)

This record compares two ways of tuning the specification the automated
search selects: the shipped route, which reads the optimal `mstop` off
one cross-validated boosting path per fold at a fixed step length
`nu = 0.3`, and dlib's global optimizer (`find_min_global`), which
searches `(nu, mstop)` as a black box. Both arms were given the same
total fit budget per panel. The protocol below was fixed before any run;
the panel list was not changed afterwards.

## Protocol

Panels: the frozen shootout panels the automated-search stopping study
ran on (`benches/stopping_ablation_509.py`'s default task list over the
frozen task table in `benches/shootout_data.py`), seed 0:

| panel | kind | development rows |
| --- | --- | --- |
| cancer | binary | 364 |
| bike_iid | regression | 4000 |
| bike_future | regression | 4000 |
| concrete | regression | 659 |
| joint_boundary | regression (synthetic) | 3200 |
| smooth_sparse | regression (synthetic) | 3200 |
| housing_iid | regression | 4000 |

Data per panel: the shootout payload's fit and validation partitions
concatenated — the rows a search may see. The calibration and test
partitions were never touched. Response is the payload's `y` throughout.

Arm 1, the current route as shipped: `autospec(data, "y", budget=200,
seed=0)` on a shared fold matrix, autospec's own default scheme
(`cv_weights(n, type="kfold", B=5, seed=0)`), passed in explicitly so
both arms use identical folds. The arm's result is the selected
specification at `(nu = 0.3, route-chosen mstop)` with its
cross-validated family risk. The arm's budget B is the run's own fit
accounting (the recommendation's `fits`): the number of fits the route
spent on the panel, specification search included.

Arm 2, dlib: `find_min_global` over `(nu, mstop)` for the specification
arm 1 selected, on the same folds and the same cross-validated family
risk (the library's `tune_mstop` with grid extensions disabled, read at
exactly the requested iteration). One objective evaluation costs one
fold fit per fold — 5 fits, the route's own unit — so the evaluation
cap is `B // 5` and the arm never spends more fits than the route.
Search ranges: `nu` in [0.02, 1.0]; `mstop` in [1, L], where L is the
end of the grid the route's own final probe evaluated for this
specification (200 to 10,000 across panels). `mstop` is declared an
integer variable to the optimizer.

Win rule, fixed in advance: dlib wins a panel when its best risk at the
equal budget beats the route's by more than the paired standard error
of the difference — the per-fold risks at the two chosen configurations
on the shared folds, differenced fold by fold, so fold difficulty
cancels. The per-fold risks were recomputed once per configuration with
`cvrisk` for this comparison; those judge fits measure the outcome and
were not charged to either arm. A Rust reimplementation of the
optimizer is proposed only if dlib wins a majority of the panels;
otherwise the study does not support a Rust reimplementation.

Anytime curves: for arm 2, the best objective after each evaluation
against cumulative fits spent. The shipped route exposes no ordered
probe trace, so for arm 1 the harness wrapped `probe_mstop` during the
run and recorded, after each mstop probe, the cumulative probe fits and
the best chosen-probe risk so far, plus the endpoint (B, route risk).
Probe fits are a subset of B: the route also spends fits on work that
is not an mstop probe.

## Environment

mboostrs 0.2.0 built from the main tree at commit
`1f7f37513dfbd41b48612ee8b1f019b51a5d382e`; Python 3.12.3; dlib 20.0.1
(Boost Software License) installed into the study environment as a
benchmark oracle only — it is not, and must not become, a dependency of
the library. Harness: `benches/bench_dlib_tuning_898.py`; aggregation:
`benches/analyze_dlib_tuning_898.py`; full per-evaluation traces:
`docs/benchmarks/2026-11-14-dlib-tuning-results.json`.

## Results

| panel | budget (fits) | route: mstop @ nu=0.3 | route risk | dlib: (nu, mstop), evals | dlib risk | margin | paired SE | dlib wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| cancer | 52 | 625 | 0.0907211 | (0.443, 479), 10 | 0.0904787 | +0.000242 | 0.00105 | no |
| bike_iid | 136 | 9993 | 0.353397 | (0.532, 9844), 27 | 0.353353 | +0.0000436 | 0.000473 | no |
| bike_future | 106 | 9994 | 0.283388 | (0.990, 9915), 21 | 0.282810 | +0.000578 | 0.000258 | yes |
| concrete | 78 | 9941 | 29.9035 | (0.609, 9868), 15 | 28.8451 | +1.058 | 0.498 | yes |
| joint_boundary | 36 | 9 | 9.35373 | (0.163, 57), 7 | 9.36527 | −0.0115 | 0.00984 | no |
| smooth_sparse | 45 | 145 | 0.250478 | (0.445, 101), 9 | 0.250489 | −0.0000108 | 0.0000226 | no |
| housing_iid | 132 | 195 | 0.0608359 | (0.258, 475), 26 | 0.0590529 | +0.00178 | 0.000914 | yes |

dlib's best risk is lower than the route's on five of seven panels, but
on cancer and bike_iid the margin is smaller than the paired standard
error, so it does not count as a win under the rule. dlib wins three
panels: bike_future, concrete, housing_iid.

Selected specifications, for reference: cancer — Binomial, 13 terms;
bike_iid — Gaussian, 11 terms; bike_future — Gaussian, 5 terms;
concrete — Gaussian, 8 terms; joint_boundary — Gaussian, 12 terms;
smooth_sparse — Gaussian, 3 terms; housing_iid — Gaussian, 6 terms.
Full formulas are in the results JSON.

## Anytime behaviour

The budget itself is the binding constraint on the dlib arm: the routes
spent 36–136 fits per panel, which is 7–27 dlib evaluations at five
folds each. Within that allowance the optimizer localized the optimum
region quickly on most panels. Its first evaluation was already within
0.1% of its final best on cancer and bike_iid, and the remaining
evaluations bought small refinements. Two panels kept improving late:
concrete's best arrived at evaluation 12 of 15 and housing_iid's at
evaluation 13 of 26. bike_future's winning value appeared at evaluation
4 of 21, with a final negligible improvement on the last evaluation. No
evaluation on any panel returned a non-finite risk.

The route's probe trace shows the corresponding shape from the other
side: each probe prices an entire risk curve for five fits, so the
route's chosen-probe risk falls in a few large steps (housing_iid: ten
probes; the final specification's probe, at 45 cumulative probe fits,
reached 0.0608359). The dlib arm spends the same fits one configuration
at a time and refines gradually.

## Observations

1. On bike_iid, bike_future and concrete, the route's chosen mstop
   (9993, 9994, 9941) lies in the boundary band of the 10,000-step cap:
   the cross-validated curve was still falling when the grid ended.
   The dlib arm, given the same grid end, also landed near the cap on
   those panels (9844, 9915, 9868). On those panels both searches are
   grid-bound, and what separates them is the step length.
2. All three dlib wins come through the `nu` dimension, the one knob
   the route never searches: the winning step lengths are 0.990, 0.609
   and 0.258 against the route's fixed 0.3. Where 0.3 is already about
   right (cancer, bike_iid, smooth_sparse), the black-box search only
   matches the path read-off.
3. The largest relative deficit for the dlib arm is
   −0.12% (joint_boundary, where the route's optimum is a very short
   path, mstop 9, and seven evaluations is a coarse search of the
   space). A black-box tuner at an equal fit budget is a viable
   substitute for the path read-off on these panels; it is not, under
   the win rule, a demonstrated improvement on the route as a whole.
4. Wall time favoured the dlib arm on every panel (0–52 s against
   7–196 s for the full route), but the protocol's currency is fits:
   most of the route's wall time is the specification search and long
   fold paths, not the final tuning step. The wall times are recorded
   in the results JSON and were not part of the verdict.

## Verdict

dlib wins 3 of 7 panels, below the prespecified majority threshold.
The study therefore does not support a Rust reimplementation of the
optimizer. At the equal fit budget, the black-box search over `(nu,
mstop)` matched the shipped route on four panels and won on the three
panels where the fixed step length was less suitable.

## Reproduction

```
python benches/bench_dlib_tuning_898.py    # writes per-panel checkpoints
python benches/analyze_dlib_tuning_898.py  # aggregates the record tables
```

Requires dlib in the study environment (`pip install dlib`); the
shootout payloads prepare on first use (concrete fetches OpenML data
id 44959; the other panels use in-repository data, scikit-learn's
bundled breast-cancer data, or synthetic generators).
