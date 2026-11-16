# Maximum-grid contract validation

Issue #574 concerns argument propagation, separate from the binary dictionary
repair (#572) and the fixed-budget shifted-code gap (#573). The mean-model
discovery loop now passes its effective maximum to both tune_mstop and the
preflight estimate. No statistical arithmetic or ordinary resource limit is
changed. The measured cap repair is source 8664413, over binary repair 4c4a270;
the native release extension remains the main-056d475 build.

Two regression cases above 10,000 fail on the parent; the 800-step guard
already passes. All three pass after repair, including actual delegated grids,
preflight and final reported stopping count. 31 focused checks pass. The full
non-R-parity suite passes **7,060 tests**, with 15 skipped, 407 deselected and
one expected failure, in 498.60 seconds. Ruff and diff checks pass.

Frozen public API validation repeats eight shifted draws with an explicit
25,600 maximum before and after. The original source overstates its actual
explored grid on all eight: the public tuner receives its default maximum
10,000 while the caller labels the grid 12,800 or 25,600. The repaired calls
forward 25,600 and every actual grid equals the declared one; these curves
turn within an explored grid of 12,800. A maximum is a ceiling, not a demand
to continue after an interior optimum. The selected repaired optima exceed
10,000 on five draws and range from 8,769 to 11,800.

All 80 original encoding-study mboostrs fits at their unchanged 1,600 limit
replay input/latent/prediction digests and MSE exactly. This is an additional
library-only contract regression check. The 200 original competitor/Ridge
predictions remain their frozen measurements, not relabelled new runs. The
full 280-job-per-phase encoding study is still the comparative experiment;
this cap repair does not close its two fixed-budget shifted gaps.

The first observer missed the warning decorator's stack frame and captured
no tuner stages. It is retained in initial_observer and was not treated as a
successful contract audit. The corrected observer walks the wrapper stack and
rejects a high-cap row without observed returns. All 96 numerical captures
(80 low-limit fits plus eight high calls per phase) repeat their original
input/prediction digests and MSE exactly. OBSERVER.md records this correction;
no statistical call, selection rule or data changed, and repeats are not new
independent confirmation draws.

The companion study branch contains bench_discover_cap.py and its frozen data
runner. Import the binary-only parent or cap-repaired checkout via PYTHONPATH,
run the same high-mode calls, then the fixed low-mode replay. The diagnostic
reporter verifies every stage and exact prediction in summary.json. No new
scorecard grade or aggregate is inferred. No R parity or full PDF build was
run for this argument-handling repair.
