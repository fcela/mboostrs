# Nonmonotone spread validation

## Source and measurement identity

The nomination protocol was committed in ec42756 before implementation or
measurement. The nomination fix and scoring addendum were committed in 008002f
before the final full rerun. Baseline sources are unchanged origin/main 19e2537
in an isolated worktree. Fixed Python sources are this branch. Both measured versions load the
same release extension; their Rust/Cargo sources match its b089d48 build. No native
source or shared editable installation was changed.

Final evidence contains 700 baseline/control and 100 proposed-screen records:
five processes, 20 independent training/test draws per process, seven baseline
methods and one proposed discovery method. Every final record completed.
All 800 input identities match their intended pair. Script, protocol, scoring
addendum and loaded-source hashes match the measured files. Prediction digests
match all corresponding successful initial records, including every one of the
100 proposed discovery results. This confirms that the scoring repair did not
change those fits or predictions.

The initial 700 baseline records retain 88 scorer rejections of crossed CQR
bounds. Initial discovery/proposed records and all six reproduction calls also
remain available. Separate snapshots preserve the exact original runner bytes.
The scoring addendum defines zero-length empty sets and their proper two-quantile
score; the complete final rerun includes every competitor, not only rejected
rows. The initial and final phases reuse draws and must not be pooled.

## Tests and notebook

The targeted suite covers the public slope/curve diagnostic, nonmonotone scale
adoption, unchanged two-SE adoption margin, sigma candidate restrictions, null
control and zero-budget behavior, existing discovery/interaction/mean recovery,
missing-data paths, search constraints, specification assistants and LSS
comparison. The final result is 441 passing tests (23 expected warnings):

```bash
PYTHONPATH=python:. .venv/bin/python -m pytest \
  tests/test_nonmonotone_spread_562.py tests/test_discover*.py \
  tests/test_search_constraints.py tests/test_specification_assistants.py \
  tests/test_compare_specifications_lss.py -q
```

The first broad run had 440 passes and one failure: the existing mean-recovery
test assumed the selected model always returned a single vector. The new scale
nomination can select a location-scale model, whose public response prediction
is a mapping. Extracting its mu prediction retains the original test's MSE <0.7
criterion; the measured value is 0.515024. The statistical ambiguity exposed by
that model selection is filed in #629, not hidden by the test repair. The original
failure log and reproducible baseline/fixed counterexample probes are retained.
The probe is development evidence from an existing test, not a fresh confirmation
batch.

The notebook executes nine code cells (at most 25 lines each), seven GT HTML
tables and no error outputs. A first attempt loaded the shared checkout through
a relative PYTHONPATH; its failure log is retained. Successful execution uses
the isolated branch through an absolute path. Table cells mark the retrospective
eligible competitor widths in bold. Both information tracks, all competitor
outcomes, conditional coverage, uncertainty, workflow costs, remaining gaps and
measurement failures are visible.

A focused strict MyST HTML build executes this page with the repository's local
book theme. Its generated page contains seven HTML outputs; the externally
stored headline table asset exists, and all ten inline computed results resolve
with status ok. This focused rendering check does not execute the other 178 book
pages. The full book source checker and navigation/archive surface tests are
run on the integrated book separately. No Altair chart is introduced.

No Rust numerical implementation changes, so Rust/R parity tests and the entire
statistical suite are not rerun for this Python orchestration change. Complete
Python Ruff and changed-file whitespace checks pass.

## Interpretation

Nomination is curve-sensitive; adoption still uses the existing held-out joint
GaussianLSS score. The 20 fresh absolute-scale draws select the true driver and
reduce width by 4.6299% (paired MC SE 0.8888 percentage points), with coverage
0.899075. The bell-scale reduction is 8.1842% (SE 1.8783). Smooth-mean Gaussian
constant-scale and heavy-tail controls have zero scale adoptions in 20 draws
each. These controls do not establish a population false-selection rate or
cover the sharp-mean counterexample #629.

The absolute-scale gap to retrospective eligible LightGBM CQR is 0.0157%
(SE 1.1652 points). A point estimate below 1% is not an upper confidence bound.
Monotone and bell gaps remain 1.8182% (SE 0.9455) and 3.5342% (SE 1.3508), filed
in #630. The known-specification counterparts also remain visible. No comparator,
process or unfavorable draw is dropped. Competitors are untuned and data use
differs by workflow. This study supports D9 score 3 within its standard-process
scope; small-n/extrapolation adaptation and other scorecard dimensions remain
unmeasured. No claim closes the broader #555 acceptance requirements.

The later source audit in BOUNDARY_AUDIT.md identifies reuse of future internal
calibration responses during discovery selection (#631). The original frozen
protocol claim is preserved and explicitly corrected. Test rows remain untouched;
empirical calibration is not a whole-procedure finite-sample guarantee. This MR
repairs the nomination mechanism but leaves #562 open for its calibration-exclusion
acceptance requirement, as well as #629, #630 and the broader #555 target.

Before publication, main 9bb7e6a was integrated. Its #627 change adds only
profile.dev opt-level=1; release settings, Rust sources and Cargo.lock are
unchanged from the measured extension. #628 adds the separate Binomial audit.
The measured discover/explore source hashes still match the integrated branch.
An additional exact scoring test verifies zero width and no coverage for an
empty set while preserving the proper two-quantile score for crossed bounds.
The integrated validation runs 465 tests: 441 discovery/specification tests,
that scoring test and 23 book navigation/archive/surface tests.
