# Discovery calibration validation

## Source and confirmation identity

Protocol cbab210 precedes implementation and measurement. The implementation
was committed as 59b2cf1 before fixed confirmation. Baseline Python sources are
immutable spread-MR head 105b1b5. Both phases load the same release extension;
its Rust sources and Cargo.lock match b089d48. Current main adds only the
previously merged dev-profile optimization; release settings are unchanged.
No shared editable installation or root workspace files were changed.

Evidence stores runner-checkout commit and loaded Python source hashes
separately. The top-level commit is the runner's working directory (cbab210 for
baseline, 59b2cf1 for fixed), not an assertion that baseline imported the runner
checkout's package. The loaded_sources paths/hashes identify the actual package:
immutable baseline versus new source. Native build_info is a third identity.
Before publication, every measured script, protocol and loaded-source hash was
compared with its exact file. The native binary hashes match between worktrees.

All 400 baseline and 2,800 fixed/control records complete: eight panels, 50
independent draws per panel, one baseline method and seven fixed/control methods.
Different panels share seed streams and are never pooled for uncertainty.
Every method/draw key is unique. Input digests pair every arm on its intended
panel/draw. Fixed discovery, supplied reference and all five competitors have
identical actual selection/calibration row lists, verified from their records.
Each reserved pair is disjoint and covers every original input row exactly once.

The JSON documents are preserved losslessly as gzip with deterministic headers.
The adjacent SHA256 files identify the uncompressed bytes; decompression equality
was checked before moving working JSON copies outside the checkout. No array,
warning, unfavorable result or failed statistical trial was discarded. Derived
summary JSON is secondary to the raw per-draw records and executable notebook.

## Executable boundary checks

Tests inspect actual row identities entering family inference, draft, mean
screening, interactions, spread, tuning and refit. They mutate only calibration
responses and assert unchanged family/formula/stopping, exact model coefficients,
model predictions after calibration, and refit predictions, while band quantiles
change. This tests fitted objects after calibration, not only a mean column
computed beforehand. Internal versus explicit copies of the same partition give
identical models and bands.

Other cases cover all-data analysis without a requested band, raw row-position
provenance and input namespaces, inadequate/same-object calibration, missing
response/predictor, non-finite observations, budget refusal without a coverage
claim, GaussianLSS central quantiles, no refit, non-cyclic total update prefixes,
step limits, fitted predictor log/imputation recipes and original-scale response
quantiles. A transformed LSS original-scale mean is left unavailable instead of
labelling an inverse location as an expectation. Existing internal-path response
unit inconsistency is tracked separately as #633.

The final targeted suite passes 730 tests (94 warnings):

```bash
PYTHONPATH=python:. .venv/bin/python -m pytest \
  tests/test_discovery_calibration_631.py tests/test_discover*.py \
  tests/test_conformal*.py tests/test_nonmonotone_spread_562.py \
  tests/test_search_constraints.py tests/test_specification_assistants.py \
  tests/test_compare_specifications_lss.py tests/test_search_session.py \
  tests/test_specification_curve.py tests/test_cost_budget.py \
  tests/test_agent_search_fixes.py -q
```

The strengthened fitted-object mutation assertions pass all 12 boundary tests.
There is no new Rust numerical implementation: the split, quantiles, conformity
scores and band widening reuse existing core APIs, so Rust/R parity suites are
not repeated. Full Python Ruff and changed-file whitespace checks pass.

## Development failures and sample-size changes

The first local run called stopping_budget as a property; it is a method. That
instrument/wrapper error was corrected before the implementation freeze and
confirmation, with the initial test log retained. Subsequent fixture failures
showed a deliberate sample-size change: an existing 300-row recovery test with
newdata now selected on 150 rows. Its null/structure assertions did not hold on
that smaller sample. Those tests now supply an additional independent 150-row
calibration frame, preserving the original 300 selection rows and assertions.
They do not establish 300-total-row recovery under the new default. The frozen
small-sample confirmation and its regressions remain unchanged and visible.
No threshold, stopping grid or DGP was altered to recover the fixtures.

## Results and interpretation

New discovery coverage ranges from 0.89496 to 0.90174 across eight panels, with
per-panel Monte Carlo SE. Absolute-scale paired width change is -3.4911%
(SE 0.6349 percentage points), interval score change -0.1621 (SE 0.0179), with
50/50 scale adoptions. This is a whole-procedure change: baseline selection
used all data and its later NumPy split, while the new rule uses the reserved
Rust partition. It does not isolate reservation from partition/data-use effects.

The 80-row panel leaves 40 selection and 40 calibration rows. Mean-term
selection falls from 37/50 to 16/50. Paired point-MSE change is +0.2221
(SE 0.0382 squared response units), width change +5.5604% (SE 2.3441 points),
and coverage changes from 0.89170 to 0.89496. This cost of reservation is recorded
in #415. Analysis-only calls still retain all input rows, and independent
external calibration permits using all data rows for selection. No silent
all-data fit replaces the selection-only model that supplies the bands.

Every new discovery point width gap is <=1% to its retrospective eligible
competitor on this battery; this is not a population upper bound or a universal
best-in-class claim. Known references retain +1.0231% (SE 0.5476) absolute-scale
width in #630 and +4.4291% (SE 1.5158) Poisson width in #636. The latter has
better point MSE and interval score than LightGBM CQR. Untuned competitors,
fixed reference stopping and the count-support interpretation of width remain
limitations; no model was retuned from these confirmation outcomes.

Mean paired width ratios can differ in direction from ratios of mean widths:
sharp-panel paired change is +1.001% (SE 3.992), while its mean width decreases.
Both summaries are retained; the protocol's paired estimand is not changed to
choose the more favorable one. Selection/scale attribution remains separate
from band coverage (#629/#464).

## Notebook and book

The executed Part V notebook has 14 code cells (maximum 21 lines), nine GT HTML
tables and no error outputs. It displays every method in both information
tracks, common row counts, independent-draw uncertainty, conditional coverage,
selection incidence, costs and failures. Eight retrospective competitor width
leaders are marked with bold HTML that survives the book's table conversion.
The live workflow uses public APIs; a separate CatBoost CQR construction shows
its actual fit/calibration calls and correction without adding a confirmation
row. All computed prose resolves from earlier cells.

A focused strict MyST execution/rendering check uses the repository book theme;
the other book notebooks are not re-executed. Integrated source, navigation and
archive checks validate the new ToC page and every declared evidence asset.
The empirical standard-process evidence supports D9 score 3 in the new known
and discovery tracks. Conditional/grouped/temporal coverage, selection-adjusted
effect inference, extrapolation adaptation and other scorecard dimensions are
not established by this study. No aggregate score or scorecard file is changed.

Final rendering result: strict focused build exits successfully with nine HTML
outputs (including referenced table assets) and nine inline computed results,
all status ok. The rendered prose uses the intended rounded values without
Python representation wrappers. Complete book source validation covers 180 ToC
notebooks and 2,285 Rust reference items. All 23 navigation/archive/surface tests
pass; together with the 730 targeted statistical/orchestration cases, 753 unique
tests pass. The new source and all declared gzip/log/protocol assets are bundled
in the downloadable notebook archive. No other 179 book notebooks were executed
by the focused rendering check.
