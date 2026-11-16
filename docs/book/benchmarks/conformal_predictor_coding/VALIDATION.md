# Predictor-coordinate validation

## Frozen diagnostic

Protocol9caae89 precedes implementation and measurement. Baseline numerical
source is4d1f883; corrected numerical source is46d14ea. The helper constructs
independent references through public model fits/prediction and NumPy order
statistics, without calling private conformal prediction/numerical helpers.
Production continues to use the existing Rust numerical APIs.

Both phases retain72 deterministic cases:64 fixed-prefix cases across two
frontends, two families, two methods, four predictor recipes and two query
modes, plus eight Gaussian tuned cases. These are shared-fixture checks,
not independent Monte Carlo draws or a coverage experiment.

All baseline calls returned intervals, but39 cases exceed the prescribed
1e-10 endpoint/point tolerance. Largest discrepancies are .0515653240 for
CV+ and .1260473523 for CQR. All72 corrected cases match the independent
references; maximum absolute error is6.5086824819e-15 response units.
Input digests and complete reference arrays, partitions, stops and inner
risk curves are identical across phases. The actually used inner CV mode
changes from fixed to refit.

Archives are lossless gzip JSON. archives.json records compressed and
uncompressed hashes; decompressed bytes equal the original measured files.
NATIVE_IDENTITY.json records the loaded release library hash and build
identity; its Rust source digest equals the source at measurement. The same freshly
built Python3.12 extension was used in both phases. No statistical Rust or
binding source changed. No shared editable installation was replaced.

## Stopping labels

One GAM/missing-predictor lower-quantile curve selects1556 in production and
76 in the independent direct refit. The curves differ by at most
5.5511151231e-17; the final interval discrepancy is6.5086824819e-15.
Both reported stops equal their own exact positive-grid argmins. Validation
retains both stops and all curve values, checks identical rows/fold weights,
and checks numerical risk agreement within1e-12. It does not require
identical integer labels for roundoff-equivalent flat curves or change the
stopping rule after observing them. This sensitivity is tracked in #653.

## Integration checks

Tests inspect actual outer/inner raw row identifiers, disjoint held-out rows,
proper-training case-weight alignment, query chunking, a241-row/four-fold
remainder, a requested80-step prefix beyond a60-step model, categorical
labels, missing predictor imputation and log query values below the positive
training minimum. CQR calibration predictor/response mutations leave the
quantile fits, coefficients, chosen stops and query predictions identical;
the correction changes. Missing/misaligned raw inputs refuse clearly.

The initial21-case regression module has16 failures on the baseline and
five passing controls. The broader test run exposed the existing one-cap-
warning-per-call contract when both quantiles reached the search cap under
refitted CV. A context-local notice set preserves that contract and respects
warnings-as-errors. This post-freeze warning-only change does not alter the
diagnostic's numerical path; none of its72 cases reaches the cap. The
ordinary numerical oracle, CV+ guarantee/remainder and warning tests remain
in the validation suite. A log-boundary test fixture initially placed zeros
in training, outside log's domain; it was corrected to positive training
values with below-minimum query inputs, preserving the domain check.

Response centering/scaling has a separate standard-model coordinate issue,
tracked in #652. The predictor-only repair does not define new response
units or fix that case. Refitted inner CV retains its explicit refusal for
response transforms whose scales cannot be aggregated. The previous
Poisson competitor evidence and protocols were not changed.

After incorporating externally merged main a0615b6 (including the separate
split correction and Poisson score study),381 targeted Python tests pass
with routine warnings. This includes all conformal tests, the new38-case
boundary suite, response/formula transforms, refit specification/reuse,
previous calibration-boundary and response-unit regressions, and book
navigation/archive checks. A matching release extension for integrated
Rust sources was loaded for that run; its separate identity is retained
in INTEGRATION_NATIVE_IDENTITY.json. Frozen measurement identities and
arrays remain unchanged. The public API reference is current; Ruff,
formatting and global source/cell checks pass (183 ToC notebooks,
2295 Rust items). No full Python suite, R parity suite or Rust test rerun
is claimed for this Python-only correction.

The final integrated notebook executes six code cells and retains five
annotated GT HTML tables, including the differently selected stop labels.
Bold tolerance results are preserved within cells. A focused strict MyST
site build passes. The other book notebooks and the full PDF were not
executed. Both measured archives and all reference values remain intact.

Reproduce into a new raw output path:

```bash
PYTHONPATH=python:. RAYON_NUM_THREADS=1 .venv/bin/python \
  benches/audit_conformal_predictor_coding.py --phase corrected \
  --output /tmp/conformal-predictor-coding.json
```

The runner refuses existing outputs. The notebook reads retained evidence,
verifies archives and references, and demonstrates the public API; it does
not rerun a competitor comparison or infer coverage from these cases.
