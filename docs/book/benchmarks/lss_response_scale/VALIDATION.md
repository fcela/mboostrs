# Response-unit diagnostic validation

The protocol was committed before implementation and measurement. Baseline
numeric source is 0d94682; its recorded HEAD includes only the protocol commit.
Corrected numeric source is 6b6851b. The separate center inverse prerequisite
is MR #640, commit 054a2b2, merged as bd5a358. No core Rust or binding source
changes are part of this fix. The repository's Python 3.12 environment and
the existing matching release extension were used in an isolated worktree.

The baseline and corrected archives retain identical data hashes and all 300
positional row identities. Each has 12 records: three response transforms,
two stopping prefixes, and internal/external calibration. Earlier internal
maximum endpoint/centre error ranges from 1.4818419114 to 2.0365126753 original
response units. Earlier external centering raises a parameter-unpacking error
in both prefixes; those failures remain in the archive. Corrected errors are
at most 8.8817841970e-16 units. No coverage or competitive-efficiency claim is
inferred from this one deterministic dataset.

After incorporating origin/main bd5a358, 303 targeted Python tests passed
(23 routine warnings): all conformal tests, the issue #633 and #639 regressions,
discovery calibration exclusion tests, response/formula transform tests, refit
specification and reuse tests, and book navigation/archive-policy tests.
The new tests cover both LSS frontends, proper-training-only response and
predictor recipes, coefficient/prediction invariance to calibration mutation,
missing predictor imputation, fixed prefixes, omitted-row alignment, missing
raw inputs and non-finite calibration responses. The existing untransformed
conformal numerical oracle tests pass unchanged.

The notebook executed all six code cells using the repository interpreter;
both GT tables have HTML output and accuracy leaders are bold within cells.
A focused MyST `build --site --strict --ci` completed. The full documentation
source checker passes with 181 ToC notebooks and 2285 Rust items. The global
cell checker reports no syntax, dense-cell or uncommented-cell errors, but
fails on one pre-existing printed dataframe in the KDD expansion notebook;
that separate defect is tracked in issue #641. Other book notebooks and the
full PDF were not executed. Ruff and `git diff --check` pass.

Reproduction from the repository root:

```bash
PYTHONPATH=python:. RAYON_NUM_THREADS=1 .venv/bin/python \
  benches/audit_lss_response_scale.py --output /tmp/lss-response-scale.json
```

The archived baseline is evidence of the earlier implementation and must not
be overwritten by that command. The diagnostic computes independent reference
quantiles and order statistics with NumPy/SciPy; production orchestration
continues to use the existing family quantiles, response inverse and Rust
conformal scoring, quantile, widening and clipping APIs.
