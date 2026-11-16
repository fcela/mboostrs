# Poisson calibration study validation

## Sources and information boundaries

Protocol f49cfa5 precedes implementation and measurement. The shared Rust
score implementation is afa9d21; the runner and reporting freeze is 9d79736.
All three phases used the same loaded Python release binary and the same
Python numerical sources. The separately built R release library has the
same Rust source digest. Binary hashes, build profiles and identities are
retained in NATIVE_IDENTITY.json and the phase metadata. CPU-specific flags
were local build environment settings, with no CI configuration change.

Audit, development and confirmation retain 350, 660 and 1650 records, all
successful. Each confirmation rate has 50 independent development/test
draws. Rates share streams, so uncertainty is computed within rate only.
The validation counter was labelled as panel draws after measurement to
avoid calling 150 rate/seed combinations independent replicates; no metric,
uncertainty calculation, score or setting changed.

Every within-draw input digest and actual 500/500 positional partition
matches across methods. Across the three scores within each information
track, every fitted mean vector and point MSE is identical. Calibration
does not refit or tune those models. All 2660 records were independently
rescored from retained endpoint/point arrays and regenerated response draws;
every recorded input digest and metric matches exactly. The original audit
also reproduces all 350 preceding prediction digests and raw metrics exactly.

The archive files are lossless gzip JSONL; metadata stores both compressed
and uncompressed SHA-256 digests. No failed method, warning, poor result or
confirmation draw was removed. Progress files are transient status, not
evidence. Source/native/input identities distinguish new score calculations
from the unchanged fitting algorithms.

## Outcomes

The earlier real-width excess remains +4.429% (paired MC SE1.516 percentage
points). Under the common count-set metric, the same supplied intervals have
-5.351% excess (SE.447) relative to the eligible count-size leader. Different
leaders can be selected for the two different estimands. The audit is
retrospective and does not establish a prospective improvement.

On the fresh original-rate confirmation, supplied/discovery normalized-score
count-size excesses are -6.232% (SE.512) and -3.932% (SE.911), with coverage
.89764/.90064. Real-width excesses are -3.739% (SE1.252) and -3.673% (SE1.525).
The normalized score leaves point MSE exactly unchanged. Supplied normalized
count-size excess is -7.675% at the low rate; the low-rate discovery variants
have the same covered count sets and -0.756% count-size excess. A point
estimate, MC SE and retrospectively chosen leader do not establish a
universal one-percent bound.

High-rate discovery retains +25.005% count-size excess (SE3.377) with the
normalized score; supplied normalized excess is +4.899% (SE2.663). All 50
discovery fits choose Poisson and both smooth terms, but 25 select one
update. This is a separate mean-fit/stopping investigation in issue #649;
the study does not isolate the cause. Central-quantile scores are overly
conservative at low/original rates (supplied coverage .97732/.97062) and
retain +30.039%/+21.411% count-size excess. Issue #650 tracks tie efficiency.
Neither extension changes the absolute-score default or relaxes eligibility.

## Validation and follow-ups

The workspace test command excluding the R crate passed 770 core unit tests,
14 Rust integration tests and seven doc tests. Workspace Clippy with warnings
denied and cargo formatting pass. Three R numerical-reference and public
integration scripts pass, including the existing conformal-core and fixed
prefix suites. Both the Python SciPy and R qpois references agree with the
new Rust count scores and bands, including zero means and ties.

After incorporating the separately published predictor-coding prerequisite
MR #647, 342 targeted Python tests pass. They include both standard frontends,
internal/external partition equivalence, learned predictor scaling, fixed
prefixes, calibration-mutation invariance, invalid responses/means, support
boundaries and empty sets, and existing conformal/refit/response-transform
oracles. One initial binding test expected the wrong exception class for a
quantile solver-range refusal; its expectation was corrected to the existing
RuntimeError mapping without altering the production error. Compile-time
error-constructor mistakes were corrected before the implementation freeze.

The #647 split prediction fix is Python orchestration over existing core
operations and has its own issue #646 and MR. Its ten new regression cases
all fail on the preceding source, and its own 202-test suite passes. The
frozen #636 score study uses supplied external calibration; its sources and
predictions were not changed by the subsequent integration. CV+ and CQR
still have the related coding problem, separately tracked in #648.

The final notebook executes 13 code cells and retains ten GT HTML tables,
including conditional-coverage groups, paired gaps and stage costs. Its
public-API example predicts independent future rows. Bold leaders are
preserved within table cells. A focused strict MyST site build passes, as
do global source/cell checks (182 ToC notebooks, 2295 Rust items) and the
286-name generated API reference check. No other book notebooks or full
PDF were executed. The new score options require the separately tracked
split-recipe correction in MR #647; the score-study MR is stacked on it.

Reproduce into a new output path rather than overwriting preserved evidence:

```bash
PYTHONPATH=python:. RAYON_NUM_THREADS=1 .venv/bin/python \
  benches/study_poisson_conformal.py --phase confirmation \
  --output /tmp/poisson-confirmation.jsonl.gz
```

The runner refuses an existing output or incomplete journal. The notebook
reads archived results, verifies checksums/partitions and demonstrates the
public score API; it does not rerun competitor fitting or choose settings.
No full-book PDF execution, universal competitor bound, conditional coverage
claim or coverage result for overdispersed/zero-inflated data is asserted.
