# Encoding-study validation

The frozen instrument is encoding-tracks/1. Five encodings reuse the same eight
latent draws; they are not forty independent samples. Each has separate known-
specification and discovery tracks, with four tuned tree competitors and Ridge.
Both primary phases contain every planned outcome: 5 × 8 × 7 = 280 successful
jobs each. Input digests match across methods and phases, and latent digests
match across encoding views. The runner and protocol remain unchanged.

The parent imports source 746ec91 (categorical repair content f290194). The
binary repair is source 4c4a270. Exact source/package/native identities and
all dependency versions are retained separately from the runner revision.
The release native extension is the matching main-056d475 build; no Rust
arithmetic changes in either proposed repair.

## Primary comparison

Integer 0/1 and boolean discovery move from +70.84% (paired SE 2.87 points)
to −10.68% (SE 1.82) against retrospectively selected complete CatBoost. MSE
falls 2.07415 → 1.08423; paired percent change −47.69% (SE 0.94 points).
The two repaired views exactly match categorical predictions on all eight
draws. All 40 known fits and 200 competitor/Ridge predictions replay exactly;
categorical discovery is unchanged.

Signed discovery retains a small regression: +0.40% MSE, paired SE 0.53 points.
It is unresolved at this Monte Carlo precision and still within the competitor
target. Every per-seed change remains in the raw/report artifacts. Fresh signed
controls have mean after/before ratio 0.98984; no cross-battery pooling is used.

Eight of ten cells meet the 1% point-gap target. Shifted coding remains +10.99%
(SE 1.33) in known specification and +14.12% (SE 2.01) in discovery. All shifted
optima are at the original 1,600 limit. Those valid-dictionary gaps are issue
#573 and are not removed or relabelled as solved by the binary repair #572.

## Repair and diagnostic checks

Fresh binary-reference controls: nine families, six new draws each, all 108
phase/draw jobs successful. Every family mean ratio is at most 1.01, null
adoptions do not increase, categorical predictions are exact, and numeric/
boolean zero-side curves become nonconstant. This is a parent non-regression
gate, not a calibrated FDR or best-competitor claim.

The separate longer known-path diagnostic keeps formula/data/folds/step/score
fixed, verifies all 16 original prefixes exactly, and raises only its search
ceiling to 25,600. Shifted known MSE changes 1.34867 → 1.03182, gap +10.99%
→ −15.05%; indicator results remain exact. Complete risk paths and all selected
iterations are retained. The figure shows each of eight paired CV paths per
encoding, not an aggregate confidence band. PNG and SVG exports are available.
This is known-data diagnosis, not a new default or a change to the primary
budget. Other eleven dimensions remain unmeasured in the D11 profiles.

The independent maximum-grid bug #574 omitted the caller's cap from tuning and
preflight. Frozen high-cap public calls expose a declared/actual mismatch on
all eight parent draws; the fixed calls forward 25,600 and each actual grid
matches its declaration. These curves turn at an explored grid of 12,800.
All 80 original low-limit mboostrs fits replay their exact hashes and MSE on
the cap-repaired package. The 200 frozen peer fits are not labelled new runs.

The initial observer missed decorator frames; it is retained separately and
was not counted as a contract-audit pass. Its corrected capture repeats all
96 numerical outcomes exactly without changing statistical settings. Every
high-mode record now requires actual observed tuner returns. OBSERVER.md
records the correction; these repeats are not independent confirmation draws.

## Tests, source and limits

Three new binary public regressions fail on the parent; signed and heredity
guards pass both. The binary repair has 32 focused checks, 16 final direct/gate
checks and a full non-R-parity suite of 7,057 passed, 15 skipped, 407 deselected
and one expected failure (546.21 seconds). The separate cap repair has three
boundary guards (two fail on the parent), 31 focused checks, 14 integration
checks and a full suite of 7,060 passed with the same skips/deselections/xfail
(498.60 seconds). Ruff and diff checks pass on affected source/test/report files.
The new study/diagnostic contract tests total 20, covering latent pairing,
missing/non-finite outcomes, incomplete peers, all eight SE observations,
changed protocols/predictions, prefix minima and actual cap traces.

The notebook is executed with the repository environment, retains its HTML
GT outputs over pandas frames and is registered in Part V. Its executed
sources, local links, rendered tables, canonical profiles, protocol/source
hashes and exact replay counts are independently checked. Only D11 is scored;
no aggregate or claim that the broader benchmark goal is complete.

No R parity or full PDF build is claimed for these Python orchestration and
argument-handling changes. Global book checking retains the same pre-existing
ordinary/orphan source defects outside this study, and its cell checker has
one existing printed-frame violation in the older KDD expansion case. Those
checks must not be represented as a clean full-book build. The new study's
own source/runtime checks and canonical scorecard consistency are reported
separately from those inherited defects.

Final scoped checks: 37 study/contract/navigation checks pass. Seven HTML
GT tables render with bold headline minima, and the scientific CV plot has
legible PNG/SVG exports. All experiment source cells remain unchanged during
execution and all local links resolve. The new study adds no global source
error: the inherited checker still reports the same 1,000 older defects.
Canonical scorecard consistency and the provenance ratchet pass; the latter
retains its existing 84 missing fields across 47 older case artifacts, with
no regression. The global cell checker has zero syntax errors and still
reports one existing printed-frame violation outside this study.
