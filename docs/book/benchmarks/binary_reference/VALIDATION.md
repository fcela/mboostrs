# Binary-reference repair validation

Issue #572 extends the categorical repair to numeric and boolean binary
inputs. The parent has package content f290194, imported at source 746ec91;
the measured repair is 4c4a270. Fitting and CV use the unchanged release
extension built from main 056d475. The change completes formula dictionaries,
preserving original by values, constraints and adoption thresholds.

Three public regressions fail on the parent and pass after repair: opposite
0/1 curves, opposite boolean curves, and log-scale zero-side curves. Signed
and strong-heredity guards pass both. There are 32 focused passing checks and
16 final direct/gate checks. The full non-R-parity suite passes **7,057 tests**,
with 15 skipped, 407 deselected and one expected failure, in 546.21 seconds.
Ruff and diff checks pass. No Rust arithmetic, R parity or PDF build changed.

The fresh frozen controls have nine families and six draws each, with 500
training rows, 2,000 test rows and noise standard deviation 0.5 response units.
All 54 jobs in each phase succeed. Every mean after/before family MSE ratio
is at most 1.01; null adoptions do not increase; categorical predictions are
byte-identical; all numeric/boolean reference curves are nonconstant. Numeric,
boolean and logged opposed curves improve by about 73–74%. Signed ratio is
0.98984 and shifted ratio is 0.42236; active, smooth and null predictions are
unchanged. This is parent non-regression, not a competitor or FDR guarantee.

The unchanged encoding-tracks/1 study completes 280 jobs per phase. Integer
and boolean discovery change from +70.84% (SE 2.87 points) to −10.68% (SE 1.82)
against retrospective complete CatBoost. MSE falls 2.07415 → 1.08423, with
paired percent change −47.69% (SE 0.94 points). Their repaired predictions
exactly match categorical predictions on all eight draws. All 40 known fits
and 200 competitor/Ridge predictions remain exact; categorical discovery is
unchanged. Signed discovery retains a small MSE increase of 0.40%, paired SE
0.53 points, unresolved at this precision. It still meets the competitor
target and its separate fresh control family passes.

Eight of ten primary cells meet 1%. Shifted coding retains known-specification
+10.99% (SE 1.33) and discovery +14.12% (SE 2.01), with final optima at the
1,600-update limit. The valid-dictionary convergence/budget gap is #573.
A frozen longer known-path diagnostic exactly replays 16 original prefixes
and reduces shifted MSE 1.34867 → 1.03182; it does not change the original
budget. The independent API ceiling bug above 10,000 is #574, addressed in
its own MR. This dictionary repair does not claim to fix that contract.

Only D11 is graded by the companion executed study; eleven dimensions remain
unmeasured and no aggregate is defined. Finite competitor grids and unequal
search designs limit transfer claims. The wider benchmark goal stays active.
