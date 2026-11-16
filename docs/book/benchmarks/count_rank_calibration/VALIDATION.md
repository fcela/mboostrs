# Validation

The frozen statistical core is unchanged from0333052. Both independent confirmation gates pass before default promotion; the conjunction is reproduced by benches/decide_count_rank_promotion.py. The original primary cohort is retained and reported separately. Default target scope is the original-rate process; high-rate fixed-mean and zero-inflated count-size limitations remain explicit.

Final integration includes main768519f (CQR response-transform tuning and fold-path cache). The final native digest2ca05527393e6153cce73642f587ed8cb0068173aa1bf4b749701ebc075e4e24 differs from the measured native digest only because an equivalent R seed range check was rewritten for Clippy. SOURCE_MANIFEST.json verifies frozen phase-specific runner and Python source hashes. No statistical core threshold, score, mean fit or confirmation draw was changed.

- Rust workspace tests excluding the R crate:774 core tests,14 integration tests and7 doc tests pass; the PyO3 target has no tests. Strict workspace Clippy passes. Independent SciPy CDF/inverse and likelihood-profile tests pass.
- Integrated Python conformal/count, observer, codec, LSS response-unit and CQR path-cache checks:283 passed. Warnings concern declared extrapolation, calibration choices and capped tuning paths.
- Audit replay:12,236 unique native calls agree exactly, both with the measured and final native build. All1,250 actual fixed competitor intervals reconstruct exactly from their observed calibration/query predictions and scores. Both identities are retained.
- R installed into a private library from the same final native source. Fixed native inputs agree exactly across Python/R, public predictions differ by at most7.95e-9, and reference nuisances by1.54e-8. Separate established contracts are1e-8 for predictions and2e-7 for the flat likelihood-profile nuisances. No statistical threshold is changed.
- Complete phase summaries independently rescore all stored predictions, including infinite outcomes, and validate hashes, method/draw coverage, partitions and fixed means across scores. Unbounded outcomes are not averaged only over finite draws.
- Focused strict MyST HTML execution passes. The public API example agrees with all1,000 frozen original-rate supplied-fixed predictions to1e-12. Tables render as GT HTML and computed prose resolves. Generated Python/R references and native stubs are synchronized.

The initial three strict-JSON writer failures are archived separately with498 completed records and original failure logs. The codec repairs evidence transport only; the public exporter remains tracked by#665. Timing is descriptive elapsed workflow/calibration cost on shared hardware; memory was not measured.
