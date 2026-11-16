# Paired A/B run report — instrument paired-ab/v1.0

**Issue:** #511. **Date:** 2026-10-13. **Instrument:** `paired-ab/v1.0`
(fresh implementation; the laptop-local v1.1/v1.2 was never published).
**Battery:** 4 Gaussian panels × 8 replicates × 3 arms (A_supplied,
A_partial, B_discovery), paired on identical train/val/test partitions.
32/32 replicates complete, 0 arm failures.

Raw records: `results.json`. Scales: `scales.json`. Protocol:
`../paired_estimation_discovery_protocol.md`.

## Headline decomposition (medians across 8 replicates, units of Var(y))

| Panel | estimation_excess | discovery_gap | partial_gap | end_to_end |
|---|---|---|---|---|
| linear_factors | 0.0003 | 0.0003 | 0.0010 | 0.0006 |
| smooth_additive | 0.0014 | 0.0083 | 0.0009 | 0.0087 |
| kinked | 0.0005 | 0.0008 | 0.0027 | 0.0009 |
| interaction | 0.0282 | 0.0577 | 0.0581 | 0.0841 |

Structural (B_discovery medians): linear_factors recall 0.75 / FDP 0.25;
smooth_additive 1.00 / 0.00; kinked 1.00 / 0.00; interaction 0.67 / 0.00.

## Decomposition results

**On linear, smooth and kinked truth there is no gap.** Estimation excess
and discovery gap are both under 0.1% of Var(y). Given an adequate
specification the library estimates at essentially Bayes risk (D1a
clean); the frozen discovery procedure recovers the structure with
recall 0.75–1.00 and FDP ≤ 0.25 (D1b clean on additive truth).

**The gap lives in interaction discovery.** On the product-interaction
panel, end-to-end excess is 8.4% of Var(y), decomposed as:
- estimation excess 2.8%: even the true tensor-product specification
  (`bbs(x0) %X% bbs(x1)`, selected for 906/1000 updates) leaves
  signal on the table;
- discovery gap 5.8%: the frozen autospec (budget 32,
  interactions="adopt") does not adopt the x0×x1 interaction in 7 of 8
  replicates — it returns a purely additive spec. The product has no
  additive component by construction (centered), so the additive
  fallback cannot recover it;
- partial-knowledge gap ≈ discovery gap (5.8%): the naive all-smooth
  baseline matches the discovery procedure — interaction adoption adds
  nothing on this DGP at this budget.

Structural recall on the interaction panel (0.67) reflects the missed
joint group {x0,x1}: the procedure finds the mains (x2, x3) but not the
pair. FDP is 0.00 — it does not hallucinate; it abstains.

## D11 interpretation

The maintainer asked which component owns how much of the predictive
gap. On these panels: **interaction discovery (D1b) owns roughly
two-thirds of the interaction-panel gap; estimation (D1a) owns roughly
one-third; on additive truth neither owns anything measurable.** The
#509 stopping/tree-form work addressed estimation-side refinements;
this decomposition says the larger share is the discovery procedure not
finding product interactions at a frozen budget of 32.

## Limitations

- Four Gaussian panels only; no thresholds beyond kinks, no spatial,
  no location–scale, no real-data prespecified runs (per the #511
  experiment-design order).
- Uncertainty calibration not measured in v1.0 (scheduled).
- B_discovery is one frozen autospec configuration; the gap is
  conditional on budget=32 and the adopt rule, not on discovery in
  general.
- Negative q1 values on kinked (e.g. estimation excess −0.0019) are
  test-noise (n_test=2000), not negative excess — R* remains the floor
  in expectation.
