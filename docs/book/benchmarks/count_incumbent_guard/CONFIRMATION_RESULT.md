# Confirmation result: production candidate (#679)

Reserved cohorts: count seeds 6799100-6799149 and Gaussian seeds 6799300-6799349,
independent test seed + 100000. Recorder: `benches/study_interaction_production.py
--phase confirmation`, run at commit 928aea8. Output: `production_confirmation_*.jsonl.xz`
(7000 records; 14 failed, all `supplied_fixed` on the two interaction panels,
where the fixed-step Poisson fit cannot handle responses in the hundreds).

## Gates

| Gate (PROTOCOL.md) | Result | Outcome |
|---|---|---|
| Deterministic: adopted tensor improves matched contrast and actual incumbent | 282 adopted, 0 violations | pass |
| Nonlinear ZI cardinality vs LightGBM CQR, upper95 <= +1% | candidate -0.16%; supplied_tuned -1.57% | pass |
| Nonlinear ZI mean error vs baseline, upper95 <= 0 | upper95 -64.3% | pass |
| Smooth x1 on >= 90% of nonlinear draws | 100% | pass |
| Interaction selected <= 10% on each additive nonlinear control | 0% on both | pass |
| Family-supplied representation >= 90% | 0.90 | pass (at floor) |
| Ordinary genuine interaction: no representation regression, mean error upper <= 10% | 0.74 vs 0.74; upper +6.7% | pass |
| Gaussian regression controls | identical to baseline (0 differing) | pass (no change; #629 not addressed) |
| Other count panels: mean-error upper95 <= +10% | low +131%; noise_zero_inflated +89%; zero_inflated +14%; interaction_supplied_family +319% | **fail** |

Overdispersed (+6.0%), high, original, null Poisson and null zero-inflated pass.

## Decision

A failed statistical gate leaves #679 open (PROTOCOL.md). The candidate is not promoted:
its default change for NBinomial mean fits is not merged.
