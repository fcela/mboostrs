# Confirmation 2 result: the shipped procedure on fresh draws (#679)

Reserved cohorts: count seeds 6799200-6799249 and Gaussian seeds 6799400-6799449,
independent test seed + 100000. Recorder: `benches/study_interaction_production.py
--phase confirmation2`, run at the frozen tree of CONFIRMATION2_DECISION.md
(c50e8193; package source digest sha256:2c35a9d8c1a21a48419c64f9ff19667d11513d4d1d83a252c815ac81b73a4503).
Candidate: the production default of that tree (incumbent guard with
`mean_step="fixed"`). Output: `production_confirmation2_*.jsonl.xz`
(5500 records; 5489 ok, 11 failed, all `supplied_fixed`: five on
`interaction_zero_inflated`, five on `interaction_supplied_family` and one on
`nonlinear_interaction_zero_inflated`, where the fixed-step Poisson fit cannot
handle responses in the hundreds; kept in the output, as in the first
confirmation). The first count attempt was interrupted by a machine reboot
before any panel finalized and the count phase was rerun in full; every record
below is from the completed runs.

Identity checks: `candidate` and `guard_only` agree on every count panel (the
production default is the fixed configuration; no systematic difference), and
`candidate` and `baseline` agree on every Gaussian panel.

## Gates

| Gate (PROTOCOL.md, CONFIRMATION2_DECISION.md) | Result | Outcome |
|---|---|---|
| Deterministic: adopted tensor strictly improves the actual incumbent | 258 adopted, 0 violations | pass |
| Nonlinear ZI efficiency vs LightGBM CQR, upper95 <= +1% | candidate width -2.70%, count size +2.73%; supplied_tuned width -6.58%, count size -1.43% | **fail** (candidate count size) |
| Nonlinear ZI mean error vs baseline, upper95 <= 0 | upper95 -15.7% | pass |
| Smooth x1 on >= 90% of draws, each additive nonlinear control | 100% on both | pass |
| Interaction selected <= 10% on each additive nonlinear control | nonlinear_poisson 0/50; nonlinear_zero_inflated 3/50 | pass |
| Other count panels: complete, coverage >= .88, mean-error upper95 <= +10% | null_poisson, null_zero_inflated, noise_zero_inflated, zero_inflated: coverage .897-.904, upper95 0.0 | pass |
| Genuine interaction: no representation regression, mean-error upper95 <= +10% | ordinary 38/50 vs 38/50 baseline; curved (new panel) 40/50 vs 40/50; upper95 0.0 | pass |
| Family-supplied interaction representation >= 90% | 50/50 | pass |
| False inclusion, paired McNemar .05 on null/noise panels | 0 added, 0 removed, p = 1.0 | pass |
| Gaussian controls: mean-error upper95 <= +10%, no false scale adoption increase, no genuine scale adoption decrease, noise inclusion unchanged | upper95 0.0 on all ten; adoption and inclusion identical to baseline | pass |

## Decision

One failed gate leaves #679 open (PROTOCOL.md). On fresh draws the shipped
configuration is confirmed for representation (curve terms on every additive
draw, interactions within bound, no false inclusions), for latent-mean error
against the pre-guard baseline, for both genuine-interaction controls
including the new curved panel, and for every other retained panel. The
remaining defect is narrow: on the curved zero-inflated panel the discovered
fit's count sets average +1.67% larger than prespecified LightGBM CQR
(one-sided 95% upper +2.73% against a +1% bound), while its interval width is
smaller (upper95 -2.70%) and the tuned supplied track passes the same
efficiency gate (count-size upper95 -1.43%), so the headroom is in the
discovered fit's calibration, not in the panel. The same gate failed on the
development draws (upper95 +3.26%): the shortfall is a stable property of the
configuration, not fresh-seed noise. A change aimed at this gate requires its
own fresh confirmation.
