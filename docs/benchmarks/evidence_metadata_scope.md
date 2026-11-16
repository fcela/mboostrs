# Diagnostic uncertainty metadata

The uncertainty block in historical evidence produced by
`benches.study_evidence.evidence_document` described independent simulation
draws even when a caller ran a deterministic numerical diagnostic. Those
historical JSON files remain unchanged. Their protocols determine their
interpretation; this note corrects the inherited metadata for these audits:

| Audit | Experimental unit | Permitted interpretation |
|---|---|---|
| `conformal-predictor-coding/1` | One deterministic fixture per response family, shared by its configurations | Numerical reference agreement; no independent sampling replicates or Monte Carlo standard errors |
| `conformal-response-units/1` | One deterministic fixture shared by all configurations | Original-response numerical agreement; no coverage or speed inference |
| `binomial-prediction-scale/1` | Three recorded splits of one public data set and a fixed synthetic probe | Descriptive scores and reference agreement; no independent-draw Monte Carlo standard errors |

The conformal protocols and raw archives are in
[`docs/book/benchmarks/conformal_predictor_coding`](../book/benchmarks/conformal_predictor_coding/PROTOCOL.md)
and
[`docs/book/benchmarks/conformal_response_units`](../book/benchmarks/conformal_response_units/PROTOCOL.md).
The binomial protocol and results are in
[`docs/book/benchmarks/binomial_prediction_scale`](../book/benchmarks/binomial_prediction_scale/PROTOCOL.md).
These are diagnostics of specified numerical contracts, rather than new
experiments establishing predictive superiority over competitors.

Future calls can supply an explicit `uncertainty=` mapping to
`evidence_document`. The three audit runners now supply their actual scope.
The default retains the historical independent-draw contract for simulation
studies. Custom contracts are copied into each evidence document, including
an explicitly empty mapping. This change does not remeasure or rewrite any
frozen result. The discrepancy and correction are tracked in
[issue #655](https://github.com/fcela/mboostrs/issues/655).
