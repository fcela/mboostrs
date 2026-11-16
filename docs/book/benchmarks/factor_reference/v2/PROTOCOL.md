# Factor-reference dictionary repair

Instrument `factor-reference/2`, frozen before control outcomes for issue #560.
The proposed repair supplies a baseline smooth alongside a treatment-coded
factor-by smooth when marginal elimination has removed the continuous partner.
It preserves existing single-column terms and their scale, numeric-indicator
behavior, factor coding and the existing 2.54-SE paired adoption threshold.
This is dictionary completion; all statistical fitting/CV stays in Rust-backed
public functions. Record the compared reference terms in the discovery log.

## Fresh controls

Use seven previously unexamined generators, six independent draws each. Seeds
are 618500 + 10000*case_index + replicate 0–5. Each has 500 training and 2,000
independent test rows. Independently x,z,n0,n1~Uniform(-1,1), g~Bernoulli(0.5),
noise Normal(0,0.5); categorical g is a/b unless specified. Means:

- opposed: 2*sin(pi*x)*(1-2*g) + 0.5*z.
- relabelled: same mean, g labels zeta/alpha, testing the opposite reference.
- smooth: 2*sin(pi*x)+0.5*z, g irrelevant.
- null: 0, all predictors irrelevant.
- numeric: 2*sin(pi*x)*g+0.5*z, g is numeric0/1, not a factor.
- factor_main: 1.5*sin(pi*x)+2*g+0.5*z, no interaction.
- logged: generate u~Uniform(0,4), replace x with exp(u),
  mean2*sin(pi*(u/2-1))*(1-2*g)+0.5*z; tests recorded log scale.

Before and after use identical rows and calls:
`discover(train,"y",seed=replicate,folds=3,max_mstop=800,spread=False)`;
no formula, truth or test data reach discovery. Predict through result.model
with clip extrapolation. Keep full logs, formulas, inference errors/warnings,
input and prediction digests and exact loaded package/native identities.
Run serially, one thread, timeout240s per draw. Retain explicit failures.

## Non-regression gate

Require all 84 phase/draw jobs complete; no family's mean after/before outer-MSE
ratio above1.01; zero new null interactions beyond the parent's count; and
byte-identical numeric-indicator predictions on every draw. Require both
opposed-label families to improve and represent a nonconstant reference-level
curve. Report paired MSE changes and Monte Carlo SE by family, every adoption
and failure. This gate compares the parent on these generators; it does not
establish the broader 1% best-competitor target or calibrated FDR.

If a rule changes after these outcomes, retain the negative trial and freeze
another version with fresh control seeds. The separate corrected
`predictive-tracks/2` study must be rerun unchanged in both information tracks,
with all 168 original method/draw jobs and the same named competitors/settings.
Retain its valid baseline, paired improvement and SE. The first v1 supplied
factor formula is invalid instrument evidence, not a library baseline.

No new scorecard grade is assigned from this control gate; other dimensions
remain unmeasured by this repair. Each actual fix gets its own MR; no merging
without explicit user authorization.

## Parent and instrument history

Version1 found an independent reporting failure (#564): three original logged
parent controls crashed before returning their result. Its full outcomes remain
retained, and its all-success confirmation gate did not pass. The separate
schema repair replays all 42 original controls, resolves all 3 failures and leaves
all 39 previously successful predictions byte-identical. It changes metadata
resolution, not fitting or the interaction dictionary.

For this version, the parent includes only that schema repair, and the after
package additionally completes the factor-reference dictionary (#560). The
seven generators, six draws per generator, calls, 1.01 per-family limit, null
adoption constraint and exact numeric prediction requirement are unchanged.
Freeze new seeds618500 plus the same family offsets before collecting any
version 2 outcomes. The original controls are development/replay evidence only.
This version's matched parent isolates the dictionary repair from #564.

For reference-curve verification, also predict a 40-point grid between the
training x quantiles 0.02 and 0.98, holding numeric partners at their training
means and the factor at its lexicographic first observed level. Record the
max-minus-min response range. For the two opposed-label families require a
range above 1e-8 response units on every after draw (the numerical definition
of nonconstant). This grid is a reporting check and selects no settings.
