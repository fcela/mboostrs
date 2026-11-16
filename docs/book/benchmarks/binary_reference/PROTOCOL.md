# Binary-reference dictionary controls

Instrument `binary-reference/1`, frozen before outcomes for #572. The proposed
rule makes the continuous partner's baseline smooth available in every
binary/continuous interaction comparison, not just string factors. Preserve
existing single-column terms, transforms, factor/scalar coding, heredity and
adoption thresholds. All fitting/CV remains in existing Rust-backed functions.

Nine fresh families, six draws each. Seed619300+10000*family_index+replicate0:5,
500 training rows and2,000 independent test rows. x,z,n0,n1 independently
Uniform(-1,1), b independent Bernoulli(0.5), noise Normal(0,0.5).

- categorical, indicator, boolean, signed, shifted: mean
  `2*sin(pi*x)*(1-2*b)+0.5*z`; g encoded a/b, integers0/1, bool,1-2*b,2+b.
- active: g=b numeric; mean `2*sin(pi*x)*b+0.5*z`.
- smooth: g=b numeric irrelevant; mean `2*sin(pi*x)+0.5*z`.
- null: g=b numeric; mean0.
- logged: replace x by exp(u), u independently Uniform(0,4), g=b numeric,
  mean `2*sin(pi*(u/2-1))*(1-2*b)+0.5*z`.

Run discover(train,"y",seed=replicate,folds3,max_mstop800,spread=False).
No truth, formula or test rows reach discovery. Predict its public model with
clip extrapolation. Retain complete logs, formulas, warnings, input/prediction
hashes, MSE and actual package/native/source identities. Serial jobs, one
thread, timeout240seconds each; explicit failures retained.

Promotion gate: every54before and54after job succeeds; each family's
mean after/before MSE ratio≤1.01 independently; null interaction adoptions do
not increase; categorical predictions replay exactly; indicator and boolean
families improve on average and have a nonconstant zero-level reference curve
on every draw. Signed, shifted and active scalar behavior has the same
non-regression gate, with every regression retained. Record per-seed values,
paired change and SE=sampleSD/sqrt6; gains cannot compensate another family's
failure. The reference curve uses40x values between training quantiles0.02/0.98,
other numerics at training means, g at its smallest observed level, and range
above1e-8response units means nonconstant. This gate protects the parent;
it does not establish competitive validity or calibrated FDR.

The parent is categorical repair #568 (sourcead77f3b, package contentf290194).
Before and after use the same newly frozen rows/calls. Existing old controls
and exploratory618211 are development evidence only. If this gate fails,
preserve it and freeze new controls before changing the rule. An unchanged
full encoding-tracks/1 rerun supplies the separate two-track competitor gaps
and paired SEs; all shifted/signed losses remain visible. No new scorecard
grade from this control gate, no aggregate. Each fix has its own branch/MR.
