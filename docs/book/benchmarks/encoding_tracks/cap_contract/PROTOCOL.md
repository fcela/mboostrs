# Discovery maximum-grid contract

Contract instrument `discover-cap/1`, frozen before measuring the API before and after repair
for #574. This validates argument propagation, not a changed statistical policy
or a new scorecard grade. Keep encoding-tracks/1 primary outcomes unchanged.

Repeat eight shifted encoding draws from the frozen generator with the same
public discover call, except an explicit requested max_mstop25600 in BOTH
parent and repaired phases. Other seed/folds/default stages/data/prediction
settings are unchanged. Observe actual public tuner returns inside _tune_mean:
record its caller's declared grid, public maximum, actual largest explored
position and chosen mstop. Retain formulas, full logs, predictions, inputs,
MSEs and all failures/identities. A profiler observes returns without changing
tuning or choosing settings. Report parent declared/actual mismatches and
whether the repaired search reaches a larger requested grid when needed.
Do not require every draw's optimum to exceed10000 if its curve already turns.

Also repeat all80original known/discovery encoding-tracks/1 fits at their
UNCHANGED1600limit on the cap-repaired package. They must exactly replay every
input digest, prediction digest and MSE from the already measured after phase.
The four tree competitors and Ridge are not refitted for this library-only
contract check; their frozen measured outputs remain unchanged and are not
relabelled as new measurements. The full280job main study remains the
performance comparison; this is an additional bounded regression check.

Jobs run serially, one thread, timeout240seconds each. Retain every planned
job including failures. No new best-competitor, FDR or scorecard claim; the
original shifted fixed-budget gap remains #573. Public-boundary unit tests
cover800,12800,25600 limits and preflight propagation. No numerical arithmetic
is added: use the existing Rust-backed tuner with the caller's true ceiling.
