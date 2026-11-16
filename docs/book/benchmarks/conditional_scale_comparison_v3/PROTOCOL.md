# Robust conditional spread comparison

Instrument conditional-scale-comparison/3, frozen before its implementation
and development. Version2's 100 development records remain retained. It keeps
zero sharp constant-variance adoptions but finds only2/10 axis and4/10 shifted
varying controls; its null controls also record a prototype-only attempt to
tune an intercept-only path. Version2 is not ready for confirmation or product
adoption, and no confirmation has been generated.

Retain version2's explicit inner mean tuning, out-of-fold nomination, identical
fold locations across scale candidates and final unchanged Gaussian joint fit.
For an intercept-only selected mean, follow discovery's existing convention:
fit the constant Gaussian mean without a meaningless positive-step CV search,
record that choice and retain its held-out predictions. This repairs the
prototype harness; original version2 refusals remain visible.

Change only the conditional scale fit/score to a Student-t reference with fixed
four degrees of freedom. Hold location0 and df4 fixed, fit sigma for200 steps
at rate.1, no stabilization, offsets(mu,sigma,log-df)=(0,0,log4), budgets(0,200,0).
Judge each scale candidate with that same fixed-df Student-t NLL on the same
held-out mean residuals. No outcome chooses df or a robustness cutoff. The
scale-score derivative is bounded in the squared standardized residual,
which limits leverage from large mean-approximation errors near estimated
sharp boundaries. Symmetry again permits absolute instead of signed residuals.
This reference selects predictive residual spread, not an identified Gaussian
noise mechanism; final returned models remain Gaussian as before.

Use the same development draws and ten processes as versions1/2, and retain
all versions, costs, warnings, failed stages, intervals, calibration calls,
fold means, tuning curves and scale scores. Production numerics/bindings and
full cost admission remain prerequisites before fresh6296100–6296149. All
version1 acceptance gates, including90% true-scale adoption, clustered false-
adoption comparison and all coverage controls, remain unchanged. A failed
development gate leaves the default untouched and confirmation ungenerated.
A #630 claim still requires the original two-predictor task, separately frozen.
