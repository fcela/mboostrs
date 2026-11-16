# Distributional conformal response units

This diagnostic for issue #633 fixes its cases before implementation or
measurement. Baseline source: 0d94682. It checks response units and calibration
independence; it does not estimate coverage or compare predictive competitors.

Use NumPy seed 6319, 300 independent x uniform[-1,1] values and
y = exp(0.3 + 0.6*x + 0.15*N(0,1)). Fit GaussianLSS with both parameters using
`TRANSFORM(y) ~ bols(x)`, for log, center and scale. Use 60 boosting iterations,
rate 0.1, no tuning. Query x = [-0.5, 0, 0.5]. Reserve the first 150 rows of
the NumPy permutation with seed 9 for calibration, the remainder for fitting.
Request level 0.9; evaluate the full refit and a fixed 30-iteration prefix.

The reference is a separately fitted model on exactly these proper-training
rows, with response preprocessing re-estimated there. For log, compute
original-response quantiles with SciPy's lognormal quantile function. For
center/scale, use Gaussian quantiles and the proper-training response mean
and sample standard deviation. Compute raw-response CQR scores and the
finite-sample corrected order statistic independently in the diagnostic.
Report maximum absolute endpoint/centre error against this reference,
separately for every transform and prefix. Retain endpoint arrays, correction
quantiles, actual row identities, source hashes and the data hash. Baseline
and corrected arms must use identical inputs, partitions and specifications.

Acceptance requires agreement within 1e-10, agreement between internal and
external calibration on that partition, and no claimed original-response mean
for a transformed LSS fit. Regression tests must also establish that changing
only calibration responses cannot alter proper-training response or predictor
transform parameters, coefficients or prediction, and exercise missing
predictors, invalid calibration responses, absent raw inputs and row alignment.
Numerical fitting, quantiles, scoring and widening continue to use existing
core APIs. The diagnostic's independent NumPy/SciPy calculations are references,
not production estimators.
