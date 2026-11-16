# Answer key: simulated arm of the SLID + LSS-simulation benchmark

Fixed and committed before any run. Do not edit after the baseline run;
a change starts a new run and is recorded as one.

## Generator (seed s in 0..9, n_train = 3000, n_test = 3000)

Covariates, drawn independently per row unless noted:

| name | distribution | role |
|---|---|---|
| x1 | Uniform(0, 1) | nonlinear location effect |
| x2 | Uniform(-1, 1) | quadratic location effect |
| x3 | Uniform(0, 1) | interaction partner (varying slope by g) |
| g  | categorical {"a","b","c"}, P = 0.4/0.35/0.25 | factor; modifies x3 slope |
| x5 | Uniform(0, 1) | scale (log-sigma) effect only |
| x6 | 0.6*x1 + 0.8*Uniform(0,1), rescaled to [0,1] | nuisance, correlated with x1 |
| x7 | Normal(0, 1) | nuisance |
| x8 | categorical, 5 levels uniform | nuisance |

Response: y ~ Normal(mu, sigma)

- mu = 1.0 + 1.5*sin(2*pi*x1) + 1.2*x2^2 + slope(g)*x3,
  slope(a) = 0, slope(b) = 1.5, slope(c) = -1.0, plus main effect of g:
  +0.0 (a), +0.3 (b), -0.2 (c)
- log(sigma) = -0.7 + 1.4*x5   (sigma from ~0.50 to ~2.0)

## True structure (scored in D1a / D1b)

Active in location: f(x1) nonlinear, f(x2) nonlinear (quadratic), g main
effect, x3-by-g interaction. Active in scale: x5 (linear on log scale).
Inactive everywhere: x6, x7, x8. x5 has NO location effect; x1..x3, g have NO
scale effect.

## Scoring rules

- D1a (known spec): mboostrs and the classical GAMLSS-style competitor get the
  true formula; score = RMSE of fitted mu(x) and log sigma(x) vs truth on test
  rows, and recovered x3 slopes per level.
- D1b (discovery): data + response name only. Score = true/false selection of
  each term in each parameter (precision, recall), and false discoveries among
  x6, x7, x8 and the x5-in-location / x1-in-scale traps.
- D9: 80% and 95% interval coverage and mean width on test rows.
- D11: test CRPS and Gaussian log score (primary), test RMSE (secondary),
  gap to the retrospective best competitor with paired SE over 10 seeds.
- D5: wall-clock per fit and per "what-if" refit.
