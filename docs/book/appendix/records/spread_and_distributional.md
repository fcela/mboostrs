(record-spread-and-distributional)=
# Conditional spread selection and distributional calibration

This record consolidates the empirical studies on heteroscedastic scale discovery, sharp-mean controls, standardized conformal scoring, and response-unit alignment supporting [](../../benchmarks/nonmonotone_spread.ipynb), [](../../benchmarks/gaussian_two_column.ipynb), [](../../benchmarks/gaussian_scale_development.ipynb), [](../../benchmarks/lss_response_scale.ipynb), [](../../benchmarks/conformal_response_units.ipynb), and [](../../benchmarks/slid_lss_sim.ipynb).

## Nonmonotone spread nomination and mean-adequacy guard

**Nomination.** Extending the heteroscedasticity screen from monotone log-squared-residual slopes to an 8-bin held-out curve test enabled discovery of nonmonotone variance patterns (absolute and bell-shaped spread). Across 20 fresh draws per process ($n = 1{,}000$, 500/500 fit/calibration split, $n_{\text{test}} = 2{,}000$), curve-sensitive nomination reduced 90% prediction-interval width by 4.63% (SE 0.89 points, coverage 0.899) on absolute scale and 8.18% (SE 1.88 points) on bell scale, with 0/20 false scale adoptions on smooth constant-variance and heavy-tail ($t_3 / \sqrt{3}$) controls.

**Mean-adequacy confirmation guard.** On sharp step-boundary means ($3\,\mathrm{sign}(x_1 - a)\,\mathrm{sign}(x_2 - b) + x_3$) with constant variance, a fixed 200-step screening mean left boundary approximation error in the residuals, causing false varying-scale adoptions on 3/10 axis-aligned and 2/10 shifted development draws. Requiring nominated scale terms to pass a second confirmation comparison with fold-tuned stopping eliminated those false adoptions. On a 4,000-record confirmation battery (10 processes × 40 fresh draws × 10 methods; 4,000/4,000 completed):

| Process group | Draws | Varying scale selected |
|---|---:|---:|
| Constant-variance controls (axis, shifted, smooth, null, heavy-tail; 5 × 40) | 200 | **0 / 200** |
| Smooth varying-scale processes (absolute, bell, monotone; 3 × 40) | 120 | **120 / 120** |
| Sharp varying-scale processes (axis, shifted; 2 × 40) | 80 | **0 / 80** |

**Documented limitation.** Automatic discovery does not recover a smooth varying scale beneath a sharp discontinuous mean (0/40 on both sharp varying-scale processes). Three alternative conditional-scale prototypes—fixed-location GaussianLSS on absolute residuals, inner-tuned cross-fitted mean residuals, and Student-$t_4$ scale likelihood—were tested in development; all held constant-variance false adoptions at 0/10, but none exceeded 4/10 recall on sharp varying-scale controls, so none was promoted.

## Gaussian conformal scoring and two-predictor confirmation

- **Standardized score vs. distributional quantile bands:** Holding fitted GaussianLSS means and scales fixed across the 10-process battery (1,600 records), standardized conformal scores $|Y - \mu| / \sigma$ failed the 1% width-efficiency gate on smooth bell scale (supplied-fixed width excess +3.41%, one-sided 95% upper +7.25%; discovery excess −2.07%, upper +1.82%). The existing distributional quantile band remains the default.
- **Two-predictor Gaussian confirmation:** On the original two-predictor bell and monotone heteroscedastic tasks (550 development + 2,750 confirmation records, all completed), both data-only discovery and supplied separate-budget GaussianLSS workflows passed their frozen coverage and width gates against the five tree/CQR comparators.

## Response-unit alignment in conformal intervals

Two deterministic numerical audits verified that conformal calibration and prediction endpoints operate in original response units when response transformations (e.g., log, square root, centering) are active:

- **LSS response scale (12 configurations: 3 transforms × 2 stopping prefixes × internal/external calibration):** Maximum endpoint discrepancy against an independent NumPy/SciPy reference dropped from 1.48–2.04 response units (with external-centering unpacking errors) to $\le 8.88 \times 10^{-16}$ units.
- **Standard conformal response units (132 cases: 112 fixed, 8 tuned, 12 explicit refusals):** All 120 supported cases agreed with the independent reference to $\le 1.33 \times 10^{-15}$ units, while the 12 center/scale-tuned cases preserved explicit refusal for fold-dependent scales.

## SLID and simulated location–scale study

Evaluating `mboostrs` on SLID wages and a simulated location–scale benchmark (260 mboostrs rows paired against 220 competitor rows) confirmed that automatic family inference, identifier/response-copy hygiene, and fitted scale exposure closed the SLID discovery CRPS gap to the best competitor from +34.07% to +1.15% (RMSE gap +28.76% → +0.85%) and restored nominal 80% coverage from 0.716 to 0.804 (95% coverage 0.950). On the simulated location–scale tracks, `mboostrs` outperformed the best competitor by −4.56% CRPS (known specification) and −4.44% CRPS (discovery), with 0.980 location precision, 1.000 location recall, and 1.000 scale precision and recall.

**Audit artifacts.** Protocols, development decisions, confirmation summaries, and validation records are archived in `docs/book/benchmarks/` (`conditional_scale_comparison*/`, `gaussian_scale_calibration/`, `gaussian_two_column/`, `lss_response_scale/`, `conformal_response_units/`, `nonmonotone_spread/`, `sharp_mean_controls/`, and `slid_lss_sim/`).
