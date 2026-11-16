# Findings report — heteroscedastic.csv (Arm C: mboostrs, no skill files)

**Analyst:** benchmark agent (mboostrs only; docstrings/`help()`, no skill files)
**Data:** 800 rows × 6 cols (`x0..x4`, `y`); all predictors on [0,1], no missing values.
**Wall clock:** 2026-10-12 03:42:55 UTC → 04:01:02 UTC (~18 min)

All claims are associations, not causal effects.

## 1. Which predictors matter

| Predictor | Verdict | Evidence |
|---|---|---|
| **x0** | **Signal — mean, linear positive** | Partial dependence +1.40 → +3.46 over x0∈[0,1]; OLS slope +2.01/unit (se small) |
| **x1** | **Signal — mean, inverted-U** | Partial dependence 1.18 → 2.60 → 1.23, peak ≈0.5; fits ≈1.56·sin(π·x1) |
| **x2** | **Signal — variance only, no mean effect** | Mean flat (PD 2.60–2.69); `heteroscedasticity` by-column t=+13.46, spread_ratio 4.29; LSS: σ(x2)≈exp(−1.2+1.7·x2) |
| x3 | Noise | PD flat (2.54–2.68); variance t=+0.99; nothing in any fit |
| x4 | Noise | PD dead flat; variance diagnostic t=−3.20 but adding x4 to σ gains only +0.5 CV logLik (se 7.7) — does not survive out-of-sample |

## 2. Functional forms and magnitudes

Final model — Gaussian LSS: mu `y ~ bbs(x0,df=4)+bbs(x1,df=4)`, sigma `y ~ bols(x2)`:

- **x0 → mean: linear, +2.0 per unit of x0** (effect range ≈2.0 over x0∈[0,1]; binned residual means show no leftover curvature).
- **x1 → mean: symmetric hump, peak at x1≈0.5, amplitude ≈1.56** (well described by 1.56·sin(π·x1); residual-by-bin check clean, ±0.17 max).
- **x2 → variance: log-linear. σ(x2) ≈ exp(−1.23 + 1.73·x2)** — σ≈0.36 at x2=0 rising to ≈1.9 at x2=1, a ~5× spread. Binned raw sd 0.78→1.81 confirms.
- **x3, x4:** no detectable mean or variance effect.

## 3. Tested and ruled out

- **Constant variance** — rejected decisively: sigma~x2 beats sigma~1 by +32 CV logLik (se ~10); overall fitted-value heteroscedasticity t=−0.05 only because the variance driver (x2) is orthogonal to the mean.
- **x4 variance effect** — the by-column t=−3.20 looked suggestive, but CV says otherwise (+0.5 ± 7.7). Ruled out; reported as noise.
- **Discontinuity** — `screen_discontinuities` top hit `x2 > −2.976+4.389·x1` had z=+4.97 but **type_z=+0.28 (not typed as a jump)**; adding the diagonal indicator to the mean changes CV logLik by exactly 0.0 (−197.8 → −197.8). No jump. This is the screen's typing doing its job: a raw z without jump typing does not survive honest modeling.
- **Interactions** — `screen_interactions` top hits (x0×x1 +0.51 etc.) do not survive: adding the centered x0·x1 product changes CV logLik −197.8 → −197.9. Ruled out.
- **x2/x3/x4 mean effects** — flat partial dependences; nothing selected.
- **Misspecified x1 shape** — sine form leaves no binned residual pattern.

## 4. Methods, libraries, seeds, fit count

- **Library:** mboostrs 0.1.3 only (PR #489 build), via `/home/hatch/workspace/mboostrs-venv/bin/python`; pandas/polars/numpy/sklearn (OLS, KFold) for handling and quantification. No skill files consulted; API learned from docstrings/`help()` (`heteroscedasticity`, `gamboostLSS` with `family=GaussianLSS()`, formula-dict `{"mu":…, "sigma":…}`, `predict(..., parameter=, type="response")`, `coef_all`).
- **Procedure:** (1) marginal screens (correlations, binned means/sds); (2) additive `gamboost` mean fit + partial dependences; (3) `mb.heteroscedasticity` overall + by-column → x2 flagged (t=+13.5); (4) `gamboostLSS` mu/sigma fits, logLik ladder sigma~1 / ~x2 / ~x2+x4; (5) 5-fold CV logLik comparisons for sigma specs, the diagonal indicator, and the x0×x1 product (seed 11; second CV seed 7); (6) `screen_interactions`, `screen_discontinuities`; (7) narrow-band check on the flagged diagonal.
- **Seeds:** 7, 11 (CV splits); screens at defaults.
- **Model fits:** ~60 total — gamboost ×3, gamboostLSS ×~8 (+50 fold-fits inside the three 5-fold CV comparisons), screens ×3. Screens run in seconds (Rust core).
- **Compliance:** analysis used only the CSV. No package-source inspection, no `mboostrs.datasets`, no truth-key files, no skill files, no prior hypotheses — every number above was computed from the CSV in this session.

## 5. Bottom line

Two mean effects (x0 linear +2.0/unit; x1 symmetric hump peaking at 0.5, amplitude ≈1.56), **non-constant variance driven by x2** as σ(x2)≈exp(−1.2+1.7·x2) (~5× spread across x2's range), x3/x4 noise. No jumps, no interactions. The final distributional model (Gaussian LSS, location + scale) beats the constant-variance version by 32 CV logLik points.

## 6. VARIANCE ANALYSIS

**Is the variance constant? No.** The residual spread grows strongly with x2 and with nothing else.

- `mb.heteroscedasticity` on the additive mean fit: overall slope t=−0.05 (spread_ratio 1.00) — the fitted-value test sees nothing, because the variance driver is orthogonal to the mean. The by-column test finds it: **x2 t=+13.46, spread_ratio=4.29**; all other columns |t|<3.3, and x4's −3.2 does not survive CV (see §3).
- **Estimated variance function** (GaussianLSS, log link on σ): log σ = −1.23 + 1.73·x2, i.e. σ ≈ 0.36 at x2=0 → ≈1.9 at x2=1.
- **What I did about it:** the final model is distributional — `gamboostLSS({"mu": "y ~ bbs(x0,df=4)+bbs(x1,df=4)", "sigma": "y ~ bols(x2)"}, family=GaussianLSS())`. Intervals from a constant-variance fit would be ~2.6× too narrow at high x2 and ~2.8× too wide at low x2; the LSS fit gives each observation its own scale.
- **What could be wrong:** the log-linear σ(x2) form is an assumption — a spline on σ would be the next check if the log-linear fit showed misfit (it doesn't: binned residual |z|-scores are flat in x2 after scaling). If the variance driver were correlated with a mean predictor, the location/scale split would be harder to identify; here x2 is clean.
