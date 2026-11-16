//! gamboostLSS: component-wise boosting for GAMLSS (location/scale/shape).
//!
//! Unlike the single-predictor [`crate::families::Family`] stack (where scale
//! parameters are *calibrated* / profiled out), gamboostLSS gives EACH
//! distribution parameter `theta_k = g_k^{-1}(eta_k)` its own additive predictor
//! `eta_k` and BOOSTS all of them. Fitting is **cyclic**: at every iteration we
//! loop over the parameters and take one component-wise boosting step on each,
//! using the other parameters' current fitted values in the partial gradient.
//! This mirrors R's `gamboostLSS` (default `method = "cyclic"`).
//!
//! This module reuses the existing learner machinery
//! ([`PreparedLearner::fit_step_ref`], `prepare_*`, [`BoostControl`],
//! [`BoostingPath`]); it only adds the multi-parameter family trait
//! ([`LssFamily`]) and the cyclic driver ([`fit_lss`]).

use ndarray::{Array1, ArrayView1};
use std::time::Instant;

use crate::control::{BoostControl, StepRule};
use crate::engine::observe_candidate;
use crate::families::{golden_grid_min, minimize_brent, R_OPTIMIZE_TOL};
use crate::grid_search::{search_grid, FoldRisks, GridSearch, GridStep};
use crate::learners::PreparedLearner;
use crate::path::BoostingPath;
use crate::report::{EngineRoute, FitReport, InstrumentationLevel, PathRow};
use crate::MboostrsError;
use libm::lgamma;

/// How a distributional fit rescales its partial gradient before selection.
///
/// The parameters of a GAMLSS live on different scales -- a location in the
/// response's units, a log-scale in dimensionless ones -- so their gradients
/// have different magnitudes and a single `nu` takes very different-sized
/// steps on each. `gamboostLSS` addresses this by dividing each gradient by a
/// measure of its own spread, which puts the parameters on comparable footing
/// and stops one of them stalling while another overshoots.
///
/// The definitions are `gamboostLSS:::stabilize_ngradient`'s, including its
/// floors and ceilings, so a fit with the same option produces the same path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stabilization {
    /// Use the gradient as the family computes it. The `gamboostLSS` default,
    /// and the only setting under which `nu` means what it means for a
    /// single-parameter fit.
    #[default]
    None,
    /// Divide by the weighted median absolute deviation from the weighted
    /// median. Robust to a few extreme gradient entries, which is what makes
    /// it the recommended setting for heavy-tailed responses.
    Mad,
    /// Divide by the root weighted mean square. Cheaper than `Mad` and
    /// sensitive to the same extreme entries `Mad` is chosen to ignore.
    L2,
}

impl Stabilization {
    /// Parse the spelling the frontends and `gamboostLSS` use.
    pub fn parse(name: &str) -> Result<Self, MboostrsError> {
        match name {
            "none" => Ok(Self::None),
            "MAD" | "mad" => Ok(Self::Mad),
            "L2" | "l2" => Ok(Self::L2),
            other => Err(MboostrsError::Control(format!(
                "stabilization must be \"none\", \"MAD\" or \"L2\", got {other:?}"
            ))),
        }
    }
}

/// `gamboostLSS`'s weighted median, including its tie rule.
///
/// Zero-weight observations are dropped, the rest are sorted, and the value at
/// the first index whose cumulative weight share exceeds one half is taken.
/// When the weights sum to an even whole number the two straddling values are
/// averaged -- the convention that makes an even number of unit weights give
/// the ordinary median -- and otherwise the larger of the two is used.
fn weighted_median(values: &[f64], weights: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .zip(weights.iter())
        .filter(|(value, &weight)| weight != 0.0 && value.is_finite())
        .map(|(&value, &weight)| (value, weight))
        .collect();
    if pairs.is_empty() {
        return 0.0;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total: f64 = pairs.iter().map(|(_, weight)| weight).sum();
    if total <= 0.0 {
        return pairs[0].0;
    }

    let mut cumulative = 0.0f64;
    let mut upper = pairs.len() - 1;
    for (index, (_, weight)) in pairs.iter().enumerate() {
        cumulative += weight;
        if cumulative / total > 0.5 {
            upper = index;
            break;
        }
    }
    // The last index whose cumulative share is at or below one half, which is
    // `upper` itself when `upper` is the first element.
    let lower = if upper == 0 {
        0
    } else {
        let mut running = 0.0f64;
        let mut found = 0usize;
        for (index, (_, weight)) in pairs.iter().enumerate() {
            running += weight;
            if running / total <= 0.5 {
                found = index;
            } else {
                break;
            }
        }
        found
    };

    let integral = (total.fract() == 0.0) && (total % 2.0 == 0.0);
    if integral {
        (pairs[upper].0 + pairs[lower].0) / 2.0
    } else {
        pairs[upper].0.max(pairs[lower].0)
    }
}

/// Rescale a partial gradient in place, matching
/// `gamboostLSS:::stabilize_ngradient`.
///
/// The floors and the `L2` ceiling are upstream's: a divisor below `1e-4`
/// would amplify a gradient that is already numerically zero, and one above
/// `1e4` would shrink a legitimately large one to nothing.
pub fn stabilize_ngradient(gradient: &mut [f64], weights: &[f64], how: Stabilization) {
    match how {
        Stabilization::None => {}
        Stabilization::Mad => {
            let centre = weighted_median(gradient, weights);
            let deviations: Vec<f64> = gradient
                .iter()
                .map(|value| (value - centre).abs())
                .collect();
            let divisor = weighted_median(&deviations, weights).max(1e-4);
            for value in gradient.iter_mut() {
                *value /= divisor;
            }
        }
        Stabilization::L2 => {
            let total_weight: f64 = weights.iter().sum();
            let mean_square = if total_weight > 0.0 {
                gradient
                    .iter()
                    .zip(weights.iter())
                    .map(|(&value, &weight)| weight * value * value)
                    .sum::<f64>()
                    / total_weight
            } else {
                0.0
            };
            let divisor = mean_square.sqrt().clamp(1e-4, 1e4);
            for value in gradient.iter_mut() {
                *value /= divisor;
            }
        }
    }
}

/// digamma (psi) — derivative of ln Γ. Recurrence up to x ≥ 6, then the
/// Bernoulli asymptotic expansion; accurate to ~1e-12, matching R's `digamma`.
///
/// Returns `NaN` for `x <= 0` and for `NaN` input. LSS callers pass
/// `sigma = exp(eta_sigma)` or `phi = exp(eta_phi)`, which is positive in exact
/// arithmetic but underflows to `0.0` (or is `NaN`) once a diverging fit drives
/// the predictor towards `-inf` (#319: GammaLSS on Meuse log-zinc). Panicking
/// there aborted the whole Python process path through pyo3; a `NaN` gradient
/// instead makes the cycle's risk non-finite, and the engine's
/// `check_lss_cycle_risk` reports that as an ordinary solver error.
pub(crate) fn digamma(mut x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln()
        - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2
                    * (1.0 / 120.0
                        - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0)))))
}

/// R's GammaLSS per-observation loss with `sigma` = shape and `f` = log(mu).
/// Includes the `+ log(y)` constant (irrelevant to the fit's argmin but needed
/// for the risk value to match R, e.g. in cvrisk).
#[inline]
fn gamma_plloss(sigma: f64, y: f64, f_mu: f64) -> f64 {
    lgamma(sigma) + sigma * y * (-f_mu).exp() - sigma * y.ln() - sigma * sigma.ln()
        + sigma * f_mu
        + y.ln()
}

/// Each observation's negative log-likelihood under `family` at the
/// link-scale predictors `etas`, with the natural-scale parameter values
/// those predictors map to.
///
/// `etas` holds one length-`n` array per parameter, in `param_names()` order;
/// `y` holds `n * response_span()` values in the column-major layout the fit
/// uses. The losses are [`LssFamily::row_nll`], the same per-observation terms
/// the fit and the core's cross-validated risk sum, so a held-out score built
/// from them is the risk the core reports for those rows. The second value
/// holds one length-`n` vector per parameter from [`LssFamily::response`].
pub fn row_nll_and_parameters(
    family: &dyn LssFamily,
    y: &[f64],
    etas: &[Array1<f64>],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), MboostrsError> {
    let k = family.n_params();
    if etas.len() != k {
        return Err(MboostrsError::Shape(format!(
            "{} has {k} parameters, got {} predictor arrays",
            family.name(),
            etas.len()
        )));
    }
    let n = etas[0].len();
    if let Some((parameter, eta)) = etas.iter().enumerate().find(|(_, e)| e.len() != n) {
        return Err(MboostrsError::Shape(format!(
            "predictor {parameter} has {} rows, predictor 0 has {n}",
            eta.len()
        )));
    }
    let span = family.response_span();
    if y.len() != n * span {
        return Err(MboostrsError::Shape(format!(
            "{} takes {span} response value(s) per observation: expected {} for {n} \
             observations, got {}",
            family.name(),
            n * span,
            y.len()
        )));
    }
    if let Some(parameter) = etas.iter().position(|e| e.iter().any(|v| !v.is_finite())) {
        return Err(MboostrsError::Shape(format!(
            "predictor {parameter} contains a non-finite value"
        )));
    }
    let mut nll = vec![0.0; n];
    family.row_nll(y, etas, &mut nll);
    let parameters = etas
        .iter()
        .enumerate()
        .map(|(parameter, eta)| {
            let mut out = vec![0.0; n];
            family.response(parameter, &eta.to_vec(), &mut out);
            out
        })
        .collect();
    Ok((nll, parameters))
}

/// Construct an LSS family from the name the Python and R frontends use
/// (`"gaussianlss"`, `"gammalss"`, ...). `n_params` is checked against the
/// family and sets the number of components for `"dirichletlss"`.
pub fn make_family(name: &str, n_params: usize) -> Result<Box<dyn LssFamily>, MboostrsError> {
    let family: Box<dyn LssFamily> = match name {
        "gaussianlss" => Box::new(GaussianLSS),
        "gaussianknownvarlss" => Box::new(GaussianKnownVarLSS),
        "gammalss" => Box::new(GammaLSS),
        "studenttlss" => Box::new(StudentTLSS),
        "betalss" => Box::new(BetaLSS),
        "nbinomiallss" => Box::new(NBinomialLSS),
        "lognormallss" => Box::new(SurvivalLSS::log_normal()),
        "logloglss" => Box::new(SurvivalLSS::log_log()),
        "weibulllss" => Box::new(SurvivalLSS::weibull()),
        "lognormallss_uncensored" => Box::new(SurvivalLSS::log_normal().uncensored()),
        "logloglss_uncensored" => Box::new(SurvivalLSS::log_log().uncensored()),
        "weibulllss_uncensored" => Box::new(SurvivalLSS::weibull().uncensored()),
        "bccglss" => Box::new(BCCGLSS),
        "binomiallss" => Box::new(BinomialLSS),
        "geometriclss" => Box::new(GeometricLSS),
        "gumbellss" => Box::new(GumbelLSS),
        "inversegaussianlss" => Box::new(InverseGaussianLSS),
        "logisticlss" => Box::new(LogisticLSS),
        "zeroadjustedgammalss" => Box::new(ZeroAdjustedGammaLSS),
        "jsulss" => Box::new(JSULSS),
        "bctlss" => Box::new(BCTLSS),
        "bcpelss" => Box::new(BCPELSS),
        "negativebinomiallss" => Box::new(NegativeBinomialLSS),
        "pareto2lss" => Box::new(Pareto2LSS),
        "piglss" => Box::new(PIGLSS),
        "poissonlss" => Box::new(PoissonLSS),
        "zeroadjustedpoissonlss" => Box::new(ZeroAdjustedPoissonLSS),
        "zipolss" => Box::new(ZIPoLSS),
        "zinblss" => Box::new(ZINBLSS),
        "dirichletlss" => Box::new(DirichletLSS::try_new(n_params)?),
        other => {
            return Err(MboostrsError::Control(format!(
                "Unknown LSS family: {other}"
            )))
        }
    };
    if n_params != family.n_params() {
        return Err(MboostrsError::Shape(format!(
            "expected {} parameter blocks, got {}",
            family.n_params(),
            n_params
        )));
    }
    Ok(family)
}

/// A GAMLSS family: `K` distribution parameters, each with its own link,
/// offset, partial negative gradient, and inverse link. The joint
/// negative-log-likelihood is the boosting risk.
pub trait LssFamily: Send + Sync {
    /// Family name (e.g. `"gaussian"`).
    fn name(&self) -> &'static str;

    /// Number of distribution parameters (e.g. 2 for Gaussian mu/sigma).
    fn n_params(&self) -> usize;

    /// Number of response values stored per observation.
    ///
    /// Ordinary families return one. Survival families store time followed by
    /// status, and Dirichlet stores one column per composition component.
    fn response_span(&self) -> usize {
        1
    }

    /// Parameter names, in order (e.g. `["mu", "sigma"]`).
    fn param_names(&self) -> &'static [&'static str];

    /// Initial predictor value (on the link scale) for parameter `k`.
    ///
    /// `prior` holds the already-computed link-scale offsets of parameters
    /// `0..k` (offsets are computed in parameter order). R's gamboostLSS
    /// computes them sequentially: parameter `k`'s offset uses the offsets just
    /// computed for earlier parameters and family defaults for later ones.
    fn offset(&self, k: usize, y: &[f64], w: &[f64], prior: &[f64]) -> f64;

    /// Negative gradient of the risk w.r.t. `eta_k`, given ALL current
    /// predictors `etas` (length-`K`, each length-`n`, on the link scale).
    /// Writes the length-`n` pseudo-residual into `out`.
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], w: &[f64], out: &mut [f64]);

    /// Joint risk = `sum_i w_i * (negative log-likelihood)_i`.
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64;

    /// Negative log-likelihood of each observation, unweighted: `out[i]` is
    /// the term that [`LssFamily::risk`] multiplies by `w[i]`, so
    /// `sum_i w_i * out[i] == risk(y, etas, w)`.
    ///
    /// `y` holds `y.len() / n` values per observation in column-major order
    /// (value `j` of observation `i` at `j * n + i`), as `risk` reads it, and
    /// `n == out.len()`. Each row is evaluated by calling `risk` on that row
    /// alone with unit weight, so the two cannot disagree; a zero weight in
    /// `risk` does not hide an infinite row loss here.
    fn row_nll(&self, y: &[f64], etas: &[Array1<f64>], out: &mut [f64]) {
        let n = out.len();
        if n == 0 {
            return;
        }
        let span = y.len() / n;
        let mut y_row = vec![0.0; span];
        let mut eta_row: Vec<Array1<f64>> = etas.iter().map(|_| Array1::zeros(1)).collect();
        let one = [1.0];
        for i in 0..n {
            for j in 0..span {
                y_row[j] = y[j * n + i];
            }
            for (row, eta) in eta_row.iter_mut().zip(etas) {
                row[0] = eta[i];
            }
            out[i] = self.risk(&y_row, &eta_row, &one);
        }
    }

    /// Inverse link for parameter `k`: maps the link-scale `eta` to the
    /// natural-scale parameter into `out`.
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]);

    /// Validate the response values against the family's domain.
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError>;

    /// Heap-allocated clone of this family as a trait object.
    fn clone_box(&self) -> Box<dyn LssFamily>;

    /// Negative Hessian of each observation's log-likelihood with respect to
    /// the `K` linear predictors, unweighted: `out[i*K*K + a*K + b] =
    /// -d2 l_i / d eta_a d eta_b`, `out.len() == n * K * K`.
    ///
    /// Used by the penalized GAMLSS refit (`gam::try_gamlss_refit`). The
    /// default takes central differences of the analytic `ngradient` (the
    /// first derivative), with a step of `1e-5 (1 + |eta|)` per row, and
    /// symmetrizes; families with closed-form second derivatives override it.
    fn neg_hessian(&self, y: &[f64], etas: &[Array1<f64>], out: &mut [f64]) {
        let k = self.n_params();
        let n = etas[0].len();
        let ones = vec![1.0; n];
        let mut plus = vec![0.0; n];
        let mut minus = vec![0.0; n];
        for b in 0..k {
            let step = etas[b].mapv(|e| 1e-5 * (1.0 + e.abs()));
            let mut up = etas.to_vec();
            let mut down = etas.to_vec();
            up[b] = &etas[b] + &step;
            down[b] = &etas[b] - &step;
            for a in 0..k {
                self.ngradient(a, y, &up, &ones, &mut plus);
                self.ngradient(a, y, &down, &ones, &mut minus);
                for i in 0..n {
                    out[i * k * k + a * k + b] = -(plus[i] - minus[i]) / (2.0 * step[i]);
                }
            }
        }
        for i in 0..n {
            for a in 0..k {
                for b in (a + 1)..k {
                    let m = 0.5 * (out[i * k * k + a * k + b] + out[i * k * k + b * k + a]);
                    out[i * k * k + a * k + b] = m;
                    out[i * k * k + b * k + a] = m;
                }
            }
        }
    }
}

/// Result of a cyclic LSS fit: one [`BoostingPath`] per parameter (each with its
/// Cycle-count → per-parameter eta snapshot from the diagonal-checkpoint LSS
/// fit. Returned by [`try_fit_lss_diagonal_checkpoints`] (and its panicking
/// compatibility wrapper [`fit_lss_diagonal_checkpoints`]).
pub type LssSnapshots = Vec<(usize, Vec<Array1<f64>>)>;

/// Full per-distribution-parameter boosting result of an LSS fit: one
/// [`BoostingPath`] per parameter (each carrying its own offset, selections,
/// coefficients, and final `eta`; per-step increments are not stored —
/// predictions reconstruct from the coefficients), plus the joint risk path
/// recorded after each full cycle (length `mstop + 1`).
#[derive(Clone, Debug)]
pub struct LssPath {
    /// One boosting path per distribution parameter, aligned with
    /// `param_names()` order.
    pub per_param: Vec<BoostingPath>,
    /// Joint risk after each full cycle, length `mstop + 1`.
    pub risk: Vec<f64>,
    /// Global per-step parameter-update order: `selection_order[s]` is the
    /// parameter updated at global step `s`. For cyclic fits this is the
    /// deterministic round-robin `0,1,..,K-1` repeated; for non-cyclic it is the
    /// risk-driven sequence. Combined with each parameter's `selected` list it
    /// gives the interleaved base-learner selection order (used by stabsel).
    pub selection_order: Vec<usize>,
}

impl LssPath {
    /// Aggregate per-parameter metadata into one model-level execution report.
    ///
    /// LSS iteration counts are expressed as total parameter updates. Thus a
    /// two-parameter cyclic fit with ten updates per parameter reports twenty,
    /// while a ten-step non-cyclic fit reports ten.
    pub fn execution_report(&self) -> FitReport {
        let Some(first) = self.per_param.first() else {
            return FitReport::default();
        };
        FitReport {
            instrumentation: first.report.instrumentation,
            route: first.report.route,
            n_observations: first.report.n_observations,
            n_learners: self
                .per_param
                .iter()
                .map(|path| path.report.n_learners)
                .sum(),
            total_basis_columns: self
                .per_param
                .iter()
                .map(|path| path.report.total_basis_columns)
                .sum(),
            requested_mstop: if first.report.route == EngineRoute::LssNoncyclic {
                first.report.requested_mstop
            } else {
                self.per_param
                    .iter()
                    .map(|path| path.report.requested_mstop)
                    .sum()
            },
            completed_mstop: self.selection_order.len(),
            effective_threads: first.report.effective_threads,
            gram_eligible: false,
            fallback_reasons: Vec::new(),
            scalar_learner_count: self
                .per_param
                .iter()
                .map(|path| path.report.scalar_learner_count)
                .sum(),
            block_learner_count: self
                .per_param
                .iter()
                .map(|path| path.report.block_learner_count)
                .sum(),
            tree_learner_count: self
                .per_param
                .iter()
                .map(|path| path.report.tree_learner_count)
                .sum(),
            cholesky_solver_count: self
                .per_param
                .iter()
                .map(|path| path.report.cholesky_solver_count)
                .sum(),
            lu_solver_count: self
                .per_param
                .iter()
                .map(|path| path.report.lu_solver_count)
                .sum(),
            regularized_solver_count: self
                .per_param
                .iter()
                .map(|path| path.report.regularized_solver_count)
                .sum(),
            max_solver_jitter: self
                .per_param
                .iter()
                .filter_map(|path| path.report.max_solver_jitter)
                .reduce(f64::max),
            factor_cache_bytes: self
                .per_param
                .iter()
                .map(|path| path.report.factor_cache_bytes)
                .sum(),
            gram_cache_bytes: self
                .per_param
                .iter()
                .map(|path| path.report.gram_cache_bytes)
                .max()
                .unwrap_or(0),
            // Summed, not maxed: an LSS fit prepares the candidate set once per
            // distribution parameter, so a three-parameter Student-t fit holds
            // three copies of every spline design at the same time.
            retained_learner_bytes: self
                .per_param
                .iter()
                .map(|path| path.report.retained_learner_bytes)
                .sum(),
            calibration_count: self
                .per_param
                .iter()
                .map(|path| path.report.calibration_count)
                .sum(),
            tie_selection_count: self
                .per_param
                .iter()
                .map(|path| path.report.tie_selection_count)
                .sum(),
            calibration_converged_count: self
                .per_param
                .iter()
                .map(|path| path.report.calibration_converged_count)
                .sum(),
            calibration_limit_count: self
                .per_param
                .iter()
                .map(|path| path.report.calibration_limit_count)
                .sum(),
            calibration_failure_count: self
                .per_param
                .iter()
                .map(|path| path.report.calibration_failure_count)
                .sum(),
            nonfinite_candidate_fallbacks: self
                .per_param
                .iter()
                .map(|path| path.report.nonfinite_candidate_fallbacks)
                .sum(),
            setup_ns: first.report.setup_ns,
            basis_ns: first.report.basis_ns,
            factorization_ns: first.report.factorization_ns,
            cross_gram_ns: first.report.cross_gram_ns,
            iteration_ns: first.report.iteration_ns,
            reconstruction_ns: first.report.reconstruction_ns,
            binding_ns: first.report.binding_ns,
            total_ns: first.report.total_ns,
            ..FitReport::default()
        }
    }

    /// Derive global LSS update rows in their original parameter-update order.
    pub fn path_rows(&self) -> Vec<PathRow> {
        let noncyclic = self
            .per_param
            .first()
            .is_some_and(|path| path.report.route == EngineRoute::LssNoncyclic);
        let mut cursors = vec![0usize; self.per_param.len()];
        self.selection_order
            .iter()
            .enumerate()
            .map(|(step, &parameter)| {
                let local_step = cursors[parameter];
                cursors[parameter] += 1;
                let cycle = local_step + 1;
                let cycle_ends = self
                    .selection_order
                    .get(step + 1)
                    .is_none_or(|&next_parameter| next_parameter <= parameter);
                let risk_index = if noncyclic {
                    Some(step + 1)
                } else if cycle_ends {
                    Some(cycle)
                } else {
                    None
                };
                let risk = risk_index
                    .and_then(|index| self.risk.get(index))
                    .copied()
                    .filter(|value| value.is_finite());
                let previous_risk = risk_index
                    .and_then(|index| index.checked_sub(1))
                    .and_then(|index| self.risk.get(index))
                    .copied()
                    .filter(|value| value.is_finite());
                let coefficient_norm = self.per_param[parameter]
                    .coefs
                    .get(local_step)
                    .map(|coef| coef.iter().map(|value| value * value).sum::<f64>().sqrt())
                    .unwrap_or(0.0);
                PathRow {
                    iteration: step + 1,
                    cycle: (!noncyclic).then_some(cycle),
                    parameter_index: Some(parameter),
                    learner_index: self.per_param[parameter].selected[local_step],
                    risk,
                    risk_change: previous_risk
                        .zip(risk)
                        .map(|(before, after)| before - after),
                    coefficient_norm,
                    nuisance: Vec::new(),
                }
            })
            .collect()
    }
}

/// Cyclic component-wise gradient boosting for a GAMLSS family.
///
/// `learners_per_param[k]` is the base-learner set for parameter `k`. Each
/// parameter gets `control.mstop` boosting steps (one per cycle). `offsets`, if
/// given, overrides the family's computed per-parameter offsets.
pub fn fit_lss(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    mstop_per_param: Option<&[usize]>,
    offsets: Option<&[f64]>,
) -> LssPath {
    try_fit_lss(
        learners_per_param,
        y,
        weights,
        family,
        control,
        mstop_per_param,
        offsets,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// Validate the inputs both LSS drivers accept, in one place.
///
/// `try_fit_lss` and `try_fit_lss_noncyclic` take the same arguments and impose
/// the same requirements on them, and each carried a verbatim copy of this
/// block. `cv.rs` sets the precedent one file over: `validate_cvrisk_inputs` is
/// called from both its panicking and its fallible entry point. Two copies of a
/// validation block is where the third copy comes from — the crate already
/// documents what that costs, in `Family::encode_response`'s note about the
/// `cvrisk` path being missed when each binding recoded `y` itself.
///
/// `n` and `k` are passed in rather than recomputed because the observation
/// count comes from the weights, not from `y`: survival families carry a
/// two-column response, so `y.len()` can be a multiple of `n`.
#[allow(clippy::too_many_arguments)]
fn validate_lss_inputs(
    learners_per_param: &[Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    ys: &[f64],
    ws: &[f64],
    family: &dyn LssFamily,
    offsets: Option<&[f64]>,
    n: usize,
    k: usize,
) -> Result<(), MboostrsError> {
    if n == 0 {
        return Err(MboostrsError::Shape(
            "LSS fit requires at least one observation".to_string(),
        ));
    }
    if ys.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::BadResponse(
            family.name().to_string(),
            "response contains NaN or infinite values".to_string(),
        ));
    }
    if ws.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::Shape(
            "LSS weights contain NaN or infinite values".to_string(),
        ));
    }
    if ws.iter().any(|&value| value < 0.0) {
        return Err(MboostrsError::Shape(
            "LSS weights must be non-negative".to_string(),
        ));
    }
    family.check_y(y)?;
    if learners_per_param.len() != k {
        return Err(MboostrsError::Shape(format!(
            "expected {k} learner sets (one per distribution parameter), got {}",
            learners_per_param.len()
        )));
    }
    for (p, learners) in learners_per_param.iter().enumerate() {
        if learners.is_empty() {
            return Err(MboostrsError::Shape(format!(
                "LSS parameter {} requires at least one base learner",
                family.param_names().get(p).copied().unwrap_or("?")
            )));
        }
        for learner in learners {
            let learner_n = learner.n_observations();
            if learner_n != n {
                return Err(MboostrsError::LearnerSetup(
                    learner.name().to_string(),
                    format!("design has {learner_n} rows but LSS fit has {n} observations"),
                ));
            }
        }
    }
    if let Some(values) = offsets {
        if values.len() != k {
            return Err(MboostrsError::Shape(format!(
                "offsets must have one entry per parameter; expected {k}, got {}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::Control(
                "LSS offsets must be finite".to_string(),
            ));
        }
    }
    Ok(())
}

/// The factor by which a cycle's joint risk may exceed its running minimum
/// before the LSS fit is declared divergent.
///
/// The increment is measured as
/// `LSS_DIVERGENCE_FACTOR * (|running_minimum| + 1)` against
/// `step_risk - running_minimum`: an absolute increment, because LSS risks
/// are signed (the Gaussian NLL goes negative once sigma^2 drops below
/// e^-1, and a ratio test on a negative minimum would fire on every
/// ordinary decrease). The constant is the observable policy: unit tests
/// pin recoverable excursions against it, and the fire/no-fire boundary is
/// part of the public behaviour of `check_lss_cycle_risk`.
pub const LSS_DIVERGENCE_FACTOR: f64 = 1e6;

/// Reject an LSS fit whose joint risk exploded relative to the best seen.
///
/// Component-wise boosting fits each base learner by least squares on the
/// negative gradient, so the effective step length scales with the gradient
/// magnitude. Location equations whose gradient carries a precision factor
/// are the hazard: GaussianLSS moves `mu` along `(y - mu) / sigma^2`, so a
/// response with standard deviation far below 1 turns one cycle's step into
/// `nu / sigma^2` times the least-squares increment -- orders of magnitude
/// past the solution -- and the joint risk grows geometrically from there
/// (observed as 1e8 -> 1e213 within tens of cycles).
///
/// Brief risk excursions are normal at aggressive step lengths (`nu` up to
/// 0.4): the sigma equation overshoots downward and recalibrates within a
/// few cycles, spiking the joint risk by factors of a few before it
/// recovers, so those paths are left alone. The guard fires when the risk
/// climbs more than [`LSS_DIVERGENCE_FACTOR`] times `|minimum| + 1` above
/// its running minimum. No recoverable excursion reaches that increment,
/// while a precision-driven explosion passes it within a cycle or two. A
/// fit whose risk is a million-fold off its best value produces meaningless
/// output even if it later recovers. Non-finite risks always stop the fit.
fn check_lss_cycle_risk(
    step_risk: f64,
    best_risk: &mut f64,
    cycle: usize,
    method: &str,
) -> Result<(), MboostrsError> {
    if !step_risk.is_finite() {
        return Err(MboostrsError::Solver(format!(
            "{method} LSS risk became non-finite after cycle {}. An update \
             pushed a predictor far enough that the risk overflowed; reduce \
             `nu`, or rescale the response so its standard deviation is of \
             order 1.",
            cycle + 1
        )));
    }
    let divergence_increment = LSS_DIVERGENCE_FACTOR * (best_risk.abs() + 1.0);
    if best_risk.is_finite() && step_risk - *best_risk > divergence_increment {
        return Err(MboostrsError::Solver(format!(
            "{method} LSS risk diverged after cycle {}: {:.6e} against a \
             running minimum of {:.6e}. Component-wise steps are proportional \
             to the gradient, and this response's scale makes them overshoot: \
             GaussianLSS, for example, updates `mu` along `(y - mu)/sigma^2`, \
             so a response with sd << 1 multiplies every step by 1/sigma^2. \
             Express the response in units where its standard deviation is of \
             order 1 (per cent rather than decimal fractions for returns), or \
             lower `nu`.",
            cycle + 1,
            step_risk,
            *best_risk
        )));
    }
    if step_risk < *best_risk {
        *best_risk = step_risk;
    }
    Ok(())
}

/// Resolve the per-parameter learning rates a distributional fit will use.
///
/// [`BoostControl::nu_per_param`], when set, overrides the scalar `nu` for
/// each parameter's updates; `None` recycles `nu` for every parameter --
/// `gamboostLSS`'s `check()` does the same recycling before the fit sees
/// the value. Every entry must satisfy `0 < nu <= 1` and there must be one
/// entry per parameter, in `param_names()` order.
pub fn resolve_nu_per_param(
    control: &BoostControl,
    n_params: usize,
) -> Result<Vec<f64>, MboostrsError> {
    match &control.nu_per_param {
        Some(nus) => {
            if nus.len() != n_params {
                return Err(MboostrsError::Control(format!(
                    "nu_per_param must have one entry per parameter; expected {n_params}, got {}",
                    nus.len()
                )));
            }
            if let Some(bad) = nus.iter().find(|&&v| !v.is_finite() || v <= 0.0 || v > 1.0) {
                return Err(MboostrsError::Control(format!(
                    "every per-parameter nu must satisfy 0 < nu <= 1, got {bad}"
                )));
            }
            Ok(nus.clone())
        }
        None => Ok(vec![control.nu; n_params]),
    }
}

/// The multiple `rho` of parameter `p`'s winning update that minimizes the
/// joint risk, for the `LineSearch` step rule on a cyclic LSS fit.
///
/// Every other parameter stays at its current value, so this is a
/// one-dimensional search on the joint likelihood along the winning update's
/// direction — the same rule the standard engine applies, read through the
/// coupled loss a distributional fit actually minimizes. `etas[p]` is used as
/// the evaluation scratch and restored before returning, so the caller
/// applies its update to the pre-search value. Non-finite joint risks count
/// as positive infinity, so the search backs away from overflow.
fn lss_line_search_rho(
    family: &dyn LssFamily,
    ys: &[f64],
    etas: &mut [Array1<f64>],
    p: usize,
    h: &[f64],
    ws: &[f64],
    saved: &mut [f64],
    max_iter: usize,
) -> f64 {
    let n = h.len();
    saved.copy_from_slice(etas[p].as_slice().expect("eta contiguous"));
    let rho = golden_grid_min(
        |rho| {
            {
                let ep = etas[p].as_slice_mut().expect("eta contiguous");
                for i in 0..n {
                    ep[i] = saved[i] + rho * h[i];
                }
            }
            let risk = family.risk(ys, etas, ws);
            if risk.is_finite() {
                risk
            } else {
                f64::INFINITY
            }
        },
        max_iter,
    );
    // Restore the pre-search values: the caller's update applies to them.
    let ep = etas[p].as_slice_mut().expect("eta contiguous");
    ep.copy_from_slice(&saved[..n]);
    rho
}

/// Fallible cyclic LSS fitting with typed validation errors.
pub fn try_fit_lss(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    mstop_per_param: Option<&[usize]>,
    offsets: Option<&[f64]>,
) -> Result<LssPath, MboostrsError> {
    let timing_enabled = control.instrumentation != InstrumentationLevel::Off;
    let fit_started = timing_enabled.then(Instant::now);
    let ys = y
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS response must be contiguous".to_string()))?;
    let ws = weights
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS weights must be contiguous".to_string()))?;
    // Observation count comes from the weights (one per observation), not from
    // `y`: survival families carry a 2-column response (time, status), so
    // `y.len()` can be a multiple of the observation count.
    let n = ws.len();
    let k = family.n_params();
    let nus = resolve_nu_per_param(control, k)?;
    validate_lss_inputs(learners_per_param, y, ys, ws, family, offsets, n, k)?;
    let (lss_line_search, ls_max_iter) = match &control.step {
        StepRule::Fixed => (false, 0usize),
        StepRule::LineSearch { max_iter } => (true, *max_iter),
    };
    let mut ls_saved: Vec<f64> = if lss_line_search {
        vec![0.0f64; n]
    } else {
        Vec::new()
    };
    // Per-parameter step budget (R's mstop vector). Cyclic: run max(mstops)
    // cycles; parameter p only updates while its own budget is not exhausted,
    // so parameter p ends with exactly mstops[p] boosting steps.
    let mstops: Vec<usize> = match mstop_per_param {
        Some(m) => {
            if m.len() != k {
                return Err(MboostrsError::Shape(format!(
                    "mstop_per_param must have one entry per parameter; expected {k}, got {}",
                    m.len()
                )));
            }
            m.to_vec()
        }
        None => vec![control.mstop as usize; k],
    };
    let max_m = mstops.iter().copied().max().unwrap_or(0);

    // Initialise each predictor to its offset.
    let mut offs: Vec<f64> = Vec::with_capacity(k);
    let mut etas: Vec<Array1<f64>> = Vec::with_capacity(k);
    for p in 0..k {
        let off = offsets
            .map(|o| o[p])
            .unwrap_or_else(|| family.offset(p, ys, ws, &offs));
        if !off.is_finite() {
            return Err(MboostrsError::Solver(format!(
                "LSS offset for parameter '{}' is non-finite",
                family.param_names().get(p).copied().unwrap_or("?")
            )));
        }
        offs.push(off);
        etas.push(Array1::from_elem(n, off));
    }

    let mut selected: Vec<Vec<usize>> = (0..k).map(|p| Vec::with_capacity(mstops[p])).collect();
    let mut coefs: Vec<Vec<Vec<f64>>> = (0..k).map(|p| Vec::with_capacity(mstops[p])).collect();

    let mut risk_path = Vec::with_capacity(max_m + 1);
    let initial_risk = family.risk(ys, &etas, ws);
    if !initial_risk.is_finite() {
        return Err(MboostrsError::Solver(
            "LSS initial risk is non-finite".to_string(),
        ));
    }
    risk_path.push(initial_risk);
    let setup_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
    let iteration_started = timing_enabled.then(Instant::now);

    let mut u = vec![0.0f64; n];
    let mut best_pred = vec![0.0f64; n];
    let mut best_beta: Vec<f64> = Vec::new();
    let mut selection_order: Vec<usize> = Vec::new();
    let mut tie_selection_counts = vec![0usize; k];
    let mut best_cycle_risk = initial_risk;
    // The joint risk after each parameter's own update, as gamboostLSS's
    // risk(merge = FALSE) reports it; the last parameter's entry in a cycle
    // is the cycle's risk.
    let mut param_risk: Vec<Vec<f64>> = (0..k)
        .map(|p| {
            let mut path = Vec::with_capacity(mstops[p] + 1);
            path.push(initial_risk);
            path
        })
        .collect();

    for m in 0..max_m {
        let mut step_risk = f64::NAN;
        for p in 0..k {
            // Skip parameter p once its own step budget is exhausted.
            if m >= mstops[p] {
                continue;
            }
            selection_order.push(p);
            // Partial pseudo-residual for parameter p (uses current etas of all).
            family.ngradient(p, ys, &etas, ws, &mut u);
            // Put the parameters' gradients on comparable footing before
            // selection, so one `nu` does not take a large step on one and a
            // negligible one on another. A no-op under the default.
            stabilize_ngradient(&mut u, ws, control.stabilization);
            let u_view = ArrayView1::from(u.as_slice());

            // Compute wu2 = sum(w_i * u_i^2) once for all learners (O(n)).
            let wu2: f64 = ws.iter().zip(u.iter()).map(|(&w, &u)| w * u * u).sum();

            // Component-wise selection via SSE-by-identity (O(p^2) per learner
            // instead of O(n)). Falls back to explicit fit_step_ref if any
            // learner produces NaN SSE (singular Gram).
            let mut best_sse = f64::INFINITY;
            let mut best_idx = 0usize;
            let mut found = false;
            let mut sse_is_nan = false;
            let mut beta_len = 0usize;
            let mut best_multiplicity = 0usize;
            for (li, learner) in learners_per_param[p].iter_mut().enumerate() {
                let (beta, sse) = learner.fit_step_select(u_view, weights, wu2);
                if li == 0 {
                    beta_len = beta.len();
                }
                if sse.is_nan() {
                    sse_is_nan = true;
                    break;
                }
                if observe_candidate(
                    li,
                    sse,
                    &mut best_idx,
                    &mut best_sse,
                    &mut best_multiplicity,
                ) {
                    found = true;
                    best_beta.clear();
                    best_beta.extend_from_slice(beta);
                }
            }

            if sse_is_nan {
                // Fallback: use explicit fit_step_ref (computes pred + SSE
                // directly), mirroring the single-param engine's all-block
                // fallback for singular Grams.
                best_beta.clear();
                best_sse = f64::INFINITY;
                best_idx = 0;
                found = false;
                best_multiplicity = 0;
                for (li, learner) in learners_per_param[p].iter_mut().enumerate() {
                    let (beta, pred, sse) = learner.fit_step_ref(u_view, weights);
                    if li == 0 {
                        beta_len = beta.len();
                    }
                    if observe_candidate(
                        li,
                        sse,
                        &mut best_idx,
                        &mut best_sse,
                        &mut best_multiplicity,
                    ) {
                        found = true;
                        best_beta.clear();
                        best_beta.extend_from_slice(beta);
                        best_pred.copy_from_slice(pred);
                    }
                }
            } else if found {
                // Compute prediction for the winner only (O(n) once).
                learners_per_param[p][best_idx].predict_into(&best_beta, &mut best_pred);
                // Btree fallback: copy from its internal pred_buf.
                let pb = learners_per_param[p][best_idx].pred_buf();
                if !pb.is_empty() && pb.len() == n {
                    best_pred.copy_from_slice(pb);
                }
            }

            if !found {
                // Every learner produced a non-finite SSE (degenerate gradient
                // or singular fold). Record an explicit no-op step instead of
                // silently reusing the previous iteration's best_beta/best_pred
                // buffers (stale data), mirroring the engine's tolerance of
                // NaN SSEs as no-op selections.
                best_beta.clear();
                best_beta.resize(beta_len, 0.0);
                best_pred.fill(0.0);
                best_idx = 0;
            }
            tie_selection_counts[p] += usize::from(found && best_multiplicity > 1);

            // Update eta_p += nu * (line-search multiple of) the winning
            // learner's prediction. The stored coefficient carries rho, so
            // every replay path applies the same effective step.
            let rho = if lss_line_search {
                lss_line_search_rho(
                    family,
                    ys,
                    &mut etas,
                    p,
                    &best_pred,
                    ws,
                    &mut ls_saved,
                    ls_max_iter,
                )
            } else {
                1.0
            };
            let ep = etas[p].as_slice_mut().expect("eta contiguous");
            for i in 0..n {
                ep[i] += nus[p] * rho * best_pred[i];
            }
            selected[p].push(best_idx);
            // A tree's coefficient encodes its structure: scale leaf means only.
            coefs[p].push(learners_per_param[p][best_idx].scale_coef(&best_beta, rho));
            step_risk = family.risk(ys, &etas, ws);
            param_risk[p].push(step_risk);
        }
        check_lss_cycle_risk(step_risk, &mut best_cycle_risk, m, "cyclic")?;
        risk_path.push(step_risk);
    }

    let iteration_ns = iteration_started.map(|started| started.elapsed().as_nanos() as u64);
    let total_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
    let mut per_param = Vec::with_capacity(k);
    for p in 0..k {
        let learner_stats = crate::learners::aggregate_report_stats(&learners_per_param[p]);
        per_param.push(BoostingPath {
            report: FitReport {
                instrumentation: control.instrumentation,
                route: EngineRoute::LssCyclic,
                n_observations: n,
                n_learners: learners_per_param[p].len(),
                total_basis_columns: learners_per_param[p]
                    .iter()
                    .map(PreparedLearner::n_parameters)
                    .sum(),
                requested_mstop: mstops[p],
                completed_mstop: selected[p].len(),
                scalar_learner_count: learner_stats.scalar_learners,
                block_learner_count: learner_stats.block_learners,
                tree_learner_count: learner_stats.tree_learners,
                cholesky_solver_count: learner_stats.cholesky_solvers,
                lu_solver_count: learner_stats.lu_solvers,
                regularized_solver_count: learner_stats.regularized_solvers,
                max_solver_jitter: learner_stats.max_solver_jitter,
                factor_cache_bytes: learner_stats.factor_cache_bytes,
                retained_learner_bytes: learner_stats.retained_data_bytes,
                tie_selection_count: tie_selection_counts[p],
                setup_ns,
                iteration_ns,
                total_ns,
                ..FitReport::default()
            },
            offset: offs[p],
            selected: std::mem::take(&mut selected[p]),
            coefs: std::mem::take(&mut coefs[p]),
            // The joint risk after each of this parameter's updates (entry 0
            // is the offset-only model); LssPath.risk is the per-cycle path.
            risk: std::mem::take(&mut param_risk[p]),
            fitted: etas[p].clone(),
            // Per-step increments are not stored: no binding consumes them for
            // LSS (predict/eta reconstruct from coefs), and LSS CV calls this
            // per fold × per grid point — storing them cost O(B·|grid|·mstop·n)
            // allocations for nothing (LSS4).
            increments: Vec::new(),
            nuisance_path: Vec::new(),
            nuisance_vec_path: Vec::new(),
            nuisance_vec: Vec::new(),
            // A cyclic LSS path is not resumable per parameter: freezing one
            // changes the later updates of the others, so there is no single
            // residual a continuation could pick up from.
            residual: Array1::zeros(0),
        });
    }

    Ok(LssPath {
        per_param,
        risk: risk_path,
        selection_order,
    })
}

/// Cyclic gamboostLSS fit that records eta snapshots at a set of "checkpoint"
/// cycle counts instead of returning only the final state.
///
/// For a *diagonal* CV grid (every parameter sharing the same per-step
/// budget `m`, i.e. grid points `(m, m, ..., m)`), a fit run to
/// `max(checkpoints)` cycles passes through, after cycle `m`, EXACTLY the
/// state a fresh [`fit_lss`] call with `mstop_per_param = [m; k]` would
/// produce: with equal budgets, [`fit_lss`]'s per-parameter skip
/// (`m >= mstops[p]`) never triggers before `max(checkpoints)`, so every
/// parameter updates on every cycle regardless of which checkpoint the grid
/// happens to land on. One pass over `0..max(checkpoints)` therefore
/// replaces one [`fit_lss`] call per diagonal grid point. Non-diagonal grid
/// points (unequal per-parameter budgets) do NOT have this prefix property
/// — one parameter's cap binding early changes the partial gradients the
/// others see afterward — so callers must keep using [`fit_lss`] for those.
/// Compatibility wrapper around [`try_fit_lss_diagonal_checkpoints`]; panics
/// on a malformed learner layout or non-contiguous inputs and is deprecated on
/// the P0.8 schedule.
#[allow(unused_assignments)]
// best_idx resets mirror fit_lss's structure; some are dead here since this function doesn't track `selected`
#[deprecated(
    since = "0.1.3",
    note = "use try_fit_lss_diagonal_checkpoints (returns MboostrsError instead of panicking); removed in 0.2.0"
)]
pub fn fit_lss_diagonal_checkpoints(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    checkpoints: &[usize],
    offsets: Option<&[f64]>,
) -> LssSnapshots {
    try_fit_lss_diagonal_checkpoints(
        learners_per_param,
        y,
        weights,
        family,
        control,
        checkpoints,
        offsets,
    )
    .expect("fit_lss_diagonal_checkpoints: valid inputs")
}

/// Fallible [`fit_lss_diagonal_checkpoints`]: never panics.
///
/// Returns `Err(MboostrsError::LearnerSetup)` when `learners_per_param.len()`
/// does not match `family.n_params()`, and `Err(MboostrsError::Shape)` when the
/// response or weight view is non-contiguous.
pub fn try_fit_lss_diagonal_checkpoints(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    checkpoints: &[usize],
    offsets: Option<&[f64]>,
) -> Result<LssSnapshots, MboostrsError> {
    let ys = match y.as_slice() {
        Some(ys) => ys,
        None => {
            return Err(MboostrsError::Shape(
                "fit_lss_diagonal_checkpoints: y must be contiguous".to_string(),
            ))
        }
    };
    let ws = match weights.as_slice() {
        Some(ws) => ws,
        None => {
            return Err(MboostrsError::Shape(
                "fit_lss_diagonal_checkpoints: weights must be contiguous".to_string(),
            ))
        }
    };
    let k = family.n_params();
    let mut path = LssDiagonalPath::start(family, control, ys, ws, offsets)?;
    if learners_per_param.len() != k {
        return Err(MboostrsError::LearnerSetup(
            "fit_lss_diagonal_checkpoints".to_string(),
            format!(
                "expected one learner set per distribution parameter, got {} learner sets \
                 for {} parameters",
                learners_per_param.len(),
                k
            ),
        ));
    }

    let max_m = checkpoints.iter().copied().max().unwrap_or(0);
    let checkpoint_set: std::collections::HashSet<usize> = checkpoints.iter().copied().collect();

    let mut snapshots: LssSnapshots = Vec::with_capacity(checkpoints.len());
    if checkpoint_set.contains(&0) {
        snapshots.push((0, path.etas.clone()));
    }
    path.advance(
        learners_per_param,
        ys,
        ws,
        weights,
        family,
        control.stabilization,
        max_m,
        |cycles_done, etas| {
            if checkpoint_set.contains(&cycles_done) {
                snapshots.push((cycles_done, etas.to_vec()));
            }
        },
    )?;

    Ok(snapshots)
}

/// A diagonal LSS fit (every parameter updated once per cycle) that can be
/// stopped after any number of cycles and resumed.
///
/// The state is the linear predictor of every parameter and the cycle count.
/// A cycle reads only that state and the learners, so running cycles `0..m`,
/// stopping, and running `m..M` gives the same predictors as one run of
/// `0..M`, to the last bit: the arithmetic of each cycle is unchanged. The
/// learners are reused, so the caller must pass the same prepared learners
/// (built against the same weights) on every call.
pub(crate) struct LssDiagonalPath {
    /// Linear predictor of each parameter after `cycles` cycles.
    pub(crate) etas: Vec<Array1<f64>>,
    /// Completed cycles.
    pub(crate) cycles: usize,
    nus: Vec<f64>,
    lss_line_search: bool,
    ls_max_iter: usize,
    /// Scratch: the negative gradient of the parameter being updated.
    u: Vec<f64>,
    /// Scratch: the fitted update of the selected learner.
    best_pred: Vec<f64>,
    /// Scratch: the coefficients of the selected learner.
    best_beta: Vec<f64>,
    /// Scratch: the line search's saved predictor.
    ls_saved: Vec<f64>,
}

impl LssDiagonalPath {
    /// The path before its first cycle: every predictor at its offset.
    ///
    /// `offsets` (one per parameter) replaces the family's offsets when given.
    pub(crate) fn start(
        family: &dyn LssFamily,
        control: &BoostControl,
        ys: &[f64],
        ws: &[f64],
        offsets: Option<&[f64]>,
    ) -> Result<Self, MboostrsError> {
        let n = ws.len();
        let k = family.n_params();
        let nus = resolve_nu_per_param(control, k)?;
        let (lss_line_search, ls_max_iter) = match &control.step {
            StepRule::Fixed => (false, 0usize),
            StepRule::LineSearch { max_iter } => (true, *max_iter),
        };
        let mut offs: Vec<f64> = Vec::with_capacity(k);
        let mut etas: Vec<Array1<f64>> = Vec::with_capacity(k);
        for p in 0..k {
            let off = offsets
                .map(|o| o[p])
                .unwrap_or_else(|| family.offset(p, ys, ws, &offs));
            offs.push(off);
            etas.push(Array1::from_elem(n, off));
        }
        Ok(Self {
            etas,
            cycles: 0,
            nus,
            lss_line_search,
            ls_max_iter,
            u: vec![0.0f64; n],
            best_pred: vec![0.0f64; n],
            best_beta: Vec::new(),
            ls_saved: if lss_line_search {
                vec![0.0f64; n]
            } else {
                Vec::new()
            },
        })
    }

    /// Run cycles until `to` are complete, calling `after_cycle(m, etas)` once
    /// each time a cycle `m` completes. Does nothing when `cycles >= to`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn advance<F: FnMut(usize, &[Array1<f64>])>(
        &mut self,
        learners_per_param: &mut [Vec<PreparedLearner>],
        ys: &[f64],
        ws: &[f64],
        weights: ArrayView1<f64>,
        family: &dyn LssFamily,
        stabilization: Stabilization,
        to: usize,
        mut after_cycle: F,
    ) -> Result<(), MboostrsError> {
        while self.cycles < to {
            self.cycle(learners_per_param, ys, ws, weights, family, stabilization);
            self.cycles += 1;
            after_cycle(self.cycles, &self.etas);
        }
        Ok(())
    }

    /// One cycle: each parameter takes its best learner's step in turn.
    // The `best_idx` resets keep the selection loop's structure readable.
    #[allow(unused_assignments)]
    fn cycle(
        &mut self,
        learners_per_param: &mut [Vec<PreparedLearner>],
        ys: &[f64],
        ws: &[f64],
        weights: ArrayView1<f64>,
        family: &dyn LssFamily,
        stabilization: Stabilization,
    ) {
        let n = ws.len();
        let k = self.etas.len();
        let Self {
            etas,
            nus,
            lss_line_search,
            ls_max_iter,
            u,
            best_pred,
            best_beta,
            ls_saved,
            ..
        } = self;
        for p in 0..k {
            family.ngradient(p, ys, etas.as_slice(), ws, u.as_mut_slice());
            // Put the parameters' gradients on comparable footing before
            // selection, so one `nu` does not take a large step on one and a
            // negligible one on another. A no-op under the default.
            stabilize_ngradient(u.as_mut_slice(), ws, stabilization);
            let u_view = ArrayView1::from(u.as_slice());
            let wu2: f64 = ws.iter().zip(u.iter()).map(|(&w, &u)| w * u * u).sum();

            let mut best_sse = f64::INFINITY;
            let mut best_idx = 0usize;
            let mut found = false;
            let mut sse_is_nan = false;
            let mut beta_len = 0usize;
            for (li, learner) in learners_per_param[p].iter_mut().enumerate() {
                let (beta, sse) = learner.fit_step_select(u_view, weights, wu2);
                if li == 0 {
                    beta_len = beta.len();
                }
                if sse.is_nan() {
                    sse_is_nan = true;
                    break;
                }
                if sse < best_sse {
                    best_sse = sse;
                    best_idx = li;
                    found = true;
                    best_beta.clear();
                    best_beta.extend_from_slice(beta);
                }
            }

            if sse_is_nan {
                best_beta.clear();
                best_sse = f64::INFINITY;
                best_idx = 0;
                found = false;
                for (li, learner) in learners_per_param[p].iter_mut().enumerate() {
                    let (beta, pred, sse) = learner.fit_step_ref(u_view, weights);
                    if li == 0 {
                        beta_len = beta.len();
                    }
                    if sse < best_sse {
                        best_sse = sse;
                        best_idx = li;
                        found = true;
                        best_beta.clear();
                        best_beta.extend_from_slice(beta);
                        best_pred.copy_from_slice(pred);
                    }
                }
            } else if found {
                learners_per_param[p][best_idx].predict_into(best_beta, best_pred);
                let pb = learners_per_param[p][best_idx].pred_buf();
                if !pb.is_empty() && pb.len() == n {
                    best_pred.copy_from_slice(pb);
                }
            }

            if !found {
                best_beta.clear();
                best_beta.resize(beta_len, 0.0);
                best_pred.fill(0.0);
            }

            let rho = if *lss_line_search {
                lss_line_search_rho(
                    family,
                    ys,
                    etas.as_mut_slice(),
                    p,
                    best_pred.as_slice(),
                    ws,
                    ls_saved.as_mut_slice(),
                    *ls_max_iter,
                )
            } else {
                1.0
            };
            let ep = etas[p].as_slice_mut().expect("eta contiguous");
            for i in 0..n {
                ep[i] += nus[p] * rho * best_pred[i];
            }
        }
    }
}

/// Shape checks of a cross-validation over LSS folds: at least one
/// observation and fold, a fold matrix with one row per observation, and a
/// judge mask shaped like the folds. Returns the fold count.
fn check_lss_cv_shapes(
    n: usize,
    folds: ndarray::ArrayView2<f64>,
    judge: Option<&ndarray::Array2<bool>>,
) -> Result<usize, MboostrsError> {
    let b = folds.ncols();
    if n == 0 || b == 0 {
        return Err(MboostrsError::Shape(
            "LSS cross-validation requires observations and folds".to_string(),
        ));
    }
    if folds.nrows() != n {
        return Err(MboostrsError::Shape(format!(
            "fold matrix has {} rows but weights have {n}",
            folds.nrows()
        )));
    }
    if let Some(mask) = judge {
        if mask.shape() != folds.shape() {
            return Err(MboostrsError::Shape(format!(
                "judge mask has shape {:?}, expected {:?} (the fold matrix)",
                mask.shape(),
                folds.shape()
            )));
        }
    }
    Ok(b)
}

/// Value checks of a cross-validation over LSS folds: a finite response,
/// finite non-negative weights and folds, then the family's response check.
fn check_lss_cv_values(
    y: ArrayView1<f64>,
    ys: &[f64],
    ws: &[f64],
    folds: ndarray::ArrayView2<f64>,
    family: &dyn LssFamily,
) -> Result<(), MboostrsError> {
    if ys.iter().any(|v| !v.is_finite())
        || ws.iter().any(|v| !v.is_finite() || *v < 0.0)
        || folds.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(MboostrsError::Shape(
            "LSS CV response, weights, and folds must be finite; weights must be non-negative"
                .to_string(),
        ));
    }
    family.check_y(y)
}

/// The out-of-bag risk of an LSS fold after the predictors `etas`, per unit of
/// out-of-bag weight, or `NaN` when `m` is not read by the schedule.
fn oob_risk(
    family: &dyn LssFamily,
    ys: &[f64],
    etas: &[Array1<f64>],
    oob_w: &[f64],
    oob_sum: f64,
    needed: &[bool],
    m: usize,
) -> f64 {
    if !needed.get(m).copied().unwrap_or(false) {
        return f64::NAN;
    }
    let r = family.risk(ys, etas, oob_w);
    if oob_sum > 0.0 {
        r / oob_sum
    } else {
        f64::NAN
    }
}

/// One fold of an LSS cross-validation search: its prepared learners, its
/// diagonal path, and the out-of-bag weights its risk is scored with.
struct LssFoldSearch {
    learners: Vec<Vec<PreparedLearner>>,
    fold_w: Array1<f64>,
    path: LssDiagonalPath,
    oob_w: Vec<f64>,
    oob_sum: f64,
    /// Whether iteration 0 has been recorded.
    started: bool,
}

impl LssFoldSearch {
    /// Advance the path to `end` cycles and return the risk at every iteration
    /// after the last call's `end` (iteration 0 on the first call).
    fn advance(
        &mut self,
        ys: &[f64],
        family: &dyn LssFamily,
        stabilization: Stabilization,
        needed: &[bool],
        end: usize,
    ) -> Result<Vec<f64>, MboostrsError> {
        let LssFoldSearch {
            learners,
            fold_w,
            path,
            oob_w,
            oob_sum,
            started,
        } = self;
        let oob: &[f64] = oob_w.as_slice();
        let oob_sum = *oob_sum;
        let mut fresh = Vec::new();
        if !*started {
            fresh.push(oob_risk(family, ys, &path.etas, oob, oob_sum, needed, 0));
            *started = true;
        }
        let ws = fold_w.as_slice().expect("fold weights are contiguous");
        path.advance(
            learners.as_mut_slice(),
            ys,
            ws,
            fold_w.view(),
            family,
            stabilization,
            end,
            |m, etas| fresh.push(oob_risk(family, ys, etas, oob, oob_sum, needed, m)),
        )?;
        Ok(fresh)
    }
}

/// The folds of an LSS cross-validation search, advanced together.
struct LssFoldsSearch<'a> {
    runs: Vec<LssFoldSearch>,
    ys: &'a [f64],
    family: &'a dyn LssFamily,
    stabilization: Stabilization,
    /// Iterations some grid of the schedule reads.
    needed: Vec<bool>,
}

impl FoldRisks for LssFoldsSearch<'_> {
    fn advance(&mut self, end: usize) -> Result<Vec<Vec<f64>>, MboostrsError> {
        use rayon::prelude::*;
        let (ys, family, stabilization) = (self.ys, self.family, self.stabilization);
        let needed = self.needed.as_slice();
        self.runs
            .par_iter_mut()
            .map(|run| run.advance(ys, family, stabilization, needed, end))
            .collect()
    }
}

/// Cross-validated LSS risk over a doubling grid search, resuming each fold.
///
/// Each fold is prepared against its in-bag weights and its diagonal path is
/// fitted once. The search reads the held-out risk at the iterations the
/// `schedule` grids name, advancing every path to the next grid's end only
/// when the current grid's optimum is in the boundary band (see
/// [`crate::grid_search::search_grid`]). Because a diagonal path's first `m`
/// cycles do not depend on how far it runs, the risks equal those of a fit
/// refitted from zero to each grid's end.
///
/// The risk for a fold is the out-of-bag negative log-likelihood per unit of
/// out-of-bag weight, as [`try_cvrisk_lss_with`] reports it.
#[allow(clippy::too_many_arguments)]
pub fn try_cvrisk_lss_search<F>(
    prepare: F,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    folds: ndarray::ArrayView2<f64>,
    judge: Option<&ndarray::Array2<bool>>,
    schedule: &[GridStep],
    boundary_share: f64,
) -> Result<GridSearch, MboostrsError>
where
    F: Fn(ArrayView1<f64>) -> Result<Vec<Vec<PreparedLearner>>, MboostrsError> + Sync,
{
    use rayon::prelude::*;

    let ys = y
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS response must be contiguous".to_string()))?;
    let ws = weights
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS weights must be contiguous".to_string()))?;
    let n = ws.len();
    let b = check_lss_cv_shapes(n, folds, judge)?;
    check_lss_cv_values(y, ys, ws, folds, family)?;
    let k = family.n_params();
    let max_end = schedule.iter().map(|step| step.end).max().unwrap_or(0);
    let mut needed = vec![false; max_end + 1];
    for step in schedule {
        for &g in &step.grid {
            if g <= max_end {
                needed[g] = true;
            }
        }
    }

    let runs = (0..b)
        .into_par_iter()
        .map(|fold| {
            let fold_w: Vec<f64> = (0..n).map(|i| folds[[i, fold]]).collect();
            let oob_w: Vec<f64> = (0..n)
                .map(|i| {
                    if fold_w[i] > 0.0 || judge.is_some_and(|mask| !mask[[i, fold]]) {
                        0.0
                    } else {
                        ws[i]
                    }
                })
                .collect();
            let oob_sum: f64 = oob_w.iter().sum();
            if fold_w.iter().all(|&value| value == 0.0) {
                return Err(MboostrsError::Shape(format!(
                    "fold {fold} has no in-bag observations"
                )));
            }
            if oob_sum <= 0.0 {
                return Err(MboostrsError::Shape(format!(
                    "fold {fold} has no positive-weight OOB observations"
                )));
            }
            let fold_w = Array1::from_vec(fold_w);
            let fold_ws = fold_w.as_slice().expect("fold weights are contiguous");
            let learners = prepare(fold_w.view())?;
            validate_lss_inputs(&learners, y, ys, fold_ws, family, None, n, k)?;
            let path = LssDiagonalPath::start(family, control, ys, fold_ws, None)?;
            Ok(LssFoldSearch {
                learners,
                fold_w,
                path,
                oob_w,
                oob_sum,
                started: false,
            })
        })
        .collect::<Result<Vec<_>, MboostrsError>>()?;

    let mut state = LssFoldsSearch {
        runs,
        ys,
        family,
        stabilization: control.stabilization,
        needed,
    };
    search_grid(&mut state, schedule, boundary_share)
}

/// Non-cyclic component-wise boosting for a GAMLSS family (R's
/// `method = "noncyclic"`, Thomas et al. 2018). At each of `mstop` steps, every
/// parameter's best base-learner update is computed, and only the single update
/// that most reduces the joint risk is applied. The per-parameter selection
/// counts therefore sum to `mstop`.
pub fn fit_lss_noncyclic(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    offsets: Option<&[f64]>,
) -> LssPath {
    try_fit_lss_noncyclic(learners_per_param, y, weights, family, control, offsets)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible non-cyclic LSS fitting with typed validation errors.
pub fn try_fit_lss_noncyclic(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    offsets: Option<&[f64]>,
) -> Result<LssPath, MboostrsError> {
    try_fit_lss_noncyclic_capped(
        learners_per_param,
        y,
        weights,
        family,
        control,
        offsets,
        None,
    )
}

/// The best base-learner update of one parameter for the negative gradient
/// `u`, written into `cand_idx`, `cand_beta` and `cand_pred`. Returns whether
/// the winning learner tied with another.
#[allow(clippy::too_many_arguments)]
fn noncyclic_candidate(
    learners: &mut [PreparedLearner],
    u: &[f64],
    ws: &[f64],
    weights: ArrayView1<f64>,
    n: usize,
    cand_idx: &mut usize,
    cand_beta: &mut Vec<f64>,
    cand_pred: &mut [f64],
) -> bool {
    let u_view = ArrayView1::from(u);
    // Compute wu2 once for all learners of this parameter (O(n)).
    let wu2: f64 = ws.iter().zip(u.iter()).map(|(&w, &u)| w * u * u).sum();

    // SSE-by-identity selection (O(p^2) per learner, not O(n)).
    let mut best_sse = f64::INFINITY;
    let mut found = false;
    let mut sse_is_nan = false;
    let mut beta_len = 0usize;
    let mut best_multiplicity = 0usize;
    for (li, learner) in learners.iter_mut().enumerate() {
        let (beta, sse) = learner.fit_step_select(u_view, weights, wu2);
        if li == 0 {
            beta_len = beta.len();
        }
        if sse.is_nan() {
            sse_is_nan = true;
            break;
        }
        if observe_candidate(li, sse, cand_idx, &mut best_sse, &mut best_multiplicity) {
            found = true;
            cand_beta.clear();
            cand_beta.extend_from_slice(beta);
        }
    }

    if sse_is_nan {
        // Fallback: explicit fit_step_ref for singular Grams.
        best_sse = f64::INFINITY;
        found = false;
        best_multiplicity = 0;
        for (li, learner) in learners.iter_mut().enumerate() {
            let (beta, pred, sse) = learner.fit_step_ref(u_view, weights);
            if li == 0 {
                beta_len = beta.len();
            }
            if observe_candidate(li, sse, cand_idx, &mut best_sse, &mut best_multiplicity) {
                found = true;
                cand_beta.clear();
                cand_beta.extend_from_slice(beta);
                cand_pred.copy_from_slice(pred);
            }
        }
    } else if found {
        // Compute prediction for the winner only (O(n) once).
        learners[*cand_idx].predict_into(cand_beta, cand_pred);
        // Btree fallback: copy from its internal pred_buf.
        let pb = learners[*cand_idx].pred_buf();
        if !pb.is_empty() && pb.len() == n {
            cand_pred.copy_from_slice(pb);
        }
    }

    if !found {
        // Every learner produced a non-finite SSE: make this
        // parameter's candidate an explicit no-op rather than reusing
        // the previous step's buffers (stale data). A zero update
        // leaves the joint risk unchanged, so it only wins step 2 if
        // every parameter is degenerate — and then applying it is
        // harmless.
        cand_beta.clear();
        cand_beta.resize(beta_len, 0.0);
        cand_pred.fill(0.0);
        *cand_idx = 0;
    }
    found && best_multiplicity > 1
}

/// Non-cyclic LSS fitting that can stop once `stop_after_distinct` distinct
/// `(parameter, learner)` components have been selected.
///
/// Stability selection reads only the order in which components first appear,
/// up to the `q`-th distinct one: R's `stabsel` rewrites every selection from
/// the `(q+1)`-th distinct component onward, and the reported frequencies
/// record each component's *first* appearance, so both a later component and a
/// repeat of an earlier one leave the answer unchanged. Stopping there returns
/// a prefix of the path the full `mstop` would have produced -- iteration `k`
/// picks the same component whatever the stopping iteration is -- and is the
/// whole reason `q` is a parameter of stability selection rather than a filter
/// applied to a finished fit.
///
/// The returned path is that prefix: its `risk_path`, `selection_order` and
/// per-parameter `selected`/`coefs` are shorter than `mstop`, and its report's
/// iteration counts describe the steps that ran. `None` runs the full `mstop`
/// and is what every fitting entry point passes.
pub fn try_fit_lss_noncyclic_capped(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    offsets: Option<&[f64]>,
    stop_after_distinct: Option<usize>,
) -> Result<LssPath, MboostrsError> {
    let timing_enabled = control.instrumentation != InstrumentationLevel::Off;
    let fit_started = timing_enabled.then(Instant::now);
    let ys = y
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS response must be contiguous".to_string()))?;
    let ws = weights
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS weights must be contiguous".to_string()))?;
    let n = ws.len();
    let k = family.n_params();
    let nus = resolve_nu_per_param(control, k)?;
    let mstop = control.mstop as usize;
    validate_lss_inputs(learners_per_param, y, ys, ws, family, offsets, n, k)?;
    if !matches!(control.step, StepRule::Fixed) {
        return Err(MboostrsError::Control(
            "step='line_search' is implemented for the cyclic LSS methods; \
             the non-cyclic method already compares candidate updates by the \
             joint risk they achieve, so a per-step search would change what \
             it compares"
                .into(),
        ));
    }

    let mut offs: Vec<f64> = Vec::with_capacity(k);
    let mut etas: Vec<Array1<f64>> = Vec::with_capacity(k);
    for p in 0..k {
        let off = offsets
            .map(|o| o[p])
            .unwrap_or_else(|| family.offset(p, ys, ws, &offs));
        if !off.is_finite() {
            return Err(MboostrsError::Solver(format!(
                "LSS offset for parameter '{}' is non-finite",
                family.param_names().get(p).copied().unwrap_or("?")
            )));
        }
        offs.push(off);
        etas.push(Array1::from_elem(n, off));
    }

    let mut selected: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    let mut coefs: Vec<Vec<Vec<f64>>> = (0..k).map(|_| Vec::new()).collect();
    let mut risk_path = Vec::with_capacity(mstop + 1);
    let initial_risk = family.risk(ys, &etas, ws);
    if !initial_risk.is_finite() {
        return Err(MboostrsError::Solver(
            "LSS initial risk is non-finite".to_string(),
        ));
    }
    risk_path.push(initial_risk);
    let setup_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
    let iteration_started = timing_enabled.then(Instant::now);

    let mut u = vec![0.0f64; n];
    let mut selection_order: Vec<usize> = Vec::with_capacity(mstop);
    // Per-parameter candidate update for this step.
    let mut cand_pred: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0; n]).collect();
    let mut cand_beta: Vec<Vec<f64>> = (0..k).map(|_| Vec::new()).collect();
    let mut cand_idx: Vec<usize> = vec![0; k];
    let mut cand_tied: Vec<bool> = vec![false; k];
    let mut tie_selection_counts = vec![0usize; k];
    let mut best_cycle_risk = initial_risk;
    // The parameter the previous step updated and its predictor before that
    // update (see step 1).
    let mut last_best: Option<usize> = None;
    let mut prev_eta = Array1::<f64>::zeros(n);
    // Distinct `(parameter, learner)` components seen so far, tracked only when
    // a caller asked to stop at a count of them.
    let mut distinct_seen: Vec<Vec<bool>> = if stop_after_distinct.is_some() {
        learners_per_param
            .iter()
            .map(|learners| vec![false; learners.len()])
            .collect()
    } else {
        Vec::new()
    };
    let mut distinct_count = 0usize;

    for m in 0..mstop {
        // 1. Best base-learner update for each parameter. gamboostLSS
        // evaluates a parameter's candidate with the gradient its own model
        // last computed, which predates the previous step's update unless
        // that step updated this parameter; follow it, so the candidate of a
        // parameter other than the last-updated one uses the gradient with
        // the last-updated parameter at its value before that update.
        for p in 0..k {
            let stale = matches!(last_best, Some(lb) if lb != p);
            if stale {
                let lb = last_best.expect("checked");
                std::mem::swap(&mut etas[lb], &mut prev_eta);
                family.ngradient(p, ys, &etas, ws, &mut u);
                std::mem::swap(&mut etas[lb], &mut prev_eta);
            } else {
                family.ngradient(p, ys, &etas, ws, &mut u);
            }
            // Put the parameters' gradients on comparable footing before
            // selection, so one `nu` does not take a large step on one and a
            // negligible one on another. A no-op under the default.
            stabilize_ngradient(&mut u, ws, control.stabilization);
            cand_tied[p] = noncyclic_candidate(
                &mut learners_per_param[p],
                &u,
                ws,
                weights,
                n,
                &mut cand_idx[p],
                &mut cand_beta[p],
                &mut cand_pred[p],
            );
        }
        // 2. Pick the parameter whose update most reduces the joint risk.
        let mut best_p = 0usize;
        let mut best_risk = f64::INFINITY;
        let mut best_param_multiplicity = 0usize;
        for p in 0..k {
            {
                let ep = etas[p].as_slice_mut().expect("eta contiguous");
                for i in 0..n {
                    ep[i] += nus[p] * cand_pred[p][i];
                }
            }
            let r = family.risk(ys, &etas, ws);
            {
                let ep = etas[p].as_slice_mut().expect("eta contiguous");
                for i in 0..n {
                    ep[i] -= nus[p] * cand_pred[p][i];
                }
            }
            if r.is_finite() {
                observe_candidate(
                    p,
                    r,
                    &mut best_p,
                    &mut best_risk,
                    &mut best_param_multiplicity,
                );
            }
        }
        if !best_risk.is_finite() {
            return Err(MboostrsError::Solver(
                "all non-cyclic LSS candidate risks are non-finite".to_string(),
            ));
        }
        // 3. Apply only the winning update. gamboostLSS recomputes the
        // winner's gradient at the current values before taking its step, so
        // a candidate scored with the earlier gradient is refitted first.
        if matches!(last_best, Some(lb) if lb != best_p) {
            family.ngradient(best_p, ys, &etas, ws, &mut u);
            stabilize_ngradient(&mut u, ws, control.stabilization);
            cand_tied[best_p] = noncyclic_candidate(
                &mut learners_per_param[best_p],
                &u,
                ws,
                weights,
                n,
                &mut cand_idx[best_p],
                &mut cand_beta[best_p],
                &mut cand_pred[best_p],
            );
        }
        prev_eta.assign(&etas[best_p]);
        last_best = Some(best_p);
        {
            let ep = etas[best_p].as_slice_mut().expect("eta contiguous");
            for i in 0..n {
                ep[i] += nus[best_p] * cand_pred[best_p][i];
            }
        }
        selection_order.push(best_p);
        tie_selection_counts[best_p] += usize::from(cand_tied[best_p]);
        tie_selection_counts[best_p] += usize::from(best_param_multiplicity > 1);
        selected[best_p].push(cand_idx[best_p]);
        coefs[best_p].push(cand_beta[best_p].clone());
        let step_risk = family.risk(ys, &etas, ws);
        check_lss_cycle_risk(step_risk, &mut best_cycle_risk, m, "non-cyclic")?;
        risk_path.push(step_risk);
        if let Some(limit) = stop_after_distinct {
            let seen = &mut distinct_seen[best_p][cand_idx[best_p]];
            if !*seen {
                *seen = true;
                distinct_count += 1;
            }
            if distinct_count >= limit {
                break;
            }
        }
    }

    let iteration_ns = iteration_started.map(|started| started.elapsed().as_nanos() as u64);
    let total_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
    let mut per_param = Vec::with_capacity(k);
    for p in 0..k {
        let learner_stats = crate::learners::aggregate_report_stats(&learners_per_param[p]);
        per_param.push(BoostingPath {
            report: FitReport {
                instrumentation: control.instrumentation,
                route: EngineRoute::LssNoncyclic,
                n_observations: n,
                n_learners: learners_per_param[p].len(),
                total_basis_columns: learners_per_param[p]
                    .iter()
                    .map(PreparedLearner::n_parameters)
                    .sum(),
                requested_mstop: mstop,
                completed_mstop: selected[p].len(),
                scalar_learner_count: learner_stats.scalar_learners,
                block_learner_count: learner_stats.block_learners,
                tree_learner_count: learner_stats.tree_learners,
                cholesky_solver_count: learner_stats.cholesky_solvers,
                lu_solver_count: learner_stats.lu_solvers,
                regularized_solver_count: learner_stats.regularized_solvers,
                max_solver_jitter: learner_stats.max_solver_jitter,
                factor_cache_bytes: learner_stats.factor_cache_bytes,
                retained_learner_bytes: learner_stats.retained_data_bytes,
                tie_selection_count: tie_selection_counts[p],
                setup_ns,
                iteration_ns,
                total_ns,
                ..FitReport::default()
            },
            offset: offs[p],
            selected: std::mem::take(&mut selected[p]),
            coefs: std::mem::take(&mut coefs[p]),
            // The joint risk after each of this parameter's updates, as
            // gamboostLSS's risk(merge = FALSE): step s updated parameter
            // selection_order[s], and risk_path[s + 1] is the risk after it.
            risk: std::iter::once(risk_path[0])
                .chain(
                    selection_order
                        .iter()
                        .enumerate()
                        .filter(|&(_, &q)| q == p)
                        .map(|(step, _)| risk_path[step + 1]),
                )
                .collect(),
            fitted: etas[p].clone(),
            // Not stored — see fit_lss (LSS4): no consumer, pure overhead.
            increments: Vec::new(),
            nuisance_path: Vec::new(),
            nuisance_vec_path: Vec::new(),
            nuisance_vec: Vec::new(),
            // A cyclic LSS path is not resumable per parameter: freezing one
            // changes the later updates of the others, so there is no single
            // residual a continuation could pick up from.
            residual: Array1::zeros(0),
        });
    }
    Ok(LssPath {
        per_param,
        risk: risk_path,
        selection_order,
    })
}

/// Cross-validated risk for a cyclic GAMLSS fit over a grid of per-parameter
/// mstop combinations (R's `cvrisk.mboostLSS`).
///
/// For each fold `b`, each grid point is fit on the in-bag weights with that
/// point's per-parameter step budget. This is intentionally not a "fit once and
/// take partial sums" implementation: in cyclic LSS, freezing one parameter
/// changes later updates of the other parameters, so prefix sums of a full path
/// are not generally equivalent to a path fit with smaller per-parameter
/// budgets. The out-of-bag risk `sum_i oob_w_i * nll_i` is then evaluated from
/// the fitted parameters for each grid point.
///
/// `folds[:, b]` holds the in-bag weights for fold `b`; the out-of-bag set is the
/// observations with zero in-bag weight (their original `weights`). `grid[g]` is
/// a per-parameter mstop vector (length `K`). Returns a `B x grid.len()` matrix
/// of per-fold out-of-bag risks.
pub fn cvrisk_lss(
    base_learners: &[Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    folds: ndarray::ArrayView2<f64>,
    grid: &[Vec<usize>],
) -> ndarray::Array2<f64> {
    try_cvrisk_lss(base_learners, y, weights, family, control, folds, grid)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible variant of [`cvrisk_lss`].
pub fn try_cvrisk_lss(
    base_learners: &[Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    folds: ndarray::ArrayView2<f64>,
    grid: &[Vec<usize>],
) -> Result<ndarray::Array2<f64>, MboostrsError> {
    // For linear (bols, lambda=0) learners, clone + reweight to the fold's
    // weights is exact. (Penalized learners need a true re-prepare so the
    // df->lambda is recomputed per fold — use `cvrisk_lss_with` for those.)
    try_cvrisk_lss_with(
        |fold_w| {
            Ok(base_learners
                .iter()
                .map(|set| {
                    set.iter()
                        .map(|l| {
                            let mut c = l.clone();
                            c.reweight(fold_w);
                            c
                        })
                        .collect()
                })
                .collect())
        },
        y,
        weights,
        family,
        control,
        folds,
        None,
        grid,
    )
}

/// Like `cvrisk_lss`, but the per-parameter learner sets are *rebuilt* from
/// scratch for each fold via `prepare(fold_weights)`. This is required for
/// penalized learners (e.g. `bbs`): R re-runs the base-learner preprocessing
/// per fold, recomputing `df_to_lambda` from the fold weights. Cloning +
/// reweighting a single full-sample learner keeps the full-sample lambda and
/// gives a small but systematic mismatch with R for smooth `gamboostLSS`.
///
/// This function still refits every grid point for the same path-dependence
/// reason documented on `cvrisk_lss`.
pub fn cvrisk_lss_with<F>(
    prepare: F,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    folds: ndarray::ArrayView2<f64>,
    grid: &[Vec<usize>],
) -> ndarray::Array2<f64>
where
    F: Fn(ArrayView1<f64>) -> Vec<Vec<PreparedLearner>> + Sync,
{
    try_cvrisk_lss_with(
        |fold_weights| Ok(prepare(fold_weights)),
        y,
        weights,
        family,
        control,
        folds,
        None,
        grid,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible variant of [`cvrisk_lss_with`].
pub fn try_cvrisk_lss_with<F>(
    prepare: F,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    folds: ndarray::ArrayView2<f64>,
    judge: Option<&ndarray::Array2<bool>>,
    grid: &[Vec<usize>],
) -> Result<ndarray::Array2<f64>, MboostrsError>
where
    F: Fn(ArrayView1<f64>) -> Result<Vec<Vec<PreparedLearner>>, MboostrsError> + Sync,
{
    use rayon::prelude::*;

    let ys = y
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS response must be contiguous".to_string()))?;
    let ws = weights
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS weights must be contiguous".to_string()))?;
    let n = ws.len();
    let b = check_lss_cv_shapes(n, folds, judge)?;
    if grid.is_empty() || grid.iter().any(|point| point.len() != family.n_params()) {
        return Err(MboostrsError::Shape(format!(
            "each LSS CV grid point must contain {} parameter budgets",
            family.n_params()
        )));
    }
    check_lss_cv_values(y, ys, ws, folds, family)?;

    // Folds are independent; run them in parallel (mirrors the non-LSS
    // `cvrisk`). The indexed collect keeps results in fold order, so the
    // output is identical to the sequential version.
    let rows: Result<Vec<Vec<f64>>, MboostrsError> = (0..b)
        .into_par_iter()
        .map(|fold| {
            let fold_w: Vec<f64> = (0..n).map(|i| folds[[i, fold]]).collect();
            let oob_w: Vec<f64> = (0..n)
                .map(|i| {
                    if fold_w[i] > 0.0 || judge.is_some_and(|mask| !mask[[i, fold]]) {
                        0.0
                    } else {
                        ws[i]
                    }
                })
                .collect();
            let oob_sum: f64 = oob_w.iter().sum();
            if fold_w.iter().all(|&value| value == 0.0) {
                return Err(MboostrsError::Shape(format!(
                    "fold {fold} has no in-bag observations"
                )));
            }
            if oob_sum <= 0.0 {
                return Err(MboostrsError::Shape(format!(
                    "fold {fold} has no positive-weight OOB observations"
                )));
            }
            let fold_w_arr = Array1::from_vec(fold_w);

            // Rebuild the learners against this fold's in-bag weights once
            // (correct per-fold lambda); fit_lss recomputes from scratch each
            // call, so the same set is reused across grid points.
            let mut learners = prepare(fold_w_arr.view())?;
            // Validate the fold-specific learner layout once before either
            // optimized checkpoint replay or individual grid fits.
            try_fit_lss(
                &mut learners,
                y,
                fold_w_arr.view(),
                family,
                control,
                Some(&vec![0; family.n_params()]),
                None,
            )?;

            let risk_from_etas = |etas: &[Array1<f64>]| -> f64 {
                let r = family.risk(ys, etas, &oob_w);
                if oob_sum > 0.0 {
                    r / oob_sum
                } else {
                    f64::NAN
                }
            };

            // Diagonal grid points (every parameter sharing the same budget
            // m) are an exact prefix of a single fit run to max(m) — see
            // fit_lss_diagonal_checkpoints. Non-diagonal points (unequal
            // per-parameter budgets) don't have this property and still
            // need their own fit_lss call.
            let is_diag: Vec<bool> = grid
                .iter()
                .map(|point| !point.is_empty() && point.iter().all(|&v| v == point[0]))
                .collect();
            let mut diag_ms: Vec<usize> = grid
                .iter()
                .zip(&is_diag)
                .filter(|(_, &d)| d)
                .map(|(point, _)| point[0])
                .collect();
            diag_ms.sort_unstable();
            diag_ms.dedup();

            let snap_map: std::collections::HashMap<usize, Vec<Array1<f64>>> = if diag_ms.is_empty()
            {
                std::collections::HashMap::new()
            } else {
                try_fit_lss_diagonal_checkpoints(
                    &mut learners,
                    y,
                    fold_w_arr.view(),
                    family,
                    control,
                    &diag_ms,
                    None,
                )?
                .into_iter()
                .collect()
            };

            grid.iter()
                .zip(&is_diag)
                .map(|(point, &diag)| {
                    if diag {
                        Ok(risk_from_etas(&snap_map[&point[0]]))
                    } else {
                        let path = try_fit_lss(
                            &mut learners,
                            y,
                            fold_w_arr.view(),
                            family,
                            control,
                            Some(point),
                            None,
                        )?;
                        let etas: Vec<Array1<f64>> =
                            path.per_param.iter().map(|bp| bp.fitted.clone()).collect();
                        Ok(risk_from_etas(&etas))
                    }
                })
                .map(|risk| {
                    let risk = risk?;
                    if risk.is_finite() {
                        Ok(risk)
                    } else {
                        Err(MboostrsError::Solver(format!(
                            "LSS OOB risk is non-finite in fold {fold}"
                        )))
                    }
                })
                .collect()
        })
        .collect();
    let rows = rows?;

    let mut out = ndarray::Array2::zeros((b, grid.len()));
    for (fold, risks) in rows.iter().enumerate() {
        for (g, r) in risks.iter().enumerate() {
            out[[fold, g]] = *r;
        }
    }
    Ok(out)
}

// ─── GaussianLSS (mu identity link, sigma log link) ──────────────────────────

/// Normal location-scale family: `y ~ N(mu, sigma^2)` with `mu` (identity link)
/// and `sigma = exp(eta_sigma)` (log link). Matches R `gamboostLSS::GaussianLSS`.
///
/// - offset: `mu` = weighted mean of y; `sigma` = `log(weighted sd of y)`
///   (R's `weighted.sd`: variance = `sum(w (y-ybar)^2) / (sum(w) - 1)`).
/// - ngradient: `u_mu = (y - mu) / sigma^2`,
///   `u_sigma = -1 + (y - mu)^2 / sigma^2`.
/// - risk per obs (negative log-lik): `log(sigma) + 0.5*log(2*pi)
///   + (y - mu)^2 / (2 sigma^2)`.
pub struct GaussianLSS;

const GAUSSIAN_LSS_PARAMS: &[&str] = &["mu", "sigma"];

impl LssFamily for GaussianLSS {
    fn name(&self) -> &'static str {
        "GaussianLSS"
    }

    fn n_params(&self) -> usize {
        2
    }

    fn param_names(&self) -> &'static [&'static str] {
        GAUSSIAN_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let sw: f64 = w.iter().sum();
        let wmean = if sw > 0.0 {
            y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
        } else {
            0.0
        };
        match k {
            0 => wmean, // mu: identity link
            1 => {
                // sigma: log link, offset = log(weighted.sd)
                let wss: f64 = y
                    .iter()
                    .zip(w)
                    .map(|(yi, wi)| wi * (yi - wmean) * (yi - wmean))
                    .sum();
                let var = if sw > 1.0 { wss / (sw - 1.0) } else { 1.0 };
                var.max(1e-12).sqrt().ln()
            }
            _ => unreachable!("GaussianLSS has 2 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // (y - mu) / sigma^2 ; sigma = exp(eta_sigma) -> 1/sigma^2 = exp(-2 eta_sigma)
                for i in 0..n {
                    out[i] = (y[i] - mu[i]) * (-2.0 * eta_sigma[i]).exp();
                }
            }
            1 => {
                // -1 + (y - mu)^2 / sigma^2
                for i in 0..n {
                    let r = y[i] - mu[i];
                    out[i] = -1.0 + r * r * (-2.0 * eta_sigma[i]).exp();
                }
            }
            _ => unreachable!("GaussianLSS has 2 parameters"),
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut s = 0.0f64;
        for i in 0..y.len() {
            let r = y[i] - mu[i];
            let inv_var = (-2.0 * eta_sigma[i]).exp(); // 1/sigma^2
                                                       // log(sigma) = eta_sigma
            let nll = eta_sigma[i] + half_log_2pi + 0.5 * r * r * inv_var;
            s += w[i] * nll;
        }
        s
    }

    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta), // identity
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp(); // sigma = exp(eta)
                }
            }
            _ => unreachable!("GaussianLSS has 2 parameters"),
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        if let Some(pos) = y.iter().position(|v| !v.is_finite()) {
            return Err(MboostrsError::BadResponse(
                "GaussianLSS".to_string(),
                format!("y must be finite; non-finite at index {pos}"),
            ));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(GaussianLSS)
    }

    fn neg_hessian(&self, y: &[f64], etas: &[Array1<f64>], out: &mut [f64]) {
        // l = -s - r^2 e^{-2s} / 2 with r = y - mu, s = eta_sigma:
        // -l_mumu = e^{-2s}, -l_mus = 2 r e^{-2s}, -l_ss = 2 r^2 e^{-2s}.
        let mu = &etas[0];
        let s = &etas[1];
        for i in 0..mu.len() {
            let inv_var = (-2.0 * s[i]).exp();
            let r = y[i] - mu[i];
            out[i * 4] = inv_var;
            out[i * 4 + 1] = 2.0 * r * inv_var;
            out[i * 4 + 2] = 2.0 * r * inv_var;
            out[i * 4 + 3] = 2.0 * r * r * inv_var;
        }
    }
}

// ─── GaussianKnownVarLSS (known sampling variance in the response) ──────────

/// Normal location-scale family whose observations carry a known sampling
/// variance: `v_i = s2_i + exp(eta_sigma_i)`, where `exp(eta_sigma_i)` is the
/// *excess* variance the sigma-submodel estimates and `s2_i` is fixed data.
/// Built for regressing small-area estimates (life expectancy, county means)
/// whose published standard errors vary by orders of magnitude: a plain
/// GaussianLSS fit mistakes sampling noise for heterogeneity and its
/// sigma-submodel selects proxies for sample size.
///
/// Response layout follows [`SurvivalLSS`]: `y = [outcome_0..n-1,
/// s2_0..n-1]` with `n = w.len()`. Riding the auxiliary vector inside `y`
/// keeps it aligned under cvrisk/stabsel, which reweight full-length arrays
/// rather than subsetting rows.
///
/// - offset: `mu` = weighted mean of y (identity link); `sigma` =
///   `log(max(weighted var(y) - weighted mean(s2), eps))` — the moment-matched
///   excess variance, floored so a fully-explained dispersion stays finite.
/// - ngradient: `u_mu = (y - mu) / v`;
///   `u_sigma = 0.5 * e^{eta} / v * [(y - mu)^2 / v - 1]`
///   (the leading 1/2 is real here — unlike GaussianLSS, where the chain
///   rule through `sigma = e^{eta}` cancels it; keeping the gradient exact
///   makes nu = 0.1 steps comparable across families. The per-observation
///   attenuation e^eta / v is what stops the model chasing noise where s2
///   dominates).
/// - risk per obs: `0.5 log(2 pi v) + (y - mu)^2 / (2 v)`.
pub struct GaussianKnownVarLSS;

const GAUSSIAN_KNOWNVAR_PARAMS: &[&str] = &["mu", "sigma"];

impl LssFamily for GaussianKnownVarLSS {
    fn name(&self) -> &'static str {
        "GaussianKnownVarLSS"
    }

    fn n_params(&self) -> usize {
        2
    }

    /// The outcome followed by its known sampling variance.
    fn response_span(&self) -> usize {
        2
    }

    fn param_names(&self) -> &'static [&'static str] {
        GAUSSIAN_KNOWNVAR_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = w.len();
        let sw: f64 = w.iter().sum();
        let wmean = if sw > 0.0 {
            (0..n).map(|i| y[i] * w[i]).sum::<f64>() / sw
        } else {
            0.0
        };
        match k {
            0 => wmean, // mu: identity link
            1 => {
                // sigma: the link scale holds log excess VARIANCE. Moment-match
                // the total dispersion, subtract the known sampling part, floor.
                let wss: f64 = (0..n).map(|i| w[i] * (y[i] - wmean) * (y[i] - wmean)).sum();
                let total_var = if sw > 1.0 { wss / (sw - 1.0) } else { 1.0 };
                let mean_s2: f64 = if sw > 0.0 {
                    (0..n).map(|i| w[i] * y[n + i]).sum::<f64>() / sw
                } else {
                    0.0
                };
                let excess = (total_var - mean_s2).max(total_var.max(1e-12) * 1e-6);
                excess.max(1e-12).ln()
            }
            _ => unreachable!("GaussianKnownVarLSS has 2 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = out.len();
        let mu = etas[0].as_slice().expect("eta contiguous");
        let eta_var = etas[1].as_slice().expect("eta contiguous");
        match k {
            0 => {
                for i in 0..n {
                    let v = y[n + i] + eta_var[i].exp();
                    out[i] = (y[i] - mu[i]) / v;
                }
            }
            1 => {
                for i in 0..n {
                    let e = eta_var[i].exp();
                    let v = y[n + i] + e;
                    let r = y[i] - mu[i];
                    out[i] = 0.5 * e / v * (r * r / v - 1.0);
                }
            }
            _ => unreachable!("GaussianKnownVarLSS has 2 parameters"),
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = w.len();
        let mu = etas[0].as_slice().expect("eta contiguous");
        let eta_var = etas[1].as_slice().expect("eta contiguous");
        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut s = 0.0f64;
        for i in 0..n {
            let v = y[n + i] + eta_var[i].exp();
            let r = y[i] - mu[i];
            s += w[i] * (0.5 * v.ln() + half_log_2pi + 0.5 * r * r / v);
        }
        s
    }

    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta), // mu: identity
            1 => {
                // eta holds log excess variance, so the natural-scale excess
                // standard deviation is exp(eta / 2). The TOTAL sd of a new
                // observation is sqrt(s2 + exp(eta)) and needs the s2 column —
                // see the Python wrapper's pdf/cdf/quantile.
                for i in 0..eta.len() {
                    out[i] = (0.5 * eta[i]).exp();
                }
            }
            _ => unreachable!("GaussianKnownVarLSS has 2 parameters"),
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        if y.len() % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                "GaussianKnownVarLSS".to_string(),
                format!(
                    "response must be the concatenated [outcome, s2] layout, \
                     got odd length {}",
                    y.len()
                ),
            ));
        }
        if let Some(pos) = y.iter().position(|v| !v.is_finite()) {
            return Err(MboostrsError::BadResponse(
                "GaussianKnownVarLSS".to_string(),
                format!("y must be finite; non-finite at index {pos}"),
            ));
        }
        let n = y.len() / 2;
        if let Some(pos) = (0..n).map(|i| n + i).find(|&i| y[i] < 0.0) {
            return Err(MboostrsError::BadResponse(
                "GaussianKnownVarLSS".to_string(),
                format!("s2 (known sampling variance) must be >= 0; negative at index {pos}"),
            ));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(GaussianKnownVarLSS)
    }
}

// ─── GammaLSS (mu log link, sigma=shape log link) ────────────────────────────

/// Gamma location-shape family: `y ~ Gamma(mean = mu, shape = sigma)` with both
/// `mu = exp(eta_mu)` and `sigma = exp(eta_sigma)` (log links). Matches R
/// `gamboostLSS::GammaLSS`.
///
/// - offset: `mu` = `log(mean y)` (the Brent argmin of the mu-risk, independent
///   of sigma); `sigma` = Brent argmin of the sigma-risk holding `mu = mean y`.
///   Both replicate R's `optimize` (same intervals, tolerance).
/// - ngradient: `u_mu = sigma (y/mu - 1)`,
///   `u_sigma = sigma (-psi(sigma) + eta_sigma + 1 - log mu + log y - y/mu)`.
/// - risk per obs: `lgamma(s) + s y/mu - s log y - s log s + s log mu + log y`.
pub struct GammaLSS;

const GAMMA_LSS_PARAMS: &[&str] = &["mu", "sigma"];

impl LssFamily for GammaLSS {
    fn name(&self) -> &'static str {
        "GammaLSS"
    }

    fn n_params(&self) -> usize {
        2
    }

    fn param_names(&self) -> &'static [&'static str] {
        GAMMA_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], prior: &[f64]) -> f64 {
        let sw: f64 = w.iter().sum();
        let wmean = if sw > 0.0 {
            y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
        } else {
            0.0
        };
        match k {
            0 => {
                // R: sigma <<- mean(y)^2/var(y); optimize mu-risk over [0, max(y^2)].
                // The argmin is log(mean y) regardless of sigma; replicate the
                // Brent call for fidelity.
                let wvar = if sw > 1.0 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * (yi - wmean) * (yi - wmean))
                        .sum::<f64>()
                        / (sw - 1.0)
                } else {
                    1.0
                };
                let sigma_mom = if wvar > 0.0 {
                    wmean * wmean / wvar
                } else {
                    1.0
                };
                let y_max_sq = y.iter().map(|yi| yi * yi).fold(0.0f64, f64::max);
                let risk = |f_mu: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * gamma_plloss(sigma_mom, *yi, f_mu))
                        .sum()
                };
                minimize_brent(&risk, 0.0, y_max_sq, R_OPTIMIZE_TOL, 500)
            }
            1 => {
                // R: optimize sigma-risk over [0, max(y)] holding mu at the
                // already-computed mu offset (= log mean y).
                let f_mu = prior
                    .first()
                    .copied()
                    .unwrap_or_else(|| wmean.max(1e-12).ln());
                let y_max = y.iter().cloned().fold(0.0f64, f64::max);
                let risk = |f_sigma: f64| -> f64 {
                    let s = f_sigma.exp();
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * gamma_plloss(s, *yi, f_mu))
                        .sum()
                };
                minimize_brent(&risk, 0.0, y_max, R_OPTIMIZE_TOL, 500)
            }
            _ => unreachable!("GammaLSS has 2 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // sigma * (y/mu - 1); mu = exp(eta_mu)
                for i in 0..n {
                    let s = eta_sigma[i].exp();
                    out[i] = s * (y[i] * (-eta_mu[i]).exp() - 1.0);
                }
            }
            1 => {
                // sigma * (-psi(sigma) + eta_sigma + 1 - log mu + log y - y/mu)
                for i in 0..n {
                    let s = eta_sigma[i].exp();
                    let mu = eta_mu[i].exp();
                    out[i] =
                        s * (-digamma(s) + eta_sigma[i] + 1.0 - eta_mu[i] + y[i].ln() - y[i] / mu);
                }
            }
            _ => unreachable!("GammaLSS has 2 parameters"),
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0f64;
        for i in 0..y.len() {
            s += w[i] * gamma_plloss(eta_sigma[i].exp(), y[i], eta_mu[i]);
        }
        s
    }

    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        // Both mu and sigma use the log link.
        for i in 0..eta.len() {
            out[i] = eta[i].exp();
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() || *v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "GammaLSS".to_string(),
                    format!("y must be finite and > 0; bad value at index {i}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(GammaLSS)
    }
}

// ─── StudentTLSS (mu id link, sigma + df log links) ──────────────────────────

/// Student's t location-scale-shape family: `mu` (identity), `sigma = exp(eta)`
/// (log), `df = exp(eta)` (log). Matches R `gamboostLSS::StudentTLSS`.
///
/// Per-obs negative log-likelihood (`r = y - mu`):
/// `lgamma(df/2) - lgamma((df+1)/2) + lgamma(1/2) + 0.5 log df + log sigma
///  + (df+1)/2 * log(1 + r^2/(df sigma^2))`.
/// Offsets are Brent minima of the risk holding the other two parameters at R's
/// defaults (mu = mean y, sigma = 1, df = 4).
pub struct StudentTLSS;

const STUDENTT_LSS_PARAMS: &[&str] = &["mu", "sigma", "df"];

/// Student-t per-observation NLL with `eta_sigma = log sigma`, `eta_df = log df`.
#[inline]
fn studentt_nll(mu: f64, eta_sigma: f64, eta_df: f64, y: f64) -> f64 {
    let sigma = eta_sigma.exp();
    let df = eta_df.exp();
    let r = y - mu;
    lgamma(df / 2.0) - lgamma((df + 1.0) / 2.0)
        + lgamma(0.5)
        + 0.5 * df.ln()
        + eta_sigma
        + (df + 1.0) / 2.0 * (1.0 + r * r / (df * sigma * sigma)).ln()
}

/// Student-t densities from response-scale location, scale and degrees of freedom.
/// Columns must have equal length or length one (scalar broadcasting).
/// Empty input returns an empty vector; nonpositive scales or df are refused.
/// Uses the fitting likelihood in log space to avoid overflowing gamma ratios.
pub fn student_t_pdf(
    y: &[f64],
    mu: &[f64],
    sigma: &[f64],
    df: &[f64],
) -> Result<Vec<f64>, MboostrsError> {
    let cols = [y, mu, sigma, df];
    if cols.iter().any(|c| c.is_empty()) {
        return Ok(Vec::new());
    }
    let n = cols.iter().map(|c| c.len()).max().unwrap();
    if cols.iter().any(|c| c.len() != 1 && c.len() != n) {
        return Err(MboostrsError::Shape(
            "StudentTLSS density columns must have equal length or be scalar".into(),
        ));
    }
    if sigma.iter().chain(df).any(|v| *v <= 0.0) {
        return Err(MboostrsError::Control(
            "StudentTLSS requires sigma > 0 and df > 0".into(),
        ));
    }
    Ok((0..n)
        .map(|i| {
            let at = |c: &[f64]| c[if c.len() == 1 { 0 } else { i }];
            (-studentt_nll(at(mu), at(sigma).ln(), at(df).ln(), at(y))).exp()
        })
        .collect())
}

impl LssFamily for StudentTLSS {
    fn name(&self) -> &'static str {
        "StudentTLSS"
    }

    fn n_params(&self) -> usize {
        3
    }

    fn param_names(&self) -> &'static [&'static str] {
        STUDENTT_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], prior: &[f64]) -> f64 {
        let sw: f64 = w.iter().sum();
        let wmean = if sw > 0.0 {
            y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
        } else {
            0.0
        };
        let log4 = 4.0_f64.ln();
        let (lo_log, hi_log) = (1e-10_f64.ln(), 1e10_f64.ln());
        // R computes the offsets in order mu->sigma->df, each using the offsets
        // already computed for earlier parameters and defaults (mu=mean y,
        // sigma=1 -> eta 0, df=4) for the not-yet-computed ones.
        let mu_prior = prior.first().copied().unwrap_or(wmean);
        let sigma_prior = prior.get(1).copied().unwrap_or(0.0);
        match k {
            0 => {
                // mu: sigma=1 (eta=0), df=4; optimize over [min y, max y].
                let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
                let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let risk = |mu: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * studentt_nll(mu, 0.0, log4, *yi))
                        .sum()
                };
                minimize_brent(&risk, y_min, y_max, R_OPTIMIZE_TOL, 500)
            }
            1 => {
                // sigma: mu=mu offset, df=4; optimize over [log 1e-10, log 1e10].
                let risk = |f_sigma: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * studentt_nll(mu_prior, f_sigma, log4, *yi))
                        .sum()
                };
                minimize_brent(&risk, lo_log, hi_log, R_OPTIMIZE_TOL, 500)
            }
            2 => {
                // df: mu=mu offset, sigma=sigma offset; optimize over the log range.
                let risk = |f_df: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * studentt_nll(mu_prior, sigma_prior, f_df, *yi))
                        .sum()
                };
                minimize_brent(&risk, lo_log, hi_log, R_OPTIMIZE_TOL, 500)
            }
            _ => unreachable!("StudentTLSS has 3 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_df = etas[2].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // (df+1)(y-mu) / (df sigma^2 + (y-mu)^2)
                for i in 0..n {
                    let df = eta_df[i].exp();
                    let s2 = (2.0 * eta_sigma[i]).exp();
                    let r = y[i] - mu[i];
                    out[i] = (df + 1.0) * r / (df * s2 + r * r);
                }
            }
            1 => {
                // -1 + (df+1)/(df sigma^2/(y-mu)^2 + 1)
                for i in 0..n {
                    let df = eta_df[i].exp();
                    let s2 = (2.0 * eta_sigma[i]).exp();
                    let r = y[i] - mu[i];
                    out[i] = -1.0 + (df + 1.0) / (df * s2 / (r * r) + 1.0);
                }
            }
            2 => {
                // df gradient (d = df): d/2 (psi((d+1)/2) - psi(d/2)) - 0.5
                //   - [ d/2 log(1 + r^2/(d sigma^2))
                //       - r^2/(sigma^2 + r^2/d) (1/d + 1)/2 ]
                for i in 0..n {
                    let d = eta_df[i].exp();
                    let s2 = (2.0 * eta_sigma[i]).exp();
                    let r2 = (y[i] - mu[i]) * (y[i] - mu[i]);
                    out[i] = d / 2.0 * (digamma((d + 1.0) / 2.0) - digamma(d / 2.0))
                        - 0.5
                        - (d / 2.0 * (1.0 + r2 / (d * s2)).ln()
                            - r2 / (s2 + r2 / d) * (1.0 / d + 1.0) / 2.0);
                }
            }
            _ => unreachable!("StudentTLSS has 3 parameters"),
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_df = etas[2].as_slice().expect("eta contiguous");
        let mut s = 0.0f64;
        for i in 0..y.len() {
            s += w[i] * studentt_nll(mu[i], eta_sigma[i], eta_df[i], y[i]);
        }
        s
    }

    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta), // mu: identity
            1 | 2 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp(); // sigma, df: log link
                }
            }
            _ => unreachable!("StudentTLSS has 3 parameters"),
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        if let Some(pos) = y.iter().position(|v| !v.is_finite()) {
            return Err(MboostrsError::BadResponse(
                "StudentTLSS".to_string(),
                format!("y must be finite; non-finite at index {pos}"),
            ));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(StudentTLSS)
    }
}

// ─── BetaLSS (mu logit link, phi log link) ───────────────────────────────────

/// Numerically stable logistic CDF (R's `plogis`).
#[inline]
fn plogis(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Logit (R's `qlogis`): `log(p / (1 - p))`.
#[inline]
fn qlogis(p: f64) -> f64 {
    p.ln() - (1.0 - p).ln()
}

/// Beta location-precision family: `y ~ Beta(mu*phi, (1-mu)*phi)` with
/// `mu = plogis(eta_mu)` (logit link, mean) and `phi = exp(eta_phi)` (log link,
/// precision). Matches R `gamboostLSS::BetaLSS`. Requires `0 < y < 1`.
pub struct BetaLSS;

const BETA_LSS_PARAMS: &[&str] = &["mu", "phi"];

/// Beta per-observation NLL with `eta_mu` (logit) and `eta_phi` (log).
#[inline]
fn beta_nll(eta_mu: f64, eta_phi: f64, y: f64) -> f64 {
    let mu = plogis(eta_mu);
    let phi = eta_phi.exp();
    -lgamma(phi) + lgamma(mu * phi) + lgamma((1.0 - mu) * phi)
        - (mu * phi - 1.0) * y.ln()
        - ((1.0 - mu) * phi - 1.0) * (1.0 - y).ln()
}

impl LssFamily for BetaLSS {
    fn name(&self) -> &'static str {
        "BetaLSS"
    }

    fn n_params(&self) -> usize {
        2
    }

    fn param_names(&self) -> &'static [&'static str] {
        BETA_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], prior: &[f64]) -> f64 {
        let sw: f64 = w.iter().sum();
        let wmean = if sw > 0.0 {
            y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
        } else {
            0.5
        };
        match k {
            0 => {
                // mu: phi = method-of-moments precision; optimize over the logit
                // range [qlogis(1e-6), qlogis(0.999999)].
                let wvar = if sw > 1.0 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * (yi - wmean) * (yi - wmean))
                        .sum::<f64>()
                        / (sw - 1.0)
                } else {
                    1.0
                };
                let phi_mom = if wvar > 0.0 {
                    wmean * (1.0 - wmean) / wvar - 1.0
                } else {
                    1.0
                };
                let eta_phi = phi_mom.max(1e-10).ln();
                let risk = |eta_mu: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * beta_nll(eta_mu, eta_phi, *yi))
                        .sum()
                };
                minimize_brent(&risk, qlogis(1e-6), qlogis(0.999999), R_OPTIMIZE_TOL, 500)
            }
            1 => {
                // phi: mu = the computed mu offset (logit scale); optimize over
                // [log 1e-10, log 1e10].
                let eta_mu = prior.first().copied().unwrap_or_else(|| qlogis(wmean));
                let risk = |eta_phi: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * beta_nll(eta_mu, eta_phi, *yi))
                        .sum()
                };
                minimize_brent(&risk, 1e-10_f64.ln(), 1e10_f64.ln(), R_OPTIMIZE_TOL, 500)
            }
            _ => unreachable!("BetaLSS has 2 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_phi = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // mu(1-mu) phi (logit(y) - (psi(mu phi) - psi((1-mu) phi)))
                for i in 0..n {
                    let mu = plogis(eta_mu[i]);
                    let phi = eta_phi[i].exp();
                    out[i] = mu
                        * (1.0 - mu)
                        * phi
                        * (qlogis(y[i]) - (digamma(mu * phi) - digamma((1.0 - mu) * phi)));
                }
            }
            1 => {
                // phi ( mu (logit(y) - (psi(mu phi) - psi((1-mu) phi)))
                //       + psi(phi) - psi((1-mu) phi) + log(1-y) )
                for i in 0..n {
                    let mu = plogis(eta_mu[i]);
                    let phi = eta_phi[i].exp();
                    out[i] = phi
                        * (mu * (qlogis(y[i]) - (digamma(mu * phi) - digamma((1.0 - mu) * phi)))
                            + digamma(phi)
                            - digamma((1.0 - mu) * phi)
                            + (1.0 - y[i]).ln());
                }
            }
            _ => unreachable!("BetaLSS has 2 parameters"),
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_phi = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0f64;
        for i in 0..y.len() {
            s += w[i] * beta_nll(eta_mu[i], eta_phi[i], y[i]);
        }
        s
    }

    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = plogis(eta[i]); // mu: logit link
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp(); // phi: log link
                }
            }
            _ => unreachable!("BetaLSS has 2 parameters"),
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() || *v <= 0.0 || *v >= 1.0 {
                return Err(MboostrsError::BadResponse(
                    "BetaLSS".to_string(),
                    format!("y must be in the open interval (0, 1); bad value at index {i}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BetaLSS)
    }
}

// ─── NBinomialLSS (mu log link = mean, sigma log link = size) ─────────────────

/// Negative-binomial location-dispersion family: `mu = exp(eta_mu)` (mean) and
/// `sigma = exp(eta_sigma)` (size / dispersion). Matches R
/// `gamboostLSS::NBinomialLSS`. Requires non-negative integer-valued `y`.
pub struct NBinomialLSS;

const NBINOMIAL_LSS_PARAMS: &[&str] = &["mu", "sigma"];

/// NB per-observation NLL with `eta_mu`, `eta_sigma` (both log links).
#[inline]
fn nbinomial_nll(eta_mu: f64, eta_sigma: f64, y: f64) -> f64 {
    let mu = eta_mu.exp();
    let s = eta_sigma.exp();
    -lgamma(y + s) + lgamma(s) + lgamma(y + 1.0) - s * s.ln() - y * eta_mu + (y + s) * (mu + s).ln()
}

impl LssFamily for NBinomialLSS {
    fn name(&self) -> &'static str {
        "NBinomialLSS"
    }

    fn n_params(&self) -> usize {
        2
    }

    fn param_names(&self) -> &'static [&'static str] {
        NBINOMIAL_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], prior: &[f64]) -> f64 {
        let sw: f64 = w.iter().sum();
        let wmean = if sw > 0.0 {
            y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / sw
        } else {
            0.0
        };
        let (lo, hi) = (1e-10_f64.ln(), 1e10_f64.ln());
        match k {
            0 => {
                // mu: sigma = method-of-moments size; optimize over the log range.
                // The argmin is log(mean y) regardless of sigma.
                let wvar = if sw > 1.0 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * (yi - wmean) * (yi - wmean))
                        .sum::<f64>()
                        / (sw - 1.0)
                } else {
                    1.0
                };
                let s_mom = if wvar > wmean {
                    wmean * wmean / (wvar - wmean)
                } else {
                    1e-10
                };
                let eta_sigma = s_mom.max(1e-10).ln();
                let risk = |eta_mu: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * nbinomial_nll(eta_mu, eta_sigma, *yi))
                        .sum()
                };
                minimize_brent(&risk, lo, hi, R_OPTIMIZE_TOL, 500)
            }
            1 => {
                // sigma: mu = the computed mu offset (= log mean y).
                let eta_mu = prior
                    .first()
                    .copied()
                    .unwrap_or_else(|| wmean.max(1e-12).ln());
                let risk = |eta_sigma: f64| -> f64 {
                    y.iter()
                        .zip(w)
                        .map(|(yi, wi)| wi * nbinomial_nll(eta_mu, eta_sigma, *yi))
                        .sum()
                };
                minimize_brent(&risk, lo, hi, R_OPTIMIZE_TOL, 500)
            }
            _ => unreachable!("NBinomialLSS has 2 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // y - (y+sigma) mu/(mu+sigma)
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let s = eta_sigma[i].exp();
                    out[i] = y[i] - (y[i] + s) * mu / (mu + s);
                }
            }
            1 => {
                // sigma (psi(y+sigma) - psi(sigma) + log sigma + 1
                //        - log(mu+sigma) - (sigma+y)/(mu+sigma))
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let s = eta_sigma[i].exp();
                    out[i] = s
                        * (digamma(y[i] + s) - digamma(s) + s.ln() + 1.0
                            - (mu + s).ln()
                            - (s + y[i]) / (mu + s));
                }
            }
            _ => unreachable!("NBinomialLSS has 2 parameters"),
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0f64;
        for i in 0..y.len() {
            s += w[i] * nbinomial_nll(eta_mu[i], eta_sigma[i], y[i]);
        }
        s
    }

    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        // mu and sigma both use the log link.
        for i in 0..eta.len() {
            out[i] = eta[i].exp();
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() || *v < 0.0 || (*v - v.round()).abs() > 1e-8 {
                return Err(MboostrsError::BadResponse(
                    "NBinomialLSS".to_string(),
                    format!("y must be a non-negative integer count; bad value at index {i}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(NBinomialLSS)
    }
}

// ─── ZIPoLSS (zero-inflated Poisson; mu log link, sigma logit link) ───────────

/// Machine epsilon, matching R's `.Machine$double.eps`.
const R_EPS: f64 = 2.220446049250313e-16;

/// gamlss.dist's `sigma.linkinv` for the logit link: clamp `eta` to
/// `[-thresh, thresh]` with `thresh = -qlogis(eps)`, then apply `plogis`. The
/// clamp keeps the inflation probability strictly inside `(0, 1)`.
#[inline]
fn zi_sigma_linkinv(eta: f64) -> f64 {
    let thresh = -qlogis(R_EPS);
    plogis(eta.clamp(-thresh, thresh))
}

/// Zero-inflated Poisson location-inflation family: `mu = exp(eta_mu)` (Poisson
/// mean, log link) and `sigma = plogis(eta_sigma)` (zero-inflation probability,
/// logit link). Matches R `gamboostLSS::ZIPoLSS()` (wrapping gamlss.dist `ZIP`).
/// Requires non-negative integer-valued `y`.
pub struct ZIPoLSS;

const ZIPO_LSS_PARAMS: &[&str] = &["mu", "sigma"];

/// ZIP per-observation NLL with `eta_mu` (log) and `eta_sigma` (logit).
#[inline]
fn zipo_nll(eta_mu: f64, eta_sigma: f64, y: f64) -> f64 {
    let mu = eta_mu.exp().max(R_EPS);
    let s = zi_sigma_linkinv(eta_sigma);
    if y == 0.0 {
        -(s + (1.0 - s) * (-mu).exp()).ln()
    } else {
        // -(log(1-sigma) - mu + y log mu - log y!)
        -((1.0 - s).ln() - mu + y * mu.ln() - lgamma(y + 1.0))
    }
}

impl LssFamily for ZIPoLSS {
    fn name(&self) -> &'static str {
        "ZIPoLSS"
    }

    fn n_params(&self) -> usize {
        2
    }

    fn param_names(&self) -> &'static [&'static str] {
        ZIPO_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], _w: &[f64], _prior: &[f64]) -> f64 {
        // R's gamlss offsets use fixed initials and (unlike the other LSS
        // families) ignore the weights: mu = log(mean(y)), sigma = qlogis(0.1).
        match k {
            0 => {
                let n = y.len() as f64;
                let mean_y = if n > 0.0 {
                    y.iter().sum::<f64>() / n
                } else {
                    0.0
                };
                mean_y.max(R_EPS).ln()
            }
            1 => qlogis(0.1),
            _ => unreachable!("ZIPoLSS has 2 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // dldm * mu.dr; mu.dr = mu = exp(eta_mu).
                for i in 0..n {
                    let mu = eta_mu[i].exp().max(R_EPS);
                    let s = zi_sigma_linkinv(eta_sigma[i]);
                    let dldm = if y[i] == 0.0 {
                        -(1.0 - s) / ((1.0 - s) + s * mu.exp())
                    } else {
                        y[i] / mu - 1.0
                    };
                    out[i] = dldm * mu;
                }
            }
            1 => {
                // dldd * sigma.dr; sigma.dr = sigma*(1-sigma) (logistic density).
                for i in 0..n {
                    let mu = eta_mu[i].exp().max(R_EPS);
                    let s = zi_sigma_linkinv(eta_sigma[i]);
                    let dldd = if y[i] == 0.0 {
                        (1.0 - (-mu).exp()) / (s + (1.0 - s) * (-mu).exp())
                    } else {
                        -1.0 / (1.0 - s)
                    };
                    out[i] = dldd * s * (1.0 - s);
                }
            }
            _ => unreachable!("ZIPoLSS has 2 parameters"),
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0f64;
        for i in 0..y.len() {
            s += w[i] * zipo_nll(eta_mu[i], eta_sigma[i], y[i]);
        }
        s
    }

    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp().max(R_EPS); // mu: log link
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = zi_sigma_linkinv(eta[i]); // sigma: logit link
                }
            }
            _ => unreachable!("ZIPoLSS has 2 parameters"),
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() || *v < 0.0 || (*v - v.round()).abs() > 1e-8 {
                return Err(MboostrsError::BadResponse(
                    "ZIPoLSS".to_string(),
                    format!("y must be a non-negative integer count; bad value at index {i}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(ZIPoLSS)
    }
}

// ─── ZINBLSS (zero-inflated negative binomial type I; mu/sigma log, nu logit) ─

/// log of the NBI (negative-binomial type I) density at `y` (R's `dNBI`):
/// `dnbinom(y, size = 1/sigma, mu)` for `sigma >= 1e-4`, else Poisson.
#[inline]
fn nbi_logpdf(y: f64, mu: f64, sigma: f64) -> f64 {
    if sigma < 1e-4 {
        -mu + y * mu.ln() - lgamma(y + 1.0)
    } else {
        let r = 1.0 / sigma; // size
        lgamma(y + r) - lgamma(r) - lgamma(y + 1.0)
            + r * (r.ln() - (r + mu).ln())
            + y * (mu.ln() - (r + mu).ln())
    }
}

/// NBI `dldm` = (y - mu) / (mu (1 + mu sigma)).
#[inline]
fn nbi_dldm(y: f64, mu: f64, sigma: f64) -> f64 {
    (y - mu) / (mu * (1.0 + mu * sigma))
}

/// NBI `dldd` = -(1/sigma)^2 (psi(y + 1/sigma) - psi(1/sigma) - log(1 + mu sigma)
///   - (y - mu) sigma / (1 + mu sigma)).
#[inline]
fn nbi_dldd(y: f64, mu: f64, sigma: f64) -> f64 {
    let inv = 1.0 / sigma;
    -(inv * inv)
        * (digamma(y + inv)
            - digamma(inv)
            - (1.0 + mu * sigma).ln()
            - (y - mu) * sigma / (1.0 + mu * sigma))
}

/// Zero-inflated negative-binomial (type I) family: `mu = exp(eta_mu)` (mean,
/// log link), `sigma = exp(eta_sigma)` (dispersion, log link), and
/// `nu = plogis(eta_nu)` (zero-inflation probability, logit link). Matches R
/// `gamboostLSS::ZINBLSS()` (wrapping gamlss.dist `ZINBI`). Requires non-negative
/// integer-valued `y`.
pub struct ZINBLSS;

const ZINB_LSS_PARAMS: &[&str] = &["mu", "sigma", "nu"];

/// ZINBI per-observation NLL with `eta_mu`/`eta_sigma` (log) and `eta_nu` (logit).
#[inline]
fn zinb_nll(eta_mu: f64, eta_sigma: f64, eta_nu: f64, y: f64) -> f64 {
    let mu = eta_mu.exp().max(R_EPS);
    let sigma = eta_sigma.exp().max(R_EPS);
    let nu = zi_sigma_linkinv(eta_nu);
    if y == 0.0 {
        -(nu + (1.0 - nu) * nbi_logpdf(0.0, mu, sigma).exp()).ln()
    } else {
        -((1.0 - nu).ln() + nbi_logpdf(y, mu, sigma))
    }
}

impl LssFamily for ZINBLSS {
    fn name(&self) -> &'static str {
        "ZINBLSS"
    }

    fn n_params(&self) -> usize {
        3
    }

    fn param_names(&self) -> &'static [&'static str] {
        ZINB_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], _w: &[f64], _prior: &[f64]) -> f64 {
        // gamlss fixed initials (weight-independent), matching R:
        //   mu    = log(mean y)
        //   sigma = log(max((var(y) - mean(y)) / mean(y)^2, 0.1))  [var: n-1 denom]
        //   nu    = qlogis(((#{y == 0} / n) + 0.01) / 2)
        let n = y.len() as f64;
        let mean_y = if n > 0.0 {
            y.iter().sum::<f64>() / n
        } else {
            0.0
        };
        match k {
            0 => mean_y.max(R_EPS).ln(),
            1 => {
                let var_y = if n > 1.0 {
                    y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>() / (n - 1.0)
                } else {
                    0.0
                };
                let sig_init = ((var_y - mean_y) / (mean_y * mean_y)).max(0.1);
                sig_init.ln()
            }
            2 => {
                let prop0 = y.iter().filter(|v| **v == 0.0).count() as f64 / n;
                qlogis((prop0 + 0.01) / 2.0)
            }
            _ => unreachable!("ZINBLSS has 3 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_nu = etas[2].as_slice().expect("eta contiguous");
        let n = y.len();
        for i in 0..n {
            let mu = eta_mu[i].exp().max(R_EPS);
            let sigma = eta_sigma[i].exp().max(R_EPS);
            let nu = zi_sigma_linkinv(eta_nu[i]);
            let f0 = nbi_logpdf(0.0, mu, sigma).exp(); // dNBI(0, mu, sigma)
            let denom0 = nu + (1.0 - nu) * f0;
            out[i] = match k {
                0 => {
                    let dldm = if y[i] == 0.0 {
                        (1.0 - nu) / denom0 * f0 * nbi_dldm(0.0, mu, sigma)
                    } else {
                        nbi_dldm(y[i], mu, sigma)
                    };
                    dldm * mu // mu.dr (log link)
                }
                1 => {
                    let dldd = if y[i] == 0.0 {
                        (1.0 - nu) / denom0 * f0 * nbi_dldd(0.0, mu, sigma)
                    } else {
                        nbi_dldd(y[i], mu, sigma)
                    };
                    dldd * sigma // sigma.dr (log link)
                }
                2 => {
                    let dldv = if y[i] == 0.0 {
                        (1.0 - f0) / denom0
                    } else {
                        -1.0 / (1.0 - nu)
                    };
                    dldv * nu * (1.0 - nu) // nu.dr (logit link)
                }
                _ => unreachable!("ZINBLSS has 3 parameters"),
            };
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_nu = etas[2].as_slice().expect("eta contiguous");
        let mut s = 0.0f64;
        for i in 0..y.len() {
            s += w[i] * zinb_nll(eta_mu[i], eta_sigma[i], eta_nu[i], y[i]);
        }
        s
    }

    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 | 1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp().max(R_EPS); // mu, sigma: log link
                }
            }
            2 => {
                for i in 0..eta.len() {
                    out[i] = zi_sigma_linkinv(eta[i]); // nu: logit link
                }
            }
            _ => unreachable!("ZINBLSS has 3 parameters"),
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() || *v < 0.0 || (*v - v.round()).abs() > 1e-8 {
                return Err(MboostrsError::BadResponse(
                    "ZINBLSS".to_string(),
                    format!("y must be a non-negative integer count; bad value at index {i}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(ZINBLSS)
    }
}

// ─── DirichletLSS (K concentration params, log links; K-column response) ──────

/// Parameter names `a1..a20` for [`DirichletLSS`] (supports up to 20 categories).
const DIRICHLET_PARAM_NAMES: [&str; 20] = [
    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13", "a14", "a15",
    "a16", "a17", "a18", "a19", "a20",
];

/// Dirichlet family for compositional responses: `K` concentration parameters
/// `alpha_k = exp(eta_k)` (log links). The response is a `K`-column matrix (each
/// row a composition summing to 1), carried as the concatenated slice
/// `y = [col0.., col1.., .., col_{K-1}..]` (length `K * n`). Matches R
/// `gamboostLSS::DirichletLSS(K)`.
///
/// - offset: `eta_k = min_i(w_i * y_{i,k})` (R's `min(w * y[, k])`).
/// - ngradient (param k): `alpha_k (psi(sum_j alpha_j) - psi(alpha_k) + log y_k)`.
/// - risk per obs: `-(lgamma(sum_j alpha_j) - sum_j lgamma(alpha_j)
///   + sum_j (alpha_j - 1) log y_j)`.
#[derive(Debug)]
pub struct DirichletLSS {
    k: usize,
}

impl DirichletLSS {
    /// Checked constructor for a Dirichlet family with `k` components.
    pub fn try_new(k: usize) -> Result<Self, MboostrsError> {
        if !(2..=DIRICHLET_PARAM_NAMES.len()).contains(&k) {
            return Err(MboostrsError::BadFamilyParameter(
                "DirichletLSS".to_string(),
                format!("K must be in 2..={}, got {k}", DIRICHLET_PARAM_NAMES.len()),
            ));
        }
        Ok(Self { k })
    }

    /// Construct a Dirichlet family with `k` compositional categories
    /// (2 ≤ k ≤ the length of `DIRICHLET_PARAM_NAMES`).
    ///
    /// Prefer [`Self::try_new`] at external input boundaries.
    pub fn new(k: usize) -> Self {
        Self::try_new(k).expect("DirichletLSS K outside the supported range")
    }

    /// Number of observations from the concatenated `K`-column response.
    #[inline]
    fn n_obs(&self, y: &[f64]) -> usize {
        y.len() / self.k
    }
}

impl LssFamily for DirichletLSS {
    fn name(&self) -> &'static str {
        "DirichletLSS"
    }

    fn n_params(&self) -> usize {
        self.k
    }

    fn response_span(&self) -> usize {
        self.k
    }

    fn param_names(&self) -> &'static [&'static str] {
        &DIRICHLET_PARAM_NAMES[..self.k]
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        // min_i (w_i * y_{i,k}); column k starts at k*n in the concatenated slice.
        let n = self.n_obs(y);
        (0..n)
            .map(|i| w[i] * y[k * n + i])
            .fold(f64::INFINITY, f64::min)
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = self.n_obs(y);
        for i in 0..n {
            let alpha_k = etas[k][i].exp();
            let sum_alpha: f64 = (0..self.k).map(|j| etas[j][i].exp()).sum();
            out[i] = alpha_k * (digamma(sum_alpha) - digamma(alpha_k) + y[k * n + i].ln());
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = self.n_obs(y);
        let mut s = 0.0f64;
        for i in 0..n {
            let mut sum_alpha = 0.0;
            let mut sum_lgamma = 0.0;
            let mut sum_term = 0.0;
            for j in 0..self.k {
                let alpha_j = etas[j][i].exp();
                sum_alpha += alpha_j;
                sum_lgamma += lgamma(alpha_j);
                sum_term += (alpha_j - 1.0) * y[j * n + i].ln();
            }
            s += w[i] * -(lgamma(sum_alpha) - sum_lgamma + sum_term);
        }
        s
    }

    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        for i in 0..eta.len() {
            out[i] = eta[i].exp(); // all params use the log link
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        if y.len() % self.k != 0 {
            return Err(MboostrsError::BadResponse(
                "DirichletLSS".to_string(),
                format!("response length must be a multiple of K = {}", self.k),
            ));
        }
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() || *v <= 0.0 || *v >= 1.0 {
                return Err(MboostrsError::BadResponse(
                    "DirichletLSS".to_string(),
                    format!("each composition entry must be in (0, 1); bad value at index {i}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(DirichletLSS { k: self.k })
    }
}

// ─── Survival AFT LSS families (LogNormal / LogLog / Weibull) ─────────────────
//
// These are accelerated-failure-time location-scale models for a right-censored
// response carried as a concatenated 2-column slice `y = [t0..t_{n-1}, d0..d_{n-1}]`
// (t = time, d = event indicator 1=event/0=censored; matches R's
// `model.response` ordering `c(time, status)`). `mu` has an identity link,
// `sigma = exp(eta_sigma)` a log link, and the standardized residual is
// `eta = (log t - mu) / sigma`. Each family differs only in the standard
// log-density `logf` and log-survival `logS`. Matches R gamboostLSS'
// {LogNormal,LogLog,Weibull}LSS.

/// log(1 + exp(x)), numerically stable.
#[inline]
fn ln1pexp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// log of the standard normal CDF, `log Phi(x)`, numerically stable in both
/// tails (matches R's `pnorm(x, log.p = TRUE)`). For the far-left tail it uses
/// the asymptotic expansion of `log Phi(x)`.
/// The Box-Cox standardized residual `z`, written to survive a small `nu`.
///
/// `((y/mu)^nu - 1) / (nu * sigma)` loses every significant digit as `nu`
/// approaches zero: the numerator is `1 + nu*l + ...` minus 1, so the leading
/// term cancels and what remains is noise scaled by `1/nu`. `expm1(nu * l)`
/// computes the same quantity directly and stays accurate, which is what lets
/// the general branch be used for any `nu` that is not exactly zero.
///
/// `h = phi(a) / Phi(a)` at `a = 1 / (sigma * |nu|)`, the correction the
/// Box-Cox-Cole-Green truncation contributes.
///
/// BCCG is defined on `y > 0`, so its Box-Cox transform maps to a normal
/// truncated below and the density carries a `1/Phi(a)` normalizer. That
/// normalizer depends on sigma and nu, so it appears in their gradients as
/// well as in the risk. It vanishes quickly as `a` grows -- `h` is below
/// 1e-11 by `a = 6.7` -- which is why omitting it was defensible for small
/// `sigma * |nu|` and wrong further out.
#[inline]
fn bccg_truncation_ratio(sigma: f64, nu: f64) -> f64 {
    if nu.abs() < 1e-10 {
        return 0.0;
    }
    let a = 1.0 / (sigma * nu.abs());
    let phi = (-0.5 * a * a).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 0.5 * libm::erfc(-a / std::f64::consts::SQRT_2);
    if cdf > 0.0 {
        phi / cdf
    } else {
        0.0
    }
}

fn log_ndtr(x: f64) -> f64 {
    if x > -1.0 {
        // Phi(x) = 0.5*erfc(-x/sqrt2); accurate and well away from 0 here.
        (0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)).ln()
    } else {
        // Phi(x) = 0.5*erfc(|x|/sqrt2). For moderately negative x erfc is fine;
        // for very negative x it underflows, so fall back to the asymptotic
        // log Phi(x) ~ -x^2/2 - ln(-x) - 0.5 ln(2pi) + ln(1 - 1/x^2 + 3/x^4 ...).
        let s = 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
        if s > 0.0 {
            s.ln()
        } else {
            let x2 = x * x;
            -0.5 * x2 - (-x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
                + (1.0 - 1.0 / x2 + 3.0 / (x2 * x2)).ln()
        }
    }
}

/// Regularized incomplete beta `I_x(a, b)`, Lentz continued-fraction treatment
/// matching the Python `_betainc` (which is checked against SciPy). Used by the
/// Student-t CDF.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (-x).ln_1p()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        return front * betacf(a, b, x) / a;
    }
    1.0 - front * betacf(b, a, 1.0 - x) / b
}

/// Continued fraction used by [`regularized_incomplete_beta`].
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const EPS: f64 = 3.0e-16;
    const TINY: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m2 = 2.0 * m as f64;
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Student-t CDF via the incomplete-beta identity `pt(q, df)`, matching R.
fn student_t_cdf(q: f64, df: f64) -> f64 {
    let x = df / (df + q * q);
    let ib = regularized_incomplete_beta(0.5 * df, 0.5, x);
    if q > 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// Regularized lower incomplete gamma `P(a, x)` — series for `x < a+1`, else
/// the continued fraction for the complement, matching R's `pgamma`/`pgamm` and
/// the Python `_gammainc_p`.
fn regularized_incomplete_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let log_front = -x + a * x.ln() - lgamma(a);
    if x < a + 1.0 {
        let mut term = 1.0 / a;
        let mut total = term;
        let mut n = a;
        for _ in 0..300 {
            n += 1.0;
            term *= x / n;
            total += term;
            if term.abs() < total.abs() * 3.0e-16 {
                break;
            }
        }
        return total * log_front.exp();
    }
    // Continued fraction for Q(a, x), then complement.
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1.0e-300;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=300 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1.0e-300 {
            d = 1.0e-300;
        }
        c = b + an / c;
        if c.abs() < 1.0e-300 {
            c = 1.0e-300;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 3.0e-16 {
            break;
        }
    }
    let q = log_front.exp() * h;
    1.0 - q
}

const SURV_LSS_PARAMS: &[&str] = &["mu", "sigma"];
const LOG_LO: f64 = -23.025850929940457; // log(1e-10)
const LOG_HI: f64 = 23.025850929940457; // log(1e10)

/// Which AFT distribution a [`SurvivalLSS`] uses. The negative log-likelihood,
/// gradients, and risk match R gamboostLSS' `{LogNormal,LogLog,Weibull}LSS`. The
/// auto-offsets are the *correct* risk minimizers (mu with sigma = 1, then sigma
/// with mu = the mu offset; see [`SurvivalLSS::offset`]) and intentionally differ
/// from gamboostLSS for LogLog (sigma) and Weibull (mu), whose offset routines
/// have upstream bugs documented in
/// `.remember/gamboostlss-survival-offset-bugs.md`. LogNormal matches exactly.
#[derive(Clone, Copy)]
enum SurvDist {
    LogNormal,
    LogLog,
    Weibull,
}

/// Shared accelerated-failure-time location-scale LSS family.
///
/// In the default (censored) mode the risk is the negative log-likelihood of
/// `log(T)`, with `y` holding `(time, status)` pairs stacked as `[t.., d..]`.
/// In the uncensored mode `y` is the plain positive response `[t..]`, every
/// observation is an event, and the risk is the negative log-likelihood of `T`
/// itself: the log-time density is multiplied by the Jacobian `1/t`, so the
/// risk gains `+ log(t)` per observation. That is the scale the other
/// distributional families (`GammaLSS`, `BCCGLSS`, ...) use, so the two kinds
/// of family can be compared by risk on a plain positive response.
pub struct SurvivalLSS {
    dist: SurvDist,
    uncensored: bool,
}

impl SurvivalLSS {
    /// AFT log-normal survival family.
    pub fn log_normal() -> Self {
        Self {
            dist: SurvDist::LogNormal,
            uncensored: false,
        }
    }
    /// AFT log-logistic survival family.
    pub fn log_log() -> Self {
        Self {
            dist: SurvDist::LogLog,
            uncensored: false,
        }
    }
    /// AFT Weibull survival family.
    pub fn weibull() -> Self {
        Self {
            dist: SurvDist::Weibull,
            uncensored: false,
        }
    }
    /// The same distribution applied to an uncensored response (every status 1).
    pub fn uncensored(mut self) -> Self {
        self.uncensored = true;
        self
    }

    /// Standard log-density at the standardized residual `e`.
    #[inline]
    fn logf(&self, e: f64) -> f64 {
        match self.dist {
            SurvDist::LogNormal => -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * e * e,
            SurvDist::LogLog => e - 2.0 * ln1pexp(e),
            SurvDist::Weibull => e - e.exp(),
        }
    }

    /// Standard log-survival at `e`.
    #[inline]
    fn logs(&self, e: f64) -> f64 {
        match self.dist {
            // 1 - Phi(e) = Phi(-e); use the stable log-CDF.
            SurvDist::LogNormal => log_ndtr(-e),
            SurvDist::LogLog => -ln1pexp(e),
            SurvDist::Weibull => -e.exp(),
        }
    }

    /// Time and status of observation `i`. An uncensored response has no status
    /// column, so every observation is an event.
    #[inline]
    fn obs(&self, y: &[f64], n: usize, i: usize) -> (f64, f64) {
        if self.uncensored {
            (y[i], 1.0)
        } else {
            (y[i], y[n + i])
        }
    }

    #[inline]
    fn nll(&self, eta_mu: f64, eta_sigma: f64, t: f64, d: f64) -> f64 {
        let e = (t.ln() - eta_mu) / eta_sigma.exp();
        let log_time_nll = -d * (self.logf(e) - eta_sigma) - (1.0 - d) * self.logs(e);
        if self.uncensored {
            // Density of T rather than of log(T): f_T(t) = f_logT(log t) / t.
            log_time_nll + t.ln()
        } else {
            log_time_nll
        }
    }
}

impl LssFamily for SurvivalLSS {
    fn name(&self) -> &'static str {
        match self.dist {
            SurvDist::LogNormal => "LogNormalLSS",
            SurvDist::LogLog => "LogLogLSS",
            SurvDist::Weibull => "WeibullLSS",
        }
    }

    fn n_params(&self) -> usize {
        2
    }

    fn response_span(&self) -> usize {
        if self.uncensored {
            1
        } else {
            2
        }
    }

    fn param_names(&self) -> &'static [&'static str] {
        SURV_LSS_PARAMS
    }

    fn offset(&self, k: usize, y: &[f64], w: &[f64], prior: &[f64]) -> f64 {
        let n = w.len();
        match k {
            0 => {
                // mu (identity link, log-time location) minimizing the family's
                // own NLL with sigma = 1. R uses interval [log 1e-10, log 1e10]
                // for LogNormal/LogLog but [0, max time] for Weibull.
                //
                // NOTE: gamboostLSS' WeibullLSS stores `log` of this location as
                // the offset (an upstream bug — the mu link is identity, not log;
                // see .remember/gamboostlss-survival-offset-bugs.md). We keep the
                // correct, un-logged location.
                let (lo, hi) = match self.dist {
                    SurvDist::Weibull => {
                        let max_t = (0..n).map(|i| y[i]).fold(0.0f64, f64::max);
                        (0.0, max_t)
                    }
                    _ => (LOG_LO, LOG_HI),
                };
                let risk = |mu: f64| -> f64 {
                    (0..n)
                        .map(|i| {
                            let (t, d) = self.obs(y, n, i);
                            w[i] * self.nll(mu, 0.0, t, d)
                        })
                        .sum()
                };
                minimize_brent(&risk, lo, hi, R_OPTIMIZE_TOL, 500)
            }
            1 => {
                // sigma (log link, scale) minimizing the family's own NLL with mu
                // fixed at the computed mu offset.
                //
                // NOTE: gamboostLSS' LogLog uses the standard-normal log-survival
                // for the censored term here (an upstream copy-paste bug; see
                // .remember/gamboostlss-survival-offset-bugs.md). We use the
                // correct log-logistic survival via the family's own `nll`.
                let mu = prior.first().copied().unwrap_or(0.0);
                let risk = |eta_sigma: f64| -> f64 {
                    (0..n)
                        .map(|i| {
                            let (t, d) = self.obs(y, n, i);
                            w[i] * self.nll(mu, eta_sigma, t, d)
                        })
                        .sum()
                };
                minimize_brent(&risk, LOG_LO, LOG_HI, R_OPTIMIZE_TOL, 500)
            }
            _ => unreachable!("survival LSS has 2 parameters"),
        }
    }

    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = w.len();
        for i in 0..n {
            let (t, d) = self.obs(y, n, i);
            let sigma = eta_sigma[i].exp();
            let e = (t.ln() - eta_mu[i]) / sigma;
            out[i] = match (self.dist, k) {
                (SurvDist::LogNormal, 0) => {
                    let ratio = (self.logf(e) - self.logs(e)).exp(); // phi/(1-Phi)
                    (d * e + (1.0 - d) * ratio) / sigma
                }
                (SurvDist::LogNormal, 1) => {
                    let ratio = (self.logf(e) - self.logs(e)).exp();
                    -(d - d * e * e + (d - 1.0) * e * ratio)
                }
                (SurvDist::LogLog, 0) => {
                    let nom = (-e).exp() + 1.0;
                    (d * (2.0 / nom - 1.0) + (1.0 - d) / nom) / sigma
                }
                (SurvDist::LogLog, 1) => -(d + d * e - (d + 1.0) * e / (1.0 + (-e).exp())),
                (SurvDist::Weibull, 0) => (d * (e.exp() - 1.0) + (1.0 - d) * e.exp()) / sigma,
                (SurvDist::Weibull, 1) => -(d * (e + 1.0) - e * e.exp()),
                _ => unreachable!("survival LSS has 2 parameters"),
            };
        }
    }

    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = w.len();
        (0..n)
            .map(|i| {
                let (t, d) = self.obs(y, n, i);
                w[i] * self.nll(eta_mu[i], eta_sigma[i], t, d)
            })
            .sum()
    }

    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta), // mu: identity
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp(); // sigma: log link
                }
            }
            _ => unreachable!("survival LSS has 2 parameters"),
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        if self.uncensored {
            for (i, &t) in y.iter().enumerate() {
                if !t.is_finite() || t <= 0.0 {
                    return Err(MboostrsError::BadResponse(
                        self.name().to_string(),
                        format!("need a finite response > 0 at obs {i}"),
                    ));
                }
            }
            return Ok(());
        }
        if y.len() % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                self.name().to_string(),
                "survival response must be an interleaved (time, status) pair per obs".to_string(),
            ));
        }
        let n = y.len() / 2;
        for i in 0..n {
            let (t, d) = (y[i], y[n + i]);
            if !t.is_finite() || t <= 0.0 || (d != 0.0 && d != 1.0) {
                return Err(MboostrsError::BadResponse(
                    self.name().to_string(),
                    format!("need time > 0 and status in {{0,1}} at obs {i}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(SurvivalLSS {
            dist: self.dist,
            uncensored: self.uncensored,
        })
    }
}

// ─── PoissonLSS (PO) — 1 param, log link ──────────────────────────

/// Poisson distribution (single parameter mu, mean, log link).
/// y >= 0 integer. Matches `gamlss.dist::PO()`.
pub struct PoissonLSS;

impl LssFamily for PoissonLSS {
    fn name(&self) -> &'static str {
        "PoissonLSS"
    }
    fn n_params(&self) -> usize {
        1
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "PoissonLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, _k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        if sw > 0.0 {
            (sy / sw).ln()
        } else {
            0.0
        }
    }
    fn ngradient(&self, _k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let mu = etas[0].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            out[i] = y[i] - mu[i].exp();
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let mu = etas[0].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let m = mu[i].exp();
            s += w[i] * (-y[i] * m.ln() + m + lgamma(y[i] + 1.0));
        }
        s
    }
    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        for i in 0..eta.len() {
            out[i] = eta[i].exp();
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(PoissonLSS)
    }
}

// ─── GeometricLSS (GEOM) — 1 param, log link ─────────────────────

/// Geometric distribution (single parameter mu, mean, log link).
/// y >= 0 integer. Matches `gamlss.dist::GEOM()`.
pub struct GeometricLSS;

impl LssFamily for GeometricLSS {
    fn name(&self) -> &'static str {
        "GeometricLSS"
    }
    fn n_params(&self) -> usize {
        1
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "GeometricLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, _k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        if sw > 0.0 {
            (sy / sw).ln()
        } else {
            0.0
        }
    }
    fn ngradient(&self, _k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            out[i] = y[i] - (y[i] + 1.0) * mu / (1.0 + mu);
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            s += w[i] * (-y[i] * mu.ln() + (y[i] + 1.0) * (1.0 + mu).ln());
        }
        s
    }
    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        for i in 0..eta.len() {
            out[i] = eta[i].exp();
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(GeometricLSS)
    }
}

// ─── BinomialLSS (BI) — 1 param, logit link ──────────────────────

/// Binomial distribution for proportion data (single parameter mu = probability,
/// logit link). `y` in `[0, 1]`, a proportion. Matches `gamlss.dist::BI()`.
pub struct BinomialLSS;

impl LssFamily for BinomialLSS {
    fn name(&self) -> &'static str {
        "BinomialLSS"
    }
    fn n_params(&self) -> usize {
        1
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v > 1.0 {
                return Err(MboostrsError::BadResponse(
                    "BinomialLSS".into(),
                    format!("y must be proportion in [0,1] at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, _k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        let p = if sw > 0.0 {
            (sy / sw).clamp(1e-8, 1.0 - 1e-8)
        } else {
            0.5
        };
        (p / (1.0 - p)).ln()
    }
    fn ngradient(&self, _k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let mu = 1.0 / (1.0 + (-eta_mu[i]).exp());
            out[i] = y[i] - mu;
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = 1.0 / (1.0 + (-eta_mu[i]).exp());
            let eps = 1e-15;
            s += w[i] * (-y[i] * (mu + eps).ln() - (1.0 - y[i]) * (1.0 - mu + eps).ln());
        }
        s
    }
    fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
        for i in 0..eta.len() {
            out[i] = 1.0 / (1.0 + (-eta[i]).exp());
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BinomialLSS)
    }
}

// ─── NegativeBinomialLSS (NBI) — 2 param, mu log, sigma log ─────

/// Negative Binomial type I (variance = mu + sigma * mu^2). Matches `gamlss.dist::NBI()`.
pub struct NegativeBinomialLSS;

impl LssFamily for NegativeBinomialLSS {
    fn name(&self) -> &'static str {
        "NegativeBinomialLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "NegativeBinomialLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                if sw > 0.0 {
                    (sy / sw).ln()
                } else {
                    0.0
                }
            }
            1 => {
                // gamlss NBI sigma.initial, as in the ZINBLSS offset:
                // max((var(y) - mean(y)) / mean(y)^2, 0.1), from variance
                // mu + sigma * mu^2; var has the n - 1 denominator, here the
                // weighted `sw - 1`.
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                let mean = sy / sw.max(1e-300);
                let ss = (0..n).fold(0.0, |s, i| s + w[i] * (y[i] - mean) * (y[i] - mean));
                let var = if sw > 1.0 { ss / (sw - 1.0) } else { 0.0 };
                let sigma = ((var - mean) / (mean * mean).max(1e-300)).max(0.1);
                sigma.ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("NBI has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    // NLL = -y*log(mu) + (y + 1/sig)*log(1 + sig*mu) + const
                    // d(NLL)/d(eta_mu) = -y + (y*sigma + 1)*mu/(1 + sigma*mu)
                    out[i] = y[i] - (y[i] * sig + 1.0) * mu / (1.0 + sig * mu);
                }
            }
            1 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    let t = sig * mu;
                    out[i] = y[i]
                        - (y[i] * sig + 1.0) * mu / (1.0 + t)
                        - digamma(y[i] + 1.0 / sig) / sig
                        + digamma(1.0 / sig) / sig
                        + (1.0 + t).ln() / sig;
                }
            }
            _ => unreachable!("NBI has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sig = eta_sigma[i].exp();
            let t = sig * mu;
            s += w[i]
                * (-y[i] * t.ln() + (t + 1.0).ln() * (y[i] + 1.0 / sig) - lgamma(y[i] + 1.0 / sig)
                    + lgamma(1.0 / sig)
                    + lgamma(y[i] + 1.0));
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("NBI has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(NegativeBinomialLSS)
    }
}

// ─── GumbelLSS (gamlss RG) — 2 param, identity/log links ───────

/// Gumbel distribution for maxima — the standard extreme-value type I
/// (location mu, identity link; scale sigma, log link). y ∈ ℝ.
///
/// Matches `gamlss.dist::RG()`, which gamlss labels "Reverse Gumbel", **not**
/// `gamlss.dist::GU()`. gamlss's naming is the reverse of the common one: its
/// `GU` is the Gumbel for *minima* (left-skewed), and its `RG` is the
/// right-skewed form almost everyone means by "Gumbel". This implementation is
/// the latter, so the mboostrs name is the conventional one and the gamlss
/// cross-reference is `RG`.
///
/// Verified numerically rather than by name: on
/// `y = [0.4, 1.3, -0.7, 2.1, 0.0]` with `mu = 0.5, sigma = 2`, this `risk()`
/// returns 9.042800226406863, which is `-sum(dRG(..., log = TRUE))` to all
/// printed digits; `-sum(dGU(...))` is 9.161943. The doc comment previously
/// claimed `GU()` parity, which was wrong in exactly the direction that would
/// have sent someone comparing against R to the wrong reference.
///
/// gamlss's `GU()` — the mirrored form — is not implemented.
pub struct GumbelLSS;

impl LssFamily for GumbelLSS {
    fn name(&self) -> &'static str {
        "GumbelLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(MboostrsError::BadResponse(
                    "GumbelLSS".into(),
                    format!("y must be finite at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                // mu: identity link, weighted mean
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                if sw > 0.0 {
                    sy / sw
                } else {
                    0.0
                }
            }
            1 => {
                // sigma: log link, approximate via std dev
                let (sw, sy, syy) = (0..n).fold((0.0, 0.0, 0.0), |(s, sy, syy), i| {
                    (s + w[i], sy + w[i] * y[i], syy + w[i] * y[i] * y[i])
                });
                let mean = sy / sw.max(1e-300);
                let var = (syy / sw - mean * mean).max(1e-12);
                // sigma: log link, offset = log(weighted.sd). The leading 0.5 was a
                // factor-of-2 bug (it produced 0.5*log(std) instead of log(std)),
                // inconsistent with GaussianLSS (lss.rs:635) and JSULSS (lss.rs:2973).
                var.sqrt().ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("GumbelLSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                for i in 0..n {
                    let sig = eta_sigma[i].exp();
                    out[i] = (1.0 - (-(y[i] - eta_mu[i]) / sig).exp()) / sig;
                }
            }
            1 => {
                for i in 0..n {
                    let x = y[i] - eta_mu[i];
                    let sig = eta_sigma[i].exp();
                    let z = (-x / sig).exp();
                    out[i] = x / sig * (1.0 - z) - 1.0;
                }
            }
            _ => unreachable!("GumbelLSS has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let x = y[i] - eta_mu[i];
            let sig = eta_sigma[i].exp();
            s += w[i] * ((-x / sig).exp() + x / sig + eta_sigma[i]);
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("GumbelLSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(GumbelLSS)
    }
}

// ─── Pareto2LSS (PARETO2) — 2 param, log link ──────────────────

/// Pareto type 2 (Lomax) distribution: mu = scale (log link), sigma = shape
/// (log link), y >= 0. Density `f(y) = sigma * mu^sigma / (y + mu)^(sigma+1)`,
/// so `NLL = (sigma+1)*ln(1 + y/mu) + ln(mu) - ln(sigma)`.
///
/// The shape parameterization differs from `gamlss.dist::PARETO2()`, which
/// uses `1/sigma`. For `y = [0.15, 0.8, 1.0, 2.6, 7.4]`, `mu = 1.6`, and
/// `sigma = 2.3`, this family's `risk()` is 10.305999389708, while the R
/// density evaluated at `sigma = 2.3` gives a negative log-likelihood of
/// 11.784358047059. Evaluating the R density at `sigma = 1/2.3` gives the
/// same distribution as this family's parameterization.
pub struct Pareto2LSS;

impl LssFamily for Pareto2LSS {
    fn name(&self) -> &'static str {
        "Pareto2LSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Pareto2LSS".into(),
                    format!("y must be >= 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
                if sw > 0.0 {
                    (sy / sw).ln().clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            }
            1 => {
                let (sw, sly) = (0..n).fold((0.0, 0.0), |(s, sly), i| {
                    (s + w[i], sly + w[i] * (y[i] / (y[i] + 1.0).max(1.0)).ln())
                });
                if sw > 0.0 {
                    (-sly / sw).ln().clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            }
            _ => unreachable!("Pareto2LSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // d(NLL)/d(eta_mu) = (mu - sig*y) / (mu+y)
                // ngradient_mu = (sig*y - mu) / (mu+y)
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    out[i] = (sig * y[i] - mu) / (mu + y[i]);
                }
            }
            1 => {
                // d(NLL)/d(eta_sig) = sig*ln(1+y/mu) - 1
                // ngradient_sig = 1 - sig*ln(1+y/mu)
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sig = eta_sigma[i].exp();
                    out[i] = 1.0 - sig * (1.0 + y[i] / mu).ln();
                }
            }
            _ => unreachable!("Pareto2LSS has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sig = eta_sigma[i].exp();
            // NLL = (sig+1)*ln(1+y/mu) + ln(mu) - ln(sig)
            s += w[i] * ((sig + 1.0) * (1.0 + y[i] / mu).ln() + mu.ln() - sig.ln());
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("Pareto2LSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(Pareto2LSS)
    }
}

// ─── ZeroAdjustedPoissonLSS (ZAP) — 2 param ─────────────────────

/// Zero-Adjusted Poisson (ZAP): mixture of a point mass at zero and a
/// zero-truncated Poisson. mu = mean of Poisson part (log link),
/// sigma = probability of zero (logit link). y >= 0 integer.
/// Matches `gamlss.dist::ZAP()`.
pub struct ZeroAdjustedPoissonLSS;

impl LssFamily for ZeroAdjustedPoissonLSS {
    fn name(&self) -> &'static str {
        "ZeroAdjustedPoissonLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(MboostrsError::BadResponse(
                    "ZeroAdjustedPoissonLSS".into(),
                    format!("y must be non-negative integer at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, _, sz) = (0..n).fold((0.0, 0.0, 0.0), |(sw, sy, sz), i| {
            (
                sw + w[i],
                sy + w[i] * y[i],
                sz + if y[i] < 0.5 { w[i] } else { 0.0 },
            )
        });
        match k {
            0 => {
                let nonzero_mean = (0..n)
                    .filter(|&i| y[i] > 0.5)
                    .fold((0.0, 0.0), |(s, sw), i| (s + w[i] * y[i], sw + w[i]));
                if nonzero_mean.1 > 0.0 {
                    (nonzero_mean.0 / nonzero_mean.1).ln()
                } else {
                    0.0
                }
            }
            1 => {
                let p0 = if sw > 0.0 { sz / sw } else { 0.0 };
                (p0 / (1.0 - p0).max(1e-10)).ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("ZAP has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                // ZAP mu gradient (log link):
                //   y=0: NLL = -ln(sig), does not depend on mu => ngradient = 0
                //   y>0: NLL = -ln(1-sig) + mu - y*ln(mu) + lgamma(y+1) - ln(1-exp(-mu))
                //         dNLL/dmu = 1 - y/mu + exp(-mu)/(1-exp(-mu))
                //   ngradient_mu = y/mu - 1 - exp(-mu)/(1-exp(-mu))
                //   But with log link eta_mu = ln(mu), d/d(eta_mu) = d/dm * mu
                //   So ngradient_eta_mu = (y/mu - 1 - exp(-mu)/(1-exp(-mu))) * mu
                //                      = y - mu - mu/(exp(mu)-1)
                for i in 0..n {
                    if y[i] < 0.5 {
                        out[i] = 0.0;
                    } else {
                        let mu = eta_mu[i].exp();
                        out[i] = y[i] - mu - mu / (mu.exp() - 1.0);
                    }
                }
            }
            1 => {
                // ZAP sigma gradient (logit link):
                //   y=0: NLL = -ln(sig), dNLL/dsig = -1/sig, ngradient_raw = 1/sig
                //   y>0: NLL = -ln(1-sig) + ..., dNLL/dsig = 1/(1-sig), ngradient_raw = -1/(1-sig)
                //   Chain rule: d/d(eta_sig) = d/d(sig) * sig*(1-sig)
                for i in 0..n {
                    let sig = 1.0 / (1.0 + (-eta_sigma[i]).exp());
                    if y[i] < 0.5 {
                        out[i] = 1.0 - sig; // (1/sig) * sig*(1-sig)
                    } else {
                        out[i] = -sig; // (-1/(1-sig)) * sig*(1-sig)
                    }
                }
            }
            _ => unreachable!("ZAP has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sig = 1.0 / (1.0 + (-eta_sigma[i]).exp());
            if y[i] < 0.5 {
                // P(Y=0) = sig  (point mass only, no Poisson zero)
                s += w[i] * (-sig.ln());
            } else {
                // P(Y=y) = (1-sig) * exp(-mu) * mu^y / (y! * (1-exp(-mu)))
                let lpois = y[i] * mu.ln() - mu - lgamma(y[i] + 1.0);
                s += w[i] * (-(1.0 - sig).ln() - lpois + (1.0 - (-mu).exp()).ln());
            }
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = 1.0 / (1.0 + (-eta[i]).exp());
                }
            }
            _ => unreachable!("ZAP has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(ZeroAdjustedPoissonLSS)
    }
}

/// `log K_{i - 1/2}(x)` for `i = 0 ..= order_max`, where `K` is the modified
/// Bessel function of the second kind.
///
/// Half-integer orders have a closed form and an exact three-term recurrence,
/// so no series or continued fraction is needed:
///
/// * `K_{1/2}(x) = K_{-1/2}(x) = sqrt(pi / 2x) exp(-x)`;
/// * `K_{v+1}(x) = K_{v-1}(x) + (2v / x) K_v(x)`.
///
/// The recurrence adds two positive terms, so it is run in logs -- `K_{y-1/2}`
/// overflows for a count of a few hundred at a small `x`, and a PIG fit on
/// count data reaches that.
///
/// Used by [`PIGLSS`], whose density needs `K_{y-1/2}` and whose gradient
/// needs the ratio `K_{y+1/2} / K_{y-1/2}`.
fn log_bessel_k_half_integer(x: f64, order_max: usize) -> Vec<f64> {
    let mut values = vec![f64::NEG_INFINITY; order_max + 1];
    if !(x.is_finite() && x > 0.0) {
        return values;
    }
    let base = 0.5 * (std::f64::consts::PI / (2.0 * x)).ln() - x;
    values[0] = base;
    if order_max == 0 {
        return values;
    }
    values[1] = base;
    for index in 2..=order_max {
        // `v = index - 3/2`, so this step produces order `index - 1/2` from
        // orders `index - 5/2` and `index - 3/2`.
        let v = index as f64 - 1.5;
        let scaled = (2.0 * v / x).ln() + values[index - 1];
        values[index] = log_add_exp(values[index - 2], scaled);
    }
    values
}

/// `ln(exp(a) + exp(b))` without overflowing.
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    high + (low - high).exp().ln_1p()
}

/// The PIG quantities every one of its methods needs, computed once.
///
/// `alpha = sqrt(1/sigma^2 + 2 mu / sigma)`, and
/// `ty = (mu / (alpha sigma)) K_{y+1/2}(alpha) / K_{y-1/2}(alpha)`, which is
/// what `gamlss.dist`'s `tofyPIG1` returns and what both of its derivatives
/// are written in terms of.
struct PigTerms {
    log_density: f64,
    ty: f64,
}

fn pig_terms(y: f64, mu: f64, sigma: f64) -> PigTerms {
    let count = y.max(0.0).round();
    let index = count as usize;
    let alpha = (1.0 / (sigma * sigma) + 2.0 * mu / sigma).sqrt();
    let logs = log_bessel_k_half_integer(alpha, index + 1);
    let log_k_lower = logs[index];
    let log_k_upper = logs[index + 1];
    let log_density =
        0.5 * (2.0 * alpha / std::f64::consts::PI).ln() + count * mu.ln() + 1.0 / sigma
            - count * (alpha * sigma).ln()
            - lgamma(count + 1.0)
            + log_k_lower;
    let ty = (mu / (alpha * sigma)) * (log_k_upper - log_k_lower).exp();
    PigTerms { log_density, ty }
}

// ─── PIGLSS (PIG) — 2 param, log links ─────────────────────────

/// Poisson-Inverse Gaussian (variance = mu + sigma * mu^2, so a larger
/// sigma means more overdispersion). y >= 0 integer. Matches
/// `gamlss.dist::PIG()`.
pub struct PIGLSS;

impl LssFamily for PIGLSS {
    fn name(&self) -> &'static str {
        "PIGLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (index, &value) in y.iter().enumerate() {
            if !value.is_finite() || value < 0.0 || value != value.round() {
                return Err(MboostrsError::BadResponse(
                    "PIGLSS".into(),
                    format!(
                        "y must be a non-negative whole count, got {value} at \
                         index {index}"
                    ),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        match k {
            0 => {
                if sw > 0.0 {
                    (sy / sw).ln()
                } else {
                    0.0
                }
            }
            1 => {
                // gamlss PIG sigma.initial: max((var(y) - mean(y)) / mean(y)^2,
                // 0.1), from variance = mu + sigma * mu^2; var has the n - 1
                // denominator, here the weighted `sw - 1`.
                let mean = sy / sw.max(1e-300);
                let sy2 = (0..n).fold(0.0, |s2, i| s2 + w[i] * (y[i] - mean) * (y[i] - mean));
                let var = if sw > 1.0 { sy2 / (sw - 1.0) } else { 0.0 };
                let sigma = ((var - mean) / (mean * mean).max(1e-300)).max(0.1);
                sigma.ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("PIG has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        // `gamlss.dist::PIG()` writes both derivatives in terms of one
        // quantity, `ty` from its `tofyPIG1`:
        //
        //   dl/dmu    = (y - ty) / mu
        //   dl/dsigma = (ty (1 + sigma mu) / mu - (1 + sigma y)) / sigma^2
        //
        // Both links are logs here, so the gradient with respect to eta picks
        // up `dmu/deta = mu` and `dsigma/deta = sigma`. The mu case reduces to
        // `y - ty` exactly, which is why the old approximation `y - mu` looked
        // plausible: it is this with `ty` replaced by the Poisson mean.
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let n = y.len();
        match k {
            0 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sigma = eta_sigma[i].exp();
                    out[i] = y[i] - pig_terms(y[i], mu, sigma).ty;
                }
            }
            1 => {
                for i in 0..n {
                    let mu = eta_mu[i].exp();
                    let sigma = eta_sigma[i].exp();
                    let ty = pig_terms(y[i], mu, sigma).ty;
                    out[i] = (ty * (1.0 + sigma * mu) / mu - (1.0 + sigma * y[i])) / sigma;
                }
            }
            _ => unreachable!("PIG has 2 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut total = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sigma = eta_sigma[i].exp();
            total -= w[i] * pig_terms(y[i], mu, sigma).log_density;
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("PIG has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(PIGLSS)
    }
}

// ─── BCCG (Box-Cox Cole-Green) — 3 param ───────────────────────

/// Box-Cox Cole-Green distribution (mu = median, sigma = CV, nu = skewness).
/// y > 0. Matches `gamlss.dist::BCCG()`.
pub struct BCCGLSS;

impl LssFamily for BCCGLSS {
    fn name(&self) -> &'static str {
        "BCCGLSS"
    }
    fn n_params(&self) -> usize {
        3
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "BCCGLSS".into(),
                    format!("y must be > 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            0 => {
                let (sw, sly) =
                    (0..n).fold((0.0, 0.0), |(s, sly), i| (s + w[i], sly + w[i] * y[i].ln()));
                if sw > 0.0 {
                    sly / sw
                } else {
                    0.0
                }
            }
            1 => {
                let (sw, sly) =
                    (0..n).fold((0.0, 0.0), |(s, sly), i| (s + w[i], sly + w[i] * y[i].ln()));
                let lmu = if sw > 0.0 { sly / sw } else { 0.0 };
                let slv = (0..n)
                    .map(|i| w[i] * (y[i].ln() - lmu) * (y[i].ln() - lmu))
                    .sum::<f64>();
                (slv / (sw - 1.0).max(1.0)).sqrt().ln().clamp(-5.0, 5.0)
            }
            2 => 0.0, // nu offset = 0 (symmetric)
            _ => unreachable!("BCCG has 3 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        match k {
            0 => {
                // mu link = log: mu = exp(eta_mu)
                // ngradient = -d(NLL)/d(eta_mu) = z/sig + nu*(z^2 - 1).
                //
                // The nu = 0 branch is the limit of that expression, which is
                // z/sig with z = log(y/mu)/sig -- the nu*(z^2 - 1) term simply
                // vanishes. It used to subtract a further 1, making the mu
                // gradient wrong by exactly 1 per observation at nu = 0. Since
                // nu's offset is 0, that is where every BCCG fit starts, so
                // the error was present from the first iteration and biased mu
                // for the whole path.
                for i in 0..n {
                    let mu = e_mu[i].exp();
                    let sig = e_sig[i].exp();
                    let nu = e_nu[i];
                    if nu.abs() > 1e-10 {
                        let z = ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig);
                        out[i] = z / sig + nu * (z * z - 1.0);
                    } else {
                        let z = (y[i] / mu).ln() / sig;
                        out[i] = z / sig;
                    }
                }
            }
            1 => {
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let mu = e_mu[i].exp();
                    let nu = e_nu[i];
                    let z = if nu.abs() > 1e-10 {
                        ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig)
                    } else {
                        (y[i] / mu).ln() / sig
                    };
                    // gamlss.dist::BCCG$dldd on the eta scale (log link, so
                    // multiplied by sigma): (z^2 - 1) + h / (sigma * |nu|).
                    let h = bccg_truncation_ratio(sig, nu);
                    out[i] = z * z - 1.0
                        + if nu.abs() > 1e-10 {
                            h / (sig * nu.abs())
                        } else {
                            0.0
                        };
                }
            }
            2 => {
                for i in 0..n {
                    let mu = e_mu[i].exp();
                    let sig = e_sig[i].exp();
                    let nu = e_nu[i];
                    if nu.abs() > 1e-10 {
                        let z = ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig);
                        let l = (y[i] / mu).ln();
                        // gamlss.dist::BCCG$dldv; nu has an identity link so
                        // the eta scale is the parameter scale.
                        let h = bccg_truncation_ratio(sig, nu);
                        out[i] = (z - l / sig) * (z / nu) - l * (z * z - 1.0)
                            + nu.signum() * h / (sig * nu * nu);
                    } else {
                        // nu = 0 is a removable singularity, not a stationary
                        // point. Expanding z = (exp(nu*l) - 1)/(nu*sigma) about
                        // nu = 0 gives z -> l/sigma and dz/dnu -> l^2/(2*sigma),
                        // so dl/dnu -> l - l^3/(2*sigma^2). The truncation term
                        // vanishes here because 1/(sigma*|nu|) -> infinity.
                        //
                        // Returning 0 instead made this a fixed point: nu's
                        // offset is 0, so a zero gradient there meant nu could
                        // never move and every BCCG fit was silently its
                        // nu = 0 special case, the log-normal.
                        let l = (y[i] / mu).ln();
                        out[i] = l - l * l * l / (2.0 * sig * sig);
                    }
                }
            }
            _ => unreachable!("BCCG has 3 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..n {
            let mu = e_mu[i].exp(); // log link
            let sig = e_sig[i].exp();
            let nu = e_nu[i];
            let z = if nu.abs() > 1e-10 {
                ((nu * (y[i] / mu).ln()).exp_m1()) / (nu * sig)
            } else {
                (y[i] / mu).ln() / sig
            };
            // NLL = 0.5*z^2 + ln(sig) + ln(mu) - (nu-1)*ln(y/mu) + ln(2*pi)/2
            // The ln(mu) is the Jacobian for the BCCG density on log(y) scale.
            //
            // The ln(2*pi)/2 was missing. It is constant in the parameters, so
            // no gradient and no selected path changes, but it made risk(),
            // logLik() and AIC differ from `gamlss.dist::dBCCG` by
            // n*0.9189385, and — the reason it matters — made BCCG's risk
            // incomparable with every other family's, which is exactly what a
            // family-ranking scan does. Same defect class as the log(y!) that
            // was missing from NegativeBinomialLSS.
            // The truncation normalizer, which depends on sigma and nu and so
            // also appears in their gradients above.
            let truncation = if nu.abs() > 1e-10 {
                let a = 1.0 / (sig * nu.abs());
                (0.5 * libm::erfc(-a / std::f64::consts::SQRT_2)).ln()
            } else {
                0.0
            };
            s += w[i]
                * (0.5 * z * z + e_sig[i] + e_mu[i] - (nu - 1.0) * (y[i] / mu).ln()
                    + 0.5 * (2.0 * std::f64::consts::PI).ln()
                    + truncation);
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => {
                // mu link = log
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            _ => unreachable!("BCCG has 3 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BCCGLSS)
    }
}

// ─── JSU (Johnson's SU) — 4 param ──────────────────────────────

/// Johnson's SU distribution. Centered parameterization matching
/// `gamlss.dist::JSU()`:
///
///   rtau = 1/tau
///   w = exp(rtau^2)                  (w=1 if rtau < 1e-7)
///   omega = -nu * rtau
///   c = 1/sqrt(0.5*(w-1)*(w*cosh(2*omega)+1))
///   mu_c = mu + c * sigma * sqrt(w) * sinh(omega)
///   sigma_c = c * sigma
///   z = (y - mu_c) / sigma_c
///   r = -nu + asinh(z) / rtau
///
/// f(y) = tau/(c * sigma * sqrt(2*pi)) * 1/sqrt(1+z^2) * exp(-0.5*r^2)
///
/// Links: mu=identity, sigma=log, nu=identity, tau=log.
pub struct JSULSS;

impl JSULSS {
    /// Common centering computation: returns (z, r, 1+z^2, sqrt(1+z^2), c, sig_c, ww, omega).
    #[inline]
    fn centered_z_r(
        y_i: f64,
        mu: f64,
        sig: f64,
        nu: f64,
        tau: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        let rtau = 1.0 / tau;
        let ww = if rtau < 1e-7 {
            1.0
        } else {
            (rtau * rtau).exp()
        };
        let omega = -nu * rtau;
        let c = if rtau < 1e-7 {
            1.0 / rtau
        } else {
            (0.5 * (ww - 1.0) * (ww * (2.0 * omega).cosh() + 1.0))
                .sqrt()
                .recip()
        };
        let sqrt_w = ww.sqrt();
        let mu_c = mu + c * sig * sqrt_w * omega.sinh();
        let sig_c = c * sig;
        let z = (y_i - mu_c) / sig_c;
        let r = -nu + z.asinh() / rtau;
        let s1 = 1.0 + z * z;
        let s1_sqrt = s1.sqrt();
        (z, r, s1, s1_sqrt, c, sig_c, ww, omega)
    }
}

impl LssFamily for JSULSS {
    fn name(&self) -> &'static str {
        "JSULSS"
    }
    fn n_params(&self) -> usize {
        4
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu", "tau"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(MboostrsError::BadResponse(
                    "JSULSS".into(),
                    format!("y must be finite at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy, _, syxx) = (0..n).fold((0.0, 0.0, 0.0, 0.0), |(sw, sy, _, sxx), i| {
            (sw + w[i], sy + w[i] * y[i], 0.0, sxx + w[i] * y[i] * y[i])
        });
        match k {
            0 => {
                if sw > 0.0 {
                    sy / sw
                } else {
                    0.0
                }
            }
            1 => {
                let mean = sy / sw.max(1e-300);
                let var = (syxx / sw - mean * mean).max(1e-12);
                0.5 * var.ln().clamp(-10.0, 10.0)
            }
            2 => 0.0,
            3 => 0.0,
            _ => unreachable!("JSU has 4 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        match k {
            0 => {
                // mu gradient: dldm = base / (c * sig), identity link
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, sig_c, _ww, _omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let base = z / s1 + r * tau / s1_sqrt; // r/(rtau*s1_sqrt) = r*tau/s1_sqrt
                    out[i] = base / sig_c;
                }
            }
            1 => {
                // sigma gradient: dldd_raw - 1 (since sigma.link = "log")
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, _sig_c, ww, omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let base = z / s1 + r * tau / s1_sqrt;
                    let dldd_raw = (z + ww.sqrt() * omega.sinh()) * base;
                    out[i] = dldd_raw - 1.0;
                }
            }
            2 => {
                // nu gradient: dldv (identity link)
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, _sig_c, ww, omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let rtau = 1.0 / tau;
                    let base = z / s1 + r * tau / s1_sqrt;
                    let dlogcdv =
                        rtau * ww * (2.0 * omega).sinh() / (ww * (2.0 * omega).cosh() + 1.0);
                    let dzdv =
                        -(z + ww.sqrt() * omega.sinh()) * dlogcdv + rtau * ww.sqrt() * omega.cosh();
                    out[i] = -dlogcdv - base * dzdv + r;
                }
            }
            3 => {
                // tau gradient: dldt * tau (since tau.link = "log")
                for i in 0..n {
                    let sig = e_sig[i].exp();
                    let tau = e_tau[i].exp();
                    let (z, r, s1, s1_sqrt, _c, _sig_c, ww, omega) =
                        Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
                    let rtau = 1.0 / tau;
                    let rtau2 = rtau * rtau;
                    let base = z / s1 + r * tau / s1_sqrt;
                    let denom = ww * (2.0 * omega).cosh() + 1.0;
                    let dlogcdt = -rtau * ww * (1.0 / (ww - 1.0) + (2.0 * omega).cosh() / denom)
                        + (e_nu[i] * ww * (2.0 * omega).sinh()) / denom;
                    let dzdt = -(z + ww.sqrt() * omega.sinh()) * dlogcdt
                        - rtau * ww.sqrt() * omega.sinh()
                        + e_nu[i] * ww.sqrt() * omega.cosh();
                    let dldt_inner = -dlogcdt - 1.0 / rtau - base * dzdt + r * (r + e_nu[i]) / rtau;
                    let dldt = -dldt_inner * rtau2;
                    out[i] = dldt * tau;
                }
            }
            _ => unreachable!("JSU has 4 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut s = 0.0;
        for i in 0..n {
            let sig = e_sig[i].exp();
            let tau = e_tau[i].exp();
            let (_z, r, s1, _s1_sqrt, c, _sig_c, _ww, _omega) =
                Self::centered_z_r(y[i], e_mu[i], sig, e_nu[i], tau);
            let log_c = c.ln();
            // NLL = log(sigma) + log(c) + log(rtau) + 0.5*(1+z^2).ln() + 0.5*r^2 + const
            //     = e_sig[i] + log_c - e_tau[i] + 0.5*s1.ln() + 0.5*r*r + half_log_2pi
            // where log(rtau) = -log(tau) = -e_tau[i].
            s += w[i] * (0.5 * r * r + e_sig[i] + log_c - e_tau[i] + 0.5 * s1.ln() + half_log_2pi);
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            3 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("JSU has 4 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(JSULSS)
    }
}

// ─── BCT (Box-Cox t) and BCPE (Box-Cox Power Exponential) — 4 param ──

/// The shared Box-Cox standardized variate behind BCT and BCPE:
///
///   z = ((y/mu)^nu - 1) / (nu sigma),          nu != 0
///   z = log(y/mu) / sigma,                    nu == 0
///
/// `nu` and the log ratio are precomputed by the callers (the `nu == 0` case is
/// handled by a guard whose threshold matches gamlss' `ifelse(nu != 0, ...)`).
#[inline]
fn boxcox_z(log_ratio: f64, nu: f64, sigma: f64) -> f64 {
    if nu.abs() < 1e-10 {
        log_ratio / sigma
    } else {
        ((nu * log_ratio).exp_m1()) / (nu * sigma)
    }
}

/// The log-normalizing constant `c` of the BCPE kernel, from `gamlss.dist`:
/// `log.c = 0.5*(-(2/tau) log2 + lgamma(1/tau) - lgamma(3/tau))`.
#[inline]
fn bcpe_log_c(tau: f64) -> f64 {
    0.5 * (-(2.0 / tau) * std::f64::consts::LN_2 + lgamma(1.0 / tau) - lgamma(3.0 / tau))
}

/// BCPE power-exponential kernel: `log f_T(t)`, matching `gamlss.dist`'s
/// inner `f.T(t, log = TRUE)`.
#[inline]
fn bcpe_log_f_t(t: f64, tau: f64, log_c: f64) -> f64 {
    tau.ln()
        - log_c
        - 0.5 * (t / log_c.exp()).abs().powf(tau)
        - (1.0 + 1.0 / tau) * std::f64::consts::LN_2
        - lgamma(1.0 / tau)
}

/// `|z/c|^power`, returning `0` at `z == 0` instead of the literal value.
///
/// BCPE's mu, nu and tau scores each multiply this factor by a term that
/// vanishes at least as fast as the power grows, so the product tends to zero
/// as `z → 0` for every `tau > 0`. Evaluated literally at `z == 0` — a
/// response exactly equal to its fitted `mu`, which the mean-valued mu offset
/// makes reachable on the first iteration — a negative `power` gives `inf`
/// and the product becomes NaN, poisoning the whole candidate scan. Returning
/// the limit keeps the score finite. For `tau < 1` the mu-score is genuinely
/// unbounded at the kernel's cusp, and zero is the subgradient there.
#[inline]
fn bcpe_kernel_power(z: f64, log_c: f64, power: f64) -> f64 {
    if z == 0.0 {
        return 0.0;
    }
    (z / log_c.exp()).abs().powf(power)
}

/// BCPE power-exponential CDF `F_T(t)`, matching `gamlss.dist`'s `F.T`:
/// `0.5*(1 + sign(t) * pgamma(0.5 |t/c|^tau, 1/tau))`.
#[inline]
fn bcpe_cdf_t(t: f64, tau: f64, log_c: f64) -> f64 {
    let s = 0.5 * (t / log_c.exp()).abs().powf(tau);
    let p = regularized_incomplete_gamma_p(1.0 / tau, s);
    0.5 * (1.0 + t.signum() * p)
}

/// BCT truncation ratio `h = dt(t0, tau) / pt(t0, tau)` at `t0 = 1/(sigma|nu|)`.
/// This is what gamlss' `dldd`/`dldv`/`dldt` add to fold in the truncation to
/// `y > 0`. Returns 0 at `nu == 0` (the truncation point recedes to infinity).
#[inline]
fn bct_truncation_ratio(sigma: f64, nu: f64, tau: f64) -> f64 {
    if nu.abs() < 1e-10 {
        return 0.0;
    }
    let t0 = 1.0 / (sigma * nu.abs());
    let log_density = lgamma((tau + 1.0) / 2.0)
        - lgamma(tau / 2.0)
        - 0.5 * (tau * std::f64::consts::PI).ln()
        - (tau + 1.0) / 2.0 * (1.0 + t0 * t0 / tau).ln();
    let cdf = student_t_cdf(t0, tau);
    (log_density - cdf.clamp(1e-300, 1.0).ln()).exp()
}

/// BCPE truncation ratio `h = f_T(t0) / F_T(t0)` at `t0 = 1/(sigma|nu|)`.
#[inline]
fn bcpe_truncation_ratio(sigma: f64, nu: f64, tau: f64) -> f64 {
    if nu.abs() < 1e-10 {
        return 0.0;
    }
    let log_c = bcpe_log_c(tau);
    let t0 = 1.0 / (sigma * nu.abs());
    let log_f = bcpe_log_f_t(t0, tau, log_c);
    let cdf = bcpe_cdf_t(t0, tau, log_c);
    (log_f - cdf.clamp(1e-300, 1.0).ln()).exp()
}

/// Box-Cox t distribution. `y > 0`. Links: mu=identity, sigma=log, nu=identity,
/// tau=log. Matches `gamlss.dist::BCT()`.
pub struct BCTLSS;

impl BCTLSS {
    /// Full per-observation negative log-density = -`dBCT(..., log = TRUE)`.
    #[inline]
    fn nll(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
        let log_ratio = (y / mu).ln();
        let z = boxcox_z(log_ratio, nu, sigma);
        let kernel = lgamma((tau + 1.0) / 2.0)
            - lgamma(tau / 2.0)
            - 0.5 * tau.ln()
            - lgamma(0.5)
            - (tau + 1.0) / 2.0 * (1.0 + z * z / tau).ln();
        let log_det = (nu - 1.0) * y.ln() - nu * mu.ln() - sigma.ln();
        let truncation = if nu.abs() < 1e-10 {
            0.0
        } else {
            student_t_cdf(1.0 / (sigma * nu.abs()), tau).ln()
        };
        -(log_det + kernel - truncation)
    }
}

impl LssFamily for BCTLSS {
    fn name(&self) -> &'static str {
        "BCTLSS"
    }
    fn n_params(&self) -> usize {
        4
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu", "tau"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "BCTLSS".into(),
                    format!("y must be > 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let sw: f64 = (0..n).map(|i| w[i]).sum();
        let mean = if sw > 0.0 {
            (0..n).map(|i| w[i] * y[i]).sum::<f64>() / sw
        } else {
            0.0
        };
        match k {
            0 => mean,
            1 => -std::f64::consts::LN_10, // sigma.initial = 0.1, log link
            2 => 0.5,                      // nu.initial = 0.5, identity link
            3 => std::f64::consts::LN_10,  // tau (= 10 on natural scale, log link)
            _ => unreachable!("BCT has 4 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        match k {
            0 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    out[i] = (w * z) / (mu * sigma) + (nu / mu) * (w * z * z - 1.0);
                }
            }
            1 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    let h = bct_truncation_ratio(sigma, nu, tau);
                    // log link: multiply the raw dldd by sigma.
                    let truncation = if nu.abs() < 1e-10 {
                        0.0
                    } else {
                        h / (sigma * sigma * nu.abs())
                    };
                    let dldd = (w * z * z - 1.0) / sigma + truncation;
                    out[i] = dldd * sigma;
                }
            }
            2 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    if nu.abs() < 1e-7 {
                        // The analytic expression has cancelling 1/nu terms.
                        // Differentiate the exact normalized density locally
                        // rather than turn the removable singularity into NaN.
                        let step = 1e-5;
                        let plus = Self::nll(y[i], mu, sigma, nu + step, tau);
                        let minus = Self::nll(y[i], mu, sigma, nu - step, tau);
                        out[i] = -(plus - minus) / (2.0 * step);
                    } else {
                        let h = bct_truncation_ratio(sigma, nu, tau);
                        out[i] = (w * z * z) / nu
                            - log_ratio * (w * z * z + (w * z) / (sigma * nu) - 1.0)
                            + nu.signum() * h / (sigma * nu * nu);
                    }
                }
            }
            3 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let w = (tau + 1.0) / (tau + z * z);
                    // j = d/dt log pt(t0, tau), by gamlss' forward
                    // difference at h=0.01.
                    let t0 = if nu.abs() < 1e-10 {
                        f64::INFINITY
                    } else {
                        1.0 / (sigma * nu.abs())
                    };
                    let j = if t0.is_finite() {
                        let inner = |df: f64| -> f64 { student_t_cdf(t0, df).ln() };
                        (inner(tau + 0.01) - inner(tau)) / 0.01
                    } else {
                        0.0
                    };
                    let dldt = -0.5 * (1.0 + z * z / tau).ln()
                        + (w * z * z) / (2.0 * tau)
                        + 0.5 * digamma((tau + 1.0) / 2.0)
                        - 0.5 * digamma(tau / 2.0)
                        - 1.0 / (2.0 * tau)
                        - j;
                    // tau log link: multiply by tau.
                    out[i] = dldt * tau;
                }
            }
            _ => unreachable!("BCT has 4 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..n {
            s += w[i] * Self::nll(y[i], e_mu[i], e_sig[i].exp(), e_nu[i], e_tau[i].exp());
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            3 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("BCT has 4 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BCTLSS)
    }
}

/// Box-Cox Power Exponential distribution. `y > 0`. Links: mu=identity,
/// sigma=log, nu=identity, tau=log. Matches `gamlss.dist::BCPE()`.
pub struct BCPELSS;

impl BCPELSS {
    /// Full per-observation negative log-density = -`dBCPE(..., log = TRUE)`.
    #[inline]
    fn nll(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
        let log_c = bcpe_log_c(tau);
        let log_ratio = (y / mu).ln();
        let z = boxcox_z(log_ratio, nu, sigma);
        let log_f_z = bcpe_log_f_t(z, tau, log_c);
        let truncation = if nu.abs() < 1e-10 {
            0.0
        } else {
            bcpe_cdf_t(1.0 / (sigma * nu.abs()), tau, log_c).ln()
        };
        let log_det = (nu - 1.0) * y.ln() - nu * mu.ln() - sigma.ln();
        -(log_det + log_f_z - truncation)
    }
}

impl LssFamily for BCPELSS {
    fn name(&self) -> &'static str {
        "BCPELSS"
    }
    fn n_params(&self) -> usize {
        4
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu", "tau"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "BCPELSS".into(),
                    format!("y must be > 0 at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let sw: f64 = (0..n).map(|i| w[i]).sum();
        let mean = if sw > 0.0 {
            (0..n).map(|i| w[i] * y[i]).sum::<f64>() / sw
        } else {
            0.0
        };
        match k {
            0 => mean,
            1 => -std::f64::consts::LN_10, // sigma.initial = 0.1, log link
            2 => 1.0,                      // nu.initial = 1, identity link
            3 => std::f64::consts::LN_2,   // tau (= 2 on natural scale, log link)
            _ => unreachable!("BCPE has 4 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        match k {
            0 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let c2 = (log_c.exp()).powi(2);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    out[i] = (tau / (2.0 * mu * sigma * c2))
                        * (z + sigma * nu * z * z)
                        * bcpe_kernel_power(z, log_c, tau - 2.0)
                        - nu / mu;
                }
            }
            1 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let h = bcpe_truncation_ratio(sigma, nu, tau);
                    let truncation = if nu.abs() < 1e-10 {
                        0.0
                    } else {
                        h / (sigma * sigma * nu.abs())
                    };
                    let dldd = (1.0 / sigma)
                        * ((tau / 2.0) * (z / log_c.exp()).abs().powf(tau) - 1.0)
                        + truncation;
                    out[i] = dldd * sigma;
                }
            }
            2 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let c2 = log_c.exp().powi(2);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    if nu.abs() < 1e-7 {
                        let step = 1e-5;
                        let plus = Self::nll(y[i], mu, sigma, nu + step, tau);
                        let minus = Self::nll(y[i], mu, sigma, nu - step, tau);
                        out[i] = -(plus - minus) / (2.0 * step);
                    } else {
                        let h = bcpe_truncation_ratio(sigma, nu, tau);
                        out[i] = -(tau / (2.0 * nu * c2))
                            * bcpe_kernel_power(z, log_c, tau - 2.0)
                            * z
                            * ((nu * z + 1.0 / sigma) * log_ratio - z)
                            + log_ratio
                            + nu.signum() * h / (sigma * nu * nu);
                    }
                }
            }
            3 => {
                for i in 0..n {
                    let mu = e_mu[i];
                    let sigma = e_sig[i].exp();
                    let nu = e_nu[i];
                    let tau = e_tau[i].exp();
                    let log_c = bcpe_log_c(tau);
                    let log_ratio = (y[i] / mu).ln();
                    let z = boxcox_z(log_ratio, nu, sigma);
                    let dlogc_dt = (1.0 / (2.0 * tau * tau))
                        * (2.0 * std::f64::consts::LN_2 - digamma(1.0 / tau)
                            + 3.0 * digamma(3.0 / tau));
                    // j = d/dt log F_T(t0, tau) at h=0.001 (matches gamlss dldt).
                    let t0 = if nu.abs() < 1e-10 {
                        f64::INFINITY
                    } else {
                        1.0 / (sigma * nu.abs())
                    };
                    let j = if t0.is_finite() {
                        let inner = |df: f64| -> f64 { bcpe_cdf_t(t0, df, bcpe_log_c(df)).ln() };
                        (inner(tau + 0.001) - inner(tau)) / 0.001
                    } else {
                        0.0
                    };
                    // `|z/c|^tau · ln|z/c|` also tends to zero as z → 0, but
                    // literally it is `0 · -inf`; take the limit there too.
                    let scaled_power = bcpe_kernel_power(z, log_c, tau);
                    let log_term = if z == 0.0 {
                        0.0
                    } else {
                        -0.5 * (z / log_c.exp()).abs().ln() * scaled_power
                    };
                    let dldt = 1.0 / tau
                        + log_term
                        + (1.0 / (tau * tau)) * (std::f64::consts::LN_2 + digamma(1.0 / tau))
                        + ((tau / 2.0) * scaled_power - 1.0) * dlogc_dt
                        - j;
                    out[i] = dldt * tau;
                }
            }
            _ => unreachable!("BCPE has 4 parameters"),
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let n = y.len();
        let e_mu = etas[0].as_slice().expect("eta contiguous");
        let e_sig = etas[1].as_slice().expect("eta contiguous");
        let e_nu = etas[2].as_slice().expect("eta contiguous");
        let e_tau = etas[3].as_slice().expect("eta contiguous");
        let mut s = 0.0;
        for i in 0..n {
            s += w[i] * Self::nll(y[i], e_mu[i], e_sig[i].exp(), e_nu[i], e_tau[i].exp());
        }
        s
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => out.copy_from_slice(eta),
            3 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("BCPE has 4 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(BCPELSS)
    }
}

// ─── InverseGaussianLSS (IG) — 2 param, log/log links ──────────

/// Inverse Gaussian distribution (mean mu, log link; dispersion sigma, log
/// link). y > 0. Matches `gamlss.dist::IG()`.
///
/// Density `f(y) = (2*pi*sigma^2*y^3)^(-1/2) * exp(-(y-mu)^2 / (2*mu^2*sigma^2*y))`,
/// so the negative log-likelihood is
/// `0.5*log(2*pi) + log(sigma) + 1.5*log(y) + (y-mu)^2/(2*mu^2*sigma^2*y)`.
///
/// The gradients below are `gamlss.dist::IG()`'s `dldm`/`dldd` with the log
/// link's chain rule applied (`d/deta = d/dparam * param`), verified against
/// the oracle before implementation.
pub struct InverseGaussianLSS;

impl LssFamily for InverseGaussianLSS {
    fn name(&self) -> &'static str {
        "InverseGaussianLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "InverseGaussianLSS".into(),
                    format!("y must be finite and strictly positive at {i}, got {v}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy) = (0..n).fold((0.0, 0.0), |(s, sy), i| (s + w[i], sy + w[i] * y[i]));
        let mean = if sw > 0.0 { sy / sw } else { 1.0 };
        match k {
            // mu: log link on the weighted mean.
            0 => mean.max(1e-10).ln().clamp(-10.0, 10.0),
            // sigma: log link. The IG dispersion satisfies Var = sigma^2 * mu^3,
            // so a moment start is sigma = sqrt(Var / mu^3).
            1 => {
                let syy = (0..n).fold(0.0, |acc, i| acc + w[i] * y[i] * y[i]);
                let var = (syy / sw.max(1e-300) - mean * mean).max(1e-12);
                let sigma = (var / mean.max(1e-10).powi(3)).max(1e-12).sqrt();
                sigma.ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("InverseGaussianLSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sigma = eta_sigma[i].exp();
            match k {
                // dl/dmu * mu = (y-mu)/(sigma^2 * mu^2)
                0 => out[i] = (y[i] - mu) / (sigma * sigma * mu * mu),
                // dl/dsigma * sigma = -1 + (y-mu)^2/(y * sigma^2 * mu^2)
                1 => {
                    let d = y[i] - mu;
                    out[i] = -1.0 + d * d / (y[i] * sigma * sigma * mu * mu);
                }
                _ => unreachable!("InverseGaussianLSS has 2 parameters"),
            }
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let half_log_two_pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut total = 0.0;
        for i in 0..y.len() {
            let mu = eta_mu[i].exp();
            let sigma = eta_sigma[i].exp();
            let d = y[i] - mu;
            total += w[i]
                * (half_log_two_pi
                    + eta_sigma[i]
                    + 1.5 * y[i].ln()
                    + d * d / (2.0 * mu * mu * sigma * sigma * y[i]));
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 | 1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("InverseGaussianLSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(InverseGaussianLSS)
    }
}

// ─── LogisticLSS (LO) — 2 param, identity/log links ────────────

/// Logistic distribution (location mu, identity link; scale sigma, log link).
/// y ∈ ℝ. Matches `gamlss.dist::LO()`.
///
/// With `z = (y - mu)/sigma`, the negative log-likelihood is
/// `z + log(sigma) + 2*log(1 + exp(-z))`. Heavier-tailed than Gaussian at the
/// same scale, so it is the usual drop-in when residuals are mildly
/// outlier-prone but symmetric.
pub struct LogisticLSS;

impl LssFamily for LogisticLSS {
    fn name(&self) -> &'static str {
        "LogisticLSS"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(MboostrsError::BadResponse(
                    "LogisticLSS".into(),
                    format!("y must be finite at {i}"),
                ));
            }
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        let (sw, sy, syy) = (0..n).fold((0.0, 0.0, 0.0), |(s, sy, syy), i| {
            (s + w[i], sy + w[i] * y[i], syy + w[i] * y[i] * y[i])
        });
        let mean = if sw > 0.0 { sy / sw } else { 0.0 };
        match k {
            0 => mean,
            1 => {
                // Var = sigma^2 * pi^2 / 3, so sigma = sd * sqrt(3)/pi.
                let var = (syy / sw.max(1e-300) - mean * mean).max(1e-12);
                let sigma = var.sqrt() * 3.0f64.sqrt() / std::f64::consts::PI;
                sigma.max(1e-12).ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("LogisticLSS has 2 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let sigma = eta_sigma[i].exp();
            let z = (y[i] - eta_mu[i]) / sigma;
            // exp(z)/(1+exp(z)) = logistic(z), computed stably for large |z|.
            let logistic = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let e = z.exp();
                e / (1.0 + e)
            };
            match k {
                // dl/dmu = (1/sigma)*(exp(z)-1)/(1+exp(z)) = (2*logistic - 1)/sigma
                0 => out[i] = (2.0 * logistic - 1.0) / sigma,
                // dl/dsigma * sigma = -1 - z + 2*z*logistic
                1 => out[i] = -1.0 - z + 2.0 * z * logistic,
                _ => unreachable!("LogisticLSS has 2 parameters"),
            }
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let mut total = 0.0;
        for i in 0..y.len() {
            let z = (y[i] - eta_mu[i]) / eta_sigma[i].exp();
            // log(1 + exp(-z)) via the numerically stable softplus.
            let softplus = if -z > 0.0 {
                -z + (1.0 + z.exp()).ln()
            } else {
                (1.0 + (-z).exp()).ln()
            };
            total += w[i] * (z + eta_sigma[i] + 2.0 * softplus);
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(eta),
            1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            _ => unreachable!("LogisticLSS has 2 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(LogisticLSS)
    }
}

// ─── ZeroAdjustedGammaLSS (ZAGA) — 3 param, log/log/logit ──────

/// Zero-adjusted Gamma (mu log, sigma log, nu logit). `y >= 0`, with `nu` the
/// probability of an exact zero. Matches `gamlss.dist::ZAGA()`.
///
/// This family models semicontinuous responses such as claim amounts, rainfall,
/// or household spending, which can be zero or positive and right-skewed. The
/// parameter `nu` models the probability of an exact zero; `mu` and `sigma`
/// model the positive response distribution.
///
/// Density is a two-part mixture:
/// `f(0) = nu`, and `f(y) = (1 - nu) * Gamma(y; mu, sigma)` for `y > 0`.
///
/// The `nu` gradient collapses under the logit link: `dl/dnu` is `1/nu` at a
/// zero and `-1/(1-nu)` otherwise, and multiplying by `dnu/deta = nu(1-nu)`
/// leaves `1 - nu` and `-nu`. Verified against `gamlss.dist` before
/// implementation, along with the other two and the risk.
pub struct ZeroAdjustedGammaLSS;

impl LssFamily for ZeroAdjustedGammaLSS {
    fn name(&self) -> &'static str {
        "ZeroAdjustedGammaLSS"
    }
    fn n_params(&self) -> usize {
        3
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["mu", "sigma", "nu"]
    }
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let mut positives = 0usize;
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() || v < 0.0 {
                return Err(MboostrsError::BadResponse(
                    "ZeroAdjustedGammaLSS".into(),
                    format!("y must be finite and non-negative at {i}, got {v}"),
                ));
            }
            if v > 0.0 {
                positives += 1;
            }
        }
        // The Gamma part is estimated from the positives alone, so an all-zero
        // response leaves mu and sigma unidentified. Say so here rather than
        // returning a fit whose two Gamma parameters mean nothing.
        if positives == 0 {
            return Err(MboostrsError::BadResponse(
                "ZeroAdjustedGammaLSS".into(),
                "y is all zeros; the Gamma part has no data to estimate mu and \
                 sigma from"
                    .to_string(),
            ));
        }
        Ok(())
    }
    fn offset(&self, k: usize, y: &[f64], w: &[f64], _prior: &[f64]) -> f64 {
        let n = y.len();
        match k {
            // mu, sigma: the Gamma part, from the positive observations only.
            0 | 1 => {
                let (sw, sy, syy) = (0..n)
                    .filter(|&i| y[i] > 0.0)
                    .fold((0.0, 0.0, 0.0), |(s, sy, syy), i| {
                        (s + w[i], sy + w[i] * y[i], syy + w[i] * y[i] * y[i])
                    });
                let mean = if sw > 0.0 { sy / sw } else { 1.0 };
                if k == 0 {
                    mean.max(1e-10).ln().clamp(-10.0, 10.0)
                } else {
                    // Gamma: Var = sigma^2 * mu^2, so sigma = sd / mean.
                    let var = (syy / sw.max(1e-300) - mean * mean).max(1e-12);
                    (var.sqrt() / mean.max(1e-10))
                        .max(1e-12)
                        .ln()
                        .clamp(-10.0, 10.0)
                }
            }
            // nu: logit of the observed zero proportion.
            2 => {
                let (sw, sz) = (0..n).fold((0.0, 0.0), |(s, sz), i| {
                    (s + w[i], sz + if y[i] == 0.0 { w[i] } else { 0.0 })
                });
                let p = (sz / sw.max(1e-300)).clamp(1e-6, 1.0 - 1e-6);
                (p / (1.0 - p)).ln().clamp(-10.0, 10.0)
            }
            _ => unreachable!("ZeroAdjustedGammaLSS has 3 parameters"),
        }
    }
    fn ngradient(&self, k: usize, y: &[f64], etas: &[Array1<f64>], _w: &[f64], out: &mut [f64]) {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_nu = etas[2].as_slice().expect("eta contiguous");
        for i in 0..y.len() {
            let is_zero = y[i] == 0.0;
            match k {
                0 => {
                    if is_zero {
                        out[i] = 0.0;
                    } else {
                        let mu = eta_mu[i].exp();
                        let sigma = eta_sigma[i].exp();
                        out[i] = (y[i] - mu) / (sigma * sigma * mu);
                    }
                }
                1 => {
                    if is_zero {
                        out[i] = 0.0;
                    } else {
                        let mu = eta_mu[i].exp();
                        let s2 = (2.0 * eta_sigma[i]).exp();
                        out[i] = (2.0 / s2)
                            * (y[i] / mu - y[i].ln() + mu.ln() + s2.ln() - 1.0 + digamma(1.0 / s2));
                    }
                }
                2 => {
                    // logistic(eta) computed stably.
                    let e = eta_nu[i];
                    let nu = if e >= 0.0 {
                        1.0 / (1.0 + (-e).exp())
                    } else {
                        let t = e.exp();
                        t / (1.0 + t)
                    };
                    out[i] = if is_zero { 1.0 - nu } else { -nu };
                }
                _ => unreachable!("ZeroAdjustedGammaLSS has 3 parameters"),
            }
        }
    }
    fn risk(&self, y: &[f64], etas: &[Array1<f64>], w: &[f64]) -> f64 {
        let eta_mu = etas[0].as_slice().expect("eta contiguous");
        let eta_sigma = etas[1].as_slice().expect("eta contiguous");
        let eta_nu = etas[2].as_slice().expect("eta contiguous");
        let mut total = 0.0;
        for i in 0..y.len() {
            let e = eta_nu[i];
            // log(nu) and log(1 - nu) from the logit eta, without forming nu
            // first: log(nu) = -softplus(-eta), log(1-nu) = -softplus(eta).
            let softplus = |v: f64| {
                if v > 0.0 {
                    v + (1.0 + (-v).exp()).ln()
                } else {
                    (1.0 + v.exp()).ln()
                }
            };
            if y[i] == 0.0 {
                total += w[i] * softplus(-e);
            } else {
                let mu = eta_mu[i].exp();
                let s2 = (2.0 * eta_sigma[i]).exp();
                let inv = 1.0 / s2;
                let log_gamma_density =
                    inv * (y[i] / (mu * s2)).ln() - y[i] / (mu * s2) - y[i].ln() - lgamma(inv);
                total += w[i] * (softplus(e) - log_gamma_density);
            }
        }
        total
    }
    fn response(&self, k: usize, eta: &[f64], out: &mut [f64]) {
        match k {
            0 | 1 => {
                for i in 0..eta.len() {
                    out[i] = eta[i].exp();
                }
            }
            2 => {
                for i in 0..eta.len() {
                    let e = eta[i];
                    out[i] = if e >= 0.0 {
                        1.0 / (1.0 + (-e).exp())
                    } else {
                        let t = e.exp();
                        t / (1.0 + t)
                    };
                }
            }
            _ => unreachable!("ZeroAdjustedGammaLSS has 3 parameters"),
        }
    }
    fn clone_box(&self) -> Box<dyn LssFamily> {
        Box::new(ZeroAdjustedGammaLSS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learners::prepare_scalar_bols;

    fn etas_from(mu: &[f64], es: &[f64]) -> Vec<Array1<f64>> {
        vec![Array1::from_vec(mu.to_vec()), Array1::from_vec(es.to_vec())]
    }

    fn simple_gaussian_lss_learners(n: usize, w: ArrayView1<f64>) -> Vec<Vec<PreparedLearner>> {
        let x = Array1::<f64>::linspace(-1.0, 1.0, n);
        vec![
            vec![prepare_scalar_bols(x.view(), w, 0.0, true, "mu", None)],
            vec![prepare_scalar_bols(x.view(), w, 0.0, true, "sigma", None)],
        ]
    }

    #[test]
    fn gaussianlss_offset_mu_sigma() {
        // mu = weighted mean; sigma = log(weighted.sd), var = sum(w(y-m)^2)/(sw-1).
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0; 5];
        let fam = GaussianLSS;
        assert!((fam.offset(0, &y, &w, &[]) - 3.0).abs() < 1e-12);
        // var = (4+1+0+1+4)/4 = 2.5; sd = sqrt(2.5); offset_sigma = 0.5*ln(2.5)
        let want = 0.5 * 2.5f64.ln();
        assert!((fam.offset(1, &y, &w, &[]) - want).abs() < 1e-12);
    }

    #[test]
    fn try_fit_lss_diverges_on_tiny_scale_response() {
        // GaussianLSS updates mu along (y - mu) / sigma^2. With sd(y) ~ 0.02
        // the effective per-cycle step is nu / sigma^2 ~ 250x the
        // least-squares increment: risk grows geometrically and the guard in
        // try_fit_lss must stop the path with a typed error naming the fix.
        let n = 200;
        let x = Array1::<f64>::linspace(-1.0, 1.0, n);
        let tiny_scale = |scale: f64| {
            let mx = x.mean().unwrap();
            let sd_x = (x.iter().map(|&v| (v - mx) * (v - mx)).sum::<f64>() / n as f64).sqrt();
            let y: Array1<f64> = x.iter().map(|&xi| 1.0 + scale * (xi - mx) / sd_x).collect();
            let w = Array1::ones(n);
            let ctrl = BoostControl::new(50, 0.1).unwrap();
            let mut learners = simple_gaussian_lss_learners(n, w.view());
            try_fit_lss(
                &mut learners,
                y.view(),
                w.view(),
                &GaussianLSS,
                &ctrl,
                None,
                None,
            )
        };

        let err = tiny_scale(0.02).expect_err("tiny-scale response must trip the divergence guard");
        let msg = format!("{err}");
        assert!(msg.contains("diverged"), "unexpected message: {msg}");
        assert!(
            msg.contains("standard deviation"),
            "unexpected message: {msg}"
        );

        // Same shape at a conventional scale must fit without tripping the
        // guard (healthy cyclic paths are non-increasing).
        assert!(tiny_scale(2.0).is_ok());
    }

    /// The divergence guard's policy, pinned where it is decided.
    mod lss_divergence_guard_policy {
        use super::*;

        /// A recoverable excursion passes: the risk climbs by factors of a
        /// few above its running minimum and comes back, which is what an
        /// aggressive `nu` does to the sigma equation before it recalibrates.
        /// The guard must admit every step of such a path.
        #[test]
        fn recoverable_excursions_pass_the_guard() {
            let mut best = 240.0_f64; // a typical Gaussian joint NLL
            let excursion = [242.0, 251.0, 268.0, 249.0, 239.5, 238.0];
            for (cycle, &risk) in excursion.iter().enumerate() {
                check_lss_cycle_risk(risk, &mut best, cycle, "cyclic").unwrap_or_else(|e| {
                    panic!("cycle {cycle}: recoverable excursion rejected: {e}")
                });
            }
            // The running minimum followed the recovery down.
            assert!((best - 238.0).abs() < 1e-12);
        }

        /// The fire boundary is the published increment, on both sides and
        /// against the constant itself rather than a restated copy of it.
        #[test]
        fn the_fire_boundary_is_the_published_increment() {
            let mut best = 240.0_f64;
            let just_under = 240.0 + LSS_DIVERGENCE_FACTOR * (240.0 + 1.0) * 0.999_999;
            check_lss_cycle_risk(just_under, &mut best, 0, "cyclic")
                .expect("an excursion under the increment must not fire");

            let mut best2 = 240.0_f64;
            let just_over = 240.0 + LSS_DIVERGENCE_FACTOR * (240.0 + 1.0) * 1.000_001;
            let err = check_lss_cycle_risk(just_over, &mut best2, 0, "cyclic")
                .expect_err("an excursion over the increment must fire");
            assert!(format!("{err}").contains("diverged"));
            // Firing does not corrupt the running minimum.
            assert!((best2 - 240.0).abs() < 1e-12);
        }

        /// LSS risks are signed: the Gaussian NLL goes negative once the
        /// variance drops below e^-1. The increment is absolute against
        /// `|min| + 1`, so an ordinary decrease from a negative minimum
        /// passes and a million-fold climb from one fires.
        #[test]
        fn signed_risks_use_an_absolute_increment() {
            let mut best = -12.0_f64;
            check_lss_cycle_risk(-12.7, &mut best, 0, "cyclic")
                .expect("a decrease from a negative minimum is ordinary");
            assert!((best - -12.7).abs() < 1e-12);

            let mut best2 = -12.0_f64;
            let climbed = -12.0 + LSS_DIVERGENCE_FACTOR * (12.0 + 1.0) * 1.1;
            assert!(
                check_lss_cycle_risk(climbed, &mut best2, 0, "cyclic").is_err(),
                "a million-fold climb from a negative minimum must fire"
            );
        }

        /// Non-finite risk stops the fit unconditionally, before the
        /// increment comparison could be made against garbage.
        #[test]
        fn non_finite_risk_always_stops() {
            let mut best = 240.0_f64;
            let err = check_lss_cycle_risk(f64::NAN, &mut best, 3, "cyclic")
                .expect_err("NaN risk must stop the fit");
            assert!(format!("{err}").contains("non-finite"));
            let err = check_lss_cycle_risk(f64::INFINITY, &mut best, 3, "cyclic")
                .expect_err("infinite risk must stop the fit");
            assert!(format!("{err}").contains("non-finite"));
        }

        /// What the policy boundary means at the fit level, on fixtures
        /// probed at the maximum `nu` the engine documents as usable:
        /// healthy cyclic paths never approach the increment. Measured
        /// across families and shapes, completed conventional-scale fits
        /// are non-increasing in joint risk, so the guard's headroom is
        /// the whole published increment, not a few percent of it.
        #[test]
        fn completed_conventional_fits_stay_far_below_the_increment() {
            let n = 150usize;
            let x = Array1::<f64>::linspace(-2.0, 2.0, n);
            let shapes: Vec<Array1<f64>> = vec![
                x.mapv(|v| 1.0 + 0.8 * v),
                x.mapv(|v| (0.4 * v).exp()),
                x.mapv(|v| 2.0 + 0.5 * v + 0.3 * (3.0 * v).sin()),
            ];
            for y in &shapes {
                let w = Array1::ones(n);
                let ctrl = BoostControl::new(60, 0.4).unwrap();
                let mut learners = simple_gaussian_lss_learners(n, w.view());
                let path = try_fit_lss(
                    &mut learners,
                    y.view(),
                    w.view(),
                    &GaussianLSS,
                    &ctrl,
                    None,
                    None,
                )
                .expect("conventional-scale fit at nu=0.4 must complete");
                let mut best = f64::INFINITY;
                for (cycle, &risk) in path.risk.iter().enumerate() {
                    let excursion = risk - best;
                    assert!(
                        excursion <= LSS_DIVERGENCE_FACTOR * (best.abs() + 1.0) * 0.001,
                        "cycle {cycle} excursion {excursion:.3e} approached the guard increment"
                    );
                    if risk < best {
                        best = risk;
                    }
                }
                assert!(path.risk.iter().all(|r| r.is_finite()));
            }
        }
    }

    #[test]
    fn try_fit_lss_diagonal_checkpoints_rejects_malformed_inputs() {
        let n = 6;
        let y = Array1::linspace(1.0, 2.0, n);
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(3, 0.1).unwrap();
        let family = GaussianLSS;

        // Wrong learner-set count → typed Err, not assert! panic.
        let mut wrong_param_count = simple_gaussian_lss_learners(n, w.view());
        wrong_param_count.pop();
        assert!(matches!(
            try_fit_lss_diagonal_checkpoints(
                &mut wrong_param_count,
                y.view(),
                w.view(),
                &family,
                &ctrl,
                &[2],
                None,
            ),
            Err(MboostrsError::LearnerSetup(_, _))
        ));

        // Valid inputs reproduce the panicking wrapper's snapshots exactly.
        let mut learners = simple_gaussian_lss_learners(n, w.view());
        let a = try_fit_lss_diagonal_checkpoints(
            &mut learners,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            &[0, 2],
            None,
        )
        .unwrap();
        let mut learners2 = simple_gaussian_lss_learners(n, w.view());
        // The wrapper is deprecated on the P0.8 schedule; this parity check
        // guards the shim until its 0.2.0 removal.
        #[allow(deprecated)]
        let b = fit_lss_diagonal_checkpoints(
            &mut learners2,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            &[0, 2],
            None,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn try_fit_lss_rejects_invalid_inputs_without_panicking() {
        let n = 6;
        let y = Array1::linspace(1.0, 2.0, n);
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(3, 0.1).unwrap();
        let family = GaussianLSS;

        let mut wrong_param_count = simple_gaussian_lss_learners(n, w.view());
        wrong_param_count.pop();
        assert!(try_fit_lss(
            &mut wrong_param_count,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            None,
            None,
        )
        .is_err());

        let mut empty_parameter = simple_gaussian_lss_learners(n, w.view());
        empty_parameter[1].clear();
        assert!(try_fit_lss(
            &mut empty_parameter,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            None,
            None,
        )
        .is_err());

        let mut learners = simple_gaussian_lss_learners(n, w.view());
        assert!(try_fit_lss(
            &mut learners,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            Some(&[1]),
            None,
        )
        .is_err());

        let mut learners = simple_gaussian_lss_learners(n, w.view());
        assert!(try_fit_lss(
            &mut learners,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            None,
            Some(&[0.0, f64::NAN]),
        )
        .is_err());

        let mut bad_y = y.clone();
        bad_y[2] = f64::NAN;
        let mut learners = simple_gaussian_lss_learners(n, w.view());
        assert!(try_fit_lss(
            &mut learners,
            bad_y.view(),
            w.view(),
            &family,
            &ctrl,
            None,
            None,
        )
        .is_err());

        let mut bad_w = w.clone();
        bad_w[1] = -1.0;
        let mut learners = simple_gaussian_lss_learners(n, w.view());
        assert!(try_fit_lss(
            &mut learners,
            y.view(),
            bad_w.view(),
            &family,
            &ctrl,
            None,
            None,
        )
        .is_err());

        let mut learners = simple_gaussian_lss_learners(n, w.view());
        assert!(try_fit_lss_noncyclic(
            &mut learners,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            Some(&[0.0, f64::NAN]),
        )
        .is_err());

        let mut empty_parameter = simple_gaussian_lss_learners(n, w.view());
        empty_parameter[0].clear();
        assert!(try_fit_lss_noncyclic(
            &mut empty_parameter,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            None,
        )
        .is_err());

        let learners = simple_gaussian_lss_learners(n, w.view());
        let no_oob = ndarray::Array2::ones((n, 2));
        assert!(try_cvrisk_lss(
            &learners,
            y.view(),
            w.view(),
            &family,
            &ctrl,
            no_oob.view(),
            &[vec![1, 1]],
        )
        .is_err());
    }

    #[test]
    fn gaussianlss_ngradient_and_risk_match_formula() {
        let y = [1.0, 2.0, 3.0];
        let w = [1.0, 1.0, 1.0];
        let mu = [0.5, 2.0, 2.5];
        let es = [0.1, -0.2, 0.0]; // eta_sigma
        let etas = etas_from(&mu, &es);
        let fam = GaussianLSS;

        let mut u = [0.0; 3];
        fam.ngradient(0, &y, &etas, &w, &mut u);
        for i in 0..3 {
            let want = (y[i] - mu[i]) * (-2.0 * es[i]).exp();
            assert!((u[i] - want).abs() < 1e-12);
        }
        fam.ngradient(1, &y, &etas, &w, &mut u);
        for i in 0..3 {
            let r = y[i] - mu[i];
            let want = -1.0 + r * r * (-2.0 * es[i]).exp();
            assert!((u[i] - want).abs() < 1e-12);
        }

        let half = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut want_risk = 0.0;
        for i in 0..3 {
            let r = y[i] - mu[i];
            want_risk += es[i] + half + 0.5 * r * r * (-2.0 * es[i]).exp();
        }
        assert!((fam.risk(&y, &etas, &w) - want_risk).abs() < 1e-10);
    }

    #[test]
    fn gaussianlss_matches_r_gamboostlss() {
        // Value-level parity vs R gamboostLSS::GaussianLSS() (R 4.5):
        // y=[1,2,3], mu_hat=[0.5,2,2.5], eta_sigma=[0.1,-0.2,0], w=1.
        let fam = GaussianLSS;
        let y = [1.0, 2.0, 3.0];
        let w = [1.0, 1.0, 1.0];
        let etas = etas_from(&[0.5, 2.0, 2.5], &[0.1, -0.2, 0.0]);

        let r_u_mu = [0.409365376538991, 0.0, 0.5];
        let r_u_sig = [-0.795317311730505, -1.0, -0.75];
        let r_risk = 2.88415694374877;

        let mut u = [0.0; 3];
        fam.ngradient(0, &y, &etas, &w, &mut u);
        for i in 0..3 {
            assert!((u[i] - r_u_mu[i]).abs() < 1e-12, "u_mu[{i}]");
        }
        fam.ngradient(1, &y, &etas, &w, &mut u);
        for i in 0..3 {
            assert!((u[i] - r_u_sig[i]).abs() < 1e-12, "u_sigma[{i}]");
        }
        assert!((fam.risk(&y, &etas, &w) - r_risk).abs() < 1e-11);

        // Weighted offsets vs R: y=[1..5], w=[1,2,1,3,1].
        let yy = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ww = [1.0, 2.0, 1.0, 3.0, 1.0];
        assert!((fam.offset(0, &yy, &ww, &[]) - 3.125).abs() < 1e-12);
        assert!((fam.offset(1, &yy, &ww, &[]) - 0.304688648747243).abs() < 1e-12);
    }

    #[test]
    fn gaussianlss_response_links() {
        let fam = GaussianLSS;
        let eta = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        fam.response(0, &eta, &mut out);
        assert_eq!(out, eta); // identity
        fam.response(1, &eta, &mut out);
        for i in 0..3 {
            assert!((out[i] - eta[i].exp()).abs() < 1e-12);
        }
    }

    #[test]
    fn gammalss_response_links_apply_log_link_once() {
        let fam = GammaLSS;
        let eta = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        for k in 0..2 {
            fam.response(k, &eta, &mut out);
            for i in 0..3 {
                assert!((out[i] - eta[i].exp()).abs() < 1e-12);
            }
        }
    }

    /// ngradient_k[j] must equal -d(risk)/d(eta_k[j]) (central difference), i.e.
    /// the analytic negative gradient is the exact derivative of the risk. This
    /// is the core correctness check for count/zero-inflated families whose
    /// gradients are intricate (and, for ZINB, where R's own gradient is buggy).
    fn assert_ngradient_matches_finite_diff(
        fam: &dyn LssFamily,
        y: &[f64],
        etas: &mut [Array1<f64>],
        n: usize,
    ) {
        let w = vec![1.0; n];
        let h = 1e-6;
        for k in 0..fam.n_params() {
            let mut ngr = vec![0.0; n];
            fam.ngradient(k, y, etas, &w, &mut ngr);
            for j in 0..n {
                let orig = etas[k][j];
                etas[k][j] = orig + h;
                let rp = fam.risk(y, etas, &w);
                etas[k][j] = orig - h;
                let rm = fam.risk(y, etas, &w);
                etas[k][j] = orig;
                let fd = -(rp - rm) / (2.0 * h);
                assert!(
                    (ngr[j] - fd).abs() < 1e-5 * (1.0 + ngr[j].abs()),
                    "{} param {k} obs {j}: ngradient {} != finite-diff {}",
                    fam.name(),
                    ngr[j],
                    fd
                );
            }
        }
        assert_hessian_matches_finite_diff(fam, y, etas, n);
    }

    /// `neg_hessian` against second differences of one row's risk (the
    /// negative log-likelihood), for every row and pair of parameters.
    fn assert_hessian_matches_finite_diff(
        fam: &dyn LssFamily,
        y: &[f64],
        etas: &mut [Array1<f64>],
        n: usize,
    ) {
        let k = fam.n_params();
        let mut hess = vec![0.0; n * k * k];
        fam.neg_hessian(y, etas, &mut hess);
        let h = 1e-4;
        for i in 0..n {
            let mut w = vec![0.0; n];
            w[i] = 1.0;
            for a in 0..k {
                for b in 0..k {
                    let mut at = |da: f64, db: f64| {
                        let (oa, ob) = (etas[a][i], etas[b][i]);
                        etas[a][i] += da;
                        etas[b][i] += db;
                        let r = fam.risk(y, etas, &w);
                        etas[a][i] = oa;
                        etas[b][i] = ob;
                        r
                    };
                    let fd = if a == b {
                        (at(h, 0.0) - 2.0 * at(0.0, 0.0) + at(-h, 0.0)) / (h * h)
                    } else {
                        (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4.0 * h * h)
                    };
                    let got = hess[i * k * k + a * k + b];
                    assert!(
                        (got - fd).abs() < 1e-3 * (1.0 + fd.abs()),
                        "{} obs {i} d2/d eta_{a} d eta_{b}: neg_hessian {got} != finite-diff {fd}",
                        fam.name()
                    );
                }
            }
        }
    }

    #[test]
    fn ngradient_and_neg_hessian_match_finite_diff_for_remaining_families() {
        let n = 6;
        let eta = |base: f64| Array1::from_shape_fn(n, |i| base + 0.15 * (i as f64) - 0.3);
        let positive = [0.7, 1.9, 3.2, 0.4, 2.5, 1.1];
        let counts = [0.0, 3.0, 1.0, 7.0, 2.0, 5.0];
        let unit = [0.2, 0.55, 0.8, 0.35, 0.9, 0.1];
        let real = [-0.8, 1.3, 0.2, 2.4, -1.5, 0.6];
        type Case<'a> = (Box<dyn LssFamily>, &'a [f64], Vec<Array1<f64>>);
        let cases: Vec<Case> = vec![
            (Box::new(GaussianLSS), &real, vec![eta(0.2), eta(-0.1)]),
            (Box::new(GammaLSS), &positive, vec![eta(0.3), eta(-0.5)]),
            (
                Box::new(StudentTLSS),
                &real,
                vec![eta(0.1), eta(-0.2), eta(1.2)],
            ),
            (Box::new(BetaLSS), &unit, vec![eta(0.1), eta(1.5)]),
            (Box::new(NBinomialLSS), &counts, vec![eta(1.0), eta(-0.4)]),
            (Box::new(PIGLSS), &counts, vec![eta(1.0), eta(-0.8)]),
        ];
        // The gradient check also runs the Hessian check.
        for (fam, y, mut etas) in cases {
            assert_ngradient_matches_finite_diff(fam.as_ref(), y, &mut etas, n);
        }
        // BCTLSS has per-parameter gradient tests of its own.
        let mut bct_etas = vec![eta(0.4), eta(-1.0), eta(0.3), eta(2.0)];
        assert_hessian_matches_finite_diff(&BCTLSS, &positive, &mut bct_etas, n);
    }

    #[test]
    fn survival_lss_ngradient_matches_finite_diff() {
        let n = 6;
        // Censored response [time.., status..]; uncensored response [time..].
        let times = [0.7, 1.9, 3.2, 0.4, 2.5, 1.1];
        let censored: Vec<f64> = times
            .iter()
            .copied()
            .chain([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
            .collect();
        let eta_mu = Array1::from(vec![-0.2, 0.5, 1.0, -0.6, 0.3, 0.1]);
        let eta_sigma = Array1::from(vec![-0.3, 0.2, -0.1, 0.4, -0.5, 0.0]);
        for fam in [
            SurvivalLSS::log_normal(),
            SurvivalLSS::log_log(),
            SurvivalLSS::weibull(),
        ] {
            let mut etas = vec![eta_mu.clone(), eta_sigma.clone()];
            assert_ngradient_matches_finite_diff(&fam, &censored, &mut etas, n);
            let fam = fam.uncensored();
            let mut etas = vec![eta_mu.clone(), eta_sigma.clone()];
            assert_ngradient_matches_finite_diff(&fam, &times, &mut etas, n);
        }
    }

    fn assert_param_ngradient_matches_finite_diff(
        fam: &dyn LssFamily,
        y: &[f64],
        etas: &mut [Array1<f64>],
        n: usize,
        k: usize,
    ) {
        let w = vec![1.0; n];
        let h = 1e-6;
        let mut ngr = vec![0.0; n];
        fam.ngradient(k, y, etas, &w, &mut ngr);
        for j in 0..n {
            let orig = etas[k][j];
            etas[k][j] = orig + h;
            let rp = fam.risk(y, etas, &w);
            etas[k][j] = orig - h;
            let rm = fam.risk(y, etas, &w);
            etas[k][j] = orig;
            let fd = -(rp - rm) / (2.0 * h);
            assert!(
                (ngr[j] - fd).abs() < 1e-5 * (1.0 + ngr[j].abs()),
                "{} param {k} obs {j}: ngradient {} != finite-diff {}",
                fam.name(),
                ngr[j],
                fd
            );
        }
    }

    /// Every family `make_family` builds, with a response in its domain laid
    /// out as the fit takes it (column-major when a row has several values).
    fn row_nll_fixtures() -> Vec<(&'static str, usize, Vec<f64>)> {
        let real = vec![-1.2, -0.3, 0.4, 1.1, 2.0];
        let positive = vec![0.3, 0.8, 1.2, 2.5, 4.0];
        let counts = vec![0.0, 1.0, 2.0, 5.0, 0.0];
        let with = |a: &[f64], b: &[f64]| [a, b].concat();
        vec![
            ("gaussianlss", 2, real.clone()),
            (
                "gaussianknownvarlss",
                2,
                with(&real, &[0.1, 0.2, 0.0, 0.1, 0.5]),
            ),
            ("gammalss", 2, positive.clone()),
            ("studenttlss", 3, real.clone()),
            ("betalss", 2, vec![0.1, 0.3, 0.5, 0.7, 0.9]),
            ("nbinomiallss", 2, counts.clone()),
            (
                "lognormallss",
                2,
                with(&positive, &[1.0, 0.0, 1.0, 1.0, 0.0]),
            ),
            ("logloglss", 2, with(&positive, &[1.0, 0.0, 1.0, 1.0, 0.0])),
            ("weibulllss", 2, with(&positive, &[1.0, 0.0, 1.0, 1.0, 0.0])),
            ("lognormallss_uncensored", 2, positive.clone()),
            ("logloglss_uncensored", 2, positive.clone()),
            ("weibulllss_uncensored", 2, positive.clone()),
            ("bccglss", 3, positive.clone()),
            ("binomiallss", 1, vec![0.0, 1.0, 1.0, 0.0, 1.0]),
            ("geometriclss", 1, counts.clone()),
            ("gumbellss", 2, real.clone()),
            ("inversegaussianlss", 2, positive.clone()),
            ("logisticlss", 2, real.clone()),
            ("zeroadjustedgammalss", 3, vec![0.0, 0.5, 1.5, 0.0, 3.0]),
            ("jsulss", 4, real.clone()),
            ("bctlss", 4, positive.clone()),
            ("bcpelss", 4, positive.clone()),
            ("negativebinomiallss", 2, counts.clone()),
            ("pareto2lss", 2, positive.clone()),
            ("piglss", 2, counts.clone()),
            ("poissonlss", 1, counts.clone()),
            ("zeroadjustedpoissonlss", 2, counts.clone()),
            ("zipolss", 2, counts.clone()),
            ("zinblss", 3, counts.clone()),
            (
                "dirichletlss",
                3,
                vec![
                    0.2, 0.5, 0.1, 0.3, 0.6, // component 1
                    0.3, 0.2, 0.6, 0.3, 0.1, // component 2
                    0.5, 0.3, 0.3, 0.4, 0.3, // component 3
                ],
            ),
        ]
    }

    #[test]
    fn row_nll_sums_to_the_weighted_risk_for_every_family() {
        let n = 5;
        let w = [0.5, 1.0, 2.0, 0.0, 1.5];
        for (name, k, y) in row_nll_fixtures() {
            let family = make_family(name, k).unwrap();
            family
                .check_y(ArrayView1::from(&y))
                .unwrap_or_else(|e| panic!("{name} fixture: {e}"));
            // The first parameter sits near 1, inside the domain of the
            // families (BCT, BCPE, BCCG) whose location has an identity link.
            let etas: Vec<Array1<f64>> = (0..k)
                .map(|p| {
                    let base = if p == 0 { 0.8 } else { -0.2 + 0.05 * p as f64 };
                    Array1::from_iter((0..n).map(|i| base + 0.1 * i as f64))
                })
                .collect();
            let (rows, parameters) = row_nll_and_parameters(family.as_ref(), &y, &etas).unwrap();
            assert_eq!(rows.len(), n);
            assert_eq!(parameters.len(), k);
            assert!(rows.iter().all(|v| v.is_finite()), "{name}: {rows:?}");
            let summed: f64 = rows.iter().zip(&w).map(|(r, w)| w * r).sum();
            let risk = family.risk(&y, &etas, &w);
            assert!(
                (summed - risk).abs() <= 1e-12 * (1.0 + risk.abs()),
                "{name}: rows sum to {summed}, risk is {risk}"
            );
        }
    }

    #[test]
    fn row_nll_refuses_a_response_of_the_wrong_length() {
        let etas = vec![Array1::zeros(3), Array1::zeros(3)];
        let err = row_nll_and_parameters(&GaussianKnownVarLSS, &[0.0, 1.0, 2.0], &etas)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 6"), "{err}");
        let err = row_nll_and_parameters(&GaussianLSS, &[0.0, 1.0, 2.0], &etas[..1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 parameters"), "{err}");
    }

    #[test]
    fn zinb_and_zipo_ngradients_match_finite_diff() {
        // Count response with zeros (exercises the zero-inflation branch).
        let y = [0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0, 0.0, 4.0];
        let n = y.len();
        // A non-trivial, per-observation eta config (not all equal) so the
        // zero-inflation density genuinely varies across observations.
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.3 + 0.1 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.2 + 0.05 * i as f64).collect();
        let eta_nu: Vec<f64> = (0..n).map(|i| -1.0 + 0.07 * i as f64).collect();

        let mut etas2 = vec![
            Array1::from_vec(eta_mu.clone()),
            Array1::from_vec(eta_sigma.clone()),
        ];
        assert_ngradient_matches_finite_diff(&ZIPoLSS, &y, &mut etas2, n);

        let mut etas3 = vec![
            Array1::from_vec(eta_mu),
            Array1::from_vec(eta_sigma),
            Array1::from_vec(eta_nu),
        ];
        assert_ngradient_matches_finite_diff(&ZINBLSS, &y, &mut etas3, n);
    }

    #[test]
    fn dirichlet_checked_constructor_rejects_invalid_dimensions() {
        for k in [0, 1, DIRICHLET_PARAM_NAMES.len() + 1] {
            let error = DirichletLSS::try_new(k).expect_err("invalid K must be rejected");
            assert!(matches!(error, MboostrsError::BadFamilyParameter(_, _)));
        }
        assert_eq!(DirichletLSS::try_new(3).unwrap().response_span(), 3);
        assert_eq!(SurvivalLSS::log_normal().response_span(), 2);
        assert_eq!(GaussianLSS.response_span(), 1);
    }

    #[test]
    fn dirichlet_lss_matches_r() {
        // K=3 composition response, concatenated columns [c0.., c1.., c2..].
        // Golden offset/risk/ngradient from R gamboostLSS::DirichletLSS(K=3).
        let cols = [
            [0.2, 0.1, 0.4, 0.25, 0.5], // a1 column
            [0.3, 0.6, 0.4, 0.25, 0.2], // a2 column
            [0.5, 0.3, 0.2, 0.50, 0.3], // a3 column
        ];
        let n = 5;
        let mut y = Vec::with_capacity(3 * n);
        for c in &cols {
            y.extend_from_slice(c);
        }
        let w = vec![1.0; n];
        let fam = DirichletLSS::new(3);
        for (k, want) in [0.1, 0.2, 0.2].into_iter().enumerate() {
            assert!(
                (fam.offset(k, &y, &w, &[]) - want).abs() < 1e-12,
                "offset a{k}"
            );
        }

        let eta = [
            [0.1, -0.2, 0.0, 0.3, -0.1],
            [0.0, 0.2, -0.1, 0.1, 0.2],
            [-0.2, 0.1, 0.3, 0.0, -0.1],
        ];
        let etas: Vec<Array1<f64>> = eta.iter().map(|e| Array1::from_vec(e.to_vec())).collect();
        assert!(
            (fam.risk(&y, &etas, &w) - (-3.233386284461)).abs() < 1e-9,
            "risk"
        );

        let want_ngr = [
            0.2655182094,
            0.8916529383,
            0.7673496898,
            0.1307016703,
            -0.5034695058,
        ];
        let mut ngr = vec![0.0; n];
        fam.ngradient(1, &y, &etas, &w, &mut ngr); // param a2 (k=1)
        for i in 0..n {
            assert!((ngr[i] - want_ngr[i]).abs() < 1e-9, "ngradient a2 obs {i}");
        }

        // And the analytic gradient is the exact derivative of the risk.
        let mut etas_mut = etas.clone();
        assert_ngradient_matches_finite_diff(&fam, &y, &mut etas_mut, n);
    }

    #[test]
    fn zinb_lss_offsets() {
        // mu = log(mean y); sigma = log(max((var - mean)/mean^2, 0.1)); nu =
        // qlogis(((#zeros/n) + 0.01)/2).
        let y = [0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0, 0.0, 4.0];
        let w = [1.0; 10];
        let n = y.len() as f64;
        let mean_y = y.iter().sum::<f64>() / n;
        let var_y = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>() / (n - 1.0);
        let fam = ZINBLSS;
        assert!((fam.offset(0, &y, &w, &[]) - mean_y.ln()).abs() < 1e-12);
        let sig_init = ((var_y - mean_y) / (mean_y * mean_y)).max(0.1);
        assert!((fam.offset(1, &y, &w, &[]) - sig_init.ln()).abs() < 1e-12);
        let prop0 = 4.0 / 10.0;
        assert!((fam.offset(2, &y, &w, &[]) - qlogis((prop0 + 0.01) / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn bccglss_ngradient_matches_finite_diff() {
        let y = [0.5, 1.2, 2.0, 3.5, 0.8, 4.1, 1.7];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.3 + 0.1 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.2 + 0.05 * i as f64).collect();
        let eta_nu: Vec<f64> = (0..n).map(|i| 0.1 + 0.03 * i as f64).collect();
        let mut etas = vec![
            Array1::from_vec(eta_mu),
            Array1::from_vec(eta_sigma),
            Array1::from_vec(eta_nu),
        ];
        assert_ngradient_matches_finite_diff(&BCCGLSS, &y, &mut etas, n);
    }

    #[test]
    fn jsulss_ngradient_matches_finite_diff() {
        let y = [-1.5, 0.2, 1.0, -0.5, 2.3, -0.8, 0.0, 1.8];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.2 + 0.08 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.1 + 0.04 * i as f64).collect();
        let eta_nu: Vec<f64> = (0..n).map(|i| 0.05 * i as f64).collect();
        let eta_tau: Vec<f64> = (0..n).map(|i| 0.1 + 0.02 * i as f64).collect();
        let mut etas = vec![
            Array1::from_vec(eta_mu),
            Array1::from_vec(eta_sigma),
            Array1::from_vec(eta_nu),
            Array1::from_vec(eta_tau),
        ];
        assert_ngradient_matches_finite_diff(&JSULSS, &y, &mut etas, n);
    }

    #[test]
    fn bctlss_ngradient_matches_finite_diff() {
        // y > 0; nu away from 0/large so the Box-Cox transform is well-behaved,
        // tau modest so the truncation ratio is not degenerate.
        let y = [0.5, 1.2, 2.0, 3.5, 0.8, 4.1, 1.7, 2.4];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.6 + 0.1 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.3 + 0.04 * i as f64).collect();
        let eta_nu: Vec<f64> = (0..n).map(|i| 0.4 + 0.03 * i as f64).collect();
        let eta_tau: Vec<f64> = (0..n).map(|i| 0.2 + 0.02 * i as f64).collect();
        let mut etas = vec![
            Array1::from_vec(eta_mu),
            Array1::from_vec(eta_sigma),
            Array1::from_vec(eta_nu),
            Array1::from_vec(eta_tau),
        ];
        // Finite-diff checks mu/sigma/nu. tau is excluded: gamlss' `dldt`
        // carries a `j` term computed by its own forward difference
        // `(log pt(t0, tau+0.01) - log pt(t0, tau))/0.01`, so R's analytic dldt
        // differs from the true derivative by ~0.7% there. Reproducing that
        // (rather than the exact derivative) is what keeps the tau path
        // bit-compatible with R, so tau is pinned by the R-parity suite instead.
        for k in 0..3 {
            assert_param_ngradient_matches_finite_diff(&BCTLSS, &y, &mut etas, n, k);
        }
        // Spot-check tau against gamlss' value at one non-trivial point (from a
        // fixed seed of the R oracle); the full sweep lives in the parity tests.
        let w = vec![1.0; n];
        let mut out = vec![0.0; n];
        BCTLSS.ngradient(3, &y, &etas, &w, &mut out);
        assert!((out[0] - 0.0972463).abs() < 1e-6, "tau gradient {}", out[0]);

        let prior = vec![0.0; n];
        assert!((BCTLSS.offset(1, &y, &w, &prior) + std::f64::consts::LN_10).abs() < 1e-15);
        assert!((BCTLSS.offset(2, &y, &w, &prior) - 0.5).abs() < 1e-15);

        // The signed 1/nu terms must retain a negative nu rather than clamp it
        // to a small positive denominator.
        etas[2].fill(-0.4);
        for k in 0..3 {
            assert_param_ngradient_matches_finite_diff(&BCTLSS, &y, &mut etas, n, k);
        }
        // nu=0 is a removable singularity; the implementation evaluates its
        // local derivative from the exact normalized density.
        etas[2].fill(0.0);
        for k in 0..3 {
            assert_param_ngradient_matches_finite_diff(&BCTLSS, &y, &mut etas, n, k);
        }
    }

    /// `z == 0` — a response exactly equal to its fitted `mu`, which the
    /// mean-valued mu offset makes reachable on the first iteration — used to
    /// give `0 · inf` in the mu/nu scores (for `tau < 2`) and `0 · -inf` in
    /// the tau score at every `tau`, including its own starting value of 2.
    #[test]
    fn bcpelss_ngradient_is_finite_where_y_equals_mu() {
        let y = [2.0, 2.5, 1.4];
        let n = y.len();
        let w = vec![1.0; n];
        // Row 0 sits exactly at mu. Sweep tau across the cusp at 1 and the
        // exponent sign change at 2.
        for eta_tau in [-0.3f64, 0.0, 0.4, std::f64::consts::LN_2, 1.1] {
            let etas = vec![
                Array1::from_vec(vec![2.0, 2.0, 2.0]),
                Array1::from_vec(vec![-1.2, -1.2, -1.2]),
                Array1::from_vec(vec![0.5, 0.5, 0.5]),
                Array1::from_vec(vec![eta_tau; n]),
            ];
            for k in 0..4 {
                let mut out = vec![0.0; n];
                BCPELSS.ngradient(k, &y, &etas, &w, &mut out);
                assert!(
                    out.iter().all(|value| value.is_finite()),
                    "tau=exp({eta_tau}) param {k}: {out:?}"
                );
            }
        }

        // Away from the tie the guard must not change anything: rows 1 and 2
        // agree with the finite difference, and the limit at row 0 does not
        // perturb them.
        let mut etas = vec![
            Array1::from_vec(vec![2.0, 2.0, 2.0]),
            Array1::from_vec(vec![-1.2, -1.2, -1.2]),
            Array1::from_vec(vec![0.5, 0.5, 0.5]),
            Array1::from_vec(vec![0.4, 0.4, 0.4]),
        ];
        let h = 1e-6;
        for k in 0..4 {
            let mut ngr = vec![0.0; n];
            BCPELSS.ngradient(k, &y, &etas, &w, &mut ngr);
            for j in 1..n {
                let orig = etas[k][j];
                etas[k][j] = orig + h;
                let rp = BCPELSS.risk(&y, &etas, &w);
                etas[k][j] = orig - h;
                let rm = BCPELSS.risk(&y, &etas, &w);
                etas[k][j] = orig;
                let fd = -(rp - rm) / (2.0 * h);
                assert!(
                    (ngr[j] - fd).abs() < 1e-5 * (1.0 + ngr[j].abs()),
                    "param {k} obs {j}: {} != {fd}",
                    ngr[j]
                );
            }
        }
    }

    #[test]
    fn bcpelss_ngradient_matches_finite_diff() {
        let y = [0.5, 1.2, 2.0, 3.5, 0.8, 4.1, 1.7, 2.4];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.6 + 0.1 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.3 + 0.04 * i as f64).collect();
        let eta_nu: Vec<f64> = (0..n).map(|i| 0.4 + 0.03 * i as f64).collect();
        let eta_tau: Vec<f64> = (0..n).map(|i| 0.2 + 0.02 * i as f64).collect();
        let mut etas = vec![
            Array1::from_vec(eta_mu),
            Array1::from_vec(eta_sigma),
            Array1::from_vec(eta_nu),
            Array1::from_vec(eta_tau),
        ];
        assert_ngradient_matches_finite_diff(&BCPELSS, &y, &mut etas, n);

        let w = vec![1.0; n];
        let prior = vec![0.0; n];
        assert!((BCPELSS.offset(1, &y, &w, &prior) + std::f64::consts::LN_10).abs() < 1e-15);
        assert!((BCPELSS.offset(2, &y, &w, &prior) - 1.0).abs() < 1e-15);

        etas[2].fill(-0.4);
        assert_ngradient_matches_finite_diff(&BCPELSS, &y, &mut etas, n);
        etas[2].fill(0.0);
        assert_ngradient_matches_finite_diff(&BCPELSS, &y, &mut etas, n);
    }

    #[test]
    fn zipo_lss_offsets_and_response() {
        // mu offset = log(mean y); sigma offset = qlogis(0.1). Response links:
        // mu = exp(eta), sigma = plogis(eta).
        let y = [0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 0.0];
        let w = [1.0; 8];
        let fam = ZIPoLSS;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        assert!((fam.offset(0, &y, &w, &[]) - mean_y.ln()).abs() < 1e-12);
        assert!((fam.offset(1, &y, &w, &[]) - qlogis(0.1)).abs() < 1e-12);

        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        fam.response(1, &[0.0, 2.0, -2.0], &mut out);
        for (o, e) in out.iter().zip([0.5, plogis(2.0), plogis(-2.0)]) {
            assert!((o - e).abs() < 1e-12);
        }
    }

    #[test]
    fn survival_lss_offsets_are_correct_risk_minimizers() {
        // The mu/sigma auto-offsets must minimize the family's OWN negative
        // log-likelihood (not gamboostLSS' buggy LogLog-sigma / Weibull-mu offset
        // routines — see .remember/gamboostlss-survival-offset-bugs.md). Verify
        // each offset is a local minimum of the correct risk by perturbing it.
        let t = [0.5, 1.2, 2.0, 3.5, 0.8, 4.1, 1.7, 2.9, 0.3, 5.0];
        let d = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let n = t.len();
        let mut y = Vec::with_capacity(2 * n);
        y.extend_from_slice(&t);
        y.extend_from_slice(&d);
        let w = vec![1.0; n];

        let mu_risk = |fam: &SurvivalLSS, mu: f64| -> f64 {
            (0..n).map(|i| fam.nll(mu, 0.0, y[i], y[n + i])).sum()
        };
        let sigma_risk = |fam: &SurvivalLSS, mu: f64, es: f64| -> f64 {
            (0..n).map(|i| fam.nll(mu, es, y[i], y[n + i])).sum()
        };

        for fam in [
            SurvivalLSS::log_log(),
            SurvivalLSS::weibull(),
            SurvivalLSS::log_normal(),
        ] {
            let mu = fam.offset(0, &y, &w, &[]);
            let sigma = fam.offset(1, &y, &w, &[mu]);
            let h = 1e-3;
            // mu offset minimizes the correct risk with sigma = 1.
            let r0 = mu_risk(&fam, mu);
            assert!(
                r0 <= mu_risk(&fam, mu + h) && r0 <= mu_risk(&fam, mu - h),
                "{} mu offset {mu} is not a local min of the correct risk",
                fam.name()
            );
            // sigma offset minimizes the correct risk with mu = mu offset.
            let s0 = sigma_risk(&fam, mu, sigma);
            assert!(
                s0 <= sigma_risk(&fam, mu, sigma + h) && s0 <= sigma_risk(&fam, mu, sigma - h),
                "{} sigma offset {sigma} is not a local min of the correct risk",
                fam.name()
            );
        }
    }

    #[test]
    fn fit_lss_cyclic_reduces_risk_and_recovers_signal() {
        // y = 2*x + N(0, sigma), sigma growing with z -> both mu and sigma have signal.
        let n = 200;
        let mut x = vec![0.0; n];
        let mut z = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = (i as f64) / n as f64 - 0.5;
            z[i] = ((i * 7) % n) as f64 / n as f64 - 0.5;
            // deterministic pseudo-noise so the test is stable
            let e = ((i as f64 * 1.3).sin()) * (0.3 + 0.5 * (z[i] + 0.5));
            y[i] = 2.0 * x[i] + e;
        }
        let w = Array1::from_elem(n, 1.0);
        let xc = Array1::from_vec(x.clone());
        let zc = Array1::from_vec(z.clone());

        // mu learners: intercept + x ; sigma learners: intercept + z
        let mut mu_learners = vec![
            prepare_scalar_bols(
                Array1::ones(n).view(),
                w.view(),
                0.0,
                false,
                "(Intercept)",
                None,
            ),
            prepare_scalar_bols(xc.view(), w.view(), 0.0, true, "x", None),
        ];
        let mut sigma_learners = vec![
            prepare_scalar_bols(
                Array1::ones(n).view(),
                w.view(),
                0.0,
                false,
                "(Intercept)",
                None,
            ),
            prepare_scalar_bols(zc.view(), w.view(), 0.0, true, "z", None),
        ];
        let mut per_param = vec![
            std::mem::take(&mut mu_learners),
            std::mem::take(&mut sigma_learners),
        ];

        let ctrl = BoostControl::new(100, 0.1).unwrap();
        let fam = GaussianLSS;
        let path = fit_lss(
            &mut per_param,
            Array1::from_vec(y.clone()).view(),
            w.view(),
            &fam,
            &ctrl,
            None,
            None,
        );

        assert_eq!(path.per_param.len(), 2);
        assert_eq!(path.risk.len(), 101);
        // Risk must decrease monotonically (component-wise boosting is a descent).
        for m in 1..path.risk.len() {
            assert!(
                path.risk[m] <= path.risk[m - 1] + 1e-9,
                "risk increased at step {m}: {} -> {}",
                path.risk[m - 1],
                path.risk[m]
            );
        }
        assert!(path.risk[100] < path.risk[0] - 1.0, "risk barely moved");
        // mu predictor should pick up the x slope (positive).
        let mu_fitted = &path.per_param[0].fitted;
        // correlation of fitted mu with x should be strongly positive
        let mx: f64 = mu_fitted.iter().sum::<f64>() / n as f64;
        let xx: f64 = xc.iter().sum::<f64>() / n as f64;
        let cov: f64 = (0..n).map(|i| (mu_fitted[i] - mx) * (xc[i] - xx)).sum();
        assert!(cov > 0.0, "fitted mu should increase with x");
    }

    #[test]
    fn fit_lss_per_parameter_mstop() {
        // Per-parameter mstop: param p ends with exactly mstops[p] steps; the
        // joint risk path has max(mstops)+1 entries. Also: once sigma stops, mu
        // keeps updating, so the result must equal the prefix of a longer run.
        let n = 120;
        let w = Array1::from_elem(n, 1.0);
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64 - 0.5).collect();
        let z: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % n) as f64 / n as f64 - 0.5)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * x[i] + (i as f64 * 1.3).sin() * (0.3 + 0.5 * (z[i] + 0.5)))
            .collect();
        let xc = Array1::from_vec(x);
        let zc = Array1::from_vec(z);
        let make = || {
            vec![
                vec![
                    prepare_scalar_bols(
                        Array1::ones(n).view(),
                        w.view(),
                        0.0,
                        false,
                        "(Intercept)",
                        None,
                    ),
                    prepare_scalar_bols(xc.view(), w.view(), 0.0, true, "x", None),
                ],
                vec![
                    prepare_scalar_bols(
                        Array1::ones(n).view(),
                        w.view(),
                        0.0,
                        false,
                        "(Intercept)",
                        None,
                    ),
                    prepare_scalar_bols(zc.view(), w.view(), 0.0, true, "z", None),
                ],
            ]
        };
        let ctrl = BoostControl::new(60, 0.1)
            .unwrap()
            .with_instrumentation(crate::report::InstrumentationLevel::Summary);
        let fam = GaussianLSS;
        let yv = Array1::from_vec(y);

        let mut lp = make();
        let path = fit_lss(
            &mut lp,
            yv.view(),
            w.view(),
            &fam,
            &ctrl,
            Some(&[60, 20]),
            None,
        );
        assert_eq!(path.per_param[0].selected.len(), 60);
        assert_eq!(path.per_param[1].selected.len(), 20);
        assert_eq!(path.risk.len(), 61); // max(60,20)+1
        let report = path.execution_report();
        assert_eq!(report.route, EngineRoute::LssCyclic);
        assert_eq!(report.n_observations, n);
        assert_eq!(report.n_learners, 4);
        assert_eq!(report.total_basis_columns, 4);
        assert_eq!(report.requested_mstop, 80);
        assert_eq!(report.completed_mstop, 80);
        assert!(report.setup_ns.is_some());
        assert!(report.iteration_ns.is_some());
        assert!(report.total_ns.is_some());
        let rows = path.path_rows();
        assert_eq!(rows.len(), 80);
        assert_eq!(rows[0].parameter_index, Some(0));
        assert_eq!(rows[0].cycle, Some(1));
        assert_eq!(rows[0].risk, None);
        assert_eq!(rows[1].parameter_index, Some(1));
        assert_eq!(rows[1].risk, Some(path.risk[1]));
        assert_eq!(rows[40].parameter_index, Some(0));
        assert_eq!(rows[40].cycle, Some(21));
        assert_eq!(rows[40].risk, Some(path.risk[21]));

        // sigma frozen after step 20: its fitted must equal a 20-step sigma-only
        // run's fitted (mu continues to evolve but doesn't feed back into a
        // frozen sigma's recorded path beyond what the cyclic order produces).
        // Sanity: risk still non-increasing.
        for m in 1..path.risk.len() {
            assert!(path.risk[m] <= path.risk[m - 1] + 1e-9);
        }
    }

    #[test]
    fn fit_lss_diagonal_checkpoints_matches_per_point_fit_lss() {
        // LSS5: a single fit_lss_diagonal_checkpoints pass to max(checkpoints)
        // must, at every checkpoint m, produce etas bit-identical to a fresh
        // fit_lss(..., mstop_per_param = [m; k], ...) call — the property
        // cvrisk_lss_with's diagonal-grid fast path relies on.
        let n = 100;
        let w = Array1::from_elem(n, 1.0);
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64 - 0.5).collect();
        let z: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % n) as f64 / n as f64 - 0.5)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * x[i] + (i as f64 * 1.3).sin() * (0.3 + 0.5 * (z[i] + 0.5)))
            .collect();
        let xc = Array1::from_vec(x);
        let zc = Array1::from_vec(z);
        let make = || {
            vec![
                vec![
                    prepare_scalar_bols(
                        Array1::ones(n).view(),
                        w.view(),
                        0.0,
                        false,
                        "(Intercept)",
                        None,
                    ),
                    prepare_scalar_bols(xc.view(), w.view(), 0.0, true, "x", None),
                ],
                vec![
                    prepare_scalar_bols(
                        Array1::ones(n).view(),
                        w.view(),
                        0.0,
                        false,
                        "(Intercept)",
                        None,
                    ),
                    prepare_scalar_bols(zc.view(), w.view(), 0.0, true, "z", None),
                ],
            ]
        };
        let ctrl = BoostControl::new(50, 0.1).unwrap();
        let fam = GaussianLSS;
        let yv = Array1::from_vec(y);

        let checkpoints = [5usize, 13, 27, 50];
        let mut lp_snap = make();
        let snapshots = try_fit_lss_diagonal_checkpoints(
            &mut lp_snap,
            yv.view(),
            w.view(),
            &fam,
            &ctrl,
            &checkpoints,
            None,
        )
        .unwrap();
        assert_eq!(snapshots.len(), checkpoints.len());
        let snap_map: std::collections::HashMap<usize, Vec<Array1<f64>>> =
            snapshots.into_iter().collect();

        for &m in &checkpoints {
            let mut lp_direct = make();
            let path = fit_lss(
                &mut lp_direct,
                yv.view(),
                w.view(),
                &fam,
                &ctrl,
                Some(&[m, m]),
                None,
            );
            let snap = &snap_map[&m];
            for p in 0..2 {
                let direct_fitted = &path.per_param[p].fitted;
                for i in 0..n {
                    assert!(
                        (direct_fitted[i] - snap[p][i]).abs() < 1e-10,
                        "checkpoint m={m}, param {p}, obs {i}: {} vs {}",
                        direct_fitted[i],
                        snap[p][i]
                    );
                }
            }
        }
    }

    #[test]
    fn seeded_lss_diagonal_checkpoints_match_independent_fits() {
        // Both paths execute the same cyclic updates; the tolerance permits
        // only the snapshot/copy roundoff visible in fitted predictors.
        const TOLERANCE: f64 = 1e-10;
        for seed in 301_u64..=305 {
            let n = 18 + seed as usize % 23;
            let max_mstop = 7 + seed as u32 % 17;
            let nu = [0.03, 0.1, 0.4][seed as usize % 3];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let x = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let z = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let y = Array1::from_shape_fn(n, |index| {
                1.3 * x[index] + (0.15 + 0.35 * (z[index] + 1.0)) * (2.0 * random() - 1.0)
            });
            let weights = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * random());
            let intercept = Array1::ones(n);
            let make = || {
                vec![
                    vec![
                        prepare_scalar_bols(
                            intercept.view(),
                            weights.view(),
                            0.0,
                            false,
                            "(Intercept)",
                            None,
                        ),
                        prepare_scalar_bols(x.view(), weights.view(), 0.0, true, "x", None),
                    ],
                    vec![
                        prepare_scalar_bols(
                            intercept.view(),
                            weights.view(),
                            0.0,
                            false,
                            "(Intercept)",
                            None,
                        ),
                        prepare_scalar_bols(z.view(), weights.view(), 0.0, true, "z", None),
                    ],
                ]
            };
            let control = BoostControl::new(max_mstop, nu).unwrap();
            let checkpoints = [1, max_mstop as usize / 2, max_mstop as usize];
            let mut snapshot_learners = make();
            let snapshots = try_fit_lss_diagonal_checkpoints(
                &mut snapshot_learners,
                y.view(),
                weights.view(),
                &GaussianLSS,
                &control,
                &checkpoints,
                None,
            )
            .unwrap();
            let snapshot_map: std::collections::HashMap<usize, Vec<Array1<f64>>> =
                snapshots.into_iter().collect();
            for checkpoint in checkpoints {
                let mut direct_learners = make();
                let direct = fit_lss(
                    &mut direct_learners,
                    y.view(),
                    weights.view(),
                    &GaussianLSS,
                    &control,
                    Some(&[checkpoint, checkpoint]),
                    None,
                );
                for parameter in 0..2 {
                    let max_difference = direct.per_param[parameter]
                        .fitted
                        .iter()
                        .zip(&snapshot_map[&checkpoint][parameter])
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f64, f64::max);
                    assert!(
                        max_difference <= TOLERANCE,
                        "LSS checkpoint mismatch {max_difference:.3e}; seed={seed} \
                         spec=n={n},mstop={max_mstop},nu={nu},checkpoint={checkpoint},\
                         parameter={parameter}"
                    );
                }
            }
        }
    }

    /// LssFamily whose pseudo-residual is all-NaN: every learner's WLS fit
    /// then produces a NaN SSE, exercising the drivers' no-winner fallback.
    /// (A degenerate gradient is the realistic trigger — singular Grams
    /// degrade to finite values in the solver.)
    struct NanGradFamily;

    impl LssFamily for NanGradFamily {
        fn name(&self) -> &'static str {
            "NanGrad"
        }
        fn n_params(&self) -> usize {
            2
        }
        fn param_names(&self) -> &'static [&'static str] {
            &["a", "b"]
        }
        fn offset(&self, _k: usize, _y: &[f64], _w: &[f64], _prior: &[f64]) -> f64 {
            0.5
        }
        fn ngradient(
            &self,
            _k: usize,
            _y: &[f64],
            _etas: &[Array1<f64>],
            _w: &[f64],
            out: &mut [f64],
        ) {
            out.fill(f64::NAN);
        }
        fn risk(&self, _y: &[f64], _etas: &[Array1<f64>], _w: &[f64]) -> f64 {
            1.0
        }
        fn response(&self, _k: usize, eta: &[f64], out: &mut [f64]) {
            out.copy_from_slice(eta);
        }
        fn check_y(&self, _y: ArrayView1<f64>) -> Result<(), MboostrsError> {
            Ok(())
        }
        fn clone_box(&self) -> Box<dyn LssFamily> {
            Box::new(NanGradFamily)
        }
    }

    #[test]
    fn fit_lss_all_nan_sse_is_noop_not_stale() {
        // When every learner's SSE is NaN (here: an all-NaN pseudo-residual),
        // the drivers must record explicit no-op steps (zero coefs, eta
        // unchanged) instead of silently reusing the previous iteration's
        // winner buffers (the engine's all-block path has the analogous guard).
        let n = 30;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 0.5).collect();
        let w = Array1::from_elem(n, 1.0);
        let xc = Array1::from_vec(x);
        let make = || {
            vec![
                vec![prepare_scalar_bols(
                    xc.view(),
                    w.view(),
                    0.0,
                    true,
                    "x",
                    None,
                )],
                vec![prepare_scalar_bols(
                    xc.view(),
                    w.view(),
                    0.0,
                    true,
                    "x",
                    None,
                )],
            ]
        };
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let fam = NanGradFamily;
        let yv = Array1::from_vec(y);

        // Precondition: a NaN residual really does yield a non-finite SSE.
        {
            let mut l = make();
            let u = Array1::from_elem(n, f64::NAN);
            let (_b, _p, sse) = l[0][0].fit_step_ref(u.view(), w.view());
            assert!(
                !sse.is_finite(),
                "expected a non-finite SSE from a NaN residual, got {sse}"
            );
        }

        let mut lp = make();
        let path = fit_lss(&mut lp, yv.view(), w.view(), &fam, &ctrl, None, None);
        for p in 0..2 {
            let bp = &path.per_param[p];
            assert_eq!(bp.selected.len(), 5);
            for c in &bp.coefs {
                assert!(
                    c.iter().all(|v| *v == 0.0),
                    "coefs must be explicit zeros, got {c:?}"
                );
            }
            for v in bp.fitted.iter() {
                assert!(
                    v.is_finite() && (v - bp.offset).abs() < 1e-12,
                    "eta must stay at the offset (param {p})"
                );
            }
        }
        for r in &path.risk {
            assert!(r.is_finite(), "risk path must stay finite");
        }

        // Non-cyclic driver: same guarantee.
        let mut lp2 = make();
        let path2 = fit_lss_noncyclic(&mut lp2, yv.view(), w.view(), &fam, &ctrl, None);
        let report2 = path2.execution_report();
        assert_eq!(report2.route, EngineRoute::LssNoncyclic);
        assert_eq!(report2.requested_mstop, 5);
        assert_eq!(report2.completed_mstop, 5);
        let rows2 = path2.path_rows();
        assert_eq!(rows2.len(), 5);
        assert!(rows2.iter().all(|row| row.cycle.is_none()));
        assert!(rows2.iter().all(|row| row.risk.is_some()));
        for p in 0..2 {
            let bp = &path2.per_param[p];
            for v in bp.fitted.iter() {
                assert!(
                    v.is_finite() && (v - bp.offset).abs() < 1e-12,
                    "noncyclic eta must stay at the offset (param {p})"
                );
            }
        }
        for r in &path2.risk {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn lss_execution_report_counts_deterministic_tie_decisions() {
        let n = 30;
        let w = Array1::ones(n);
        let x = Array1::<f64>::linspace(-1.0, 1.0, n);
        let y = x.mapv(|value| 1.0 + value + 0.2 * (3.0 * value).sin());
        let make = || {
            (0..2)
                .map(|parameter| {
                    vec![
                        prepare_scalar_bols(
                            x.view(),
                            w.view(),
                            0.0,
                            true,
                            &format!("p{parameter}_a"),
                            None,
                        ),
                        prepare_scalar_bols(
                            x.view(),
                            w.view(),
                            0.0,
                            true,
                            &format!("p{parameter}_b"),
                            None,
                        ),
                    ]
                })
                .collect::<Vec<_>>()
        };
        let control = BoostControl::new(3, 0.1).unwrap();

        let mut cyclic_learners = make();
        let cyclic = fit_lss(
            &mut cyclic_learners,
            y.view(),
            w.view(),
            &GaussianLSS,
            &control,
            None,
            None,
        );
        assert_eq!(cyclic.execution_report().tie_selection_count, 6);

        let mut noncyclic_learners = make();
        let noncyclic = fit_lss_noncyclic(
            &mut noncyclic_learners,
            y.view(),
            w.view(),
            &GaussianLSS,
            &control,
            None,
        );
        // Every applied parameter update has a two-learner tie. An additional
        // count would be valid if the joint-risk parameter decision also ties.
        assert!(noncyclic.execution_report().tie_selection_count >= 3);
    }

    #[test]
    fn poisson_lss_ngradient_matches_finite_diff() {
        let y = [0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 4.0];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.2 + 0.1 * i as f64).collect();
        let mut etas = vec![Array1::from_vec(eta_mu)];
        assert_ngradient_matches_finite_diff(&PoissonLSS, &y, &mut etas, n);
    }

    #[test]
    fn poisson_lss_offset_response_check_y() {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 5];
        let fam = PoissonLSS;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        assert!((fam.offset(0, &y, &w, &[]) - mean_y.ln()).abs() < 1e-12);
        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        assert!(fam.check_y(ArrayView1::from(&[0.0, 1.0, 2.0])).is_ok());
        assert!(fam.check_y(ArrayView1::from(&[-1.0])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[1.5])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[f64::NAN])).is_err());
    }

    #[test]
    fn geometric_lss_ngradient_matches_finite_diff() {
        let y = [0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.2 + 0.1 * i as f64).collect();
        let mut etas = vec![Array1::from_vec(eta_mu)];
        assert_ngradient_matches_finite_diff(&GeometricLSS, &y, &mut etas, n);
    }

    #[test]
    fn geometric_lss_offset_response_check_y() {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 5];
        let fam = GeometricLSS;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        assert!((fam.offset(0, &y, &w, &[]) - mean_y.ln()).abs() < 1e-12);
        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        assert!(fam.check_y(ArrayView1::from(&[0.0, 1.0, 2.0])).is_ok());
        assert!(fam.check_y(ArrayView1::from(&[-1.0])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[1.5])).is_err());
    }

    #[test]
    fn binomial_lss_ngradient_matches_finite_diff() {
        let y = [0.0, 1.0, 0.3, 0.7, 0.0, 1.0, 0.5, 0.2];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| -0.5 + 0.15 * i as f64).collect();
        let mut etas = vec![Array1::from_vec(eta_mu)];
        assert_ngradient_matches_finite_diff(&BinomialLSS, &y, &mut etas, n);
    }

    #[test]
    fn binomial_lss_offset_response_check_y() {
        let y = [0.0, 0.0, 1.0, 1.0, 0.0];
        let w = [1.0; 5];
        let fam = BinomialLSS;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        assert!((fam.offset(0, &y, &w, &[]) - (mean_y / (1.0 - mean_y)).ln()).abs() < 1e-12);
        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 2.0, -2.0], &mut out);
        for (o, e) in out.iter().zip([0.5, plogis(2.0), plogis(-2.0)]) {
            assert!((o - e).abs() < 1e-12);
        }
        assert!(fam.check_y(ArrayView1::from(&[0.0, 0.5, 1.0])).is_ok());
        assert!(fam.check_y(ArrayView1::from(&[-0.1])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[1.1])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[f64::NAN])).is_err());
    }

    #[test]
    fn negative_binomial_lss_ngradient_matches_finite_diff() {
        let y = [0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 4.0];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.3 + 0.1 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.2 + 0.05 * i as f64).collect();
        let mut etas = vec![Array1::from_vec(eta_mu), Array1::from_vec(eta_sigma)];
        assert_ngradient_matches_finite_diff(&NegativeBinomialLSS, &y, &mut etas, n);
    }

    #[test]
    fn negative_binomial_lss_offset_response_check_y() {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0];
        let w = [1.0; 7];
        let n = y.len() as f64;
        let fam = NegativeBinomialLSS;
        let mean_y = y.iter().sum::<f64>() / n;
        assert!((fam.offset(0, &y, &w, &[]) - mean_y.ln()).abs() < 1e-12);
        // Was log(var / mean) with the n denominator; #817 uses gamlss NBI's rule.
        let var_y = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>() / (n - 1.0);
        let sig_init = ((var_y - mean_y) / (mean_y * mean_y)).max(0.1).ln();
        assert!((fam.offset(1, &y, &w, &[]) - sig_init).abs() < 1e-10);
        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        fam.response(1, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        assert!(fam.check_y(ArrayView1::from(&[0.0, 1.0, 2.0])).is_ok());
        assert!(fam.check_y(ArrayView1::from(&[-1.0])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[1.5])).is_err());
    }

    #[test]
    fn zero_adjusted_gamma_lss_ngradient_matches_finite_diff() {
        // Mixed zeros and positives: the gradient must be right on both arms.
        let y = [0.0, 1.3, 0.0, 0.9, 3.1, 2.2];
        let n = y.len();
        let mut etas = vec![
            Array1::from(vec![0.0, 0.4, 0.7, 0.2, 0.9, 0.6]),
            Array1::from(vec![-0.2, 0.1, -0.1, 0.3, -0.3, 0.0]),
            Array1::from(vec![-1.4, -0.8, -1.1, -0.4, -1.7, -0.6]),
        ];
        assert_ngradient_matches_finite_diff(&ZeroAdjustedGammaLSS, &y, &mut etas, n);
    }

    #[test]
    fn zero_adjusted_gamma_lss_risk_matches_gamlss_dist() {
        // -sum(dZAGA(y, mu, sigma, nu, log = TRUE)) from gamlss.dist, computed
        // on this point before the family was written.
        let y = [0.0, 1.3, 0.0, 0.9, 3.1, 2.2];
        let mu = [1.0, 1.5, 2.0, 1.2, 2.5, 1.8];
        let sigma = [0.8, 1.1, 0.9, 1.3, 0.7, 1.0];
        let nu = [0.2, 0.3, 0.25, 0.4, 0.15, 0.35];
        let etas = vec![
            Array1::from(mu.iter().map(|v: &f64| v.ln()).collect::<Vec<_>>()),
            Array1::from(sigma.iter().map(|v: &f64| v.ln()).collect::<Vec<_>>()),
            Array1::from(
                nu.iter()
                    .map(|v: &f64| (v / (1.0 - v)).ln())
                    .collect::<Vec<_>>(),
            ),
        ];
        let w = vec![1.0; y.len()];
        let risk = ZeroAdjustedGammaLSS.risk(&y, &etas, &w);
        assert!(
            (risk - 10.663_543_049_1).abs() < 1e-9,
            "ZAGA risk {risk} != gamlss.dist 10.6635430491"
        );
    }

    #[test]
    fn zero_adjusted_gamma_lss_rejects_an_all_zero_response() {
        // The Gamma part has no data, so mu and sigma would be unidentified.
        let y = Array1::from(vec![0.0, 0.0, 0.0]);
        let err = ZeroAdjustedGammaLSS
            .check_y(y.view())
            .expect_err("an all-zero response must be refused");
        assert!(format!("{err}").contains("all zeros"), "unexpected: {err}");
        // A single positive is enough to identify it.
        let mixed = Array1::from(vec![0.0, 0.0, 1.5]);
        assert!(ZeroAdjustedGammaLSS.check_y(mixed.view()).is_ok());
    }

    #[test]
    fn zero_adjusted_gamma_lss_rejects_negative_y() {
        let y = Array1::from(vec![0.0, 1.0, -0.5]);
        assert!(ZeroAdjustedGammaLSS.check_y(y.view()).is_err());
    }

    #[test]
    fn inverse_gaussian_lss_ngradient_matches_finite_diff() {
        let y = [0.4, 1.3, 2.7, 0.9, 3.1, 1.8];
        let n = y.len();
        let mut etas = vec![
            Array1::from(vec![0.0, 0.4, 0.7, 0.2, 0.9, 0.5]),
            Array1::from(vec![-0.2, 0.1, -0.1, 0.3, -0.3, 0.0]),
        ];
        assert_ngradient_matches_finite_diff(&InverseGaussianLSS, &y, &mut etas, n);
    }

    #[test]
    fn logistic_lss_ngradient_matches_finite_diff() {
        let y = [0.4, -1.3, 2.7, 0.9, -3.1, 1.8];
        let n = y.len();
        let mut etas = vec![
            Array1::from(vec![0.5, 0.2, 1.0, 0.7, -1.0, 0.3]),
            Array1::from(vec![-0.2, 0.1, -0.1, 0.3, -0.3, 0.0]),
        ];
        assert_ngradient_matches_finite_diff(&LogisticLSS, &y, &mut etas, n);
    }

    #[test]
    fn inverse_gaussian_lss_risk_matches_gamlss_dist() {
        // -sum(dIG(y, mu, sigma, log = TRUE)) from gamlss.dist, computed on the
        // same point before this family was written.
        let y = [0.4, 1.3, 2.7, 0.9, 3.1];
        let mu = [1.0, 1.5, 2.0, 1.2, 2.5];
        let sigma = [0.8, 1.1, 0.9, 1.3, 0.7];
        let etas = vec![
            Array1::from(mu.iter().map(|v: &f64| v.ln()).collect::<Vec<_>>()),
            Array1::from(sigma.iter().map(|v: &f64| v.ln()).collect::<Vec<_>>()),
        ];
        let w = vec![1.0; y.len()];
        let risk = InverseGaussianLSS.risk(&y, &etas, &w);
        assert!(
            (risk - 7.091_526_303_4).abs() < 1e-9,
            "IG risk {risk} != gamlss.dist 7.0915263034"
        );
    }

    #[test]
    fn logistic_lss_risk_matches_gamlss_dist() {
        // -sum(dLO(y, mu, sigma, log = TRUE)) from gamlss.dist.
        let y = [0.4, -1.3, 2.7, 0.9, -3.1];
        let mu = [0.5, 0.2, 1.0, 0.7, -1.0];
        let sigma = [0.8, 1.1, 0.9, 1.3, 0.7];
        let etas = vec![
            Array1::from(mu.to_vec()),
            Array1::from(sigma.iter().map(|v: &f64| v.ln()).collect::<Vec<_>>()),
        ];
        let w = vec![1.0; y.len()];
        let risk = LogisticLSS.risk(&y, &etas, &w);
        assert!(
            (risk - 9.541_710_797_0).abs() < 1e-9,
            "LO risk {risk} != gamlss.dist 9.5417107970"
        );
    }

    #[test]
    fn gumbel_lss_is_gamlss_rg_not_gu() {
        // The doc comment claimed `gamlss.dist::GU()` parity and was wrong: this
        // family is `RG()`, the right-skewed extreme-value form almost everyone
        // means by "Gumbel". gamlss's naming is the reverse of the common one.
        // -sum(dRG(...)) = 9.042800226406863; -sum(dGU(...)) = 9.161943.
        let y = [0.4, 1.3, -0.7, 2.1, 0.0];
        let etas = vec![Array1::from_elem(5, 0.5), Array1::from_elem(5, 2.0f64.ln())];
        let w = vec![1.0; 5];
        let risk = GumbelLSS.risk(&y, &etas, &w);
        assert!(
            (risk - 9.042_800_226_406_863).abs() < 1e-12,
            "GumbelLSS risk {risk} should equal gamlss RG, not GU (9.161943)"
        );
        assert!((risk - 9.161_943).abs() > 1e-3, "GumbelLSS must not be GU");
    }

    #[test]
    fn gumbel_lss_ngradient_matches_finite_diff() {
        let y = [-1.5, 0.2, 1.0, -0.5, 2.3, -0.8, 0.0, 1.8];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.2 + 0.08 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.1 + 0.04 * i as f64).collect();
        let mut etas = vec![Array1::from_vec(eta_mu), Array1::from_vec(eta_sigma)];
        assert_ngradient_matches_finite_diff(&GumbelLSS, &y, &mut etas, n);
    }

    #[test]
    fn gumbel_lss_offset_response_check_y() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0; 5];
        let n = y.len() as f64;
        let fam = GumbelLSS;
        let mean_y = y.iter().sum::<f64>() / n;
        assert!((fam.offset(0, &y, &w, &[]) - mean_y).abs() < 1e-12);
        let var_y = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f64>() / n;
        let want_sig = var_y.sqrt().ln();
        assert!((fam.offset(1, &y, &w, &[]) - want_sig).abs() < 1e-12);
        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 1.0, -1.0], &mut out);
        assert_eq!(out, [0.0, 1.0, -1.0]);
        fam.response(1, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        assert!(fam.check_y(ArrayView1::from(&[-1.0, 0.0, 2.5])).is_ok());
        assert!(fam.check_y(ArrayView1::from(&[f64::NAN])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[f64::INFINITY])).is_err());
    }

    #[test]
    fn pareto2_lss_ngradient_matches_finite_diff() {
        let y = [0.5, 1.2, 2.0, 0.0, 3.5, 0.8, 4.1, 1.7];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.3 + 0.1 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.2 + 0.05 * i as f64).collect();
        let mut etas = vec![Array1::from_vec(eta_mu), Array1::from_vec(eta_sigma)];
        assert_ngradient_matches_finite_diff(&Pareto2LSS, &y, &mut etas, n);
    }

    #[test]
    fn pareto2_lss_offset_response_check_y() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0; 5];
        let fam = Pareto2LSS;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        assert!((fam.offset(0, &y, &w, &[]) - mean_y.ln()).abs() < 1e-12);
        let sly: f64 = y.iter().map(|&v| (v / (v + 1.0)).ln()).sum();
        let want_sig = (-sly / y.len() as f64).ln();
        assert!((fam.offset(1, &y, &w, &[]) - want_sig).abs() < 1e-12);
        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        fam.response(1, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        assert!(fam.check_y(ArrayView1::from(&[0.0, 1.0, 2.5])).is_ok());
        assert!(fam.check_y(ArrayView1::from(&[-1.0])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[f64::NAN])).is_err());
    }

    #[test]
    fn zap_lss_ngradient_matches_finite_diff() {
        let y = [0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0, 0.0, 4.0];
        let n = y.len();
        let eta_mu: Vec<f64> = (0..n).map(|i| 0.3 + 0.1 * i as f64).collect();
        let eta_sigma: Vec<f64> = (0..n).map(|i| -0.2 + 0.05 * i as f64).collect();
        let mut etas = vec![Array1::from_vec(eta_mu), Array1::from_vec(eta_sigma)];
        assert_ngradient_matches_finite_diff(&ZeroAdjustedPoissonLSS, &y, &mut etas, n);
    }

    #[test]
    fn zap_lss_offset_response_check_y() {
        let y = [0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0];
        let w = [1.0; 7];
        let n = y.len() as f64;
        let fam = ZeroAdjustedPoissonLSS;
        let nonzero_y: Vec<f64> = y.iter().filter(|&&v| v > 0.5).copied().collect();
        let nonzero_mean = nonzero_y.iter().sum::<f64>() / nonzero_y.len() as f64;
        assert!((fam.offset(0, &y, &w, &[]) - nonzero_mean.ln()).abs() < 1e-12);
        let p0 = y.iter().filter(|&&v| v < 0.5).count() as f64 / n;
        assert!((fam.offset(1, &y, &w, &[]) - (p0 / (1.0 - p0)).ln()).abs() < 1e-12);
        let mut out = [0.0; 3];
        fam.response(0, &[0.0, 1.0, -1.0], &mut out);
        for (o, e) in out.iter().zip([1.0, 1.0_f64.exp(), (-1.0_f64).exp()]) {
            assert!((o - e).abs() < 1e-12);
        }
        fam.response(1, &[0.0, 2.0, -2.0], &mut out);
        for (o, e) in out.iter().zip([0.5, plogis(2.0), plogis(-2.0)]) {
            assert!((o - e).abs() < 1e-12);
        }
        assert!(fam.check_y(ArrayView1::from(&[0.0, 1.0, 2.0])).is_ok());
        assert!(fam.check_y(ArrayView1::from(&[-1.0])).is_err());
        assert!(fam.check_y(ArrayView1::from(&[1.5])).is_err());
    }

    #[test]
    fn pig_lss_check_y_accepts_counts_and_nothing_else() {
        // This asserted that PIG rejected *every* response, which was right
        // while the family was a Poisson approximation behind a closed door
        // (P4.1 item 1). Now it is a count distribution and refuses what a
        // count distribution refuses.
        assert!(PIGLSS.check_y(ArrayView1::from(&[0.0])).is_ok());
        assert!(PIGLSS.check_y(ArrayView1::from(&[1.0, 2.0])).is_ok());
        assert!(PIGLSS.check_y(ArrayView1::from(&[-1.0])).is_err());
        assert!(PIGLSS.check_y(ArrayView1::from(&[1.5])).is_err());
        assert!(PIGLSS.check_y(ArrayView1::from(&[f64::NAN])).is_err());
    }

    #[test]
    fn negative_binomial_lss_sigma_offset_matches_gamlss_sigma_initial() {
        // mean 5, var (n - 1 denominator) 15. gamlss.dist::NBI()$sigma.initial
        // gives max((15 - 5) / 5^2, 0.1) = 0.4. Before issue #817 the start was
        // var / mean, about 2.7 here.
        let y = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let w = [1.0; 11];
        let sigma = NegativeBinomialLSS.offset(1, &y, &w, &[]).exp();
        assert!((sigma - 0.4).abs() < 1e-10, "sigma start {sigma}");
        // The same value as the ZINBLSS sigma offset on these data.
        let zinb = ZINBLSS.offset(1, &y, &w, &[]).exp();
        assert!((sigma - zinb).abs() < 1e-12, "NBI {sigma}, ZINB {zinb}");
        let flat = [2.0, 2.0, 3.0, 3.0];
        let sigma = NegativeBinomialLSS.offset(1, &flat, &[1.0; 4], &[]).exp();
        assert!((sigma - 0.1).abs() < 1e-12, "sigma floor {sigma}");
    }

    #[test]
    fn pig_lss_sigma_offset_matches_gamlss_sigma_initial() {
        // mean 5, var (n - 1 denominator) 15. gamlss.dist::PIG()$sigma.initial
        // gives max((15 - 5) / 5^2, 0.1) = 0.4. Before issue #817 the start was
        // sqrt(mu^2 / (var - mu)) = 1.58, which inverts the variance relation.
        let y = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let w = [1.0; 11];
        assert!((PIGLSS.offset(0, &y, &w, &[]) - 5.0_f64.ln()).abs() < 1e-12);
        let sigma = PIGLSS.offset(1, &y, &w, &[]).exp();
        assert!((sigma - 0.4).abs() < 1e-10, "sigma start {sigma}");
        // Underdispersed data take gamlss's floor of 0.1.
        let flat = [2.0, 2.0, 3.0, 3.0];
        let sigma = PIGLSS.offset(1, &flat, &[1.0; 4], &[]).exp();
        assert!((sigma - 0.1).abs() < 1e-12, "sigma floor {sigma}");
    }

    #[test]
    fn digamma_known_values() {
        // Reference values from R's digamma(); implementation is accurate to ~1e-11
        // due to the recurrence+asymptotic expansion.
        assert!(
            (digamma(1.0) - (-0.5772156649015329)).abs() < 1e-10,
            "digamma(1) = -gamma"
        );
        assert!(
            (digamma(2.0) - 0.4227843351).abs() < 1e-10,
            "digamma(2) = 1 - gamma"
        );
        assert!(
            (digamma(0.5) - (-1.96351002602834)).abs() < 1e-8,
            "digamma(0.5) = -gamma - 2*ln(2)"
        );
        assert!(
            (digamma(3.0) - 0.9227843351).abs() < 1e-10,
            "digamma(3) = 1/1+1/2-gamma"
        );
        assert!(
            (digamma(10.0) - 2.251752589).abs() < 1e-8,
            "digamma(10) from R"
        );
        assert!(
            (digamma(100.0) - 4.600161853).abs() < 1e-8,
            "digamma(100) from R"
        );
        // x >= 6 hits the asymptotic expansion directly (no recurrence)
        assert!(
            (digamma(6.0) - 1.706117668).abs() < 1e-8,
            "digamma(6) from R (no recurrence)"
        );
    }

    #[test]
    fn digamma_is_nan_on_zero() {
        // An underflowed `exp(eta)` must not abort the process (#319).
        assert!(digamma(0.0).is_nan());
    }

    #[test]
    fn digamma_is_nan_on_negative_and_nan() {
        assert!(digamma(-1.0).is_nan());
        assert!(digamma(f64::NAN).is_nan());
    }

    #[test]
    fn gamma_plloss_known_values() {
        let sigma = 0.8;
        let y = 2.0;
        let f_mu = 1.5f64.ln();
        let loss = crate::lss::gamma_plloss(sigma, y, f_mu);
        assert!(
            loss.is_finite(),
            "gamma_plloss should be finite for valid inputs"
        );
        let loss_neg = crate::lss::gamma_plloss(sigma, 0.01, f_mu);
        assert!(
            loss_neg.is_finite(),
            "gamma_plloss should be finite for small y"
        );
    }

    #[test]
    fn gaussian_knownvar_ngradient_matches_finite_diff() {
        // Doubled response [outcome, s2]; per-observation s2 spanning an
        // order of magnitude so the attenuation term e/(v) varies strongly.
        let outcome = [1.2, -0.4, 2.9, 0.3, -1.7, 3.3, 0.9, -0.6];
        let s2 = [0.02, 0.5, 0.07, 1.4, 0.11, 0.03, 0.9, 0.25];
        let y: Vec<f64> = outcome.iter().chain(s2.iter()).copied().collect();
        let n = outcome.len();
        let mut etas = [
            Array1::from(vec![0.4, -0.2, 1.1, 0.0, -0.8, 1.5, 0.2, -0.3]),
            Array1::from(vec![-0.5, 0.3, -1.2, 0.8, -0.1, 0.6, -0.9, 0.1]),
        ];
        let fam = GaussianKnownVarLSS;
        fam.check_y(ndarray::arr1(&y).view()).unwrap();
        assert_ngradient_matches_finite_diff(&fam, &y, &mut etas, n);
    }

    #[test]
    fn gaussian_knownvar_zero_s2_reduces_to_gaussianlss() {
        // With s2 = 0 the variance is exp(eta) — GaussianLSS under the
        // reparameterization eta_gauss = eta_knownvar / 2 (there sigma =
        // e^{2 eta_gauss}). Risks must coincide, and both gradients must
        // coincide in natural coordinates.
        let outcome = [0.7, -1.3, 2.1, 0.4, -0.9, 1.8];
        let n = outcome.len();
        let y_kv: Vec<f64> = outcome
            .iter()
            .copied()
            .chain(std::iter::repeat_n(0.0, n))
            .collect();
        let eta_mu: Vec<f64> = vec![0.2, -0.5, 1.0, 0.0, -0.7, 0.9];
        let eta_kv_sigma: Vec<f64> = vec![-0.4, 0.1, -1.1, 0.6, -0.2, 0.3];
        let eta_gauss_sigma: Vec<f64> = eta_kv_sigma.iter().map(|v| v / 2.0).collect();
        let w = vec![1.0; n];

        let etas_kv = [Array1::from(eta_mu.clone()), Array1::from(eta_kv_sigma)];
        let etas_g = [Array1::from(eta_mu), Array1::from(eta_gauss_sigma)];
        let r_kv = GaussianKnownVarLSS.risk(&y_kv, &etas_kv, &w);
        let r_g = GaussianLSS.risk(&outcome, &etas_g, &w);
        assert!(
            (r_kv - r_g).abs() < 1e-10 * (1.0 + r_g.abs()),
            "risk mismatch at s2=0: knownvar {r_kv} vs gaussian {r_g}"
        );

        for k in 0..2 {
            let mut u_kv = vec![0.0; n];
            GaussianKnownVarLSS.ngradient(k, &y_kv, &etas_kv, &w, &mut u_kv);
            let mut u_g = vec![0.0; n];
            GaussianLSS.ngradient(k, &outcome, &etas_g, &w, &mut u_g);
            // Chain rule: eta_gauss = eta_kv / 2, so for k = 1 the knownvar
            // gradient is half the gaussian one (u_kv = 0.5 * u_g); k = 0
            // shares the identity link and must match exactly.
            let scale: f64 = if k == 0 { 1.0 } else { 0.5 };
            for i in 0..n {
                assert!(
                    (u_kv[i] - scale * u_g[i]).abs() < 1e-9 * (1.0 + u_g[i].abs()),
                    "k={k} obs {i}: knownvar {} vs gaussian*{scale} {}",
                    u_kv[i],
                    u_g[i]
                );
            }
        }
    }

    #[test]
    fn gaussian_knownvar_rejects_negative_s2_and_odd_length() {
        let fam = GaussianKnownVarLSS;
        let bad = ndarray::arr1(&[1.0, 2.0, 3.0]);
        assert!(fam.check_y(bad.view()).is_err(), "odd length must fail");
        let negative = ndarray::arr1(&[1.0, 2.0, -0.1, 3.0]);
        assert!(
            fam.check_y(negative.view()).is_err(),
            "negative s2 must fail"
        );
        let good = ndarray::arr1(&[1.0, 2.0, 0.0, 3.0]);
        assert!(fam.check_y(good.view()).is_ok());
    }

    /// Per-parameter learning rates: the recycling equivalence, the effect,
    /// and the validation, on the distributional drivers.
    mod nu_per_param {
        use super::*;

        fn fixture(n: usize) -> Array1<f64> {
            let x = Array1::<f64>::linspace(-1.5, 1.5, n);
            let mx = x.mean().unwrap();
            let sd_x = (x.iter().map(|&v| (v - mx) * (v - mx)).sum::<f64>() / n as f64).sqrt();
            x.iter().map(|&xi| 1.0 + 0.9 * (xi - mx) / sd_x).collect()
        }

        /// A control whose nu_per_param recycles the scalar nu reproduces the
        /// plain control's path exactly: selection order, coefficients, and
        /// joint risk.
        #[test]
        fn recycled_nu_reproduces_the_scalar_path() {
            let y = fixture(120);
            let w = Array1::<f64>::ones(120);
            let plain = BoostControl::new(30, 0.1).unwrap();
            let mut recycled = plain.clone();
            recycled.nu_per_param = Some(vec![0.1, 0.1]);

            let mut a = simple_gaussian_lss_learners(120, w.view());
            let path_a =
                try_fit_lss(&mut a, y.view(), w.view(), &GaussianLSS, &plain, None, None).unwrap();
            let mut b = simple_gaussian_lss_learners(120, w.view());
            let path_b = try_fit_lss(
                &mut b,
                y.view(),
                w.view(),
                &GaussianLSS,
                &recycled,
                None,
                None,
            )
            .unwrap();
            assert_eq!(path_a.risk, path_b.risk);
            assert_eq!(path_a.per_param[0].selected, path_b.per_param[0].selected);
            for (ca, cb) in path_a
                .per_param
                .iter()
                .zip(path_b.per_param.iter())
                .flat_map(|(pa, pb)| pa.coefs.iter().zip(pb.coefs.iter()))
            {
                assert_eq!(ca, cb);
            }
        }

        /// A smaller mu learning rate changes the mu trajectory -- the joint
        /// risk descends differently in early cycles -- while the sigma path,
        /// same rate and same selections, is unchanged. (By a large mstop both
        /// converge to the same least-squares limit, so the discriminator is
        /// the trajectory, not the endpoint.)
        #[test]
        fn a_smaller_mu_rate_slows_only_mu() {
            let y = fixture(120);
            let w = Array1::<f64>::ones(120);
            let mut slow_mu = BoostControl::new(30, 0.2).unwrap();
            slow_mu.nu_per_param = Some(vec![0.05, 0.2]);
            let mut sigma_only = BoostControl::new(30, 0.2).unwrap();
            sigma_only.nu_per_param = Some(vec![0.2, 0.2]);

            let mut a = simple_gaussian_lss_learners(120, w.view());
            let path_a = try_fit_lss(
                &mut a,
                y.view(),
                w.view(),
                &GaussianLSS,
                &slow_mu,
                None,
                None,
            )
            .unwrap();
            let mut b = simple_gaussian_lss_learners(120, w.view());
            let path_b = try_fit_lss(
                &mut b,
                y.view(),
                w.view(),
                &GaussianLSS,
                &sigma_only,
                None,
                None,
            )
            .unwrap();

            // The joint risk descends more slowly while mu catches up.
            let early = 4usize;
            assert!(
                path_a.risk[early] > path_b.risk[early],
                "cycle {early}: slow-mu risk {} should exceed {}",
                path_a.risk[early],
                path_b.risk[early]
            );
            // The sigma path keeps its own selections and, at the same rate,
            // coefficients that agree to numerical noise: the residual
            // coupling through the joint gradient moves them only at the
            // last-bits level, not by the rate change.
            assert_eq!(path_a.per_param[1].selected, path_b.per_param[1].selected);
            for (ca, cb) in path_a.per_param[1]
                .coefs
                .iter()
                .zip(path_b.per_param[1].coefs.iter())
            {
                for (va, vb) in ca.iter().zip(cb.iter()) {
                    assert!(
                        (va - vb).abs() < 1e-12,
                        "sigma coefficient moved by the mu rate change: {va} vs {vb}"
                    );
                }
            }
        }

        /// The validation the frontends rely on: one entry per parameter,
        /// each finite in (0, 1].
        #[test]
        fn invalid_per_parameter_rates_are_refused() {
            let mut control = BoostControl::new(10, 0.1).unwrap();
            control.nu_per_param = Some(vec![0.1]);
            let err = resolve_nu_per_param(&control, 2).expect_err("length mismatch must fire");
            assert!(format!("{err}").contains("one entry per parameter"));

            for bad in [0.0, -0.1, 1.5, f64::NAN] {
                let mut control = BoostControl::new(10, 0.1).unwrap();
                control.nu_per_param = Some(vec![0.1, bad]);
                let err = resolve_nu_per_param(&control, 2)
                    .expect_err("an out-of-range per-parameter nu must be refused");
                assert!(
                    format!("{err}").contains("0 < nu <= 1"),
                    "nu = {bad}: {err}"
                );
            }
            let mut control = BoostControl::new(10, 0.1).unwrap();
            control.nu_per_param = Some(vec![0.1, 1.0]);
            assert_eq!(resolve_nu_per_param(&control, 2).unwrap(), vec![0.1, 1.0]);
            assert_eq!(
                resolve_nu_per_param(&BoostControl::new(10, 0.1).unwrap(), 3).unwrap(),
                vec![0.1; 3]
            );
        }

        /// The diagonal-checkpoint driver honours the rates too: a recycled
        /// vector reproduces its plain path exactly.
        #[test]
        fn diagonal_checkpoints_honour_the_rates() {
            let y = fixture(60);
            let w = Array1::<f64>::ones(60);
            let plain = BoostControl::new(10, 0.1).unwrap();
            let mut recycled = plain.clone();
            recycled.nu_per_param = Some(vec![0.1, 0.1]);

            let mut a = simple_gaussian_lss_learners(60, w.view());
            let a = try_fit_lss_diagonal_checkpoints(
                &mut a,
                y.view(),
                w.view(),
                &GaussianLSS,
                &plain,
                &[3, 7],
                None,
            )
            .unwrap();
            let mut b = simple_gaussian_lss_learners(60, w.view());
            let b = try_fit_lss_diagonal_checkpoints(
                &mut b,
                y.view(),
                w.view(),
                &GaussianLSS,
                &recycled,
                &[3, 7],
                None,
            )
            .unwrap();
            for ((_, fitted_a), (_, fitted_b)) in a.iter().zip(b.iter()) {
                assert_eq!(fitted_a, fitted_b);
            }
        }
    }

    // ─── line-search step rule (E1) ──────────────────────────────────────

    #[test]
    fn line_search_keeps_a_tiny_scale_lss_fit_converging() {
        // The same fixture the divergence guard is probed on: sd(y) ~ 0.02
        // makes the fixed mu step nu / sigma^2 ~ 250x the least-squares
        // increment, so the fixed rule diverges and the guard stops the fit.
        // Searching each step on the joint likelihood takes steps the
        // coupled loss actually supports, and the fit completes.
        let n = 200;
        let x = Array1::<f64>::linspace(-1.0, 1.0, n);
        let mx = x.mean().unwrap();
        let sd_x = (x.iter().map(|&v| (v - mx) * (v - mx)).sum::<f64>() / n as f64).sqrt();
        let y: Array1<f64> = x.iter().map(|&xi| 1.0 + 0.02 * (xi - mx) / sd_x).collect();
        let w = Array1::ones(n);

        let fixed = BoostControl::new(50, 0.1).unwrap();
        let mut learners = simple_gaussian_lss_learners(n, w.view());
        let error = try_fit_lss(
            &mut learners,
            y.view(),
            w.view(),
            &GaussianLSS,
            &fixed,
            None,
            None,
        )
        .expect_err("the fixed step must diverge on this fixture");
        assert!(format!("{error}").contains("diverged"), "{error}");

        let searched = BoostControl::new(50, 0.1)
            .unwrap()
            .with_step(StepRule::LineSearch { max_iter: 4 });
        let mut learners = simple_gaussian_lss_learners(n, w.view());
        let path = try_fit_lss(
            &mut learners,
            y.view(),
            w.view(),
            &GaussianLSS,
            &searched,
            None,
            None,
        )
        .expect("the searched step must fit this fixture");
        assert!(path.risk.iter().all(|r| r.is_finite()));
        assert!(
            *path.risk.last().unwrap() < path.risk[0],
            "the searched path must reduce the joint risk: {} -> {}",
            path.risk[0],
            path.risk.last().unwrap()
        );
    }

    #[test]
    fn line_search_coefficients_replay_the_lss_fitted_path() {
        // As in the standard engine: the stored coefficient carries rho, so
        // nu * sum(coefs) replays each parameter's fitted predictor.
        let n = 120;
        let x = Array1::<f64>::linspace(-1.0, 1.0, n);
        let y: Array1<f64> = x.iter().map(|&xi| 2.0 + xi + 0.4 * xi * xi).collect();
        let w = Array1::ones(n);
        let control = BoostControl::new(30, 0.1)
            .unwrap()
            .with_step(StepRule::LineSearch { max_iter: 4 });
        let mut learners = simple_gaussian_lss_learners(n, w.view());
        let path = try_fit_lss(
            &mut learners,
            y.view(),
            w.view(),
            &GaussianLSS,
            &control,
            None,
            None,
        )
        .unwrap();
        for p in 0..2 {
            let replayed: Vec<f64> = (0..n)
                .map(|i| {
                    path.per_param[p].offset
                        + 0.1
                            * path.per_param[p]
                                .coefs
                                .iter()
                                .map(|coef| coef[0] * x[i])
                                .sum::<f64>()
                })
                .collect();
            let fitted: Vec<f64> = path.per_param[p].fitted.iter().copied().collect();
            let worst = fitted
                .iter()
                .zip(&replayed)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-9, "parameter {p} replay misses by {worst:.3e}");
        }
    }

    #[test]
    fn noncyclic_refuses_the_line_search_rule() {
        let n = 100;
        let x = Array1::<f64>::linspace(-1.0, 1.0, n);
        let y = x.mapv(|v: f64| 1.0 + 0.5 * v);
        let w = Array1::ones(n);
        let control = BoostControl::new(20, 0.1)
            .unwrap()
            .with_step(StepRule::LineSearch { max_iter: 4 });
        let mut learners = simple_gaussian_lss_learners(n, w.view());
        let error = try_fit_lss_noncyclic(
            &mut learners,
            y.view(),
            w.view(),
            &GaussianLSS,
            &control,
            None,
        )
        .expect_err("the non-cyclic method must refuse the rule");
        assert!(format!("{error}").contains("non-cyclic"), "{error}");
    }
}

#[cfg(test)]
mod stabilization_tests {
    use super::*;

    /// Reference values from `gamboostLSS:::weighted.median` and
    /// `stabilize_ngradient`, computed in R on the same inputs. Recomputing
    /// them here would only check this implementation against itself.
    #[test]
    fn weighted_median_matches_upstream_on_unit_weights() {
        // An odd count takes the middle value.
        let odd = [3.0, 1.0, 2.0];
        assert_eq!(weighted_median(&odd, &[1.0, 1.0, 1.0]), 2.0);
        // An even count averages the two middle values, because the weights
        // sum to an even whole number.
        let even = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(weighted_median(&even, &[1.0; 4]), 2.5);
    }

    /// R: `weighted.median(c(1, 2, 3), w = c(1, 1, 10))` is 2.5.
    ///
    /// Weight concentrated on the largest value pulls the median toward it,
    /// and the weights sum to 12 -- an even whole number -- so the two
    /// straddling values are averaged rather than the larger taken.
    #[test]
    fn weighted_median_uses_the_weights() {
        let values = [1.0, 2.0, 3.0];
        assert_eq!(weighted_median(&values, &[1.0, 1.0, 10.0]), 2.5);
    }

    #[test]
    fn weighted_median_ignores_zero_weight_observations() {
        let values = [100.0, 1.0, 2.0, 3.0];
        // Dropping the zero-weight 100 leaves an odd count of three.
        assert_eq!(weighted_median(&values, &[0.0, 1.0, 1.0, 1.0]), 2.0);
    }

    #[test]
    fn none_leaves_the_gradient_alone() {
        let mut gradient = [1.0, -2.0, 3.0];
        stabilize_ngradient(&mut gradient, &[1.0; 3], Stabilization::None);
        assert_eq!(gradient, [1.0, -2.0, 3.0]);
    }

    /// R:
    ///   ngr <- c(1, -2, 3, -4)
    ///   gamboostLSS:::stabilize_ngradient(ngr, w = rep(1, 4), "MAD")
    ///   # 0.4 -0.8 1.2 -1.6
    ///
    /// The intermediate values are median -0.5 and divisor 2.5. Both are
    /// upstream's, taken from R rather than recomputed here, because
    /// recomputing them would only check this implementation against itself.
    #[test]
    fn mad_divides_by_the_weighted_median_absolute_deviation() {
        let mut gradient = [1.0, -2.0, 3.0, -4.0];
        stabilize_ngradient(&mut gradient, &[1.0; 4], Stabilization::Mad);
        for (got, want) in gradient.iter().zip([0.4, -0.8, 1.2, -1.6]) {
            assert!((got - want).abs() < 1e-12, "{gradient:?}");
        }
    }

    /// R:
    ///   ngr <- c(3, 4)
    ///   div <- sqrt(weighted.mean(ngr^2, w = c(1, 1)))    # 3.535534
    ///   ngr / div   # 0.8485281 1.1313708
    #[test]
    fn l2_divides_by_the_root_weighted_mean_square() {
        let mut gradient = [3.0, 4.0];
        stabilize_ngradient(&mut gradient, &[1.0, 1.0], Stabilization::L2);
        let divisor = (12.5f64).sqrt();
        assert!((gradient[0] - 3.0 / divisor).abs() < 1e-12);
        assert!((gradient[1] - 4.0 / divisor).abs() < 1e-12);
    }

    /// The floor stops a gradient that is already numerically zero from being
    /// amplified into noise.
    #[test]
    fn a_vanishing_gradient_is_not_amplified() {
        let mut gradient = [1e-12, -1e-12];
        stabilize_ngradient(&mut gradient, &[1.0, 1.0], Stabilization::Mad);
        assert!(
            gradient.iter().all(|value| value.abs() <= 1e-7),
            "{gradient:?}"
        );

        let mut l2 = [1e-12, -1e-12];
        stabilize_ngradient(&mut l2, &[1.0, 1.0], Stabilization::L2);
        assert!(l2.iter().all(|value| value.abs() <= 1e-7), "{l2:?}");
    }

    /// The L2 ceiling stops a legitimately huge gradient from being shrunk to
    /// nothing. `Mad` has no ceiling upstream, and does not get one here.
    #[test]
    fn l2_caps_its_divisor_where_mad_does_not() {
        let mut l2 = [1e9, -1e9];
        stabilize_ngradient(&mut l2, &[1.0, 1.0], Stabilization::L2);
        // Divisor capped at 1e4, so the result is 1e5, not 1.
        assert!((l2[0] - 1e5).abs() / 1e5 < 1e-9, "{l2:?}");

        let mut mad = [1e9, -1e9];
        stabilize_ngradient(&mut mad, &[1.0, 1.0], Stabilization::Mad);
        assert!((mad[0] - 1.0).abs() < 1e-9, "{mad:?}");
    }

    #[test]
    fn parse_accepts_the_spellings_both_frontends_use() {
        assert_eq!(Stabilization::parse("none").unwrap(), Stabilization::None);
        assert_eq!(Stabilization::parse("MAD").unwrap(), Stabilization::Mad);
        assert_eq!(Stabilization::parse("mad").unwrap(), Stabilization::Mad);
        assert_eq!(Stabilization::parse("L2").unwrap(), Stabilization::L2);
        assert!(Stabilization::parse("robust").is_err());
    }
}

/// The exact PIG contract, against `gamlss.dist` and against itself.
///
/// P4.1 item 1: "Implement the exact PIG negative log-likelihood and gradients
/// with a tested modified-Bessel kernel; keep PIG fitting rejected until
/// finite differences, density normalization, simulation recovery, and R
/// parity pass."
///
/// What was there approximated the mu gradient as `y - mu` -- the Poisson one
/// -- with a comment saying so, and computed the risk as a Poisson negative
/// log-likelihood outright. Both ran and neither was PIG, so a fit would have
/// converged to the wrong model quietly; `check_y` refusing every response was
/// the only thing keeping that off the road.
#[cfg(test)]
mod pig_contract {
    use super::*;

    /// `log dPIG(y, mu, sigma, log = TRUE)` from `gamlss.dist` 6.1-1,
    /// `(y, mu, sigma, value)`.
    const R_LOG_DENSITY: &[(f64, f64, f64, f64)] = &[
        (0.0, 0.5, 0.2, -0.47722557505166),
        (1.0, 0.5, 0.2, -1.26153353400858),
        (3.0, 0.5, 0.2, -4.12251485486425),
        (0.0, 2.0, 1.0, -1.23606797749979),
        (1.0, 2.0, 1.0, -1.34763975315689),
        (4.0, 2.0, 1.0, -2.77785691633558),
        (0.0, 10.0, 3.0, -2.27008322530222),
        (2.0, 10.0, 3.0, -2.14387628799603),
        (6.0, 10.0, 3.0, -3.05526543028052),
    ];

    fn log_density(y: f64, mu: f64, sigma: f64) -> f64 {
        pig_terms(y, mu, sigma).log_density
    }

    #[test]
    fn the_bessel_kernel_matches_its_closed_form() {
        // K_{1/2}(x) = K_{-1/2}(x) = sqrt(pi / 2x) e^{-x}, and
        // K_{3/2}(x) = K_{1/2}(x) (1 + 1/x).
        for x in [0.05_f64, 0.5, 1.0, 7.0, 90.0] {
            let logs = log_bessel_k_half_integer(x, 3);
            let half = 0.5 * (std::f64::consts::PI / (2.0 * x)).ln() - x;
            assert!((logs[0] - half).abs() < 1e-13, "K_-1/2 at x={x}");
            assert!((logs[1] - half).abs() < 1e-13, "K_1/2 at x={x}");
            let three_halves = half + (1.0 + 1.0 / x).ln();
            assert!(
                (logs[2] - three_halves).abs() < 1e-12,
                "K_3/2 at x={x}: {} vs {three_halves}",
                logs[2]
            );
        }
    }

    #[test]
    fn the_kernel_stays_finite_where_the_direct_recurrence_overflows() {
        // K_{y-1/2} at a small argument grows past f64::MAX well before the
        // counts a real fit sees, which is why the recurrence runs in logs.
        let logs = log_bessel_k_half_integer(0.3, 400);
        assert!(logs[400].is_finite());
        assert!(logs[400] > 700.0, "the direct form would have overflowed");
        for pair in logs.windows(2).skip(1) {
            assert!(pair[1] >= pair[0], "K_v is increasing in v for v > 0");
        }
    }

    #[test]
    fn the_log_density_matches_gamlss() {
        for &(y, mu, sigma, expected) in R_LOG_DENSITY {
            let got = log_density(y, mu, sigma);
            assert!(
                (got - expected).abs() < 1e-7,
                "dPIG({y}, {mu}, {sigma}): {got} against R's {expected}"
            );
        }
    }

    #[test]
    fn the_density_sums_to_one() {
        for (mu, sigma) in [(0.5_f64, 0.2_f64), (2.0, 1.0), (10.0, 3.0)] {
            // Far enough out that the truncated tail is below the
            // tolerance. R puts 4.4e-09 beyond y = 800 for (10, 3) -- the
            // heaviest tail here -- and 5.6e-16 beyond 2,000.
            let total: f64 = (0..2_000)
                .map(|y| log_density(y as f64, mu, sigma).exp())
                .sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "sum_y f(y | {mu}, {sigma}) = {total}"
            );
        }
    }

    #[test]
    fn the_moments_are_the_ones_pig_claims() {
        // E(Y) = mu, Var(Y) = mu + sigma mu^2. The struct doc says the
        // variance is `mu + mu^2/sigma`, which is a different distribution;
        // the parametrisation here is gamlss's.
        for (mu, sigma) in [(0.5_f64, 0.2_f64), (2.0, 1.0), (10.0, 3.0)] {
            let mut mean = 0.0;
            let mut second = 0.0;
            for y in 0..2_000 {
                let p = log_density(y as f64, mu, sigma).exp();
                mean += y as f64 * p;
                second += (y as f64) * (y as f64) * p;
            }
            let variance = second - mean * mean;
            assert!((mean - mu).abs() < 1e-7, "mean {mean} against {mu}");
            let expected = mu + sigma * mu * mu;
            assert!(
                (variance - expected).abs() < 1e-6,
                "variance {variance} against {expected}"
            );
        }
    }

    #[test]
    fn the_gradients_match_a_finite_difference_of_the_risk() {
        let family = PIGLSS;
        let step = 1e-6;
        for (mu, sigma) in [(0.5_f64, 0.2_f64), (2.0, 1.0), (10.0, 3.0)] {
            for y in [0.0_f64, 1.0, 3.0, 9.0] {
                let eta = [mu.ln(), sigma.ln()];
                for parameter in 0..2 {
                    let risk_at = |shift: f64| {
                        let mut moved = eta;
                        moved[parameter] += shift;
                        family.risk(
                            &[y],
                            &[
                                Array1::from_elem(1, moved[0]),
                                Array1::from_elem(1, moved[1]),
                            ],
                            &[1.0],
                        )
                    };
                    let numeric = (risk_at(step) - risk_at(-step)) / (2.0 * step);
                    let mut analytic = [0.0];
                    family.ngradient(
                        parameter,
                        &[y],
                        &[Array1::from_elem(1, eta[0]), Array1::from_elem(1, eta[1])],
                        &[1.0],
                        &mut analytic,
                    );
                    // The negative gradient is -d(risk)/d(eta).
                    assert!(
                        (analytic[0] + numeric).abs() < 1e-5,
                        "parameter {parameter} at y={y}, mu={mu}, sigma={sigma}: \
                         analytic {} against numeric {}",
                        analytic[0],
                        -numeric
                    );
                }
            }
        }
    }

    #[test]
    fn a_non_count_response_is_refused_by_name() {
        let family = PIGLSS;
        let bad = ndarray::array![0.0, 1.0, 2.5];
        let error = family.check_y(bad.view()).unwrap_err();
        assert!(
            format!("{error}").contains("non-negative whole count"),
            "{error}"
        );
        let good = ndarray::array![0.0, 1.0, 7.0];
        assert!(family.check_y(good.view()).is_ok());
    }
}
