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

use crate::control::{BoostControl, ResponseScale, StepRule};
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

/// How one distribution parameter's linear predictor changes when the
/// response is divided by a scale `s`: the model fitted to `y / s` has
/// predictor `eta_s`, and the same model for `y` has `eta = a * eta_s + b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamScale {
    /// `eta = s * eta_s`: an identity-link location or scale parameter.
    Multiply,
    /// `eta = eta_s + c * ln(s)`: a log-link parameter proportional to
    /// `s^c` (a mean or scale, `c = 1`; a variance, `c = 2`), or an
    /// identity-link location on the log scale of the response.
    Shift(f64),
    /// `eta = eta_s`: a shape parameter that does not depend on the units.
    Invariant,
}

impl ParamScale {
    /// The `(a, b)` of `eta = a * eta_s + b` for scale `s`.
    pub fn affine(self, s: f64) -> (f64, f64) {
        match self {
            ParamScale::Multiply => (s, 0.0),
            ParamScale::Shift(c) => (1.0, c * s.ln()),
            ParamScale::Invariant => (1.0, 0.0),
        }
    }
}

/// A distributional fit run on `y / s` and mapped back (see
/// [`BoostControl::response_scale`]).
///
/// The family's [`LssFamily::scale_map`] says how each parameter transforms;
/// [`LssFamily::scale_jacobian_weight`] gives the risk shift
/// `risk(y) = risk(y / s) + ln(s) * J`, the change of variables of every
/// density on the response.
#[derive(Clone, Debug)]
pub struct ResponseScaling {
    scale: f64,
    affine: Vec<(f64, f64)>,
}

impl ResponseScaling {
    /// The scaling `control` asks for on the response `y` with weights `w`,
    /// or `None` when it asks for none, the family has no scale map (counts,
    /// proportions, compositions), or an automatic scale finds no spread.
    pub fn resolve(
        family: &dyn LssFamily,
        control: &BoostControl,
        y: &[f64],
        w: &[f64],
    ) -> Result<Option<Self>, MboostrsError> {
        let scale = match control.response_scale {
            ResponseScale::Raw => return Ok(None),
            ResponseScale::Fixed(s) => s,
            ResponseScale::Auto => match auto_response_scale(family, y, w) {
                Some(s) => s,
                None => return Ok(None),
            },
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(MboostrsError::Control(format!(
                "response_scale must be finite and positive, got {scale}"
            )));
        }
        let Some(map) = family.scale_map() else {
            return Ok(None);
        };
        if map.len() != family.n_params() {
            return Err(MboostrsError::Shape(format!(
                "{} scale map has {} entries for {} parameters",
                family.name(),
                map.len(),
                family.n_params()
            )));
        }
        Ok(Some(Self {
            scale,
            affine: map.iter().map(|m| m.affine(scale)).collect(),
        }))
    }

    /// The scale `s`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The response the internal fit sees.
    pub fn internal_response(&self, family: &dyn LssFamily, y: &[f64], n: usize) -> Vec<f64> {
        family.scale_response(y, n, self.scale)
    }

    /// Caller-supplied offsets, converted to the internal units.
    pub fn internal_offsets(&self, offsets: Option<&[f64]>) -> Option<Vec<f64>> {
        offsets.map(|o| {
            o.iter()
                .zip(&self.affine)
                .map(|(off, (a, b))| (off - b) / a)
                .collect()
        })
    }

    /// The risk shift `ln(s) * J` for the original response `y` and weights `w`.
    pub fn risk_shift(&self, family: &dyn LssFamily, y: &[f64], w: &[f64]) -> f64 {
        self.scale.ln() * family.scale_jacobian_weight(y, w)
    }

    /// Map internal predictors to the original units, in place.
    pub fn restore_etas(&self, etas: &mut [Array1<f64>]) {
        for (eta, (a, b)) in etas.iter_mut().zip(&self.affine) {
            eta.mapv_inplace(|e| a * e + b);
        }
    }

    /// Map a fitted path to the original units: offsets and fitted
    /// predictors through `a * eta + b`, every stored coefficient through the
    /// selected learner's [`PreparedLearner::scale_coef`] by `a`, and every
    /// risk by `shift` (see [`ResponseScaling::risk_shift`]).
    pub fn restore_path(
        &self,
        path: &mut LssPath,
        learners_per_param: &[Vec<PreparedLearner>],
        shift: f64,
    ) {
        for (p, bp) in path.per_param.iter_mut().enumerate() {
            let (a, b) = self.affine[p];
            bp.offset = a * bp.offset + b;
            bp.fitted.mapv_inplace(|e| a * e + b);
            if a != 1.0 {
                for (coef, &idx) in bp.coefs.iter_mut().zip(&bp.selected) {
                    *coef = learners_per_param[p][idx].scale_coef(coef, a);
                }
            }
            for r in bp.risk.iter_mut() {
                *r += shift;
            }
        }
        for r in path.risk.iter_mut() {
            *r += shift;
        }
    }
}

/// The weighted SD of the part of `y` a family's scale applies to, the
/// automatic `response_scale` ([`ResponseScale::Auto`], `"auto"` in the
/// Python and R interfaces): `sqrt(sum w (y - m)^2 / sum w)` with `m` the
/// weighted mean. A fit sees the rows its weights give, so a fold, a weighted
/// refit and a fit on the same rows as a subset take the same scale. `None`
/// when the family has no scale map, or the SD is zero or not finite.
pub fn auto_response_scale(family: &dyn LssFamily, y: &[f64], w: &[f64]) -> Option<f64> {
    family.scale_map()?;
    let n = w.len();
    let source = family.scale_source(y, n);
    let (mut sw, mut swy) = (0.0f64, 0.0f64);
    for (&v, &wi) in source.iter().zip(w) {
        if wi > 0.0 && v.is_finite() {
            sw += wi;
            swy += wi * v;
        }
    }
    if sw <= 0.0 {
        return None;
    }
    let mean = swy / sw;
    let mut ss = 0.0f64;
    for (&v, &wi) in source.iter().zip(w) {
        if wi > 0.0 && v.is_finite() {
            ss += wi * (v - mean) * (v - mean);
        }
    }
    let sd = (ss / sw).sqrt();
    (sd.is_finite() && sd > 0.0).then_some(sd)
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

    /// How each parameter's predictor transforms when the response is divided
    /// by a scale (see [`ParamScale`]), or `None` when the family's response
    /// has no unit to change (counts, proportions, compositions). `None` by
    /// default.
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        None
    }

    /// The response divided by `s`. `n` is the observation count; by default
    /// every stored value is divided. Families whose response carries other
    /// data (a censoring status, a known variance) override this.
    fn scale_response(&self, y: &[f64], n: usize, s: f64) -> Vec<f64> {
        let _ = n;
        y.iter().map(|v| v / s).collect()
    }

    /// The values whose spread sets the automatic scale: the first `n` stored
    /// values (the outcome, or the time of a survival response).
    fn scale_source<'a>(&self, y: &'a [f64], n: usize) -> &'a [f64] {
        &y[..n.min(y.len())]
    }

    /// The weight `J` of the rows whose likelihood is a density on the
    /// response, so that `risk(y) = risk(y / s) + ln(s) * J` at matching
    /// predictors. All rows by default.
    fn scale_jacobian_weight(&self, y: &[f64], w: &[f64]) -> f64 {
        let _ = y;
        w.iter().sum()
    }

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
             pushed a predictor far enough that the risk overflowed. Without \
             stabilization a step is proportional to the gradient, which \
             GaussianLSS divides by sigma^2, so rows with a spread far below \
             the response's overall spread take steps far larger than the \
             others; set `stabilization=\"MAD\"` on the family, lower `nu`, or \
             use a family for the response's shape (GammaLSS or LogNormalLSS \
             for a positive, right-skewed response). Without \
             `response_scale=\"auto\"` (the default), also express the \
             response in units where its standard deviation is of order 1.",
            cycle + 1
        )));
    }
    let divergence_increment = LSS_DIVERGENCE_FACTOR * (best_risk.abs() + 1.0);
    if best_risk.is_finite() && step_risk - *best_risk > divergence_increment {
        return Err(MboostrsError::Solver(format!(
            "{method} LSS risk diverged after cycle {}: {:.6e} against a \
             running minimum of {:.6e}. Component-wise steps are proportional \
             to the gradient and overshoot: GaussianLSS, for example, updates \
             `mu` along `(y - mu)/sigma^2`, so rows whose spread is far below \
             the response's overall spread take steps far larger than the \
             others. Set `stabilization=\"MAD\"` on the family, lower `nu`, or \
             use a family for the response's shape (GammaLSS or LogNormalLSS \
             for a positive, right-skewed response). Without \
             `response_scale=\"auto\"` (the default), also express the \
             response in units where its standard deviation is of order 1 \
             (per cent rather than decimal fractions for returns).",
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

/// The response and weights as contiguous slices.
fn contiguous_response<'a>(
    y: &'a ArrayView1<f64>,
    weights: &'a ArrayView1<f64>,
) -> Result<(&'a [f64], &'a [f64]), MboostrsError> {
    let ys = y
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS response must be contiguous".to_string()))?;
    let ws = weights
        .as_slice()
        .ok_or_else(|| MboostrsError::Shape("LSS weights must be contiguous".to_string()))?;
    Ok((ys, ws))
}

/// Fallible cyclic LSS fitting with typed validation errors.
///
/// With [`BoostControl::response_scale`] set (and a family with a
/// [`LssFamily::scale_map`]), the fit runs on `y / s` and the returned path is
/// mapped back to the original units: offsets, fitted predictors,
/// coefficients and risks.
pub fn try_fit_lss(
    learners_per_param: &mut [Vec<PreparedLearner>],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    family: &dyn LssFamily,
    control: &BoostControl,
    mstop_per_param: Option<&[usize]>,
    offsets: Option<&[f64]>,
) -> Result<LssPath, MboostrsError> {
    let (ys, ws) = contiguous_response(&y, &weights)?;
    let Some(scaling) = ResponseScaling::resolve(family, control, ys, ws)? else {
        return try_fit_lss_raw(
            learners_per_param,
            y,
            weights,
            family,
            control,
            mstop_per_param,
            offsets,
        );
    };
    let internal = scaling.internal_response(family, ys, ws.len());
    let internal_offsets = scaling.internal_offsets(offsets);
    let mut path = try_fit_lss_raw(
        learners_per_param,
        ArrayView1::from(&internal[..]),
        weights,
        family,
        control,
        mstop_per_param,
        internal_offsets.as_deref(),
    )?;
    scaling.restore_path(
        &mut path,
        learners_per_param,
        scaling.risk_shift(family, ys, ws),
    );
    Ok(path)
}

/// [`try_fit_lss`] on the response as given, without
/// [`BoostControl::response_scale`].
fn try_fit_lss_raw(
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
    let (ys, ws) = contiguous_response(&y, &weights)?;
    let Some(scaling) = ResponseScaling::resolve(family, control, ys, ws)? else {
        return try_fit_lss_diagonal_checkpoints_raw(
            learners_per_param,
            y,
            weights,
            family,
            control,
            checkpoints,
            offsets,
        );
    };
    let internal = scaling.internal_response(family, ys, ws.len());
    let internal_offsets = scaling.internal_offsets(offsets);
    let mut snapshots = try_fit_lss_diagonal_checkpoints_raw(
        learners_per_param,
        ArrayView1::from(&internal[..]),
        weights,
        family,
        control,
        checkpoints,
        internal_offsets.as_deref(),
    )?;
    for (_, etas) in snapshots.iter_mut() {
        scaling.restore_etas(etas);
    }
    Ok(snapshots)
}

/// [`try_fit_lss_diagonal_checkpoints`] on the response as given.
fn try_fit_lss_diagonal_checkpoints_raw(
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
/// A fold's path in a resumable search: cyclic (one step per parameter per
/// iteration) or non-cyclic (one step in total per iteration).
enum FoldPath {
    Cyclic(LssDiagonalPath),
    Noncyclic(LssNoncyclicPath),
}

impl FoldPath {
    fn etas(&self) -> &[Array1<f64>] {
        match self {
            FoldPath::Cyclic(path) => &path.etas,
            FoldPath::Noncyclic(path) => &path.etas,
        }
    }

    /// Advance to iteration `to` (cycles, or total steps), calling
    /// `after(iteration, etas)` after each.
    #[allow(clippy::too_many_arguments)]
    fn advance<F: FnMut(usize, &[Array1<f64>])>(
        &mut self,
        learners: &mut [Vec<PreparedLearner>],
        ys: &[f64],
        ws: &[f64],
        weights: ArrayView1<f64>,
        family: &dyn LssFamily,
        stabilization: Stabilization,
        to: usize,
        mut after: F,
    ) -> Result<(), MboostrsError> {
        match self {
            FoldPath::Cyclic(path) => {
                path.advance(learners, ys, ws, weights, family, stabilization, to, after)
            }
            FoldPath::Noncyclic(path) => {
                while path.steps < to {
                    path.step(learners, ys, ws, weights, family)?;
                    after(path.steps, &path.etas);
                }
                Ok(())
            }
        }
    }
}

struct LssFoldSearch {
    learners: Vec<Vec<PreparedLearner>>,
    fold_w: Array1<f64>,
    path: FoldPath,
    oob_w: Vec<f64>,
    oob_sum: f64,
    /// Added to every held-out risk: the change of units of a scaled fit,
    /// per unit of out-of-bag weight (0 when the response is not scaled).
    risk_shift: f64,
    /// The response the fold's path fits, when it is scaled.
    ys_fit: Option<Vec<f64>>,
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
            risk_shift,
            ys_fit,
            started,
        } = self;
        let ys: &[f64] = ys_fit.as_deref().unwrap_or(ys);
        let oob: &[f64] = oob_w.as_slice();
        let oob_sum = *oob_sum;
        let shift = *risk_shift;
        let mut fresh = Vec::new();
        if !*started {
            fresh.push(oob_risk(family, ys, path.etas(), oob, oob_sum, needed, 0) + shift);
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
            |m, etas| fresh.push(oob_risk(family, ys, etas, oob, oob_sum, needed, m) + shift),
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
/// With `noncyclic`, each fold runs the non-cyclic algorithm and an
/// iteration is one step in total (one parameter updated), so the grid counts
/// total steps; its path is prefix-consistent in the same way.
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
    noncyclic: bool,
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

    if noncyclic && !matches!(control.step, StepRule::Fixed) {
        return Err(MboostrsError::Control(
            "step='line_search' is implemented for the cyclic LSS methods only".into(),
        ));
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
            // A scaled fold fits `y / s`, with `s` from its own in-bag rows as a
            // fit on those rows would take it, and reports each held-out risk
            // in the original units (`risk(y) = risk(y / s) + ln(s) * J`).
            let scaling = ResponseScaling::resolve(family, control, ys, fold_ws)?;
            let ys_fit: Option<Vec<f64>> = scaling
                .as_ref()
                .map(|sc| sc.internal_response(family, ys, n));
            let fit_y: &[f64] = ys_fit.as_deref().unwrap_or(ys);
            let path = if noncyclic {
                FoldPath::Noncyclic(
                    LssNoncyclicPath::start(family, control, fit_y, fold_ws, None)?.0,
                )
            } else {
                FoldPath::Cyclic(LssDiagonalPath::start(
                    family, control, fit_y, fold_ws, None,
                )?)
            };
            let risk_shift = scaling
                .as_ref()
                .map_or(0.0, |sc| sc.risk_shift(family, ys, &oob_w) / oob_sum);
            Ok(LssFoldSearch {
                learners,
                fold_w,
                path,
                oob_w,
                oob_sum,
                risk_shift,
                ys_fit,
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
    let (ys, ws) = contiguous_response(&y, &weights)?;
    let Some(scaling) = ResponseScaling::resolve(family, control, ys, ws)? else {
        return try_fit_lss_noncyclic_capped_raw(
            learners_per_param,
            y,
            weights,
            family,
            control,
            offsets,
            stop_after_distinct,
        );
    };
    let internal = scaling.internal_response(family, ys, ws.len());
    let internal_offsets = scaling.internal_offsets(offsets);
    let mut path = try_fit_lss_noncyclic_capped_raw(
        learners_per_param,
        ArrayView1::from(&internal[..]),
        weights,
        family,
        control,
        internal_offsets.as_deref(),
        stop_after_distinct,
    )?;
    scaling.restore_path(
        &mut path,
        learners_per_param,
        scaling.risk_shift(family, ys, ws),
    );
    Ok(path)
}

/// The state of a non-cyclic LSS fit between steps: the predictors, the
/// parameter the last step updated and its predictor before that update, and
/// the candidate buffers. [`LssNoncyclicPath::step`] takes one step, so a fit
/// can be run to any length and resumed: the first `m` steps do not depend on
/// how far the fit runs. The fit and the resumable cross-validation search
/// share it.
pub(crate) struct LssNoncyclicPath {
    pub(crate) etas: Vec<Array1<f64>>,
    nus: Vec<f64>,
    stabilization: Stabilization,
    last_best: Option<usize>,
    prev_eta: Array1<f64>,
    best_cycle_risk: f64,
    /// Steps taken.
    pub(crate) steps: usize,
    u: Vec<f64>,
    cand_pred: Vec<Vec<f64>>,
    pub(crate) cand_beta: Vec<Vec<f64>>,
    pub(crate) cand_idx: Vec<usize>,
    pub(crate) cand_tied: Vec<bool>,
}

/// What one non-cyclic step did.
pub(crate) struct NoncyclicStep {
    /// The parameter updated.
    pub(crate) param: usize,
    /// Whether another parameter's update reached the same joint risk.
    pub(crate) multiple_best: bool,
    /// The joint risk after the step.
    pub(crate) risk: f64,
}

impl LssNoncyclicPath {
    /// Start at the offsets (computed on `ys`/`ws` unless given), returning
    /// the path, the offsets and the initial risk.
    pub(crate) fn start(
        family: &dyn LssFamily,
        control: &BoostControl,
        ys: &[f64],
        ws: &[f64],
        offsets: Option<&[f64]>,
    ) -> Result<(Self, Vec<f64>, f64), MboostrsError> {
        let n = ws.len();
        let k = family.n_params();
        let nus = resolve_nu_per_param(control, k)?;
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
        let initial_risk = family.risk(ys, &etas, ws);
        if !initial_risk.is_finite() {
            return Err(MboostrsError::Solver(
                "LSS initial risk is non-finite".to_string(),
            ));
        }
        Ok((
            Self {
                etas,
                nus,
                stabilization: control.stabilization,
                last_best: None,
                prev_eta: Array1::<f64>::zeros(n),
                best_cycle_risk: initial_risk,
                steps: 0,
                u: vec![0.0f64; n],
                cand_pred: (0..k).map(|_| vec![0.0; n]).collect(),
                cand_beta: (0..k).map(|_| Vec::new()).collect(),
                cand_idx: vec![0; k],
                cand_tied: vec![false; k],
            },
            offs,
            initial_risk,
        ))
    }

    /// Take one step: score every parameter's best base-learner update by the
    /// joint risk it reaches and apply only the best.
    pub(crate) fn step(
        &mut self,
        learners_per_param: &mut [Vec<PreparedLearner>],
        ys: &[f64],
        ws: &[f64],
        weights: ArrayView1<f64>,
        family: &dyn LssFamily,
    ) -> Result<NoncyclicStep, MboostrsError> {
        let LssNoncyclicPath {
            etas,
            nus,
            stabilization,
            last_best,
            prev_eta,
            best_cycle_risk,
            steps,
            u,
            cand_pred,
            cand_beta,
            cand_idx,
            cand_tied,
        } = self;
        let n = ws.len();
        let k = family.n_params();
        let m = *steps;
        // 1. Best base-learner update for each parameter. gamboostLSS
        // evaluates a parameter's candidate with the gradient its own model
        // last computed, which predates the previous step's update unless
        // that step updated this parameter; follow it, so the candidate of a
        // parameter other than the last-updated one uses the gradient with
        // the last-updated parameter at its value before that update.
        for p in 0..k {
            let stale = matches!(*last_best, Some(lb) if lb != p);
            if stale {
                let lb = last_best.expect("checked");
                std::mem::swap(&mut etas[lb], prev_eta);
                family.ngradient(p, ys, etas, ws, u);
                std::mem::swap(&mut etas[lb], prev_eta);
            } else {
                family.ngradient(p, ys, etas, ws, u);
            }
            // Put the parameters' gradients on comparable footing before
            // selection, so one `nu` does not take a large step on one and a
            // negligible one on another. A no-op under the default.
            stabilize_ngradient(u, ws, *stabilization);
            cand_tied[p] = noncyclic_candidate(
                &mut learners_per_param[p],
                u,
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
            let r = family.risk(ys, etas, ws);
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
        if matches!(*last_best, Some(lb) if lb != best_p) {
            family.ngradient(best_p, ys, etas, ws, u);
            stabilize_ngradient(u, ws, *stabilization);
            cand_tied[best_p] = noncyclic_candidate(
                &mut learners_per_param[best_p],
                u,
                ws,
                weights,
                n,
                &mut cand_idx[best_p],
                &mut cand_beta[best_p],
                &mut cand_pred[best_p],
            );
        }
        prev_eta.assign(&etas[best_p]);
        *last_best = Some(best_p);
        {
            let ep = etas[best_p].as_slice_mut().expect("eta contiguous");
            for i in 0..n {
                ep[i] += nus[best_p] * cand_pred[best_p][i];
            }
        }
        let step_risk = family.risk(ys, etas, ws);
        check_lss_cycle_risk(step_risk, best_cycle_risk, m, "non-cyclic")?;
        *steps += 1;
        Ok(NoncyclicStep {
            param: best_p,
            multiple_best: best_param_multiplicity > 1,
            risk: step_risk,
        })
    }
}

/// [`try_fit_lss_noncyclic_capped`] on the response as given.
fn try_fit_lss_noncyclic_capped_raw(
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

    let (mut path, offs, initial_risk) = LssNoncyclicPath::start(family, control, ys, ws, offsets)?;

    let mut selected: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    let mut coefs: Vec<Vec<Vec<f64>>> = (0..k).map(|_| Vec::new()).collect();
    let mut risk_path = Vec::with_capacity(mstop + 1);
    risk_path.push(initial_risk);
    let setup_ns = fit_started.map(|started| started.elapsed().as_nanos() as u64);
    let iteration_started = timing_enabled.then(Instant::now);

    let mut selection_order: Vec<usize> = Vec::with_capacity(mstop);
    let mut tie_selection_counts = vec![0usize; k];
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

    for _ in 0..mstop {
        let step = path.step(learners_per_param, ys, ws, weights, family)?;
        let best_p = step.param;
        selection_order.push(best_p);
        tie_selection_counts[best_p] += usize::from(path.cand_tied[best_p]);
        tie_selection_counts[best_p] += usize::from(step.multiple_best);
        selected[best_p].push(path.cand_idx[best_p]);
        coefs[best_p].push(path.cand_beta[best_p].clone());
        risk_path.push(step.risk);
        if let Some(limit) = stop_after_distinct {
            let seen = &mut distinct_seen[best_p][path.cand_idx[best_p]];
            if !*seen {
                *seen = true;
                distinct_count += 1;
            }
            if distinct_count >= limit {
                break;
            }
        }
    }
    let etas = path.etas;

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
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Multiply, ParamScale::Shift(1.0)];
        Some(MAP)
    }

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
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Multiply, ParamScale::Shift(2.0)];
        Some(MAP)
    }

    fn scale_response(&self, y: &[f64], n: usize, s: f64) -> Vec<f64> {
        // `[outcome.., s2..]`: the known sampling variance is in squared units.
        y.iter()
            .enumerate()
            .map(|(i, v)| if i < n { v / s } else { v / (s * s) })
            .collect()
    }

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
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Shift(1.0), ParamScale::Invariant];
        Some(MAP)
    }

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
                // R searches [0, max(y^2)], which cannot reach a negative
                // log-mean: a response whose mean is below 1 gets the boundary,
                // a mean of 1. The lower end is opened only then, so every
                // offset R can represent is unchanged.
                let lower = if wmean > 0.0 && wmean < 1.0 {
                    wmean.ln() - 1.0
                } else {
                    0.0
                };
                minimize_brent(&risk, lower, y_max_sq.max(lower + 1.0), R_OPTIMIZE_TOL, 500)
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
                // R searches the log shape over [0, max(y)]: a shape below 1
                // (a response more skewed than an exponential) is out of reach,
                // and so is any shape when max(y) is small. Opened only when
                // R's interval ends at the optimum's side of a boundary.
                let (lower, upper) = (0.0, y_max);
                let slope_at = |x: f64| risk(x + 1e-6) - risk(x - 1e-6);
                let lower = if slope_at(lower) > 0.0 { -10.0 } else { lower };
                let upper = if upper <= lower || slope_at(upper) < 0.0 {
                    upper.max(lower) + 10.0
                } else {
                    upper
                };
                minimize_brent(&risk, lower, upper, R_OPTIMIZE_TOL, 500)
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
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[
            ParamScale::Multiply,
            ParamScale::Shift(1.0),
            ParamScale::Invariant,
        ];
        Some(MAP)
    }

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
    fn scale_map(&self) -> Option<&'static [ParamScale]> {
        const MAP: &[ParamScale] = &[ParamScale::Shift(1.0), ParamScale::Invariant];
        Some(MAP)
    }

    fn scale_response(&self, y: &[f64], n: usize, s: f64) -> Vec<f64> {
        // `[time.., status..]` for censored data: only the times have units.
        y.iter()
            .enumerate()
            .map(|(i, v)| if i < n { v / s } else { *v })
            .collect()
    }

    fn scale_jacobian_weight(&self, _y: &[f64], w: &[f64]) -> f64 {
        // The censored likelihood is a density on log(T), which moves with
        // the location and needs no Jacobian; the uncensored one is a density
        // on T itself.
        if self.uncensored {
            w.iter().sum()
        } else {
            0.0
        }
    }

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

mod gamlss_families;
pub use self::gamlss_families::*;
#[allow(unused_imports)]
use self::gamlss_families::*;

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

#[cfg(test)]
mod test_suite;
