//! Boost control parameters.

use crate::report::InstrumentationLevel;
use crate::MboostrsError;

/// Risk estimation method used during boosting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RiskKind {
    #[default]
    /// In-bag risk: evaluate on the same data used to fit.
    Inbag,
    /// Out-of-bag risk: evaluate on observations left out per iteration.
    Oobag,
    /// No risk is computed.
    None,
}

/// How the step length is chosen once the winning base learner is selected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum StepRule {
    #[default]
    /// `f += nu · h`, with `h` the winner's least-squares fit to the negative
    /// gradient: the constant step Bühlmann & Hothorn (2007) describe and R's
    /// `mboost` implements. The default.
    Fixed,
    /// Find `rho` minimizing the training risk along the winner's direction,
    /// `R(f + rho·h)`, then take `f += nu·rho·h`: `nu` becomes a fraction of
    /// the loss-optimal step, so its meaning no longer depends on the
    /// response scale. Selection is unchanged. Friedman's (2001) original
    /// gradient boosting multiplies each fitted learner by this `rho` before
    /// shrinking by `nu`; mboost omits the search.
    ///
    /// The search probes `rho` on the grid 0, ⅛, ¼, ½, 1, 2, 4 (doubling to at most
    /// 64 while the risk still falls and halving below the grid if needed), then
    /// refines the best interval with `max_iter` golden-section iterations.
    /// Gamma uses a stable risk difference rather than subtracting nearly equal
    /// losses. Each evaluation is `O(n)` on
    /// the training weights. For the
    /// squared-error families the least-squares step is already optimal
    /// (`rho = 1` exactly), the covariance/Gram route is untouched, and a
    /// line-search fit is bit-identical to `Fixed`.
    LineSearch {
        /// Golden-section refinement iterations per boosting step. Small
        /// values (2–8) suffice: the grid has already bracketed the minimum.
        max_iter: usize,
    },
}

/// Control parameters for the boosting algorithm.
///
/// Mirrors R's `mboost::boost_control(mstop = …, nu = …, risk = …)`.
#[derive(Clone, Debug)]
pub struct BoostControl {
    /// Maximum number of boosting iterations. Must be > 0.
    pub mstop: u32,
    /// Step size (shrinkage factor). Must satisfy 0 < nu ≤ 1.
    pub nu: f64,
    /// Risk estimation method.
    pub risk: RiskKind,
    /// Whether to print progress information.
    pub trace: bool,
    /// Structured instrumentation level. Route metadata is always recorded;
    /// clocks are read only for `Summary` and `Iterations`.
    pub instrumentation: InstrumentationLevel,
    /// Stochastic component (base-learner) subsampling. `None` evaluates every
    /// candidate at each iteration (the default, and R `mboost`'s behavior).
    /// `Some(f)` with `0 < f < 1` evaluates a random fraction `f` of the
    /// candidates for selection at each iteration, reducing per-iteration cost
    /// on wide-`p` models and perturbing selection frequencies. The subset is
    /// drawn deterministically from `subsample_seed`, so a fixed seed gives a
    /// reproducible path independent of thread count.
    pub component_fraction: Option<f64>,
    /// Seed for component subsampling. Ignored when `component_fraction` is
    /// `None`.
    pub subsample_seed: u64,
    /// How a distributional fit rescales each partial gradient before
    /// selection. **Distributional fits only**: a single-parameter fit has one
    /// gradient on one scale, so there is nothing to put on comparable
    /// footing, and the standard engine ignores this.
    ///
    /// The parameters of a GAMLSS live on different scales, so their gradients
    /// have different magnitudes and one `nu` takes very different-sized steps
    /// on each. `Stabilization::Mad` or `L2` divides each by a measure of its
    /// own spread. Defaults to `None`, which is `gamboostLSS`'s default too.
    pub stabilization: crate::lss::Stabilization,
    /// Per-parameter learning rates for distributional fits, in
    /// `param_names()` order. **Distributional fits only**: when set, entry
    /// `p` replaces `nu` for parameter `p`'s updates, matching
    /// `gamboostLSS`'s `boost_control(nu = c(mu = ..., sigma = ...))`,
    /// where the frontend recycles a scalar and reorders a named vector to
    /// family order before the fit sees it. Every entry must satisfy
    /// `0 < nu <= 1`, and the number of entries must equal the family's
    /// parameter count. `None` (the default) recycles the scalar `nu`.
    /// The standard engine ignores this and uses `nu`.
    pub nu_per_param: Option<Vec<f64>>,
    /// How the step length is chosen once the winning base learner is
    /// selected. `Fixed` (the default) takes the classic constant step
    /// `nu·h`; `LineSearch` first finds the loss-optimal multiple of the
    /// winner's direction and shrinks that by `nu` instead.
    pub step: StepRule,
    /// Divide the response by this positive scale before a distributional fit
    /// and map the fitted model back to the original units afterwards.
    /// **Distributional fits only**, and only for families whose
    /// [`crate::lss::LssFamily::scale_map`] is `Some`: the others ignore it.
    ///
    /// Boosted LSS fitting is not unit-free. Each step moves a location
    /// parameter by about `nu` in response units once gradients are
    /// stabilized (and by much less when they are not), so the same data in
    /// dollars and in thousands of dollars give different fits at the same
    /// step count, and a response measured in large units may not leave its
    /// offset within any practical `mstop`. Fitting `y / s` and mapping each
    /// parameter's predictor back by `eta = a * eta_s + b` makes the fit
    /// equivariant: predictions, coefficients and risks are in the original
    /// units, and the path is the one R `gamboostLSS` takes on `y / s`.
    /// [`ResponseScale::Raw`] (the default) fits the raw response, R's path.
    pub response_scale: ResponseScale,
}

/// The scale a distributional fit divides its response by (see
/// [`BoostControl::response_scale`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ResponseScale {
    /// Fit the response as given (R `gamboostLSS`).
    #[default]
    Raw,
    /// The weighted SD of the response each fit sees, so a fit on a subset,
    /// a cross-validation fold or a weighted refit uses the scale of its own
    /// data (see [`crate::lss::auto_response_scale`]).
    Auto,
    /// A fixed positive scale.
    Fixed(f64),
}

impl BoostControl {
    /// Create a new `BoostControl` with the given `mstop` and `nu`,
    /// using the default risk method (`Inbag`) and `trace = false`.
    pub fn new(mstop: u32, nu: f64) -> Result<Self, MboostrsError> {
        if mstop == 0 {
            return Err(MboostrsError::Control("mstop must be > 0".to_string()));
        }
        if nu.is_nan() || nu <= 0.0 || nu > 1.0 {
            return Err(MboostrsError::Control(
                "nu must satisfy 0 < nu ≤ 1".to_string(),
            ));
        }
        Ok(Self {
            mstop,
            nu,
            risk: RiskKind::default(),
            trace: false,
            instrumentation: InstrumentationLevel::Off,
            component_fraction: None,
            stabilization: crate::lss::Stabilization::None,
            subsample_seed: 0,
            nu_per_param: None,
            step: StepRule::Fixed,
            response_scale: ResponseScale::Raw,
        })
    }

    /// Fit distributional models on `y / s` and map the model back (see
    /// [`BoostControl::response_scale`]). A fixed scale must be finite and
    /// positive.
    pub fn with_response_scale(mut self, scale: ResponseScale) -> Result<Self, MboostrsError> {
        if let ResponseScale::Fixed(s) = scale {
            if !s.is_finite() || s <= 0.0 {
                return Err(MboostrsError::Control(format!(
                    "response_scale must be finite and positive, got {s}"
                )));
            }
        }
        self.response_scale = scale;
        Ok(self)
    }

    /// Choose the step rule: the constant `nu·h` (`Fixed`), or `nu` as a
    /// fraction of the loss-optimal step along the winner's direction
    /// (`LineSearch`).
    pub fn with_step(mut self, step: StepRule) -> Self {
        self.step = step;
        self
    }

    /// Enable stochastic component subsampling at fraction `fraction` with the
    /// given `seed`. `fraction` must satisfy `0 < fraction < 1`.
    pub fn with_component_subsampling(
        mut self,
        fraction: f64,
        seed: u64,
    ) -> Result<Self, MboostrsError> {
        if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 {
            return Err(MboostrsError::Control(
                "component_fraction must satisfy 0 < fraction < 1".to_string(),
            ));
        }
        self.component_fraction = Some(fraction);
        self.subsample_seed = seed;
        Ok(self)
    }

    /// Set the risk estimation method.
    pub fn with_risk(mut self, risk: RiskKind) -> Self {
        self.risk = risk;
        self
    }

    /// Enable or disable tracing.
    pub fn with_trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        if trace && self.instrumentation == InstrumentationLevel::Off {
            self.instrumentation = InstrumentationLevel::Summary;
        }
        self
    }

    /// Select structured instrumentation independently of textual tracing.
    pub fn with_instrumentation(mut self, instrumentation: InstrumentationLevel) -> Self {
        self.instrumentation = instrumentation;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let ctrl = BoostControl::new(100, 0.1).unwrap();
        assert_eq!(ctrl.mstop, 100);
        assert!((ctrl.nu - 0.1).abs() < f64::EPSILON);
        assert_eq!(ctrl.risk, RiskKind::Inbag);
        assert!(!ctrl.trace);
    }

    #[test]
    fn new_zero_mstop() {
        assert!(BoostControl::new(0, 0.1).is_err());
    }

    #[test]
    fn new_zero_nu() {
        assert!(BoostControl::new(100, 0.0).is_err());
    }

    #[test]
    fn new_nu_above_one() {
        assert!(BoostControl::new(100, 1.5).is_err());
    }

    #[test]
    fn new_nu_exactly_one() {
        assert!(BoostControl::new(100, 1.0).is_ok());
    }

    #[test]
    fn new_nu_nan_rejected() {
        assert!(BoostControl::new(100, f64::NAN).is_err());
    }

    #[test]
    fn new_nu_inf_rejected() {
        assert!(BoostControl::new(100, f64::INFINITY).is_err());
    }

    #[test]
    fn with_risk_oobag() {
        let ctrl = BoostControl::new(50, 0.1)
            .unwrap()
            .with_risk(RiskKind::Oobag);
        assert_eq!(ctrl.risk, RiskKind::Oobag);
    }

    #[test]
    fn with_trace() {
        let ctrl = BoostControl::new(50, 0.1).unwrap().with_trace(true);
        assert!(ctrl.trace);
    }
}

/// Discovery's family-specific fixed learning rate (Rust family keys).
/// This affects screening, tuning, cost planning and the returned fit together.
pub fn discovery_learning_rate(family: &str) -> f64 {
    match family {
        "gaussian" => 0.1,
        _ => 0.02,
    }
}

/// Ordinary-mean discovery optimizer, using Rust family keys.
/// Negative-binomial gradients can leave curved means underfitted within the
/// fixed budget; use the existing loss-optimal direction search before shrinkage.
pub fn discovery_mean_step(family: &str) -> &'static str {
    if family == "nbinomial" {
        "line_search"
    } else {
        "fixed"
    }
}

#[cfg(test)]
mod discovery_rate_tests {
    #[test]
    fn negative_binomial_alone_uses_automatic_line_search() {
        assert_eq!(super::discovery_mean_step("nbinomial"), "line_search");
        for family in ["gaussian", "poisson", "binomial", "gammasreg", "other"] {
            assert_eq!(super::discovery_mean_step(family), "fixed");
        }
    }
    #[test]
    fn preserves_existing_family_rates() {
        assert_eq!(super::discovery_learning_rate("gaussian"), 0.1);
        assert_eq!(super::discovery_learning_rate("nbinomial"), 0.02);
        for family in ["poisson", "gammasreg", "binomial", "gaussreg", "other"] {
            assert_eq!(super::discovery_learning_rate(family), 0.02);
        }
    }
}
