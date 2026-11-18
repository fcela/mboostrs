//! Penalized GAM refits with REML smoothing parameters.
//!
//! A boosted model selects terms and shrinks their effects; this module
//! refits the selected terms once as a penalized regression, the construction
//! of mgcv's `gam(method = "REML")` (Wood 2011):
//!
//! * each term contributes a design block `X_b` and one or more penalties
//!   `S_bj` (a tensor term has one per margin);
//! * ordinary smooth blocks are constrained to sum to zero over the fitting
//!   rows (one Householder reflection of `c = X_b' w`), and the model has
//!   one intercept; nonconstant numeric `by` terms retain their full basis
//!   and lose only directions reproduced by the intercept or nested terms;
//! * a block whose columns reproduce part of an earlier block, as a tensor
//!   term `x1 %X% x2` reproduces the linear effect of `x1` that a `bbs(x1)`
//!   term also contains, loses the dependent columns before fitting (mgcv's
//!   `gam.side`, Wood 2017, section 5.6.3), so the coefficients are
//!   identified; the result reports them at full width with zeros there;
//! * the coefficients minimize `||z - X beta||^2_W + beta' S(lambda) beta`
//!   with `S(lambda) = sum_k lambda_k S_k`;
//! * `rho = log(lambda)` minimizes the restricted likelihood criterion by
//!   Newton's method with exact first and second derivatives
//!   (`reml`);
//! * non-Gaussian families are fitted by penalized iteratively reweighted
//!   least squares with REML re-estimated on each working model, the
//!   performance iteration (`pirls`);
//! * the Bayesian covariance `phi * A^-1`, `A = X'WX + S(lambda)`, gives
//!   pointwise intervals for each term's effect (Wahba 1983; Nychka 1988).
//!
//! The designs are supplied by the caller (the Python and R bindings build
//! them from the boosted model's learners); everything numeric after that is
//! here.

mod bootstrap;
mod concurvity;
mod design;
mod json;
mod laml;
mod lss_refit;
mod pirls;
mod reml;
mod term_test;

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use crate::MboostrsError;

pub use bootstrap::{bootstrap_linear_predictor, refit_row_bootstrap};
pub use concurvity::{
    boosted_concurvity, concurvity, model_concurvity, Concurvity, ModelConcurvity,
};
pub use design::{kron, row_tensor, TermBasis};
pub use lss_refit::{try_gamlss_refit, LssRefitParam, LssRefitResult, NewRowsTerm};
pub use pirls::breslow_cumulative_hazard;

/// One term of the refit: its raw design, its penalties and an optional
/// varying-coefficient column.
#[derive(Clone, Debug)]
pub struct RefitTerm {
    /// Label reported back with the term's results.
    pub name: String,
    /// The learner's design at the fitting rows, before the sum-to-zero
    /// constraint (`n x q`).
    pub design: Array2<f64>,
    /// The term's penalties on the raw basis (`q x q` each). An all-zero
    /// penalty (an unpenalized `bols`) is allowed and ignored.
    pub penalties: Vec<Array2<f64>>,
    /// A numeric `by=` column, multiplying the design's rows to give
    /// `f(x) * by`. A nonconstant column retains the raw basis, since its
    /// constant-in-x direction becomes the identifiable `by` effect.
    pub by: Option<Array1<f64>>,
    /// The data columns the term reads (its `by=` column included). A term
    /// is checked for columns repeated by the terms whose columns it
    /// contains, as mgcv's `gam.side` does; empty means unknown and skips
    /// checks against other terms. Numeric by terms are always checked
    /// against the global intercept.
    pub variables: Vec<String>,
}

/// The response distribution and link of the refit.
#[derive(Clone, Debug)]
pub enum RefitFamily {
    /// Normal errors, identity link, scale estimated.
    Gaussian,
    /// 0/1 response (a -1/+1 response is recoded), logit link.
    Binomial,
    /// Counts, log link, scale 1.
    Poisson,
    /// Counts with the negative binomial variance `mu + mu^2 / theta`, log
    /// link; `theta` is taken as known.
    NegativeBinomial {
        /// The dispersion parameter, taken as known.
        theta: f64,
    },
    /// Positive response, log link, scale estimated.
    Gamma,
    /// Right-censored survival: `y` is the event indicator and `time` the
    /// observed time. Fitted through the Poisson form of the Cox likelihood
    /// with a Breslow baseline hazard; effects are log hazard ratios.
    CoxPH {
        /// Observed (event or censoring) time of each row.
        time: Array1<f64>,
    },
}

impl RefitFamily {
    /// Whether the scale parameter is fixed at 1 (true) or estimated.
    pub fn known_scale(&self) -> bool {
        !matches!(self, RefitFamily::Gaussian | RefitFamily::Gamma)
    }

    /// The link the effects are reported on: `identity`, `logit` or `log`.
    pub fn link(&self) -> &'static str {
        match self {
            RefitFamily::Gaussian => "identity",
            RefitFamily::Binomial => "logit",
            _ => "log",
        }
    }
}

/// Values of [`RefitResult::stop_reason`]: `gradient` (the gradient met the
/// tolerance), `stationary` (the parameters and criterion stopped moving),
/// `flat` (no descent step, gradient below 1e-4 of the criterion's size),
/// `no_smoothing_parameters`, and the non-converged `no_descent`,
/// `iteration_limit`, `pirls_iteration_limit` and `outer_iteration_limit`.
pub const STOP_REASONS: [&str; 8] = [
    "gradient",
    "stationary",
    "flat",
    "no_smoothing_parameters",
    "no_descent",
    "iteration_limit",
    "pirls_iteration_limit",
    "outer_iteration_limit",
];

/// Relative size of the smallest change in a REML or LAML criterion that the
/// line searches treat as a real change. The computed criterion carries
/// rounding error well above machine precision: on the 400-row Gaussian
/// model of `tests/test_direct_reml_gam.py`, near its optimum with one
/// smoothing parameter close to its bound, the criterion (about 1496) moved
/// by up to 6.6e-6 (4.4e-9 relative) between trial points 1e-6 apart, while
/// the Newton step predicted a decrease of 1.4e-7. Whether the value test
/// accepted that step then depended on the CPU. The resolution is about 20
/// times the measured noise.
const CRITERION_RESOLUTION: f64 = 1e-7;

/// Changes in a criterion of size `v` smaller than this are rounding.
pub(crate) fn criterion_resolution(v: f64) -> f64 {
    CRITERION_RESOLUTION * (1.0 + v.abs())
}

/// Outcome of the slope test a line search applies when a trial point's
/// criterion value is within [`criterion_resolution`] of the current one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlopeTest {
    /// The slope along the step has fallen to near zero: accept the point.
    Accept,
    /// The slope has turned strongly positive: the step went past the
    /// minimum along the line, so try a shorter one.
    Overshoot,
    /// The slope is still nearly the starting slope: the step is too short
    /// to make progress, and shorter ones would make less.
    TooShort,
}

/// The approximate Wolfe conditions of Hager and Zhang (2005). When two
/// criterion values cannot be ordered because they differ by less than the
/// rounding in them, the directional derivative can still be compared: a
/// step along `s` from slope `slope0 = g0's < 0` is accepted when the slope
/// at the new point satisfies `SIGMA slope0 <= g1's <= (2 DELTA - 1) slope0`,
/// that is, it has risen from the starting slope by at least a tenth and is
/// not more positive than 0.8 of the starting slope's magnitude. The value
/// test cannot decide this near an optimum; the slope test can, because the
/// gradient is computed with a relative error far smaller than the change
/// in the criterion it measures.
pub(crate) fn approximate_wolfe(slope0: f64, slope1: f64) -> SlopeTest {
    const DELTA: f64 = 0.1;
    const SIGMA: f64 = 0.9;
    if slope1 < SIGMA * slope0 {
        SlopeTest::TooShort
    } else if slope1 > (2.0 * DELTA - 1.0) * slope0 {
        SlopeTest::Overshoot
    } else {
        SlopeTest::Accept
    }
}

/// Iteration limits and tolerances.
#[derive(Clone, Debug)]
pub struct RefitControl {
    /// Newton iterations per REML optimization.
    pub max_newton: usize,
    /// Convergence: largest free gradient component below
    /// `newton_tol * (1 + |V|)`.
    pub newton_tol: f64,
    /// Penalized IRLS iterations for non-Gaussian families.
    pub max_pirls: usize,
    /// PIRLS convergence: the linear predictor moves less than
    /// `pirls_tol * (1 + max |eta|)`.
    pub pirls_tol: f64,
    /// Bound on each internal log smoothing parameter (penalties are
    /// rescaled to the scale of their block's `X'WX`, so `+-rho_bound`
    /// means effectively unpenalized / fully smoothed).
    pub rho_bound: f64,
    /// For the known-scale likelihoods (Binomial, Poisson, negative
    /// binomial, Cox): after the performance iteration, optimize the
    /// Laplace-approximate REML directly (mgcv's default). Off, the
    /// performance iteration's fixed point is returned.
    pub outer_iteration: bool,
    /// Negative binomial with the outer iteration: estimate `theta` by the
    /// same criterion, starting from the supplied value. Off, the supplied
    /// value is used as known.
    pub estimate_theta: bool,
    /// Factor each REML evaluation from the penalized Gram matrix
    /// `X'WX + S` by Cholesky, and compute the penalized residual sum of
    /// squares from the Gram (`z'Wz - beta' X'Wz`), instead of from a QR
    /// factor of the n-row design. Costs one `X'WX` (O(n p^2)) and then
    /// O(p^3) per evaluation, independent of n. The QR form (off) keeps the
    /// conditioning of `X` rather than `X'X`, which is the default for the
    /// boosted refit; the Gram form is the one `mb.gam` uses.
    pub gram: bool,
}

impl Default for RefitControl {
    fn default() -> Self {
        Self {
            max_newton: 200,
            newton_tol: 1e-7,
            max_pirls: 100,
            pirls_tol: 1e-7,
            rho_bound: 20.0,
            outer_iteration: true,
            estimate_theta: true,
            gram: false,
        }
    }
}

/// One term's part of the result.
#[derive(Clone, Debug)]
pub struct RefitBlock {
    /// The term's label, as supplied in [`RefitTerm::name`].
    pub name: String,
    /// Columns of the full coefficient vector (after the intercept) that
    /// belong to this term: `start..end`.
    pub start: usize,
    /// End (exclusive) of the term's columns.
    pub end: usize,
    /// Householder vector of the sum-to-zero constraint; empty when the
    /// block needed no constraint.
    pub householder: Array1<f64>,
    /// Whether the design was multiplied by a `by=` column.
    pub has_by: bool,
    /// Rank of the summed penalty (the term's penalized dimension).
    pub rank: usize,
    /// Effective degrees of freedom of the term.
    pub edf: f64,
    /// Smoothing parameter of each penalty, on the caller's penalty scale;
    /// empty for an unpenalized term.
    pub lambda: Vec<f64>,
    /// The term's test of `f = 0` (Wood 2013), mgcv's `summary.gam` row:
    /// `ref_df`, the statistic and the p-value. `None` for refits that do
    /// not compute it (the distributional refit).
    pub test: Option<TermTest>,
    /// mgcv's concurvity measures of the term against the rest of the
    /// model (see [`concurvity`]); `None` where not computed.
    pub concurvity: Option<Concurvity>,
}

pub use term_test::{term_test, TermTest};

/// The fitted refit.
#[derive(Clone, Debug)]
pub struct RefitResult {
    /// One entry per term, in the order supplied.
    pub blocks: Vec<RefitBlock>,
    /// Coefficients: the intercept, then each block's constrained
    /// coefficients.
    pub coef: Array1<f64>,
    /// Bayesian covariance `phi * A^-1` of `coef`.
    pub cov: Array2<f64>,
    /// The Bayesian covariance corrected, to first order, for the
    /// uncertainty of the estimated smoothing parameters (mgcv's
    /// `vcov(unconditional = TRUE)` without its second-order term).
    pub cov_corrected: Array2<f64>,
    /// Scale parameter (residual variance for Gaussian, 1 for the
    /// known-scale families).
    pub scale: f64,
    /// Criterion at the optimum: -2 x restricted log-likelihood up to a
    /// constant (the scale profiled out when it is estimated).
    pub reml_score: f64,
    /// Internal log smoothing parameters, one per penalty.
    pub rho: Array1<f64>,
    /// Hessian of the criterion in `rho` at the optimum.
    pub rho_hessian: Array2<f64>,
    /// Largest absolute gradient component over the free parameters.
    pub gradient_norm: f64,
    /// Total effective degrees of freedom, including the intercept.
    pub edf_total: f64,
    /// Log-likelihood of the fitted model, as mgcv's `logLik.gam` computes
    /// it (Gaussian with the maximum-likelihood variance; Cox partial
    /// likelihood).
    pub loglik: f64,
    /// Degrees of freedom for AIC: `edf_total` plus one for an estimated
    /// scale, theta or gamma shape.
    pub df: f64,
    /// Newton iterations of the last REML optimization.
    pub newton_iterations: usize,
    /// Penalized IRLS iterations (1 for Gaussian).
    pub pirls_iterations: usize,
    /// Whether both the REML optimization and PIRLS converged.
    pub converged: bool,
    /// Why the smoothing-parameter iteration stopped, one of
    /// [`STOP_REASONS`]. `"no_descent"` with `converged = false` means no
    /// step along the Newton direction lowered the criterion although its
    /// gradient was not small: the criterion is not computed accurately
    /// enough there, which happens when terms are nearly the same function of
    /// the data (concurvity) and several are left almost unpenalized.
    pub stop_reason: &'static str,
    /// Penalty-free directions held at zero because the data cannot identify
    /// them (see `Model::fixed_penalty`): two terms whose unpenalized parts
    /// span the same functions on the fitting rows.
    pub unidentified_directions: usize,
    /// The terms whose unpenalized directions take part in those
    /// dependencies, named as the refit received them, in term order. Empty
    /// when `unidentified_directions` is zero.
    pub unidentified_terms: Vec<String>,
    /// `"performance"` (REML on the working model; Gaussian, or
    /// `outer_iteration = false`) or `"outer"` (Laplace-approximate REML).
    pub method: &'static str,
    /// The negative binomial `theta` the refit used: estimated with the
    /// outer iteration and `estimate_theta`, otherwise the supplied value.
    pub theta: Option<f64>,
    /// `identity`, `logit` or `log`.
    pub link: &'static str,
    /// Number of observations.
    pub n: usize,
    /// Residual degrees of freedom `n - edf_total` when the scale was
    /// estimated (the term tests are then F tests); `None` for a known
    /// scale (chi-square tests).
    pub residual_df: Option<f64>,
}

/// Refit `terms` as one penalized GAM with REML smoothing parameters.
///
/// `y` is the response (the event indicator for `CoxPH`), `weights` the
/// prior weights, `offset` a per-row offset added to the linear predictor
/// (log exposure), or `None`.
pub fn try_gam_refit(
    terms: Vec<RefitTerm>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    offset: Option<ArrayView1<f64>>,
    family: &RefitFamily,
    control: &RefitControl,
) -> Result<RefitResult, MboostrsError> {
    let n = y.len();
    if terms.is_empty() {
        return Err(MboostrsError::Control("no terms to refit".into()));
    }
    let w = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(MboostrsError::Shape(format!(
                    "weights have {} entries for {n} observations",
                    w.len()
                )));
            }
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(MboostrsError::Control(
                    "weights must be finite and non-negative".into(),
                ));
            }
            w.to_owned()
        }
        None => Array1::ones(n),
    };
    let offset = match offset {
        Some(o) => {
            if o.len() != n {
                return Err(MboostrsError::Shape(format!(
                    "offset has {} entries for {n} observations",
                    o.len()
                )));
            }
            o.to_owned()
        }
        None => Array1::zeros(n),
    };
    if let RefitFamily::CoxPH { time } = family {
        if time.len() != n {
            return Err(MboostrsError::Shape(format!(
                "survival times have {} entries for {n} observations",
                time.len()
            )));
        }
    }
    if let RefitFamily::NegativeBinomial { theta } = family {
        if !(theta.is_finite() && *theta > 0.0) {
            return Err(MboostrsError::Control(format!(
                "negative binomial theta must be positive and finite, got {theta}"
            )));
        }
    }
    let model = build_model(terms, w.view(), n)?;
    pirls::fit(&model, y, w.view(), offset.view(), family, control)
}

/// The assembled model matrix and penalties.
pub(crate) struct Model {
    /// `n x p`: intercept column, then the constrained blocks, each without
    /// the columns its side constraint removed.
    pub x: Array2<f64>,
    /// Width of the coefficient vector the result reports: the intercept and
    /// every block's constrained columns, removed ones included.
    pub full_p: usize,
    pub blocks: Vec<BlockLayout>,
    /// One entry per penalized (block, penalty) pair.
    pub penalties: Vec<Penalty>,
    /// Null space dimension of the total penalty (includes the intercept),
    /// less the directions `fixed_penalty` removes.
    pub null_dim: usize,
    /// How many directions `fixed_penalty` holds at zero.
    pub unidentified: usize,
    /// The blocks (by name) whose directions the dependency uses.
    pub unidentified_terms: Vec<String>,
    /// A fixed penalty (`p x p`) on penalty-free directions the data cannot
    /// identify: two terms whose unpenalized parts span the same functions,
    /// such as smooths of two columns with identical values. `A` is singular
    /// along such a direction at every smoothing parameter. The penalty fixes
    /// the coefficient there at zero, which is what mgcv's rank detection
    /// does by dropping it; it does not depend on `rho`.
    pub fixed_penalty: Option<Array2<f64>>,
}

pub(crate) struct BlockLayout {
    pub name: String,
    /// Columns of the block in `Model::x`.
    pub start: usize,
    pub end: usize,
    /// Columns of the block in the reported (full-width) coefficients.
    pub full_start: usize,
    pub full_end: usize,
    /// Positions, within the block's constrained columns, of the columns
    /// kept by the side constraint; `None` when all are kept.
    pub kept: Option<Vec<usize>>,
    pub householder: Array1<f64>,
    pub has_by: bool,
    pub rank: usize,
    /// Indices into `Model::penalties`.
    pub penalty_ids: Vec<usize>,
    /// For a single-penalty block: log pseudo-determinant of its scaled
    /// penalty (so `log|lambda S|+ = rank * rho + this`).
    pub logdet_const: f64,
    /// For a two-penalty block (a tensor term): the simultaneous
    /// diagonalization that makes `log|S|+` a sum of logs (see
    /// [`PenaltyPair`]).
    pub pair: Option<PenaltyPair>,
}

/// Two penalties `P1`, `P2` projected on their joint range space, written as
/// `P1 = T^1/2 A T^1/2`, `P2 = T^1/2 (I - A) T^1/2` with `T = P1 + P2` and
/// `A = Q diag(mu) Q'`. Then
/// `log|l1 P1 + l2 P2| = log|T| + sum_i log(l1 mu_i + l2 (1 - mu_i))`,
/// exact, O(rank) per evaluation, and free of cancellation however far apart
/// `l1` and `l2` are.
pub(crate) struct PenaltyPair {
    pub logdet_t: f64,
    pub mu: Array1<f64>,
}

pub(crate) struct Penalty {
    pub block: usize,
    /// The constrained penalty, rescaled by `scale` (`q' x q'`).
    pub matrix: Array2<f64>,
    /// Factor applied to the caller's penalty: `lambda_caller = scale *
    /// exp(rho)`.
    pub scale: f64,
    /// The penalty projected on the block's range basis (multi-penalty
    /// blocks only).
    pub projected: Option<Array2<f64>>,
}

pub(crate) fn build_model(
    terms: Vec<RefitTerm>,
    w: ArrayView1<f64>,
    n: usize,
) -> Result<Model, MboostrsError> {
    // Pass 1: validate each term, constrain it to sum to zero and apply its
    // `by=` column.
    let mut prepared = Vec::with_capacity(terms.len());
    for term in terms {
        prepared.push(constrain_term(term, w, n)?);
    }
    // Pass 2: side constraints, lower-order terms first.
    let kept = side_constraints(&prepared, w, n)?;
    // Pass 3: penalties and layout, in the caller's order.
    let mut designs = Vec::with_capacity(prepared.len());
    let mut blocks = Vec::with_capacity(prepared.len());
    let mut penalties = Vec::new();
    let mut start = 1usize;
    let mut full_start = 1usize;
    let mut null_bases = Vec::with_capacity(prepared.len());
    for (b, (term, kept)) in prepared.into_iter().zip(kept).enumerate() {
        let full_width = term.xc.ncols();
        let (xc, constrained) = match &kept {
            Some(cols) => (
                term.xc.select(ndarray::Axis(1), cols),
                term.penalties
                    .iter()
                    .map(|p| {
                        p.select(ndarray::Axis(0), cols)
                            .select(ndarray::Axis(1), cols)
                    })
                    .collect::<Vec<_>>(),
            ),
            None => (term.xc, term.penalties),
        };
        let qc = xc.ncols();
        // Scale each penalty to the size of the block's X'WX (ratio of
        // traces, O(n q)), so rho = 0 is a balanced start whatever units the
        // caller's penalty has.
        let xtwx_trace: f64 = xc
            .rows()
            .into_iter()
            .zip(w.iter())
            .map(|(row, &wi)| wi * row.dot(&row))
            .sum();
        let mut ids = Vec::new();
        let mut scaled = Vec::new();
        for pc in constrained {
            if frobenius(&pc) == 0.0 {
                continue;
            }
            let pen_trace = pc.diag().sum();
            let scale = if xtwx_trace > 0.0 && pen_trace > 0.0 {
                xtwx_trace / pen_trace
            } else {
                1.0
            };
            scaled.push((pc.mapv(|x| x * scale), scale));
        }
        let rank;
        let mut logdet_const = 0.0;
        let mut range_basis = None;
        // Columns: the block's penalty-free directions in its coefficients.
        let null_basis: Array2<f64>;
        if scaled.is_empty() {
            rank = 0;
            null_basis = Array2::eye(qc);
        } else {
            // The penalized dimension is the rank of the (normalized) sum,
            // which does not depend on the positive smoothing parameters.
            let mut sum = Array2::<f64>::zeros((qc, qc));
            for (m, _) in &scaled {
                sum.scaled_add(1.0 / frobenius(m), m);
            }
            let (vals, vecs) = reml::symmetric_eigen(&sum);
            let top = vals.iter().cloned().fold(0.0f64, f64::max);
            let tol = top * (qc as f64) * f64::EPSILON * 10.0;
            let keep: Vec<usize> = (0..qc).filter(|&i| vals[i] > tol).collect();
            rank = keep.len();
            let free: Vec<usize> = (0..qc).filter(|&i| vals[i] <= tol).collect();
            let mut basis = Array2::<f64>::zeros((qc, free.len()));
            for (j, &i) in free.iter().enumerate() {
                basis.column_mut(j).assign(&vecs.column(i));
            }
            null_basis = basis;
            if scaled.len() == 1 {
                // The sum is the penalty divided by its norm.
                let norm = frobenius(&scaled[0].0);
                logdet_const = keep
                    .iter()
                    .map(|&i| (vals[i] * norm).max(1e-300).ln())
                    .sum();
            } else {
                let mut basis = Array2::<f64>::zeros((qc, rank));
                for (j, &i) in keep.iter().enumerate() {
                    basis.column_mut(j).assign(&vecs.column(i));
                }
                range_basis = Some(basis);
            }
        }
        let projections: Vec<Option<Array2<f64>>> = scaled
            .iter()
            .map(|(m, _)| {
                range_basis.as_ref().map(|u: &Array2<f64>| {
                    reml::matmul(reml::matmul(u.t(), m.view()).view(), u.view())
                })
            })
            .collect();
        let pair = match projections.as_slice() {
            [Some(p1), Some(p2)] => Some(penalty_pair(p1, p2)?),
            _ => None,
        };
        for ((m, scale), projected) in scaled.into_iter().zip(projections) {
            ids.push(penalties.len());
            penalties.push(Penalty {
                block: b,
                matrix: m,
                scale,
                projected,
            });
        }
        blocks.push(BlockLayout {
            name: term.name,
            start,
            end: start + qc,
            full_start,
            full_end: full_start + full_width,
            kept,
            householder: term.householder,
            has_by: term.has_by,
            rank,
            penalty_ids: ids,
            logdet_const,
            pair,
        });
        start += qc;
        full_start += full_width;
        designs.push(xc);
        null_bases.push(null_basis);
    }
    let p = start;
    let mut x = Array2::<f64>::zeros((n, p));
    x.column_mut(0).fill(1.0);
    for (block, design) in blocks.iter().zip(designs) {
        x.slice_mut(s![.., block.start..block.end]).assign(&design);
    }
    let null_dim = p - blocks.iter().map(|b| b.rank).sum::<usize>();
    let (fixed_penalty, dropped, dropped_terms) =
        unidentified_directions(&x, w, &blocks, &null_bases);
    Ok(Model {
        x,
        full_p: full_start,
        blocks,
        penalties,
        null_dim: null_dim - dropped,
        fixed_penalty,
        unidentified: dropped,
        unidentified_terms: dropped_terms,
    })
}

/// Penalty-free directions of the whole model that the data cannot separate,
/// as a fixed penalty `kappa * sum d d'` and their count.
///
/// The penalty-free directions are the intercept and each block's penalty
/// null space (a `bbs` term's linear part, every column of an unpenalized
/// `bols`). Their columns of the design, `Z`, are scaled to unit weighted
/// norm; an eigenvalue of `Z'WZ` below `1e-10` of the largest marks a
/// combination of them that is zero on every fitting row up to rounding. Its
/// coefficient direction is penalized by `kappa`, the mean diagonal of
/// `X'WX`, so the system stays as well scaled as before while the direction,
/// which carries no information in the data, is held at zero.
///
/// Also returns the names of the blocks whose directions take part in a
/// dependency (the intercept is in no block), in block order.
fn unidentified_directions(
    x: &Array2<f64>,
    w: ArrayView1<f64>,
    blocks: &[BlockLayout],
    null_bases: &[Array2<f64>],
) -> (Option<Array2<f64>>, usize, Vec<String>) {
    let p = x.ncols();
    // Coefficient directions (p-vectors), intercept first, and the block
    // each one belongs to (`None` for the intercept).
    let mut directions: Vec<Array1<f64>> = Vec::new();
    let mut owners: Vec<Option<usize>> = Vec::new();
    let mut intercept = Array1::<f64>::zeros(p);
    intercept[0] = 1.0;
    directions.push(intercept);
    owners.push(None);
    for (b, (block, basis)) in blocks.iter().zip(null_bases).enumerate() {
        for j in 0..basis.ncols() {
            let mut d = Array1::<f64>::zeros(p);
            d.slice_mut(s![block.start..block.end])
                .assign(&basis.column(j));
            directions.push(d);
            owners.push(Some(b));
        }
    }
    let m = directions.len();
    if m < 2 {
        return (None, 0, Vec::new());
    }
    // Z'WZ for the columns Z = X D equals D'(X'WX)D: one pass over the rows
    // for X'WX, then p x m products, instead of m passes over the n x p design.
    let mut dmat = Array2::<f64>::zeros((p, m));
    for (j, d) in directions.iter().enumerate() {
        dmat.column_mut(j).assign(d);
    }
    let xtwx = reml::weighted_gram(x.view(), w);
    let gram = dmat.t().dot(&xtwx).dot(&dmat);
    let scale: Vec<f64> = (0..m)
        .map(|j| {
            let v = gram[[j, j]];
            if v > 0.0 {
                1.0 / v.sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let normalized = Array2::from_shape_fn((m, m), |(a, b)| gram[[a, b]] * scale[a] * scale[b]);
    let (vals, vecs) = reml::symmetric_eigen(&normalized);
    let top = vals.iter().cloned().fold(0.0f64, f64::max);
    if top <= 0.0 {
        return (None, 0, Vec::new());
    }
    let mut dependent: Vec<Array1<f64>> = Vec::new();
    // Blocks whose directions take part in a dependency, by block index.
    let mut involved: Vec<usize> = Vec::new();
    for i in 0..m {
        if vals[i] > 1e-10 * top {
            continue;
        }
        // Each direction's column of Z has unit weighted norm after scaling,
        // so its share of the dependency is the eigenvector entry.
        let share: Vec<f64> = (0..m).map(|j| vecs[[j, i]].abs()).collect();
        let peak = share.iter().cloned().fold(0.0f64, f64::max);
        for j in 0..m {
            if share[j] > 1e-6 * peak {
                if let Some(b) = owners[j] {
                    involved.push(b);
                }
            }
        }
        let mut d = Array1::<f64>::zeros(p);
        for j in 0..m {
            d.scaled_add(vecs[[j, i]] * scale[j], &directions[j]);
        }
        let norm = d.dot(&d).sqrt();
        if norm > 0.0 {
            dependent.push(d / norm);
        }
    }
    // A column with zero weighted norm on its own (a constant `by=` column
    // of zeros, say) is dependent as well.
    for j in 0..m {
        if scale[j] == 0.0 {
            dependent.push(directions[j].clone());
            if let Some(b) = owners[j] {
                involved.push(b);
            }
        }
    }
    if dependent.is_empty() {
        return (None, 0, Vec::new());
    }
    involved.sort_unstable();
    involved.dedup();
    let terms: Vec<String> = involved.iter().map(|&b| blocks[b].name.clone()).collect();
    let xtwx_diag: f64 = x
        .rows()
        .into_iter()
        .zip(w.iter())
        .map(|(row, &wi)| wi * row.dot(&row))
        .sum::<f64>()
        / p as f64;
    let kappa = if xtwx_diag > 0.0 { xtwx_diag } else { 1.0 };
    let mut fixed = Array2::<f64>::zeros((p, p));
    for d in &dependent {
        for a in 0..p {
            if d[a] == 0.0 {
                continue;
            }
            for b in 0..p {
                fixed[[a, b]] += kappa * d[a] * d[b];
            }
        }
    }
    (Some(fixed), dependent.len(), terms)
}

/// A term after its identifiability constraint and `by=` column.
struct ConstrainedTerm {
    name: String,
    variables: Vec<String>,
    /// The constrained design (`n x qc`).
    xc: Array2<f64>,
    /// The constrained penalties, all-zero ones dropped.
    penalties: Vec<Array2<f64>>,
    householder: Array1<f64>,
    has_by: bool,
}

fn constrain_term(
    term: RefitTerm,
    w: ArrayView1<f64>,
    n: usize,
) -> Result<ConstrainedTerm, MboostrsError> {
    let (nr, q) = term.design.dim();
    if nr != n {
        return Err(MboostrsError::Shape(format!(
            "term {} has {nr} design rows for {n} observations",
            term.name
        )));
    }
    if q == 0 {
        return Err(MboostrsError::Shape(format!(
            "term {} has no columns",
            term.name
        )));
    }
    if term.design.iter().any(|v| !v.is_finite()) {
        return Err(MboostrsError::Control(format!(
            "term {} has a non-finite design entry",
            term.name
        )));
    }
    for pen in &term.penalties {
        if pen.dim() != (q, q) {
            return Err(MboostrsError::Shape(format!(
                "term {} has a {:?} penalty for {q} design columns",
                term.name,
                pen.dim()
            )));
        }
    }
    if let Some(by) = &term.by {
        if by.len() != n {
            return Err(MboostrsError::Shape(format!(
                "term {} has a by column of length {} for {n} observations",
                term.name,
                by.len()
            )));
        }
    }
    let c = term.design.t().dot(&w);
    // A one-column term (a slope without an intercept, `bols(x,
    // intercept=False)`, or its product with a numeric `by`) is not
    // confounded with the intercept unless the column is constant, so it
    // takes no sum-to-zero constraint; applying one would remove its only
    // column. A constant column is refused below as before.
    let column_varies = q == 1 && {
        let col = term.design.column(0);
        let first = col[0];
        col.iter()
            .any(|&x| (x - first).abs() > 1e-12 * first.abs().max(1.0))
    };
    // With a nonconstant numeric by, the raw constant direction becomes
    // the by variable, not a duplicate intercept. Retain it (mgcv's numeric
    // by convention); side_constraints checks the multiplied design for
    // actual overlap with the intercept and nested terms instead.
    let by_varies = term.by.as_ref().is_some_and(|by| {
        let first = by[0];
        by.iter()
            .any(|&x| (x - first).abs() > 1e-12 * first.abs().max(1.0))
    });
    let v = if column_varies || by_varies {
        Array1::zeros(0)
    } else {
        householder_vector(c.view())
    };
    let mut xc = apply_constraint(term.design.view(), v.view());
    if let Some(by) = &term.by {
        for (mut row, &f) in xc.rows_mut().into_iter().zip(by.iter()) {
            row *= f;
        }
    }
    if xc.ncols() == 0 {
        return Err(MboostrsError::Shape(format!(
            "term {} has one column, which the sum-to-zero constraint removes",
            term.name
        )));
    }
    let penalties = term
        .penalties
        .iter()
        .filter(|pen| pen.iter().any(|v| *v != 0.0))
        .map(|pen| constrain_penalty(pen, v.view()))
        .collect();
    Ok(ConstrainedTerm {
        name: term.name,
        variables: term.variables,
        xc,
        penalties,
        householder: v,
        has_by: term.by.is_some(),
    })
}

/// Squared residual norm, relative to the column's own, below which a
/// column counts as reproduced by the earlier terms (a norm ratio of 1e-4).
/// Exact reproduction, such as a spline basis reproducing a straight line,
/// leaves about 1e-12; two different terms of real data leave far more.
const SIDE_TOLERANCE: f64 = 1e-8;

/// For each term, the columns to keep so that no term reproduces what the
/// intercept and the terms nested in it already span (`None`: keep all).
///
/// A term is checked against the terms whose columns are a subset of its
/// own (mgcv's `gam.side`, Wood 2017, section 5.6.3): `x1 %X% x2` against
/// `bbs(x1)` and `bbs(x2)`, a second smooth of `x1` against the first. Terms
/// are placed in order of their number of columns read, penalties and basis
/// columns, so columns are removed from the higher-order term. Terms over
/// unrelated columns are not compared, so a coincidence of the data (two
/// spline columns each supported by the same single row) removes nothing.
/// Within a term, a pivoted Cholesky factorization of its residual
/// cross-product after projection on the nested terms' columns finds the
/// directions they already span; see [`SIDE_TOLERANCE`].
fn side_constraints(
    terms: &[ConstrainedTerm],
    w: ArrayView1<f64>,
    n: usize,
) -> Result<Vec<Option<Vec<usize>>>, MboostrsError> {
    let mut order: Vec<usize> = (0..terms.len()).collect();
    order.sort_by_key(|&b| {
        (
            terms[b].variables.len(),
            terms[b].penalties.len(),
            terms[b].xc.ncols(),
            b,
        )
    });
    // The columns each placed term adds to the span (full column rank).
    let mut spans: Vec<Option<Array2<f64>>> = vec![None; terms.len()];
    let mut kept = vec![None; terms.len()];
    for b in order {
        let x = &terms[b].xc;
        let q = x.ncols();
        let vars = &terms[b].variables;
        let nested: Vec<&Array2<f64>> = if vars.is_empty() {
            Vec::new()
        } else {
            terms
                .iter()
                .zip(&spans)
                .filter(|(t, span)| {
                    span.is_some()
                        && !t.variables.is_empty()
                        && t.variables.iter().all(|v| vars.contains(v))
                })
                .map(|(_, span)| span.as_ref().expect("filtered"))
                .collect()
        };
        if nested.is_empty() && !terms[b].has_by {
            spans[b] = Some(x.clone());
            continue;
        }
        let mut blocks = vec![Array2::<f64>::ones((n, 1))];
        blocks.extend(nested.into_iter().cloned());
        let views: Vec<_> = blocks.iter().map(|m| m.view()).collect();
        let placed = ndarray::concatenate(ndarray::Axis(1), &views).expect("same number of rows");
        // Residual cross-product R = X'WX - C' G^-1 C, with G = A'WA and
        // C = A'WX for the placed columns A.
        let g = reml::weighted_gram(placed.view(), w);
        let c = reml::weighted_cross(placed.view(), x.view(), w);
        let h = reml::weighted_gram(x.view(), w);
        let pa = g.nrows();
        let ridge = 1e-12 * (g.diag().sum() / pa as f64).max(1e-300);
        let mut gr = g.clone();
        gr.diag_mut().mapv_inplace(|v| v + ridge);
        let solved = solve_spd(&gr, &c).ok_or_else(|| {
            MboostrsError::Solver(format!(
                "cannot check term {} against the terms before it",
                terms[b].name
            ))
        })?;
        let mut r = &h - &reml::matmul(c.t(), solved.view());
        let norms: Vec<f64> = (0..q).map(|j| h[[j, j]].max(0.0)).collect();
        // Columns that add to the span of the placed columns, largest
        // residual first.
        let mut remaining: Vec<usize> = (0..q).collect();
        let chosen = pivoted_cholesky(&mut r, &norms, &mut remaining);
        if chosen.is_empty() {
            return Err(MboostrsError::LearnerSetup(
                terms[b].name.clone(),
                "the term is reproduced entirely by the other terms of the refit \
                 (a duplicate term?)"
                    .to_string(),
            ));
        }
        // The term's own span can exceed `chosen` only through directions
        // the placed columns already hold: those are the overlap, and one
        // column per direction is removed. Columns dependent within the term
        // itself, such as a spline column with no data in its support, stay:
        // the penalty identifies them, as in mgcv. Continuing the
        // factorization of X'WX (without the placed columns) past `chosen`
        // finds the columns that carry the overlap.
        let mut own = h.clone();
        for &k in &chosen {
            let d = own[[k, k]];
            if d <= 0.0 {
                continue;
            }
            let l: Vec<f64> = (0..q).map(|j| own[[j, k]] / d.sqrt()).collect();
            for i in 0..q {
                for j in 0..q {
                    own[[i, j]] -= l[i] * l[j];
                }
            }
        }
        let overlap = pivoted_cholesky(&mut own, &norms, &mut remaining);
        let reduced = if overlap.is_empty() {
            x.select(ndarray::Axis(1), &{
                let mut c = chosen.clone();
                c.sort_unstable();
                c
            })
        } else {
            let mut keep: Vec<usize> = (0..q).filter(|j| !overlap.contains(j)).collect();
            keep.sort_unstable();
            kept[b] = Some(keep);
            let mut c = chosen.clone();
            c.sort_unstable();
            x.select(ndarray::Axis(1), &c)
        };
        // The span this term adds is its chosen columns alone, which keeps
        // every placed design of full column rank.
        spans[b] = Some(reduced);
    }
    Ok(kept)
}

/// Greedy pivoted Cholesky factorization of the residual cross-product `r`
/// over the columns in `remaining`: repeatedly take the column whose
/// residual, relative to its own squared norm in `norms`, is largest, and
/// stop when every remaining ratio is below [`SIDE_TOLERANCE`]. Returns the
/// columns taken, in order; `remaining` keeps the rest and `r` is updated.
fn pivoted_cholesky(r: &mut Array2<f64>, norms: &[f64], remaining: &mut Vec<usize>) -> Vec<usize> {
    let q = r.nrows();
    let mut chosen = Vec::new();
    while !remaining.is_empty() {
        let (pos, ratio) = remaining
            .iter()
            .enumerate()
            .map(|(pos, &j)| {
                let ratio = if norms[j] > 0.0 {
                    r[[j, j]] / norms[j]
                } else {
                    0.0
                };
                (pos, ratio)
            })
            .fold((0, f64::NEG_INFINITY), |best, cur| {
                if cur.1 > best.1 {
                    cur
                } else {
                    best
                }
            });
        if ratio < SIDE_TOLERANCE {
            break;
        }
        let k = remaining.swap_remove(pos);
        chosen.push(k);
        let d = r[[k, k]].sqrt();
        let l: Vec<f64> = (0..q).map(|j| r[[j, k]] / d).collect();
        for &i in remaining.iter() {
            for &j in remaining.iter() {
                r[[i, j]] -= l[i] * l[j];
            }
        }
    }
    chosen
}

/// `G^-1 C` for a symmetric positive definite `G`, by Cholesky.
fn solve_spd(g: &Array2<f64>, c: &Array2<f64>) -> Option<Array2<f64>> {
    use faer::linalg::solvers::Cholesky;
    use faer::prelude::SpSolver;
    let p = g.nrows();
    let fg = faer::Mat::from_fn(p, p, |i, j| g[[i, j]]);
    let chol = Cholesky::try_new(fg.as_ref(), faer::Side::Lower).ok()?;
    let fc = faer::Mat::from_fn(c.nrows(), c.ncols(), |i, j| c[[i, j]]);
    let sol = chol.solve(fc.as_ref());
    let out = Array2::from_shape_fn((c.nrows(), c.ncols()), |(i, j)| sol.read(i, j));
    out.iter().all(|v| v.is_finite()).then_some(out)
}

impl Model {
    /// Position in the full-width coefficient vector of each column of `x`.
    pub(crate) fn full_positions(&self) -> Vec<usize> {
        let mut out = vec![0usize];
        for block in &self.blocks {
            match &block.kept {
                Some(cols) => out.extend(cols.iter().map(|&c| block.full_start + c)),
                None => out.extend(block.full_start..block.full_end),
            }
        }
        out
    }
}

/// `v` placed at `positions` of a zero vector of length `len`.
pub(crate) fn expand_vector(v: &Array1<f64>, positions: &[usize], len: usize) -> Array1<f64> {
    let mut out = Array1::zeros(len);
    for (&p, &x) in positions.iter().zip(v.iter()) {
        out[p] = x;
    }
    out
}

/// `m` placed at `positions` (rows and columns) of a zero `len x len` matrix.
pub(crate) fn expand_matrix(m: &Array2<f64>, positions: &[usize], len: usize) -> Array2<f64> {
    let mut out = Array2::zeros((len, len));
    for (i, &pi) in positions.iter().enumerate() {
        for (j, &pj) in positions.iter().enumerate() {
            out[[pi, pj]] = m[[i, j]];
        }
    }
    out
}

fn penalty_pair(p1: &Array2<f64>, p2: &Array2<f64>) -> Result<PenaltyPair, MboostrsError> {
    let t = p1 + p2;
    let (d, v) = reml::symmetric_eigen(&t);
    if d.iter().any(|x| x.is_nan() || *x <= 0.0) {
        return Err(MboostrsError::Solver(
            "a tensor term's penalties are not positive definite on their range".into(),
        ));
    }
    // T^-1/2 = V diag(d^-1/2) V'
    let scaled = &v / &d.mapv(f64::sqrt).view().insert_axis(ndarray::Axis(0));
    let t_inv_half = reml::matmul(scaled.view(), v.t());
    let a = reml::matmul(
        reml::matmul(t_inv_half.view(), p1.view()).view(),
        t_inv_half.view(),
    );
    let (mu, _) = reml::symmetric_eigen(&a);
    Ok(PenaltyPair {
        logdet_t: d.iter().map(|x| x.ln()).sum(),
        mu: mu.mapv(|x| x.clamp(0.0, 1.0)),
    })
}

/// Householder vector `v` with `(I - 2 v v' / v'v) c = alpha e_1`; empty when
/// `c` is zero (no constraint needed).
fn householder_vector(c: ArrayView1<f64>) -> Array1<f64> {
    let norm = c.dot(&c).sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Array1::zeros(0);
    }
    let mut v = c.to_owned();
    let alpha = if c[0] >= 0.0 { -norm } else { norm };
    v[0] -= alpha;
    v
}

/// `X H` without its first column: the design on the null space of `c'`.
pub(crate) fn apply_constraint(x: ArrayView2<f64>, v: ArrayView1<f64>) -> Array2<f64> {
    if v.is_empty() {
        return x.to_owned();
    }
    let vv = v.dot(&v);
    let xv = x.dot(&v);
    let q = x.ncols();
    let mut out = Array2::<f64>::zeros((x.nrows(), q - 1));
    for j in 1..q {
        let f = 2.0 * v[j] / vv;
        let mut col = out.column_mut(j - 1);
        col.assign(&x.column(j));
        col.scaled_add(-f, &xv);
    }
    out
}

/// `H S H` without its first row and column.
fn constrain_penalty(pen: &Array2<f64>, v: ArrayView1<f64>) -> Array2<f64> {
    let sym = (pen + &pen.t()) / 2.0;
    if v.is_empty() {
        return sym;
    }
    let vv = v.dot(&v);
    let q = sym.nrows();
    // H S H = S - 2/vv (S v v' + v v' S) + 4/vv^2 (v' S v) v v'
    let sv = sym.dot(&v);
    let vsv = v.dot(&sv);
    let mut out = Array2::<f64>::zeros((q - 1, q - 1));
    for i in 1..q {
        for j in 1..q {
            out[[i - 1, j - 1]] = sym[[i, j]] - 2.0 / vv * (sv[i] * v[j] + v[i] * sv[j])
                + 4.0 / (vv * vv) * vsv * v[i] * v[j];
        }
    }
    (&out + &out.t()) / 2.0
}

fn frobenius(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

impl RefitResult {
    /// The constrained design of block `b` at new rows: `raw` is the
    /// learner's raw design there, `by` the new rows' `by=` values.
    ///
    /// With `zero_rows_have_no_effect`, a row whose raw design is all zero
    /// (a group the fit never saw, encoded as zero indicators) gets a zero
    /// design row, so its effect is zero with zero standard error: the
    /// average over groups, the prediction lme4 and mgcv make for a new
    /// group.
    pub fn block_design(
        &self,
        b: usize,
        raw: ArrayView2<f64>,
        by: Option<ArrayView1<f64>>,
        zero_rows_have_no_effect: bool,
    ) -> Result<Array2<f64>, MboostrsError> {
        let block = self
            .blocks
            .get(b)
            .ok_or_else(|| MboostrsError::Control(format!("no refit term {b}")))?;
        let width = block.end - block.start;
        let expected = if block.householder.is_empty() {
            width
        } else {
            width + 1
        };
        if raw.ncols() != expected {
            return Err(MboostrsError::Shape(format!(
                "term {} needs {expected} raw design columns, got {}",
                block.name,
                raw.ncols()
            )));
        }
        let mut x = apply_constraint(raw, block.householder.view());
        if zero_rows_have_no_effect {
            for (i, row) in raw.rows().into_iter().enumerate() {
                if row.iter().all(|v| *v == 0.0) {
                    x.row_mut(i).fill(0.0);
                }
            }
        }
        match (block.has_by, by) {
            (true, Some(by)) => {
                if by.len() != raw.nrows() {
                    return Err(MboostrsError::Shape(format!(
                        "term {} has a by column of length {} for {} rows",
                        block.name,
                        by.len(),
                        raw.nrows()
                    )));
                }
                for (mut row, &f) in x.rows_mut().into_iter().zip(by.iter()) {
                    row *= f;
                }
            }
            (true, None) => {
                return Err(MboostrsError::Control(format!(
                    "term {} was fitted with a by column; supply its values",
                    block.name
                )))
            }
            (false, _) => {}
        }
        Ok(x)
    }

    /// Effect of block `b` and its standard error at new rows (see
    /// [`block_design`](Self::block_design)).
    pub fn effect(
        &self,
        b: usize,
        raw: ArrayView2<f64>,
        by: Option<ArrayView1<f64>>,
        zero_rows_have_no_effect: bool,
    ) -> Result<(Array1<f64>, Array1<f64>), MboostrsError> {
        let block = self
            .blocks
            .get(b)
            .ok_or_else(|| MboostrsError::Control(format!("no refit term {b}")))?;
        let range = block.start..block.end;
        block_effect(
            raw,
            block.householder.view(),
            self.coef.slice(s![range.clone()]),
            self.cov.slice(s![range.clone(), range]),
            block.has_by,
            by,
            zero_rows_have_no_effect,
        )
    }

    /// The intercept.
    pub fn intercept(&self) -> f64 {
        self.coef[0]
    }

    /// The summed linear predictor (intercept plus every term's constrained
    /// effect) at new rows, with its standard error from the full covariance:
    /// `se_i = sqrt(x_i' V x_i)`, where `x_i` holds the intercept and each
    /// term's constrained design row. Terms are given in the order of
    /// `self.blocks`; `corrected` uses the covariance corrected for
    /// smoothing-parameter uncertainty.
    pub fn linear_predictor(
        &self,
        terms: &[lss_refit::NewRowsTerm],
        corrected: bool,
    ) -> Result<(Array1<f64>, Array1<f64>), MboostrsError> {
        let x = self.new_rows_design(terms, true)?;
        let cov = if corrected {
            &self.cov_corrected
        } else {
            &self.cov
        };
        predictor_with_se(&x, self.coef.view(), cov.view())
    }

    /// The average change in the linear predictor from the rows `lo` to the
    /// rows `hi`, as `(estimate, se)`.
    ///
    /// Each set holds one raw design per term, as for
    /// [`linear_predictor`](Self::linear_predictor), with the same number of
    /// rows in both. The estimate is the mean over the rows of the predictor at
    /// `hi` minus the predictor at `lo`; the intercept cancels, so the contrast
    /// is `d' coef` for `d`, the mean over rows of the constrained design
    /// difference, and its standard error is `sqrt(d' V d)`. A term whose
    /// covariates do not change between the two sets contributes zero to `d`.
    pub fn linear_contrast(
        &self,
        hi: &[lss_refit::NewRowsTerm],
        lo: &[lss_refit::NewRowsTerm],
        corrected: bool,
    ) -> Result<(f64, f64), MboostrsError> {
        let x_hi = self.new_rows_design(hi, false)?;
        let x_lo = self.new_rows_design(lo, false)?;
        if x_hi.nrows() != x_lo.nrows() || x_hi.nrows() == 0 {
            return Err(MboostrsError::Shape(format!(
                "the two row sets of a contrast need the same nonzero row count, got {} and {}",
                x_hi.nrows(),
                x_lo.nrows()
            )));
        }
        let n = x_hi.nrows() as f64;
        let d = (&x_hi - &x_lo).sum_axis(ndarray::Axis(0)) / n;
        let d = d.insert_axis(ndarray::Axis(0));
        let cov = if corrected {
            &self.cov_corrected
        } else {
            &self.cov
        };
        let (eta, se) = predictor_with_se(&d, self.coef.view(), cov.view())?;
        Ok((eta[0], se[0]))
    }

    /// The design of the new rows: one row per row of the terms, with the
    /// intercept column first when `intercept` is true.
    fn new_rows_design(
        &self,
        terms: &[lss_refit::NewRowsTerm],
        intercept: bool,
    ) -> Result<Array2<f64>, MboostrsError> {
        if terms.len() != self.blocks.len() {
            return Err(MboostrsError::Shape(format!(
                "terms for {} blocks, expected {}",
                terms.len(),
                self.blocks.len()
            )));
        }
        let n_rows = terms.first().map_or(0, |t| t.raw.nrows());
        let mut x = Array2::<f64>::zeros((n_rows, self.coef.len()));
        if intercept {
            x.column_mut(0).fill(1.0);
        }
        for (block, term) in self.blocks.iter().zip(terms) {
            if term.raw.nrows() != n_rows {
                return Err(MboostrsError::Shape(
                    "terms differ in their row counts".into(),
                ));
            }
            let xb = constrained_design(
                term.raw.view(),
                block.householder.view(),
                block.end - block.start,
                block.has_by,
                term.by.as_ref().map(|b| b.view()),
                term.zero_rows_have_no_effect,
            )?;
            x.slice_mut(s![.., block.start..block.end]).assign(&xb);
        }
        Ok(x)
    }
}

/// The linear predictor `x coef` and its standard error
/// `sqrt(diag(x V x'))` at the rows of the design `x`, where `V` is the
/// covariance of `coef`. `x` has one column per coefficient, intercept first.
pub fn predictor_with_se(
    x: &Array2<f64>,
    coef: ArrayView1<f64>,
    cov: ArrayView2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), MboostrsError> {
    let p = coef.len();
    if x.ncols() != p || cov.dim() != (p, p) {
        return Err(MboostrsError::Shape(format!(
            "design has {} columns, covariance is {:?}, for {p} coefficients",
            x.ncols(),
            cov.dim()
        )));
    }
    let eta = x.dot(&coef);
    let xv = x.dot(&cov);
    let se = (&xv * x)
        .sum_axis(ndarray::Axis(1))
        .mapv(|v| v.max(0.0).sqrt());
    Ok((eta, se))
}

/// A term's effect and standard error at new rows, from its raw design
/// there and the stored pieces of a refit: the constraint's Householder
/// vector, the term's coefficients and their covariance block.
///
/// This is what [`RefitResult::effect`] computes; it takes the pieces
/// separately so a binding can keep them as plain data (an R object that
/// survives `saveRDS`) and still evaluate effects in Rust.
pub fn block_effect(
    raw: ArrayView2<f64>,
    householder: ArrayView1<f64>,
    coef: ArrayView1<f64>,
    cov: ArrayView2<f64>,
    has_by: bool,
    by: Option<ArrayView1<f64>>,
    zero_rows_have_no_effect: bool,
) -> Result<(Array1<f64>, Array1<f64>), MboostrsError> {
    let width = coef.len();
    if cov.dim() != (width, width) {
        return Err(MboostrsError::Shape(format!(
            "covariance block is {:?} for {width} coefficients",
            cov.dim()
        )));
    }
    let x = constrained_design(
        raw,
        householder,
        width,
        has_by,
        by,
        zero_rows_have_no_effect,
    )?;
    let estimate = x.dot(&coef);
    let xv = x.dot(&cov);
    let se = (&xv * &x)
        .sum_axis(ndarray::Axis(1))
        .mapv(|v| v.max(0.0).sqrt());
    Ok((estimate, se))
}

/// A term's constrained design at new rows (`width` columns): the raw
/// design reflected by the term's Householder vector, zero rows kept at
/// zero when requested (unseen groups), and multiplied by the `by=` column.
pub fn constrained_design(
    raw: ArrayView2<f64>,
    householder: ArrayView1<f64>,
    width: usize,
    has_by: bool,
    by: Option<ArrayView1<f64>>,
    zero_rows_have_no_effect: bool,
) -> Result<Array2<f64>, MboostrsError> {
    let expected = if householder.is_empty() {
        width
    } else {
        width + 1
    };
    if raw.ncols() != expected {
        return Err(MboostrsError::Shape(format!(
            "the term needs {expected} raw design columns, got {}",
            raw.ncols()
        )));
    }
    let mut x = apply_constraint(raw, householder);
    if zero_rows_have_no_effect {
        for (i, row) in raw.rows().into_iter().enumerate() {
            if row.iter().all(|v| *v == 0.0) {
                x.row_mut(i).fill(0.0);
            }
        }
    }
    match (has_by, by) {
        (true, Some(by)) => {
            if by.len() != raw.nrows() {
                return Err(MboostrsError::Shape(format!(
                    "by column of length {} for {} rows",
                    by.len(),
                    raw.nrows()
                )));
            }
            for (mut row, &f) in x.rows_mut().into_iter().zip(by.iter()) {
                row *= f;
            }
        }
        (true, None) => {
            return Err(MboostrsError::Control(
                "the term was fitted with a by column; supply its values".into(),
            ))
        }
        (false, _) => {}
    }
    Ok(x)
}

#[cfg(test)]
mod tests;
