//! Maximum-likelihood linear fit in the same family as a boost.
//!
//! `same_family_gap` (Python) compares a linear boost's log-likelihood at its
//! stopping step with the maximum of the same log-likelihood over the same
//! linear predictor: the unpenalized GLM on the boost's design, the
//! proportional-odds model for `PropOdds`, and the beta regression for a
//! linear `BetaLSS`. [`fit_same_family`] computes that maximum.
//!
//! The objective is the family's own risk from [`crate::families`] or
//! [`crate::lss`], the quantity `logLik` negates, so the maximum and the
//! boost's `logLik` are values of one function. Two additions turn a risk
//! into the reported log-likelihood: `GammaReg`'s risk omits `log(y)` on each
//! row, which is added back, and the Gaussian squared-error risk is turned
//! into the profile likelihood with `sigma^2 = RSS / n`.
//!
//! The optimizer is a damped Newton iteration over the coefficients and the
//! family's nuisance parameters jointly (the gamma shape and negative-binomial
//! size on the log scale, the proportional-odds thresholds through R's `d2s`
//! parameterization). For a standard family the derivatives with respect to
//! the linear predictor are central differences of each row's loss; the
//! families' negative gradients follow mboost and are scaled for some
//! families (Binomial by `1 / log 2`, Gaussian by `1 / 2`), so they are not
//! used as derivatives of the risk. For `BetaLSS` the family's analytic
//! gradient and Hessian are used. Directions with a near-zero Hessian
//! eigenvalue, such as those of a collinear design, are left out of each
//! step, which is what a minimum-norm least-squares solve does.

use faer::{Mat, Side};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use crate::families::{
    delta_to_sigma, sigma_to_delta, Binomial, Family, GammaReg, Gaussian, NBinomial, Poisson,
    PropOdds,
};
use crate::lss::{BetaLSS, LssFamily};
use crate::MboostrsError;

/// Relative step for the per-row central differences of the loss.
const ROW_STEP: f64 = 1e-4;
/// Relative step for differences of the total risk in a nuisance coordinate.
const NUISANCE_STEP: f64 = 1e-4;
/// Newton iterations before the fit is reported as not converged.
const MAX_ITER: usize = 200;
/// Converged when the Newton decrement, relative to `1 + |risk|`, is below this.
const DECREMENT_TOL: f64 = 1e-14;
/// After a failed line search, the decrement still counts as converged below this.
const STALL_TOL: f64 = 1e-8;
/// Bounds for a log-scale nuisance (gamma shape, negative-binomial size).
const LOG_NUISANCE_BOUNDS: (f64, f64) = (-13.815_510_557_964_274, 18.420_680_743_952_367);

/// The likelihoods [`fit_same_family`] maximizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SameFamily {
    /// Gaussian linear model; profile likelihood with `sigma^2 = RSS / n`.
    Gaussian,
    /// Binomial logit GLM.
    Binomial,
    /// Poisson log-linear GLM.
    Poisson,
    /// Gamma log-linear GLM with the shape estimated.
    GammaReg,
    /// Negative-binomial log-linear GLM with the size estimated.
    NBinomial,
    /// Proportional-odds model with the thresholds estimated.
    PropOdds,
    /// Beta regression: logit-linear mean, log-linear precision.
    BetaLss,
}

impl SameFamily {
    /// The comparison for a Python family class name (`"GaussReg"` is the
    /// Gaussian linear model).
    pub fn from_name(name: &str) -> Result<Self, MboostrsError> {
        match name {
            "Gaussian" | "GaussReg" => Ok(Self::Gaussian),
            "Binomial" => Ok(Self::Binomial),
            "Poisson" => Ok(Self::Poisson),
            "GammaReg" => Ok(Self::GammaReg),
            "NBinomial" => Ok(Self::NBinomial),
            "PropOdds" => Ok(Self::PropOdds),
            "BetaLSS" => Ok(Self::BetaLss),
            other => Err(MboostrsError::Control(format!(
                "no same-family fit for {other}"
            ))),
        }
    }

    fn n_predictors(self) -> usize {
        match self {
            Self::BetaLss => 2,
            _ => 1,
        }
    }
}

/// One linear predictor `offset + design * beta`, and the boost's own
/// link-scale predictor for it, which seeds the fit.
pub struct LinearPredictor<'a> {
    /// `n x p` design; `p` may be zero.
    pub design: ArrayView2<'a, f64>,
    /// Length-`n` offset, held fixed.
    pub offset: ArrayView1<'a, f64>,
    /// Length-`n` link-scale predictor of the boost being compared.
    pub start: ArrayView1<'a, f64>,
}

/// Result of [`fit_same_family`].
#[derive(Clone, Debug)]
pub struct SameFamilyFit {
    /// The maximized log-likelihood, in nats.
    pub loglik: f64,
    /// The same log-likelihood at the supplied start predictors and nuisance:
    /// the boost's value, for checking against its `logLik`.
    pub start_loglik: f64,
    /// Whether Newton's decrement reached its tolerance at an interior point.
    pub converged: bool,
    /// Newton iterations taken.
    pub iterations: usize,
    /// Fitted coefficients, one vector per linear predictor.
    pub coefficients: Vec<Array1<f64>>,
    /// Fitted nuisance on its natural scale: the gamma shape, the
    /// negative-binomial size, or the proportional-odds thresholds.
    pub nuisance: Vec<f64>,
}

/// Maximize the `family` log-likelihood over the coefficients of `predictors`
/// and the family's nuisance parameters.
///
/// `y` is the response as the boost received it (Binomial `0/1`, PropOdds
/// codes `1..K`). `nuisance` is the boost's nuisance at the step compared:
/// one gamma shape or negative-binomial size, the `K - 1` proportional-odds
/// thresholds, or empty. A Gaussian comparison requires uniform `w`, because
/// the Gaussian `logLik` is the unweighted residual likelihood.
///
/// Errors on inconsistent lengths, a response outside the family's domain, or
/// a nuisance of the wrong length. A fit that does not reach a maximum is
/// returned with `converged = false`.
pub fn fit_same_family(
    family: SameFamily,
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
    predictors: &[LinearPredictor<'_>],
    nuisance: &[f64],
) -> Result<SameFamilyFit, MboostrsError> {
    let problem = Problem::new(family, y, w, predictors, nuisance)?;
    let start_risk = problem.risk_at(&problem.start_etas(), &problem.start_nuisance);
    let start_loglik = problem.loglik(start_risk);

    let mut theta = problem.initial_theta();
    let mut risk = problem.risk(&theta);
    let mut converged = false;
    let mut iterations = 0;
    if risk.is_finite() {
        for _ in 0..MAX_ITER {
            iterations += 1;
            let (gradient, hessian) = problem.derivatives(&theta);
            if !gradient.iter().all(|v| v.is_finite()) {
                break;
            }
            let direction = -pseudo_solve(&hessian, &gradient);
            let slope = gradient.dot(&direction);
            let decrement = -slope;
            if decrement <= DECREMENT_TOL * (1.0 + risk.abs()) {
                converged = true;
                break;
            }
            let mut step = 1.0;
            let mut accepted = None;
            for _ in 0..50 {
                let candidate = problem.clamp(&theta + &(step * &direction));
                let value = problem.risk(&candidate);
                if value.is_finite() && value <= risk + 1e-4 * step * slope {
                    accepted = Some((candidate, value));
                    break;
                }
                step *= 0.5;
            }
            match accepted {
                Some((candidate, value)) => {
                    theta = candidate;
                    risk = value;
                }
                None => {
                    converged = decrement <= STALL_TOL * (1.0 + risk.abs());
                    break;
                }
            }
        }
    }
    if problem.at_nuisance_bound(&theta) {
        converged = false;
    }

    let coefficients = problem.split_coefficients(&theta);
    let nuisance = problem.natural_nuisance(&theta.slice(s![problem.n_coef..]).to_vec());
    Ok(SameFamilyFit {
        loglik: if risk.is_finite() {
            problem.loglik(risk)
        } else {
            f64::NAN
        },
        start_loglik,
        converged: converged && risk.is_finite(),
        iterations,
        coefficients,
        nuisance,
    })
}

struct Problem<'p, 'a> {
    family: SameFamily,
    /// Response in the family's internal coding.
    y: Array1<f64>,
    /// Weights the risk is evaluated with (ones for Gaussian).
    w: Array1<f64>,
    predictors: &'p [LinearPredictor<'a>],
    /// Start of each predictor's block in the parameter vector.
    starts: Vec<usize>,
    n_coef: usize,
    start_nuisance: Vec<f64>,
    /// `sum(w log y)`, the term GammaReg's risk omits.
    gamma_constant: f64,
}

impl<'p, 'a> Problem<'p, 'a> {
    fn new(
        family: SameFamily,
        y: ArrayView1<f64>,
        w: ArrayView1<f64>,
        predictors: &'p [LinearPredictor<'a>],
        nuisance: &[f64],
    ) -> Result<Self, MboostrsError> {
        let n = y.len();
        if predictors.len() != family.n_predictors() {
            return Err(MboostrsError::Shape(format!(
                "{family:?} takes {} linear predictor(s), got {}",
                family.n_predictors(),
                predictors.len()
            )));
        }
        if w.len() != n {
            return Err(MboostrsError::Shape(format!(
                "weights have length {} but the response has {n}",
                w.len()
            )));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(MboostrsError::Control(
                "weights must be finite and non-negative".into(),
            ));
        }
        for (k, p) in predictors.iter().enumerate() {
            if p.design.nrows() != n || p.offset.len() != n || p.start.len() != n {
                return Err(MboostrsError::Shape(format!(
                    "linear predictor {k}: design has {} rows, offset {} and start {} \
                     entries; the response has {n}",
                    p.design.nrows(),
                    p.offset.len(),
                    p.start.len()
                )));
            }
        }
        let expected_nuisance = match family {
            SameFamily::GammaReg | SameFamily::NBinomial => Some(1),
            SameFamily::PropOdds => None,
            _ => Some(0),
        };
        match expected_nuisance {
            Some(q) if nuisance.len() != q => {
                return Err(MboostrsError::Shape(format!(
                    "{family:?} takes {q} nuisance value(s), got {}",
                    nuisance.len()
                )));
            }
            None if nuisance.len() < 2 => {
                return Err(MboostrsError::Shape(format!(
                    "PropOdds takes at least 2 thresholds, got {}",
                    nuisance.len()
                )));
            }
            _ => {}
        }
        if nuisance.iter().any(|v| !v.is_finite()) {
            return Err(MboostrsError::Control(
                "nuisance values must be finite".into(),
            ));
        }
        if matches!(family, SameFamily::GammaReg | SameFamily::NBinomial) && nuisance[0] <= 0.0 {
            return Err(MboostrsError::Control(format!(
                "{family:?} nuisance must be positive, got {}",
                nuisance[0]
            )));
        }

        let mut w = w.to_owned();
        if family == SameFamily::Gaussian {
            let first = w.first().copied().unwrap_or(1.0);
            if w.iter()
                .any(|v| (v - first).abs() > 1e-8 * (1.0 + first.abs()))
            {
                return Err(MboostrsError::Control(
                    "Gaussian logLik() is the unweighted residual likelihood, so a \
                     weighted comparison would not be the same number"
                        .into(),
                ));
            }
            w.fill(1.0);
        }

        let mut starts = Vec::with_capacity(predictors.len());
        let mut n_coef = 0;
        for p in predictors {
            starts.push(n_coef);
            n_coef += p.design.ncols();
        }

        let mut problem = Self {
            family,
            y: y.to_owned(),
            w,
            predictors,
            starts,
            n_coef,
            start_nuisance: nuisance.to_vec(),
            gamma_constant: 0.0,
        };
        match family {
            SameFamily::BetaLss => BetaLSS.check_y(y)?,
            _ => {
                let fam = problem.standard_family(nuisance);
                fam.check_y(y)?;
                if let Some(encoded) = fam.encode_response(y) {
                    problem.y = encoded;
                }
            }
        }
        if family == SameFamily::GammaReg {
            problem.gamma_constant = problem
                .y
                .iter()
                .zip(problem.w.iter())
                .map(|(yi, wi)| wi * yi.ln())
                .sum();
        }
        Ok(problem)
    }

    fn n_nuisance(&self) -> usize {
        self.start_nuisance.len()
    }

    /// The standard family with its nuisance at natural-scale `nuisance`.
    fn standard_family(&self, nuisance: &[f64]) -> Box<dyn Family> {
        match self.family {
            SameFamily::Gaussian => Box::new(Gaussian),
            SameFamily::Binomial => Box::new(Binomial),
            SameFamily::Poisson => Box::new(Poisson),
            SameFamily::GammaReg => Box::new(GammaReg { sigma: nuisance[0] }),
            SameFamily::NBinomial => Box::new(NBinomial { sigma: nuisance[0] }),
            SameFamily::PropOdds => Box::new(PropOdds {
                sigma: nuisance.to_vec(),
            }),
            SameFamily::BetaLss => unreachable!("BetaLSS is not a standard family"),
        }
    }

    /// Nuisance in the unconstrained coordinates the optimizer moves.
    fn free_nuisance(&self, natural: &[f64]) -> Vec<f64> {
        match self.family {
            SameFamily::GammaReg | SameFamily::NBinomial => vec![natural[0].ln()],
            SameFamily::PropOdds => sigma_to_delta(natural),
            _ => Vec::new(),
        }
    }

    fn natural_nuisance(&self, free: &[f64]) -> Vec<f64> {
        match self.family {
            SameFamily::GammaReg | SameFamily::NBinomial => vec![free[0].exp()],
            SameFamily::PropOdds => delta_to_sigma(free),
            _ => Vec::new(),
        }
    }

    fn start_etas(&self) -> Vec<Array1<f64>> {
        self.predictors.iter().map(|p| p.start.to_owned()).collect()
    }

    /// Coefficients that reproduce each start predictor as closely as the
    /// design allows, and the start nuisance in free coordinates.
    ///
    /// For PropOdds the design has no intercept, because the thresholds play
    /// that role; the constant part of the start predictor is moved into the
    /// thresholds, which keeps the start's fitted probabilities.
    fn initial_theta(&self) -> Array1<f64> {
        let mut theta = Array1::<f64>::zeros(self.n_coef + self.n_nuisance());
        let mut nuisance = self.start_nuisance.clone();
        for (k, p) in self.predictors.iter().enumerate() {
            let target = &p.start - &p.offset;
            if self.family == SameFamily::PropOdds {
                let n = target.len();
                let mut augmented = Array2::<f64>::ones((n, p.design.ncols() + 1));
                augmented.slice_mut(s![.., 1..]).assign(&p.design);
                let coef = least_squares(augmented.view(), target.view());
                let constant = coef[0];
                theta
                    .slice_mut(s![self.starts[k]..self.starts[k] + p.design.ncols()])
                    .assign(&coef.slice(s![1..]));
                for t in nuisance.iter_mut() {
                    *t -= constant;
                }
                for j in 1..nuisance.len() {
                    if nuisance[j] <= nuisance[j - 1] {
                        nuisance[j] = nuisance[j - 1] + 1e-4;
                    }
                }
            } else {
                let coef = least_squares(p.design, target.view());
                theta
                    .slice_mut(s![self.starts[k]..self.starts[k] + p.design.ncols()])
                    .assign(&coef);
            }
        }
        let free = self.free_nuisance(&nuisance);
        theta
            .slice_mut(s![self.n_coef..])
            .assign(&Array1::from(free));
        self.clamp(theta)
    }

    fn split_coefficients(&self, theta: &Array1<f64>) -> Vec<Array1<f64>> {
        self.predictors
            .iter()
            .enumerate()
            .map(|(k, p)| {
                theta
                    .slice(s![self.starts[k]..self.starts[k] + p.design.ncols()])
                    .to_owned()
            })
            .collect()
    }

    fn etas(&self, theta: &Array1<f64>) -> Vec<Array1<f64>> {
        self.predictors
            .iter()
            .zip(self.split_coefficients(theta))
            .map(|(p, beta)| {
                if beta.is_empty() {
                    p.offset.to_owned()
                } else {
                    &p.offset + &p.design.dot(&beta)
                }
            })
            .collect()
    }

    fn risk(&self, theta: &Array1<f64>) -> f64 {
        let nuisance = self.natural_nuisance(&theta.as_slice().unwrap()[self.n_coef..]);
        self.risk_at(&self.etas(theta), &nuisance)
    }

    fn risk_at(&self, etas: &[Array1<f64>], nuisance: &[f64]) -> f64 {
        match self.family {
            SameFamily::BetaLss => {
                BetaLSS.risk(self.y.as_slice().unwrap(), etas, self.w.as_slice().unwrap())
            }
            _ => self
                .standard_family(nuisance)
                .risk(self.y.view(), etas[0].view(), self.w.view()),
        }
    }

    /// The log-likelihood `logLik` reports, from the risk.
    fn loglik(&self, risk: f64) -> f64 {
        match self.family {
            SameFamily::Gaussian => {
                let n = self.y.len() as f64;
                if risk <= 0.0 {
                    f64::INFINITY
                } else {
                    -0.5 * n * ((2.0 * std::f64::consts::PI * risk / n).ln() + 1.0)
                }
            }
            SameFamily::GammaReg => -risk - self.gamma_constant,
            _ => -risk,
        }
    }

    fn clamp(&self, mut theta: Array1<f64>) -> Array1<f64> {
        if matches!(self.family, SameFamily::GammaReg | SameFamily::NBinomial) {
            let (lo, hi) = LOG_NUISANCE_BOUNDS;
            let v = &mut theta[self.n_coef];
            *v = v.clamp(lo, hi);
        }
        theta
    }

    fn at_nuisance_bound(&self, theta: &Array1<f64>) -> bool {
        if matches!(self.family, SameFamily::GammaReg | SameFamily::NBinomial) {
            let (lo, hi) = LOG_NUISANCE_BOUNDS;
            let v = theta[self.n_coef];
            v <= lo + 1e-9 || v >= hi - 1e-9
        } else {
            false
        }
    }

    /// Gradient and Hessian of the risk in the parameter vector.
    fn derivatives(&self, theta: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        match self.family {
            SameFamily::BetaLss => self.lss_derivatives(theta),
            _ => self.standard_derivatives(theta),
        }
    }

    fn lss_derivatives(&self, theta: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        let etas = self.etas(theta);
        let n = self.y.len();
        let k = etas.len();
        let y = self.y.as_slice().unwrap();
        let ones = vec![1.0; n];
        let m = self.n_coef;
        let mut gradient = Array1::<f64>::zeros(m);
        let mut hessian = Array2::<f64>::zeros((m, m));
        let mut ngrad = vec![0.0; n];
        for a in 0..k {
            BetaLSS.ngradient(a, y, &etas, &ones, &mut ngrad);
            let weighted: Array1<f64> = (0..n).map(|i| -self.w[i] * ngrad[i]).collect();
            let p = &self.predictors[a];
            gradient
                .slice_mut(s![self.starts[a]..self.starts[a] + p.design.ncols()])
                .assign(&p.design.t().dot(&weighted));
        }
        let mut nh = vec![0.0; n * k * k];
        BetaLSS.neg_hessian(y, &etas, &mut nh);
        for a in 0..k {
            for b in 0..k {
                let row_weights: Array1<f64> = (0..n)
                    .map(|i| self.w[i] * nh[i * k * k + a * k + b])
                    .collect();
                let block = weighted_cross(
                    self.predictors[a].design,
                    self.predictors[b].design,
                    &row_weights,
                );
                let (ra, rb) = (self.starts[a], self.starts[b]);
                hessian
                    .slice_mut(s![ra..ra + block.nrows(), rb..rb + block.ncols()])
                    .assign(&block);
            }
        }
        (gradient, symmetrize(hessian))
    }

    fn standard_derivatives(&self, theta: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        let eta = self.etas(theta).swap_remove(0);
        let x = self.predictors[0].design;
        let p = self.n_coef;
        let q = self.n_nuisance();
        let free = theta.slice(s![p..]).to_vec();
        let family = self.standard_family(&self.natural_nuisance(&free));
        let (d1, d2) = row_derivatives(family.as_ref(), &self.y, &eta);

        let mut gradient = Array1::<f64>::zeros(p + q);
        let mut hessian = Array2::<f64>::zeros((p + q, p + q));
        let wd1 = &self.w * &d1;
        let wd2 = &self.w * &d2;
        gradient.slice_mut(s![..p]).assign(&x.t().dot(&wd1));
        hessian
            .slice_mut(s![..p, ..p])
            .assign(&weighted_cross(x, x, &wd2));

        if q > 0 {
            let risk_with = |free: &[f64]| {
                self.standard_family(&self.natural_nuisance(free)).risk(
                    self.y.view(),
                    eta.view(),
                    self.w.view(),
                )
            };
            let shifted = |deltas: &[(usize, f64)]| {
                let mut v = free.clone();
                for &(j, d) in deltas {
                    v[j] += d;
                }
                v
            };
            let r0 = family.risk(self.y.view(), eta.view(), self.w.view());
            let steps: Vec<f64> = free
                .iter()
                .map(|v| NUISANCE_STEP * (1.0 + v.abs()))
                .collect();
            for j in 0..q {
                let h = steps[j];
                let up = shifted(&[(j, h)]);
                let down = shifted(&[(j, -h)]);
                let r_up = risk_with(&up);
                let r_down = risk_with(&down);
                gradient[p + j] = (r_up - r_down) / (2.0 * h);
                hessian[[p + j, p + j]] = (r_up - 2.0 * r0 + r_down) / (h * h);

                let fam_up = self.standard_family(&self.natural_nuisance(&up));
                let fam_down = self.standard_family(&self.natural_nuisance(&down));
                let (d1_up, _) = row_derivatives(fam_up.as_ref(), &self.y, &eta);
                let (d1_down, _) = row_derivatives(fam_down.as_ref(), &self.y, &eta);
                let cross = (&d1_up - &d1_down) * &self.w / (2.0 * h);
                let column = x.t().dot(&cross);
                hessian.slice_mut(s![..p, p + j]).assign(&column);
                hessian.slice_mut(s![p + j, ..p]).assign(&column);

                for l in 0..j {
                    let g = steps[l];
                    let pp = risk_with(&shifted(&[(j, h), (l, g)]));
                    let pm = risk_with(&shifted(&[(j, h), (l, -g)]));
                    let mp = risk_with(&shifted(&[(j, -h), (l, g)]));
                    let mm = risk_with(&shifted(&[(j, -h), (l, -g)]));
                    let value = (pp - pm - mp + mm) / (4.0 * h * g);
                    hessian[[p + j, p + l]] = value;
                    hessian[[p + l, p + j]] = value;
                }
            }
        }
        (gradient, symmetrize(hessian))
    }
}

/// First and second derivatives of each row's loss in its linear predictor,
/// by central differences with step `ROW_STEP * (1 + |eta|)`.
fn row_derivatives(
    family: &dyn Family,
    y: &Array1<f64>,
    eta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = eta.len();
    let one = Array1::<f64>::ones(1);
    let mut d1 = Array1::<f64>::zeros(n);
    let mut d2 = Array1::<f64>::zeros(n);
    let mut point = Array1::<f64>::zeros(1);
    let loss = |yi: ArrayView1<f64>, value: f64, point: &mut Array1<f64>| {
        point[0] = value;
        family.risk(yi, point.view(), one.view())
    };
    for i in 0..n {
        let yi = y.slice(s![i..i + 1]);
        let h = ROW_STEP * (1.0 + eta[i].abs());
        let up = loss(yi, eta[i] + h, &mut point);
        let mid = loss(yi, eta[i], &mut point);
        let down = loss(yi, eta[i] - h, &mut point);
        d1[i] = (up - down) / (2.0 * h);
        d2[i] = (up - 2.0 * mid + down) / (h * h);
    }
    (d1, d2)
}

/// `a' diag(weights) b`.
fn weighted_cross(a: ArrayView2<f64>, b: ArrayView2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let mut wb = b.to_owned();
    for (mut row, w) in wb.rows_mut().into_iter().zip(weights.iter()) {
        row.mapv_inplace(|v| v * w);
    }
    a.t().dot(&wb)
}

fn symmetrize(m: Array2<f64>) -> Array2<f64> {
    let t = m.t().to_owned();
    (&m + &t) * 0.5
}

/// `sum_k (u_k' b / |lambda_k|) u_k` over the eigenpairs of symmetric `a`
/// whose eigenvalue exceeds `1e-10` of the largest in magnitude. For a
/// positive semidefinite `a` this is the minimum-norm solution of `a x = b`;
/// taking `|lambda|` makes `-x` a descent direction when `a` is an indefinite
/// Hessian and `b` the gradient.
fn pseudo_solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let m = a.nrows();
    let mut x = Array1::<f64>::zeros(m);
    if m == 0 {
        return x;
    }
    let fm = Mat::from_fn(m, m, |i, j| a[[i, j]]);
    let eig = fm.selfadjoint_eigendecomposition(Side::Lower);
    let s = eig.s().column_vector();
    let u = eig.u();
    let largest = (0..m).map(|k| s.read(k).abs()).fold(0.0f64, f64::max);
    if largest == 0.0 || !largest.is_finite() {
        return x;
    }
    let tol = 1e-10 * largest;
    for k in 0..m {
        let lambda = s.read(k).abs();
        if lambda <= tol {
            continue;
        }
        let c: f64 = (0..m).map(|i| u.read(i, k) * b[i]).sum::<f64>() / lambda;
        for i in 0..m {
            x[i] += c * u.read(i, k);
        }
    }
    x
}

/// Minimum-norm least-squares coefficients of `target` on `design`.
fn least_squares(design: ArrayView2<f64>, target: ArrayView1<f64>) -> Array1<f64> {
    if design.ncols() == 0 {
        return Array1::zeros(0);
    }
    let gram = design.t().dot(&design);
    let rhs = design.t().dot(&target);
    pseudo_solve(&gram, &rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Deterministic covariates and a fixed pseudo-random sequence, so the
    /// fixtures need no random-number crate.
    fn fixture(n: usize) -> (Array2<f64>, Array1<f64>) {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut design = Array2::<f64>::ones((n, 3));
        let mut u = Array1::<f64>::zeros(n);
        for i in 0..n {
            design[[i, 1]] = (i as f64 / n as f64) * 2.0 - 1.0;
            design[[i, 2]] = uniform() - 0.5;
            u[i] = uniform();
        }
        (design, u)
    }

    fn fit_one(
        family: SameFamily,
        design: &Array2<f64>,
        y: &Array1<f64>,
        start: &Array1<f64>,
        nuisance: &[f64],
    ) -> SameFamilyFit {
        let n = y.len();
        let offset = Array1::<f64>::zeros(n);
        let w = Array1::<f64>::ones(n);
        let predictors = [LinearPredictor {
            design: design.view(),
            offset: offset.view(),
            start: start.view(),
        }];
        fit_same_family(family, y.view(), w.view(), &predictors, nuisance).unwrap()
    }

    #[test]
    fn gaussian_matches_ordinary_least_squares() {
        let (x, u) = fixture(60);
        let y: Array1<f64> = (0..60)
            .map(|i| 1.0 + 2.0 * x[[i, 1]] - x[[i, 2]] + (u[i] - 0.5))
            .collect();
        let start = Array1::<f64>::from_elem(60, y.mean().unwrap());
        let fit = fit_one(SameFamily::Gaussian, &x, &y, &start, &[]);
        assert!(fit.converged);
        let beta = least_squares(x.view(), y.view());
        let rss: f64 = (&y - &x.dot(&beta)).mapv(|r| r * r).sum();
        let n = 60.0;
        let expected = -0.5 * n * ((2.0 * std::f64::consts::PI * rss / n).ln() + 1.0);
        assert!((fit.loglik - expected).abs() < 1e-9 * expected.abs());
        assert!(fit.loglik > fit.start_loglik);
    }

    #[test]
    fn poisson_score_is_zero_at_the_fit() {
        let (x, u) = fixture(80);
        let y: Array1<f64> = (0..80)
            .map(|i| ((0.3 + 0.8 * x[[i, 1]]).exp() * 2.0 * u[i]).floor())
            .collect();
        let start = Array1::<f64>::zeros(80);
        let fit = fit_one(SameFamily::Poisson, &x, &y, &start, &[]);
        assert!(fit.converged);
        // The Poisson score equations: X'(y - mu) = 0.
        let mu = x.dot(&fit.coefficients[0]).mapv(f64::exp);
        let score = x.t().dot(&(&y - &mu));
        assert!(score.iter().all(|s| s.abs() < 1e-6), "{score:?}");
        assert!(
            (fit.start_loglik + Poisson.risk(y.view(), start.view(), Array1::ones(80).view()))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn gamma_shape_and_coefficients_are_stationary() {
        let (x, u) = fixture(100);
        let y: Array1<f64> = (0..100)
            .map(|i| (0.5 + 0.6 * x[[i, 1]]).exp() * (0.4 + 1.2 * u[i]))
            .collect();
        let start = Array1::<f64>::zeros(100);
        let fit = fit_one(SameFamily::GammaReg, &x, &y, &start, &[1.0]);
        assert!(fit.converged);
        let shape = fit.nuisance[0];
        for factor in [0.99, 1.01] {
            let other = fit_one(SameFamily::GammaReg, &x, &y, &start, &[shape * factor]);
            assert!((other.loglik - fit.loglik).abs() < 1e-8 * fit.loglik.abs());
            let fixed = Problem {
                family: SameFamily::GammaReg,
                y: y.clone(),
                w: Array1::ones(100),
                predictors: &[],
                starts: vec![],
                n_coef: 0,
                start_nuisance: vec![shape * factor],
                gamma_constant: y.mapv(f64::ln).sum(),
            };
            let eta = x.dot(&fit.coefficients[0]);
            let ll = fixed.loglik(fixed.risk_at(&[eta], &[shape * factor]));
            assert!(ll < fit.loglik);
        }
    }

    #[test]
    fn proportional_odds_threshold_shift_is_absorbed() {
        let (x, u) = fixture(120);
        let design = x.slice(s![.., 1..]).to_owned();
        let y: Array1<f64> = (0..120)
            .map(|i| {
                let eta = 1.5 * x[[i, 1]];
                let p1 = 1.0 / (1.0 + (eta + 0.3).exp());
                let p2 = 1.0 / (1.0 + (eta - 0.7).exp());
                1.0 + f64::from(u[i] > p1) + f64::from(u[i] > p2)
            })
            .collect();
        let zero = Array1::<f64>::zeros(120);
        let shifted = Array1::<f64>::from_elem(120, 2.0);
        let a = fit_one(SameFamily::PropOdds, &design, &y, &zero, &[-0.3, 0.7]);
        let b = fit_one(SameFamily::PropOdds, &design, &y, &shifted, &[1.7, 2.7]);
        assert!(a.converged && b.converged);
        assert!((a.loglik - b.loglik).abs() < 1e-8 * a.loglik.abs());
        assert!((a.start_loglik - b.start_loglik).abs() < 1e-9 * a.start_loglik.abs());
        assert!(a.loglik >= a.start_loglik);
    }

    #[test]
    fn beta_regression_is_stationary() {
        let (x, u) = fixture(90);
        let y: Array1<f64> = u.mapv(|v| 0.05 + 0.9 * v);
        let offset = Array1::<f64>::zeros(90);
        let w = Array1::<f64>::ones(90);
        let start_mu = Array1::<f64>::zeros(90);
        let start_phi = Array1::<f64>::from_elem(90, 1.0);
        let predictors = [
            LinearPredictor {
                design: x.view(),
                offset: offset.view(),
                start: start_mu.view(),
            },
            LinearPredictor {
                design: x.view(),
                offset: offset.view(),
                start: start_phi.view(),
            },
        ];
        let fit =
            fit_same_family(SameFamily::BetaLss, y.view(), w.view(), &predictors, &[]).unwrap();
        assert!(fit.converged);
        assert!(fit.loglik > fit.start_loglik);
        let n = 90;
        let ones = vec![1.0; n];
        let etas = vec![x.dot(&fit.coefficients[0]), x.dot(&fit.coefficients[1])];
        let mut ng = vec![0.0; n];
        for k in 0..2 {
            BetaLSS.ngradient(k, y.as_slice().unwrap(), &etas, &ones, &mut ng);
            let score = x.t().dot(&Array1::from(ng.clone()));
            assert!(score.iter().all(|s| s.abs() < 1e-6), "{score:?}");
        }
    }

    #[test]
    fn rejects_inputs_outside_the_family() {
        let (x, _) = fixture(10);
        let y = Array1::<f64>::from_elem(10, -1.0);
        let start = Array1::<f64>::zeros(10);
        let offset = Array1::<f64>::zeros(10);
        let w = Array1::<f64>::ones(10);
        let predictors = [LinearPredictor {
            design: x.view(),
            offset: offset.view(),
            start: start.view(),
        }];
        assert!(
            fit_same_family(SameFamily::Poisson, y.view(), w.view(), &predictors, &[]).is_err()
        );
        assert!(
            fit_same_family(SameFamily::NBinomial, y.view(), w.view(), &predictors, &[]).is_err()
        );
        let uneven = array![1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let y = Array1::<f64>::zeros(10);
        assert!(fit_same_family(
            SameFamily::Gaussian,
            y.view(),
            uneven.view(),
            &predictors,
            &[]
        )
        .is_err());
        assert!(SameFamily::from_name("Laplace").is_err());
    }

    /// Values the NumPy fitter this module replaced (IRLS for the GLMs, a
    /// profile Newton step for the gamma shape and negative-binomial size,
    /// BFGS for PropOdds) returned on these fixtures.
    #[test]
    fn agrees_with_the_replaced_numpy_fitter() {
        let close = |got: f64, want: f64| (got - want).abs() <= 1e-8 * want.abs();

        let (x, u) = fixture(100);
        let y: Array1<f64> = (0..100)
            .map(|i| {
                let p = 1.0 / (1.0 + (-(0.2 + 1.5 * x[[i, 1]])).exp());
                f64::from(u[i] < p)
            })
            .collect();
        let fit = fit_one(SameFamily::Binomial, &x, &y, &Array1::zeros(100), &[]);
        assert!(
            fit.converged && close(fit.loglik, -63.71721157917473),
            "{fit:?}"
        );

        let (x, u) = fixture(120);
        let y: Array1<f64> = (0..120)
            .map(|i| ((0.3 + 0.8 * x[[i, 1]]).exp() * 3.0 * u[i]).floor())
            .collect();
        let fit = fit_one(SameFamily::NBinomial, &x, &y, &Array1::zeros(120), &[1.0]);
        assert!(
            fit.converged && close(fit.loglik, -192.17752826339284),
            "{fit:?}"
        );
        assert!((fit.nuisance[0] - 10.63050848560736).abs() < 1e-5);

        let (x, u) = fixture(100);
        let y: Array1<f64> = (0..100)
            .map(|i| (0.5 + 0.6 * x[[i, 1]]).exp() * (0.4 + 1.2 * u[i]))
            .collect();
        let fit = fit_one(SameFamily::GammaReg, &x, &y, &Array1::zeros(100), &[1.0]);
        assert!(
            fit.converged && close(fit.loglik, -89.09327463859388),
            "{fit:?}"
        );

        let (x, u) = fixture(120);
        let design = x.slice(s![.., 1..]).to_owned();
        let y: Array1<f64> = (0..120)
            .map(|i| {
                let eta = 1.5 * x[[i, 1]];
                let p1 = 1.0 / (1.0 + (eta + 0.3).exp());
                let p2 = 1.0 / (1.0 + (eta - 0.7).exp());
                1.0 + f64::from(u[i] > p1) + f64::from(u[i] > p2)
            })
            .collect();
        let fit = fit_one(
            SameFamily::PropOdds,
            &design,
            &y,
            &Array1::zeros(120),
            &[-0.3, 0.7],
        );
        assert!(
            fit.converged && close(fit.loglik, -120.53738564155557),
            "{fit:?}"
        );
    }
}
