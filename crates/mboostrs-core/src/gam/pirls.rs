//! Penalized iteratively reweighted least squares with REML re-estimated on
//! each working model (the performance iteration; Wood 2017, section 6.9.2).
//!
//! Each iteration forms, from the current linear predictor `eta`, a working
//! response `z = eta + (y - mu) / (dmu/deta)` and working weights
//! `w (dmu/deta)^2 / V(mu)`, then solves that weighted Gaussian problem with
//! its smoothing parameters chosen by REML, until `eta` stops moving. The
//! Gaussian family needs one pass.
//!
//! Cox models use the Poisson form of the partial likelihood (Whitehead
//! 1980): with `H0` the Breslow cumulative baseline hazard at the current
//! fit, the event indicator is treated as Poisson with mean
//! `H0(t) exp(eta)`; alternating the `H0` update with the working-model step
//! converges to the penalized partial-likelihood fit.

use ndarray::{Array1, ArrayView1};

use super::laml::{self, Data, Likelihood};
use super::reml::{optimize, weighted_gram, Problem};
use super::{Model, RefitBlock, RefitControl, RefitFamily, RefitResult};
use crate::MboostrsError;

/// Breslow cumulative baseline hazard at each row's own time.
///
/// Rows are ordered by time; tied times share the risk set of their first
/// member, and a row's hazard is the cumulative sum through the last of its
/// ties.
pub fn breslow_cumulative_hazard(
    time: ArrayView1<f64>,
    status: ArrayView1<f64>,
    eta: ArrayView1<f64>,
    w: ArrayView1<f64>,
) -> Array1<f64> {
    let n = time.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let t: Vec<f64> = order.iter().map(|&i| time[i]).collect();
    let d: Vec<f64> = order.iter().map(|&i| w[i] * status[i]).collect();
    let risk: Vec<f64> = order.iter().map(|&i| w[i] * eta[i].exp()).collect();
    let mut at_risk = vec![0.0; n];
    let mut acc = 0.0;
    for i in (0..n).rev() {
        acc += risk[i];
        at_risk[i] = acc;
    }
    // first[i]: index of the first row with time t[i]; last[i]: the last.
    let mut first = vec![0usize; n];
    let mut last = vec![0usize; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && t[j + 1] == t[i] {
            j += 1;
        }
        for k in i..=j {
            first[k] = i;
            last[k] = j;
        }
        i = j + 1;
    }
    let mut cumulative = vec![0.0; n];
    let mut acc = 0.0;
    for i in 0..n {
        if d[i] > 0.0 {
            acc += d[i] / at_risk[first[i]];
        }
        cumulative[i] = acc;
    }
    let mut hazard = Array1::<f64>::zeros(n);
    for (pos, &row) in order.iter().enumerate() {
        hazard[row] = cumulative[last[pos]];
    }
    hazard
}

fn check_response(y: &Array1<f64>, family: &RefitFamily) -> Result<(), MboostrsError> {
    if y.iter().any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            family_name(family).into(),
            "the response has non-finite values".into(),
        ));
    }
    let bad = match family {
        RefitFamily::Gaussian => None,
        RefitFamily::Binomial => y
            .iter()
            .any(|v| !(0.0..=1.0).contains(v))
            .then_some("a Binomial refit needs a 0/1 (or -1/+1) response"),
        RefitFamily::Poisson | RefitFamily::NegativeBinomial { .. } => y
            .iter()
            .any(|v| *v < 0.0)
            .then_some("a count refit needs a non-negative response"),
        RefitFamily::Gamma => y
            .iter()
            .any(|v| *v <= 0.0)
            .then_some("a Gamma refit needs a positive response"),
        RefitFamily::CoxPH { .. } => y
            .iter()
            .any(|v| *v != 0.0 && *v != 1.0)
            .then_some("a CoxPH refit needs a 0/1 event indicator"),
    };
    match bad {
        Some(msg) => Err(MboostrsError::BadResponse(
            family_name(family).into(),
            msg.into(),
        )),
        None => Ok(()),
    }
}

fn family_name(family: &RefitFamily) -> &'static str {
    match family {
        RefitFamily::Gaussian => "Gaussian",
        RefitFamily::Binomial => "Binomial",
        RefitFamily::Poisson => "Poisson",
        RefitFamily::NegativeBinomial { .. } => "NBinomial",
        RefitFamily::Gamma => "GammaReg",
        RefitFamily::CoxPH { .. } => "CoxPH",
    }
}

pub(crate) fn fit(
    model: &Model,
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    family: &RefitFamily,
    control: &RefitControl,
) -> Result<RefitResult, MboostrsError> {
    let n = y.len();
    let mut y = y.to_owned();
    if matches!(family, RefitFamily::Binomial) && y.iter().any(|v| *v < 0.0) {
        // mboostrs codes a binary response -1/+1 internally.
        y.mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });
    }
    check_response(&y, family)?;
    let known_scale = family.known_scale();
    let solve = |z: &Array1<f64>, wz: &Array1<f64>, rho: &Array1<f64>| {
        let xtwx = weighted_gram(model.x.view(), wz.view());
        let wzz = wz * z;
        let xtwz = model.x.t().dot(&wzz);
        let zwz = wzz.dot(z);
        // The Gram form takes no n-row factor: the criterion is evaluated
        // from X'WX and X'Wz alone (see `RefitControl::gram`).
        let (data, root) = if control.gram {
            (None, None)
        } else {
            (
                Some((z.view(), wz.view())),
                Some(super::reml::SquareRoot::new(model, wz.view())),
            )
        };
        let problem = Problem {
            model,
            xtwx,
            xtwz,
            zwz,
            n,
            known_scale,
            data,
            root,
        };
        let optimum = optimize(&problem, rho, control).ok_or_else(|| {
            MboostrsError::Solver(
                "the penalized normal equations are singular at every smoothing parameter \
                 tried; a term may have basis columns with no data"
                    .into(),
            )
        })?;
        Ok::<_, MboostrsError>((optimum, problem.xtwx))
    };

    let mut rho = Array1::<f64>::zeros(model.penalties.len());
    let (optimum, xtwx, pirls_iterations, pirls_converged) = match family {
        RefitFamily::Gaussian => {
            let z = &y - &offset;
            let (opt, xtwx) = solve(&z, &w.to_owned(), &rho)?;
            (opt, xtwx, 1, true)
        }
        _ => {
            let mut eta = start_eta(&y, offset, family);
            let mut last = None;
            let mut converged = false;
            let mut iterations = 0;
            for _ in 0..control.max_pirls {
                iterations += 1;
                let (z, weight) = working(&y, &eta, offset, w, family);
                let wz = &weight * &w;
                let (opt, xtwx) = solve(&z, &wz, &rho)?;
                rho = opt.rho.clone();
                let new_eta = model.x.dot(&opt.eval.beta);
                let moved = (&new_eta - &eta).iter().fold(0.0f64, |a, v| a.max(v.abs()));
                let scale = new_eta.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                eta = new_eta;
                last = Some((opt, xtwx));
                if moved < control.pirls_tol * (1.0 + scale) {
                    converged = true;
                    break;
                }
            }
            let (opt, xtwx) = last.expect("at least one PIRLS iteration");
            (opt, xtwx, iterations, converged)
        }
    };

    // The performance iteration's fixed point is the answer for Gaussian
    // (one pass) and GammaReg (unknown scale). For the known-scale
    // likelihoods it is the start of the outer LAML iteration.
    let mut fin = Final {
        beta: optimum.eval.beta.clone(),
        ainv: optimum
            .eval
            .ainv
            .clone()
            .expect("the optimum is evaluated with derivatives"),
        xtwx,
        scale: if known_scale {
            1.0
        } else {
            optimum.eval.pen_rss / (n as f64 - model.null_dim as f64).max(1.0)
        },
        dispersion: if known_scale {
            1.0
        } else {
            optimum.eval.pen_rss / (n as f64 - model.null_dim as f64).max(1.0)
        },
        rho: optimum.rho.clone(),
        hessian: optimum.eval.hess.clone(),
        value: optimum.eval.value,
        gradient_norm: optimum.gradient_norm,
        iterations: optimum.iterations,
        converged: optimum.converged && pirls_converged,
        stop: if pirls_converged {
            optimum.stop
        } else {
            "pirls_iteration_limit"
        },
        method: "performance",
        theta: match family {
            RefitFamily::NegativeBinomial { theta } => Some(*theta),
            _ => None,
        },
    };
    let likelihood = match family {
        RefitFamily::Binomial => Some(Likelihood::Binomial),
        RefitFamily::Poisson | RefitFamily::CoxPH { .. } => Some(Likelihood::Poisson),
        RefitFamily::NegativeBinomial { theta } => Some(Likelihood::NegativeBinomial(*theta)),
        // The gamma shape starts at the inverse of the performance
        // iteration's scale estimate.
        RefitFamily::Gamma => Some(Likelihood::Gamma(1.0 / fin.scale.max(1e-12))),
        _ => None,
    };
    if let (true, Some(likelihood)) = (control.outer_iteration, likelihood) {
        fin = outer(model, &y, w, offset, family, likelihood, fin, control)?;
    }

    let cov = fin.ainv.mapv(|v| v * fin.scale);
    let cov_corrected = corrected_covariance(model, &fin, control.rho_bound);
    // edf: the diagonal of F = A^-1 X'WX, summed over each block.
    let p = fin.ainv.nrows();
    let influence_diag: Vec<f64> = (0..p)
        .map(|i| fin.ainv.row(i).dot(&fin.xtwx.column(i)))
        .collect();
    let edf_total: f64 = influence_diag.iter().sum();
    let residual_df = (!known_scale).then(|| (n as f64 - edf_total).max(1.0));
    let spans: Vec<(usize, usize)> = model.blocks.iter().map(|b| (b.start, b.end)).collect();
    let tests =
        super::term_test::block_tests(&spans, &fin.beta, &fin.ainv, &fin.xtwx, &cov, residual_df);
    // The Gram path takes the factor of the unweighted design from X'X, so
    // no serial n-row QR runs after the fit.
    let concurvities = if control.gram {
        let ones = Array1::<f64>::ones(n);
        let xtx = super::reml::weighted_gram(model.x.view(), ones.view());
        super::concurvity::concurvity_from_gram(xtx.view(), &spans, fin.beta.view()).unwrap_or_else(
            || super::concurvity::concurvity(model.x.view(), &spans, fin.beta.view()),
        )
    } else {
        super::concurvity::concurvity(model.x.view(), &spans, fin.beta.view())
    };
    let blocks = model
        .blocks
        .iter()
        .zip(tests)
        .zip(concurvities)
        .map(|((b, (_, test)), concurvity)| RefitBlock {
            name: b.name.clone(),
            start: b.full_start,
            end: b.full_end,
            householder: b.householder.clone(),
            has_by: b.has_by,
            rank: b.rank,
            edf: influence_diag[b.start..b.end].iter().sum(),
            lambda: b
                .penalty_ids
                .iter()
                .map(|&k| model.penalties[k].scale * fin.rho[k].exp())
                .collect(),
            test: Some(test),
            concurvity: Some(concurvity),
        })
        .collect();
    let (loglik, extra_df) = log_likelihood(model, &y, w, offset, family, &fin);
    // Columns a side constraint removed are reported with zero coefficient
    // and zero variance.
    let positions = model.full_positions();
    Ok(RefitResult {
        loglik,
        df: influence_diag.iter().sum::<f64>() + extra_df,
        blocks,
        coef: super::expand_vector(&fin.beta, &positions, model.full_p),
        cov: super::expand_matrix(&cov, &positions, model.full_p),
        cov_corrected: super::expand_matrix(&cov_corrected, &positions, model.full_p),
        scale: fin.dispersion,
        reml_score: fin.value,
        rho: fin.rho,
        rho_hessian: fin.hessian,
        gradient_norm: fin.gradient_norm,
        edf_total: influence_diag.iter().sum(),
        newton_iterations: fin.iterations,
        pirls_iterations,
        converged: fin.converged,
        stop_reason: fin.stop,
        unidentified_directions: model.unidentified,
        unidentified_terms: model.unidentified_terms.clone(),
        method: fin.method,
        theta: fin.theta,
        link: family.link(),
        n,
        residual_df,
    })
}

/// What the result is built from, whichever iteration produced it.
struct Final {
    beta: Array1<f64>,
    ainv: ndarray::Array2<f64>,
    xtwx: ndarray::Array2<f64>,
    /// Factor applied to `A^-1` for the covariance (the performance
    /// iteration's profiled scale; 1 when the likelihood carries it).
    scale: f64,
    /// Reported scale parameter: residual variance (Gaussian), gamma
    /// dispersion `1/shape`, or 1.
    dispersion: f64,
    rho: Array1<f64>,
    hessian: ndarray::Array2<f64>,
    value: f64,
    gradient_norm: f64,
    iterations: usize,
    converged: bool,
    stop: &'static str,
    method: &'static str,
    theta: Option<f64>,
}

/// The outer LAML iteration from the performance-iteration solution. For
/// Cox models the Breslow baseline is held fixed during each LAML fit (the
/// event indicator is then Poisson with offset `log H0`) and updated
/// between fits until the linear predictor stops moving.
#[allow(clippy::too_many_arguments)]
fn outer(
    model: &Model,
    y: &Array1<f64>,
    w: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    family: &RefitFamily,
    likelihood: Likelihood,
    start: Final,
    control: &RefitControl,
) -> Result<Final, MboostrsError> {
    let mut rho = start.rho.clone();
    let mut hessian = start.hessian.clone();
    let mut beta = start.beta.clone();
    let rounds = if matches!(family, RefitFamily::CoxPH { .. }) {
        control.max_pirls
    } else {
        1
    };
    let mut last = None;
    let mut settled = rounds == 1;
    let mut total_iterations = 0;
    for round in 0..rounds {
        let (prior, off) = outer_data(model, y, w, offset, family, &beta);
        let data = Data {
            model,
            y: y.view(),
            prior: prior.view(),
            offset: off.view(),
            likelihood,
        };
        let mut opt = laml::optimize(
            &|r, b| data.evaluate(r, b),
            &rho,
            &hessian,
            &beta,
            control.rho_bound,
            control.max_newton,
            control.newton_tol,
        )
        .ok_or_else(|| {
            MboostrsError::Solver(
                "the outer REML iteration could not evaluate the criterion; the penalized \
                 information matrix is singular"
                    .into(),
            )
        })?;
        // A smoothing parameter that ended at its upper bound (the term
        // reduced to its unpenalized part) may sit in the basin of that
        // limit while an interior minimum has a lower criterion: the REML
        // criterion is not convex in rho. Restart those coordinates from the
        // balanced value 0 and keep the lower criterion. The basin is chosen
        // once, in the first round; later Cox rounds only move the baseline.
        let at_bound: Vec<usize> = (0..opt.rho.len())
            .filter(|_| round == 0)
            .filter(|&k| opt.rho[k] >= control.rho_bound - 1e-6)
            .collect();
        if !at_bound.is_empty() {
            let mut restart = opt.rho.clone();
            for &k in &at_bound {
                restart[k] = 0.0;
            }
            let identity = ndarray::Array2::<f64>::eye(restart.len());
            if let Some(alt) = laml::optimize(
                &|r, b| data.evaluate(r, b),
                &restart,
                &identity,
                &opt.eval.beta,
                control.rho_bound,
                control.max_newton,
                control.newton_tol,
            ) {
                if alt.eval.value < opt.eval.value {
                    opt = alt;
                }
            }
        }
        total_iterations += opt.iterations;
        let moved = (&model.x.dot(&opt.eval.beta) - &model.x.dot(&beta))
            .iter()
            .fold(0.0f64, |a, v| a.max(v.abs()));
        rho = opt.rho.clone();
        hessian = opt.hessian.clone();
        beta = opt.eval.beta.clone();
        last = Some(opt);
        if rounds > 1
            && moved < control.pirls_tol * (1.0 + beta.iter().fold(0.0f64, |a, v| a.max(v.abs())))
        {
            settled = true;
            break;
        }
    }
    let mut opt = last.expect("at least one outer round");
    let mut likelihood = likelihood;
    // Negative binomial theta or gamma shape: estimated by the same
    // criterion, alternating a one-dimensional search over its logarithm at
    // fixed smoothing parameters with the smoothing-parameter optimization at
    // fixed dispersion (mgcv's nb() estimates theta jointly with them). The
    // gamma shape is always estimated; theta when `estimate_theta`.
    let estimate = match likelihood {
        Likelihood::Gamma(_) => true,
        Likelihood::NegativeBinomial(_) => control.estimate_theta,
        _ => false,
    };
    if let (Some(theta0), true) = (likelihood.dispersion(), estimate) {
        let (prior, off) = outer_data(model, y, w, offset, family, &opt.eval.beta);
        let mut log_theta = theta0.ln();
        for _ in 0..30 {
            let at = |lt: f64| {
                Data {
                    model,
                    y: y.view(),
                    prior: prior.view(),
                    offset: off.view(),
                    likelihood: likelihood.with_dispersion(lt.exp()),
                }
                .evaluate(&opt.rho, &opt.eval.beta)
                .map_or(f64::INFINITY, |e| e.value)
            };
            let next =
                crate::families::minimize_brent(&at, log_theta - 3.0, log_theta + 3.0, 1e-7, 200);
            let data = Data {
                model,
                y: y.view(),
                prior: prior.view(),
                offset: off.view(),
                likelihood: likelihood.with_dispersion(next.exp()),
            };
            if let Some(refined) = laml::optimize(
                &|r, b| data.evaluate(r, b),
                &opt.rho,
                &opt.hessian,
                &opt.eval.beta,
                control.rho_bound,
                control.max_newton,
                control.newton_tol,
            ) {
                total_iterations += refined.iterations;
                opt = refined;
            }
            let moved = (next - log_theta).abs();
            log_theta = next;
            if moved < 1e-6 {
                break;
            }
        }
        likelihood = likelihood.with_dispersion(log_theta.exp());
    }
    // The exact Hessian at the optimum (BFGS only approximates it), for the
    // smoothing-parameter-uncertainty correction of the covariance.
    let (prior, off) = outer_data(model, y, w, offset, family, &opt.eval.beta);
    let data = Data {
        model,
        y: y.view(),
        prior: prior.view(),
        offset: off.view(),
        likelihood,
    };
    let hessian = data
        .hessian_fd(&opt.rho, &opt.eval.beta)
        .unwrap_or_else(|| opt.hessian.clone());
    Ok(Final {
        beta: opt.eval.beta,
        ainv: opt.eval.ainv,
        xtwx: opt.eval.xtwx,
        scale: 1.0,
        dispersion: match likelihood {
            Likelihood::Gamma(shape) => 1.0 / shape,
            _ => 1.0,
        },
        rho: opt.rho,
        hessian,
        value: opt.eval.value,
        gradient_norm: opt.gradient_norm,
        iterations: total_iterations,
        converged: opt.converged && settled,
        stop: if settled {
            opt.stop
        } else {
            "outer_iteration_limit"
        },
        method: "outer",
        theta: match likelihood {
            Likelihood::NegativeBinomial(theta) => Some(theta),
            _ => None,
        },
    })
}

/// Log-likelihood of the fitted model (unpenalized) and the number of
/// estimated parameters it adds beyond the coefficients' effective degrees
/// of freedom (a scale, theta or gamma shape).
///
/// As in mgcv's `logLik.gam`: Gaussian uses the maximum-likelihood variance
/// `RSS / n`; Cox uses the partial likelihood with Breslow's treatment of
/// ties; GammaReg uses the refit's estimated shape.
fn log_likelihood(
    model: &Model,
    y: &Array1<f64>,
    w: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    family: &RefitFamily,
    fin: &Final,
) -> (f64, f64) {
    let eta = model.x.dot(&fin.beta) + offset;
    let n = y.len();
    let lgamma = libm::lgamma;
    match family {
        RefitFamily::Gaussian => {
            let rss: f64 = (0..n).map(|i| w[i] * (y[i] - eta[i]).powi(2)).sum();
            let nn = n as f64;
            (
                -0.5 * nn * ((2.0 * std::f64::consts::PI * rss / nn).ln() + 1.0),
                1.0,
            )
        }
        RefitFamily::Binomial => {
            let l = (0..n)
                .map(|i| {
                    let e = eta[i];
                    let softplus = if e > 0.0 {
                        e + (-e).exp().ln_1p()
                    } else {
                        e.exp().ln_1p()
                    };
                    w[i] * (y[i] * e - softplus)
                })
                .sum();
            (l, 0.0)
        }
        RefitFamily::Poisson => {
            let l = (0..n)
                .map(|i| w[i] * (y[i] * eta[i] - eta[i].exp() - lgamma(y[i] + 1.0)))
                .sum();
            (l, 0.0)
        }
        RefitFamily::NegativeBinomial { .. } => {
            let theta = fin.theta.unwrap_or(1.0);
            let l = (0..n)
                .map(|i| {
                    let mu = eta[i].exp();
                    w[i] * (lgamma(y[i] + theta) - lgamma(theta) - lgamma(y[i] + 1.0)
                        + theta * theta.ln()
                        + y[i] * eta[i]
                        - (y[i] + theta) * (mu + theta).ln())
                })
                .sum();
            (l, 1.0)
        }
        RefitFamily::Gamma => {
            let shape = 1.0 / fin.dispersion.max(1e-300);
            let l = (0..n)
                .map(|i| {
                    w[i] * (shape * shape.ln() - lgamma(shape) + (shape - 1.0) * y[i].ln()
                        - shape * eta[i]
                        - shape * y[i] * (-eta[i]).exp())
                })
                .sum();
            (l, 1.0)
        }
        RefitFamily::CoxPH { time } => {
            // Breslow partial likelihood: sum over events of
            // eta_i - log sum_{j: t_j >= t_i} w_j exp(eta_j).
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                time[a]
                    .partial_cmp(&time[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut at_risk = vec![0.0; n];
            let mut acc = 0.0;
            for pos in (0..n).rev() {
                acc += w[order[pos]] * eta[order[pos]].exp();
                at_risk[pos] = acc;
            }
            let mut l = 0.0;
            let mut pos = 0;
            while pos < n {
                let mut end = pos;
                while end + 1 < n && time[order[end + 1]] == time[order[pos]] {
                    end += 1;
                }
                for &row in &order[pos..=end] {
                    if y[row] > 0.0 {
                        l += w[row] * (eta[row] - at_risk[pos].ln());
                    }
                }
                pos = end + 1;
            }
            (l, 0.0)
        }
    }
}

/// Prior weights and offset of the LAML fit. For Cox the event indicator is
/// Poisson with offset `log H0` at the Breslow baseline of `beta`; rows whose
/// time precedes every event have `H0 = 0` and get weight 0.
fn outer_data(
    model: &Model,
    y: &Array1<f64>,
    w: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    family: &RefitFamily,
    beta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    match family {
        RefitFamily::CoxPH { time } => {
            let eta = model.x.dot(beta) + offset;
            let h0 = breslow_cumulative_hazard(time.view(), y.view(), eta.view(), w);
            let prior = Array1::from_shape_fn(y.len(), |i| if h0[i] > 0.0 { w[i] } else { 0.0 });
            let off = Array1::from_shape_fn(y.len(), |i| {
                if h0[i] > 0.0 {
                    offset[i] + h0[i].ln()
                } else {
                    0.0
                }
            });
            (prior, off)
        }
        _ => (w.to_owned(), offset.to_owned()),
    }
}

/// The covariance corrected for smoothing-parameter uncertainty, to first
/// order (Wood, Pya and Saefken 2016, section 3.1):
/// `Vc = Vp + J V_rho J'`, `J_k = d beta / d rho_k = -lambda_k A^-1 S_k beta`,
/// `V_rho = 2 H^-1` with `H` the Hessian of the criterion (-2 x the log
/// restricted likelihood). Smoothing parameters at a bound, or along
/// directions where the criterion is flat, are treated as known, as mgcv
/// does; mgcv's second-order term is not included.
fn corrected_covariance(model: &Model, fin: &Final, bound: f64) -> ndarray::Array2<f64> {
    let vp = fin.ainv.mapv(|v| v * fin.scale);
    let m = fin.rho.len();
    let free: Vec<usize> = (0..m)
        .filter(|&k| fin.rho[k].abs() < bound - 1e-6)
        .collect();
    if free.is_empty() || fin.hessian.dim() != (m, m) {
        return vp;
    }
    let f = free.len();
    let h = ndarray::Array2::from_shape_fn((f, f), |(a, b)| fin.hessian[[free[a], free[b]]]);
    let (vals, vecs) = super::reml::symmetric_eigen(&h);
    let top = vals.iter().fold(0.0f64, |a, v| a.max(*v));
    if top.is_nan() || top <= 0.0 {
        return vp;
    }
    // V_rho = 2 H^-1 on the directions with clearly positive curvature.
    let inv = vals.mapv(|v| if v > 1e-7 * top { 2.0 / v } else { 0.0 });
    let scaled = &vecs * &inv.view().insert_axis(ndarray::Axis(0));
    let v_rho = scaled.dot(&vecs.t());
    let p = fin.beta.len();
    let mut j = ndarray::Array2::<f64>::zeros((p, f));
    for (col, &k) in free.iter().enumerate() {
        let pen = &model.penalties[k];
        let block = &model.blocks[pen.block];
        let r = block.start..block.end;
        let sb = pen.matrix.dot(&fin.beta.slice(ndarray::s![r.clone()])) * fin.rho[k].exp();
        let dbeta = fin.ainv.slice(ndarray::s![.., r]).dot(&sb).mapv(|v| -v);
        j.column_mut(col).assign(&dbeta);
    }
    vp + j.dot(&v_rho).dot(&j.t())
}

/// Starting linear predictor (excluding the offset).
fn start_eta(y: &Array1<f64>, offset: ArrayView1<f64>, family: &RefitFamily) -> Array1<f64> {
    match family {
        RefitFamily::Binomial => y
            .iter()
            .zip(offset.iter())
            .map(|(&v, &o)| {
                let mu = (v + 0.5) / 2.0;
                (mu / (1.0 - mu)).ln() - o
            })
            .collect(),
        RefitFamily::CoxPH { .. } => Array1::zeros(y.len()),
        _ => {
            let mean = y.mean().unwrap_or(1.0);
            let floor = 0.1 * mean.max(1e-8);
            y.iter()
                .zip(offset.iter())
                .map(|(&v, &o)| v.max(floor).ln() - o)
                .collect()
        }
    }
}

/// Working response and working weights (without the prior weights).
fn working(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    offset: ArrayView1<f64>,
    w: ArrayView1<f64>,
    family: &RefitFamily,
) -> (Array1<f64>, Array1<f64>) {
    let n = y.len();
    let mut z = Array1::<f64>::zeros(n);
    let mut weight = Array1::<f64>::zeros(n);
    match family {
        RefitFamily::Binomial => {
            for i in 0..n {
                let mu = 1.0 / (1.0 + (-(eta[i] + offset[i])).exp());
                let v = (mu * (1.0 - mu)).max(1e-10);
                weight[i] = v;
                z[i] = eta[i] + (y[i] - mu) / v;
            }
        }
        RefitFamily::Poisson => {
            for i in 0..n {
                let mu = (eta[i] + offset[i]).exp();
                weight[i] = mu;
                z[i] = eta[i] + (y[i] - mu) / mu;
            }
        }
        RefitFamily::NegativeBinomial { theta } => {
            for i in 0..n {
                let mu = (eta[i] + offset[i]).exp();
                weight[i] = mu / (1.0 + mu / theta);
                z[i] = eta[i] + (y[i] - mu) / mu;
            }
        }
        RefitFamily::Gamma => {
            for i in 0..n {
                let mu = (eta[i] + offset[i]).exp();
                weight[i] = 1.0;
                z[i] = eta[i] + (y[i] - mu) / mu;
            }
        }
        RefitFamily::CoxPH { time } => {
            let full: Array1<f64> = eta + &offset;
            let h0 = breslow_cumulative_hazard(time.view(), y.view(), full.view(), w);
            for i in 0..n {
                // A row whose time precedes every event has H0 = 0 and no
                // weight.
                let mu = h0[i] * full[i].exp();
                weight[i] = mu;
                z[i] = if mu > 0.0 {
                    eta[i] + (y[i] - mu) / mu
                } else {
                    eta[i]
                };
            }
        }
        RefitFamily::Gaussian => unreachable!("the Gaussian refit is not iterated"),
    }
    (z, weight)
}
