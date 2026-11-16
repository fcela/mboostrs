use super::reml::{weighted_gram, Problem};
use super::*;
use crate::learners::{bspline_basis, bspline_knots, diff_penalty};
use ndarray::Array1;

/// A deterministic pseudo-random sequence in [0, 1).
fn uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        })
        .collect()
}

fn normal(n: usize, seed: u64) -> Vec<f64> {
    let a = uniform(n, seed);
    let b = uniform(n, seed + 7);
    a.iter()
        .zip(&b)
        .map(|(u, v)| (-2.0 * (u.max(1e-12)).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos())
        .collect()
}

fn bbs_term(name: &str, x: &[f64], inner: usize) -> RefitTerm {
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let knots = bspline_knots((lo, hi), inner, 3);
    let design = bspline_basis(x, &knots, 3);
    let q = design.ncols();
    RefitTerm {
        name: name.into(),
        design,
        penalties: vec![diff_penalty(q, 2)],
        by: None,
        variables: vec![name.into()],
    }
}

fn tensor_term(x1: &[f64], x2: &[f64], inner: usize) -> RefitTerm {
    let a = bbs_term("a", x1, inner);
    let b = bbs_term("b", x2, inner);
    let (qa, qb) = (a.design.ncols(), b.design.ncols());
    let n = x1.len();
    let mut design = Array2::<f64>::zeros((n, qa * qb));
    for i in 0..n {
        for j in 0..qa {
            for k in 0..qb {
                design[[i, j * qb + k]] = a.design[[i, j]] * b.design[[i, k]];
            }
        }
    }
    let kron = |left: &Array2<f64>, right: &Array2<f64>| {
        let (r1, c1) = left.dim();
        let (r2, c2) = right.dim();
        Array2::from_shape_fn((r1 * r2, c1 * c2), |(i, j)| {
            left[[i / r2, j / c2]] * right[[i % r2, j % c2]]
        })
    };
    let pa = kron(&a.penalties[0], &Array2::eye(qb));
    let pb = kron(&Array2::eye(qa), &b.penalties[0]);
    RefitTerm {
        name: "te(x1,x2)".into(),
        design,
        penalties: vec![pa, pb],
        by: None,
        variables: vec!["x1".into(), "x2".into()],
    }
}

fn gaussian_data(n: usize) -> (Vec<f64>, Vec<f64>, Array1<f64>) {
    let x1 = uniform(n, 1);
    let x2 = uniform(n, 2);
    let e = normal(n, 3);
    let y = Array1::from_shape_fn(n, |i| {
        (2.0 * std::f64::consts::PI * x1[i]).sin() + 0.5 * x2[i] * x2[i] + 0.3 * e[i]
    });
    (x1, x2, y)
}

fn problem_for<'a>(model: &'a Model, y: &Array1<f64>, known_scale: bool) -> Problem<'a> {
    let w = Array1::<f64>::ones(y.len());
    let xtwx = weighted_gram(model.x.view(), w.view());
    let xtwz = model.x.t().dot(y);
    Problem {
        model,
        xtwx,
        xtwz,
        zwz: y.dot(y),
        n: y.len(),
        known_scale,
        data: None,
        root: None,
    }
}

fn assert_derivatives(problem: &Problem<'_>, rho: &Array1<f64>) {
    let e = problem.evaluate(rho, true).expect("evaluable");
    let m = rho.len();
    let h = 1e-5;
    for k in 0..m {
        let mut up = rho.clone();
        let mut down = rho.clone();
        up[k] += h;
        down[k] -= h;
        let eu = problem.evaluate(&up, true).unwrap();
        let ed = problem.evaluate(&down, true).unwrap();
        let fd = (eu.value - ed.value) / (2.0 * h);
        assert!(
            (fd - e.grad[k]).abs() < 1e-5 * (1.0 + fd.abs()),
            "gradient {k}: analytic {} vs finite difference {fd}",
            e.grad[k]
        );
        for j in 0..m {
            let fdh = (eu.grad[j] - ed.grad[j]) / (2.0 * h);
            assert!(
                (fdh - e.hess[[k, j]]).abs() < 1e-4 * (1.0 + fdh.abs()),
                "hessian ({k},{j}): analytic {} vs finite difference {fdh}",
                e.hess[[k, j]]
            );
        }
    }
}

#[test]
fn reml_derivatives_match_finite_differences() {
    let n = 300;
    let (x1, x2, y) = gaussian_data(n);
    // The tensor's unpenalized space contains the linear functions of x1 and
    // x2, so a separate smooth of either would make A singular; x3 is new.
    let x3 = uniform(n, 9);
    let terms = vec![tensor_term(&x1, &x2, 5), bbs_term("x3", &x3, 8)];
    let w = Array1::<f64>::ones(n);
    let model = build_model(terms, w.view(), n).unwrap();
    assert_eq!(model.penalties.len(), 3);
    let rho = Array1::from(vec![0.3, -1.2, 2.0]);
    assert_derivatives(&problem_for(&model, &y, false), &rho);
    assert_derivatives(&problem_for(&model, &y, true), &rho);
}

#[test]
fn newton_finds_a_stationary_minimum() {
    let n = 400;
    let (x1, x2, y) = gaussian_data(n);
    let terms = vec![bbs_term("x1", &x1, 20), bbs_term("x2", &x2, 20)];
    let fit = try_gam_refit(
        terms,
        y.view(),
        None,
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .unwrap();
    assert!(fit.converged, "gradient norm {}", fit.gradient_norm);
    // A sine over one period needs several edf; the quadratic about two.
    assert!(fit.blocks[0].edf > 4.0, "edf x1 {}", fit.blocks[0].edf);
    assert!(fit.blocks[1].edf < 3.5, "edf x2 {}", fit.blocks[1].edf);
    // The criterion does not decrease when any rho moves.
    let w = Array1::<f64>::ones(n);
    let model = build_model(
        vec![bbs_term("x1", &x1, 20), bbs_term("x2", &x2, 20)],
        w.view(),
        n,
    )
    .unwrap();
    let problem = problem_for(&model, &y, false);
    let best = problem.evaluate(&fit.rho, false).unwrap().value;
    for k in 0..fit.rho.len() {
        for delta in [-0.2, 0.2] {
            let mut r = fit.rho.clone();
            r[k] += delta;
            assert!(problem.evaluate(&r, false).unwrap().value >= best - 1e-9);
        }
    }
    // Residual variance close to the simulated 0.09.
    assert!((fit.scale - 0.09).abs() < 0.03, "scale {}", fit.scale);
}

#[test]
fn every_block_sums_to_zero_over_the_weighted_rows() {
    let n = 200;
    let (x1, x2, y) = gaussian_data(n);
    let w = Array1::from_shape_fn(n, |i| 0.5 + (i % 3) as f64);
    let terms = vec![bbs_term("x1", &x1, 10), bbs_term("x2", &x2, 10)];
    let raw: Vec<Array2<f64>> = terms.iter().map(|t| t.design.clone()).collect();
    let fit = try_gam_refit(
        terms,
        y.view(),
        Some(w.view()),
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .unwrap();
    for (b, raw) in raw.iter().enumerate() {
        let (effect, se) = fit.effect(b, raw.view(), None, false).unwrap();
        assert!(
            w.dot(&effect).abs() < 1e-8,
            "block {b} weighted sum {}",
            w.dot(&effect)
        );
        assert!(se.iter().all(|v| *v > 0.0));
    }
}

#[test]
fn breslow_ties_share_the_first_risk_set() {
    let time = Array1::from(vec![1.0, 2.0, 2.0, 3.0]);
    let status = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
    let eta = Array1::zeros(4);
    let w = Array1::ones(4);
    let h = breslow_cumulative_hazard(time.view(), status.view(), eta.view(), w.view());
    // t=1: 1/4. t=2: two events, each 1/3 (risk set of the first tie).
    let expected = [0.25, 0.25 + 2.0 / 3.0, 0.25 + 2.0 / 3.0, 0.25 + 2.0 / 3.0];
    for (a, b) in h.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

#[test]
fn poisson_refit_recovers_a_log_rate_curve() {
    let n = 600;
    let x = uniform(n, 11);
    let u = uniform(n, 12);
    // Poisson draws by inversion.
    let y = Array1::from_shape_fn(n, |i| {
        let mu = (1.0 + (2.0 * std::f64::consts::PI * x[i]).sin()).exp();
        let (mut k, mut p, mut cdf) = (0.0, (-mu).exp(), (-mu).exp());
        while u[i] > cdf && k < 200.0 {
            k += 1.0;
            p *= mu / k;
            cdf += p;
        }
        k
    });
    let term = bbs_term("x", &x, 20);
    let raw = term.design.clone();
    let fit = try_gam_refit(
        vec![term],
        y.view(),
        None,
        None,
        &RefitFamily::Poisson,
        &RefitControl::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert_eq!(fit.scale, 1.0);
    let (effect, _) = fit.effect(0, raw.view(), None, false).unwrap();
    let truth: Vec<f64> = x
        .iter()
        .map(|v| (2.0 * std::f64::consts::PI * v).sin())
        .collect();
    let mean = truth.iter().sum::<f64>() / n as f64;
    let rmse = (effect
        .iter()
        .zip(&truth)
        .map(|(e, t)| (e - (t - mean)).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert!(rmse < 0.15, "rmse {rmse}");
}

#[test]
fn reml_derivatives_match_for_a_block_with_three_penalties() {
    // Three penalties on one block take the general Cholesky route for
    // log|S|+ rather than the two-penalty diagonalization.
    let n = 300;
    let (x1, x2, y) = gaussian_data(n);
    let mut term = tensor_term(&x1, &x2, 4);
    let q = term.design.ncols();
    term.penalties.push(Array2::eye(q));
    let w = Array1::<f64>::ones(n);
    let model = build_model(vec![term], w.view(), n).unwrap();
    assert_eq!(model.penalties.len(), 3);
    assert!(model.blocks[0].pair.is_none());
    let rho = Array1::from(vec![0.5, -0.7, -3.0]);
    assert_derivatives(&problem_for(&model, &y, false), &rho);
}

#[test]
fn two_penalty_diagonalization_matches_the_direct_log_determinant() {
    let n = 200;
    let (x1, x2, _) = gaussian_data(n);
    let w = Array1::<f64>::ones(n);
    let model = build_model(vec![tensor_term(&x1, &x2, 4)], w.view(), n).unwrap();
    let pair = model.blocks[0].pair.as_ref().unwrap();
    let (p1, p2) = (
        model.penalties[0].projected.as_ref().unwrap(),
        model.penalties[1].projected.as_ref().unwrap(),
    );
    for (l1, l2) in [(1.0, 1.0), (1e-4, 30.0), (1e6, 1e-3)] {
        let s = p1 * l1 + p2 * l2;
        let (vals, _) = super::reml::symmetric_eigen(&s);
        let direct: f64 = vals.iter().map(|v| v.ln()).sum();
        let closed: f64 = pair.logdet_t
            + pair
                .mu
                .iter()
                .map(|m| (l1 * m + l2 * (1.0 - m)).ln())
                .sum::<f64>();
        assert!(
            (direct - closed).abs() < 1e-6 * (1.0 + direct.abs()),
            "{direct} vs {closed}"
        );
    }
}

#[test]
fn term_basis_rebuilds_the_fitted_design_and_penalties() {
    let n = 150;
    let (x1, x2, y) = gaussian_data(n);
    let (a, b) = (Array1::from(x1.clone()), Array1::from(x2.clone()));
    let bbs = TermBasis::Bbs {
        boundary: (0.0, 1.0),
        n_knots: 6,
        degree: 3,
        differences: 2,
        cyclic: false,
    };
    let tensor = TermBasis::Tensor(Box::new(bbs.clone()), Box::new(bbs.clone()));
    let design = tensor.design(&[a.view(), b.view()], false).unwrap();
    assert_eq!(design.ncols(), tensor.width());
    assert_eq!(tensor.penalties().len(), 2);
    // Rows of a B-spline basis sum to one inside the boundary; so do the
    // tensor's.
    for row in design.rows() {
        assert!((row.sum() - 1.0).abs() < 1e-10);
    }
    let group = TermBasis::Brandom {
        levels: vec![1.0, 2.0, 3.0],
    };
    let g = Array1::from(vec![1.0, 3.0, 7.0]);
    assert!(group.design(&[g.view()], false).is_err());
    let zeroed = group.design(&[g.view()], true).unwrap();
    assert_eq!(zeroed.row(2).sum(), 0.0);
    // A fit on the recipe's design gives the same effect through
    // `block_effect` as through `RefitResult::effect`.
    let term = RefitTerm {
        name: "t".into(),
        design: design.clone(),
        penalties: tensor.penalties(),
        by: None,
        variables: Vec::new(),
    };
    let fit = try_gam_refit(
        vec![term],
        y.view(),
        None,
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .unwrap();
    let block = &fit.blocks[0];
    let r = block.start..block.end;
    let (e1, s1) = fit.effect(0, design.view(), None, false).unwrap();
    let (e2, s2) = block_effect(
        design.view(),
        block.householder.view(),
        fit.coef.slice(s![r.clone()]),
        fit.cov.slice(s![r.clone(), r]),
        false,
        None,
        false,
    )
    .unwrap();
    assert_eq!(e1, e2);
    assert_eq!(s1, s2);
}

#[test]
fn laml_gradient_matches_finite_differences() {
    use super::laml::{Data, Likelihood};
    let n = 400;
    let x = uniform(n, 31);
    let u = uniform(n, 32);
    let x2 = uniform(n, 33);
    let w = Array1::<f64>::ones(n);
    for likelihood in [
        Likelihood::Binomial,
        Likelihood::Poisson,
        Likelihood::NegativeBinomial(2.5),
        Likelihood::Gamma(4.0),
    ] {
        let y = Array1::from_shape_fn(n, |i| {
            let eta = (2.0 * std::f64::consts::PI * x[i]).sin() + 0.5 * x2[i];
            match likelihood {
                Likelihood::Binomial => f64::from(u[i] < 1.0 / (1.0 + (-eta).exp())),
                Likelihood::Gamma(_) => eta.exp() * (0.3 + 1.4 * u[i]),
                _ => (eta.exp() * (0.5 + u[i]) * 1.5).floor(),
            }
        });
        let model = build_model(
            vec![bbs_term("x", &x, 10), bbs_term("x2", &x2, 8)],
            w.view(),
            n,
        )
        .unwrap();
        let offset = Array1::<f64>::zeros(n);
        let data = Data {
            model: &model,
            y: y.view(),
            prior: w.view(),
            offset: offset.view(),
            likelihood,
        };
        let beta0 = Array1::<f64>::zeros(model.x.ncols());
        let rho = Array1::from(vec![0.4, 1.5]);
        let e = data.evaluate(&rho, &beta0).unwrap();
        for k in 0..rho.len() {
            let h = 1e-5;
            let mut up = rho.clone();
            let mut down = rho.clone();
            up[k] += h;
            down[k] -= h;
            let vu = data.evaluate(&up, &e.beta).unwrap().value;
            let vd = data.evaluate(&down, &e.beta).unwrap().value;
            let fd = (vu - vd) / (2.0 * h);
            assert!(
                (fd - e.grad[k]).abs() < 1e-4 * (1.0 + fd.abs()),
                "{likelihood:?} gradient {k}: analytic {} vs finite difference {fd}",
                e.grad[k]
            );
        }
    }
}

#[test]
fn refit_results_round_trip_through_json() {
    let n = 200;
    let (x1, _, y) = gaussian_data(n);
    let term = bbs_term("x1", &x1, 10);
    let raw = term.design.clone();
    let fit = try_gam_refit(
        vec![term],
        y.view(),
        None,
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .unwrap();
    let text = fit.to_json().to_string();
    let back = RefitResult::from_json(&serde_json::from_str(&text).unwrap()).unwrap();
    assert_eq!(back.coef, fit.coef);
    assert_eq!(back.cov, fit.cov);
    assert_eq!(back.blocks[0].householder, fit.blocks[0].householder);
    assert_eq!(back.method, fit.method);
    assert_eq!(
        back.effect(0, raw.view(), None, false).unwrap(),
        fit.effect(0, raw.view(), None, false).unwrap()
    );
}

#[test]
fn a_tensor_term_loses_the_columns_its_main_effects_reproduce() {
    // bbs(x1) + bbs(x2) + te(x1, x2): cubic B-splines on any knots reproduce
    // cubic polynomials, and the tensor basis contains every product of its
    // margins, so after centring it repeats x, x^2 and x^3 of each main
    // effect. Without the side constraint the design has six dependent
    // columns and the refit is not identified.
    let n = 500;
    let x1 = uniform(n, 21);
    let x2 = uniform(n, 22);
    let e = normal(n, 23);
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * x1[i]).sin() + 1.5 * x1[i] * x2[i])
        .collect();
    let y = Array1::from_shape_fn(n, |i| signal[i] + 0.3 * e[i]);
    let terms = vec![
        bbs_term("x1", &x1, 10),
        bbs_term("x2", &x2, 10),
        tensor_term(&x1, &x2, 4),
    ];
    let raws: Vec<Array2<f64>> = terms.iter().map(|t| t.design.clone()).collect();
    let w = Array1::<f64>::ones(n);
    let model = build_model(terms.clone(), w.view(), n).unwrap();
    assert!(model.blocks[0].kept.is_none() && model.blocks[1].kept.is_none());
    let tensor = &model.blocks[2];
    let full = tensor.full_end - tensor.full_start;
    assert_eq!(full - tensor.kept.as_ref().unwrap().len(), 6);

    let fit = try_gam_refit(
        terms,
        y.view(),
        None,
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .unwrap();
    assert!(fit.converged);
    assert_eq!(fit.coef.len(), model.full_p);
    for block in &fit.blocks {
        assert!(block.edf > 0.5, "{} has edf {}", block.name, block.edf);
    }
    // The summed effects reproduce the signal up to its mean.
    let mut total = vec![0.0; n];
    for (b, raw) in raws.iter().enumerate() {
        let (effect, _) = fit.effect(b, raw.view(), None, false).unwrap();
        for i in 0..n {
            total[i] += effect[i];
        }
    }
    let mean_signal = signal.iter().sum::<f64>() / n as f64;
    let mean_total = total.iter().sum::<f64>() / n as f64;
    let rmse = ((0..n)
        .map(|i| ((total[i] - mean_total) - (signal[i] - mean_signal)).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert!(rmse < 0.1, "rmse {rmse}");
}

/// Smooths of two columns with identical values have the same linear part,
/// so `A` is singular along it at every smoothing parameter. The refit holds
/// that direction at zero (mgcv drops it) instead of failing, and the fitted
/// values match the refit with one copy of the column.
#[test]
fn identical_columns_in_two_terms_are_identified_by_a_fixed_penalty() {
    let n = 400;
    let (x1, x2, y) = gaussian_data(n);
    let twin = x1.clone();
    let both = try_gam_refit(
        vec![
            bbs_term("x1", &x1, 20),
            bbs_term("twin", &twin, 20),
            bbs_term("x2", &x2, 20),
        ],
        y.view(),
        None,
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .expect("a duplicated column must not make the refit fail");
    assert_eq!(both.unidentified_directions, 1);
    // The duplicated pair is named; the third term is not in the dependency.
    assert_eq!(
        both.unidentified_terms,
        vec!["x1".to_string(), "twin".to_string()]
    );
    let single = try_gam_refit(
        vec![bbs_term("x1", &x1, 20), bbs_term("x2", &x2, 20)],
        y.view(),
        None,
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .unwrap();
    assert_eq!(single.unidentified_directions, 0);
    assert_eq!(single.stop_reason, "gradient");
    // The two copies share the smooth between them; the fit is the same
    // function of the data up to the optimizer's tolerance.
    let rss = |fit: &RefitResult| fit.scale;
    assert!(
        (rss(&both) - rss(&single)).abs() < 1e-3 * rss(&single),
        "scale {} vs {}",
        rss(&both),
        rss(&single)
    );
}

/// The penalized residual sum of squares from residuals agrees with the
/// sufficient-statistics form where the latter is accurate.
#[test]
fn penalized_rss_from_residuals_matches_sufficient_statistics() {
    let n = 300;
    let (x1, x2, y) = gaussian_data(n);
    let w = Array1::<f64>::ones(n);
    let model = build_model(
        vec![bbs_term("x1", &x1, 12), bbs_term("x2", &x2, 12)],
        w.view(),
        n,
    )
    .unwrap();
    let rho = Array1::from(vec![0.5, -1.0]);
    let stats = problem_for(&model, &y, false);
    let mut direct = problem_for(&model, &y, false);
    direct.data = Some((y.view(), w.view()));
    let a = stats.evaluate(&rho, true).unwrap();
    let b = direct.evaluate(&rho, true).unwrap();
    assert!((a.pen_rss - b.pen_rss).abs() < 1e-9 * a.pen_rss);
    assert!((a.value - b.value).abs() < 1e-8 * (1.0 + a.value.abs()));
}

/// Naval-propulsion-like data: twelve predictors that are near-deterministic
/// functions of one operating variable with nine distinct values, and a
/// response fitted almost exactly. The REML surface is nearly degenerate in
/// the smoothing parameters of the redundant terms, so Newton may stop
/// without meeting the gradient test (mgcv reports a step failure in the
/// same situation). The refit must still return, recover the noise level,
/// and name why it stopped.
#[test]
fn reml_on_concurved_low_noise_data_returns_and_reports_its_stop() {
    let n = 3000;
    let u = uniform(n, 11);
    let e = normal(n, 12);
    let lever: Vec<f64> = u.iter().map(|v| (v * 9.0).floor() / 8.0).collect();
    let shapes: [fn(f64) -> f64; 12] = [
        |l| l,
        |l| l * l,
        |l| (2.0 * l).exp(),
        |l| (3.0 * l).sin(),
        |l| (l + 0.1).sqrt(),
        |l| (1.0 + l).ln(),
        |l| l * l * l,
        |l| (1.5 * l).cos(),
        |l| 1.0 / (1.0 + l),
        |l| (l - 0.5).abs(),
        |l| (4.0 * l).tanh(),
        |l| l.powf(1.5),
    ];
    let mut terms = Vec::new();
    for (k, f) in shapes.iter().enumerate() {
        let jitter = normal(n, 30 + k as u64);
        let x: Vec<f64> = lever
            .iter()
            .zip(&jitter)
            .map(|(l, j)| f(*l) + 1e-5 * j)
            .collect();
        terms.push(bbs_term(&format!("x{k}"), &x, 20));
    }
    let y = Array1::from_shape_fn(n, |i| 0.95 + 0.05 * lever[i] * lever[i] + 1e-5 * e[i]);
    let fit = try_gam_refit(
        terms,
        y.view(),
        None,
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .expect("the refit must return");
    assert!(fit.scale.sqrt() < 3e-5, "sigma {}", fit.scale.sqrt());
    if fit.converged {
        assert!(["gradient", "stationary", "flat"].contains(&fit.stop_reason));
    } else {
        assert_eq!(fit.stop_reason, "no_descent");
    }
}

#[test]
fn numeric_by_retains_the_constant_coefficient_direction() {
    let n = 200;
    let x = uniform(n, 81);
    let by = Array1::from(uniform(n, 82)).mapv(|g| 0.5 + 1.5 * g);
    let term = RefitTerm {
        name: "x_by_g".into(),
        design: Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] }),
        penalties: vec![],
        by: Some(by.clone()),
        variables: vec!["x".into(), "g".into()],
    };
    let y = Array1::from_shape_fn(n, |i| 0.4 + (2.0 + 3.0 * x[i]) * by[i]);
    let w = Array1::from_shape_fn(n, |i| 0.5 + (i % 5) as f64);
    let model = build_model(vec![term.clone()], w.view(), n).unwrap();
    assert_eq!(model.x.ncols(), 3); // intercept, g, x*g
    let raw = term.design.clone();
    let fit = try_gam_refit(
        vec![term],
        y.view(),
        Some(w.view()),
        None,
        &RefitFamily::Gaussian,
        &RefitControl::default(),
    )
    .unwrap();
    let (effect, _) = fit.effect(0, raw.view(), Some(by.view()), false).unwrap();
    for i in 0..n {
        assert!((fit.coef[0] + effect[i] - y[i]).abs() < 1e-7);
    }
}

#[test]
fn numeric_by_checks_overlap_with_the_global_intercept() {
    let n = 100;
    let x = uniform(n, 91);
    let w = Array1::ones(n);
    let term = RefitTerm {
        name: "x_by_inverse".into(),
        design: Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { 1.0 + x[i] }),
        penalties: vec![],
        by: Some(Array1::from_shape_fn(n, |i| 1.0 / (1.0 + x[i]))),
        variables: vec!["x".into(), "g".into()],
    };
    let model = build_model(vec![term], w.view(), n).unwrap();
    assert_eq!(model.x.ncols(), 2); // x*g = 1 is already the intercept
    assert_eq!(model.blocks[0].kept.as_ref().unwrap().len(), 1);
}
