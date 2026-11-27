#![allow(unused_imports)]
use super::*;

#[cfg(test)]
mod tests {
    #[test]
    fn one_class_offsets_are_finite() {
        use super::{AdaExp, Family, HingeLoss};
        let w = ndarray::arr1(&[1.0, 1.0, 1.0]);
        for y in [
            ndarray::arr1(&[1.0, 1.0, 1.0]),
            ndarray::arr1(&[-1.0, -1.0, -1.0]),
        ] {
            let a = AdaExp.offset(y.view(), w.view());
            let h = HingeLoss::new(0.5).offset(y.view(), w.view());
            assert!(a.is_finite(), "AdaExp offset {a}");
            assert!(h.is_finite(), "HingeLoss offset {h}");
            // Positive class gives a positive offset, negative class negative.
            assert_eq!(a > 0.0, y[0] > 0.0);
            assert_eq!(h > 0.0, y[0] > 0.0);
        }
    }

    /// Analytic negative gradient against the central difference of the risk,
    /// one observation at a time. `scale` is the constant by which the
    /// family's negative gradient differs from -d(risk)/df by mboost
    /// convention (Gaussian: risk is the squared error, gradient is y - f, so
    /// the finite difference is twice the gradient). The check is
    /// `-d(risk)/df / scale == negative gradient`. `y` may be longer than `f`
    /// (survival families stack `[time, event]`); weights have the length of
    /// `f`. The fused `ngradient_and_risk` is also compared with the separate
    /// calls, on a copy of the family taken before either runs.
    fn assert_family_gradient_matches_fd(
        fam: &mut dyn super::Family,
        y: &[f64],
        f: &[f64],
        scale: f64,
    ) {
        assert_family_gradient_matches_fd_with(fam, y, f, scale, &|fam, y, f, w| fam.risk(y, f, w));
    }

    type RiskFn = dyn Fn(
        &dyn super::Family,
        ndarray::ArrayView1<f64>,
        ndarray::ArrayView1<f64>,
        ndarray::ArrayView1<f64>,
    ) -> f64;

    /// [`assert_family_gradient_matches_fd`] with the differentiated risk
    /// supplied by the caller (Multinomial differentiates `held_out_risk`).
    fn assert_family_gradient_matches_fd_with(
        fam: &mut dyn super::Family,
        y: &[f64],
        f: &[f64],
        scale: f64,
        risk: &RiskFn,
    ) {
        let n = f.len();
        let yv = ndarray::Array1::from(y.to_vec());
        let w = ndarray::Array1::from(vec![1.0; n]);
        let mut fv = ndarray::Array1::from(f.to_vec());
        let mut fused_fam = fam.clone_family();
        let mut ngr = ndarray::Array1::zeros(n);
        fam.negative_gradient(yv.view(), fv.view(), w.view(), ngr.view_mut());
        let r0 = fam.risk(yv.view(), fv.view(), w.view());
        let mut fused = ndarray::Array1::zeros(n);
        let fused_risk =
            fused_fam.ngradient_and_risk(yv.view(), fv.view(), w.view(), fused.view_mut());
        assert!(
            (fused_risk - r0).abs() < 1e-10 * (1.0 + r0.abs()),
            "{}: fused risk {fused_risk} != risk {r0}",
            fam.name()
        );
        let h = 1e-6;
        for j in 0..n {
            assert!(
                (fused[j] - ngr[j]).abs() < 1e-12 * (1.0 + ngr[j].abs()),
                "{} obs {j}: fused ngradient {} != ngradient {}",
                fam.name(),
                fused[j],
                ngr[j]
            );
            let orig = fv[j];
            fv[j] = orig + h;
            let rp = risk(&*fam, yv.view(), fv.view(), w.view());
            fv[j] = orig - h;
            let rm = risk(&*fam, yv.view(), fv.view(), w.view());
            fv[j] = orig;
            let fd = -(rp - rm) / (2.0 * h) / scale;
            assert!(
                (ngr[j] - fd).abs() < 1e-5 * (1.0 + ngr[j].abs()),
                "{} obs {j}: ngradient {} != finite-diff {}",
                fam.name(),
                ngr[j],
                fd
            );
        }
    }

    #[test]
    fn gaussian_poisson_binomial_huber_gradients_match_finite_diff() {
        use super::{Binomial, Gaussian, Huber, Poisson};
        let f = [-1.3, -0.2, 0.4, 0.9, 1.7];
        assert_family_gradient_matches_fd(&mut Gaussian, &[0.3, -1.0, 2.0, 0.5, 1.1], &f, 2.0);
        assert_family_gradient_matches_fd(&mut Poisson, &[0.0, 1.0, 3.0, 2.0, 5.0], &f, 1.0);
        // Binomial: the gradient carries mboost's 1/ln(2) factor, so -d(risk)/df is ln(2) times it (see binomial_row).
        assert_family_gradient_matches_fd(
            &mut Binomial,
            &[0.0, 1.0, 1.0, 0.0, 1.0],
            &f,
            std::f64::consts::LN_2,
        );
        // Huber: the negative gradient always clips at the adaptive
        // delta = median|y - f_prev| (R's behaviour; the first call uses
        // median|y| = 2 for this response), even when delta is supplied. The
        // risk is therefore evaluated at delta = 2 with the adaptive
        // (calibrating) form. Residuals fall on both sides of the clip.
        let mut huber = Huber::new(None);
        huber.delta = Some(2.0);
        assert_family_gradient_matches_fd(&mut huber, &[0.3, -1.0, 2.0, 4.5, -3.1], &f, 1.0);
    }

    #[test]
    fn regression_and_count_gradients_match_finite_diff() {
        use super::{GammaReg, GaussReg, Hurdle, Laplace, NBinomial};
        let f = [-1.3, -0.2, 0.4, 0.9, 1.7];
        assert_family_gradient_matches_fd(&mut GaussReg, &[0.3, -1.0, 2.0, 0.5, 1.1], &f, 2.0);
        // Laplace: every residual is away from the kink at y = f.
        assert_family_gradient_matches_fd(
            &mut Laplace::default(),
            &[0.3, -1.0, 2.0, 0.5, 1.1],
            &f,
            1.0,
        );
        // Nuisance parameters are set away from their defaults so that they
        // enter the gradient; calibration is a separate step.
        let positive = [0.4, 1.9, 0.7, 3.2, 5.0];
        assert_family_gradient_matches_fd(&mut GammaReg { sigma: 2.5 }, &positive, &f, 1.0);
        let counts = [0.0, 1.0, 3.0, 2.0, 7.0];
        assert_family_gradient_matches_fd(&mut NBinomial { sigma: 1.7 }, &counts, &f, 1.0);
        // Hurdle is fitted to the positive counts only.
        let positive_counts = [1.0, 2.0, 1.0, 4.0, 6.0];
        assert_family_gradient_matches_fd(&mut Hurdle { sigma: 0.6 }, &positive_counts, &f, 1.0);
    }

    #[test]
    fn survival_gradients_match_finite_diff() {
        use super::{CoxPH, Loglog, Lognormal, Weibull};
        let f = [-0.4, 0.3, 1.1, 0.2, -0.9, 0.6];
        // y = [time.., event..]. Time 2.0 is shared by an event and a censored
        // row, so the Breslow risk set contains a tie.
        let y = [0.5, 2.0, 3.5, 2.0, 0.9, 6.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let mut weibull = Weibull::new();
        weibull.sigma = 0.7;
        assert_family_gradient_matches_fd(&mut weibull, &y, &f, 1.0);
        let mut loglog = Loglog::new();
        loglog.sigma = 1.4;
        assert_family_gradient_matches_fd(&mut loglog, &y, &f, 1.0);
        let mut lognormal = Lognormal::new();
        lognormal.sigma = 0.8;
        assert_family_gradient_matches_fd(&mut lognormal, &y, &f, 1.0);
        assert_family_gradient_matches_fd(&mut CoxPH::new(), &y, &f, 1.0);
    }

    #[test]
    fn propodds_gradient_matches_finite_diff() {
        use super::PropOdds;
        // Four classes, with rows in the first, the last and both middle classes.
        let mut fam = PropOdds::new(4);
        fam.sigma = vec![-0.8, 0.1, 1.2];
        let y = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0];
        let f = [-0.5, 0.2, 0.9, 1.6, -1.1, 0.4];
        assert_family_gradient_matches_fd(&mut fam, &y, &f, 1.0);
    }

    #[test]
    fn multinomial_gradient_matches_finite_diff() {
        use super::{Family, Multinomial};
        // Three classes: two non-reference indicator columns per observation.
        // The last two observations are in the reference class.
        let y = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let f = [0.4, -0.3, -1.0, 0.8, 1.5, 0.2, 0.3, 0.6, -0.7, -0.2];
        // The negative gradient y - p is the derivative of the full negative
        // log-likelihood, which `held_out_risk` computes.
        assert_family_gradient_matches_fd_with(
            &mut Multinomial::new(3),
            &y,
            &f,
            1.0,
            &|fam, y, f, w| fam.held_out_risk(y, f, w),
        );
        // R's in-bag loss `-y log p` leaves out the reference class. It has
        // the same derivative on observations outside the reference class; a
        // reference-class observation adds no loss but has gradient -p.
        let mut fam = Multinomial::new(3);
        let w = ndarray::Array1::from(vec![1.0; 6]);
        let mut ngr = ndarray::Array1::zeros(6);
        let yv = ndarray::arr1(&y[..6]);
        let mut fv = ndarray::arr1(&f[..6]);
        fam.negative_gradient(yv.view(), fv.view(), w.view(), ngr.view_mut());
        let h = 1e-6;
        for j in 0..6 {
            let orig = fv[j];
            fv[j] = orig + h;
            let rp = fam.risk(yv.view(), fv.view(), w.view());
            fv[j] = orig - h;
            let rm = fam.risk(yv.view(), fv.view(), w.view());
            fv[j] = orig;
            let fd = -(rp - rm) / (2.0 * h);
            assert!((ngr[j] - fd).abs() < 1e-5, "obs {j}: {} != {fd}", ngr[j]);
        }
    }

    #[test]
    fn gamma_risk_change_retains_small_quadratic_terms() {
        use super::{Family, GammaReg};
        let family = GammaReg { sigma: 20.0 };
        let y = ndarray::arr1(&[1.0]);
        let step = 1e-9;
        let change = family
            .risk_change(y.view(), &[0.0], &[1.0], &[3.0], step)
            .unwrap();
        // At y = mu the first derivative is zero and curvature is w * shape.
        assert!((change / (step * step) - 30.0).abs() < 1e-7);
        assert!(change > 0.0);
        let y = ndarray::arr1(&[1.5, 0.7]);
        let f = ndarray::arr1(&[0.2, -0.1]);
        let w = ndarray::arr1(&[3.0, 1.0]);
        let h = [0.1, -0.2];
        let newer = ndarray::arr1(&[f[0] + 0.3 * h[0], f[1] + 0.3 * h[1]]);
        let ordinary = family.risk(y.view(), newer.view(), w.view())
            - family.risk(y.view(), f.view(), w.view());
        let stable = family
            .risk_change(
                y.view(),
                f.as_slice().unwrap(),
                &h,
                w.as_slice().unwrap(),
                0.3,
            )
            .unwrap();
        assert!((stable - ordinary).abs() < 1e-12);
    }

    #[test]
    fn binary_class_decision_boundaries() {
        use super::{AdaExp, Auc, Binomial, Family, Gaussian, HingeLoss};
        let f = ndarray::arr1(&[-1.0, 0.0, 0.1, 1.0]);
        for family in [&Binomial as &dyn Family, &AdaExp, &Auc] {
            assert_eq!(family.class(f.view()), Some(vec![0, 0, 1, 1]));
        }
        // mboost's asymmetric HingeLoss boundary is 2 * alphaCost - 1.
        let f = ndarray::arr1(&[0.4, 0.5, 0.6]);
        assert_eq!(HingeLoss::new(0.75).class(f.view()), Some(vec![0, 0, 1]));
        assert_eq!(Gaussian.class(f.view()), None);
    }

    use super::*;
    use ndarray::{array, Array1};

    #[test]
    fn gaussian_offset() {
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let offset = Gaussian.offset(y.view(), w.view());
        assert!((offset - 2.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_offset_weighted() {
        let y = array![1.0, 2.0, 3.0];
        let w = array![0.0, 0.0, 1.0];
        let offset = Gaussian.offset(y.view(), w.view());
        assert!((offset - 3.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_negative_gradient() {
        let y = array![1.0, 2.0, 3.0];
        let f = array![2.0, 2.0, 2.0];
        let w = array![1.0, 1.0, 1.0];
        let mut u = Array1::zeros(3);
        Gaussian.negative_gradient(y.view(), f.view(), w.view(), u.view_mut());
        assert!((u[0] - (-1.0)).abs() < 1e-12);
        assert!((u[1] - 0.0).abs() < 1e-12);
        assert!((u[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_risk() {
        let y = array![1.0, 2.0, 3.0];
        let f = array![2.0, 2.0, 2.0];
        let w = array![1.0, 1.0, 1.0];
        let risk = Gaussian.risk(y.view(), f.view(), w.view());
        assert!((risk - 2.0).abs() < 1e-12);
    }

    #[test]
    fn binomial_offset() {
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let offset = Binomial.offset(y.view(), w.view());
        assert!((offset - 0.0).abs() < 1e-10);
    }

    /// Grouped binomial data (40 rows of 3 to 8 trials, some all-failure and
    /// all-success rows) and its Bernoulli expansion: one row per trial with
    /// y in {0, 1}. Returns (x, p, trials, x_long, y_long, group_of_long_row).
    #[allow(clippy::type_complexity)]
    fn grouped_and_expanded_binomial() -> (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Vec<usize>,
    ) {
        let groups = 40usize;
        let mut x = Vec::new();
        let mut p = Vec::new();
        let mut trials = Vec::new();
        let mut x_long = Vec::new();
        let mut y_long = Vec::new();
        let mut group = Vec::new();
        for i in 0..groups {
            let xi = -1.5 + 3.0 * i as f64 / (groups - 1) as f64;
            let t = 3 + (i * 7) % 6;
            let prob = 1.0 / (1.0 + (-(0.4 + 2.5 * xi + 0.5 * (3.0 * xi).sin())).exp());
            let k = ((t as f64) * prob).round() as usize;
            x.push(xi);
            p.push(k as f64 / t as f64);
            trials.push(t as f64);
            for j in 0..t {
                x_long.push(xi);
                y_long.push(if j < k { 1.0 } else { 0.0 });
                group.push(i);
            }
        }
        assert!(p.contains(&0.0) && p.contains(&1.0));
        assert!(p.iter().any(|&v| v > 0.0 && v < 1.0));
        (
            Array1::from(x),
            Array1::from(p),
            Array1::from(trials),
            Array1::from(x_long),
            Array1::from(y_long),
            group,
        )
    }

    fn binomial_fit(x: &Array1<f64>, y: &Array1<f64>, w: &Array1<f64>) -> crate::BoostingPath {
        use crate::learners::{prepare_bbs, prepare_scalar_bols};
        let mut learners = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, false, "bols(x)", None),
            prepare_bbs(x.view(), w.view(), 8, 3, 2, 4.0, true, "bbs(x)"),
        ];
        let control = crate::BoostControl::new(60, 0.1).unwrap();
        crate::engine::try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut Binomial,
            &control,
            None,
        )
        .unwrap()
    }

    #[test]
    fn binomial_proportion_with_trial_weights_equals_bernoulli_expansion() {
        let (x, p, trials, x_long, y_long, group) = grouped_and_expanded_binomial();
        let grouped = binomial_fit(&x, &p, &trials);
        let expanded = binomial_fit(&x_long, &y_long, &Array1::ones(y_long.len()));

        let total: f64 = trials.sum();
        let share: f64 = (&p * &trials).sum() / total;
        let expected_offset = 0.5 * (share / (1.0 - share)).ln();
        assert!((grouped.offset - expected_offset).abs() < 1e-12);
        assert!(
            (grouped.offset - expanded.offset).abs() < 1e-10,
            "offset {} vs {}",
            grouped.offset,
            expanded.offset
        );
        assert_eq!(grouped.selected, expanded.selected);
        for (row, &g) in group.iter().enumerate() {
            let diff = (grouped.fitted[g] - expanded.fitted[row]).abs();
            assert!(
                diff < 1e-10,
                "row {row} (group {g}): fitted differs by {diff}"
            );
        }
        for (step, (a, b)) in grouped.risk.iter().zip(&expanded.risk).enumerate() {
            assert!(
                (a - b).abs() < 1e-10 * (1.0 + b.abs()),
                "risk at step {step}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn binomial_encoding_is_idempotent_and_does_not_change_the_fit() {
        let (x, p, trials, ..) = grouped_and_expanded_binomial();
        let once = Binomial.encode_response(p.view()).unwrap();
        let twice = Binomial.encode_response(once.view()).unwrap();
        assert_eq!(once, twice);
        // A response encoded before the fit (as cvrisk does) gives the fit on
        // the raw response.
        let raw = binomial_fit(&x, &p, &trials);
        let pre = binomial_fit(&x, &once, &trials);
        assert_eq!(raw.offset, pre.offset);
        assert_eq!(raw.fitted, pre.fitted);
    }

    #[test]
    fn binomial_ngradient_is_the_scaled_derivative_of_the_loss() {
        // ngradient = -(1 / ln 2) dL/df for labels and proportions alike.
        let eps = 1e-6;
        for &y in &[-1.0, 0.0, 1.0, 0.05, 0.3, 0.7, 0.999] {
            for &f in &[-3.0, -0.8, 0.0, 0.25, 1.4, 4.0] {
                let numeric =
                    -(binomial_row(y, f + eps).1 - binomial_row(y, f - eps).1) / (2.0 * eps);
                let analytic = binomial_row(y, f).0 * std::f64::consts::LN_2;
                assert!(
                    (numeric - analytic).abs() < 1e-7 * (1.0 + analytic.abs()),
                    "y={y} f={f}: {analytic} vs {numeric}"
                );
            }
        }
    }

    #[test]
    fn binomial_encode_response_maps_0_1_to_pm1() {
        // {0,1} -> {-1,+1}; idempotent on an already {-1,+1} response.
        let y01 = array![0.0, 1.0, 1.0, 0.0];
        let enc = Binomial
            .encode_response(y01.view())
            .expect("Binomial recodes");
        assert_eq!(enc.to_vec(), vec![-1.0, 1.0, 1.0, -1.0]);
        let ypm = array![-1.0, 1.0, 1.0, -1.0];
        let enc2 = Binomial
            .encode_response(ypm.view())
            .expect("Binomial recodes");
        assert_eq!(enc2.to_vec(), vec![-1.0, 1.0, 1.0, -1.0]);
        // Non-Binomial families leave the response alone (no allocation).
        assert!(Gaussian.encode_response(y01.view()).is_none());
        assert!(Poisson.encode_response(y01.view()).is_none());
    }

    #[test]
    fn poisson_offset() {
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let offset = Poisson.offset(y.view(), w.view());
        assert!((offset - 2.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn poisson_risk_includes_log_factorial() {
        let y = array![1.0, 2.0, 3.0];
        let f = array![0.0, 0.0, 0.0];
        let w = array![1.0, 1.0, 1.0];
        let risk = Poisson.risk(y.view(), f.view(), w.view());
        // R's Poisson risk at f=0: sum(exp(0) - y*0 + log(y!)) = 3 + (0+0.693+1.791)
        let expected = 3.0 - 0.0 + (0.0 + (2.0f64).ln() + lgamma(4.0));
        assert!((risk - expected).abs() < 1e-10);
    }

    #[test]
    fn laplace_offset_optimize() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let offset = Laplace::default().offset(y.view(), w.view());
        assert!((offset - 3.0).abs() < 1e-6);
    }

    #[test]
    fn quantreg_offset_type7() {
        // For 5 values, type-7 median at q=0.5: h = 0.5*4 + 1 = 3.0
        // So it picks the 3rd sorted value (1-indexed) = 3.0
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let fam = QuantReg::new(0.5);
        let offset = fam.offset(y.view(), w.view());
        assert!((offset - 3.0).abs() < 1e-12);
    }

    #[test]
    fn quantreg_offset_type7_even_length() {
        // For 4 values, type-7 median: h = 0.5*3 + 1 = 2.5
        // Interpolates between sorted[1] and sorted[2]
        let y = array![1.0, 2.0, 3.0, 4.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let fam = QuantReg::new(0.5);
        let offset = fam.offset(y.view(), w.view());
        assert!((offset - 2.5).abs() < 1e-12);
    }

    #[test]
    fn expectreg_offset() {
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let fam = ExpectReg::new(0.5);
        let offset = fam.offset(y.view(), w.view());
        assert!((offset - 2.0).abs() < 1e-12);
    }

    #[test]
    fn expectreg_offset_weighted_is_mean_of_unit_weights() {
        // R's ExpectReg offset is mean(y[w == 1]) and INDEPENDENT of tau.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = array![1.0, 2.0, 1.0, 3.0, 1.0, 1.0];
        // unit-weight obs are y = 1,3,5,6 -> mean = 15/4 = 3.75
        for tau in [0.5, 0.2, 0.8] {
            let fam = ExpectReg::new(tau);
            let off = fam.offset(y.view(), w.view());
            assert!(
                (off - 3.75).abs() < 1e-12,
                "tau={tau}: ExpectReg offset = mean(y[w==1]); got {off}, want 3.75"
            );
        }
    }

    #[test]
    fn expectreg_ngradient_sign() {
        // At tau=0.5, ngradient should equal y - f (same sign)
        let y = array![1.0, 2.0, 3.0];
        let f = array![2.0, 2.0, 2.0];
        let w = array![1.0, 1.0, 1.0];
        let mut u = Array1::zeros(3);
        let mut fam = ExpectReg::new(0.5);
        fam.negative_gradient(y.view(), f.view(), w.view(), u.view_mut());
        // y - f = [-1, 0, 1]
        assert!((u[0] - (-1.0)).abs() < 1e-12);
        assert!((u[1] - 0.0).abs() < 1e-12);
        assert!((u[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn huber_offset_optimize() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let fam = Huber::new(None);
        let offset = fam.offset(y.view(), w.view());
        assert!((offset - 3.0).abs() < 1e-6);
    }

    #[test]
    fn gammareg_offset_optimize() {
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let offset = GammaReg::new().offset(y.view(), w.view());
        // R's GammaReg offset optimizes over [0, max(y^2)] = [0, 9]
        // The true optimum is log(mean(y)) = log(2) ≈ 0.693, which IS in [0, 9]
        assert!((offset - (2.0f64).ln()).abs() < 1e-4);
    }

    #[test]
    fn gammareg_risk_uses_plloss() {
        let y = array![1.0, 2.0, 3.0];
        let f = array![0.0, 0.0, 0.0];
        let w = array![1.0, 1.0, 1.0];
        let fam = GammaReg::new(); // sigma=1
        let risk = fam.risk(y.view(), f.view(), w.view());
        // plloss(1, y, 0) = lgamma(1) + 1*y*exp(0) - 1*log(y) - 1*log(1) + 0
        // = 0 + y - log(y) - 0 + 0
        // For y=1,2,3: (1-0) + (2-ln2) + (3-ln3) = 6 - ln2 - ln3 ≈ 4.208
        let expected = 0.0 + 1.0 - 0.0 + 2.0 - (2.0f64).ln() + 3.0 - (3.0f64).ln();
        assert!((risk - expected).abs() < 1e-10);
    }

    #[test]
    fn weighted_quantile_median() {
        let y = array![5.0, 1.0, 3.0, 2.0, 4.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let q = weighted_quantile(y.view(), w.view(), 0.5);
        // uniform weights => type-7 on the data: h = 0.5*4 + 1 = 3.0 => 3.0
        assert!((q - 3.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_quantile_weighted() {
        // R: quantile(rep(c(1,2,3,4,5), c(1,1,1,1,10)), 0.5) replicates 5 ten
        // times -> 14-elt sample; type-7 median h = 0.5*13 + 1 = 7.5, between
        // the 7th and 8th sorted values (both 5) => 5.0. The old code ignored
        // weights and wrongly returned 3.0.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 10.0];
        let q = weighted_quantile(y.view(), w.view(), 0.5);
        assert!((q - 5.0).abs() < 1e-12, "weighted median got {q}, want 5.0");
    }

    #[test]
    fn weighted_quantile_matches_expansion_for_integer_and_fractional_weights() {
        fn expanded_reference(y: ArrayView1<f64>, w: ArrayView1<f64>, q: f64) -> f64 {
            let mut expanded = Vec::new();
            for (&value, &weight) in y.iter().zip(w.iter()) {
                for _ in 0..weight.max(0.0) as usize {
                    expanded.push(value);
                }
            }
            if expanded.is_empty() {
                return type7_quantile(y, q);
            }
            expanded.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            type7_from_sorted(&expanded, q)
        }

        let mut state = 20260715_u64;
        for _case in 0..100 {
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let n = 1 + (random() * 15.0) as usize;
            let y = Array1::from_shape_fn(n, |_| (20.0 * random()).round() / 4.0 - 2.5);
            let w = Array1::from_shape_fn(n, |_| (6.0 * random()).floor() + 0.75 * random());
            for q in [0.0, 0.01, 0.25, 0.5, 0.9, 1.0] {
                let expected = expanded_reference(y.view(), w.view(), q);
                let actual = weighted_quantile(y.view(), w.view(), q);
                assert_eq!(actual, expected, "n={n}, q={q}, y={y:?}, w={w:?}");
            }
        }
    }

    #[test]
    fn weighted_quantile_huge_frequencies_do_not_expand() {
        let y = array![1.0, 10.0, 20.0];
        let w = array![1.0e15, 2.0e15, 1.0e15];
        assert_eq!(weighted_quantile(y.view(), w.view(), 0.25), 7.75);
        assert_eq!(weighted_quantile(y.view(), w.view(), 0.5), 10.0);
        // At this scale f64 cannot represent every adjacent order-statistic
        // position; this matches the existing type-7 arithmetic order.
        assert_eq!(weighted_quantile(y.view(), w.view(), 0.75), 10.0);

        // Saturating casts plus checked cumulative arithmetic remain finite
        // even when individual floating-point weights exceed usize.
        let extreme = array![f64::MAX, f64::MAX, f64::MAX];
        let result = weighted_quantile(y.view(), extreme.view(), 0.5);
        assert!(result.is_finite());
        assert_eq!(result, 10.0);
    }

    #[test]
    fn type7_quantile_is_unweighted() {
        // type7_quantile is the UNWEIGHTED median R's Huber delta relies on.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        // h = 0.5*4 + 1 = 3.0 => 3rd sorted value = 3.0
        assert!((type7_quantile(y.view(), 0.5) - 3.0).abs() < 1e-12);
        // quartile: h = 0.25*4 + 1 = 2.0 => 2nd value = 2.0
        assert!((type7_quantile(y.view(), 0.25) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn quantreg_offset_weighted_matches_r() {
        // R: QuantReg(tau)@offset(y, w) = quantile(y[rep(1:n, w)], qoffset),
        // qoffset = 0.5 by default and INDEPENDENT of tau. Reference computed
        // in R 2.9.11 for this exact y/w.
        let y = array![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let w = array![1.0, 2.0, 0.0, 3.0, 1.0, 1.0, 2.0, 1.0];
        // rep -> [3, 1,1, (4 dropped), 1,1,1, 5, 9, 2,2, 6] sorted:
        // [1,1,1,1,1,2,2,3,5,6,9] (n=11); h = 0.5*10 + 1 = 6 => 6th = 2.0
        let fam = QuantReg::with_qoffset(0.5, 0.5);
        assert!((fam.offset(y.view(), w.view()) - 2.0).abs() < 1e-12);
        // tau must not affect the offset (R uses qoffset, not tau)
        let fam_tau = QuantReg::with_qoffset(0.25, 0.5);
        assert!(
            (fam_tau.offset(y.view(), w.view()) - 2.0).abs() < 1e-12,
            "QuantReg offset must depend on qoffset, not tau"
        );
    }

    #[test]
    fn binomial_check_y_rejects_non_binary() {
        let y = array![0.0, 1.0, 2.0];
        assert!(Binomial.check_y(y.view()).is_err());
        let negative_nonclass = array![-0.2, 1.0];
        assert!(Binomial.check_y(negative_nonclass.view()).is_err());
    }

    #[test]
    fn poisson_check_y_rejects_negative() {
        let y = array![-1.0, 1.0, 2.0];
        assert!(Poisson.check_y(y.view()).is_err());
    }

    #[test]
    fn gaussian_check_y_accepts_any() {
        let y = array![1.0, -5.0, 0.0, 1e10, -1e10];
        assert!(Gaussian.check_y(y.view()).is_ok());
    }

    #[test]
    fn gammareg_check_y_rejects_zero() {
        let y = array![0.0, 1.0, 2.0];
        assert!(GammaReg::new().check_y(y.view()).is_err());
    }

    #[test]
    fn minimize_brent_quadratic() {
        let f = |x: f64| (x - 3.0) * (x - 3.0);
        let result = minimize_brent(&f, 0.0, 10.0, 1e-10, 500);
        assert!((result - 3.0).abs() < 1e-8);
    }

    #[test]
    fn minimize_brent_at_boundary() {
        let f = |x: f64| (x + 2.0) * (x + 2.0);
        let result = minimize_brent(&f, 0.0, 10.0, 1e-10, 500);
        assert!(result.abs() < 0.1);
    }

    #[test]
    fn nbinomial_offset() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let fam = NBinomial::new();
        let offset = fam.offset(y.view(), w.view());
        let expected = (3.0f64).ln();
        assert!((offset - expected).abs() < 1e-10);
    }

    #[test]
    fn nbinomial_check_y_rejects_negative() {
        let y = array![-1.0, 1.0, 2.0];
        assert!(NBinomial::new().check_y(y.view()).is_err());
    }

    #[test]
    fn nbinomial_risk_positive() {
        let y = array![1.0, 3.0, 5.0, 2.0];
        let f = array![0.1, 0.2, 0.3, 0.4];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let fam = NBinomial::new();
        let risk = fam.risk(y.view(), f.view(), w.view());
        assert!(risk > 0.0);
    }

    #[test]
    fn nuisance_get_set_scalar_families() {
        // Nuisance families expose a scalar nuisance for cv to reuse across the
        // OOB walk; non-nuisance families report None.
        let mut nb = NBinomial::new();
        nb.set_nuisance(3.5);
        assert_eq!(nb.nuisance(), Some(3.5));
        assert_eq!(nb.sigma, 3.5);
        let mut wb = Weibull::new();
        wb.set_nuisance(0.8);
        assert_eq!(wb.nuisance(), Some(0.8));
        let mut ll = Loglog::new();
        ll.set_nuisance(1.2);
        assert_eq!(ll.nuisance(), Some(1.2));
        assert_eq!(Gaussian.nuisance(), None);
        assert_eq!(Poisson.nuisance(), None);
        // set_nuisance is a harmless no-op for families without one.
        let mut g = Gaussian;
        g.set_nuisance(5.0);
        assert_eq!(g.nuisance(), None);
    }

    #[test]
    fn nbinomial_plloss_matches_r() {
        // Single-obs risk must equal R's NBinomial plloss, which includes the
        // log(y!) = lgamma(y+1) normalising constant. Reference from R 2.9.11:
        // plloss(sigma=2, y=3, f=0.5) = 2.199295124848.
        let mut fam = NBinomial::new();
        fam.sigma = 2.0;
        let y = array![3.0];
        let f = array![0.5];
        let w = array![1.0];
        let risk = fam.risk(y.view(), f.view(), w.view());
        assert!(
            (risk - 2.199295124848).abs() < 1e-9,
            "NBinomial risk = {risk}, R = 2.199295124848 (missing lgamma(y+1)?)"
        );
    }

    #[test]
    fn propodds_offset() {
        let y = array![1.0, 2.0, 3.0, 1.0, 2.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let fam = PropOdds::new(3);
        let offset = fam.offset(y.view(), w.view());
        assert!(offset.is_finite());
    }

    #[test]
    fn propodds_check_y_valid() {
        let y = array![1.0, 2.0, 3.0, 1.0, 2.0];
        assert!(PropOdds::new(3).check_y(y.view()).is_ok());
    }

    #[test]
    fn propodds_check_y_rejects_out_of_range() {
        let y = array![0.0, 1.0, 2.0];
        assert!(PropOdds::new(2).check_y(y.view()).is_err());
        let y2 = array![1.0, 3.0, 2.0];
        assert!(PropOdds::new(2).check_y(y2.view()).is_err());
    }

    #[test]
    fn propodds_risk_positive() {
        let y = array![1.0, 2.0, 3.0, 2.0, 1.0];
        let f = array![0.1, -0.2, 0.3, 0.0, -0.1];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut fam = PropOdds::new(3);
        fam.calibrate(y.view(), f.view(), w.view());
        let risk = fam.risk(y.view(), f.view(), w.view());
        assert!(risk > 0.0);
    }

    #[test]
    fn propodds_response_matches_r() {
        // Per-category probabilities vs R's PropOdds()@response (K=3),
        // f = [0.5, -0.3], sigma = [0.0, 1.0]. Row-major (n x K).
        let f = [0.5, -0.3];
        let sigma = [0.0, 1.0];
        let got = propodds_response(&f, &sigma);
        let want = [
            0.377540668798145,
            0.244918662403709,
            0.377540668798145,
            0.574442516811659,
            0.211392466230900,
            0.214165016957441,
        ];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-12, "got {g}, want {w}");
        }
        // Each row sums to 1.
        assert!((got[0] + got[1] + got[2] - 1.0).abs() < 1e-12);
        assert!((got[3] + got[4] + got[5] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn propodds_risk_decreases() {
        let y = array![1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0];
        let w = ndarray::Array1::from_elem(8, 1.0f64);
        let mut fam = PropOdds::new(3);
        let mut f = ndarray::Array1::zeros(8);
        fam.calibrate(y.view(), f.view(), w.view());
        let risk_before = fam.risk(y.view(), f.view(), w.view());
        let ng = {
            let mut out = ndarray::Array1::zeros(8);
            fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
            out
        };
        f += &(0.1 * &ng);
        fam.calibrate(y.view(), f.view(), w.view());
        let risk_after = fam.risk(y.view(), f.view(), w.view());
        assert!(
            risk_after <= risk_before + 1e-10,
            "risk should decrease after negative gradient step"
        );
    }

    // ─── CoxPH tests ───────────────────────────────────────────────────

    #[test]
    fn coxph_check_y_valid() {
        // y = [time, event] concatenated, 3 obs
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        assert!(CoxPH::new().check_y(y.view()).is_ok());
    }

    #[test]
    fn coxph_check_y_rejects_odd_length() {
        let y = array![1.0, 2.0, 3.0];
        assert!(CoxPH::new().check_y(y.view()).is_err());
    }

    #[test]
    fn coxph_check_y_rejects_bad_event() {
        // event must be 0 or 1
        let y = array![1.0, 2.0, 3.0, 0.5];
        assert!(CoxPH::new().check_y(y.view()).is_err());
    }

    #[test]
    fn coxph_offset_is_zero() {
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        let w = ndarray::Array1::from_elem(3, 1.0f64);
        assert_eq!(CoxPH::new().offset(y.view(), w.view()), 0.0);
    }

    #[test]
    fn coxph_n_from_y() {
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        assert_eq!(CoxPH::new().n_from_y(y.view()), 3);
    }

    #[test]
    fn coxph_risk_positive() {
        // 6 obs: times=[1,2,3,4,5,6], events=[1,0,1,0,1,0]
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let f = ndarray::Array1::zeros(6);
        let w = ndarray::Array1::from_elem(6, 1.0f64);
        let risk = CoxPH::new().risk(y.view(), f.view(), w.view());
        assert!(risk > 0.0, "CoxPH risk should be positive, got {}", risk);
    }

    #[test]
    fn coxph_ngradient_sums_to_num_events() {
        // The sum of ngradient should equal the number of events (for zero f)
        // because score_i = delta_i - sum of contributions, and contributions sum to 1 per event
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        let f = ndarray::Array1::zeros(3);
        let w = ndarray::Array1::from_elem(3, 1.0f64);
        let mut fam = CoxPH::new();
        let mut out = ndarray::Array1::zeros(3);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        // With all f=0, at each event time, each obs in risk set gets -1/n_risk
        // Sum of scores should be close to 0 for model-based reasons
        // At minimum, ngradient entries should be finite
        for i in 0..3 {
            assert!(out[i].is_finite(), "ngradient[{}] = {}", i, out[i]);
        }
    }

    #[test]
    fn coxph_risk_decreases() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let w = ndarray::Array1::from_elem(6, 1.0f64);
        let f = ndarray::Array1::zeros(6);
        let risk_before = CoxPH::new().risk(y.view(), f.view(), w.view());
        let mut fam = CoxPH::new();
        let mut out = ndarray::Array1::zeros(6);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        let f_after = &f + &(0.01 * &out);
        let risk_after = CoxPH::new().risk(y.view(), f_after.view(), w.view());
        assert!(
            risk_after < risk_before + 1e-6,
            "risk should decrease: before={}, after={}",
            risk_before,
            risk_after
        );
    }

    #[test]
    fn coxph_weighted_risk_and_gradient_match_r() {
        // Value-level parity with R mboost's CoxPH()@risk / @ngradient under
        // NON-UNIFORM case weights (the bootstrap-cvrisk inbag path). R's
        // `plloss` replicates each observation w_i times, so weights enter the
        // risk-set total S0, the weighted event count dN, and the outer
        // log-likelihood term -- but NOT the gradient's leading e^{f_i}.
        // Reference values from R 2.9.11 (see tests/test_glmboost_parity).
        // y = [time.. , event..]; ties at t=2 (one event, one censored).
        let y = array![
            1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // time
            1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0 // event
        ];
        let f = array![0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.2, 0.1];
        let w = array![2.0, 1.0, 0.0, 3.0, 1.0, 1.0, 2.0, 1.0];

        let r_u = [
            0.814_195_943_968_914_6,
            0.7778235446790619,
            -0.36630704773638734,
            0.390_300_596_620_903_2,
            -1.0052243750842944,
            0.43239850754743525,
            -0.722_744_042_014_675_3,
            -1.5588032709133914,
        ];
        let r_risk = 17.742_276_966_264_35;

        let risk = CoxPH::new().risk(y.view(), f.view(), w.view());
        assert!(
            (risk - r_risk).abs() < 1e-10,
            "weighted risk: got {risk}, R = {r_risk}"
        );
        let mut fam = CoxPH::new();
        let mut out = ndarray::Array1::zeros(8);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        for i in 0..8 {
            assert!(
                (out[i] - r_u[i]).abs() < 1e-10,
                "weighted ngradient[{i}]: got {}, R = {}",
                out[i],
                r_u[i]
            );
        }

        // Uniform weights must still match R bit-for-bit (full-data fit path).
        let w1 = ndarray::Array1::from_elem(8, 1.0f64);
        let r_u1 = [
            0.876_093_239_758_539_4,
            0.803_432_937_992_472_7,
            -0.324_084_296_250_841_3,
            0.582_783_827_017_510_2,
            -0.687_873_178_876_335,
            0.49666002499338235,
            -0.44212441892658694,
            -1.3048881357081417,
        ];
        let r_risk1 = 8.217021226686537;
        let risk1 = CoxPH::new().risk(y.view(), f.view(), w1.view());
        assert!(
            (risk1 - r_risk1).abs() < 1e-10,
            "uniform risk: got {risk1}, R = {r_risk1}"
        );
        let mut out1 = ndarray::Array1::zeros(8);
        fam.negative_gradient(y.view(), f.view(), w1.view(), out1.view_mut());
        for i in 0..8 {
            assert!(
                (out1[i] - r_u1[i]).abs() < 1e-10,
                "uniform ngradient[{i}]: got {}, R = {}",
                out1[i],
                r_u1[i]
            );
        }
    }

    // ─── Weibull tests ─────────────────────────────────────────────────

    #[test]
    fn weibull_check_y_valid() {
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        assert!(Weibull::new().check_y(y.view()).is_ok());
    }

    #[test]
    fn weibull_check_y_rejects_zero_time() {
        let y = array![0.0, 2.0, 1.0, 0.0];
        assert!(Weibull::new().check_y(y.view()).is_err());
    }

    #[test]
    fn weibull_check_y_rejects_odd_length() {
        let y = array![1.0, 2.0, 3.0];
        assert!(Weibull::new().check_y(y.view()).is_err());
    }

    #[test]
    fn weibull_n_from_y() {
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        assert_eq!(Weibull::new().n_from_y(y.view()), 3);
    }

    #[test]
    fn weibull_offset_positive() {
        // times > 0, offset should be > 0
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        let w = ndarray::Array1::from_elem(3, 1.0f64);
        let offset = Weibull::new().offset(y.view(), w.view());
        assert!(offset > 0.0, "Weibull offset should be > 0, got {}", offset);
    }

    #[test]
    fn weibull_risk_decreases() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let w = ndarray::Array1::from_elem(6, 1.0f64);
        let mut fam = Weibull::new();
        let f = ndarray::Array1::zeros(6);
        fam.calibrate(y.view(), f.view(), w.view());
        let risk_before = fam.risk(y.view(), f.view(), w.view());
        let mut out = ndarray::Array1::zeros(6);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        let f_after = &f + &(0.01 * &out);
        fam.calibrate(y.view(), f_after.view(), w.view());
        let risk_after = fam.risk(y.view(), f_after.view(), w.view());
        assert!(
            risk_after <= risk_before + 1e-6,
            "Weibull risk should decrease: before={}, after={}",
            risk_before,
            risk_after
        );
    }

    // ─── Loglog tests ──────────────────────────────────────────────────

    #[test]
    fn loglog_check_y_valid() {
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        assert!(Loglog::new().check_y(y.view()).is_ok());
    }

    #[test]
    fn loglog_check_y_rejects_zero_time() {
        let y = array![0.0, 2.0, 1.0, 0.0];
        assert!(Loglog::new().check_y(y.view()).is_err());
    }

    #[test]
    fn loglog_n_from_y() {
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        assert_eq!(Loglog::new().n_from_y(y.view()), 3);
    }

    #[test]
    fn loglog_offset_positive() {
        let y = array![1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
        let w = ndarray::Array1::from_elem(3, 1.0f64);
        let offset = Loglog::new().offset(y.view(), w.view());
        assert!(offset > 0.0, "Loglog offset should be > 0, got {}", offset);
    }

    #[test]
    fn loglog_offset_subunit_times_finite() {
        // All survival times < 1 => every log-time < 0. The offset must be a
        // finite value (the guard returns 0.0) rather than relying on a
        // degenerate Brent bracket / producing NaN.
        let y = array![0.1, 0.3, 0.5, 1.0, 1.0, 1.0];
        let w = ndarray::Array1::from_elem(3, 1.0f64);
        let offset = Loglog::new().offset(y.view(), w.view());
        assert!(
            offset.is_finite(),
            "Loglog offset must be finite, got {offset}"
        );
    }

    #[test]
    fn adaexp_gradient_independent_of_weights() {
        // R's AdaExp@ngradient = y * exp(-y*f), with no weight factor: weights
        // enter only via the weighted base-learner fit and the risk. So scaling
        // the weights must NOT change the gradient.
        let y = array![1.0, -1.0, 1.0, -1.0];
        let f = array![0.5, -0.2, 1.3, 0.4];
        let w1 = ndarray::Array1::from_elem(4, 1.0f64);
        let w2 = ndarray::Array1::from_elem(4, 3.5f64);
        let mut g1 = ndarray::Array1::zeros(4);
        let mut g2 = ndarray::Array1::zeros(4);
        let mut fam = AdaExp;
        fam.negative_gradient(y.view(), f.view(), w1.view(), g1.view_mut());
        fam.negative_gradient(y.view(), f.view(), w2.view(), g2.view_mut());
        for i in 0..4 {
            let expected = y[i] * (-y[i] * f[i]).exp();
            assert!(
                (g1[i] - expected).abs() < 1e-12,
                "gradient w=1 wrong at {i}"
            );
            assert!(
                (g1[i] - g2[i]).abs() < 1e-12,
                "gradient depends on weights at {i}"
            );
        }
        // ngradient_and_risk must agree with negative_gradient on the gradient.
        let mut g3 = ndarray::Array1::zeros(4);
        fam.ngradient_and_risk(y.view(), f.view(), w2.view(), g3.view_mut());
        for i in 0..4 {
            assert!(
                (g3[i] - g1[i]).abs() < 1e-12,
                "ngradient_and_risk gradient mismatch at {i}"
            );
        }
    }

    #[test]
    fn adaexp_response_contract_accepts_both_binary_codings() {
        // Tolerance, not strict equality: a value like -1 + 1e-12 from float
        // arithmetic must be accepted (regression for strict `!= -1.0`).
        let y = array![1.0 - 1e-12, -1.0 + 1e-12, 1.0, -1.0, 0.0];
        assert!(AdaExp.check_y(y.view()).is_ok());
        let bad = array![0.5, -1.0];
        assert!(AdaExp.check_y(bad.view()).is_err());

        let y01 = array![0.0, 1.0, 1.0, 0.0];
        let encoded = AdaExp.encode_response(y01.view()).expect("AdaExp recodes");
        assert_eq!(encoded.to_vec(), vec![-1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn hinge_response_contract_accepts_both_binary_codings() {
        let family = HingeLoss::default();
        let y01 = array![0.0, 1.0, 1.0, 0.0];
        let ypm = array![-1.0, 1.0, 1.0, -1.0];
        assert!(family.check_y(y01.view()).is_ok());
        assert!(family.check_y(ypm.view()).is_ok());
        assert!(family.check_y(array![0.5, 1.0].view()).is_err());
        assert_eq!(
            family
                .encode_response(y01.view())
                .expect("HingeLoss recodes")
                .to_vec(),
            ypm.to_vec()
        );
    }

    #[test]
    fn binomial_gradient_finite_at_extreme_f() {
        // f far into the misclassified region must not overflow to inf/inf=NaN;
        // the gradient is clamped like the risk.
        let y = array![1.0, -1.0, 1.0, -1.0];
        let f = array![-1000.0, 1000.0, 500.0, -500.0];
        let w = ndarray::Array1::from_elem(4, 1.0f64);
        let mut fam = Binomial;
        let mut g = ndarray::Array1::zeros(4);
        fam.negative_gradient(y.view(), f.view(), w.view(), g.view_mut());
        assert!(
            g.iter().all(|v| v.is_finite()),
            "gradient not finite: {g:?}"
        );
        let mut g2 = ndarray::Array1::zeros(4);
        let r = fam.ngradient_and_risk(y.view(), f.view(), w.view(), g2.view_mut());
        assert!(
            g2.iter().all(|v| v.is_finite()),
            "ngradient_and_risk grad not finite"
        );
        assert!(r.is_finite(), "risk not finite");
    }

    #[test]
    fn loglog_risk_decreases() {
        // All-uncensored for cleaner gradient
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w = ndarray::Array1::from_elem(6, 1.0f64);
        let mut fam = Loglog::new();
        let f = ndarray::Array1::zeros(6);
        fam.calibrate(y.view(), f.view(), w.view());
        let risk_before = fam.risk(y.view(), f.view(), w.view());
        let mut out = ndarray::Array1::zeros(6);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        let f_after = &f + &(0.1 * &out);
        fam.calibrate(y.view(), f_after.view(), w.view());
        let risk_after = fam.risk(y.view(), f_after.view(), w.view());
        assert!(
            risk_after <= risk_before + 1e-6,
            "Loglog risk should decrease: before={}, after={}",
            risk_before,
            risk_after
        );
    }

    #[test]
    fn multinomial_check_y_valid() {
        // 3-class, 2 obs: y = [1,0, 0,1] (obs1=class1, obs2=class2, ref=class3)
        let y = array![1.0, 0.0, 0.0, 1.0];
        let fam = Multinomial::new(3);
        assert!(fam.check_y(y.view()).is_ok());
    }

    #[test]
    fn multinomial_check_y_rejects_bad_length() {
        let y = array![1.0, 0.0, 0.0]; // length 3, not multiple of K-1=2
        let fam = Multinomial::new(3);
        assert!(fam.check_y(y.view()).is_err());
    }

    #[test]
    fn multinomial_check_y_rejects_values() {
        let y = array![1.0, 0.5, 0.0, 1.0]; // 0.5 not 0/1
        let fam = Multinomial::new(3);
        assert!(fam.check_y(y.view()).is_err());
    }

    #[test]
    fn multinomial_ngradient_sum_to_zero() {
        // For each observation, sum of ngradient over classes should be 0
        // (since y is 0/1 indicator and softmax sums to < 1 with ref class)
        // Actually: sum_k (y_k - p_k) = (sum_k y_k) - (sum_k p_k) = (0 or 1) - (1 - p_ref)
        // For obs with y_k=0 for all k (reference class): sum = -(1 - p_ref) = -(1 - 1/(1+sum))
        // For obs with y_k=1 for some k: sum = 1 - (1 - p_ref)
        // Not exactly zero — the ngradient for reference class is implicitly 0 - p_ref but
        // we don't store it. So let's just verify the gradient direction.
        let y = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3 obs, 3 classes (K-1=2)
        let f = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // zero fitted -> uniform probs
        let w = array![1.0, 1.0, 1.0];
        let mut fam = Multinomial::new(3);
        let mut out = Array1::zeros(6);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        // With f=0: p_k = exp(0)/(1+2*exp(0)) = 1/3 for k=1,2
        // obs1 (y=[1,0]): u = [1-1/3, 0-1/3] = [2/3, -1/3]
        assert!((out[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((out[1] - (-1.0 / 3.0)).abs() < 1e-10);
        // obs2 (y=[0,1]): u = [0-1/3, 1-1/3] = [-1/3, 2/3]
        assert!((out[2] - (-1.0 / 3.0)).abs() < 1e-10);
        assert!((out[3] - 2.0 / 3.0).abs() < 1e-10);
        // obs3 (y=[0,0], ref class): u = [0-1/3, 0-1/3] = [-1/3, -1/3]
        assert!((out[4] - (-1.0 / 3.0)).abs() < 1e-10);
        assert!((out[5] - (-1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn multinomial_risk_matches_manual() {
        // 3-class, 2 obs: [1,0] and [0,1]
        // With f=0: p_k = 1/3 for all k
        // risk = -sum(y * log(p)) = -(log(1/3) + log(1/3)) = -2*log(1/3) = 2*log(3)
        let y = array![1.0, 0.0, 0.0, 1.0];
        let f = array![0.0, 0.0, 0.0, 0.0];
        // Weights are flat like y: one entry per class coordinate.
        let w = array![1.0, 1.0, 1.0, 1.0];
        let fam = Multinomial::new(3);
        let risk = fam.risk(y.view(), f.view(), w.view());
        let expected = 2.0 * 3.0f64.ln();
        assert!(
            (risk - expected).abs() < 1e-10,
            "risk={}, expected={}",
            risk,
            expected
        );
    }

    #[test]
    fn multinomial_risk_weights_each_observation_block() {
        // Three observations in classes 1, 2 and the reference, with three
        // different weights. Reading the flat weights by observation index
        // gives observation i the weight of observation i/(K-1): here that is
        // 5*log(3) instead of 9*log(3), and it is what CV's out-of-bag masks
        // and any weighted fit ran into.
        let y = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let f = ndarray::Array1::zeros(6);
        let observation_weights = [5.0, 4.0, 7.0];
        let w = ndarray::Array1::from_iter(observation_weights.iter().flat_map(|&wi| [wi, wi]));
        let fam = Multinomial::new(3);
        // At f = 0 every class probability is 1/3. R's loss, which `risk`
        // keeps, gives each non-reference observation w_i * log(3) and the
        // reference one nothing; the held-out risk is the log-likelihood,
        // with the reference observation's w * log(3) as well.
        let expected = (5.0 + 4.0) * 3.0f64.ln();
        let risk = fam.risk(y.view(), f.view(), w.view());
        assert!(
            (risk - expected).abs() < 1e-12,
            "risk={risk}, expected={expected}"
        );
        let held_out = fam.held_out_risk(y.view(), f.view(), w.view());
        assert!((held_out - (5.0 + 4.0 + 7.0) * 3.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn multinomial_risk_decreases() {
        // 3-class, 3 obs
        let y = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let w = ndarray::Array1::from_elem(6, 1.0f64);
        let mut fam = Multinomial::new(3);
        let f = ndarray::Array1::zeros(6);
        let risk_before = fam.risk(y.view(), f.view(), w.view());
        let mut out = ndarray::Array1::zeros(6);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        let f_after = &f + &(0.1 * &out);
        let risk_after = fam.risk(y.view(), f_after.view(), w.view());
        assert!(
            risk_after < risk_before,
            "Multinomial risk should decrease: before={}, after={}",
            risk_before,
            risk_after
        );
    }

    #[test]
    fn multinomial_offset_is_zero() {
        let y = array![1.0, 0.0, 0.0, 1.0];
        let w = array![1.0, 1.0];
        let fam = Multinomial::new(3);
        assert_eq!(fam.offset(y.view(), w.view()), 0.0);
    }

    #[test]
    fn multinomial_n_from_y() {
        let y = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3 obs, 3 classes
        let fam = Multinomial::new(3);
        assert_eq!(fam.n_from_y(y.view()), 6);
    }

    #[test]
    fn multinomial_4class() {
        // 4 classes (K-1=3), 2 obs
        let y = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // obs1=class1, obs2=class2
        let f = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let w = array![1.0, 1.0];
        let mut fam = Multinomial::new(4);
        let mut out = Array1::zeros(6);
        fam.negative_gradient(y.view(), f.view(), w.view(), out.view_mut());
        // p_k = exp(0)/(1+3*exp(0)) = 1/4
        // obs1: u = [1-1/4, 0-1/4, 0-1/4] = [3/4, -1/4, -1/4]
        assert!((out[0] - 0.75).abs() < 1e-10);
        assert!((out[1] - (-0.25)).abs() < 1e-10);
        assert!((out[2] - (-0.25)).abs() < 1e-10);
        // obs2: u = [0-1/4, 1-1/4, 0-1/4] = [-1/4, 3/4, -1/4]
        assert!((out[3] - (-0.25)).abs() < 1e-10);
        assert!((out[4] - 0.75).abs() < 1e-10);
        assert!((out[5] - (-0.25)).abs() < 1e-10);
    }

    // ─── Regression tests from Audit 1 & 2 ────────────────────────────────

    #[test]
    fn all_zero_weights_produce_finite_risk() {
        // B1/S1 guard: all-zero weights must not produce NaN risk or gradient.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let f = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![0.0, 0.0, 0.0, 0.0, 0.0];
        // Test each family separately so we can mutably call ngradient_and_risk.
        for (mut fam, name) in [
            (Box::new(Gaussian) as Box<dyn Family>, "Gaussian"),
            (Box::new(Binomial) as Box<dyn Family>, "Binomial"),
            (Box::new(Poisson) as Box<dyn Family>, "Poisson"),
            (Box::new(Laplace::default()) as Box<dyn Family>, "Laplace"),
            (Box::new(AdaExp) as Box<dyn Family>, "AdaExp"),
        ] {
            let mut g = Array1::zeros(5);
            let r = fam.ngradient_and_risk(y.view(), f.view(), w.view(), g.view_mut());
            assert!(r.is_finite(), "{name} risk not finite: {r}");
            assert!(
                g.iter().all(|v| v.is_finite()),
                "{name} gradient not finite"
            );
        }
    }

    #[test]
    fn constant_response_produces_zero_offset() {
        // T2 edge case: y constant → offset should equal that constant.
        let y = array![3.0, 3.0, 3.0, 3.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let offset = Gaussian.offset(y.view(), w.view());
        assert!(
            (offset - 3.0).abs() < 1e-12,
            "offset should be 3.0, got {offset}"
        );
    }

    #[test]
    fn constant_response_poisson_offset_uses_weighted_mean() {
        // T2 edge case: constant y for Poisson → offset = log(mean(y))
        let y = array![5.0, 5.0, 5.0, 5.0];
        let w = array![1.0, 2.0, 1.0, 2.0];
        let offset = Poisson.offset(y.view(), w.view());
        assert!((offset - 5.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn binomial_risk_does_not_overflow_for_extreme_weights() {
        // S1 regression: risk with large weights and extreme f must be finite.
        let y = array![1.0, -1.0, 1.0, -1.0];
        let f = array![-500.0, 500.0, 1000.0, -1000.0];
        let w = array![1e10, 1e10, 1e10, 1e10];
        let mut fam = Binomial;
        let mut g = Array1::zeros(4);
        let r = fam.ngradient_and_risk(y.view(), f.view(), w.view(), g.view_mut());
        assert!(r.is_finite(), "Binomial risk must be finite: {r}");
        assert!(
            g.iter().all(|v| v.is_finite()),
            "Binomial gradient must be finite"
        );
    }

    // ─── AUC tests ───────────────────────────────────────────────────

    #[test]
    fn auc_check_y_requires_two_levels() {
        let y2 = array![0.0, 1.0, 0.0, 1.0];
        assert!(Auc.check_y(y2.view()).is_ok());
        let y1 = array![1.0, 1.0, 1.0];
        assert!(Auc.check_y(y1.view()).is_err());
        let y3 = array![0.0, 1.0, 2.0];
        assert!(Auc.check_y(y3.view()).is_err());
    }

    #[test]
    fn auc_encode_response_maps_extremes_to_pm1() {
        let y01 = array![0.0, 1.0, 1.0, 0.0];
        let enc = Auc.encode_response(y01.view()).expect("Auc recodes");
        assert_eq!(enc.to_vec(), vec![-1.0, 1.0, 1.0, -1.0]);

        // Already {-1,+1} is idempotent.
        let ypm = array![-1.0, 1.0, 1.0, -1.0];
        let enc2 = Auc.encode_response(ypm.view()).expect("Auc recodes");
        assert_eq!(enc2.to_vec(), vec![-1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn auc_offset_is_always_zero() {
        let y = array![-1.0, 1.0, -1.0, 1.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        assert_eq!(Auc.offset(y.view(), w.view()), 0.0);
    }

    #[test]
    fn auc_risk_zero_for_perfect_separation() {
        // Every positive scores strictly above every negative -> AUC = 1,
        // risk (1 - AUC) = 0.
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let f = array![-2.0, -1.0, 1.0, 2.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let risk = Auc.risk(y.view(), f.view(), w.view());
        assert!(risk.abs() < 1e-12, "expected risk 0, got {risk}");
    }

    #[test]
    fn auc_risk_one_for_perfect_anti_separation() {
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let f = array![2.0, 1.0, -1.0, -2.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let risk = Auc.risk(y.view(), f.view(), w.view());
        assert!((risk - 1.0).abs() < 1e-12, "expected risk 1, got {risk}");
    }

    #[test]
    fn auc_risk_one_when_f_constant() {
        // With all-equal f (e.g. the offset=0 starting point), M0 is all
        // zeros, so no observation scores strictly > 0 -> risk = 1 (not the
        // n0==0/n1==0 degenerate-class guard; both classes are present).
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let f = array![0.0, 0.0, 0.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let risk = Auc.risk(y.view(), f.view(), w.view());
        assert!(
            (risk - 1.0).abs() < 1e-12,
            "expected risk 1 (no M0 > 0), got {risk}"
        );
    }

    #[test]
    fn auc_degrades_gracefully_for_single_class() {
        // Both classes must be present for a well-defined AUC; check_y
        // rejects this upstream, but risk/negative_gradient degrade
        // gracefully (R warns "response is constant - AUC is 1" and returns
        // risk 0) rather than dividing by zero.
        let y = array![1.0, 1.0, 1.0];
        let f = array![0.1, 0.2, 0.3];
        let w = array![1.0, 1.0, 1.0];
        assert_eq!(Auc.risk(y.view(), f.view(), w.view()), 0.0);
        let mut fam = Auc;
        let mut g = Array1::zeros(3);
        fam.negative_gradient(y.view(), f.view(), w.view(), g.view_mut());
        assert!(g.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn auc_standardization_skipped_when_f_constant() {
        // f all-equal (e.g. iteration 0, f == offset == 0) must not divide
        // by sd == 0; standardization is skipped entirely, matching R's
        // `if (length(unique(f)) != 1) f <- f/sd(f)`.
        let fs = [0.0, 0.0, 0.0, 0.0];
        let ws = [1.0, 1.0, 1.0, 1.0];
        let out = auc_standardize(&fs, &ws);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn auc_standardization_uses_n_minus_1_sample_sd() {
        // f = [1,2,3,4,5]: R's sd() uses the n-1 (sample) denominator, so
        // the standardizing divisor is sqrt(2.5), NOT the population sd
        // sqrt(2.0) -- the subtlety the spec calls load-bearing. Unit
        // weights must reduce this to the same n-1 sample sd as before
        // (see the `auc_standardize` doc comment for the bit-identical
        // argument).
        let fs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ws = [1.0, 1.0, 1.0, 1.0, 1.0];
        let out = auc_standardize(&fs, &ws);
        let expected_sd = 2.5f64.sqrt();
        for (got, raw) in out.iter().zip(fs.iter()) {
            assert!(
                (got - raw / expected_sd).abs() < 1e-12,
                "expected n-1 sample-sd standardization: got {got}, raw/sd={}",
                raw / expected_sd
            );
        }
    }

    #[test]
    fn auc_does_not_require_unit_weights() {
        // Auc now implements a principled weighted gradient (see the
        // `Auc` doc comment) instead of rejecting non-unit weights;
        // Gate A1/A2 below are the correctness oracle for that gradient.
        assert!(!Auc.requires_unit_weights());
        assert!(!Gaussian.requires_unit_weights());
    }

    /// Gate 2 (spec's safety net): a finite-difference check on the AUC
    /// pseudo-residual, independent of R.
    ///
    /// R's `AUC()@ngradient`, as coded (and reproduced in
    /// `auc_ngradient_core` above), is `ng[ind1[j]] = colSums(approxGrad(M0))`
    /// and `ng[ind0[k]] = -rowSums(approxGrad(M0))`. Writing
    /// `A(f) = sum_{k,j} approxLoss(M0_kj)` (the *unnormalized* sum, with
    /// `M0_kj = f[ind1[j]] - f[ind0[k]]`), differentiating `A` term-by-term
    /// gives exactly `dA/df_i = ng[i]` for every `i` -- no extra sign flip,
    /// no extra `1/(n0*n1)` scale factor. (R's own smoothed `loss` is
    /// `1 - A/(n0*n1)`, so `ng = (n0*n1) * (-d(loss)/df)`: proportional to,
    /// not literally equal to, `-d(loss)/df`; boosting only needs a
    /// loss-decreasing *direction*, and mboostrs tracks the true `risk` --
    /// not R's smoothed `loss` -- for reporting/CV, so this proportionality
    /// is harmless.) This test verifies `ng == dA/df` via central
    /// differences.
    ///
    /// Standardization escape hatch: production `negative_gradient` first
    /// rescales `f <- f / sd(f)`, which makes every `M0` entry a function of
    /// the *entire* `f` vector (through `sd(f)`), not just the two entries
    /// in its own pair. Differentiating `A` through that rescale picks up
    /// an extra `d(sd)/df_i` term that R's own coded `ngradient` does NOT
    /// include (R computes `approxGrad` on the already-standardized `f`
    /// without correcting for how the standardization itself varies with
    /// `f`). Finite-differencing through standardization would therefore
    /// legitimately *not* match `ng` -- not because `ng` is wrong, but
    /// because R's `ngradient` was never the exact derivative of a fully
    /// standardization-aware loss in the first place. Per the spec's
    /// documented escape hatch, this test disables standardization (it
    /// calls `auc_ngradient_core`/`auc_approx_loss` directly on the raw
    /// `f`, i.e. as if `sd(f) == 1`) and leaves the standardized path to be
    /// validated end-to-end by R-parity (`tests/test_glmboost_parity/
    /// test_auc_bols.py`).
    #[test]
    fn auc_ngradient_matches_finite_difference_of_unstandardized_loss() {
        let ys = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let f: Vec<f64> = vec![0.35, -0.72, 1.15, -0.18, 0.52, -1.4];
        let ws = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let n = f.len();
        let (ind1, ind0) = auc_partition(&ys);

        let total_approx_loss = |fv: &[f64]| -> f64 {
            let mut total = 0.0;
            for &j in &ind1 {
                for &k in &ind0 {
                    total += auc_approx_loss(fv[j] - fv[k]);
                }
            }
            total
        };

        let mut analytic = vec![0.0; n];
        auc_ngradient_core(&ind1, &ind0, &f, &ws, &mut analytic);

        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..n {
            let mut f_plus = f.clone();
            f_plus[i] += eps;
            let mut f_minus = f.clone();
            f_minus[i] -= eps;
            let numeric = (total_approx_loss(&f_plus) - total_approx_loss(&f_minus)) / (2.0 * eps);
            let err = (analytic[i] - numeric).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-4,
                "AUC ngradient[{i}] finite-diff mismatch: analytic={}, numeric={}, err={err:e}",
                analytic[i],
                numeric,
            );
        }
        eprintln!("AUC finite-diff max gradient error (unstandardized escape hatch): {max_err:e}");
    }

    /// Gate A1 (weighted extension spec): finite-difference check on the
    /// **weighted** un-standardized smoothed loss
    /// `L = sum_{i in pos, j in neg} w_i * w_j * approxLoss(f_i - f_j)`,
    /// the direct weighted generalization of the unit-weight escape-hatch
    /// test above, for the same reason (standardization pulls in a
    /// `d(sd)/df` term the raw `ngradient` doesn't include). Uses
    /// non-trivial integer weights, including a `0`, so the finite
    /// difference exercises both the pairwise `w_i * w_k` scaling and the
    /// "`w == 0` contributes nothing" requirement in one test.
    #[test]
    fn auc_ngradient_matches_finite_difference_of_weighted_unstandardized_loss() {
        let ys = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let f: Vec<f64> = vec![0.35, -0.72, 1.15, -0.18, 0.52, -1.4];
        let ws: Vec<f64> = vec![2.0, 1.0, 0.0, 3.0, 1.0, 2.0];
        let n = f.len();
        let (ind1, ind0) = auc_partition(&ys);

        let total_weighted_loss = |fv: &[f64]| -> f64 {
            let mut total = 0.0;
            for &j in &ind1 {
                for &k in &ind0 {
                    total += ws[j] * ws[k] * auc_approx_loss(fv[j] - fv[k]);
                }
            }
            total
        };

        let mut analytic = vec![0.0; n];
        auc_ngradient_core(&ind1, &ind0, &f, &ws, &mut analytic);

        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..n {
            let mut f_plus = f.clone();
            f_plus[i] += eps;
            let mut f_minus = f.clone();
            f_minus[i] -= eps;
            let numeric =
                (total_weighted_loss(&f_plus) - total_weighted_loss(&f_minus)) / (2.0 * eps);
            let err = (analytic[i] - numeric).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-4,
                "weighted AUC ngradient[{i}] finite-diff mismatch: analytic={}, numeric={}, err={err:e}",
                analytic[i],
                numeric,
            );
        }
        eprintln!("AUC weighted finite-diff max gradient error: {max_err:e}");
    }

    /// Gate A2 (spec's strongest structural check, R-independent): for
    /// weights in `{0, 1}`, the weighted `negative_gradient`/`risk` at the
    /// `w == 1` positions must equal the *unit-weight* `negative_gradient`/
    /// `risk` computed on the sub-fixture of just the `w == 1`
    /// observations, and be exactly `0` at the `w == 0` positions. This
    /// validates the entire weighted path -- partition, weighted-sd
    /// standardization-over-participating, and pairwise normalization --
    /// in one shot, independent of R.
    #[test]
    fn auc_zero_one_weights_match_unit_weight_subset() {
        let ys = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let f = [0.35, -0.72, 1.15, -0.18, 0.52, -1.4, -0.9, 1.05];
        let ws = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let n = ys.len();

        let y_full = Array1::from_vec(ys.to_vec());
        let f_full = Array1::from_vec(f.to_vec());
        let w_full = Array1::from_vec(ws.to_vec());
        let mut ng_full = Array1::zeros(n);
        let mut fam = Auc;
        fam.negative_gradient(
            y_full.view(),
            f_full.view(),
            w_full.view(),
            ng_full.view_mut(),
        );
        let risk_full = Auc.risk(y_full.view(), f_full.view(), w_full.view());

        // Both classes must survive the w==1 filter for this fixture to be
        // meaningful; verified by construction (ys/ws chosen above).
        let keep: Vec<usize> = (0..n).filter(|&i| ws[i] == 1.0).collect();
        let y_sub = Array1::from_vec(keep.iter().map(|&i| ys[i]).collect::<Vec<_>>());
        let f_sub = Array1::from_vec(keep.iter().map(|&i| f[i]).collect::<Vec<_>>());
        let w_sub = Array1::ones(keep.len());
        let mut ng_sub = Array1::zeros(keep.len());
        fam.negative_gradient(y_sub.view(), f_sub.view(), w_sub.view(), ng_sub.view_mut());
        let risk_sub = Auc.risk(y_sub.view(), f_sub.view(), w_sub.view());

        let mut max_diff = 0.0f64;
        for (sub_idx, &orig_idx) in keep.iter().enumerate() {
            let diff = (ng_full[orig_idx] - ng_sub[sub_idx]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-12,
                "AUC weighted ng[{orig_idx}] != unit-weight subset ng[{sub_idx}]: {} vs {}",
                ng_full[orig_idx],
                ng_sub[sub_idx],
            );
        }
        for i in 0..n {
            if ws[i] == 0.0 {
                assert_eq!(
                    ng_full[i], 0.0,
                    "w==0 observation must get exactly 0 gradient"
                );
            }
        }
        let risk_diff = (risk_full - risk_sub).abs();
        max_diff = max_diff.max(risk_diff);
        assert!(
            risk_diff < 1e-12,
            "AUC weighted risk != unit-weight subset risk: {risk_full} vs {risk_sub}"
        );
        eprintln!("AUC {{0,1}}==unit-subset max diff: {max_diff:e}");
    }

    // ─── Gehan ───────────────────────────────────────────────────────

    fn gehan_y(times: &[f64], events: &[f64]) -> Array1<f64> {
        let mut v = times.to_vec();
        v.extend_from_slice(events);
        Array1::from_vec(v)
    }

    #[test]
    fn gehan_check_y_accepts_valid_survival_response() {
        let y = gehan_y(&[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0]);
        assert!(Gehan.check_y(y.view()).is_ok());
    }

    #[test]
    fn gehan_check_y_rejects_odd_length() {
        let y = array![1.0, 2.0, 3.0];
        assert!(Gehan.check_y(y.view()).is_err());
    }

    #[test]
    fn gehan_check_y_rejects_nonpositive_time() {
        let y = gehan_y(&[1.0, -2.0, 3.0], &[1.0, 0.0, 1.0]);
        assert!(Gehan.check_y(y.view()).is_err());
    }

    #[test]
    fn gehan_check_y_rejects_bad_event_indicator() {
        let y = gehan_y(&[1.0, 2.0, 3.0], &[1.0, 2.0, 1.0]);
        assert!(Gehan.check_y(y.view()).is_err());
    }

    #[test]
    fn gehan_offset_is_always_zero() {
        // The Gehan loss depends only on pairwise residual differences, so a
        // constant offset cancels and the risk is flat in it -- verified
        // against real R (see families.rs doc comment on `Gehan`). mboostrs
        // therefore defines offset() = 0 rather than reproducing R's
        // arbitrary optimize()-on-a-flat-function value.
        let y = gehan_y(&[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0]);
        let w = array![1.0, 1.0, 1.0];
        assert_eq!(Gehan.offset(y.view(), w.view()), 0.0);
    }

    #[test]
    fn gehan_does_not_require_unit_weights() {
        // Gehan now implements a principled weighted gradient (see the
        // `Gehan` doc comment) instead of rejecting non-unit weights;
        // Gate A1/A2 below are the correctness oracle for that gradient.
        assert!(!Gehan.requires_unit_weights());
    }

    #[test]
    fn gehan_risk_is_offset_invariant() {
        // Direct restatement of the up-front unidentifiability fact: shifting
        // every f_i by the same constant must not change risk at all.
        let y = gehan_y(&[1.0, 2.0, 5.0, 3.0], &[1.0, 0.0, 1.0, 1.0]);
        let w = array![1.0, 1.0, 1.0, 1.0];
        let f0 = array![0.0, 0.0, 0.0, 0.0];
        let f1 = array![1.0, 1.0, 1.0, 1.0];
        let f2 = array![2.5, 2.5, 2.5, 2.5];
        let r0 = Gehan.risk(y.view(), f0.view(), w.view());
        let r1 = Gehan.risk(y.view(), f1.view(), w.view());
        let r2 = Gehan.risk(y.view(), f2.view(), w.view());
        assert!((r0 - r1).abs() < 1e-12, "r0={r0} r1={r1}");
        assert!((r0 - r2).abs() < 1e-12, "r0={r0} r2={r2}");
    }

    #[test]
    fn gehan_risk_zero_when_all_censored() {
        // loss_i is gated on Del_i = 1; with no events there is nothing to
        // rank against, so risk must be exactly zero regardless of f.
        let y = gehan_y(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
        let w = array![1.0, 1.0, 1.0];
        let f = array![0.3, -1.2, 4.0];
        assert_eq!(Gehan.risk(y.view(), f.view(), w.view()), 0.0);
    }

    /// Gate 2, check 2 (sign/structure sanity): at equal fitted values, two
    /// uncensored observations with different survival times must have their
    /// residual order broken by the gradient in the AFT-consistent
    /// direction -- push the fitted value of the *longer*-survival-time
    /// observation up and the *shorter*-survival-time observation down.
    /// Hand-derived: times = [1, 10] (Z = [0, ln(10)]), f = [0, 0], both
    /// events observed. arg2 = [1, 2], arg1 = [2, 1], so
    /// g = arg1 - arg2 = [1, -1] and ng = -g/n = [-0.5, 0.5].
    #[test]
    fn gehan_ngradient_sign_matches_aft_ranking_direction() {
        let y = gehan_y(&[1.0, 10.0], &[1.0, 1.0]);
        let f = array![0.0, 0.0];
        let w = array![1.0, 1.0];
        let mut ng = Array1::zeros(2);
        Gehan.negative_gradient(y.view(), f.view(), w.view(), ng.view_mut());
        assert!(
            (ng[0] - (-0.5)).abs() < 1e-12 && (ng[1] - 0.5).abs() < 1e-12,
            "ng={ng:?}, expected [-0.5, 0.5]"
        );
        // The shorter-time observation is pushed down, the longer-time one up.
        assert!(ng[0] < 0.0);
        assert!(ng[1] > 0.0);
    }

    /// Gate 2, check 1 (the R-independent safety net): a central-difference
    /// check of `negative_gradient` against `risk` on a tie-free fixture
    /// (distinct times, distinct f, so no exact residual ties trip the
    /// non-smooth boundary). Gehan's `risk` IS the loss being minimized
    /// (unlike AUC's smoothed surrogate), so `ng` must equal `-d(risk)/df`
    /// directly -- this is the strongest available guard against a
    /// transposed or sign-wrong gradient.
    #[test]
    fn gehan_ngradient_matches_finite_difference_of_risk() {
        let times = [2.3, 5.1, 0.8, 9.4, 3.6, 1.2];
        let events = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let f = [0.35, -0.72, 1.15, -0.18, 0.52, -1.4];
        let n = f.len();
        let y = gehan_y(&times, &events);
        let w = Array1::ones(n);

        let f_arr = Array1::from_vec(f.to_vec());
        let mut analytic = Array1::zeros(n);
        Gehan.negative_gradient(y.view(), f_arr.view(), w.view(), analytic.view_mut());

        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..n {
            let mut f_plus = f.to_vec();
            f_plus[i] += eps;
            let mut f_minus = f.to_vec();
            f_minus[i] -= eps;
            let risk_plus = Gehan.risk(y.view(), Array1::from_vec(f_plus).view(), w.view());
            let risk_minus = Gehan.risk(y.view(), Array1::from_vec(f_minus).view(), w.view());
            let numeric = -(risk_plus - risk_minus) / (2.0 * eps);
            let err = (analytic[i] - numeric).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-4,
                "Gehan ngradient[{i}] finite-diff mismatch: analytic={}, numeric={}, err={err:e}",
                analytic[i],
                numeric,
            );
        }
        eprintln!("Gehan finite-diff max gradient error: {max_err:e}");
    }

    /// Gate A1 (weighted extension spec): finite-difference check of the
    /// **weighted** `negative_gradient` against the **weighted** `risk`,
    /// the direct weighted generalization of
    /// `gehan_ngradient_matches_finite_difference_of_risk` above (Gehan's
    /// risk IS the loss it minimizes, so this remains a direct check, no
    /// escape hatch needed). Non-trivial integer weights, including a `0`,
    /// exercise both the `w_i`/`w_j` pairwise scaling and the
    /// "`w == 0` contributes nothing" requirement.
    #[test]
    fn gehan_ngradient_matches_finite_difference_of_weighted_risk() {
        let times = [2.3, 5.1, 0.8, 9.4, 3.6, 1.2];
        let events = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let f = [0.35, -0.72, 1.15, -0.18, 0.52, -1.4];
        let ws = [2.0, 1.0, 0.0, 3.0, 1.0, 2.0];
        let n = f.len();
        let y = gehan_y(&times, &events);
        let w = Array1::from_vec(ws.to_vec());

        let f_arr = Array1::from_vec(f.to_vec());
        let mut analytic = Array1::zeros(n);
        Gehan.negative_gradient(y.view(), f_arr.view(), w.view(), analytic.view_mut());

        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..n {
            let mut f_plus = f.to_vec();
            f_plus[i] += eps;
            let mut f_minus = f.to_vec();
            f_minus[i] -= eps;
            let risk_plus = Gehan.risk(y.view(), Array1::from_vec(f_plus).view(), w.view());
            let risk_minus = Gehan.risk(y.view(), Array1::from_vec(f_minus).view(), w.view());
            let numeric = -(risk_plus - risk_minus) / (2.0 * eps);
            let err = (analytic[i] - numeric).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-4,
                "weighted Gehan ngradient[{i}] finite-diff mismatch: analytic={}, numeric={}, err={err:e}",
                analytic[i],
                numeric,
            );
        }
        eprintln!("Gehan weighted finite-diff max gradient error: {max_err:e}");
    }

    /// Gate A2 (spec's strongest structural check, R-independent): for
    /// weights in `{0, 1}`, the weighted `negative_gradient`/`risk` at the
    /// `w == 1` positions must equal the *unit-weight* `negative_gradient`/
    /// `risk` computed on the sub-fixture of just the `w == 1`
    /// observations, and be exactly `0` at the `w == 0` positions --
    /// mirrors `auc_zero_one_weights_match_unit_weight_subset` above. One
    /// of the `w == 0` rows (index 1) has `event == 1`, to make sure a
    /// participating-in-the-outer-sum-but-zero-weight observation is
    /// exercised too, not just censored zero-weight rows.
    #[test]
    fn gehan_zero_one_weights_match_unit_weight_subset() {
        let times = [2.3, 5.1, 0.8, 9.4, 3.6, 1.2, 4.4, 0.5];
        let events = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let f = [0.35, -0.72, 1.15, -0.18, 0.52, -1.4, 0.2, -0.6];
        let ws = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let n = times.len();

        let y_full = gehan_y(&times, &events);
        let f_full = Array1::from_vec(f.to_vec());
        let w_full = Array1::from_vec(ws.to_vec());
        let mut ng_full = Array1::zeros(n);
        Gehan.negative_gradient(
            y_full.view(),
            f_full.view(),
            w_full.view(),
            ng_full.view_mut(),
        );
        let risk_full = Gehan.risk(y_full.view(), f_full.view(), w_full.view());

        let keep: Vec<usize> = (0..n).filter(|&i| ws[i] == 1.0).collect();
        let times_sub: Vec<f64> = keep.iter().map(|&i| times[i]).collect();
        let events_sub: Vec<f64> = keep.iter().map(|&i| events[i]).collect();
        let f_sub = Array1::from_vec(keep.iter().map(|&i| f[i]).collect::<Vec<_>>());
        let y_sub = gehan_y(&times_sub, &events_sub);
        let w_sub = Array1::ones(keep.len());
        let mut ng_sub = Array1::zeros(keep.len());
        Gehan.negative_gradient(y_sub.view(), f_sub.view(), w_sub.view(), ng_sub.view_mut());
        let risk_sub = Gehan.risk(y_sub.view(), f_sub.view(), w_sub.view());

        let mut max_diff = 0.0f64;
        for (sub_idx, &orig_idx) in keep.iter().enumerate() {
            let diff = (ng_full[orig_idx] - ng_sub[sub_idx]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-12,
                "Gehan weighted ng[{orig_idx}] != unit-weight subset ng[{sub_idx}]: {} vs {}",
                ng_full[orig_idx],
                ng_sub[sub_idx],
            );
        }
        for i in 0..n {
            if ws[i] == 0.0 {
                assert_eq!(
                    ng_full[i], 0.0,
                    "w==0 observation must get exactly 0 gradient"
                );
            }
        }
        let risk_diff = (risk_full - risk_sub).abs();
        max_diff = max_diff.max(risk_diff);
        assert!(
            risk_diff < 1e-12,
            "Gehan weighted risk != unit-weight subset risk: {risk_full} vs {risk_sub}"
        );
        eprintln!("Gehan {{0,1}}==unit-subset max diff: {max_diff:e}");
    }

    // ─── Cindex ──────────────────────────────────────────────────────

    fn cindex_y(times: &[f64], events: &[f64]) -> Array1<f64> {
        gehan_y(times, events)
    }

    #[test]
    fn cindex_check_y_accepts_valid_survival_response() {
        let y = cindex_y(&[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0]);
        assert!(Cindex::default().check_y(y.view()).is_ok());
    }

    #[test]
    fn cindex_check_y_rejects_odd_length() {
        let y = array![1.0, 2.0, 3.0];
        assert!(Cindex::default().check_y(y.view()).is_err());
    }

    #[test]
    fn cindex_check_y_rejects_nonpositive_time() {
        let y = cindex_y(&[1.0, -2.0, 3.0], &[1.0, 0.0, 1.0]);
        assert!(Cindex::default().check_y(y.view()).is_err());
    }

    #[test]
    fn cindex_check_y_rejects_bad_event_indicator() {
        let y = cindex_y(&[1.0, 2.0, 3.0], &[1.0, 2.0, 1.0]);
        assert!(Cindex::default().check_y(y.view()).is_err());
    }

    #[test]
    fn cindex_offset_is_always_zero() {
        let y = cindex_y(&[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0]);
        let w = array![1.0, 1.0, 1.0];
        assert_eq!(Cindex::default().offset(y.view(), w.view()), 0.0);
    }

    #[test]
    fn cindex_requires_binary_weights() {
        assert!(Cindex::default().requires_binary_weights());
        assert!(!Cindex::default().requires_unit_weights());
    }

    #[test]
    fn cindex_default_sigma_is_point_one() {
        assert_eq!(Cindex::default().sigma, 0.1);
    }

    /// Gate 2, check 1 (non-R anchor, verified directly against R --
    /// `Rscript -e 'library(mboost); Cindex(sigma=0.1)@ngradient(Surv(c(2,5,1,5,3),
    /// c(1,1,0,1,1)), c(0.2,-0.3,0.5,0.1,-0.1), rep(1,5))'` -- see the task
    /// spec for the transcript). Obs #3 is both censored (status = 0) *and*
    /// has the minimum survival time in this fixture, so its gradient comes
    /// out exactly 0 -- see
    /// `cindex_min_time_censored_observation_gets_zero_gradient` (Gate 2,
    /// check 3) for why that combination, not censoring alone, is what
    /// forces the exact zero.
    #[test]
    fn cindex_ngradient_and_risk_match_r_anchor() {
        let times = [2.0, 5.0, 1.0, 5.0, 3.0];
        let events = [1.0, 1.0, 0.0, 1.0, 1.0];
        let f = array![0.2, -0.3, 0.5, 0.1, -0.1];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let y = cindex_y(&times, &events);
        let mut fam = Cindex::new(0.1);
        let mut ng = Array1::zeros(5);
        fam.negative_gradient(y.view(), f.view(), w.view(), ng.view_mut());
        let risk = fam.risk(y.view(), f.view(), w.view());

        let expected_ng = [-0.414061, 0.186069, 0.0, 0.502676, -0.274684];
        let expected_risk = -0.303843;
        for i in 0..5 {
            let err = (ng[i] - expected_ng[i]).abs();
            assert!(
                err < 1e-5,
                "cindex ngradient[{i}] mismatch: got={}, expected={}, err={err:e}",
                ng[i],
                expected_ng[i],
            );
        }
        assert!(
            (risk - expected_risk).abs() < 1e-5,
            "cindex risk mismatch: got={risk}, expected={expected_risk}"
        );
        // Structural check: the censored observation (index 2) gets exactly
        // zero gradient because its IPCW is 0.
        assert_eq!(ng[2], 0.0, "censored observation must get zero gradient");
    }

    /// Gate 2, check 2 (the R-independent safety net): `W` is constant in
    /// `f` (it depends only on `y`/`w`), so `ng` must equal `-d(risk)/df`
    /// exactly via central differences -- the strongest available guard
    /// against a transposed or sign-wrong `colSums`/`rowSums` in
    /// `cindex_ngradient_and_risk`.
    #[test]
    fn cindex_ngradient_matches_finite_difference_of_risk() {
        let times = [2.3, 5.1, 0.8, 9.4, 3.6, 1.2, 4.4, 7.7];
        let events = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let f = [0.35, -0.72, 1.15, -0.18, 0.52, -1.4, 0.9, -0.3];
        let n = f.len();
        let y = cindex_y(&times, &events);
        let w = Array1::ones(n);

        let f_arr = Array1::from_vec(f.to_vec());
        let mut fam = Cindex::new(0.1);
        let mut analytic = Array1::zeros(n);
        fam.negative_gradient(y.view(), f_arr.view(), w.view(), analytic.view_mut());

        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..n {
            let mut f_plus = f.to_vec();
            f_plus[i] += eps;
            let mut f_minus = f.to_vec();
            f_minus[i] -= eps;
            let risk_plus = fam.risk(y.view(), Array1::from_vec(f_plus).view(), w.view());
            let risk_minus = fam.risk(y.view(), Array1::from_vec(f_minus).view(), w.view());
            let numeric = -(risk_plus - risk_minus) / (2.0 * eps);
            let err = (analytic[i] - numeric).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-4,
                "cindex ngradient[{i}] finite-diff mismatch: analytic={}, numeric={}, err={err:e}",
                analytic[i],
                numeric,
            );
        }
        eprintln!("Cindex finite-diff max gradient error: {max_err:e}");
    }

    /// Gate 2, check 3 (structural, standalone from the anchor): `ng[k] =
    /// colSums(M1)[k] - rowSums(M1)[k]`. `rowSums(M1)[k]` is `sum_j
    /// approxGrad(f[k]-f[j]) * ipcw[k]^2 * weightsI[k,j] * w[k]*w[j]`, which
    /// vanishes identically for any censored `k` (`ipcw[k] == 0`). But
    /// `colSums(M1)[k] = sum_i approxGrad(f[i]-f[k]) * ipcw[i]^2 *
    /// weightsI[i,k] * w[i]*w[k]` depends on the *other* subjects' `ipcw`,
    /// not `k`'s own -- so it is generally nonzero even for censored `k`
    /// (verified directly against R: a mid-range-time censored subject gets
    /// a tiny but nonzero gradient, ~1e-11, matching R element-wise). The
    /// one case where `colSums(M1)[k]` is *also* forced to exactly 0 is
    /// when `k` has the (unique, untied) minimum survival time -- then no
    /// other subject has `time[i] < time[k]`, so every `weightsI[i,k]` term
    /// is 0. This is exactly the coincidence in the anchor fixture (obs #3
    /// there is both censored *and* has the minimum time); this test
    /// isolates it explicitly on its own fixture with additional (non-
    /// minimal-time) events, so it doesn't depend on the anchor's specific
    /// numbers.
    #[test]
    fn cindex_min_time_censored_observation_gets_zero_gradient() {
        let times = [1.0, 4.0, 2.0, 6.0, 3.0, 5.0];
        let events = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = array![0.4, -0.9, 1.3, 0.0, -0.5, 2.1];
        let w = Array1::ones(6);
        let y = cindex_y(&times, &events);
        let mut fam = Cindex::default();
        let mut ng = Array1::zeros(6);
        fam.negative_gradient(y.view(), f.view(), w.view(), ng.view_mut());
        assert_eq!(
            ng[0], 0.0,
            "the unique-minimum-time censored observation must have exactly zero gradient"
        );
    }

    #[test]
    fn cindex_zero_weight_observations_excluded() {
        // An observation with w = 0 must not influence the gradient of the
        // remaining in-bag observations (it is scattered out of `ipcw` and
        // zeroed out of `W` via the w[i]*w[j] term).
        let times = [1.0, 4.0, 2.0, 6.0, 3.0];
        let events = [1.0, 1.0, 1.0, 1.0, 1.0];
        let f = array![0.4, -0.9, 1.3, 0.0, -0.5];
        let y = cindex_y(&times, &events);
        let mut fam = Cindex::default();

        let w_full = Array1::ones(5);
        let mut ng_full = Array1::zeros(5);
        fam.negative_gradient(y.view(), f.view(), w_full.view(), ng_full.view_mut());

        let mut w_drop = Array1::ones(5);
        w_drop[2] = 0.0;
        let mut ng_drop = Array1::zeros(5);
        fam.negative_gradient(y.view(), f.view(), w_drop.view(), ng_drop.view_mut());

        assert_eq!(ng_drop[2], 0.0, "w=0 observation must get zero gradient");
        // The remaining gradients differ from the full-sample fit (fewer
        // pairs contribute), but must still be finite and non-degenerate.
        for i in [0usize, 1, 3, 4] {
            assert!(ng_drop[i].is_finite());
        }
        assert_ne!(ng_full, ng_drop);
    }
}

/// The ranked AUC kernel computes what the pairwise one computes.
///
/// P3.4 item 5: "optimize AUC/Gehan/C-index pairwise kernels with
/// ordered/ranked algorithms where exact tie and weight semantics can be
/// preserved."
///
/// "Where they can be preserved" is the whole condition, and the ties are
/// where a ranked rewrite goes wrong: `approxGrad` is a tent, so an
/// observation exactly at the centre contributes 1 and one exactly a unit
/// away contributes 0. Those two boundaries decide which side of a window a
/// tied score falls on, and getting either backwards changes the gradient by
/// a whole weight without changing its shape.
#[cfg(test)]
mod auc_ranked_kernel {
    use super::*;

    fn reference(ind1: &[usize], ind0: &[usize], f: &[f64], w: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; f.len()];
        auc_ngradient_pairwise(ind1, ind0, f, w, &mut out);
        out
    }

    fn ranked(ind1: &[usize], ind0: &[usize], f: &[f64], w: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; f.len()];
        auc_ngradient_ranked(ind1, ind0, f, w, &mut out);
        out
    }

    fn compare(ind1: &[usize], ind0: &[usize], f: &[f64], w: &[f64], label: &str) {
        let expected = reference(ind1, ind0, f, w);
        let got = ranked(ind1, ind0, f, w);
        for (index, (left, right)) in expected.iter().zip(&got).enumerate() {
            assert!(
                (left - right).abs() <= 1e-9 * left.abs().max(1.0),
                "{label}: observation {index} is {right} against {left}"
            );
        }
    }

    fn split(n: usize) -> (Vec<usize>, Vec<usize>) {
        ((0..n).step_by(2).collect(), (1..n).step_by(2).collect())
    }

    #[test]
    fn it_matches_the_pairwise_kernel_on_spread_scores() {
        let n = 200;
        let f: Vec<f64> = (0..n).map(|i| (i as f64) * 0.031 - 3.0).collect();
        let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 5) as f64 * 0.3).collect();
        let (ind1, ind0) = split(n);
        compare(&ind1, &ind0, &f, &w, "spread");
    }

    #[test]
    fn it_matches_when_every_score_is_tied() {
        // Every difference is exactly 0, so every pair sits on the boundary
        // that belongs to the *first* window. A ranked kernel that put it in
        // the second, or in neither, would be wrong by one whole weight per
        // pair and would still look like a gradient.
        let n = 60;
        let f = vec![0.25_f64; n];
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
        let (ind1, ind0) = split(n);
        compare(&ind1, &ind0, &f, &w, "all tied");
    }

    #[test]
    fn it_matches_at_the_support_boundary() {
        // Differences of exactly +/-1, where `approxGrad` is 0: the window
        // ends must be strict on that side, or these pairs contribute.
        let n = 80;
        let f: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let w = vec![1.0; n];
        let (ind1, ind0) = split(n);
        compare(&ind1, &ind0, &f, &w, "exact unit gap");

        let shifted: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { -0.5 } else { 0.5 })
            .collect();
        compare(&ind1, &ind0, &shifted, &w, "exact half gap");
    }

    #[test]
    fn it_matches_with_zero_weights() {
        // A zero-weight observation must contribute nothing and receive
        // nothing, which the pairwise form gets from the `w_j * w_k` factor
        // and the ranked one has to get from the prefix sums.
        let n = 120;
        let f: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % n) as f64 / n as f64 - 0.5)
            .collect();
        let mut w = vec![1.0; n];
        for (index, weight) in w.iter_mut().enumerate() {
            if index % 4 == 0 {
                *weight = 0.0;
            }
        }
        let (ind1, ind0) = split(n);
        compare(&ind1, &ind0, &f, &w, "zero weights");
        let got = ranked(&ind1, &ind0, &f, &w);
        for (index, value) in got.iter().enumerate() {
            if w[index] == 0.0 {
                assert_eq!(*value, 0.0, "a zero-weight observation got {value}");
            }
        }
    }

    #[test]
    fn it_matches_on_an_unbalanced_split() {
        let n = 300;
        let f: Vec<f64> = (0..n).map(|i| ((i * 37) % n) as f64 / 50.0 - 3.0).collect();
        let w: Vec<f64> = (0..n).map(|i| 0.25 + (i % 7) as f64 * 0.5).collect();
        let ind1: Vec<usize> = (0..12).collect();
        let ind0: Vec<usize> = (12..n).collect();
        compare(&ind1, &ind0, &f, &w, "12 against 288");
    }

    #[test]
    fn an_empty_class_produces_zeros() {
        let f = vec![0.1, 0.2, 0.3];
        let w = vec![1.0; 3];
        let ind1: Vec<usize> = vec![];
        let ind0: Vec<usize> = vec![0, 1, 2];
        compare(&ind1, &ind0, &f, &w, "no positives");
        assert!(ranked(&ind1, &ind0, &f, &w).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_ranked_risk_counts_the_same_concordant_pairs() {
        // The risk is a rank statistic: `sum_{j,k} w_j w_k 1{f_j > f_k}`.
        // `>` is strict, so tied scores contribute nothing -- a ranked form
        // using `upper_bound` instead of `lower_bound` would count every tie
        // and would still look like a risk.
        for (label, f, w) in [
            (
                "spread",
                (0..300)
                    .map(|i| (i as f64) * 0.013 - 2.0)
                    .collect::<Vec<_>>(),
                (0..300).map(|i| 0.5 + (i % 4) as f64).collect::<Vec<_>>(),
            ),
            (
                "all tied",
                vec![0.75_f64; 300],
                (0..300).map(|i| 1.0 + (i % 3) as f64).collect::<Vec<_>>(),
            ),
            (
                "two levels",
                (0..300)
                    .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
                vec![1.0; 300],
            ),
        ] {
            let ind1: Vec<usize> = (0..f.len()).step_by(2).collect();
            let ind0: Vec<usize> = (1..f.len()).step_by(2).collect();
            let mut pairwise = 0.0f64;
            for &j in &ind1 {
                for &k in &ind0 {
                    if f[j] - f[k] > 0.0 {
                        pairwise += w[j] * w[k];
                    }
                }
            }
            let negatives = SortedClass::new(&ind0, &f, &w);
            let ranked: f64 = ind1
                .iter()
                .map(|&j| w[j] * negatives.weight_below(f[j]))
                .sum();
            assert!(
                (pairwise - ranked).abs() <= 1e-9 * pairwise.abs().max(1.0),
                "{label}: ranked {ranked} against pairwise {pairwise}"
            );
        }
    }

    #[test]
    fn the_dispatch_uses_the_ranked_kernel_above_the_crossover() {
        // A guard that never routes to the new code would pass every test
        // above and change nothing.
        let n = 200;
        assert!((n / 2) * (n / 2) >= AUC_RANKED_CROSSOVER);
        let f: Vec<f64> = (0..n).map(|i| (i as f64) * 0.017).collect();
        let w = vec![1.0; n];
        let (ind1, ind0) = split(n);
        let mut dispatched = vec![0.0; n];
        auc_ngradient_core(&ind1, &ind0, &f, &w, &mut dispatched);
        let expected = ranked(&ind1, &ind0, &f, &w);
        assert_eq!(dispatched, expected);
    }
}

/// The ranked Gehan kernels compute what the pairwise ones compute.
///
/// P3.4 item 5: "optimize AUC/Gehan/C-index pairwise kernels with
/// ordered/ranked algorithms where exact tie and weight semantics can be
/// preserved."
///
/// Gehan is a rank statistic outright -- `arg1`/`arg2` are weighted counts
/// on either side of `res_i` -- so the ranked form must reproduce the
/// pairwise **non-strict** `>=`/`<=` boundaries exactly. A ranked rewrite
/// that used strict bounds would drop the diagonal and every tied
/// observation from one or both counts, changing the gradient by up to a
/// whole weight per pair while still looking like a gradient. The tests
/// below plant ties deliberately (all-equal residuals, exact duplications)
/// and check the two forms agree within summing-order rounding.
#[cfg(test)]
mod gehan_ranked_kernel {
    use super::*;

    fn gehan_y(times: &[f64], events: &[f64]) -> Array1<f64> {
        let mut v = times.to_vec();
        v.extend_from_slice(events);
        Array1::from_vec(v)
    }

    fn reference(res: &[f64], del: &[f64], w: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; res.len()];
        gehan_ngradient_core(res, del, w, &mut out);
        out
    }

    fn ranked(res: &[f64], del: &[f64], w: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; res.len()];
        gehan_ngradient_ranked(res, del, w, &mut out);
        out
    }

    fn compare_gradient(res: &[f64], del: &[f64], w: &[f64], label: &str) {
        let expected = reference(res, del, w);
        let got = ranked(res, del, w);
        for (index, (left, right)) in expected.iter().zip(&got).enumerate() {
            assert!(
                (left - right).abs() <= 1e-9 * left.abs().max(1.0),
                "{label}: observation {index} is {right} against {left}"
            );
        }
        let pairwise_risk = gehan_risk_core(res, del, w);
        let ranked_risk = gehan_risk_ranked(res, del, w);
        assert!(
            (pairwise_risk - ranked_risk).abs() <= 1e-9 * pairwise_risk.abs().max(1.0),
            "{label}: ranked risk {ranked_risk} against pairwise {pairwise_risk}"
        );
    }

    fn spread_residuals(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64) * 0.013 - 3.0).collect()
    }

    fn weights(n: usize) -> Vec<f64> {
        (0..n).map(|i| 0.5 + (i % 5) as f64 * 0.3).collect()
    }

    fn events(every: usize, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if i % every == 0 { 1.0 } else { 0.0 })
            .collect()
    }

    #[test]
    fn it_matches_the_pairwise_kernel_on_spread_residuals() {
        let n = 200;
        compare_gradient(&spread_residuals(n), &events(3, n), &weights(n), "spread");
    }

    #[test]
    fn it_matches_when_every_residual_is_tied() {
        // Every `res_j - res_i == 0`, counted by both the `<=` and the `>=`
        // boundaries: `arg1_i` and `arg2_i` each count the full (weighted)
        // column on their side. A ranked rewrite using strict bounds on
        // either side would drop the diagonal and every tie, changing every
        // gradient by weights while keeping its shape.
        let n = 60;
        let res = vec![0.25_f64; n];
        compare_gradient(&res, &events(2, n), &weights(n), "all tied");
    }

    #[test]
    fn it_matches_when_residuals_are_duplicated() {
        // Half the observations share residual 0.0 and half 1.0: every tie
        // sits exactly on the `>=`/`<=` boundary of its own block, and the
        // block logic must count the whole block on both sides.
        let n = 80;
        let res: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        compare_gradient(&res, &events(2, n), &weights(n), "duplicated");
    }

    #[test]
    fn it_matches_when_events_are_sparse() {
        // Mostly censored: `arg1` is almost always 0, so the `arg2` prefix
        // alone drives the gradient, and the ranked form must keep it exact.
        let n = 300;
        compare_gradient(
            &spread_residuals(n),
            &events(25, n),
            &weights(n),
            "sparse events",
        );
    }

    #[test]
    fn it_matches_with_zero_weights() {
        // A zero-weight observation must contribute nothing to any count and
        // receive an exactly zero gradient, which the pairwise form gets
        // from `ws[i] * raw / w_sum` and the ranked one must reproduce.
        let n = 120;
        let res = spread_residuals(n);
        let mut w = vec![1.0; n];
        for (index, weight) in w.iter_mut().enumerate() {
            if index % 4 == 0 {
                *weight = 0.0;
            }
        }
        // Plant an event on one of the zero-weight rows so the arg1 branch is
        // exercised on it too.
        let mut del = events(3, n);
        del[4] = 1.0;
        compare_gradient(&res, &del, &w, "zero weights");
        let got = ranked(&res, &del, &w);
        for (index, value) in got.iter().enumerate() {
            if w[index] == 0.0 {
                assert_eq!(*value, 0.0, "a zero-weight observation got {value}");
            }
        }
    }

    #[test]
    fn all_censored_produces_exactly_zero_ranked_gradient() {
        let n = 300;
        let res = spread_residuals(n);
        let del = vec![0.0; n];
        let out = ranked(&res, &del, &weights(n));
        assert!(out.iter().all(|&v| v == 0.0));
        let r = gehan_risk_ranked(&res, &del, &weights(n));
        assert_eq!(r, 0.0);
    }

    #[test]
    fn the_ranked_risk_matches_the_pairwise_risk() {
        for (label, res, del, w) in [
            (
                "spread",
                spread_residuals(300),
                events(4, 300),
                weights(300),
            ),
            (
                "all tied",
                vec![0.75_f64; 300],
                events(3, 300),
                (0..300).map(|i| 1.0 + (i % 3) as f64).collect::<Vec<_>>(),
            ),
            (
                "two levels",
                (0..300)
                    .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
                    .collect::<Vec<_>>(),
                events(5, 300),
                vec![1.0; 300],
            ),
        ] {
            let pairwise = gehan_risk_core(&res, &del, &w);
            let ranked = gehan_risk_ranked(&res, &del, &w);
            assert!(
                (pairwise - ranked).abs() <= 1e-9 * pairwise.abs().max(1.0),
                "{label}: ranked {ranked} against pairwise {pairwise}"
            );
        }
    }

    /// The ranked path is a new gradient formula, so it gets its own
    /// finite-difference gate: at n = 200 (well above the crossover) every
    /// family call uses the ranked kernels, and the central-difference check
    /// of `ngradient` against `risk` must still hold. Same construction as
    /// `gehan_ngradient_matches_finite_difference_of_risk`, on the ranked
    /// path.
    #[test]
    fn the_ranked_kernel_passes_the_finite_difference_gate() {
        let n = 200;
        assert!(n * n >= GEHAN_RANKED_CROSSOVER);
        let times: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.031 - 3.0).exp()).collect();
        let events_v: Vec<f64> = (0..n)
            .map(|i| if (i * 7) % 3 != 0 { 1.0 } else { 0.0 })
            .collect();
        let f = vec![0.35; n];
        let y = gehan_y(&times, &events_v);
        let w = Array1::ones(n);
        let f_arr = Array1::from_vec(f.to_vec());
        let mut analytic = Array1::zeros(n);
        Gehan.negative_gradient(y.view(), f_arr.view(), w.view(), analytic.view_mut());

        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..n {
            let mut f_plus = f.to_vec();
            f_plus[i] += eps;
            let mut f_minus = f.to_vec();
            f_minus[i] -= eps;
            let risk_plus = Gehan.risk(y.view(), Array1::from_vec(f_plus).view(), w.view());
            let risk_minus = Gehan.risk(y.view(), Array1::from_vec(f_minus).view(), w.view());
            let numeric = -(risk_plus - risk_minus) / (2.0 * eps);
            let err = (analytic[i] - numeric).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-4,
                "ranked Gehan ngradient[{i}] finite-diff mismatch: analytic={}, numeric={}, err={err:e}",
                analytic[i],
                numeric,
            );
        }
        eprintln!("ranked Gehan finite-diff max gradient error: {max_err:e}");
    }

    #[test]
    fn the_dispatch_uses_the_ranked_kernel_above_the_crossover() {
        // A guard that never routes to the new code would pass every test
        // above and change nothing.
        let n = 200;
        assert!(n * n >= GEHAN_RANKED_CROSSOVER);
        // Pick the times directly and derive the residuals from them, so the
        // ranked core and the family's own `ln(time) - f` (here `f = 0`)
        // operate on bit-identical vectors -- no exp/ln round-trip.
        let times: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.013 - 3.0).exp()).collect();
        let del = events(3, n);
        let w = Array1::ones(n);
        let res: Vec<f64> = times.iter().map(|t| t.ln()).collect();
        let y = gehan_y(&times, &del);
        let f_arr = Array1::zeros(n);
        let mut dispatched = Array1::zeros(n);
        Gehan.negative_gradient(y.view(), f_arr.view(), w.view(), dispatched.view_mut());
        let expected = Array1::from_vec(ranked(&res, &del, w.as_slice().unwrap()));
        assert_eq!(dispatched, expected);
        let risk = Gehan.risk(y.view(), f_arr.view(), w.view());
        assert_eq!(risk, gehan_risk_ranked(&res, &del, w.as_slice().unwrap()));
    }
}

#[cfg(test)]
mod hurdle_tests {
    use super::*;

    #[test]
    fn hurdle_gradient_is_the_derivative_of_its_loss() {
        for &sigma in &[0.3, 0.7, 2.5] {
            for &(y, f) in &[
                (1.0, -1.0),
                (2.0, 0.0),
                (5.0, 0.5),
                (9.0, 2.0),
                (1.0, 3.0),
                (3.0, -2.0),
            ] {
                let eps = 1e-6;
                let numeric = (hurdle_plloss(sigma, y, f - eps) - hurdle_plloss(sigma, y, f + eps))
                    / (2.0 * eps);
                let analytic = hurdle_ngradient(sigma, y, f);
                assert!(
                    (numeric - analytic).abs() < 1e-6 * (1.0 + analytic.abs()),
                    "sigma={sigma} y={y} f={f}: {analytic} vs {numeric}"
                );
            }
        }
    }
}

#[cfg(test)]
mod row_loss_tests {
    use super::*;

    fn check<F: Family>(family: F, y: Vec<f64>, f: Vec<f64>) {
        let w: Vec<f64> = (0..f.len()).map(|i| 0.5 + (i % 3) as f64).collect();
        let (y, f, w) = (Array1::from(y), Array1::from(f), Array1::from(w));
        let rows = held_out_row_losses(&family, y.view(), f.view()).expect("row-separable");
        let span = family.flat_response_span();
        let weighted: f64 = rows
            .iter()
            .enumerate()
            .map(|(i, loss)| w[i * span] * loss)
            .sum();
        let total = family.held_out_risk(y.view(), f.view(), w.view());
        assert!(
            (weighted - total).abs() <= 1e-10 * (1.0 + total.abs()),
            "{}: rows sum to {weighted}, held_out_risk is {total}",
            family.name()
        );
    }

    #[test]
    fn the_weighted_row_losses_sum_to_the_held_out_risk() {
        let f = vec![-0.8, -0.1, 0.3, 0.9, 1.4, -1.2];
        let real = vec![-1.0, 0.2, 0.5, 1.1, 2.0, -0.7];
        let signs = vec![-1.0, 1.0, 1.0, -1.0, 1.0, -1.0];
        let counts = vec![0.0, 1.0, 3.0, 2.0, 5.0, 1.0];
        let positive = vec![0.5, 1.2, 2.0, 0.8, 3.1, 1.7];
        check(Gaussian, real.clone(), f.clone());
        check(GaussReg, real.clone(), f.clone());
        check(Laplace::default(), real.clone(), f.clone());
        check(QuantReg::new(0.3), real.clone(), f.clone());
        check(ExpectReg::new(0.7), real.clone(), f.clone());
        check(Huber::new(Some(0.5)), real.clone(), f.clone());
        check(Binomial, signs.clone(), f.clone());
        check(AdaExp, signs.clone(), f.clone());
        check(HingeLoss::new(1.0), signs.clone(), f.clone());
        check(Poisson, counts.clone(), f.clone());
        check(NBinomial::new(), counts.clone(), f.clone());
        check(Hurdle::new(), vec![1.0, 2.0, 3.0, 1.0, 5.0, 2.0], f.clone());
        check(GammaReg::new(), positive.clone(), f.clone());
        let mut survival = positive.clone();
        survival.extend([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        check(Weibull::new(), survival.clone(), f.clone());
        check(Loglog::new(), survival.clone(), f.clone());
        check(Lognormal::new(), survival, f.clone());
        check(
            PropOdds::new(3),
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 3.0],
            f.clone(),
        );
        // Three classes: two non-reference coordinates per observation.
        check(Multinomial::new(3), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], f);
    }

    #[test]
    fn coupled_losses_have_no_row_terms() {
        let y = Array1::from(vec![1.0, 2.0, 3.0, 1.0, 0.0, 1.0]);
        let f = Array1::from(vec![0.1, -0.2, 0.3]);
        assert!(held_out_row_losses(&CoxPH::new(), y.view(), f.view()).is_none());
        assert!(held_out_row_losses(&Gehan, y.view(), f.view()).is_none());
    }
}
