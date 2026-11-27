#![allow(unused_imports)]
use super::*;

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
            (digamma(1.0) - (-std::f64::consts::EULER_GAMMA)).abs() < 1e-10,
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
mod response_scale_tests {
    use super::*;

    /// A deterministic pseudo-random stream in (0, 1).
    fn uniforms(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            })
            .collect()
    }

    /// Responses on each family's support, with the scaled part first.
    fn response_for(family: &dyn LssFamily, n: usize) -> Vec<f64> {
        let u = uniforms(2 * n, 7);
        let positive: Vec<f64> = u[..n].iter().map(|v| 0.2 + 5.0 * v).collect();
        match family.name() {
            "GaussianLSS" | "StudentTLSS" | "GumbelLSS" | "LogisticLSS" | "JSULSS" => {
                u[..n].iter().map(|v| 10.0 * (v - 0.5)).collect()
            }
            "GaussianKnownVarLSS" => {
                let mut y: Vec<f64> = u[..n].iter().map(|v| 4.0 * (v - 0.5)).collect();
                y.extend(u[n..].iter().map(|v| 0.05 + v));
                y
            }
            "ZeroAdjustedGammaLSS" => positive
                .iter()
                .enumerate()
                .map(|(i, v)| if i % 4 == 0 { 0.0 } else { *v })
                .collect(),
            _ => positive,
        }
    }

    fn check_identity(family: &dyn LssFamily, y: &[f64], n: usize) {
        let map = family.scale_map().expect("mapped family");
        let k = family.n_params();
        let w: Vec<f64> = uniforms(n, 11).iter().map(|v| 0.5 + v).collect();
        let u = uniforms(n * k, 13);
        // Internal predictors in a range every family evaluates finitely.
        let etas_s: Vec<Array1<f64>> = (0..k)
            .map(|p| {
                Array1::from_iter((0..n).map(|i| {
                    let base = 0.6 * (u[p * n + i] - 0.5);
                    if family.name() == "StudentTLSS" && p == 2 {
                        1.5 + base // df > 1
                    } else if matches!(family.name(), "BCTLSS" | "BCPELSS") && p == 0 {
                        2.0 + base // identity-link median, positive
                    } else {
                        base
                    }
                }))
            })
            .collect();
        for s in [0.01, 3.7, 1.0e4] {
            let y_s = family.scale_response(y, n, s);
            let etas: Vec<Array1<f64>> = etas_s
                .iter()
                .zip(map)
                .map(|(e, m)| {
                    let (a, b) = m.affine(s);
                    e.mapv(|v| a * v + b)
                })
                .collect();
            let original = family.risk(y, &etas, &w);
            let internal = family.risk(&y_s, &etas_s, &w);
            let shift = s.ln() * family.scale_jacobian_weight(y, &w);
            assert!(
                original.is_finite() && internal.is_finite(),
                "{} risk not finite at s={s}",
                family.name()
            );
            let tol = 1e-9 * (1.0 + original.abs());
            assert!(
                (original - (internal + shift)).abs() < tol,
                "{} at s={s}: risk(y)={original} but risk(y/s)+shift={}",
                family.name(),
                internal + shift
            );
        }
    }

    #[test]
    fn every_scale_map_reproduces_the_likelihood_in_original_units() {
        let n = 40;
        let families: Vec<Box<dyn LssFamily>> = vec![
            Box::new(GaussianLSS),
            Box::new(GaussianKnownVarLSS),
            Box::new(GammaLSS),
            Box::new(StudentTLSS),
            Box::new(BCCGLSS),
            Box::new(GumbelLSS),
            Box::new(InverseGaussianLSS),
            Box::new(LogisticLSS),
            Box::new(ZeroAdjustedGammaLSS),
            Box::new(JSULSS),
            Box::new(BCTLSS),
            Box::new(BCPELSS),
            Box::new(Pareto2LSS),
            Box::new(SurvivalLSS::log_normal().uncensored()),
            Box::new(SurvivalLSS::log_log().uncensored()),
            Box::new(SurvivalLSS::weibull().uncensored()),
        ];
        for family in &families {
            let y = response_for(family.as_ref(), n);
            check_identity(family.as_ref(), &y, n);
        }
        // Censored survival: `[time.., status..]`.
        for family in [
            SurvivalLSS::log_normal(),
            SurvivalLSS::log_log(),
            SurvivalLSS::weibull(),
        ] {
            let mut y: Vec<f64> = uniforms(n, 5).iter().map(|v| 0.2 + 5.0 * v).collect();
            y.extend((0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }));
            check_identity(&family, &y, n);
        }
    }

    fn bols_learners(n: usize, w: ArrayView1<f64>) -> Vec<Vec<PreparedLearner>> {
        let x = Array1::<f64>::linspace(-1.0, 1.0, n);
        vec![
            vec![crate::learners::prepare_scalar_bols(
                x.view(),
                w,
                0.0,
                true,
                "mu",
                None,
            )],
            vec![crate::learners::prepare_scalar_bols(
                x.view(),
                w,
                0.0,
                true,
                "sigma",
                None,
            )],
        ]
    }

    /// A heteroscedastic Gaussian response in units of `unit`.
    fn response(n: usize, unit: f64) -> Array1<f64> {
        let u = uniforms(n, 3);
        Array1::from_iter((0..n).map(|i| {
            let x = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let z = (u[i] - 0.5) * 3.4;
            unit * (1.0 + 0.5 * x + (-1.0 + 0.5 * x).exp() * z)
        }))
    }

    fn assert_close(a: f64, b: f64, what: &str) {
        assert!(
            (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs())),
            "{what}: {a} vs {b}"
        );
    }

    /// Fitting `c * y` with scale `c * s` is the fit of `y` with scale `s`,
    /// mapped: mu by `c`, log sigma by `ln c`, the risk by `n ln c`.
    #[test]
    fn a_scaled_fit_is_equivariant_in_the_response_units() {
        let n = 120;
        let c = 1.0e5;
        let w = Array1::ones(n);
        let y = response(n, 1.0);
        let yc = response(n, c);
        let s = auto_response_scale(&GaussianLSS, y.as_slice().unwrap(), w.as_slice().unwrap())
            .unwrap();
        for method in ["cyclic", "noncyclic"] {
            let fit = |yv: &Array1<f64>, scale: f64| {
                let control = BoostControl::new(60, 0.1)
                    .unwrap()
                    .with_response_scale(ResponseScale::Fixed(scale))
                    .unwrap();
                let mut learners = bols_learners(n, w.view());
                if method == "cyclic" {
                    try_fit_lss(
                        &mut learners,
                        yv.view(),
                        w.view(),
                        &GaussianLSS,
                        &control,
                        None,
                        None,
                    )
                } else {
                    try_fit_lss_noncyclic(
                        &mut learners,
                        yv.view(),
                        w.view(),
                        &GaussianLSS,
                        &control,
                        None,
                    )
                }
                .unwrap()
            };
            let unit = fit(&y, s);
            let big = fit(&yc, c * s);
            assert_eq!(unit.selection_order, big.selection_order, "{method}");
            for i in 0..n {
                assert_close(
                    big.per_param[0].fitted[i],
                    c * unit.per_param[0].fitted[i],
                    "mu",
                );
                assert_close(
                    big.per_param[1].fitted[i],
                    unit.per_param[1].fitted[i] + c.ln(),
                    "log sigma",
                );
            }
            assert_close(
                big.per_param[0].offset,
                c * unit.per_param[0].offset,
                "mu offset",
            );
            for (a, b) in big.per_param[0].coefs.iter().zip(&unit.per_param[0].coefs) {
                for (x, y) in a.iter().zip(b) {
                    assert_close(*x, c * y, "mu coefficient");
                }
            }
            for (a, b) in big.per_param[1].coefs.iter().zip(&unit.per_param[1].coefs) {
                for (x, y) in a.iter().zip(b) {
                    assert_close(*x, *y, "sigma coefficient");
                }
            }
            for (a, b) in big.risk.iter().zip(&unit.risk) {
                assert_close(*a, b + n as f64 * c.ln(), "risk");
            }
            // The reported risk is the likelihood of the mapped predictors.
            let etas: Vec<Array1<f64>> = big.per_param.iter().map(|p| p.fitted.clone()).collect();
            assert_close(
                *big.risk.last().unwrap(),
                GaussianLSS.risk(yc.as_slice().unwrap(), &etas, w.as_slice().unwrap()),
                "final risk",
            );
        }
    }

    /// The same scaled fit through the checkpointed CV path reproduces the
    /// mapped predictors of a plain fit at each checkpoint.
    #[test]
    fn checkpoint_snapshots_are_in_the_original_units() {
        let n = 80;
        let c = 250.0;
        let w = Array1::ones(n);
        let y = response(n, c);
        let control = BoostControl::new(30, 0.1)
            .unwrap()
            .with_response_scale(ResponseScale::Fixed(c))
            .unwrap();
        let mut learners = bols_learners(n, w.view());
        let snaps = try_fit_lss_diagonal_checkpoints(
            &mut learners,
            y.view(),
            w.view(),
            &GaussianLSS,
            &control,
            &[10, 30],
            None,
        )
        .unwrap();
        for (m, etas) in snaps {
            let control_m = BoostControl::new(m as u32, 0.1)
                .unwrap()
                .with_response_scale(ResponseScale::Fixed(c))
                .unwrap();
            let mut learners = bols_learners(n, w.view());
            let path = try_fit_lss(
                &mut learners,
                y.view(),
                w.view(),
                &GaussianLSS,
                &control_m,
                None,
                None,
            )
            .unwrap();
            for p in 0..2 {
                for i in 0..n {
                    assert_close(etas[p][i], path.per_param[p].fitted[i], "snapshot");
                }
            }
        }
    }

    /// In large units the raw fit's mean hardly leaves its offset; the scaled
    /// fit reaches the same mean it reaches in unit scale.
    #[test]
    fn the_raw_fit_stalls_in_large_units_and_the_scaled_fit_does_not() {
        let n = 120;
        let c = 1.0e5;
        let w = Array1::ones(n);
        let yc = response(n, c);
        let offset = yc.mean().unwrap();
        let spread = |path: &LssPath| {
            let mu = &path.per_param[0].fitted;
            mu.iter().map(|m| (m - offset).abs()).fold(0.0, f64::max) / c
        };
        let fit = |scale: ResponseScale| {
            let control = BoostControl::new(200, 0.1)
                .unwrap()
                .with_response_scale(scale)
                .unwrap();
            let mut learners = bols_learners(n, w.view());
            try_fit_lss(
                &mut learners,
                yc.view(),
                w.view(),
                &GaussianLSS,
                &control,
                None,
                None,
            )
            .unwrap()
        };
        assert!(
            spread(&fit(ResponseScale::Raw)) < 1e-6,
            "the raw fit moved {}",
            spread(&fit(ResponseScale::Raw))
        );
        assert!(
            spread(&fit(ResponseScale::Auto)) > 0.3,
            "the scaled fit moved {}",
            spread(&fit(ResponseScale::Auto))
        );
    }

    /// A non-cyclic fit to 10 steps is the first 10 steps of the fit to 25.
    #[test]
    fn a_noncyclic_fit_is_a_prefix_of_a_longer_one() {
        let n = 90;
        let w = Array1::ones(n);
        let y = response(n, 1.0);
        let fit = |m: u32| {
            let control = BoostControl::new(m, 0.1).unwrap();
            let mut learners = bols_learners(n, w.view());
            try_fit_lss_noncyclic(
                &mut learners,
                y.view(),
                w.view(),
                &GaussianLSS,
                &control,
                None,
            )
            .unwrap()
        };
        let short = fit(10);
        let long = fit(25);
        assert_eq!(short.selection_order[..], long.selection_order[..10]);
        assert_eq!(short.risk[..], long.risk[..11]);
    }

    /// The resumable non-cyclic search reads the same held-out risk as a fit
    /// refitted from zero to each total, with and without a response scale.
    #[test]
    fn the_noncyclic_search_matches_refits_from_zero() {
        use crate::grid_search::GridStep;
        let n = 90;
        let w = Array1::ones(n);
        for (unit, scale) in [
            (1.0, ResponseScale::Raw),
            (300.0, ResponseScale::Fixed(300.0)),
            (300.0, ResponseScale::Auto),
        ] {
            let y = response(n, unit);
            let b = 3;
            let mut folds = ndarray::Array2::<f64>::ones((n, b));
            for i in 0..n {
                folds[[i, i % b]] = 0.0;
            }
            let control = BoostControl::new(1, 0.1)
                .unwrap()
                .with_response_scale(scale)
                .unwrap();
            let schedule = vec![
                GridStep {
                    end: 8,
                    grid: (1..=8).collect(),
                },
                GridStep {
                    end: 16,
                    grid: (1..=16).collect(),
                },
            ];
            let found = try_cvrisk_lss_search(
                |fw| Ok(bols_learners(n, fw)),
                y.view(),
                w.view(),
                &GaussianLSS,
                &control,
                true,
                folds.view(),
                None,
                &schedule,
                0.05,
            )
            .unwrap();
            for fold in 0..b {
                let fold_w = folds.column(fold).to_owned();
                let oob: Vec<f64> = (0..n).map(|i| 1.0 - fold_w[i]).collect();
                let oob_sum: f64 = oob.iter().sum();
                for (j, &m) in found.grid.iter().enumerate() {
                    let control_m = BoostControl::new(m as u32, 0.1)
                        .unwrap()
                        .with_response_scale(scale)
                        .unwrap();
                    let mut learners = bols_learners(n, fold_w.view());
                    let path = try_fit_lss_noncyclic(
                        &mut learners,
                        y.view(),
                        fold_w.view(),
                        &GaussianLSS,
                        &control_m,
                        None,
                    )
                    .unwrap();
                    let etas: Vec<Array1<f64>> =
                        path.per_param.iter().map(|p| p.fitted.clone()).collect();
                    let want = GaussianLSS.risk(y.as_slice().unwrap(), &etas, &oob) / oob_sum;
                    assert_close(found.risk[fold][j], want, "held-out risk");
                }
            }
        }
    }

    #[test]
    fn unmapped_families_are_left_alone() {
        let control = BoostControl::new(10, 0.1)
            .unwrap()
            .with_response_scale(ResponseScale::Fixed(5.0))
            .unwrap();
        for family in [
            Box::new(PoissonLSS) as Box<dyn LssFamily>,
            Box::new(BetaLSS),
            Box::new(NBinomialLSS),
        ] {
            assert!(
                ResponseScaling::resolve(family.as_ref(), &control, &[1.0, 2.0], &[1.0; 2])
                    .unwrap()
                    .is_none()
            );
            assert!(auto_response_scale(family.as_ref(), &[1.0, 2.0, 3.0], &[1.0; 3]).is_none());
        }
    }
}
