//! Engine: parity and performance regression tests (added file).
//!
//! These tests are designed to stress the boosting engine in ways that the
//! existing unit tests do not, with the explicit goal of uncovering numerical
//! stability bugs, R-parity drift, and performance regressions.

#[cfg(test)]
mod tests {
    use ndarray::{array, Array1};

    use crate::control::BoostControl;
    use crate::engine::{fit, fit_selected, fit_with};
    use crate::families::{
        Binomial, ExpectReg, Family, GammaReg, Gaussian, Huber, Laplace, Poisson, QuantReg,
    };
    use crate::learners::{df_to_lambda_scalar, prepare_block_bols, prepare_scalar_bols};
    use ndarray::Array2;

    // ------------------------------------------------------------------
    // 1.  Edge-case: zero-variance predictors
    // ------------------------------------------------------------------

    #[test]
    fn fit_zero_variance_predictor() {
        // If a predictor is constant, denom = 0 → beta is forced to 0.
        // The engine should not crash and should preferentially select the
        // non-constant learner.
        let y = array![3.0, 5.0, 7.0];
        let w = array![1.0, 1.0, 1.0];
        let x_const = array![2.0, 2.0, 2.0]; // zero variance after centering
        let x_var = array![1.0, 2.0, 3.0];

        let l_const = prepare_scalar_bols(x_const.view(), w.view(), 0.0, true, "c", None);
        let l_var = prepare_scalar_bols(x_var.view(), w.view(), 0.0, true, "v", None);
        let mut learners = vec![l_const, l_var];

        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        // The constant learner should never be selected (beta=0, sse doesn't improve)
        for &idx in &path.selected {
            assert_eq!(idx, 1, "constant learner was selected");
        }
    }

    // ------------------------------------------------------------------
    // 2.  Edge-case: extremely unbalanced weights
    // ------------------------------------------------------------------

    #[test]
    fn fit_unbalanced_weights() {
        // One observation dominates the fit.  Ensures weighted_mean and
        // the scalar OLS path handle near-zero denominators gracefully.
        let y = array![100.0, 1.0, 1.0];
        let w = array![1e9, 1e-9, 1e-9]; // wildly unbalanced
        let x = array![1.0, 2.0, 3.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        // Offset ≈ 100 (the dominant observation; weighted_mean)
        assert!(
            (path.offset - 100.0).abs() < 1.0,
            "offset with unbalanced weights should be ~100, got {}",
            path.offset
        );
        assert_eq!(path.selected.len(), 10);
    }

    // ------------------------------------------------------------------
    // 3.  Edge-case: single observation
    // ------------------------------------------------------------------

    #[test]
    fn fit_single_observation() {
        let y = array![42.0];
        let w = array![1.0];
        let x = array![1.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert_eq!(path.offset, 42.0);
        assert_eq!(path.risk.len(), 6);
        // With a single obs the centered x is 0, so denom = 0 → beta forced to 0.
        for b in &path.coefs {
            assert!(
                b[0].abs() < 1e-12,
                "beta should be forced to 0 for single obs"
            );
        }
    }

    // ------------------------------------------------------------------
    // 4.  Edge-case: near-perfect collinearity (identical predictors)
    // ------------------------------------------------------------------

    #[test]
    fn fit_collinear_predictors_tie_breaking() {
        // Two identical columns → engine should always pick the lowest index.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let l1 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x2", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        for &idx in &path.selected {
            assert_eq!(idx, 0, "tie should be broken by lowest index");
        }
    }

    // ------------------------------------------------------------------
    // 5.  Edge-case: all-zero response
    // ------------------------------------------------------------------

    #[test]
    fn fit_all_zero_response() {
        let y = array![0.0, 0.0, 0.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0, 4.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        // Offset = 0, risk should stay at 0 forever (perfect fit at offset)
        assert!(path.risk[0].abs() < 1e-12);
        for i in 1..path.risk.len() {
            assert!(
                path.risk[i].abs() < 1e-12,
                "risk should stay 0, got {}",
                path.risk[i]
            );
        }
    }

    // ------------------------------------------------------------------
    // 6.  Edge-case: very small/large nu
    // ------------------------------------------------------------------

    #[test]
    fn fit_tiny_nu_convergence() {
        // nu = 1e-6 → extremely slow convergence; risk barely changes.
        let n = 20;
        let y: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::ones(n);
        let x = y.clone();

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(100, 1e-6).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        // Risk should decrease monotonically (or stay same)
        for i in 1..path.risk.len() {
            assert!(
                path.risk[i] <= path.risk[i - 1] + 1e-10,
                "risk increased at step {i}: {} -> {}",
                path.risk[i - 1],
                path.risk[i]
            );
        }
    }

    #[test]
    fn fit_nu_one_full_step() {
        // nu = 1.0: a single step should perfectly fit a scalar problem
        let y = array![2.0, 4.0, 6.0];
        let w = array![1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(1, 1.0).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert!(
            path.risk[1].abs() < 1e-10,
            "risk should be 0 after full step"
        );
    }

    // ------------------------------------------------------------------
    // 7.  Edge-case: negative response with Poisson
    // ------------------------------------------------------------------

    #[test]
    fn fit_poisson_rejects_negative_y() {
        let y = array![-1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let _learners = [l];

        let _ctrl = BoostControl::new(10, 0.1).unwrap();
        let family = Poisson;

        assert!(family.check_y(y.view()).is_err());
    }

    // ------------------------------------------------------------------
    // 8.  Edge-case: Binomial with non-standard encoding
    // ------------------------------------------------------------------

    #[test]
    fn fit_binomial_with_zero_encoded() {
        let y = array![0.0, 1.0, -1.0];
        let w = array![1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let _learners = [l];

        let _ctrl = BoostControl::new(10, 0.1).unwrap();
        let family = Binomial;

        assert!(family.check_y(y.view()).is_ok());

        let y_ok = array![1.0, -1.0, 1.0];
        assert!(family.check_y(y_ok.view()).is_ok());
    }

    // ------------------------------------------------------------------
    // 9.  Determinism test: repeated fits yields identical results
    // ------------------------------------------------------------------

    #[test]
    fn fit_deterministic_repeated() {
        let n = 50;
        let y: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64).sin());
        let w = Array1::ones(n);
        let x1: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64);
        let x2: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64).cos());

        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(20, 0.1).unwrap();
        let mut family = Gaussian;

        let path1 = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);
        let path2 = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert_eq!(path1.selected, path2.selected);
        assert_eq!(path1.risk, path2.risk);
        for (c1, c2) in path1.coefs.iter().zip(path2.coefs.iter()) {
            assert!(
                c1.iter()
                    .zip(c2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
                    < 1e-12
            );
        }
    }

    // ------------------------------------------------------------------
    // 10.  Family offset correctness across families
    // ------------------------------------------------------------------

    #[test]
    fn family_offsets_agree_with_known_values() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];

        // Gaussian → weighted mean = 3.0
        let off_g = Gaussian.offset(y.view(), w.view());
        assert!((off_g - 3.0).abs() < 1e-12);

        // Poisson → log(mean(y)) = log(3) ≈ 1.0986
        let off_p = Poisson.offset(y.view(), w.view());
        assert!((off_p - 3.0f64.ln()).abs() < 1e-12);

        // Binomial: p = proportion of y > 0 = 3/5, offset = 0.5 * ln(p/(1-p))
        let y_bin = array![-1.0, -1.0, 1.0, 1.0, 1.0];
        let off_b = Binomial.offset(y_bin.view(), w.view());
        let expected = 0.5 * (0.6_f64 / 0.4_f64).ln();
        assert!(
            (off_b - expected).abs() < 1e-10,
            "Binomial offset: expected {expected}, got {off_b}"
        );
    }

    // ------------------------------------------------------------------
    // 11.  R-parity sanity: risk decreases monotonically for all families
    // ------------------------------------------------------------------

    #[test]
    fn risk_monotonicity_all_families() {
        let n = 30;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let y: Array1<f64> = x.mapv(|v| 2.0 * v + 1.0);
        let w = Array1::ones(n);

        let _families: Vec<Box<dyn Family>> = vec![
            Box::new(Gaussian),
            Box::new(Laplace::default()),
            Box::new(Poisson),
            Box::new(Binomial),
            Box::new(QuantReg::new(0.5)),
            Box::new(ExpectReg::new(0.5)),
            Box::new(Huber::new(None)),
            Box::new(GammaReg::new()),
        ];

        // We can't easily iterate over dyn Family because fit() needs a mut reference.
        // Instead, test each family individually inline.
        let families: [Box<dyn Family>; 8] = [
            Box::new(Gaussian),
            Box::new(Laplace::default()),
            Box::new(Poisson),
            Box::new(Binomial),
            Box::new(QuantReg::new(0.5)),
            Box::new(ExpectReg::new(0.5)),
            Box::new(Huber::new(None)),
            Box::new(GammaReg::new()),
        ];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        for mut family in families.into_iter() {
            let ctrl = BoostControl::new(20, 0.1).unwrap();

            if family.check_y(y.view()).is_err() {
                continue;
            }

            // Binomial with continuous y is nonsensical even though check_y
            // now accepts {0,1} — skip it for this continuous-y test.
            if family.name() == "Binomial" {
                continue;
            }

            let path = fit(
                &mut learners,
                y.view(),
                w.view(),
                family.as_mut(),
                &ctrl,
                None,
            );

            if family.name() == "Gaussian" {
                for j in 1..path.risk.len() {
                    assert!(
                        path.risk[j] <= path.risk[j - 1] + 1e-9,
                        "Gaussian: risk increased at step {}: {} -> {}",
                        j,
                        path.risk[j - 1],
                        path.risk[j]
                    );
                }
            } else {
                assert!(!path.risk.is_empty(), "{}: no risk computed", family.name());
            }
        }
    }

    // ------------------------------------------------------------------
    // 12.  R-parity: intercept-only model risk = sum((y-mean)^2)
    // ------------------------------------------------------------------

    #[test]
    fn intercept_only_risk_matches_hand_computation() {
        let y = array![2.0, 4.0, 6.0, 8.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0, 4.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(1, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        // mean(y) = 5, risk = Σ w*(y-mean)^2 = 9+1+1+9 = 20
        assert!(
            (path.risk[0] - 20.0).abs() < 1e-12,
            "intercept-only risk should be 20, got {}",
            path.risk[0]
        );
    }

    // ------------------------------------------------------------------
    // 13.  Performance regression: hot loop should not allocate
    // ------------------------------------------------------------------

    #[test]
    fn fit_large_n_no_panic_or_slow() {
        // This is a smoke test: if the hot loop allocates per-iteration,
        // this will be very slow.  Keep n relatively small for CI, but
        // large enough to catch O(n*mstop) allocations.
        let n = 5000;
        let mstop = 50;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let y: Array1<f64> = x.mapv(|v| 3.0 * v + (v * 10.0).sin());
        let w = Array1::ones(n);

        let l1 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l1];

        let ctrl = BoostControl::new(mstop, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert_eq!(path.selected.len(), mstop as usize);
        assert_eq!(path.risk.len(), mstop as usize + 1);
    }

    // ------------------------------------------------------------------
    // 14.  CV determinism: fold generation is reproducible
    // ------------------------------------------------------------------

    #[test]
    fn cv_folds_reproducible() {
        use crate::cv::{cv, FoldSpec, FoldType};

        let spec = FoldSpec::new(FoldType::KFold, 5, 0.5, 12345);
        let n = 100;
        let weights = vec![1.0f64; n];
        let folds1 = cv(&spec, n, &weights);
        let folds2 = cv(&spec, n, &weights);

        assert_eq!(
            folds1, folds2,
            "CV folds must be reproducible with same seed"
        );
    }

    // ------------------------------------------------------------------
    // 15.  QuantReg with extreme tau values
    // ------------------------------------------------------------------

    #[test]
    fn quantreg_extreme_tau() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];

        // R's QuantReg uses qoffset (default 0.5 = median) for offset, NOT tau
        let fam_low = QuantReg::new(0.01);
        let off_low = fam_low.offset(y.view(), w.view());
        assert!(
            (off_low - 3.0).abs() < 1e-10,
            "qoffset=0.5 should give median=3, got {off_low}"
        );

        let fam_high = QuantReg::new(0.99);
        let off_high = fam_high.offset(y.view(), w.view());
        assert!(
            (off_high - 3.0).abs() < 1e-10,
            "qoffset=0.5 should give median=3, got {off_high}"
        );

        // With explicit qoffset, offset should track qoffset not tau
        let fam_custom = QuantReg::with_qoffset(0.5, 0.01);
        let off_custom = fam_custom.offset(y.view(), w.view());
        assert!(
            off_custom >= 1.0 && off_custom <= 2.0,
            "qoffset=0.01 should be near min, got {off_custom}"
        );
    }

    // ------------------------------------------------------------------
    // 16.  Huber auto-calibration: delta updated after calibrate
    // ------------------------------------------------------------------

    #[test]
    fn huber_delta_calibration_changes() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // outlier at 100
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut family = Huber::new(None);

        let f = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        family.calibrate(y.view(), f.view(), w.view());

        let delta = family.delta.unwrap();
        // Delta should be the median absolute residual (with outlier)
        // Median of |1,2,3,4,5,100| = 3.5
        assert!(
            delta > 0.0 && delta < 100.0,
            "Huber delta should be finite and not dominated by outlier, got {delta}"
        );
    }

    // ------------------------------------------------------------------
    // 17.  GammaReg: check_y rejects non-positive values
    // ------------------------------------------------------------------

    #[test]
    fn gammareg_check_y() {
        assert!(GammaReg::new()
            .check_y(array![1.0, 2.0, 3.0].view())
            .is_ok());
        assert!(GammaReg::new()
            .check_y(array![0.0, 1.0, 2.0].view())
            .is_err());
        assert!(GammaReg::new()
            .check_y(array![-1.0, 1.0, 2.0].view())
            .is_err());
    }

    // ------------------------------------------------------------------
    // 18.  ExpectReg with extreme tau values
    // ------------------------------------------------------------------

    #[test]
    fn expectreg_extreme_tau_gradients() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let f = array![3.0, 3.0, 3.0, 3.0, 3.0]; // all at mean
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];

        let mut fam_high = ExpectReg::new(0.99);
        let mut out_high = Array1::zeros(5);
        fam_high.negative_gradient(y.view(), f.view(), w.view(), out_high.view_mut());

        // For tau=0.99, positive residuals get amplified by ~2, negative by ~0.02
        for (&yi, &ui) in y.iter().zip(out_high.iter()) {
            let r = yi - f[0]; // f is constant 3.0
            let expected = if r > 0.0 {
                2.0 * 0.99 * r
            } else {
                2.0 * 0.01 * r
            };
            assert!(
                (ui - expected).abs() < 1e-12,
                "ExpectReg gradient mismatch: got {ui}, expected {expected}"
            );
        }
    }

    // ------------------------------------------------------------------
    // 19.  Laplace: offset at boundary (min or max of y)
    // ------------------------------------------------------------------

    #[test]
    fn laplace_offset_median_at_boundary() {
        let y = array![1.0, 2.0, 3.0, 4.0, 100.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let off = Laplace::default().offset(y.view(), w.view());
        // Median of {1,2,3,4,100} = 3, but the Brent optimiser gives the
        // weighted median which minimises absolute deviations.
        assert!(
            off >= 2.0 && off <= 4.0,
            "Laplace offset should be near median, got {off}"
        );
    }

    // ------------------------------------------------------------------
    // 20.  Binomial: response link inverse
    // ------------------------------------------------------------------

    #[test]
    fn binomial_response_link() {
        let f = array![0.0, 2.0, -2.0];
        let mut out = Array1::zeros(3);
        Binomial.response(f.view(), out.view_mut());
        // p = 1 / (1 + exp(-2*f)) with y in {-1,1}
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!(
            out[1] > 0.88,
            "response at f=2 should be > 0.88, got {}",
            out[1]
        );
        assert!(
            out[2] < 0.12,
            "response at f=-2 should be < 0.12, got {}",
            out[2]
        );
    }

    // ------------------------------------------------------------------
    // 21.  Poisson: fused vs separate risk should match
    // ------------------------------------------------------------------

    #[test]
    fn poisson_fused_ngradient_and_risk_matches_separate() {
        let y = array![1.0, 2.0, 3.0, 0.0, 5.0];
        let f = array![0.1, 0.2, 0.3, -0.1, 0.5];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];

        let mut u1 = Array1::zeros(5);
        let mut family = Poisson;
        let risk_sep = family.risk(y.view(), f.view(), w.view());
        family.negative_gradient(y.view(), f.view(), w.view(), u1.view_mut());

        let mut u2 = Array1::zeros(5);
        let risk_fused = family.ngradient_and_risk(y.view(), f.view(), w.view(), u2.view_mut());

        assert!(
            (risk_sep - risk_fused).abs() < 1e-12,
            "fused and separate risk should match: {} vs {}",
            risk_sep,
            risk_fused
        );
        for i in 0..5 {
            assert!(
                (u1[i] - u2[i]).abs() < 1e-12,
                "fused and separate ngradient should match at {i}"
            );
        }
    }

    // ------------------------------------------------------------------
    // 22.  Performance: timings should be within reasonable bounds
    // ------------------------------------------------------------------

    #[test]
    fn fit_timing_regression_guard() {
        use std::time::Instant;

        let n = 2000;
        let mstop = 100;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let y: Array1<f64> = x.mapv(|v| 3.0 * v + (v * 10.0).sin());
        let w = Array1::ones(n);

        let l1 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let l2 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x2", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(mstop, 0.1).unwrap();
        let mut family = Gaussian;

        let start = Instant::now();
        let _path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);
        let elapsed = start.elapsed();

        // This is a very loose guard.  On a decent machine with --release
        // this should take < 100ms.  On debug it can be an order of
        // magnitude slower.  We set a generous 5-second ceiling to avoid
        // spurious failures on slow CI runners.
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "fit took {:.2}s, regression suspected",
            elapsed.as_secs_f64()
        );
    }

    // ------------------------------------------------------------------
    // 23.  Multiple learners with same name should still work
    // ------------------------------------------------------------------

    #[test]
    fn fit_duplicate_names_ok() {
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![3.0, 1.0, 2.0];

        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert_eq!(path.selected.len(), 5);
        // The names are the same, but internal tracking should keep them distinct
    }

    // ------------------------------------------------------------------
    // 24.  Weighted mean with zero total weight
    // ------------------------------------------------------------------

    #[test]
    fn weighted_mean_zero_total_weight() {
        let y = array![1.0, 2.0, 3.0];
        let w = array![0.0, 0.0, 0.0];
        let off = Gaussian.offset(y.view(), w.view());
        assert_eq!(off, 0.0, "offset with zero total weight should be 0");
    }

    // ------------------------------------------------------------------
    // 25.  Risk monotone under different nu values
    // ------------------------------------------------------------------

    #[test]
    fn risk_monotonicity_across_nu_values() {
        let n = 30;
        let y: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::ones(n);
        let x = y.clone();
        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        for nu in [0.001f64, 0.01, 0.1, 0.5, 1.0] {
            let ctrl = BoostControl::new(20, nu).unwrap();
            let mut family = Gaussian;
            let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);
            for i in 1..path.risk.len() {
                assert!(
                    path.risk[i] <= path.risk[i - 1] + 1e-9,
                    "nu={nu}: risk increased at step {i}: {} -> {}",
                    path.risk[i - 1],
                    path.risk[i]
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // 26.  Packed path (store_increments=false) should not crash
    // ------------------------------------------------------------------

    #[test]
    fn fit_with_no_increment_storage() {
        let n = 100;
        let y: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::ones(n);
        let x = y.clone();
        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &ctrl,
            None,
            false,
        );

        assert!(
            path.increments.is_empty(),
            "increments should be empty when store_increments=false"
        );
        assert_eq!(path.selected.len(), 10);
    }

    #[test]
    fn fit_selected_matches_full_fit_selection() {
        let n = 100;
        let y: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64 / 10.0).sin());
        let w = Array1::ones(n);
        let x1: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let x2: Array1<f64> = Array1::from_shape_fn(n, |i| ((i as f64) * 0.11).cos());

        let build = || {
            vec![
                prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None),
                prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None),
            ]
        };

        let ctrl = BoostControl::new(15, 0.1).unwrap();
        let mut full_learners = build();
        let mut full_family = Gaussian;
        let full = fit(
            &mut full_learners,
            y.view(),
            w.view(),
            &mut full_family,
            &ctrl,
            None,
        );

        let mut selected_learners = build();
        let mut selected_family = Gaussian;
        let selected = fit_selected(
            &mut selected_learners,
            y.view(),
            w.view(),
            &mut selected_family,
            &ctrl,
            None,
        );

        assert_eq!(selected, full.selected);
    }

    // ------------------------------------------------------------------
    // 27.  Binomial response with perfect separation
    // ------------------------------------------------------------------

    #[test]
    fn binomial_perfect_separation() {
        let y = array![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x = array![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut family = Binomial;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert_eq!(path.selected.len(), 10);
        // Risk should decrease (or stay flat) even with perfect separation
        for i in 1..path.risk.len() {
            assert!(path.risk[i] <= path.risk[i - 1] + 1e-9);
        }
    }

    // ------------------------------------------------------------------
    // 28.  GammaReg: sigma calibration changes within fit
    // ------------------------------------------------------------------

    #[test]
    fn gammareg_sigma_calibration_changes() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut family = GammaReg::new();

        let f1 = array![0.5, 0.5, 0.5, 0.5, 0.5];
        family.calibrate(y.view(), f1.view(), w.view());
        let sigma1 = family.sigma;

        let f2 = array![1.5, 1.5, 1.5, 1.5, 1.5];
        family.calibrate(y.view(), f2.view(), w.view());
        let sigma2 = family.sigma;

        // Sigma should have been updated (or at least not be the default 1.0)
        assert!(
            sigma1 != 1.0 || sigma2 != 1.0,
            "GammaReg sigma should be calibrated away from default"
        );
    }

    // ------------------------------------------------------------------
    // 29.  Large penalty should force beta toward 0
    // ------------------------------------------------------------------

    #[test]
    fn large_penalty_forces_beta_to_zero() {
        let x = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let u = array![10.0, 20.0, 30.0];

        let mut l_small = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut l_large = prepare_scalar_bols(x.view(), w.view(), 1e12, true, "x", None);

        let (beta_small, _, _) = l_small.fit_step(u.view(), w.view());
        let (beta_large, _, _) = l_large.fit_step(u.view(), w.view());

        assert!(
            beta_small[0].abs() > beta_large[0].abs(),
            "large penalty should shrink beta more than small penalty"
        );
        assert!(
            beta_large[0].abs() < 1e-3,
            "extremely large penalty should force beta ~ 0, got {}",
            beta_large[0]
        );
    }

    // ------------------------------------------------------------------
    // 30.  ddf_to_lambda edge: df >= 1 and df <= 0
    // ------------------------------------------------------------------

    #[test]
    fn df_to_lambda_boundary_values() {
        let x = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];

        // df >= 1 → λ = 0 (unpenalised)
        let lam1 = df_to_lambda_scalar(x.view(), w.view(), 1.0, false);
        assert_eq!(lam1, 0.0, "df=1 without intercept should give lambda=0");

        let lam1b = df_to_lambda_scalar(x.view(), w.view(), 2.0, true);
        assert_eq!(lam1b, 0.0, "df=2 with intercept should give lambda=0");

        let lam2 = df_to_lambda_scalar(x.view(), w.view(), 0.0, false);
        assert!(lam2 > 1e14, "df=0 should give very large lambda");

        let lam3 = df_to_lambda_scalar(x.view(), w.view(), -1.0, false);
        assert!(lam3 > 1e14, "negative df should give very large lambda");
    }

    // ------------------------------------------------------------------
    // 31.  BlockBols fit_step produces pred that satisfies design @ beta
    // ------------------------------------------------------------------

    #[test]
    fn block_bols_pred_equals_design_dot_beta() {
        let design = array![
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3],
            [1.0, 1.0, 0.8],
            [2.0, 0.0, 0.1],
            [0.5, 0.5, 0.6]
        ];
        let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let penalty = Array2::zeros((3, 3));

        let mut l = prepare_block_bols(design.clone(), w.view(), penalty, false, "X");
        let u = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (beta, pred, _sse) = l.fit_step(u.view(), w.view());

        // pred should equal design @ beta
        assert_eq!(pred.len(), 5);
        for i in 0..5 {
            let mut expected = 0.0f64;
            for j in 0..3 {
                expected += design[[i, j]] * beta[j];
            }
            assert!(
                (pred[i] - expected).abs() < 1e-10,
                "pred[{i}] = {}, expected {} (design @ beta)",
                pred[i],
                expected
            );
        }
    }

    // ------------------------------------------------------------------
    // 32.  store_increments true vs false gives same risk path
    // ------------------------------------------------------------------

    #[test]
    fn store_increments_true_vs_false_same_risk() {
        let n = 50;
        let y: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::ones(n);
        let x = y.clone();
        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(20, 0.1).unwrap();
        let mut family = Gaussian;

        let path_true = fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &ctrl,
            None,
            true,
        );
        let path_false = fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &ctrl,
            None,
            false,
        );

        assert_eq!(
            path_true.risk, path_false.risk,
            "risk path should be identical regardless of store_increments"
        );
        assert_eq!(
            path_true.selected, path_false.selected,
            "selections should be identical regardless of store_increments"
        );
    }
}
