#![allow(unused_imports)]
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmin_skip_nan_ignores_nan_after_minimum() {
        // Previously [1, NaN, 5] selected index 2 (value 5) because a NaN
        // following the minimum replaced it.
        assert_eq!(argmin_skip_nan(&[1.0, f64::NAN, 5.0]), Some(0));
        assert_eq!(argmin_skip_nan(&[f64::NAN, 3.0, 2.0]), Some(2));
        // All NaN: no minimum; the caller falls back to mstop = 1.
        assert_eq!(argmin_skip_nan(&[f64::NAN, f64::NAN]), None);
    }

    // ─── per-observation offsets (R `offset=` semantics) ──────────────────

    #[test]
    fn vector_offset_is_not_estimated_and_recovers_the_known_contribution() {
        // y = o + 0.5*x with a per-observation o: fitting x with the offset
        // supplied must recover 0.5*x exactly, proving the offset entered the
        // start and was never re-estimated.
        let x = Array1::linspace(0.0, 1.0, 60);
        let o = Array1::from_iter((0..60).map(|i| (i as f64) * 0.2 - 4.0));
        let y = &o + &(0.5 * &x);
        let w = Array1::ones(60);
        let control = BoostControl::new(400, 1.0).unwrap();
        // The block form `[1, x]` is what a vector-offset fit needs: the
        // centered single column leans on the model offset for its constant
        // (`gamboost`'s equivalence, see `learner_spec`), and a per-observation
        // offset is not a free constant, so the Python layer routes vector
        // offset fits to the block form. This test uses it for the same
        // reason.
        let design =
            ndarray::Array2::from_shape_fn((60, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] });
        let penalty = ndarray::Array2::zeros((2, 2));
        let mut learners = vec![prepare_block_bols(design, w.view(), penalty, false, "x")];
        let mut family = Gaussian;
        let path = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            o.view(),
            false,
        )
        .unwrap();
        let max = path
            .fitted
            .iter()
            .zip(y.iter())
            .map(|(fitted, observed)| (fitted - observed).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-10, "fitted differs from the offset model by {max}");
        // The scalar field reports the scalar start this fit never used; it
        // is not part of the model, which is why the caller keeps the vector.
        let _ = path.offset;
    }

    #[test]
    fn vector_offset_poisson_rates_track_exposure() {
        // Counts over unequal exposure: mu_i = exposure_i * exp(b0 + b1 x).
        // The offset log(exposure) makes the model a rate model, and the fit
        // above the offset tracks x rather than absorbing the exposure.
        let mut rng: u64 = 7;
        let mut uniform = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let n = 200;
        let x = Array1::from_iter((0..n).map(|_| 2.0 * uniform() - 1.0));
        let exposure = Array1::from_iter((0..n).map(|_| 0.5 + 3.5 * uniform()));
        let log_rate = 0.4 + &(0.6 * &x);
        let y = Array1::from_iter((0..n).map(|i| {
            // Knuth's Poisson sampler around lambda = exposure * rate.
            let lambda = exposure[i] * log_rate[i].exp();
            let threshold = (-lambda).exp();
            let mut count = 0.0_f64;
            let mut product = uniform();
            while product > threshold {
                count += 1.0;
                product *= uniform();
            }
            count
        }));
        let w = Array1::ones(n);
        let control = BoostControl::new(300, 0.1).unwrap();
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut family = crate::families::Poisson;
        let offset = exposure.map(|e| e.ln());
        let path = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            offset.view(),
            false,
        )
        .unwrap();
        // eta_i = log(exposure_i) + b0 + b1 x_i: the fit above the offset is
        // a smooth function of x with real spread, not a constant that would
        // mean the exposure was absorbed into the estimate.
        let rate_fit = &path.fitted - &offset;
        let mean_rate = rate_fit.sum() / n as f64;
        let spread = rate_fit
            .iter()
            .map(|v| (v - mean_rate).abs())
            .fold(0.0, f64::max);
        assert!(
            spread > 0.1,
            "the fit above the offset tracks x (spread {spread})"
        );
    }

    #[test]
    fn vector_offset_length_and_finiteness_are_refused() {
        let x = Array1::linspace(0.0, 1.0, 10);
        let y = x.clone();
        let w = Array1::ones(10);
        let control = BoostControl::new(5, 0.1).unwrap();
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut family = Gaussian;
        let wrong_length = Array1::zeros(9);
        let error = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            wrong_length.view(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("9 entries"));

        let zero_exposure =
            Array1::from_iter((0..10).map(|i| if i == 3 { f64::NEG_INFINITY } else { 0.0 }));
        let error = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            zero_exposure.view(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("log(0)"));
    }

    #[test]
    fn unknown_information_criterion_is_rejected() {
        let error = try_compute_aic("typo", &[2.0, 1.0], &[0.1], 10.0, 2.0, 2.0)
            .expect_err("an unknown method must not return a NaN path");
        assert!(error.to_string().contains("aic method"));
    }
    use crate::families::{AdaExp, Gaussian, HingeLoss, Huber, Poisson};
    use crate::learners::{
        prepare_bbs, prepare_binned_scalar_bols, prepare_block_bols, prepare_scalar_bols,
        quantile_bin_column,
    };
    // `prepare_block_bols` is also used directly by the vector-offset tests.
    use ndarray::{array, Array1, Array2};

    fn assert_differential_paths_close(
        reference: &BoostingPath,
        optimized: &BoostingPath,
        seed: u64,
        specification: &str,
        tolerance: f64,
    ) {
        assert_eq!(
            reference.selected, optimized.selected,
            "selection mismatch; seed={seed} spec={specification}"
        );
        let max_coefficient_difference = reference
            .coefs
            .iter()
            .zip(&optimized.coefs)
            .flat_map(|(left, right)| left.iter().zip(right))
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_coefficient_difference <= tolerance,
            "coefficient mismatch {max_coefficient_difference:.3e}; seed={seed} spec={specification}"
        );
        let risk_scale = reference
            .risk
            .first()
            .copied()
            .unwrap_or(1.0)
            .abs()
            .max(1.0);
        let max_relative_risk_difference = reference
            .risk
            .iter()
            .zip(&optimized.risk)
            .map(|(left, right)| (left - right).abs() / risk_scale)
            .fold(0.0_f64, f64::max);
        assert!(
            max_relative_risk_difference <= tolerance,
            "risk mismatch {max_relative_risk_difference:.3e}; seed={seed} spec={specification}"
        );
        let max_fitted_difference = reference
            .fitted
            .iter()
            .zip(&optimized.fitted)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_fitted_difference <= tolerance,
            "fitted mismatch {max_fitted_difference:.3e}; seed={seed} spec={specification}"
        );
    }

    #[test]
    fn fit_empty_learners_is_offset_only() {
        // No learners: zero boosting steps, no selections, every fitted value
        // is the family offset (the weighted mean for Gaussian).
        let mut learners: Vec<crate::learners::PreparedLearner> = Vec::new();
        let y = array![1.0, 2.0, 6.0];
        let w = array![1.0, 1.0, 1.0];
        let mut fam = Gaussian;
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let path = fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            true,
        );
        assert!(path.selected.is_empty());
        assert_eq!(path.mstop(), 0);
        assert_eq!(path.risk.len(), 1);
        assert!((path.offset - 3.0).abs() < 1e-12);
        assert!(path.fitted.iter().all(|&f| (f - 3.0).abs() < 1e-12));
    }

    #[test]
    fn try_fit_empty_learners_offset_respects_vector_offset() {
        let mut learners: Vec<crate::learners::PreparedLearner> = Vec::new();
        let y = array![1.0, 2.0, 6.0];
        let w = array![1.0, 1.0, 1.0];
        let offset = array![0.5, 1.5, 2.5];
        let mut fam = Gaussian;
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let path = try_fit_with_vector_offset(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            offset.view(),
            true,
        )
        .expect("offset-only fit");
        assert!(path.selected.is_empty());
        assert!(path
            .fitted
            .iter()
            .zip(offset.iter())
            .all(|(f, o)| (f - o).abs() < 1e-12));
    }

    #[test]
    fn margin_family_binary_codings_share_the_core_fit_path() {
        let x = array![-2.0, -1.0, 1.0, 2.0];
        let y01 = array![0.0, 0.0, 1.0, 1.0];
        let ypm = array![-1.0, -1.0, 1.0, 1.0];
        let weights = Array1::ones(4);
        let control = BoostControl::new(20, 0.1).unwrap();

        let fit = |y: &Array1<f64>, family: &mut dyn Family| {
            let mut learners = vec![prepare_scalar_bols(
                x.view(),
                weights.view(),
                0.0,
                true,
                "x",
                None,
            )];
            try_fit(
                &mut learners,
                y.view(),
                weights.view(),
                family,
                &control,
                None,
            )
            .unwrap()
        };

        for (mut raw_family, mut encoded_family) in [
            (
                Box::new(AdaExp) as Box<dyn Family>,
                Box::new(AdaExp) as Box<dyn Family>,
            ),
            (
                Box::new(HingeLoss::default()) as Box<dyn Family>,
                Box::new(HingeLoss::default()) as Box<dyn Family>,
            ),
        ] {
            let encoded_by_family = fit(&y01, raw_family.as_mut());
            let already_encoded = fit(&ypm, encoded_family.as_mut());
            assert_eq!(encoded_by_family.selected, already_encoded.selected);
            assert_eq!(encoded_by_family.risk, already_encoded.risk);
            assert_eq!(encoded_by_family.fitted, already_encoded.fitted);
        }
    }

    #[test]
    fn binned_scalar_selection_path_matches_exact_at_distinct_count() {
        let n = 400usize;
        let columns: Vec<Array1<f64>> = (0..6)
            .map(|column| {
                Array1::from_iter((0..n).map(|row| {
                    let phase = (row * (column + 3) + column * 17) as f64;
                    (phase * 0.017).sin() + 0.002 * row as f64 * (column + 1) as f64
                }))
            })
            .collect();
        let y = Array1::from_iter((0..n).map(|row| {
            1.4 * columns[0][row] - 0.9 * columns[2][row]
                + 0.35 * columns[4][row]
                + ((row * 37 % 101) as f64 / 101.0 - 0.5) * 0.2
        }));
        let weights = Array1::from_iter((0..n).map(|row| 0.5 + (row % 9) as f64 / 7.0));
        let mut exact = columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                prepare_scalar_bols(
                    column.view(),
                    weights.view(),
                    0.3,
                    true,
                    &format!("x{index}"),
                    None,
                )
            })
            .collect::<Vec<_>>();
        let mut binned = columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                let bins = quantile_bin_column(column.view(), n).unwrap();
                prepare_binned_scalar_bols(bins, weights.view(), 0.3, true, &format!("x{index}"))
            })
            .collect::<Vec<_>>();
        // Stop before the signal learners become numerically tied at the
        // floating-point floor. The two routes use different candidate-scan
        // kernels (packed GEMM versus one-learner bin scans), so once every
        // remaining improvement is roundoff-sized, a path identity assertion
        // tests the BLAS reduction order rather than the binned design.
        let control = BoostControl::new(100, 0.1).unwrap();
        let mut exact_family = Gaussian;
        let exact_path = try_fit(
            &mut exact,
            y.view(),
            weights.view(),
            &mut exact_family,
            &control,
            None,
        )
        .unwrap();
        let mut binned_family = Gaussian;
        let binned_path = try_fit(
            &mut binned,
            y.view(),
            weights.view(),
            &mut binned_family,
            &control,
            None,
        )
        .unwrap();

        assert_eq!(exact_path.selected, binned_path.selected);
        let fitted_difference = exact_path
            .fitted
            .iter()
            .zip(&binned_path.fitted)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        let coefficient_difference = exact_path
            .coefs
            .iter()
            .zip(&binned_path.coefs)
            .flat_map(|(left, right)| left.iter().zip(right))
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        let risk_difference = exact_path
            .risk
            .iter()
            .zip(&binned_path.risk)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        assert!(
            fitted_difference < 1e-12,
            "fitted={fitted_difference}, coef={coefficient_difference}"
        );
        assert!(risk_difference < 1e-9, "{risk_difference}");
    }

    #[test]
    fn packed_scalar_rejects_a_round_without_a_finite_candidate() {
        // Squaring this finite Poisson gradient overflows `wu2`, so every
        // scalar candidate has a non-finite SSE. The packed path must report
        // that numerical failure instead of recording learner zero with a
        // zero coefficient for every requested iteration.
        let x1 = array![-2.0, -1.0, 1.0, 2.0];
        let x2 = array![1.0, -1.0, 1.0, -1.0];
        let y = Array1::from_elem(4, 1e300);
        let weights = Array1::ones(4);
        let mut learners = vec![
            prepare_scalar_bols(x1.view(), weights.view(), 0.0, true, "x1", None),
            prepare_scalar_bols(x2.view(), weights.view(), 0.0, true, "x2", None),
        ];
        let control = BoostControl::new(5, 0.1).unwrap();
        let mut family = Poisson;

        let error = try_fit_with(
            &mut learners,
            y.view(),
            weights.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .expect_err("non-finite candidate scores must not produce a fake selection");

        assert!(
            matches!(error, MboostrsError::Solver(message) if message.contains("no finite candidate SSE"))
        );
    }

    #[test]
    fn a_non_finite_offset_only_risk_is_reported_rather_than_stored() {
        // Enhancement-hunt 7.2: `try_fit_lss` validated the initial risk and
        // every step's risk and returned a typed error; the single-parameter
        // engine did neither, so a fit whose very first risk evaluation
        // overflowed returned "successfully" with an infinite risk path.
        // Gaussian squares its residuals, and 1e200^2 overflows an f64.
        let x = array![-2.0, -1.0, 1.0, 2.0];
        let y = array![1e200, -1e200, 1e200, -1e200];
        let weights = Array1::ones(4);
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            weights.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let control = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;

        let error = try_fit_with(
            &mut learners,
            y.view(),
            weights.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .expect_err("an offset-only risk that overflows must not produce a fit");

        assert!(
            matches!(&error, MboostrsError::Solver(message)
                     if message.contains("offset-only model is not finite")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn the_gram_fast_path_declines_rather_than_publishing_a_non_finite_risk() {
        // The Gram path returns `Option`, so it cannot carry the typed error.
        // It must decline instead, routing the fit to the general path, which
        // owns the check. Either way the caller never receives a path with a
        // non-finite risk in it.
        let x1 = array![1e200, -1e200, 1e200, -1e200, 1e200, -1e200];
        let x2 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![1e200, -1e200, 1e200, -1e200, 1e200, -1e200];
        let weights = Array1::ones(6);
        let mut learners = vec![
            prepare_scalar_bols(x1.view(), weights.view(), 0.0, true, "x1", None),
            prepare_scalar_bols(x2.view(), weights.view(), 0.0, true, "x2", None),
        ];
        let control = BoostControl::new(20, 0.1).unwrap();
        let mut family = Gaussian;

        match try_fit_with(
            &mut learners,
            y.view(),
            weights.view(),
            &mut family,
            &control,
            None,
            false,
        ) {
            Err(MboostrsError::Solver(_)) => {}
            Err(other) => panic!("unexpected error kind: {other:?}"),
            Ok(path) => assert!(
                path.risk.iter().all(|value| value.is_finite()),
                "a successful fit must not carry a non-finite risk: {:?}",
                path.risk
            ),
        }
    }

    #[test]
    fn execution_report_routes_and_instrumentation_are_deterministic() {
        use crate::report::{EngineRoute, FallbackReason, InstrumentationLevel};

        let y = array![1.0, 2.0, 4.0, 3.0, 5.0, 7.0];
        let w = Array1::ones(y.len());
        let x = Array1::linspace(-1.0, 1.0, y.len());
        let scalar = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);

        let mut off_learners = vec![scalar.clone()];
        let mut summary_learners = vec![scalar];
        let off_control = BoostControl::new(5, 0.1).unwrap();
        let mut off_family = Gaussian;
        let mut summary_family = Gaussian;
        let off = try_fit_with(
            &mut off_learners,
            y.view(),
            w.view(),
            &mut off_family,
            &off_control,
            None,
            false,
        )
        .unwrap();
        let summary = try_fit_profiled(
            &mut summary_learners,
            y.view(),
            w.view(),
            &mut summary_family,
            &off_control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(off.selected, summary.selected);
        assert_eq!(off.coefs, summary.coefs);
        assert_eq!(off.risk, summary.risk);
        assert_eq!(off.offset, summary.offset);
        assert_eq!(off.fitted, summary.fitted);
        assert_eq!(off.nuisance_path, summary.nuisance_path);
        assert_eq!(off.nuisance_vec_path, summary.nuisance_vec_path);
        assert_eq!(off.report.route, EngineRoute::PackedScalar);
        assert!(!off.report.gram_eligible);
        assert!(off
            .report
            .fallback_reasons
            .contains(&FallbackReason::PureScalarModel));
        assert_eq!(off.report.scalar_learner_count, 1);
        assert_eq!(off.report.block_learner_count, 0);
        assert_eq!(off.report.tree_learner_count, 0);
        assert_eq!(off.report.cholesky_solver_count, 0);
        assert_eq!(off.report.lu_solver_count, 0);
        assert_eq!(off.report.regularized_solver_count, 0);
        assert_eq!(off.report.max_solver_jitter, None);
        assert_eq!(off.report.factor_cache_bytes, 0);
        assert_eq!(off.report.gram_cache_bytes, 0);
        assert_eq!(off.report.calibration_count, 0);
        assert_eq!(off.report.total_ns, None);
        assert_eq!(off_control.instrumentation, InstrumentationLevel::Off);
        assert_eq!(
            summary.report.instrumentation,
            InstrumentationLevel::Summary
        );
        assert!(summary.report.setup_ns.is_some());
        assert!(summary.report.iteration_ns.is_some());
        assert!(summary.report.total_ns.is_some());
        let rows = off.path_rows();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].iteration, 1);
        assert_eq!(rows[0].learner_index, off.selected[0]);
        assert_eq!(rows[0].risk, Some(off.risk[1]));
        assert_eq!(rows[0].risk_change, Some(off.risk[0] - off.risk[1]));
        assert!(rows[0].coefficient_norm.is_finite());
    }

    #[test]
    fn seeded_packed_scalar_matches_reference_general() {
        // The tolerance covers only alternate floating-point summation order:
        // both routes solve the same scalar normal equations.
        const TOLERANCE: f64 = 2e-11;
        for seed in 1_u64..=12 {
            let n = 3 + (seed as usize % 29);
            let mstop = 1 + (seed as u32 % 19);
            let nu = [0.01, 0.1, 0.5, 1.0][seed as usize % 4];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let mut x0 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            x0[0] = -1.0;
            x0[n - 1] = 1.0;
            let x1 = x0.clone(); // exact tie: lowest learner index must win
            let x2 = Array1::from_elem(n, 0.25); // rank-deficient after centering
            let x3 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let y = Array1::from_shape_fn(n, |index| {
                1.7 * x0[index] - 0.4 * x3[index] + 0.05 * (random() - 0.5)
            });
            let mut weights = Array1::from_shape_fn(n, |_| 0.1 + 1.9 * random());
            if seed % 2 == 0 {
                weights[0] = 0.0;
            }
            let build = || {
                [&x0, &x1, &x2, &x3]
                    .into_iter()
                    .enumerate()
                    .map(|(index, x)| {
                        prepare_scalar_bols(
                            x.view(),
                            weights.view(),
                            0.0,
                            true,
                            &format!("x{index}"),
                            None,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let control = BoostControl::new(mstop, nu).unwrap();
            let mut reference_learners = build();
            let mut reference_family = Gaussian;
            let reference = try_fit_with_engine_policy(
                &mut reference_learners,
                y.view(),
                weights.view(),
                &mut reference_family,
                &control,
                None,
                false,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();
            let mut packed_learners = build();
            let mut packed_family = Gaussian;
            let packed = try_fit_with_engine_policy(
                &mut packed_learners,
                y.view(),
                weights.view(),
                &mut packed_family,
                &control,
                None,
                false,
                EnginePolicy::ForcePackedScalar,
            )
            .unwrap();
            let specification = format!("n={n},mstop={mstop},nu={nu}");
            assert_eq!(reference.report.route, EngineRoute::General);
            assert_eq!(packed.report.route, EngineRoute::PackedScalar);
            assert_differential_paths_close(&reference, &packed, seed, &specification, TOLERANCE);
        }
    }

    #[test]
    fn seeded_parallel_block_matches_sequential_block() {
        // Block solves are identical; the tolerance allows BLAS reduction
        // order differences between sequential and rayon execution.
        const TOLERANCE: f64 = 2e-10;
        for seed in 101_u64..=106 {
            let n = 24 + (seed as usize % 17);
            let mstop = 4 + (seed as u32 % 13);
            let nu = [0.05, 0.1, 0.3][seed as usize % 3];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let mut x0 = Array1::from_shape_fn(n, |_| random());
            x0[0] = 0.0;
            x0[n - 1] = 1.0; // exercise both spline boundary knots
            let x1 = x0.clone(); // exact tied smoothers
            let x2 = Array1::from_shape_fn(n, |_| random());
            let x3 = Array1::from_shape_fn(n, |_| random());
            let y = Array1::from_shape_fn(n, |index| {
                (std::f64::consts::TAU * x0[index]).sin() + x2[index] + 0.02 * (random() - 0.5)
            });
            let weights = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * random());
            let build = || {
                [&x0, &x1, &x2, &x3]
                    .into_iter()
                    .enumerate()
                    .map(|(index, x)| {
                        prepare_bbs(
                            x.view(),
                            weights.view(),
                            6,
                            3,
                            2,
                            4.0,
                            true,
                            &format!("bbs(x{index})"),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let control = BoostControl::new(mstop, nu).unwrap();
            let mut sequential_learners = build();
            let mut sequential_family = Gaussian;
            let sequential = try_fit_with_engine_policy(
                &mut sequential_learners,
                y.view(),
                weights.view(),
                &mut sequential_family,
                &control,
                None,
                false,
                EnginePolicy::ForceSequentialBlock,
            )
            .unwrap();
            let mut parallel_learners = build();
            let mut parallel_family = Gaussian;
            let parallel = try_fit_with_engine_policy(
                &mut parallel_learners,
                y.view(),
                weights.view(),
                &mut parallel_family,
                &control,
                None,
                false,
                EnginePolicy::ForceParallelBlock,
            )
            .unwrap();
            let specification = format!("n={n},mstop={mstop},nu={nu}");
            assert_eq!(sequential.report.route, EngineRoute::General);
            assert_eq!(parallel.report.route, EngineRoute::ParallelBlock);
            assert_differential_paths_close(
                &sequential,
                &parallel,
                seed,
                &specification,
                TOLERANCE,
            );
        }
    }

    #[test]
    fn reusing_the_subsample_buffers_draws_the_same_learners() {
        // P3.4 item 3: the component-subsample draw allocated a shuffle
        // buffer and a mask on every boosting step -- `2 * mstop` allocations
        // for any fit that uses it, where nothing about the buffers changes
        // between steps except their contents.
        //
        // Reusing them is only safe if the draw is unchanged, and "unchanged"
        // here means the same learners, not a similar number of them: the
        // seed mixes the base seed with the iteration index, and a buffer
        // that was not cleared between steps would leave the previous
        // iteration's marks in place and quietly widen the candidate set.
        for fraction in [0.1_f64, 0.34, 0.5, 0.9] {
            for n_learners in [1_usize, 2, 7, 40] {
                let control = BoostControl::new(50, 0.1)
                    .unwrap()
                    .with_component_subsampling(fraction, 20260905)
                    .unwrap();
                let k = active_learner_count(&control, n_learners);
                let mut scratch = ComponentSubsampleScratch::with_capacity(n_learners);
                for iteration in 0..50usize {
                    let fresh = active_learner_mask(&control, n_learners, iteration);
                    match (k, fresh) {
                        (None, None) => {}
                        (Some(k), Some(expected)) => {
                            scratch.draw(&control, n_learners, iteration, k);
                            assert_eq!(
                                scratch.mask, expected,
                                "fraction={fraction} n={n_learners} iteration={iteration}"
                            );
                            assert_eq!(
                                scratch.mask.iter().filter(|&&on| on).count(),
                                k,
                                "the reused mask must mark exactly k learners"
                            );
                        }
                        (a, b) => panic!(
                            "the two paths disagree about whether to subsample: \
                             {a:?} against {}",
                            b.is_some()
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn seeded_hybrid_matches_reference_general() {
        // The route pairs already covered are packed-scalar against general,
        // parallel-block against sequential-block, and Gaussian-Gram against
        // general. `Hybrid` -- a learner set mixing packed scalars with dense
        // blocks -- had none, and it is the route a mixed `bols() + bbs()`
        // formula takes for every non-Gaussian family. P3.2 asks for
        // differential route tests at the module boundaries *before* the
        // monoliths are split, because a split that reordered selection
        // would otherwise be invisible.
        //
        // There is no `ForceHybrid`: `Auto` picks it from the learner mix, so
        // asserting the route came out `Hybrid` pins the routing decision as
        // well as the arithmetic.
        const TOLERANCE: f64 = 5e-11;
        for seed in 201_u64..=206 {
            let n = 30 + (seed as usize % 23);
            let mstop = 5 + (seed as u32 % 11);
            let nu = [0.05, 0.1, 0.3][seed as usize % 3];
            let mut state = seed;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64
            };
            let mut x0 = Array1::from_shape_fn(n, |_| random());
            x0[0] = 0.0;
            x0[n - 1] = 1.0;
            let x1 = Array1::from_shape_fn(n, |_| 2.0 * random() - 1.0);
            let x2 = x1.clone(); // an exact tie across the two kinds
            let x3 = Array1::from_shape_fn(n, |_| random());
            let y = Array1::from_shape_fn(n, |index| {
                (std::f64::consts::TAU * x0[index]).sin()
                    + 1.3 * x1[index]
                    + 0.02 * (random() - 0.5)
            });
            let weights = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * random());
            let build = || {
                let mut learners = vec![
                    prepare_bbs(x0.view(), weights.view(), 6, 3, 2, 4.0, true, "bbs(x0)"),
                    prepare_bbs(x3.view(), weights.view(), 6, 3, 2, 4.0, true, "bbs(x3)"),
                ];
                learners.push(prepare_scalar_bols(
                    x1.view(),
                    weights.view(),
                    0.0,
                    true,
                    "x1",
                    None,
                ));
                learners.push(prepare_scalar_bols(
                    x2.view(),
                    weights.view(),
                    0.0,
                    true,
                    "x2",
                    None,
                ));
                learners
            };
            let control = BoostControl::new(mstop, nu).unwrap();

            let mut reference_learners = build();
            let mut reference_family = Huber::new(Some(1.0));
            let reference = try_fit_with_engine_policy(
                &mut reference_learners,
                y.view(),
                weights.view(),
                &mut reference_family,
                &control,
                None,
                false,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();

            let mut hybrid_learners = build();
            let mut hybrid_family = Huber::new(Some(1.0));
            let hybrid = try_fit_with_engine_policy(
                &mut hybrid_learners,
                y.view(),
                weights.view(),
                &mut hybrid_family,
                &control,
                None,
                false,
                EnginePolicy::Auto,
            )
            .unwrap();

            let specification = format!("n={n},mstop={mstop},nu={nu}");
            assert_eq!(reference.report.route, EngineRoute::General);
            assert_eq!(
                hybrid.report.route,
                EngineRoute::Hybrid,
                "a mixed scalar/block set under a non-Gaussian family should \
                 route Hybrid; seed={seed} spec={specification}"
            );
            assert_differential_paths_close(&reference, &hybrid, seed, &specification, TOLERANCE);
        }
    }

    #[test]
    fn candidate_scan_merge_is_partition_independent() {
        // Enhancement-hunt 2.2 asked whether `winning_multiplicity` is
        // reduction-safe: Rayon merges partial accumulators over an arbitrary
        // binary tree, so if `merge_candidate_scans` disagreed with the serial
        // fold for some tree shape, the reported `tie_selection_count` would
        // depend on how the work was partitioned.
        //
        // Enumerate every binary tree shape over each score vector by
        // recursively splitting at every position, and require all of them to
        // equal the serial fold `observe_candidate` performs — index, score,
        // and multiplicity.
        fn serial(scores: &[f64]) -> (usize, f64, usize) {
            let mut best_index = 0usize;
            let mut best_sse = f64::INFINITY;
            let mut multiplicity = 0usize;
            for (index, &sse) in scores.iter().enumerate() {
                observe_candidate(
                    index,
                    sse,
                    &mut best_index,
                    &mut best_sse,
                    &mut multiplicity,
                );
            }
            (best_index, best_sse, multiplicity)
        }

        fn all_tree_shapes(leaves: &[(usize, f64, usize)]) -> Vec<(usize, f64, usize)> {
            if leaves.len() == 1 {
                return vec![leaves[0]];
            }
            let mut results = Vec::new();
            for split in 1..leaves.len() {
                for left in all_tree_shapes(&leaves[..split]) {
                    for right in all_tree_shapes(&leaves[split..]) {
                        results.push(merge_candidate_scans(left, right));
                    }
                }
            }
            results
        }

        let inf = f64::INFINITY;
        let nan = f64::NAN;
        let cases: Vec<Vec<f64>> = vec![
            vec![5.0, 5.0, 7.0],
            vec![7.0, 5.0, 5.0],
            vec![5.0, 7.0, 5.0],
            vec![3.0, 3.0, 3.0],
            vec![3.0, 5.0, 3.0, 5.0],
            vec![5.0, 3.0, 3.0, 5.0, 3.0],
            vec![inf, inf, inf],
            vec![inf, 4.0, inf, 4.0],
            vec![nan, 2.0, 2.0],
            vec![2.0, nan, 2.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
        ];

        for scores in cases {
            let expected = serial(&scores);
            // Rayon may fold the identity in at any point, so include it.
            let mut leaves: Vec<(usize, f64, usize)> = vec![CANDIDATE_SCAN_IDENTITY];
            leaves.extend(
                scores
                    .iter()
                    .enumerate()
                    .map(|(index, &sse)| (index, sse, usize::from(sse.is_finite()))),
            );
            for actual in all_tree_shapes(&leaves) {
                assert_eq!(
                    actual.0, expected.0,
                    "index differs for {scores:?}: {actual:?} vs {expected:?}"
                );
                assert_eq!(
                    actual.2, expected.2,
                    "multiplicity differs for {scores:?}: {actual:?} vs {expected:?}"
                );
                assert!(
                    (actual.1 == expected.1) || (actual.1.is_nan() && expected.1.is_nan()),
                    "score differs for {scores:?}: {actual:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn parallel_block_ties_match_sequential_across_learner_counts() {
        let n = 64;
        let x = Array1::linspace(-1.0, 1.0, n);
        let design = Array2::from_shape_fn((n, 3), |(row, column)| match column {
            0 => x[row],
            1 => x[row] * x[row],
            _ => x[row] * x[row] * x[row],
        });
        let y = x.mapv(|value| 1.2 * value - 0.7 * value * value);
        let weights = Array1::ones(n);
        let penalty = Array2::eye(3) * 0.01;
        let control = BoostControl::new(3, 0.1).unwrap();

        for learner_count in [1, 2, 4, 10, 50] {
            let build = || {
                (0..learner_count)
                    .map(|index| {
                        prepare_block_bols(
                            design.clone(),
                            weights.view(),
                            penalty.clone(),
                            false,
                            &format!("block{index}"),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let mut sequential_learners = build();
            let sequential = try_fit_with_engine_policy(
                &mut sequential_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                None,
                false,
                EnginePolicy::ForceSequentialBlock,
            )
            .unwrap();
            let mut parallel_learners = build();
            let parallel = try_fit_with_engine_policy(
                &mut parallel_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                None,
                false,
                EnginePolicy::ForceParallelBlock,
            )
            .unwrap();

            assert_eq!(sequential.selected, vec![0; 3], "J={learner_count}");
            assert_differential_paths_close(
                &sequential,
                &parallel,
                learner_count as u64,
                &format!("J={learner_count},exact_tie"),
                1e-12,
            );
            let expected_ties = if learner_count > 1 { 3 } else { 0 };
            assert_eq!(parallel.report.tie_selection_count, expected_ties);
        }
    }

    #[test]
    fn execution_report_counts_only_active_family_calibrations() {
        use crate::families::Huber;

        let y = array![1.0, 2.0, 4.0, 3.0, 5.0, 7.0];
        let w = Array1::ones(y.len());
        let x = Array1::linspace(-1.0, 1.0, y.len());
        let control = BoostControl::new(4, 0.1).unwrap();

        let mut adaptive_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut adaptive_family = Huber::new(None);
        let adaptive = try_fit_with(
            &mut adaptive_learners,
            y.view(),
            w.view(),
            &mut adaptive_family,
            &control,
            None,
            false,
        )
        .unwrap();
        assert_eq!(adaptive.report.calibration_count, 5);

        let mut fixed_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut fixed_family = Huber::new(Some(1.5));
        let fixed = try_fit_with(
            &mut fixed_learners,
            y.view(),
            w.view(),
            &mut fixed_family,
            &control,
            None,
            false,
        )
        .unwrap();
        assert_eq!(fixed.report.calibration_count, 0);
    }

    #[test]
    fn execution_report_tracks_propodds_calibration_stops() {
        use crate::families::PropOdds;

        let y = array![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0];
        let w = Array1::ones(y.len());
        let x = Array1::linspace(-1.0, 1.0, y.len());
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut family = PropOdds::new(3);
        let control = BoostControl::new(3, 0.1).unwrap();
        let path = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(path.report.calibration_count, 4);
        assert_eq!(
            path.report.calibration_converged_count
                + path.report.calibration_limit_count
                + path.report.calibration_failure_count,
            path.report.calibration_count
        );
        assert_eq!(path.report.calibration_failure_count, 0);
    }

    #[test]
    fn execution_report_identifies_gaussian_gram_route() {
        use crate::report::{EngineRoute, InstrumentationLevel};

        let n = 20;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|value| value * value + 0.5 * value);
        let w = Array1::ones(n);
        let design =
            Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { x[i] * x[i] });
        let block = prepare_block_bols(design, w.view(), Array2::zeros((2, 2)), true, "block");
        let mut learners = vec![block];
        let mut family = Gaussian;
        let control = BoostControl::new(5, 0.1)
            .unwrap()
            .with_instrumentation(InstrumentationLevel::Summary);
        let path = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(path.report.route, EngineRoute::GaussianGram);
        assert!(path.report.gram_eligible);
        assert!(path.report.fallback_reasons.is_empty());
        assert_eq!(path.report.total_basis_columns, 2);
        assert_eq!(path.report.scalar_learner_count, 0);
        assert_eq!(path.report.block_learner_count, 1);
        assert_eq!(path.report.cholesky_solver_count, 1);
        assert_eq!(path.report.lu_solver_count, 0);
        assert_eq!(path.report.regularized_solver_count, 0);
        assert_eq!(path.report.max_solver_jitter, None);
        assert_eq!(path.report.factor_cache_bytes, 2 * 2 * 8);
        assert_eq!(path.report.gram_cache_bytes, 2 * 2 * 8);
        assert!(path.report.setup_ns.is_some());
        assert!(path.report.cross_gram_ns.is_some());
        assert!(path.report.iteration_ns.is_some());
        assert!(path.report.reconstruction_ns.is_some());
        assert!(path.report.total_ns.is_some());
    }

    #[test]
    fn gaussian_gram_fit_never_builds_block_sparse_copies() {
        // #738: the Gram route works from the dense designs and the
        // cross-Grams only, so it must not trigger the lazy sparse-copy
        // build — otherwise an exact Gaussian fit retains about 46 MB per
        // bbs term at 10^6 rows for an index it never reads.
        use crate::learners::{prepare_bbs, PreparedLearner, SparseRowsCache};

        fn cache_of(learner: &PreparedLearner) -> &SparseRowsCache {
            match learner {
                PreparedLearner::BlockBols { sparse, .. } => sparse
                    .as_ref()
                    .expect("bbs design must carry a sparse cache"),
                _ => panic!("bbs must prepare as BlockBols"),
            }
        }

        let n = 3000;
        let x = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let y = x.mapv(|v| (4.0 * v).sin() + v * v);
        let w = Array1::ones(n);
        let mut learners = vec![
            prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x)"),
            prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x) twin"),
        ];
        let control = BoostControl::new(10, 0.1).unwrap();
        try_fit_with_engine_policy(
            &mut learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        for learner in &learners {
            assert!(
                !cache_of(learner).is_built(),
                "the Gram route must not build the sparse copy"
            );
        }
        // Prediction after the fit is still served by the copy, built on
        // that first use.
        let p = match &learners[0] {
            PreparedLearner::BlockBols { design, .. } => design.ncols(),
            _ => unreachable!(),
        };
        let mut out = vec![0.0; n];
        learners[0].predict_into(&vec![0.0; p], &mut out);
        assert!(
            cache_of(&learners[0]).is_built(),
            "prediction must build the sparse copy on first use"
        );
    }

    #[cfg(feature = "instrumentation-baseline")]
    #[test]
    fn no_report_benchmark_baseline_preserves_gram_fit() {
        let n = 30;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|value| value * value + 0.5 * value);
        let w = Array1::ones(n);
        let build = || {
            let design =
                Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { x[i] * x[i] });
            vec![prepare_block_bols(
                design,
                w.view(),
                Array2::zeros((2, 2)),
                true,
                "block",
            )]
        };
        let control = BoostControl::new(10, 0.1).unwrap();
        let mut reported_learners = build();
        let reported = try_fit_with(
            &mut reported_learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
            false,
        )
        .unwrap();
        let mut baseline_learners = build();
        let baseline = try_fit_without_report(
            &mut baseline_learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
        )
        .unwrap();

        assert_eq!(reported.selected, baseline.selected);
        assert_eq!(reported.coefs, baseline.coefs);
        assert_eq!(reported.risk, baseline.risk);
        assert_eq!(reported.fitted, baseline.fitted);
        assert_eq!(baseline.report.route, EngineRoute::Unknown);
    }

    #[test]
    fn execution_report_exposes_regularized_block_solver() {
        let n = 20;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|value| 0.5 * value);
        let w = Array1::ones(n);
        // Rank-deficient by an exactly-zero second column, not by duplicating the
        // first. Duplicate columns make the Gram singular only up to accumulation
        // order: on a target with FMA (for example a local target-cpu=native build
        // with on Linux) the duplicated entries can differ by one ULP, leaving the
        // matrix merely ill-conditioned, so plain Cholesky succeeds with a tiny
        // pivot and the regularized route is never reached. A zero column puts an
        // exact zero on the diagonal, which no rounding mode can perturb, so the
        // fallback triggers identically on every platform.
        let design = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { 0.0 });
        let block = prepare_block_bols(design, w.view(), Array2::zeros((2, 2)), false, "rank1");
        let mut learners = vec![block];
        let control = BoostControl::new(3, 0.1).unwrap();
        let path = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &control,
            None,
            false,
        )
        .unwrap();

        assert_eq!(path.report.regularized_solver_count, 1);
        assert_eq!(path.report.cholesky_solver_count, 0);
        assert_eq!(path.report.lu_solver_count, 0);
        assert!(path
            .report
            .max_solver_jitter
            .is_some_and(|value| value > 0.0));
        assert!(path.fitted.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn try_fit_y_weights_mismatch_returns_err() {
        let x = array![1.0, 2.0, 3.0];
        let w0 = array![0.0, 0.0, 0.0];
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w0.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0]; // wrong length
        let mut fam = Gaussian;
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let result = try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            true,
        );
        assert!(matches!(result, Err(MboostrsError::Shape(_))));
    }

    #[test]
    fn try_fit_rejects_nonfinite_response_and_weights() {
        let x = array![1.0, 2.0, 3.0];
        let setup_w = array![1.0, 1.0, 1.0];
        let ctrl = BoostControl::new(5, 0.1).unwrap();

        for (y, w) in [
            (array![1.0, f64::NAN, 3.0], array![1.0, 1.0, 1.0]),
            (array![1.0, 2.0, 3.0], array![1.0, f64::INFINITY, 1.0]),
        ] {
            let mut learners = vec![prepare_scalar_bols(
                x.view(),
                setup_w.view(),
                0.0,
                true,
                "x",
                None,
            )];
            let result = try_fit_with(
                &mut learners,
                y.view(),
                w.view(),
                &mut Gaussian,
                &ctrl,
                None,
                true,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn try_fit_rejects_negative_weights_nonfinite_offset_and_bad_learner_rows() {
        let x = array![1.0, 2.0, 3.0];
        let setup_w = array![1.0, 1.0, 1.0];
        let y = array![1.0, 2.0, 3.0];
        let ctrl = BoostControl::new(5, 0.1).unwrap();

        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            setup_w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let negative_w = array![1.0, -1.0, 1.0];
        assert!(try_fit_with(
            &mut learners,
            y.view(),
            negative_w.view(),
            &mut Gaussian,
            &ctrl,
            None,
            true,
        )
        .is_err());

        assert!(try_fit_with(
            &mut learners,
            y.view(),
            setup_w.view(),
            &mut Gaussian,
            &ctrl,
            Some(f64::NAN),
            true,
        )
        .is_err());

        let short_x = array![1.0, 2.0];
        let short_w = array![1.0, 1.0];
        let mut short_learner = vec![prepare_scalar_bols(
            short_x.view(),
            short_w.view(),
            0.0,
            true,
            "short",
            None,
        )];
        assert!(matches!(
            try_fit_with(
                &mut short_learner,
                y.view(),
                setup_w.view(),
                &mut Gaussian,
                &ctrl,
                None,
                true,
            ),
            Err(MboostrsError::LearnerSetup(_, _))
        ));
    }

    #[test]
    fn try_fit_rejects_empty_response() {
        let x = array![1.0];
        let setup_w = array![1.0];
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            setup_w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let y = Array1::<f64>::zeros(0);
        let w = Array1::<f64>::zeros(0);
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        assert!(try_fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut Gaussian,
            &ctrl,
            None,
            true,
        )
        .is_err());
    }

    #[test]
    fn hybrid_parallel_block_ties_choose_lowest_index() {
        let n = 24;
        let y = Array1::from_iter((0..n).map(|i| (i as f64 + 1.0) / n as f64));
        let w = Array1::ones(n);
        let zero_scalar = Array1::zeros(n);
        let block_design = Array2::from_shape_vec((n, 1), y.to_vec()).unwrap();
        let penalty = Array2::zeros((1, 1));

        let mut learners = vec![prepare_scalar_bols(
            zero_scalar.view(),
            w.view(),
            0.0,
            false,
            "z",
            None,
        )];
        for idx in 0..4 {
            learners.push(prepare_block_bols(
                block_design.clone(),
                w.view(),
                penalty.clone(),
                false,
                &format!("b{idx}"),
            ));
        }

        let mut fam = Gaussian;
        let ctrl = BoostControl::new(1, 0.1).unwrap();
        let path = fit_with(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            false,
        );

        assert_eq!(path.selected, vec![1]);
        assert_eq!(path.report.tie_selection_count, 1);
    }

    #[test]
    fn sparse_gram_products_keep_exact_support_and_matrix_layout() {
        use ndarray::ShapeBuilder;
        for fortran in [false, true] {
            let mut matrix = if fortran {
                Array2::zeros((80, 90).f())
            } else {
                Array2::zeros((80, 90))
            };
            for i in 0..79 {
                matrix[[i, i]] = 2.0;
                matrix[[i, i + 3]] = -0.25;
            }
            // A completely empty row and a tiny remote entry exercise CSR
            // row offsets and ensure no numerical sparsification occurs.
            matrix[[0, 89]] = 1e-100;
            let product = GramMatVec::new(&matrix);
            assert!(product.sparse.is_some());
            let mut rhs = Array1::from_shape_fn(90, |j| (j as f64).sin());
            let mut actual = Array1::zeros(80);
            product.mul_into(&rhs, actual.view_mut());
            let expected = matrix.dot(&rhs);
            assert!((&actual - &expected).iter().all(|v| v.abs() < 1e-14));
            assert_eq!(actual[79], 0.0);
            rhs.fill(0.0);
            rhs[89] = 1.0;
            product.mul_into(&rhs, actual.view_mut());
            assert_eq!(actual[0], 1e-100);
        }
        let dense = Array2::from_elem((80, 90), 1.0);
        assert!(GramMatVec::new(&dense).sparse.is_none());
        assert!(GramMatVec::new(&Array2::eye(24)).sparse.is_none());
    }

    #[test]
    fn gaussian_gram_spatial_matches_general_with_weights_centering_and_by() {
        use crate::learners::{prepare_bbs, prepare_bspatial};
        let n = 240;
        let x = Array1::from_shape_fn(n, |i| ((i * 37) % 241) as f64 / 241.0);
        let z = Array1::from_shape_fn(n, |i| ((i * 83) % 239) as f64 / 239.0);
        let by = Array1::from_shape_fn(n, |i| 0.5 + x[i]);
        let y = Array1::from_shape_fn(n, |i| (6.0 * x[i]).sin() * (5.0 * z[i]).cos() + x[i]);
        let weights = Array1::from_shape_fn(n, |i| if i % 7 == 0 { 0.0 } else { 0.5 + z[i] });
        for center in [false, true] {
            for nu in [0.1, 0.5, 1.0] {
                for lambda in [None, Some(0.7)] {
                    let spatial = prepare_bspatial(
                        x.view(),
                        z.view(),
                        weights.view(),
                        5,
                        3,
                        2,
                        6.0,
                        lambda,
                        center,
                        "spatial",
                        Some(by.view()),
                    );
                    let smooth = prepare_bbs(x.view(), weights.view(), 8, 3, 2, 4.0, true, "x");
                    let learners = vec![spatial, smooth];
                    let control = BoostControl::new(200, nu).unwrap();
                    let run = |policy| {
                        try_fit_with_engine_policy(
                            &mut learners.clone(),
                            y.view(),
                            weights.view(),
                            &mut Gaussian,
                            &control,
                            None,
                            false,
                            policy,
                        )
                        .unwrap()
                    };
                    let reference = run(EnginePolicy::ReferenceGeneral);
                    let optimized = run(EnginePolicy::ForceGaussianGram);
                    assert_differential_paths_close(
                        &reference,
                        &optimized,
                        270,
                        &format!(
                            "spatial center={center}, nu={nu}, lambda={lambda:?}, weighted by"
                        ),
                        1e-8,
                    );
                }
            }
        }
    }

    #[test]
    fn gaussian_gram_sparse_twins_stay_tied() {
        use crate::learners::prepare_bspatial;
        let n = 180;
        let x = Array1::from_shape_fn(n, |i| ((i * 37) % 181) as f64 / 181.0);
        let z = Array1::from_shape_fn(n, |i| ((i * 83) % 179) as f64 / 179.0);
        let y = Array1::from_shape_fn(n, |i| (6.0 * x[i]).sin() * (5.0 * z[i]).cos());
        let weights = Array1::ones(n);
        let spatial = prepare_bspatial(
            x.view(),
            z.view(),
            weights.view(),
            8,
            3,
            2,
            6.0,
            Some(1.0),
            false,
            "spatial",
            None,
        );
        let mut learners = vec![spatial.clone(), spatial];
        let control = BoostControl::new(1200, 0.5).unwrap();
        let path = try_fit_with_engine_policy(
            &mut learners,
            y.view(),
            weights.view(),
            &mut Gaussian,
            &control,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        assert!(path.selected.iter().all(|&j| j == 0));
    }

    #[test]
    fn gaussian_gram_duplicate_designs_share_blocks_and_stay_tied() {
        // Three copies of one bspatial learner — two clones (distinct
        // allocations, so the elementwise comparison path runs) and the
        // original. The Gram route must give all three bit-identical
        // blocks, so the fit is bit-identical to the two-learner fit and
        // the first learner wins every tie.
        use crate::learners::prepare_bspatial;
        let n = 180;
        let x = Array1::from_shape_fn(n, |i| ((i * 37) % 181) as f64 / 181.0);
        let z = Array1::from_shape_fn(n, |i| ((i * 83) % 179) as f64 / 179.0);
        let y = Array1::from_shape_fn(n, |i| (6.0 * x[i]).sin() * (5.0 * z[i]).cos());
        let weights = Array1::ones(n);
        let spatial = prepare_bspatial(
            x.view(),
            z.view(),
            weights.view(),
            8,
            3,
            2,
            6.0,
            Some(1.0),
            false,
            "spatial",
            None,
        );
        let control = BoostControl::new(1200, 0.5).unwrap();
        let run = |learners: &mut [PreparedLearner]| {
            try_fit_with_engine_policy(
                learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                None,
                false,
                EnginePolicy::ForceGaussianGram,
            )
            .unwrap()
        };
        let pair = run(&mut [spatial.clone(), spatial.clone()]);
        let trio = run(&mut [spatial.clone(), spatial.clone(), spatial]);
        assert!(trio.selected.iter().all(|&j| j == 0));
        assert_eq!(pair.selected, trio.selected);
        assert_eq!(pair.coefs, trio.coefs);
        assert!(pair
            .fitted
            .iter()
            .zip(trio.fitted.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));

        // Same design, different penalties: here the duplicate is not a
        // tie partner — its weaker penalty should win selections the
        // general path also gives it. This exercises the filled blocks
        // in the duplicate's own row and column (the source's diagonal
        // and the transposed cross blocks), which the tie scenario above
        // never touches because its duplicates are never selected.
        use crate::learners::prepare_bbs;
        let xs1 = Array1::from_shape_fn(n, |i| ((i * 53) % 181) as f64 / 181.0);
        let ys = Array1::from_shape_fn(n, |i| (6.0 * xs1[i]).sin());
        let build = || {
            vec![
                prepare_bbs(xs1.view(), weights.view(), 20, 3, 2, 0.05, true, "bbs(x)"),
                prepare_bbs(xs1.view(), weights.view(), 20, 3, 2, 50.0, true, "bbs(x)"),
            ]
        };
        let ctrl = BoostControl::new(120, 0.1).unwrap();
        let mut lg = build();
        let general = try_fit_with_engine_policy(
            &mut lg,
            ys.view(),
            weights.view(),
            &mut Gaussian,
            &ctrl,
            None,
            false,
            EnginePolicy::ReferenceGeneral,
        )
        .unwrap();
        let mut lk = build();
        let gram = try_fit_with_engine_policy(
            &mut lk,
            ys.view(),
            weights.view(),
            &mut Gaussian,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(gram.selected.contains(&1));
        assert_eq!(general.selected, gram.selected);
        let mut max_coef_d = 0.0f64;
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            for k in 0..a.len().min(b.len()) {
                max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
            }
        }
        assert!(max_coef_d < 1e-8, "coef mismatch: {max_coef_d:.3e}");
    }

    #[test]
    fn gaussian_gram_matches_general_path() {
        // Explicit policies keep route selection independent from path
        // storage choices.
        use crate::control::RiskKind;
        use crate::learners::prepare_bbs;

        let n = 300usize;
        let mut rng = 7u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin() + (tau * x2[i]).cos() + (xs() - 0.5) * 0.3
        });
        let ctrl = BoostControl::new(120, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        // Run with uniform and with non-uniform observation weights.
        let w_unit = Array1::ones(n);
        let w_rand = Array1::from_shape_fn(n, |_| 0.2 + 1.8 * xs());
        for w in [w_unit, w_rand] {
            let build = || {
                vec![
                    prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                    prepare_bbs(x2.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x2)"),
                ]
            };

            let mut lg = build();
            let mut fam_g = Gaussian;
            let general = try_fit_with_engine_policy(
                &mut lg,
                y.view(),
                w.view(),
                &mut fam_g,
                &ctrl,
                None,
                false,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();

            let mut lk = build();
            let mut fam_k = Gaussian;
            let gram = try_fit_with_engine_policy(
                &mut lk,
                y.view(),
                w.view(),
                &mut fam_k,
                &ctrl,
                None,
                false,
                EnginePolicy::ForceGaussianGram,
            )
            .unwrap();

            assert_eq!(general.report.route, EngineRoute::General);
            assert_eq!(gram.report.route, EngineRoute::GaussianGram);
            assert!(general.increments.is_empty());
            assert!(gram.increments.is_empty());

            assert_eq!(
                general.selected, gram.selected,
                "selected sequence diverged; seed=7 spec=n=300,mstop=120,nu=0.1"
            );

            let mut max_coef_d = 0.0f64;
            for (a, b) in general.coefs.iter().zip(&gram.coefs) {
                for k in 0..a.len().min(b.len()) {
                    max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
                }
            }
            assert!(
                max_coef_d < 1e-8,
                "coef mismatch: {max_coef_d:.3e}; seed=7 spec=n=300,mstop=120,nu=0.1"
            );

            let mut max_risk_d = 0.0f64;
            for (a, b) in general.risk.iter().zip(&gram.risk) {
                max_risk_d = max_risk_d.max((a - b).abs());
            }
            // risk magnitude scales with sum(w); use a relative bar.
            let risk0 = general.risk[0].abs().max(1.0);
            assert!(
                max_risk_d / risk0 < 1e-9,
                "risk mismatch: {max_risk_d:.3e}; seed=7 spec=n=300,mstop=120,nu=0.1"
            );

            let mut max_fit_d = 0.0f64;
            for i in 0..n {
                max_fit_d = max_fit_d.max((general.fitted[i] - gram.fitted[i]).abs());
            }
            assert!(
                max_fit_d < 1e-8,
                "fitted mismatch: {max_fit_d:.3e}; seed=7 spec=n=300,mstop=120,nu=0.1"
            );
        }
    }

    #[test]
    fn gaussian_gram_mixed_scalar_and_block() {
        // Mix of scalar bols + bbs learners. Exercises the ScalarBols
        // extraction branch of try_fit_gaussian_gram (1-column design, scalar
        // solve/penalty), which the all-bbs parity test does not cover.
        use crate::control::RiskKind;
        use crate::learners::{prepare_bbs, prepare_scalar_bols};

        let n = 300usize;
        let mut rng = 99u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let x3 = Array1::from_shape_fn(n, |_| xs());
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin() + 2.0 * x2[i] + (tau * x3[i]).cos() + (xs() - 0.5) * 0.3
        });
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(100, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None),
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                prepare_bbs(x3.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x3)"),
            ]
        };

        let mut lg = build();
        let mut fam_g = Gaussian;
        let general = try_fit_with_engine_policy(
            &mut lg,
            y.view(),
            w.view(),
            &mut fam_g,
            &ctrl,
            None,
            false,
            EnginePolicy::ReferenceGeneral,
        )
        .unwrap();

        let mut lk = build();
        let mut fam_k = Gaussian;
        let gram = try_fit_with_engine_policy(
            &mut lk,
            y.view(),
            w.view(),
            &mut fam_k,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();

        assert_eq!(general.report.route, EngineRoute::General);
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(gram.increments.is_empty());
        assert_eq!(
            general.selected, gram.selected,
            "selected sequence diverged; seed=99 spec=n=300,mstop=100,nu=0.1"
        );

        let mut max_coef_d = 0.0f64;
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            for k in 0..a.len().min(b.len()) {
                max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
            }
        }
        assert!(
            max_coef_d < 1e-8,
            "coef mismatch: {max_coef_d:.3e}; seed=99 spec=n=300,mstop=100,nu=0.1"
        );

        let mut max_fit_d = 0.0f64;
        for (g, c) in general.fitted.iter().zip(gram.fitted.iter()) {
            max_fit_d = max_fit_d.max((g - c).abs());
        }
        assert!(
            max_fit_d < 1e-8,
            "fitted mismatch: {max_fit_d:.3e}; seed=99 spec=n=300,mstop=100,nu=0.1"
        );
    }

    #[test]
    fn gaussian_gram_twin_learners_stay_tied() {
        // Two learners over bit-identical columns have bit-identical candidate
        // SSEs at every iteration, so the selection criterion is exactly tied
        // and the lowest index must win each time — R's `which.min` behaviour.
        // The Gram path maintains rhsⱼ incrementally through cross-Gram
        // blocks; any asymmetry between the twins' blocks or updates breaks
        // the tie and sends the path to the wrong learner. Found by
        // tests/test_selection_sequence_parity.py (config GAUSS_TIE).
        use crate::control::RiskKind;

        let n = 200usize;
        let mut rng = 11u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| 10.0 * xs());
        let y = Array1::from_shape_fn(n, |i| 2.0 * x1[i] + (xs() - 0.5) * 2.0);
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(50, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x1)"),
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x2)"),
            ]
        };
        let mut learners = build();
        let mut fam = Gaussian;
        let gram = try_fit_with_engine_policy(
            &mut learners,
            y.view(),
            w.view(),
            &mut fam,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(
            gram.selected.iter().all(|&j| j == 0),
            "twin tie broken: learner 1 first picked at iteration {}",
            gram.selected.iter().position(|&j| j == 1).unwrap_or(0) + 1
        );
    }

    #[test]
    fn gaussian_gram_matches_general_path_with_trees() {
        // Two bbs terms and two trees (one a genuine step), Gaussian. The Gram
        // route fits the trees to an explicit residual beside the projections;
        // it must select the same sequence as the reference general route and
        // agree on risk, fitted values and coefficients.
        use crate::control::RiskKind;
        use crate::learners::{prepare_bbs, prepare_btree};

        let n = 300usize;
        let mut rng = 17u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let x3 = Array1::from_shape_fn(n, |_| xs());
        let x4 = Array1::from_shape_fn(n, |_| xs());
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin()
                + 0.5 * x2[i]
                + if x3[i] > 0.4 { 1.0 } else { 0.0 }
                + (xs() - 0.5) * 0.3
        });
        // Fold-style weights: a fifth of the rows held out.
        let w = Array1::from_shape_fn(n, |i| if i % 5 == 0 { 0.0 } else { 1.0 });
        let ctrl = BoostControl::new(200, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);
        let x3m = x3.clone().insert_axis(ndarray::Axis(1));
        let x4m = x4.clone().insert_axis(ndarray::Axis(1));
        let build = || {
            vec![
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                prepare_btree(x3m.view(), w.view(), 7, 20, 1, None, "btree(x3)"),
                prepare_bbs(x2.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x2)"),
                prepare_btree(x4m.view(), w.view(), 7, 20, 1, None, "btree(x4)"),
            ]
        };
        let run = |policy| {
            let mut learners = build();
            let mut family = Gaussian;
            try_fit_with_engine_policy(
                &mut learners,
                y.view(),
                w.view(),
                &mut family,
                &ctrl,
                None,
                false,
                policy,
            )
            .unwrap()
        };
        let general = run(EnginePolicy::ReferenceGeneral);
        let gram = run(EnginePolicy::ForceGaussianGram);
        assert_eq!(general.report.route, EngineRoute::General);
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert_eq!(
            general.selected, gram.selected,
            "selected sequence diverged"
        );
        assert!(
            gram.selected.contains(&1),
            "the step tree was never selected"
        );
        for (a, b) in general.risk.iter().zip(&gram.risk) {
            assert!((a - b).abs() <= 1e-8 * a.abs().max(1.0), "risk {a} vs {b}");
        }
        for (a, b) in general.fitted.iter().zip(gram.fitted.iter()) {
            assert!((a - b).abs() <= 1e-9, "fitted {a} vs {b}");
        }
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            assert_eq!(a.len(), b.len());
            for (u, v) in a.iter().zip(b) {
                assert!((u - v).abs() <= 1e-9 * u.abs().max(1.0), "coef {u} vs {v}");
            }
        }
    }

    #[test]
    fn gaussian_gram_matches_general_path_with_bmono() {
        // B1: 2×bbs + 2×bmono (one constrained), Gaussian. The explicitly
        // forced Gram path must match the reference-general route.
        use crate::control::RiskKind;
        use crate::learners::{prepare_bbs, prepare_bmono, MonoConstraint};

        let n = 300usize;
        let mut rng = 11u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let x3 = Array1::from_shape_fn(n, |_| xs());
        // x4: decreasing signal under an Increasing constraint, so the
        // constrained solve actually binds (beta genuinely non-stationary).
        let x4 = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let tau = std::f64::consts::TAU;
        let y = Array1::from_shape_fn(n, |i| {
            (tau * x1[i]).sin() + (tau * x2[i]).cos() - (2.0 * tau * x4[i]).sin()
                + 0.5 * x3[i]
                + (xs() - 0.5) * 0.3
        });
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(150, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x1)"),
                prepare_bbs(x2.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x2)"),
                prepare_bmono(
                    x3.view(),
                    w.view(),
                    20,
                    3,
                    2,
                    4.0,
                    MonoConstraint::Increasing,
                    BmonoSolver::Iterative,
                    1e6,
                    10,
                    "bmono(x3)",
                    None,
                    None,
                    false,
                ),
                prepare_bmono(
                    x4.view(),
                    w.view(),
                    20,
                    3,
                    2,
                    4.0,
                    MonoConstraint::Increasing,
                    BmonoSolver::Iterative,
                    1e6,
                    10,
                    "bmono(x4)",
                    None,
                    None,
                    false,
                ),
            ]
        };

        let mut lg = build();
        let mut fam_g = Gaussian;
        let general = try_fit_with_engine_policy(
            &mut lg,
            y.view(),
            w.view(),
            &mut fam_g,
            &ctrl,
            None,
            false,
            EnginePolicy::ReferenceGeneral,
        )
        .unwrap();

        let mut lk = build();
        let mut fam_k = Gaussian;
        let gram = try_fit_with_engine_policy(
            &mut lk,
            y.view(),
            w.view(),
            &mut fam_k,
            &ctrl,
            None,
            false,
            EnginePolicy::ForceGaussianGram,
        )
        .unwrap();

        assert_eq!(general.report.route, EngineRoute::General);
        assert_eq!(gram.report.route, EngineRoute::GaussianGram);
        assert!(general.increments.is_empty());
        assert!(gram.increments.is_empty());

        assert_eq!(
            general.selected, gram.selected,
            "selected sequence diverged; seed=11 spec=n=300,mstop=150,nu=0.1"
        );

        let mut max_coef_d = 0.0f64;
        for (a, b) in general.coefs.iter().zip(&gram.coefs) {
            for k in 0..a.len().min(b.len()) {
                max_coef_d = max_coef_d.max((a[k] - b[k]).abs());
            }
        }
        assert!(
            max_coef_d < 1e-8,
            "coef mismatch: {max_coef_d:.3e}; seed=11 spec=n=300,mstop=150,nu=0.1"
        );

        let mut max_risk_d = 0.0f64;
        for (a, b) in general.risk.iter().zip(&gram.risk) {
            max_risk_d = max_risk_d.max((a - b).abs());
        }
        assert!(
            max_risk_d < 1e-8,
            "risk mismatch: {max_risk_d:.3e}; seed=11 spec=n=300,mstop=150,nu=0.1"
        );

        let mut max_fit_d = 0.0f64;
        for (g, c) in general.fitted.iter().zip(gram.fitted.iter()) {
            max_fit_d = max_fit_d.max((g - c).abs());
        }
        assert!(
            max_fit_d < 1e-8,
            "fitted mismatch: {max_fit_d:.3e}; seed=11 spec=n=300,mstop=150,nu=0.1"
        );
    }

    #[test]
    fn gaussian_gram_selection_only_matches_reference_without_unused_paths() {
        use crate::control::RiskKind;
        use crate::learners::{prepare_bmono, MonoConstraint};

        let n = 180usize;
        let x1 = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let x2 = Array1::from_shape_fn(n, |i| ((i * 47) % n) as f64 / (n - 1) as f64);
        let x3 = Array1::from_shape_fn(n, |i| ((i * 89) % n) as f64 / (n - 1) as f64);
        let y = Array1::from_shape_fn(n, |i| {
            (std::f64::consts::TAU * x1[i]).sin() + 0.8 * x2[i] + 0.4 * x3[i]
        });
        let weights = Array1::from_shape_fn(n, |i| 0.5 + (i % 7) as f64 / 7.0);
        let control = BoostControl::new(75, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let compare = |build: &dyn Fn() -> Vec<PreparedLearner>, label: &str| {
            let mut reference_learners = build();
            let reference = try_fit_selected_with_engine_policy(
                &mut reference_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                EnginePolicy::ReferenceGeneral,
            )
            .unwrap();
            let mut gram_learners = build();
            let gram = try_fit_selected_with_engine_policy(
                &mut gram_learners,
                y.view(),
                weights.view(),
                &mut Gaussian,
                &control,
                EnginePolicy::ForceGaussianGram,
            )
            .unwrap();

            assert_eq!(reference.report.route, EngineRoute::General, "{label}");
            assert_eq!(gram.report.route, EngineRoute::GaussianGram, "{label}");
            assert_eq!(reference.selected, gram.selected, "{label}");
            for path in [&reference, &gram] {
                assert!(path.coefs.is_empty(), "{label}");
                assert!(path.risk.is_empty(), "{label}");
                assert!(path.fitted.is_empty(), "{label}");
                assert!(path.increments.is_empty(), "{label}");
                assert!(path.nuisance_path.is_empty(), "{label}");
                assert!(path.nuisance_vec_path.is_empty(), "{label}");
                assert!(path.nuisance_vec.is_empty(), "{label}");
            }
        };

        compare(
            &|| {
                vec![
                    prepare_bbs(x1.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x1)"),
                    prepare_bbs(x2.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x2)"),
                ]
            },
            "block",
        );
        compare(
            &|| {
                vec![
                    prepare_scalar_bols(x2.view(), weights.view(), 0.0, true, "x2", None),
                    prepare_bbs(x1.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x1)"),
                ]
            },
            "mixed",
        );
        compare(
            &|| {
                vec![
                    prepare_bbs(x1.view(), weights.view(), 12, 3, 2, 4.0, true, "bbs(x1)"),
                    prepare_bmono(
                        x3.view(),
                        weights.view(),
                        12,
                        3,
                        2,
                        4.0,
                        MonoConstraint::Increasing,
                        BmonoSolver::Iterative,
                        1e6,
                        10,
                        "bmono(x3)",
                        None,
                        None,
                        false,
                    ),
                ]
            },
            "monotone",
        );
    }

    #[test]
    fn fit_gaussian_two_scalar_predictors() {
        // y = [3, 5, 7], x1 = [1, 2, 3], x2 = [3, 1, 2]
        // offset = weighted mean of y = 5.0
        // x1 centered: [-1, 0, 1], x2 centered: [1, -1, 0]
        let y = array![3.0, 5.0, 7.0];
        let w = array![1.0, 1.0, 1.0];
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![3.0, 1.0, 2.0];

        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(3, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert!((path.offset - 5.0).abs() < 1e-12);
        assert_eq!(path.selected.len(), 3);
        assert_eq!(path.risk.len(), 4); // offset risk + 3 steps

        // Risk should decrease monotonically (or at least not increase)
        for i in 1..path.risk.len() {
            assert!(path.risk[i] <= path.risk[i - 1] + 1e-10);
        }
    }

    #[test]
    fn fit_tie_breaker_lowest_index() {
        // Two identical columns → lowest index always selected
        let y = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0];

        let l1 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x2", None);
        let mut learners = vec![l1, l2];

        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        for &idx in &path.selected {
            assert_eq!(idx, 0);
        }
        assert_eq!(path.report.tie_selection_count, 5);
    }

    #[test]
    fn fit_single_step_hand_computed() {
        // y = [2, 4, 6], x = [1, 2, 3], w = [1, 1, 1], nu = 1.0
        // offset = 4.0, f0 = [4, 4, 4]
        // u = y - f = [-2, 0, 2]
        // x centered: [-1, 0, 1], denom = 2
        // numer = (-1)*(-2) + 0*0 + 1*2 = 4
        // beta = 4/2 = 2.0, pred = [-2, 0, 2]
        // f1 = f0 + 1.0 * pred = [2, 4, 6] = y
        // risk[1] = Σ (y-f)² = 0
        let y = array![2.0, 4.0, 6.0];
        let w = array![1.0, 1.0, 1.0];
        let x = array![1.0, 2.0, 3.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];

        let ctrl = BoostControl::new(1, 1.0).unwrap();
        let mut family = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut family, &ctrl, None);

        assert!((path.offset - 4.0).abs() < 1e-12);
        assert_eq!(path.selected.len(), 1);
        assert!((path.coefs[0][0] - 2.0).abs() < 1e-12);
        assert!((path.risk[1]).abs() < 1e-10); // perfect fit after one step
        assert!((path.fitted[0] - 2.0).abs() < 1e-12);
        assert!((path.fitted[1] - 4.0).abs() < 1e-12);
        assert!((path.fitted[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_incremental_risk_matches_full() {
        use crate::control::RiskKind;
        use crate::learners::prepare_scalar_bols;

        let n = 200usize;
        let mut rng = 42u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let y = Array1::from_shape_fn(n, |i| 2.0 * x1[i] - x2[i] + (xs() - 0.5));
        let w = Array1::ones(n);
        let ctrl = BoostControl::new(50, 0.1)
            .unwrap()
            .with_risk(RiskKind::Inbag);

        let build = || {
            vec![
                prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None),
                prepare_scalar_bols(x2.view(), w.view(), 0.0, true, "x2", None),
            ]
        };

        // store_increments=true forces the general path (not Gram).
        let mut l1 = build();
        let mut fam1 = Gaussian;
        let path = fit_with(&mut l1, y.view(), w.view(), &mut fam1, &ctrl, None, true);

        // Recompute full risk from fitted values at each step.
        let mut f_check = Array1::from_elem(n, path.offset);
        let nu = 0.1f64;
        let _full_risks: Vec<f64> = vec![path.risk[0]];
        for m in 0..path.selected.len() {
            let sel = path.selected[m];
            let coef = &path.coefs[m];
            let col = match sel {
                0 => &x1,
                1 => &x2,
                _ => unreachable!(),
            };
            // Scalar bols with centering: the coef is the raw beta;
            // the increment is nu * (coef[0] * (x - mean_x)).
            // But the fitted values from the path already incorporate centering,
            // so we compare against path.fitted directly.
            for i in 0..n {
                f_check[i] += nu * coef[0] * col[i];
            }
        }
        // Full risk at final fitted values.
        let final_full_risk: f64 = y
            .iter()
            .zip(path.fitted.iter())
            .map(|(yi, fi)| (yi - fi) * (yi - fi))
            .sum();
        let final_incr_risk = *path.risk.last().unwrap();
        assert!(
            (final_full_risk - final_incr_risk).abs() < 1e-6,
            "final risk: full={final_full_risk:.6}, incr={final_incr_risk:.6}"
        );
    }

    #[test]
    fn hat_matrix_nonuniform_weights_maintains_df() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 100usize;
        let mut rng = 12345u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::from_shape_fn(n, |_| 0.5 + xs()); // non-uniform
        let w_f64 = w.view();

        // === Scalar Bols without centering (lambda=0, df=1) ===
        let l = prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None);
        let h = l.hat_matrix().unwrap();
        let tr: f64 = (0..n).map(|i| h[[i, i]]).sum();
        assert!(
            (tr - 1.0).abs() < 1e-12,
            "scalar bols no-center trace: {tr}"
        );

        // === Scalar Bols with centering (lambda=0, df=2) ===
        let lc = prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None);
        let hc = lc.hat_matrix().unwrap();
        let trc: f64 = (0..n).map(|i| hc[[i, i]]).sum();
        assert!((trc - 2.0).abs() < 1e-12, "scalar bols center trace: {trc}");

        // === Scalar Bols with penalty (df < 1) ===
        let lp = prepare_scalar_bols(x.view(), w_f64, 10.0, false, "x", None);
        let hp = lp.hat_matrix().unwrap();
        let trp: f64 = (0..n).map(|i| hp[[i, i]]).sum();
        let xsq = x.mapv(|xi| xi * xi);
        let expected_df = w.dot(&xsq) / (w.dot(&xsq) + 10.0);
        assert!(
            (trp - expected_df).abs() < 1e-12,
            "scalar bols penalized trace: {trp} vs {expected_df}"
        );

        // === BlockBols: bbs with non-uniform weights ===
        let lb = prepare_bbs(x.view(), w_f64, 20, 3, 2, 4.0, false, "bbs(x)");
        let hb = lb.hat_matrix().unwrap();
        let trb: f64 = (0..n).map(|i| hb[[i, i]]).sum();
        assert!(trb > 0.0 && trb < n as f64, "bbs trace out of range: {trb}");
        assert!((trb - 4.0).abs() < 2.0, "bbs trace ~4: {trb}");

        // === AIC trace: 2 iterations, verify monotonic ===
        let l1 = prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None);
        let l2 = prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None);
        let learners = vec![l1, l2];
        let selected = vec![0usize, 1, 0, 1];
        let trace = try_compute_aic_trace(&learners, &selected, 0.1)
            .unwrap()
            .unwrap();
        assert!(
            trace.windows(2).all(|w| w[1] > w[0] - 1e-12),
            "AIC trace not monotonic: {trace:?}"
        );
        assert!(
            (trace[0] - 0.2).abs() < 1e-12,
            "first trace: {} vs 0.2",
            trace[0]
        );

        // === Verify hat matrix elements directly (penalized λ=10) ===
        let denom_pen = w.dot(&x.mapv(|xi| xi * xi)) + 10.0;
        let inv_denom_pen = 1.0 / denom_pen;
        for i in [0, 1, n / 2, n - 1] {
            for j in [0, n / 3, n - 1] {
                let expected = w[j] * x[i] * x[j] * inv_denom_pen;
                let actual = hp[[i, j]];
                assert!(
                    (actual - expected).abs() < 1e-14,
                    "H[{i},{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn approximate_hat_diag_covers_multi_column_learners() {
        use crate::learners::{prepare_bbs, prepare_scalar_bols};

        let n = 80usize;
        let mut rng = 848u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x = Array1::from_shape_fn(n, |_| xs());
        let z = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::ones(n);
        let learners = vec![
            prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x)"),
            prepare_scalar_bols(z.view(), w.view(), 0.0, true, "z", None),
        ];

        // The learner diagonal matches the full hat matrix's.
        let full = learners[0].hat_matrix().unwrap();
        let diag = learners[0].hat_diag().unwrap();
        for i in 0..n {
            assert!((diag[i] - full[[i, i]]).abs() < 1e-12);
        }

        // One step of the spline: the exact hat matrix is nu * H, so the
        // approximation is it plus the offset's 1/n.
        let nu = 0.1;
        let approx = try_approximate_hat_diag(&learners, &[0], nu)
            .unwrap()
            .unwrap();
        let exact = try_compute_hat_diag(&learners, &[0], nu).unwrap().unwrap();
        for i in 0..n {
            assert!((approx[i] - 1.0 / n as f64 - exact[i]).abs() < 1e-12);
        }
        let exact_trace = try_compute_aic_trace(&learners, &[0], nu).unwrap().unwrap();
        let approx_trace: f64 = approx.iter().sum();
        assert!((approx_trace - 1.0 - exact_trace[0]).abs() < 1e-10);

        // Several small steps of the spline: the exact trace is
        // sum_k 1 - (1 - nu e_k)^m over H's eigenvalues e_k, at most the
        // first-order m nu tr(H), and within the second-order term of it.
        let selected = [0usize, 0, 0, 0];
        let nu = 0.01;
        let approx = try_approximate_hat_diag(&learners, &selected, nu)
            .unwrap()
            .unwrap();
        let exact_trace = *try_compute_aic_trace(&learners, &selected, nu)
            .unwrap()
            .unwrap()
            .last()
            .unwrap();
        let approx_trace: f64 = approx.iter().sum::<f64>() - 1.0;
        assert!(approx_trace >= exact_trace - 1e-12);
        assert!(
            (approx_trace - exact_trace) / exact_trace < 0.05,
            "approximate {approx_trace} vs exact {exact_trace}"
        );

        // The single-column learner contributes its slope leverage only.
        let approx_z = try_approximate_hat_diag(&learners, &[1], 1.0)
            .unwrap()
            .unwrap();
        let zc = z.mapv(|v| v - z.mean().unwrap());
        let ss: f64 = zc.iter().map(|v| v * v).sum();
        for i in 0..n {
            let want = (1.0 / n as f64 + zc[i] * zc[i] / ss).clamp(0.0, 1.0);
            assert!((approx_z[i] - want).abs() < 1e-12);
        }
        assert!(try_approximate_hat_diag(&learners, &[2], 0.1).is_err());
    }

    // ─── AIC1: hat-factor verification ──────────────────────────────────

    /// The try_ AIC/hat variants must reject malformed inputs with a typed
    /// `Err` instead of panicking, and reproduce the wrapper's result on
    /// valid inputs.
    #[test]
    fn try_hat_trace_and_aic_variants_reject_bad_inputs() {
        use crate::learners::prepare_scalar_bols;

        let n = 20usize;
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));
        let w = Array1::ones(n);
        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let learners = vec![l];

        // Out-of-bounds selected index → Err, not a slice-index panic.
        assert!(matches!(
            try_compute_aic_trace(&learners, &[0, 7], 0.1),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_hat_diag(&learners, &[7], 0.1),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic_trace_fast(&learners, &[7], 0.1),
            Err(MboostrsError::Solver(_))
        ));

        // Empty learner set / empty selection → Err.
        assert!(matches!(
            try_compute_aic_trace(&[], &[0], 0.1),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic_trace_fast(&learners, &[], 0.1),
            Err(MboostrsError::Solver(_))
        ));

        // Non-finite nu → Err.
        assert!(matches!(
            try_compute_aic_trace(&learners, &[0], f64::NAN),
            Err(MboostrsError::Solver(_))
        ));

        // from_hats: no hat matrices → Err.
        assert!(matches!(
            try_compute_aic_trace_from_hats(&[], &[0], 0.1),
            Err(MboostrsError::Solver(_))
        ));

        // compute_aic: risk too short for df (risk[i+1] would be out of
        // bounds), or a non-finite/negative sumw.
        assert!(matches!(
            try_compute_aic("corrected", &[1.0, 2.0], &[1.0, 1.0, 1.0], 10.0, 2.0, 100.0),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic("corrected", &[1.0], &[1.0], 10.0, 2.0, 100.0),
            Err(MboostrsError::Solver(_))
        ));
        assert!(matches!(
            try_compute_aic("corrected", &[0.0, 5.0], &[1.0], -1.0, 2.0, 100.0),
            Err(MboostrsError::Solver(_))
        ));

        // Valid inputs reproduce the panicking wrapper's result exactly.
        // The wrapper is deprecated on the P0.8 schedule; this parity check
        // guards the shim until its 0.2.0 removal.
        let good = try_compute_aic("classical", &[0.0, 5.0], &[1.0], 10.0, 2.0, 100.0).unwrap();
        #[allow(deprecated)]
        let wrapper_result = compute_aic("classical", &[0.0, 5.0], &[1.0], 10.0, 2.0, 100.0);
        assert_eq!(wrapper_result, good);
        assert!(matches!(
            try_compute_hat_diag(&learners, &[0, 0, 0], 0.1),
            Ok(Some(_))
        ));
    }

    // ─── btree step hat matrices ─────────────────────────────────────

    /// One tree step's matrix, reconstructed independently of the code
    /// under test: decode the step's recorded tree, group the rows by
    /// the value the tree predicts for them (rows in one leaf share the
    /// leaf mean; the helper asserts the leaf means are distinct, so the
    /// groups ARE the leaves), and average the working response over
    /// each group with the learner's stored fit weights, in plain loops.
    fn brute_force_tree_step_matrix(learner: &PreparedLearner, coef: &[f64]) -> Array2<f64> {
        use crate::learners::FittedTree;
        let PreparedLearner::Btree {
            design, weights, ..
        } = learner
        else {
            panic!("brute_force_tree_step_matrix needs a tree learner");
        };
        let n = design.nrows();
        let tree = FittedTree::from_coef_vec(ndarray::ArrayView1::from(coef));
        let pred = tree.predict(design.view());
        // The leaf count, from the encoding's leaf tags.
        let mut leaves = 0usize;
        let mut idx = 1usize;
        for _ in 0..coef[0] as usize {
            match coef[idx] {
                0.0 => idx += 5,
                2.0 => idx += 5 + coef[idx + 2] as usize,
                _ => {
                    leaves += 1;
                    idx += 2;
                }
            }
        }
        let mut distinct: Vec<f64> = pred.to_vec();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            leaves,
            "leaf means must be distinct for grouping by prediction to recover the leaves"
        );
        let mut h = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let total: f64 = (0..n)
                .filter(|&j| pred[j] == pred[i])
                .map(|j| weights[j])
                .sum();
            assert!(total > 0.0, "a fitted leaf has positive fit weight");
            for j in 0..n {
                if pred[j] == pred[i] {
                    h[[i, j]] = weights[j] / total;
                }
            }
        }
        h
    }

    /// The boosting hat recursion over a recorded path, composed in the
    /// test from per-step matrices: a smooth learner's established
    /// `hat_matrix()`, a tree step's brute-force matrix above. With
    /// `step_weights`, each step's update is row-scaled after the fact,
    /// as the IRLS-weighted recursion prescribes.
    fn brute_force_hat_path(
        learners: &[PreparedLearner],
        selected: &[usize],
        coefs: &[Vec<f64>],
        nu: f64,
        step_weights: Option<&[Vec<f64>]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = learners[0].n();
        let mut b = Array2::<f64>::zeros((n, n));
        let mut trace = Vec::with_capacity(selected.len());
        for (step, &sel) in selected.iter().enumerate() {
            let h = match &learners[sel] {
                PreparedLearner::Btree { .. } => {
                    brute_force_tree_step_matrix(&learners[sel], &coefs[step])
                }
                learner => learner.hat_matrix().expect("smooth learner matrix"),
            };
            // z = (nu H)(I - B) = nu H - nu H B
            let mut z = Array2::<f64>::zeros((n, n));
            ndarray::linalg::general_mat_mul(-nu, &h, &b, 0.0, &mut z);
            z.zip_mut_with(&h, |zj, &hj| *zj += nu * hj);
            if let Some(all_w) = step_weights {
                let w = &all_w[step];
                for (i, mut row) in z.rows_mut().into_iter().enumerate() {
                    row.mapv_inplace(|v| v * w[i]);
                }
            }
            b += &z;
            trace.push(b.diag().sum());
        }
        (trace, b.diag().to_vec())
    }

    fn two_column_threshold_data(n: usize) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut state = 0x1F123BB5u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f64 / (1u32 << 24) as f64
        };
        let x = Array1::from_shape_fn(n, |_| next() * 4.0 - 2.0);
        let z = Array1::from_shape_fn(n, |_| next() * 2.0 - 1.0);
        let design = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { x[i] } else { z[i] });
        (x, z, design)
    }

    /// A Gaussian fit whose path selects a tree repeatedly: the trace
    /// and the diagonal from the steps variants agree with the
    /// brute-force composition of the same recorded path, and the
    /// coefficient-free calls still refuse, as they did before.
    #[test]
    fn btree_steps_in_hat_recursion_match_brute_force() {
        use crate::learners::{prepare_btree, prepare_scalar_bols};

        let n = 36usize;
        let (x, z, design) = two_column_threshold_data(n);
        let mut noise = 0x0D15EA5Eu32;
        let mut next_noise = move || {
            noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
            (noise >> 8) as f64 / (1u32 << 24) as f64
        };
        let y = Array1::from_shape_fn(n, |i| {
            (if x[i] > 0.25 { 2.0 } else { -1.0 }) + 0.3 * z[i] + 0.05 * next_noise()
        });
        let w = Array1::ones(n);
        let mut learners = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_btree(design.view(), w.view(), 3, 6, 2, None, "tree"),
        ];
        let nu = 0.3;
        let control = BoostControl::new(20, nu).unwrap();
        let mut family = Gaussian;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
        )
        .unwrap();
        assert!(
            path.selected.iter().filter(|&&s| s == 1).count() >= 2,
            "the tree must be selected repeatedly for this test: {:?}",
            path.selected
        );

        let (trace_bf, diag_bf) =
            brute_force_hat_path(&learners, &path.selected, &path.coefs, nu, None);
        let trace = try_compute_aic_trace_steps(&learners, &path.selected, Some(&path.coefs), nu)
            .unwrap()
            .expect("tree steps resolve from the recorded path");
        let diag = try_compute_hat_diag_steps(&learners, &path.selected, Some(&path.coefs), nu)
            .unwrap()
            .expect("tree steps resolve from the recorded path");
        for (got, want) in trace.iter().zip(&trace_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "trace: {got} vs brute force {want}"
            );
        }
        for (got, want) in diag.iter().zip(&diag_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "diag: {got} vs brute force {want}"
            );
        }
        // The fast trace takes its dense fallback for a tree path and
        // lands on the same numbers.
        let fast =
            try_compute_aic_trace_fast_steps(&learners, &path.selected, Some(&path.coefs), nu)
                .unwrap()
                .expect("fast steps trace");
        for (got, want) in fast.iter().zip(&trace) {
            assert!(
                (got - want).abs() < 1e-12,
                "fast trace: {got} vs dense {want}"
            );
        }
        // Without the coefficients the same calls still refuse, as before.
        assert_eq!(
            try_compute_aic_trace(&learners, &path.selected, nu).unwrap(),
            None
        );
        assert_eq!(
            try_compute_hat_diag(&learners, &path.selected, nu).unwrap(),
            None
        );
    }

    /// The IRLS-weighted recursion over a Binomial fit with tree steps:
    /// the steps variant agrees with the brute-force composition using
    /// the step weights at the fit after each step, rebuilt from the
    /// path's offset and increments exactly as the bindings rebuild them.
    #[test]
    fn btree_steps_in_weighted_hat_match_brute_force() {
        use crate::families::Binomial;
        use crate::learners::{prepare_btree, prepare_scalar_bols};

        let n = 36usize;
        let (x, z, design) = two_column_threshold_data(n);
        let mut noise = 0x0BADF00Du32;
        let mut next_noise = move || {
            noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
            (noise >> 8) as f64 / (1u32 << 24) as f64
        };
        let y = Array1::from_shape_fn(n, |i| {
            let latent =
                (if x[i] > 0.25 { 1.3 } else { -1.3 }) + 0.3 * z[i] + (next_noise() - 0.5) * 0.8;
            if latent > 0.0 {
                1.0
            } else {
                0.0
            }
        });
        let w = Array1::ones(n);
        let mut learners = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_btree(design.view(), w.view(), 3, 6, 2, None, "tree"),
        ];
        let nu = 0.3;
        let control = BoostControl::new(20, nu).unwrap();
        let mut family = Binomial;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
        )
        .unwrap();
        assert!(
            path.selected.contains(&1),
            "the tree must be selected for this test: {:?}",
            path.selected
        );

        let mut f = vec![path.offset; n];
        let mut step_weights: Vec<Vec<f64>> = Vec::with_capacity(path.increments.len());
        for inc in &path.increments {
            for i in 0..n {
                f[i] += inc[i];
            }
            step_weights.push(
                f.iter()
                    .map(|&fi| glm_hat_weight("binomial", fi).expect("binomial weight"))
                    .collect(),
            );
        }
        let (trace_bf, diag_bf) = brute_force_hat_path(
            &learners,
            &path.selected,
            &path.coefs,
            nu,
            Some(&step_weights),
        );
        let (trace, diag) = try_compute_hat_weighted_steps(
            &learners,
            &path.selected,
            Some(&path.coefs),
            nu,
            &step_weights,
        )
        .unwrap()
        .expect("tree steps resolve from the recorded path");
        for (got, want) in trace.iter().zip(&trace_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "weighted trace: {got} vs brute force {want}"
            );
        }
        for (got, want) in diag.iter().zip(&diag_bf) {
            assert!(
                (got - want).abs() < 1e-10,
                "weighted diag: {got} vs brute force {want}"
            );
        }
        // Without the coefficients the weighted call still refuses.
        assert_eq!(
            try_compute_hat_weighted(&learners, &path.selected, nu, &step_weights).unwrap(),
            None
        );
    }

    /// A selected `bmono` learner still refuses in every steps variant:
    /// the tree convention does not extend to it.
    #[test]
    fn bmono_still_refuses_in_steps_variants() {
        use crate::learners::{prepare_bmono, prepare_scalar_bols, BmonoSolver, MonoConstraint};

        let n = 24usize;
        let x = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let w = Array1::ones(n);
        let learners = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_bmono(
                x.view(),
                w.view(),
                10,
                3,
                2,
                4.0,
                MonoConstraint::Increasing,
                BmonoSolver::Iterative,
                1e6,
                10,
                "z",
                None,
                None,
                false,
            ),
        ];
        let selected = vec![0usize, 1, 0];
        let coefs: Vec<Vec<f64>> = vec![vec![0.0], vec![0.0; 8], vec![0.0]];
        let weights: Vec<Vec<f64>> = vec![vec![1.0; n]; 3];
        assert_eq!(
            try_compute_aic_trace_steps(&learners, &selected, Some(&coefs), 0.1).unwrap(),
            None
        );
        assert_eq!(
            try_compute_hat_diag_steps(&learners, &selected, Some(&coefs), 0.1).unwrap(),
            None
        );
        assert_eq!(
            try_compute_hat_weighted_steps(&learners, &selected, Some(&coefs), 0.1, &weights)
                .unwrap(),
            None
        );
    }

    /// Verify that the factor X·M·(XᵀW) reconstructs hat_matrix() for each
    /// learner variant (scalar, centered scalar, block, penalised).
    #[test]
    fn hat_factor_matches_hat_matrix() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 50usize;
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));
        let w = Array1::ones(n);
        let w_f64 = w.view();

        // Scalar Bols: no centering, no penalty
        let l = prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None);
        verify_factor(&l, w.as_slice().unwrap());

        // Scalar Bols: with centering
        let lc = prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None);
        verify_factor(&lc, w.as_slice().unwrap());

        // Scalar Bols: penalised
        let lp = prepare_scalar_bols(x.view(), w_f64, 5.0, false, "x", None);
        verify_factor(&lp, w.as_slice().unwrap());

        // Scalar Bols: penalised + centered
        let lpc = prepare_scalar_bols(x.view(), w_f64, 5.0, true, "x", None);
        verify_factor(&lpc, w.as_slice().unwrap());

        // BBS (BlockBols)
        let lb = prepare_bbs(x.view(), w_f64, 20, 3, 2, 4.0, false, "bbs(x)");
        verify_factor(&lb, w.as_slice().unwrap());

        // BBS with non-uniform weights
        let w2 = Array1::from_shape_fn(n, |i| 0.5 + (i as f64) / (n as f64));
        let lb2 = prepare_bbs(x.view(), w2.view(), 20, 3, 2, 4.0, false, "bbs(x)");
        verify_factor(&lb2, w2.as_slice().unwrap());
    }

    fn verify_factor(l: &PreparedLearner, w_slice: &[f64]) {
        let Some(factor) = l.hat_factor() else {
            panic!("hat_factor returned None for a learner that should support it");
        };
        let Some(h_mat) = l.hat_matrix() else {
            panic!("hat_matrix returned None");
        };
        let n = factor.design.nrows();
        let p = factor.design.ncols();
        // Reconstruct H_rebuilt = X · M · (XᵀW)
        // H_rebuilt[i,j] = Σ_k Σ_l X[i,k] · M[k,l] · (X[j,l] · w[j])
        //                = Σ_k X[i,k] · Σ_l M[k,l] · X[j,l] · w[j]
        let mut h_rebuilt = ndarray::Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..p {
                    let xik = factor.design.read(i, k);
                    let mut inner = 0.0;
                    for l in 0..p {
                        inner += factor.hat_inv[k * p + l] * factor.design.read(j, l);
                    }
                    val += xik * inner * w_slice[j];
                }
                h_rebuilt[[i, j]] = val;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (h_rebuilt[[i, j]] - h_mat[[i, j]]).abs();
                assert!(diff < 1e-12, "rebuild mismatch at ({i},{j}): {:.2e}", diff);
            }
        }
        // Also verify the trace matches
        let tr_rebuilt: f64 = (0..n).map(|i| h_rebuilt[[i, i]]).sum();
        assert!(
            (tr_rebuilt - factor.tr_h).abs() < 1e-12,
            "trace mismatch: rebuilt={tr_rebuilt}, factor={}",
            factor.tr_h
        );
    }

    /// `df()` / `df_mboost()` via the low-rank factor path equals the dense
    /// hat-matrix computation, and both match what `verify_factor` builds.
    #[test]
    fn df_fast_matches_dense_hat_matrix() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 50usize;
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));
        let w = Array1::ones(n);
        let w_f64 = w.view();

        let cases: Vec<PreparedLearner> = vec![
            prepare_scalar_bols(x.view(), w_f64, 0.0, false, "x", None),
            prepare_scalar_bols(x.view(), w_f64, 0.0, true, "x", None),
            prepare_scalar_bols(x.view(), w_f64, 5.0, false, "x", None),
            prepare_scalar_bols(x.view(), w_f64, 5.0, true, "x", None),
            prepare_bbs(x.view(), w_f64, 20, 3, 2, 4.0, false, "bbs(x)"),
        ];

        for l in &cases {
            let df_fast = l.df().expect("df fast None");
            let df_dense = dense_df(l).expect("df dense None");
            assert!(
                (df_fast - df_dense).abs() < 1e-9,
                "df fast {df_fast} != dense {df_dense} for {}",
                l.name()
            );

            let dfm_fast = l.df_mboost().expect("df_mboost fast None");
            let dfm_dense = dense_df_mboost(l).expect("df_mboost dense None");
            assert!(
                (dfm_fast - dfm_dense).abs() < 1e-9,
                "df_mboost fast {dfm_fast} != dense {dfm_dense} for {}",
                l.name()
            );
        }
    }

    fn dense_df(l: &PreparedLearner) -> Option<f64> {
        let h = l.hat_matrix()?;
        Some((0..h.nrows()).map(|i| h[[i, i]]).sum())
    }

    fn dense_df_mboost(l: &PreparedLearner) -> Option<f64> {
        let h = l.hat_matrix()?;
        let n = h.nrows();
        let mut value = 0.0;
        for i in 0..n {
            value += 2.0 * h[[i, i]];
            for j in 0..n {
                value -= h[[j, i]] * h[[j, i]];
            }
        }
        Some(value)
    }

    /// Verify that the fast AIC trace matches the dense reference trace.
    #[test]
    fn aic_trace_fast_matches_dense() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 100usize;
        let mut rng = 12345u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::ones(n);

        // Build a mixed learner set: centered scalar, uncentered scalar, BBS
        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 3.0, false, "x2", None);
        let l3 = prepare_bbs(x1.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x1)");
        let l4 = prepare_scalar_bols(x1.view(), w.view(), 0.0, false, "x1_plain", None);

        let learners = vec![l1, l2, l3, l4];
        let selected: Vec<usize> = (0..40).map(|i| i % 4).collect();

        let dense = try_compute_aic_trace(&learners, &selected, 0.1)
            .unwrap()
            .unwrap();
        let fast = try_compute_aic_trace_fast(&learners, &selected, 0.1)
            .unwrap()
            .unwrap();

        assert_eq!(
            dense.len(),
            fast.len(),
            "trace length mismatch: {} vs {}",
            dense.len(),
            fast.len()
        );
        for (i, (d, f)) in dense.iter().zip(fast.iter()).enumerate() {
            let diff = (d - f).abs();
            assert!(
                diff < 1e-10,
                "trace mismatch at step {i}: dense={d:.12e}, fast={f:.12e}, diff={diff:.2e}"
            );
        }

        // Also test with the fallback guard: construct enough learners to make P large
        let many_learners: Vec<_> = (0..10)
            .map(|i| {
                prepare_scalar_bols(
                    Array1::from_shape_fn(n, |_| xs()).view(),
                    w.view(),
                    if i % 2 == 0 { 0.0 } else { 1.0 },
                    i % 3 == 0,
                    &format!("x{i}"),
                    None,
                )
            })
            .collect();
        let selected2: Vec<usize> = (0..30).map(|i| i % 10).collect();
        let dense2 = try_compute_aic_trace(&many_learners, &selected2, 0.1)
            .unwrap()
            .unwrap();
        let fast2 = try_compute_aic_trace_fast(&many_learners, &selected2, 0.1)
            .unwrap()
            .unwrap();
        for (i, (d, f)) in dense2.iter().zip(fast2.iter()).enumerate() {
            let diff = (d - f).abs();
            assert!(
                diff < 1e-10,
                "many-learner trace mismatch at step {i}: dense={d:.12e}, fast={f:.12e}, diff={diff:.2e}"
            );
        }
    }

    /// Verify that compute_hat_diag returns a vector whose sum equals the
    /// final trace from compute_aic_trace.
    #[test]
    fn hat_diag_sum_matches_trace() {
        use crate::learners::prepare_bbs;
        use crate::learners::prepare_scalar_bols;

        let n = 50usize;
        let mut rng = 12345u64;
        let mut xs = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64)
        };
        let x1 = Array1::from_shape_fn(n, |_| xs());
        let x2 = Array1::from_shape_fn(n, |_| xs());
        let w = Array1::ones(n);

        let l1 = prepare_scalar_bols(x1.view(), w.view(), 0.0, true, "x1", None);
        let l2 = prepare_scalar_bols(x2.view(), w.view(), 2.0, false, "x2", None);
        let l3 = prepare_bbs(x1.view(), w.view(), 10, 3, 2, 4.0, false, "bbs(x1)");

        let learners = vec![l1, l2, l3];
        let selected: Vec<usize> = (0..20).map(|i| i % 3).collect();
        let nu = 0.1f64;

        let trace = try_compute_aic_trace(&learners, &selected, nu)
            .unwrap()
            .unwrap();
        let diag = try_compute_hat_diag(&learners, &selected, nu)
            .unwrap()
            .unwrap();

        assert_eq!(diag.len(), n, "hat diag length mismatch");
        let diag_sum: f64 = diag.iter().sum();
        let final_trace = *trace.last().unwrap();
        assert!(
            (diag_sum - final_trace).abs() < 1e-12,
            "hat diag sum ({diag_sum:.12}) != final trace ({final_trace:.12})"
        );
        // All leverages should be non-negative and ≤ 1
        for (i, &h) in diag.iter().enumerate() {
            assert!(h >= 0.0, "negative leverage at obs {i}: {h}");
            assert!(h <= 1.0 + 1e-10, "leverage > 1 at obs {i}: {h}");
        }
    }

    // ─── Regression tests from Audit 1 & 2 ────────────────────────────────

    #[test]
    fn fit_constant_response_produces_zero_selection() {
        // T2 edge case: all y identical → offset is that value,
        // all residuals are zero → no learner should reduce SSE.
        let n = 20usize;
        let y = Array1::from_elem(n, 5.0);
        let w = Array1::ones(n);
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        assert!((path.offset - 5.0).abs() < 1e-12);
        // Risk should never increase (stays at ~0 since residuals are 0)
        for i in 1..path.risk.len() {
            assert!(
                path.risk[i] <= path.risk[i - 1] + 1e-10,
                "risk increased at step {i}: {} > {}",
                path.risk[i],
                path.risk[i - 1]
            );
        }
        // Fitted values should remain at the offset (residuals were 0)
        for i in 0..n {
            assert!(
                (path.fitted[i] - 5.0).abs() < 1e-10,
                "fitted[{i}] should stay at offset, got {}",
                path.fitted[i]
            );
        }
    }

    #[test]
    fn fit_single_observation() {
        // T2 edge case: n=1 — single obs with one predictor.
        let y = array![3.5];
        let w = array![1.0];
        let x = array![2.0];

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(3, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        assert!((path.offset - 3.5).abs() < 1e-12);
        assert_eq!(path.selected.len(), 3);
        assert!(
            path.risk.iter().all(|r| r.is_finite()),
            "risk not finite: {:?}",
            path.risk
        );
        assert!(path.fitted.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fit_all_zero_weights_produces_finite_output() {
        // T2 edge case: all weights zero → offset = 0, risk = 0, no update.
        let n = 10usize;
        let y = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::zeros(n);
        let x = Array1::from_shape_fn(n, |i| (i as f64) * 2.0);

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(5, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        assert!(path.offset.is_finite());
        assert!(path.risk.iter().all(|r| r.is_finite()));
        assert!(path.fitted.iter().all(|v| v.is_finite()));
        // All risk values should be 0 (zero-weight SSE = 0)
        for r in &path.risk {
            assert!(
                r.abs() < 1e-12,
                "risk should be 0 with zero weights, got {r}"
            );
        }
    }

    #[test]
    fn fit_zero_variance_predictor() {
        // T2 edge case: predictor is constant → centered design is all-zero
        // → denom = 0 → beta stays 0 → no update.
        let n = 20usize;
        let y = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let w = Array1::ones(n);
        let x = Array1::from_elem(n, 5.0);

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(10, 0.1).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        // Fitted values should stay at offset (no update from zero-variance predictor)
        for i in 0..n {
            assert!(
                (path.fitted[i] - path.offset).abs() < 1e-12,
                "fitted[{i}] should equal offset, got {}",
                path.fitted[i]
            );
        }
    }

    #[test]
    fn fit_no_centering_zero_variance_predictor() {
        // Without centering, a constant predictor gets β = 0 because numer = 0
        // (x'Wu = 0 when x constant and u sums to 0 due to centered offset).
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let w = ndarray::Array1::from_elem(10, 1.0);
        let x = ndarray::Array1::from_elem(10, 5.0); // constant column

        let l = prepare_scalar_bols(x.view(), w.view(), 0.0, false, "x", None);
        let mut learners = vec![l];
        let ctrl = BoostControl::new(3, 1.0).unwrap();
        let mut fam = Gaussian;
        let path = fit(&mut learners, y.view(), w.view(), &mut fam, &ctrl, None);

        // β = numer/denom where numer = Σ w_i x_i u_i.
        // With x all 5 and spread parameter, numer = 5 * Σ w_i u_i.
        // The engine's centering inside fit_with ensures Σ u_i = 0 per step.
        // Since we're NOT centering here, check that coef is finite.
        assert!(path.coefs.iter().all(|c| c[0].is_finite()));
        assert!(path.risk.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_compute_aic_edge_cases() {
        // sumw = 0 → INFINITY for all iterations
        let (aic, _) = try_compute_aic("corrected", &[0.0, 10.0], &[1.0], 0.0, 2.0, 100.0).unwrap();
        assert!(aic[0].is_infinite());

        // corrected: denominator approaches zero when sumw ≈ d + 2
        let (aic, _) = try_compute_aic("corrected", &[0.0, 5.0], &[8.0], 10.0, 2.0, 100.0).unwrap();
        assert!(
            aic[0].is_finite(),
            "corrected AIC should clamp denom -> finite"
        );

        // gmdl: d = 0 should not panic or NaN
        let (aic, _) = try_compute_aic("gmdl", &[0.0, 5.0], &[0.0], 10.0, 2.0, 100.0).unwrap();
        assert!(aic[0].is_finite());

        // classical: always finite for finite inputs
        let (aic, _) = try_compute_aic("classical", &[0.0, 5.0], &[1.0], 10.0, 2.0, 100.0).unwrap();
        assert!(aic[0].is_finite());

        // No NaN anywhere for any method
        for method in &["corrected", "classical", "gmdl"] {
            let (aic, _) =
                try_compute_aic(method, &[0.0, 5.0, 4.0], &[1.0, 1.5], 10.0, 2.0, 100.0).unwrap();
            assert!(
                aic.iter().all(|v| !v.is_nan()),
                "NaN in {method} AIC: {aic:?}"
            );
        }
    }

    #[test]
    fn test_compute_aic_finite_normal_case() {
        let risk = vec![0.0, 10.0, 9.0, 8.0];
        let df = vec![2.0, 2.0, 2.0];
        let (aic, _) = try_compute_aic("corrected", &risk, &df, 10.0, 2.0, 100.0).unwrap();
        assert!(aic.iter().all(|v| v.is_finite()));
    }

    // ── Stochastic component subsampling ────────────────────────────────
    mod subsampling {
        use super::*;
        use crate::learners::prepare_scalar_bols;

        fn scalar_dataset(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
            let mut state = seed.max(1);
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let cols = Array2::from_shape_fn((n, p), |_| next() - 0.5);
            // Only the first three columns carry signal.
            let y = Array1::from_shape_fn(n, |i| {
                1.5 * cols[[i, 0]] - 0.9 * cols[[i, 1]] + 0.6 * cols[[i, 2]] + 0.05 * (next() - 0.5)
            });
            (cols, y)
        }

        fn build(cols: &Array2<f64>, w: &Array1<f64>) -> Vec<PreparedLearner> {
            (0..cols.ncols())
                .map(|j| {
                    prepare_scalar_bols(cols.column(j), w.view(), 0.0, true, &format!("x{j}"), None)
                })
                .collect()
        }

        #[test]
        fn full_fraction_equals_no_subsampling() {
            // fraction so high it rounds to every learner -> None path -> identical.
            let (cols, y) = scalar_dataset(200, 8, 7);
            let w = Array1::ones(200);
            let base = BoostControl::new(50, 0.1).unwrap();
            let mut a = build(&cols, &w);
            let mut fam_a = Gaussian;
            let ref_path = try_fit(&mut a, y.view(), w.view(), &mut fam_a, &base, None).unwrap();

            // 0.99 * 8 rounds to 8 == n_learners, so the mask collapses to None.
            let sub = base.with_component_subsampling(0.99, 123).unwrap();
            let mut b = build(&cols, &w);
            let mut fam_b = Gaussian;
            let sub_path = try_fit(&mut b, y.view(), w.view(), &mut fam_b, &sub, None).unwrap();

            assert_eq!(ref_path.selected, sub_path.selected);
        }

        #[test]
        fn same_seed_gives_identical_path() {
            let (cols, y) = scalar_dataset(200, 12, 3);
            let w = Array1::ones(200);
            let control = BoostControl::new(80, 0.1)
                .unwrap()
                .with_component_subsampling(0.5, 999)
                .unwrap();

            let mut a = build(&cols, &w);
            let mut fam_a = Gaussian;
            let pa = try_fit(&mut a, y.view(), w.view(), &mut fam_a, &control, None).unwrap();
            let mut b = build(&cols, &w);
            let mut fam_b = Gaussian;
            let pb = try_fit(&mut b, y.view(), w.view(), &mut fam_b, &control, None).unwrap();

            assert_eq!(pa.selected, pb.selected);
            assert_eq!(pa.fitted, pb.fitted);
        }

        #[test]
        fn different_seed_changes_the_path() {
            let (cols, y) = scalar_dataset(200, 12, 5);
            let w = Array1::ones(200);
            let base = BoostControl::new(80, 0.1).unwrap();
            let c1 = base.clone().with_component_subsampling(0.4, 1).unwrap();
            let c2 = base.with_component_subsampling(0.4, 2).unwrap();

            let mut a = build(&cols, &w);
            let mut fam_a = Gaussian;
            let p1 = try_fit(&mut a, y.view(), w.view(), &mut fam_a, &c1, None).unwrap();
            let mut b = build(&cols, &w);
            let mut fam_b = Gaussian;
            let p2 = try_fit(&mut b, y.view(), w.view(), &mut fam_b, &c2, None).unwrap();

            assert_ne!(p1.selected, p2.selected);
        }

        #[test]
        fn subsampled_fit_still_reduces_risk_and_recovers_signal() {
            let (cols, y) = scalar_dataset(400, 20, 9);
            let w = Array1::ones(400);
            let control = BoostControl::new(300, 0.1)
                .unwrap()
                .with_component_subsampling(0.3, 42)
                .unwrap();
            let mut learners = build(&cols, &w);
            let mut fam = Gaussian;
            let path =
                try_fit(&mut learners, y.view(), w.view(), &mut fam, &control, None).unwrap();

            // Risk falls monotonically overall.
            assert!(path.risk.last().unwrap() < &path.risk[0]);
            // The three signal columns are among the most-selected.
            let mut counts = vec![0usize; 20];
            for &s in &path.selected {
                counts[s] += 1;
            }
            let top3: std::collections::HashSet<usize> = {
                let mut idx: Vec<usize> = (0..20).collect();
                idx.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
                idx.into_iter().take(3).collect()
            };
            let signal: std::collections::HashSet<usize> = [0, 1, 2].into_iter().collect();
            assert!(
                top3.intersection(&signal).count() >= 2,
                "subsampled fit did not concentrate on the signal columns: {counts:?}"
            );
        }

        #[test]
        fn invalid_fraction_is_rejected() {
            let base = BoostControl::new(10, 0.1).unwrap();
            assert!(base.clone().with_component_subsampling(0.0, 1).is_err());
            assert!(base.clone().with_component_subsampling(1.0, 1).is_err());
            assert!(base.with_component_subsampling(f64::NAN, 1).is_err());
        }

        #[test]
        fn report_counts_subsampled_iterations() {
            let (cols, y) = scalar_dataset(200, 10, 4);
            let w = Array1::ones(200);
            let control = BoostControl::new(60, 0.1)
                .unwrap()
                .with_component_subsampling(0.5, 7)
                .unwrap();
            let mut learners = build(&cols, &w);
            let mut fam = Gaussian;
            let path =
                try_fit(&mut learners, y.view(), w.view(), &mut fam, &control, None).unwrap();
            assert_eq!(path.report.component_subsample_iterations, 60);
        }
    }
}

#[cfg(test)]
mod warm_continuation {
    //! A continued path must equal a cold path of the same length.
    //!
    //! Not approximately: the first `m` iterations of a boosting fit are a
    //! deterministic function of the data, so resuming from the state they
    //! left has to reproduce the rest bit for bit. Anything less and
    //! `continue_to` would return a different model depending on how the
    //! caller got there, which is worse than being slow.

    use super::*;
    use crate::families::{Gaussian, Huber, NBinomial, Poisson, PropOdds};
    use crate::learners::prepare_scalar_bols;
    use ndarray::{Array1, Array2};

    fn design(n: usize, p: usize, seed: u64) -> Array2<f64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        Array2::from_shape_fn((n, p), |_| next())
    }

    fn learners(x: &Array2<f64>, w: ArrayView1<f64>) -> Vec<PreparedLearner> {
        (0..x.ncols())
            .map(|j| prepare_scalar_bols(x.column(j), w, 0.0, false, &format!("x{j}"), None))
            .collect()
    }

    fn assert_same_path(cold: &BoostingPath, warm: &BoostingPath) {
        assert_eq!(cold.selected, warm.selected, "selection sequence");
        assert_eq!(cold.offset.to_bits(), warm.offset.to_bits(), "offset");
        assert_eq!(cold.risk.len(), warm.risk.len(), "risk path length");
        for (index, (a, b)) in cold.risk.iter().zip(warm.risk.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "risk at iteration {index}");
        }
        assert_eq!(cold.coefs.len(), warm.coefs.len(), "coefficient count");
        for (index, (a, b)) in cold.coefs.iter().zip(warm.coefs.iter()).enumerate() {
            assert_eq!(a, b, "coefficients at iteration {index}");
        }
        for (index, (a, b)) in cold.fitted.iter().zip(warm.fitted.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "fitted value {index}");
        }
        assert_eq!(cold.nuisance_path, warm.nuisance_path, "nuisance path");
        assert_eq!(
            cold.nuisance_vec_path, warm.nuisance_vec_path,
            "vector nuisance path"
        );
    }

    /// Fit cold to `total`, fit cold to `part` then continue, compare.
    fn check<F: Family + Clone>(family: F, y: Array1<f64>, part: u32, total: u32) {
        let n = y.len();
        let x = design(n, 4, 7);
        let w = Array1::ones(n);
        let control_part = BoostControl::new(part, 0.1).unwrap();
        let control_total = BoostControl::new(total, 0.1).unwrap();

        let mut cold_family = family.clone();
        let mut cold_learners = learners(&x, w.view());
        let cold = try_fit(
            &mut cold_learners,
            y.view(),
            w.view(),
            &mut cold_family,
            &control_total,
            None,
        )
        .expect("cold fit");

        let mut warm_family = family.clone();
        let mut warm_learners = learners(&x, w.view());
        let partial = try_fit(
            &mut warm_learners,
            y.view(),
            w.view(),
            &mut warm_family,
            &control_part,
            None,
        )
        .expect("partial fit");
        assert_eq!(partial.selected.len(), part as usize);

        let mut continued_learners = learners(&x, w.view());
        let state = WarmStart::from_path(&partial);
        let continued = try_fit_continue(
            &mut continued_learners,
            y.view(),
            w.view(),
            &mut warm_family,
            &control_total,
            &state,
        )
        .expect("continued fit");

        assert_same_path(&cold, &continued);
    }

    fn response(n: usize, seed: u64) -> Array1<f64> {
        let x = design(n, 1, seed);
        Array1::from_shape_fn(n, |i| 2.0 * x[[i, 0]] + 0.3 * ((i % 7) as f64 - 3.0))
    }

    #[test]
    fn gaussian_continuation_equals_a_cold_path() {
        check(Gaussian, response(120, 3), 40, 100);
    }

    #[test]
    fn poisson_continuation_equals_a_cold_path() {
        let counts = Array1::from_shape_fn(120, |i| ((i * 7) % 5) as f64);
        check(Poisson, counts, 30, 90);
    }

    #[test]
    fn a_family_that_calibrates_continues_exactly() {
        // Huber recomputes its delta from the median absolute residual, a pure
        // function of `(y, f, w)`, so resuming reproduces it.
        check(Huber::new(None), response(120, 11), 25, 80);
    }

    #[test]
    fn a_family_with_a_fitted_nuisance_continues_exactly() {
        let counts = Array1::from_shape_fn(150, |i| ((i * 13) % 9) as f64);
        check(NBinomial::new(), counts, 20, 60);
    }

    #[test]
    fn a_path_dependent_calibration_continues_exactly() {
        // PropOdds seeds its coordinate descent from the current thresholds,
        // so the resumed fit restores them instead of recalibrating. Without
        // that restore this test fails on the nuisance path.
        let levels = Array1::from_shape_fn(200, |i| ((i % 4) + 1) as f64);
        check(PropOdds::new(4), levels, 15, 50);
    }

    #[test]
    fn continuing_past_the_retained_length_is_required() {
        let y = response(80, 5);
        let x = design(80, 3, 7);
        let w = Array1::ones(80);
        let control = BoostControl::new(30, 0.1).unwrap();
        let mut family = Gaussian;
        let mut prepared = learners(&x, w.view());
        let path = try_fit(
            &mut prepared,
            y.view(),
            w.view(),
            &mut family,
            &control,
            None,
        )
        .expect("fit");
        let state = WarmStart::from_path(&path);
        let mut again = learners(&x, w.view());
        let error = try_fit_continue(
            &mut again,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(30, 0.1).unwrap(),
            &state,
        )
        .expect_err("continuing to the same length must be refused");
        assert!(
            format!("{error}").contains("slice the retained path"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn a_warm_state_from_different_data_is_refused() {
        let y = response(90, 5);
        let other = response(90, 6);
        let x = design(90, 3, 7);
        let w = Array1::ones(90);
        let mut family = Gaussian;
        let mut prepared = learners(&x, w.view());
        let path = try_fit(
            &mut prepared,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(20, 0.1).unwrap(),
            None,
        )
        .expect("fit");

        let state = WarmStart::from_path(&path);
        let mut again = learners(&x, w.view());
        let error = try_fit_continue(
            &mut again,
            other.view(),
            w.view(),
            &mut family,
            &BoostControl::new(40, 0.1).unwrap(),
            &state,
        )
        .expect_err("a state fitted on other data must be refused");
        assert!(
            format!("{error}").contains("does not describe this fit"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn a_warm_state_of_the_wrong_length_is_refused() {
        let y = response(60, 5);
        let x = design(60, 2, 7);
        let w = Array1::ones(60);
        let mut family = Gaussian;
        let mut prepared = learners(&x, w.view());
        let path = try_fit(
            &mut prepared,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(10, 0.1).unwrap(),
            None,
        )
        .expect("fit");
        let short = Array1::<f64>::zeros(30);
        let state = WarmStart {
            fitted: short.view(),
            ..WarmStart::from_path(&path)
        };
        let mut again = learners(&x, w.view());
        let error = try_fit_continue(
            &mut again,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(20, 0.1).unwrap(),
            &state,
        )
        .expect_err("a mismatched fitted vector must be refused");
        assert!(
            format!("{error}").contains("fitted values"),
            "unexpected message: {error}"
        );
    }

    /// A `Gaussian` that counts how many times the engine asked it for a
    /// gradient, which is once per iteration plus once at setup.
    #[derive(Clone)]
    struct CountingGaussian {
        inner: Gaussian,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CountingGaussian {
        fn new() -> Self {
            Self {
                inner: Gaussian,
                calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }

        fn count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl Family for CountingGaussian {
        fn name(&self) -> &'static str {
            "counting-gaussian"
        }

        fn clone_family(&self) -> Box<dyn Family> {
            Box::new(self.clone())
        }

        fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
            self.inner.offset(y, w)
        }

        fn negative_gradient(
            &mut self,
            y: ArrayView1<f64>,
            f: ArrayView1<f64>,
            w: ArrayView1<f64>,
            out: ndarray::ArrayViewMut1<f64>,
        ) {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.negative_gradient(y, f, w, out);
        }

        fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
            self.inner.risk(y, f, w)
        }

        fn held_out_risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
            self.inner.held_out_risk(y, f, w)
        }

        fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
            self.inner.check_y(y)
        }
    }

    #[test]
    fn a_continuation_runs_only_the_iterations_it_adds() {
        // The point of resuming: the retained iterations are not replayed.
        let y = response(100, 21);
        let x = design(100, 3, 7);
        let w = Array1::ones(100);
        let control_part = BoostControl::new(30, 0.1).unwrap();
        let control_total = BoostControl::new(100, 0.1).unwrap();

        let mut cold = CountingGaussian::new();
        let mut cold_learners = learners(&x, w.view());
        try_fit(
            &mut cold_learners,
            y.view(),
            w.view(),
            &mut cold,
            &control_total,
            None,
        )
        .expect("cold");
        let cold_calls = cold.count();

        let mut warm = CountingGaussian::new();
        let mut warm_learners = learners(&x, w.view());
        let partial = try_fit(
            &mut warm_learners,
            y.view(),
            w.view(),
            &mut warm,
            &control_part,
            None,
        )
        .expect("partial");
        let after_partial = warm.count();

        let state = WarmStart::from_path(&partial);
        let mut more = learners(&x, w.view());
        try_fit_continue(
            &mut more,
            y.view(),
            w.view(),
            &mut warm,
            &control_total,
            &state,
        )
        .expect("continue");
        let continuation_calls = warm.count() - after_partial;

        // 70 added iterations plus the one setup evaluation, against 101 for
        // the cold path.
        assert_eq!(cold_calls, 101);
        assert_eq!(continuation_calls, 71);
    }

    #[test]
    fn continuing_twice_equals_one_cold_path() {
        // The interactive loop: fit, look, extend, look, extend.
        let y = response(140, 9);
        let x = design(140, 4, 7);
        let w = Array1::ones(140);
        let mut family = Gaussian;

        let mut cold_learners = learners(&x, w.view());
        let cold = try_fit(
            &mut cold_learners,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(120, 0.1).unwrap(),
            None,
        )
        .expect("cold");

        let mut step = learners(&x, w.view());
        let mut path = try_fit(
            &mut step,
            y.view(),
            w.view(),
            &mut family,
            &BoostControl::new(30, 0.1).unwrap(),
            None,
        )
        .expect("first");
        for target in [70u32, 120] {
            let state = WarmStart::from_path(&path);
            let mut more = learners(&x, w.view());
            path = try_fit_continue(
                &mut more,
                y.view(),
                w.view(),
                &mut family,
                &BoostControl::new(target, 0.1).unwrap(),
                &state,
            )
            .expect("continue");
        }
        assert_same_path(&cold, &path);
    }

    // ─── line-search step rule (E1) ──────────────────────────────────────

    fn line_search(mstop: u32, nu: f64) -> BoostControl {
        BoostControl::new(mstop, nu)
            .unwrap()
            .with_step(crate::control::StepRule::LineSearch { max_iter: 4 })
    }

    #[test]
    fn line_search_gamma_reaches_the_weighted_log_linear_mean() {
        let n = 120;
        let x = Array1::linspace(-1.0, 1.0, n);
        let truth = x.mapv(|v: f64| 5000.0_f64.ln() + 0.2 * v);
        let y = truth.mapv(f64::exp);
        let w = Array1::from_shape_fn(n, |i| {
            if i < 8 {
                1000.0 + 100.0 * i as f64
            } else {
                1.0
            }
        });
        let ones = Array1::ones(n);
        let mut learners = vec![
            prepare_scalar_bols(ones.view(), w.view(), 0.0, false, "intercept", None),
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
        ];
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut crate::families::GammaReg::new(),
            &line_search(400, 1.0),
            None,
        )
        .unwrap();
        let worst = path
            .fitted
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-7,
            "weighted gamma mean did not converge: {worst}"
        );
        assert!(path.risk.windows(2).all(|p| p[1] <= p[0] + 1e-7));
    }

    #[test]
    fn line_search_leaves_a_squared_error_fit_bit_identical() {
        // For a squared-error family the least-squares step is already the
        // loss-optimal one, so enabling the search must change nothing --
        // not selection, not coefficients, not one bit of the fitted values.
        let n = 80usize;
        let x = Array1::from_shape_fn(n, |i| ((i as f64) / (n as f64)) * 2.0 - 1.0);
        let y = x.mapv(|v: f64| 1.0 + 0.7 * v)
            + Array1::from_shape_fn(n, |i| (((i * 37) % 19) as f64 / 19.0 - 0.5) * 0.4);
        let w = Array1::ones(n);
        let mut learners_fixed = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_scalar_bols((2.0 * &x).view(), w.view(), 0.0, true, "x2", None),
        ];
        let mut learners_searched = vec![
            prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
            prepare_scalar_bols((2.0 * &x).view(), w.view(), 0.0, true, "x2", None),
        ];
        let mut fixed = Gaussian;
        let mut searched = Gaussian;
        let fixed_path = try_fit(
            &mut learners_fixed,
            y.view(),
            w.view(),
            &mut fixed,
            &BoostControl::new(200, 0.1).unwrap(),
            None,
        )
        .unwrap();
        let searched_path = try_fit(
            &mut learners_searched,
            y.view(),
            w.view(),
            &mut searched,
            &line_search(200, 0.1),
            None,
        )
        .unwrap();
        assert_eq!(fixed_path.selected, searched_path.selected);
        assert_eq!(fixed_path.coefs, searched_path.coefs);
        let worst = fixed_path
            .fitted
            .iter()
            .zip(&searched_path.fitted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert_eq!(worst, 0.0, "the line search must be inactive for Gaussian");
    }

    #[test]
    fn line_search_keeps_a_large_count_poisson_fit_finite() {
        // Counts around 150 with nu=0.1: the constant step is about
        // nu*mean Newton steps, so iteration 2 pushes the linear predictor
        // past exp overflow and the fixed rule stops with the R-style
        // divergence error. Searching the step keeps every risk finite.
        let n = 200usize;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|v| 150.0 + 40.0 * v) + Array1::from_shape_fn(n, |i| (i % 7) as f64);
        let w = Array1::ones(n);
        let make = || {
            vec![
                prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None),
                prepare_scalar_bols(x.mapv(|v| v * v).view(), w.view(), 0.0, true, "x2", None),
            ]
        };
        let control = BoostControl::new(50, 0.1).unwrap();
        let error = try_fit(
            &mut make(),
            y.view(),
            w.view(),
            &mut Poisson,
            &control,
            None,
        )
        .expect_err("the fixed step must diverge at these counts");
        assert!(
            error.to_string().contains("risk became non-finite"),
            "{error}"
        );
        let text = error.to_string();
        assert!(
            text.contains("Poisson risk became non-finite after boosting iteration "),
            "{text}"
        );
        assert!(text.contains("nu = 0.1"), "{text}");
        assert!(text.contains("step=\"line_search\""), "{text}");

        let path = try_fit(
            &mut make(),
            y.view(),
            w.view(),
            &mut Poisson,
            &line_search(50, 0.1),
            None,
        )
        .expect("the line search must keep the fit finite");
        assert!(path.risk.iter().all(|r| r.is_finite()));
        assert!(
            *path.risk.last().unwrap() < path.risk[0],
            "the searched path must reduce the risk: {} -> {}",
            path.risk[0],
            path.risk.last().unwrap()
        );
    }

    #[test]
    fn line_search_reaches_a_step_below_the_grid_for_counts_near_1e5() {
        // Counts around 2e5 (house values stored as whole numbers): the
        // loss-optimal multiple of the least-squares direction is about
        // 1/mu = 5e-6, far below the grid's smallest point 0.125, at which
        // every step overflows. The search must halve down to it.
        let n = 200usize;
        let x = Array1::linspace(-1.0, 1.0, n);
        let y = x.mapv(|v| 2e5 + 8e4 * v) + Array1::from_shape_fn(n, |i| (i % 7) as f64);
        let w = Array1::ones(n);
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut Poisson,
            &line_search(50, 0.1),
            None,
        )
        .expect("the line search must keep the fit finite");
        assert!(path.risk.iter().all(|r| r.is_finite()));
        assert!(
            *path.risk.last().unwrap() < path.risk[0],
            "the searched path must reduce the risk: {} -> {}",
            path.risk[0],
            path.risk.last().unwrap()
        );
    }

    #[test]
    fn line_search_reaches_a_rare_event_risk_in_fewer_iterations() {
        // A 2% event rate: the curvature is 4p(1-p) ~ 0.08, so the fixed
        // step is about 12x smaller than a Newton step and the fixed rule
        // spends thousands of iterations. Searching the step must reach a
        // lower risk within 50 iterations than the fixed rule manages in
        // 300.
        let n = 500usize;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = Array1::from_shape_fn(n, |i| if i < 10 { 1.0 } else { 0.0 });
        let w = Array1::ones(n);
        let mut fixed_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut searched_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut fixed = crate::families::Binomial;
        let mut searched = crate::families::Binomial;
        let fixed_path = try_fit(
            &mut fixed_learners,
            y.view(),
            w.view(),
            &mut fixed,
            &BoostControl::new(300, 0.1).unwrap(),
            None,
        )
        .unwrap();
        let searched_path = try_fit(
            &mut searched_learners,
            y.view(),
            w.view(),
            &mut searched,
            &line_search(50, 0.1),
            None,
        )
        .unwrap();
        // At an equal budget the searched path is strictly ahead, and 50
        // searched iterations reach the fixed rule's 300-iteration risk to
        // within a thousandth of the reduction either of them achieves.
        let mut short_learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            true,
            "x",
            None,
        )];
        let mut short_fixed = crate::families::Binomial;
        let short_path = try_fit(
            &mut short_learners,
            y.view(),
            w.view(),
            &mut short_fixed,
            &BoostControl::new(50, 0.1).unwrap(),
            None,
        )
        .unwrap();
        assert!(
            *searched_path.risk.last().unwrap() < *short_path.risk.last().unwrap(),
            "at an equal budget the searched risk ({}) must beat the fixed one ({})",
            searched_path.risk.last().unwrap(),
            short_path.risk.last().unwrap()
        );
        let total_reduction = fixed_path.risk[0] - *fixed_path.risk.last().unwrap();
        assert!(
            *searched_path.risk.last().unwrap()
                <= *fixed_path.risk.last().unwrap() + 1e-3 * total_reduction,
            "50 searched iterations ({}) must reach the 300 fixed ones ({})",
            searched_path.risk.last().unwrap(),
            fixed_path.risk.last().unwrap()
        );
        // Each searched step moves downhill on the risk it minimizes, so the
        // path the search walks is non-increasing.
        assert!(searched_path
            .risk
            .windows(2)
            .all(|step| step[1] <= step[0] + 1e-9 * step[0].abs()));
    }

    #[test]
    fn line_search_coefficients_replay_the_fitted_path() {
        // The stored coefficient is rho*beta, so replaying it with only `nu`
        // in front must reconstruct the fitted values exactly. A stored
        // coefficient missing rho would replay nu*beta and miss by nu*(1-rho).
        let n = 120usize;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|v: f64| 2.0 + (3.0 * v).sin())
            + Array1::from_shape_fn(n, |i| (i % 5) as f64 * 0.1);
        let w = Array1::ones(n);
        let mut learners = vec![prepare_scalar_bols(
            x.view(),
            w.view(),
            0.0,
            false,
            "x",
            None,
        )];
        let mut family = Poisson;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &line_search(40, 0.1),
            None,
        )
        .unwrap();
        let nu = 0.1;
        let replayed: Vec<f64> = (0..n)
            .map(|i| path.offset + nu * path.coefs.iter().map(|coef| coef[0] * x[i]).sum::<f64>())
            .collect();
        let worst = path
            .fitted
            .iter()
            .zip(&replayed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "coefficient replay misses by {worst:.3e}");
    }

    #[test]
    fn line_search_keeps_a_tree_coefficient_decodable() {
        // A tree's coefficient vector encodes its node count, split variables
        // and split points. The line search used to multiply the whole vector
        // by rho, so a count of 3 nodes became 0.6 -> 0 nodes and prediction
        // indexed an empty tree. Only the leaf means carry rho.
        let n = 200usize;
        let x = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|v: f64| 600.0 + 200.0 * f64::from(u8::from(v > 0.5)));
        let w = Array1::ones(n);
        let design = x.clone().insert_axis(ndarray::Axis(1));
        let mut learners = vec![crate::learners::prepare_btree(
            design.view(),
            w.view(),
            7,
            20,
            1,
            None,
            "x",
        )];
        let mut family = Poisson;
        let path = try_fit(
            &mut learners,
            y.view(),
            w.view(),
            &mut family,
            &line_search(30, 0.1),
            None,
        )
        .unwrap();
        let nu = 0.1;
        let mut replayed = vec![path.offset; n];
        for coef in &path.coefs {
            let tree = crate::learners::FittedTree::from_coef_vec(ndarray::ArrayView1::from(
                coef.as_slice(),
            ));
            let pred = tree.predict(design.view());
            for i in 0..n {
                replayed[i] += nu * pred[i];
            }
        }
        let worst = path
            .fitted
            .iter()
            .zip(&replayed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "tree replay misses by {worst:.3e}");
    }
}

#[cfg(test)]
mod penalty_drift_tests {
    use super::*;
    use crate::learners::{prepare_bbs, smoother_spectrum};

    #[test]
    fn spectrum_reproduces_the_hat_trace_and_drift_rises_with_selections() {
        let n = 300;
        let x = Array1::from_shape_fn(n, |i| (i as f64 * 0.731).sin() * 2.0);
        let w = Array1::from_shape_fn(n, |i| 1.0 + (i % 3) as f64);
        let learner = prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, false, "bbs(x)");
        let spectrum = smoother_spectrum(&learner).unwrap();
        let trace: f64 = spectrum.iter().sum();
        let df: f64 = spectrum.iter().map(|s| 2.0 * s - s * s).sum();
        assert!((trace - learner.hat_factor().unwrap().tr_h).abs() < 1e-8);
        // mboost's df2lambda definition, 2 tr(S) - tr(S^2), at which the
        // learner was calibrated.
        assert!((df - 4.0).abs() < 1e-4, "{df}");

        let learners = vec![learner];
        let at = |m: usize| penalty_drift(&learners, &vec![0; m], 0.1).unwrap()[0].unwrap();
        let (few, many, huge) = (at(10), at(1_000), at(10_000_000));
        assert_eq!(few.selections, 10);
        assert!(few.boosted < many.boosted && many.boosted < huge.boosted);
        assert_eq!(huge.unpenalized, 24.0);
        assert!(huge.share() > 0.9, "{huge:?}");
        assert!(few.share() == 0.0);
        assert!(penalty_drift(&learners, &[1], 0.1).is_err());
        assert!(penalty_drift(&learners, &[0], 0.0).is_err());
    }
}
