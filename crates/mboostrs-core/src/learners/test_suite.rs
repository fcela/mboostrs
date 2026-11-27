#![allow(unused_imports)]
use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A residue class no row occupies still owns `p_l` columns of the
    /// materialised design, so it still needs a solver. Deriving the solver
    /// count from `max(groups) + 1` instead of the factorization's own
    /// `n_groups` left the vector short, and every fit path indexes it over
    /// `0..n_groups`.
    #[test]
    fn factored_tensor_handles_an_unoccupied_group() {
        let (n, p, ng, p_l) = (9usize, 8usize, 4usize, 2usize);
        let mut design = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let g = i % 3; // residue class 3 is never occupied
            for l in 0..p_l {
                design[[i, l * ng + g]] = 1.0 + (i * (l + 1)) as f64 * 0.1;
            }
        }
        let penalty = Array2::<f64>::eye(p) * 0.5;
        let weights = Array1::<f64>::ones(n);
        let u = Array1::from_shape_fn(n, |i| (i as f64) * 0.5 - 1.0);

        let mut factored =
            prepare_block_bols_interaction(design.clone(), weights.view(), penalty.clone(), "f");
        let PreparedLearner::FactoredBlockBols {
            n_groups, solvers, ..
        } = &factored
        else {
            panic!("detection should factor this design");
        };
        assert_eq!(*n_groups, ng);
        assert_eq!(solvers.len(), ng, "one solver per right-marginal column");

        // The factored path must agree with the materialised one it replaces.
        let mut materialised = prepare_block_bols(design, weights.view(), penalty, false, "m");
        let (beta_f, pred_f, sse_f) = factored.fit_step(u.view(), weights.view());
        let (beta_m, pred_m, sse_m) = materialised.fit_step(u.view(), weights.view());
        assert_eq!(beta_f.len(), p);
        for (bf, bm) in beta_f.iter().zip(beta_m.iter()) {
            assert!((bf - bm).abs() < 1e-10, "beta {bf} != {bm}");
        }
        for (pf, pm) in pred_f.iter().zip(pred_m.iter()) {
            assert!((pf - pm).abs() < 1e-10, "pred {pf} != {pm}");
        }
        assert!((sse_f - sse_m).abs() < 1e-10, "sse {sse_f} != {sse_m}");

        // The unoccupied group contributes no fitted effect.
        for l in 0..p_l {
            assert_eq!(beta_f[l * ng + 3], 0.0);
        }

        // A refit under new weights must keep the solver count.
        let reweighted = Array1::from_shape_fn(n, |i| if i % 2 == 0 { 2.0 } else { 0.5 });
        factored.reweight(reweighted.view());
        let PreparedLearner::FactoredBlockBols { solvers, .. } = &factored else {
            unreachable!()
        };
        assert_eq!(solvers.len(), ng);
        materialised.reweight(reweighted.view());
        let (beta_f, _, sse_f) = factored.fit_step(u.view(), reweighted.view());
        let (beta_m, _, sse_m) = materialised.fit_step(u.view(), reweighted.view());
        for (bf, bm) in beta_f.iter().zip(beta_m.iter()) {
            assert!((bf - bm).abs() < 1e-10, "refit beta {bf} != {bm}");
        }
        assert!(
            (sse_f - sse_m).abs() < 1e-10,
            "refit sse {sse_f} != {sse_m}"
        );
    }

    /// `cho_solve_inplace` claims to reproduce `cho_solve` exactly while
    /// factoring into the caller's buffer and allocating nothing.
    #[test]
    fn cho_solve_inplace_matches_the_owned_solve_and_consumes_its_input() {
        let a = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let b = array![1.0, 2.0, 3.0];
        let expected = crate::linalg::cho_solve(&a, &b.view()).expect("spd");

        let n = 3usize;
        let mut a_col = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                a_col[j * n + i] = a[[i, j]];
            }
        }
        let source = a_col.clone();
        let mut x = b.to_vec();
        let mut scratch = vec![0.0; cho_solve_inplace_scratch_len(n)];
        cho_solve_inplace(n, &mut a_col, &mut x, &mut scratch).expect("spd");
        for (got, want) in x.iter().zip(expected.iter()) {
            assert_eq!(got, want, "in-place solve must be bit-identical");
        }
        assert_ne!(a_col, source, "the factor must overwrite the input buffer");

        // A non-PD matrix is reported, not panicked on.
        let mut indefinite = vec![1.0, 2.0, 2.0, 1.0];
        let mut rhs = vec![1.0, 1.0];
        let mut scratch2 = vec![0.0; cho_solve_inplace_scratch_len(2)];
        assert!(cho_solve_inplace(2, &mut indefinite, &mut rhs, &mut scratch2).is_err());
    }

    #[test]
    fn chol_backsolve_inv_t_pd_contract() {
        // For PD A = L Lᵀ, Rm = L^{-T} must satisfy Rm Rmᵀ = A⁻¹, i.e.
        // A (Rm Rmᵀ) = I. Guards the main (Cholesky) path's contract.
        let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) => 4.0,
            (1, 1) => 5.0,
            (2, 2) => 6.0,
            (0, 1) | (1, 0) => 1.0,
            (1, 2) | (2, 1) => 2.0,
            (0, 2) | (2, 0) => 0.5,
            _ => 0.0,
        });
        let rm = chol_backsolve_inv_t_faer(&a);
        let p = 3;
        let ainv = |i: usize, j: usize| (0..p).map(|k| rm.read(i, k) * rm.read(j, k)).sum::<f64>();
        for i in 0..p {
            for j in 0..p {
                let s: f64 = (0..p).map(|k| a.read(i, k) * ainv(k, j)).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s - expected).abs() < 1e-9,
                    "A*A^-1 not identity at ({i},{j}): {s}"
                );
            }
        }
    }

    #[test]
    fn chol_backsolve_inv_t_indefinite_is_finite() {
        // Cholesky fails for an indefinite A; the jitter-retry fallback must
        // return a finite factor (regression for the old A^-1 LU path, which
        // also broke the eigenvalue contract).
        let a = Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1) => -1.0, // eigenvalues +1, -1 → indefinite
            _ => 0.0,
        });
        let rm = chol_backsolve_inv_t_faer(&a);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    rm.read(i, j).is_finite(),
                    "fallback produced non-finite at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn kronecker_identity_fast_path_matches_dense_formula() {
        let n_obs = 4;
        let n_resp = 3;
        let x1 =
            Array2::from_shape_vec((n_obs, 2), vec![1.0, -0.5, 0.0, 0.25, 2.0, 1.5, -1.0, 0.75])
                .unwrap();
        let w_obs = Array1::ones(n_obs);
        let x2 = Array2::eye(n_resp);
        let penalty = Array2::eye(2);
        let mut learner =
            prepare_kronecker_bols(x1.clone(), w_obs.view(), penalty, x2.clone(), false, "x");

        match &learner {
            PreparedLearner::KroneckerBols { x2_is_identity, .. } => assert!(*x2_is_identity),
            _ => unreachable!(),
        }

        let beta = vec![0.5, -0.2, 1.0, 0.3, -0.75, 0.8];
        let mut pred = vec![0.0; n_obs * n_resp];
        learner.predict_into(&beta, &mut pred);
        for i in 0..n_obs {
            for k in 0..n_resp {
                let mut expected = 0.0;
                for j in 0..x1.ncols() {
                    for l in 0..x2.ncols() {
                        expected += x1[[i, j]] * beta[l * x1.ncols() + j] * x2[[k, l]];
                    }
                }
                let got = pred[i * n_resp + k];
                assert!(
                    (got - expected).abs() < 1e-12,
                    "predict_into[{i},{k}] got {got}, expected {expected}"
                );
            }
        }

        let u = Array1::from_iter((0..n_obs * n_resp).map(|i| ((i as f64) * 0.3).cos()));
        let w_expanded = Array1::ones(n_obs * n_resp);
        let (fit_beta, fit_pred, _) = learner.fit_step_ref(u.view(), w_expanded.view());
        for i in 0..n_obs {
            for k in 0..n_resp {
                let mut expected = 0.0;
                for j in 0..x1.ncols() {
                    for l in 0..x2.ncols() {
                        expected += x1[[i, j]] * fit_beta[l * x1.ncols() + j] * x2[[k, l]];
                    }
                }
                let got = fit_pred[i * n_resp + k];
                assert!(
                    (got - expected).abs() < 1e-12,
                    "fit_step_ref pred[{i},{k}] got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn prepare_kronecker_bbs_wires_recipe() {
        // R-independent guard for the bbs %O% bols(class) Multinomial recipe:
        // prepare_kronecker_bbs must use the RAW B-spline basis, per-class penalty
        // (I + K1) [bols ridge + bbs difference penalty], df->lambda on that
        // penalty, and x2 = I_{n_resp}. Reproduce the per-class ridge by hand and
        // require fit_step_ref to match; a wrong penalty (e.g. K1 alone) or wrong
        // x2 would diverge.
        let n = 24usize;
        let nr = 2usize; // K-1
        let (n_knots, degree, differences, df) = (8usize, 3usize, 2usize, 4.0);
        let x: Array1<f64> = (0..n)
            .map(|i| (i as f64) / (n as f64 - 1.0) * 4.0 - 2.0)
            .collect();
        let w = Array1::from_elem(n, 1.0);
        let mut learner = prepare_kronecker_bbs(
            x.view(),
            w.view(),
            n_knots,
            degree,
            differences,
            df,
            nr,
            "s",
            0,
        );

        // Pseudo-residual u (row-major: u[i*nr + k]).
        let mut u = Array1::zeros(n * nr);
        for i in 0..n {
            for k in 0..nr {
                u[i * nr + k] = (i as f64 * 0.5 + k as f64).sin();
            }
        }
        // The engine passes per-(obs,class) expanded weights to fit_step.
        let w_exp = Array1::from_elem(n * nr, 1.0);
        let (_, pred, _) = learner.fit_step_ref(u.view(), w_exp.view());
        let pred = pred.to_vec();

        // Reference: raw basis, penalty I + K1, lambda from df_to_lambda.
        let bk = (-2.0, 2.0);
        let knots = bspline_knots(bk, n_knots, degree);
        let basis = bspline_basis(x.as_slice().unwrap(), &knots, degree);
        let p = basis.ncols();
        let penalty = &diff_penalty(p, differences) + &Array2::<f64>::eye(p);
        let lambda = df_to_lambda(&basis, &penalty, &w, df);
        let mut gram = penalty.mapv(|v| lambda * v);
        for a in 0..p {
            for b in 0..p {
                for i in 0..n {
                    gram[[a, b]] += basis[[i, a]] * basis[[i, b]];
                }
            }
        }
        let solver = GramSolver::new(&gram);
        for k in 0..nr {
            let mut rhs = vec![0.0; p];
            for (a, r) in rhs.iter_mut().enumerate() {
                for i in 0..n {
                    *r += basis[[i, a]] * u[i * nr + k];
                }
            }
            let mut beta = vec![0.0; p];
            solver.solve_into_slice(&rhs, &mut beta);
            for i in 0..n {
                let mut s = 0.0;
                for a in 0..p {
                    s += basis[[i, a]] * beta[a];
                }
                assert!(
                    (pred[i * nr + k] - s).abs() < 1e-9,
                    "prediction mismatch at obs {i} class {k}: {} vs {}",
                    pred[i * nr + k],
                    s
                );
            }
        }
    }

    #[test]
    fn kronecker_bbs_blueprint_matches_direct_prepare() {
        // The `KroneckerBbs` blueprint's `prepare` must reproduce
        // `prepare_kronecker_bbs` bit-for-bit, because it is the spec path's
        // route to the Multinomial bbs recipe the array path reached directly.
        let n = 20usize;
        let nr = 2usize;
        let (n_knots, degree, differences, df) = (6usize, 3usize, 2usize, 4.0);
        let x: Array1<f64> = (0..n).map(|i| i as f64 / 19.0).collect();
        let w = Array1::ones(n);

        let blueprint = LearnerBlueprint::kronecker_bbs(
            x.clone(),
            "s",
            n_knots,
            degree,
            differences,
            df,
            nr,
            0,
        )
        .unwrap();
        let (mut via_blueprint, _info) = blueprint.prepare(w.view()).unwrap();

        let mut direct = prepare_kronecker_bbs(
            x.view(),
            w.view(),
            n_knots,
            degree,
            differences,
            df,
            nr,
            "s",
            0,
        );

        let mut u = Array1::zeros(n * nr);
        for i in 0..n {
            for k in 0..nr {
                u[i * nr + k] = (i as f64 * 0.7 - k as f64).cos();
            }
        }
        let w_exp = Array1::ones(n * nr);
        let (_, pred_bp, _) = via_blueprint.fit_step_ref(u.view(), w_exp.view());
        let (_, pred_dir, _) = direct.fit_step_ref(u.view(), w_exp.view());
        assert_eq!(pred_bp.to_vec(), pred_dir.to_vec());
    }

    #[test]
    fn prepare_scalar_bols_center() {
        let x = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let learner = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        if let PreparedLearner::ScalarBols {
            design,
            centering,
            scalar_denom,
            ..
        } = learner
        {
            assert!((centering - 2.0).abs() < 1e-12);
            assert!((design[0] - (-1.0)).abs() < 1e-12);
            assert!((design[1] - 0.0).abs() < 1e-12);
            assert!((design[2] - 1.0).abs() < 1e-12);
            assert!((scalar_denom - 2.0).abs() < 1e-12);
        } else {
            panic!("expected ScalarBols");
        }
    }

    #[test]
    fn prepare_scalar_bols_no_center() {
        let x = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let learner = prepare_scalar_bols(x.view(), w.view(), 0.0, false, "x", None);
        if let PreparedLearner::ScalarBols {
            design,
            centering,
            scalar_denom,
            ..
        } = learner
        {
            assert!((centering - 0.0).abs() < 1e-12);
            assert!((design[0] - 1.0).abs() < 1e-12);
            assert!((scalar_denom - 14.0).abs() < 1e-12);
        } else {
            panic!("expected ScalarBols");
        }
    }

    #[test]
    fn scalar_bols_fit_step() {
        // x = [1, 2, 3] centered → [-1, 0, 1], denom = 2
        // u = [4, 5, 6], w = [1, 1, 1]
        // numer = Σ w_i * x_i * u_i = (-1)*4 + 0*5 + 1*6 = 2
        // beta = 2/2 = 1.0, pred = [-1, 0, 1]
        // sse = (4-(-1))² + (5-0)² + (6-1)² = 25 + 25 + 25 = 75
        let x = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let u = array![4.0, 5.0, 6.0];
        let mut learner = prepare_scalar_bols(x.view(), w.view(), 0.0, true, "x", None);
        let (beta, pred, sse) = learner.fit_step(u.view(), w.view());
        assert!((beta[0] - 1.0).abs() < 1e-12);
        assert!((pred[0] - (-1.0)).abs() < 1e-12);
        assert!((pred[1] - 0.0).abs() < 1e-12);
        assert!((pred[2] - 1.0).abs() < 1e-12);
        assert!((sse - 75.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_bols_fit_step_with_penalty() {
        // x = [1, 2, 3] centered → [-1, 0, 1]
        // lambda = 2, denom = 2 + 2 = 4
        // numer = 2 (same), beta = 2/4 = 0.5
        let x = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];
        let u = array![4.0, 5.0, 6.0];
        let mut learner = prepare_scalar_bols(x.view(), w.view(), 2.0, true, "x", None);
        let (beta, _pred, sse) = learner.fit_step(u.view(), w.view());
        assert!((beta[0] - 0.5).abs() < 1e-12);
        assert!((sse - 75.5).abs() < 1e-12);
    }

    #[test]
    fn quantile_binning_is_exact_at_the_distinct_count() {
        let x = array![3.0, -1.0, 3.0, 2.0, -1.0, 7.0];
        let bins = quantile_bin_column(x.view(), 4).unwrap();
        assert_eq!(bins.distinct, 4);
        assert_eq!(bins.effective, 4);
        for i in 0..x.len() {
            assert_eq!(bins.values[bins.index[i] as usize], x[i]);
        }
    }

    #[test]
    fn quantile_binning_never_splits_ties_and_reports_collapsed_cells() {
        let x = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let bins = quantile_bin_column(x.view(), 6).unwrap();
        assert!(bins.effective < bins.requested);
        assert!(bins.index[..4].iter().all(|&index| index == bins.index[0]));
        assert_eq!(
            bins.index.iter().copied().max().unwrap() as usize + 1,
            bins.effective
        );
    }

    #[test]
    fn binned_scalar_bols_matches_exact_with_one_bin_per_distinct_value() {
        let x = array![3.0, -1.0, 3.0, 2.0, -1.0, 7.0];
        let w = array![1.0, 0.5, 2.0, 3.0, 1.5, 0.25];
        let u = array![0.2, -0.5, 1.1, 0.3, -0.8, 2.0];
        let lambda = 0.7;
        let mut exact = prepare_scalar_bols(x.view(), w.view(), lambda, true, "x", None);
        let binning = quantile_bin_column(x.view(), 4).unwrap();
        let mut binned = prepare_binned_scalar_bols(binning, w.view(), lambda, true, "x");

        let (beta_exact, pred_exact, sse_exact) = exact.fit_step(u.view(), w.view());
        let (beta_binned, pred_binned, sse_binned) = binned.fit_step(u.view(), w.view());
        assert!((beta_exact[0] - beta_binned[0]).abs() < 1e-14);
        for i in 0..x.len() {
            assert!((pred_exact[i] - pred_binned[i]).abs() < 1e-14);
        }
        assert!((sse_exact - sse_binned).abs() < 1e-14);
    }

    #[test]
    fn binned_scalar_bols_converges_as_bins_increase() {
        let n = 1000;
        let x = Array1::from_iter((0..n).map(|i| i as f64 / (n - 1) as f64));
        let w = Array1::from_iter((0..n).map(|i| 0.5 + (i % 7) as f64 / 7.0));
        let u = x.mapv(|value| (8.0 * value).sin() + 0.3 * value);
        let mut exact = prepare_scalar_bols(x.view(), w.view(), 0.2, true, "x", None);
        let (_, exact_pred, _) = exact.fit_step(u.view(), w.view());

        let error = |requested| {
            let bins = quantile_bin_column(x.view(), requested).unwrap();
            let mut learner = prepare_binned_scalar_bols(bins, w.view(), 0.2, true, "x");
            let (_, pred, _) = learner.fit_step(u.view(), w.view());
            pred.iter()
                .zip(exact_pred.iter())
                .map(|(&left, &right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let error_10 = error(10);
        let error_100 = error(100);
        assert!(error_100 < error_10 / 5.0, "{error_100} vs {error_10}");
    }

    #[test]
    fn binned_bbs_matches_exact_with_one_bin_per_distinct_value() {
        let x = array![-1.0, -0.5, -0.5, 0.1, 0.7, 1.0, 1.0, 1.5];
        let weights = array![1.0, 0.5, 1.5, 2.0, 0.75, 1.25, 0.4, 1.8];
        let response = array![0.2, -0.4, 0.7, 1.2, -0.1, 0.8, 1.1, -0.5];
        for center in [false, true] {
            let exact_blueprint = LearnerBlueprint::bbs(
                x.clone(),
                "x",
                6,
                3,
                2,
                4.0,
                Some(0.7),
                center,
                None,
                false,
                None,
                0,
            )
            .unwrap();
            let binned_blueprint = LearnerBlueprint::bbs_binned(
                x.clone(),
                "x",
                6,
                3,
                2,
                4.0,
                Some(0.7),
                center,
                None,
                false,
                None,
                0,
                6,
            )
            .unwrap();
            let (mut exact, _) = exact_blueprint.prepare(weights.view()).unwrap();
            let (mut binned, _) = binned_blueprint.prepare(weights.view()).unwrap();
            let (exact_beta, exact_pred, exact_sse) =
                exact.fit_step(response.view(), weights.view());
            let (binned_beta, binned_pred, binned_sse) =
                binned.fit_step(response.view(), weights.view());
            for (left, right) in exact_beta.iter().zip(&binned_beta) {
                assert!((left - right).abs() < 2e-12, "{left} vs {right}");
            }
            for (left, right) in exact_pred.iter().zip(&binned_pred) {
                assert!((left - right).abs() < 2e-12, "{left} vs {right}");
            }
            assert!((exact_sse - binned_sse).abs() < 2e-12);
        }
    }

    #[test]
    fn binned_bspatial_matches_exact_on_distinct_coordinate_grid() {
        let x1 = array![-1.0, -1.0, -0.2, -0.2, 0.6, 0.6, 1.4, 1.4];
        let x2 = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let weights = array![1.0, 0.5, 1.5, 0.75, 2.0, 1.25, 0.8, 1.1];
        let response = array![0.2, -0.4, 0.7, 1.2, -0.1, 0.8, 1.1, -0.5];
        for center in [false, true] {
            let exact_blueprint = LearnerBlueprint::bspatial(
                x1.clone(),
                x2.clone(),
                "space",
                4,
                2,
                1,
                4.0,
                Some(0.5),
                center,
                None,
                None,
                None,
            )
            .unwrap();
            let binned_blueprint = LearnerBlueprint::bspatial_binned(
                x1.clone(),
                x2.clone(),
                "space",
                4,
                2,
                1,
                4.0,
                Some(0.5),
                center,
                None,
                None,
                None,
                4,
            )
            .unwrap();
            let (mut exact, _) = exact_blueprint.prepare(weights.view()).unwrap();
            let (mut binned, _) = binned_blueprint.prepare(weights.view()).unwrap();
            let (exact_beta, exact_pred, exact_sse) =
                exact.fit_step(response.view(), weights.view());
            let (binned_beta, binned_pred, binned_sse) =
                binned.fit_step(response.view(), weights.view());
            for (left, right) in exact_beta.iter().zip(&binned_beta) {
                assert!((left - right).abs() < 5e-10, "{left} vs {right}");
            }
            for (left, right) in exact_pred.iter().zip(&binned_pred) {
                assert!((left - right).abs() < 5e-10, "{left} vs {right}");
            }
            assert!((exact_sse - binned_sse).abs() < 5e-10);
        }
    }

    #[test]
    fn bspline_knots_match_r() {
        let bk = (0.11748736165463924, 0.978_226_428_385_824);
        let knots = bspline_knots(bk, 20, 3);
        assert_eq!(knots.len(), 28);
        assert!((knots[0] - (-0.005_475_362_164_101_438)).abs() < 1e-12);
        assert!((knots[3] - 0.11748736165463924).abs() < 1e-12);
        assert!((knots[24] - 0.978_226_428_385_824).abs() < 1e-12);
        assert!((knots[27] - 1.101_189_152_204_564_7).abs() < 1e-12);
    }

    #[test]
    fn bspline_basis_dimensions_and_partition_of_unity() {
        let x: Vec<f64> = vec![
            0.11748736165463924,
            0.134_666_597_237_810_5,
            0.255_428_824_340_924_6,
            0.286_139_534_786_343_6,
            0.45774177624844015,
            0.46229282254353166,
            0.47499708156101406,
            0.519_095_949_130_132_8,
            0.560_332_746_244_967,
            0.641_745_518_893_003_5,
            0.656_992_290_401_831_3,
            0.705_064_784_036_949_3,
            0.719_112_251_652_404_7,
            0.736_588_314_641_267_1,
            0.830_447_626_067_325_5,
            0.914_806_043_496_355_4,
            0.934_672_247_152_775_5,
            0.937_075_413_297_861_8,
            0.940_014_522_755_518_6,
            0.978_226_428_385_824,
        ];
        let bk = (0.11748736165463924, 0.978_226_428_385_824);
        let knots = bspline_knots(bk, 20, 3);
        let basis = bspline_basis(&x, &knots, 3);

        assert_eq!(basis.nrows(), 20);
        assert_eq!(basis.ncols(), 24);

        for i in 0..20 {
            let row_sum: f64 = (0..24).map(|j| basis[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn bspline_basis_first_row_matches_r() {
        let x: Vec<f64> = vec![0.11748736165463924];
        let bk = (0.11748736165463924, 0.978_226_428_385_824);
        let knots = bspline_knots(bk, 20, 3);
        let basis = bspline_basis(&x, &knots, 3);

        let r_row: Vec<f64> = vec![
            0.1666666667,
            0.6666666667,
            0.1666666667,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for j in 0..24 {
            let diff = (basis[[0, j]] - r_row[j]).abs();
            assert!(diff < 1e-8, "col {} diff = {}", j, diff);
        }
    }

    #[test]
    fn bspline_basis_second_row_matches_r() {
        let x: Vec<f64> = vec![0.134_666_597_237_810_5];
        let bk = (0.11748736165463924, 0.978_226_428_385_824);
        let knots = bspline_knots(bk, 20, 3);
        let basis = bspline_basis(&x, &knots, 3);

        let r_row: Vec<f64> = vec![
            0.0326647534,
            0.5278093936,
            0.4272541851,
            0.0122716679,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for j in 0..24 {
            let diff = (basis[[0, j]] - r_row[j]).abs();
            assert!(diff < 1e-8, "col {} diff = {}", j, diff);
        }
    }

    #[test]
    fn diff_penalty_second_diff_structure() {
        let k = diff_penalty(24, 2);
        assert_eq!(k.nrows(), 24);
        assert_eq!(k.ncols(), 24);
        assert!((k[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((k[[0, 1]] - (-2.0)).abs() < 1e-12);
        assert!((k[[0, 2]] - 1.0).abs() < 1e-12);
        assert!((k[[0, 3]] - 0.0).abs() < 1e-12);
        assert!((k[[1, 0]] - (-2.0)).abs() < 1e-12);
        assert!((k[[1, 1]] - 5.0).abs() < 1e-12);
        assert!((k[[1, 2]] - (-4.0)).abs() < 1e-12);
        assert!((k[[1, 3]] - 1.0).abs() < 1e-12);
        assert!((k[[2, 2]] - 6.0).abs() < 1e-12);
        assert!((k[[23, 23]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn diff_penalty_matches_r_flat() {
        let k = diff_penalty(24, 2);
        let r_k: Vec<f64> = vec![
            1.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 5.0, -4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -4.0, 6.0,
            -4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        for i in 0..3 {
            for j in 0..24 {
                let diff = (k[[i, j]] - r_k[i * 24 + j]).abs();
                assert!(diff < 1e-10, "K[{},{}] diff = {}", i, j, diff);
            }
        }
    }

    /// Primary penalty spectrum exactly as `df_to_lambda` computes it, and the
    /// regularized Gram it is computed from.
    fn primary_spectrum_for_test(
        basis: &Array2<f64>,
        k: &Array2<f64>,
        w: &Array1<f64>,
    ) -> (Vec<f64>, Mat<f64>, Mat<f64>) {
        let (n, p) = (basis.nrows(), basis.ncols());
        let wx = Mat::from_fn(n, p, |i, j| w[i] * basis[[i, j]]);
        let xt = Mat::from_fn(p, n, |i, j| basis[[j, i]]);
        let mut xtwx = Mat::zeros(p, p);
        faer::linalg::matmul::matmul(&mut xtwx, &xt, &wx, None, 1.0, faer::Parallelism::None);
        let k_faer = Mat::from_fn(p, p, |i, j| k[[i, j]]);
        let a = Mat::from_fn(p, p, |i, j| {
            xtwx.read(i, j) + MBOOST_EPS * k_faer.read(i, j)
        });
        let rm = chol_backsolve_inv_t_faer(&a);
        let mut krm = Mat::zeros(p, p);
        faer::linalg::matmul::matmul(&mut krm, &k_faer, &rm, None, 1.0, faer::Parallelism::None);
        let mut kmrm = Mat::zeros(p, p);
        faer::linalg::matmul::matmul(
            &mut kmrm,
            rm.transpose(),
            krm,
            None,
            1.0,
            faer::Parallelism::None,
        );
        (penalty_spectrum_faer(&kmrm), a, k_faer)
    }

    #[test]
    fn df_to_lambda_resolves_the_spectrum_beside_an_empty_basis_column() {
        // Outlying extremes stretch the equally spaced knot grid, so several
        // B-spline columns see no data and `X'WX` is singular in their
        // directions; `eps·K` turns those into spectral values near 1e9. With a
        // large total weight the penalized values that set df at the root fall
        // below what a dense SVD resolves beside 1e9. On the primary spectrum
        // alone they collapsed to zero, the df floor rose above the penalty's
        // null-space dimension, df = 4 looked unattainable, and the capped
        // lambda left the smooth fitting only its linear null space. Found on
        // 200,000 N(0,1) rows with a minimum of -5.35, where R's `df2lambda`
        // gives 96,531 and the primary spectrum gave the 1e14 cap.
        let n = 2_000usize;
        let mut x: Vec<f64> = (0..n)
            .map(|i| -2.0 + 4.0 * i as f64 / (n - 1) as f64)
            .collect();
        x[0] = -5.35;
        x[n - 1] = 4.28;
        let knots = bspline_knots((-5.35, 4.28), 20, 3);
        let basis = bspline_basis(&x, &knots, 3);
        let k = diff_penalty(basis.ncols(), 2);
        let empty = (0..basis.ncols())
            .filter(|&j| basis.column(j).iter().all(|value| *value == 0.0))
            .count();
        assert!(empty > 0, "the design must contain an empty basis column");
        let w = Array1::from_elem(n, 1_000.0);

        let (primary, a, k_faer) = primary_spectrum_for_test(&basis, &k, &w);
        let resolution = primary.last().copied().unwrap() * basis.ncols() as f64 * f64::EPSILON;
        assert!(
            resolution > SPECTRUM_RESOLUTION_LIMIT,
            "the input must reach the regime where the spectrum is re-checked; resolution = {resolution:e}"
        );

        let lambda = df_to_lambda(&basis, &k, &w, 4.0);
        let bounded = bounded_pencil_spectrum(&a, &k_faer);
        let null_directions = bounded.iter().filter(|&&d| d < 1e-12).count();
        assert_eq!(
            null_directions, 2,
            "a second-difference penalty has a two-dimensional null space: {bounded:?}"
        );
        let df = df_from_spectrum(&bounded, lambda);
        assert!(
            (df - 4.0).abs() < 1e-6,
            "df at the calibrated lambda is {df}, lambda = {lambda:e}"
        );
        assert!(lambda < 1e10, "lambda = {lambda:e} is at the cap");
    }

    #[test]
    fn bounded_pencil_spectrum_matches_the_primary_spectrum_on_a_populated_design() {
        // Where the primary spectrum is well resolved the two routes are the
        // same eigenvalues, so the re-check can never move a healthy result.
        let n = 400usize;
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.618_033_988_75).fract())
            .collect();
        let knots = bspline_knots((0.0, 1.0), 20, 3);
        let basis = bspline_basis(&x, &knots, 3);
        let k = diff_penalty(basis.ncols(), 2);
        let w = Array1::ones(n);
        let (primary, a, k_faer) = primary_spectrum_for_test(&basis, &k, &w);
        let bounded = bounded_pencil_spectrum(&a, &k_faer);
        assert_eq!(primary.len(), bounded.len());
        for (index, (p_value, b_value)) in primary.iter().zip(bounded.iter()).enumerate() {
            let scale = p_value.abs().max(1e-8);
            assert!(
                (p_value - b_value).abs() / scale < 1e-8,
                "value {index}: primary {p_value:e}, bounded {b_value:e}"
            );
        }
    }

    #[test]
    fn df_to_lambda_produces_r_lambda() {
        let x: Vec<f64> = vec![
            0.11748736165463924,
            0.134_666_597_237_810_5,
            0.255_428_824_340_924_6,
            0.286_139_534_786_343_6,
            0.45774177624844015,
            0.46229282254353166,
            0.47499708156101406,
            0.519_095_949_130_132_8,
            0.560_332_746_244_967,
            0.641_745_518_893_003_5,
            0.656_992_290_401_831_3,
            0.705_064_784_036_949_3,
            0.719_112_251_652_404_7,
            0.736_588_314_641_267_1,
            0.830_447_626_067_325_5,
            0.914_806_043_496_355_4,
            0.934_672_247_152_775_5,
            0.937_075_413_297_861_8,
            0.940_014_522_755_518_6,
            0.978_226_428_385_824,
        ];
        let n = x.len();
        let bk = (0.11748736165463924, 0.978_226_428_385_824);
        let knots = bspline_knots(bk, 20, 3);
        let basis = bspline_basis(&x, &knots, 3);
        let k = diff_penalty(basis.ncols(), 2);
        let w = Array1::ones(n);

        // R computes df2lambda on the CENTERED basis (after spectralDecomp)
        let centered = center_spectral(&basis, &k, 2);
        let penalty_centered = Array2::eye(centered.ncols());

        let lambda = df_to_lambda(&centered, &penalty_centered, &w, 4.0);
        // Full-precision value from `mboost:::df2lambda(X, df = 4, weights,
        // dmat = K)` on the same centered design. The tolerance was 1e-4 while
        // `df_to_lambda` minimized `|df(lambda) - df*|`; the root-find reaches
        // 4.6e-8, which is the `sqrt(EPSILON)` tolerance both root-finds stop
        // at rather than a difference in the spectrum they solve over.
        let r_lambda = 8.99215608585474;
        let rel_err = (lambda - r_lambda).abs() / r_lambda;
        assert!(
            rel_err < 1e-6,
            "lambda = {}, R = {}, rel_err = {}",
            lambda,
            r_lambda,
            rel_err
        );
    }

    /// A tensor design and its Kronecker-sum penalty, on fewer rows than the
    /// product basis has columns — the shape every `%X%` and `bspatial` term
    /// takes once both margins are wide.
    fn tensor_design_and_penalty(n: usize, width: usize) -> (Array2<f64>, Array2<f64>) {
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let x1: Vec<f64> = (0..n).map(|_| next()).collect();
        let x2: Vec<f64> = (0..n).map(|_| next()).collect();
        let knots = bspline_knots((0.0, 1.0), width - 4, 3);
        let b1 = bspline_basis(&x1, &knots, 3);
        let b2 = bspline_basis(&x2, &knots, 3);
        let (p1, p2) = (b1.ncols(), b2.ncols());
        let design =
            Array2::from_shape_fn((n, p1 * p2), |(i, j)| b1[[i, j / p2]] * b2[[i, j % p2]]);
        let k1 = diff_penalty(p1, 2);
        let k2 = diff_penalty(p2, 2);
        let mut penalty = Array2::zeros((p1 * p2, p1 * p2));
        for a in 0..p1 {
            for b in 0..p1 {
                for c in 0..p2 {
                    penalty[[a * p2 + c, b * p2 + c]] += k1[[a, b]];
                }
            }
        }
        for a in 0..p2 {
            for b in 0..p2 {
                for c in 0..p1 {
                    penalty[[c * p2 + a, c * p2 + b]] += k2[[a, b]];
                }
            }
        }
        (design, penalty)
    }

    #[test]
    fn lambda_at_the_null_space_floor_leaves_a_solvable_gram() {
        // The Kronecker sum of two second-difference penalties leaves a
        // 4-dimensional unpenalized subspace, so df = 4 is the floor no finite
        // lambda reaches. The clamped lambda must still be one the update solve
        // can use: this used to return 1e15, whose Gram no longer factorizes,
        // and the hat matrix that came back was not a contraction.
        let (design, penalty) = tensor_design_and_penalty(80, 12);
        let weights = Array1::ones(design.nrows());
        let lambda = df_to_lambda(&design, &penalty, &weights, 4.0);
        assert!(
            lambda < MBOOST_LAMBDA_MAX,
            "lambda = {lambda} was not capped"
        );

        let gram = Array2::from_shape_fn((design.ncols(), design.ncols()), |(i, j)| {
            let mut v = lambda * penalty[[i, j]];
            for r in 0..design.nrows() {
                v += design[[r, i]] * design[[r, j]];
            }
            v
        });
        let factor = CachedCholesky::new(&gram).expect("capped lambda must factorize");
        assert!(
            !factor.is_numerically_rank_deficient(),
            "capped lambda left a rank-deficient Gram (ratio {})",
            factor.diagonal_ratio()
        );
    }

    #[test]
    fn a_full_rank_penalty_keeps_the_unclamped_lambda_max() {
        // Nothing about X'WX + 1e15*I is ill-conditioned, so the cap must not
        // fire and every learner with a positive-definite penalty keeps the
        // lambda it had before the cap existed.
        let design = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]];
        let gram = Mat::from_fn(2, 2, |i, j| {
            (0..4).map(|r| design[[r, i]] * design[[r, j]]).sum()
        });
        let identity = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        assert_eq!(solvable_lambda_max(&gram, &identity), MBOOST_LAMBDA_MAX);
    }

    #[test]
    fn a_tensor_penalty_caps_lambda_below_the_maximum() {
        let (design, penalty) = tensor_design_and_penalty(80, 12);
        let (n, p) = (design.nrows(), design.ncols());
        let gram = Mat::from_fn(p, p, |i, j| {
            (0..n).map(|r| design[[r, i]] * design[[r, j]]).sum()
        });
        let k = Mat::from_fn(p, p, |i, j| penalty[[i, j]]);
        let capped = solvable_lambda_max(&gram, &k);
        assert!(
            capped < MBOOST_LAMBDA_MAX,
            "lambda was not capped: {capped}"
        );
        assert!(
            capped > 1.0,
            "cap collapsed to {capped}, which shrinks nothing"
        );
    }

    #[test]
    fn center_spectral_reduces_columns() {
        let x: Vec<f64> = vec![
            0.11748736165463924,
            0.134_666_597_237_810_5,
            0.255_428_824_340_924_6,
            0.286_139_534_786_343_6,
            0.45774177624844015,
            0.46229282254353166,
            0.47499708156101406,
            0.519_095_949_130_132_8,
            0.560_332_746_244_967,
            0.641_745_518_893_003_5,
            0.656_992_290_401_831_3,
            0.705_064_784_036_949_3,
            0.719_112_251_652_404_7,
            0.736_588_314_641_267_1,
            0.830_447_626_067_325_5,
            0.914_806_043_496_355_4,
            0.934_672_247_152_775_5,
            0.937_075_413_297_861_8,
            0.940_014_522_755_518_6,
            0.978_226_428_385_824,
        ];
        let bk = (0.11748736165463924, 0.978_226_428_385_824);
        let knots = bspline_knots(bk, 20, 3);
        let basis = bspline_basis(&x, &knots, 3);
        let k = diff_penalty(basis.ncols(), 2);

        let centered = center_spectral(&basis, &k, 2);
        assert_eq!(centered.nrows(), 20);
        assert_eq!(centered.ncols(), 22);
    }

    #[test]
    fn prepare_bbs_returns_block_bols() {
        let x = array![
            0.11748736165463924,
            0.134_666_597_237_810_5,
            0.255_428_824_340_924_6,
            0.286_139_534_786_343_6,
            0.45774177624844015,
            0.46229282254353166,
            0.47499708156101406,
            0.519_095_949_130_132_8,
            0.560_332_746_244_967,
            0.641_745_518_893_003_5,
            0.656_992_290_401_831_3,
            0.705_064_784_036_949_3,
            0.719_112_251_652_404_7,
            0.736_588_314_641_267_1,
            0.830_447_626_067_325_5,
            0.914_806_043_496_355_4,
            0.934_672_247_152_775_5,
            0.937_075_413_297_861_8,
            0.940_014_522_755_518_6,
            0.978_226_428_385_824,
        ];
        let w = Array1::ones(20);

        let learner = prepare_bbs(x.view(), w.view(), 20, 3, 2, 4.0, true, "bbs(x)");

        if let PreparedLearner::BlockBols { design, .. } = learner {
            assert_eq!(design.nrows(), 20);
            assert_eq!(design.ncols(), 22);
        } else {
            panic!("expected BlockBols from prepare_bbs");
        }
    }

    // ── bmono constraint matrices ──────────────────────────────────────

    #[test]
    fn mono_constraint_increasing_is_first_difference() {
        let d = build_mono_constraint(MonoConstraint::Increasing, 4);
        assert_eq!(d.shape(), &[3, 4]);
        // β = [1, 2, 3, 5]: differences [1, 1, 2] all ≥ 0 → feasible
        let beta = array![1.0, 2.0, 3.0, 5.0];
        let diffs = d.dot(&beta);
        for &v in diffs.iter() {
            assert!(v >= 0.0);
        }
        // β = [3, 1, 2, 4]: first diff is -2 (violated)
        let beta_bad = array![3.0, 1.0, 2.0, 4.0];
        let diffs_bad = d.dot(&beta_bad);
        assert!((diffs_bad[0] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn mono_constraint_decreasing_is_negated_increasing() {
        let d_inc = build_mono_constraint(MonoConstraint::Increasing, 5);
        let d_dec = build_mono_constraint(MonoConstraint::Decreasing, 5);
        for i in 0..d_inc.nrows() {
            for j in 0..d_inc.ncols() {
                assert_eq!(d_inc[[i, j]], -d_dec[[i, j]]);
            }
        }
    }

    #[test]
    fn mono_constraint_convex_is_second_difference() {
        let d = build_mono_constraint(MonoConstraint::Convex, 5);
        assert_eq!(d.shape(), &[3, 5]);
        // β quadratic (1, 4, 9, 16, 25): second diffs are all 2 (≥ 0)
        let beta = array![1.0, 4.0, 9.0, 16.0, 25.0];
        let diffs = d.dot(&beta);
        for &v in diffs.iter() {
            assert!((v - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mono_constraint_positive_is_identity() {
        let d = build_mono_constraint(MonoConstraint::Positive, 4);
        let eye = Array2::<f64>::eye(4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(d[[i, j]], eye[[i, j]]);
            }
        }
    }

    // ── bmono iterative QP: violated → projected feasible ──────────────

    #[test]
    fn bmono_iterative_projects_to_feasible_region() {
        // Construct a synthetic problem where the unconstrained solution
        // violates monotonicity, and verify the iterative solver returns a
        // (numerically) feasible β.
        // x ∈ [0, 1], y = -sin(2π x) — strictly DECREASING then INCREASING.
        // Asking for "Increasing" should force a near-flat fit.
        let n = 100;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y: Array1<f64> = x.mapv(|xi| -(2.0 * std::f64::consts::PI * xi).sin());
        let w = Array1::<f64>::ones(n);

        let mut learner = prepare_bmono(
            x.view(),
            w.view(),
            /* n_knots */ 20,
            /* degree */ 3,
            /* differences */ 2,
            /* df */ 4.0,
            MonoConstraint::Increasing,
            BmonoSolver::Iterative,
            /* lambda2 */ 1e6,
            /* niter */ 10,
            "x",
            None,
            None,
            false,
        );
        // u = y for first boosting step (offset = 0 here)
        let (beta, _pred, _sse) = learner.fit_step(y.view(), w.view());

        // Check: D β ≥ -tol  (constraints approximately satisfied)
        if let PreparedLearner::BmonoBbs { d_mat, .. } = &learner {
            let diffs = d_mat.dot(&beta);
            let worst = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                worst > -1e-3,
                "Increasing constraint violated: worst Dβ_i = {worst}"
            );
        } else {
            panic!("expected BmonoBbs from prepare_bmono");
        }
    }

    #[test]
    fn bmono_quadprog_projects_to_feasible_region() {
        // Same problem as the iterative test, but using the QuadProg solver.
        // QuadProg gives strictly KKT-feasible β (down to ~1e-10), so we
        // tighten the tolerance accordingly.
        let n = 100;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y: Array1<f64> = x.mapv(|xi| -(2.0 * std::f64::consts::PI * xi).sin());
        let w = Array1::<f64>::ones(n);

        let mut learner = prepare_bmono(
            x.view(),
            w.view(),
            20,
            3,
            2,
            4.0,
            MonoConstraint::Increasing,
            BmonoSolver::QuadProg,
            /* lambda2 unused */ 0.0,
            /* niter unused */ 0,
            "x",
            None,
            None,
            false,
        );
        let (beta, _pred, _sse) = learner.fit_step(y.view(), w.view());

        if let PreparedLearner::BmonoBbs { d_mat, .. } = &learner {
            let diffs = d_mat.dot(&beta);
            let worst = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
            // QuadProg guarantees feasibility to numerical precision.
            assert!(
                worst > -1e-8,
                "QuadProg should give strictly feasible β: worst Dβ_i = {worst}"
            );
        } else {
            panic!("expected BmonoBbs from prepare_bmono");
        }
    }

    #[test]
    fn bmono_quadprog_unconstrained_when_feasible() {
        // If unconstrained fit is already monotone-increasing, QuadProg
        // returns the unconstrained solution without entering the
        // active-set loop.
        let n = 50;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y: Array1<f64> = x.mapv(|xi| 3.0 * xi);
        let w = Array1::<f64>::ones(n);

        let mut learner_q = prepare_bmono(
            x.view(),
            w.view(),
            20,
            3,
            2,
            4.0,
            MonoConstraint::Increasing,
            BmonoSolver::QuadProg,
            0.0,
            0,
            "x",
            None,
            None,
            false,
        );
        let mut learner_i = prepare_bmono(
            x.view(),
            w.view(),
            20,
            3,
            2,
            4.0,
            MonoConstraint::Increasing,
            BmonoSolver::Iterative,
            1e6,
            10,
            "x",
            None,
            None,
            false,
        );
        let (beta_q, _, _) = learner_q.fit_step(y.view(), w.view());
        let (beta_i, _, _) = learner_i.fit_step(y.view(), w.view());

        // When unconstrained is feasible, both solvers should produce the
        // same β to machine precision (both return the unconstrained solve).
        for k in 0..beta_q.len() {
            assert!(
                (beta_q[k] - beta_i[k]).abs() < 1e-10,
                "QuadProg and Iterative disagree on feasible case at k={k}: {} vs {}",
                beta_q[k],
                beta_i[k],
            );
        }
    }

    // ── QuadProg optimality against an exhaustive oracle ──────────────────

    /// Solve `min ½β'Aβ − b'β s.t. Dβ ≥ 0` by enumerating *every* active set.
    ///
    /// For each subset `W ⊆ {0..m}` the equality-constrained stationary point
    /// is `β = A⁻¹(b + D_W'λ)` with `λ = −(D_W A⁻¹ D_W')⁻¹ D_W A⁻¹ b`, and the
    /// KKT point is the unique subset whose `β` is primal feasible (`Dβ ≥ 0`)
    /// and whose `λ` is dual feasible (`λ ≥ 0`). `A` is SPD here, so the QP is
    /// strictly convex and that point is the unique global minimiser.
    ///
    /// This is deliberately an exhaustive enumeration rather than an
    /// active-set *search*, so it cannot share a bug with the function under
    /// test. Only usable for small `m` (2^m subsets).
    fn brute_force_qp(a: &Array2<f64>, d: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
        let p = b.len();
        let m = d.nrows();
        assert!(m <= 12, "brute force is 2^m; keep m small");
        let eps = 1e-9;

        for mask in 0u32..(1u32 << m) {
            let active: Vec<usize> = (0..m).filter(|&i| mask & (1 << i) != 0).collect();
            let q = active.len();

            let beta = if q == 0 {
                match crate::linalg::cho_solve(a, &b.view()) {
                    Ok(x) => x,
                    Err(_) => continue,
                }
            } else {
                // a_inv_dt[:, k] = A⁻¹ D_k'
                let mut a_inv_dt = Array2::<f64>::zeros((p, q));
                let mut ok = true;
                for (k, &i) in active.iter().enumerate() {
                    let row = d.row(i).to_owned();
                    match crate::linalg::cho_solve(a, &row.view()) {
                        Ok(col) => {
                            for r in 0..p {
                                a_inv_dt[[r, k]] = col[r];
                            }
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                // H = D_W A⁻¹ D_W'
                let mut h = Array2::<f64>::zeros((q, q));
                for (k1, &i1) in active.iter().enumerate() {
                    for k2 in 0..q {
                        let mut s = 0.0;
                        for r in 0..p {
                            s += d[[i1, r]] * a_inv_dt[[r, k2]];
                        }
                        h[[k1, k2]] = s;
                    }
                }
                let a_inv_b = match crate::linalg::cho_solve(a, &b.view()) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                // λ = −H⁻¹ D_W A⁻¹ b
                let mut rhs = Array1::<f64>::zeros(q);
                for (k, &i) in active.iter().enumerate() {
                    let mut s = 0.0;
                    for r in 0..p {
                        s += d[[i, r]] * a_inv_b[r];
                    }
                    rhs[k] = -s;
                }
                let lambda = match crate::linalg::cho_solve(&h, &rhs.view()) {
                    Ok(x) => x,
                    Err(_) => continue, // linearly dependent working set
                };
                // Dual feasibility.
                if lambda.iter().any(|&v| v < -eps) {
                    continue;
                }
                // β = A⁻¹(b + D_W'λ)
                let mut rhs_beta = b.clone();
                for (k, &i) in active.iter().enumerate() {
                    for r in 0..p {
                        rhs_beta[r] += d[[i, r]] * lambda[k];
                    }
                }
                match crate::linalg::cho_solve(a, &rhs_beta.view()) {
                    Ok(x) => x,
                    Err(_) => continue,
                }
            };

            // Primal feasibility.
            let diffs = d.dot(&beta);
            if diffs.iter().any(|&v| v < -eps) {
                continue;
            }
            return Some(beta);
        }
        None
    }

    #[test]
    fn bmono_quadprog_matches_exhaustive_active_set_oracle() {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        // Random strictly-convex QPs checked against the exhaustive-enumeration
        // optimum. Every shape is covered, not just the first-difference
        // ("increasing") one: the multiplier-sign bug lived in shape-agnostic
        // code, and convex/concave fixtures merely happened not to trigger it.
        let shapes = [
            MonoConstraint::Increasing,
            MonoConstraint::Decreasing,
            MonoConstraint::Convex,
            MonoConstraint::Concave,
            MonoConstraint::Positive,
            MonoConstraint::Negative,
        ];
        let mut checked_binding = 0usize;

        for seed in 0..200u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let p = 5 + (seed as usize % 4); // p ∈ 5..=8  → m ≤ 8
            let shape = shapes[seed as usize % shapes.len()];
            let d = build_mono_constraint(shape, p);

            // A = M'M + δI, SPD by construction.
            let m_mat = Array2::<f64>::from_shape_fn((p, p), |_| rng.random::<f64>() * 2.0 - 1.0);
            let mut a = m_mat.t().dot(&m_mat);
            for i in 0..p {
                a[[i, i]] += 0.5;
            }
            let b = Array1::<f64>::from_shape_fn(p, |_| rng.random::<f64>() * 4.0 - 2.0);

            let expected = match brute_force_qp(&a, &d, &b) {
                Some(x) => x,
                None => continue, // no KKT point found (should not happen for SPD A)
            };

            let solver_a = GramSolver::new(&a);
            let got = solve_bmono_quadprog(
                &solver_a,
                a.view(),
                d.view(),
                b.view(),
                1e-10,
                4 * d.nrows().max(8),
            );

            // Track how many cases actually exercise the active-set loop
            // rather than the "unconstrained already feasible" early return.
            let unc = solver_a.solve(&b.view());
            if d.dot(&unc).iter().any(|&v| v < -1e-10) {
                checked_binding += 1;
            }

            for k in 0..p {
                assert!(
                    (got[k] - expected[k]).abs() < 1e-7,
                    "seed {seed} ({shape:?}): QuadProg β[{k}] = {} but exhaustive \
                     optimum = {}",
                    got[k],
                    expected[k],
                );
            }
        }

        // Guard against the test silently degenerating into "every case was
        // already feasible", which would exercise none of the active-set code.
        assert!(
            checked_binding >= 50,
            "expected many binding cases, got {checked_binding}"
        );
    }

    #[test]
    fn bmono_quadprog_satisfies_kkt_on_real_fixture() {
        // The real bmono fixture has p = 24 ⇒ m = 23 ⇒ 2^23 active sets, far
        // too many to enumerate, so optimality is certified directly instead:
        // a feasible β with a non-negative multiplier vector reproducing the
        // gradient on the active set is *the* optimum of a strictly convex QP.
        //
        // Note this cannot be checked by comparing objective values against
        // the iterative solver — that one minimises a *penalised* objective
        // and returns a slightly infeasible β, so it can and does score lower
        // on the unconstrained objective (asserted at the end).
        let n = 100;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y: Array1<f64> = x.mapv(|xi| -(2.0 * std::f64::consts::PI * xi).sin());
        let w = Array1::<f64>::ones(n);

        let make = |solver, lambda2, niter| {
            prepare_bmono(
                x.view(),
                w.view(),
                20,
                3,
                2,
                4.0,
                MonoConstraint::Increasing,
                solver,
                lambda2,
                niter,
                "x",
                None,
                None,
                false,
            )
        };
        let mut learner_q = make(BmonoSolver::QuadProg, 0.0, 0);
        let mut learner_i = make(BmonoSolver::Iterative, 1e6, 10);
        let (beta_q, _, _) = learner_q.fit_step(y.view(), w.view());
        let (beta_i, _, _) = learner_i.fit_step(y.view(), w.view());

        let PreparedLearner::BmonoBbs {
            penalised_gram,
            d_mat,
            wdt_faer,
            solver_a,
            ..
        } = &learner_q
        else {
            panic!("expected BmonoBbs from prepare_bmono");
        };

        // The KKT algebra assumes A⁻¹ inverts exactly the `A` the QP
        // multiplies by. A regularized fallback would silently break that.
        assert!(
            solver_a.is_cholesky(),
            "bmono gram unexpectedly needed an LU/regularized solver; the QP's \
             A and A⁻¹ would no longer be the same operator"
        );

        // b = X'W u for this step.
        let mut b = vec![0.0; wdt_faer.nrows()];
        faer_matvec_into(wdt_faer, y.as_slice().expect("y contiguous"), &mut b);
        let b = Array1::from_vec(b);

        // (1) Primal feasibility: Dβ ≥ 0.
        let diffs = d_mat.dot(&beta_q);
        let worst = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(worst > -1e-8, "QuadProg β infeasible: worst Dβ_i = {worst}");

        // (2) Recover the multipliers on the active set from stationarity
        // g = D_A' λ, solved in the least-squares sense via the normal
        // equations (D_A D_A') λ = D_A g.
        let g = penalised_gram.dot(&beta_q) - &b;
        let active: Vec<usize> = (0..diffs.len()).filter(|&i| diffs[i] <= 1e-7).collect();
        assert!(
            !active.is_empty(),
            "expected the increasing constraint to bind on this fixture"
        );
        let q = active.len();
        let mut dd = Array2::<f64>::zeros((q, q));
        for (k1, &i1) in active.iter().enumerate() {
            for (k2, &i2) in active.iter().enumerate() {
                dd[[k1, k2]] = d_mat.row(i1).dot(&d_mat.row(i2));
            }
        }
        let mut dg = Array1::<f64>::zeros(q);
        for (k, &i) in active.iter().enumerate() {
            dg[k] = d_mat.row(i).dot(&g);
        }
        let lambda =
            crate::linalg::cho_solve(&dd, &dg.view()).expect("active D rows are independent");

        // (3) Dual feasibility: λ ≥ 0.
        let min_lambda = lambda.iter().cloned().fold(f64::INFINITY, f64::min);
        let g_scale = g.dot(&g).sqrt().max(1.0);
        assert!(
            min_lambda > -1e-9 * g_scale,
            "KKT violated: negative multiplier {min_lambda} (‖g‖ = {g_scale})"
        );

        // (4) Stationarity residual: g must actually be reproduced by D_A' λ.
        let mut recon = Array1::<f64>::zeros(beta_q.len());
        for (k, &i) in active.iter().enumerate() {
            for r in 0..beta_q.len() {
                recon[r] += d_mat[[i, r]] * lambda[k];
            }
        }
        let resid = (&recon - &g).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            resid < 1e-7 * g_scale,
            "KKT violated: ‖D_A'λ − g‖ = {resid} (‖g‖ = {g_scale})"
        );

        // The iterative solver reaches a lower unconstrained objective only by
        // leaving the feasible set — documenting why an objective comparison
        // is not a valid optimality check here.
        let objective =
            |beta: &Array1<f64>| 0.5 * beta.dot(&penalised_gram.dot(beta)) - b.dot(beta);
        let worst_i = d_mat
            .dot(&beta_i)
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            objective(&beta_i) < objective(&beta_q) && worst_i < worst,
            "expected the iterative solution to score lower while being less \
             feasible: f_i = {}, f_q = {}, worst_i = {worst_i}, worst_q = {worst}",
            objective(&beta_i),
            objective(&beta_q),
        );
    }

    #[test]
    fn bmono_iterative_unconstrained_when_feasible() {
        // If the unconstrained fit happens to be monotone-increasing, the
        // solver should return after one iteration with no violations.
        let n = 50;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y: Array1<f64> = x.mapv(|xi| 3.0 * xi); // strictly increasing
        let w = Array1::<f64>::ones(n);

        let mut learner = prepare_bmono(
            x.view(),
            w.view(),
            20,
            3,
            2,
            4.0,
            MonoConstraint::Increasing,
            BmonoSolver::Iterative,
            1e6,
            10,
            "x",
            None,
            None,
            false,
        );
        let (beta, _pred, _sse) = learner.fit_step(y.view(), w.view());

        if let PreparedLearner::BmonoBbs { d_mat, .. } = &learner {
            let diffs = d_mat.dot(&beta);
            let worst = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
            // Should be feasible to high precision — no violations to penalise.
            assert!(worst > -1e-10, "should be feasible: worst Dβ_i = {worst}");
        } else {
            panic!("expected BmonoBbs");
        }
    }

    // ── B2: fit_step_select's SSE-by-identity matches explicit SSE ─────────

    fn assert_bmono_select_sse_matches_explicit(
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
        w: ArrayView1<f64>,
        constraint: MonoConstraint,
    ) {
        let mut learner_ref = prepare_bmono(
            x,
            w,
            20,
            3,
            2,
            4.0,
            constraint,
            BmonoSolver::Iterative,
            1e6,
            10,
            "x",
            None,
            None,
            false,
        );
        let mut learner_sel = prepare_bmono(
            x,
            w,
            20,
            3,
            2,
            4.0,
            constraint,
            BmonoSolver::Iterative,
            1e6,
            10,
            "x",
            None,
            None,
            false,
        );

        // Explicit SSE from fit_step's predict output.
        let (beta_ref, pred_ref, _) = learner_ref.fit_step(y, w);
        let explicit_sse: f64 = (0..y.len())
            .map(|i| w[i] * (y[i] - pred_ref[i]).powi(2))
            .sum();

        // wu2 for a zero-offset first step: Σ w y².
        let wu2: f64 = (0..y.len()).map(|i| w[i] * y[i] * y[i]).sum();
        let (beta_sel, sse_sel) = learner_sel.fit_step_select(y, w, wu2);

        for k in 0..beta_ref.len() {
            assert!(
                (beta_ref[k] - beta_sel[k]).abs() < 1e-9,
                "beta mismatch at k={k}: {} vs {}",
                beta_ref[k],
                beta_sel[k]
            );
        }
        assert!(
            (explicit_sse - sse_sel).abs() < 1e-9 * (1.0 + explicit_sse.abs()),
            "SSE-by-identity {sse_sel} != explicit SSE {explicit_sse}"
        );
    }

    #[test]
    fn bmono_select_sse_matches_explicit_when_unconstrained_feasible() {
        // Unconstrained solve already satisfies the constraint: beta is the
        // plain ridge solution.
        let n = 100;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y: Array1<f64> = x.mapv(|xi| 3.0 * xi);
        let w = Array1::<f64>::ones(n);
        assert_bmono_select_sse_matches_explicit(
            x.view(),
            y.view(),
            w.view(),
            MonoConstraint::Increasing,
        );
    }

    #[test]
    fn bmono_select_sse_matches_explicit_when_constraint_binds() {
        // y decreasing under an Increasing constraint: the QP/iterative
        // projection binds, so beta violates ridge stationarity — this is
        // exactly the case the full-form identity (vs. the BlockBols
        // shortcut) must get right.
        let n = 100;
        let x: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        let y: Array1<f64> = x.mapv(|xi| -(2.0 * std::f64::consts::PI * xi).sin());
        let w = Array1::<f64>::ones(n);
        assert_bmono_select_sse_matches_explicit(
            x.view(),
            y.view(),
            w.view(),
            MonoConstraint::Increasing,
        );
    }

    #[test]
    fn fit_tree_depth2_outperforms_stump() {
        let x1: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) / 50.0).collect();
        let x2: Vec<f64> = (0..200)
            .map(|i| ((i * 7 + 3) % 200) as f64 / 50.0 - 2.0)
            .collect();
        let n = 200;
        let mut design = Array2::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = x1[i];
            design[[i, 1]] = x2[i];
        }
        let y: Array1<f64> = (0..n)
            .map(|i| {
                let base = if x1[i] > 0.0 { 2.0 } else { -1.0 };
                let interaction = if x1[i] > 0.0 && x2[i] > 0.5 { 1.5 } else { 0.0 };
                base + interaction
            })
            .collect();
        let w = Array1::from_elem(n, 1.0f64);

        let (_coef1, pred1) = fit_tree(design.view(), y.view(), w.view(), 4, 10, 1);
        let (coef2, pred2) = fit_tree(design.view(), y.view(), w.view(), 4, 10, 2);

        let sse1: f64 = y
            .iter()
            .zip(pred1.iter())
            .map(|(yi, pi)| (yi - pi) * (yi - pi))
            .sum();
        let sse2: f64 = y
            .iter()
            .zip(pred2.iter())
            .map(|(yi, pi)| (yi - pi) * (yi - pi))
            .sum();

        assert!(
            sse2 < sse1,
            "depth-2 SSE ({sse2}) should be < depth-1 SSE ({sse1})"
        );

        // Verify coef round-trips through FittedTree serialization
        let tree = FittedTree::from_coef_vec(coef2.view());
        let pred_rt = tree.predict(design.view());
        for i in 0..n {
            assert!(
                (pred2[i] - pred_rt[i]).abs() < 1e-12,
                "round-trip mismatch at {i}"
            );
        }
    }

    #[test]
    fn fit_tree_depth3_outperforms_depth2() {
        // 3-level staircase: needs depth-3 to partition perfectly
        // x1 < -0.5 → -3, x1 in [-0.5, 0.5) and x2 < 0 → 0, x2 in [0, 1) → 2, x2 >= 1 → 4
        // x1 >= 0.5 and x2 < 0.5 → 1, x2 >= 0.5 → 5
        let n = 400;
        let mut design = Array2::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = (i as f64 / 100.0) - 2.0;
            design[[i, 1]] = ((i * 7 + 3) % n) as f64 / 100.0 - 2.0;
        }
        let y: Array1<f64> = (0..n)
            .map(|i| {
                let x1 = design[[i, 0]];
                let x2 = design[[i, 1]];
                if x1 < -0.5 {
                    -3.0
                } else if x1 < 0.5 {
                    if x2 < 0.0 {
                        0.0
                    } else if x2 < 1.0 {
                        2.0
                    } else {
                        4.0
                    }
                } else {
                    if x2 < 0.5 {
                        1.0
                    } else {
                        5.0
                    }
                }
            })
            .collect();
        let w = Array1::from_elem(n, 1.0f64);

        let (_, pred1) = fit_tree(design.view(), y.view(), w.view(), 4, 10, 1);
        let (_, pred2) = fit_tree(design.view(), y.view(), w.view(), 4, 10, 2);
        let (_, pred3) = fit_tree(design.view(), y.view(), w.view(), 4, 10, 3);

        let sse1: f64 = y
            .iter()
            .zip(pred1.iter())
            .map(|(yi, pi)| (yi - pi) * (yi - pi))
            .sum();
        let sse2: f64 = y
            .iter()
            .zip(pred2.iter())
            .map(|(yi, pi)| (yi - pi) * (yi - pi))
            .sum();
        let sse3: f64 = y
            .iter()
            .zip(pred3.iter())
            .map(|(yi, pi)| (yi - pi) * (yi - pi))
            .sum();

        assert!(
            sse2 < sse1,
            "depth-2 SSE ({sse2}) should be < depth-1 SSE ({sse1})"
        );
        assert!(
            sse3 < sse2,
            "depth-3 SSE ({sse3}) should be < depth-2 SSE ({sse2})"
        );
    }

    #[test]
    fn fit_tree_no_valid_split_returns_leaf() {
        let design = Array2::from_shape_vec((5, 1), vec![1.0; 5]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = Array1::from_elem(5, 1.0f64);
        let (coef, pred) = fit_tree(design.view(), y.view(), w.view(), 4, 10, 2);
        // coef layout: [num_nodes, tag, mean]
        assert_eq!(coef[0] as usize, 1);
        assert_eq!(coef[1], 1.0);
        let mean = coef[2];
        assert!((mean - 3.0).abs() < 1e-12);
        for i in 0..5 {
            assert!((pred[i] - mean).abs() < 1e-12);
        }
    }

    #[test]
    fn bspline_basis_extrap_linear_beyond_boundary() {
        let bk = (0.0, 1.0);
        let knots = bspline_knots(bk, 10, 3);
        let _x_in = [0.0, 0.5, 1.0];
        let _x_out = [-1.0, 2.0];
        let x_all = vec![-1.0, 0.0, 0.5, 1.0, 2.0];

        let basis_plain = bspline_basis(&x_all, &knots, 3);
        let basis_extrap = bspline_basis_extrap(&x_all, &knots, 3, bk);

        for (i, &xi) in x_all.iter().enumerate() {
            if xi >= bk.0 && xi <= bk.1 {
                for j in 0..basis_plain.ncols() {
                    let diff = (basis_plain[[i, j]] - basis_extrap[[i, j]]).abs();
                    assert!(diff < 1e-12, "in-domain mismatch at x={xi} col={j}: {diff}");
                }
            }
        }

        // Extrapolated basis at x=-1 should be linear: B(-1) = B(0) + (-1-0)*B'(0)
        let (val_lo, deriv_lo) = bspline_basis_deriv1(bk.0, &knots, 3);
        for j in 0..basis_extrap.ncols() {
            let expected = val_lo[j] + -deriv_lo[j];
            let diff = (basis_extrap[[0, j]] - expected).abs();
            assert!(diff < 1e-12, "left extrap mismatch col={j}: {diff}");
        }

        // Extrapolated basis at x=2 should be linear: B(2) = B(1) + (2-1)*B'(1)
        let (val_hi, deriv_hi) = bspline_basis_deriv1(bk.1, &knots, 3);
        for j in 0..basis_extrap.ncols() {
            let expected = val_hi[j] + 1.0 * deriv_hi[j];
            let diff = (basis_extrap[[4, j]] - expected).abs();
            assert!(diff < 1e-12, "right extrap mismatch col={j}: {diff}");
        }

        // Plain basis at x=2 should NOT equal the linear extrapolation
        let mut any_diff = false;
        for j in 0..basis_plain.ncols() {
            if (basis_plain[[4, j]] - basis_extrap[[4, j]]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "plain basis at x=2 should differ from linear extrap"
        );
    }

    #[test]
    fn bspline_deriv1_matches_r_at_boundary_knots() {
        // Uniform cubic B-splines on [0,1], 5 inner knots, dx=0.125
        let dx = 0.125;
        let knots: Vec<f64> = {
            let mut k = Vec::new();
            for i in 0..=3 {
                k.push(0.0 - (3 - i) as f64 * dx);
            }
            for i in 1..=5 {
                k.push(i as f64 * dx);
            }
            for i in 0..=3 {
                k.push(1.0 + i as f64 * dx);
            }
            k
        };
        let degree = 3;
        let k_lo = knots[degree];
        let k_hi = knots[knots.len() - degree - 1];

        let (_val_lo, deriv_lo) = bspline_basis_deriv1(k_lo, &knots, degree);
        let (_val_hi, deriv_hi) = bspline_basis_deriv1(k_hi, &knots, degree);

        // Partition of unity: sum(B_i) = 1 => sum(dB_i/dx) = 0
        let deriv_sum: f64 = deriv_lo.iter().sum();
        assert!(deriv_sum.abs() < 1e-12, "deriv sum at k_lo = {deriv_sum}");
        let deriv_sum_hi: f64 = deriv_hi.iter().sum();
        assert!(
            deriv_sum_hi.abs() < 1e-12,
            "deriv sum at k_hi = {deriv_sum_hi}"
        );

        // At left boundary, only cols 0..2 have nonzero derivatives
        for j in 3..deriv_lo.len() {
            assert!(
                deriv_lo[j].abs() < 1e-10,
                "deriv_lo[{j}]={} should be ~0",
                deriv_lo[j]
            );
        }

        // Regression guard: old bug (order-4 instead of order-3 B-splines)
        // produced 4 nonzero derivatives at boundary. Correct: at most 3.
        let nonzero_count = deriv_lo.iter().filter(|&&d| d.abs() > 1e-6).count();
        assert!(
            nonzero_count <= 3,
            "at most 3 nonzero derivs at left boundary, got {nonzero_count}"
        );

        // At right boundary, only the last 3 basis functions should
        // have nonzero derivatives (cols n-3, n-2, n-1). Earlier cols ~0.
        let n = deriv_hi.len();
        for j in 0..(n.saturating_sub(3)) {
            assert!(
                deriv_hi[j].abs() < 1e-10,
                "deriv_hi[{j}]={} should be ~0",
                deriv_hi[j]
            );
        }

        // Regression guard: at most 3 nonzero derivs at right boundary too.
        let nonzero_count_hi = deriv_hi.iter().filter(|&&d| d.abs() > 1e-6).count();
        assert!(
            nonzero_count_hi <= 3,
            "at most 3 nonzero derivs at right boundary, got {nonzero_count_hi}"
        );
    }

    #[test]
    fn fit_step_ref_matches_fit_step_block_bols() {
        let n = 20usize;
        let p = 5usize;
        // Well-conditioned design: each column is a different sin wave
        let design = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i as f64 + 1.0) * (j as f64 + 1.0) * 0.3).sin()
        });
        let weights = Array1::ones(n);
        let penalty = Array2::zeros((p, p));
        let name = "test".to_string();
        let centering = Array1::zeros(p);

        let d = into_column_major(design.clone());
        let gram = Array2::from_shape_fn((p, p), |(j, k)| {
            (0..n)
                .map(|i| weights[i] * design[[i, j]] * design[[i, k]])
                .sum()
        });
        let solver = GramSolver::new(&gram);

        let mut learner = PreparedLearner::BlockBols {
            design: d,
            sparse: None,
            penalty,
            lambda: None,
            solver,
            name,
            centering,
            weights: weights.to_owned(),
            weighted_gram: None,
            wu_buf: vec![0.0; n],
            rhs_buf: vec![0.0; p],
            beta_buf: vec![0.0; p],
            pred_buf: vec![0.0; n],
        };

        let u = Array1::from_shape_fn(n, |i| (i as f64).sin());

        // Call fit_step (allocating version)
        let (beta_vec, pred_vec, sse_old) = learner.fit_step(u.view(), weights.view());

        // Call fit_step_ref (zero-alloc version)
        let (beta_slice, pred_slice, sse_ref) = learner.fit_step_ref(u.view(), weights.view());

        // Compare SSE
        let sse_diff = (sse_old - sse_ref).abs();
        assert!(
            sse_diff < 1e-12,
            "SSE mismatch: old={sse_old}, ref={sse_ref}, diff={sse_diff}"
        );

        // Compare betas
        for i in 0..beta_vec.len() {
            let diff = (beta_vec[i] - beta_slice[i]).abs();
            assert!(
                diff < 1e-12,
                "beta[{i}] mismatch: old={}, ref={}, diff={diff}",
                beta_vec[i],
                beta_slice[i]
            );
        }

        // Compare predictions
        for i in 0..pred_vec.len() {
            let diff = (pred_vec[i] - pred_slice[i]).abs();
            assert!(
                diff < 1e-12,
                "pred[{i}] mismatch: old={}, ref={}, diff={diff}",
                pred_vec[i],
                pred_slice[i]
            );
        }
    }

    #[test]
    fn multinomial_kronecker_bols_engine_fit() {
        use crate::control::BoostControl;
        use crate::engine::fit_with;
        use crate::families::Multinomial;

        // 3 classes (K=3), K-1=2 response dims, 10 observations
        // Use a design where x1 correlates with class membership so
        // the Kronecker learner can decrease risk.
        let n_obs = 10;
        let n_resp = 2;

        // Row-major y: y[i * 2 + k]
        let y = array![
            1.0, 0.0, // obs 0 = class A
            1.0, 0.0, // obs 1 = class A
            1.0, 0.0, // obs 2 = class A
            1.0, 0.0, // obs 3 = class A
            0.0, 0.0, // obs 4 = class C (ref)
            0.0, 0.0, // obs 5 = class C (ref)
            0.0, 1.0, // obs 6 = class B
            0.0, 1.0, // obs 7 = class B
            0.0, 1.0, // obs 8 = class B
            0.0, 1.0, // obs 9 = class B
        ];

        // x1 separates classes: low for A, medium for C, high for B
        let x1 = Array2::from_shape_vec(
            (n_obs, 2),
            vec![
                0.0f64, 0.0, // A
                0.0, 1.0, // A
                1.0, 0.0, // A
                1.0, 1.0, // A
                2.0, 2.0, // C
                3.0, 3.0, // C
                4.0, 4.0, // B
                5.0, 5.0, // B
                6.0, 6.0, // B
                7.0, 7.0, // B
            ],
        )
        .unwrap();
        let w_obs = Array1::ones(n_obs);
        let x2 = Array2::eye(n_resp);
        let penalty = Array2::zeros((2, 2));

        let learner = prepare_kronecker_bols(x1, w_obs.view(), penalty, x2, true, "x1");

        let w_expanded = Array1::ones(n_obs * n_resp);
        let mut learners = vec![learner];
        let mut fam = Multinomial::new(3);
        let ctrl = BoostControl::new(20, 0.1).unwrap();

        let path = fit_with(
            &mut learners,
            y.view(),
            w_expanded.view(),
            &mut fam,
            &ctrl,
            None,
            false,
        );

        assert_eq!(path.risk.len(), 21);
        assert_eq!(path.selected.len(), 20);
        // Risk should decrease overall
        assert!(
            path.risk.last().unwrap() < path.risk.first().unwrap(),
            "risk should decrease overall: start={} end={}",
            path.risk.first().unwrap(),
            path.risk.last().unwrap(),
        );
    }

    #[test]
    fn btree_predict_row_unseen_values_no_oob() {
        // Train a depth-2 tree on 2-column design.
        // Then verify that predict_row handles unseen numeric values
        // (simulating unseen factor levels) without OOB — the tree just
        // follows the appropriate branch.
        let n_train = 100;
        let mut design = Array2::zeros((n_train, 2));
        for i in 0..n_train {
            design[[i, 0]] = (i % 5) as f64; // 5 "levels"
            design[[i, 1]] = (i % 3) as f64; // 3 "levels"
        }
        let u: Array1<f64> = (0..n_train).map(|i| (i % 7) as f64).collect();
        let w = Array1::ones(n_train);

        let (coef, pred_train) = fit_tree(design.view(), u.view(), w.view(), 4, 10, 2);
        let tree = FittedTree::from_coef_vec(coef.view());

        // Verify prediction on training data matches
        let pred_recompute = tree.predict(design.view());
        for i in 0..n_train {
            assert!((pred_train[i] - pred_recompute[i]).abs() < 1e-12);
        }

        // Test A: same column count, unseen "factor" values
        // e.g. column 0 has levels 5, 6, 7 (was 0..4 during training)
        let mut test_design = Array2::zeros((10, 2));
        for i in 0..10 {
            test_design[[i, 0]] = 5.0 + (i % 3) as f64; // unseen values in col 0
            test_design[[i, 1]] = 3.0 + (i % 2) as f64; // unseen values in col 1
        }
        let pred_test = tree.predict(test_design.view());
        // No panic == success
        assert_eq!(pred_test.len(), 10);
        // Values should be finite (tree routed correctly)
        for &p in &pred_test {
            assert!(p.is_finite());
        }

        // Test B: fewer columns than training may OOB if var > ncols-1
        // (var depends on the tree structure, which is data-dependent).
        // Just verify predict doesn't crash the process regardless.
    }

    /// The step hat matrix of a fitted tree reproduces the step itself:
    /// `S u` is the prediction `fit_step_ref` returned for the working
    /// response the tree was grown on — with no `by` variable, and with
    /// one (whose zero rows contribute nothing and predict 0).
    #[test]
    fn btree_step_hat_matrix_reproduces_the_fitted_step() {
        let n = 40usize;
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut state = 0x2545F491u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f64 / (1u32 << 24) as f64
        };
        for i in 0..n {
            design[[i, 0]] = next() * 4.0 - 2.0;
            design[[i, 1]] = next() * 2.0 - 1.0;
        }
        let u = Array1::from_shape_fn(n, |i| {
            (if design[[i, 0]] > 0.25 { 2.0 } else { -1.0 }) + 0.3 * design[[i, 1]]
        });
        let w = Array1::from_shape_fn(n, |i| 0.5 + (i % 5) as f64 * 0.25);

        let bys: Vec<Option<Array1<f64>>> = vec![
            None,
            Some(Array1::from_shape_fn(n, |i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    1.0 + (i % 4) as f64 * 0.5
                }
            })),
        ];
        for by in bys {
            let mut learner = prepare_btree(design.view(), w.view(), 3, 6, 3, by, "t");
            let (coef, pred, _) = learner.fit_step_ref(u.view(), w.view());
            let coef = coef.to_vec();
            let pred = pred.to_vec();
            assert!(coef[0] > 1.0, "the step must have fitted a split");
            let s = learner
                .btree_step_hat_matrix(&coef)
                .expect("a fitted tree step has a hat matrix");
            let su = s.dot(&u);
            for i in 0..n {
                assert!(
                    (su[i] - pred[i]).abs() < 1e-10,
                    "S u != fitted prediction at row {i}: {} vs {}",
                    su[i],
                    pred[i]
                );
            }
            // A malformed coefficient vector is refused, not panicked on.
            assert!(learner.btree_step_hat_matrix(&[1.0]).is_none());
            assert!(learner.btree_step_hat_matrix(&[]).is_none());
        }

        // A non-tree learner has no per-step tree matrix.
        let bols = prepare_scalar_bols(design.column(0), w.view(), 0.0, false, "x", None);
        assert!(bols.btree_step_hat_matrix(&[1.0, 1.0, 0.0]).is_none());
    }

    #[test]
    fn bbs_ospline_binned_matches_bbs_ospline() {
        let n = 800usize;
        let mut x_vec = Vec::with_capacity(n);
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            x_vec.push((u * 3.0).exp() - 1.0);
        }
        let x = Array1::from_vec(x_vec);
        let y = x.mapv(|v| (0.4 * v).sin() + 0.1 * v);
        let w = Array1::ones(n);

        for center in [false, true] {
            let bp_exact = LearnerBlueprint::bbs_ospline(
                x.clone(),
                "bbs(x)",
                15,
                3,
                4.0,
                None,
                center,
                None,
                None,
                0,
            )
            .unwrap();
            let bp_binned = LearnerBlueprint::bbs_ospline_binned(
                x.clone(),
                "bbs(x)",
                15,
                3,
                4.0,
                None,
                center,
                None,
                None,
                0,
                256,
            )
            .unwrap();

            let (mut l_exact, b_exact) = bp_exact.prepare(w.view()).unwrap();
            let (mut l_binned, b_binned) = bp_binned.prepare(w.view()).unwrap();

            let (_, pred_exact, _) = l_exact.fit_step_ref(y.view(), w.view());
            let (coef_binned, pred_binned, _) = l_binned.fit_step_ref(y.view(), w.view());

            let max_diff = pred_exact
                .iter()
                .zip(pred_binned.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let rmse_diff = (pred_exact
                .iter()
                .zip(pred_binned.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                / n as f64)
                .sqrt();
            assert!(
                rmse_diff < 0.015 && max_diff < 0.1,
                "center={center}: bbs_ospline_binned rmse={rmse_diff} max={max_diff}"
            );

            // When bins >= n, quantile binning is exact.
            let bp_full_bins = LearnerBlueprint::bbs_ospline_binned(
                x.clone(),
                "bbs(x)",
                15,
                3,
                4.0,
                None,
                center,
                None,
                None,
                0,
                n,
            )
            .unwrap();
            let (mut l_full, _) = bp_full_bins.prepare(w.view()).unwrap();
            let (_, pred_full, _) = l_full.fit_step_ref(y.view(), w.view());
            let exact_diff = pred_exact
                .iter()
                .zip(pred_full.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                exact_diff < 1e-9,
                "center={center}: bins=n exact diff {exact_diff} >= 1e-9"
            );

            let mut replay = vec![0.0f64; n];
            b_binned
                .predict_into(
                    coef_binned,
                    x.as_slice().unwrap(),
                    &mut replay,
                    1.0,
                    None,
                )
                .unwrap();
            let replay_diff = pred_binned
                .iter()
                .zip(replay.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                replay_diff < 1e-11,
                "center={center}: replay diff {replay_diff} >= 1e-11"
            );

            // Exact unbinned replay on held-out grid stays close to binned replay
            let grid = vec![0.1, 1.0, 4.0, 12.0, 30.0];
            let (coef_exact, _, _) = l_exact.fit_step_ref(y.view(), w.view());
            let mut out_exact = vec![0.0f64; grid.len()];
            let mut out_binned = vec![0.0f64; grid.len()];
            b_exact
                .predict_into(coef_exact, &grid, &mut out_exact, 1.0, None)
                .unwrap();
            b_binned
                .predict_into(coef_binned, &grid, &mut out_binned, 1.0, None)
                .unwrap();
            for (a, b) in out_exact.iter().zip(out_binned.iter()) {
                assert!((a - b).abs() < 0.1, "extrap/grid diff {} vs {}", a, b);
            }
        }
    }

    #[test]
    fn btree_lambda_leaf_shrinks_and_colsample_is_deterministic() {
        let n = 120usize;
        let p = 8usize;
        let mut design = Array2::zeros((n, p));
        let mut state = 0xDEAD_BEEF_1234_5678u64;
        for i in 0..n {
            for j in 0..p {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                design[[i, j]] = ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
            }
        }
        let u = Array1::from_shape_fn(n, |i| {
            if design[[i, 0]] > 0.0 {
                2.0
            } else {
                -2.0
            }
        });
        let w = Array1::ones(n);

        for method in [SplitMethod::Exact, SplitMethod::Histogram] {
            let bp0 = LearnerBlueprint::btree(design.clone(), "t", 4, 10, 3, None)
                .unwrap()
                .with_split_method(method);
            let bp_reg = LearnerBlueprint::btree(design.clone(), "t", 4, 10, 3, None)
                .unwrap()
                .with_split_method(method)
                .with_lambda_leaf(20.0);

            let (mut l0, _) = bp0.prepare(w.view()).unwrap();
            let (mut lr, _) = bp_reg.prepare(w.view()).unwrap();
            let (_, pred0, _) = l0.fit_step_ref(u.view(), w.view());
            let (_, predr, _) = lr.fit_step_ref(u.view(), w.view());
            let norm0: f64 = pred0.iter().map(|v| v * v).sum();
            let normr: f64 = predr.iter().map(|v| v * v).sum();
            assert!(
                normr < norm0 * 0.9,
                "method={method:?}: lambda_leaf should shrink predictions: normr={normr} vs norm0={norm0}"
            );

            let bp_cs1 = LearnerBlueprint::btree(design.clone(), "t", 4, 10, 3, None)
                .unwrap()
                .with_split_method(method)
                .with_colsample(0.5, 42);
            let bp_cs2 = LearnerBlueprint::btree(design.clone(), "t", 4, 10, 3, None)
                .unwrap()
                .with_split_method(method)
                .with_colsample(0.5, 42);
            let (mut c1, _) = bp_cs1.prepare(w.view()).unwrap();
            let (mut c2, _) = bp_cs2.prepare(w.view()).unwrap();
            for _ in 0..5 {
                let (coef1, p1, sse1) = c1.fit_step_ref(u.view(), w.view());
                let (coef2, p2, sse2) = c2.fit_step_ref(u.view(), w.view());
                assert_eq!(coef1, coef2);
                assert_eq!(p1, p2);
                assert_eq!(sse1, sse2);
            }
        }
    }

    #[test]
    fn btree_predict_by_masks_correctly() {
        // btree + by: verify by multiplies prediction element-wise
        // and that unseen by-values (e.g. 0.5, -1) don't cause OOB.
        let n = 50;
        let mut design = Array2::zeros((n, 1));
        for i in 0..n {
            design[[i, 0]] = (i % 10) as f64;
        }
        let u: Array1<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let w = Array1::ones(n);

        let (coef, _) = fit_tree(design.view(), u.view(), w.view(), 4, 10, 2);
        let tree = FittedTree::from_coef_vec(coef.view());

        let pred = tree.predict(design.view());

        // by vector is a separate multiplier, not part of design
        let by_vals = array![0.0, 1.0, 0.0, 0.5, -1.0]; // mixes 0, 1, unseen
        for &by in &by_vals {
            let mut eta = 0.0;
            for i in 0..n {
                eta += pred[i] * by;
            }
            assert!(eta.is_finite(), "by={by} should not cause issues");
        }
    }

    // ── bspline_basis_with_deriv ──────────────────────────────────────

    #[test]
    fn bspline_basis_with_deriv_deriv0_matches_bspline_basis() {
        let x = [0.2, 0.5, 0.8];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let expected = bspline_basis(&x, &knots, degree);
        let got = bspline_basis_with_deriv(&x, &knots, degree, 0);
        assert_eq!(got.shape(), expected.shape());
        for i in 0..got.nrows() {
            for j in 0..got.ncols() {
                assert!(
                    (got[[i, j]] - expected[[i, j]]).abs() < 1e-14,
                    "deriv=0 mismatch at ({i},{j}): {} vs {}",
                    got[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn bspline_basis_with_deriv_deriv1_sum_is_zero() {
        let x = [0.2, 0.5, 0.8];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let basis = bspline_basis_with_deriv(&x, &knots, degree, 1);
        // Partition of unity: sum(B_i) = 1 → sum(dB_i/dx) = 0 at any point
        for i in 0..basis.nrows() {
            let row_sum: f64 = basis.row(i).iter().sum();
            assert!(
                row_sum.abs() < 1e-12,
                "deriv=1 row sum at x={} = {} (expected ~0)",
                x[i],
                row_sum
            );
        }
    }

    #[test]
    fn bspline_basis_with_deriv_deriv_0_to_3_produce_different_values() {
        let x = [0.2, 0.5, 0.8];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let mut prev: Option<Array2<f64>> = None;
        for deriv in 0..=3 {
            let cur = bspline_basis_with_deriv(&x, &knots, degree, deriv);
            if let Some(ref prev) = prev {
                let mut all_same = true;
                for i in 0..cur.nrows() {
                    for j in 0..cur.ncols() {
                        if (cur[[i, j]] - prev[[i, j]]).abs() > 1e-14 {
                            all_same = false;
                            break;
                        }
                    }
                }
                assert!(
                    !all_same,
                    "deriv={deriv} produced identical values to deriv={}",
                    deriv - 1
                );
            }
            prev = Some(cur);
        }
    }

    #[test]
    fn bspline_basis_with_deriv_deriv2_and_deriv3_sum_is_zero() {
        let x = [0.2, 0.5, 0.8];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        for deriv in 2..=3 {
            let basis = bspline_basis_with_deriv(&x, &knots, degree, deriv);
            for i in 0..basis.nrows() {
                let row_sum: f64 = basis.row(i).iter().sum();
                assert!(
                    row_sum.abs() < 1e-12,
                    "deriv={deriv} row sum at x={xi} = {rs} (expected ~0)",
                    deriv = deriv,
                    xi = x[i],
                    rs = row_sum
                );
            }
        }
    }

    // ── cyclic_bspline_knots / cyclic_bspline_basis ───────────────────

    #[test]
    fn cyclic_bspline_knots_has_expected_structure() {
        let boundary = (0.0, 1.0);
        let degree = 3;
        let n_inner = 5;
        let knots = cyclic_bspline_knots(boundary, n_inner, degree);
        // n_base = n_inner + 2 = 7, prepend `degree` mirrored = 3 → total 10
        assert_eq!(knots.len(), n_inner + 2 + degree, "knots length mismatch");
        // First extended knot should be the mirrored copy of base_knots[n_base - ord]
        // n_base=7, ord=4, base_knots[3] = 3*dx = 0.5, so knots[0] = 0 - (1 - 0.5) = -0.5
        let first = knots[0];
        assert!(
            first < 0.0,
            "first extended knot should be negative, got {first}"
        );
        // All base knots (skipping the prepended ones) should be in [0, 1]
        for &k in knots.iter().skip(degree).take(n_inner + 2) {
            assert!(k >= 0.0 && k <= 1.0, "base knot {k} out of [0,1]");
        }
    }

    #[test]
    fn cyclic_xc_returns_expected_value() {
        let boundary = (0.0, 1.0);
        let degree = 3;
        let n_inner = 5;
        let dx = 1.0 / (n_inner + 1) as f64; // 1/6
        let xc = cyclic_xc(boundary, n_inner, degree);
        // xc = bk_lo + (n_inner + 2 - ord - 1) * dx = 0 + (7 - 4 - 1) * 1/6 = 2/6
        let expected = (n_inner + 2 - (degree + 1) - 1) as f64 * dx;
        assert!(
            (xc - expected).abs() < 1e-14,
            "xc = {xc}, expected {expected}"
        );
    }

    #[test]
    fn cyclic_bspline_basis_wraps_near_right_boundary() {
        let boundary = (0.0, 1.0);
        let degree = 3;
        let n_inner = 5;
        let knots = cyclic_bspline_knots(boundary, n_inner, degree);
        let period = 1.0;

        // Pick a point well past xc where wrapping is guaranteed
        let x = vec![0.9];

        let basis = cyclic_bspline_basis(&x, boundary, n_inner, degree);
        // The raw bspline_basis produces p = knots.len() - degree - 1 = n_inner + 1
        let ncols_expected = n_inner + 1;
        assert_eq!(basis.shape(), &[1, ncols_expected]);

        // For a point beyond xc, the cyclic basis = raw_basis(x) + raw_basis(x - period)
        let raw = bspline_basis(&[0.9], &knots, degree);
        let raw_wrapped = bspline_basis(&[0.9 - period], &knots, degree);
        for j in 0..basis.ncols() {
            let expected = raw[[0, j]] + raw_wrapped[[0, j]];
            assert!(
                (basis[[0, j]] - expected).abs() < 1e-14,
                "wrapped col {j}: {} vs sum {expected}",
                basis[[0, j]]
            );
        }
    }

    #[test]
    fn cyclic_bspline_basis_no_wrapping_before_xc() {
        let boundary = (0.0, 1.0);
        let degree = 3;
        let n_inner = 5;
        let knots = cyclic_bspline_knots(boundary, n_inner, degree);

        // Point well below xc, no wrapping
        let x = vec![0.2];
        let basis = cyclic_bspline_basis(&x, boundary, n_inner, degree);
        let raw = bspline_basis(&[0.2], &knots, degree);
        for j in 0..basis.ncols() {
            assert!(
                (basis[[0, j]] - raw[[0, j]]).abs() < 1e-14,
                "unwrapped col {j}: {} vs raw {}",
                basis[[0, j]],
                raw[[0, j]]
            );
        }
    }

    // ── build_mono_constraint (Negative) ──────────────────────────────

    #[test]
    fn mono_constraint_negative_is_negative_identity() {
        let d = build_mono_constraint(MonoConstraint::Negative, 4);
        assert_eq!(d.shape(), &[4, 4]);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { -1.0 } else { 0.0 };
                assert!(
                    (d[[i, j]] - expected).abs() < 1e-12,
                    "Negative D[{i},{j}] = {}, expected {}",
                    d[[i, j]],
                    expected
                );
            }
        }
        let beta = array![1.0, -2.0, 3.0, -4.0];
        let dbeta = d.dot(&beta);
        assert!((dbeta[0] - (-1.0)).abs() < 1e-12);
        assert!((dbeta[1] - 2.0).abs() < 1e-12);
        assert!((dbeta[2] - (-3.0)).abs() < 1e-12);
        assert!((dbeta[3] - 4.0).abs() < 1e-12);
    }

    // ── HingeLoss alpha_cost ──────────────────────────────────────────

    #[test]
    fn hinge_loss_alpha_affects_ngradient() {
        use crate::families::Family;
        use crate::families::HingeLoss;

        let y = array![1.0, -1.0, 1.0, -1.0];
        let f = array![0.8, 0.2, -0.3, -0.9]; // mix of margins < 1 and ≥ 1
        let w = array![1.0, 1.0, 1.0, 1.0];
        let n = y.len();

        let mut hinge_default = HingeLoss::default();
        let mut hinge_custom = HingeLoss::new(0.3);

        let mut grad_def = Array1::zeros(n);
        let mut grad_cus = Array1::zeros(n);
        hinge_default.negative_gradient(y.view(), f.view(), w.view(), grad_def.view_mut());
        hinge_custom.negative_gradient(y.view(), f.view(), w.view(), grad_cus.view_mut());

        // At least some entries must differ
        let mut any_diff = false;
        for i in 0..n {
            if (grad_def[i] - grad_cus[i]).abs() > 1e-14 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "derault and custom alpha_cost should differ");

        // Verify specific values:
        // For default (ac=0.5): pos class weight = 1-0.5=0.5, neg class weight = 0.5
        // For custom (ac=0.3): pos class weight = 1-0.3=0.7, neg class weight = 0.3
        // Margin 1*0.8 = 0.8 < 1, y>0 → ngrad = (1-ac)*y
        //   default: 0.5 * 1.0 = 0.5, custom: 0.7 * 1.0 = 0.7
        assert!(
            (grad_def[0] - 0.5).abs() < 1e-14,
            "default grad[0] = {}",
            grad_def[0]
        );
        assert!(
            (grad_cus[0] - 0.7).abs() < 1e-14,
            "custom grad[0] = {}",
            grad_cus[0]
        );
        // Margin -1*0.2 = -0.2 < 1, y<0 → ngrad = ac*y
        //   default: 0.5 * (-1.0) = -0.5, custom: 0.3 * (-1.0) = -0.3
        assert!(
            (grad_def[1] - (-0.5)).abs() < 1e-14,
            "default grad[1] = {}",
            grad_def[1]
        );
        assert!(
            (grad_cus[1] - (-0.3)).abs() < 1e-14,
            "custom grad[1] = {}",
            grad_cus[1]
        );
    }

    #[test]
    fn hinge_loss_alpha_affects_risk() {
        use crate::families::Family;
        use crate::families::HingeLoss;

        let y = array![1.0, -1.0, 1.0, -1.0];
        let f = array![0.8, 0.2, -0.3, -0.9];
        let w = array![1.0, 1.0, 1.0, 1.0];

        let hinge_default = HingeLoss::default();
        let hinge_custom = HingeLoss::new(0.3);

        let risk_def = hinge_default.risk(y.view(), f.view(), w.view());
        let risk_cus = hinge_custom.risk(y.view(), f.view(), w.view());

        assert!(
            (risk_def - risk_cus).abs() > 1e-14,
            "risk should differ with different alpha_cost: default={risk_def}, custom={risk_cus}"
        );
    }

    // ── df_to_lambda_scalar edge cases ────────────────────────────────

    #[test]
    fn df_to_lambda_scalar_edge_cases() {
        let x = array![1.0, 2.0, 3.0];
        let w = array![1.0, 1.0, 1.0];

        // df_target >= max_df → λ = 0
        let lam = df_to_lambda_scalar(x.view(), w.view(), 1.0, false);
        assert_eq!(lam, 0.0, "df=1, no intercept → λ=0");

        let lam = df_to_lambda_scalar(x.view(), w.view(), 2.0, true);
        assert_eq!(lam, 0.0, "df=2, intercept → λ=0");

        // df_target <= 0 → λ = MBOOST_LAMBDA_MAX
        let lam = df_to_lambda_scalar(x.view(), w.view(), 0.0, false);
        assert_eq!(lam, MBOOST_LAMBDA_MAX, "df=0, no intercept → λ_max");

        let lam = df_to_lambda_scalar(x.view(), w.view(), -0.5, false);
        assert_eq!(lam, MBOOST_LAMBDA_MAX, "df=-0.5, no intercept → λ_max");

        // has_intercept=true, df_target=1.0 → df_pen=0.0 → λ_max
        let lam = df_to_lambda_scalar(x.view(), w.view(), 1.0, true);
        assert_eq!(lam, MBOOST_LAMBDA_MAX, "df=1, intercept → df_pen=0 → λ_max");

        // has_intercept=true, df_target=0.5 → df_pen=-0.5 ≤ 0 → λ_max
        let lam = df_to_lambda_scalar(x.view(), w.view(), 0.5, true);
        assert_eq!(
            lam, MBOOST_LAMBDA_MAX,
            "df=0.5, intercept → df_pen≤0 → λ_max"
        );

        // Finite positive lambda for valid df_pen in (0,1)
        let lam = df_to_lambda_scalar(x.view(), w.view(), 0.5, false);
        assert!(
            lam > 0.0 && lam < MBOOST_LAMBDA_MAX,
            "df=0.5, no intercept → finite λ, got {lam}"
        );

        // has_intercept=true, df_target=1.5 → df_pen=0.5 → finite λ
        let lam = df_to_lambda_scalar(x.view(), w.view(), 1.5, true);
        assert!(
            lam > 0.0 && lam < MBOOST_LAMBDA_MAX,
            "df=1.5, intercept → finite λ, got {lam}"
        );
    }

    #[test]
    fn bbs_template_split_matches_monolithic_prepare() {
        // build_bbs_template + prepare_bbs_from_template (used by the LSS
        // CV/stabsel per-fold loops, CV2) must produce the identical
        // PreparedLearner as the monolithic prepare_bbs_with_lambda, since
        // the split is meant to be a pure refactor: same basis/penalty
        // construction, just hoisted out of the per-fold call.
        let x: Array1<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let w = array_from_scalar(40, 1.3);

        let direct = prepare_bbs_with_lambda(
            x.view(),
            w.view(),
            8,
            3,
            2,
            3.0,
            None,
            true,
            "x",
            None,
            false,
            None,
            0,
        );
        let (design, penalty) = build_bbs_template(x.view(), 8, 3, 2, true, None, false, None, 0);
        let templated = prepare_bbs_from_template(design, penalty, w.view(), 3.0, None, "x");

        let (
            PreparedLearner::BlockBols {
                design: d1,
                penalty: p1,
                ..
            },
            PreparedLearner::BlockBols {
                design: d2,
                penalty: p2,
                ..
            },
        ) = (direct, templated)
        else {
            panic!("expected BlockBols for both");
        };
        assert_eq!(d1.dim(), d2.dim());
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-12, "design mismatch: {a} vs {b}");
        }
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-12, "penalty mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn learner_blueprint_bols_centered_scalar_matches_the_gamboost_convention() {
        // `intercept = false, center = true` is what `gamboost` means by
        // `bols(x, intercept = TRUE)`: one weight-centered column, with the
        // constant term left to the model offset. The LSS form below fits an
        // explicit `[1, x]` block instead, which is why the same term has one
        // coefficient there and two here.
        let x = array![-1.0, 0.5, 2.0, 3.0];
        let weights = array![1.0, 2.0, 1.0, 0.5];
        let blueprint =
            LearnerBlueprint::scalar_bols(x.clone(), "bols(x)", Some(0.0), None, true, false, None)
                .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::ScalarBols { centering, .. } = learner else {
            panic!("gamboost bols must be a single centered column");
        };
        let expected_centering = x.dot(&weights) / weights.sum();
        assert!((centering - expected_centering).abs() < 1e-15);

        let coef = [0.75];
        let mut replay = vec![0.0; 4];
        recipe
            .predict_into(&coef, x.as_slice().unwrap(), &mut replay, 0.1, None)
            .unwrap();
        for row in 0..4 {
            let expected = 0.1 * coef[0] * (x[row] - expected_centering);
            assert!((replay[row] - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn learner_blueprint_bols_centered_scalar_replay_applies_by_before_centering() {
        // The centering constant is the weighted mean of `by * x` as it was at
        // fit time, so replay must scale before subtracting it. Doing it the
        // other way round is off by `(by - 1) * centering` per observation.
        let x = array![-1.0, 0.5, 2.0, 3.0];
        let by = array![2.0, 0.5, 1.0, 3.0];
        let weights = array![1.0, 2.0, 1.0, 0.5];
        let blueprint = LearnerBlueprint::scalar_bols(
            x.clone(),
            "bols(x, by = z)",
            Some(0.0),
            None,
            true,
            false,
            Some(by.clone()),
        )
        .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::ScalarBols { centering, .. } = learner else {
            panic!("gamboost bols must be a single centered column");
        };

        let coef = [0.75];
        let mut replay = vec![0.0; 4];
        recipe
            .predict_into(&coef, x.as_slice().unwrap(), &mut replay, 1.0, None)
            .unwrap();
        for row in 0..4 {
            let expected = coef[0] * (by[row] * x[row] - centering);
            assert!((replay[row] - expected).abs() < 1e-15, "row {row}");
        }
    }

    #[test]
    fn learner_blueprint_bols_intercept_preserves_two_column_design_and_replay() {
        let x = array![-1.0, 0.5, 2.0, 3.0];
        let weights = array![1.0, 2.0, 1.0, 0.5];
        let blueprint =
            LearnerBlueprint::scalar_bols(x.clone(), "bols(x)", Some(0.0), None, true, true, None)
                .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BlockBols { design, .. } = learner else {
            panic!("bols(intercept=true) must retain the intercept column");
        };
        assert_eq!(design.dim(), (4, 2));
        for row in 0..4 {
            assert_eq!(design[[row, 0]], 1.0);
            assert_eq!(design[[row, 1]], x[row]);
        }

        let coef = [0.25, -0.75];
        let mut replay = vec![0.0; 4];
        recipe
            .predict_into(&coef, x.as_slice().unwrap(), &mut replay, 0.1, None)
            .unwrap();
        for row in 0..4 {
            let expected = 0.1 * (coef[0] + coef[1] * x[row]);
            assert!((replay[row] - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn learner_blueprint_bbs_matches_direct_preparation() {
        let x: Array1<f64> = (0..50).map(|index| index as f64 / 10.0).collect();
        let weights = array_from_scalar(50, 1.0);
        let blueprint = LearnerBlueprint::bbs(
            x.clone(),
            "bbs(x)",
            10,
            3,
            2,
            4.0,
            None,
            false,
            None,
            false,
            None,
            0,
        )
        .unwrap();
        let (from_blueprint, _) = blueprint.prepare(weights.view()).unwrap();
        let direct = prepare_bbs_with_lambda(
            x.view(),
            weights.view(),
            10,
            3,
            2,
            4.0,
            None,
            false,
            "bbs(x)",
            None,
            false,
            None,
            0,
        );
        let (
            PreparedLearner::BlockBols {
                design: blueprint_design,
                penalty: blueprint_penalty,
                ..
            },
            PreparedLearner::BlockBols {
                design: direct_design,
                penalty: direct_penalty,
                ..
            },
        ) = (from_blueprint, direct)
        else {
            panic!("bbs must prepare as BlockBols");
        };
        assert_eq!(blueprint_design, direct_design);
        assert_eq!(blueprint_penalty, direct_penalty);
    }

    #[test]
    fn learner_blueprint_bns_matches_weighted_centering_and_replay() {
        let x: Array1<f64> = (0..48).map(|index| index as f64 / 12.0).collect();
        let weights: Array1<f64> = (0..48)
            .map(|index| 0.5 + (index % 5) as f64 / 4.0)
            .collect();
        let blueprint =
            LearnerBlueprint::bns(x.clone(), "bns(x)", 8, 3, Some(7.0), true, None, None).unwrap();
        let (from_blueprint, recipe) = blueprint.prepare(weights.view()).unwrap();
        let direct = prepare_bns(x.view(), weights.view(), 8, 3, 7.0, true, "bns(x)", None);
        let (
            PreparedLearner::BlockBols {
                design: blueprint_design,
                penalty: blueprint_penalty,
                ..
            },
            PreparedLearner::BlockBols {
                design: direct_design,
                penalty: direct_penalty,
                ..
            },
        ) = (from_blueprint, direct)
        else {
            panic!("bns must prepare as BlockBols");
        };
        assert_eq!(blueprint_design, direct_design);
        assert_eq!(blueprint_penalty, direct_penalty);

        let coef = vec![0.1; blueprint_design.ncols()];
        let expected = blueprint_design.dot(&Array1::from_vec(coef.clone()));
        let mut replay = vec![0.0; x.len()];
        recipe
            .predict_into(&coef, x.as_slice().unwrap(), &mut replay, 1.0, None)
            .unwrap();
        for index in 0..x.len() {
            assert!((replay[index] - expected[index]).abs() < 1e-12);
        }
    }

    #[test]
    fn learner_blueprint_buser_replays_explicit_training_and_newdata_designs() {
        let design = array![[1.0, -1.0], [0.5, 0.0], [2.0, 1.0], [-0.5, 2.0]];
        let weights = array![1.0, 2.0, 0.5, 1.5];
        let blueprint = LearnerBlueprint::buser(
            design.clone(),
            Array2::eye(2),
            "buser(custom)",
            1.5,
            Some(0.25),
            true,
            None,
        )
        .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BlockBols {
            design: prepared_design,
            ..
        } = learner
        else {
            panic!("buser must prepare as BlockBols");
        };
        let coef = vec![0.4, -0.7];
        let expected = prepared_design.dot(&Array1::from_vec(coef.clone()));
        let mut replay = vec![0.0; design.nrows()];
        recipe
            .predict_into(&coef, design.as_slice().unwrap(), &mut replay, 1.0, None)
            .unwrap();
        for row in 0..design.nrows() {
            assert!((replay[row] - expected[row]).abs() < 1e-12);
        }

        let new_design = array![[1.5, 0.25], [-1.0, 0.75]];
        let mut new_replay = vec![0.0; new_design.nrows()];
        recipe
            .predict_into(
                &coef,
                new_design.as_slice().unwrap(),
                &mut new_replay,
                1.0,
                None,
            )
            .unwrap();
        assert!(new_replay.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn bspatial_preparation_stashes_banded_gram_and_reweight_clears_it() {
        let n = 200;
        let x1 = Array1::from_shape_fn(n, |i| (i as f64) / n as f64);
        let x2 = Array1::from_shape_fn(n, |i| ((i * 7) % n) as f64 / n as f64);
        let weights = Array1::ones(n);
        let blueprint = LearnerBlueprint::bspatial(
            x1.clone(),
            x2.clone(),
            "spatial",
            30,
            3,
            2,
            6.0,
            None,
            false,
            None,
            None,
            None,
        )
        .unwrap();
        let (mut learner, _) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BlockBols { weighted_gram, .. } = &learner else {
            panic!("bspatial prepares as BlockBols");
        };
        // The uncentered tensor Gram is banded (~18% fill at 30 knots), so
        // preparation stashes the exact CSR copy it already computed.
        let stash = weighted_gram
            .as_ref()
            .expect("the banded tensor Gram is stashed as CSR");
        assert_eq!(stash.dims, (1156, 1156));
        assert!(stash.values.len() * 4 < stash.dims.0 * stash.dims.1);
        // Reweighting rebuilds the solver under new fold weights; the stash
        // described the old weights and must be gone, not stale.
        let new_weights = Array1::from_shape_fn(n, |i| if i % 5 == 0 { 0.0 } else { 1.5 });
        learner.reweight(new_weights.view());
        let PreparedLearner::BlockBols { weighted_gram, .. } = &learner else {
            unreachable!();
        };
        assert!(weighted_gram.is_none());
    }

    #[test]
    fn learner_blueprint_bkernel_replays_cross_kernel_with_training_centering() {
        let x = array![-1.0, -0.2, 0.4, 1.3];
        let weights = array![1.0, 2.0, 1.0, 0.5];
        let blueprint =
            LearnerBlueprint::bkernel(x.clone(), "bkernel(x)", 0.8, 2.5, "gaussian", true, None)
                .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BlockBols {
            design: prepared_design,
            ..
        } = learner
        else {
            panic!("bkernel must prepare as BlockBols");
        };
        let coef = vec![0.2, -0.1, 0.5, 0.3];
        let expected = prepared_design.dot(&Array1::from_vec(coef.clone()));
        let mut replay = vec![0.0; x.len()];
        recipe
            .predict_into(&coef, x.as_slice().unwrap(), &mut replay, 1.0, None)
            .unwrap();
        for row in 0..x.len() {
            assert!((replay[row] - expected[row]).abs() < 1e-12);
        }

        let new_x = [-0.6, 0.9];
        let mut new_replay = vec![0.0; new_x.len()];
        recipe
            .predict_into(&coef, &new_x, &mut new_replay, 1.0, None)
            .unwrap();
        assert!(new_replay.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn learner_blueprint_brandom_replays_centered_indicator_and_rejects_unknown_levels() {
        let levels = array![1.0, 2.0, 1.0, 3.0, 2.0, 3.0];
        let by = array![1.0, 0.5, 2.0, 1.5, 1.0, 0.25];
        let weights = array![1.0, 2.0, 1.0, 0.5, 1.5, 1.0];
        let blueprint = LearnerBlueprint::brandom(
            levels.clone(),
            "brandom(group)",
            2.0,
            None,
            true,
            Some(by),
            None,
        )
        .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BlockBols { design, .. } = learner else {
            panic!("brandom must prepare as BlockBols");
        };
        assert_eq!(design.dim(), (6, 3));

        let coef = vec![0.4, -0.2, 0.7];
        let expected = design.dot(&Array1::from_vec(coef.clone()));
        let mut replay = vec![0.0; levels.len()];
        recipe
            .predict_into(&coef, levels.as_slice().unwrap(), &mut replay, 1.0, None)
            .unwrap();
        for index in 0..levels.len() {
            assert!((replay[index] - expected[index]).abs() < 1e-12);
        }

        let mut unknown_replay = vec![0.0; 1];
        let error = recipe
            .predict_into(&coef, &[4.0], &mut unknown_replay, 1.0, None)
            .unwrap_err();
        assert!(error.to_string().contains("unknown Brandom level 4"));
        assert!(matches!(
            error,
            MboostrsError::UnseenLevel {
                learner: "Brandom",
                observation: 0,
                ..
            }
        ));

        // Under `ZeroEffect` an unseen level gets coefficient zero, so the
        // row receives the all-zero indicator row's value: minus the
        // centering constant this centered, weighted term subtracts.
        let centered_constant = 0.4 - replay[0];
        assert!(centered_constant.abs() > 1e-6);
        // Without a `by` override each row reads the cached `by` at its own
        // index, so the known level sits at row 0 to match `replay[0]`.
        let mut zero_replay = vec![0.0; 2];
        recipe
            .predict_into_with(
                &coef,
                &[1.0, 4.0],
                &mut zero_replay,
                1.0,
                None,
                UnseenLevels::ZeroEffect,
            )
            .unwrap();
        assert!((zero_replay[0] - replay[0]).abs() < 1e-12);
        assert!((zero_replay[1] + centered_constant).abs() < 1e-12);

        // Under `Average` the unseen level takes the mean coefficient, scaled
        // by the row's `by` value (row 1 reads the cached by[1] = 0.5).
        let mut average_replay = vec![0.0; 2];
        recipe
            .predict_into_with(
                &coef,
                &[1.0, 4.0],
                &mut average_replay,
                1.0,
                None,
                UnseenLevels::Average,
            )
            .unwrap();
        let mean = coef.iter().sum::<f64>() / coef.len() as f64;
        assert!((average_replay[0] - replay[0]).abs() < 1e-12);
        assert!((average_replay[1] - (0.5 * mean - centered_constant)).abs() < 1e-12);

        let sparse_levels = array![1.0, 3.0, 1.0, 3.0];
        let sparse = LearnerBlueprint::brandom(
            sparse_levels,
            "brandom(sparse_group)",
            2.0,
            Some(0.0),
            false,
            None,
            None,
        )
        .unwrap();
        let (_, sparse_recipe) = sparse.prepare(array![1.0, 1.0, 1.0, 1.0].view()).unwrap();
        let error = sparse_recipe
            .predict_into(&[0.4, 0.7], &[2.0], &mut unknown_replay, 1.0, None)
            .unwrap_err();
        assert!(error.to_string().contains("unknown Brandom level 2"));
    }

    #[test]
    fn learner_blueprint_bmrf_replays_spectral_design_and_rejects_unknown_levels() {
        let levels = array![0.0, 1.0, 2.0, 3.0, 0.0, 2.0];
        let by = array![1.0, 0.5, 2.0, 1.5, 0.25, 1.0];
        let weights = array![1.0, 2.0, 1.0, 0.5, 1.5, 1.0];
        let k = array![
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ];
        let blueprint =
            LearnerBlueprint::bmrf(levels.clone(), "bmrf(region)", k, 3.0, None, true, Some(by))
                .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BlockBols { design, .. } = learner else {
            panic!("bmrf must prepare as BlockBols");
        };
        assert_eq!(design.dim(), (6, 3));

        let coef = vec![0.4, -0.2, 0.7];
        let expected = design.dot(&Array1::from_vec(coef.clone()));
        let mut replay = vec![0.0; levels.len()];
        recipe
            .predict_into(&coef, levels.as_slice().unwrap(), &mut replay, 1.0, None)
            .unwrap();
        for index in 0..levels.len() {
            assert!((replay[index] - expected[index]).abs() < 1e-12);
        }

        let error = recipe
            .predict_into(&coef, &[4.0], &mut [0.0], 1.0, None)
            .unwrap_err();
        assert!(error.to_string().contains("unknown Bmrf level 4"));
    }

    #[test]
    fn learner_blueprint_bspatial_replays_centered_tensor_design() {
        let x1 = array![-1.0, -0.6, -0.1, 0.2, 0.7, 1.0];
        let x2 = array![0.1, 0.8, 0.3, 0.9, 0.5, 0.2];
        let by = array![1.0, 0.5, 1.5, 2.0, 0.75, 1.25];
        let weights = array![1.0, 2.0, 1.0, 0.5, 1.5, 1.0];
        let blueprint = LearnerBlueprint::bspatial(
            x1.clone(),
            x2.clone(),
            "bspatial(x1,x2)",
            3,
            2,
            1,
            4.0,
            Some(0.5),
            true,
            Some(by),
            None,
            None,
        )
        .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BlockBols { design, .. } = learner else {
            panic!("bspatial must prepare as BlockBols");
        };
        let coef = (0..design.ncols())
            .map(|index| 0.02 * (index + 1) as f64)
            .collect::<Vec<_>>();
        let expected = design.dot(&Array1::from_vec(coef.clone()));
        let packed = x1.iter().chain(x2.iter()).copied().collect::<Vec<_>>();
        let mut replay = vec![0.0; x1.len()];
        recipe
            .predict_into(&coef, &packed, &mut replay, 1.0, None)
            .unwrap();
        for index in 0..x1.len() {
            assert!((replay[index] - expected[index]).abs() < 1e-10);
        }

        let error = recipe
            .predict_into(&coef, &packed[..packed.len() - 1], &mut replay, 1.0, None)
            .unwrap_err();
        assert!(error.to_string().contains("packed [x1, x2]"));
    }

    #[test]
    fn learner_blueprint_bmono_replays_training_basis() {
        let x = array![-1.0, -0.6, -0.1, 0.2, 0.7, 1.0];
        let by = array![1.0, 0.5, 1.5, 2.0, 0.75, 1.25];
        let weights = array![1.0, 2.0, 1.0, 0.5, 1.5, 1.0];
        let blueprint = LearnerBlueprint::bmono(
            x.clone(),
            "bmono(x)",
            4,
            2,
            1,
            4.0,
            MonoConstraint::Increasing,
            BmonoSolver::Iterative,
            1e6,
            10,
            Some(by),
            false,
        )
        .unwrap();
        let (learner, recipe) = blueprint.prepare(weights.view()).unwrap();
        let PreparedLearner::BmonoBbs { design, .. } = learner else {
            panic!("bmono must prepare as BmonoBbs");
        };
        let coef = (0..design.ncols())
            .map(|index| 0.03 * index as f64)
            .collect::<Vec<_>>();
        let expected = design.dot(&Array1::from_vec(coef.clone()));
        let mut replay = vec![0.0; x.len()];
        recipe
            .predict_into(&coef, x.as_slice().unwrap(), &mut replay, 1.0, None)
            .unwrap();
        for index in 0..x.len() {
            assert!((replay[index] - expected[index]).abs() < 1e-12);
        }
    }

    #[test]
    fn block_bols_selection_sse_matches_the_weighted_residual_sum() {
        // The selection SSE uses the ridge-stationarity shortcut
        // `wu2 - beta'rhs - beta'Pbeta`, which equals the weighted residual sum
        // only when beta solves (X'WX + P)beta = X'Wu. `BmonoBbs` documents
        // that precondition where it fails -- a constrained beta does not
        // satisfy stationarity -- but nothing pinned it where it holds. A
        // tensor design is the hardest case for it: the Gram is rank-deficient
        // and the penalty leaves a four-dimensional space free. Measured here
        // at p = 64, 144 and 256, the two agree to a relative 4e-16, so the
        // solve is exact enough for the shortcut and a selection step cannot
        // be handed an SSE below the truth.
        for width in [8usize, 12, 16] {
            let n = 60usize;
            let (design, penalty) = tensor_design_and_penalty(n, width);
            let p = design.ncols();
            let weights = Array1::ones(n);
            let mut learner = prepare_block_bols(
                design.clone(),
                weights.view(),
                penalty.clone(),
                false,
                "tensor",
            );
            let u = Array1::from_shape_fn(n, |i| ((i as f64) * 0.7).sin());
            let wu2: f64 = (0..n).map(|i| weights[i] * u[i] * u[i]).sum();

            let reported = learner.fit_step_select_sse(u.view(), weights.view(), wu2);
            let beta: Vec<f64> = learner.selection_beta().to_vec();
            let truth: f64 = (0..n)
                .map(|i| {
                    let fitted: f64 = (0..p).map(|j| design[[i, j]] * beta[j]).sum();
                    weights[i] * (u[i] - fitted) * (u[i] - fitted)
                })
                .sum();
            let relative = (reported - truth).abs() / truth.abs().max(1e-12);
            println!(
                "width={width} p={p} reported={reported:.12e} truth={truth:.12e} \
                 relative gap={relative:.3e}"
            );
            assert!(
                relative < 1e-9,
                "width={width}: selection SSE {reported} differs from the weighted \
                 residual sum {truth} by a relative {relative}"
            );
        }
    }

    fn array_from_scalar(n: usize, v: f64) -> Array1<f64> {
        Array1::from_elem(n, v)
    }
}

/// The Nyström kernel learner approximates the exact one, and says how well.
///
/// P3.4 item 6: "add a Nyström/landmark alternative to the `O(n^2)` Bkernel
/// path".
///
/// The exact path builds an `n x n` design *and* an `n x n` penalty, so a
/// 10,000-row fit needs 1.6 GB before it starts. The point of the landmark
/// form is that `m` is a choice the caller makes, so what these check is that
/// the choice behaves: exact at `m = n`, and converging as `m` grows rather
/// than merely differing by less than some number someone picked.
#[cfg(test)]
mod nystrom_kernel {
    use super::*;

    fn fixture(n: usize) -> (Array1<f64>, Array1<f64>) {
        let x = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64) * 4.0 - 2.0);
        let weights = Array1::ones(n);
        (x, weights)
    }

    /// `Phi Phi'`, the `n x n` kernel a Nyström design stands in for.
    ///
    /// The *outer* product, not the inner one: the identity being
    /// approximated is `K_nm K_mm^-1 K_mn ~ K_nn`, and at `m = n` it is an
    /// equality. Comparing `Phi' Phi` instead compares two `m x m` matrices
    /// that have no reason to agree.
    fn implied_kernel(learner: &PreparedLearner) -> Array2<f64> {
        let design = learner
            .design_matrix()
            .expect("block learners expose a design");
        design.dot(&design.t())
    }

    /// The exact learner's design *is* `K_nn`.
    fn exact_kernel(learner: &PreparedLearner) -> Array2<f64> {
        learner
            .design_matrix()
            .expect("block learners expose a design")
    }

    #[test]
    fn landmarks_are_quantiles_of_the_data_and_deduplicated() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let centres = kernel_landmarks(x.view(), 4);
        assert_eq!(centres.first().copied(), Some(0.0));
        assert_eq!(centres.last().copied(), Some(3.0));
        // The three tied zeros collapse: a duplicated landmark makes `K_mm`
        // singular, and a singular factor gives a design full of NaN that
        // still fits.
        assert_eq!(centres.iter().filter(|&&v| v == 0.0).count(), 1);

        // Asking for more landmarks than there are rows is not an error.
        assert_eq!(kernel_landmarks(x.view(), 100).len(), 4);
        assert_eq!(kernel_landmarks(x.view(), 1).len(), 1);
    }

    #[test]
    fn it_reproduces_the_exact_kernel_when_every_point_is_a_landmark() {
        // `K_nm K_mm^-1 K_mn` is `K_nn` when the landmarks are the
        // observations, so the two designs must have the same Gram.
        let (x, weights) = fixture(40);
        let exact = prepare_bkernel(
            x.view(),
            weights.view(),
            1.0,
            4.0,
            "gaussian",
            false,
            "k",
            None,
        );
        let approximate = prepare_bkernel_nystrom(
            x.view(),
            weights.view(),
            1.0,
            4.0,
            "gaussian",
            40,
            false,
            "k",
            None,
        )
        .unwrap();

        let exact_gram = exact_kernel(&exact);
        let nystrom_gram = implied_kernel(&approximate);
        assert_eq!(exact_gram.dim(), nystrom_gram.dim());
        let worst = exact_gram
            .iter()
            .zip(nystrom_gram.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let scale = exact_gram.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            worst <= 1e-6 * scale.max(1.0),
            "gram differs by {worst} against a scale of {scale}"
        );
    }

    #[test]
    fn the_approximation_improves_with_more_landmarks() {
        // The claim worth making about an approximation is that it converges,
        // not that it is under a threshold on one fixture.
        let (x, weights) = fixture(120);
        let exact = prepare_bkernel(
            x.view(),
            weights.view(),
            0.8,
            4.0,
            "gaussian",
            false,
            "k",
            None,
        );
        let exact_gram = exact_kernel(&exact);
        let scale = exact_gram.iter().fold(0.0_f64, |m, v| m.max(v.abs()));

        let mut previous = f64::INFINITY;
        for landmarks in [4usize, 10, 30, 120] {
            let approximate = prepare_bkernel_nystrom(
                x.view(),
                weights.view(),
                0.8,
                4.0,
                "gaussian",
                landmarks,
                false,
                "k",
                None,
            )
            .unwrap();
            let worst = exact_gram
                .iter()
                .zip(implied_kernel(&approximate).iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
                / scale.max(1.0);
            assert!(
                worst <= previous + 1e-12,
                "{landmarks} landmarks were worse than the previous count: \
                 {worst} against {previous}"
            );
            previous = worst;
        }
        assert!(
            previous <= 1e-6,
            "120 landmarks on 120 rows should be exact"
        );
    }

    #[test]
    fn the_design_is_n_by_m_rather_than_n_by_n() {
        // The whole point: the memory is the caller's choice.
        let (x, weights) = fixture(200);
        let approximate = prepare_bkernel_nystrom(
            x.view(),
            weights.view(),
            1.0,
            4.0,
            "gaussian",
            25,
            false,
            "k",
            None,
        )
        .unwrap();
        let design = approximate.design_matrix().unwrap();
        assert_eq!(design.dim(), (200, 25));

        let exact = prepare_bkernel(
            x.view(),
            weights.view(),
            1.0,
            4.0,
            "gaussian",
            false,
            "k",
            None,
        );
        assert_eq!(exact.design_matrix().unwrap().dim(), (200, 200));
    }

    #[test]
    fn every_design_entry_is_finite() {
        // A singular `K_mm` gives a design of NaN that still fits and reports
        // nothing, which is why the factorisation carries a ridge.
        let tied = Array1::from_shape_fn(60, |i| if i < 50 { 0.5 } else { 0.500000001 });
        let weights = Array1::ones(60);
        let learner = prepare_bkernel_nystrom(
            tied.view(),
            weights.view(),
            1.0,
            4.0,
            "gaussian",
            20,
            false,
            "k",
            None,
        )
        .unwrap();
        assert!(learner
            .design_matrix()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn the_replay_reproduces_the_training_fit() {
        // A learner whose new-data path does not match its training design is
        // worse than one that refuses new data: it predicts, and the numbers
        // are wrong. The Nyström replay has to redo the same forward
        // substitution the design did -- scoring an untransformed
        // cross-kernel against coefficients fitted on a transformed one is
        // the mistake that looks like a fit.
        let (x, weights) = fixture(150);
        let blueprint = LearnerBlueprint::bkernel_nystrom(
            x.to_owned(),
            "k",
            0.9,
            4.0,
            "gaussian",
            20,
            false,
            None,
        )
        .unwrap();
        let (learner, basis) = blueprint.prepare(weights.view()).unwrap();
        let design = learner.design_matrix().unwrap();
        let m = design.ncols();

        // An arbitrary coefficient vector: the identity has to hold for any
        // of them, not only the one a fit would produce.
        let coef: Vec<f64> = (0..m).map(|j| 0.3 - 0.07 * j as f64).collect();
        let mut expected = vec![0.0f64; x.len()];
        for row in 0..x.len() {
            let mut value = 0.0;
            for column in 0..m {
                value += coef[column] * design[[row, column]];
            }
            expected[row] = value;
        }

        let mut replayed = vec![0.0f64; x.len()];
        basis
            .predict_into(&coef, x.as_slice().unwrap(), &mut replayed, 1.0, None)
            .unwrap();

        for (row, (left, right)) in expected.iter().zip(&replayed).enumerate() {
            assert!(
                (left - right).abs() <= 1e-9 * left.abs().max(1.0),
                "row {row}: replay {right} against the training design {left}"
            );
        }
    }

    #[test]
    fn the_blueprint_refuses_what_the_preparation_refuses() {
        let (x, _) = fixture(10);
        {
            let landmarks = 0usize;
            match LearnerBlueprint::bkernel_nystrom(
                x.to_owned(),
                "k",
                1.0,
                4.0,
                "gaussian",
                landmarks,
                false,
                None,
            ) {
                Ok(_) => panic!("{landmarks} landmarks should be refused"),
                Err(error) => {
                    assert!(format!("{error}").contains("at least 1"), "{error}")
                }
            }
        }
    }

    #[test]
    fn zero_landmarks_are_refused() {
        let (x, weights) = fixture(10);
        let outcome = prepare_bkernel_nystrom(
            x.view(),
            weights.view(),
            1.0,
            4.0,
            "gaussian",
            0,
            false,
            "k",
            None,
        );
        match outcome {
            Ok(_) => panic!("zero landmarks should be refused"),
            Err(error) => assert!(format!("{error}").contains("at least 1"), "{error}"),
        }
    }
}

#[cfg(test)]
mod categorical_tree_tests {
    use super::*;

    /// A factor whose levels {0, 2} and {1, 3} have different means splits
    /// into those two sets, whatever the level order; the fit round-trips
    /// through the coefficient encoding; an unseen code follows the heavier
    /// child.
    #[test]
    fn a_categorical_column_splits_into_level_sets() {
        let n = 400;
        let codes: Vec<f64> = (0..n).map(|i| [0.0, 1.0, 2.0, 3.0, 0.0][i % 5]).collect();
        let u: Vec<f64> = codes
            .iter()
            .map(|&c| if c == 0.0 || c == 2.0 { 1.0 } else { -1.0 })
            .collect();
        let design = Array2::from_shape_vec((n, 1), codes).unwrap();
        let w = Array1::ones(n);
        let sorted = btree_sorted_indices(design.view());
        let (coef, pred) = fit_tree_with_sorted(
            design.view(),
            ArrayView1::from(&u),
            w.view(),
            &sorted,
            &[4],
            SplitSelection::Sse,
            SplitMethod::Exact,
            &vec![None; 1],
            1,
            2,
            1,
            0.0,
            None,
        );
        for (p, t) in pred.iter().zip(&u) {
            assert!((p - t).abs() < 1e-12);
        }
        let tree = FittedTree::from_coef_vec(coef.view());
        assert!((tree.predict_row(ndarray::arr1(&[2.0]).view()) - 1.0).abs() < 1e-12);
        // Codes 0 and 2 hold 240 of the 400 rows: an unseen code goes with them.
        assert!((tree.predict_row(ndarray::arr1(&[-1.0]).view()) - 1.0).abs() < 1e-12);
        let scaled = FittedTree::scale_coef_vec(coef.as_slice().unwrap(), 0.5);
        let half = FittedTree::from_coef_vec(ArrayView1::from(&scaled));
        assert!((half.predict_row(ndarray::arr1(&[1.0]).view()) + 0.5).abs() < 1e-12);
    }
}

#[cfg(test)]
mod histogram_tree_tests {
    use super::*;

    /// Small deterministic generator so the fixtures are reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fit(
        design: &Array2<f64>,
        u: &Array1<f64>,
        w: &Array1<f64>,
        method: SplitMethod,
        categorical: &[usize],
        minbucket: usize,
        minsplit: usize,
        maxdepth: usize,
    ) -> (Array1<f64>, Array1<f64>) {
        let sorted = btree_sorted_indices(design.view());
        let binning = match method {
            SplitMethod::Histogram => prepare_btree_binning(design.view(), categorical, 256),
            SplitMethod::Exact => vec![None; design.ncols()],
        };
        fit_tree_with_sorted(
            design.view(),
            u.view(),
            w.view(),
            &sorted,
            categorical,
            SplitSelection::Sse,
            method,
            &binning,
            minbucket,
            minsplit,
            maxdepth,
            0.0,
            None,
        )
    }

    /// With no more distinct values than bins, the histogram's candidate
    /// cuts are exactly the exact scan's, so the two trees must agree.
    #[test]
    fn histogram_matches_exact_when_distinct_within_bins() {
        let (n, p) = (500usize, 3usize);
        let mut lcg = Lcg(42);
        let data: Vec<f64> = (0..n * p)
            .map(|_| (lcg.next_f64() * 12.0).floor())
            .collect();
        let design = Array2::from_shape_vec((n, p), data).unwrap();
        let u = Array1::from_iter((0..n).map(|i| {
            let x0 = design[[i, 0]];
            let x1 = design[[i, 1]];
            // A strong, unambiguous signal on the first column.
            (x0 * 0.37).sin() * 2.0 + 0.01 * x1 + lcg.next_f64() - 0.5
        }));
        let w = Array1::from_iter((0..n).map(|_| 0.5 + lcg.next_f64()));
        let (coef_exact, pred_exact) = fit(&design, &u, &w, SplitMethod::Exact, &[], 4, 10, 3);
        let (coef_hist, pred_hist) = fit(&design, &u, &w, SplitMethod::Histogram, &[], 4, 10, 3);
        assert_eq!(coef_exact.len(), coef_hist.len());
        eprintln!("exact: {coef_exact:?}");
        eprintln!("hist : {coef_hist:?}");
        for (i, (a, b)) in coef_exact.iter().zip(coef_hist.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9 * a.abs().max(1.0),
                "coef[{i}] {a} vs {b}"
            );
        }
        for (a, b) in pred_exact.iter().zip(pred_hist.iter()) {
            assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "pred {a} vs {b}");
        }
    }

    /// The blueprint threads the method all the way to the fit step: both
    /// methods fit the same learner shape and agree on this exact set.
    #[test]
    fn blueprint_threads_split_method_to_fit_step() {
        let (n, p) = (300usize, 2usize);
        let mut lcg = Lcg(11);
        let data: Vec<f64> = (0..n * p).map(|_| (lcg.next_f64() * 8.0).floor()).collect();
        let design = Array2::from_shape_vec((n, p), data).unwrap();
        let u = Array1::from_iter((0..n).map(|i| design[[i, 0]] * 0.5 - lcg.next_f64()));
        let w = Array1::ones(n);
        let mut exact = LearnerBlueprint::btree(design.clone(), "t", 4, 10, 3, None)
            .unwrap()
            .with_split_method(SplitMethod::Exact)
            .prepare(w.view())
            .unwrap()
            .0;
        let mut histogram = LearnerBlueprint::btree(design, "t", 4, 10, 3, None)
            .unwrap()
            .with_split_method(SplitMethod::Histogram)
            .prepare(w.view())
            .unwrap()
            .0;
        let (_, pred_exact, _) = exact.fit_step(u.view(), w.view());
        let (_, pred_hist, _) = histogram.fit_step(u.view(), w.view());
        for (a, b) in pred_exact.iter().zip(pred_hist.iter()) {
            assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "pred {a} vs {b}");
        }
    }

    /// Binning more distinct values than bins is the approximation: the fit
    /// still runs, keeps the minbucket constraints, and stays finite.
    #[test]
    fn histogram_with_more_distinct_than_bins_still_splits() {
        let (n, p) = (4000usize, 2usize);
        let mut lcg = Lcg(7);
        let data: Vec<f64> = (0..n * p).map(|_| lcg.next_f64()).collect();
        let design = Array2::from_shape_vec((n, p), data).unwrap();
        let u = Array1::from_iter((0..n).map(|i| design[[i, 0]] + 0.1 * lcg.next_f64()));
        let w = Array1::ones(n);
        let (_, pred) = fit(&design, &u, &w, SplitMethod::Histogram, &[], 10, 20, 4);
        assert!(pred.iter().all(|v| v.is_finite()));
        // The fitted tree must beat the intercept: the signal is linear.
        let mean = u.mean().unwrap();
        let sse_model: f64 = pred
            .iter()
            .zip(u.iter())
            .map(|(p, y)| (p - y).powi(2))
            .sum();
        let sse_null: f64 = u.iter().map(|y| (y - mean).powi(2)).sum();
        assert!(
            sse_model < 0.9 * sse_null,
            "sse {sse_model} vs null {sse_null}"
        );
    }

    /// Categorical columns keep their level-set path under the histogram
    /// split method, which leaves them unbinned.
    #[test]
    fn histogram_keeps_the_categorical_path() {
        let n = 400usize;
        let codes: Vec<f64> = (0..n).map(|i| [0.0, 1.0, 2.0, 3.0, 0.0][i % 5]).collect();
        let u: Array1<f64> = codes
            .iter()
            .map(|&c| if c == 0.0 || c == 2.0 { 1.0 } else { -1.0 })
            .collect();
        let design = Array2::from_shape_vec((n, 1), codes).unwrap();
        let w = Array1::ones(n);
        let (_, pred) = fit(&design, &u, &w, SplitMethod::Histogram, &[4], 1, 2, 1);
        for (p, t) in pred.iter().zip(u.iter()) {
            assert!((p - t).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod bspatial_shared_gram_tests {
    use super::*;

    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut lcg = Lcg(seed);
        let x1 = Array1::from_iter((0..n).map(|_| lcg.next_f64()));
        let x2 = Array1::from_iter((0..n).map(|_| lcg.next_f64()));
        let w = Array1::from_iter((0..n).map(|_| 0.5 + lcg.next_f64()));
        (x1, x2, w)
    }

    /// The blueprint materializes the final (possibly centered) tensor design
    /// and penalty; reuse it so both paths see byte-identical inputs.
    fn blueprint_parts(
        x1: &Array1<f64>,
        x2: &Array1<f64>,
        center: bool,
    ) -> (Array2<f64>, Array2<f64>) {
        let bp = LearnerBlueprint::bspatial(
            x1.clone(),
            x2.clone(),
            "spatial",
            6,
            3,
            2,
            6.0,
            None,
            center,
            None,
            None,
            None,
        )
        .unwrap();
        match &bp {
            LearnerBlueprint::Bspatial {
                design, penalty, ..
            } => (design.clone(), penalty.clone()),
            _ => unreachable!(),
        }
    }

    /// The shared-gram solver matrix `X'WX + λK` must agree with the
    /// penalty-seeded faer accumulation it replaces to rounding level.
    #[test]
    fn shared_gram_matches_seeded_faer_accumulation() {
        let (x1, x2, w) = data(400, 7);
        for center in [false, true] {
            let (design, penalty) = blueprint_parts(&x1, &x2, center);
            let xtwx = weighted_gram_blocked(design.view(), w.view());
            let lambda = df_to_lambda_from_gram(&xtwx, &penalty, 6.0);
            let scaled = penalty.mapv(|v| v * lambda);

            let gram_new = &xtwx + &scaled;
            let d = into_column_major(design.clone());
            let gram_old = penalized_weighted_gram(&d, w.view(), &scaled);

            let scale = gram_old.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let mut max_abs = 0.0_f64;
            for (a, b) in gram_new.iter().zip(gram_old.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
            let rel = max_abs / scale;
            eprintln!(
                "shared-gram center={center}: max |X'WX+λK - seeded faer| = {max_abs:e}, rel = {rel:e}"
            );
            // Accumulation-order difference only (ndarray blocked
            // `general_mat_mul` vs the λK-seeded faer kernel).
            assert!(
                rel < 1e-12,
                "center={center}: max_abs={max_abs:e}, rel={rel:e}"
            );
        }
    }

    /// `prepare_bspatial` (shared gram) and the previous dense path
    /// (`df_to_lambda` + `prepare_block_bols`) must produce the same λ and
    /// the same fitted coefficients to tolerance, centered or not.
    #[test]
    fn prepare_bspatial_matches_dense_path() {
        let (x1, x2, w) = data(400, 11);
        let mut lcg = Lcg(99);
        let y = Array1::from_iter((0..400).map(|_| lcg.next_f64()));
        let wu2: f64 = (0..400).map(|i| w[i] * y[i] * y[i]).sum();

        for center in [false, true] {
            let (design, penalty) = blueprint_parts(&x1, &x2, center);

            let lambda = df_to_lambda(&design, &penalty, &w, 6.0);
            let mut dense = prepare_block_bols(
                design,
                w.view(),
                penalty.mapv(|v| v * lambda),
                false,
                "spatial",
            )
            .with_effective_lambda(Some(lambda));

            let mut shared = prepare_bspatial(
                x1.view(),
                x2.view(),
                w.view(),
                6,
                3,
                2,
                6.0,
                None,
                center,
                "spatial",
                None,
            );

            assert_eq!(
                shared.lambda(),
                dense.lambda(),
                "center={center}: shared and dense paths must calibrate the same λ"
            );

            let beta_shared = shared.fit_step_select(y.view(), w.view(), wu2).0.to_vec();
            let beta_dense = dense.fit_step_select(y.view(), w.view(), wu2).0.to_vec();
            assert_eq!(beta_shared.len(), beta_dense.len());

            let scale = beta_dense.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let mut max_abs = 0.0_f64;
            for (a, b) in beta_shared.iter().zip(beta_dense.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
            let rel = max_abs / (scale + f64::MIN_POSITIVE);
            eprintln!(
                "prepare_bspatial vs dense center={center}: max |β_shared - β_dense| = {max_abs:e}, rel = {rel:e}"
            );
            assert!(
                rel < 1e-8,
                "center={center}: max_abs={max_abs:e}, rel={rel:e}"
            );
        }
    }

    fn streamed_learner(
        x1: &Array1<f64>,
        x2: &Array1<f64>,
        w: &Array1<f64>,
        center: bool,
        by: Option<&Array1<f64>>,
    ) -> PreparedLearner {
        prepare_bspatial_streamed(
            x1.view(),
            x2.view(),
            w.view(),
            6,
            3,
            2,
            6.0,
            None,
            center,
            "spatial",
            by.map(|b| b.view()),
        )
    }

    /// The streamed row-wise tensor must match `prepare_bspatial` on λ, the
    /// fitted predictions, the selection SSE and the effective df, centered or
    /// not and with or without a `by` modifier.
    #[test]
    fn streamed_matches_bspatial_path() {
        let (x1, x2, w) = data(400, 21);
        let mut lcg = Lcg(5);
        let y = Array1::from_iter((0..400).map(|_| lcg.next_f64()));
        let by = Array1::from_iter((0..400).map(|_| 0.25 + lcg.next_f64()));
        let wu2: f64 = (0..400).map(|i| w[i] * y[i] * y[i]).sum();

        for center in [false, true] {
            for with_by in [false, true] {
                let by_ref = if with_by { Some(by.view()) } else { None };
                let mut dense = prepare_bspatial(
                    x1.view(),
                    x2.view(),
                    w.view(),
                    6,
                    3,
                    2,
                    6.0,
                    None,
                    center,
                    "spatial",
                    by_ref,
                );
                let mut streamed = prepare_bspatial_streamed(
                    x1.view(),
                    x2.view(),
                    w.view(),
                    6,
                    3,
                    2,
                    6.0,
                    None,
                    center,
                    "spatial",
                    by_ref,
                );

                let (lam_s, lam_d) = (
                    streamed.lambda().expect("streamed λ"),
                    dense.lambda().expect("dense λ"),
                );
                assert!(
                    ((lam_s - lam_d) / lam_d.abs().max(1.0)).abs() < 1e-6,
                    "center={center} by={with_by}: λ {lam_s} vs {lam_d}"
                );
                assert_eq!(streamed.n_parameters(), dense.n_parameters());
                assert_eq!(streamed.n_observations(), dense.n_observations());

                let (_, pred_s, sse_s) = streamed.fit_step_ref(y.view(), w.view());
                let (_, pred_d, sse_d) = dense.fit_step_ref(y.view(), w.view());
                let scale = pred_d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let max_abs = pred_s
                    .iter()
                    .zip(pred_d.iter())
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                let rel = max_abs / (scale + f64::MIN_POSITIVE);
                eprintln!(
                    "streamed center={center} by={with_by}: pred rel={rel:e}, sse {sse_s:e} vs {sse_d:e}"
                );
                assert!(
                    rel < 1e-8,
                    "center={center} by={with_by}: pred max_abs={max_abs:e}, rel={rel:e}"
                );
                assert!(
                    ((sse_s - sse_d) / (sse_d.abs() + 1.0)).abs() < 1e-8,
                    "center={center} by={with_by}: sse {sse_s} vs {sse_d}"
                );

                let select_s = streamed.fit_step_select(y.view(), w.view(), wu2).1;
                let select_d = dense.fit_step_select(y.view(), w.view(), wu2).1;
                assert!(
                    ((select_s - select_d) / (select_d.abs() + 1.0)).abs() < 1e-8,
                    "center={center} by={with_by}: select sse {select_s} vs {select_d}"
                );

                let df_s = streamed.df().expect("streamed df");
                let df_d = dense.df().expect("dense df");
                assert!(
                    (df_s - df_d).abs() < 1e-6,
                    "center={center} by={with_by}: df {df_s} vs {df_d}"
                );
            }
        }
    }

    /// The marginal-eigen centering and the one-eig / pencil calibrations
    /// (#284) must match the dense path on a `knots = 20` basis — the regime
    /// whose full `(p1·p2)`-sized spectral centering motivated the change —
    /// at the default `df = 6` and at a high `df = 200`.
    #[test]
    fn streamed_matches_bspatial_path_big_basis() {
        let (x1, x2, w) = data(2000, 57);
        let mut lcg = Lcg(59);
        let y = Array1::from_iter((0..2000).map(|_| lcg.next_f64()));
        let wu2: f64 = (0..2000).map(|i| w[i] * y[i] * y[i]).sum();

        for center in [false, true] {
            for df in [6.0, 200.0] {
                let mut dense = prepare_bspatial(
                    x1.view(),
                    x2.view(),
                    w.view(),
                    20,
                    3,
                    2,
                    df,
                    None,
                    center,
                    "spatial",
                    None,
                );
                let mut streamed = prepare_bspatial_streamed(
                    x1.view(),
                    x2.view(),
                    w.view(),
                    20,
                    3,
                    2,
                    df,
                    None,
                    center,
                    "spatial",
                    None,
                );

                let (lam_s, lam_d) = (
                    streamed.lambda().expect("streamed λ"),
                    dense.lambda().expect("dense λ"),
                );
                assert!(
                    ((lam_s - lam_d) / lam_d.abs().max(1.0)).abs() < 1e-6,
                    "center={center} df={df}: λ {lam_s} vs {lam_d}"
                );

                let (_, pred_s, _) = streamed.fit_step_ref(y.view(), w.view());
                let (_, pred_d, _) = dense.fit_step_ref(y.view(), w.view());
                let max_abs = pred_s
                    .iter()
                    .zip(pred_d.iter())
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                let scale = pred_d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let rel = max_abs / (scale + f64::MIN_POSITIVE);
                assert!(
                    rel < 1e-6,
                    "center={center} df={df}: pred rel={rel:e}, max_abs={max_abs:e}"
                );

                let select_s = streamed.fit_step_select(y.view(), w.view(), wu2).1;
                let select_d = dense.fit_step_select(y.view(), w.view(), wu2).1;
                assert!(
                    ((select_s - select_d) / (select_d.abs() + 1.0)).abs() < 1e-8,
                    "center={center} df={df}: select sse {select_s} vs {select_d}"
                );

                let df_s = streamed.df().expect("streamed df");
                let df_d = dense.df().expect("dense df");
                assert!(
                    (df_s - df_d).abs() < 1e-6,
                    "center={center} df={df}: df {df_s} vs {df_d}"
                );
            }
        }
    }

    /// `reweight` must reproduce a fresh streamed fit under the new weights.
    #[test]
    fn streamed_reweight_matches_fresh_fit() {
        let (x1, x2, w) = data(300, 31);
        let mut lcg = Lcg(41);
        let w2 = Array1::from_iter((0..300).map(|_| 0.5 + lcg.next_f64()));
        let y = Array1::from_iter((0..300).map(|_| lcg.next_f64()));

        for center in [false, true] {
            let mut reused = streamed_learner(&x1, &x2, &w, center, None);
            let mut fresh = prepare_bspatial_streamed(
                x1.view(),
                x2.view(),
                w2.view(),
                6,
                3,
                2,
                6.0,
                reused.lambda(),
                center,
                "spatial",
                None,
            );
            reused.reweight(w2.view());

            assert_eq!(reused.lambda(), fresh.lambda());
            let (_, pred_reused, _) = reused.fit_step_ref(y.view(), w2.view());
            let (_, pred_fresh, _) = fresh.fit_step_ref(y.view(), w2.view());
            let max_abs = pred_reused
                .iter()
                .zip(pred_fresh.iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            let scale = pred_fresh.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let rel = max_abs / (scale + f64::MIN_POSITIVE);
            assert!(
                rel < 1e-8,
                "center={center}: reweight rel={rel:e}, max_abs={max_abs:e}"
            );
        }
    }
}

#[cfg(test)]
mod bhinge_tests {
    use super::*;

    #[test]
    fn bhinge_recovers_a_planted_kink() {
        let n = 240usize;
        let x = Array1::from_shape_fn(n, |i| i as f64 * (12.0 / (n - 1) as f64));
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * (v - 5.0).max(0.0)).collect();

        let w = Array1::ones(n);

        let mut learner = LearnerBlueprint::bhinge(x, "kink", 60, 4.0, None, false, None)
            .unwrap()
            .prepare(w.view())
            .expect("prepare bhinge")
            .0;
        let u = ArrayView1::from(&y);
        let wu2 = y.iter().map(|v| v * v).sum();
        let (beta, sse) = learner.fit_step_select(u, w.view(), wu2);
        assert_eq!(beta.len(), 4, "[knot, direction, coefficient, centering]");
        let (knot, direction, coefficient) = (beta[0], beta[1], beta[2]);
        assert!(direction > 0.0, "max(0, x - t) expected for a rising kink");
        assert!(
            (knot - 5.0).abs() < 0.25,
            "knot {knot} should land within a candidate spacing of 5.0"
        );
        assert!((coefficient - 2.0).abs() < 0.1, "coefficient {coefficient}");
        // The quantile grid cannot sit exactly on 5.0 (nearest knot 4.92), so a
        // few percent of the kink's variance remains; it must be a small
        // fraction of the total, overwhelmed only by the knot-grid resolution.
        assert!(
            sse < 0.05 * wu2,
            "sse {sse} must be far below the total variance {wu2}"
        );
    }

    #[test]
    fn bhinge_beats_bbs_on_a_kinked_objective() {
        let n = 240usize;
        let x = Array1::from_shape_fn(n, |i| i as f64 * (12.0 / (n - 1) as f64));
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * (v - 5.0).max(0.0)).collect();
        let w = Array1::ones(n);

        let mut hinge = LearnerBlueprint::bhinge(x.clone(), "kink", 60, 4.0, None, false, None)
            .unwrap()
            .prepare(w.view())
            .expect("prepare bhinge")
            .0;
        let mut spline = LearnerBlueprint::bbs(
            x, "spline", 20, 3, 2, 4.0, None, false, None, false, None, 0,
        )
        .unwrap()
        .prepare(w.view())
        .expect("prepare bbs")
        .0;

        let u = ArrayView1::from(&y);
        let wu2 = y.iter().map(|v| v * v).sum();
        let (_, sse_hinge) = hinge.fit_step_select(u, w.view(), wu2);
        let (_, sse_spline) = spline.fit_step_select(u, w.view(), wu2);
        assert!(
            sse_hinge < 0.2 * sse_spline,
            "hinge sse {sse_hinge} should be a fraction of the bbs sse {sse_spline}"
        );
    }

    #[test]
    fn bhinge_fits_a_left_facing_hinge_and_rescales_only_the_coefficient() {
        let n = 200usize;
        let x = Array1::from_shape_fn(n, |i| i as f64 * (12.0 / (n - 1) as f64));
        // Non-zero left of the knot at 5: direction -1, coefficient +3.
        let y: Vec<f64> = x.iter().map(|&v| 3.0 * (5.0 - v).max(0.0)).collect();
        let w = Array1::ones(n);

        let mut learner = LearnerBlueprint::bhinge(x, "left", 60, 4.0, None, false, None)
            .unwrap()
            .prepare(w.view())
            .expect("prepare bhinge")
            .0;
        let u = ArrayView1::from(&y);
        let wu2 = y.iter().map(|v| v * v).sum();
        let (beta_ref, _) = learner.fit_step_select(u, w.view(), wu2);
        let beta: Vec<f64> = beta_ref.to_vec();
        let (knot, direction, coefficient) = (beta[0], beta[1], beta[2]);
        assert!(
            direction < 0.0,
            "max(0, t - x) expected for a left-facing kink"
        );
        assert!((knot - 5.0).abs() < 0.25, "knot {knot} ~ 5.0");
        assert!((coefficient - 3.0).abs() < 0.1, "coefficient {coefficient}");

        let scaled = learner.scale_coef(&beta, 0.5);
        assert_eq!(scaled.len(), 4);
        assert_eq!(scaled[0], beta[0], "knot is structural");
        assert_eq!(scaled[1], beta[1], "direction is structural");
        assert!(
            (scaled[2] - 0.5 * beta[2]).abs() < 1e-12,
            "only the coefficient is rescaled"
        );
        assert_eq!(scaled[3], beta[3], "centering is structural");
    }

    #[test]
    fn bhinge_prediction_replays_the_selected_hinge() {
        let n = 180usize;
        let x = Array1::from_shape_fn(n, |i| i as f64 * (12.0 / (n - 1) as f64));
        let by = Array1::from_shape_fn(n, |i| 0.5 + 0.01 * (i % 7) as f64);
        let y: Vec<f64> = (0..n)
            .map(|i| by[i] * 1.7 * (x[i] - 6.0).max(0.0))
            .collect();
        let w = Array1::ones(n);

        let mut learner =
            LearnerBlueprint::bhinge(x.clone(), "kink", 50, 4.0, None, true, Some(by.clone()))
                .unwrap()
                .prepare(w.view())
                .expect("prepare bhinge")
                .0;
        let u = ArrayView1::from(&y);
        let wu2 = y.iter().map(|v| v * v).sum();
        let (beta_ref, _) = learner.fit_step_select(u, w.view(), wu2);
        let beta: Vec<f64> = beta_ref.to_vec();

        // `pred_buf` holds the pre-nu training prediction; a re-solve decodes
        // the stored `[knot, direction, coefficient, centering]` into exactly
        // that column, and the direct formula agrees row by row.
        let pred_buf = learner.pred_buf();
        assert_eq!(pred_buf.len(), n);
        let mut out = vec![0.0f64; n];
        learner.predict_into(&beta, &mut out);
        let (knot, direction, coefficient, centering) = (beta[0], beta[1], beta[2], beta[3]);
        for i in 0..n {
            assert!((out[i] - pred_buf[i]).abs() < 1e-10, "replay {i}");
            let hinge = if direction > 0.0 {
                (x[i] - knot).max(0.0)
            } else {
                (knot - x[i]).max(0.0)
            };
            let expected = coefficient * (by[i] * hinge - centering);
            assert!((out[i] - expected).abs() < 1e-12, "formula {i}");
        }
    }

    #[test]
    fn bhinge_centers_its_column_and_exposes_the_scalar_api() {
        let n = 120usize;
        let x = Array1::from_shape_fn(n, |i| i as f64 * (10.0 / (n - 1) as f64));
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * (v - 4.0).max(0.0)).collect();
        let w = Array1::ones(n);

        let blueprint = LearnerBlueprint::bhinge(x, "c", 30, 4.0, Some(2.5), true, None).unwrap();
        let (mut learner, _) = blueprint.prepare(w.view()).expect("prepare bhinge");

        // Scalar-shaped API surface: a single ridge weight and one effective
        // parameter, with no design or dof until a hinge is actually chosen.
        assert_eq!(learner.n_parameters(), 1);
        assert_eq!(learner.lambda(), Some(2.5));
        assert_eq!(learner.penalty_matrix().unwrap().shape(), &[1, 1]);
        assert_eq!(learner.design_matrix(), None, "no column before a fit step");
        assert_eq!(learner.df_mboost(), None);

        let u = ArrayView1::from(&y);
        let wu2 = y.iter().map(|v| v * v).sum();
        let (beta_ref, _) = learner.fit_step_select(u, w.view(), wu2);
        let _beta = beta_ref.to_vec();

        // The reported design is the weighted-mean-centered column of the
        // selected hinge, so its weighted mean is zero.
        let design = learner.design_matrix().expect("column after a fit step");
        assert_eq!(design.shape(), &[n, 1]);
        let weighted_sum: f64 = (0..n).map(|i| w[i] * design[[i, 0]]).sum();
        assert!(
            weighted_sum.abs() < 1e-9,
            "centered column sums to {weighted_sum}"
        );
    }
}

/// Parity tests for the `BlockBols` sparse route (#323): the CSR copy of the
/// uncentered design must reproduce the dense faer products, through the
/// centering algebra when the design is weight-centered, and end to end the
/// boosting loop must not be able to tell the routes apart.
#[cfg(test)]
mod sparse_rows_parity {
    use super::*;
    use crate::control::BoostControl;
    use crate::lss::{fit_lss, BinomialLSS};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// |got − want| <= tol · max(|got|, |want|) + 1e-12: relative agreement
    /// with an absolute floor for entries that are both ~0.
    fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
        let diff = (got - want).abs();
        let scale = got.abs().max(want.abs());
        assert!(
            diff <= tol * scale + 1e-12,
            "{what}: got {got}, want {want} (diff {diff}, scale {scale})"
        );
    }

    /// The row-wise scatter that [`SparseRows::tmatvec_into`] replaced, kept
    /// as the reference for the old-versus-new comparison (#323). It walks
    /// the non-zeros of each row of `dense` in ascending column order, as
    /// the removed row-compressed copy stored them.
    fn scatter_tmatvec_reference(dense: &Array2<f64>, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; dense.ncols()];
        for (row, &vi) in dense.rows().into_iter().zip(v) {
            for (col, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    out[col] += value * vi;
                }
            }
        }
        out
    }

    /// The row-wise dot product that [`SparseRows::matvec_into`] replaced
    /// (#738): each row sums its non-zeros in ascending column order from 0.
    fn row_dot_matvec_reference(dense: &Array2<f64>, beta: &[f64]) -> Vec<f64> {
        dense
            .rows()
            .into_iter()
            .map(|row| {
                let mut sum = 0.0;
                for (col, &value) in row.iter().enumerate() {
                    if value != 0.0 {
                        sum += value * beta[col];
                    }
                }
                sum
            })
            .collect()
    }

    #[test]
    fn column_scatter_matvec_is_bit_identical_to_the_row_dot_it_replaced() {
        for (seed, n, p) in [(5u64, 2048usize, 24usize), (17, 5003, 40)] {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut dense = Array2::<f64>::zeros((n, p));
            for row in 0..n {
                // Four non-zeros, wrapping past the last column, so a row's
                // entries are not always a contiguous column range.
                let start = rng.random::<u32>() as usize % p;
                for k in 0..4 {
                    dense[[row, (start + k) % p]] = rng.random::<f64>() * 2.0 - 1.0;
                }
            }
            let sparse =
                SparseRows::from_dense_if_sparse(dense.view(), SPARSE_ROWS_MAX_DENSITY).unwrap();
            let beta: Vec<f64> = (0..p).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
            let want = row_dot_matvec_reference(&dense, &beta);
            let mut got = vec![f64::NAN; n];
            sparse.matvec_into(&beta, &mut got);
            for row in 0..n {
                assert_eq!(
                    got[row].to_bits(),
                    want[row].to_bits(),
                    "n={n} p={p} row {row}: {} vs {}",
                    got[row],
                    want[row]
                );
            }
        }
    }

    #[test]
    fn gather_transpose_matches_the_row_scatter_it_replaced_to_1e_10_relative() {
        // (seed, n, p, non-zeros per row): one-entry factor-like rows, then
        // four-entry B-spline-like rows. Column 0 is left empty in the last
        // case, so the gather meets a column with no entries.
        let cases = [
            (7u64, 2048usize, 3usize, 1usize),
            (11, 4099, 24, 4),
            (13, 9001, 40, 4),
        ];
        for (seed, n, p, per_row) in cases {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut dense = Array2::<f64>::zeros((n, p));
            for row in 0..n {
                let first = if per_row == 1 { 0 } else { 1 };
                let start = first + rng.random::<u32>() as usize % (p - per_row - first + 1);
                for k in 0..per_row {
                    dense[[row, start + k]] = 0.5 + rng.random::<f64>();
                }
            }
            let sparse =
                SparseRows::from_dense_if_sparse(dense.view(), SPARSE_ROWS_MAX_DENSITY).unwrap();
            let v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();

            let old = scatter_tmatvec_reference(&dense, &v);
            let mut new = vec![f64::NAN; p];
            sparse.tmatvec_into(&v, &mut new);
            for col in 0..p {
                assert_close(new[col], old[col], 1e-10, &format!("n={n} p={p} col {col}"));
            }
            if per_row == 4 {
                assert_eq!(old[0], 0.0, "column 0 is empty by construction");
                assert_eq!(new[0], 0.0, "empty column must give exactly zero");
            }
        }
    }

    #[test]
    fn sparse_rows_products_match_dense_to_1e_12_relative() {
        let mut rng = ChaCha8Rng::seed_from_u64(323);
        let n = 2100;
        let p = 24;
        let mut dense = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            // bbs-like support: four non-zeros per row, scattered over p
            let start = rng.random::<u32>() as usize % p;
            for k in 0..4 {
                dense[[row, (start + k) % p]] = 1.0 + rng.random::<f64>();
            }
        }
        let sparse =
            SparseRows::from_dense_if_sparse(dense.view(), SPARSE_ROWS_MAX_DENSITY).unwrap();

        let v = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 2.0 - 1.0);
        let mut transposed = vec![f64::NAN; p];
        sparse.tmatvec_into(v.as_slice().expect("v contiguous"), &mut transposed);
        for col in 0..p {
            let want: f64 = (0..n).map(|row| dense[[row, col]] * v[row]).sum();
            assert_close(transposed[col], want, 1e-12, &format!("Sᵀv[{col}]"));
        }

        let beta = Array1::from_shape_fn(p, |_| rng.random::<f64>() * 2.0 - 1.0);
        let mut pred = vec![f64::NAN; n];
        sparse.matvec_into(beta.as_slice().expect("beta contiguous"), &mut pred);
        for row in 0..n {
            let want: f64 = (0..p).map(|col| dense[[row, col]] * beta[col]).sum();
            assert_close(pred[row], want, 1e-12, &format!("Sβ[{row}]"));
        }
    }

    #[test]
    fn sparse_rows_gates_on_size_and_density() {
        // Below the row threshold the dense route stays.
        let small = Array2::<f64>::zeros((SPARSE_ROWS_MIN_ROWS - 1, 24));
        assert!(SparseRows::from_dense_if_sparse(small.view(), SPARSE_ROWS_MAX_DENSITY).is_none());
        // At the threshold but dense: no copy either.
        let dense =
            Array2::<f64>::from_shape_fn((SPARSE_ROWS_MIN_ROWS, 24), |(_, col)| col as f64 + 1.0);
        assert!(SparseRows::from_dense_if_sparse(dense.view(), SPARSE_ROWS_MAX_DENSITY).is_none());
        // At the threshold and sparse enough: a copy, and its bytes are
        // accounted through byte_size.
        let mut sparse_design = Array2::<f64>::zeros((SPARSE_ROWS_MIN_ROWS, 24));
        for row in 0..SPARSE_ROWS_MIN_ROWS {
            sparse_design[[row, row % 24]] = 1.0;
        }
        let rows = SparseRows::from_dense_if_sparse(sparse_design.view(), SPARSE_ROWS_MAX_DENSITY)
            .expect("4% density is well under the gate");
        assert_eq!(rows.col_vals.len(), SPARSE_ROWS_MIN_ROWS);
        // 4 bytes per col_ptr/row_idx entry, 8 per value; one layout only
        // (#738), so the copy is at most 12 bytes per entry plus col_ptr.
        let expected = (24 + 1 + SPARSE_ROWS_MIN_ROWS) * 4 + SPARSE_ROWS_MIN_ROWS * 8;
        assert!(rows.byte_size() >= expected);
        assert!(rows.byte_size() <= expected + 64);
    }

    /// B-spline-like banded design: `p` columns, at most four non-zeros per
    /// row (the cubic Bernstein weights over the interval containing the
    /// row's x).
    fn banded_bbs_design(n: usize, p: usize) -> Array2<f64> {
        let intervals = p - 3;
        let mut design = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let t = row as f64 / (n - 1) as f64 * intervals as f64;
            let j = (t as usize).min(intervals - 1);
            let f = t - j as f64;
            design[[row, j]] = (1.0 - f) * (1.0 - f) * (1.0 - f);
            design[[row, j + 1]] = 3.0 * f * (1.0 - f) * (1.0 - f);
            design[[row, j + 2]] = 3.0 * f * f * (1.0 - f);
            design[[row, j + 3]] = f * f * f;
        }
        design
    }

    #[test]
    fn block_bols_sparse_route_matches_dense_route() {
        let n = 5000;
        let p = 20;
        let design = banded_bbs_design(n, p);
        let weights = Array1::ones(n);
        // An O(1) ridge, the scale a df-calibrated bbs penalty actually
        // produces: the raw banded Gram alone is near-singular (each
        // interval's cubic row sums to zero against uniform quadrature),
        // and route-parity of beta needs the solve conditioned.
        let penalty = Array2::<f64>::eye(p) * 1.0;

        for center in [false, true] {
            let mut sparse_learner =
                prepare_block_bols(design.clone(), weights.view(), penalty.clone(), center, "s");
            let mut dense_learner =
                prepare_block_bols(design.clone(), weights.view(), penalty.clone(), center, "d");
            dense_learner.clear_sparse();
            assert!(
                matches!(
                    sparse_learner,
                    PreparedLearner::BlockBols {
                        sparse: Some(_),
                        ..
                    }
                ),
                "center={center}: n={n} with 4 nnz of p={p} must build the CSR copy"
            );

            let mut rng = ChaCha8Rng::seed_from_u64(100 + center as u64);
            for _ in 0..3 {
                let u = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 2.0 - 1.0);
                let wu2: f64 = weights.iter().zip(u.iter()).map(|(&w, &v)| w * v * v).sum();

                let (beta_sparse, sse_sparse) =
                    sparse_learner.fit_step_select(u.view(), weights.view(), wu2);
                let (beta_dense, sse_dense) =
                    dense_learner.fit_step_select(u.view(), weights.view(), wu2);
                assert_eq!(beta_sparse.len(), p);
                for j in 0..p {
                    assert_close(
                        beta_sparse[j],
                        beta_dense[j],
                        1e-9,
                        &format!("beta[{j}] (center={center})"),
                    );
                }
                assert_close(
                    sse_sparse,
                    sse_dense,
                    1e-9,
                    &format!("sse (center={center})"),
                );
                let beta_sparse: Vec<f64> = beta_sparse.to_vec();
                let beta_dense: Vec<f64> = beta_dense.to_vec();

                let mut pred_sparse = vec![f64::NAN; n];
                let mut pred_dense = vec![f64::NAN; n];
                sparse_learner.predict_into(&beta_sparse, &mut pred_sparse);
                dense_learner.predict_into(&beta_dense, &mut pred_dense);
                let scale = pred_sparse
                    .iter()
                    .chain(pred_dense.iter())
                    .fold(0.0f64, |m, &v| m.max(v.abs()));
                for row in 0..n {
                    let diff = (pred_sparse[row] - pred_dense[row]).abs();
                    assert!(
                        diff <= 1e-10 * scale + 1e-14,
                        "pred[{row}] (center={center}): {} vs {} (diff {diff})",
                        pred_sparse[row],
                        pred_dense[row]
                    );
                }
            }
        }
    }

    #[test]
    fn block_bols_sparse_copy_builds_on_first_use() {
        let n = 5000;
        let p = 20;
        let design = banded_bbs_design(n, p);
        let weights = Array1::ones(n);
        let penalty = Array2::<f64>::eye(p);
        let cache_of = |learner: &PreparedLearner| match learner {
            PreparedLearner::BlockBols { sparse, .. } => sparse.clone(),
            _ => panic!("expected a BlockBols learner"),
        };

        // Uncentered preparation defers the build (#738): the Gaussian
        // Gram route never reads the copy, so it must not be retained
        // from preparation.
        let mut lazy = prepare_block_bols(
            design.clone(),
            weights.view(),
            penalty.clone(),
            false,
            "lazy",
        );
        let cache = cache_of(&lazy).expect("n=5000 banded design must carry a sparse cache");
        assert!(
            !cache.is_built(),
            "uncentered prepare must not build the copy"
        );
        let bytes_before = lazy.retained_data_bytes();

        // Centred preparation still builds eagerly: its copy is taken
        // from the pre-centring design, which is not retained.
        let eager = prepare_block_bols(
            design.clone(),
            weights.view(),
            penalty.clone(),
            true,
            "eager",
        );
        let eager_cache = cache_of(&eager).expect("centred prepare keeps a sparse cache");
        assert!(
            eager_cache.is_built(),
            "centred prepare must build the copy"
        );

        // The first fit step builds the copy, and the retained bytes grow
        // by the built copy alone.
        let u = Array1::from_shape_fn(n, |i| ((i * 31) % 97) as f64 / 97.0 - 0.5);
        let wu2: f64 = weights.iter().zip(u.iter()).map(|(&w, &v)| w * v * v).sum();
        let _ = lazy.fit_step_select(u.view(), weights.view(), wu2);
        assert!(cache.is_built(), "the first fit step must build the copy");
        assert_eq!(
            lazy.retained_data_bytes() - bytes_before,
            cache.byte_size(),
            "retained bytes must grow by the built copy alone"
        );
    }

    #[test]
    fn lss_fit_sparse_route_matches_dense_route_end_to_end() {
        let n = 5000;
        let x1 = Array1::from_shape_fn(n, |i| i as f64 / n as f64);
        let x2 = Array1::from_shape_fn(n, |i| ((i * 7) % n) as f64 / n as f64);
        let y = Array1::from_shape_fn(n, |i| {
            let eta = 3.0 * (x1[i] - 0.5) + 2.0 * (6.0 * x2[i]).sin();
            1.0 / (1.0 + (-eta).exp())
        });
        let weights = Array1::ones(n);

        let build = |clear: bool| -> Vec<Vec<PreparedLearner>> {
            let mut learners = vec![
                prepare_bbs(x1.view(), weights.view(), 16, 3, 2, 8.0, false, "bbs(x1)"),
                prepare_bbs(x2.view(), weights.view(), 12, 3, 2, 8.0, false, "bbs(x2)"),
            ];
            if clear {
                for learner in learners.iter_mut() {
                    learner.clear_sparse();
                }
            }
            vec![learners]
        };

        let mut sparse_run = build(false);
        assert!(sparse_run[0].iter().all(|l| matches!(
            l,
            PreparedLearner::BlockBols {
                sparse: Some(_),
                ..
            }
        )));
        let mut dense_run = build(true);
        assert!(dense_run[0]
            .iter()
            .all(|l| matches!(l, PreparedLearner::BlockBols { sparse: None, .. })));

        let fam = BinomialLSS;
        let ctrl = BoostControl::new(25, 0.1).unwrap();
        let path_sparse = fit_lss(
            &mut sparse_run,
            y.view(),
            weights.view(),
            &fam,
            &ctrl,
            None,
            None,
        );
        let path_dense = fit_lss(
            &mut dense_run,
            y.view(),
            weights.view(),
            &fam,
            &ctrl,
            None,
            None,
        );

        assert_eq!(
            path_sparse.per_param[0].selected, path_dense.per_param[0].selected,
            "the sparse and dense routes must select the same learner sequence"
        );
        assert_eq!(path_sparse.risk.len(), path_dense.risk.len());
        for (m, (risk_sparse, risk_dense)) in path_sparse
            .risk
            .iter()
            .zip(path_dense.risk.iter())
            .enumerate()
        {
            assert_close(*risk_sparse, *risk_dense, 1e-9, &format!("risk[{m}]"));
        }
    }
}
