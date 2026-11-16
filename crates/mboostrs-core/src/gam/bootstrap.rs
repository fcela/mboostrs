//! Row bootstrap of a penalized REML refit.
//!
//! A bootstrap draw of the rows is a vector of multiplicities `c_i`: the
//! number of times row `i` appears in the resample. A multinomial draw of `n`
//! rows gives `c_i` that sum to `n`. Refitting the resampled rows is the same
//! fit as refitting the full sample with prior weights `w_i c_i`, so every
//! draw keeps the full-sample designs, penalties and spline knots and changes
//! only the weights. The smoothing parameters are re-estimated by REML on
//! each draw, so the draws carry the variation of the smoothing choice too.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

use super::{try_gam_refit, NewRowsTerm, RefitControl, RefitFamily, RefitResult, RefitTerm};
use crate::MboostrsError;

/// Refit `terms` once for each row of `counts`.
///
/// `counts` is `B x n`: `counts[(b, i)]` is the multiplicity of row `i` in
/// draw `b`, and must be finite and non-negative. Draw `b` is fitted with
/// prior weights `weights_i * counts[(b, i)]` (`weights` defaults to one). The
/// response, the designs and the penalties are those of the full sample. The
/// draws are fitted in parallel and returned in the order of `counts`; a draw
/// that fails to converge is still returned, with its `converged` flag false,
/// so the caller can count it.
pub fn refit_row_bootstrap(
    terms: &[RefitTerm],
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    counts: ArrayView2<f64>,
    family: &RefitFamily,
    control: &RefitControl,
) -> Result<Vec<RefitResult>, MboostrsError> {
    let n = y.len();
    let draws = counts.nrows();
    if draws < 2 {
        return Err(MboostrsError::Shape(
            "a row bootstrap needs at least two draws".to_string(),
        ));
    }
    if counts.ncols() != n {
        return Err(MboostrsError::Shape(format!(
            "counts have {} columns for {n} observations",
            counts.ncols()
        )));
    }
    if counts.iter().any(|c| !(c.is_finite() && *c >= 0.0)) {
        return Err(MboostrsError::BadResponse(
            "bootstrap".to_string(),
            "row multiplicities must be finite and non-negative".to_string(),
        ));
    }
    let base: Array1<f64> = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(MboostrsError::Shape(format!(
                    "weights have {} entries for {n} observations",
                    w.len()
                )));
            }
            w.to_owned()
        }
        None => Array1::ones(n),
    };
    (0..draws)
        .into_par_iter()
        .map(|b| {
            let row = counts.row(b);
            let weights_b: Array1<f64> = &base * &row;
            try_gam_refit(
                terms.to_vec(),
                y,
                Some(weights_b.view()),
                None,
                family,
                control,
            )
        })
        .collect()
}

/// The linear predictor of each refit at the same new rows, as a
/// `draws x rows` matrix: entry `(b, j)` is the intercept plus every term's
/// effect at new row `j` under draw `b`.
///
/// Every draw must have the same terms as the others, in the same order;
/// `terms` gives the new rows' design of each term, as
/// [`RefitResult::linear_predictor`] takes it.
pub fn bootstrap_linear_predictor(
    draws: &[RefitResult],
    terms: &[NewRowsTerm],
) -> Result<Array2<f64>, MboostrsError> {
    let rows: Vec<Array1<f64>> = draws
        .par_iter()
        .map(|draw| draw.linear_predictor(terms, false).map(|(eta, _)| eta))
        .collect::<Result<_, _>>()?;
    let m = rows.first().map_or(0, |r| r.len());
    let mut out = Array2::<f64>::zeros((draws.len(), m));
    for (b, eta) in rows.iter().enumerate() {
        out.row_mut(b).assign(eta);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// A smooth-free design: one column of x and one of x squared, each a
    /// term with a ridge-like penalty, on a fixed grid of rows.
    fn toy_terms(n: usize) -> (Vec<RefitTerm>, Array1<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let design = Array2::from_shape_fn((n, 4), |(i, j)| {
            let t = x[i];
            [1.0, t, t * t, (3.0 * t).sin()][j]
        });
        let y = Array1::from_shape_fn(n, |i| {
            let t = x[i];
            2.0 * t + (4.0 * t).cos() + 0.1 * ((i * 7 % 11) as f64 - 5.0)
        });
        let penalty =
            Array2::from_shape_fn((4, 4), |(i, j)| if i == j && i > 0 { 1.0 } else { 0.0 });
        let term = RefitTerm {
            name: "x".to_string(),
            design,
            penalties: vec![penalty],
            by: None,
            variables: vec!["x".to_string()],
        };
        (vec![term], y)
    }

    #[test]
    fn unit_multiplicities_reproduce_the_full_sample_fit() {
        let n = 60;
        let (terms, y) = toy_terms(n);
        let control = RefitControl::default();
        let full = try_gam_refit(
            terms.clone(),
            y.view(),
            None,
            None,
            &RefitFamily::Gaussian,
            &control,
        )
        .unwrap();
        let counts = Array2::from_elem((3, n), 1.0);
        let draws = refit_row_bootstrap(
            &terms,
            y.view(),
            None,
            counts.view(),
            &RefitFamily::Gaussian,
            &control,
        )
        .unwrap();
        assert_eq!(draws.len(), 3);
        for draw in &draws {
            for (a, b) in draw.coef.iter().zip(full.coef.iter()) {
                assert!((a - b).abs() < 1e-10, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn zero_multiplicities_drop_rows_from_the_fit() {
        let n = 60;
        let (terms, y) = toy_terms(n);
        let control = RefitControl::default();
        let mut counts = Array2::from_elem((2, n), 1.0);
        for i in n / 2..n {
            counts[(1, i)] = 0.0;
        }
        let draws = refit_row_bootstrap(
            &terms,
            y.view(),
            None,
            counts.view(),
            &RefitFamily::Gaussian,
            &control,
        )
        .unwrap();
        let diff: f64 = draws[0]
            .coef
            .iter()
            .zip(draws[1].coef.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "dropping half of the rows must change the fit");
    }

    #[test]
    fn linear_predictor_has_one_row_per_draw() {
        let n = 40;
        let (terms, y) = toy_terms(n);
        let control = RefitControl::default();
        let counts = Array2::from_shape_fn((4, n), |(b, i)| ((b + i) % 3) as f64);
        let draws = refit_row_bootstrap(
            &terms,
            y.view(),
            None,
            counts.view(),
            &RefitFamily::Gaussian,
            &control,
        )
        .unwrap();
        let raw = Array2::from_shape_fn((5, 4), |(j, c)| {
            [
                1.0,
                0.2 * j as f64,
                0.04 * (j * j) as f64,
                (0.6 * j as f64).sin(),
            ][c]
        });
        let new_terms = vec![NewRowsTerm {
            raw,
            by: None,
            zero_rows_have_no_effect: false,
        }];
        let eta = bootstrap_linear_predictor(&draws, &new_terms).unwrap();
        assert_eq!(eta.dim(), (4, 5));
        assert!(eta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn refuses_negative_multiplicities_and_one_draw() {
        let n = 20;
        let (terms, y) = toy_terms(n);
        let control = RefitControl::default();
        let mut bad = Array2::from_elem((2, n), 1.0);
        bad[(0, 3)] = -1.0;
        assert!(refit_row_bootstrap(
            &terms,
            y.view(),
            None,
            bad.view(),
            &RefitFamily::Gaussian,
            &control
        )
        .is_err());
        let one = Array2::from_elem((1, n), 1.0);
        assert!(refit_row_bootstrap(
            &terms,
            y.view(),
            None,
            one.view(),
            &RefitFamily::Gaussian,
            &control
        )
        .is_err());
    }
}
