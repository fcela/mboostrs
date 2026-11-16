//! Joint support of new rows relative to training rows.
//!
//! A new row can sit inside every predictor's training range and still lie
//! where the training data has no rows: a high `x1` with a low `x2` when the
//! two are strongly correlated. Marginal range checks miss it. This measures
//! each new row's Mahalanobis distance from the training rows,
//! `d^2 = (x - m)' S^+ (x - m)`, with `m` the training mean and `S^+` the
//! pseudo-inverse of the training covariance (directions with no training
//! variance are dropped), and reports the share of training rows whose own
//! distance is at least as large. A share near zero means the new row is
//! farther from the training data's centre, in the data's own metric, than
//! almost every training row. The distance is linear in the predictors, so it
//! detects linear joint-support gaps; a curved or clustered training cloud can
//! leave a gap it does not see.

use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Mahalanobis distance of a new row and the share of training rows at
/// least as far from the training mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointSupport {
    /// `d^2`, the squared Mahalanobis distance from the training mean.
    pub distance_sq: f64,
    /// Share of training rows with `d^2` at least this row's.
    pub training_share_beyond: f64,
}

/// Joint support of each row of `new` relative to `train` (rows are
/// observations, columns predictors; both must have the same columns).
/// Returns `None` when `train` has fewer than two rows or the column counts
/// differ.
pub fn joint_support(train: ArrayView2<f64>, new: ArrayView2<f64>) -> Option<Vec<JointSupport>> {
    let (n, p) = train.dim();
    if n < 2 || new.ncols() != p || p == 0 {
        return None;
    }
    let mean: Array1<f64> = train.mean_axis(Axis(0))?;
    let centred = &train - &mean;
    let cov: Array2<f64> = centred.t().dot(&centred) / (n as f64 - 1.0);
    // Pseudo-inverse through the eigendecomposition of the covariance,
    // dropping directions whose variance is negligible against the largest.
    let fm = faer::Mat::from_fn(p, p, |i, j| 0.5 * (cov[[i, j]] + cov[[j, i]]));
    let eig = fm.selfadjoint_eigendecomposition(faer::Side::Lower);
    let vals = eig.s().column_vector();
    let vecs = eig.u();
    let largest = (0..p).map(|k| vals.read(k)).fold(0.0f64, f64::max);
    let tol = largest * (p as f64) * f64::EPSILON * 1e3;
    let keep: Vec<usize> = (0..p).filter(|&k| vals.read(k) > tol).collect();
    let distance = |row: ndarray::ArrayView1<f64>| -> f64 {
        keep.iter()
            .map(|&k| {
                let proj: f64 = (0..p).map(|j| vecs.read(j, k) * (row[j] - mean[j])).sum();
                proj * proj / vals.read(k)
            })
            .sum()
    };
    let mut train_d: Vec<f64> = train.rows().into_iter().map(distance).collect();
    train_d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(
        new.rows()
            .into_iter()
            .map(|row| {
                let d = distance(row);
                // Training rows with distance >= d: those from the first
                // index at or above d (a small tolerance absorbs rounding for
                // a new row equal to a training row).
                let first = train_d.partition_point(|&t| t < d * (1.0 - 1e-12));
                JointSupport {
                    distance_sq: d,
                    training_share_beyond: (n - first) as f64 / n as f64,
                }
            })
            .collect(),
    )
}

/// Same-cell nearest-neighbour distance of each new row to the training rows.
///
/// A new row is a counterfactual combination: one predictor is set to a value
/// and the others keep the row's own values. Its categorical predictors form a
/// cell, and the training rows of that cell are the only evidence about the
/// combination. `train_cells` and `new_cells` hold the categorical predictors
/// as integer codes stored in `f64`; a level never seen in training carries a
/// negative code and matches no cell. The numeric predictors (`train_numeric`,
/// `new_numeric`) are standardized by the training means and standard
/// deviations, and each new row's value is the Euclidean distance to its
/// nearest training row in the same cell. A row with no same-cell training
/// row, or with a non-finite numeric value, gets `f64::INFINITY`. Training rows
/// with a non-finite numeric value are ignored.
///
/// Distances are in training standard-deviation units. A nonlinear or
/// clustered training cloud can leave a gap that this distance does not show
/// when the nearest same-cell row happens to be close in the other predictors.
pub fn cell_nearest_distance(
    train_cells: ArrayView2<f64>,
    train_numeric: ArrayView2<f64>,
    new_cells: ArrayView2<f64>,
    new_numeric: ArrayView2<f64>,
) -> Result<Vec<f64>, crate::MboostrsError> {
    use crate::MboostrsError;
    use rayon::prelude::*;

    let n = train_cells.nrows();
    if train_numeric.nrows() != n
        || new_cells.ncols() != train_cells.ncols()
        || new_numeric.ncols() != train_numeric.ncols()
        || new_cells.nrows() != new_numeric.nrows()
    {
        return Err(MboostrsError::Shape(format!(
            "train cells {:?} and numeric {:?} must share rows; new cells {:?} and numeric {:?} \
             must match them in columns",
            train_cells.dim(),
            train_numeric.dim(),
            new_cells.dim(),
            new_numeric.dim()
        )));
    }
    let p = train_numeric.ncols();
    let usable: Vec<usize> = (0..n)
        .filter(|&i| train_numeric.row(i).iter().all(|v| v.is_finite()))
        .collect();
    if usable.is_empty() {
        return Err(MboostrsError::Shape(
            "no training row has finite numeric predictors; support cannot be assessed".into(),
        ));
    }
    let count = usable.len() as f64;
    let mut mean = vec![0.0; p];
    let mut sd = vec![1.0; p];
    for j in 0..p {
        let m = usable.iter().map(|&i| train_numeric[[i, j]]).sum::<f64>() / count;
        let var = usable
            .iter()
            .map(|&i| (train_numeric[[i, j]] - m).powi(2))
            .sum::<f64>()
            / count;
        mean[j] = m;
        // A constant column has no spread to standardize by.
        sd[j] = if var > 0.0 { var.sqrt() } else { 1.0 };
    }
    let standardized = |row: ndarray::ArrayView1<f64>| -> Vec<f64> {
        (0..p).map(|j| (row[j] - mean[j]) / sd[j]).collect()
    };
    let train_rows: Vec<(Vec<f64>, Vec<f64>)> = usable
        .iter()
        .map(|&i| {
            (
                train_cells.row(i).to_vec(),
                standardized(train_numeric.row(i)),
            )
        })
        .collect();
    let out: Vec<f64> = (0..new_cells.nrows())
        .into_par_iter()
        .map(|r| {
            let cell = new_cells.row(r);
            let numeric = new_numeric.row(r);
            if numeric.iter().any(|v| !v.is_finite()) || cell.iter().any(|c| *c < 0.0) {
                return f64::INFINITY;
            }
            let z = standardized(numeric);
            let mut best = f64::INFINITY;
            for (codes, zt) in &train_rows {
                if codes.iter().zip(cell.iter()).any(|(a, b)| a != b) {
                    continue;
                }
                let d2: f64 = z.iter().zip(zt).map(|(a, b)| (a - b).powi(2)).sum();
                if d2 < best {
                    best = d2;
                }
            }
            best.sqrt()
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_counterfactual_needs_a_training_row_in_its_own_cell() {
        // Cell code 0 is observed with x near 0 and 1; cell code 1 only with x
        // near 9 and 10. Query (cell 0, x = 10) has no same-cell row nearby;
        // query (cell 1, x = 10) does.
        let train_cells = Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let train_num = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 9.0, 10.0]).unwrap();
        let new_cells = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, -1.0]).unwrap();
        let new_num = Array2::from_shape_vec((3, 1), vec![10.0, 10.0, 10.0]).unwrap();
        let d = cell_nearest_distance(
            train_cells.view(),
            train_num.view(),
            new_cells.view(),
            new_num.view(),
        )
        .unwrap();
        assert!(d[1].is_finite() && d[1] < 0.5, "{d:?}");
        // Cell 0 has training rows, so the distance is finite: its nearest row
        // is x = 1, nine units away. The training sd of x is sqrt(20.5), so
        // the distance is 9 / sqrt(20.5) = 1.99 standard deviations.
        assert!((d[0] - 9.0 / 20.5f64.sqrt()).abs() < 1e-12, "{d:?}");
        assert!(d[2].is_infinite(), "an unseen level matches no cell: {d:?}");
    }

    #[test]
    fn the_distance_is_in_training_standard_deviations() {
        // One cell; training x = 0, 2 has mean 1 and sd 1, so a query at 3 is
        // 1 sd from the nearest training row (x = 2).
        let train_cells = Array2::zeros((2, 1));
        let train_num = Array2::from_shape_vec((2, 1), vec![0.0, 2.0]).unwrap();
        let new_cells = Array2::zeros((1, 1));
        let new_num = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let d = cell_nearest_distance(
            train_cells.view(),
            train_num.view(),
            new_cells.view(),
            new_num.view(),
        )
        .unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12, "{d:?}");
    }

    #[test]
    fn a_constant_column_is_ignored() {
        let train = Array2::from_shape_fn((50, 2), |(i, j)| if j == 0 { i as f64 } else { 3.0 });
        let new = Array2::from_shape_vec((1, 2), vec![25.0, 7.0]).unwrap();
        let out = joint_support(train.view(), new.view()).unwrap();
        assert!(out[0].distance_sq.is_finite() && out[0].distance_sq < 1.0);
    }

    #[test]
    fn a_point_off_a_correlated_cloud_is_flagged_inside_the_marginals() {
        // x2 = x1 + small noise on [0, 1]: (0.9, 0.1) is inside both ranges
        // and far from the line the data lies on.
        let n = 400;
        let train = Array2::from_shape_fn((n, 2), |(i, j)| {
            let x1 = (i as f64 + 0.5) / n as f64;
            let noise = (((i * 7919) % 101) as f64 / 101.0 - 0.5) * 0.04;
            if j == 0 {
                x1
            } else {
                x1 + noise
            }
        });
        let new = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.9, 0.1]).unwrap();
        let out = joint_support(train.view(), new.view()).unwrap();
        assert!(out[0].training_share_beyond > 0.5, "{:?}", out[0]);
        assert_eq!(out[1].training_share_beyond, 0.0);
        assert!(out[1].distance_sq > 100.0);
    }
}
