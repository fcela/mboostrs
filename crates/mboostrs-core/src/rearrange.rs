//! Rearrangement of predictions at increasing levels.
//!
//! A quantile (or expectile) model fitted separately at each of several
//! levels can predict a higher level below a lower one at some covariate
//! values. Chernozhukov, Fernandez-Val and Galichon (2010) repair this at
//! prediction time by sorting each row's predictions into increasing order.
//! [`rearrange_levels`] does that sort and reports how often, and by how much,
//! the unsorted predictions crossed.

use ndarray::{Array2, ArrayView2};

use crate::MboostrsError;

/// Row-wise sorted predictions and the crossings found before sorting.
#[derive(Debug, Clone, PartialEq)]
pub struct Rearranged {
    /// `n x K` matrix whose every row is in non-decreasing order.
    pub sorted: Array2<f64>,
    /// Number of rows with at least one crossing.
    pub crossing_rows: usize,
    /// `crossing_rows / n`; `0` when there are no rows.
    pub crossing_share: f64,
    /// Entry `k` counts rows where level `k` was predicted strictly above
    /// level `k + 1` (length `K - 1`).
    pub pair_crossings: Vec<usize>,
    /// Largest `prediction[k] - prediction[k + 1]` over all rows and adjacent
    /// pairs; `0` when nothing crossed.
    pub max_crossing: f64,
}

/// Sort each row of `predictions` (`n x K`, columns at increasing levels) and
/// report the crossings of adjacent levels in the input.
///
/// A crossing is an adjacent pair where the lower level is predicted strictly
/// above the next level. Ties are not crossings. Rows already in order are
/// returned unchanged. NaN anywhere in the input is an error, because a NaN
/// has no position in a sorted row.
pub fn rearrange_levels(predictions: ArrayView2<'_, f64>) -> Result<Rearranged, MboostrsError> {
    let (n, k) = predictions.dim();
    if k == 0 {
        return Err(MboostrsError::Shape(
            "predictions must have at least one level (column)".into(),
        ));
    }
    if predictions.iter().any(|v| v.is_nan()) {
        return Err(MboostrsError::BadResponse(
            "predictions".into(),
            "values must not contain NaN; a row with a NaN cannot be sorted".into(),
        ));
    }
    let mut sorted = predictions.to_owned();
    let mut pair_crossings = vec![0usize; k - 1];
    let mut crossing_rows = 0usize;
    let mut max_crossing = 0.0f64;
    let mut buf = vec![0.0f64; k];
    for i in 0..n {
        let mut crossed = false;
        for j in 0..k {
            buf[j] = predictions[[i, j]];
        }
        for j in 0..k - 1 {
            let gap = buf[j] - buf[j + 1];
            if gap > 0.0 {
                crossed = true;
                pair_crossings[j] += 1;
                if gap > max_crossing {
                    max_crossing = gap;
                }
            }
        }
        if crossed {
            crossing_rows += 1;
            buf.sort_by(|a, b| a.total_cmp(b));
            for j in 0..k {
                sorted[[i, j]] = buf[j];
            }
        }
    }
    let crossing_share = if n == 0 {
        0.0
    } else {
        crossing_rows as f64 / n as f64
    };
    Ok(Rearranged {
        sorted,
        crossing_rows,
        crossing_share,
        pair_crossings,
        max_crossing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sorted_input_is_unchanged_and_reports_no_crossings() {
        let m = array![[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]];
        let r = rearrange_levels(m.view()).unwrap();
        assert_eq!(r.sorted, m);
        assert_eq!(r.crossing_rows, 0);
        assert_eq!(r.crossing_share, 0.0);
        assert_eq!(r.pair_crossings, vec![0, 0]);
        assert_eq!(r.max_crossing, 0.0);
    }

    #[test]
    fn crafted_crossings_are_sorted_and_counted() {
        // row 0: pair 0 crosses by 2 (3 > 1); row 1 is ordered;
        // row 2: pair 1 crosses by 1 (5 > 4).
        let m = array![[3.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 5.0, 4.0]];
        let r = rearrange_levels(m.view()).unwrap();
        assert_eq!(
            r.sorted,
            array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 4.0, 5.0]]
        );
        assert_eq!(r.crossing_rows, 2);
        assert!((r.crossing_share - 2.0 / 3.0).abs() < 1e-15);
        assert_eq!(r.pair_crossings, vec![1, 1]);
        assert_eq!(r.max_crossing, 2.0);
    }

    #[test]
    fn ties_are_not_crossings() {
        let m = array![[1.0, 1.0, 1.0]];
        let r = rearrange_levels(m.view()).unwrap();
        assert_eq!(r.crossing_rows, 0);
    }

    #[test]
    fn nan_is_refused() {
        let m = array![[1.0, f64::NAN]];
        assert!(rearrange_levels(m.view()).is_err());
    }

    #[test]
    fn empty_columns_are_refused_and_empty_rows_are_fine() {
        assert!(rearrange_levels(Array2::<f64>::zeros((2, 0)).view()).is_err());
        let r = rearrange_levels(Array2::<f64>::zeros((0, 3)).view()).unwrap();
        assert_eq!(r.crossing_share, 0.0);
    }
}
