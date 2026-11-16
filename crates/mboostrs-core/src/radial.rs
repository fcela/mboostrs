//! Matérn radial bases with a retained new-location prediction recipe.

use faer::Mat;
use ndarray::{Array2, ArrayView2};

use crate::MboostrsError;

/// Matérn correlation at smoothness 3/2 and positive range `theta`.
pub fn matern32(distance: f64, theta: f64) -> f64 {
    let r = 3.0_f64.sqrt() * distance / theta;
    if r.is_infinite() {
        return 0.0;
    }
    (1.0 + r) * (-r).exp()
}

fn checked_coordinates(x: ArrayView2<'_, f64>) -> Result<(), MboostrsError> {
    if x.ncols() == 0 || x.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::Shape(
            "brad coordinates must have columns and be finite".into(),
        ));
    }
    Ok(())
}

fn squared_distance(a: ndarray::ArrayView1<'_, f64>, b: ndarray::ArrayView1<'_, f64>) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Range making Matérn correlation `eps` at the training data's diameter.
/// Pair distances are streamed, using constant auxiliary memory.
pub fn effective_range(x: ArrayView2<'_, f64>, eps: f64) -> Result<f64, MboostrsError> {
    checked_coordinates(x)?;
    if !eps.is_finite() || !(0.0..1.0).contains(&eps) || eps == 0.0 {
        return Err(MboostrsError::Control(
            "brad range tolerance must be in (0, 1)".into(),
        ));
    }
    let mut diameter_sq: f64 = 0.0;
    for i in 0..x.nrows() {
        for j in 0..i {
            diameter_sq = diameter_sq.max(squared_distance(x.row(i), x.row(j)));
        }
    }
    if diameter_sq == 0.0 {
        return Ok(1.0);
    }
    let diameter = diameter_sq.sqrt();
    if !diameter.is_finite() {
        return Err(MboostrsError::Shape(
            "brad coordinate distances overflow".into(),
        ));
    }
    let (mut lo, mut hi) = (1e-6 * diameter, 1e3 * diameter);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if matern32(diameter, mid) > eps {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Radial-basis geometry retained independently of fit weights and coefficients.
#[derive(Clone, Debug)]
pub struct RadialBasis {
    /// Lexicographically ordered, distinct, space-filling training centres.
    pub centres: Array2<f64>,
    /// Symmetric inverse square root of the centres' covariance matrix.
    pub whitening: Array2<f64>,
    /// Matérn range.
    pub theta: f64,
}

impl RadialBasis {
    /// Choose centres by deterministic greedy max-min coverage and whiten
    /// their covariance. This retains the project's existing construction;
    /// upstream mboost instead uses fields' stochastic cover.design.
    pub fn new(
        x: ArrayView2<'_, f64>,
        knots: usize,
        theta: Option<f64>,
    ) -> Result<Self, MboostrsError> {
        checked_coordinates(x)?;
        if knots == 0 || x.nrows() == 0 {
            return Err(MboostrsError::Shape(
                "brad requires training rows and at least one knot".into(),
            ));
        }
        let mut rows: Vec<usize> = (0..x.nrows()).collect();
        rows.sort_by(|&i, &j| {
            for k in 0..x.ncols() {
                let order = x[[i, k]].partial_cmp(&x[[j, k]]).unwrap();
                if !order.is_eq() {
                    return order;
                }
            }
            std::cmp::Ordering::Equal
        });
        rows.dedup_by(|a, b| x.row(*a).iter().zip(x.row(*b)).all(|(u, v)| u == v));
        let unique = Array2::from_shape_fn((rows.len(), x.ncols()), |(i, j)| x[[rows[i], j]]);
        let count = knots.min(unique.nrows());
        let mut chosen = if count == unique.nrows() {
            (0..count).collect::<Vec<_>>()
        } else {
            let centroid = ndarray::Array1::from_shape_fn(x.ncols(), |j| {
                unique.column(j).sum() / unique.nrows() as f64
            });
            let first = (0..unique.nrows())
                .min_by(|&a, &b| {
                    squared_distance(unique.row(a), centroid.view())
                        .total_cmp(&squared_distance(unique.row(b), centroid.view()))
                })
                .unwrap();
            let mut selected = vec![first];
            let mut closest: Vec<f64> = unique
                .rows()
                .into_iter()
                .map(|row| squared_distance(row, unique.row(first)))
                .collect();
            for _ in 1..count {
                let mut next = 0;
                for i in 1..closest.len() {
                    if closest[i] > closest[next] {
                        next = i;
                    }
                }
                selected.push(next);
                for i in 0..closest.len() {
                    closest[i] = closest[i].min(squared_distance(unique.row(i), unique.row(next)));
                }
            }
            selected
        };
        chosen.sort_unstable();
        let centres = Array2::from_shape_fn((count, x.ncols()), |(i, j)| unique[[chosen[i], j]]);
        let theta = match theta {
            Some(value) => value,
            None => effective_range(x, 1e-3)?,
        };
        if !theta.is_finite() || theta <= 0.0 {
            return Err(MboostrsError::Control(
                "brad theta must be finite and positive".into(),
            ));
        }
        let covariance = Mat::from_fn(count, count, |i, j| {
            matern32(
                squared_distance(centres.row(i), centres.row(j)).sqrt(),
                theta,
            )
        });
        let eigen = covariance.selfadjoint_eigendecomposition(faer::Side::Lower);
        let vectors = eigen.u();
        let values = eigen.s().column_vector();
        let whitening = Array2::from_shape_fn((count, count), |(i, j)| {
            (0..count)
                .map(|k| vectors.read(i, k) * vectors.read(j, k) / values.read(k).max(1e-10).sqrt())
                .sum()
        });
        Ok(Self {
            centres,
            whitening,
            theta,
        })
    }

    /// Rebuild the uncentered, unscaled design at arbitrary new coordinates,
    /// preserving the training centres, range and whitening transform.
    pub fn design(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, MboostrsError> {
        checked_coordinates(x)?;
        checked_coordinates(self.centres.view())?;
        let k = self.centres.nrows();
        if k == 0
            || x.ncols() != self.centres.ncols()
            || self.whitening.dim() != (k, k)
            || self.whitening.iter().any(|v| !v.is_finite())
            || !self.theta.is_finite()
            || self.theta <= 0.0
        {
            return Err(MboostrsError::Shape(
                "invalid brad prediction recipe or coordinate dimensions".into(),
            ));
        }
        let covariance = Array2::from_shape_fn((x.nrows(), k), |(i, j)| {
            matern32(
                squared_distance(x.row(i), self.centres.row(j)).sqrt(),
                self.theta,
            )
        });
        let design = covariance.dot(&self.whitening);
        if design.iter().any(|v| !v.is_finite()) {
            return Err(MboostrsError::Solver("brad design is non-finite".into()));
        }
        Ok(design)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_basis_replays_and_whitens_its_centres() {
        let x = ndarray::arr2(&[[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., 0.]]);
        let recipe = RadialBasis::new(x.view(), 100, Some(0.5)).unwrap();
        assert_eq!(recipe.centres.nrows(), 4);
        let design = recipe.design(recipe.centres.view()).unwrap();
        let reproduced = design.dot(&design.t());
        for i in 0..4 {
            for j in 0..4 {
                let expected = matern32(
                    squared_distance(recipe.centres.row(i), recipe.centres.row(j)).sqrt(),
                    0.5,
                );
                assert!((reproduced[[i, j]] - expected).abs() < 1e-12);
            }
        }
        let range = effective_range(x.view(), 1e-3).unwrap();
        assert!((matern32(2.0_f64.sqrt(), range) - 1e-3).abs() < 1e-12);
    }
}
