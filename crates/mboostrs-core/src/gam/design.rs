//! Designs and penalties of refit terms, built from a recorded recipe so the
//! same basis can be evaluated at the fitting rows and at new rows.
//!
//! A [`TermBasis`] is what a base learner records about its basis: the
//! boundary and knots of a B-spline, whether a linear term has an
//! intercept column, the levels of a random effect, or two such margins
//! combined into a row-wise tensor product. [`TermBasis::design`] evaluates
//! the raw (unconstrained) design at given variable values;
//! [`TermBasis::penalties`] returns the penalties on that raw basis. The
//! refit's sum-to-zero constraint is applied afterwards by
//! [`try_gam_refit`](super::try_gam_refit) and
//! [`block_effect`](super::block_effect).

use ndarray::{Array2, ArrayView1};

use crate::learners::{
    bspline_basis_extrap, bspline_knots, cyclic_bspline_basis_from_knots, cyclic_bspline_knots,
    cyclic_diff_penalty, diff_penalty,
};
use crate::MboostrsError;

/// The recipe of one term's basis.
#[derive(Clone, Debug, PartialEq)]
pub enum TermBasis {
    /// Cubic (or `degree`) B-splines with `n_knots` equally spaced inner
    /// knots on `boundary`, extrapolated linearly outside it, and a
    /// difference penalty of order `differences`. When `cyclic`, the basis
    /// wraps at the boundary knots (periodic) and the penalty is the
    /// circulant difference penalty, mirroring the cyclic `bbs` learner.
    Bbs {
        /// Lower and upper boundary knots.
        boundary: (f64, f64),
        /// Number of inner knots.
        n_knots: usize,
        /// Spline degree.
        degree: usize,
        /// Order of the difference penalty.
        differences: usize,
        /// Whether the basis is cyclic (wraps at the boundary knots).
        cyclic: bool,
    },
    /// A linear term: the column `x`, with a leading column of ones when
    /// `intercept`. Unpenalized, or with a ridge penalty over every column
    /// when `ridge` (R `bols` with a `lambda`).
    Bols {
        /// Whether the design has a column of ones.
        intercept: bool,
        /// Whether the design carries a ridge penalty.
        ridge: bool,
    },
    /// Treatment-coded indicators of a factor used as a `by=` margin: one
    /// column per recorded level except the first (the reference), with a
    /// ridge penalty.
    ByFactor {
        /// The level codes seen at fitting time; the first is the reference.
        levels: Vec<f64>,
    },
    /// One indicator column per level, with a ridge penalty (a Gaussian
    /// random effect).
    Brandom {
        /// The level codes seen at fitting time, in column order.
        levels: Vec<f64>,
    },
    /// Row-wise tensor product of two margins; one penalty per margin
    /// penalty, `P_l (x) I` and `I (x) P_r` (mgcv's `te()` construction).
    Tensor(Box<TermBasis>, Box<TermBasis>),
}

impl TermBasis {
    /// How many variables the basis reads (a tensor reads its margins').
    pub fn n_vars(&self) -> usize {
        match self {
            TermBasis::Tensor(l, r) => l.n_vars() + r.n_vars(),
            _ => 1,
        }
    }

    /// Number of raw design columns.
    pub fn width(&self) -> usize {
        match self {
            TermBasis::Bbs {
                n_knots,
                degree,
                cyclic,
                ..
            } => {
                if *cyclic {
                    // The wrapped cyclic basis has one column per inner knot
                    // interval (n_knots + 1), not the n_knots + degree + 1 of
                    // the open basis; see cyclic_bspline_knots.
                    n_knots + 1
                } else {
                    n_knots + degree + 1
                }
            }
            TermBasis::Bols { intercept, .. } => 1 + usize::from(*intercept),
            TermBasis::ByFactor { levels } => levels.len().saturating_sub(1),
            TermBasis::Brandom { levels } => levels.len(),
            TermBasis::Tensor(l, r) => l.width() * r.width(),
        }
    }

    /// The raw design at the rows given by `vars` (one slice per variable,
    /// in the order the margins read them).
    ///
    /// A `Brandom` value that is not a recorded level gives an all-zero row
    /// when `unseen_zero`, and an error otherwise.
    pub fn design(
        &self,
        vars: &[ArrayView1<f64>],
        unseen_zero: bool,
    ) -> Result<Array2<f64>, MboostrsError> {
        if vars.len() != self.n_vars() {
            return Err(MboostrsError::Shape(format!(
                "the term reads {} variable(s); got {}",
                self.n_vars(),
                vars.len()
            )));
        }
        let n = vars[0].len();
        if vars.iter().any(|v| v.len() != n) {
            return Err(MboostrsError::Shape(
                "the term's variables have different lengths".into(),
            ));
        }
        if vars.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
            return Err(MboostrsError::Control(
                "a refit term's variable has a non-finite value".into(),
            ));
        }
        match self {
            TermBasis::Bbs {
                boundary,
                n_knots,
                degree,
                cyclic,
                ..
            } => {
                if boundary.0.is_nan() || boundary.1.is_nan() || boundary.0 >= boundary.1 {
                    return Err(MboostrsError::Control(format!(
                        "boundary knots {boundary:?} must be increasing"
                    )));
                }
                if *cyclic {
                    let knots = cyclic_bspline_knots(*boundary, *n_knots, *degree);
                    let x = vars[0].to_vec();
                    Ok(cyclic_bspline_basis_from_knots(
                        &x, &knots, *degree, *boundary,
                    ))
                } else {
                    let knots = bspline_knots(*boundary, *n_knots, *degree);
                    let x = vars[0].to_vec();
                    Ok(bspline_basis_extrap(&x, &knots, *degree, *boundary))
                }
            }
            TermBasis::ByFactor { levels } => {
                let full = TermBasis::Brandom {
                    levels: levels.clone(),
                }
                .design(vars, unseen_zero)?;
                Ok(full.slice(ndarray::s![.., 1..]).to_owned())
            }
            TermBasis::Bols { intercept, .. } => {
                let x = vars[0];
                Ok(if *intercept {
                    Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] })
                } else {
                    Array2::from_shape_fn((n, 1), |(i, _)| x[i])
                })
            }
            TermBasis::Brandom { levels } => {
                let mut design = Array2::<f64>::zeros((n, levels.len()));
                for (i, &value) in vars[0].iter().enumerate() {
                    match levels.iter().position(|&l| l == value) {
                        Some(j) => design[[i, j]] = 1.0,
                        None if unseen_zero => {}
                        None => {
                            return Err(MboostrsError::Control(format!(
                                "level {value} was not in the data the refit was fitted \
                                 on; pass unseen_levels='zero_effect' to predict the \
                                 average group"
                            )))
                        }
                    }
                }
                Ok(design)
            }
            TermBasis::Tensor(left, right) => {
                let k = left.n_vars();
                let a = left.design(&vars[..k], unseen_zero)?;
                let b = right.design(&vars[k..], unseen_zero)?;
                Ok(row_tensor(a.view(), b.view()))
            }
        }
    }

    /// Penalties on the raw basis. An unpenalized basis has none.
    pub fn penalties(&self) -> Vec<Array2<f64>> {
        match self {
            TermBasis::Bbs {
                differences,
                cyclic,
                ..
            } => {
                if *cyclic {
                    vec![cyclic_diff_penalty(self.width(), *differences)]
                } else {
                    vec![diff_penalty(self.width(), *differences)]
                }
            }
            TermBasis::Bols { ridge, .. } => {
                if *ridge {
                    vec![Array2::eye(self.width())]
                } else {
                    Vec::new()
                }
            }
            TermBasis::ByFactor { .. } => vec![Array2::eye(self.width())],
            TermBasis::Brandom { levels } => vec![Array2::eye(levels.len())],
            TermBasis::Tensor(left, right) => {
                let (qa, qb) = (left.width(), right.width());
                let mut out = Vec::new();
                for p in left.penalties() {
                    out.push(kron(&p, &Array2::eye(qb)));
                }
                for p in right.penalties() {
                    out.push(kron(&Array2::eye(qa), &p));
                }
                out
            }
        }
    }
}

/// Row-wise (Khatri-Rao) tensor product: row `i` is the Kronecker product
/// of row `i` of `a` and row `i` of `b`.
pub fn row_tensor(a: ndarray::ArrayView2<f64>, b: ndarray::ArrayView2<f64>) -> Array2<f64> {
    let (n, qa) = a.dim();
    let qb = b.ncols();
    let mut out = Array2::<f64>::zeros((n, qa * qb));
    for ((mut row, ra), rb) in out.rows_mut().into_iter().zip(a.rows()).zip(b.rows()) {
        for (ja, &va) in ra.iter().enumerate() {
            let base = ja * qb;
            for (jb, &vb) in rb.iter().enumerate() {
                row[base + jb] = va * vb;
            }
        }
    }
    out
}

/// Kronecker product `a (x) b`.
pub fn kron(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (r1, c1) = a.dim();
    let (r2, c2) = b.dim();
    Array2::from_shape_fn((r1 * r2, c1 * c2), |(i, j)| {
        a[[i / r2, j / c2]] * b[[i % r2, j % c2]]
    })
}
