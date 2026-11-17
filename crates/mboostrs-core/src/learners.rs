//! Base learners for component-wise gradient boosting.

use faer::Mat;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Ix1, Ix2};
use std::sync::{Arc, OnceLock};

use crate::linalg::{cho_solve_inplace, cho_solve_inplace_scratch_len, CachedCholesky, GramSolver};
use crate::MboostrsError;

/// Metadata to rebuild the x1 factor of a Kronecker design from raw x values.
///
/// For Multinomial models, each base-learner is wrapped as KroneckerBols with
/// x2 = I_{K-1}.  When predicting on new data, we need to rebuild x1 from the
/// raw input x values using the same B-spline knots / bols structure that was
/// used during fitting.
#[derive(Clone, Debug)]
pub enum KroneckerRebuild {
    /// bols(x, intercept=True): x1 = [1, x]  (2 columns)
    BolsIntercept,
    /// `bols(x, intercept=False)`: `x1 = [x]` (1 column)
    BolsNoIntercept,
    /// brandom(g) or a categorical bols(g): one indicator column per level
    /// code, minus the fit-time column means. A code the fit never saw gets a
    /// zero row, the average level's effect.
    Levels {
        /// Sorted integer level codes, one per column.
        level_codes: Vec<i64>,
        /// Column means subtracted at fit time (zeros when uncentered).
        centering: Vec<f64>,
    },
    /// bbs(x, df=...): B-spline basis with stored knots/degree
    Bbs {
        /// Knot sequence used to rebuild the B-spline basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots for extrapolation.
        boundary_knots: (f64, f64),
        /// Derivative order requested for the rebuilt basis.
        deriv: usize,
        /// Spectral centering applied after the basis, as the row-major
        /// `(p, q)` transform and `q`; `None` for the raw basis.
        transform: Option<(Vec<f64>, usize)>,
    },
}

/// Per-learner metadata for rebuilding the design matrix on new data.
///
/// Stored alongside each `PreparedLearner` and used by `predict_newdata`
/// to avoid the expensive Python-to-Rust argument marshalling of the
/// legacy `predict_gamboost` path.
#[derive(Clone, Debug)]
pub enum BasisInfo {
    /// Scalar bols: predict via `coef * (by * x - centering)`.
    ///
    /// `centering` is the weighted mean of the design column as it was at fit
    /// time, so it already accounts for `by`. Replay must therefore apply `by`
    /// before subtracting it, not after.
    ScalarBols {
        /// Weighted mean of the (by-scaled) design column at fit time.
        centering: f64,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// A formula-level `bols` learner. With `intercept=true`, coefficients are
    /// `[intercept, slope]`; otherwise the single coefficient multiplies `x`.
    Bols {
        /// Whether the design is `[1, x]` (`true`) or the single column `[x]`.
        intercept: bool,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// Block bols (including centered B-spline via prepare_block_bols):
    /// predict via `design_new @ coef` where design_new is rebuilt from
    /// the original x values using the cached parameters.
    BlockBols {
        /// Per-column design means subtracted at replay.
        centering: Vec<f64>,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
        /// B-spline derivative order used when rebuilding the block design.
        deriv: usize,
    },
    /// Centered B-spline (bbs with center=true). Stores the knot sequence,
    /// degree, boundary knots, and the centering matrix C such that
    /// `centered_design = bspline_basis(x) · C`.
    BbsCentered {
        /// Knot sequence used to rebuild the B-spline basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots for extrapolation.
        boundary_knots: (f64, f64),
        /// Flat column-major centering matrix: C is `(p, r)` where `p` is the
        /// B-spline dimension and `r` is the number of non-null eigenvalues.
        centering_mat: Vec<f64>,
        /// Row count of the centering matrix (the raw B-spline dimension).
        centering_rows: usize,
        /// Column count of the centering matrix (the centered dimension).
        centering_cols: usize,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
        /// B-spline derivative order used when rebuilding the basis.
        deriv: usize,
    },
    /// Uncentered B-spline (bbs with center=false).
    BbsUncentered {
        /// Knot sequence used to rebuild the B-spline basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots for extrapolation.
        boundary_knots: (f64, f64),
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
        /// Whether the basis is cyclic (wraps at the boundary knots).
        cyclic: bool,
        /// B-spline derivative order used when rebuilding the basis.
        deriv: usize,
    },
    /// Unconstrained B-spline basis with an identity ridge penalty. Unlike
    /// `BbsCentered`, Bns centering is the fold-weighted column mean and is
    /// therefore produced during `prepare(weights)`.
    Bns {
        /// Knot sequence used to rebuild the B-spline basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots for extrapolation.
        boundary_knots: (f64, f64),
        /// Per-column fold-weighted means subtracted at replay.
        centering: Vec<f64>,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// Monotone P-spline (bmono, always uncentered).
    BmonoUncentered {
        /// Knot sequence used to rebuild the B-spline basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots for extrapolation.
        boundary_knots: (f64, f64),
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// Monotone P-spline with spectral centering (bmono with center=TRUE).
    /// Stores the centering matrix C such that `centered_design = bspline_basis(x) · C`.
    BmonoCentered {
        /// Knot sequence used to rebuild the B-spline basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots for extrapolation.
        boundary_knots: (f64, f64),
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
        /// Flat column-major centering matrix C (`centered = basis · C`).
        centering_mat: Vec<f64>,
        /// Row count of the centering matrix (the raw B-spline dimension).
        centering_rows: usize,
        /// Column count of the centering matrix (the centered dimension).
        centering_cols: usize,
    },
    /// Tree learner. New-data values are packed row-major with one row per
    /// observation and `n_features` columns.
    Btree {
        /// Number of design columns; new-data values are packed row-major.
        n_features: usize,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
        /// Level count of each categorical column (codes 0..L-1), 0 for a
        /// numeric one; empty when no column is categorical. Read by the
        /// `Average` unseen-level policy.
        categorical: Vec<usize>,
    },
    /// Adaptive hinge learner (`bhinge`). Coefficients are one encoded
    /// selection per boosting step, `[knot, direction, coefficient,
    /// centering]`: `direction` is +1 for `max(0, x - t)` and -1 for
    /// `max(0, t - x)`, and `centering` is the weight-centering constant the
    /// fit subtracted from the (`by`-scaled) hinge column. Replay evaluates
    /// `coefficient * (by * hinge(x) - centering)` per step and sums.
    Bhinge {
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// Markov-random-field indicator basis. `transform` is the optional
    /// spectral map from level indicators into the non-null penalty space.
    Bmrf {
        /// Number of region levels.
        n_levels: usize,
        /// Optional spectral map from level indicators into the non-null
        /// penalty space, as `(matrix, columns)`.
        transform: Option<(Vec<f64>, usize)>,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// Kronecker bols (%O%) — stores the two factor designs.
    KroneckerBols {
        /// Per-column means of the x1 factor design.
        x1_centering: Vec<f64>,
        /// Per-column means of the x2 factor design.
        x2_centering: Vec<f64>,
    },
    /// Kronecker bbs (Multinomial `bbs(x) %O% bols(class)`). Stores the raw
    /// B-spline recipe so new-data replay can rebuild the x1 factor the same
    /// way [`prepare_kronecker_bbs`] does.
    KroneckerBbs {
        /// Knot sequence for the raw B-spline basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots.
        boundary_knots: (f64, f64),
        /// Derivative order for the rebuilt basis.
        deriv: usize,
    },
    /// Bspatial — 2D B-spline.
    Bspatial {
        /// Knot sequence for the first marginal B-spline.
        knots1: Vec<f64>,
        /// Knot sequence for the second marginal B-spline.
        knots2: Vec<f64>,
        /// Spline degree (shared by both marginals).
        degree: usize,
        /// Boundary knots for the first marginal.
        boundary_knots1: (f64, f64),
        /// Boundary knots for the second marginal.
        boundary_knots2: (f64, f64),
        /// Optional spectral centering as `(transform, rows, columns)`.
        centering_mat: Option<(Vec<f64>, usize, usize)>,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// Random-intercept indicator basis. The sorted integer codes define the
    /// coefficient-column mapping. Unknown or non-integral levels are rejected
    /// during replay instead of being silently mapped.
    Brandom {
        /// Sorted integer level codes mapping observations to coefficient columns.
        level_codes: Vec<i64>,
        /// Per-level means subtracted at replay.
        centering: Vec<f64>,
        /// Optional varying-coefficient multiplier column.
        by: Option<Vec<f64>>,
    },
    /// Explicit kernel-ridge learner. Coefficients index the training rows;
    /// new-data replay builds the cross-kernel against `training_x` and then
    /// applies the fold-specific column centering used during preparation.
    Bkernel {
        /// Training covariate rows; coefficients index them.
        training_x: Vec<f64>,
        /// Kernel bandwidth used for the cross-kernel.
        bandwidth: f64,
        /// Kernel name: "gaussian", "epanechnikov", or "uniform".
        kernel_type: String,
        /// Per-training-row column centering applied at fit time.
        centering: Vec<f64>,
    },
    /// Nyström-approximated kernel learner. Coefficients index the landmarks
    /// rather than the training rows, and the cross-kernel is mapped through
    /// the same `L^-T` the training design used -- so the replay needs the
    /// landmarks and that factor, not the training covariate.
    BkernelNystrom {
        /// Landmark positions; coefficients index them.
        landmarks: Vec<f64>,
        /// Lower-triangular Cholesky factor of `K_mm`, column-major.
        cholesky_l: Vec<f64>,
        /// Kernel bandwidth used for the cross-kernel.
        bandwidth: f64,
        /// Kernel name: "gaussian", "epanechnikov", or "uniform".
        kernel_type: String,
        /// Per-column centering applied at fit time.
        centering: Vec<f64>,
    },
    /// Caller-supplied design learner. New-data values are a row-major
    /// `(n, n_features)` matrix supplied explicitly by the frontend.
    Buser {
        /// Number of design columns in the caller-supplied matrix.
        n_features: usize,
        /// Per-column design means subtracted at replay.
        centering: Vec<f64>,
        /// Varying-coefficient multiplier applied before subtracting the means.
        by: Option<Vec<f64>>,
    },
    /// Replay a one-dimensional binned learner by snapping in-range values to
    /// the training bin representative, then delegate to its ordinary basis
    /// recipe. Values outside the training range remain unsnapped so the
    /// underlying learner's established extrapolation rule still applies.
    Binned {
        /// Training bin boundaries; in-range values are snapped to the bin.
        cutpoints: Arc<[f64]>,
        /// Per-bin representative values (bin centers).
        values: Arc<[f64]>,
        /// Lower extrapolation bound of the binned range.
        lower: f64,
        /// Upper extrapolation bound of the binned range.
        upper: f64,
        /// Inner basis recipe applied after snapping.
        inner: Box<BasisInfo>,
    },
    /// Two-dimensional occupied-cell replay for `bspatial`.
    BinnedSpatial {
        /// Binning along the first spatial marginal.
        marginal1: QuantileBinning,
        /// Binning along the second spatial marginal.
        marginal2: QuantileBinning,
        /// Occupied-cell lookup table, flattened row-major over the grid.
        cell_for_pair: Arc<[i32]>,
        /// First-marginal representative value for each occupied cell.
        cell_values1: Arc<[f64]>,
        /// Second-marginal representative value for each occupied cell.
        cell_values2: Arc<[f64]>,
        /// Inner basis recipe applied at occupied-cell representatives.
        inner: Box<BasisInfo>,
    },
    /// This learner is handled by another prediction path or is unsupported by
    /// raw one-slice basis reconstruction.
    Unsupported {
        /// Learner kind that cannot be rebuilt from raw inputs.
        kind: String,
        /// Human-readable reason the learner is unsupported.
        reason: String,
    },
}

/// What a `Brandom` replay does with a level code the fit never saw.
///
/// Grouped cross-validation produces such codes by construction: every
/// held-out group is absent from its fold's training rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnseenLevels {
    /// Refuse the row with [`MboostrsError::UnseenLevel`].
    #[default]
    Error,
    /// Give the level a zero coefficient, the mean the ridge penalty shrinks
    /// every level towards. The row receives what an all-zero indicator row
    /// receives: nothing from an uncentered term, and minus the centering
    /// constant from a centered one. A fit on the full frame in which the
    /// level's rows carry zero weight estimates exactly that zero
    /// coefficient, so this policy reproduces the in-bag-weights route that
    /// `cvrisk` takes.
    ZeroEffect,
    /// Predict the level as an average group: its coefficient is the mean of
    /// the term's level coefficients. For a random intercept that mean is
    /// usually near zero; for a random slope (`by=`) it is the part of the
    /// average slope the group term absorbed, which a zero coefficient would
    /// drop from every new group's prediction.
    Average,
}

impl BasisInfo {
    /// Predict using the given coefficient vector, writing into `out`.
    /// `x` is the raw input slice for this learner (e.g. B-spline x values
    /// for a single-column learner).
    ///
    /// Returns [`MboostrsError`] (with [`MboostrsError::Shape`] variants)
    /// when the supplied coefficient vector, raw input, or level codes do not
    /// match the stored basis, rather than mixing a bare `String` into the
    /// crate's typed error surface. A level code the fit never saw is refused
    /// with [`MboostrsError::UnseenLevel`]; [`BasisInfo::predict_into_with`]
    /// takes a policy for it.
    pub fn predict_into(
        &self,
        coef: &[f64],
        x: &[f64],
        out: &mut [f64],
        nu: f64,
        by_override: Option<&[f64]>,
    ) -> Result<(), MboostrsError> {
        self.predict_into_with(coef, x, out, nu, by_override, UnseenLevels::Error)
    }

    /// [`BasisInfo::predict_into`] with an explicit policy for level codes
    /// the fit never saw. Only `Brandom` (exact or binned) reads `unseen`.
    pub fn predict_into_with(
        &self,
        coef: &[f64],
        x: &[f64],
        out: &mut [f64],
        nu: f64,
        by_override: Option<&[f64]>,
        unseen: UnseenLevels,
    ) -> Result<(), MboostrsError> {
        let n = out.len();
        debug_assert!(
            x.len() >= n,
            "predict_into: x.len() {} < out.len() {}",
            x.len(),
            n
        );
        debug_assert!(!coef.is_empty(), "predict_into: coef is empty");
        match self {
            BasisInfo::Binned {
                cutpoints,
                values,
                lower,
                upper,
                inner,
            } => {
                let mut snapped = Vec::with_capacity(n);
                for &value in &x[..n] {
                    if value < *lower || value > *upper {
                        snapped.push(value);
                    } else {
                        let bin = cutpoints.partition_point(|cutpoint| value > *cutpoint);
                        snapped.push(values[bin]);
                    }
                }
                inner.predict_into_with(coef, &snapped, out, nu, by_override, unseen)
            }
            BasisInfo::BinnedSpatial {
                marginal1,
                marginal2,
                cell_for_pair,
                cell_values1,
                cell_values2,
                inner,
            } => {
                if x.len() != 2 * n {
                    return Err(MboostrsError::Shape(format!(
                        "binned spatial replay requires packed [x1, x2] data of length {}; got {}",
                        2 * n,
                        x.len()
                    )));
                }
                let mut snapped = vec![0.0f64; 2 * n];
                for row in 0..n {
                    let left = x[row];
                    let right = x[n + row];
                    let inside_left = left >= marginal1.lower && left <= marginal1.upper;
                    let inside_right = right >= marginal2.lower && right <= marginal2.upper;
                    let left_bin = marginal1
                        .cutpoints
                        .partition_point(|cutpoint| left > *cutpoint);
                    let right_bin = marginal2
                        .cutpoints
                        .partition_point(|cutpoint| right > *cutpoint);
                    let cell = if inside_left && inside_right {
                        cell_for_pair[left_bin * marginal2.effective + right_bin]
                    } else {
                        -1
                    };
                    if cell >= 0 {
                        snapped[row] = cell_values1[cell as usize];
                        snapped[n + row] = cell_values2[cell as usize];
                    } else {
                        snapped[row] = if inside_left {
                            marginal1.values[left_bin]
                        } else {
                            left
                        };
                        snapped[n + row] = if inside_right {
                            marginal2.values[right_bin]
                        } else {
                            right
                        };
                    }
                }
                inner.predict_into_with(coef, &snapped, out, nu, by_override, unseen)
            }
            BasisInfo::ScalarBols { centering, by } => {
                let c = coef[0];
                let scale = by_override.or(by.as_deref());
                for i in 0..n {
                    let value = scale.map_or(x[i], |values| x[i] * values[i]);
                    out[i] += nu * c * (value - centering);
                }
                Ok(())
            }
            BasisInfo::Bols { intercept, by } => {
                for i in 0..n {
                    let value = if *intercept {
                        coef[0] + coef[1] * x[i]
                    } else {
                        coef[0] * x[i]
                    };
                    let scale = by_override
                        .or(by.as_deref())
                        .map_or(1.0, |values| values[i]);
                    out[i] += nu * scale * value;
                }
                Ok(())
            }
            BasisInfo::BlockBols { centering, by, .. } => {
                let p = centering.len();
                for i in 0..n {
                    let mut s = 0.0;
                    for j in 0..p {
                        s += coef[j] * (x[i * p + j] - centering[j]);
                    }
                    let scale = by_override.or(by.as_deref()).map_or(1.0, |b| b[i]);
                    out[i] += nu * scale * s;
                }
                Ok(())
            }
            BasisInfo::BbsCentered {
                knots,
                degree,
                boundary_knots,
                centering_mat,
                centering_rows,
                centering_cols,
                by,
                deriv,
            } => {
                // p (knots.len() - degree - 1) equals *centering_rows
                let basis = if *deriv > 0 {
                    bspline_basis_with_deriv(x, knots, *degree, *deriv)
                } else {
                    bspline_basis_extrap(x, knots, *degree, *boundary_knots)
                };
                // Apply centering: centered = basis · C
                let r = *centering_cols;
                let cr = *centering_rows;
                let mut centered = vec![0.0f64; n * r];
                for i in 0..n {
                    for k in 0..r {
                        let mut s = 0.0;
                        for j in 0..cr {
                            s += basis[[i, j]] * centering_mat[k * cr + j];
                        }
                        centered[i * r + k] = s;
                    }
                }
                // Multiply by coefficient vector
                for i in 0..n {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += coef[k] * centered[i * r + k];
                    }
                    let scale = by_override.or(by.as_deref()).map_or(1.0, |b| b[i]);
                    out[i] += nu * scale * s;
                }
                Ok(())
            }
            BasisInfo::BbsUncentered {
                knots,
                degree,
                boundary_knots,
                by,
                cyclic,
                deriv,
            } => {
                let basis = if *cyclic {
                    cyclic_bspline_basis_from_knots(x, knots, *degree, *boundary_knots)
                } else if *deriv > 0 {
                    bspline_basis_with_deriv(x, knots, *degree, *deriv)
                } else {
                    bspline_basis_extrap(x, knots, *degree, *boundary_knots)
                };
                let p = basis.ncols();
                for i in 0..n {
                    let mut s = 0.0;
                    for j in 0..p {
                        s += coef[j] * basis[[i, j]];
                    }
                    let scale = by_override.or(by.as_deref()).map_or(1.0, |b| b[i]);
                    out[i] += nu * scale * s;
                }
                Ok(())
            }
            BasisInfo::Bns {
                knots,
                degree,
                boundary_knots,
                centering,
                by,
            } => {
                let basis = bspline_basis_extrap(x, knots, *degree, *boundary_knots);
                let p = basis.ncols();
                for i in 0..n {
                    let scale = by_override
                        .or(by.as_deref())
                        .map_or(1.0, |values| values[i]);
                    let mut value = 0.0;
                    for j in 0..p {
                        value += coef[j] * (scale * basis[[i, j]] - centering[j]);
                    }
                    out[i] += nu * value;
                }
                Ok(())
            }
            BasisInfo::BmonoUncentered {
                knots,
                degree,
                boundary_knots,
                by,
            } => {
                let basis = bspline_basis_extrap(x, knots, *degree, *boundary_knots);
                let p = basis.ncols();
                for i in 0..n {
                    let mut s = 0.0;
                    for j in 0..p {
                        s += coef[j] * basis[[i, j]];
                    }
                    let scale = by_override.or(by.as_deref()).map_or(1.0, |b| b[i]);
                    out[i] += nu * scale * s;
                }
                Ok(())
            }
            BasisInfo::BmonoCentered {
                knots,
                degree,
                boundary_knots,
                by,
                centering_mat,
                centering_rows,
                centering_cols,
            } => {
                let basis = bspline_basis_extrap(x, knots, *degree, *boundary_knots);
                let cr = *centering_rows;
                let cc = *centering_cols;
                let mut centered = vec![0.0f64; n * cc];
                for i in 0..n {
                    for k in 0..cc {
                        let mut s = 0.0;
                        for j in 0..cr {
                            s += basis[[i, j]] * centering_mat[k * cr + j];
                        }
                        centered[i * cc + k] = s;
                    }
                }
                for i in 0..n {
                    let mut s = 0.0;
                    for k in 0..cc {
                        s += coef[k] * centered[i * cc + k];
                    }
                    let scale = by_override.or(by.as_deref()).map_or(1.0, |b| b[i]);
                    out[i] += nu * scale * s;
                }
                Ok(())
            }
            BasisInfo::Btree {
                n_features,
                by,
                categorical,
            } => {
                if x.len() != n * *n_features {
                    return Err(MboostrsError::Shape(format!(
                        "Btree replay requires a row-major design of length {}; got {}",
                        n * *n_features,
                        x.len()
                    )));
                }
                let design = ArrayView2::from_shape((n, *n_features), x).map_err(|error| {
                    MboostrsError::Shape(format!("invalid Btree replay design: {error}"))
                })?;
                let tree = FittedTree::from_coef_vec(ArrayView1::from(coef));
                let prediction =
                    if matches!(unseen, UnseenLevels::Average) && !categorical.is_empty() {
                        Array1::from_shape_fn(n, |i| {
                            tree.predict_row_level_average(design.row(i), categorical)
                        })
                    } else {
                        tree.predict(design)
                    };
                for row in 0..n {
                    let scale = by_override
                        .or(by.as_deref())
                        .map_or(1.0, |values| values[row]);
                    out[row] += nu * scale * prediction[row];
                }
                Ok(())
            }
            BasisInfo::Bhinge { by } => {
                if coef.len() != 4 {
                    return Err(MboostrsError::Shape(format!(
                        "Bhinge replay requires a 4-element [knot, direction, coefficient, centering] vector; got {}",
                        coef.len()
                    )));
                }
                let (knot, direction, coefficient, centering) =
                    (coef[0], coef[1], coef[2], coef[3]);
                let scale = by_override.or(by.as_deref());
                for i in 0..n {
                    let hinge = if direction > 0.0 {
                        (x[i] - knot).max(0.0)
                    } else {
                        (knot - x[i]).max(0.0)
                    };
                    let column = scale.map_or(hinge, |values| hinge * values[i]);
                    out[i] += nu * coefficient * (column - centering);
                }
                Ok(())
            }
            BasisInfo::Bmrf {
                n_levels,
                transform,
                by,
            } => {
                let expected = transform
                    .as_ref()
                    .map_or(*n_levels, |(_, columns)| *columns);
                if coef.len() != expected {
                    return Err(MboostrsError::Shape(format!(
                        "Bmrf coefficient length {} does not match replay dimension {expected}",
                        coef.len()
                    )));
                }
                for i in 0..n {
                    let code = x[i];
                    if !code.is_finite() || code.fract() != 0.0 {
                        return Err(MboostrsError::Shape(format!(
                            "Bmrf level at observation {i} must be a finite integer; got {code}"
                        )));
                    }
                    if code < 0.0 || code >= *n_levels as f64 {
                        return Err(MboostrsError::Shape(format!(
                            "unknown Bmrf level {code} at observation {i}; expected an integer code in 0..{}",
                            n_levels - 1
                        )));
                    }
                    let level = code as usize;
                    let value = match transform {
                        Some((matrix, columns)) => {
                            let mut result = 0.0;
                            for column in 0..*columns {
                                result += coef[column] * matrix[column * *n_levels + level];
                            }
                            result
                        }
                        None => coef[level],
                    };
                    let scale = by_override
                        .or(by.as_deref())
                        .map_or(1.0, |values| values[i]);
                    out[i] += nu * scale * value;
                }
                Ok(())
            }
            BasisInfo::KroneckerBols { .. } => Err(MboostrsError::Shape(
                "KroneckerBols prediction uses multiple factor designs".to_string(),
            )),
            BasisInfo::KroneckerBbs { .. } => Err(MboostrsError::Shape(
                "KroneckerBbs prediction uses the multiple-factor Kronecker path".to_string(),
            )),
            BasisInfo::Bspatial {
                knots1,
                knots2,
                degree,
                boundary_knots1,
                boundary_knots2,
                centering_mat,
                by,
            } => {
                if x.len() != 2 * n {
                    return Err(MboostrsError::Shape(format!(
                        "Bspatial replay requires packed [x1, x2] data of length {}; got {}",
                        2 * n,
                        x.len()
                    )));
                }
                let b1 = bspline_basis_extrap(&x[..n], knots1, *degree, *boundary_knots1);
                let b2 = bspline_basis_extrap(&x[n..], knots2, *degree, *boundary_knots2);
                let p1 = b1.ncols();
                let p2 = b2.ncols();
                let raw_columns = p1 * p2;
                let replay_columns = centering_mat
                    .as_ref()
                    .map_or(raw_columns, |(_, _, columns)| *columns);
                if coef.len() != replay_columns {
                    return Err(MboostrsError::Shape(format!(
                        "Bspatial coefficient length {} does not match replay dimension {replay_columns}",
                        coef.len()
                    )));
                }
                for row in 0..n {
                    let mut value = 0.0;
                    match centering_mat {
                        Some((transform, rows, columns)) => {
                            if *rows != raw_columns || *columns != replay_columns {
                                return Err(MboostrsError::Shape(
                                    "invalid cached Bspatial centering transform".to_string(),
                                ));
                            }
                            for column in 0..*columns {
                                let mut centered_value = 0.0;
                                for column1 in 0..p1 {
                                    for column2 in 0..p2 {
                                        let raw_column = column1 * p2 + column2;
                                        centered_value += b1[[row, column1]]
                                            * b2[[row, column2]]
                                            * transform[column * *rows + raw_column];
                                    }
                                }
                                value += coef[column] * centered_value;
                            }
                        }
                        None => {
                            for column1 in 0..p1 {
                                for column2 in 0..p2 {
                                    value += coef[column1 * p2 + column2]
                                        * b1[[row, column1]]
                                        * b2[[row, column2]];
                                }
                            }
                        }
                    }
                    let scale = by_override
                        .or(by.as_deref())
                        .map_or(1.0, |values| values[row]);
                    out[row] += nu * scale * value;
                }
                Ok(())
            }
            BasisInfo::Brandom {
                level_codes,
                centering,
                by,
            } => {
                let n_levels = level_codes.len();
                if coef.len() != n_levels || centering.len() != n_levels {
                    return Err(MboostrsError::Shape(format!(
                        "Brandom coefficient/centering length does not match {n_levels} levels"
                    )));
                }
                let centered_constant: f64 = coef
                    .iter()
                    .zip(centering.iter())
                    .map(|(coefficient, mean)| coefficient * mean)
                    .sum();
                let average_coefficient = coef.iter().sum::<f64>() / n_levels as f64;
                for i in 0..n {
                    let code = x[i];
                    if !code.is_finite() || code.fract() != 0.0 {
                        return Err(MboostrsError::Shape(format!(
                            "Brandom level at observation {i} must be a finite integer; got {code}"
                        )));
                    }
                    let level = match level_codes.binary_search(&(code as i64)) {
                        Ok(level) => level,
                        Err(_) => match unseen {
                            UnseenLevels::Error => {
                                return Err(MboostrsError::UnseenLevel {
                                    learner: "Brandom",
                                    code,
                                    observation: i,
                                })
                            }
                            UnseenLevels::ZeroEffect => {
                                out[i] -= nu * centered_constant;
                                continue;
                            }
                            UnseenLevels::Average => {
                                let scale = by_override
                                    .or(by.as_deref())
                                    .map_or(1.0, |values| values[i]);
                                out[i] += nu * (scale * average_coefficient - centered_constant);
                                continue;
                            }
                        },
                    };
                    let scale = by_override
                        .or(by.as_deref())
                        .map_or(1.0, |values| values[i]);
                    out[i] += nu * (scale * coef[level] - centered_constant);
                }
                Ok(())
            }
            BasisInfo::Bkernel {
                training_x,
                bandwidth,
                kernel_type,
                centering,
            } => {
                let p = training_x.len();
                if coef.len() != p || centering.len() != p {
                    return Err(MboostrsError::Shape(format!(
                        "Bkernel coefficient/centering length does not match {p} training rows"
                    )));
                }
                let kernel: fn(f64, f64) -> f64 = match kernel_type.as_str() {
                    "gaussian" => kernel_gaussian,
                    "epanechnikov" => kernel_epanechnikov,
                    "uniform" => kernel_uniform,
                    other => {
                        return Err(MboostrsError::Shape(format!(
                            "unknown Bkernel type {other:?}"
                        )))
                    }
                };
                for row in 0..n {
                    let mut value = 0.0;
                    for column in 0..p {
                        let basis_value = kernel((x[row] - training_x[column]).abs(), *bandwidth);
                        value += coef[column] * (basis_value - centering[column]);
                    }
                    out[row] += nu * value;
                }
                Ok(())
            }
            BasisInfo::BkernelNystrom {
                landmarks,
                cholesky_l,
                bandwidth,
                kernel_type,
                centering,
            } => {
                let m = landmarks.len();
                if coef.len() != m || centering.len() != m || cholesky_l.len() != m * m {
                    return Err(MboostrsError::Shape(format!(
                        "BkernelNystrom coefficient/centering/factor length does not \
                         match {m} landmarks"
                    )));
                }
                let kernel: fn(f64, f64) -> f64 = match kernel_type.as_str() {
                    "gaussian" => kernel_gaussian,
                    "epanechnikov" => kernel_epanechnikov,
                    "uniform" => kernel_uniform,
                    other => {
                        return Err(MboostrsError::Shape(format!(
                            "unknown Bkernel type {other:?}"
                        )))
                    }
                };
                let mut row_basis = vec![0.0f64; m];
                for row in 0..n {
                    // `phi = L^-1 k`, the same forward substitution the
                    // training design used. Skipping it would score new data
                    // against an untransformed cross-kernel while the
                    // coefficients were fitted against a transformed one.
                    for (column, landmark) in landmarks.iter().enumerate() {
                        row_basis[column] = kernel((x[row] - landmark).abs(), *bandwidth);
                    }
                    for column in 0..m {
                        let mut value = row_basis[column];
                        for earlier in 0..column {
                            value -= cholesky_l[earlier * m + column] * row_basis[earlier];
                        }
                        row_basis[column] = value / cholesky_l[column * m + column];
                    }
                    let mut value = 0.0;
                    for column in 0..m {
                        value += coef[column] * (row_basis[column] - centering[column]);
                    }
                    out[row] += nu * value;
                }
                Ok(())
            }
            BasisInfo::Buser {
                n_features,
                centering,
                by,
            } => {
                if x.len() != n * *n_features {
                    return Err(MboostrsError::Shape(format!(
                        "Buser replay requires a row-major design of length {}; got {}",
                        n * *n_features,
                        x.len()
                    )));
                }
                if coef.len() != *n_features || centering.len() != *n_features {
                    return Err(MboostrsError::Shape(format!(
                        "Buser coefficient/centering length does not match {n_features} design columns"
                    )));
                }
                if by_override
                    .or(by.as_deref())
                    .is_some_and(|values| values.len() != n)
                {
                    return Err(MboostrsError::Shape(
                        "Buser by-values must align with prediction rows".into(),
                    ));
                }
                for row in 0..n {
                    let scale = by_override
                        .or(by.as_deref())
                        .map_or(1.0, |values| values[row]);
                    let mut value = 0.0;
                    for column in 0..*n_features {
                        value += coef[column]
                            * (scale * x[row * *n_features + column] - centering[column]);
                    }
                    out[row] += nu * value;
                }
                Ok(())
            }
            BasisInfo::Unsupported { kind, reason } => Err(MboostrsError::Shape(format!(
                "BasisInfo for {kind} is unsupported: {reason}"
            ))),
        }
    }
}

/// Weight-independent description of a base learner.
///
/// A blueprint owns the raw or precomputed design information that is stable
/// across fits. Calling [`LearnerBlueprint::prepare`] supplies the observation
/// weights and constructs the weight-dependent centering, smoothing parameter,
/// Gram factorization, and reusable work buffers. This is the common boundary
/// used by LSS fitting, cross-validation, and stability selection; those paths
/// must not reconstruct learner types independently.
#[derive(Clone, Debug)]
pub enum LearnerBlueprint {
    /// A `bols` learner over one covariate, in one of the two forms the
    /// frontends use for `bols(x, intercept = TRUE)`.
    ///
    /// `intercept = true` fits an explicit two-column block `[1, x]`, so the
    /// learner carries its own constant term. This is what `glmboostLSS` and
    /// `gamboostLSS` mean by an intercept: each distribution parameter's
    /// predictor gets one.
    ///
    /// `intercept = false` with `center = true` fits a single column, weight-
    /// centered, with the constant term left to the model offset. This is what
    /// `gamboost` means by it, and it is why a `gamboost` `bols` term has one
    /// coefficient where the LSS one has two.
    ///
    /// `intercept = false` with `center = false` is the raw uncentered column.
    ScalarBols {
        /// Raw covariate column.
        x: Array1<f64>,
        /// Formula name of the learner.
        name: String,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// Target effective degrees of freedom used to derive lambda.
        df: Option<f64>,
        /// Whether the single surviving column is weight-centered.
        center: bool,
        /// Whether the design is the two-column block `[1, x]`.
        intercept: bool,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
    },
    /// Exact discrete random-intercept learner: one occupied bin per factor
    /// level and an identity basis, avoiding the dense n×levels indicator.
    BinnedBrandom {
        /// Quantile binning shared by every resampling fold.
        binning: QuantileBinning,
        /// Compact basis evaluated at bin representatives, one column per level.
        basis: Arc<Array2<f64>>,
        /// Identity ridge penalty over the level basis.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// Whether the level columns are weight-centered.
        center: bool,
        /// Sorted integer level codes mapping observations to columns.
        level_codes: Vec<i64>,
    },
    /// Weight-independent compact representation for a binned, one-column
    /// `bols`. Bin membership and representatives are computed once from the
    /// full data and shared by every CV/stability-selection preparation.
    BinnedScalarBols {
        /// Quantile binning shared by every resampling fold.
        binning: QuantileBinning,
        /// Formula name of the learner.
        name: String,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// Target effective degrees of freedom used to derive lambda.
        df: Option<f64>,
        /// Whether the single surviving column is weight-centered.
        center: bool,
    },
    /// B-spline learner with a fold-invariant design and penalty template.
    Bbs {
        /// Fold-invariant B-spline design matrix.
        design: Array2<f64>,
        /// Difference ridge penalty on the B-spline coefficients.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// New-data replay recipe for this basis.
        basis_info: BasisInfo,
    },
    /// B-spline learner Kronecker-wrapped for the Multinomial family
    /// (`bbs(x, df) %O% bols(class, K-1)`). Unlike [`Self::Bbs`], it keeps the
    /// raw covariate and knots rather than a pre-built design, because
    /// [`prepare_kronecker_bbs`] reconstructs the **un-centred** raw basis and
    /// a per-class `I + K1` penalty that the ordinary `Bbs` recipe does not.
    KroneckerBbs {
        /// Raw covariate column.
        x: Array1<f64>,
        /// Formula name of the learner.
        name: String,
        /// Number of inner knots.
        n_knots: usize,
        /// B-spline degree.
        degree: usize,
        /// Order of the difference penalty.
        differences: usize,
        /// Target effective degrees of freedom per class block.
        df: f64,
        /// Number of non-reference classes (`K - 1`).
        n_resp: usize,
        /// Derivative order for the new-data replay recipe.
        deriv: usize,
        /// Spectrally centre the basis (`bbs(x, center=TRUE)`): fit only the
        /// deviation from the penalty's null space, with an identity penalty.
        center: bool,
    },
    /// Tensor-product spline evaluated at occupied quantile-grid cells.
    BinnedBspatial {
        /// Quantile grid binning shared by every resampling fold.
        grid: QuantileGridBinning,
        /// Compact tensor basis evaluated at occupied grid cells.
        basis: Arc<Array2<f64>>,
        /// Additive tensor ridge penalty on the surface coefficients.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// New-data replay recipe for this basis.
        basis_info: BasisInfo,
    },
    /// P-spline template evaluated only at occupied quantile-bin
    /// representatives. The observation index and compact basis are shared by
    /// every resampling fold.
    BinnedBbs {
        /// Quantile binning shared by every resampling fold.
        binning: QuantileBinning,
        /// Compact B-spline basis evaluated at bin representatives.
        basis: Arc<Array2<f64>>,
        /// Difference ridge penalty on the B-spline coefficients.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// New-data replay recipe for this basis.
        basis_info: BasisInfo,
    },
    /// Shape-constrained P-spline. The raw covariate and constraint options are
    /// stable, while df-to-lambda conversion and the constrained solver state
    /// are rebuilt from each fit or resample's weights.
    Bmono {
        /// Raw covariate column.
        x: Array1<f64>,
        /// Formula name of the learner.
        name: String,
        /// Number of inner knots.
        n_knots: usize,
        /// Spline degree.
        degree: usize,
        /// Penalty difference order.
        differences: usize,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// Monotonicity constraint applied per coefficient.
        constraint: MonoConstraint,
        /// Reusable constrained solver state.
        solver: BmonoSolver,
        /// Secondary penalty weight for the constraint term.
        lambda2: f64,
        /// Iteration cap for the constraint solver.
        niter: usize,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
        /// Explicit constraint matrix (Bconstrained); overrides `constraint`.
        constraint_matrix: Option<Array2<f64>>,
        /// Whether the constrained design is weight-centered.
        center: bool,
        /// New-data replay recipe for this basis.
        basis_info: BasisInfo,
    },
    /// Regression-tree learner over one or more columns. The design is stable;
    /// weights and tree work state are rebuilt for each fit or resample.
    Btree {
        /// Fold-invariant design matrix, one column per predictor.
        design: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Minimum observations per terminal node.
        minbucket: usize,
        /// Minimum observations per split attempt.
        minsplit: usize,
        /// Maximum tree depth.
        maxdepth: usize,
        /// Level count of each categorical column, 0 for a numeric one.
        categorical: Vec<usize>,
        /// How a node chooses the column it splits.
        selection: SplitSelection,
        /// How a node finds split points.
        split_method: SplitMethod,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
        /// New-data replay recipe for this basis.
        basis_info: BasisInfo,
    },
    /// Adaptive hinge (`bhinge`) learner over one covariate: at every fit
    /// step it scans candidate knots in both hinge directions and adds the
    /// single best hinge term, MARS-style. Unlike [`Self::Bbs`] there is no
    /// fold-invariant design to store — the basis is chosen per iteration —
    /// so the blueprint keeps the raw covariate and knot-grid size, the way
    /// [`Self::Bmono`] keeps a raw `x`.
    Bhinge {
        /// Raw covariate column.
        x: Array1<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom of one selected hinge
        /// coefficient. A one-column hinge fit can absorb at most 1 df, so a
        /// target above that (the default, reflecting the ~2 effective
        /// parameters a knot-plus-coefficient selection spends) yields an
        /// unpenalised fit whose complexity is governed by `mstop`/`nu` the
        /// way a tree's is.
        df: f64,
        /// Penalty weight for the hinge coefficient; overrides `df` when given.
        lambda: Option<f64>,
        /// Whether the selected hinge column is weight-centered.
        center: bool,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
        /// Number of candidate knots (interior quantiles of `x`) scanned per
        /// fit step.
        n_candidates: usize,
    },
    /// B-spline basis with an identity ridge penalty (`bns`). The raw basis is
    /// cached; weighted centering, lambda, and factorization are fold-specific.
    Bns {
        /// Fold-invariant B-spline design matrix.
        design: Array2<f64>,
        /// Identity ridge penalty on the B-spline coefficients.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Knot sequence of the basis.
        knots: Vec<f64>,
        /// Spline degree.
        degree: usize,
        /// Lower and upper boundary knots for extrapolation.
        boundary_knots: (f64, f64),
        /// Whether the design columns are weight-centered.
        center: bool,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
    },
    /// Random intercept represented by a full indicator matrix and ridge
    /// penalty. Indicator construction is cached; centering, df-to-lambda, and
    /// factorization are recomputed for each fit or resample.
    Brandom {
        /// Indicator design matrix, one column per level.
        design: Array2<f64>,
        /// Identity ridge penalty on the level indicator columns.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// Whether the indicator columns are weight-centered.
        center: bool,
        /// Sorted integer level codes mapping observations to columns.
        level_codes: Vec<i64>,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
    },
    /// Markov random field over integer-coded regions. The graph penalty and
    /// optional spectral transform are fold-invariant; only `df -> lambda` and
    /// the weighted Gram factorization are recomputed for each resample.
    Bmrf {
        /// Indicator design matrix, one column per region.
        design: Array2<f64>,
        /// Graph Laplacian penalty on the region coefficients.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// New-data replay recipe for this basis.
        basis_info: BasisInfo,
    },
    /// Tensor-product P-spline surface. The marginal bases, tensor design,
    /// additive penalty, and optional spectral map are invariant across
    /// resamples; only lambda selection and the weighted factorization vary.
    Bspatial {
        /// Tensor-product design matrix over the two marginal bases.
        design: Array2<f64>,
        /// Additive ridge penalty (sum of the two marginal penalties).
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// New-data replay recipe for this basis.
        basis_info: BasisInfo,
    },
    /// Explicit one-dimensional kernel-ridge learner. This is an mboostrs
    /// extension and deliberately does not claim the pathway/GWAS semantics
    /// of upstream extension packages named `bkernel`.
    Bkernel {
        /// Raw covariate column used to build the training kernel.
        x: Array1<f64>,
        /// Formula name of the learner.
        name: String,
        /// Kernel bandwidth.
        bandwidth: f64,
        /// Target effective degrees of freedom.
        df: f64,
        /// Kernel name: "gaussian", "epanechnikov", or "uniform".
        kernel_type: String,
        /// Whether the kernel columns are weight-centered.
        center: bool,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
    },
    /// Nyström-approximated kernel learner: the same kernel, represented
    /// against `landmarks` centres instead of every training row.
    BkernelNystrom {
        /// Raw covariate column.
        x: Array1<f64>,
        /// Formula name of the learner.
        name: String,
        /// Kernel bandwidth.
        bandwidth: f64,
        /// Target effective degrees of freedom.
        df: f64,
        /// Kernel name: "gaussian", "epanechnikov", or "uniform".
        kernel_type: String,
        /// How many landmark centres to represent the kernel against.
        landmarks: usize,
        /// Whether the columns are weight-centered.
        center: bool,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
    },
    /// Caller-owned design and penalty matrices. Observation resampling keeps
    /// the full design in its original row order and changes only `weights`.
    Buser {
        /// Caller-owned design matrix.
        design: Array2<f64>,
        /// Caller-owned penalty matrix.
        penalty: Array2<f64>,
        /// Formula name of the learner.
        name: String,
        /// Target effective degrees of freedom.
        df: f64,
        /// Penalty weight; overrides `df` when both are given.
        lambda: Option<f64>,
        /// Whether the design columns are weight-centered.
        center: bool,
        /// Optional varying-coefficient multiplier column.
        by: Option<Array1<f64>>,
    },
}

impl LearnerBlueprint {
    #[allow(clippy::too_many_arguments)]
    /// Build a `bols` blueprint over a single covariate column.
    pub fn scalar_bols(
        x: Array1<f64>,
        name: impl Into<String>,
        lambda: Option<f64>,
        df: Option<f64>,
        center: bool,
        intercept: bool,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != x.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but x has length {}",
                        values.len(),
                        x.len()
                    ),
                ));
            }
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        if let Some(value) = df {
            if !value.is_finite() || value <= 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "df must be finite and positive".to_string(),
                ));
            }
        }
        Ok(Self::ScalarBols {
            x,
            name,
            lambda,
            df,
            center,
            intercept,
            by,
        })
    }

    /// Construct a scalar `bols` blueprint with quantile-binned training
    /// storage. The initial slice supports the one-column, no-`by` form; the
    /// explicit restrictions keep unsupported block/joint binning from
    /// silently changing a model.
    #[allow(clippy::too_many_arguments)]
    /// Build a weight-independent binned `bols` blueprint from a `QuantileBinning`.
    pub fn scalar_bols_binned(
        x: Array1<f64>,
        name: impl Into<String>,
        lambda: Option<f64>,
        df: Option<f64>,
        center: bool,
        intercept: bool,
        by: Option<Array1<f64>>,
        bins: usize,
    ) -> Result<Self, MboostrsError> {
        let blueprint = Self::scalar_bols(x, name, lambda, df, center, intercept, by)?;
        if bins == 0 {
            return Err(MboostrsError::LearnerSetup(
                blueprint.name().to_string(),
                "bins must be a positive integer".to_string(),
            ));
        }
        if intercept {
            return Err(MboostrsError::LearnerSetup(
                blueprint.name().to_string(),
                "binned bols currently requires intercept=false".to_string(),
            ));
        }
        if matches!(&blueprint, Self::ScalarBols { by: Some(_), .. }) {
            return Err(MboostrsError::LearnerSetup(
                blueprint.name().to_string(),
                "binned bols with by= is not implemented; use the exact learner".to_string(),
            ));
        }
        let Self::ScalarBols {
            x,
            name,
            lambda,
            df,
            center,
            ..
        } = blueprint
        else {
            unreachable!()
        };
        let binning = quantile_bin_column(x.view(), bins)
            .map_err(|error| MboostrsError::LearnerSetup(name.clone(), error.to_string()))?;
        Ok(Self::BinnedScalarBols {
            binning,
            name,
            lambda,
            df,
            center,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a B-spline (`bbs`) blueprint from raw design and penalty arrays.
    pub fn bbs(
        x: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
        cyclic: bool,
        boundary_knots: Option<(f64, f64)>,
        deriv: usize,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if n_knots == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "knots must be positive".to_string(),
            ));
        }
        if differences > degree + 1 {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "differences={differences} exceeds degree + 1={}",
                    degree + 1
                ),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != x.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but x has length {}",
                        values.len(),
                        x.len()
                    ),
                ));
            }
        }
        if center && cyclic {
            return Err(MboostrsError::LearnerSetup(
                name,
                "centered cyclic bbs prediction is not implemented".to_string(),
            ));
        }

        let boundary_knots = boundary_knots.unwrap_or_else(|| {
            (
                x.iter().copied().fold(f64::INFINITY, f64::min),
                x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        });
        if !boundary_knots.0.is_finite()
            || !boundary_knots.1.is_finite()
            || boundary_knots.0 >= boundary_knots.1
        {
            return Err(MboostrsError::LearnerSetup(
                name,
                if x.len() < 2 {
                    // The usual cause: not enough data to have a range at all.
                    // Say so in those terms rather than reporting the derived
                    // knots, which are a symptom.
                    format!(
                        "a spline basis needs at least 2 samples; got {} sample{}",
                        x.len(),
                        if x.len() == 1 { "" } else { "s" }
                    )
                } else {
                    "boundary_knots must contain two finite increasing values".to_string()
                },
            ));
        }

        let by_view = by.as_ref().map(|values| values.view());
        let (design, penalty) = build_bbs_template(
            x.view(),
            n_knots,
            degree,
            differences,
            center,
            by_view,
            cyclic,
            Some(boundary_knots),
            deriv,
        );
        let by_values = by.as_ref().map(|values| values.to_vec());
        let basis_info = if center {
            let knots = bspline_knots(boundary_knots, n_knots, degree);
            let p = knots.len() - degree - 1;
            // Same differenceMatrix transform the fit path applies
            // (build_bbs_template), so replayed designs match fitted ones.
            let (centering_mat, centering_cols) = raw_diff_matrix(p, differences)
                .and_then(|d| compute_centering_transform_diff(&d, p))
                .unwrap_or_else(|| compute_centering_transform(&diff_penalty(p, differences), p));
            BasisInfo::BbsCentered {
                knots,
                degree,
                boundary_knots,
                centering_mat,
                centering_rows: p,
                centering_cols,
                by: by_values,
                deriv,
            }
        } else if cyclic {
            BasisInfo::BbsUncentered {
                knots: cyclic_bspline_knots(boundary_knots, n_knots, degree),
                degree,
                boundary_knots,
                by: by_values,
                cyclic: true,
                deriv,
            }
        } else {
            BasisInfo::BbsUncentered {
                knots: bspline_knots(boundary_knots, n_knots, degree),
                degree,
                boundary_knots,
                by: by_values,
                cyclic: false,
                deriv,
            }
        };

        Ok(Self::Bbs {
            // Column-major now, so every fold's `prepare` clones it into the
            // layout `BlockBols` keeps instead of copying it a second time.
            design: into_column_major(design),
            penalty,
            name,
            df,
            lambda,
            basis_info,
        })
    }

    /// Build a bbs blueprint that [`prepare`](Self::prepare) lifts into a
    /// Kronecker-wrapped learner via [`prepare_kronecker_bbs`], matching R's
    /// `bbs(x, df) %O% bols(class, K-1)`. Keeps the raw covariate rather than
    /// a pre-built design, because the Kronecker bbs recipe reconstructs the
    /// un-centred raw basis with the per-class `I + K1` penalty.
    #[allow(clippy::too_many_arguments)]
    pub fn kronecker_bbs(
        x: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        n_resp: usize,
        deriv: usize,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if n_knots == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "knots must be positive".to_string(),
            ));
        }
        if differences > degree + 1 {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "differences={differences} exceeds degree + 1={}",
                    degree + 1
                ),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if n_resp == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "n_resp must be positive".to_string(),
            ));
        }
        Ok(Self::KroneckerBbs {
            x,
            name,
            n_knots,
            degree,
            differences,
            df,
            n_resp,
            deriv,
            center: false,
        })
    }

    /// Set spectral centering on a Kronecker bbs blueprint (a no-op on any
    /// other kind).
    pub fn with_center(mut self, value: bool) -> Self {
        if let Self::KroneckerBbs { center, .. } = &mut self {
            *center = value;
        }
        self
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a binned B-spline (`bbs`) blueprint sharing basis and penalty across resampling folds.
    pub fn bbs_binned(
        x: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
        cyclic: bool,
        boundary_knots: Option<(f64, f64)>,
        deriv: usize,
        bins: usize,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if bins == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "bins must be a positive integer".to_string(),
            ));
        }
        if by.is_some() {
            return Err(MboostrsError::LearnerSetup(
                name,
                "binned bbs with by= is not implemented; use the exact learner".to_string(),
            ));
        }
        if n_knots == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "knots must be positive".to_string(),
            ));
        }
        if differences > degree + 1 {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "differences={differences} exceeds degree + 1={}",
                    degree + 1
                ),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        if center && cyclic {
            return Err(MboostrsError::LearnerSetup(
                name,
                "centered cyclic bbs prediction is not implemented".to_string(),
            ));
        }
        let boundary_knots = boundary_knots.unwrap_or_else(|| {
            (
                x.iter().copied().fold(f64::INFINITY, f64::min),
                x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        });
        if !boundary_knots.0.is_finite()
            || !boundary_knots.1.is_finite()
            || boundary_knots.0 >= boundary_knots.1
        {
            return Err(MboostrsError::LearnerSetup(
                name,
                "boundary_knots must contain two finite increasing values".to_string(),
            ));
        }
        let binning = quantile_bin_column(x.view(), bins)
            .map_err(|error| MboostrsError::LearnerSetup(name.clone(), error.to_string()))?;
        let representatives = ArrayView1::from(binning.values.as_ref());
        let (basis, penalty) = build_bbs_template(
            representatives,
            n_knots,
            degree,
            differences,
            center,
            None,
            cyclic,
            Some(boundary_knots),
            deriv,
        );
        let inner_basis_info = if center {
            let knots = bspline_knots(boundary_knots, n_knots, degree);
            let p = knots.len() - degree - 1;
            let (centering_mat, centering_cols) = raw_diff_matrix(p, differences)
                .and_then(|matrix| compute_centering_transform_diff(&matrix, p))
                .unwrap_or_else(|| compute_centering_transform(&diff_penalty(p, differences), p));
            BasisInfo::BbsCentered {
                knots,
                degree,
                boundary_knots,
                centering_mat,
                centering_rows: p,
                centering_cols,
                by: None,
                deriv,
            }
        } else if cyclic {
            BasisInfo::BbsUncentered {
                knots: cyclic_bspline_knots(boundary_knots, n_knots, degree),
                degree,
                boundary_knots,
                by: None,
                cyclic: true,
                deriv,
            }
        } else {
            BasisInfo::BbsUncentered {
                knots: bspline_knots(boundary_knots, n_knots, degree),
                degree,
                boundary_knots,
                by: None,
                cyclic: false,
                deriv,
            }
        };
        let basis_info = BasisInfo::Binned {
            cutpoints: binning.cutpoints.clone(),
            values: binning.values.clone(),
            lower: binning.lower,
            upper: binning.upper,
            inner: Box::new(inner_basis_info),
        };
        Ok(Self::BinnedBbs {
            binning,
            basis: Arc::new(basis),
            penalty,
            name,
            df,
            lambda,
            basis_info,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a natural-spline (`bns`) blueprint with an identity ridge penalty.
    pub fn bns(
        x: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        df: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
        boundary_knots: Option<(f64, f64)>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if n_knots == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "knots must be positive".to_string(),
            ));
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != x.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but x has length {}",
                        values.len(),
                        x.len()
                    ),
                ));
            }
        }
        let boundary_knots = boundary_knots.unwrap_or_else(|| {
            (
                x.iter().copied().fold(f64::INFINITY, f64::min),
                x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        });
        if !boundary_knots.0.is_finite()
            || !boundary_knots.1.is_finite()
            || boundary_knots.0 >= boundary_knots.1
        {
            return Err(MboostrsError::LearnerSetup(
                name,
                if x.len() < 2 {
                    // The usual cause: not enough data to have a range at all.
                    // Say so in those terms rather than reporting the derived
                    // knots, which are a symptom.
                    format!(
                        "a spline basis needs at least 2 samples; got {} sample{}",
                        x.len(),
                        if x.len() == 1 { "" } else { "s" }
                    )
                } else {
                    "boundary_knots must contain two finite increasing values".to_string()
                },
            ));
        }
        let knots = bspline_knots(boundary_knots, n_knots, degree);
        let x_slice = x.as_slice().ok_or_else(|| {
            MboostrsError::LearnerSetup(name.clone(), "x must be contiguous".to_string())
        })?;
        let mut design = bspline_basis_extrap(x_slice, &knots, degree, boundary_knots);
        if let Some(ref by_values) = by {
            for i in 0..design.nrows() {
                for j in 0..design.ncols() {
                    design[[i, j]] *= by_values[i];
                }
            }
        }
        let penalty = Array2::eye(design.ncols());
        let df = df.unwrap_or(design.ncols() as f64);
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        Ok(Self::Bns {
            design,
            penalty,
            name,
            df,
            knots,
            degree,
            boundary_knots,
            center,
            by,
        })
    }

    /// Build a random-intercept (`brandom`) blueprint from a full indicator design.
    pub fn brandom(
        levels: Array1<f64>,
        name: impl Into<String>,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
        bins: Option<usize>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&levels, &name, "levels")?;
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != levels.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but levels has length {}",
                        values.len(),
                        levels.len()
                    ),
                ));
            }
        }
        if levels.iter().any(|value| {
            value.fract() != 0.0 || *value < i64::MIN as f64 || *value > i64::MAX as f64
        }) {
            return Err(MboostrsError::LearnerSetup(
                name,
                "levels must contain integer codes representable as signed 64-bit values"
                    .to_string(),
            ));
        }
        let mut level_codes = levels.iter().map(|value| *value as i64).collect::<Vec<_>>();
        level_codes.sort_unstable();
        level_codes.dedup();
        let n_levels = level_codes.len();
        if let Some(requested) = bins {
            if by.is_some() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "binned brandom with by= is not implemented; use the exact learner".to_string(),
                ));
            }
            if requested < n_levels {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "bins={requested} is smaller than the {n_levels} factor levels; factor levels cannot be merged"
                    ),
                ));
            }
            let binning = quantile_bin_column(levels.view(), n_levels)
                .map_err(|error| MboostrsError::LearnerSetup(name.clone(), error.to_string()))?;
            return Ok(Self::BinnedBrandom {
                binning,
                basis: Arc::new(Array2::eye(n_levels)),
                penalty: Array2::eye(n_levels),
                name,
                df,
                lambda,
                center,
                level_codes,
            });
        }
        let mut design = Array2::zeros((levels.len(), n_levels));
        for (row, value) in levels.iter().enumerate() {
            let level = level_codes.binary_search(&(*value as i64)).map_err(|_| {
                MboostrsError::LearnerSetup(
                    name.clone(),
                    format!("level code {value} could not be indexed"),
                )
            })?;
            design[[row, level]] = 1.0;
        }
        let penalty = Array2::eye(n_levels);
        Ok(Self::Brandom {
            design,
            penalty,
            name,
            df,
            lambda,
            center,
            level_codes,
            by,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a Markov-random-field (`bmrf`) blueprint from the indicator design and graph penalty.
    pub fn bmrf(
        levels: Array1<f64>,
        name: impl Into<String>,
        k: Array2<f64>,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&levels, &name, "levels")?;
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        if k.nrows() == 0 || k.nrows() != k.ncols() {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "K must be a non-empty square matrix; got {}x{}",
                    k.nrows(),
                    k.ncols()
                ),
            ));
        }
        if k.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::LearnerSetup(
                name,
                "K contains NaN or infinite values".to_string(),
            ));
        }
        for row in 0..k.nrows() {
            for column in 0..row {
                if (k[[row, column]] - k[[column, row]]).abs() > 1e-10 {
                    return Err(MboostrsError::LearnerSetup(
                        name,
                        "K must be symmetric".to_string(),
                    ));
                }
            }
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != levels.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but levels has length {}",
                        values.len(),
                        levels.len()
                    ),
                ));
            }
        }
        let n_levels = k.nrows();
        let mut design = Array2::zeros((levels.len(), n_levels));
        for (row, value) in levels.iter().enumerate() {
            if value.fract() != 0.0 || *value < 0.0 || *value >= n_levels as f64 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "levels must be integer codes in 0..{}; got {value} at observation {row}",
                        n_levels - 1
                    ),
                ));
            }
            design[[row, *value as usize]] = 1.0;
        }
        if let Some(ref values) = by {
            for row in 0..design.nrows() {
                for column in 0..design.ncols() {
                    design[[row, column]] *= values[row];
                }
            }
        }
        let (design, penalty, transform) = if center {
            let (matrix, columns) = compute_centering_transform(&k, n_levels);
            if columns == 0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "K has no positive eigenvalues after removing its null space".to_string(),
                ));
            }
            (
                center_spectral(&design, &k, 1),
                Array2::eye(columns),
                Some((matrix, columns)),
            )
        } else {
            (design, k, None)
        };
        let basis_info = BasisInfo::Bmrf {
            n_levels,
            transform,
            by: by.as_ref().map(|values| values.to_vec()),
        };
        Ok(Self::Bmrf {
            design,
            penalty,
            name,
            df,
            lambda,
            basis_info,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a tensor-product P-spline (`bspatial`) blueprint from design, penalty, and replay metadata.
    pub fn bspatial(
        x1: Array1<f64>,
        x2: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
        boundary_knots1: Option<(f64, f64)>,
        boundary_knots2: Option<(f64, f64)>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x1, &name, "x1")?;
        validate_blueprint_vector(&x2, &name, "x2")?;
        if x1.len() != x2.len() {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!("x1 has length {} but x2 has length {}", x1.len(), x2.len()),
            ));
        }
        if n_knots == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "knots must be positive".to_string(),
            ));
        }
        if differences > degree + 1 {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "differences={differences} exceeds degree + 1={}",
                    degree + 1
                ),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != x1.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but coordinates have length {}",
                        values.len(),
                        x1.len()
                    ),
                ));
            }
        }
        let resolve_boundary = |values: &Array1<f64>, supplied: Option<(f64, f64)>| {
            supplied.unwrap_or_else(|| {
                (
                    values.iter().copied().fold(f64::INFINITY, f64::min),
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            })
        };
        let boundary_knots1 = resolve_boundary(&x1, boundary_knots1);
        let boundary_knots2 = resolve_boundary(&x2, boundary_knots2);
        for (label, boundary) in [
            ("boundary_knots1", boundary_knots1),
            ("boundary_knots2", boundary_knots2),
        ] {
            if !boundary.0.is_finite() || !boundary.1.is_finite() || boundary.0 >= boundary.1 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!("{label} must contain two finite increasing values"),
                ));
            }
        }

        let knots1 = bspline_knots(boundary_knots1, n_knots, degree);
        let knots2 = bspline_knots(boundary_knots2, n_knots, degree);
        let b1 = bspline_basis_extrap(
            x1.as_slice().expect("validated Array1 is contiguous"),
            &knots1,
            degree,
            boundary_knots1,
        );
        let b2 = bspline_basis_extrap(
            x2.as_slice().expect("validated Array1 is contiguous"),
            &knots2,
            degree,
            boundary_knots2,
        );
        let p1 = b1.ncols();
        let p2 = b2.ncols();
        let raw_columns = p1 * p2;
        let mut design = Array2::zeros((x1.len(), raw_columns));
        for row in 0..x1.len() {
            for column1 in 0..p1 {
                for column2 in 0..p2 {
                    design[[row, column1 * p2 + column2]] = b1[[row, column1]] * b2[[row, column2]];
                }
            }
        }
        if let Some(ref values) = by {
            for row in 0..design.nrows() {
                for column in 0..design.ncols() {
                    design[[row, column]] *= values[row];
                }
            }
        }
        let k1 = diff_penalty(p1, differences);
        let k2 = diff_penalty(p2, differences);
        let raw_penalty = kron(&k1, &Array2::eye(p2)) + kron(&Array2::eye(p1), &k2);
        let (design, penalty, centering_mat) = if center {
            let (transform, columns) = compute_centering_transform(&raw_penalty, raw_columns);
            if columns == 0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "tensor penalty has no positive eigenvalues after removing its null space"
                        .to_string(),
                ));
            }
            (
                center_spectral(&design, &raw_penalty, differences),
                Array2::eye(columns),
                Some((transform, raw_columns, columns)),
            )
        } else {
            (design, raw_penalty, None)
        };
        let basis_info = BasisInfo::Bspatial {
            knots1,
            knots2,
            degree,
            boundary_knots1,
            boundary_knots2,
            centering_mat,
            by: by.as_ref().map(|values| values.to_vec()),
        };
        Ok(Self::Bspatial {
            design,
            penalty,
            name,
            df,
            lambda,
            basis_info,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a binned `bspatial` blueprint evaluated at occupied quantile-grid cells.
    pub fn bspatial_binned(
        x1: Array1<f64>,
        x2: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
        boundary_knots1: Option<(f64, f64)>,
        boundary_knots2: Option<(f64, f64)>,
        bins_per_axis: usize,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x1, &name, "x1")?;
        validate_blueprint_vector(&x2, &name, "x2")?;
        if x1.len() != x2.len() {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!("x1 has length {} but x2 has length {}", x1.len(), x2.len()),
            ));
        }
        if by.is_some() {
            return Err(MboostrsError::LearnerSetup(
                name,
                "binned bspatial with by= is not implemented; use the exact learner".to_string(),
            ));
        }
        if bins_per_axis == 0 || n_knots == 0 || differences > degree + 1 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "bins and knots must be positive, and differences must not exceed degree + 1"
                    .to_string(),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if lambda.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(MboostrsError::LearnerSetup(
                name,
                "lambda must be finite and non-negative".to_string(),
            ));
        }
        let resolve_boundary = |values: &Array1<f64>, supplied: Option<(f64, f64)>| {
            supplied.unwrap_or_else(|| {
                (
                    values.iter().copied().fold(f64::INFINITY, f64::min),
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            })
        };
        let boundary_knots1 = resolve_boundary(&x1, boundary_knots1);
        let boundary_knots2 = resolve_boundary(&x2, boundary_knots2);
        for (label, boundary) in [
            ("boundary_knots1", boundary_knots1),
            ("boundary_knots2", boundary_knots2),
        ] {
            if !boundary.0.is_finite() || !boundary.1.is_finite() || boundary.0 >= boundary.1 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!("{label} must contain two finite increasing values"),
                ));
            }
        }
        let grid = quantile_bin_grid(x1.view(), x2.view(), bins_per_axis)
            .map_err(|error| MboostrsError::LearnerSetup(name.clone(), error.to_string()))?;
        let knots1 = bspline_knots(boundary_knots1, n_knots, degree);
        let knots2 = bspline_knots(boundary_knots2, n_knots, degree);
        let b1 = bspline_basis_extrap(&grid.values1, &knots1, degree, boundary_knots1);
        let b2 = bspline_basis_extrap(&grid.values2, &knots2, degree, boundary_knots2);
        let p1 = b1.ncols();
        let p2 = b2.ncols();
        let raw_columns = p1 * p2;
        let mut basis = Array2::zeros((grid.effective, raw_columns));
        for row in 0..grid.effective {
            for column1 in 0..p1 {
                for column2 in 0..p2 {
                    basis[[row, column1 * p2 + column2]] = b1[[row, column1]] * b2[[row, column2]];
                }
            }
        }
        let k1 = diff_penalty(p1, differences);
        let k2 = diff_penalty(p2, differences);
        let raw_penalty = kron(&k1, &Array2::eye(p2)) + kron(&Array2::eye(p1), &k2);
        let (basis, penalty, centering_mat) = if center {
            let (transform, columns) = compute_centering_transform(&raw_penalty, raw_columns);
            if columns == 0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "tensor penalty has no positive eigenvalues after removing its null space"
                        .to_string(),
                ));
            }
            (
                center_spectral(&basis, &raw_penalty, differences),
                Array2::eye(columns),
                Some((transform, raw_columns, columns)),
            )
        } else {
            (basis, raw_penalty, None)
        };
        let inner = BasisInfo::Bspatial {
            knots1,
            knots2,
            degree,
            boundary_knots1,
            boundary_knots2,
            centering_mat,
            by: None,
        };
        let basis_info = BasisInfo::BinnedSpatial {
            marginal1: grid.marginal1.clone(),
            marginal2: grid.marginal2.clone(),
            cell_for_pair: grid.cell_for_pair.clone(),
            cell_values1: grid.values1.clone(),
            cell_values2: grid.values2.clone(),
            inner: Box::new(inner),
        };
        Ok(Self::BinnedBspatial {
            grid,
            basis: Arc::new(basis),
            penalty,
            name,
            df,
            lambda,
            basis_info,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a monotone P-spline (`bmono`) blueprint from raw covariate values.
    pub fn bmono(
        x: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        constraint: MonoConstraint,
        solver: BmonoSolver,
        lambda2: f64,
        niter: usize,
        by: Option<Array1<f64>>,
        center: bool,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if n_knots == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "knots must be positive".to_string(),
            ));
        }
        if differences > degree + 1 {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "differences={differences} exceeds degree + 1={}",
                    degree + 1
                ),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if !lambda2.is_finite() || lambda2 < 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "lambda2 must be finite and non-negative".to_string(),
            ));
        }
        if niter == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "niter must be positive".to_string(),
            ));
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != x.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but x has length {}",
                        values.len(),
                        x.len()
                    ),
                ));
            }
        }
        let boundary_knots = (
            x.iter().copied().fold(f64::INFINITY, f64::min),
            x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        let knots = bspline_knots(boundary_knots, n_knots, degree);
        let basis_info = if center {
            // Build a temporary design to determine p (number of basis columns),
            // then compute the centering matrix from the eigendecomposition of K.
            let tmp_design = bspline_basis(x.as_slice().expect("x contiguous"), &knots, degree);
            let p = tmp_design.ncols();
            let k = diff_penalty(p, differences);
            let (centering_mat, centering_cols) = compute_centering_transform(&k.to_owned(), p);
            BasisInfo::BmonoCentered {
                knots,
                degree,
                boundary_knots,
                by: by.as_ref().map(|values| values.to_vec()),
                centering_mat,
                centering_rows: p,
                centering_cols,
            }
        } else {
            BasisInfo::BmonoUncentered {
                knots,
                degree,
                boundary_knots,
                by: by.as_ref().map(|values| values.to_vec()),
            }
        };
        Ok(Self::Bmono {
            x,
            name,
            n_knots,
            degree,
            differences,
            df,
            lambda: None,
            constraint,
            solver,
            lambda2,
            niter,
            by,
            constraint_matrix: None,
            center,
            basis_info,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a shape-constrained P-spline blueprint with an explicit constraint matrix (Bconstrained).
    pub fn bconstrained(
        x: Array1<f64>,
        name: impl Into<String>,
        n_knots: usize,
        degree: usize,
        differences: usize,
        df: f64,
        lambda: Option<f64>,
        constraint_matrix: Array2<f64>,
        lambda2: f64,
        niter: usize,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        let expected_columns = n_knots + degree + 1;
        if constraint_matrix.nrows() == 0
            || constraint_matrix.ncols() != expected_columns
            || constraint_matrix.iter().any(|value| !value.is_finite())
        {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "C must be a non-empty finite matrix with {expected_columns} columns; got {}x{}",
                    constraint_matrix.nrows(),
                    constraint_matrix.ncols()
                ),
            ));
        }
        let mut blueprint = Self::bmono(
            x,
            name,
            n_knots,
            degree,
            differences,
            df,
            MonoConstraint::Constrained,
            BmonoSolver::Iterative,
            lambda2,
            niter,
            by,
            false,
        )?;
        if let Self::Bmono {
            constraint_matrix: stored,
            ..
        } = &mut blueprint
        {
            *stored = Some(constraint_matrix);
        }
        if let Self::Bmono { lambda: stored, .. } = &mut blueprint {
            *stored = lambda;
        }
        Ok(blueprint)
    }

    /// Choose the constrained solver of a `bmono` / `bconstrained` blueprint
    /// (a no-op on any other kind). [`Self::bconstrained`] starts from the
    /// iterative violation penalty, which leaves a small same-signed
    /// violation each step that boosting accumulates (3e-6 below zero on a
    /// non-negativity constraint after one step); `QuadProg` is exact.
    pub fn with_bmono_solver(mut self, value: BmonoSolver) -> Self {
        if let Self::Bmono { solver, .. } = &mut self {
            *solver = value;
        }
        self
    }

    /// Build a regression-tree (`btree`) blueprint from a stable design matrix.
    pub fn btree(
        design: Array2<f64>,
        name: impl Into<String>,
        minbucket: usize,
        minsplit: usize,
        maxdepth: usize,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        if design.nrows() == 0 || design.ncols() == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "tree design must have at least one row and one column".to_string(),
            ));
        }
        if design.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::LearnerSetup(
                name,
                "tree design contains NaN or infinite values".to_string(),
            ));
        }
        if minbucket == 0 || minsplit == 0 || maxdepth == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "minbucket, minsplit, and maxdepth must be positive".to_string(),
            ));
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != design.nrows() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but tree design has {} rows",
                        values.len(),
                        design.nrows()
                    ),
                ));
            }
        }
        let basis_info = BasisInfo::Btree {
            n_features: design.ncols(),
            by: by.as_ref().map(|values| values.to_vec()),
            categorical: Vec::new(),
        };
        Ok(Self::Btree {
            design,
            name,
            minbucket,
            minsplit,
            maxdepth,
            categorical: Vec::new(),
            selection: SplitSelection::Sse,
            split_method: SplitMethod::Exact,
            by,
            basis_info,
        })
    }

    /// Set a tree blueprint's column-selection rule; other blueprints are
    /// returned unchanged.
    pub fn with_selection(mut self, rule: SplitSelection) -> Self {
        if let Self::Btree { selection, .. } = &mut self {
            *selection = rule;
        }
        self
    }

    /// [`Self::btree`] with an explicit split-finding method
    /// ([`SplitMethod`]): the exhaustive exact scan, or histogram binning.
    pub fn with_split_method(mut self, method: SplitMethod) -> Self {
        if let Self::Btree { split_method, .. } = &mut self {
            *split_method = method;
        }
        self
    }

    /// Mark tree design columns as categorical: `levels[j]` is the level count
    /// of column `j` (codes 0..L-1), 0 for a numeric column. Other
    /// blueprints are returned unchanged.
    pub fn with_categorical(mut self, levels: Vec<usize>) -> Self {
        if let Self::Btree {
            categorical,
            basis_info,
            ..
        } = &mut self
        {
            if let BasisInfo::Btree {
                categorical: replay,
                ..
            } = basis_info
            {
                *replay = levels.clone();
            }
            *categorical = levels;
        }
        self
    }

    #[allow(clippy::too_many_arguments)]
    /// Build an adaptive hinge (`bhinge`) blueprint over a single covariate
    /// column (mboostrs extension, issue #279).
    ///
    /// At each fit step the prepared learner scans `n_candidates` interior
    /// quantile knots of `x` in both hinge directions, fits every candidate
    /// hinge by weighted least squares, and adds the one with the smallest
    /// residual SSE — so one boosting selection adds exactly one listable
    /// hinge term.
    pub fn bhinge(
        x: Array1<f64>,
        name: impl Into<String>,
        n_candidates: usize,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != x.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but x has {} observations",
                        values.len(),
                        x.len()
                    ),
                ));
            }
        }
        if n_candidates == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "n_candidates must be positive".to_string(),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        Ok(Self::Bhinge {
            x,
            name,
            n_candidates,
            df,
            lambda,
            center,
            by,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a kernel-ridge (`bkernel`) blueprint (mboostrs extension).
    pub fn bkernel(
        x: Array1<f64>,
        name: impl Into<String>,
        bandwidth: f64,
        df: f64,
        kernel_type: impl Into<String>,
        center: bool,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "bandwidth must be finite and positive".to_string(),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        let kernel_type = kernel_type.into();
        if !matches!(
            kernel_type.as_str(),
            "gaussian" | "epanechnikov" | "uniform"
        ) {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!("unknown kernel type {kernel_type:?}"),
            ));
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != x.len() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but x has length {}",
                        values.len(),
                        x.len()
                    ),
                ));
            }
        }
        Ok(Self::Bkernel {
            x,
            name,
            bandwidth,
            df,
            kernel_type,
            center,
            by,
        })
    }

    /// Build a Nyström-approximated kernel blueprint.
    ///
    /// The exact form's memory is `O(n^2)` in the design *and* the penalty;
    /// this one is `O(n m)` and `O(m^2)`, with `m` the caller's choice.
    #[allow(clippy::too_many_arguments)]
    pub fn bkernel_nystrom(
        x: Array1<f64>,
        name: impl Into<String>,
        bandwidth: f64,
        df: f64,
        kernel_type: impl Into<String>,
        landmarks: usize,
        center: bool,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        validate_blueprint_vector(&x, &name, "x")?;
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "bandwidth must be finite and positive".to_string(),
            ));
        }
        if landmarks == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "landmarks must be at least 1".to_string(),
            ));
        }
        Ok(Self::BkernelNystrom {
            x,
            name,
            bandwidth,
            df,
            kernel_type: kernel_type.into(),
            landmarks,
            center,
            by,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a caller-owned design/penalty (`buser`) blueprint.
    pub fn buser(
        design: Array2<f64>,
        penalty: Array2<f64>,
        name: impl Into<String>,
        df: f64,
        lambda: Option<f64>,
        center: bool,
        by: Option<Array1<f64>>,
    ) -> Result<Self, MboostrsError> {
        let name = name.into();
        if design.nrows() == 0 || design.ncols() == 0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "X must have at least one row and one column".to_string(),
            ));
        }
        if design.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::LearnerSetup(
                name,
                "X contains NaN or infinite values".to_string(),
            ));
        }
        if penalty.nrows() != design.ncols() || penalty.ncols() != design.ncols() {
            return Err(MboostrsError::LearnerSetup(
                name,
                format!(
                    "K must be {}x{} to match X; got {}x{}",
                    design.ncols(),
                    design.ncols(),
                    penalty.nrows(),
                    penalty.ncols()
                ),
            ));
        }
        if penalty.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::LearnerSetup(
                name,
                "K contains NaN or infinite values".to_string(),
            ));
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(MboostrsError::LearnerSetup(
                name,
                "df must be finite and positive".to_string(),
            ));
        }
        if let Some(value) = lambda {
            if !value.is_finite() || value < 0.0 {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    "lambda must be finite and non-negative".to_string(),
                ));
            }
        }
        if let Some(ref values) = by {
            validate_blueprint_vector(values, &name, "by")?;
            if values.len() != design.nrows() {
                return Err(MboostrsError::LearnerSetup(
                    name,
                    format!(
                        "by has length {} but X has {} rows",
                        values.len(),
                        design.nrows()
                    ),
                ));
            }
        }
        Ok(Self::Buser {
            design,
            penalty,
            name,
            df,
            lambda,
            center,
            by,
        })
    }

    /// Name of this learner's formula term.
    pub fn name(&self) -> &str {
        match self {
            Self::ScalarBols { name, .. }
            | Self::BinnedScalarBols { name, .. }
            | Self::Bbs { name, .. }
            | Self::KroneckerBbs { name, .. }
            | Self::BinnedBbs { name, .. }
            | Self::Bns { name, .. }
            | Self::Brandom { name, .. }
            | Self::BinnedBrandom { name, .. }
            | Self::Bmrf { name, .. }
            | Self::Bspatial { name, .. }
            | Self::BinnedBspatial { name, .. }
            | Self::Bmono { name, .. }
            | Self::Btree { name, .. }
            | Self::Bhinge { name, .. }
            | Self::Bkernel { name, .. }
            | Self::BkernelNystrom { name, .. }
            | Self::Buser { name, .. } => name,
        }
    }

    /// Number of observations this learner's design is indexed by.
    pub fn n_observations(&self) -> usize {
        match self {
            Self::ScalarBols { x, .. } => x.len(),
            Self::BinnedScalarBols { binning, .. } => binning.index.len(),
            Self::Bbs { design, .. } => design.nrows(),
            Self::BinnedBbs { binning, .. } => binning.index.len(),
            Self::KroneckerBbs { x, .. } => x.len(),
            Self::Bns { design, .. }
            | Self::Brandom { design, .. }
            | Self::Bmrf { design, .. }
            | Self::Bspatial { design, .. }
            | Self::Buser { design, .. } => design.nrows(),
            Self::BinnedBrandom { binning, .. } => binning.index.len(),
            Self::BinnedBspatial { grid, .. } => grid.index.len(),
            Self::Bmono { x, .. } => x.len(),
            Self::Bkernel { x, .. } => x.len(),
            Self::BkernelNystrom { x, .. } => x.len(),
            Self::Btree { design, .. } => design.nrows(),
            Self::Bhinge { x, .. } => x.len(),
        }
    }

    /// [`prepare`](Self::prepare) for a single fit: the blueprint is consumed,
    /// so a `bbs` design moves into the learner rather than being cloned from
    /// the template a cross-validation loop would reuse.
    pub fn into_prepared(
        self,
        weights: ArrayView1<f64>,
    ) -> Result<(PreparedLearner, BasisInfo), MboostrsError> {
        match self {
            Self::Bbs {
                design,
                penalty,
                name,
                df,
                lambda,
                basis_info,
            } if weights.len() == design.nrows()
                && weights
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0) =>
            {
                Ok((
                    prepare_bbs_from_template(design, penalty, weights, df, lambda, &name),
                    basis_info,
                ))
            }
            other => other.prepare(weights),
        }
    }

    /// Construct a prepared learner and the matching new-data replay recipe.
    pub fn prepare(
        &self,
        weights: ArrayView1<f64>,
    ) -> Result<(PreparedLearner, BasisInfo), MboostrsError> {
        if weights.len() != self.n_observations() {
            return Err(MboostrsError::LearnerSetup(
                self.name().to_string(),
                format!(
                    "weights have length {} but learner has {} observations",
                    weights.len(),
                    self.n_observations()
                ),
            ));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(MboostrsError::LearnerSetup(
                self.name().to_string(),
                "weights must be finite and non-negative".to_string(),
            ));
        }

        match self {
            Self::ScalarBols {
                x,
                name,
                lambda,
                df,
                center,
                intercept,
                by,
            } => {
                if *intercept {
                    let n = x.len();
                    let mut design = Array2::ones((n, 2));
                    for i in 0..n {
                        design[[i, 1]] = x[i];
                    }
                    if let Some(by_values) = by {
                        for i in 0..n {
                            design[[i, 0]] *= by_values[i];
                            design[[i, 1]] *= by_values[i];
                        }
                    }
                    // R `bols` penalizes every column it holds — `X_ols` builds
                    // `K = diag(ncol(X))` — so the intercept shrinks with the
                    // slope under one lambda. Leaving the intercept out of the
                    // penalty calibrates lambda against a different matrix than
                    // it is then applied to, and fits a model R cannot express.
                    let penalty = Array2::eye(2);
                    let lambda = lambda.unwrap_or_else(|| {
                        df.map_or(0.0, |target| {
                            df_to_lambda(&design, &penalty, &weights.to_owned(), target)
                        })
                    });
                    Ok((
                        prepare_block_bols(
                            design,
                            weights,
                            penalty.mapv(|value| value * lambda),
                            false,
                            name,
                        )
                        .with_effective_lambda(Some(lambda)),
                        BasisInfo::Bols {
                            intercept: true,
                            by: by.as_ref().map(|values| values.to_vec()),
                        },
                    ))
                } else {
                    let lambda = lambda.unwrap_or_else(|| {
                        df.map_or(0.0, |target| {
                            df_to_lambda_scalar(x.view(), weights, target, *center)
                        })
                    });
                    let learner = prepare_scalar_bols(
                        x.view(),
                        weights,
                        lambda,
                        *center,
                        name,
                        by.as_ref().map(|values| values.view()),
                    );
                    let centering = match &learner {
                        PreparedLearner::ScalarBols { centering, .. } => *centering,
                        _ => 0.0,
                    };
                    Ok((
                        learner,
                        BasisInfo::ScalarBols {
                            centering,
                            by: by.as_ref().map(|values| values.to_vec()),
                        },
                    ))
                }
            }
            Self::BinnedScalarBols {
                binning,
                name,
                lambda,
                df,
                center,
            } => {
                let lambda = lambda.unwrap_or_else(|| {
                    df.map_or(0.0, |target| {
                        let sumw: f64 = weights.sum();
                        let centering = if *center && sumw > 0.0 {
                            let mut total = 0.0f64;
                            for i in 0..binning.index.len() {
                                total += weights[i] * binning.values[binning.index[i] as usize];
                            }
                            total / sumw
                        } else {
                            0.0
                        };
                        let mut xtwx = 0.0f64;
                        for i in 0..binning.index.len() {
                            let value = binning.values[binning.index[i] as usize] - centering;
                            xtwx += weights[i] * value * value;
                        }
                        df_to_lambda_scalar_from_xtwx(xtwx, target, *center)
                    })
                });
                let learner =
                    prepare_binned_scalar_bols(binning.clone(), weights, lambda, *center, name);
                let centering = match &learner {
                    PreparedLearner::BinnedScalarBols { centering, .. } => *centering,
                    _ => unreachable!(),
                };
                Ok((
                    learner,
                    BasisInfo::Binned {
                        cutpoints: binning.cutpoints.clone(),
                        values: binning.values.clone(),
                        lower: binning.lower,
                        upper: binning.upper,
                        inner: Box::new(BasisInfo::ScalarBols {
                            centering,
                            by: None,
                        }),
                    },
                ))
            }
            Self::Bbs {
                design,
                penalty,
                name,
                df,
                lambda,
                basis_info,
            } => Ok((
                prepare_bbs_from_template(
                    design.clone(),
                    penalty.clone(),
                    weights,
                    *df,
                    *lambda,
                    name,
                ),
                basis_info.clone(),
            )),
            Self::KroneckerBbs {
                x,
                name,
                n_knots,
                degree,
                differences,
                df,
                n_resp,
                deriv,
                center,
            } => {
                let bk = (
                    x.iter().cloned().fold(f64::INFINITY, f64::min),
                    x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                );
                let knots = bspline_knots(bk, *n_knots, *degree);
                let learner = prepare_kronecker_bbs_centered(
                    x.view(),
                    weights,
                    *n_knots,
                    *degree,
                    *differences,
                    *df,
                    *n_resp,
                    name,
                    *deriv,
                    *center,
                );
                Ok((
                    learner,
                    BasisInfo::KroneckerBbs {
                        knots,
                        degree: *degree,
                        boundary_knots: bk,
                        deriv: *deriv,
                    },
                ))
            }
            Self::BinnedBbs {
                binning,
                basis,
                penalty,
                name,
                df,
                lambda,
                basis_info,
            } => {
                let mut bin_weights = Array1::zeros(binning.effective);
                for i in 0..binning.index.len() {
                    bin_weights[binning.index[i] as usize] += weights[i];
                }
                let lambda = lambda
                    .unwrap_or_else(|| df_to_lambda(basis.as_ref(), penalty, &bin_weights, *df));
                Ok((
                    prepare_binned_block_bols(
                        binning,
                        basis.clone(),
                        weights,
                        penalty.mapv(|value| value * lambda),
                        false,
                        name,
                    )
                    .with_effective_lambda(Some(lambda)),
                    basis_info.clone(),
                ))
            }
            Self::Bns {
                design,
                penalty,
                name,
                df,
                knots,
                degree,
                boundary_knots,
                center,
                by,
            } => {
                let lambda = df_to_lambda(design, penalty, &weights.to_owned(), *df);
                let learner = prepare_block_bols(
                    design.clone(),
                    weights,
                    penalty.mapv(|value| value * lambda),
                    *center,
                    name,
                )
                .with_effective_lambda(Some(lambda));
                let centering = match &learner {
                    PreparedLearner::BlockBols { centering, .. } => centering.to_vec(),
                    _ => {
                        return Err(MboostrsError::LearnerSetup(
                            name.clone(),
                            "internal bns preparation did not produce a block learner".to_string(),
                        ))
                    }
                };
                Ok((
                    learner,
                    BasisInfo::Bns {
                        knots: knots.clone(),
                        degree: *degree,
                        boundary_knots: *boundary_knots,
                        centering,
                        by: by.as_ref().map(|values| values.to_vec()),
                    },
                ))
            }
            Self::Brandom {
                design,
                penalty,
                name,
                df,
                lambda,
                center,
                level_codes,
                by,
            } => {
                // The df is calibrated on the design the learner fits, after
                // the `by=` modifier scales it, as mboost does.
                let mut prepared_design = design.clone();
                if let Some(by_values) = by {
                    for row in 0..prepared_design.nrows() {
                        for column in 0..prepared_design.ncols() {
                            prepared_design[[row, column]] *= by_values[row];
                        }
                    }
                }
                let lambda = match lambda {
                    Some(value) => *value,
                    None => {
                        let design_for_df = if *center {
                            let sum_weights = weights.sum();
                            if sum_weights <= 0.0 {
                                return Err(MboostrsError::LearnerSetup(
                                    name.clone(),
                                    "weights must have positive total mass".to_string(),
                                ));
                            }
                            let mut centered = prepared_design.clone();
                            for column in 0..centered.ncols() {
                                let mean = centered.column(column).dot(&weights) / sum_weights;
                                centered
                                    .column_mut(column)
                                    .mapv_inplace(|value| value - mean);
                            }
                            centered
                        } else {
                            prepared_design.clone()
                        };
                        df_to_lambda(&design_for_df, penalty, &weights.to_owned(), *df)
                    }
                };
                let learner = prepare_block_bols(
                    prepared_design,
                    weights,
                    penalty.mapv(|value| value * lambda),
                    *center,
                    name,
                )
                .with_effective_lambda(Some(lambda));
                let centering = match &learner {
                    PreparedLearner::BlockBols { centering, .. } => centering.to_vec(),
                    _ => {
                        return Err(MboostrsError::LearnerSetup(
                            name.clone(),
                            "internal brandom preparation did not produce a block learner"
                                .to_string(),
                        ))
                    }
                };
                Ok((
                    learner,
                    BasisInfo::Brandom {
                        level_codes: level_codes.clone(),
                        centering,
                        by: by.as_ref().map(|values| values.to_vec()),
                    },
                ))
            }
            Self::BinnedBrandom {
                binning,
                basis,
                penalty,
                name,
                df,
                lambda,
                center,
                level_codes,
            } => {
                let mut bin_weights = Array1::zeros(binning.effective);
                for i in 0..binning.index.len() {
                    bin_weights[binning.index[i] as usize] += weights[i];
                }
                let lambda = match lambda {
                    Some(value) => *value,
                    None => {
                        let mut design_for_df = basis.as_ref().clone();
                        if *center {
                            let sumw = bin_weights.sum();
                            if sumw <= 0.0 {
                                return Err(MboostrsError::LearnerSetup(
                                    name.clone(),
                                    "weights must have positive total mass".to_string(),
                                ));
                            }
                            for column in 0..design_for_df.ncols() {
                                let mean = design_for_df.column(column).dot(&bin_weights) / sumw;
                                design_for_df
                                    .column_mut(column)
                                    .mapv_inplace(|value| value - mean);
                            }
                        }
                        df_to_lambda(&design_for_df, penalty, &bin_weights, *df)
                    }
                };
                let learner = prepare_binned_block_bols(
                    binning,
                    basis.clone(),
                    weights,
                    penalty.mapv(|value| value * lambda),
                    *center,
                    name,
                )
                .with_effective_lambda(Some(lambda));
                let centering = match &learner {
                    PreparedLearner::BinnedBlockBols { centering, .. } => centering.to_vec(),
                    _ => unreachable!(),
                };
                Ok((
                    learner,
                    BasisInfo::Brandom {
                        level_codes: level_codes.clone(),
                        centering,
                        by: None,
                    },
                ))
            }
            Self::Bmrf {
                design,
                penalty,
                name,
                df,
                lambda,
                basis_info,
            } => {
                let lambda = lambda
                    .unwrap_or_else(|| df_to_lambda(design, penalty, &weights.to_owned(), *df));
                Ok((
                    prepare_block_bols(
                        design.clone(),
                        weights,
                        penalty.mapv(|value| value * lambda),
                        false,
                        name,
                    )
                    .with_effective_lambda(Some(lambda)),
                    basis_info.clone(),
                ))
            }
            Self::Bspatial {
                design,
                penalty,
                name,
                df,
                lambda,
                basis_info,
            } => {
                // A single `X'WX` serves the df→λ calibration and the solver
                // Gram (`X'WX + λK`), so the second O(n·p²) pass the previous
                // `df_to_lambda` + `prepare_block_bols` pair paid is gone.
                // The unpenalized Gram is also kept as an exact CSR copy for
                // the Gaussian Gram route's self-block.
                //
                // The calibration Gram comes from `weighted_gram_blocked`, not
                // the parallel faer kernel, because the df→λ calibration is
                // hypersensitive at tensor sizes: its penalty spectrum has a
                // noise band of order 1e-4 in λ, and every R-parity fit is
                // pinned to the λ this exact accumulation order produces. The
                // solver Gram is now `X'WX + λK` rather than a λK-seeded faer
                // accumulation; the two differ only by rounding, bounded by
                // the tolerance-gated tensor-parity tests.
                Ok((
                    prepare_block_bols_shared_gram(
                        design.clone(),
                        weights,
                        penalty.clone(),
                        *df,
                        *lambda,
                        name,
                    ),
                    basis_info.clone(),
                ))
            }
            Self::BinnedBspatial {
                grid,
                basis,
                penalty,
                name,
                df,
                lambda,
                basis_info,
            } => {
                let mut bin_weights = Array1::zeros(grid.effective);
                for i in 0..grid.index.len() {
                    bin_weights[grid.index[i] as usize] += weights[i];
                }
                let lambda = lambda
                    .unwrap_or_else(|| df_to_lambda(basis.as_ref(), penalty, &bin_weights, *df));
                Ok((
                    prepare_indexed_block_bols(
                        grid.index.clone(),
                        basis.clone(),
                        weights,
                        penalty.mapv(|value| value * lambda),
                        false,
                        name,
                    )
                    .with_effective_lambda(Some(lambda)),
                    basis_info.clone(),
                ))
            }
            Self::Bmono {
                x,
                name,
                n_knots,
                degree,
                differences,
                df,
                lambda,
                constraint,
                solver,
                lambda2,
                niter,
                by,
                constraint_matrix,
                center,
                basis_info,
            } => {
                let learner = prepare_bmono_with_lambda(
                    x.view(),
                    weights,
                    *n_knots,
                    *degree,
                    *differences,
                    *df,
                    *lambda,
                    *constraint,
                    *solver,
                    *lambda2,
                    *niter,
                    name,
                    by.as_ref().map(|values| values.view()),
                    constraint_matrix.as_ref().map(|matrix| matrix.view()),
                    *center,
                );

                // When the Ts_constraint path fires (center=true,
                // Increasing/Decreasing, no custom C), the training design
                // is  raw_bspline @ (-D_cumsum) @ D'@inv(D·D')  but the
                // pre-stored BmonoCentered BasisInfo would rebuild only
                // raw_bspline @ spectral_centering.  Compute the combined
                // centering matrix so predict matches the training pipeline.
                let use_ts = *center
                    && constraint_matrix.is_none()
                    && matches!(
                        *constraint,
                        MonoConstraint::Increasing | MonoConstraint::Decreasing
                    );
                let actual_basis = if use_ts {
                    let bk = (
                        x.iter().cloned().fold(f64::INFINITY, f64::min),
                        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    );
                    let knots = bspline_knots(bk, *n_knots, *degree);
                    let p = knots.len() - degree - 1;
                    let d_raw = raw_diff_matrix(p, *differences)
                        .expect("differences >= p not supported for centering");

                    // Combined centering:  C = (-D_cumsum) @ D' @ inv(D·D')
                    let d_cum = ts_constraint_cumsum_matrix(p);
                    let sign = if matches!(*constraint, MonoConstraint::Decreasing) {
                        -1.0
                    } else {
                        1.0
                    };
                    // C_cols = D' @ inv(D·D')  — standard difference centering
                    let (c_raw, r) = compute_centering_transform_diff(&d_raw, p)
                        .expect("D·D' must be invertible for centering");
                    // Combined: sign * D_cumsum @ C, where C = D' @ inv(D·D').
                    // c_raw is column-major: c_raw[col * p + row] = C[row, col].
                    // d_cum is (p, p) row-major.  Result must be column-major
                    // (p, r) for BbsCentered::predict_into.
                    let d_signed = d_cum.mapv(|v| sign * v);
                    let mut centering_mat = vec![0.0f64; p * r];
                    for col in 0..r {
                        for row in 0..p {
                            let mut s = 0.0;
                            for k in 0..p {
                                s += d_signed[[row, k]] * c_raw[col * p + k];
                            }
                            centering_mat[col * p + row] = s;
                        }
                    }
                    BasisInfo::BbsCentered {
                        knots,
                        degree: *degree,
                        boundary_knots: bk,
                        centering_mat,
                        centering_rows: p,
                        centering_cols: r,
                        by: by.as_ref().map(|values| values.to_vec()),
                        deriv: 0,
                    }
                } else {
                    basis_info.clone()
                };

                Ok((learner, actual_basis))
            }
            Self::Btree {
                design,
                name,
                minbucket,
                minsplit,
                maxdepth,
                categorical,
                selection,
                split_method,
                by,
                basis_info,
            } => Ok((
                prepare_btree_selected_with_method(
                    design.view(),
                    weights,
                    *minbucket,
                    *minsplit,
                    *maxdepth,
                    by.clone(),
                    name,
                    categorical.clone(),
                    *selection,
                    *split_method,
                ),
                basis_info.clone(),
            )),
            Self::Bhinge {
                x,
                name,
                df,
                lambda,
                center,
                by,
                n_candidates,
            } => {
                // The knot grid is fold-invariant (built from the raw
                // covariate), but the effective ridge must be calibrated on
                // the current weights, like every `df -> lambda` path.
                let knots =
                    hinge_knot_grid(x.as_slice().expect("x must be contiguous"), *n_candidates);
                let lambda =
                    lambda.unwrap_or_else(|| df_to_lambda_scalar(x.view(), weights, *df, *center));
                Ok((
                    prepare_bhinge(
                        x.clone(),
                        lambda,
                        *df,
                        *center,
                        by.clone(),
                        name.clone(),
                        knots,
                    ),
                    BasisInfo::Bhinge {
                        by: by.as_ref().map(|values| values.to_vec()),
                    },
                ))
            }
            Self::Bkernel {
                x,
                name,
                bandwidth,
                df,
                kernel_type,
                center,
                by,
            } => {
                let learner = prepare_bkernel(
                    x.view(),
                    weights,
                    *bandwidth,
                    *df,
                    kernel_type,
                    *center,
                    name,
                    by.as_ref().map(|values| values.view()),
                );
                let centering = match &learner {
                    PreparedLearner::BlockBols { centering, .. } => centering.to_vec(),
                    _ => {
                        return Err(MboostrsError::LearnerSetup(
                            name.clone(),
                            "internal bkernel preparation did not produce a block learner"
                                .to_string(),
                        ))
                    }
                };
                Ok((
                    learner,
                    BasisInfo::Bkernel {
                        training_x: x.to_vec(),
                        bandwidth: *bandwidth,
                        kernel_type: kernel_type.clone(),
                        centering,
                    },
                ))
            }
            Self::BkernelNystrom {
                x,
                name,
                bandwidth,
                df,
                kernel_type,
                landmarks,
                center,
                by,
            } => {
                let parts = prepare_bkernel_nystrom_parts(
                    x.view(),
                    weights,
                    *bandwidth,
                    *df,
                    kernel_type,
                    *landmarks,
                    *center,
                    name,
                    by.as_ref().map(|values| values.view()),
                )?;
                let centering = match &parts.learner {
                    PreparedLearner::BlockBols { centering, .. } => centering.to_vec(),
                    _ => {
                        return Err(MboostrsError::LearnerSetup(
                            name.clone(),
                            "internal bkernel preparation did not produce a block learner"
                                .to_string(),
                        ))
                    }
                };
                Ok((
                    parts.learner,
                    BasisInfo::BkernelNystrom {
                        landmarks: parts.landmarks,
                        cholesky_l: parts.cholesky_l,
                        bandwidth: *bandwidth,
                        kernel_type: kernel_type.clone(),
                        centering,
                    },
                ))
            }
            Self::Buser {
                design,
                penalty,
                name,
                df,
                lambda,
                center,
                by,
            } => {
                let learner = prepare_buser(
                    design.clone(),
                    penalty.clone(),
                    weights,
                    *df,
                    *lambda,
                    *center,
                    name,
                    by.as_ref().map(|values| values.view()),
                );
                let centering = match &learner {
                    PreparedLearner::BlockBols { centering, .. } => centering.to_vec(),
                    // Detection only fires for `center = false`, so the
                    // factored tensor's centering is identically zero.
                    PreparedLearner::FactoredBlockBols { p_l, n_groups, .. } => {
                        vec![0.0; p_l * n_groups]
                    }
                    _ => {
                        return Err(MboostrsError::LearnerSetup(
                            name.clone(),
                            "internal buser preparation did not produce a block learner"
                                .to_string(),
                        ))
                    }
                };
                Ok((
                    learner,
                    BasisInfo::Buser {
                        n_features: design.ncols(),
                        centering,
                        by: by.as_ref().map(|values| values.to_vec()),
                    },
                ))
            }
        }
    }
}

fn validate_blueprint_vector(
    values: &Array1<f64>,
    learner_name: &str,
    field: &str,
) -> Result<(), MboostrsError> {
    if values.is_empty() {
        return Err(MboostrsError::LearnerSetup(
            learner_name.to_string(),
            format!("{field} must contain at least one observation"),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::LearnerSetup(
            learner_name.to_string(),
            format!("{field} contains NaN or infinite values"),
        ));
    }
    Ok(())
}

// ─── btree: tree node encoding ──────────────────────────────────

/// A node in a fitted regression tree.
///
/// Stored in a flat `Vec<TreeNode>`; children are referenced by index.
/// Split nodes store the variable index and split point; leaf nodes
/// store the predicted mean.
#[derive(Clone, Debug)]
pub enum TreeNode {
    /// Internal split node: route on `x[var] <= point`.
    Split {
        /// Index of the column used for the split.
        var: usize,
        /// Split threshold value.
        point: f64,
        /// Index of the left child node.
        left: usize,
        /// Index of the right child node.
        right: usize,
    },
    /// Internal split on a categorical column coded 0..L-1: rows whose code
    /// is in `left_levels` go left, all others go right. The fit lists the
    /// lighter child's levels, so a code the fit never saw goes to the child
    /// that held more of the training weight.
    SplitSet {
        /// Index of the column used for the split.
        var: usize,
        /// Level codes routed to the left child, ascending.
        left_levels: Vec<usize>,
        /// Index of the left child node.
        left: usize,
        /// Index of the right child node.
        right: usize,
    },
    /// Terminal node: predicts a constant mean.
    Leaf {
        /// Predicted response mean for observations reaching this node.
        mean: f64,
    },
}

/// A fitted regression tree that can predict on new data.
#[derive(Clone, Debug)]
pub struct FittedTree {
    nodes: Vec<TreeNode>,
}

impl FittedTree {
    /// Predict for a single observation described by its row in `design`.
    pub fn predict_row(&self, row: ArrayView1<f64>) -> f64 {
        match &self.nodes[self.leaf_index(row)] {
            TreeNode::Leaf { mean } => *mean,
            _ => unreachable!("leaf_index stops only at a leaf"),
        }
    }

    /// The index of the leaf node `row` is routed to: the descent of
    /// [`FittedTree::predict_row`], returning where it ends rather than
    /// the leaf's mean there. Two rows are in the same leaf exactly when
    /// the fitted step averages their working responses together.
    pub fn leaf_index(&self, row: ArrayView1<f64>) -> usize {
        let mut node_idx = 0;
        loop {
            match &self.nodes[node_idx] {
                TreeNode::Leaf { .. } => return node_idx,
                TreeNode::Split {
                    var,
                    point,
                    left,
                    right,
                } => {
                    let val = row[*var];
                    node_idx = if val <= *point { *left } else { *right };
                }
                TreeNode::SplitSet {
                    var,
                    left_levels,
                    left,
                    right,
                } => {
                    let val = row[*var];
                    let code = val as usize;
                    let goes_left = val >= 0.0
                        && val.fract() == 0.0
                        && left_levels.binary_search(&code).is_ok();
                    node_idx = if goes_left { *left } else { *right };
                }
            }
        }
    }

    /// The prediction for a row averaged over the levels of each categorical
    /// column whose code the fit never saw (negative, fractional, or at least
    /// the column's level count `categorical[var]`): the mean of the
    /// predictions with the column set to each of its levels in turn, the
    /// "average group" the `Average` unseen-level policy asks for. The descent
    /// keeps, per such column, the levels that can still reach the node; at a
    /// level-set split on it both branches are taken, weighted by the share of
    /// those levels on each side. Known codes and numeric splits descend as in
    /// [`FittedTree::predict_row`].
    pub fn predict_row_level_average(&self, row: ArrayView1<f64>, categorical: &[usize]) -> f64 {
        let mut possible: Vec<Option<Vec<bool>>> = (0..row.len())
            .map(|var| {
                let n_levels = categorical.get(var).copied().unwrap_or(0);
                let val = row[var];
                let known = val >= 0.0 && val.fract() == 0.0 && (val as usize) < n_levels;
                (n_levels > 0 && !known).then(|| vec![true; n_levels])
            })
            .collect();
        self.level_average_from(0, row, &mut possible)
    }

    fn level_average_from(
        &self,
        node_idx: usize,
        row: ArrayView1<f64>,
        possible: &mut [Option<Vec<bool>>],
    ) -> f64 {
        match &self.nodes[node_idx] {
            TreeNode::Leaf { mean } => *mean,
            TreeNode::Split {
                var,
                point,
                left,
                right,
            } => {
                let next = if row[*var] <= *point { *left } else { *right };
                self.level_average_from(next, row, possible)
            }
            TreeNode::SplitSet {
                var,
                left_levels,
                left,
                right,
            } => {
                let Some(levels) = possible[*var].clone() else {
                    let val = row[*var];
                    let goes_left = val >= 0.0
                        && val.fract() == 0.0
                        && left_levels.binary_search(&(val as usize)).is_ok();
                    let next = if goes_left { *left } else { *right };
                    return self.level_average_from(next, row, possible);
                };
                let in_left = |code: usize| left_levels.binary_search(&code).is_ok();
                let left_set: Vec<bool> = levels
                    .iter()
                    .enumerate()
                    .map(|(code, &p)| p && in_left(code))
                    .collect();
                let right_set: Vec<bool> = levels
                    .iter()
                    .enumerate()
                    .map(|(code, &p)| p && !in_left(code))
                    .collect();
                let n_left = left_set.iter().filter(|&&p| p).count();
                let n_right = right_set.iter().filter(|&&p| p).count();
                let total = (n_left + n_right) as f64;
                let mut value = 0.0;
                for (count, set, next) in [(n_left, left_set, *left), (n_right, right_set, *right)]
                {
                    if count > 0 {
                        possible[*var] = Some(set);
                        value +=
                            count as f64 / total * self.level_average_from(next, row, possible);
                    }
                }
                possible[*var] = Some(levels);
                value
            }
        }
    }

    /// Predict for all rows of `design`.
    pub fn predict(&self, design: ArrayView2<f64>) -> Array1<f64> {
        Array1::from_shape_fn(design.nrows(), |i| self.predict_row(design.row(i)))
    }

    /// Encode as a flat coefficient vector for storage in `BoostingPath.coefs`.
    ///
    /// Layout: `[num_nodes, node0_tag, node0_fields..., node1_tag, node1_fields..., ...]`
    /// - Split tag = 0.0, fields = `[var, point, left, right]`
    /// - Leaf tag = 1.0, fields = `[mean]`
    /// - Level-set split tag = 2.0, fields = `[var, m, level_1..level_m, left, right]`
    pub fn to_coef_vec(&self) -> Array1<f64> {
        let mut v = Vec::with_capacity(self.nodes.len() * 5 + 1);
        v.push(self.nodes.len() as f64);
        for node in &self.nodes {
            match node {
                TreeNode::Split {
                    var,
                    point,
                    left,
                    right,
                } => {
                    v.push(0.0);
                    v.push(*var as f64);
                    v.push(*point);
                    v.push(*left as f64);
                    v.push(*right as f64);
                }
                TreeNode::Leaf { mean } => {
                    v.push(1.0);
                    v.push(*mean);
                }
                TreeNode::SplitSet {
                    var,
                    left_levels,
                    left,
                    right,
                } => {
                    v.push(2.0);
                    v.push(*var as f64);
                    v.push(left_levels.len() as f64);
                    v.extend(left_levels.iter().map(|&l| l as f64));
                    v.push(*left as f64);
                    v.push(*right as f64);
                }
            }
        }
        Array1::from_vec(v)
    }

    /// The encoded tree with every leaf mean multiplied by `factor` and the
    /// structure (node count, split variables, split points, child indices)
    /// unchanged: the tree whose prediction is `factor` times this one's.
    pub fn scale_coef_vec(coef: &[f64], factor: f64) -> Vec<f64> {
        let mut out = coef.to_vec();
        let num_nodes = coef[0] as usize;
        let mut idx = 1;
        for _ in 0..num_nodes {
            if coef[idx] == 0.0 {
                idx += 5;
            } else if coef[idx] == 2.0 {
                idx += 5 + coef[idx + 2] as usize;
            } else {
                out[idx + 1] *= factor;
                idx += 2;
            }
        }
        out
    }

    /// Decode a coefficient vector back into a `FittedTree`.
    pub fn from_coef_vec(coef: ArrayView1<f64>) -> Self {
        let mut idx = 0;
        let num_nodes = coef[idx] as usize;
        idx += 1;
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let tag = coef[idx];
            idx += 1;
            if tag == 0.0 {
                let var = coef[idx] as usize;
                idx += 1;
                let point = coef[idx];
                idx += 1;
                let left = coef[idx] as usize;
                idx += 1;
                let right = coef[idx] as usize;
                idx += 1;
                nodes.push(TreeNode::Split {
                    var,
                    point,
                    left,
                    right,
                });
            } else if tag == 2.0 {
                let var = coef[idx] as usize;
                let m = coef[idx + 1] as usize;
                let left_levels: Vec<usize> = (0..m).map(|k| coef[idx + 2 + k] as usize).collect();
                let left = coef[idx + 2 + m] as usize;
                let right = coef[idx + 3 + m] as usize;
                idx += 4 + m;
                nodes.push(TreeNode::SplitSet {
                    var,
                    left_levels,
                    left,
                    right,
                });
            } else {
                let mean = coef[idx];
                idx += 1;
                nodes.push(TreeNode::Leaf { mean });
            }
        }
        FittedTree { nodes }
    }
}

/// Whether `coef` is a well-formed tree encoding for a design with
/// `n_vars` columns: every node tag one `to_coef_vec` writes, every split
/// variable in range, every child index inside the node list, and the
/// walk consuming the vector exactly. [`FittedTree::from_coef_vec`]
/// indexes without checking, so a caller holding coefficients it did not
/// produce itself validates first.
fn valid_tree_coef(coef: &[f64], n_vars: usize) -> bool {
    if coef.is_empty() {
        return false;
    }
    let num_nodes = coef[0];
    if num_nodes < 1.0 || num_nodes.fract() != 0.0 {
        return false;
    }
    let num_nodes = num_nodes as usize;
    let in_range = |v: f64, bound: usize| v >= 0.0 && v.fract() == 0.0 && (v as usize) < bound;
    let mut idx = 1usize;
    for _ in 0..num_nodes {
        let Some(&tag) = coef.get(idx) else {
            return false;
        };
        idx += 1;
        match tag {
            0.0 => {
                let Some(fields) = coef.get(idx..idx + 4) else {
                    return false;
                };
                if !in_range(fields[0], n_vars)
                    || !in_range(fields[2], num_nodes)
                    || !in_range(fields[3], num_nodes)
                {
                    return false;
                }
                idx += 4;
            }
            2.0 => {
                let Some(&m) = coef.get(idx + 1) else {
                    return false;
                };
                if m < 0.0 || m.fract() != 0.0 {
                    return false;
                }
                let m = m as usize;
                let Some(fields) = coef.get(idx..idx + 4 + m) else {
                    return false;
                };
                if !in_range(fields[0], n_vars)
                    || !in_range(fields[2 + m], num_nodes)
                    || !in_range(fields[3 + m], num_nodes)
                {
                    return false;
                }
                idx += 4 + m;
            }
            1.0 => {
                if coef.get(idx).is_none() {
                    return false;
                }
                idx += 1;
            }
            _ => return false,
        }
    }
    idx == coef.len()
}

/// Build a column-major `faer::Mat<f64>` mirror of an `ndarray::Array2<f64>`.
/// One-time cost at prep; pays for itself on the first hot-loop GEMV because
/// faer's matvec is far faster than ndarray's no-BLAS fallback.
fn ndarray2_to_faer(a: &Array2<f64>) -> Mat<f64> {
    let (r, c) = (a.nrows(), a.ncols());
    // Read from a contiguous row-major view with flat indexing, avoiding
    // ndarray's per-element 2D bounds-checked `a[[i, j]]`. `as_standard_layout`
    // is a no-op borrow when `a` is already row-major (the common case).
    let std = a.as_standard_layout();
    let s = std.as_slice().expect("standard layout is contiguous");
    Mat::from_fn(r, c, |i, j| s[i * c + j])
}

/// No-alloc GEMV: writes `a · v` into `buf` (length must equal `a.nrows()`).
/// Zeros `buf` first then uses faer's `matmul` with β=1 (additive).
/// β=0 is broken in faer 0.20 — it silently writes nothing.
#[inline]
fn faer_matvec_into(a: &Mat<f64>, v: &[f64], buf: &mut [f64]) {
    let n = v.len();
    let p = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(buf.len(), p);
    for x in buf.iter_mut() {
        *x = 0.0;
    }
    let v_view = faer::mat::from_column_major_slice::<f64, usize, usize>(v, n, 1);
    let mut out_view = faer::mat::from_column_major_slice_mut::<f64, usize, usize>(buf, p, 1);
    faer::linalg::matmul::matmul(
        &mut out_view,
        a,
        v_view,
        Some(1.0),
        1.0,
        faer::Parallelism::None,
    );
}

/// `buf = a · v` for a borrowed faer matrix.
#[inline]
fn faer_ref_matvec_into(a: faer::MatRef<'_, f64>, v: &[f64], buf: &mut [f64]) {
    debug_assert_eq!(a.ncols(), v.len());
    debug_assert_eq!(a.nrows(), buf.len());
    buf.fill(0.0);
    let v_view = faer::mat::from_column_major_slice::<f64, usize, usize>(v, v.len(), 1);
    let rows = buf.len();
    let mut out_view = faer::mat::from_column_major_slice_mut::<f64, usize, usize>(buf, rows, 1);
    faer::linalg::matmul::matmul(
        &mut out_view,
        a,
        v_view,
        Some(1.0),
        1.0,
        faer::Parallelism::None,
    );
}

/// `design` in column-major (Fortran) order, copied only when it is not
/// already. `BlockBols` stores its design once, in this layout, and faer
/// products read it through [`design_ref`] without a second copy.
fn into_column_major(design: Array2<f64>) -> Array2<f64> {
    if design.t().is_standard_layout() {
        return design;
    }
    let mut out = Array2::zeros(ndarray::ShapeBuilder::f((design.nrows(), design.ncols())));
    out.assign(&design);
    out
}

/// Zero-copy faer view of a column-major design.
fn design_ref(design: &Array2<f64>) -> faer::MatRef<'_, f64> {
    debug_assert!(design.t().is_standard_layout());
    faer::mat::from_column_major_slice::<f64, usize, usize>(
        design
            .as_slice_memory_order()
            .expect("a column-major design is contiguous"),
        design.nrows(),
        design.ncols(),
    )
}

/// `out = Xᵀ (w ∘ u)`, the right-hand side of a weighted least-squares step,
/// read from the design itself; `wu` is an `n`-length scratch buffer. This
/// replaces a stored `p × n` copy of `XᵀW`.
fn weighted_crossprod_into(
    design: &Array2<f64>,
    w: &[f64],
    u: &[f64],
    wu: &mut [f64],
    out: &mut [f64],
) {
    for ((slot, &wi), &ui) in wu.iter_mut().zip(w).zip(u) {
        *slot = wi * ui;
    }
    faer_ref_matvec_into(design_ref(design).transpose(), wu, out);
}

/// Smallest row count for which a [`SparseRows`] copy of a block design is
/// considered. Small fits keep the dense faer route, which is fast enough
/// and keeps the existing small-data tests exercising it.
const SPARSE_ROWS_MIN_ROWS: usize = 2048;

/// Largest nonzero fraction of a block design for which a [`SparseRows`]
/// copy pays (`prepare_block_bols` passes 0.35). B-spline rows carry
/// `degree + 1` non-zeros of `p` columns and factor dummies one, so real
/// block designs sit far below it; a near-dense design would only double
/// the learner's memory.
const SPARSE_ROWS_MAX_DENSITY: f64 = 0.35;

/// Column-compressed (CSC) copy of the *uncentered* block design `S`, kept
/// by [`PreparedLearner::BlockBols`] for the per-step products of the
/// boosting loop (#323).
///
/// The dense `design` `D = S − 1 cᵀ` stays the learner's source of truth;
/// this is a derived cache built before centering is subtracted, and the
/// centered products are recovered algebraically — `Dᵀ v = Sᵀ v − c·sum(v)`
/// and `D β = S β − (c∙β)·1` — by `block_bols_crossprod_into` and
/// `block_bols_matvec_into`. Reading `Sᵀ(w∘u)` and `S β` from the sparse
/// arrays costs O(nnz) per step instead of a dense O(n·p) faer product,
/// which profiling put at over half of LSS/Binomial fit time.
///
/// Only the column-compressed layout is stored. A second, row-compressed
/// copy of the same entries cost 52 bytes per row for a cubic B-spline
/// design (4 non-zeros), 0.52 GB for ten `bbs` terms at 10^6 rows (#738).
#[derive(Clone, Debug)]
pub struct SparseRows {
    /// Row count of the dense original.
    nrows: usize,
    /// Column count of the dense original.
    ncols: usize,
    /// Column offsets, length `ncols + 1`. `Sᵀ v` reads the entries as one
    /// gather-dot per column (#323): a row-wise scatter `out[col] += v·val`
    /// updates the same few accumulators through memory on every row, which
    /// profiling put at about half of the per-step cost of a block learner.
    col_ptr: Vec<u32>,
    /// Row index per stored entry, ascending within each column.
    row_idx: Vec<u32>,
    /// Value per stored entry.
    col_vals: Vec<f64>,
}

impl SparseRows {
    /// Exact column-compressed copy of `d`, or `None` when it is too small
    /// (`n <` `SPARSE_ROWS_MIN_ROWS`) or too dense
    /// (`nnz > max_density · n · p`) for the sparse products to pay. A value
    /// is a non-zero iff it is not exactly `0.0`, the same support
    /// [`crate::linalg::CsrMatrix::try_from_dense`] discovers.
    pub fn from_dense_if_sparse(d: ArrayView2<f64>, max_density: f64) -> Option<Self> {
        let (n, p) = (d.nrows(), d.ncols());
        if n < SPARSE_ROWS_MIN_ROWS || p > u32::MAX as usize || n > u32::MAX as usize {
            return None;
        }
        let nonzeros = d.iter().filter(|&&value| value != 0.0).count();
        if nonzeros > u32::MAX as usize || nonzeros as f64 > max_density * (n as f64) * (p as f64) {
            return None;
        }
        let mut rows = SparseRows {
            nrows: n,
            ncols: p,
            col_ptr: Vec::with_capacity(p + 1),
            row_idx: Vec::with_capacity(nonzeros),
            col_vals: Vec::with_capacity(nonzeros),
        };
        // Each column is filled in ascending row order, so the gather reads
        // `v` forward and the scatter in `matvec_into` writes `out` forward.
        rows.col_ptr.push(0);
        for column in d.columns() {
            for (row, &value) in column.iter().enumerate() {
                if value != 0.0 {
                    rows.row_idx.push(row as u32);
                    rows.col_vals.push(value);
                }
            }
            rows.col_ptr.push(rows.col_vals.len() as u32);
        }
        Some(rows)
    }

    /// `out = Sᵀ v`; `out` is overwritten.
    ///
    /// One gather-dot per column over the column-compressed copy. Each column
    /// sums into four independent accumulators, so consecutive entries do not
    /// wait on one floating-point add chain; the summation order differs from
    /// a row-wise scatter by rounding only.
    pub fn tmatvec_into(&self, v: &[f64], out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.ncols);
        debug_assert_eq!(self.col_ptr.len(), self.ncols + 1);
        for (column, slot) in out.iter_mut().enumerate() {
            let lo = self.col_ptr[column] as usize;
            let hi = self.col_ptr[column + 1] as usize;
            let rows = &self.row_idx[lo..hi];
            let vals = &self.col_vals[lo..hi];
            let mut acc = [0.0f64; 4];
            let (row_chunks, row_rem) = rows.as_chunks::<4>();
            let (val_chunks, val_rem) = vals.as_chunks::<4>();
            for (r4, x4) in row_chunks.iter().zip(val_chunks) {
                for lane in 0..4 {
                    acc[lane] += x4[lane] * v[r4[lane] as usize];
                }
            }
            let mut tail = 0.0f64;
            for (&r, &x) in row_rem.iter().zip(val_rem) {
                tail += x * v[r as usize];
            }
            *slot = (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail;
        }
    }

    /// `out = S β`; every entry of `out` is overwritten.
    ///
    /// `out` is zeroed, then each column adds `β_j · S[r, j]` to the rows `r`
    /// of its entries. Columns are visited in ascending order, so each row's
    /// sum is formed as `((0 + S[r,j₁]β_{j₁}) + S[r,j₂]β_{j₂}) + …` over its
    /// non-zeros in ascending column order: the same operations, in the same
    /// order, as a row-wise dot product, and therefore bit-identical to it.
    pub fn matvec_into(&self, beta: &[f64], out: &mut [f64]) {
        debug_assert_eq!(beta.len(), self.ncols);
        debug_assert_eq!(out.len(), self.nrows);
        out.fill(0.0);
        for (column, &b) in beta.iter().enumerate() {
            let lo = self.col_ptr[column] as usize;
            let hi = self.col_ptr[column + 1] as usize;
            for (&r, &x) in self.row_idx[lo..hi].iter().zip(&self.col_vals[lo..hi]) {
                out[r as usize] += x * b;
            }
        }
    }

    /// Heap bytes retained by the sparse arrays.
    pub fn byte_size(&self) -> usize {
        (self.col_ptr.capacity() + self.row_idx.capacity()) * std::mem::size_of::<u32>()
            + self.col_vals.capacity() * std::mem::size_of::<f64>()
    }
}

/// A [`SparseRows`] copy of a block learner's design, built on first use.
///
/// The copy is read only by the general boosting path's per-step products
/// and by prediction. The Gaussian Gram route never reads it — it works
/// from the dense design and the cross-Grams — so building the copy at
/// preparation made every exact Gaussian fit retain about 46 MB per
/// `bbs` term at 10^6 rows for an index it never touched (#738). A
/// learner whose stored design *is* the design the copy is taken from
/// (its centering is all zeros) therefore defers the build to the first
/// product that needs it; the built copy is exactly the one
/// [`SparseRows::from_dense_if_sparse`] would have built at preparation,
/// so products through it are unchanged. A learner prepared from a
/// design that preparation then centres keeps the eager build through
/// [`SparseRowsCache::prebuilt`]: its copy is taken from the pre-centring
/// design, which is not retained.
#[derive(Debug, Default)]
pub struct SparseRowsCache {
    cell: OnceLock<Option<SparseRows>>,
}

impl SparseRowsCache {
    /// An empty cache; the copy is built on first use.
    pub fn new() -> Self {
        SparseRowsCache {
            cell: OnceLock::new(),
        }
    }

    /// A cache already holding `rows` — or the settled decision that no
    /// copy pays, when `rows` is `None`.
    pub fn prebuilt(rows: Option<SparseRows>) -> Self {
        let cache = SparseRowsCache::new();
        let _ = cache.cell.set(rows);
        cache
    }

    /// The sparse copy of `design`, built on the first call; `None` when
    /// the design is too small or too dense for the sparse products to
    /// pay. `design` must be the design the copy is taken from (the
    /// learner's uncentered design); the result is cached, so later
    /// calls are plain reads.
    pub fn get_or_build(&self, design: ArrayView2<f64>) -> Option<&SparseRows> {
        self.cell
            .get_or_init(|| SparseRows::from_dense_if_sparse(design, SPARSE_ROWS_MAX_DENSITY))
            .as_ref()
    }

    /// Heap bytes of the built copy; 0 while it has not been built.
    pub fn byte_size(&self) -> usize {
        self.cell
            .get()
            .and_then(Option::as_ref)
            .map_or(0, SparseRows::byte_size)
    }

    /// Whether the copy (or the decision that none pays) has been built.
    #[cfg(test)]
    pub(crate) fn is_built(&self) -> bool {
        self.cell.get().is_some()
    }
}

/// `out = Dᵀ (w ∘ u)` for a `BlockBols` design, through the sparse uncentered
/// copy when the learner has one: `Dᵀ v = Sᵀ v − c·sum(v)` with `v = w ∘ u`.
/// `wu` is the `n`-length `w ∘ u` scratch, as in [`weighted_crossprod_into`],
/// which the dense route is equivalent to.
fn block_bols_crossprod_into(
    design: &Array2<f64>,
    sparse: Option<&SparseRows>,
    centering: &[f64],
    w: &[f64],
    u: &[f64],
    wu: &mut [f64],
    out: &mut [f64],
) {
    for ((slot, &wi), &ui) in wu.iter_mut().zip(w).zip(u) {
        *slot = wi * ui;
    }
    match sparse {
        Some(s) => {
            s.tmatvec_into(wu, out);
            let sum: f64 = wu.iter().sum();
            for (slot, &c) in out.iter_mut().zip(centering) {
                *slot -= c * sum;
            }
        }
        None => faer_ref_matvec_into(design_ref(design).transpose(), wu, out),
    }
}

/// `out = D β` for a `BlockBols` design, through the sparse uncentered copy
/// when the learner has one: `D β = S β − (c∙β)·1`.
fn block_bols_matvec_into(
    design: &Array2<f64>,
    sparse: Option<&SparseRows>,
    centering: &[f64],
    beta: &[f64],
    out: &mut [f64],
) {
    match sparse {
        Some(s) => {
            s.matvec_into(beta, out);
            let shift: f64 = centering.iter().zip(beta).map(|(&c, &b)| c * b).sum();
            for slot in out.iter_mut() {
                *slot -= shift;
            }
        }
        None => faer_ref_matvec_into(design_ref(design), beta, out),
    }
}

/// `penalty + XᵀWX`, accumulated over blocks of rows so that no weighted
/// copy of the whole design is formed.
/// Parallelism for one-time setup kernels: weighted Grams, the df→λ spectrum
/// products, and factorizations inside learner preparation. These
/// O(n·p²)/O(p³) kernels are paid once per prepared learner, so they run in
/// parallel, with the fixed task count of [`crate::linalg::FAER_TASKS`] so
/// that the result does not depend on the thread count. Per-iteration
/// boosting kernels keep passing `Parallelism::None` explicitly (#276).
fn setup_parallelism() -> faer::Parallelism<'static> {
    // Learner preparation runs on rayon's pool (mboostrs-py lib.rs `par_iter`);
    // nested faer parallelism there risks stack overflow in recursive algorithms
    // (faer's divide-and-conquer SVD, #518) on the 2MB worker stacks, and only
    // oversubscribes — the outer iterator already saturates all cores.
    if rayon::current_thread_index().is_some() {
        faer::Parallelism::None
    } else {
        crate::linalg::faer_parallelism()
    }
}

fn penalized_weighted_gram(
    design: &Array2<f64>,
    w: ArrayView1<f64>,
    penalty: &Array2<f64>,
) -> Array2<f64> {
    blocked_weighted_gram_faer(design, w, ndarray2_to_faer(penalty))
}

/// `X'WX` accumulated over blocks of rows through faer, seeded with `seed`.
///
/// Column-major designs are read zero-copy; the seed receives the block
/// products in order, so a penalty-seeded call reproduces the exact
/// accumulation order every block learner has always used.
fn blocked_weighted_gram_faer(
    design: &Array2<f64>,
    w: ArrayView1<f64>,
    seed: Mat<f64>,
) -> Array2<f64> {
    const ROWS: usize = 4096;
    let (n, p) = (design.nrows(), design.ncols());
    let x = design_ref(design);
    let mut gram = seed;
    let w_owned;
    let w_all: &[f64] = match w.as_slice() {
        Some(slice) => slice,
        None => {
            w_owned = w.to_vec();
            &w_owned
        }
    };
    let mut scaled = Mat::<f64>::zeros(ROWS.min(n.max(1)), p);
    let mut start = 0;
    while start < n {
        let len = ROWS.min(n - start);
        let rows = x.subrows(start, len);
        let w_rows = &w_all[start..start + len];
        for j in 0..p {
            let column = &design
                .column(j)
                .to_slice()
                .expect("a column-major design has contiguous columns")[start..start + len];
            let out = &mut scaled.col_as_slice_mut(j)[..len];
            for ((slot, &xij), &wi) in out.iter_mut().zip(column).zip(w_rows) {
                *slot = wi * xij;
            }
        }
        faer::linalg::matmul::matmul(
            &mut gram,
            rows.transpose(),
            scaled.as_ref().subrows(0, len),
            Some(1.0),
            1.0,
            setup_parallelism(),
        );
        start += len;
    }
    Array2::from_shape_fn((p, p), |(j, k)| gram.read(j, k))
}

/// `X'WX` for a design of any layout, accumulated over blocks of rows.
fn weighted_gram_blocked(x: ArrayView2<f64>, w: ArrayView1<f64>) -> Array2<f64> {
    use ndarray::s;
    const ROWS: usize = 4096;
    let (n, p) = (x.nrows(), x.ncols());
    let mut acc = Array2::<f64>::zeros((p, p));
    let mut scaled = Array2::<f64>::zeros((ROWS.min(n.max(1)), p));
    let mut start = 0;
    while start < n {
        let end = (start + ROWS).min(n);
        let rows = x.slice(s![start..end, ..]);
        let mut block = scaled.slice_mut(s![..end - start, ..]);
        block.assign(&rows);
        block *= &w.slice(s![start..end]).insert_axis(ndarray::Axis(1));
        ndarray::linalg::general_mat_mul(1.0, &rows.t(), &block, 1.0, &mut acc);
        start = end;
    }
    acc
}

/// Shape-constraint kinds for `bmono`. The constraint is always `Dβ ≥ 0`
/// with `D` chosen per the variant — see [`build_mono_constraint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoConstraint {
    /// `β[i+1] ≥ β[i]`  → spline is non-decreasing.
    Increasing,
    /// `β[i+1] ≤ β[i]`  → spline is non-increasing.
    Decreasing,
    /// Second differences ≥ 0  → spline is convex.
    Convex,
    /// Second differences ≤ 0  → spline is concave.
    Concave,
    /// `β ≥ 0`  → spline is non-negative.
    Positive,
    /// `β ≤ 0`  → spline is non-positive.
    Negative,
    /// Custom constraint matrix D β ≥ 0.
    Constrained,
}

/// Which inner solver to use for `bmono`'s per-step constrained QP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmonoSolver {
    /// R `type = "iterative"` — violation-penalty fixed point. Cheap, ~50
    /// LOC, λ₂=1e6 makes it numerically close to the QP solution on
    /// well-conditioned problems but feasibility is approximate.
    Iterative,
    /// R `type = "quad.prog"` — primal active-set QP. Slower per step (~5-10x)
    /// but returns the exact minimiser of the constrained problem, so β is
    /// strictly KKT-feasible rather than approximately so. Use when the
    /// constraint must be enforced exactly (mis-specified models, sparse
    /// data, downstream uses that depend on feasibility), or when matching
    /// R's own default, which is `quad.prog`.
    ///
    /// Matches R's `mboost:::solveLSEI` (which calls
    /// `quadprog::solve.QP(Dmat=XtX, dvec=Xty, Amat=t(D), bvec=0)`) to
    /// floating-point precision — measured ~5e-10 on the parity fixtures for
    /// every constraint shape.
    ///
    /// Fixed 2026-07-20: an inverted Lagrange-multiplier sign made the
    /// optimality test and the constraint-drop rule the exact opposite of
    /// the KKT condition, stranding the search on a feasible but non-optimal
    /// face (~1e-3 to 4e-3 off on fitted values). It was originally recorded
    /// as affecting only the first-difference shapes, but that was an
    /// artefact of which fixtures happened to trigger it: the faulty branch
    /// is shape-agnostic and a randomised sweep found it wrong on ~10% of
    /// binding cases for the difference shapes and ~23% for
    /// `Positive`/`Negative`. Regression guards:
    /// `bmono_quadprog_matches_exhaustive_active_set_oracle` (exhaustive
    /// active-set enumeration) and `bmono_quadprog_satisfies_kkt_on_real_fixture`.
    QuadProg,
}

/// Build the `(m, p)` constraint matrix `D` so that the desired shape
/// constraint is `Dβ ≥ 0`. R `mboost::bmono` uses the same matrices
/// internally (`differences()` helper in `bmono.R`).
pub fn build_mono_constraint(constraint: MonoConstraint, _p: usize) -> Array2<f64> {
    match constraint {
        MonoConstraint::Increasing => {
            // (p-1, p): row i has -1 at col i and +1 at col i+1
            let p = _p;
            assert!(p >= 2, "Increasing constraint needs p >= 2");
            let mut d = Array2::zeros((p - 1, p));
            for i in 0..p - 1 {
                d[[i, i]] = -1.0;
                d[[i, i + 1]] = 1.0;
            }
            d
        }
        MonoConstraint::Decreasing => {
            let mut d = build_mono_constraint(MonoConstraint::Increasing, _p);
            d.mapv_inplace(|v| -v);
            d
        }
        MonoConstraint::Convex => {
            // (p-2, p): second-difference rows [1, -2, 1]
            let p = _p;
            assert!(p >= 3, "Convex constraint needs p >= 3");
            let mut d = Array2::zeros((p - 2, p));
            for i in 0..p - 2 {
                d[[i, i]] = 1.0;
                d[[i, i + 1]] = -2.0;
                d[[i, i + 2]] = 1.0;
            }
            d
        }
        MonoConstraint::Concave => {
            let mut d = build_mono_constraint(MonoConstraint::Convex, _p);
            d.mapv_inplace(|v| -v);
            d
        }
        MonoConstraint::Positive => Array2::eye(_p),
        MonoConstraint::Negative => {
            let mut d = Array2::<f64>::eye(_p);
            d.mapv_inplace(|v| -v);
            d
        }
        MonoConstraint::Constrained => {
            unreachable!(
                "Constrained must provide external matrix; use build_mono_constraint_with_matrix"
            )
        }
    }
}

/// Build a mono constraint matrix from an external D matrix.
pub fn build_mono_constraint_with_matrix(custom_d: Array2<f64>) -> Array2<f64> {
    custom_d
}

/// Build the cumulative-sum design transform matrix matching R mboost's
/// `bsplines()` `Ts_constraint` handling.
///
/// In R's `bsplines()`, when `Ts_constraint = "decreasing"`, the B-spline
/// design is transformed as `X = -X %*% D` where `D` is a lower-triangular
/// matrix of ones:
///
/// ```text
///     | 1 0 0 ... 0 |
///     | 1 1 0 ... 0 |
/// D = | 1 1 1 ... 0 |
///     | ...         |
///     | 1 1 1 ... 1 |
/// ```
///
/// This embeds the monotonicity constraint in the design: each column `j`
/// of `-X %*% D` is the negated cumulative sum of columns `j..p` of `X`.
/// When `center=TRUE`, the centered design inherits this transform and the
/// fitting is plain ridge (no QP/iterative enforcement needed).
pub fn ts_constraint_cumsum_matrix(p: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            d[[i, j]] = 1.0;
        }
    }
    d
}

/// Apply R's `bsplines()` `Ts_constraint` transform to a design matrix.
///
/// For `"decreasing"`: `X = -X %*% D` (cumulative sum, negated)
/// For `"increasing"`:  `X =  X %*% D` (cumulative sum, unsigned)
/// For `"positive"`:    `X =  X` (identity, non-negativity on coefficients)
/// For `"negative"`:    `X =  X` (identity, non-positivity on coefficients)
/// For others:          `X =  X` (no transform)
pub fn apply_ts_constraint(design: &Array2<f64>, constraint: MonoConstraint) -> Array2<f64> {
    match constraint {
        MonoConstraint::Decreasing => {
            let d = ts_constraint_cumsum_matrix(design.ncols());
            -(design.dot(&d))
        }
        MonoConstraint::Increasing => {
            let d = ts_constraint_cumsum_matrix(design.ncols());
            design.dot(&d)
        }
        _ => design.clone(),
    }
}

// ── B-spline constants (matching R's mboost defaults) ──────────────────

const MBOOST_EPS: f64 = 1e-9;
const MBOOST_LAMBDA_MAX: f64 = 1e15;

/// A base-learner prepared with known observation weights.
///
/// Design matrices are built, penalties computed, and caches populated
/// so that `fit_step` runs in O(n) for scalar or O(p²·n + p³) for block.
///
/// `BmonoBbs` keeps its constraint-solver scratch inline: the variant is
/// large, but boxing it would add a pointer dereference to every candidate
/// solve in the boosting loop and no other variant pays that cost.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum PreparedLearner {
    /// Scalar bols: single-column design (n×1), scalar ridge penalty.
    /// Fast path: closed-form β = (x'Wu) / (x'Wx + λ).
    ScalarBols {
        /// Single-column design vector.
        design: Array1<f64>,
        /// Scalar ridge penalty `λ`.
        penalty: f64,
        /// Weighted transpose `W·x`, cached for the fit-step dot product.
        weighted_design_t: Array1<f64>,
        /// Diagonals needed for the closed-form denominator `xᵀWx + λ`.
        scalar_denom: f64,
        /// Formula name of the learner.
        name: String,
        /// Weighted mean subtracted from the design column.
        centering: f64,
        /// Total weight when this learner carries its own intercept column,
        /// and 0.0 when it does not. Only `hat_matrix` reads it, to decide
        /// whether the projection includes the intercept direction and so
        /// whether the learner costs one degree of freedom or two.
        sumw: f64,
        /// Observation weights for this fit.
        weights: Array1<f64>,
        /// Scratch buffer for the solved coefficients.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values.
        pred_buf: Vec<f64>,
    },
    /// Multi-column ridge learner evaluated on a compact set of occupied
    /// one-dimensional bins. The basis and observation bin index are shared
    /// across resampling folds; only weighted bin sums and the Gram solver are
    /// fold-specific.
    BinnedBlockBols {
        /// Per-observation bin assignment.
        bin_index: Arc<[u32]>,
        /// Compact basis evaluated at bin representatives.
        basis: Arc<Array2<f64>>,
        /// Ridge penalty matrix.
        penalty: Array2<f64>,
        /// Effective smoothing parameter λ this learner's `penalty = λK` was
        /// derived from (df→λ calibration or an explicit `lambda=`), `None`
        /// when the learner was built from a pre-scaled penalty with no
        /// recoverable scalar. `extract("lambda")` reports this.
        lambda: Option<f64>,
        /// Cholesky solver over the compact weighted Gram matrix.
        solver: GramSolver,
        /// Formula name of the learner.
        name: String,
        /// Per-column design means subtracted at fit time.
        centering: Array1<f64>,
        /// Number of observations in the training sample.
        n_observations: usize,
        /// Per-bin sum of `weights * u` for the fit-step rhs.
        bin_weighted_u: Vec<f64>,
        /// Scratch buffer for the weighted-transpose product.
        rhs_buf: Vec<f64>,
        /// Scratch buffer for the solved coefficients.
        beta_buf: Vec<f64>,
        /// Scratch buffer for bin-level fitted values.
        bin_pred_buf: Vec<f64>,
        /// Scratch buffer for observation-level fitted values.
        pred_buf: Vec<f64>,
    },
    /// Scalar `bols` whose training design is constant inside quantile bins.
    ///
    /// The observation-length state is one `u32` bin index per row. Design
    /// values, weighted pseudo-response sums, and the solve all live at the
    /// effective bin count. `pred_buf` stays empty on the engine's selection
    /// path and is allocated lazily only when a caller explicitly requests
    /// [`fit_step_ref`](Self::fit_step_ref); the boosting loop gathers the
    /// winning learner into its one shared prediction buffer instead.
    BinnedScalarBols {
        /// Per-observation bin assignment.
        bin_index: Arc<[u32]>,
        /// Design value at each bin representative.
        bin_values: Arc<[f64]>,
        /// Whether binning is exact (no approximation loss).
        exact: bool,
        /// Scalar ridge penalty `λ`.
        penalty: f64,
        /// Denominator `xᵀWx + λ` in bin-count space.
        scalar_denom: f64,
        /// Formula name of the learner.
        name: String,
        /// Weighted mean of the design column.
        centering: f64,
        /// Whether the design column was weight-centered.
        center: bool,
        /// Total observation weight for this fit.
        sumw: f64,
        /// Number of observations in the training sample.
        n_observations: usize,
        /// Per-bin sum of `weights * u` for the fit-step rhs.
        bin_weighted_u: Vec<f64>,
        /// Scratch buffer for the solved coefficients.
        beta_buf: Vec<f64>,
        /// Scratch buffer for fitted values, allocated lazily.
        pred_buf: Vec<f64>,
    },
    /// Multi-column bols: (n, p_j) design with ridge penalty matrix.
    /// Solved via GramSolver (Cholesky or LU fallback): (X'WX + λK)β = X'Wu
    BlockBols {
        /// Design matrix `(n, p_j)`, column-major. It is the learner's only
        /// copy: faer products read it through `design_ref`, and `X'Wu` is
        /// formed as `Xᵀ(w∘u)` rather than from a stored `X'W`. Keeping one
        /// layout instead of three (ndarray, faer, and faer `X'W`) cut the
        /// exact route's learner memory to a third.
        design: Array2<f64>,
        /// Column-compressed copy of the uncentered design
        /// ([`SparseRows`]), shared because resampling folds clone
        /// learners wholesale. `None` when the design is too small or too
        /// dense for the sparse products to pay, and on cold routes (the
        /// streamed-materialised learner, tests). Depends only on the
        /// design, so `reweight` keeps it; the products apply the
        /// centering algebra so results match the dense route.
        ///
        /// Held as a [`SparseRowsCache`]: when the stored design is the
        /// uncentered design (centering all zeros) the copy is built on
        /// first use, so routes that never read it — the Gaussian Gram
        /// route above all — do not retain it (#738).
        sparse: Option<Arc<SparseRowsCache>>,
        /// Ridge penalty matrix.
        penalty: Array2<f64>,
        /// Effective smoothing parameter λ this learner's `penalty = λK` was
        /// derived from (df→λ calibration or an explicit `lambda=`), `None`
        /// when the learner was built from a pre-scaled penalty with no
        /// recoverable scalar. `extract("lambda")` reports this.
        lambda: Option<f64>,
        /// Cholesky solver over the weighted Gram matrix.
        solver: GramSolver,
        /// Formula name of the learner.
        name: String,
        /// Per-column design means subtracted at fit time.
        centering: Array1<f64>,
        /// Observation weights for this fit.
        weights: Array1<f64>,
        /// The unpenalized weighted Gram `X'WX` as an exact CSR copy, when
        /// preparation already computed it and it is sparse enough for CSR
        /// storage to pay (the uncentered tensor spatial learner: its Gram is
        /// banded). The Gaussian Gram route materializes it as the learner's
        /// self-Gram block instead of recomputing the O(n·p²) product — at
        /// 30 knots that product is the route setup's critical path (#276).
        /// `None` when it was not computed, was dense, or no longer matches
        /// `weights` (a cross-validation refit clears it).
        weighted_gram: Option<crate::linalg::CsrMatrix>,
        /// Scratch buffer for `w ∘ u`, length `n`.
        wu_buf: Vec<f64>,
        /// Scratch buffer for the weighted-transpose product.
        rhs_buf: Vec<f64>,
        /// Scratch buffer for the solved coefficients.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values.
        pred_buf: Vec<f64>,
    },
    /// Monotone-constrained P-spline (R `bmono`, `type = "iterative"`).
    ///
    /// At fit time, solves the constrained QP
    /// min ½ β' A β − b' β s.t. Dβ ≥ 0
    /// where `A = X'WX + λK` (pre-stored) and `b = X'Wu`.
    ///
    /// Algorithm: R's iterative violation-penalty method. At each inner
    /// iteration, augment `A` with `λ₂ · D' diag(V) D` where `V[i] = 1` for
    /// each currently-violated constraint, re-solve, refresh `V`. Converges
    /// when `V` stops changing or `niter` is reached. `λ₂` defaults to `1e6`
    /// (matches R), large enough that violations are penalised to ≈ machine
    /// precision on well-conditioned problems.
    ///
    /// Trade-off vs R's `type = "quad.prog"`: ~50 LOC instead of a full
    /// active-set QP, and good enough for the boosting inner loop where the
    /// constraint is one step in a long path and small projection error is
    /// dominated by mstop noise. A true active-set QP can be added later as a
    /// new variant if needed.
    BmonoBbs {
        /// Design matrix `(n, p_j)`.
        design: Array2<f64>,
        /// `XᵀW` in faer layout.
        wdt_faer: Mat<f64>,
        /// Design in faer layout (column-major).
        design_faer: Mat<f64>,
        /// Penalty matrix `λK`.
        penalty: Array2<f64>,
        /// Precomputed `A = XᵀWX + λK` before constraint augmentation.
        penalised_gram: Array2<f64>,
        /// Solver over `A` (pre-constraint).
        solver_a: GramSolver,
        /// Constraint difference matrix `D`.
        d_mat: Array2<f64>,
        /// Per-column design means subtracted at fit time.
        centering: Array1<f64>,
        /// Formula name of the learner.
        name: String,
        /// Iterative violation-penalty solver state.
        solver: BmonoSolver,
        /// Penalty weight for the constraint augmentation term.
        lambda2: f64,
        /// Iteration cap for the constraint solver.
        niter: usize,
        /// Scratch buffer for the weighted-transpose product.
        rhs_buf: Vec<f64>,
        /// Scratch buffer for the solved coefficients.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values.
        pred_buf: Vec<f64>,
        /// Iterative-solver violation set, reused across candidate solves.
        solver_violations: Vec<bool>,
        /// Iterative-solver violation set, swapped with `solver_violations`.
        solver_violations_new: Vec<bool>,
        /// Iterative-solver augmented matrix `A + λ₂·Σ Dᵥ'Dᵥ`, column-major
        /// `p×p`, factored in place by `cho_solve_inplace` each iteration.
        solver_aug: Vec<f64>,
        /// Iterative-solver in-place solve buffer (`b` in, `x` out).
        solver_x: Vec<f64>,
        /// QuadProg working gradient `g = Aβ − b`.
        qp_g: Vec<f64>,
        /// QuadProg working `A⁻¹g`.
        qp_a_inv_g: Vec<f64>,
        /// QuadProg working set of active constraints.
        qp_active: Vec<usize>,
        /// QuadProg working `A⁻¹D_W'` (p×m column-major scratch; used region
        /// is the first q columns).
        qp_a_inv_dt: Vec<f64>,
        /// QuadProg working Schur complement `H = D_W A⁻¹ D_W'` (dense q×q
        /// column-major scratch at the head of an m×m buffer).
        qp_h: Vec<f64>,
        /// QuadProg working `H⁻¹·rhs` multipliers (length ≤ m).
        qp_lambda: Vec<f64>,
        /// QuadProg working search direction `Δ`.
        qp_delta: Vec<f64>,
        /// faer's Cholesky working memory, shared by the iterative solver's
        /// `p×p` augmented factorization and QuadProg's `q×q` Schur solve, so
        /// sized for the larger of the two.
        chol_scratch: Vec<f64>,
    },
    /// Tree base-learner (R `btree`). At each fit step, finds the best
    /// binary stump split across all input columns and returns node means.
    ///
    /// The coefficient vector encodes the fitted stump:
    /// `coef = [split_var_idx, split_point, left_mean, right_mean]`.
    Btree {
        /// Training data matrix `(n, p)`.
        design: Array2<f64>,
        /// Observation indices sorted by each split column. The design is fixed
        /// across boosting iterations, so these orders can be reused.
        sorted_indices: Vec<Vec<usize>>,
        /// Observation weights (from the outer boosting problem).
        weights: Array1<f64>,
        /// Minimum number of obs in a terminal node (R `minbucket`).
        minbucket: usize,
        /// Minimum number of obs to attempt a split (R `minsplit`).
        minsplit: usize,
        /// Maximum tree depth. 1 = stumps (the R `btree` default).
        maxdepth: usize,
        /// Level count of each categorical design column (codes 0..L-1), 0
        /// for a numeric column; empty when every column is numeric.
        categorical: Vec<usize>,
        /// How a node chooses the column it splits.
        selection: SplitSelection,
        /// How splits are found.
        split_method: SplitMethod,
        /// Optional `by` variable for varying-coefficient models.
        by: Option<Array1<f64>>,
        /// Formula name of the learner.
        name: String,
        /// Scratch buffer for the encoded stump coefficients.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values.
        pred_buf: Vec<f64>,
        /// Precomputed binning for numeric columns (histogram path).
        binning: Vec<Option<QuantileBinning>>,
    },
    /// Adaptive hinge base-learner (`bhinge`, mboostrs extension for kinked
    /// DGPs, issue #279). At each fit step, scans the candidate knots in both
    /// hinge directions, fits every candidate hinge column by weighted least
    /// squares, and keeps the one with the smallest residual SSE — so one
    /// boosting selection adds exactly one hinge term and the final model is
    /// a sum of listable hinges.
    ///
    /// The coefficient vector encodes the selected hinge:
    /// `coef = [knot, direction, coefficient, centering]` where `direction`
    /// is +1 for `max(0, x - t)` and -1 for `max(0, t - x)`. Only the
    /// `coefficient` element is scaled by `nu`/line-search factors; the rest
    /// describe the selected structure.
    Bhinge {
        /// Raw covariate column.
        x: Array1<f64>,
        /// Candidate knots (interior quantiles of `x`), ascending,
        /// deduplicated. Fold-invariant: built from the raw covariate once,
        /// shared by every resampling fold.
        knots: Vec<f64>,
        /// Observation weights (from the outer boosting problem).
        weights: Array1<f64>,
        /// Explicit ridge weight for the hinge coefficient; overrides `df`.
        lambda: Option<f64>,
        /// Target effective degrees of freedom of one selected hinge
        /// coefficient. A one-column hinge fit can absorb at most 1 df, so a
        /// target above that yields an unpenalised fit (lambda = 0) whose
        /// selection-stage complexity is governed by `mstop`/`nu` the way a
        /// tree's is.
        df: f64,
        /// Whether the selected hinge column is weight-centered.
        center: bool,
        /// Optional varying-coefficient multiplier column. The hinge column
        /// is `by * hinge(x)`; centering is the weighted mean of that column,
        /// matching the `bols(x, by = z)` convention.
        by: Option<Array1<f64>>,
        /// Formula name of the learner.
        name: String,
        /// Scratch buffer for the encoded hinge coefficients
        /// `[knot, direction, coefficient, centering]`.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values.
        pred_buf: Vec<f64>,
    },
    /// Kronecker-product base-learner for matrix-response models (R `%O%`).
    ///
    /// Fits a linear array model with design `kronecker(X2, X1)`, where
    /// X1 is `(n_obs, p1)` and X2 is `(n_resp, p2)`. The response `u`
    /// has length `n_obs * n_resp`.
    ///
    /// When X2 = I (the identity), this is equivalent to fitting `n_resp`
    /// independent Ridge regressions with shared design X1 and penalty,
    /// which is the standard Multinomial setup.
    KroneckerBols {
        /// X1: covariate design `(n_obs, p1)`.
        x1: Array2<f64>,
        /// X1 faer representation (column-major).
        x1_faer: Mat<f64>,
        /// W*X1^T in faer format `(p1, n_obs)`.
        wx1t_faer: Mat<f64>,
        /// Gram solver for the marginal X1 problem: `(X1'WX1 + λK1)⁻¹`.
        solver: GramSolver,
        /// X2: class design `(n_resp, p2)`. For Multinomial, this is `(K-1, K-1)` identity.
        x2: Array2<f64>,
        /// Whether X2 is exactly the identity matrix. This is the dominant
        /// Multinomial path and avoids the general Kronecker prediction loops.
        x2_is_identity: bool,
        /// Penalty matrix (already multiplied by lambda).
        penalty: Array2<f64>,
        /// Centering means for X1 columns (length p1).
        centering: Array1<f64>,
        /// Number of observations (n_obs).
        n_obs: usize,
        /// Number of response columns (K-1 for Multinomial).
        n_resp: usize,
        /// Formula name of the learner.
        name: String,
        /// Scratch buffer for the weighted-transpose product.
        rhs_buf: Vec<f64>,
        /// Scratch buffer for the solved coefficients.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values.
        pred_buf: Vec<f64>,
        /// Metadata to rebuild x1 from raw x values for new-data prediction.
        /// `None` for legacy fitted models; `Some(...)` when set after fitting.
        x1_rebuild: Option<KroneckerRebuild>,
    },
    /// A `bbs %X% brandom`-style interaction stored as a Kronecker-sum ridge
    /// system instead of the materialised row-wise tensor.
    ///
    /// The materialised tensor `T[i, l·ng + g] = x_l[i, l]·δ(g, groups[i])`
    /// (`x_l` n×p_l, one-hot right margin with `ng` groups) carries the
    /// Kronecker-sum penalty `λ·kron(K_l, I_ng) + λ·kron(I_p_l, K_r)`. For a
    /// right margin with an identity penalty (brandom, a factor `by=`) the
    /// cross terms vanish and every group's normal equations decouple into its
    /// own `p_l×p_l` ridge system (per group `g`: `gram_g = X_g'WX_g + K_0`
    /// with `rhs_g[l] = Σ_{i∈g} w_i·u_i·x_l[i,l]`), sharing `K_0 = λ·(K_l + I)`:
    ///
    /// Solver cost drops from one `O((p_l·ng)³)` factorization to `ng` small
    /// `O(ng·p_l³)` ones, and retained memory from `O(n·p_l·ng)` for the
    /// materialised tensor to `O(n·p_l)` for the marginal design plus smaller
    /// per-group state. Eigen-decomposable only in ridge form: with `center =
    /// true` (which densifies the columns) or a block-crossing penalty the
    /// structure is absent, detection fails, and the materialised `BlockBols`
    /// path is used instead. Coefficients are kept in the materialised column
    /// order `l·ng + g`, so `coef()`/predictions are identical to the
    /// materialised learner bit-for-bit.
    FactoredBlockBols {
        /// Left marginal design `(n, p_l)`; `x_l[i, l] = T[i, l·ng + groups[i]]`.
        x_l: Array2<f64>,
        /// Per-row right-marginal column (`0..n_groups`).
        groups: Vec<u32>,
        /// Left marginal column count `p_l`.
        p_l: usize,
        /// Right marginal column count `ng`.
        n_groups: usize,
        /// Scaled `p_l×p_l` per-group penalty block `λ·(K_l + I)`, identical
        /// across groups. First diagonal block of the materialised `λK`.
        penalty_block: Array2<f64>,
        /// Cholesky solver per group over `X_g'WX_g + penalty_block`.
        solvers: Vec<GramSolver>,
        /// Formula name of the learner.
        name: String,
        /// Observation weights for this fit.
        weights: Array1<f64>,
        /// Scratch buffer for the per-group weighted-transpose products,
        /// length `n_groups·p_l`, group-major (`[gr·p_l + l]`).
        rhs_buf: Vec<f64>,
        /// Scratch buffer for one group's solve (`p_l`).
        sol_buf: Vec<f64>,
        /// Scratch buffer for the solved coefficients, length `p_l·n_groups`
        /// in the materialised column order `l·ng + g`.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values.
        pred_buf: Vec<f64>,
    },
    /// Streamed tensor-product spline (`bspatial`) learner.
    ///
    /// Stores the two marginal B-spline bases and the optional `by` factor
    /// instead of the materialised `(n, p1*p2)` tensor design. Every `X'WX`,
    /// `X'W(u)` and `Xβ` product is formed row by row from the marginal
    /// supports, so the setup Gram costs `O(n·(d+1)⁴)` (each B-spline row has
    /// only `degree+1` nonzeros) instead of the dense `O(n·(p1·p2)²)`, and no
    /// `n×p` matrix is retained. The products are the exact row-wise
    /// (Khatri–Rao) tensor, *not* the separable `kron(B1'WB1, B2'WB2)` — the
    /// latter is only valid for the full-grid Kronecker design. Results match
    /// the materialised `BlockBols` path to floating-point tolerance; only the
    /// accumulation order of `X'WX` differs.
    BspatialStreamed {
        /// Left marginal basis `(n, p1)`.
        b1: Array2<f64>,
        /// Right marginal basis `(n, p2)`.
        b2: Array2<f64>,
        /// Compact nonzero `(index, value)` support of each row of `b1`.
        supports1: std::sync::Arc<Vec<Vec<(u32, f64)>>>,
        /// Compact nonzero `(index, value)` support of each row of `b2`.
        supports2: std::sync::Arc<Vec<Vec<(u32, f64)>>>,
        /// Optional `by` modifier, length `n`.
        by: Option<Array1<f64>>,
        /// Marginal spectral centering: the two small marginal eigenbases plus
        /// the kept, rescaled eigen-coordinates. `None` when the basis was
        /// not centered, in which case `p_out == p1*p2`.
        centering: Option<BspatialCentering>,
        /// Number of columns of the left marginal.
        p1: usize,
        /// Number of columns of the right marginal.
        p2: usize,
        /// Number of columns of the final (possibly centered) design.
        p_out: usize,
        /// Scaled penalty `λK` in the final space `(p_out, p_out)`. For the
        /// centered path `K` is the identity.
        penalty: Array2<f64>,
        /// Effective λ the penalty was derived from, for `extract("lambda")`.
        lambda: Option<f64>,
        /// Solver over `X'WX + penalty`.
        solver: GramSolver,
        /// Formula name of the learner.
        name: String,
        /// Observation weights of the fit.
        weights: Array1<f64>,
        /// Scratch buffer for the streamed right-hand side `(p_out)`.
        rhs_buf: Vec<f64>,
        /// Scratch buffer for the raw `(p)` right-hand side before centering.
        raw_rhs_buf: Vec<f64>,
        /// Scratch buffer for the raw `(p)` value of `C·beta` before the
        /// streamed prediction pass.
        raw_g_buf: Vec<f64>,
        /// Scratch buffer for the solved coefficients `(p_out)`.
        beta_buf: Vec<f64>,
        /// Scratch buffer for the fitted values `(n)`.
        pred_buf: Vec<f64>,
    },
}

/// Factor decomposition of a learner's hat matrix: H_j = X·M·(XᵀW).
///
/// X is n×p_j, M = (XᵀWX + λK)⁻¹ (p_j×p_j, stored row-major in `hat_inv`), tr_h = tr(H_j).
pub struct HatFactor {
    /// Design matrix X in faer layout.
    pub design: Mat<f64>,
    /// Weighted inverse `M = (XᵀWX + λK)⁻¹`, stored row-major with padding.
    pub hat_inv: Vec<f64>,
    /// Column dimension of `M`, including row padding.
    pub hat_inv_p: usize,
    /// Trace `tr(H_j)` of the learner's hat matrix.
    pub tr_h: f64,
}

/// Deterministic prepared-learner metadata used by [`FitReport`](crate::FitReport).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LearnerReportStats {
    pub scalar_learners: usize,
    pub block_learners: usize,
    pub tree_learners: usize,
    pub cholesky_solvers: usize,
    pub lu_solvers: usize,
    pub regularized_solvers: usize,
    pub max_solver_jitter: Option<f64>,
    pub factor_cache_bytes: usize,
    pub retained_data_bytes: usize,
}

impl LearnerReportStats {
    fn with_solver(mut self, solver: &GramSolver) -> Self {
        if let Some(jitter) = solver.regularization_jitter() {
            self.regularized_solvers += 1;
            self.max_solver_jitter = Some(
                self.max_solver_jitter
                    .map_or(jitter, |current| current.max(jitter)),
            );
        } else if solver.is_cholesky() {
            self.cholesky_solvers += 1;
        } else {
            self.lu_solvers += 1;
        }
        self.factor_cache_bytes += solver.cached_factor_bytes();
        self
    }
}

pub(crate) fn aggregate_report_stats(learners: &[PreparedLearner]) -> LearnerReportStats {
    learners
        .iter()
        .fold(LearnerReportStats::default(), |mut total, learner| {
            let stats = learner.report_stats();
            total.scalar_learners += stats.scalar_learners;
            total.block_learners += stats.block_learners;
            total.tree_learners += stats.tree_learners;
            total.cholesky_solvers += stats.cholesky_solvers;
            total.lu_solvers += stats.lu_solvers;
            total.regularized_solvers += stats.regularized_solvers;
            if let Some(jitter) = stats.max_solver_jitter {
                total.max_solver_jitter = Some(
                    total
                        .max_solver_jitter
                        .map_or(jitter, |current| current.max(jitter)),
                );
            }
            total.factor_cache_bytes += stats.factor_cache_bytes;
            total.retained_data_bytes += stats.retained_data_bytes;
            total
        })
}

impl PreparedLearner {
    /// Materialise the dense (possibly centered) tensor design of a
    /// `BspatialStreamed` learner. Only the hat-matrix / design-matrix paths
    /// need this; the boosting hot path never does.
    fn streamed_dense_design(&self) -> Array2<f64> {
        let PreparedLearner::BspatialStreamed {
            supports1,
            supports2,
            by,
            centering,
            p1,
            p2,
            ..
        } = self
        else {
            unreachable!("streamed_dense_design on a non-streamed learner")
        };
        let centering_dense = centering.as_ref().map(|c| c.dense_matrix(*p1, *p2));
        streamed_dense_design(
            supports1,
            supports2,
            by.as_ref(),
            centering_dense.as_ref(),
            *p1,
            *p2,
        )
    }

    /// Build the equivalent materialised `BlockBols` for a `BspatialStreamed`
    /// learner, so the hat-matrix, design-matrix and AIC paths can reuse the
    /// dense code (never on the boosting hot path).
    fn streamed_to_block_bols(&self) -> PreparedLearner {
        let PreparedLearner::BspatialStreamed {
            penalty,
            lambda,
            name,
            weights,
            p_out,
            ..
        } = self
        else {
            unreachable!("streamed_to_block_bols on a non-streamed learner")
        };
        let design = into_column_major(self.streamed_dense_design());
        let n = design.nrows();
        let gram = penalized_weighted_gram(&design, weights.view(), penalty);
        PreparedLearner::BlockBols {
            design,
            sparse: None,
            penalty: penalty.clone(),
            lambda: *lambda,
            solver: GramSolver::new(&gram),
            name: name.clone(),
            centering: Array1::zeros(*p_out),
            weights: weights.clone(),
            weighted_gram: None,
            wu_buf: vec![0.0; n],
            rhs_buf: vec![0.0; *p_out],
            beta_buf: vec![0.0; *p_out],
            pred_buf: vec![0.0; n],
        }
    }

    /// This learner's coefficient scaled so its prediction is `factor` times
    /// the original. A linear learner multiplies every coefficient; a tree's
    /// coefficient vector encodes its structure, so only the leaf means are
    /// scaled.
    pub fn scale_coef(&self, coef: &[f64], factor: f64) -> Vec<f64> {
        match self {
            PreparedLearner::Btree { .. } => FittedTree::scale_coef_vec(coef, factor),
            // A hinge's `[knot, direction, coefficient, centering]` vector
            // encodes its structure: only the coefficient is rescaled, the
            // way a tree scales its leaf means while keeping its splits.
            PreparedLearner::Bhinge { .. } => {
                let mut scaled = coef.to_vec();
                if scaled.len() == 4 {
                    scaled[2] *= factor;
                } else {
                    for value in scaled.iter_mut() {
                        *value *= factor;
                    }
                }
                scaled
            }
            _ => coef.iter().map(|b| factor * b).collect(),
        }
    }

    pub(crate) fn report_stats(&self) -> LearnerReportStats {
        let mut stats = match self {
            PreparedLearner::ScalarBols { .. } | PreparedLearner::BinnedScalarBols { .. } => {
                LearnerReportStats {
                    scalar_learners: 1,
                    ..LearnerReportStats::default()
                }
            }
            PreparedLearner::BlockBols { solver, .. }
            | PreparedLearner::BinnedBlockBols { solver, .. }
            | PreparedLearner::BspatialStreamed { solver, .. }
            | PreparedLearner::KroneckerBols { solver, .. } => LearnerReportStats {
                block_learners: 1,
                ..LearnerReportStats::default()
            }
            .with_solver(solver),
            PreparedLearner::BmonoBbs { solver_a, .. } => LearnerReportStats {
                block_learners: 1,
                ..LearnerReportStats::default()
            }
            .with_solver(solver_a),
            PreparedLearner::FactoredBlockBols { solvers, .. } => solvers.iter().fold(
                LearnerReportStats {
                    block_learners: 1,
                    ..LearnerReportStats::default()
                },
                |stats, solver| stats.with_solver(solver),
            ),
            PreparedLearner::Btree { .. } => LearnerReportStats {
                tree_learners: 1,
                ..LearnerReportStats::default()
            },
            // A hinge is an adaptive non-linear learner selected once per
            // step, so it is counted with the trees rather than with the
            // fixed-design scalar or block learners.
            PreparedLearner::Bhinge { .. } => LearnerReportStats {
                tree_learners: 1,
                ..LearnerReportStats::default()
            },
        };
        stats.retained_data_bytes = self.retained_data_bytes();
        stats
    }

    /// Dynamic learner data retained during the boosting loop, excluding the
    /// separately reported cached solver factorization.
    pub fn retained_data_bytes(&self) -> usize {
        let f64_bytes = std::mem::size_of::<f64>();
        let usize_bytes = std::mem::size_of::<usize>();
        match self {
            PreparedLearner::ScalarBols {
                design,
                weighted_design_t,
                weights,
                beta_buf,
                pred_buf,
                ..
            } => {
                f64_bytes
                    * (design.len()
                        + weighted_design_t.len()
                        + weights.len()
                        + beta_buf.capacity()
                        + pred_buf.capacity())
            }
            PreparedLearner::BinnedScalarBols {
                bin_index,
                bin_values,
                bin_weighted_u,
                beta_buf,
                pred_buf,
                ..
            } => {
                std::mem::size_of_val(bin_index.as_ref())
                    + f64_bytes
                        * (bin_values.len()
                            + bin_weighted_u.capacity()
                            + beta_buf.capacity()
                            + pred_buf.capacity())
            }
            PreparedLearner::BlockBols {
                design,
                penalty,
                weights,
                weighted_gram,
                sparse,
                wu_buf,
                rhs_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                f64_bytes
                    * (design.len()
                        + penalty.len()
                        + weights.len()
                        + wu_buf.capacity()
                        + rhs_buf.capacity()
                        + beta_buf.capacity()
                        + pred_buf.capacity())
                    + weighted_gram
                        .as_ref()
                        .map_or(0, crate::linalg::CsrMatrix::byte_size)
                    + sparse.as_ref().map_or(0, |cache| cache.byte_size())
            }
            PreparedLearner::BspatialStreamed {
                b1,
                b2,
                supports1,
                supports2,
                by,
                centering,
                penalty,
                weights,
                rhs_buf,
                raw_rhs_buf,
                raw_g_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                let support_entries = supports1.iter().map(Vec::len).sum::<usize>()
                    + supports2.iter().map(Vec::len).sum::<usize>();
                std::mem::size_of::<(u32, f64)>() * support_entries
                    + f64_bytes
                        * (b1.len()
                            + b2.len()
                            + by.as_ref().map_or(0, Array1::len)
                            + centering.as_ref().map_or(0, |c| {
                                c.u1.len() + c.u2.len() + c.kept.capacity() + c.scale.capacity()
                            })
                            + penalty.len()
                            + weights.len()
                            + rhs_buf.capacity()
                            + raw_rhs_buf.capacity()
                            + raw_g_buf.capacity()
                            + beta_buf.capacity()
                            + pred_buf.capacity())
            }
            PreparedLearner::BinnedBlockBols {
                bin_index,
                basis,
                penalty,
                bin_weighted_u,
                rhs_buf,
                beta_buf,
                bin_pred_buf,
                pred_buf,
                ..
            } => {
                std::mem::size_of_val(bin_index.as_ref())
                    + f64_bytes
                        * (basis.len()
                            + penalty.len()
                            + bin_weighted_u.capacity()
                            + rhs_buf.capacity()
                            + beta_buf.capacity()
                            + bin_pred_buf.capacity()
                            + pred_buf.capacity())
            }
            PreparedLearner::BmonoBbs {
                design,
                penalty,
                penalised_gram,
                d_mat,
                rhs_buf,
                beta_buf,
                pred_buf,
                solver_violations,
                solver_violations_new,
                solver_aug,
                solver_x,
                qp_g,
                qp_a_inv_g,
                qp_active,
                qp_a_inv_dt,
                qp_h,
                qp_lambda,
                qp_delta,
                chol_scratch,
                ..
            } => {
                // The constraint solvers' scratch is retained for the life of
                // the learner (that is the point of it), and `qp_a_inv_dt`
                // (p·m) and `qp_h` (m²) are not small next to the design, so
                // they are counted rather than treated as noise.
                f64_bytes
                    * (2 * design.len()
                        + penalty.len()
                        + penalised_gram.len()
                        + d_mat.len()
                        + rhs_buf.capacity()
                        + beta_buf.capacity()
                        + pred_buf.capacity()
                        + solver_aug.capacity()
                        + solver_x.capacity()
                        + qp_g.capacity()
                        + qp_a_inv_g.capacity()
                        + qp_a_inv_dt.capacity()
                        + qp_h.capacity()
                        + qp_lambda.capacity()
                        + qp_delta.capacity()
                        + chol_scratch.capacity())
                    + solver_violations.capacity()
                    + solver_violations_new.capacity()
                    + usize_bytes * qp_active.capacity()
            }
            PreparedLearner::Btree {
                design,
                sorted_indices,
                weights,
                beta_buf,
                pred_buf,
                ..
            } => {
                f64_bytes
                    * (design.len() + weights.len() + beta_buf.capacity() + pred_buf.capacity())
                    + usize_bytes * sorted_indices.iter().map(Vec::capacity).sum::<usize>()
            }
            PreparedLearner::Bhinge {
                x,
                knots,
                weights,
                beta_buf,
                pred_buf,
                ..
            } => {
                f64_bytes
                    * (x.len()
                        + knots.len()
                        + weights.len()
                        + beta_buf.capacity()
                        + pred_buf.capacity())
            }
            PreparedLearner::KroneckerBols {
                x1,
                x2,
                penalty,
                rhs_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                f64_bytes
                    * (3 * x1.len()
                        + x2.len()
                        + penalty.len()
                        + rhs_buf.capacity()
                        + beta_buf.capacity()
                        + pred_buf.capacity())
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                penalty_block,
                weights,
                rhs_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                f64_bytes
                    * (x_l.len()
                        + penalty_block.len()
                        + weights.len()
                        + rhs_buf.capacity()
                        + beta_buf.capacity()
                        + pred_buf.capacity())
                    + std::mem::size_of::<u32>() * groups.capacity()
            }
        }
    }

    /// Number of fitted coefficients in this learner's update.
    pub fn n_parameters(&self) -> usize {
        match self {
            PreparedLearner::ScalarBols { .. } | PreparedLearner::BinnedScalarBols { .. } => 1,
            PreparedLearner::BinnedBlockBols { basis, .. } => basis.ncols(),
            PreparedLearner::BlockBols { design, .. }
            | PreparedLearner::BmonoBbs { design, .. } => design.ncols(),
            PreparedLearner::BspatialStreamed { p_out, .. } => *p_out,
            PreparedLearner::Btree { .. } => 1,
            PreparedLearner::Bhinge { .. } => 1,
            PreparedLearner::KroneckerBols { x1, x2, .. } => x1.ncols() * x2.ncols(),
            PreparedLearner::FactoredBlockBols { p_l, n_groups, .. } => p_l * n_groups,
        }
    }

    /// Number of response entries consumed by one fit step.
    pub fn n_observations(&self) -> usize {
        match self {
            PreparedLearner::ScalarBols { design, .. } => design.len(),
            PreparedLearner::BinnedScalarBols { n_observations, .. } => *n_observations,
            PreparedLearner::BinnedBlockBols { n_observations, .. } => *n_observations,
            PreparedLearner::BlockBols { design, .. }
            | PreparedLearner::BmonoBbs { design, .. }
            | PreparedLearner::Btree { design, .. } => design.nrows(),
            PreparedLearner::BspatialStreamed { b1, .. } => b1.nrows(),
            PreparedLearner::Bhinge { x, .. } => x.len(),

            PreparedLearner::KroneckerBols { n_obs, n_resp, .. } => n_obs * n_resp,
            PreparedLearner::FactoredBlockBols { x_l, .. } => x_l.nrows(),
        }
    }

    /// Observation weights stored by this learner, when it caches them.
    pub fn weights(&self) -> Option<&[f64]> {
        match self {
            PreparedLearner::ScalarBols { weights, .. }
            | PreparedLearner::BlockBols { weights, .. }
            | PreparedLearner::BspatialStreamed { weights, .. }
            | PreparedLearner::Btree { weights, .. }
            | PreparedLearner::Bhinge { weights, .. }
            | PreparedLearner::FactoredBlockBols { weights, .. } => weights.as_slice(),
            _ => None,
        }
    }

    /// Name of this learner's formula term.
    pub fn name(&self) -> &str {
        match self {
            PreparedLearner::ScalarBols { name, .. }
            | PreparedLearner::BinnedScalarBols { name, .. } => name,
            PreparedLearner::BlockBols { name, .. } => name,
            PreparedLearner::BinnedBlockBols { name, .. } => name,
            PreparedLearner::BmonoBbs { name, .. } => name,
            PreparedLearner::Btree { name, .. } => name,
            PreparedLearner::Bhinge { name, .. } => name,
            PreparedLearner::KroneckerBols { name, .. } => name,
            PreparedLearner::BspatialStreamed { name, .. } => name,
            PreparedLearner::FactoredBlockBols { name, .. } => name,
        }
    }

    /// Zero-alloc fit: computes into internal buffers, returns slices.
    ///
    /// The returned slices borrow `self.beta_buf` and `self.pred_buf` respectively,
    /// so they are valid until the next call to `fit_step_ref` on this learner.
    pub fn fit_step_ref(
        &mut self,
        u: ArrayView1<f64>,
        weights: ArrayView1<f64>,
    ) -> (&[f64], &[f64], f64) {
        match self {
            PreparedLearner::ScalarBols {
                design,
                penalty: _,
                weighted_design_t,
                scalar_denom,
                beta_buf,
                pred_buf,
                ..
            } => {
                let numer: f64 = weighted_design_t.dot(&u);
                let beta_val = if *scalar_denom > 0.0 {
                    numer / *scalar_denom
                } else {
                    0.0
                };
                beta_buf.clear();
                beta_buf.push(beta_val);
                let n = design.len();
                if pred_buf.len() != n {
                    pred_buf.resize(n, 0.0);
                }
                for i in 0..n {
                    pred_buf[i] = design[i] * beta_val;
                }
                let sse = weighted_sse_slices(
                    u.as_slice().expect("u contiguous"),
                    pred_buf.as_slice(),
                    weights.as_slice().expect("w contiguous"),
                );
                (beta_buf.as_slice(), pred_buf.as_slice(), sse)
            }
            PreparedLearner::BinnedScalarBols {
                bin_index,
                bin_values,
                exact,
                scalar_denom,
                n_observations,
                bin_weighted_u,
                beta_buf,
                pred_buf,
                ..
            } => {
                let numer = if *exact {
                    let mut total = 0.0f64;
                    for i in 0..*n_observations {
                        total += weights[i] * bin_values[bin_index[i] as usize] * u[i];
                    }
                    total
                } else {
                    bin_weighted_u.fill(0.0);
                    for i in 0..*n_observations {
                        bin_weighted_u[bin_index[i] as usize] += weights[i] * u[i];
                    }
                    bin_values
                        .iter()
                        .zip(bin_weighted_u.iter())
                        .map(|(&value, &sum)| value * sum)
                        .sum::<f64>()
                };
                let beta_val = if *scalar_denom > 0.0 {
                    numer / *scalar_denom
                } else {
                    0.0
                };
                beta_buf.clear();
                beta_buf.push(beta_val);
                if pred_buf.len() != *n_observations {
                    pred_buf.resize(*n_observations, 0.0);
                }
                for i in 0..*n_observations {
                    pred_buf[i] = bin_values[bin_index[i] as usize] * beta_val;
                }
                let sse = weighted_sse_slices(
                    u.as_slice().expect("u contiguous"),
                    pred_buf.as_slice(),
                    weights.as_slice().expect("w contiguous"),
                );
                (beta_buf.as_slice(), pred_buf.as_slice(), sse)
            }
            PreparedLearner::BinnedBlockBols {
                bin_index,
                basis,
                solver,
                n_observations,
                bin_weighted_u,
                rhs_buf,
                beta_buf,
                bin_pred_buf,
                pred_buf,
                ..
            } => {
                bin_weighted_u.fill(0.0);
                for i in 0..*n_observations {
                    bin_weighted_u[bin_index[i] as usize] += weights[i] * u[i];
                }
                for column in 0..basis.ncols() {
                    let mut total = 0.0f64;
                    for bin in 0..basis.nrows() {
                        total += basis[[bin, column]] * bin_weighted_u[bin];
                    }
                    rhs_buf[column] = total;
                }
                beta_buf.fill(0.0);
                solver.solve_into_slice(rhs_buf, beta_buf);
                for bin in 0..basis.nrows() {
                    let mut total = 0.0f64;
                    for column in 0..basis.ncols() {
                        total += basis[[bin, column]] * beta_buf[column];
                    }
                    bin_pred_buf[bin] = total;
                }
                if pred_buf.len() != *n_observations {
                    pred_buf.resize(*n_observations, 0.0);
                }
                for i in 0..*n_observations {
                    pred_buf[i] = bin_pred_buf[bin_index[i] as usize];
                }
                let sse = weighted_sse_slices(
                    u.as_slice().expect("u contiguous"),
                    pred_buf.as_slice(),
                    weights.as_slice().expect("w contiguous"),
                );
                (beta_buf.as_slice(), pred_buf.as_slice(), sse)
            }
            PreparedLearner::BlockBols {
                solver,
                design,
                sparse,
                centering,
                weights: stored_weights,
                wu_buf,
                rhs_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let stored = stored_weights.as_slice().expect("weights contiguous");
                block_bols_crossprod_into(
                    design,
                    sparse
                        .as_ref()
                        .and_then(|cache| cache.get_or_build(design.view())),
                    centering.as_slice().expect("centering contiguous"),
                    stored,
                    u_slice,
                    wu_buf,
                    rhs_buf,
                );
                beta_buf.fill(0.0);
                solver.solve_into_slice(rhs_buf, beta_buf);
                block_bols_matvec_into(
                    design,
                    sparse
                        .as_ref()
                        .and_then(|cache| cache.get_or_build(design.view())),
                    centering.as_slice().expect("centering contiguous"),
                    beta_buf,
                    pred_buf,
                );
                let w_slice = weights.as_slice().expect("weights must be contiguous");
                let sse = weighted_sse_slices(u_slice, pred_buf, w_slice);
                (beta_buf.as_slice(), pred_buf.as_slice(), sse)
            }
            PreparedLearner::BspatialStreamed {
                supports1,
                supports2,
                by,
                centering,
                p1,
                p2,
                solver,
                weights: stored_weights,
                rhs_buf,
                raw_rhs_buf,
                raw_g_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let stored = stored_weights.as_slice().expect("weights contiguous");
                streamed_raw_rhs(
                    supports1,
                    supports2,
                    by.as_ref(),
                    stored,
                    u_slice,
                    *p2,
                    raw_rhs_buf,
                );
                match centering {
                    Some(c) => c.rhs_transform(raw_rhs_buf, *p1, *p2, rhs_buf),
                    None => rhs_buf.copy_from_slice(raw_rhs_buf),
                }
                beta_buf.fill(0.0);
                solver.solve_into_slice(rhs_buf, beta_buf);
                let g: &[f64] = match centering {
                    Some(c) => {
                        c.back_transform(beta_buf, *p1, *p2, raw_g_buf);
                        raw_g_buf.as_slice()
                    }
                    None => beta_buf.as_slice(),
                };
                streamed_predict(supports1, supports2, by.as_ref(), g, *p2, pred_buf);
                let w_slice = weights.as_slice().expect("weights must be contiguous");
                let sse = weighted_sse_slices(u_slice, pred_buf, w_slice);
                (beta_buf.as_slice(), pred_buf.as_slice(), sse)
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                n_groups,
                solvers,
                rhs_buf,
                sol_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let n = x_l.nrows();
                rhs_buf.fill(0.0);
                for i in 0..n {
                    let gr = groups[i] as usize;
                    let wi = weights[i];
                    let ui = u_slice[i];
                    for l in 0..*p_l {
                        rhs_buf[gr * *p_l + l] += wi * x_l[[i, l]] * ui;
                    }
                }
                beta_buf.fill(0.0);
                for gr in 0..*n_groups {
                    let rhs = &rhs_buf[gr * *p_l..(gr + 1) * *p_l];
                    solvers[gr].solve_into_slice(rhs, sol_buf);
                    for l in 0..*p_l {
                        beta_buf[l * *n_groups + gr] = sol_buf[l];
                    }
                }
                for i in 0..n {
                    let gr = groups[i] as usize;
                    let mut value = 0.0;
                    for l in 0..*p_l {
                        value += x_l[[i, l]] * beta_buf[l * *n_groups + gr];
                    }
                    pred_buf[i] = value;
                }
                let w_slice = weights.as_slice().expect("weights must be contiguous");
                let sse = weighted_sse_slices(u_slice, pred_buf, w_slice);
                (beta_buf.as_slice(), pred_buf.as_slice(), sse)
            }
            PreparedLearner::BmonoBbs {
                wdt_faer,
                design_faer,
                penalised_gram,
                solver_a,
                d_mat,
                solver,
                lambda2,
                niter,
                rhs_buf,
                beta_buf,
                pred_buf,
                solver_violations,
                solver_violations_new,
                solver_aug,
                solver_x,
                qp_g,
                qp_a_inv_g,
                qp_active,
                qp_a_inv_dt,
                qp_h,
                qp_lambda,
                qp_delta,
                chol_scratch,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                faer_matvec_into(wdt_faer, u_slice, rhs_buf);
                let p = penalised_gram.nrows();
                let beta_slice = &mut beta_buf[..p];
                let rhs_view = ArrayView1::from(rhs_buf.as_slice());
                match solver {
                    BmonoSolver::Iterative => solve_bmono_iterative_into(
                        solver_a,
                        penalised_gram.view(),
                        d_mat.view(),
                        rhs_view.as_slice().expect("rhs contiguous"),
                        *lambda2,
                        *niter,
                        beta_slice,
                        solver_violations,
                        solver_violations_new,
                        solver_aug.as_mut_slice(),
                        solver_x.as_mut_slice(),
                        chol_scratch.as_mut_slice(),
                    ),
                    BmonoSolver::QuadProg => solve_bmono_quadprog_into(
                        solver_a,
                        penalised_gram.view(),
                        d_mat.view(),
                        rhs_view.as_slice().expect("rhs contiguous"),
                        1e-10,
                        4 * d_mat.nrows().max(8),
                        beta_slice,
                        qp_g.as_mut_slice(),
                        qp_a_inv_g.as_mut_slice(),
                        qp_active,
                        qp_a_inv_dt.as_mut_slice(),
                        qp_h.as_mut_slice(),
                        qp_lambda.as_mut_slice(),
                        qp_delta.as_mut_slice(),
                        chol_scratch.as_mut_slice(),
                    ),
                };
                faer_matvec_into(design_faer, beta_slice, pred_buf);
                let w_slice = weights.as_slice().expect("weights must be contiguous");
                let sse = weighted_sse_slices(u_slice, pred_buf, w_slice);
                (&beta_buf[..p], pred_buf.as_slice(), sse)
            }
            PreparedLearner::Btree {
                design,
                sorted_indices,
                weights: w,
                minbucket,
                minsplit,
                maxdepth,
                categorical,
                selection,
                split_method,
                by,
                beta_buf,
                pred_buf,
                binning,
                ..
            } => {
                let target = btree_target(u, by.as_ref());
                let (coef, pred) = fit_tree_with_sorted(
                    design.view(),
                    match &target {
                        Some(target) => target.view(),
                        None => u.view(),
                    },
                    w.view(),
                    sorted_indices,
                    categorical,
                    *selection,
                    *split_method,
                    binning,
                    *minbucket,
                    *minsplit,
                    *maxdepth,
                );
                let pred_by = if let Some(by_vec) = by {
                    &pred * &*by_vec
                } else {
                    pred
                };
                let sse = weighted_sse(u, pred_by.view(), weights);
                beta_buf.clear();
                beta_buf.extend_from_slice(coef.as_slice().expect("coef must be contiguous"));
                let n = pred_by.len();
                if pred_buf.len() != n {
                    pred_buf.resize(n, 0.0);
                }
                pred_buf[..n].copy_from_slice(pred_by.as_slice().expect("pred_by contiguous"));
                (beta_buf.as_slice(), &pred_buf[..n], sse)
            }
            PreparedLearner::Bhinge {
                x,
                by,
                knots,
                lambda,
                center,
                beta_buf,
                pred_buf,
                ..
            } => {
                let sse = fit_bhinge(
                    x.as_slice().expect("x must be contiguous"),
                    by.as_ref()
                        .map(|values| values.as_slice().expect("by contiguous")),
                    u,
                    weights,
                    knots,
                    lambda.unwrap_or(0.0),
                    *center,
                    beta_buf,
                    pred_buf,
                );
                let n = x.len();
                (beta_buf.as_slice(), &pred_buf[..n], sse)
            }
            PreparedLearner::KroneckerBols {
                x1,
                x1_faer: _,
                wx1t_faer: _,
                solver,
                x2,
                x2_is_identity,
                n_obs,
                n_resp,
                rhs_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let w_slice = weights.as_slice().expect("weights must be contiguous");
                let p1 = x1.ncols();
                let p2 = x2.ncols();
                // β is stored as p1 * p2 flat vector (column-major: β[:,k] for k=0..p2)
                beta_buf.fill(0.0);
                // For each response dimension k, solve (X1'WX1 + λK) β_k = X1' W u_k
                // where u_k = u[k::n_resp] (strided) and W are the obs weights.
                for k in 0..p2 {
                    // Build RHS: X1' * (W * u_k)
                    // u_k[i] = u_slice[i * n_resp + k]
                    // w_k[i] = w_slice[i * n_resp + k] (expanded weights, all same per obs)
                    let rhs_offset = rhs_buf.len().min(p1);
                    let rhs = &mut rhs_buf[..rhs_offset];
                    rhs.fill(0.0);
                    for j in 0..p1 {
                        let mut s = 0.0f64;
                        for i in 0..*n_obs {
                            let wi = w_slice[i * *n_resp + k];
                            let ui = u_slice[i * *n_resp + k];
                            s += wi * x1[[i, j]] * ui;
                        }
                        rhs[j] = s;
                    }
                    // Solve for β_k
                    let beta_k = &mut beta_buf[k * p1..(k + 1) * p1];
                    beta_k.copy_from_slice(rhs);
                    solver.solve_into_slice(rhs, beta_k);
                }
                // Predict: vec(X1 * β * X2^T) — row-major
                // For X2=I: pred[i*n_resp+k] = sum_j X1[i,j] * β[j,k]
                // General case: pred[i*n_resp+k] = sum_j sum_l X1[i,j] * β[j*p2+l] * X2[k,l]
                pred_buf.fill(0.0);
                if *x2_is_identity {
                    debug_assert_eq!(p2, *n_resp);
                    for i in 0..*n_obs {
                        let row_offset = i * *n_resp;
                        for k in 0..*n_resp {
                            let beta_k = &beta_buf[k * p1..(k + 1) * p1];
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                s += x1[[i, j]] * beta_k[j];
                            }
                            pred_buf[row_offset + k] = s;
                        }
                    }
                } else {
                    for i in 0..*n_obs {
                        for kk in 0..*n_resp {
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                for l in 0..p2 {
                                    s += x1[[i, j]] * beta_buf[l * p1 + j] * x2[[kk, l]];
                                }
                            }
                            pred_buf[i * *n_resp + kk] = s;
                        }
                    }
                }
                let sse = weighted_sse_slices(u_slice, pred_buf, w_slice);
                (beta_buf.as_slice(), pred_buf.as_slice(), sse)
            }
        }
    }

    /// Fit this learner to pseudo-response `u` with observation weights `w`.
    ///
    /// Returns `(coefficient, prediction on training data, weighted SSE vs u)`.
    /// Selection uses weighted SSE — the only scoring rule in classical mboost.
    pub fn fit_step(
        &mut self,
        u: ArrayView1<f64>,
        weights: ArrayView1<f64>,
    ) -> (Array1<f64>, Array1<f64>, f64) {
        match self {
            PreparedLearner::ScalarBols {
                design,
                penalty: _,
                weighted_design_t,
                scalar_denom,
                ..
            } => {
                let numer: f64 = weighted_design_t.dot(&u);
                let beta_val = if *scalar_denom > 0.0 {
                    numer / *scalar_denom
                } else {
                    0.0
                };
                let beta = Array1::from_vec(vec![beta_val]);
                let pred = &*design * beta_val;
                let sse = weighted_sse(u, pred.view(), weights);
                (beta, pred, sse)
            }
            PreparedLearner::BinnedScalarBols { .. } | PreparedLearner::BinnedBlockBols { .. } => {
                let (beta, pred, sse) = self.fit_step_ref(u, weights);
                (
                    Array1::from_vec(beta.to_vec()),
                    Array1::from_vec(pred.to_vec()),
                    sse,
                )
            }
            PreparedLearner::BlockBols {
                solver,
                design,
                weights: stored_weights,
                wu_buf,
                rhs_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let stored = stored_weights.as_slice().expect("weights contiguous");
                weighted_crossprod_into(design, stored, u_slice, wu_buf, rhs_buf);
                beta_buf.fill(0.0);
                solver.solve_into_slice(rhs_buf, beta_buf);
                let beta = Array1::from_vec(beta_buf.clone());
                faer_ref_matvec_into(design_ref(design), beta_buf, pred_buf);
                let w_slice = weights.as_slice().expect("weights must be contiguous");
                let sse = weighted_sse_slices(u_slice, pred_buf, w_slice);
                let pred = Array1::from_vec(pred_buf.clone());
                (beta, pred, sse)
            }
            PreparedLearner::BmonoBbs {
                wdt_faer,
                design_faer,
                penalised_gram,
                solver_a,
                d_mat,
                solver,
                lambda2,
                niter,
                rhs_buf,
                beta_buf,
                pred_buf,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                faer_matvec_into(wdt_faer, u_slice, rhs_buf);
                let rhs_view = ArrayView1::from(rhs_buf.as_slice());
                let beta = match solver {
                    BmonoSolver::Iterative => solve_bmono_iterative(
                        solver_a,
                        penalised_gram.view(),
                        d_mat.view(),
                        rhs_view,
                        *lambda2,
                        *niter,
                    ),
                    BmonoSolver::QuadProg => solve_bmono_quadprog(
                        solver_a,
                        penalised_gram.view(),
                        d_mat.view(),
                        rhs_view,
                        1e-10,
                        4 * d_mat.nrows().max(8),
                    ),
                };
                beta_buf.copy_from_slice(beta.as_slice().expect("beta must be contiguous"));
                faer_matvec_into(design_faer, beta_buf, pred_buf);
                let w_slice = weights.as_slice().expect("weights must be contiguous");
                let sse = weighted_sse_slices(u_slice, pred_buf, w_slice);
                let pred = Array1::from_vec(pred_buf.clone());
                (beta, pred, sse)
            }
            PreparedLearner::Btree {
                design,
                sorted_indices,
                weights: w,
                minbucket,
                minsplit,
                maxdepth,
                categorical,
                selection,
                split_method,
                by,
                binning,
                ..
            } => {
                let target = btree_target(u, by.as_ref());
                let (coef, pred) = fit_tree_with_sorted(
                    design.view(),
                    match &target {
                        Some(target) => target.view(),
                        None => u.view(),
                    },
                    w.view(),
                    sorted_indices,
                    categorical,
                    *selection,
                    *split_method,
                    binning,
                    *minbucket,
                    *minsplit,
                    *maxdepth,
                );
                let pred_by = if let Some(by_vec) = by {
                    &pred * &*by_vec
                } else {
                    pred
                };
                let sse = weighted_sse(u, pred_by.view(), weights);
                (coef, pred_by, sse)
            }
            PreparedLearner::Bhinge { .. } => {
                // Delegate to fit_step_ref and clone the results
                let (b, p, sse) = self.fit_step_ref(u, weights);
                (
                    Array1::from_vec(b.to_vec()),
                    Array1::from_vec(p.to_vec()),
                    sse,
                )
            }
            PreparedLearner::KroneckerBols { .. } | PreparedLearner::BspatialStreamed { .. } => {
                // Delegate to fit_step_ref and clone the results
                let (b, p, sse) = self.fit_step_ref(u, weights);
                (
                    Array1::from_vec(b.to_vec()),
                    Array1::from_vec(p.to_vec()),
                    sse,
                )
            }
            PreparedLearner::FactoredBlockBols { .. } => {
                // Delegate to fit_step_ref and clone the results
                let (b, p, sse) = self.fit_step_ref(u, weights);
                (
                    Array1::from_vec(b.to_vec()),
                    Array1::from_vec(p.to_vec()),
                    sse,
                )
            }
        }
    }

    /// SSE by identity: `wu2 - beta'rhs - beta'P*beta` (O(p^2) instead of O(n)).
    /// BmonoBbs uses the full form `wu2 - 2*beta'rhs + beta'G*beta` instead,
    /// since a constrained beta does not satisfy the ridge stationarity the
    /// shortcut relies on (see the BmonoBbs arm below).
    ///
    /// Returns `(beta_slice, sse)` where `beta_slice` borrows `self.beta_buf`.
    /// Used for selection only — the winner's prediction is computed separately
    /// via `predict_into`. Falls back to explicit SSE for Btree only (no
    /// closed-form penalty contribution for the tree split).
    pub fn fit_step_select(
        &mut self,
        u: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        wu2: f64,
    ) -> (&[f64], f64) {
        match self {
            PreparedLearner::ScalarBols {
                weighted_design_t,
                scalar_denom,
                beta_buf,
                penalty,
                ..
            } => {
                let numer: f64 = weighted_design_t.dot(&u);
                let beta_val = if *scalar_denom > 0.0 {
                    numer / *scalar_denom
                } else {
                    0.0
                };
                beta_buf.clear();
                beta_buf.push(beta_val);
                let sse = if *scalar_denom > 0.0 {
                    wu2 - beta_val * numer - *penalty * beta_val * beta_val
                } else {
                    wu2
                };
                (beta_buf.as_slice(), sse)
            }
            PreparedLearner::BinnedBlockBols {
                bin_index,
                basis,
                penalty,
                solver,
                n_observations,
                bin_weighted_u,
                rhs_buf,
                beta_buf,
                ..
            } => {
                bin_weighted_u.fill(0.0);
                for i in 0..*n_observations {
                    bin_weighted_u[bin_index[i] as usize] += weights[i] * u[i];
                }
                for column in 0..basis.ncols() {
                    let mut total = 0.0f64;
                    for bin in 0..basis.nrows() {
                        total += basis[[bin, column]] * bin_weighted_u[bin];
                    }
                    rhs_buf[column] = total;
                }
                beta_buf.fill(0.0);
                solver.solve_into_slice(rhs_buf, beta_buf);
                let p = beta_buf.len();
                let mut beta_dot_rhs = 0.0f64;
                let mut beta_dot_pbeta = 0.0f64;
                for row in 0..p {
                    beta_dot_rhs += beta_buf[row] * rhs_buf[row];
                    let mut penalty_row = 0.0f64;
                    for column in 0..p {
                        penalty_row += penalty[[row, column]] * beta_buf[column];
                    }
                    beta_dot_pbeta += beta_buf[row] * penalty_row;
                }
                (beta_buf.as_slice(), wu2 - beta_dot_rhs - beta_dot_pbeta)
            }
            PreparedLearner::BinnedScalarBols {
                bin_index,
                bin_values,
                exact,
                penalty,
                scalar_denom,
                n_observations,
                bin_weighted_u,
                beta_buf,
                ..
            } => {
                let numer = if *exact {
                    let mut total = 0.0f64;
                    for i in 0..*n_observations {
                        total += weights[i] * bin_values[bin_index[i] as usize] * u[i];
                    }
                    total
                } else {
                    bin_weighted_u.fill(0.0);
                    for i in 0..*n_observations {
                        bin_weighted_u[bin_index[i] as usize] += weights[i] * u[i];
                    }
                    bin_values
                        .iter()
                        .zip(bin_weighted_u.iter())
                        .map(|(&value, &sum)| value * sum)
                        .sum::<f64>()
                };
                let beta_val = if *scalar_denom > 0.0 {
                    numer / *scalar_denom
                } else {
                    0.0
                };
                beta_buf.clear();
                beta_buf.push(beta_val);
                let sse = if *scalar_denom > 0.0 {
                    wu2 - beta_val * numer - *penalty * beta_val * beta_val
                } else {
                    wu2
                };
                (beta_buf.as_slice(), sse)
            }
            PreparedLearner::BlockBols {
                solver,
                design,
                sparse,
                centering,
                weights: stored_weights,
                wu_buf,
                rhs_buf,
                beta_buf,
                penalty,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let stored = stored_weights.as_slice().expect("weights contiguous");
                block_bols_crossprod_into(
                    design,
                    sparse
                        .as_ref()
                        .and_then(|cache| cache.get_or_build(design.view())),
                    centering.as_slice().expect("centering contiguous"),
                    stored,
                    u_slice,
                    wu_buf,
                    rhs_buf,
                );
                beta_buf.fill(0.0);
                solver.solve_into_slice(rhs_buf, beta_buf);
                let p = beta_buf.len();
                let mut beta_dot_rhs = 0.0f64;
                let mut beta_dot_pbeta = 0.0f64;
                for i in 0..p {
                    beta_dot_rhs += beta_buf[i] * rhs_buf[i];
                    let mut row_sum = 0.0f64;
                    for j in 0..p {
                        row_sum += penalty[[i, j]] * beta_buf[j];
                    }
                    beta_dot_pbeta += beta_buf[i] * row_sum;
                }
                let sse = wu2 - beta_dot_rhs - beta_dot_pbeta;
                (beta_buf.as_slice(), sse)
            }
            PreparedLearner::BspatialStreamed {
                supports1,
                supports2,
                by,
                centering,
                p1,
                p2,
                solver,
                weights: stored_weights,
                rhs_buf,
                raw_rhs_buf,
                beta_buf,
                penalty,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let stored = stored_weights.as_slice().expect("weights contiguous");
                streamed_raw_rhs(
                    supports1,
                    supports2,
                    by.as_ref(),
                    stored,
                    u_slice,
                    *p2,
                    raw_rhs_buf,
                );
                match centering {
                    Some(c) => c.rhs_transform(raw_rhs_buf, *p1, *p2, rhs_buf),
                    None => rhs_buf.copy_from_slice(raw_rhs_buf),
                }
                beta_buf.fill(0.0);
                solver.solve_into_slice(rhs_buf, beta_buf);
                let p_out = beta_buf.len();
                let mut beta_dot_rhs = 0.0f64;
                let mut beta_dot_pbeta = 0.0f64;
                for i in 0..p_out {
                    beta_dot_rhs += beta_buf[i] * rhs_buf[i];
                    let mut row_sum = 0.0f64;
                    for j in 0..p_out {
                        row_sum += penalty[[i, j]] * beta_buf[j];
                    }
                    beta_dot_pbeta += beta_buf[i] * row_sum;
                }
                let sse = wu2 - beta_dot_rhs - beta_dot_pbeta;
                (beta_buf.as_slice(), sse)
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                n_groups,
                solvers,
                penalty_block,
                rhs_buf,
                sol_buf,
                beta_buf,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let n = x_l.nrows();
                rhs_buf.fill(0.0);
                for i in 0..n {
                    let gr = groups[i] as usize;
                    let wi = weights[i];
                    let ui = u_slice[i];
                    for l in 0..*p_l {
                        rhs_buf[gr * *p_l + l] += wi * x_l[[i, l]] * ui;
                    }
                }
                beta_buf.fill(0.0);
                let mut beta_dot_rhs = 0.0f64;
                let mut beta_dot_pbeta = 0.0f64;
                for gr in 0..*n_groups {
                    let rhs = &rhs_buf[gr * *p_l..(gr + 1) * *p_l];
                    solvers[gr].solve_into_slice(rhs, sol_buf);
                    for l in 0..*p_l {
                        let b = sol_buf[l];
                        beta_buf[l * *n_groups + gr] = b;
                        beta_dot_rhs += b * rhs[l];
                    }
                    for l in 0..*p_l {
                        let mut row_sum = 0.0f64;
                        for j in 0..*p_l {
                            row_sum += penalty_block[[l, j]] * sol_buf[j];
                        }
                        beta_dot_pbeta += sol_buf[l] * row_sum;
                    }
                }
                (beta_buf.as_slice(), wu2 - beta_dot_rhs - beta_dot_pbeta)
            }
            PreparedLearner::BmonoBbs {
                wdt_faer,
                penalised_gram,
                penalty,
                solver_a,
                d_mat,
                solver,
                lambda2,
                niter,
                rhs_buf,
                beta_buf,
                solver_violations,
                solver_violations_new,
                solver_aug,
                solver_x,
                qp_g,
                qp_a_inv_g,
                qp_active,
                qp_a_inv_dt,
                qp_h,
                qp_lambda,
                qp_delta,
                chol_scratch,
                ..
            } => {
                let u_slice = u.as_slice().expect("u must be contiguous");
                let _w_slice = weights.as_slice().expect("weights must be contiguous");
                faer_matvec_into(wdt_faer, u_slice, rhs_buf);
                let rhs_view = ArrayView1::from(rhs_buf.as_slice());
                let p = penalised_gram.nrows();
                let beta_slice = &mut beta_buf[..p];
                match solver {
                    BmonoSolver::Iterative => solve_bmono_iterative_into(
                        solver_a,
                        penalised_gram.view(),
                        d_mat.view(),
                        rhs_view.as_slice().expect("rhs contiguous"),
                        *lambda2,
                        *niter,
                        beta_slice,
                        solver_violations,
                        solver_violations_new,
                        solver_aug.as_mut_slice(),
                        solver_x.as_mut_slice(),
                        chol_scratch.as_mut_slice(),
                    ),
                    BmonoSolver::QuadProg => solve_bmono_quadprog_into(
                        solver_a,
                        penalised_gram.view(),
                        d_mat.view(),
                        rhs_view.as_slice().expect("rhs contiguous"),
                        1e-10,
                        4 * d_mat.nrows().max(8),
                        beta_slice,
                        qp_g.as_mut_slice(),
                        qp_a_inv_g.as_mut_slice(),
                        qp_active,
                        qp_a_inv_dt.as_mut_slice(),
                        qp_h.as_mut_slice(),
                        qp_lambda.as_mut_slice(),
                        qp_delta.as_mut_slice(),
                        chol_scratch.as_mut_slice(),
                    ),
                };
                // SSE = wu2 - 2*beta'rhs + beta'G*beta, G = A - P (full form:
                // the ridge-stationarity shortcut beta'Gbeta = beta'rhs -
                // beta'Pbeta used by BlockBols is invalid here because a
                // constrained bmono beta does not satisfy stationarity).
                // O(p^2), no O(n) predict + weighted-SSE pass.
                let mut beta_rhs = 0.0f64;
                let mut beta_a_beta = 0.0f64;
                let mut beta_p_beta = 0.0f64;
                for i in 0..p {
                    beta_rhs += beta_buf[i] * rhs_buf[i];
                    let mut a_row = 0.0f64;
                    let mut p_row = 0.0f64;
                    for k in 0..p {
                        a_row += penalised_gram[[i, k]] * beta_buf[k];
                        p_row += penalty[[i, k]] * beta_buf[k];
                    }
                    beta_a_beta += beta_buf[i] * a_row;
                    beta_p_beta += beta_buf[i] * p_row;
                }
                let sse = wu2 - 2.0 * beta_rhs + (beta_a_beta - beta_p_beta);
                (&beta_buf[..p], sse)
            }
            PreparedLearner::Btree {
                design,
                sorted_indices,
                weights: w,
                minbucket,
                minsplit,
                maxdepth,
                categorical,
                selection,
                split_method,
                by,
                beta_buf,
                pred_buf,
                binning,
                ..
            } => {
                let _w_slice = weights.as_slice().expect("weights must be contiguous");
                let target = btree_target(u, by.as_ref());
                let (coef, pred) = fit_tree_with_sorted(
                    design.view(),
                    match &target {
                        Some(target) => target.view(),
                        None => u.view(),
                    },
                    w.view(),
                    sorted_indices,
                    categorical,
                    *selection,
                    *split_method,
                    binning,
                    *minbucket,
                    *minsplit,
                    *maxdepth,
                );
                let pred_by = if let Some(by_vec) = by {
                    &pred * &*by_vec
                } else {
                    pred
                };
                let sse = weighted_sse(u, pred_by.view(), weights);
                beta_buf.clear();
                beta_buf.extend_from_slice(coef.as_slice().expect("coef must be contiguous"));
                let n = pred_by.len();
                if pred_buf.len() != n {
                    pred_buf.resize(n, 0.0);
                }
                pred_buf[..n].copy_from_slice(pred_by.as_slice().expect("pred_by contiguous"));
                (beta_buf.as_slice(), sse)
            }
            PreparedLearner::Bhinge {
                x,
                by,
                knots,
                lambda,
                center,
                beta_buf,
                pred_buf,
                ..
            } => {
                let sse = fit_bhinge(
                    x.as_slice().expect("x must be contiguous"),
                    by.as_ref()
                        .map(|values| values.as_slice().expect("by contiguous")),
                    u,
                    weights,
                    knots,
                    lambda.unwrap_or(0.0),
                    *center,
                    beta_buf,
                    pred_buf,
                );
                (beta_buf.as_slice(), sse)
            }
            PreparedLearner::KroneckerBols { .. } => {
                let (b, _, sse) = self.fit_step_ref(u, weights);
                (b, sse)
            }
        }
    }

    /// Run the selection solve and retain its coefficient vector in this
    /// learner's scratch buffer, returning only the candidate SSE. This shape
    /// is suitable for `rayon::par_iter_mut`: parallel reductions move only
    /// indices/scores, and the caller copies the winning beta after the borrow
    /// boundary instead of cloning every losing candidate.
    pub fn fit_step_select_sse(
        &mut self,
        u: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        wu2: f64,
    ) -> f64 {
        self.fit_step_select(u, weights, wu2).1
    }

    /// Coefficients retained by the most recent selection solve.
    pub(crate) fn selection_beta(&self) -> &[f64] {
        match self {
            PreparedLearner::ScalarBols { beta_buf, .. }
            | PreparedLearner::BinnedScalarBols { beta_buf, .. }
            | PreparedLearner::BinnedBlockBols { beta_buf, .. }
            | PreparedLearner::BlockBols { beta_buf, .. }
            | PreparedLearner::BmonoBbs { beta_buf, .. }
            | PreparedLearner::Btree { beta_buf, .. }
            | PreparedLearner::Bhinge { beta_buf, .. }
            | PreparedLearner::KroneckerBols { beta_buf, .. }
            | PreparedLearner::BspatialStreamed { beta_buf, .. }
            | PreparedLearner::FactoredBlockBols { beta_buf, .. } => beta_buf,
        }
    }

    /// Compute prediction for a given beta into the provided buffer.
    /// Used after selection to compute the winner's prediction only.
    pub fn predict_into(&self, beta: &[f64], out: &mut [f64]) {
        debug_assert!(
            !beta.is_empty(),
            "PreparedLearner::predict_into: beta is empty"
        );
        match self {
            PreparedLearner::ScalarBols { design, .. } => {
                let bv = beta[0];
                let n = design.len();
                for i in 0..n {
                    out[i] = design[i] * bv;
                }
            }
            PreparedLearner::BinnedScalarBols {
                bin_index,
                bin_values,
                n_observations,
                ..
            } => {
                let beta = beta[0];
                debug_assert_eq!(out.len(), *n_observations);
                for i in 0..*n_observations {
                    out[i] = bin_values[bin_index[i] as usize] * beta;
                }
            }
            PreparedLearner::BinnedBlockBols {
                bin_index,
                basis,
                n_observations,
                ..
            } => {
                debug_assert_eq!(out.len(), *n_observations);
                let mut bin_prediction = vec![0.0f64; basis.nrows()];
                for bin in 0..basis.nrows() {
                    for column in 0..basis.ncols() {
                        bin_prediction[bin] += basis[[bin, column]] * beta[column];
                    }
                }
                for i in 0..*n_observations {
                    out[i] = bin_prediction[bin_index[i] as usize];
                }
            }
            PreparedLearner::BlockBols {
                design,
                sparse,
                centering,
                ..
            } => {
                block_bols_matvec_into(
                    design,
                    sparse
                        .as_ref()
                        .and_then(|cache| cache.get_or_build(design.view())),
                    centering.as_slice().expect("centering contiguous"),
                    beta,
                    out,
                );
            }
            PreparedLearner::BspatialStreamed {
                supports1,
                supports2,
                by,
                centering,
                p1,
                p2,
                ..
            } => match centering {
                Some(c) => {
                    let mut g = vec![0.0f64; *p1 * *p2];
                    c.back_transform(beta, *p1, *p2, &mut g);
                    streamed_predict(supports1, supports2, by.as_ref(), &g, *p2, out);
                }
                None => streamed_predict(supports1, supports2, by.as_ref(), beta, *p2, out),
            },
            PreparedLearner::BmonoBbs { design_faer, .. } => {
                faer_matvec_into(design_faer, beta, out);
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                n_groups,
                ..
            } => {
                let n = x_l.nrows();
                for i in 0..n {
                    let gr = groups[i] as usize;
                    let mut value = 0.0f64;
                    for l in 0..*p_l {
                        value += x_l[[i, l]] * beta[l * *n_groups + gr];
                    }
                    out[i] = value;
                }
            }
            PreparedLearner::Btree { design, by, .. } => {
                let tree = FittedTree::from_coef_vec(ArrayView1::from(beta));
                let prediction = tree.predict(design.view());
                for (i, value) in prediction.iter().enumerate() {
                    out[i] = match by {
                        Some(by_vec) => value * by_vec[i],
                        None => *value,
                    };
                }
            }
            PreparedLearner::Bhinge { x, by, .. } => {
                debug_assert_eq!(beta.len(), 4, "Bhinge coefficient width");
                let mut coef = [0.0f64; 4];
                coef.copy_from_slice(&beta[..beta.len().min(4)]);
                hinge_column(
                    x.as_slice().expect("x must be contiguous"),
                    by.as_ref()
                        .map(|values| values.as_slice().expect("by contiguous")),
                    &coef,
                    out,
                );
            }
            PreparedLearner::KroneckerBols {
                x1,
                x2,
                x2_is_identity,
                n_obs,
                n_resp,
                ..
            } => {
                let p1 = x1.ncols();
                let p2 = x2.ncols();
                if *x2_is_identity {
                    debug_assert_eq!(p2, *n_resp);
                    for i in 0..*n_obs {
                        let row_offset = i * *n_resp;
                        for k in 0..*n_resp {
                            let beta_k = &beta[k * p1..(k + 1) * p1];
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                s += x1[[i, j]] * beta_k[j];
                            }
                            out[row_offset + k] = s;
                        }
                    }
                } else {
                    for i in 0..*n_obs {
                        for k in 0..*n_resp {
                            let mut s = 0.0f64;
                            for j in 0..p1 {
                                for l in 0..p2 {
                                    s += x1[[i, j]] * beta[l * p1 + j] * x2[[k, l]];
                                }
                            }
                            out[i * n_resp + k] = s;
                        }
                    }
                }
            }
        }
    }

    /// Access the pred_buf from the most recent fit_step_select call.
    /// Used by Btree which can't reconstruct prediction from beta alone.
    pub fn pred_buf(&self) -> &[f64] {
        match self {
            PreparedLearner::Btree { pred_buf, .. }
            | PreparedLearner::Bhinge { pred_buf, .. }
            | PreparedLearner::KroneckerBols { pred_buf, .. } => pred_buf.as_slice(),
            _ => &[],
        }
    }

    /// Test-only: drop a `BlockBols` learner's sparse copy so parity
    /// tests can run the same learner through the dense faer route.
    #[cfg(test)]
    pub(crate) fn clear_sparse(&mut self) {
        if let PreparedLearner::BlockBols { sparse, .. } = self {
            *sparse = None;
        }
    }

    /// Recomputed enough fold-specific state to fit under new observation
    /// weights: weighted Gram terms, denominators, and solver factorizations,
    /// without rebuilding the design or penalties.
    pub fn reweight(&mut self, weights: ArrayView1<f64>) {
        match self {
            PreparedLearner::ScalarBols {
                design,
                penalty,
                weighted_design_t,
                scalar_denom,
                weights: w,
                ..
            } => {
                let n = design.len();
                let mut wxt = Array1::zeros(n);
                let mut denom = *penalty;
                for i in 0..n {
                    wxt[i] = weights[i] * design[i];
                    denom += weights[i] * design[i] * design[i];
                }
                *weighted_design_t = wxt;
                *scalar_denom = denom;
                *w = weights.to_owned();
            }
            PreparedLearner::BinnedScalarBols {
                bin_index,
                bin_values,
                penalty,
                scalar_denom,
                centering,
                center,
                sumw,
                n_observations,
                ..
            } => {
                let new_sumw: f64 = weights.sum();
                let bin_values = Arc::make_mut(bin_values);
                if *center {
                    let old_center = *centering;
                    let new_center = if new_sumw > 0.0 {
                        let mut total = 0.0f64;
                        for i in 0..*n_observations {
                            total += weights[i] * (bin_values[bin_index[i] as usize] + old_center);
                        }
                        total / new_sumw
                    } else {
                        0.0
                    };
                    let shift = old_center - new_center;
                    for value in bin_values.iter_mut() {
                        *value += shift;
                    }
                    *centering = new_center;
                    *sumw = new_sumw;
                }
                let mut denom = *penalty;
                for i in 0..*n_observations {
                    let value = bin_values[bin_index[i] as usize];
                    denom += weights[i] * value * value;
                }
                *scalar_denom = denom;
            }
            PreparedLearner::BinnedBlockBols {
                bin_index,
                basis,
                penalty,
                solver,
                n_observations,
                ..
            } => {
                let mut bin_weights = vec![0.0f64; basis.nrows()];
                for i in 0..*n_observations {
                    bin_weights[bin_index[i] as usize] += weights[i];
                }
                let mut gram = penalty.clone();
                for bin in 0..basis.nrows() {
                    let weight = bin_weights[bin];
                    for left in 0..basis.ncols() {
                        let scaled = weight * basis[[bin, left]];
                        for right in 0..basis.ncols() {
                            gram[[left, right]] += scaled * basis[[bin, right]];
                        }
                    }
                }
                *solver = GramSolver::new(&gram);
            }
            PreparedLearner::BlockBols {
                design,
                penalty,
                solver,
                weights: w,
                weighted_gram,
                ..
            } => {
                let gram = penalized_weighted_gram(design, weights, penalty);
                *solver = GramSolver::new(&gram);
                *w = weights.to_owned();
                // The stashed unpenalized Gram was computed under the old
                // weights and is no longer valid.
                *weighted_gram = None;
            }
            PreparedLearner::BspatialStreamed {
                supports1,
                supports2,
                by,
                centering,
                p1,
                p2,
                penalty,
                solver,
                weights: w,
                ..
            } => {
                let p = *p1 * *p2;
                let raw = streamed_raw_gram(
                    supports1,
                    supports2,
                    by.as_ref(),
                    weights.as_slice().expect("weights contiguous"),
                    *p2,
                    p,
                );
                let gram = match centering {
                    Some(c) => {
                        let g_tilde = bspatial_tensor_transform(&raw, &c.u1, &c.u2);
                        c.centered_gram(&g_tilde)
                    }
                    None => raw,
                };
                let solver_gram = &gram + &*penalty;
                *solver = GramSolver::new(&solver_gram);
                *w = weights.to_owned();
            }
            PreparedLearner::BmonoBbs {
                design,
                penalty,
                penalised_gram,
                solver_a,
                wdt_faer,
                design_faer,
                ..
            } => {
                let (n, p) = (design.nrows(), design.ncols());
                *wdt_faer = Mat::from_fn(p, n, |j, i| weights[i] * design[[i, j]]);
                let mut gram_faer = ndarray2_to_faer(penalty);
                faer::linalg::matmul::matmul(
                    &mut gram_faer,
                    wdt_faer,
                    design_faer,
                    Some(1.0),
                    1.0,
                    faer::Parallelism::None,
                );
                let gram = Array2::from_shape_fn((p, p), |(j, k)| gram_faer.read(j, k));
                *penalised_gram = gram.clone();
                *solver_a = GramSolver::new(&gram);
            }
            PreparedLearner::Btree { weights: w, by, .. } => {
                *w = btree_fit_weights(weights, by.as_ref());
            }
            PreparedLearner::Bhinge { weights: w, .. } => {
                // The hinge is re-fit from the passed weights at every step,
                // so only the reported weight slice needs updating.
                *w = weights.to_owned();
            }
            PreparedLearner::KroneckerBols {
                x1,
                penalty,
                wx1t_faer,
                x1_faer,
                solver,
                n_resp,
                ..
            } => {
                // weights has length n_obs * n_resp (expanded).
                // Extract per-observation weights by averaging across response dims.
                let n_obs = x1.nrows();
                let p1 = x1.ncols();
                let w_expanded = weights.as_slice().expect("weights must be contiguous");
                let w_obs: Vec<f64> = (0..n_obs).map(|i| w_expanded[i * *n_resp]).collect();
                *wx1t_faer = Mat::from_fn(p1, n_obs, |j, i| w_obs[i] * x1[[i, j]]);
                let mut gram_faer = ndarray2_to_faer(penalty);
                faer::linalg::matmul::matmul(
                    &mut gram_faer,
                    &*wx1t_faer,
                    x1_faer,
                    Some(1.0),
                    1.0,
                    faer::Parallelism::None,
                );
                let gram = Array2::from_shape_fn((p1, p1), |(j, k)| gram_faer.read(j, k));
                *solver = GramSolver::new(&gram);
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                n_groups,
                penalty_block,
                solvers,
                weights: w,
                ..
            } => {
                *w = weights.to_owned();
                let mut grams = vec![penalty_block.clone(); *n_groups];
                for i in 0..x_l.nrows() {
                    let gr = groups[i] as usize;
                    let wi = weights[i];
                    let gram = &mut grams[gr];
                    for j in 0..*p_l {
                        let scaled = wi * x_l[[i, j]];
                        for k in 0..*p_l {
                            gram[[j, k]] += scaled * x_l[[i, k]];
                        }
                    }
                }
                // Rebuild rather than overwrite in place: `solvers` must stay
                // one per right-marginal column, including a column no row
                // occupies, because every fit path indexes it over
                // `0..n_groups`.
                solvers.clear();
                solvers.extend(grams.iter().map(GramSolver::new));
            }
        }
    }

    /// Hat matrix of this learner at the current weights, `H = X(XᵀWX+λK)⁻¹XᵀW`,
    /// or `None` for learners that do not expose one (trees, kernels). A
    /// tree exposes none here because its matrix depends on the partition
    /// fitted at a step; that per-step matrix is
    /// [`PreparedLearner::btree_step_hat_matrix`].
    pub fn hat_matrix(&self) -> Option<Array2<f64>> {
        match self {
            PreparedLearner::ScalarBols {
                design,
                scalar_denom,
                centering,
                sumw,
                weights,
                ..
            } => {
                let _ = centering;
                let n = design.len();
                let denom = *scalar_denom;
                if denom <= 0.0 {
                    return Some(Array2::zeros((n, n)));
                }
                let inv_denom = 1.0 / denom;
                // `sumw` is the weight total when this learner carries its own
                // intercept column, and 0.0 when it does not. It was previously
                // inferred from `centering != 0`, which is a different question:
                // glmboost's per-predictor learners are centred but hold no
                // intercept (that is a separate learner), so every centred
                // glmboost fit counted an extra degree of freedom. R's AIC df
                // for glmboost is nu per selected learner; ours was 2*nu.
                let inv_sumw = if *sumw > 0.0 { 1.0 / *sumw } else { 0.0 };
                let mut h = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        // H_ij = w_j * d_i * d_j / denom (+ w_j / sumw for intercept)
                        // inv_sumw is 0.0 for a learner without an intercept,
                        // so the second term drops out without a branch.
                        let mut val = weights[j] * design[i] * design[j] * inv_denom;
                        val += weights[j] * inv_sumw;
                        h[[i, j]] = val;
                    }
                }
                Some(h)
            }
            PreparedLearner::BlockBols {
                design,
                solver,
                weights,
                ..
            } => {
                let (n, p) = (design.nrows(), design.ncols());
                // H = X · (X^T W X + λK)^{-1} · (X^T W)
                // Solve S^{-1} · (w_i X_i·) for each row of the design
                // (solve_mat requires contiguous input, so solve per-column)
                let mut m = Array2::zeros((p, n));
                let mut rhs = vec![0.0f64; p];
                let mut sol = vec![0.0f64; p];
                for i in 0..n {
                    for j in 0..p {
                        rhs[j] = weights[i] * design[[i, j]];
                    }
                    solver.solve_into_slice(&rhs, &mut sol);
                    for j in 0..p {
                        m[[j, i]] = sol[j];
                    }
                }
                let mut h = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let mut val = 0.0;
                        for k in 0..p {
                            val += design[[i, k]] * m[[k, j]];
                        }
                        h[[i, j]] = val;
                    }
                }
                Some(h)
            }
            PreparedLearner::BmonoBbs { .. } => None,
            PreparedLearner::Btree { .. } => None,
            // A hinge's basis is chosen per step, so it has no stable hat
            // matrix; it is an adaptive learner, like a tree.
            PreparedLearner::Bhinge { .. } => None,
            PreparedLearner::BinnedScalarBols { .. } | PreparedLearner::BinnedBlockBols { .. } => {
                None
            }
            PreparedLearner::BspatialStreamed { .. } => self.streamed_to_block_bols().hat_matrix(),
            // Kronecker array model: no simple per-observation hat matrix
            // (so AIC/df via the trace is unavailable, like btree/bmono).
            PreparedLearner::KroneckerBols { .. } => None,
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                solvers,
                weights,
                ..
            } => {
                // Rows of different groups do not share columns (D is column
                // group-disjoint), so H_ij = 0 unless both observations are in
                // the same group, and H = block-diagonal in the group sense:
                // H_ij = w_j · x_i' M_gr x_j for groups[i] == groups[j] = gr.
                let n = x_l.nrows();
                let inverses: Vec<Array2<f64>> = solvers.iter().map(GramSolver::inverse).collect();
                let mut h = Array2::zeros((n, n));
                for i in 0..n {
                    let gi = groups[i] as usize;
                    for j in 0..n {
                        if groups[j] as usize != gi {
                            continue;
                        }
                        let mut val = 0.0;
                        for l in 0..*p_l {
                            let mut row = 0.0;
                            for k in 0..*p_l {
                                row += inverses[gi][[l, k]] * x_l[[j, k]];
                            }
                            val += x_l[[i, l]] * row;
                        }
                        h[[i, j]] = weights[j] * val;
                    }
                }
                Some(h)
            }
        }
    }

    /// Diagonal of [`PreparedLearner::hat_matrix`], computed without forming
    /// the `n x n` matrix: `H_ii = w_i x_i' (XᵀWX + λK)⁻¹ x_i`. `None` for
    /// the learners that have no hat matrix.
    pub fn hat_diag(&self) -> Option<Vec<f64>> {
        match self {
            PreparedLearner::ScalarBols {
                design,
                scalar_denom,
                sumw,
                weights,
                ..
            } => {
                if *scalar_denom <= 0.0 {
                    return Some(vec![0.0; design.len()]);
                }
                let inv_sumw = if *sumw > 0.0 { 1.0 / *sumw } else { 0.0 };
                Some(
                    design
                        .iter()
                        .zip(weights.iter())
                        .map(|(&d, &w)| w * d * d / scalar_denom + w * inv_sumw)
                        .collect(),
                )
            }
            PreparedLearner::BlockBols {
                design,
                solver,
                weights,
                ..
            } => {
                let (n, p) = (design.nrows(), design.ncols());
                let mut rhs = vec![0.0f64; p];
                let mut sol = vec![0.0f64; p];
                let mut diag = Vec::with_capacity(n);
                for i in 0..n {
                    for j in 0..p {
                        rhs[j] = weights[i] * design[[i, j]];
                    }
                    solver.solve_into_slice(&rhs, &mut sol);
                    diag.push((0..p).map(|k| design[[i, k]] * sol[k]).sum());
                }
                Some(diag)
            }
            PreparedLearner::BspatialStreamed { .. } => self.streamed_to_block_bols().hat_diag(),
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                solvers,
                weights,
                ..
            } => {
                let inverses: Vec<Array2<f64>> = solvers.iter().map(GramSolver::inverse).collect();
                Some(
                    (0..x_l.nrows())
                        .map(|i| {
                            let m = &inverses[groups[i] as usize];
                            let mut val = 0.0;
                            for l in 0..*p_l {
                                let mut row = 0.0;
                                for k in 0..*p_l {
                                    row += m[[l, k]] * x_l[[i, k]];
                                }
                                val += x_l[[i, l]] * row;
                            }
                            weights[i] * val
                        })
                        .collect(),
                )
            }
            _ => self.hat_matrix().map(|h| h.diag().to_vec()),
        }
    }

    /// The hat matrix of one fitted `btree` step: the averaging matrix of
    /// the partition the step was fitted with, decoded from the step's
    /// recorded coefficient vector `coef`. `None` for every other learner
    /// kind, and for a coefficient vector that does not decode.
    ///
    /// Conditional on its fitted partition, a tree step is a linear map of
    /// the working response `u`. The fit grows the tree on the target
    /// `u / by` (plain `u` without a `by` variable) with the fit weights
    /// `ŵ = w · by²` this learner stores, and a leaf predicts the
    /// `ŵ`-weighted mean of its rows' targets, so the step's prediction is
    /// `S u` with, for row `j` in row `i`'s leaf `L`,
    ///
    /// `S[i, j] = by_i · ŵ_j / (by_j · Ŵ_L)`,  `Ŵ_L = Σ_{k ∈ L} ŵ_k`,
    ///
    /// and 0 outside the leaf. The `by` factors drop out without a `by`
    /// variable. A row with `by_j = 0` has `ŵ_j = 0` and a zero target, so
    /// it contributes nothing; a leaf with `Ŵ_L = 0` predicts the constant
    /// 0 the fit assigns it, and its block of `S` is 0.
    ///
    /// The convention is the one the smooth learners already get from
    /// [`PreparedLearner::hat_matrix`]: the matrix is conditional on the
    /// fitted structure — here the partition, there the knot placement —
    /// and the search that chose the structure (split selection, like
    /// learner selection) is not counted in the degrees of freedom.
    pub fn btree_step_hat_matrix(&self, coef: &[f64]) -> Option<Array2<f64>> {
        let PreparedLearner::Btree {
            design,
            weights,
            by,
            ..
        } = self
        else {
            return None;
        };
        if !valid_tree_coef(coef, design.ncols()) {
            return None;
        }
        let tree = FittedTree::from_coef_vec(ArrayView1::from(coef));
        let n = design.nrows();
        let num_nodes = coef[0] as usize;
        let mut leaf_of = vec![0usize; n];
        let mut leaf_weight = vec![0.0f64; num_nodes];
        for i in 0..n {
            let leaf = tree.leaf_index(design.row(i));
            leaf_of[i] = leaf;
            leaf_weight[leaf] += weights[i];
        }
        let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for (i, &leaf) in leaf_of.iter().enumerate() {
            rows_of[leaf].push(i);
        }
        let mut h = Array2::<f64>::zeros((n, n));
        for rows in &rows_of {
            if rows.is_empty() {
                continue;
            }
            let total = leaf_weight[leaf_of[rows[0]]];
            if total <= 0.0 {
                continue;
            }
            for &i in rows {
                let by_i = by.as_ref().map_or(1.0, |b| b[i]);
                for &j in rows {
                    if weights[j] == 0.0 {
                        continue;
                    }
                    h[[i, j]] = match by {
                        Some(b) if b[j] != 0.0 => by_i * weights[j] / (b[j] * total),
                        Some(_) => 0.0,
                        None => weights[j] / total,
                    };
                }
            }
        }
        Some(h)
    }

    /// Number of fitted values this learner produces (`n`, or `n·n_resp` for
    /// Kronecker learners).
    pub fn n(&self) -> usize {
        match self {
            PreparedLearner::ScalarBols { design, .. } => design.len(),
            PreparedLearner::BinnedScalarBols { n_observations, .. } => *n_observations,
            PreparedLearner::BinnedBlockBols { n_observations, .. } => *n_observations,
            PreparedLearner::BlockBols { design, .. } => design.nrows(),
            PreparedLearner::BmonoBbs { design, .. } => design.nrows(),
            PreparedLearner::Btree { design, .. } => design.nrows(),
            PreparedLearner::BspatialStreamed { b1, .. } => b1.nrows(),
            PreparedLearner::Bhinge { x, .. } => x.len(),

            PreparedLearner::KroneckerBols { n_obs, n_resp, .. } => n_obs * n_resp,
            PreparedLearner::FactoredBlockBols { x_l, .. } => x_l.nrows(),
        }
    }

    /// Design matrix `(n, p_j)` of this learner, reconstructed where the variant
    /// stores it in compact form, or `None` for Kronecker learners.
    pub fn design_matrix(&self) -> Option<Array2<f64>> {
        match self {
            PreparedLearner::ScalarBols { design, .. } => Some(
                design
                    .clone()
                    .into_shape_with_order((design.len(), 1))
                    .unwrap(),
            ),
            PreparedLearner::BinnedScalarBols {
                bin_index,
                bin_values,
                n_observations,
                ..
            } => Some(Array2::from_shape_fn((*n_observations, 1), |(row, _)| {
                bin_values[bin_index[row] as usize]
            })),
            PreparedLearner::BinnedBlockBols {
                bin_index,
                basis,
                n_observations,
                ..
            } => Some(Array2::from_shape_fn(
                (*n_observations, basis.ncols()),
                |(row, column)| basis[[bin_index[row] as usize, column]],
            )),
            PreparedLearner::BlockBols { design, .. } => Some(design.clone()),
            PreparedLearner::BmonoBbs { design, .. } => Some(design.clone()),
            PreparedLearner::Btree { design, .. } => Some(design.clone()),
            PreparedLearner::BspatialStreamed { .. } => Some(self.streamed_dense_design()),
            // A hinge has no fixed design; report the column selected by the
            // most recent fit step (`by * max(0, ±(x - knot)) - centering`),
            // or `None` before the first step.
            PreparedLearner::Bhinge {
                x, by, beta_buf, ..
            } => {
                if beta_buf.len() != 4 {
                    return None;
                }
                let (knot, direction, _, centering) =
                    (beta_buf[0], beta_buf[1], beta_buf[2], beta_buf[3]);
                let n = x.len();
                let mut d = Array2::zeros((n, 1));
                for i in 0..n {
                    let hinge = if direction > 0.0 {
                        (x[i] - knot).max(0.0)
                    } else {
                        (knot - x[i]).max(0.0)
                    };
                    let raw = match by.as_ref() {
                        Some(values) => hinge * values[i],
                        None => hinge,
                    };
                    d[[i, 0]] = raw - centering;
                }
                Some(d)
            }

            PreparedLearner::KroneckerBols { x1, n_resp, .. } => {
                // Return the effective design (Kronecker product) for AIC/df computation
                let n_obs = x1.nrows();
                Some(Array2::from_shape_vec((n_obs * n_resp, 1), vec![0.0]).unwrap())
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                n_groups,
                ..
            } => {
                // Reconstruct the materialised row-wise tensor
                // T[i, l·ng + g] = x_l[i, l]·δ(g, groups[i]).
                let n = x_l.nrows();
                let mut d = Array2::zeros((n, *p_l * *n_groups));
                for i in 0..n {
                    let gr = groups[i] as usize;
                    for l in 0..*p_l {
                        d[[i, l * *n_groups + gr]] = x_l[[i, l]];
                    }
                }
                Some(d)
            }
        }
    }

    /// Penalty matrix `λK` of this learner, or `None` for scalar learners that
    /// store only the scalar ridge value and for trees.
    pub fn penalty_matrix(&self) -> Option<Array2<f64>> {
        match self {
            PreparedLearner::ScalarBols { penalty, .. }
            | PreparedLearner::BinnedScalarBols { penalty, .. } => {
                Some(Array2::from_shape_vec((1, 1), vec![*penalty]).unwrap())
            }
            PreparedLearner::BlockBols { penalty, .. } => Some(penalty.clone()),
            PreparedLearner::BinnedBlockBols { penalty, .. } => Some(penalty.clone()),
            PreparedLearner::BspatialStreamed { penalty, .. } => Some(penalty.clone()),
            PreparedLearner::BmonoBbs { penalty, .. } => Some(penalty.clone()),
            PreparedLearner::Btree { .. } => None,
            // The hinge's ridge is a single scalar on the selected column.
            PreparedLearner::Bhinge { lambda, .. } => {
                Some(Array2::from_shape_vec((1, 1), vec![lambda.unwrap_or(0.0)]).unwrap())
            }
            PreparedLearner::KroneckerBols { penalty, .. } => Some(penalty.clone()),
            PreparedLearner::FactoredBlockBols {
                penalty_block,
                p_l,
                n_groups,
                ..
            } => {
                // Reconstruct the block-diagonal λK with p_l×p_l copies of the
                // per-group penalty block on the diagonal.
                let mut pen = Array2::zeros((*p_l * *n_groups, *p_l * *n_groups));
                for gr in 0..*n_groups {
                    for l in 0..*p_l {
                        for k in 0..*p_l {
                            pen[[l * *n_groups + gr, k * *n_groups + gr]] = penalty_block[[l, k]];
                        }
                    }
                }
                Some(pen)
            }
        }
    }

    /// Effective ridge weight `λ` for scalar learners and for block learners
    /// whose penalty was derived from a single scalar λ (df→λ calibration or
    /// an explicit `lambda=`), `None` otherwise (bmono, btree, Kronecker,
    /// factored interactions, and block learners built from pre-scaled
    /// penalties).
    ///
    /// R's `extract(x, what = "lambda")` reports exactly this: the effective
    /// λ per base-learner, absent for learners without one.
    pub fn lambda(&self) -> Option<f64> {
        match self {
            PreparedLearner::ScalarBols { penalty, .. }
            | PreparedLearner::BinnedScalarBols { penalty, .. } => Some(*penalty),
            PreparedLearner::BlockBols { lambda, .. }
            | PreparedLearner::BinnedBlockBols { lambda, .. }
            | PreparedLearner::BspatialStreamed { lambda, .. } => *lambda,
            PreparedLearner::BmonoBbs { .. } => None,
            PreparedLearner::Btree { .. } => None,
            PreparedLearner::Bhinge { lambda, .. } => *lambda,
            PreparedLearner::KroneckerBols { .. } => None,
            PreparedLearner::FactoredBlockBols { .. } => None,
        }
    }

    /// Builder that records the effective smoothing parameter `λ` a block
    /// learner's penalty was derived from. Makes `extract("lambda")` report
    /// the calibrated value instead of `None`. No-op on every other variant
    /// (scalar learners already report their penalty; the rest have no scalar
    /// λ), so a call site can chain it unconditionally.
    pub fn with_effective_lambda(mut self, lambda: Option<f64>) -> Self {
        match &mut self {
            PreparedLearner::BlockBols { lambda: slot, .. }
            | PreparedLearner::BinnedBlockBols { lambda: slot, .. } => *slot = lambda,
            _ => {}
        }
        self
    }

    /// Effective degrees of freedom as `tr(H)`.
    ///
    /// Uses the cached hat-factor `X·M·(XᵀW)` decomposition when the learner
    /// has one, so the cost is O(n·pⱼ²) in the learner's design rather than
    /// O(n²) in the hat matrix itself -- at n = 200,000 a dense `n×n` trace
    /// is 320 GB of transient storage, while the factor is a `pⱼ×pⱼ` solve
    /// already paid once at fit time. Falls back to the dense hat matrix only
    /// where no factor exists (bmono, btree, Kronecker, binned block
    /// learners): those either return `None` already or are scalar and tiny.
    pub fn df(&self) -> Option<f64> {
        if let Some(f) = self.hat_factor() {
            return Some(f.tr_h);
        }
        let h = self.hat_matrix()?;
        let n = h.nrows();
        let mut tr = 0.0;
        for i in 0..n {
            tr += h[[i, i]];
        }
        Some(tr)
    }

    /// Degrees of freedom the way `mboost` counts them: `tr(2H - HᵀH)`.
    ///
    /// [`Self::df`] returns `tr(H)`, which is the other common definition and
    /// a different number -- on a 140-row cubic P-spline with `lambda = 5`,
    /// `tr(H)` is 7.876 and this is 9.265. `mboost`'s `df2lambda` inverts
    /// this definition, so a learner requested at `df = 4` uses the λ that
    /// makes this value 4, even though `tr(H)` for that learner is 3.376.
    ///
    /// Same factor-first cost model as [`Self::df`]: `tr(HᵀH)` is evaluated
    /// as `Σᵢ wᵢ² (X·C·Xᵀ)ᵢᵢ` with `C = M·G·M`, `G = XᵀX`, all terms
    /// `pⱼ×pⱼ` or `pⱼ²·n`, never an `n×n` matrix.
    ///
    /// With observation weights the smoother `H = X M XᵀW` is not symmetric.
    /// mboost uses `2 tr(H) - tr(H²)`, whereas this method computes
    /// `tr(2H - HᵀH)`. These agree for equal weights; with weights, this
    /// definition gave 3.63 for a learner fitted at `df = 4` (mboost: 4).
    pub fn df_mboost(&self) -> Option<f64> {
        if let Some(spectrum) = smoother_spectrum(self) {
            return Some(spectrum.iter().map(|&s| 2.0 * s - s * s).sum());
        }
        let h = self.hat_matrix()?;
        let n = h.nrows();
        let mut value = 0.0;
        for i in 0..n {
            value += 2.0 * h[[i, i]];
            for j in 0..n {
                value -= h[[i, j]] * h[[j, i]];
            }
        }
        Some(value)
    }

    /// Factor decomposition: H_j = X · M · (XᵀW).
    ///
    /// Returns (design X (n×p_j), M = (XᵀWX + λK)⁻¹ (p_j×p_j), tr(Hⱼ)).
    /// Returns `None` for base-learners without a closed-form hat matrix
    /// (bmono, btree, Kronecker).
    pub fn hat_factor(&self) -> Option<HatFactor> {
        match self {
            PreparedLearner::ScalarBols {
                design,
                penalty,
                weights,
                scalar_denom,
                sumw,
                ..
            } => {
                let n = design.len();
                let denom = *scalar_denom;
                if denom <= 0.0 {
                    return None;
                }
                let lambda = *penalty;
                // Whether the learner carries its own intercept column, not
                // whether its design happens to be centred. glmboost centres
                // every predictor but keeps the intercept as a separate
                // learner, so inferring it from `centering` counted a second
                // column that is not there and reported 2*nu per selected
                // learner where R reports nu.
                let has_intercept = *sumw > 0.0;
                let p: usize = if has_intercept { 2 } else { 1 };
                let w_slice = weights.as_slice()?;

                // Design X: [d] (n×1) or [d, 1] (n×2)
                let mut x_faer = Mat::<f64>::zeros(n, p);
                for i in 0..n {
                    x_faer.write(i, 0, design[i]);
                }
                if has_intercept {
                    for i in 0..n {
                        x_faer.write(i, 1, 1.0);
                    }
                }

                // Weighted Gram: G_pen = XᵀWX + λK  (p×p, row-major)
                let mut gram = vec![0.0f64; p * p];
                for i in 0..n {
                    let wi = w_slice[i];
                    for k in 0..p {
                        let xik = x_faer.read(i, k);
                        for l in 0..p {
                            gram[k * p + l] += wi * xik * x_faer.read(i, l);
                        }
                    }
                }
                // λK: penalise only the design coefficient (first column)
                gram[0] += lambda;

                // Invert G_pen → M (small p, direct formula, row-major vec)
                let hat_inv = if p == 1 {
                    vec![1.0 / gram[0]]
                } else {
                    let det = gram[0] * gram[3] - gram[1] * gram[2];
                    if det.abs() < 1e-30 {
                        return None;
                    }
                    let inv_det = 1.0 / det;
                    vec![
                        gram[3] * inv_det,
                        -gram[1] * inv_det,
                        -gram[2] * inv_det,
                        gram[0] * inv_det,
                    ]
                };

                // tr(H) = tr(M · G_unpen) where G_unpen = G_pen - λK
                let gram_00_unpen = gram[0] - lambda;
                let mut tr_h = 0.0;
                for k in 0..p {
                    for l in 0..p {
                        let gul = if l == 0 && k == 0 {
                            gram_00_unpen
                        } else {
                            gram[l * p + k]
                        };
                        tr_h += hat_inv[l * p + k] * gul;
                    }
                }

                Some(HatFactor {
                    design: x_faer,
                    hat_inv,
                    hat_inv_p: p,
                    tr_h,
                })
            }
            PreparedLearner::BlockBols {
                design,
                solver,
                weights,
                ..
            } => {
                let design_faer = design_ref(design).to_owned();
                let p = design_faer.ncols();
                let w_slice = weights.as_slice()?;

                // M = (XᵀWX + λK)⁻¹  (from the cached solver, row-major vec)
                let inv_arr = solver.inverse();
                let mut hat_inv = vec![0.0f64; p * p];
                for i in 0..p {
                    for j in 0..p {
                        hat_inv[i * p + j] = inv_arr[[i, j]];
                    }
                }

                // tr(H) = tr(M · G_raw) with the unpenalized Gram G_raw = XᵀWX.
                let gram = weighted_gram_faer(design_faer.as_ref(), w_slice);
                let mut tr_h = 0.0;
                for k in 0..p {
                    for l in 0..p {
                        tr_h += hat_inv[l * p + k] * gram.read(l, k);
                    }
                }

                Some(HatFactor {
                    design: design_faer,
                    hat_inv,
                    hat_inv_p: p,
                    tr_h,
                })
            }
            PreparedLearner::FactoredBlockBols {
                x_l,
                groups,
                p_l,
                n_groups,
                solvers,
                weights,
                ..
            } => {
                let n = x_l.nrows();
                let p = *p_l * *n_groups;
                let w_slice = weights.as_slice()?;

                // Design D: the materialised row-wise tensor, n×p row-major.
                let mut design = Mat::<f64>::zeros(n, p);
                for i in 0..n {
                    let gr = groups[i] as usize;
                    for l in 0..*p_l {
                        design.write(i, l * *n_groups + gr, x_l[[i, l]]);
                    }
                }

                // M = (XᵀWX + λK)⁻¹: block-diagonal of per-group inverses in
                // the materialised column order l·ng + g.
                let mut hat_inv = vec![0.0f64; p * p];
                for gr in 0..*n_groups {
                    let inv_arr = solvers[gr].inverse();
                    for l in 0..*p_l {
                        for k in 0..*p_l {
                            hat_inv[(l * *n_groups + gr) * p + (k * *n_groups + gr)] =
                                inv_arr[[l, k]];
                        }
                    }
                }

                // Unpenalized Gram G_raw = XᵀWX (block-diagonal by group).
                let mut gram = vec![0.0f64; p * p];
                for i in 0..n {
                    let wi = w_slice[i];
                    let gr = groups[i] as usize;
                    for l in 0..*p_l {
                        let xil = x_l[[i, l]];
                        for k in 0..*p_l {
                            gram[(l * *n_groups + gr) * p + (k * *n_groups + gr)] +=
                                wi * xil * x_l[[i, k]];
                        }
                    }
                }

                // tr(H) = tr(M · G_raw)
                let mut tr_h = 0.0;
                for k in 0..p {
                    for l in 0..p {
                        tr_h += hat_inv[l * p + k] * gram[l * p + k];
                    }
                }

                Some(HatFactor {
                    design,
                    hat_inv,
                    hat_inv_p: p,
                    tr_h,
                })
            }
            PreparedLearner::BspatialStreamed { .. } => self.streamed_to_block_bols().hat_factor(),
            PreparedLearner::BinnedScalarBols { .. }
            | PreparedLearner::BinnedBlockBols { .. }
            | PreparedLearner::BmonoBbs { .. }
            | PreparedLearner::Btree { .. }
            | PreparedLearner::Bhinge { .. }
            | PreparedLearner::KroneckerBols { .. } => None,
        }
    }

    /// Set x1 reconstruction metadata for new-data prediction.
    ///
    /// After constructing a `KroneckerBols` learner, call this to store
    /// the information needed to rebuild the x1 design from raw x values
    /// when `predict_newdata_kronecker` is called.
    pub fn set_x1_rebuild(&mut self, rebuild: KroneckerRebuild) {
        if let PreparedLearner::KroneckerBols { x1_rebuild, .. } = self {
            *x1_rebuild = Some(rebuild);
        }
    }
}

/// `XᵀWX` for a faer design, accumulated over 4,096-row blocks so no
/// weighted copy of a large design is formed.
fn weighted_gram_faer(design: faer::MatRef<f64>, weights: &[f64]) -> Mat<f64> {
    const ROWS: usize = 4096;
    let (n, p) = (design.nrows(), design.ncols());
    let mut gram = Mat::<f64>::zeros(p, p);
    let mut start = 0;
    while start < n {
        let rows = (n - start).min(ROWS);
        let block = design.subrows(start, rows);
        let scaled = Mat::from_fn(rows, p, |i, k| weights[start + i] * block.read(i, k));
        gram += block.transpose() * &scaled;
        start += rows;
    }
    gram
}

/// Eigenvalues of the learner's smoother `S = X M XᵀW`, `M = (XᵀWX + λK)⁻¹`,
/// ascending and clamped to `[0, 1]`. They are the eigenvalues of the
/// symmetric `M^½ (XᵀWX) M^½`. `None` for learners without a closed-form hat
/// matrix (trees, monotone splines, Kronecker learners).
pub(crate) fn smoother_spectrum(learner: &PreparedLearner) -> Option<Vec<f64>> {
    let f = learner.hat_factor()?;
    let weights = learner.weights()?;
    let n = f.design.nrows();
    let p = f.design.ncols();
    if weights.len() != n || p == 0 {
        return None;
    }
    let g = weighted_gram_faer(f.design.as_ref(), weights);
    // The smoother S = X M XᵀW has the eigenvalues of M G, which for any
    // factor M = R Rᵀ are those of Rᵀ G R: the Cholesky factor when M is
    // positive definite, otherwise the symmetric root from M's eigenvectors.
    let symmetric =
        |a: faer::MatRef<f64>| Mat::from_fn(p, p, |i, j| 0.5 * (a.read(i, j) + a.read(j, i)));
    let m = symmetric(Mat::from_fn(p, p, |i, j| f.hat_inv[i * f.hat_inv_p + j]).as_ref());
    let a = match m.cholesky(faer::Side::Lower) {
        Ok(chol) => {
            let l = chol.compute_l();
            l.transpose() * &g * &l
        }
        Err(_) => {
            let em = m.selfadjoint_eigendecomposition(faer::Side::Lower);
            let (vals, vecs) = (em.s().column_vector(), em.u());
            let scaled_vecs =
                Mat::from_fn(p, p, |i, k| vecs.read(i, k) * vals.read(k).max(0.0).sqrt());
            let root = &scaled_vecs * vecs.transpose();
            &root * &g * &root
        }
    };
    let ea = symmetric(a.as_ref()).selfadjoint_eigendecomposition(faer::Side::Lower);
    let spectrum = ea.s().column_vector();
    Some((0..p).map(|k| spectrum.read(k).clamp(0.0, 1.0)).collect())
}

/// Solve `min ½β'Aβ − b'β s.t. Dβ ≥ 0` via the iterative
/// violation-penalty heuristic from R `mboost::bmono` (`type = "iterative"`).
///
/// At iteration k, let `V_k = diag(1{Dβ_k < 0})`. Update
///     β_{k+1} = (A + λ₂ · D' V_k D)⁻¹ b
/// and stop when `V_k` stops changing or `niter` is hit.
///
/// On a well-conditioned problem, `λ₂ = 1e6` drives violations to roughly
/// machine precision in 3–5 iterations. The first iteration starts with
/// `V₀ = 0` (i.e. the unconstrained ridge solve); if no constraint is
/// violated we return immediately.
pub(crate) fn solve_bmono_iterative(
    solver_a: &GramSolver,
    a: ndarray::ArrayView2<f64>,
    d: ndarray::ArrayView2<f64>,
    b: ndarray::ArrayView1<f64>,
    lambda2: f64,
    niter: usize,
) -> Array1<f64> {
    let p = a.nrows();
    let m = d.nrows();
    let mut beta = vec![0.0; p];
    let mut v = Vec::with_capacity(m);
    let mut v_new = Vec::with_capacity(m);
    let mut a_aug = vec![0.0; p * p];
    let mut x = vec![0.0; p];
    let mut chol_scratch = vec![0.0; cho_solve_inplace_scratch_len(p)];
    solve_bmono_iterative_into(
        solver_a,
        a,
        d,
        b.as_slice().expect("b contiguous"),
        lambda2,
        niter,
        &mut beta,
        &mut v,
        &mut v_new,
        &mut a_aug,
        &mut x,
        &mut chol_scratch,
    );
    Array1::from_vec(beta)
}

/// Allocation-free form of [`solve_bmono_iterative`]: writes `β*` into
/// `beta` and reuses caller-owned scratch so a candidate solve in the
/// boosting loop allocates nothing (P3.4 item 2). `a_aug` is `p²`
/// column-major, factored and solved in place by `cho_solve_inplace`; the
/// arithmetic is identical to the owned form, so the result is bit-identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_bmono_iterative_into(
    solver_a: &GramSolver,
    a: ndarray::ArrayView2<f64>,
    d: ndarray::ArrayView2<f64>,
    b: &[f64],
    lambda2: f64,
    niter: usize,
    beta: &mut [f64],
    v: &mut Vec<bool>,
    v_new: &mut Vec<bool>,
    a_aug: &mut [f64],
    x: &mut [f64],
    chol_scratch: &mut [f64],
) {
    let p = a.nrows();
    let m = d.nrows();

    // Iteration 0: unconstrained solve. `A` is fixed across every boosting
    // iteration (only `b` changes), so reuse the cached factorization
    // (`solver_a`, built once at `prepare_bmono` time) instead of a fresh
    // p×p copy + full Cholesky of the same matrix every call.
    solver_a.solve_into_slice(b, beta);
    compute_violations_into(d, ndarray::ArrayView1::from(&beta[..]), v);
    if !v.iter().any(|&violated| violated) {
        return;
    }

    // Fill the column-major augmented matrix from the base `A` (row-major).
    for j in 0..p {
        for i in 0..p {
            a_aug[j * p + i] = a[[i, j]];
        }
    }
    for _ in 0..niter {
        // Reset a_aug to base A, then accumulate the λ₂·D_V'D_V updates.
        for j in 0..p {
            for i in 0..p {
                a_aug[j * p + i] = a[[i, j]];
            }
        }
        for i in 0..m {
            if !v[i] {
                continue;
            }
            // rank-1 update: a_aug += λ₂ · d_i ⊗ d_i
            let row = d.row(i);
            for j in 0..p {
                let dj = row[j];
                if dj == 0.0 {
                    continue;
                }
                let coef = lambda2 * dj;
                for k in 0..p {
                    a_aug[k * p + j] += coef * row[k];
                }
            }
        }

        x.copy_from_slice(b);
        if cho_solve_inplace(p, a_aug, x, chol_scratch).is_err() {
            return; // fall back to previous iterate
        }
        beta.copy_from_slice(x);

        compute_violations_into(d, ndarray::ArrayView1::from(&beta[..]), v_new);
        if *v == *v_new {
            break;
        }
        std::mem::swap(v, v_new);
    }
}

#[inline]
fn compute_violations_into(
    d: ndarray::ArrayView2<f64>,
    beta: ndarray::ArrayView1<f64>,
    out: &mut Vec<bool>,
) {
    out.clear();
    for i in 0..d.nrows() {
        // (Dβ)_i < 0   ⇒  violated  (we want Dβ ≥ 0)
        let dot: f64 = d.row(i).dot(&beta);
        out.push(dot < 0.0);
    }
}

/// Solve `min ½β'Aβ − b'β   s.t.   Dβ ≥ 0` via a primal active-set QP
/// (Nocedal–Wright Algorithm 16.3, specialised for `D β ≥ 0`).
///
/// Starts at β = 0 with empty working set W. β = 0 is feasible for every
/// shape constraint we use (`Dβ = 0 ≥ 0` regardless of the constraint kind).
/// Per outer iteration:
///   1. Compute search direction Δ = argmin ½Δ'AΔ + (Aβ − b)'Δ s.t. D_W Δ = 0
///      via the block-elimination KKT solve
///         λ_W = (D_W A⁻¹ D_W')⁻¹ (D_W A⁻¹ g)
///         Δ   = −A⁻¹ g + A⁻¹ D_W' λ_W
///      where g = Aβ − b.
///   2. If ‖Δ‖ ≈ 0: at constrained optimum for the current W. `λ_W` (already
///      produced by the KKT solve) is then the Lagrange-multiplier estimate
///      for the original inequality problem, whose KKT condition is λ_W ≥ 0.
///      If all λ_W ≥ 0, return. Else drop the constraint with the most
///      negative multiplier from W.
///
/// The multiplier sign convention matters: stationarity for `Dβ ≥ 0` is
/// `g = D_W' λ_W` with `λ_W ≥ 0`, so `λ_W` must be solved from `+D_W A⁻¹ g`,
/// not `−D_W A⁻¹ g`. Solving for `−λ_W` and then testing `≥ 0` inverts both
/// the stopping rule and the drop rule, which strands the iterate on a
/// feasible but non-optimal face (fixed 2026-07-20; regression test
/// `bmono_quadprog_matches_exhaustive_active_set_oracle`).
///   3. Else: compute the max-feasible step length
///         α = min(1, min_{i ∉ W, Dᵢ Δ < 0} −(Dᵢ β) / (Dᵢ Δ))
///      Take β ← β + αΔ. If α < 1 a blocking inequality bound has been hit;
///      add that constraint to W.
///
/// Finite termination: standard non-degeneracy assumption (Bland-style
/// anti-cycling not needed for SPD A and modest dimensions).
pub(crate) fn solve_bmono_quadprog(
    solver_a: &GramSolver,
    gram: ndarray::ArrayView2<f64>,
    d: ndarray::ArrayView2<f64>,
    b: ndarray::ArrayView1<f64>,
    tol: f64,
    max_iter: usize,
) -> Array1<f64> {
    let p = b.len();
    let m = d.nrows();
    let mut beta = vec![0.0; p];
    let mut g = vec![0.0; p];
    let mut a_inv_g = vec![0.0; p];
    let mut active: Vec<usize> = Vec::with_capacity(m);
    let mut a_inv_dt = vec![0.0; p * m];
    let mut h = vec![0.0; m * m];
    let mut lambda = vec![0.0; m];
    let mut delta = vec![0.0; p];
    let mut chol_scratch = vec![0.0; cho_solve_inplace_scratch_len(p.max(m))];
    solve_bmono_quadprog_into(
        solver_a,
        gram,
        d,
        b.as_slice().expect("b contiguous"),
        tol,
        max_iter,
        &mut beta,
        &mut g,
        &mut a_inv_g,
        &mut active,
        &mut a_inv_dt,
        &mut h,
        &mut lambda,
        &mut delta,
        &mut chol_scratch,
    );
    Array1::from_vec(beta)
}

/// Allocation-free form of [`solve_bmono_quadprog`]: writes `β*` into `beta`
/// and works in caller-owned scratch so a candidate solve in the boosting
/// loop allocates nothing (P3.4 item 2). Scratch is pre-sized to the
/// worst-case active-set width `m`; the used regions are the first `q` rows
/// of each buffer. All arithmetic mirrors the owned form — same `h`/`rhs`
/// values in column-major order, same in-place faer factorization — so the
/// result is bit-identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_bmono_quadprog_into(
    solver_a: &GramSolver,
    gram: ndarray::ArrayView2<f64>,
    d: ndarray::ArrayView2<f64>,
    b: &[f64],
    tol: f64,
    max_iter: usize,
    beta: &mut [f64],
    g: &mut [f64],
    a_inv_g: &mut [f64],
    active: &mut Vec<usize>,
    a_inv_dt: &mut [f64],
    h: &mut [f64],
    lambda: &mut [f64],
    delta: &mut [f64],
    chol_scratch: &mut [f64],
) {
    let p = b.len();
    let m = d.nrows();

    solver_a.solve_into_slice(b, beta);
    let mut all_feasible = true;
    for i in 0..m {
        if d.row(i).dot(&ndarray::ArrayView1::from(&beta[..])) < -tol {
            all_feasible = false;
            break;
        }
    }
    if all_feasible {
        return;
    }

    // Otherwise drop back to β = 0 (always feasible: Dβ = 0) and run the
    // active-set loop. Cost of starting at 0 vs a smarter feasible point is
    // negligible for our problem sizes (p ≤ 40).
    for r in 0..p {
        beta[r] = 0.0;
    }
    active.clear();

    // Scratch reused across iterations.
    for r in 0..p {
        g[r] = 0.0;
        a_inv_g[r] = 0.0;
    }

    for _ in 0..max_iter {
        // g = Aβ − b  (A = penalised_gram, stored directly)
        let mut g_view = ndarray::ArrayViewMut1::from(&mut g[..]);
        ndarray::linalg::general_mat_vec_mul(
            1.0,
            &gram,
            &ndarray::ArrayView1::from(&beta[..]),
            0.0,
            &mut g_view,
        );
        for r in 0..p {
            g[r] -= b[r];
        }

        solver_a.solve_into_slice(g, a_inv_g);

        let q = active.len();
        if q == 0 {
            for r in 0..p {
                delta[r] = -a_inv_g[r];
            }
        } else {
            // a_inv_dt = A^{-1} D_W' (p × q), written directly into the
            // caller-owned scratch columns.
            for (k, &i) in active.iter().enumerate() {
                let d_row = d.row(i);
                let col = &mut a_inv_dt[k * p..(k + 1) * p];
                solver_a.solve_into_slice(d_row.as_slice().expect("d row must be contiguous"), col);
            }
            // h = D_W A^{-1} D_W' (q × q, symmetric PSD). Dense column-major
            // (k1, k2) at `k2*q + k1`, matching what cho_solve_inplace expects.
            for k1 in 0..q {
                let i1 = active[k1];
                let d_row1 = d.row(i1);
                for k2 in 0..q {
                    let mut s = 0.0;
                    for r in 0..p {
                        s += d_row1[r] * a_inv_dt[k2 * p + r];
                    }
                    h[k2 * q + k1] = s;
                }
            }
            // rhs = +D_W A^{-1} g  (length q). The sign is load-bearing: it
            // makes `lambda` the actual Lagrange multiplier for `Dβ ≥ 0`,
            // which the optimality test below assumes is non-negative.
            for (k, &i) in active.iter().enumerate() {
                let mut s = 0.0;
                for r in 0..p {
                    s += d[[i, r]] * a_inv_g[r];
                }
                lambda[k] = s;
            }
            // λ_W = H^{-1} rhs, factored and solved in place. If H is
            // singular (linearly dependent active constraints — possible for
            // degenerate D shapes), drop the last-added active and retry.
            if cho_solve_inplace(q, &mut h[..q * q], &mut lambda[..q], chol_scratch).is_err() {
                active.pop();
                continue;
            }
            // Δ = −A^{-1} g + A^{-1} D_W' λ_W
            for r in 0..p {
                let mut s = -a_inv_g[r];
                for k in 0..q {
                    s += a_inv_dt[k * p + r] * lambda[k];
                }
                delta[r] = s;
            }
        }

        let delta_norm_sq: f64 = delta.iter().map(|v| v * v).sum();

        // ‖Δ‖ carries the units of β, so the convergence test is scaled by the
        // current iterate: an absolute threshold would let a large-β problem
        // spin to `max_iter` without ever reaching the optimality check below.
        let beta_norm_sq: f64 = beta.iter().map(|v| v * v).sum();
        let delta_tol_sq = tol * tol * beta_norm_sq.max(1.0);

        if delta_norm_sq < delta_tol_sq {
            // At constrained optimum for current W. Check multipliers.
            if q == 0 {
                return;
            }
            // λ_W carries the units of g = Aβ − b, so scale the sign test the
            // same way rather than comparing against an absolute 1e-10.
            let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            let lambda_tol = tol * g_norm.max(1.0);
            // Find the most negative multiplier.
            let mut worst_k: usize = 0;
            let mut worst_lambda: f64 = f64::INFINITY;
            for k in 0..q {
                let v = lambda[k];
                if v < worst_lambda {
                    worst_lambda = v;
                    worst_k = k;
                }
            }
            if worst_lambda >= -lambda_tol {
                return; // all λ ≥ 0 → KKT satisfied
            }
            active.remove(worst_k);
            continue;
        }

        // Step length: α = min(1, max-feasible step before hitting an
        // inactive inequality boundary). For i ∉ W with Dᵢ Δ < 0, the
        // boundary i is hit when β + αΔ satisfies Dᵢ (β + αΔ) = 0, i.e.
        //     α_i = −(Dᵢ β) / (Dᵢ Δ).
        let mut alpha: f64 = 1.0;
        let mut blocking: Option<usize> = None;
        for i in 0..m {
            if active.contains(&i) {
                continue;
            }
            let mut d_delta = 0.0;
            let mut d_beta = 0.0;
            for r in 0..p {
                d_delta += d[[i, r]] * delta[r];
                d_beta += d[[i, r]] * beta[r];
            }
            if d_delta < -tol {
                let alpha_i = -d_beta / d_delta;
                if alpha_i < alpha {
                    alpha = alpha_i;
                    blocking = Some(i);
                }
            }
        }

        for r in 0..p {
            beta[r] += alpha * delta[r];
        }

        if alpha < 1.0 - tol {
            if let Some(idx) = blocking {
                active.push(idx);
            }
        }
        // else: full step, loop again to check optimality.
    }
    // safety fallback if max_iter is hit: beta holds the current iterate.
}

fn weighted_sse(u: ArrayView1<f64>, pred: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
    let mut sse = 0.0f64;
    for i in 0..u.len() {
        let r = u[i] - pred[i];
        sse += w[i] * r * r;
    }
    sse
}

fn weighted_sse_slices(u: &[f64], pred: &[f64], w: &[f64]) -> f64 {
    let mut sse = 0.0f64;
    for i in 0..u.len() {
        let r = u[i] - pred[i];
        sse += w[i] * r * r;
    }
    sse
}

/// Convert target df to ridge penalty λ for a single-column `bols` learner.
///
/// For scalar bols (p=1, K=I), R's `df2lambda` simplifies to a closed form:
/// - `A = x'Wx + eps`   (1×1 "gram + eps·K")
/// - `d = 1/A`           (single eigenvalue of `R_m^T K R_m`)
/// - `dfFun(λ) = 2/(1+λd) - 1/(1+λd)²`   (R's `mboost_dftraceS=FALSE`)
/// Setting `v = 1/(1+λd)`, `df = v(2−v)`, so `v = 1 − √(1−df)`.
/// Then `λ = A·(1/v − 1) = A·√(1−df) / (1−√(1−df))`.
///
/// Returns 0.0 if `df >= 1` (unpenalised) or `MBOOST_LAMBDA_MAX` if
/// `df` is very small.
pub fn df_to_lambda_scalar(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    df_target: f64,
    has_intercept: bool,
) -> f64 {
    let mut xtwx = 0.0f64;
    for i in 0..x.len() {
        xtwx += weights[i] * x[i] * x[i];
    }
    df_to_lambda_scalar_from_xtwx(xtwx, df_target, has_intercept)
}

fn df_to_lambda_scalar_from_xtwx(xtwx: f64, df_target: f64, has_intercept: bool) -> f64 {
    let max_df = if has_intercept { 2.0 } else { 1.0 };
    if df_target >= max_df {
        return 0.0;
    }
    if df_target <= 0.0 {
        return MBOOST_LAMBDA_MAX;
    }

    let df_pen = if has_intercept {
        df_target - 1.0
    } else {
        df_target
    };

    let a = xtwx + MBOOST_EPS;

    if df_pen <= 0.0 {
        return MBOOST_LAMBDA_MAX;
    }
    let v = 1.0 - (1.0 - df_pen.min(1.0)).sqrt();
    if v <= 0.0 {
        return MBOOST_LAMBDA_MAX;
    }
    let lambda = a * (1.0 / v - 1.0);
    if lambda > MBOOST_LAMBDA_MAX {
        MBOOST_LAMBDA_MAX
    } else {
        lambda
    }
}

/// Minimum positive-weight observations a candidate hinge knot must leave on
/// each side of the cut. Mirrors `btree`'s `minbucket`, so a selected hinge is
/// never a knife-edge cut backed by a handful of rows.
const HINGE_MIN_BIN: usize = 5;
/// Candidate knots for the adaptive hinge learner: evenly spaced interior
/// quantiles of the distinct values of `x`, ascending and deduplicated.
///
/// Knots sit on actual observed values between the extremes, so every
/// candidate splits the sample into two non-empty sides. An empty grid means
/// the column has fewer than three distinct values and cannot bend.
pub fn hinge_knot_grid(x: &[f64], n_candidates: usize) -> Vec<f64> {
    let mut unique: Vec<f64> = x.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique.dedup();
    if unique.len() < 3 {
        return Vec::new();
    }
    let m = unique.len();
    let mut knots = Vec::with_capacity(n_candidates);
    for k in 1..=n_candidates {
        let position = k as f64 * (m - 1) as f64 / (n_candidates as f64 + 1.0);
        let index = (position.round() as usize).clamp(1, m - 2);
        let knot = unique[index];
        match knots.last() {
            Some(&last) if last >= knot => {}
            _ => knots.push(knot),
        }
    }
    knots
}

/// Weighted least squares of one hinge candidate: fit `u` on the hinge column
/// `by * max(0, direction * (x - knot))`, optionally weight-centered.
///
/// Returns `(coefficient, centering, residual SSE)` where the coefficient is
/// `numer / (denom + lambda)` with the (centered) column in both quadratic
/// terms and `lambda` a scalar ridge on the hinge coefficient — the same
/// closed form `mboost`'s `df2lambda` family solves for a single column.
///
/// Returns `None` when the cut leaves fewer than [`HINGE_MIN_BIN`]
/// positive-weight observations on either side, or when the penalized Gram is
/// degenerate.
fn hinge_wls(
    x: &[f64],
    by: Option<&[f64]>,
    knot: f64,
    direction: f64,
    u: &[f64],
    weights: &[f64],
    lambda: f64,
    center: bool,
    wu2: f64,
) -> Option<(f64, f64, f64)> {
    let n = x.len();
    let mut left_count = 0usize;
    let mut right_count = 0usize;
    let mut sumw = 0.0f64;
    let mut sum_wu = 0.0f64;
    let mut wnumer = 0.0f64;
    let mut wdenom = 0.0f64;
    let mut wu_col = 0.0f64;
    for i in 0..n {
        let wi = weights[i];
        if wi <= 0.0 {
            continue;
        }
        let hinge = if direction > 0.0 {
            (x[i] - knot).max(0.0)
        } else {
            (knot - x[i]).max(0.0)
        };
        if hinge > 0.0 {
            right_count += 1;
        } else {
            left_count += 1;
        }
        let column = match by {
            Some(values) => hinge * values[i],
            None => hinge,
        };
        sumw += wi;
        sum_wu += wi * u[i];
        wnumer += wi * column;
        wdenom += wi * column * column;
        wu_col += wi * column * u[i];
    }
    if right_count < HINGE_MIN_BIN || left_count < HINGE_MIN_BIN {
        return None;
    }
    let centering = if center && sumw > 0.0 {
        wnumer / sumw
    } else {
        0.0
    };
    let denom = wdenom - 2.0 * centering * wnumer + centering * centering * sumw + lambda;
    if denom <= f64::EPSILON {
        return None;
    }
    let numer = wu_col - centering * sum_wu;
    let beta = numer / denom;
    let sse = wu2 - beta * numer - lambda * beta * beta;
    Some((beta, centering, sse))
}

/// The best-fit hinge (n×1) column at `[knot, direction, coefficient,
/// centering]`, scaled by `by` and centered like the fit that chose it.
fn hinge_column(x: &[f64], by: Option<&[f64]>, coef: &[f64; 4], out: &mut [f64]) {
    let (knot, direction, coefficient, centering) = (coef[0], coef[1], coef[2], coef[3]);
    for i in 0..x.len() {
        let hinge = if direction > 0.0 {
            (x[i] - knot).max(0.0)
        } else {
            (knot - x[i]).max(0.0)
        };
        let column = match by {
            Some(values) => hinge * values[i],
            None => hinge,
        };
        out[i] = coefficient * (column - centering);
    }
}

/// Run the adaptive-hinge selection scan over every candidate knot in both
/// directions, keeping the least-residual-SSE fit.
///
/// Returns `(coefficient-vector, prediction, sse)`: the coefficient vector is
/// `[knot, direction, coefficient, centering]`, `prediction` is
/// `coefficient * (column - centering)` (pre-`nu`, matching every other
/// learner's `fit_step` contract), and `sse` is the residual sum of squares.
/// Returns `None` when no knot splits the sample into at least
/// [`HINGE_MIN_BIN`] positive-weight rows per side.
fn select_hinge(
    x: &[f64],
    by: Option<&[f64]>,
    u: &[f64],
    weights: &[f64],
    knots: &[f64],
    lambda: f64,
    center: bool,
) -> Option<([f64; 4], Vec<f64>, f64)> {
    if knots.is_empty() {
        return None;
    }
    let n = x.len();
    let wu2: f64 = u
        .iter()
        .zip(weights.iter())
        .map(|(&ui, &wi)| wi * ui * ui)
        .sum();
    let mut best: Option<([f64; 4], f64)> = None;
    for &knot in knots {
        for direction in [1.0f64, -1.0f64] {
            if let Some((beta, centering, sse)) =
                hinge_wls(x, by, knot, direction, u, weights, lambda, center, wu2)
            {
                let better = match &best {
                    None => true,
                    Some((_, best_sse)) => sse < *best_sse,
                };
                if better {
                    best = Some(([knot, direction, beta, centering], sse));
                }
            }
        }
    }
    let (coef, sse) = best?;
    let mut pred = vec![0.0f64; n];
    hinge_column(x, by, &coef, &mut pred);
    Some((coef, pred, sse))
}

/// Prepare the adaptive hinge learner (`bhinge`) from a raw covariate column.
///
/// The knot grid is built once from the raw covariate (fold-invariant, so CV
/// and stability-selection folds share it); the selected hinge and its
/// column-centering are fit per step with the current weights.
pub fn prepare_bhinge(
    x: Array1<f64>,
    lambda: f64,
    df: f64,
    center: bool,
    by: Option<Array1<f64>>,
    name: String,
    knots: Vec<f64>,
) -> PreparedLearner {
    let n = x.len();
    PreparedLearner::Bhinge {
        x: x.clone(),
        knots,
        weights: Array1::zeros(n),
        lambda: Some(lambda),
        df,
        center,
        by: by.clone(),
        name,
        beta_buf: Vec::new(),
        pred_buf: vec![0.0; n],
    }
}

/// Fit the adaptive hinge for the current residual `u`: run
/// [`select_hinge`] into the learner's scratch buffers and return the residual
/// SSE. With no eligible candidate (or no knot grid), writes a zero update so
/// the learner stays a valid — if inactive — candidate.
fn fit_bhinge(
    x: &[f64],
    by: Option<&[f64]>,
    u: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    knots: &[f64],
    lambda: f64,
    center: bool,
    beta_buf: &mut Vec<f64>,
    pred_buf: &mut [f64],
) -> f64 {
    let (coef, pred, sse) = match select_hinge(
        x,
        by,
        u.as_slice().expect("u must be contiguous"),
        weights.as_slice().expect("weights must be contiguous"),
        knots,
        lambda,
        center,
    ) {
        Some(selected) => selected,
        None => {
            let u_slice = u.as_slice().expect("u must be contiguous");
            let wu2: f64 = u_slice
                .iter()
                .zip(weights.iter())
                .map(|(&ui, &wi)| wi * ui * ui)
                .sum();
            ([0.0f64; 4], vec![0.0; u_slice.len()], wu2)
        }
    };
    beta_buf.clear();
    beta_buf.extend_from_slice(&coef);
    pred_buf.copy_from_slice(&pred);
    sse
}

/// One-dimensional, observation-count quantile binning.
///
/// Equal input values are never split. When `requested >= n_distinct`, every
/// distinct value gets its own bin and `values[index[i]] == x[i]` exactly.
/// Otherwise a tied run is assigned by its midpoint rank to one of the
/// requested quantile cells; empty cells caused by large tied runs are removed
/// and reported through `effective`. A bin representative is the ordinary
/// mean of its observations, keeping this weight-independent so CV and
/// stability-selection folds can share the structure.
#[derive(Clone, Debug)]
pub struct QuantileBinning {
    /// Per-observation bin assignment.
    pub index: Arc<[u32]>,
    /// Representative value per bin (the mean of its observations).
    pub values: Arc<[f64]>,
    /// Bin boundaries.
    pub cutpoints: Arc<[f64]>,
    /// Lower bound of the binned range.
    pub lower: f64,
    /// Upper bound of the binned range.
    pub upper: f64,
    /// Number of bins requested.
    pub requested: usize,
    /// Number of distinct observed bins after merging.
    pub effective: usize,
    /// Number of distinct covariate values.
    pub distinct: usize,
}

/// 2-D quantile grid binning shared across resampling folds.
#[derive(Clone, Debug)]
pub struct QuantileGridBinning {
    /// Per-observation occupied-cell assignment.
    pub index: Arc<[u32]>,
    /// First-marginal value per occupied cell.
    pub values1: Arc<[f64]>,
    /// Second-marginal value per occupied cell.
    pub values2: Arc<[f64]>,
    /// Marginal binning along the first axis.
    pub marginal1: QuantileBinning,
    /// Marginal binning along the second axis.
    pub marginal2: QuantileBinning,
    /// Lookup from a marginal pair to the occupied cell.
    pub cell_for_pair: Arc<[i32]>,
    /// Number of occupied cells.
    pub effective: usize,
}

/// Build a 2-D quantile grid binning from two covariate columns.
pub fn quantile_bin_grid(
    x1: ArrayView1<f64>,
    x2: ArrayView1<f64>,
    requested_per_axis: usize,
) -> Result<QuantileGridBinning, MboostrsError> {
    if x1.len() != x2.len() {
        return Err(MboostrsError::LearnerSetup(
            "bins".to_string(),
            format!("grid columns have lengths {} and {}", x1.len(), x2.len()),
        ));
    }
    let marginal1 = quantile_bin_column(x1, requested_per_axis)?;
    let marginal2 = quantile_bin_column(x2, requested_per_axis)?;
    let width = marginal2.effective;
    let mut pair_to_cell = std::collections::BTreeMap::<(u32, u32), usize>::new();
    let mut index = vec![0u32; x1.len()];
    let mut sums1: Vec<f64> = Vec::new();
    let mut sums2: Vec<f64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for row in 0..x1.len() {
        let pair = (marginal1.index[row], marginal2.index[row]);
        let next = pair_to_cell.len();
        let cell = *pair_to_cell.entry(pair).or_insert(next);
        if cell == sums1.len() {
            sums1.push(0.0);
            sums2.push(0.0);
            counts.push(0);
        }
        sums1[cell] += x1[row];
        sums2[cell] += x2[row];
        counts[cell] += 1;
        index[row] = cell as u32;
    }
    let values1 = sums1
        .iter()
        .zip(&counts)
        .map(|(&sum, &count)| sum / count as f64)
        .collect::<Vec<_>>();
    let values2 = sums2
        .iter()
        .zip(&counts)
        .map(|(&sum, &count)| sum / count as f64)
        .collect::<Vec<_>>();
    let mut cell_for_pair = vec![-1i32; marginal1.effective * width];
    for (&(left, right), &cell) in &pair_to_cell {
        cell_for_pair[left as usize * width + right as usize] = cell as i32;
    }
    let effective = values1.len();
    Ok(QuantileGridBinning {
        index: index.into(),
        values1: values1.into(),
        values2: values2.into(),
        marginal1,
        marginal2,
        cell_for_pair: cell_for_pair.into(),
        effective,
    })
}

/// Build a one-dimensional quantile binning from a covariate column.
pub fn quantile_bin_column(
    x: ArrayView1<f64>,
    requested: usize,
) -> Result<QuantileBinning, MboostrsError> {
    if requested == 0 {
        return Err(MboostrsError::LearnerSetup(
            "bins".to_string(),
            "bins must be a positive integer".to_string(),
        ));
    }
    if x.is_empty() {
        return Err(MboostrsError::LearnerSetup(
            "bins".to_string(),
            "cannot bin an empty covariate".to_string(),
        ));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(MboostrsError::LearnerSetup(
            "bins".to_string(),
            "cannot bin a covariate containing NaN or infinity".to_string(),
        ));
    }

    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&left, &right| x[left].total_cmp(&x[right]));

    let mut runs: Vec<(usize, usize, f64)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let value = x[order[start]];
        let mut end = start + 1;
        while end < n && x[order[end]] == value {
            end += 1;
        }
        runs.push((start, end, value));
        start = end;
    }
    let distinct = runs.len();
    if distinct > u32::MAX as usize {
        return Err(MboostrsError::LearnerSetup(
            "bins".to_string(),
            "the effective bin count exceeds the u32 index capacity".to_string(),
        ));
    }

    let mut index = vec![0u32; n];
    if requested >= distinct {
        let values = runs.iter().map(|run| run.2).collect::<Vec<_>>();
        let cutpoints = values
            .windows(2)
            .map(|pair| pair[0] + (pair[1] - pair[0]) * 0.5)
            .collect::<Vec<_>>();
        for (bin, &(run_start, run_end, _)) in runs.iter().enumerate() {
            for position in run_start..run_end {
                index[order[position]] = bin as u32;
            }
        }
        return Ok(QuantileBinning {
            index: index.into(),
            values: values.into(),
            cutpoints: cutpoints.into(),
            lower: x[order[0]],
            upper: x[order[n - 1]],
            requested,
            effective: distinct,
            distinct,
        });
    }

    // Raw quantile-cell ids may skip when one tied run crosses several
    // cutpoints. Compress them so every stored basis row is occupied.
    let mut raw_to_dense = std::collections::BTreeMap::<usize, usize>::new();
    let mut sums: Vec<f64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut lows: Vec<f64> = Vec::new();
    let mut highs: Vec<f64> = Vec::new();
    for &(run_start, run_end, value) in &runs {
        let midpoint_twice = run_start + run_end - 1;
        let raw = ((midpoint_twice as u128 * requested as u128) / (2 * n) as u128)
            .min((requested - 1) as u128) as usize;
        let next_dense = raw_to_dense.len();
        let dense = *raw_to_dense.entry(raw).or_insert(next_dense);
        if dense == sums.len() {
            sums.push(0.0);
            counts.push(0);
            lows.push(value);
            highs.push(value);
        }
        let count = run_end - run_start;
        sums[dense] += value * count as f64;
        counts[dense] += count;
        highs[dense] = value;
        for position in run_start..run_end {
            index[order[position]] = dense as u32;
        }
    }
    let values = sums
        .iter()
        .zip(&counts)
        .map(|(&sum, &count)| sum / count as f64)
        .collect::<Vec<_>>();
    let effective = values.len();
    let cutpoints = highs
        .iter()
        .zip(lows.iter().skip(1))
        .map(|(&left, &right)| left + (right - left) * 0.5)
        .collect::<Vec<_>>();
    Ok(QuantileBinning {
        index: index.into(),
        values: values.into(),
        cutpoints: cutpoints.into(),
        lower: x[order[0]],
        upper: x[order[n - 1]],
        requested,
        effective,
        distinct,
    })
}

/// Prepare a one-column binned `bols` learner.
///
/// This is the no-intercept scalar form used by `gamboost` decompositions and
/// by each predictor in `glmboost`. Prior weights affect centering and the
/// denominator, but not bin membership, so a [`QuantileBinning`] can be shared
/// by resampling folds and re-prepared with their fold weights.
pub fn prepare_binned_scalar_bols(
    binning: QuantileBinning,
    weights: ArrayView1<f64>,
    lambda: f64,
    center: bool,
    name: &str,
) -> PreparedLearner {
    let n = binning.index.len();
    assert_eq!(weights.len(), n, "binned bols weights length mismatch");
    let k = binning.values.len();
    let sumw: f64 = weights.sum();
    let centering = if center && sumw > 0.0 {
        let mut total = 0.0f64;
        for i in 0..n {
            total += weights[i] * binning.values[binning.index[i] as usize];
        }
        total / sumw
    } else {
        0.0
    };
    let bin_values = binning
        .values
        .iter()
        .map(|&value| value - centering)
        .collect::<Vec<_>>();
    let mut scalar_denom = lambda;
    for i in 0..n {
        let value = bin_values[binning.index[i] as usize];
        scalar_denom += weights[i] * value * value;
    }

    PreparedLearner::BinnedScalarBols {
        bin_index: binning.index,
        bin_values: bin_values.into(),
        exact: binning.requested >= binning.distinct,
        penalty: lambda,
        scalar_denom,
        name: name.to_string(),
        centering,
        center,
        sumw: if center { sumw } else { 0.0 },
        n_observations: n,
        bin_weighted_u: vec![0.0; k],
        beta_buf: Vec::new(),
        pred_buf: Vec::new(),
    }
}

/// Prepare a scalar bols learner from a single covariate column.
///
/// `center=true` subtracts the weighted mean from `x` (R's default for
/// bare identifiers in a formula). `lambda` is the ridge penalty (0 = OLS).
pub fn prepare_scalar_bols(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    lambda: f64,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> PreparedLearner {
    let n = x.len();

    let mut design = x.to_owned();
    if let Some(by_vec) = by {
        for i in 0..n {
            design[i] *= by_vec[i];
        }
    }

    let sw: f64 = weights.sum();
    let centering = if center {
        if sw > 0.0 {
            design.dot(&weights) / sw
        } else {
            0.0
        }
    } else {
        0.0
    };

    if center {
        design.mapv_inplace(|v| v - centering);
    }

    let mut weighted_design_t = Array1::zeros(n);
    let mut scalar_denom = lambda;
    for i in 0..n {
        weighted_design_t[i] = weights[i] * design[i];
        scalar_denom += weights[i] * design[i] * design[i];
    }

    PreparedLearner::ScalarBols {
        design,
        penalty: lambda,
        weighted_design_t,
        scalar_denom,
        name: name.to_string(),
        centering,
        // Centring here is R's bols(x, intercept = TRUE): the learner holds
        // both an intercept and a slope, so the hat matrix gets both.
        sumw: if center { sw } else { 0.0 },
        weights: weights.to_owned(),
        beta_buf: Vec::new(),
        pred_buf: Vec::new(),
    }
}

/// Prepare a scalar bols learner with a **fixed** centering value.
///
/// This matches R's `glmboost` behavior: the centering (unweighted column means)
/// is computed once from the full data and reused across all fold refits in cvrisk.
/// Only the weighted OLS quantities (weighted_design_t, scalar_denom) are
/// recomputed with the new fold weights; the design matrix centering stays fixed.
pub fn prepare_scalar_bols_fixed_center(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    lambda: f64,
    centering: f64,
    name: &str,
) -> PreparedLearner {
    let n = x.len();
    let mut design = x.to_owned();
    design.mapv_inplace(|v| v - centering);

    let mut weighted_design_t = Array1::zeros(n);
    let mut scalar_denom = lambda;
    for i in 0..n {
        weighted_design_t[i] = weights[i] * design[i];
        scalar_denom += weights[i] * design[i] * design[i];
    }

    PreparedLearner::ScalarBols {
        design,
        penalty: lambda,
        weighted_design_t,
        scalar_denom,
        name: name.to_string(),
        centering,
        // glmboost's per-predictor learner is a single centred column; the
        // intercept is its own learner, so this one contributes one degree of
        // freedom, not two.
        sumw: 0.0,
        weights: weights.to_owned(),
        beta_buf: Vec::new(),
        pred_buf: Vec::new(),
    }
}

/// Prepare a binned block bols learner by snapping each observation to its
/// bin representative of the shared compact basis.
pub fn prepare_binned_block_bols(
    binning: &QuantileBinning,
    basis: Arc<Array2<f64>>,
    weights: ArrayView1<f64>,
    penalty: Array2<f64>,
    center: bool,
    name: &str,
) -> PreparedLearner {
    prepare_indexed_block_bols(binning.index.clone(), basis, weights, penalty, center, name)
}

/// Prepare a block bols learner over a shared compact basis indexed by an
/// explicit per-observation bin assignment.
pub fn prepare_indexed_block_bols(
    bin_index: Arc<[u32]>,
    basis: Arc<Array2<f64>>,
    weights: ArrayView1<f64>,
    penalty: Array2<f64>,
    center: bool,
    name: &str,
) -> PreparedLearner {
    let n = bin_index.len();
    let (k, p) = basis.dim();
    assert_eq!(weights.len(), n, "binned block weights length mismatch");
    assert_eq!(
        k,
        bin_index.iter().copied().max().unwrap_or(0) as usize + 1,
        "binned block basis/bin mismatch"
    );
    assert_eq!(penalty.dim(), (p, p), "binned block penalty shape mismatch");
    let mut bin_weights = vec![0.0f64; k];
    for i in 0..n {
        bin_weights[bin_index[i] as usize] += weights[i];
    }
    let sumw: f64 = bin_weights.iter().sum();
    let centering = if center && sumw > 0.0 {
        Array1::from_iter((0..p).map(|column| {
            (0..k)
                .map(|bin| bin_weights[bin] * basis[[bin, column]])
                .sum::<f64>()
                / sumw
        }))
    } else {
        Array1::zeros(p)
    };
    let basis = if center {
        let mut centered = basis.as_ref().clone();
        for column in 0..p {
            let mean = centering[column];
            centered
                .column_mut(column)
                .mapv_inplace(|value| value - mean);
        }
        Arc::new(centered)
    } else {
        basis
    };
    let mut gram = penalty.clone();
    for bin in 0..k {
        let weight = bin_weights[bin];
        if weight == 0.0 {
            continue;
        }
        for left in 0..p {
            let scaled = weight * basis[[bin, left]];
            for right in 0..p {
                gram[[left, right]] += scaled * basis[[bin, right]];
            }
        }
    }
    PreparedLearner::BinnedBlockBols {
        bin_index,
        basis,
        penalty,
        lambda: None,
        solver: GramSolver::new(&gram),
        name: name.to_string(),
        centering,
        n_observations: n,
        bin_weighted_u: vec![0.0; k],
        rhs_buf: vec![0.0; p],
        beta_buf: vec![0.0; p],
        bin_pred_buf: vec![0.0; k],
        pred_buf: Vec::new(),
    }
}

/// Result of detecting the Kronecker-sum ridge structure inside a materialised
/// `bbs %X% brandom`-style interaction design.
pub(crate) struct FactoredTensor {
    /// Left marginal design `(n, p_l)`.
    x_l: Array2<f64>,
    /// Per-row right-marginal column (`0..n_groups`), as `groups[i]`.
    groups: Vec<u32>,
    /// Left marginal column count.
    p_l: usize,
    /// Right marginal column count.
    n_groups: usize,
    /// Scaled `p_l×p_l` penalty block, identical across groups.
    penalty_block: Array2<f64>,
}

/// Try to factor a materialised row-wise tensor `T[i, l·ng + g] = x_l[i,l]·δ(g,
/// groups[i])` out of a plain (`n × p`) design with its (`p × p`) penalty.
///
/// Returns `Some` only when both directions of the interaction structure are
/// exact within `1e-12`:
///
/// - **design**: every row's non-zero columns fall in a single residue class
///   modulo `ng`, so `x_l[i,l] = design[i, l·ng + groups[i]]` is a lossless
///   compression;
/// - **penalty**: the matrix is block-diagonal over the `ng` residue classes
///   and all blocks equal the first, so `λK = kron(penalty_block, I_ng)` and
///   each group's normal equations decouple.
///
/// Candidates are tried with the largest `ng` first, which pins the split for
/// both penalized and unpenalized interactions: a genuine interaction is the
/// only divisor whose block-equality check survives, and when the penalty is
/// zero the design lattice alone requires `ng | stride` (for a left marginal
/// whose rows span several columns) or falls through to the materialised path.
pub(crate) fn detect_factored_tensor(
    design: &Array2<f64>,
    penalty: &Array2<f64>,
) -> Option<FactoredTensor> {
    let (n, p) = design.dim();
    if n == 0 || p < 4 {
        return None;
    }
    if penalty.dim() != (p, p) {
        return None;
    }
    let mut candidates: Vec<usize> = (2..p).filter(|ng| p % ng == 0 && p / ng >= 2).collect();
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    if candidates.is_empty() {
        return None;
    }
    let d = design.as_slice()?;
    const TOL: f64 = 1e-12;
    'candidate: for &ng in &candidates {
        let p_l = p / ng;
        // Design lattice: non-zero columns of each row share one residue mod ng.
        let mut groups = vec![0u32; n];
        for i in 0..n {
            let row = &d[i * p..(i + 1) * p];
            let mut residue = None;
            for (column, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    let r = column % ng;
                    match residue {
                        None => {
                            residue = Some(r);
                            groups[i] = r as u32;
                        }
                        Some(previous) if previous != r => continue 'candidate,
                        Some(_) => {}
                    }
                }
            }
        }
        // Penalty: off-block entries zero, blocks identical to the first.
        for a in 0..p {
            for b in 0..p {
                if a % ng != b % ng && penalty[[a, b]].abs() > TOL {
                    continue 'candidate;
                }
            }
        }
        for g in 1..ng {
            for l1 in 0..p_l {
                for l2 in 0..p_l {
                    let va = penalty[[l1 * ng, l2 * ng]];
                    let vb = penalty[[l1 * ng + g, l2 * ng + g]];
                    if (vb - va).abs() > TOL * va.abs().max(vb.abs()).max(1.0) {
                        continue 'candidate;
                    }
                }
            }
        }
        let mut x_l = Array2::<f64>::zeros((n, p_l));
        for i in 0..n {
            let g = groups[i] as usize;
            for l in 0..p_l {
                x_l[[i, l]] = d[i * p + l * ng + g];
            }
        }
        let mut penalty_block = Array2::<f64>::zeros((p_l, p_l));
        for l1 in 0..p_l {
            for l2 in 0..p_l {
                penalty_block[[l1, l2]] = penalty[[l1 * ng, l2 * ng]];
            }
        }
        return Some(FactoredTensor {
            x_l,
            groups,
            p_l,
            n_groups: ng,
            penalty_block,
        });
    }
    None
}

/// Build the per-group ridge solvers for a factored interaction learner.
///
/// `n_groups` is the right-marginal column count the factorization was built
/// with, not the number of groups the rows happen to occupy. A residue class
/// with no observations still owns `p_l` columns of the materialised design
/// and therefore still needs its solver: every fit path indexes `solvers` over
/// `0..n_groups`. Its Gram is the penalty block alone, so the coefficients it
/// produces are zero, matching the materialised learner.
fn build_factored_solvers(
    x_l: &Array2<f64>,
    groups: &[u32],
    weights: ArrayView1<f64>,
    penalty_block: &Array2<f64>,
    n_groups: usize,
) -> Vec<GramSolver> {
    let n = x_l.nrows();
    let p_l = x_l.ncols();
    let mut solvers = Vec::with_capacity(n_groups);
    for gr in 0..n_groups {
        let mut gram = penalty_block.clone();
        for i in 0..n {
            if groups[i] as usize != gr {
                continue;
            }
            let wi = weights[i];
            if wi == 0.0 {
                continue;
            }
            for j in 0..p_l {
                let xj = x_l[[i, j]];
                if xj == 0.0 {
                    continue;
                }
                let scaled = wi * xj;
                for k in 0..p_l {
                    gram[[j, k]] += scaled * x_l[[i, k]];
                }
            }
        }
        solvers.push(GramSolver::new(&gram));
    }
    solvers
}

/// Prepare a block-bols learner that first tries to factor an interaction
/// tensor.
///
/// A `bbs %X% brandom`-style interaction with an identity right penalty
/// decouples into `n_groups` independent ridge systems sharing one
/// `p_l×p_l` penalty block; when the design and penalty carry that exact
/// structure the result is a compact `FactoredBlockBols`. Otherwise the
/// plain materialised `BlockBols` is returned. Only the interaction
/// assembly path calls this — plain `brandom`/`bns`/`bmrf`/`bspatial`
/// bases are always materialised, because their bookkeeping (level codes,
/// centering, per-region penalties) assumes the block form.
pub(crate) fn prepare_block_bols_interaction(
    design: Array2<f64>,
    weights: ArrayView1<f64>,
    penalty: Array2<f64>,
    name: &str,
) -> PreparedLearner {
    let n = design.nrows();
    if let Some(FactoredTensor {
        x_l,
        groups,
        p_l,
        n_groups,
        penalty_block,
    }) = detect_factored_tensor(&design, &penalty)
    {
        let solvers = build_factored_solvers(&x_l, &groups, weights, &penalty_block, n_groups);
        return PreparedLearner::FactoredBlockBols {
            x_l,
            groups,
            p_l,
            n_groups,
            penalty_block,
            solvers,
            name: name.to_string(),
            weights: weights.to_owned(),
            rhs_buf: vec![0.0; n_groups * p_l],
            sol_buf: vec![0.0; p_l],
            beta_buf: vec![0.0; p_l * n_groups],
            pred_buf: vec![0.0; n],
        };
    }
    prepare_block_bols(design, weights, penalty, false, name)
}

/// Prepare a block bols learner from a multi-column design matrix.
///
/// `penalty` is the λK matrix (typically λ·I). Centering subtracts
/// the weighted column means from each column.
pub fn prepare_block_bols(
    design: Array2<f64>,
    weights: ArrayView1<f64>,
    penalty: Array2<f64>,
    center: bool,
    name: &str,
) -> PreparedLearner {
    let (n, p) = (design.nrows(), design.ncols());
    // The sparse copy is of the uncentered design. When the design is not
    // centred here (`center == false`) the stored design *is* that design,
    // so the copy can be built lazily on first use — the Gaussian Gram
    // route never reads it and no longer retains it (#738). When centring
    // follows, the stored design differs from the copy's source, so the
    // copy is built now, before the weighted means are subtracted. Small
    // or dense designs return None and keep the dense faer route.
    let sparse = if center {
        SparseRows::from_dense_if_sparse(design.view(), SPARSE_ROWS_MAX_DENSITY)
            .map(|rows| Arc::new(SparseRowsCache::prebuilt(Some(rows))))
    } else if n >= SPARSE_ROWS_MIN_ROWS && n <= u32::MAX as usize && p <= u32::MAX as usize {
        Some(Arc::new(SparseRowsCache::new()))
    } else {
        None
    };
    let mut d = design;

    let centering = if center {
        let sw: f64 = weights.sum();
        let c = if sw > 0.0 {
            (0..p)
                .map(|j| d.column(j).dot(&weights) / sw)
                .collect::<Array1<f64>>()
        } else {
            Array1::zeros(p)
        };
        for j in 0..p {
            let cj = c[j];
            d.column_mut(j).mapv_inplace(|v| v - cj);
        }
        c
    } else {
        Array1::zeros(p)
    };

    let d = into_column_major(d);
    let gram = penalized_weighted_gram(&d, weights, &penalty);
    let solver = GramSolver::new(&gram);

    PreparedLearner::BlockBols {
        design: d,
        sparse,
        penalty,
        lambda: None,
        solver,
        name: name.to_string(),
        centering,
        weights: weights.to_owned(),
        weighted_gram: None,
        wu_buf: vec![0.0; n],
        rhs_buf: vec![0.0; p],
        beta_buf: vec![0.0; p],
        pred_buf: vec![0.0; n],
    }
}

/// Prepare a block bols learner from a materialized design and penalty,
/// sharing one `X'WX` between the df→λ calibration and the solver.
///
/// This is the tensor-spatial preparation core. `X'WX` is computed once
/// (`weighted_gram_blocked`); the smoothing parameter is calibrated from it
/// ([`df_to_lambda_from_gram`], unless `lambda_override` is set); and the solver
/// Gram is assembled as `X'WX + λK`, skipping the second O(n·p²) pass that a
/// λK-seeded `penalized_weighted_gram` would pay. The unpenalized Gram is
/// retained as an exact CSR copy for the Gaussian Gram route's self-block.
///
/// `design` must already be the final (centered) design and `penalty` the
/// unscaled `K`. The result is mathematically identical to routing through
/// `penalized_weighted_gram`; only the accumulation order of the dense
/// `X'WX + λK` sum differs (ndarray `general_mat_mul` vs the λK-seeded faer
/// kernel), a rounding-level difference bounded by the tensor-parity tests.
pub fn prepare_block_bols_shared_gram(
    design: Array2<f64>,
    weights: ArrayView1<f64>,
    penalty: Array2<f64>,
    df: f64,
    lambda_override: Option<f64>,
    name: &str,
) -> PreparedLearner {
    let xtwx = weighted_gram_blocked(design.view(), weights.view());
    let lambda = lambda_override.unwrap_or_else(|| df_to_lambda_from_gram(&xtwx, &penalty, df));
    let stash = crate::linalg::CsrMatrix::try_from_dense(xtwx.view());
    // The learner's centering is zeros, so the sparse copy of the design is
    // the stored design itself; build it lazily, as in the other
    // uncentered preparations (#738).
    let sparse = if design.nrows() >= SPARSE_ROWS_MIN_ROWS
        && design.nrows() <= u32::MAX as usize
        && design.ncols() <= u32::MAX as usize
    {
        Some(Arc::new(SparseRowsCache::new()))
    } else {
        None
    };
    let d = into_column_major(design);
    let (n, p) = (d.nrows(), d.ncols());
    let scaled_penalty = penalty.mapv(|value| value * lambda);
    let gram = &xtwx + &scaled_penalty;
    let solver = GramSolver::new(&gram);

    PreparedLearner::BlockBols {
        design: d,
        sparse,
        penalty: scaled_penalty,
        lambda: None,
        solver,
        name: name.to_string(),
        centering: Array1::zeros(p),
        weights: weights.to_owned(),
        weighted_gram: stash,
        wu_buf: vec![0.0; n],
        rhs_buf: vec![0.0; p],
        beta_buf: vec![0.0; p],
        pred_buf: vec![0.0; n],
    }
    .with_effective_lambda(Some(lambda))
}

/// Bulk-fit all scalar bols learners in a single pass over the pseudo-response.
///
/// Takes **column-major** packed matrices `X_T` (J, n) and `WXT_T` (J, n),
/// where row `j` is the contiguous column `j` of the original (n, J) matrices.
/// Also takes `denoms[J]`. Returns:
/// - `betas[J]`: coefficient for each learner
/// - `sses[J]`: weighted SSE for each learner
///
/// Column-major storage ensures contiguous memory access in the inner loop
/// over observations, which is critical for cache performance at large n.
pub fn componentwise_linear_fit(
    x_t: ArrayView2<f64>,
    wxt_t: ArrayView2<f64>,
    denoms: ArrayView1<f64>,
    u: ArrayView1<f64>,
    w: ArrayView1<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let j = x_t.nrows();
    let n = x_t.ncols();

    let x_std = x_t.as_standard_layout();
    let wxt_std = wxt_t.as_standard_layout();

    let mut betas = Vec::with_capacity(j);
    let mut sses = Vec::with_capacity(j);

    for col in 0..j {
        let mut numer = 0.0f64;
        for i in 0..n {
            numer += wxt_std[[col, i]] * u[i];
        }
        let beta = if denoms[col] > 0.0 {
            numer / denoms[col]
        } else {
            0.0
        };
        let mut sse = 0.0f64;
        for i in 0..n {
            let pred_i = x_std[[col, i]] * beta;
            let r = u[i] - pred_i;
            sse += w[i] * r * r;
        }
        betas.push(beta);
        sses.push(sse);
    }
    (betas, sses)
}

/// Extract packed arrays from a slice of ScalarBols learners.
/// Returns transposed (column-major) matrices: `X_T` (J, n) and `WXT_T` (J, n),
/// where row j = column j of the original design. Also returns `denoms[J]`.
/// Returns `None` if any learner is not a ScalarBols.
pub fn pack_scalar_bols(
    learners: &[PreparedLearner],
) -> Option<(Array2<f64>, Array2<f64>, Array1<f64>)> {
    let n = match learners.first()? {
        PreparedLearner::ScalarBols { design, .. } => design.len(),
        PreparedLearner::BinnedScalarBols { .. } => return None,
        PreparedLearner::BinnedBlockBols { .. } => return None,
        PreparedLearner::BlockBols { .. } => return None,
        PreparedLearner::BmonoBbs { .. } => return None,
        PreparedLearner::Btree { .. } => return None,
        PreparedLearner::Bhinge { .. } => return None,
        PreparedLearner::KroneckerBols { .. } => return None,
        PreparedLearner::BspatialStreamed { .. } => return None,
        PreparedLearner::FactoredBlockBols { .. } => return None,
    };
    let j = learners.len();
    let mut x_t = Array2::zeros((j, n));
    let mut wxt_t = Array2::zeros((j, n));
    let mut denoms = Array1::zeros(j);

    for (col, learner) in learners.iter().enumerate() {
        match learner {
            PreparedLearner::ScalarBols {
                design,
                weighted_design_t,
                scalar_denom,
                ..
            } => {
                for i in 0..n {
                    x_t[[col, i]] = design[i];
                    wxt_t[[col, i]] = weighted_design_t[i];
                }
                denoms[col] = *scalar_denom;
            }
            PreparedLearner::BinnedScalarBols { .. } => return None,
            PreparedLearner::BinnedBlockBols { .. } => return None,
            PreparedLearner::BlockBols { .. } => return None,
            PreparedLearner::BmonoBbs { .. } => return None,
            PreparedLearner::Btree { .. } => return None,
            PreparedLearner::Bhinge { .. } => return None,
            PreparedLearner::KroneckerBols { .. } => return None,
            PreparedLearner::BspatialStreamed { .. } => return None,
            PreparedLearner::FactoredBlockBols { .. } => return None,
        }
    }
    Some((x_t, wxt_t, denoms))
}

/// Packed scalar data for the hybrid engine path.
///
/// `scalar_indices[j]` maps packed column `j` → original learner index.
/// `block_indices` lists the non-scalar learner indices that need `fit_step()`.
pub struct PackedScalar {
    /// Packed design columns, one row per learner (`p_max × n`).
    pub x_t: Array2<f64>,
    /// Packed weighted transpose columns, one row per learner.
    pub wxt_t: Array2<f64>,
    /// Per-learner denominators `xᵀWx + λ`.
    pub denoms: Array1<f64>,
    /// Original learner indices of packed scalar columns.
    pub scalar_indices: Vec<usize>,
    /// Original learner indices that need the general `fit_step` path.
    pub block_indices: Vec<usize>,
}

/// Pack the scalar learners of a prepared set into column-stacked buffers,
/// returning `None` when no scalar learner is present.
pub fn pack_scalar_bols_partial(learners: &[PreparedLearner]) -> Option<PackedScalar> {
    let n = match learners.first()? {
        PreparedLearner::ScalarBols { design, .. } => design.len(),
        PreparedLearner::BinnedScalarBols { n_observations, .. } => *n_observations,
        PreparedLearner::BinnedBlockBols { n_observations, .. } => *n_observations,
        PreparedLearner::BlockBols { design, .. } => design.nrows(),
        PreparedLearner::BmonoBbs { design, .. } => design.nrows(),
        PreparedLearner::Btree { design, .. } => design.nrows(),
        PreparedLearner::Bhinge { x, .. } => x.len(),
        PreparedLearner::KroneckerBols { x1, n_resp, .. } => x1.nrows() * *n_resp,
        PreparedLearner::BspatialStreamed { b1, .. } => b1.nrows(),
        PreparedLearner::FactoredBlockBols { x_l, .. } => x_l.nrows(),
    };
    let mut scalar_indices = Vec::new();
    let mut block_indices = Vec::new();
    for (i, l) in learners.iter().enumerate() {
        if let PreparedLearner::ScalarBols { .. } = l {
            scalar_indices.push(i);
        } else {
            block_indices.push(i);
        }
    }
    if scalar_indices.is_empty() {
        return None;
    }
    let j = scalar_indices.len();
    let mut x_t = Array2::zeros((j, n));
    let mut wxt_t = Array2::zeros((j, n));
    let mut denoms = Array1::zeros(j);
    for (col, &si) in scalar_indices.iter().enumerate() {
        if let PreparedLearner::ScalarBols {
            design,
            weighted_design_t,
            scalar_denom,
            ..
        } = &learners[si]
        {
            for i in 0..n {
                x_t[[col, i]] = design[i];
                wxt_t[[col, i]] = weighted_design_t[i];
            }
            denoms[col] = *scalar_denom;
        }
    }
    Some(PackedScalar {
        x_t,
        wxt_t,
        denoms,
        scalar_indices,
        block_indices,
    })
}

// ── B-spline basis (de Boor recursion matching R's splineDesign) ────────

/// Build the full knot sequence for a B-spline basis, matching R's `bsplines()`.
///
/// R's construction:
///   dx = diff(boundary_knots) / (n_inner + 1)
///   `bk_lower = seq(bk[0] - degree*dx, bk[0], length.out = degree+1)`
///   `bk_upper = seq(bk[1], bk[1] + degree*dx, length.out = degree+1)`
///   `inner_knots = seq(bk[0], bk[1], length.out = n_inner+2)[2:(n_inner+1)]`
///   k_full = c(bk_lower, inner_knots, bk_upper)
pub fn bspline_knots(boundary_knots: (f64, f64), n_inner: usize, degree: usize) -> Vec<f64> {
    let (bk_lo, bk_hi) = boundary_knots;
    let dx = (bk_hi - bk_lo) / (n_inner + 1) as f64;

    let mut knots = Vec::with_capacity(2 * (degree + 1) + n_inner);

    for i in 0..=degree {
        knots.push(bk_lo - (degree - i) as f64 * dx);
    }
    for i in 1..=n_inner {
        knots.push(bk_lo + i as f64 * dx);
    }
    for i in 0..=degree {
        knots.push(bk_hi + i as f64 * dx);
    }

    knots
}

/// Evaluate B-spline basis matrix using the de Boor recursion.
///
/// Matches R's `splines::splineDesign(knots, x, ord=degree+1, outer.ok=TRUE)`.
/// Returns an (n, p) matrix where p = knots.len() - degree - 1.
///
/// The algorithm computes the (degree+1)-th order B-splines via the
/// Cox–de Boor recursion on the (degree+1) non-zero basis functions
/// that are active at each evaluation point.
pub fn bspline_basis(x: &[f64], knots: &[f64], degree: usize) -> Array2<f64> {
    /// Largest degree the per-row recursion holds in its stack arrays.
    const MAX_DEGREE: usize = 31;
    assert!(
        degree <= MAX_DEGREE,
        "B-spline degree {degree} exceeds the supported maximum {MAX_DEGREE}"
    );
    let n = x.len();
    let p = knots.len().saturating_sub(degree + 1);
    let mut basis = Array2::zeros((n, p));

    let ord = degree + 1;
    let r = degree;
    let nk = knots.len();
    // The recursion's scratch arrays stay on the stack: a heap allocation per
    // row was most of the cost of a large basis. Entries of `b` are written
    // before they are read, so only the first entry is reset for each row.
    let mut deltar = [0.0f64; MAX_DEGREE];
    let mut deltal = [0.0f64; MAX_DEGREE];
    let mut b = [0.0f64; MAX_DEGREE + 1];

    for (i, &xi) in x.iter().enumerate() {
        // Clamp left so that left - j (j = 0..r) and left + j + 1 stay in
        // bounds. When xi is outside the domain, find_interval can return 0,
        // which would underflow left - j; R clamps such points to the boundary,
        // and so do we, by pinning left to [r, nk - ord).
        let left = find_interval(knots, xi).clamp(r, nk.saturating_sub(ord));

        // deltar[j] = knots[left+j+1] - xi, deltal[j] = xi - knots[left-j].
        for j in 0..r {
            deltar[j] = knots[left + j + 1] - xi;
            deltal[j] = xi - knots[left - j];
        }

        // Order 1: only the first basis function is 1.
        b[0] = 1.0;
        // Cox-de Boor recursion, order k = 1..degree.
        for k in 1..ord {
            let mut saved = 0.0f64;
            for j in 0..k {
                let denom = deltar[j] + deltal[k - 1 - j];
                let term = if denom != 0.0 { b[j] / denom } else { 0.0 };
                b[j] = saved + deltar[j] * term;
                saved = deltal[k - 1 - j] * term;
            }
            b[k] = saved;
        }

        // b[0..ord] are the basis functions left-r, ..., left.
        let mut row = basis.row_mut(i);
        for k in 0..ord {
            let col = left - r + k;
            if col < p {
                row[col] = b[k];
            }
        }
    }

    basis
}

/// Evaluate B-spline basis value AND first derivative at a single point.
/// Returns (value_row, deriv_row), each of length p = knots.len() - degree - 1.
/// Used to compute boundary slopes for linear extrapolation.
fn bspline_basis_deriv1(xi: f64, knots: &[f64], degree: usize) -> (Vec<f64>, Vec<f64>) {
    let p = knots.len().saturating_sub(degree + 1);
    let ord = degree + 1;
    let r = degree;
    let nk = knots.len();

    let left_raw = find_interval(knots, xi);
    let left = left_raw.clamp(r, nk.saturating_sub(ord));

    let mut deltar = vec![0.0f64; r];
    let mut deltal = vec![0.0f64; r];
    for j in 0..r {
        deltar[j] = knots[left + j + 1] - xi;
        deltal[j] = xi - knots[left - j];
    }

    // Full de Boor triangle for the value (order = degree+1)
    let mut bdb: Vec<Vec<f64>> = Vec::with_capacity(ord);
    bdb.push(vec![0.0; 1]);
    bdb[0][0] = 1.0;

    for k in 1..ord {
        let mut row = vec![0.0; k + 1];
        let mut saved = 0.0;
        for j in 0..k {
            let denom = deltar[j] + deltal[k - 1 - j];
            let term = if denom != 0.0 {
                bdb[k - 1][j] / denom
            } else {
                0.0
            };
            row[j] = saved + deltar[j] * term;
            saved = deltal[k - 1 - j] * term;
        }
        row[k] = saved;
        bdb.push(row);
    }

    let mut val = vec![0.0f64; p];
    for j in 0..ord {
        let col = left - r + j;
        if col < p {
            val[col] = bdb[degree][j];
        }
    }

    // Derivative: dB_{i,ord}/dx = (ord-1) * (B_{i,ord-1}/(t_{i+ord-1}-t_i)
    // - B_{i+1,ord-1}/(t_{i+ord}-t_{i+1}))
    //
    // Evaluate the order-degree (degree-1) B-splines via bspline_basis,
    // then apply the formula directly.  Using degree-1 (not degree) is
    // critical — the old two-triangle merge passed `degree` producing
    // order-(degree+1) values and ~10× derivative errors at boundaries.
    let mut deriv = vec![0.0f64; p];
    if degree == 0 {
        return (val, deriv);
    }

    let b_low = bspline_basis(&[xi], knots, degree - 1);
    // b_low is (1, p_low) where p_low = knots.len() - degree.
    for col in 0..p {
        let i0 = col; // B_{col, ord-1}
        let i1 = col + 1; // B_{col+1, ord-1}
        let b_left = if i0 < b_low.ncols() {
            b_low[[0, i0]]
        } else {
            0.0
        };
        let b_right = if i1 < b_low.ncols() {
            b_low[[0, i1]]
        } else {
            0.0
        };
        let denom_left = knots[col + degree] - knots[col];
        let denom_right = if col + 1 + degree < nk {
            knots[col + 1 + degree] - knots[col + 1]
        } else {
            1.0
        };
        let term_left = if denom_left != 0.0 {
            degree as f64 * b_left / denom_left
        } else {
            0.0
        };
        let term_right = if denom_right != 0.0 {
            degree as f64 * b_right / denom_right
        } else {
            0.0
        };
        deriv[col] = term_left - term_right;
    }

    (val, deriv)
}

/// Evaluate B-spline basis with linear extrapolation beyond boundary knots.
///
/// Matches R's `bsplines(extrapolation=TRUE)`: in-domain points use the
/// standard B-spline basis; out-of-domain points use linear extrapolation
/// from the value and slope at the nearest boundary knot.
///
/// `boundary_knots` is `(lo, hi)` — the original data range used during fitting.
pub fn bspline_basis_extrap(
    x: &[f64],
    knots: &[f64],
    degree: usize,
    boundary_knots: (f64, f64),
) -> Array2<f64> {
    let (bk_lo, bk_hi) = boundary_knots;
    let p = knots.len().saturating_sub(degree + 1);

    // Evaluate value + derivative at the EXACT knot boundary values,
    // not the user-supplied boundary_knots which can differ by ~1e-9
    // due to floating-point, causing find_interval to return the wrong
    // interval and producing wrong derivatives.
    let k_lo = knots[degree];
    let k_hi = knots[knots.len() - degree - 1];

    let (val_lo, deriv_lo) = bspline_basis_deriv1(k_lo, knots, degree);
    let (val_hi, deriv_hi) = bspline_basis_deriv1(k_hi, knots, degree);

    let mut basis = bspline_basis(x, knots, degree);

    for (i, &xi) in x.iter().enumerate() {
        if xi < bk_lo {
            let dx = xi - k_lo;
            for j in 0..p {
                basis[[i, j]] = val_lo[j] + dx * deriv_lo[j];
            }
        } else if xi > bk_hi {
            let dx = xi - k_hi;
            for j in 0..p {
                basis[[i, j]] = val_hi[j] + dx * deriv_hi[j];
            }
        }
    }

    basis
}

/// Evaluate B-spline basis with k-th derivative (R `bbs(deriv = k)`).
///
/// For `deriv = 0`, returns the standard B-spline basis (same as
/// [`bspline_basis`]). For `deriv = 1`, returns the first derivative
/// of each basis function. `deriv` must be ≤ `degree`.
///
/// This uses the analytical derivative formula:
/// `B'_{i,d}(x) = d·[B_{i,d-1}(x)/(t_{i+d}−t_i) − B_{i+1,d-1}(x)/(t_{i+d+1}−t_{i+1})]`
///
/// Higher derivatives are obtained by applying the formula recursively on
/// the reduced-degree basis.
pub fn bspline_basis_with_deriv(
    x: &[f64],
    knots: &[f64],
    degree: usize,
    deriv: usize,
) -> Array2<f64> {
    assert!(
        deriv <= degree,
        "deriv ({deriv}) must not exceed degree ({degree})"
    );
    let n = x.len();
    if deriv == 0 {
        return bspline_basis(x, knots, degree);
    }

    // For deriv ≥ 1, compute the derivative basis per point.
    // Recurrence: B^{(k)}_{i,d}(x) = d · [ B^{(k-1)}_{i,d-1}(x) / (t_{i+d} - t_i)
    //                                  - B^{(k-1)}_{i+1,d-1}(x) / (t_{i+d+1} - t_{i+1}) ]
    //
    // For each x, start with the degree-(d-k) basis and apply k derivative
    // steps. This is simpler: compute N_{i,d-k}(x) (standard basis of reduced
    // degree), then use the closed-form k-th derivative coefficients.
    let p = knots.len().saturating_sub(degree + 1);
    let mut design = Array2::<f64>::zeros((n, p));

    for (ix, &xi) in x.iter().enumerate() {
        // Work buffer: derivative values at this x, one entry per basis column
        // For step s (1..=deriv), we compute B^{(s)} from B^{(s-1)} of degree-(s-1) lower.
        // Start with B_{i, degree-deriv+1} (deriv = 1 case: degree) ... no.
        //
        // Strategy: use the per-point derivative routine for deriv=1.
        // For higher derivs, recursively lower degree.

        let current = if deriv == 1 {
            let (_, dv) = bspline_basis_deriv1(xi, knots, degree);
            dv
        } else {
            // Higher deriv: build basis of degree d-1 with deriv-1
            // then apply one more derivative step
            let lower_deriv = bspline_basis_with_deriv(&[xi], knots, degree - 1, deriv - 1);
            let nk = knots.len();
            let d = degree;
            let p_higher = lower_deriv.ncols();
            let mut result = vec![0.0f64; p];
            for col in 0..p {
                let i0 = col;
                let i1 = col + 1;
                let b_left = if i0 < p_higher {
                    lower_deriv[[0, i0]]
                } else {
                    0.0
                };
                let b_right = if i1 < p_higher {
                    lower_deriv[[0, i1]]
                } else {
                    0.0
                };
                let denom_left = knots[col + d] - knots[col];
                let denom_right = if col + 1 + d < nk {
                    knots[col + 1 + d] - knots[col + 1]
                } else {
                    1.0
                };
                let term_left = if denom_left != 0.0 {
                    d as f64 * b_left / denom_left
                } else {
                    0.0
                };
                let term_right = if denom_right != 0.0 {
                    d as f64 * b_right / denom_right
                } else {
                    0.0
                };
                result[col] = term_left - term_right;
            }
            result
        };

        for j in 0..p {
            design[[ix, j]] = current[j];
        }
    }

    design
}

/// Find the knot interval index for a point, matching R's `splineDesign`.
///
/// Returns the index `left` such that `knots[left] <= xi < knots[left+1]`.
/// When `xi` equals a knot value, we choose the **rightmost** such interval
/// (i.e. the interval starting at that knot), matching R/scipy convention.
/// Points at or beyond the right boundary are placed in the last valid interval.
fn find_interval(knots: &[f64], xi: f64) -> usize {
    let nk = knots.len();
    if nk <= 1 {
        return 0;
    }

    // Left boundary: always in the first interval
    if xi <= knots[0] {
        return 0;
    }
    // Right boundary: last valid interval
    if xi >= knots[nk - 1] {
        return nk - 2;
    }

    // Binary search for the rightmost index `left` where knots[left] <= xi.
    // This gives the interval [knots[left], knots[left+1]) containing xi.
    // When xi equals a knot, we get the interval starting at that knot.
    let mut lo = 0usize;
    let mut hi = nk - 1;
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2; // upper mid to avoid infinite loop
        if knots[mid] <= xi {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Build a second-difference penalty matrix K = D'D, matching R's
/// `crossprod(diff(diag(p), differences=d))`.
///
/// For `differences=2` and `p` columns, D is (p-2, p) with the
/// second-difference operator rows, and K = D'D is (p, p) banded.
pub fn diff_penalty(p: usize, differences: usize) -> Array2<f64> {
    let d = raw_diff_matrix(p, differences);
    match d {
        Some(d_mat) => d_mat.t().dot(&d_mat),
        None => singular_penalty_guard(p, differences),
    }
}

fn singular_penalty_guard(p: usize, differences: usize) -> Array2<f64> {
    if p == 0 || differences == 0 {
        Array2::eye(p)
    } else {
        Array2::zeros((p, p))
    }
}

/// The raw difference matrix D itself — `(p − differences, p)` rows of the
/// `differences`-th finite-difference operator. `diff_penalty` is D'D. R's
/// default `bbs(..., center = TRUE)` centering ("differenceMatrix" in
/// `mboost:::X_bbs`) multiplies by D' and the inverse of D·D', so the raw
/// matrix, not the square penalty, is what that path needs.
///
/// Returns `None` for the degenerate cases where no rectangular difference
/// matrix exists (`differences == 0`, or `differences >= p`).
pub fn raw_diff_matrix(p: usize, differences: usize) -> Option<Array2<f64>> {
    if p == 0 || differences == 0 || differences >= p {
        return None;
    }

    let d_rows = p - differences;
    let mut d_mat = Array2::zeros((d_rows, p));

    let mut sign: f64 = 1.0;
    let mut binom = vec![0i64; differences + 1];
    binom[0] = 1;
    for k in 1..=differences {
        binom[k] = binom[k - 1] * (differences - k + 1) as i64 / k as i64;
    }

    for k in (0..=differences).rev() {
        sign = -sign;
        let val = sign * binom[k] as f64;
        for row in 0..d_rows {
            d_mat[[row, row + k]] = val;
        }
    }

    Some(d_mat)
}

/// Build a cyclic (periodic) difference penalty matrix.
///
/// Matches R mboost's cyclic P-spline penalty construction:
/// 1. D = diff(I_{p+d}, differences=d) — (p, p+d) difference matrix
/// 2. Save first `d` columns, drop them → (p, p)
/// 3. Add saved columns to last `d` columns (wraps the differencing)
/// 4. K = D'D — (p, p) penalty matrix
pub fn cyclic_diff_penalty(p: usize, differences: usize) -> Array2<f64> {
    match raw_cyclic_diff_matrix(p, differences) {
        Some(d) => d.t().dot(&d),
        None => singular_penalty_guard(p, differences),
    }
}

/// The wrapped square difference matrix D of the cyclic penalty, before it is
/// squared into `cyclic_diff_penalty` (see there for the construction).
pub fn raw_cyclic_diff_matrix(p: usize, differences: usize) -> Option<Array2<f64>> {
    if p == 0 || differences == 0 || differences >= p {
        return None;
    }

    let total_cols = p + differences;
    let d_rows = total_cols - differences; // = p

    let mut d_mat = Array2::zeros((d_rows, total_cols));

    let mut sign: f64 = 1.0;
    let mut binom = vec![0i64; differences + 1];
    binom[0] = 1;
    for k in 1..=differences {
        binom[k] = binom[k - 1] * (differences - k + 1) as i64 / k as i64;
    }

    for k in (0..=differences).rev() {
        sign = -sign;
        let val = sign * binom[k] as f64;
        for row in 0..d_rows {
            d_mat[[row, row + k]] = val;
        }
    }

    // Wrap: add first `differences` columns to last `differences` columns
    for row in 0..d_rows {
        for j in 0..differences {
            d_mat[[row, p - differences + j]] += d_mat[[row, j]];
        }
    }

    // Drop first `differences` columns → (p, p)
    Some(d_mat.slice(ndarray::s![.., differences..]).to_owned())
}

/// Build cyclic B-spline knot sequence for given boundary knots and inner knot count.
///
/// Matches R mboost's `cbs()` knot construction:
/// 1. Base knots: c(bk_lo, inner_knots, bk_hi)
/// 2. Prepend `degree` mirrored knots: `bk_lo - (bk_hi - base_knots[j])`
///
/// Returns the extended knot sequence (degree + n_inner + 2 knots).
pub fn cyclic_bspline_knots(boundary_knots: (f64, f64), n_inner: usize, degree: usize) -> Vec<f64> {
    let (bk_lo, bk_hi) = boundary_knots;
    let dx = (bk_hi - bk_lo) / (n_inner + 1) as f64;

    // Base knots: c(bk_lo, inner_knots, bk_hi)
    let mut base_knots = vec![bk_lo];
    for i in 1..=n_inner {
        base_knots.push(bk_lo + i as f64 * dx);
    }
    base_knots.push(bk_hi);
    let n_base = base_knots.len(); // n_inner + 2
    let ord = degree + 1;

    // Prepend `degree` mirrored knots
    let mut knots = Vec::with_capacity(n_base + degree);
    for j in (n_base - ord)..(n_base - 1) {
        knots.push(bk_lo - (bk_hi - base_knots[j]));
    }
    knots.extend(&base_knots);
    knots
}

/// The cutoff point `xc` for cyclic wrapping.
///
/// Points with `x > xc` are wrapped: evaluate at `x - period` and add.
/// Matches R's `knots[nKnots - ord + 1]` from the base (non-extended) knot set.
pub fn cyclic_xc(boundary_knots: (f64, f64), n_inner: usize, degree: usize) -> f64 {
    let (bk_lo, bk_hi) = boundary_knots;
    let dx = (bk_hi - bk_lo) / (n_inner + 1) as f64;
    let ord = degree + 1;
    // base_knots[n_base - ord] in 0-based, where n_base = n_inner + 2
    // = bk_lo + (n_inner + 2 - ord - 1) * dx = bk_lo + (n_inner - degree) * dx
    bk_lo + (n_inner + 2 - ord - 1) as f64 * dx
}

/// Evaluate the cyclic B-spline basis on the given extended knot sequence.
///
/// `knots` must be the output of `cyclic_bspline_knots`. For points beyond
/// `xc`, the basis is evaluated at `x - period` and added, enforcing
/// periodicity. Matches R mboost's `cbs()`.
pub fn cyclic_bspline_basis_from_knots(
    x: &[f64],
    knots: &[f64],
    degree: usize,
    boundary_knots: (f64, f64),
) -> Array2<f64> {
    let (bk_lo, bk_hi) = boundary_knots;
    let period = bk_hi - bk_lo;

    // Determine xc from the knot sequence: it's the knot at index (n_base - ord)
    // in the original base knot set. The base knot set starts at index `degree`
    // in the extended knot sequence. So xc = knots[degree + n_base - ord]
    // where n_base = knots.len() - degree.
    let n_base = knots.len() - degree; // n_inner + 2
    let ord = degree + 1;
    let xc = knots[degree + n_base - ord];

    let mut basis = bspline_basis(x, knots, degree);

    for (i, &xi) in x.iter().enumerate() {
        if xi > xc {
            let x_wrapped = xi - period;
            let basis_wrapped = bspline_basis(&[x_wrapped], knots, degree);
            for j in 0..basis.ncols() {
                basis[[i, j]] += basis_wrapped[[0, j]];
            }
        }
    }

    basis
}

/// Evaluate the cyclic B-spline basis, building knots internally.
///
/// Convenience wrapper that builds the knot sequence and delegates to
/// `cyclic_bspline_basis_from_knots`.
pub fn cyclic_bspline_basis(
    x: &[f64],
    boundary_knots: (f64, f64),
    n_inner: usize,
    degree: usize,
) -> Array2<f64> {
    let knots = cyclic_bspline_knots(boundary_knots, n_inner, degree);
    cyclic_bspline_basis_from_knots(x, &knots, degree, boundary_knots)
}
/// algorithm, matching R's `df2lambda()`.
///
/// Algorithm:
/// 1. A = X'WX + eps·K, eps = 1e-9
/// 2. R = chol(A)
/// 3. Rm = R^{-T} (back-solve of identity against R)
/// 4. d = eigenvalues of Rm'·K·Rm (sorted ascending)
/// 5. dfFun(λ) = 2·Σ 1/(1+λ·d_k) - Σ 1/(1+λ·d_k)²
/// 6. Brent search on [0, λ_max] for dfFun(λ) = df_target
pub fn df_to_lambda<S1, S2, S3>(
    x: &ArrayBase<S1, Ix2>,
    k: &ArrayBase<S2, Ix2>,
    weights: &ArrayBase<S3, Ix1>,
    df_target: f64,
) -> f64
where
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
    S3: ndarray::Data<Elem = f64>,
{
    // X'WX over blocks of rows: no weighted or transposed copy of the whole
    // design, which at 10^6 rows was two design-sized allocations per term.
    let xtwx = weighted_gram_blocked(x.view(), weights.view());
    df_to_lambda_from_gram(&xtwx, k, df_target)
}

/// [`df_to_lambda`] over a caller-supplied `X'WX`, so a preparation path
/// that already holds the weighted Gram (the tensor spatial learner, whose
/// Gram serves both the df→λ calibration and the penalized solver matrix)
/// does not compute the O(n·p²) product a second time (#276).
pub fn df_to_lambda_from_gram<S1: ndarray::Data<Elem = f64>, S2: ndarray::Data<Elem = f64>>(
    xtwx: &ArrayBase<S1, Ix2>,
    k: &ArrayBase<S2, Ix2>,
    df_target: f64,
) -> f64 {
    let p = xtwx.ncols();
    let xtwx_faer = Mat::from_fn(p, p, |i, j| xtwx[[i, j]]);

    let k_faer = Mat::from_fn(p, p, |i, j| k[[i, j]]);
    let lambda_max = solvable_lambda_max(&xtwx_faer, &k_faer);

    // A = X'WX + eps·K
    let mut a_faer = xtwx_faer;
    for j in 0..p {
        for l in 0..p {
            let v = a_faer.read(j, l) + MBOOST_EPS * k_faer.read(j, l);
            a_faer.write(j, l, v);
        }
    }

    let rm_faer = chol_backsolve_inv_t_faer(&a_faer);

    // kmrm = rm^T · K · rm  (all faer)
    let mut krm = Mat::zeros(p, p);
    faer::linalg::matmul::matmul(&mut krm, &k_faer, &rm_faer, None, 1.0, setup_parallelism());
    let mut kmrm = Mat::zeros(p, p);
    {
        let rm_t = rm_faer.transpose();
        faer::linalg::matmul::matmul(&mut kmrm, rm_t, krm, None, 1.0, setup_parallelism());
    }

    // Singular values are non-negative by construction, so unlike the
    // eigenvalues this replaced they need no clamp to keep `1/(1 + λd)` sane.
    let d = penalty_spectrum_faer(&kmrm);
    let lambda = lambda_from_spectrum(&d, df_target, lambda_max);

    // A design column with (almost) no data in its support -- a B-spline whose
    // knot interval an outlying minimum or maximum has left empty -- makes
    // `X'WX` singular in that direction, and `eps·K` turns it into a spectral
    // value near `1/eps = 1e9`. A dense SVD resolves values only to about
    // `p·EPSILON` times the largest one, so beside `1e9` everything below
    // roughly `5e-6` is noise. Those are exactly the penalized directions that
    // decide `df` at the root once `X'WX` is large (many rows, or frequency
    // weights). When they collapse to zero, the curve's floor rises above the
    // penalty's null-space dimension, the target looks unattainable, and the
    // capped lambda fits only the null space: a smooth term fits nothing.
    //
    // Where the primary spectrum's resolution is that coarse, check its answer
    // against the bounded pencil, whose values all lie in `[0, 1)`, and defer
    // to it whenever the two disagree beyond rounding. A well-resolved spectrum
    // is never re-derived, so the ordinary case is untouched.
    let resolution = d.last().copied().unwrap_or(0.0) * p as f64 * f64::EPSILON;
    if resolution > SPECTRUM_RESOLUTION_LIMIT {
        let bounded = bounded_pencil_spectrum(&a_faer, &k_faer);
        let df_at_lambda = df_from_spectrum(&bounded, lambda);
        if (df_at_lambda - df_target).abs() > DF_CALIBRATION_TOLERANCE * df_target.max(1.0) {
            return lambda_from_spectrum(&bounded, df_target, lambda_max);
        }
    }
    lambda
}

/// Resolution, relative to one, below which the primary penalty spectrum is
/// re-checked on the bounded pencil. The primary spectrum's absolute error is
/// about `p·EPSILON·d_max`; a well-populated design keeps `d_max` in the
/// hundreds, which puts that error near `1e-12`.
const SPECTRUM_RESOLUTION_LIMIT: f64 = 1e-10;

/// Relative df error at which a calibrated lambda is rejected and recomputed
/// from the bounded pencil.
///
/// The two spectra estimate the same eigenvalues, so agreement within rounding
/// is the only reason to prefer the primary one — and a df error is not the
/// scale the answer is read on. Near the floor `lambda` behaves like
/// `c / (df - rank)`, so a df discrepancy `δ` between the two spectra moves the
/// root by `lambda · δ / (df - rank)`. On the fixture's `clustered_200` at
/// `df = 2.05` (rank 2, margin 0.05) the primary spectrum disagreed with the
/// bounded pencil by 3.7e-7 of df — invisible at this tolerance's previous
/// value of 1e-3, but enough to put `lambda` 7.3e-6 relative from R's while
/// the bounded pencil's root sat 3.2e-7 away, the accuracy the fixture was
/// measured at. The tolerance therefore sits at rounding level: when the
/// resolution check runs at all, the two must actually agree before the
/// primary answer is kept. No cost decides the case — the bounded spectrum is
/// computed for the check either way.
const DF_CALIBRATION_TOLERANCE: f64 = 1e-9;

/// `2·Σ 1/(1+λd) − Σ 1/(1+λd)²`, R's `dfFun`, over a penalty spectrum.
fn df_from_spectrum(d: &[f64], lambda: f64) -> f64 {
    let mut df_val = 0.0f64;
    for &dk in d {
        let v = 1.0 / (1.0 + lambda * dk);
        df_val += 2.0 * v - v * v;
    }
    df_val
}

/// The spectrum of the pencil `(K, X'WX + eps·K)`, computed through the
/// pencil `(K, X'WX + (1 + eps)·K)` instead.
///
/// If `K v = d (A) v` with `A = X'WX + eps·K`, then `K v = σ (A + K) v` with
/// `σ = d / (1 + d)`: the two pencils share eigenvectors, and `d = σ/(1 − σ)`
/// recovers the original values exactly. Every `σ` lies in `[0, 1)`, so the
/// matrix whose singular values are taken has norm below one, and the small
/// values -- the ones `df` is read from -- keep about `p·EPSILON` absolute
/// accuracy however large the largest `d` is. A `d` of order `1e12` or more
/// loses relative accuracy through `1 − σ`, but at any lambda where `df` is
/// near a target such a direction contributes nothing.
///
/// `A + K` is also better conditioned than `A` in the direction that made the
/// primary spectrum coarse: there `X'WX` is zero and `A` is only `eps·K`,
/// while `A + K` is `(1 + eps)·K`.
fn bounded_pencil_spectrum(a: &Mat<f64>, k: &Mat<f64>) -> Vec<f64> {
    let p = a.nrows();
    let b = Mat::from_fn(p, p, |i, j| a.read(i, j) + k.read(i, j));
    let rb = chol_backsolve_inv_t_faer(&b);
    let mut krb = Mat::zeros(p, p);
    faer::linalg::matmul::matmul(&mut krb, k, &rb, None, 1.0, setup_parallelism());
    let mut rkr = Mat::zeros(p, p);
    faer::linalg::matmul::matmul(
        &mut rkr,
        rb.transpose(),
        krb,
        None,
        1.0,
        setup_parallelism(),
    );
    penalty_spectrum_faer(&rkr)
        .into_iter()
        .map(|sigma| {
            if sigma >= 1.0 {
                1.0 / f64::EPSILON
            } else {
                sigma / (1.0 - sigma)
            }
        })
        .collect()
}

/// Lambda at which the df curve of spectrum `d` equals `df_target`, following
/// `mboost:::df2lambda`'s branches, capped at `lambda_max`.
fn lambda_from_spectrum(d: &[f64], df_target: f64, lambda_max: f64) -> f64 {
    if d.is_empty() {
        return 0.0;
    }

    let df_fun = |lambda: f64| df_from_spectrum(d, lambda);

    // The search runs on R's bracket, `[0, options("mboost_lambdaMax")]`, not on
    // the conditioning-limited `lambda_max` computed above. Where the two
    // differ, `lambda_max` is the smaller, and using it as the upper end moves
    // every iterate of the search, which changes the answer on a flat df curve
    // even though both brackets contain the root. `lambda_max` is applied at
    // the end instead, as a cap on the value returned.
    let df_zero = df_fun(0.0);
    let df_at_r_max = df_fun(MBOOST_LAMBDA_MAX);

    if df_target >= df_zero {
        return 0.0;
    }
    // R: `if (df2l(lambdaMax) > 0) return(lambdaMax)`. The df curve never falls
    // to the target, so no finite lambda attains it.
    if df_at_r_max - df_target > 0.0 {
        return lambda_max.min(MBOOST_LAMBDA_MAX);
    }

    // `df_fun` is monotone decreasing in lambda (each term `2v - v^2` with
    // `v = 1/(1 + lambda*d)` decreases for `d > 0`), and the two early returns
    // above establish `df_fun(0) > df_target >= df_fun(1e15)`. That is a
    // bracketed monotone root, found exactly as `mboost:::df2lambda` finds it:
    // `uniroot(df2l, c(0, lambdaMax), tol = sqrt(.Machine$double.eps))`.
    //
    // `r_zeroin` is a statement-by-statement port of the algorithm R's
    // `uniroot` calls rather than an equivalent root-finder, because for a `df`
    // target at or near the dimension of `K`'s null space the curve is flat and
    // the returned lambda is decided by the sequence of iterates, not by the
    // tolerance. A scipy-style `brentq` on the same bracket, to the same
    // tolerance, solves the same equation and returns a different lambda there:
    // on the case study's 24-column basis at `df = 2` it lands three decades
    // away from R's 1.29e10. Both answer an ill-posed question; matching R
    // makes the fitted values agree, which is what the parity fixtures check.
    let lambda = crate::families::r_zeroin(
        &|lambda: f64| df_fun(lambda) - df_target,
        0.0,
        MBOOST_LAMBDA_MAX,
        crate::families::R_UNIROOT_TOL,
        1000,
    );

    // Cap at the largest lambda whose Gram still factorizes accurately. This
    // binds only where R's answer is itself past the resolution of a double —
    // a df at the floor of a rank-deficient penalty, such as a bivariate tensor
    // at its default df — and there the cap is what keeps the hat matrix inside
    // [0, 1]. See `solvable_lambda_max`.
    lambda.min(lambda_max)
}

/// The largest `lambda` at which `X'WX + lambda·K` is still accurate enough to
/// solve with, capped at `MBOOST_LAMBDA_MAX`.
///
/// `df_to_lambda` inverts a `df(lambda)` curve that flattens out: as `lambda`
/// grows, `df` falls to the dimension of `K`'s null space and stops. A `df` at
/// or below that floor is attained by no finite `lambda`, and both R and this
/// implementation then clamp at `MBOOST_LAMBDA_MAX = 1e15`. That clamp is where
/// the arithmetic breaks. The penalized directions of the Gram are then up to
/// `1e15` times larger than the unpenalized ones, which is past the resolution
/// of a double, so the unpenalized directions — the ones that carry the whole
/// fit, since the penalty has stopped shrinking them — are lost to rounding.
/// The Cholesky fails, `GramSolver` falls back to LU, and the hat matrix that
/// comes out has eigenvalues outside `[0, 1]`, some of them negative. A boosting
/// step through a negative hat eigenvalue moves the fit *against* the residual,
/// so the training risk rises instead of falling and the model diverges. This
/// was reachable through any bivariate tensor penalty at its default `df`:
/// `bspatial(x1, x2, df = 4)` and the `%X%` tensor of two `bbs` marginals both
/// have a 4-dimensional null space.
///
/// Past the point where the penalized directions are already shrunk to rounding
/// error, more `lambda` buys no further shrinkage — it only costs conditioning.
/// So back `lambda` off by decades until the Gram factorizes with a margin, and
/// return the first value that does. The margin is `EPSILON^(1/3)` on the
/// Cholesky diagonal ratio (`κ ≈ 2.7e10`), stricter than the
/// `EPSILON^(1/2)` (`κ ≈ 4.5e15`) at which [`CachedCholesky`] merely declares a
/// factorization non-degenerate: this `lambda` has to survive the update solve
/// at every boosting step, not just factorize once. The looser threshold still
/// leaves hat eigenvalues around `1.01`; this one holds them to `1 + 4e-6`.
///
/// A well-conditioned penalty (`brandom`'s identity, any `K` of full rank)
/// passes on the first probe and keeps `MBOOST_LAMBDA_MAX` unchanged, so this
/// costs one extra Cholesky and changes no result outside the degenerate case.
/// If no `lambda` down to `1e-5` factorizes — a design whose rank deficiency
/// lies inside `K`'s null space, where no penalty can help — it returns
/// `MBOOST_LAMBDA_MAX` and leaves the outcome to `GramSolver`'s own
/// regularization.
fn solvable_lambda_max(xtwx: &Mat<f64>, k: &Mat<f64>) -> f64 {
    /// Cholesky diagonal ratio a probe must clear to be considered solvable.
    const SOLVABLE_RATIO: f64 = 6.055454452393343e-6; // f64::EPSILON.powf(1.0 / 3.0)
    /// Decades below `MBOOST_LAMBDA_MAX` the backoff is willing to walk.
    const MAX_BACKOFF_DECADES: u32 = 20;

    let p = xtwx.nrows();
    if p == 0 {
        return MBOOST_LAMBDA_MAX;
    }

    let mut lambda = MBOOST_LAMBDA_MAX;
    for _ in 0..=MAX_BACKOFF_DECADES {
        let gram = Array2::from_shape_fn((p, p), |(i, j)| xtwx.read(i, j) + lambda * k.read(i, j));
        if let Ok(factor) = CachedCholesky::new(&gram) {
            if factor.diagonal_ratio() > SOLVABLE_RATIO {
                return lambda;
            }
        }
        lambda /= 10.0;
    }
    MBOOST_LAMBDA_MAX
}

/// Computes `L^{-T}` where `A = L·Lᵀ` (lower Cholesky), matching R's
/// `backsolve(chol(A), diag(p))`. Downstream this `Rm = L^{-T}` is used to form
/// `Rmᵀ K Rm`, whose eigenvalues equal those of the generalized problem
/// `(K, A)` — `df_to_lambda` depends on getting exactly those eigenvalues.
///
/// `A` is `X'WX + ε·K`, which is positive semidefinite; with `ε > 0` it is
/// normally positive-definite, so the Cholesky path is taken. If Cholesky fails
/// (tiny negative eigenvalues from round-off), we add a small diagonal jitter
/// scaled to `A`'s magnitude and retry — this yields a genuine `L^{-T}` of a PD
/// matrix close to `A`, preserving the eigenvalue contract. (A plain LU
/// `A^{-1}` would NOT: eigenvalues of `A^{-1} K A^{-1}` differ from those of
/// `A^{-1} K`.)
fn chol_backsolve_inv_t_faer(a: &Mat<f64>) -> Mat<f64> {
    let p = a.nrows();

    // L^{-T}: solve L Y = I (Y = L^{-1}) with faer's blocked triangular
    // solve, then transpose. The element-wise triple loop this replaces was
    // most of `df_to_lambda`'s time on a 576-column spatial tensor (0.26 s of
    // a 0.29 s learner setup, paid once per cross-validation fold).
    let backsolve = |l: &Mat<f64>| -> Mat<f64> {
        let mut inverse = Mat::<f64>::identity(p, p);
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            l.as_ref(),
            inverse.as_mut(),
            setup_parallelism(),
        );
        inverse.transpose().to_owned()
    };

    if let Ok(chol) = faer::linalg::solvers::Cholesky::try_new(a.as_ref(), faer::Side::Lower) {
        return backsolve(&chol.compute_l());
    }

    // Indefinite due to round-off: jitter the diagonal and retry. Scale the
    // jitter to the mean absolute diagonal so it is meaningful at any magnitude.
    let diag_scale = {
        let s: f64 = (0..p).map(|i| a.read(i, i).abs()).sum();
        if p > 0 && s > 0.0 {
            s / p as f64
        } else {
            1.0
        }
    };
    let mut eps = diag_scale * 1e-10;
    for _ in 0..12 {
        let mut aj = a.clone();
        for i in 0..p {
            let v = aj.read(i, i) + eps;
            aj.write(i, i, v);
        }
        if let Ok(chol) = faer::linalg::solvers::Cholesky::try_new(aj.as_ref(), faer::Side::Lower) {
            return backsolve(&chol.compute_l());
        }
        eps *= 10.0;
    }

    // Should be unreachable for a PSD matrix; return identity (λ falls back to
    // the unpenalised df) rather than a mathematically inconsistent A^{-1}.
    let mut id = Mat::zeros(p, p);
    for i in 0..p {
        id.write(i, i, 1.0);
    }
    id
}

/// The spectrum of `Rmᵀ K Rm`, as singular values, sorted ascending.
///
/// `df_to_lambda` reads this spectrum through `2/(1+λd) - 1/(1+λd)²`, a
/// function of `λd`, so the `d` near `1/λ` set the answer and the large ones
/// contribute nothing. That makes the *small* values the accurate ones this
/// has to deliver, and they are the hard ones: `Rmᵀ K Rm` inherits the range of
/// `(K, X'WX + εK)`, which for a design of less than full column rank — every
/// tensor interaction on fewer rows than the product basis has columns — spans
/// nine orders of magnitude or more.
///
/// faer's self-adjoint eigendecomposition resolves those small values to about
/// three digits on such a matrix, enough to move the calibrated `λ` by 1% and
/// the fit it produces by 1e-3. Its singular values hold them to full
/// precision. Since the matrix is symmetric positive semidefinite the two
/// quantities coincide mathematically, and R's `df2lambda` reads the same
/// spectrum through `svd()`, so this is also the closer match to the reference
/// implementation. It costs roughly twice the decomposition time, once per
/// learner at setup rather than per boosting step.
fn penalty_spectrum_faer(m: &Mat<f64>) -> Vec<f64> {
    let mut values: Vec<f64> = m.singular_values();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values
}

/// R's default centering for a single-covariate `bbs(..., center = TRUE)`.
///
/// `mboost:::X_bbs` maps `center = TRUE` to the `"differenceMatrix"` method:
/// `X_centered = X · D'·(D·D')⁻¹` built from the RAW difference matrix D,
/// after which the penalty becomes the identity. (R's explicit
/// `"spectralDecomp"` option — `X · V · diag(1/sqrt(d))` over the non-null
/// eigenvectors of D'D — is what [`center_spectral`] implements; it spans the
/// same column space but rescales the basis, and because the centered penalty
/// is the identity the df→λ ridge acts in a different metric, so the two
/// centerings produce different fitted values at the same df. Found by the
/// study-02 parity harness: `bbs(x, center=TRUE, df=1)` diverged from R at
/// the first update of such a learner.)
///
/// Also handles the cyclic raw difference matrix, which is square; there
/// `D'·(D·D')⁻¹ = D⁻¹` and all p columns are kept, matching R's single
/// switch for both branches.
///
/// Returns `None` when D·D' is not factorizable (not reachable for a
/// well-formed difference matrix; the caller falls back to spectral).
pub fn center_difference_matrix(x: &Array2<f64>, d_raw: &Array2<f64>) -> Option<Array2<f64>> {
    let m = d_raw.nrows();
    let dd_t = d_raw.dot(&d_raw.t().to_owned());
    let factor = CachedCholesky::new(&dd_t).ok()?;
    // Solve (D·D')·Y = D → Y = (D·D')⁻¹·D (m × p); C = Y' is the (p, m)
    // transform and X·C the centered design.
    let y = factor.solve_mat(d_raw);
    let c = y.t().to_owned();
    debug_assert_eq!(c.dim(), (x.ncols(), m));
    Some(x.dot(&c))
}

/// The centering transform C = D'·(D·D')⁻¹ for R's `"differenceMatrix"`
/// method: `(p, m)` where m = D's rows. Returns C column-major with leading
/// dimension p, plus the number of columns, mirroring
/// [`compute_centering_transform`]'s layout contract.
pub fn compute_centering_transform_diff(
    d_raw: &Array2<f64>,
    p: usize,
) -> Option<(Vec<f64>, usize)> {
    debug_assert_eq!(d_raw.ncols(), p);
    let m = d_raw.nrows();
    let dd_t = d_raw.dot(&d_raw.t().to_owned());
    let factor = CachedCholesky::new(&dd_t).ok()?;
    // Solve (D·D') · Y = D  →  Y = (D·D')⁻¹·D (m × p); C = Y' (p × m).
    let y = factor.solve_mat(d_raw);
    let mut c = vec![0.0f64; p * m];
    for row in 0..p {
        for col in 0..m {
            c[col * p + row] = y[[col, row]];
        }
    }
    Some((c, m))
}

/// Center a B-spline basis using spectral decomposition, matching R's
/// `spectralDecomp` centering method for `bbs(..., center=TRUE)`.
///
/// Algorithm:
/// 1. Eigendecompose K = V·D·V' where D has p-d non-null and d null eigenvalues
/// 2. Drop the null-space columns (eigenvalues ≈ 0) — these are the SMALLEST
///    eigenvalues, at the start of an ascending-sorted array
/// 3. X_centered = X · V_{non-null} · diag(1/sqrt(d_{non-null}))
///
/// After centering, the penalty becomes the identity: K_centered = I.
///
/// The number of columns kept is `p` minus the rank of K's null space, which
/// is measured from the eigenvalues rather than assumed. For a difference
/// penalty of order d that null space has dimension d, but for an MRF penalty
/// it is the number of connected components in the neighbourhood graph — 3,
/// for instance, on the Munich district graph, whose islands are not reachable
/// from the mainland. The `differences` argument is therefore ignored; it is
/// retained only so callers can pass the value they know.
pub fn center_spectral(x: &Array2<f64>, k: &Array2<f64>, _differences: usize) -> Array2<f64> {
    let p = x.ncols();

    let faer_k = faer::Mat::from_fn(p, p, |i, j| k[[i, j]]);
    let eig = faer_k.selfadjoint_eigendecomposition(faer::Side::Lower);

    let s_col = eig.s().column_vector();
    let u_mat = eig.u();

    // Sort eigenvalues descending; keep only those > threshold.
    // Near-zero eigenvalues correspond to the null space and produce
    // all-zero columns that make the Gram matrix singular.
    let ev_threshold = 1e-10;
    let mut indexed: Vec<(usize, f64)> = (0..p).map(|i| (i, s_col.read(i))).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let nonnull: Vec<(usize, f64)> = indexed
        .into_iter()
        .filter(|(_, ev)| *ev > ev_threshold)
        .collect();
    let n_nonnull = nonnull.len();

    let mut v_nonnull = Array2::zeros((p, n_nonnull));
    let mut sqrt_inv_d = Array1::zeros(n_nonnull);

    for (j, &(idx, ev)) in nonnull.iter().enumerate() {
        sqrt_inv_d[j] = 1.0 / ev.sqrt();
        for i in 0..p {
            v_nonnull[[i, j]] = u_mat.read(i, idx);
        }
    }

    let x_v = x.dot(&v_nonnull);

    let mut centered = Array2::zeros((x.nrows(), n_nonnull));
    for i in 0..x.nrows() {
        for j in 0..n_nonnull {
            centered[[i, j]] = x_v[[i, j]] * sqrt_inv_d[j];
        }
    }

    centered
}

/// Compute the centering transformation matrix C = V_nonnull · diag(1/sqrt(d))
/// from the eigendecomposition of the penalty matrix K.
///
/// Returns `(C, r)` where C has shape `(p, r)` and r is the number of
/// non-null eigenvalues. The centered design for new data is `X_new · C`.
pub fn compute_centering_transform(k: &Array2<f64>, p: usize) -> (Vec<f64>, usize) {
    let faer_k = faer::Mat::from_fn(p, p, |i, j| k[[i, j]]);
    let eig = faer_k.selfadjoint_eigendecomposition(faer::Side::Lower);

    let s_col = eig.s().column_vector();
    let u_mat = eig.u();

    let ev_threshold = 1e-10;
    let mut indexed: Vec<(usize, f64)> = (0..p).map(|i| (i, s_col.read(i))).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let nonnull: Vec<(usize, f64)> = indexed
        .into_iter()
        .filter(|(_, ev)| *ev > ev_threshold)
        .collect();
    let r = nonnull.len();

    let mut c = vec![0.0f64; p * r];
    for (j, &(idx, ev)) in nonnull.iter().enumerate() {
        let scale = 1.0 / ev.sqrt();
        for i in 0..p {
            c[j * p + i] = u_mat.read(i, idx) * scale;
        }
    }

    (c, r)
}

/// Prepare a `bbs` (B-spline) base-learner, matching R's `bbs(x, df, ...)`.
///
/// This orchestrates the full pipeline:
/// 1. Build knot sequence from boundary knots and `n_knots` inner knots
/// 2. Evaluate B-spline basis via de Boor recursion
/// 3. Build difference penalty matrix K = D'D
/// 4. Center the basis via spectral decomposition (if `center=true`)
/// 5. Convert target df to smoothing parameter λ via Demmler-Reinsch
/// 6. Return a `PreparedLearner::BlockBols` ready for the engine loop
/// Prepare a monotone-constrained P-spline learner (R `bmono`).
///
/// Builds the same B-spline basis and difference penalty as
/// [`prepare_bbs`] (so `df → λ` is identical), then attaches a shape
/// constraint `Dβ ≥ 0` to be enforced at fit time via the iterative
/// violation-penalty method (see [`PreparedLearner::BmonoBbs`]).
///
/// When `center` is `true`, the design is spectrally centered via
/// [`center_spectral`] before the constraint matrix is built, removing
/// the penalty's null space so that `df=1` is meaningful.
#[allow(clippy::too_many_arguments)]
pub fn prepare_bmono(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    constraint: MonoConstraint,
    solver: BmonoSolver,
    lambda2: f64,
    niter: usize,
    name: &str,
    by: Option<ArrayView1<f64>>,
    constraint_matrix: Option<ArrayView2<f64>>,
    center: bool,
) -> PreparedLearner {
    prepare_bmono_with_lambda(
        x,
        weights,
        n_knots,
        degree,
        differences,
        df,
        None,
        constraint,
        solver,
        lambda2,
        niter,
        name,
        by,
        constraint_matrix,
        center,
    )
}

#[allow(clippy::too_many_arguments)]
/// Prepare a monotone P-spline learner, using an explicit lambda override
/// instead of deriving it from the requested degrees of freedom.
pub fn prepare_bmono_with_lambda(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    lambda_override: Option<f64>,
    constraint: MonoConstraint,
    solver: BmonoSolver,
    lambda2: f64,
    niter: usize,
    name: &str,
    by: Option<ArrayView1<f64>>,
    constraint_matrix: Option<ArrayView2<f64>>,
    center: bool,
) -> PreparedLearner {
    // R's `bsplines(Ts_constraint="decreasing")` applies a cumulative-sum
    // design transform (-X·D_cumsum) BEFORE centering. The centered design
    // inherits the constraint, and the fitting is plain ridge — no QP or
    // iterative enforcement.  When center=TRUE and the constraint is
    // Increasing/Decreasing, we replicate this: transform the design, center
    // it, and return a plain BlockBols.
    let use_ts_transform = center
        && constraint_matrix.is_none()
        && matches!(
            constraint,
            MonoConstraint::Increasing | MonoConstraint::Decreasing
        );

    let bk = (
        x.iter().cloned().fold(f64::INFINITY, f64::min),
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    let knots = bspline_knots(bk, n_knots, degree);
    let mut design = bspline_basis(x.as_slice().expect("x contiguous"), &knots, degree);
    let mut k = diff_penalty(design.ncols(), differences);

    if let Some(by_vec) = by {
        let n = design.nrows();
        for i in 0..n {
            for j in 0..design.ncols() {
                design[[i, j]] *= by_vec[i];
            }
        }
    }

    if use_ts_transform {
        // Embed monotonicity in the design via R's cumulative-sum transform.
        // This replicates R's bsplines(Ts_constraint="decreasing"): X = -X %*% D_cumsum.
        design = apply_ts_constraint(&design, constraint);
        if center {
            // Center the Ts_constraint-transformed design, matching R's X_bbs.
            let p_orig = design.ncols();
            let d_raw = raw_diff_matrix(p_orig, differences)
                .expect("differences >= p not supported for centering");
            design = center_difference_matrix(&design, &d_raw)
                .unwrap_or_else(|| center_spectral(&design, &k, differences));
            k = Array2::eye(design.ncols());
        } else {
            let d_cum = ts_constraint_cumsum_matrix(design.ncols());
            k = d_cum.t().dot(&k).dot(&d_cum);
        }
        // Delegate to the standard bbs ridge path (df→λ + BlockBols).
        return prepare_bbs_from_template(design, k, weights, df, lambda_override, name);
    }

    let lambda =
        lambda_override.unwrap_or_else(|| df_to_lambda(&design, &k, &weights.to_owned(), df));

    // Apply centering for non-ts-transform paths (convex, concave, etc.).
    let (design, k, centering_c) = if center {
        let p_orig = design.ncols();
        let d_raw = raw_diff_matrix(p_orig, differences)
            .expect("differences >= p not supported for centering");
        let centered = center_difference_matrix(&design, &d_raw)
            .unwrap_or_else(|| center_spectral(&design, &k, differences));
        let r = centered.ncols();
        let (cmat, _cr) = compute_centering_transform_diff(&d_raw, p_orig)
            .expect("D·D' must be invertible for centering");
        (centered, Array2::eye(r), Some((cmat, p_orig, r)))
    } else {
        (design, k, None)
    };

    let p = k.nrows();
    let mut lambda_k = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            lambda_k[[i, j]] = lambda * k[[i, j]];
        }
    }

    // Build constraint matrix D (use custom matrix if provided).
    // When centering is active, transform D into the centered space: D_effective = D_orig @ C
    // where C is the centering matrix. This ensures monotonicity on the
    // original-scale curve translates correctly to the centered coefficients.
    let d_mat = match (constraint, constraint_matrix, &centering_c) {
        (MonoConstraint::Constrained, Some(c), _) => c.to_owned(),
        (_, _, Some((c_mat, c_rows, c_cols))) => {
            // Build D for the original p-column basis, then transform to centered space.
            let p_orig = *c_rows;
            let d_orig = build_mono_constraint(constraint, p_orig);
            let c_arr = Array2::from_shape_vec((*c_rows, *c_cols), c_mat.clone())
                .expect("centering matrix shape mismatch");
            d_orig.dot(&c_arr)
        }
        _ => build_mono_constraint(constraint, p),
    };

    let n = design.nrows();
    let design_faer = ndarray2_to_faer(&design);
    let wdt_faer = Mat::from_fn(p, n, |j, i| weights[i] * design[[i, j]]);

    let mut gram_faer = ndarray2_to_faer(&lambda_k);
    faer::linalg::matmul::matmul(
        &mut gram_faer,
        &wdt_faer,
        &design_faer,
        Some(1.0),
        1.0,
        faer::Parallelism::None,
    );

    let gram = Array2::from_shape_fn((p, p), |(j, k)| gram_faer.read(j, k));
    let solver_a = GramSolver::new(&gram);

    let rhs_buf_len = wdt_faer.nrows();
    let pred_buf_len = design_faer.nrows();
    let m = d_mat.nrows();

    PreparedLearner::BmonoBbs {
        design,
        wdt_faer,
        design_faer,
        penalty: lambda_k,
        penalised_gram: gram,
        solver_a,
        d_mat,
        centering: Array1::zeros(p),
        name: name.to_string(),
        solver,
        lambda2,
        niter,
        rhs_buf: vec![0.0; rhs_buf_len],
        beta_buf: vec![0.0; rhs_buf_len],
        pred_buf: vec![0.0; pred_buf_len],
        // Constraint-solver scratch, pre-sized so the hot candidate solve
        // allocates nothing per call (P3.4 item 2).
        solver_violations: Vec::with_capacity(m),
        solver_violations_new: Vec::with_capacity(m),
        solver_aug: vec![0.0; p * p],
        solver_x: vec![0.0; p],
        qp_g: vec![0.0; p],
        qp_a_inv_g: vec![0.0; p],
        qp_active: Vec::with_capacity(m),
        qp_a_inv_dt: vec![0.0; p * m],
        qp_h: vec![0.0; m * m],
        qp_lambda: vec![0.0; m],
        qp_delta: vec![0.0; p],
        chol_scratch: vec![0.0; cho_solve_inplace_scratch_len(p.max(m))],
    }
}

/// Build a prepared B-spline learner from raw covariate values and weights.
pub fn prepare_bbs(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    center: bool,
    name: &str,
) -> PreparedLearner {
    prepare_bbs_with_lambda(
        x,
        weights,
        n_knots,
        degree,
        differences,
        df,
        None,
        center,
        name,
        None,
        false,
        None,
        0,
    )
}

/// Build the fold-invariant part of a `bbs` learner: the B-spline design
/// and its difference penalty, after `by`-multiplication and (if
/// requested) spectral centering. Depends only on `x`/`by`/the spline
/// spec, never on observation weights — so it can be computed once and
/// reused across every cross-validation fold instead of being rebuilt
/// from the raw `x` values (knots, basis, centering) on every fold, which
/// is all `prepare_bbs_with_lambda` used to do internally. Only the
/// weight-dependent `df -> lambda` solve and the resulting `GramSolver`
/// factorization (in [`prepare_bbs_from_template`]) need to run per fold.
#[allow(clippy::too_many_arguments)]
pub fn build_bbs_template(
    x: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    center: bool,
    by: Option<ArrayView1<f64>>,
    cyclic: bool,
    boundary_knots_override: Option<(f64, f64)>,
    deriv: usize,
) -> (Array2<f64>, Array2<f64>) {
    let bk = boundary_knots_override.unwrap_or_else(|| {
        (
            x.iter().cloned().fold(f64::INFINITY, f64::min),
            x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    });

    let (design, penalty_mat) = if cyclic {
        let knots = cyclic_bspline_knots(bk, n_knots, degree);
        let basis = cyclic_bspline_basis_from_knots(
            x.as_slice().expect("x contiguous"),
            &knots,
            degree,
            bk,
        );
        let k = cyclic_diff_penalty(basis.ncols(), differences);
        (basis, k)
    } else {
        let knots = bspline_knots(bk, n_knots, degree);
        let basis = if deriv > 0 {
            bspline_basis_with_deriv(x.as_slice().expect("x contiguous"), &knots, degree, deriv)
        } else {
            bspline_basis(x.as_slice().expect("x contiguous"), &knots, degree)
        };
        let k = diff_penalty(basis.ncols(), differences);
        (basis, k)
    };

    let mut design = design;

    if let Some(by_vec) = by {
        let n = design.nrows();
        for i in 0..n {
            for j in 0..design.ncols() {
                design[[i, j]] *= by_vec[i];
            }
        }
    }

    if center {
        // R's default single-covariate centering ("differenceMatrix" in
        // mboost:::X_bbs): X · D'·(D·D')⁻¹ from the RAW difference matrix,
        // then an identity penalty. Spectral centering remains the fallback
        // for degenerate inputs — and stays the method for two-covariate
        // tensors (`bspatial`), where R itself uses the eigen form.
        let p_uncentered = design.ncols();
        let d_raw = if cyclic {
            raw_cyclic_diff_matrix(p_uncentered, differences)
        } else {
            raw_diff_matrix(p_uncentered, differences)
        };
        let centered = d_raw
            .as_ref()
            .and_then(|d| center_difference_matrix(&design, d))
            .unwrap_or_else(|| center_spectral(&design, &penalty_mat, differences));
        let p = centered.ncols();
        return (centered, Array2::eye(p));
    }

    (design, penalty_mat)
}

/// Finish preparing a `bbs` learner from a precomputed (fold-invariant)
/// design/penalty template (see [`build_bbs_template`]): recompute
/// `df -> lambda` under the given weights (which may be a fold's in-bag
/// weights), then build the penalized Gram and its factorization. This is
/// the only per-fold work a `bbs` learner needs once its template is
/// cached.
pub fn prepare_bbs_from_template(
    design: Array2<f64>,
    penalty_mat: Array2<f64>,
    weights: ArrayView1<f64>,
    df: f64,
    lambda_override: Option<f64>,
    name: &str,
) -> PreparedLearner {
    let lambda = lambda_override
        .unwrap_or_else(|| df_to_lambda(&design, &penalty_mat, &weights.to_owned(), df));
    let penalized_k = penalty_mat.mapv(|v| lambda * v);
    prepare_block_bols(design, weights, penalized_k, false, name)
        .with_effective_lambda(Some(lambda))
}

/// Prepare a `bbs` learner, optionally with an explicit smoothing parameter.
///
/// If `lambda_override` is `Some(λ)`, the value is used directly and `df` is ignored
/// (matching R's convention: `bbs(..., lambda=λ)` overrides `df`).
///
/// Rebuilds the template (design/penalty) from scratch every call — fine
/// for a one-time fit, wasteful in a per-fold CV loop. Callers that
/// re-prepare the same `x`/spline-spec under different fold weights should
/// call [`build_bbs_template`] once and [`prepare_bbs_from_template`] per
/// fold instead.
#[allow(clippy::too_many_arguments)]
pub fn prepare_bbs_with_lambda(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    lambda_override: Option<f64>,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
    cyclic: bool,
    boundary_knots_override: Option<(f64, f64)>,
    deriv: usize,
) -> PreparedLearner {
    let (design, penalty_mat) = build_bbs_template(
        x,
        n_knots,
        degree,
        differences,
        center,
        by,
        cyclic,
        boundary_knots_override,
        deriv,
    );
    prepare_bbs_from_template(design, penalty_mat, weights, df, lambda_override, name)
}

// ─── bmrf: Markov random field base-learner ──────────────────────────

/// Reject a `bmrf` penalty that is not in graph-Laplacian form.
///
/// The failure this catches is silent: an adjacency matrix (zero diagonal,
/// `+1` for adjacent regions) has the right shape and the right sparsity
/// pattern, so it fits without complaint, but it is indefinite and penalises
/// the wrong directions. R refuses it at the door with `sum(bnd) == 0`; the
/// same check here turns a wrong answer into an error message.
///
/// The tolerance is relative to the matrix scale, since a Laplacian built by
/// summing floating-point weights need not have exactly zero row sums.
fn check_mrf_penalty(k: &Array2<f64>) {
    let l = k.nrows();
    if k.ncols() != l {
        panic!(
            "bmrf penalty K must be square; got {} x {}",
            k.nrows(),
            k.ncols()
        );
    }
    let scale = k.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if scale == 0.0 {
        return; // A zero penalty is degenerate but not ambiguous.
    }
    let tol = 1e-8 * scale * (l as f64);

    for i in 0..l {
        let row_sum: f64 = (0..l).map(|j| k[[i, j]]).sum();
        if row_sum.abs() > tol {
            panic!(
                "bmrf penalty K must be a graph Laplacian: row {i} sums to \
                 {row_sum}, not 0. Use K[i,i] = number of neighbours of i and \
                 K[i,j] = -1 for adjacent i, j (the format of R's \
                 BayesX::bnd2gra). An adjacency matrix — zero diagonal with +1 \
                 for neighbours — has the same sparsity pattern but is \
                 indefinite, and fits silently to the wrong answer."
            );
        }
    }
}

/// Prepare a `bmrf` (Markov random field) base-learner, matching R's
/// `bmrf(..., bnd = K, df, lambda, center)`.
///
/// The design matrix X is an indicator matrix: each row has a 1 in the
/// column corresponding to the factor level (region) of that observation.
/// The penalty K is the graph Laplacian of the neighbourhood structure,
/// supplied by the caller: `K[i,i]` is the number of neighbours of region i,
/// `K[i,j] = -1` when regions i and j are adjacent, and 0 otherwise. Every
/// row therefore sums to zero, and so does the whole matrix. This is the
/// `gra` format that R's `BayesX::bnd2gra` produces and that `mboost::bmrf`
/// requires — it rejects any matrix with `sum(bnd) != 0`.
///
/// The Laplacian is what makes `lambda * K` a usable penalty:
/// `β' K β = Σ_{i~j} (β_i - β_j)^2` over adjacent pairs, which is
/// non-negative and grows as neighbouring effects diverge. The adjacency
/// matrix (zero diagonal, `+1` off-diagonal) is *not* interchangeable with
/// it: adjacency is indefinite, so `β' K β` can be negative and the
/// "penalty" then rewards the directions it should shrink.
///
/// When `center = TRUE`, spectral decomposition of K drops the null space
/// (1 eigenvector — the constant vector, since K has zero row sums), and
/// the design is transformed to X_centered = X · V_{non-null} · diag(1/√d),
/// with penalty becoming the identity — matching R's `X_bmrf(center=TRUE)`.
/// That step assumes the null space really is the constants, which again
/// holds for the Laplacian and not for adjacency.
///
/// `levels` maps observation indices to 0-based level indices (0..L).
/// `K` is the (L, L) symmetric Laplacian described above.
#[allow(clippy::too_many_arguments)]
pub fn prepare_bmrf(
    levels: ArrayView1<usize>,
    weights: ArrayView1<f64>,
    k: &Array2<f64>,
    df: f64,
    lambda_override: Option<f64>,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> PreparedLearner {
    let n = levels.len();
    let l = k.nrows();

    check_mrf_penalty(k);

    let mut design = Array2::zeros((n, l));
    for i in 0..n {
        design[[i, levels[i]]] = 1.0;
    }

    if let Some(by_vec) = by {
        for i in 0..n {
            let b = by_vec[i];
            for j in 0..l {
                design[[i, j]] *= b;
            }
        }
    }

    let (design_final, penalty_mat) = if center {
        let centered = center_spectral(&design, k, 1);
        let p_centered = centered.ncols();
        (centered, Array2::eye(p_centered))
    } else {
        (design, k.clone())
    };

    let lambda = if let Some(lam) = lambda_override {
        lam
    } else {
        df_to_lambda(&design_final, &penalty_mat, &weights.to_owned(), df)
    };

    let penalized_k = penalty_mat.mapv(|v| lambda * v);

    prepare_block_bols(design_final, weights, penalized_k, false, name)
        .with_effective_lambda(Some(lambda))
}

// ─── bspatial: tensor-product spline base-learner ────────────────────

/// Compute the Kronecker product of two matrices: A ⊗ B.
///
/// If A is (m, n) and B is (p, q), the result is (m·p, n·q).
pub fn kron(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (p, q) = (b.nrows(), b.ncols());
    let mut result = Array2::zeros((m * p, n * q));
    for i in 0..m {
        for j in 0..n {
            let aij = a[[i, j]];
            if aij == 0.0 {
                continue;
            }
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = aij * b[[k, l]];
                }
            }
        }
    }
    result
}

fn is_identity_matrix(x: &Array2<f64>) -> bool {
    if x.nrows() != x.ncols() {
        return false;
    }
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (x[[i, j]] - expected).abs() > 1e-12 {
                return false;
            }
        }
    }
    true
}

/// Prepare a `bspatial` (tensor-product B-spline) base-learner, matching
/// R's `bspatial(x1, x2, df=6, ...)`.
///
/// The design is the Kronecker product X = B1 ⊗ B2 of two B-spline bases
/// evaluated at `x1` and `x2` respectively. The penalty is the additive
/// Kronecker penalty K = K1 ⊗ I2 + I1 ⊗ K2, which penalises roughness
/// in each marginal direction independently (Wood 2006, §4.1.8).
///
/// With `center = TRUE`, spectral decomposition of K drops 1 null
/// eigenvalue (the constant function), matching R's `bbs(..., center=TRUE)`
/// applied to the bivariate case.
#[allow(clippy::too_many_arguments)]
pub fn prepare_bspatial(
    x1: ArrayView1<f64>,
    x2: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    lambda_override: Option<f64>,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> PreparedLearner {
    let n = x1.len();

    let bk1 = (
        x1.iter().cloned().fold(f64::INFINITY, f64::min),
        x1.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let bk2 = (
        x2.iter().cloned().fold(f64::INFINITY, f64::min),
        x2.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let knots1 = bspline_knots(bk1, n_knots, degree);
    let knots2 = bspline_knots(bk2, n_knots, degree);
    let b1 = bspline_basis(x1.as_slice().expect("x1 contiguous"), &knots1, degree);
    let b2 = bspline_basis(x2.as_slice().expect("x2 contiguous"), &knots2, degree);

    let p1 = b1.ncols();
    let p2 = b2.ncols();

    let p = p1 * p2;
    let mut design = Array2::zeros((n, p));
    for i in 0..n {
        for j1 in 0..p1 {
            let v1 = b1[[i, j1]];
            if v1 == 0.0 {
                continue;
            }
            for j2 in 0..p2 {
                design[[i, j1 * p2 + j2]] = v1 * b2[[i, j2]];
            }
        }
    }

    if let Some(by_vec) = by {
        for i in 0..n {
            let b = by_vec[i];
            for j in 0..p {
                design[[i, j]] *= b;
            }
        }
    }

    // Build marginal penalty matrices
    let k1 = diff_penalty(p1, differences);
    let k2 = diff_penalty(p2, differences);

    // Additive Kronecker penalty: K = K1⊗I2 + I1⊗K2
    let i1 = Array2::eye(p1);
    let i2 = Array2::eye(p2);
    let penalty = kron(&k1, &i2) + kron(&i1, &k2);

    let (design_final, penalty_mat) = if center {
        // Drop 1 null eigenvalue (the constant surface), matching R
        let centered = center_spectral(&design, &penalty, 1);
        let p_c = centered.ncols();
        (centered, Array2::eye(p_c))
    } else {
        (design, penalty)
    };

    // Share one `X'WX` between the df→λ calibration and the solver Gram
    // (`X'WX + λK`); the second O(n·p²) pass the previous
    // `df_to_lambda` + `prepare_block_bols` pair paid was most of this
    // learner's setup time.
    prepare_block_bols_shared_gram(
        design_final,
        weights,
        penalty_mat,
        df,
        lambda_override,
        name,
    )
}

// ─── bspatial marginal spectral machinery (#284) ──────────────────────

/// Eigenvectors and eigenvalues of the two marginal difference penalties,
/// from which the tensor penalty's spectral data follows without ever
/// forming the `(p1·p2)`-sized eigenproblem: the eigenvalues of
/// `K = K1 ⊗ I + I ⊗ K2` are `kappa[(a, b)] = d1[a] + d2[b]` indexed
/// `a·p2 + b`, and its eigenvectors are the tensor products
/// `U1[:, a] ⊗ U2[:, b]` (#284).
fn bspatial_marginal_eigs(
    k1: &Array2<f64>,
    k2: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Vec<f64>) {
    let eigs = |k: &Array2<f64>| -> (Array2<f64>, Vec<f64>) {
        let p = k.nrows();
        let fm = Mat::from_fn(p, p, |i, j| k[[i, j]]);
        let eig = fm.selfadjoint_eigendecomposition(faer::Side::Lower);
        let vals: Vec<f64> = (0..p).map(|i| eig.s().column_vector().read(i)).collect();
        let mut u = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                u[[i, j]] = eig.u().read(i, j);
            }
        }
        (u, vals)
    };
    let (u1, d1) = eigs(k1);
    let (u2, d2) = eigs(k2);
    let p1 = u1.nrows();
    let p2 = u2.nrows();
    let mut kappa = Vec::with_capacity(p1 * p2);
    for a in 0..p1 {
        for b in 0..p2 {
            kappa.push(d1[a] + d2[b]);
        }
    }
    (u1, u2, kappa)
}

/// `G~ = (U1 ⊗ U2)' · G · (U1 ⊗ U2)` for the raw tensor Gram `G`, by
/// contracting each of the four tensor modes with its marginal eigenvector
/// matrix. Costs `O(p^2·(p1 + p2))` instead of the dense `O(p^3)`
/// congruence through the explicit Kronecker basis (#284).
fn bspatial_tensor_transform(g: &Array2<f64>, u1: &Array2<f64>, u2: &Array2<f64>) -> Array2<f64> {
    use ndarray::s;

    let p1 = u1.nrows();
    let p2 = u2.nrows();
    let p = p1 * p2;
    let u1t = u1.t().to_owned();
    let u2t = u2.t().to_owned();

    // Row modes: contract mode a with U1', then mode b with U2'.
    let gv = g
        .view()
        .into_shape_with_order((p1, p2 * p))
        .expect("gram row reshape");
    let a1 = u1t.dot(&gv); // (p1, p2·p)
    let mut a2 = ndarray::Array3::<f64>::zeros((p1, p2, p));
    for aa in 0..p1 {
        let sl = a1
            .slice(s![aa, ..])
            .into_shape_with_order((p2, p))
            .expect("row slice");
        a2.slice_mut(s![aa, .., ..]).assign(&u2t.dot(&sl));
    }
    // a2[(a, b), j] = Σ_{i1,i2} U1[a,i1]·U2[b,i2]·G[(i1,i2), j]
    let a2m = a2
        .view()
        .into_shape_with_order((p, p))
        .expect("row flatten")
        .to_owned();

    // Column modes: view each column index j = (c, d); contract c with U1,
    // then d with U2. The two gathers repack between pair-major and
    // single-major layouts for the mode splits.
    let mut a2p = Array2::<f64>::zeros((p * p2, p1));
    for r in 0..p {
        for c in 0..p1 {
            for d in 0..p2 {
                a2p[[r * p2 + d, c]] = a2m[[r, c * p2 + d]];
            }
        }
    }
    let a3 = a2p.dot(u1); // rows (r, d), columns c'
    let mut a3p = Array2::<f64>::zeros((p * p1, p2));
    for r in 0..p {
        for cp in 0..p1 {
            for d in 0..p2 {
                a3p[[r * p1 + cp, d]] = a3[[r * p2 + d, cp]];
            }
        }
    }
    a3p.dot(u2)
        .into_shape_with_order((p, p))
        .expect("final reshape")
}

/// Marginal spectral centering for a centered `bspatial` learner.
///
/// The centering map `C` is `(U1 ⊗ U2)` restricted to the eigen-coordinates
/// with `kappa > 1e-10` and rescaled by `1/sqrt(kappa)`, so the penalty in
/// the kept coordinates is exactly the identity. Storing the two small
/// marginal factors plus the kept index list replaces the dense
/// `(p, p_out)` map of the full spectral decomposition (#284).
#[derive(Clone)]
pub struct BspatialCentering {
    /// Eigenvectors of the left marginal penalty `(p1, p1)`.
    u1: Array2<f64>,
    /// Eigenvectors of the right marginal penalty `(p2, p2)`.
    u2: Array2<f64>,
    /// Raw eigen-index of each kept coordinate, `kept[c] = a·p2 + b`.
    kept: Vec<usize>,
    /// `1/sqrt(kappa)` of each kept coordinate.
    scale: Vec<f64>,
}

impl BspatialCentering {
    /// Threshold on the penalty eigenvalue below which a coordinate counts as
    /// a null-space direction and is dropped by centering. Matches
    /// [`compute_centering_transform`]'s `ev_threshold` so the kept set is
    /// identical.
    const NULL_THRESHOLD: f64 = 1e-10;

    fn from_kappa(u1: Array2<f64>, u2: Array2<f64>, kappa: &[f64]) -> Self {
        let kept: Vec<usize> = (0..kappa.len())
            .filter(|&i| kappa[i] > Self::NULL_THRESHOLD)
            .collect();
        let scale: Vec<f64> = kept.iter().map(|&i| 1.0 / kappa[i].sqrt()).collect();
        Self {
            u1,
            u2,
            kept,
            scale,
        }
    }

    /// Number of kept (centered) coordinates.
    fn p_out(&self) -> usize {
        self.kept.len()
    }

    /// The centered Gram `C'GC` from the transformed Gram
    /// `G~ = (U1 ⊗ U2)'G(U1 ⊗ U2)`: keep the `kept` rows/cols and rescale by
    /// `1/sqrt(kappa)` on both sides.
    fn centered_gram(&self, g_tilde: &Array2<f64>) -> Array2<f64> {
        let p_c = self.kept.len();
        let mut g_c = Array2::<f64>::zeros((p_c, p_c));
        for (ci, &ri) in self.kept.iter().enumerate() {
            for (cj, &rj) in self.kept.iter().enumerate() {
                g_c[[ci, cj]] = g_tilde[[ri, rj]] * self.scale[ci] * self.scale[cj];
            }
        }
        g_c
    }

    /// `C'·rhs`: project the raw right-hand side onto the kept
    /// eigen-coordinates and rescale. `rhs[(a, b)]` contracts against
    /// `U1[:, a]·U2[:, b]`.
    fn rhs_transform(&self, raw_rhs: &[f64], p1: usize, p2: usize, out: &mut [f64]) {
        let rv = ArrayView2::from_shape((p1, p2), raw_rhs).expect("rhs reshape");
        let rt = self.u1.t().dot(&rv).dot(&self.u2);
        for (c, &ri) in self.kept.iter().enumerate() {
            out[c] = rt[[ri / p2, ri % p2]] * self.scale[c];
        }
    }

    /// `C·beta`: back-transform centered coefficients to raw coordinates,
    /// `c_raw = (U1 ⊗ U2)·v` with `v[kept[c]] = beta[c]·scale[c]`. The final
    /// contraction uses `U2'`, not `U2`: the kept index of `v` is an
    /// eigenvector index while the output index is a raw one.
    fn back_transform(&self, beta: &[f64], p1: usize, p2: usize, out: &mut [f64]) {
        let mut v = vec![0.0f64; p1 * p2];
        for (c, &ri) in self.kept.iter().enumerate() {
            v[ri] = beta[c] * self.scale[c];
        }
        let vm = ArrayView2::from_shape((p1, p2), &v).expect("v reshape");
        let cb = self.u1.dot(&vm).dot(&self.u2.t());
        for a in 0..p1 {
            for b in 0..p2 {
                out[a * p2 + b] = cb[[a, b]];
            }
        }
    }

    /// The dense centering map `C` as a `(p, p_out)` matrix, for the
    /// hat-matrix and design-materialisation paths only (never the boosting
    /// hot path).
    fn dense_matrix(&self, p1: usize, p2: usize) -> Array2<f64> {
        let p = p1 * p2;
        let mut c = Array2::<f64>::zeros((p, self.kept.len()));
        for (col, &ri) in self.kept.iter().enumerate() {
            let (a, b) = (ri / p2, ri % p2);
            for r1 in 0..p1 {
                for r2 in 0..p2 {
                    c[[r1 * p2 + r2, col]] = self.u1[[r1, a]] * self.u2[[r2, b]] * self.scale[col];
                }
            }
        }
        c
    }
}

/// df→λ for a centered bspatial Gram, whose final-space penalty is the
/// identity. The pencil spectrum is then `d_k = 1/(gamma_k + MBOOST_EPS)`
/// with `gamma` the eigenvalues of the centered Gram: one symmetric
/// eigensolve replaces the Cholesky / triangular inverse / two products /
/// SVD chain of [`df_to_lambda_from_gram`] (#284). The values are exactly
/// those of the bounded pencil (`sigma = 1/(gamma + eps + 1)`,
/// `d = sigma/(1 - sigma)`), so the resolution re-check the SVD path needs
/// cannot fire. Falls back to [`df_to_lambda_from_gram`] when round-off
/// puts an eigenvalue at or below `-MBOOST_EPS`, where the shortcut would
/// emit a negative spectrum value and break the monotone df curve.
fn df_to_lambda_centered_eig(g_c: &Array2<f64>, df_target: f64) -> f64 {
    let p = g_c.nrows();
    let g_faer = Mat::from_fn(p, p, |i, j| g_c[[i, j]]);
    let eye_faer = Mat::from_fn(p, p, |i, j| if i == j { 1.0 } else { 0.0 });
    let lambda_max = solvable_lambda_max(&g_faer, &eye_faer);

    let eig = g_faer.selfadjoint_eigendecomposition(faer::Side::Lower);
    let mut gamma: Vec<f64> = (0..p).map(|i| eig.s().column_vector().read(i)).collect();
    gamma.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if gamma.iter().any(|&g| g <= -MBOOST_EPS) {
        return df_to_lambda_from_gram(g_c, &Array2::eye(p), df_target);
    }
    let mut d: Vec<f64> = gamma.iter().map(|&g| 1.0 / (g + MBOOST_EPS)).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    lambda_from_spectrum(&d, df_target, lambda_max)
}

/// df→λ for the raw (uncentered) bspatial pencil `(K, X'WX + eps·K)`,
/// computed in the marginal eigenbasis of `K`. The `r`-dimensional null space
/// of `K` contributes exactly `r` zeros; the remaining values are `1/m`
/// where `m` runs over the eigenvalues of
/// `D^{-1/2}·(S + eps·D)·D^{-1/2}` with `D = diag(kappa)` on the non-null
/// coordinates and `S` the Gram's Schur complement onto them. One
/// `(p - r)`-sized eigensolve replaces the Cholesky / triangular inverse /
/// two products / SVD chain (#284). Falls back to [`df_to_lambda_from_gram`]
/// when the null-space block does not factor or an eigenvalue comes out
/// non-positive.
fn df_to_lambda_pencil(
    g_raw: &Array2<f64>,
    k1: &Array2<f64>,
    k2: &Array2<f64>,
    df_target: f64,
) -> f64 {
    let p1 = k1.nrows();
    let p2 = k2.nrows();
    let p = p1 * p2;
    let k_raw = kron(k1, &Array2::eye(p2)) + kron(&Array2::eye(p1), k2);

    let fallback = || df_to_lambda_from_gram(g_raw, &k_raw, df_target);

    let (u1, u2, kappa) = bspatial_marginal_eigs(k1, k2);
    let zidx: Vec<usize> = (0..p)
        .filter(|&i| kappa[i] <= BspatialCentering::NULL_THRESHOLD)
        .collect();
    let pidx: Vec<usize> = (0..p)
        .filter(|&i| kappa[i] > BspatialCentering::NULL_THRESHOLD)
        .collect();
    let (r, q) = (zidx.len(), pidx.len());
    if r == 0 || q == 0 {
        return fallback();
    }

    let g_tilde = bspatial_tensor_transform(g_raw, &u1, &u2);

    // Null-space block of the Gram, G_ZZ (r x r).
    let zz = {
        let mut m = Array2::<f64>::zeros((r, r));
        for (zi, &i) in zidx.iter().enumerate() {
            for (zj, &j) in zidx.iter().enumerate() {
                m[[zi, zj]] = g_tilde[[i, j]];
            }
        }
        m
    };
    let zz_chol = match CachedCholesky::new(&zz) {
        Ok(factor) => factor,
        Err(_) => return fallback(),
    };

    // S = G_PP + eps·D - G_PZ·G_ZZ^{-1}·G_ZP (q x q).
    let pz = {
        let mut m = Array2::<f64>::zeros((q, r));
        for (pi, &i) in pidx.iter().enumerate() {
            for (zj, &j) in zidx.iter().enumerate() {
                m[[pi, zj]] = g_tilde[[i, j]];
            }
        }
        m
    };
    let sol = zz_chol.solve_mat(&pz.t().to_owned()); // ZZ^{-1}·G_PZ' (r, q)
    let s_mat = {
        let mut s = Array2::<f64>::zeros((q, q));
        for (pi, &i) in pidx.iter().enumerate() {
            for (pj, &j) in pidx.iter().enumerate() {
                let mut value = g_tilde[[i, j]];
                if pi == pj {
                    value += MBOOST_EPS * kappa[i];
                }
                let mut cross = 0.0;
                for zi in 0..r {
                    cross += pz[[pi, zi]] * sol[[zi, pj]];
                }
                s[[pi, pj]] = value - cross;
            }
        }
        s
    };

    // M = D^{-1/2}·S·D^{-1/2}; its eigenvalues are the pencil values plus eps.
    let m_mat = {
        let mut m = Array2::<f64>::zeros((q, q));
        for (pi, &i) in pidx.iter().enumerate() {
            let si = 1.0 / kappa[i].sqrt();
            for (pj, &j) in pidx.iter().enumerate() {
                m[[pi, pj]] = s_mat[[pi, pj]] * si * (1.0 / kappa[j].sqrt());
            }
        }
        m
    };
    let fm = Mat::from_fn(q, q, |i, j| m_mat[[i, j]]);
    let eig = fm.selfadjoint_eigendecomposition(faer::Side::Lower);
    let m_vals: Vec<f64> = (0..q).map(|i| eig.s().column_vector().read(i)).collect();
    if m_vals.iter().any(|&m| m <= 0.0) {
        return fallback();
    }
    let mut d: Vec<f64> = m_vals.iter().map(|&m| 1.0 / m).collect();
    d.extend(std::iter::repeat_n(0.0, r));
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let g_faer = Mat::from_fn(p, p, |i, j| g_raw[[i, j]]);
    let k_faer = Mat::from_fn(p, p, |i, j| k_raw[[i, j]]);
    let lambda_max = solvable_lambda_max(&g_faer, &k_faer);
    lambda_from_spectrum(&d, df_target, lambda_max)
}

/// Compact nonzero `(index, value)` support of every row of a marginal basis.
fn marginal_supports(basis: &Array2<f64>) -> Vec<Vec<(u32, f64)>> {
    let (n, p) = (basis.nrows(), basis.ncols());
    (0..n)
        .map(|i| {
            (0..p)
                .filter_map(|j| {
                    let value = basis[[i, j]];
                    if value != 0.0 {
                        Some((j as u32, value))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect()
}

/// Raw tensor Gram `X'WX = Σ_i w_i·by_i²·(b1_i b1_i')⊗(b2_i b2_i')`, built
/// from the marginal supports of the row-wise (Khatri–Rao) tensor design.
///
/// This is the true Gram of the row-wise Kronecker design. It is *not*
/// `kron(B1'WB1, B2'WB2)`: that identity holds only for the full-grid
/// Kronecker, whose rows are all `(i, j)` pairs, whereas `bspatial` evaluates
/// both marginals at the same observation index.
fn streamed_raw_gram(
    supports1: &[Vec<(u32, f64)>],
    supports2: &[Vec<(u32, f64)>],
    by: Option<&Array1<f64>>,
    weights: &[f64],
    p2: usize,
    p: usize,
) -> Array2<f64> {
    let mut gram = Array2::<f64>::zeros((p, p));
    for i in 0..supports1.len() {
        let by_i = by.map_or(1.0, |values| values[i]);
        let factor = weights[i] * by_i * by_i;
        if factor == 0.0 {
            continue;
        }
        let row = &supports1[i];
        let col = &supports2[i];
        for &(j1, v1) in row {
            let base = j1 as usize * p2;
            for &(j2, v2) in col {
                let index = base + j2 as usize;
                let value = factor * v1 * v2;
                for &(l1, u1) in row {
                    let lbase = l1 as usize * p2;
                    for &(l2, u2) in col {
                        gram[[index, lbase + l2 as usize]] += value * u1 * u2;
                    }
                }
            }
        }
    }
    gram
}

/// Raw tensor right-hand side `X'W(u∘by)` from the marginal supports.
fn streamed_raw_rhs(
    supports1: &[Vec<(u32, f64)>],
    supports2: &[Vec<(u32, f64)>],
    by: Option<&Array1<f64>>,
    weights: &[f64],
    u: &[f64],
    p2: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    for i in 0..supports1.len() {
        let by_i = by.map_or(1.0, |values| values[i]);
        let factor = weights[i] * u[i] * by_i;
        if factor == 0.0 {
            continue;
        }
        for &(j1, v1) in &supports1[i] {
            let base = j1 as usize * p2;
            for &(j2, v2) in &supports2[i] {
                out[base + j2 as usize] += factor * v1 * v2;
            }
        }
    }
}

/// Streamed `out = design · g` for the row-wise tensor design, where `g` is
/// `beta` (uncentered) or `C·beta` (centered).
fn streamed_predict(
    supports1: &[Vec<(u32, f64)>],
    supports2: &[Vec<(u32, f64)>],
    by: Option<&Array1<f64>>,
    g: &[f64],
    p2: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), supports1.len());
    for i in 0..supports1.len() {
        let by_i = by.map_or(1.0, |values| values[i]);
        let mut value = 0.0;
        for &(j1, v1) in &supports1[i] {
            let base = j1 as usize * p2;
            for &(j2, v2) in &supports2[i] {
                value += v1 * v2 * g[base + j2 as usize];
            }
        }
        out[i] = by_i * value;
    }
}

/// Materialise the (possibly centered) dense tensor design from the marginals.
fn streamed_dense_design(
    supports1: &[Vec<(u32, f64)>],
    supports2: &[Vec<(u32, f64)>],
    by: Option<&Array1<f64>>,
    centering: Option<&Array2<f64>>,
    p1: usize,
    p2: usize,
) -> Array2<f64> {
    let n = supports1.len();
    let mut design = Array2::zeros((n, p1 * p2));
    for i in 0..n {
        let by_i = by.map_or(1.0, |values| values[i]);
        for &(j1, v1) in &supports1[i] {
            let base = j1 as usize * p2;
            for &(j2, v2) in &supports2[i] {
                design[[i, base + j2 as usize]] = by_i * v1 * v2;
            }
        }
    }
    match centering {
        Some(c) => design.dot(c),
        None => design,
    }
}

/// Prepare a `bspatial` (tensor-product B-spline) base-learner in the streamed
/// marginal-basis form.
///
/// Mathematically identical to [`prepare_bspatial`], but the setup Gram is
/// accumulated from the marginal supports as the row-wise Kronecker
/// `X'WX = Σ_i w_i·by_i²·(b1_i b1_i')⊗(b2_i b2_i')`, costing
/// `O(n·(degree+1)⁴)` instead of the dense `O(n·(p1·p2)²)`, and no `n×p`
/// design is retained. The marginal Gram identity `kron(B1'WB1, B2'WB2)` does
/// **not** hold here (it is only valid for the full-grid Kronecker), so the
/// pair-wise accumulation above is used.
///
/// Parity with [`prepare_bspatial`] is tolerance-level: the Gram accumulation
/// order differs from the dense `general_mat_mul` path, so the two agree to
/// floating-point round-off rather than bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub fn prepare_bspatial_streamed(
    x1: ArrayView1<f64>,
    x2: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    lambda_override: Option<f64>,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> PreparedLearner {
    let n = x1.len();

    let bk1 = (
        x1.iter().cloned().fold(f64::INFINITY, f64::min),
        x1.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let bk2 = (
        x2.iter().cloned().fold(f64::INFINITY, f64::min),
        x2.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let knots1 = bspline_knots(bk1, n_knots, degree);
    let knots2 = bspline_knots(bk2, n_knots, degree);
    let b1 = bspline_basis(x1.as_slice().expect("x1 contiguous"), &knots1, degree);
    let b2 = bspline_basis(x2.as_slice().expect("x2 contiguous"), &knots2, degree);

    let p1 = b1.ncols();
    let p2 = b2.ncols();
    let p = p1 * p2;

    let k1 = diff_penalty(p1, differences);
    let k2 = diff_penalty(p2, differences);
    let raw_penalty = kron(&k1, &Array2::eye(p2)) + kron(&Array2::eye(p1), &k2);

    let supports1 = std::sync::Arc::new(marginal_supports(&b1));
    let supports2 = std::sync::Arc::new(marginal_supports(&b2));
    let by_owned = by.map(|values| values.to_owned());
    let weights_owned = weights.to_owned();

    let raw_gram = streamed_raw_gram(
        &supports1,
        &supports2,
        by_owned.as_ref(),
        weights.as_slice().expect("weights contiguous"),
        p2,
        p,
    );

    let (gram, unscaled_penalty, centering, p_out) = if center {
        // Marginal spectral centering (#284): diagonalize K through the two
        // small marginal eigenproblems instead of the (p1·p2)-sized one,
        // transform the Gram into that basis by four mode contractions, and
        // keep the eigen-coordinates with kappa > 1e-10 rescaled by
        // 1/sqrt(kappa) so the kept penalty is the identity.
        let (u1, u2, kappa) = bspatial_marginal_eigs(&k1, &k2);
        let centering = BspatialCentering::from_kappa(u1, u2, &kappa);
        let g_tilde = bspatial_tensor_transform(&raw_gram, &centering.u1, &centering.u2);
        let centered_gram = centering.centered_gram(&g_tilde);
        let columns = centering.p_out();
        (
            centered_gram,
            Array2::eye(columns),
            Some(centering),
            columns,
        )
    } else {
        (raw_gram, raw_penalty, None, p)
    };

    let lambda = if let Some(value) = lambda_override {
        value
    } else if center {
        df_to_lambda_centered_eig(&gram, df)
    } else {
        df_to_lambda_pencil(&gram, &k1, &k2, df)
    };
    let scaled_penalty = unscaled_penalty.mapv(|value| value * lambda);
    let solver_gram = &gram + &scaled_penalty;
    let solver = GramSolver::new(&solver_gram);

    PreparedLearner::BspatialStreamed {
        b1,
        b2,
        supports1,
        supports2,
        by: by_owned,
        centering,
        p1,
        p2,
        p_out,
        penalty: scaled_penalty,
        lambda: Some(lambda),
        solver,
        name: name.to_string(),
        weights: weights_owned,
        rhs_buf: vec![0.0; p_out],
        raw_rhs_buf: vec![0.0; p],
        raw_g_buf: vec![0.0; p],
        beta_buf: vec![0.0; p_out],
        pred_buf: vec![0.0; n],
    }
}

// ─── btree: tree-based base-learner ──────────────────────────────────

/// Prepare a `btree` base-learner.
///
/// `design` is an `(n, p)` matrix of covariates the tree may split on.
/// `weights` are observation weights from the outer boosting problem.
/// `minbucket` / `minsplit` / `maxdepth` mirror R's `ctree_control` defaults
/// used inside `btree()`. `by` is an optional varying-coefficient modifier.
pub fn prepare_btree(
    design: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
    by: Option<Array1<f64>>,
    name: &str,
) -> PreparedLearner {
    prepare_btree_categorical(
        design,
        weights,
        minbucket,
        minsplit,
        maxdepth,
        by,
        name,
        Vec::new(),
    )
}

/// [`prepare_btree`] with categorical columns: `categorical[j]` is the level
/// count of design column `j`, whose values are the codes 0..L-1, and 0 for a
/// numeric column. A categorical column splits into two sets of levels.
#[allow(clippy::too_many_arguments)]
pub fn prepare_btree_categorical(
    design: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
    by: Option<Array1<f64>>,
    name: &str,
    categorical: Vec<usize>,
) -> PreparedLearner {
    prepare_btree_selected(
        design,
        weights,
        minbucket,
        minsplit,
        maxdepth,
        by,
        name,
        categorical,
        SplitSelection::Sse,
    )
}

/// [`prepare_btree_categorical`] with split method (histogram support).
#[allow(clippy::too_many_arguments)]
pub fn prepare_btree_categorical_with_method(
    design: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
    by: Option<Array1<f64>>,
    name: &str,
    categorical: Vec<usize>,
    split_method: SplitMethod,
) -> PreparedLearner {
    prepare_btree_selected_with_method(
        design,
        weights,
        minbucket,
        minsplit,
        maxdepth,
        by,
        name,
        categorical,
        SplitSelection::Sse,
        split_method,
    )
}

/// [`prepare_btree_categorical`] with an explicit column-selection rule
/// ([`SplitSelection`]): CART's squared error, or partykit's `ctree`
/// association statistic as mboost's `btree` uses.
#[allow(clippy::too_many_arguments)]
pub fn prepare_btree_selected(
    design: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
    by: Option<Array1<f64>>,
    name: &str,
    categorical: Vec<usize>,
    selection: SplitSelection,
) -> PreparedLearner {
    prepare_btree_selected_with_method(
        design,
        weights,
        minbucket,
        minsplit,
        maxdepth,
        by,
        name,
        categorical,
        selection,
        SplitMethod::Exact,
    )
}

/// [`prepare_btree_categorical`] with an explicit column-selection rule and split method.
#[allow(clippy::too_many_arguments)]
pub fn prepare_btree_selected_with_method(
    design: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
    by: Option<Array1<f64>>,
    name: &str,
    categorical: Vec<usize>,
    selection: SplitSelection,
    split_method: SplitMethod,
) -> PreparedLearner {
    let sorted_indices = btree_sorted_indices(design);
    let binning = match split_method {
        SplitMethod::Histogram => prepare_btree_binning(design, &categorical, 256),
        SplitMethod::Exact => vec![None; design.ncols()],
    };
    PreparedLearner::Btree {
        design: design.to_owned(),
        sorted_indices,
        weights: btree_fit_weights(weights, by.as_ref()),
        minbucket,
        minsplit,
        maxdepth,
        categorical,
        selection,
        split_method,
        by,
        name: name.to_string(),
        beta_buf: Vec::new(),
        pred_buf: Vec::new(),
        binning,
    }
}

/// The weights a tree is grown with. A varying-coefficient tree fits
/// `f(t) * z` to the residual `u` by least squares, which is a tree fitted to
/// `u / z` with weights `w * z^2` (see `btree_target`). For a 0/1 `z` this is
/// R's `btree(..., by = z)`: grown on the rows where `z` is 1 (`weights *
/// by`), prediction multiplied by `z`. Growing it on every row fitted the
/// splits to rows whose contribution is then zeroed, and on a treatment
/// indicator gave effects of -0.49 and 0.50 where mboost gives -0.22 and 0.75.
/// A continuous `z` (the search session's one-split varying slope) used
/// `weights * z`, which is negative wherever `z` is.
fn btree_fit_weights(weights: ArrayView1<f64>, by: Option<&Array1<f64>>) -> Array1<f64> {
    match by {
        Some(by) => &weights * &by.mapv(|z| z * z),
        None => weights.to_owned(),
    }
}

/// The working response a varying-coefficient tree is grown on: `u / z`,
/// and 0 where `z` is 0 (those rows carry no weight). `None` without `by`.
fn btree_target(u: ArrayView1<f64>, by: Option<&Array1<f64>>) -> Option<Array1<f64>> {
    let by = by?;
    Some(Array1::from_iter(
        u.iter()
            .zip(by.iter())
            .map(|(&value, &z)| if z == 0.0 { 0.0 } else { value / z }),
    ))
}

// ─── bns: unpenalised B-spline base-learner ──────────────────────────

/// Prepare a ridge-penalized B-spline base-learner (`bns`).
///
/// Constructs a B-spline basis from `x` with the given knot count and degree,
/// then wraps it as a `BlockBols` with an identity penalty and a regularization
/// weight λ derived from `df`. Unlike `bbs`, `bns` does not penalize differences
/// between adjacent coefficients; the identity penalty shrinks coefficients
/// toward zero, so it can use more degrees of freedom than `bbs`.
pub fn prepare_bns(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    df: f64,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> PreparedLearner {
    let bk = (
        x.iter().cloned().fold(f64::INFINITY, f64::min),
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let knots = bspline_knots(bk, n_knots, degree);
    let mut basis = bspline_basis_extrap(x.as_slice().expect("x contiguous"), &knots, degree, bk);

    if let Some(by_vec) = by {
        let n = basis.nrows();
        for i in 0..n {
            for j in 0..basis.ncols() {
                basis[[i, j]] *= by_vec[i];
            }
        }
    }

    let p = basis.ncols();
    // Identity penalty — unpenalised in the P-spline sense
    let penalty = Array2::eye(p);
    let lambda = df_to_lambda(&basis, &penalty, &weights.to_owned(), df);
    let penalized_k = penalty.mapv(|v| lambda * v);

    prepare_block_bols(basis, weights, penalized_k, center, name)
        .with_effective_lambda(Some(lambda))
}

// ─── bkernel: kernel base-learner ────────────────────────────────────

/// Gaussian kernel function: K(u) = exp(-u²/2).
pub fn kernel_gaussian(d: f64, bandwidth: f64) -> f64 {
    let u = d / bandwidth;
    (-0.5 * u * u).exp()
}

/// Epanechnikov kernel function: K(u) = max(0, 1 - u²) · 3/4.
pub fn kernel_epanechnikov(d: f64, bandwidth: f64) -> f64 {
    let u = d / bandwidth;
    if u.abs() < 1.0 {
        0.75 * (1.0 - u * u)
    } else {
        0.0
    }
}

/// Uniform kernel function: K(u) = 1 if |u| < 1, else 0.
pub fn kernel_uniform(d: f64, bandwidth: f64) -> f64 {
    if (d / bandwidth).abs() < 1.0 {
        1.0
    } else {
        0.0
    }
}

/// Prepare a kernel base-learner (R `bkernel`).
///
/// Constructs an n×n kernel matrix `K` where `K[i,j] = kernel(|x_i - x_j| / h)`,
/// then wraps it as a `BlockBols` with an identity penalty and a
/// bandwidth-derived regularisation λ. The kernel design matrix is the
/// kernel matrix itself (n columns, one per observation).
///
/// Because the kernel matrix is n×n, this learner is O(n²) in memory.
/// For large n, consider `bbs` instead.
///
/// `kernel_type`: "gaussian" (default), "epanechnikov", or "uniform".
#[allow(clippy::too_many_arguments)]
pub fn prepare_bkernel(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    bandwidth: f64,
    df: f64,
    kernel_type: &str,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> PreparedLearner {
    let n = x.len();

    let kernel_fn: fn(f64, f64) -> f64 = match kernel_type {
        "gaussian" => kernel_gaussian,
        "epanechnikov" => kernel_epanechnikov,
        "uniform" => kernel_uniform,
        _ => kernel_gaussian,
    };

    // Build n×n kernel matrix
    let mut design = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            design[[i, j]] = kernel_fn((x[i] - x[j]).abs(), bandwidth);
        }
    }

    if let Some(by_vec) = by {
        for i in 0..n {
            let b = by_vec[i];
            for j in 0..n {
                design[[i, j]] *= b;
            }
        }
    }

    let penalty = Array2::eye(n);
    let lambda = df_to_lambda(&design, &penalty, &weights.to_owned(), df);
    let penalized_k = penalty.mapv(|v| lambda * v);

    prepare_block_bols(design, weights, penalized_k, center, name)
        .with_effective_lambda(Some(lambda))
}

/// Prepare a Nyström-approximated kernel base-learner.
///
/// P3.4 item 6: "add a Nyström/landmark alternative to the `O(n^2)` Bkernel
/// path".
///
/// [`prepare_bkernel`] builds the full `n x n` kernel design *and* an `n x n`
/// identity penalty, so a 10,000-row fit needs 1.6 GB before it starts. This
/// builds an `n x m` design instead, for `m` landmarks chosen as evenly
/// spaced quantiles of `x` -- 8 MB at the same size with `m = 100`.
///
/// The construction is the standard one. With `K_nm` the kernel between every
/// observation and the landmarks and `K_mm` the kernel among the landmarks,
/// the design is
///
/// ```text
/// Phi = K_nm L^-T,   where K_mm = L L'
/// ```
///
/// so `Phi Phi' = K_nm K_mm^-1 K_mn`, which is the Nyström approximation of
/// `K_nn`. Only a triangular solve is needed -- no inverse square root and no
/// eigendecomposition -- and a ridge on the `m` coefficients is the same
/// penalty the exact path puts on `n` of them.
///
/// The approximation is exact when the landmarks are the observations
/// (`m = n` at distinct `x`), and a test checks that rather than asserting it.
#[allow(clippy::too_many_arguments)]
pub fn prepare_bkernel_nystrom(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    bandwidth: f64,
    df: f64,
    kernel_type: &str,
    landmarks: usize,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> Result<PreparedLearner, MboostrsError> {
    prepare_bkernel_nystrom_parts(
        x,
        weights,
        bandwidth,
        df,
        kernel_type,
        landmarks,
        center,
        name,
        by,
    )
    .map(|parts| parts.learner)
}

/// What a Nyström preparation produces: the learner, and the two pieces its
/// new-data replay needs.
///
/// The replay cannot be rebuilt from the covariate alone the way the exact
/// kernel's can: the coefficients index the landmarks and the design was
/// mapped through `L^-T`, so both travel with the fit.
pub struct NystromParts {
    /// The prepared block learner.
    pub learner: PreparedLearner,
    /// Landmark positions; coefficients index them.
    pub landmarks: Vec<f64>,
    /// Lower-triangular Cholesky factor of `K_mm`, column-major.
    pub cholesky_l: Vec<f64>,
}

/// [`prepare_bkernel_nystrom`], keeping the landmarks and the factor.
#[allow(clippy::too_many_arguments)]
pub fn prepare_bkernel_nystrom_parts(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    bandwidth: f64,
    df: f64,
    kernel_type: &str,
    landmarks: usize,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> Result<NystromParts, MboostrsError> {
    let n = x.len();
    if landmarks == 0 {
        return Err(MboostrsError::LearnerSetup(
            name.to_string(),
            "landmarks must be at least 1".to_string(),
        ));
    }
    let kernel_fn: fn(f64, f64) -> f64 = match kernel_type {
        "gaussian" => kernel_gaussian,
        "epanechnikov" => kernel_epanechnikov,
        "uniform" => kernel_uniform,
        _ => kernel_gaussian,
    };

    let centres = kernel_landmarks(x, landmarks);
    let m = centres.len();

    // `K_mm`, with a small ridge so the factorisation survives landmarks that
    // are close together -- the alternative is a rank-deficient factor and a
    // design full of NaN, which fits and reports nothing.
    let mut k_mm = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            k_mm[[i, j]] = kernel_fn((centres[i] - centres[j]).abs(), bandwidth);
        }
        k_mm[[i, i]] += 1e-8;
    }
    let factor = crate::linalg::CachedCholesky::new(&k_mm)?;
    let l = factor.l_slice();

    // `Phi = K_nm L^-T`: for each row `k` of `K_nm`, solve `L phi = k`.
    let mut design = Array2::zeros((n, m));
    let mut row = vec![0.0f64; m];
    for i in 0..n {
        for (j, centre) in centres.iter().enumerate() {
            row[j] = kernel_fn((x[i] - centre).abs(), bandwidth);
        }
        for j in 0..m {
            let mut value = row[j];
            for k in 0..j {
                value -= l[k * m + j] * design[[i, k]];
            }
            design[[i, j]] = value / l[j * m + j];
        }
    }

    if let Some(by_vec) = by {
        for i in 0..n {
            let scale = by_vec[i];
            for j in 0..m {
                design[[i, j]] *= scale;
            }
        }
    }

    let penalty = Array2::eye(m);
    let lambda = df_to_lambda(&design, &penalty, &weights.to_owned(), df);
    let penalized = penalty.mapv(|value| lambda * value);
    Ok(NystromParts {
        learner: prepare_block_bols(design, weights, penalized, center, name)
            .with_effective_lambda(Some(lambda)),
        landmarks: centres,
        cholesky_l: l.to_vec(),
    })
}

/// `m` landmark positions: evenly spaced quantiles of the observed `x`.
///
/// Deterministic, so two fits on the same data get the same basis, and
/// data-driven, so the landmarks sit where the observations are rather than
/// on a grid that may be mostly empty. Duplicates are dropped -- a tied `x`
/// would make `K_mm` singular -- so the returned count can be below `m`.
fn kernel_landmarks(x: ArrayView1<f64>, landmarks: usize) -> Vec<f64> {
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return Vec::new();
    }
    let m = landmarks.min(sorted.len());
    let mut centres = Vec::with_capacity(m);
    for index in 0..m {
        let position = if m == 1 {
            (sorted.len() - 1) / 2
        } else {
            (index * (sorted.len() - 1)) / (m - 1)
        };
        centres.push(sorted[position]);
    }
    centres.dedup();
    centres
}

// ─── bconstrained: constrained P-spline base-learner ─────────────────

/// Prepare a constrained P-spline base-learner (R `bconstrained`).
///
/// Generalises `bmono` to arbitrary linear constraints of the form `C β ≥ 0`,
/// where C is a constraint matrix supplied by the caller. For monotone
/// constraints, C = D (first-difference matrix); for convexity, C = D²;
/// for sign constraints, C = I; etc.
///
/// Uses the same iterative violation-penalty algorithm as `bmono`.
#[allow(clippy::too_many_arguments)]
/// Prepare a user-supplied base-learner (R `buser`).
///
/// Accepts an arbitrary `(n × p)` design matrix `X` and a `(p × p)` penalty
/// matrix `K`, then wraps them as a `BlockBols` with λ derived from `df`
/// (or λ provided by `lambda_override`). This is the most flexible
/// base-learner — callers can define custom design/penalty pairs without
/// writing Rust code.
///
/// When `by` is supplied, each column of `X` is element-wise multiplied by
/// the by-variable vector.
pub fn prepare_buser(
    design: Array2<f64>,
    penalty: Array2<f64>,
    weights: ArrayView1<f64>,
    df: f64,
    lambda_override: Option<f64>,
    center: bool,
    name: &str,
    by: Option<ArrayView1<f64>>,
) -> PreparedLearner {
    let mut design = design;

    if let Some(by_vec) = by {
        let n = design.nrows();
        for i in 0..n {
            for j in 0..design.ncols() {
                design[[i, j]] *= by_vec[i];
            }
        }
    }

    let lambda =
        lambda_override.unwrap_or_else(|| df_to_lambda(&design, &penalty, &weights.to_owned(), df));
    let penalized_penalty = penalty.mapv(|v| lambda * v);

    if center {
        prepare_block_bols(design, weights, penalized_penalty, true, name)
            .with_effective_lambda(Some(lambda))
    } else {
        prepare_block_bols_interaction(design, weights, penalized_penalty, name)
            .with_effective_lambda(Some(lambda))
    }
}

/// Prepare a Kronecker-product base-learner (R `%O%` operator).
///
/// Fits a linear array model with design `kronecker(X2, X1)`, where
/// X1 is `(n_obs, p1)` and X2 is `(n_resp, p2)`. The response vector
/// has length `n_obs * n_resp`, stored in **row-major** (interleaved) order:
/// `y[i * n_resp + k]` is the value for observation `i`, response dimension `k`.
///
/// For Multinomial with `X2 = I_{K-1}`, this is equivalent to fitting
/// `K-1` independent Ridge regressions with shared design X1 and penalty.
///
/// `weights` has length `n_obs` (per-observation).
/// `penalty_x1` is the penalty matrix for X1 (typically `lambda * K`).
/// `x2` is the second Kronecker factor (use identity for Multinomial).
#[allow(clippy::too_many_arguments)]
pub fn prepare_kronecker_bols(
    x1: Array2<f64>,
    weights: ArrayView1<f64>,
    penalty_x1: Array2<f64>,
    x2: Array2<f64>,
    center: bool,
    name: &str,
) -> PreparedLearner {
    let n_obs = x1.nrows();
    let p1 = x1.ncols();
    let n_resp = x2.nrows();
    let p2 = x2.ncols();
    let x2_is_identity = is_identity_matrix(&x2);

    let mut d = x1;
    let centering = if center {
        let sw: f64 = weights.sum();
        let c = if sw > 0.0 {
            (0..p1)
                .map(|j| d.column(j).dot(&weights) / sw)
                .collect::<Array1<f64>>()
        } else {
            Array1::zeros(p1)
        };
        for j in 0..p1 {
            let cj = c[j];
            d.column_mut(j).mapv_inplace(|v| v - cj);
        }
        c
    } else {
        Array1::zeros(p1)
    };

    let x1_faer = ndarray2_to_faer(&d);
    let wx1t_faer = Mat::from_fn(p1, n_obs, |j, i| weights[i] * d[[i, j]]);

    let mut gram_faer = ndarray2_to_faer(&penalty_x1);
    faer::linalg::matmul::matmul(
        &mut gram_faer,
        &wx1t_faer,
        &x1_faer,
        Some(1.0),
        1.0,
        faer::Parallelism::None,
    );
    let gram = Array2::from_shape_fn((p1, p1), |(j, k)| gram_faer.read(j, k));
    let solver = GramSolver::new(&gram);

    let rhs_buf = vec![0.0; p1];
    let beta_buf = vec![0.0; p1 * p2];
    let pred_buf = vec![0.0; n_obs * n_resp];

    PreparedLearner::KroneckerBols {
        x1: d,
        x1_faer,
        wx1t_faer,
        solver,
        x2,
        x2_is_identity,
        penalty: penalty_x1,
        centering,
        n_obs,
        n_resp,
        name: name.to_string(),
        rhs_buf,
        beta_buf,
        pred_buf,
        x1_rebuild: None,
    }
}

/// Prepare a `bbs` base-learner Kronecker-wrapped for the Multinomial family
/// (R's `bbs(x, df) %O% bols(class, df = K-1, contr.dummy)`).
///
/// Reverse-engineered from mboost's `%O%`: the marginal design is the **raw**
/// B-spline basis `X1` (un-centered), and the per-class penalty is `I + K1`
/// (the bols(class) ridge identity penalty added to the bbs difference penalty
/// `K1`), with a single `lambda` chosen so the per-class effective df equals
/// `df` (the combined `df1*df2` divided by the `K-1` identical class blocks).
/// `x2 = I_{K-1}`. This reproduces R to ~1e-11.
#[allow(clippy::too_many_arguments)]
pub fn prepare_kronecker_bbs(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    n_resp: usize,
    name: &str,
    deriv: usize,
) -> PreparedLearner {
    prepare_kronecker_bbs_centered(
        x,
        weights,
        n_knots,
        degree,
        differences,
        df,
        n_resp,
        name,
        deriv,
        false,
    )
}

/// [`prepare_kronecker_bbs`] with optional centering (`bbs(x, center=TRUE)`):
/// the basis is multiplied by [`centering_transform`], the same
/// reparametrization an ordinary centred `bbs` uses, so the term fits only
/// the deviation from the penalty's null space (a line, for second
/// differences) and its penalty becomes the identity. The class ridge adds another identity, which only rescales the
/// single lambda, so the per-class df target is unchanged. The transform is
/// stored for new-data replay.
#[allow(clippy::too_many_arguments)]
pub fn prepare_kronecker_bbs_centered(
    x: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_knots: usize,
    degree: usize,
    differences: usize,
    df: f64,
    n_resp: usize,
    name: &str,
    deriv: usize,
    center: bool,
) -> PreparedLearner {
    let bk = (
        x.iter().cloned().fold(f64::INFINITY, f64::min),
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let knots = bspline_knots(bk, n_knots, degree);
    let raw_basis = bspline_basis(x.as_slice().expect("x contiguous"), &knots, degree);
    let p1 = raw_basis.ncols();
    let raw_penalty = diff_penalty(p1, differences);
    let (basis, penalty, transform) = if center {
        let t = centering_transform(p1, differences);
        let q = t.ncols();
        let flat: Vec<f64> = t.iter().copied().collect();
        (raw_basis.dot(&t), Array2::<f64>::eye(q), Some((flat, q)))
    } else {
        // Per-class penalty: bbs difference penalty K1 plus the bols(class)
        // ridge identity penalty (the `kron(K2, I) + kron(I, K1)` of mboost's
        // %O%, with K2 = I for bols(class, contr.dummy)).
        (raw_basis, &raw_penalty + &Array2::<f64>::eye(p1), None)
    };
    // Single lambda so the per-class effective df matches `df` (mboost targets
    // the combined df1*df2 over the K-1 identical class blocks).
    let lambda = df_to_lambda(&basis, &penalty, &weights.to_owned(), df);
    let penalized = penalty.mapv(|v| lambda * v);
    let x2 = Array2::<f64>::eye(n_resp);
    let mut learner = prepare_kronecker_bols(basis, weights, penalized, x2, false, name);
    learner.set_x1_rebuild(KroneckerRebuild::Bbs {
        knots,
        degree,
        boundary_knots: bk,
        deriv,
        transform,
    });
    learner
}

/// The `(p, q)` matrix a centred `bbs` multiplies its basis by: `D'(DD')^-1`
/// for the difference matrix `D` (mboost's `"differenceMatrix"` method), or,
/// when `DD'` cannot be factored, the spectral transform `V Lambda^{-1/2}` of
/// the penalty's non-null eigenvectors -- the order `prepare_bbs` uses, so a
/// Kronecker term and an ordinary one land in the same reduced space.
pub fn centering_transform(p: usize, differences: usize) -> Array2<f64> {
    let identity = Array2::<f64>::eye(p);
    let penalty = diff_penalty(p, differences);
    raw_diff_matrix(p, differences)
        .and_then(|d_raw| center_difference_matrix(&identity, &d_raw))
        .unwrap_or_else(|| center_spectral(&identity, &penalty, differences))
}

/// Fit a regression tree (stump or deeper) to the pseudo-response `u`.
///
/// Returns `(coef, pred)` where:
/// - `coef` encodes the tree via [`FittedTree::to_coef_vec`].
/// - `pred` is the fitted values on training data (node means × by if present).
///
/// **Algorithm** (CART-style, matching R `btree` with `ctree_control(mincriterion=0)`):
/// 1. For each column of `design`, sort and evaluate every valid binary split.
/// 2. A split is valid if both children have ≥ `minbucket` weighted observations
///    and the parent has ≥ `minsplit`.
/// 3. Pick the split that minimises weighted SSE of `u`.
/// 4. For `maxdepth > 1`, recursively split each child up to `maxdepth`.
fn btree_sorted_indices(design: ArrayView2<f64>) -> Vec<Vec<usize>> {
    let n = design.nrows();
    let mut sorted = Vec::with_capacity(design.ncols());
    for j in 0..design.ncols() {
        let col = design.column(j);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            col[a]
                .partial_cmp(&col[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.push(idx);
    }
    sorted
}

fn prepare_btree_binning(
    design: ArrayView2<f64>,
    categorical: &[usize],
    requested_bins: usize,
) -> Vec<Option<QuantileBinning>> {
    let mut binning = Vec::with_capacity(design.ncols());
    for j in 0..design.ncols() {
        if categorical.get(j).copied().unwrap_or(0) > 0 {
            binning.push(None);
            continue;
        }
        let col = design.column(j);
        match quantile_bin_column(col, requested_bins) {
            Ok(b) => binning.push(Some(b)),
            Err(_) => binning.push(None),
        }
    }
    binning
}

#[cfg(test)]
fn fit_tree(
    design: ArrayView2<f64>,
    u: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
) -> (Array1<f64>, Array1<f64>) {
    let sorted_indices = btree_sorted_indices(design);
    fit_tree_with_sorted(
        design,
        u,
        weights,
        &sorted_indices,
        &[],
        SplitSelection::Sse,
        SplitMethod::Exact,
        &vec![None; design.ncols()],
        minbucket,
        minsplit,
        maxdepth,
    )
}

fn fit_tree_with_sorted(
    design: ArrayView2<f64>,
    u: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    sorted_indices: &[Vec<usize>],
    categorical: &[usize],
    selection: SplitSelection,
    split_method: SplitMethod,
    binning: &[Option<QuantileBinning>],
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
) -> (Array1<f64>, Array1<f64>) {
    let mut nodes: Vec<TreeNode> = Vec::new();
    let spec = TreeSpec {
        design: design.reborrow(),
        u: u.reborrow(),
        weights: weights.reborrow(),
        categorical,
        selection,
        split_method,
        binning,
        minbucket,
        minsplit,
        maxdepth,
    };
    let n = design.nrows();
    let mut sorted = vec![0usize; n * design.ncols()];
    for j in 0..design.ncols() {
        // Histogram mode reads binned codes instead of the sorted order. The
        // exact fallback for an unbinned column still needs its order.
        let unbinned = binning.get(j).and_then(|b| b.as_ref()).is_none();
        if categorical.get(j).copied().unwrap_or(0) == 0
            && (split_method == SplitMethod::Exact || unbinned)
        {
            sorted[j * n..(j + 1) * n].copy_from_slice(&sorted_indices[j]);
        }
    }
    let hist_bins = binning
        .iter()
        .filter_map(|b| b.as_ref().map(|q| q.effective))
        .max()
        .unwrap_or(0);
    let max_levels = categorical.iter().copied().max().unwrap_or(0);
    let mut work = TreeWork {
        n,
        rows: (0..n).collect(),
        sorted,
        scratch: vec![0usize; n],
        goes_left: vec![false; n],
        hist: vec![(0.0, 0.0, 0.0); hist_bins],
        hist_counts: vec![0usize; hist_bins],
        level_stats: vec![(0.0, 0.0, 0.0); max_levels],
        level_present: Vec::with_capacity(max_levels),
    };
    build_tree_recursive(&spec, &mut work, 0, n, 0, &mut nodes);
    let tree = FittedTree { nodes };
    let pred = tree.predict(design);
    let coef = tree.to_coef_vec();
    (coef, pred)
}

/// How a tree node chooses the column to split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SplitSelection {
    /// The split, over every column, that most reduces the weighted squared
    /// error (CART). Favours columns with many candidate split points.
    #[default]
    Sse,
    /// partykit's `ctree` as mboost's `btree` runs it (`teststat = "quad"`,
    /// `testtype = "Teststatistic"`): the column with the largest quadratic
    /// statistic of association with the gradient, `n r^2` for a number and
    /// the between-level sum of squares over the variance for a factor.
    CtreeStatistic,
    /// The same statistic referred to its chi-squared distribution (1 degree
    /// of freedom for a number, `L - 1` for a factor with `L` levels at the
    /// node): the smallest p-value, which removes the preference for factors
    /// with many levels.
    CtreePvalue,
}

/// How a tree finds splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SplitMethod {
    /// Exhaustive search over sorted unique values (exact).
    #[default]
    Exact,
    /// Histogram search over binned values.
    Histogram,
}

/// The fixed inputs of one tree fit.
struct TreeSpec<'a> {
    design: ArrayView2<'a, f64>,
    u: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    /// Level count of each categorical column (codes 0..L-1); 0, or an empty
    /// slice, for a numeric column.
    categorical: &'a [usize],
    selection: SplitSelection,
    split_method: SplitMethod,
    minbucket: usize,
    minsplit: usize,
    maxdepth: usize,
    /// Precomputed quantile binning for numeric columns when using histogram method.
    binning: &'a [Option<QuantileBinning>],
}

/// The best split found at a node.
enum BestSplit {
    /// `x <= point`, with the last left position in the column's order at
    /// the node.
    Numeric {
        var: usize,
        point: f64,
        last_left: usize,
        /// The highest bin sent left, in histogram mode; `None` on the exact
        /// path, where the split is reconstructed from `work.sorted`.
        bin: Option<usize>,
    },
    /// Level codes sent left.
    Levels { var: usize, left_levels: Vec<usize> },
}

/// The column ctree splits at this node: the largest quadratic association
/// statistic between the column and the gradient `u` (or the smallest
/// chi-squared p-value of it), `None` when no column varies.
fn ctree_column(
    spec: &TreeSpec<'_>,
    indices: &[usize],
    rule: SplitSelection,
    total_w: f64,
    total_wsum: f64,
    total_sqsum: f64,
) -> Option<usize> {
    let (design, u, weights) = (spec.design, spec.u, spec.weights);
    let u_mean = total_wsum / total_w;
    let v_u = (total_sqsum / total_w - u_mean * u_mean).max(0.0);
    if v_u <= 0.0 || total_w <= 1.0 {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for j in 0..design.ncols() {
        let col = design.column(j);
        let n_levels = spec.categorical.get(j).copied().unwrap_or(0);
        let (statistic, df) = if n_levels > 0 {
            let mut w_l = vec![0.0f64; n_levels];
            let mut s_l = vec![0.0f64; n_levels];
            for &ii in indices {
                let code = col[ii];
                if code >= 0.0 && code < n_levels as f64 {
                    w_l[code as usize] += weights[ii];
                    s_l[code as usize] += weights[ii] * u[ii];
                }
            }
            let present = w_l.iter().filter(|&&w| w > 0.0).count();
            if present < 2 {
                continue;
            }
            // Between-level sum of squares over the permutation variance.
            let between: f64 = (0..n_levels)
                .filter(|&l| w_l[l] > 0.0)
                .map(|l| (s_l[l] - w_l[l] * u_mean).powi(2) / w_l[l])
                .sum();
            (
                between / v_u * (total_w - 1.0) / total_w,
                (present - 1) as f64,
            )
        } else {
            let (mut sw_x, mut sw_xx, mut t) = (0.0, 0.0, 0.0);
            for &ii in indices {
                let (wi, xi) = (weights[ii], col[ii]);
                sw_x += wi * xi;
                sw_xx += wi * xi * xi;
                t += wi * xi * u[ii];
            }
            let var_x = sw_xx - sw_x * sw_x / total_w;
            if var_x <= 1e-12 * sw_xx.abs().max(1.0) {
                continue;
            }
            // Strasser-Weber: E(T) = sum(w x) mean(u), Var(T) =
            // W/(W-1) v_u (sum(w x^2) - sum(w x)^2 / W).
            let mu = sw_x * u_mean;
            let var_t = total_w / (total_w - 1.0) * v_u * var_x;
            ((t - mu).powi(2) / var_t, 1.0)
        };
        let score = match rule {
            SplitSelection::CtreePvalue => -crate::causal::chi_square_sf(statistic, df),
            _ => statistic,
        };
        if best.is_none_or(|(_, b)| score > b) {
            best = Some((j, score));
        }
    }
    best.map(|(j, _)| j)
}

/// Buffers one tree fit works in, allocated once per tree.
///
/// A node is a range `lo..hi`: `rows[lo..hi]` are its rows, and for each
/// numeric column `j`, `sorted[j * n + lo..j * n + hi]` are the same rows in
/// the column's order. A split partitions every range in place (stably, so
/// each column stays sorted) and the children take the two halves.
struct TreeWork {
    n: usize,
    rows: Vec<usize>,
    sorted: Vec<usize>,
    scratch: Vec<usize>,
    goes_left: Vec<bool>,
    /// Histogram accumulation buffers, one entry per bin of the current
    /// numeric column: `(Σw, Σw·u, Σw·u²)`. Reused across nodes and columns.
    hist: Vec<(f64, f64, f64)>,
    /// Row counts per bin, used to reconstruct `last_left` without `sorted`.
    hist_counts: Vec<usize>,
    /// Per-level `(Σw, Σw·u, Σw·u²)` buffers for the categorical path.
    level_stats: Vec<(f64, f64, f64)>,
    /// Level indices with positive weight at the current node.
    level_present: Vec<usize>,
}

impl TreeWork {
    /// Stably move the rows marked `goes_left` to the front of `range`.
    fn partition(buf: &mut [usize], scratch: &mut [usize], goes_left: &[bool]) {
        let mut front = 0;
        let mut back = 0;
        for k in 0..buf.len() {
            let ii = buf[k];
            if goes_left[ii] {
                buf[front] = ii;
                front += 1;
            } else {
                scratch[back] = ii;
                back += 1;
            }
        }
        buf[front..].copy_from_slice(&scratch[..back]);
    }
}

/// Recursively build a CART regression tree over the node `lo..hi` of
/// `work`.
///
/// A numeric column is scanned in its presorted order. A categorical column
/// (codes 0..L-1) has its levels ordered by the mean of `u` at this node and
/// scanned in that order: for squared error, the best split of the levels
/// into two groups is one of these L-1 prefixes (Breiman et al. 1984,
/// section 9.4), so a level set is found without trying all 2^(L-1)
/// partitions.
///
/// Returns the index of this node in `nodes`.
fn build_tree_recursive(
    spec: &TreeSpec<'_>,
    work: &mut TreeWork,
    lo: usize,
    hi: usize,
    depth: usize,
    nodes: &mut Vec<TreeNode>,
) -> usize {
    let (design, u, weights) = (spec.design, spec.u, spec.weights);
    let n = hi - lo;
    let p = design.ncols();
    let stride = work.n;

    let mut total_w = 0.0f64;
    let mut total_wsum = 0.0f64;
    let mut total_sqsum = 0.0f64;
    for &ii in &work.rows[lo..hi] {
        let wi = weights[ii];
        let ui = u[ii];
        total_w += wi;
        total_wsum += wi * ui;
        total_sqsum += wi * ui * ui;
    }
    let leaf = |nodes: &mut Vec<TreeNode>| {
        let mean = if total_w > 0.0 {
            total_wsum / total_w
        } else {
            0.0
        };
        nodes.push(TreeNode::Leaf { mean });
        nodes.len() - 1
    };

    if total_w < spec.minsplit as f64 || n < spec.minsplit || depth >= spec.maxdepth {
        return leaf(nodes);
    }

    let minbucket = spec.minbucket as f64;
    let split_sse = |lw: f64, lws: f64, lsq: f64| {
        let rw = total_w - lw;
        let rws = total_wsum - lws;
        let lsse = lsq - lws * lws / lw;
        let rsse = (total_sqsum - lsq) - rws * rws / rw;
        (lsse + rsse).max(0.0)
    };

    let mut best_sse = f64::INFINITY;
    let mut best: Option<BestSplit> = None;
    // ctree chooses the column first, by its association with the gradient;
    // only that column's split points are then searched.
    let only = match spec.selection {
        SplitSelection::Sse => None,
        rule => ctree_column(
            spec,
            &work.rows[lo..hi],
            rule,
            total_w,
            total_wsum,
            total_sqsum,
        ),
    };
    if spec.selection != SplitSelection::Sse && only.is_none() {
        return leaf(nodes);
    }

    for j in 0..p {
        if only.is_some_and(|chosen| chosen != j) {
            continue;
        }
        let col = design.column(j);
        let n_levels = spec.categorical.get(j).copied().unwrap_or(0);
        if n_levels > 0 {
            // Per-level weight, weighted sum and weighted sum of squares,
            // accumulated in reusable per-node buffers (bins = levels).
            let TreeWork {
                rows,
                level_stats: stats,
                level_present: present,
                ..
            } = &mut *work;
            for entry in stats[..n_levels].iter_mut() {
                *entry = (0.0, 0.0, 0.0);
            }
            for &ii in &rows[lo..hi] {
                let code = col[ii];
                if code < 0.0 || code >= n_levels as f64 {
                    continue;
                }
                let (w, ws, sq) = &mut stats[code as usize];
                let (wi, ui) = (weights[ii], u[ii]);
                *w += wi;
                *ws += wi * ui;
                *sq += wi * ui * ui;
            }
            present.clear();
            for (l, entry) in stats[..n_levels].iter().enumerate() {
                if entry.0 > 0.0 {
                    present.push(l);
                }
            }
            if present.len() < 2 {
                continue;
            }
            present.sort_by(|&a, &b| {
                (stats[a].1 / stats[a].0)
                    .partial_cmp(&(stats[b].1 / stats[b].0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let (mut lw, mut lws, mut lsq) = (0.0, 0.0, 0.0);
            let mut col_best_k: Option<usize> = None;
            let mut col_best_sse = best_sse;
            for k in 0..present.len() - 1 {
                let (w, ws, sq) = stats[present[k]];
                lw += w;
                lws += ws;
                lsq += sq;
                if lw < minbucket || total_w - lw < minbucket {
                    continue;
                }
                let sse = split_sse(lw, lws, lsq);
                if sse < col_best_sse {
                    col_best_sse = sse;
                    col_best_k = Some(k);
                }
            }
            if let Some(k) = col_best_k {
                best_sse = col_best_sse;
                // Build the winning level set once per column, not once per
                // improving candidate.
                let mut left_levels = present[..=k].to_vec();
                left_levels.sort_unstable();
                best = Some(BestSplit::Levels {
                    var: j,
                    left_levels,
                });
            }
            continue;
        }

        // This node's rows in the column's order.
        if spec.split_method == SplitMethod::Histogram {
            if let Some(binning) = spec.binning.get(j).and_then(|b| b.as_ref()) {
                let effective = binning.effective;
                let TreeWork {
                    rows,
                    hist,
                    hist_counts,
                    level_present: present,
                    ..
                } = &mut *work;
                for entry in hist[..effective].iter_mut() {
                    *entry = (0.0, 0.0, 0.0);
                }
                hist_counts[..effective].fill(0);
                for &ii in &rows[lo..hi] {
                    let code = binning.index[ii] as usize;
                    let entry = &mut hist[code];
                    let wi = weights[ii];
                    let ui = u[ii];
                    entry.0 += wi;
                    entry.1 += wi * ui;
                    entry.2 += wi * ui * ui;
                    hist_counts[code] += 1;
                }
                // Candidate boundaries sit only between bins that actually hold
                // this node's rows. This keeps the candidate set identical to
                // the exact scan when there is at most one distinct value per
                // bin (a boundary is otherwise placed at a midpoint between two
                // absent-in-this-node values, which no exact scan would emit).
                let present = &mut *present;
                present.clear();
                for (b, &count) in hist_counts[..effective].iter().enumerate() {
                    if count > 0 {
                        present.push(b);
                    }
                }
                let (mut lw, mut lws, mut lsq) = (0.0f64, 0.0f64, 0.0f64);
                let mut lcount = 0usize;
                for idx in 0..present.len().saturating_sub(1) {
                    let b = present[idx];
                    lw += hist[b].0;
                    lws += hist[b].1;
                    lsq += hist[b].2;
                    lcount += hist_counts[b];
                    if lw < minbucket || total_w - lw < minbucket {
                        continue;
                    }
                    let sse = split_sse(lw, lws, lsq);
                    if sse < best_sse {
                        best_sse = sse;
                        let next = present[idx + 1];
                        best = Some(BestSplit::Numeric {
                            var: j,
                            point: (binning.values[b] + binning.values[next]) / 2.0,
                            last_left: lcount - 1,
                            bin: Some(b),
                        });
                    }
                }
                continue;
            }
        }

        let idx = &work.sorted[j * stride + lo..j * stride + hi];
        let (mut lw, mut lws, mut lsq) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..n - 1 {
            let ii = idx[k];
            let ii_next = idx[k + 1];
            let wi = weights[ii];
            let ui = u[ii];
            lws += wi * ui;
            lw += wi;
            lsq += wi * ui * ui;
            if col[ii] == col[ii_next] {
                continue;
            }
            if lw < minbucket || total_w - lw < minbucket {
                continue;
            }
            let sse = split_sse(lw, lws, lsq);
            if sse < best_sse {
                best_sse = sse;
                best = Some(BestSplit::Numeric {
                    var: j,
                    point: (col[ii] + col[ii_next]) / 2.0,
                    last_left: k,
                    bin: None,
                });
            }
        }
    }

    // Mark the rows that go left, and fix the node.
    let (node, n_left, numeric_var) = match best {
        None => return leaf(nodes),
        Some(BestSplit::Numeric {
            var,
            point,
            last_left,
            bin,
        }) => {
            {
                let TreeWork {
                    rows,
                    goes_left,
                    sorted,
                    ..
                } = &mut *work;
                match bin {
                    Some(b) => {
                        // Histogram mode: the marker comes from the binned
                        // code, `work.sorted` is not maintained.
                        let index = &spec.binning[var]
                            .as_ref()
                            .expect("histogram split requires a binning")
                            .index;
                        for &ii in &rows[lo..hi] {
                            goes_left[ii] = (index[ii] as usize) <= b;
                        }
                    }
                    None => {
                        let order = &sorted[var * stride + lo..var * stride + hi];
                        for (k, &ii) in order.iter().enumerate() {
                            goes_left[ii] = k <= last_left;
                        }
                    }
                }
            }
            (
                TreeNode::Split {
                    var,
                    point,
                    left: 0,
                    right: 0,
                },
                last_left + 1,
                Some(var),
            )
        }
        Some(BestSplit::Levels {
            var,
            mut left_levels,
        }) => {
            let col = design.column(var);
            let in_set = |levels: &[usize], ii: usize| {
                let v = col[ii];
                v >= 0.0 && levels.binary_search(&(v as usize)).is_ok()
            };
            let rows = &work.rows[lo..hi];
            // Store the lighter child's levels: a level the fit never saw is
            // routed away from the stored set, so it follows the child that
            // held more of the training weight.
            let listed_weight: f64 = rows
                .iter()
                .filter(|&&ii| in_set(&left_levels, ii))
                .map(|&ii| weights[ii])
                .sum();
            if listed_weight > total_w - listed_weight {
                let mut other: Vec<usize> = rows
                    .iter()
                    .filter(|&&ii| !in_set(&left_levels, ii))
                    .map(|&ii| col[ii] as usize)
                    .collect();
                other.sort_unstable();
                other.dedup();
                left_levels = other;
            }
            let mut n_left = 0;
            for &ii in rows {
                let left = in_set(&left_levels, ii);
                work.goes_left[ii] = left;
                n_left += usize::from(left);
            }
            (
                TreeNode::SplitSet {
                    var,
                    left_levels,
                    left: 0,
                    right: 0,
                },
                n_left,
                None,
            )
        }
    };

    let split_idx = nodes.len();
    nodes.push(node);
    // Hand each child its rows in every column's order: a stable partition of
    // this node's ranges, so each level of the tree costs O(n p) and no
    // buffer is allocated per node.
    let TreeWork {
        rows,
        sorted,
        scratch,
        goes_left,
        ..
    } = work;
    if spec.split_method == SplitMethod::Exact {
        for j in 0..p {
            if spec.categorical.get(j).copied().unwrap_or(0) == 0 {
                TreeWork::partition(
                    &mut sorted[j * stride + lo..j * stride + hi],
                    scratch,
                    goes_left,
                );
            }
        }
    }
    match numeric_var {
        // A numeric split's children list their rows in the split column's
        // order, the order their totals are summed in. The histogram path
        // keeps only `rows`, which it partitions directly.
        Some(var) if spec.split_method == SplitMethod::Exact => {
            rows[lo..hi].copy_from_slice(&sorted[var * stride + lo..var * stride + hi])
        }
        _ => TreeWork::partition(&mut rows[lo..hi], scratch, goes_left),
    }
    let left_child = build_tree_recursive(spec, work, lo, lo + n_left, depth + 1, nodes);
    let right_child = build_tree_recursive(spec, work, lo + n_left, hi, depth + 1, nodes);
    match &mut nodes[split_idx] {
        TreeNode::Split { left, right, .. } | TreeNode::SplitSet { left, right, .. } => {
            *left = left_child;
            *right = right_child;
        }
        TreeNode::Leaf { .. } => unreachable!("a split node was pushed at split_idx"),
    }
    split_idx
}

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
