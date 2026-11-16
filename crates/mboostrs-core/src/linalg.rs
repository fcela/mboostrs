use crate::MboostrsError;
use faer::dyn_stack::PodStack;
use faer::linalg::cholesky::llt::compute::{
    cholesky_in_place, cholesky_in_place_req, LltParams, LltRegularization,
};
use faer::linalg::cholesky::llt::solve::solve_in_place_with_conj;
use faer::linalg::solvers::Cholesky;
use faer::prelude::SpSolver;
use faer::{Conj, Mat, Parallelism, Side};

/// Solve `a · x = b` for square positive-definite `a` via Cholesky.
pub fn cho_solve(
    a: &ndarray::Array2<f64>,
    b: &ndarray::ArrayView1<f64>,
) -> Result<ndarray::Array1<f64>, MboostrsError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(MboostrsError::Solver(
            "dimension mismatch in cho_solve".into(),
        ));
    }

    let faer_a = Mat::from_fn(n, n, |i, j| a[[i, j]]);

    let chol = Cholesky::try_new(faer_a.as_ref(), Side::Lower)
        .map_err(|_| MboostrsError::Solver("Cholesky factorisation failed".into()))?;

    let mut x = vec![0.0f64; n];
    // Copy b into x, then solve in-place
    x.copy_from_slice(b.as_slice().expect("b contiguous"));
    let mut rhs = faer::mat::from_column_major_slice_mut(&mut x, n, 1);
    chol.solve_in_place(&mut rhs);

    Ok(ndarray::Array1::from_vec(x))
}

/// Length of the `f64` scratch buffer [`cho_solve_inplace`] needs for an
/// `n × n` system. Sized once at learner-preparation time so the solve itself
/// allocates nothing; `n` is the largest dimension the caller will pass.
pub(crate) fn cho_solve_inplace_scratch_len(n: usize) -> usize {
    let request = cholesky_in_place_req::<f64>(n, Parallelism::None, LltParams::default())
        .expect("Cholesky scratch request overflowed");
    // dyn-stack measures in bytes and consumes leading padding to reach its
    // alignment. An `f64` buffer is already 8-aligned, which is what faer
    // hands out here, so the unaligned requirement plus one alignment unit is
    // a sufficient bound.
    let bytes = request.unaligned_bytes_required() + request.align_bytes();
    bytes.div_ceil(std::mem::size_of::<f64>())
}

/// Solve `A·x = b` writing the Cholesky factor over `a_col` and the solution
/// over `b_x`. `a_col` must be `n*n` dense column-major (`(i, j)` at
/// `j*n + i`); `b_x` holds `b` on entry and `x` on exit. Both are destroyed.
///
/// Allocation-free sibling of [`cho_solve`] for a matrix rebuilt every
/// iteration: faer factors into the caller's buffer and takes its working
/// memory from `scratch`, which [`cho_solve_inplace_scratch_len`] sizes. The
/// arithmetic is faer's same LLᵀ on the same values, so the result matches
/// `cho_solve` bit for bit.
///
/// # Panics
///
/// If `scratch` is shorter than `cho_solve_inplace_scratch_len(n)`.
pub(crate) fn cho_solve_inplace(
    n: usize,
    a_col: &mut [f64],
    b_x: &mut [f64],
    scratch: &mut [f64],
) -> Result<(), MboostrsError> {
    debug_assert_eq!(a_col.len(), n * n);
    debug_assert_eq!(b_x.len(), n);
    let stack = PodStack::new_any(scratch);
    let mut a_mat = faer::mat::from_column_major_slice_mut(a_col, n, n);
    cholesky_in_place(
        a_mat.as_mut(),
        LltRegularization::default(),
        Parallelism::None,
        stack,
        LltParams::default(),
    )
    .map_err(|_| MboostrsError::Solver("Cholesky factorisation failed".into()))?;
    let mut rhs = faer::mat::from_column_major_slice_mut(b_x, n, 1);
    solve_in_place_with_conj(
        a_mat.as_ref(),
        Conj::No,
        rhs.as_mut(),
        Parallelism::None,
        stack,
    );
    Ok(())
}

/// Exact CSR copy of a sparse matrix, retained so a product or a Gram block
/// can be reconstructed without recomputing it.
///
/// Extraction discovers support from exact zeros — including any fill
/// introduced by spectral centering — rather than assuming a band, and keeps
/// every nonzero, however small. The size thresholds select a kernel, not
/// entries: index/value storage and indirect loads only pay for sufficiently
/// sparse matrices, so small or dense inputs return `None` and their callers
/// keep the dense route.
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    /// Row offsets, length `nrows + 1`.
    pub offsets: Vec<usize>,
    /// Column index per stored entry.
    pub columns: Vec<usize>,
    /// Value per stored entry, in row-major order.
    pub values: Vec<f64>,
    /// Dense dimensions, kept so [`CsrMatrix::to_dense`] can rebuild the
    /// matrix even when the last rows are empty.
    pub dims: (usize, usize),
}

/// Smallest matrix (total entries) for which CSR storage is considered.
const CSR_MIN_ENTRIES: usize = 4096;
/// Largest nonzero fraction for which CSR storage is considered.
const CSR_MAX_FILL: usize = 4;

impl CsrMatrix {
    /// Extract an exact CSR copy of `matrix`, or `None` when it is too small
    /// or too dense for indirect access to pay.
    pub fn try_from_dense(matrix: ndarray::ArrayView2<f64>) -> Option<Self> {
        if matrix.len() < CSR_MIN_ENTRIES {
            return None;
        }
        let nonzeros = matrix.iter().filter(|&&value| value != 0.0).count();
        if nonzeros > matrix.len() / CSR_MAX_FILL {
            return None;
        }
        let mut rows = CsrMatrix {
            offsets: Vec::with_capacity(matrix.nrows() + 1),
            columns: Vec::with_capacity(nonzeros),
            values: Vec::with_capacity(nonzeros),
            dims: (matrix.nrows(), matrix.ncols()),
        };
        rows.offsets.push(0);
        for row in matrix.rows() {
            for (column, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    rows.columns.push(column);
                    rows.values.push(value);
                }
            }
            rows.offsets.push(rows.values.len());
        }
        Some(rows)
    }

    /// Materialize the matrix as a dense row-major `Array2`, the same layout
    /// a fresh dense accumulation would have produced.
    pub fn to_dense(&self) -> ndarray::Array2<f64> {
        let (nrows, ncols) = self.dims;
        let mut out = ndarray::Array2::<f64>::zeros((nrows, ncols));
        for row in 0..nrows {
            for entry in self.offsets[row]..self.offsets[row + 1] {
                out[[row, self.columns[entry]]] = self.values[entry];
            }
        }
        out
    }

    /// Heap bytes retained by the CSR arrays.
    pub fn byte_size(&self) -> usize {
        (self.offsets.capacity() + self.columns.capacity()) * std::mem::size_of::<usize>()
            + self.values.capacity() * std::mem::size_of::<f64>()
    }
}

/// Cached Cholesky factorization: stores the lower-triangular L factor
/// where `A = L·L^T`, as a flat column-major `Vec<f64>`. Avoids O(p³)
/// re-factorization per iteration; solving is only O(p²) forward-backward
/// substitution. Column-major layout means L(i,j) = data[j * p + i],
/// which is the same memory order faer uses internally.
///
/// The hot solve path (`solve_into_slice`) does not touch the dense factor:
/// it uses `band`, a packed LAPACK-style band with `band[k + j*(bw+1)] =
/// L(j+k, j)` for `k = 0..=bw`. The packed band is `(bw+1)*p` doubles
/// (L2-resident for the tensor-product smooths) instead of streaming the
/// dense `p*p` factor, and its columns are contiguous for vectorized axpy
/// substitution.
#[derive(Clone)]
pub struct CachedCholesky {
    l: Vec<f64>,
    p: usize,
    // Exact support of the computed factor; no tolerance-based truncation.
    half_bandwidth: usize,
    // Packed band of L; see the struct docs. Built once in `new`.
    band: Vec<f64>,
}

impl CachedCholesky {
    /// Compute the Cholesky factorization once and cache L.
    pub fn new(a: &ndarray::Array2<f64>) -> Result<Self, MboostrsError> {
        let p = a.nrows();
        if a.ncols() != p {
            return Err(MboostrsError::Solver(
                "CachedCholesky: matrix must be square".into(),
            ));
        }
        let faer_a = Mat::from_fn(p, p, |i, j| a[[i, j]]);
        let chol = Cholesky::try_new(faer_a.as_ref(), Side::Lower)
            .map_err(|_| MboostrsError::Solver("Cholesky factorisation failed".into()))?;
        let l_faer = chol.compute_l();
        // Copy L into flat column-major Vec<f64> for cache-friendly access.
        let mut l = vec![0.0f64; p * p];
        let mut half_bandwidth = 0;
        for j in 0..p {
            for i in j..p {
                let value = l_faer.read(i, j);
                l[j * p + i] = value;
                if value != 0.0 {
                    half_bandwidth = half_bandwidth.max(i - j);
                }
            }
        }
        // Pack the band for the hot solve: band[k + j*(bw+1)] = L(j+k, j).
        // Column j of L occupies one contiguous segment, so both the
        // forward axpy and the backward dot stream contiguously out of L2.
        let bw = half_bandwidth;
        let stride = bw + 1;
        let mut band = vec![0.0f64; stride * p];
        for j in 0..p {
            let max_k = bw.min(p - 1 - j);
            for k in 0..=max_k {
                band[k + j * stride] = l[j * p + (j + k)];
            }
        }
        Ok(Self {
            l,
            p,
            half_bandwidth,
            band,
        })
    }

    /// Read-only access to the cached lower-triangular `L` factor column-major
    /// flat buffer. `L(i,j)` = `l[j * p + i]`.
    pub fn l_slice(&self) -> &[f64] {
        &self.l
    }

    /// Whether this factorization is numerically rank-deficient — that is,
    /// whether it succeeded only because rounding left a pivot barely positive.
    ///
    /// For `A = L·Lᵀ` the pivots are `L(j,j)²`, so `min L(j,j) / max L(j,j)`
    /// squared is a lower-bound proxy for `1/κ(A)`. Flagging at
    /// `sqrt(f64::EPSILON)` on the `L` diagonal therefore draws the line at
    /// `κ ≈ 1/EPSILON ≈ 4.5e15`: the point where the smallest pivot has sunk to
    /// the double-precision resolution of the largest and the solve has no
    /// reliable digits left.
    ///
    /// This matters because a successful Cholesky is not by itself evidence that
    /// the system is solvable. `diag(1, 1e-20)` factorizes perfectly happily and
    /// then returns a coefficient of `1e20`. Whether a rank-deficient Gram
    /// crosses into "Cholesky fails" or merely "Cholesky succeeds with a tiny
    /// pivot" depends on rounding, and so on the target: a design with duplicate
    /// columns is exactly singular under one accumulation order and singular only
    /// to within an ULP under another, which is why this same matrix took the
    /// regularized route on aarch64 and the plain Cholesky route on the
    /// FMA-enabled local build.
    pub fn is_numerically_rank_deficient(&self) -> bool {
        self.diagonal_ratio() <= f64::EPSILON.sqrt()
    }

    /// `min L(j,j) / max L(j,j)`, the proxy for `1/sqrt(κ(A))` that
    /// [`Self::is_numerically_rank_deficient`] thresholds. Zero when the factor
    /// is degenerate. Callers that need a stricter margin than "any reliable
    /// digits at all" — a factorization whose *solutions* must stay accurate,
    /// not merely exist — compare this against their own threshold.
    pub fn diagonal_ratio(&self) -> f64 {
        let p = self.p;
        if p == 0 {
            return 1.0;
        }
        let mut min_diagonal = f64::INFINITY;
        let mut max_diagonal = 0.0f64;
        for j in 0..p {
            let value = self.l[j * p + j].abs();
            min_diagonal = min_diagonal.min(value);
            max_diagonal = max_diagonal.max(value);
        }
        if max_diagonal == 0.0 {
            return 0.0;
        }
        min_diagonal / max_diagonal
    }

    /// Matrix dimension `p` of the cached factorization.
    pub fn dim(&self) -> usize {
        self.p
    }

    /// Solve A·x = b using cached L factor. A = L·L^T, so L·L^T·x = b.
    /// Forward sub: L·y = b, Backward sub: L^T·x = y.
    /// Only O(p) conversion from ndarray + O(p²) triangular solve.
    pub fn solve(&self, b: &ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        let p = self.p;
        let mut x = vec![0.0f64; p];
        self.solve_into_slice(b.as_slice().expect("b must be contiguous"), &mut x);
        ndarray::Array1::from_vec(x)
    }

    /// Solve A·x = b writing the result into `x_out`.
    ///
    /// Zero-alloc hot-path variant used by the boosting engine.
    /// `b` and `x_out` must both have length `p`.
    ///
    /// Runs on the packed band (`band[k + j*(bw+1)] = L(j+k, j)`), which is
    /// L2-resident for the bandwidths the engine sees. The forward pass is a
    /// column axpy — the same flops in the same order as the dense form, so
    /// bit-identical. The backward pass unrolls its dot product over four
    /// independent accumulators: the single-accumulator form is latency-bound
    /// (~4 cycles per FMA with a loop-carried dependency), while the
    /// unrolled form keeps several FMAs in flight. This reassociates the dot
    /// product, so results can differ in the last ulp from the dense form;
    /// the engine's parity tests pin the accepted tolerance.
    pub fn solve_into_slice(&self, b: &[f64], x_out: &mut [f64]) {
        let p = self.p;
        debug_assert_eq!(b.len(), p);
        debug_assert_eq!(x_out.len(), p);
        let band = &self.band;
        let bw = self.half_bandwidth;
        let stride = bw + 1;
        x_out.copy_from_slice(b);
        // Forward: L·y = b. Column j of L is band[0..=bw, j], contiguous.
        for j in 0..p {
            let col = j * stride;
            let value = x_out[j] / band[col];
            x_out[j] = value;
            let k_max = bw.min(p - 1 - j);
            for k in 1..=k_max {
                x_out[j + k] -= band[k + col] * value;
            }
        }
        // Backward: Lᵀ·x = y. Row i of Lᵀ is column i of L, i.e. the
        // contiguous segment band[1..=bw, i]; dot it against x[i+1..].
        for i in (0..p).rev() {
            let col = i * stride;
            let k_max = bw.min(p - 1 - i);
            let mut s0 = 0.0f64;
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;
            let mut s3 = 0.0f64;
            let mut k = 1usize;
            let k_block = 1 + (k_max / 4) * 4;
            while k < k_block {
                s0 += band[k + col] * x_out[i + k];
                s1 += band[k + 1 + col] * x_out[i + k + 1];
                s2 += band[k + 2 + col] * x_out[i + k + 2];
                s3 += band[k + 3 + col] * x_out[i + k + 3];
                k += 4;
            }
            while k <= k_max {
                s0 += band[k + col] * x_out[i + k];
                k += 1;
            }
            x_out[i] = (x_out[i] - (s0 + s1 + s2 + s3)) / band[col];
        }
    }

    /// Solve A·X = B where B is (p, q), returning X as (p, q).
    ///
    /// Fused forward-backward substitution over all q columns in one pass,
    /// avoiding per-column allocation and dispatch overhead.
    pub fn solve_mat(&self, b: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        let p = self.p;
        let q = b.ncols();
        debug_assert_eq!(b.nrows(), p);
        let l = &self.l;
        let mut x = ndarray::Array2::zeros((p, q));
        // Forward: L·Y = B
        for j in 0..q {
            for i in 0..p {
                let mut sum = b[[i, j]];
                for k in 0..i {
                    sum -= l[k * p + i] * x[[k, j]];
                }
                x[[i, j]] = sum / l[i * p + i];
            }
        }
        // Backward: L^T·X = Y
        for j in 0..q {
            for i in (0..p).rev() {
                let mut sum = x[[i, j]];
                for k in (i + 1)..p {
                    sum -= l[i * p + k] * x[[k, j]];
                }
                x[[i, j]] = sum / l[i * p + i];
            }
        }
        x
    }
}

/// Solve using a cached Cholesky factorization — convenience wrapper.
pub fn cho_solve_cached(
    chol: &CachedCholesky,
    b: &ndarray::ArrayView1<f64>,
) -> ndarray::Array1<f64> {
    chol.solve(b)
}

// ─── Cached LU factorization (fallback for indefinite systems) ──────

/// Cached LU factorization with partial (row) pivoting.
///
/// Used as a fallback when Cholesky fails on indefinite gram matrices
/// (e.g. bmrf where K has negative eigenvalues). R's `solve()` uses
/// LU internally, so this matches R's behaviour.
///
/// Stores the original gram matrix plus a packed column-major LU factor and
/// row permutation. Keeping the numeric factors in plain buffers lets the hot
/// solve path use the caller's output slice as scratch, with no temporary
/// matrix or solver-workspace allocation.
#[derive(Clone)]
pub struct CachedLu {
    gram: ndarray::Array2<f64>,
    factors: Vec<f64>,
    row_perm: Vec<usize>,
}

impl CachedLu {
    /// Dimension of the cached matrix.
    pub fn dim(&self) -> usize {
        self.gram.nrows()
    }

    /// Compute LU factorization with partial pivoting and cache the result.
    pub fn new(a: &ndarray::Array2<f64>) -> Result<Self, MboostrsError> {
        let p = a.nrows();
        if a.ncols() != p {
            return Err(MboostrsError::Solver(
                "CachedLu: matrix must be square".into(),
            ));
        }
        let faer_a = Mat::from_fn(p, p, |i, j| a[[i, j]]);
        let lu = faer_a.partial_piv_lu();
        let l = lu.compute_l();
        let u = lu.compute_u();
        let mut factors = vec![0.0; p * p];
        for j in 0..p {
            for i in 0..p {
                factors[j * p + i] = if i > j { l.read(i, j) } else { u.read(i, j) };
            }
        }
        let (row_perm, _) = lu.row_permutation().arrays();
        Ok(Self {
            gram: a.clone(),
            factors,
            row_perm: row_perm.to_vec(),
        })
    }

    /// Solve A·x = b using the cached LU factorization.
    /// Delegates to faer's `SpSolver::solve()`.
    pub fn solve(&self, b: &ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        let p = b.len();
        let mut x = vec![0.0f64; p];
        self.solve_into_slice(b.as_slice().expect("b must be contiguous"), &mut x);
        ndarray::Array1::from_vec(x)
    }

    /// Solve A·x = b writing the result into `x_out`.
    ///
    /// Zero-allocation hot-path variant. `b` and `x_out` must both have length
    /// `p`. The packed factors satisfy `P·A = L·U`; `x_out` first holds `P·b`,
    /// then the forward and backward substitutions in place.
    pub fn solve_into_slice(&self, b: &[f64], x_out: &mut [f64]) {
        let p = self.dim();
        debug_assert_eq!(b.len(), p);
        debug_assert_eq!(x_out.len(), p);

        // Apply the factorization's row permutation: x_out <- P·b.
        for i in 0..p {
            x_out[i] = b[self.row_perm[i]];
        }
        // Unit-lower solve: L·y = P·b.
        for i in 0..p {
            let mut sum = x_out[i];
            for j in 0..i {
                sum -= self.factors[j * p + i] * x_out[j];
            }
            x_out[i] = sum;
        }
        // Upper solve: U·x = y.
        for i in (0..p).rev() {
            let mut sum = x_out[i];
            for j in (i + 1)..p {
                sum -= self.factors[j * p + i] * x_out[j];
            }
            x_out[i] = sum / self.factors[i * p + i];
        }
    }

    /// Solve A·x = B for a matrix RHS using the cached LU factorization.
    pub fn solve_mat(&self, b: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        let p = b.nrows();
        let q = b.ncols();
        let mut x = ndarray::Array2::zeros((p, q));
        for column in 0..q {
            for i in 0..p {
                x[[i, column]] = b[[self.row_perm[i], column]];
            }
            for i in 0..p {
                let mut sum = x[[i, column]];
                for j in 0..i {
                    sum -= self.factors[j * p + i] * x[[j, column]];
                }
                x[[i, column]] = sum;
            }
            for i in (0..p).rev() {
                let mut sum = x[[i, column]];
                for j in (i + 1)..p {
                    sum -= self.factors[j * p + i] * x[[j, column]];
                }
                x[[i, column]] = sum / self.factors[i * p + i];
            }
        }
        x
    }

    /// Verify that the cached factorization behaves as an inverse of its
    /// source matrix. LU construction alone does not reject singular systems.
    fn has_acceptable_inverse_residual(&self) -> bool {
        let p = self.dim();
        let inverse = self.solve_mat(&ndarray::Array2::<f64>::eye(p));
        if inverse.iter().any(|value| !value.is_finite()) {
            return false;
        }
        let product = self.gram.dot(&inverse);
        if product.iter().any(|value| !value.is_finite()) {
            return false;
        }
        let mut residual = 0.0f64;
        for i in 0..p {
            for j in 0..p {
                let expected = if i == j { 1.0 } else { 0.0 };
                residual = residual.max((product[[i, j]] - expected).abs());
            }
        }
        let a_scale = self
            .gram
            .iter()
            .fold(0.0f64, |scale, value| scale.max(value.abs()));
        let inverse_scale = inverse
            .iter()
            .fold(0.0f64, |scale, value| scale.max(value.abs()));
        residual <= 1e-10 * (a_scale * inverse_scale).max(1.0) * p.max(1) as f64
    }
}

// ─── GramSolver: Cholesky with LU fallback ──────────────────────────

/// A cached solver for the gram system `(X'WX + λK)·x = rhs`.
///
/// Tries Cholesky first (O(p²) solve, fast for positive-definite systems).
/// Falls back to LU with partial pivoting when the gram matrix is indefinite
/// (e.g. bmrf where K has negative eigenvalues).
///
/// This matches R's `solve(XtX)` behaviour which uses LU internally.
#[derive(Clone)]
pub enum GramSolver {
    /// Solve via a cached Cholesky factorization.
    Cholesky(CachedCholesky),
    /// Solve via LU with partial pivoting (indefinite Gram matrices).
    Lu(CachedLu),
    /// Solve via Cholesky after adding `jitter` to the diagonal.
    RegularizedCholesky {
        /// The Cholesky factor of `gram + jitter·I`.
        factor: CachedCholesky,
        /// Amount added to the diagonal before factorization.
        jitter: f64,
    },
}

impl GramSolver {
    /// Fallibly create and verify a solver for the given gram matrix.
    ///
    /// Tries Cholesky first, then a residual-verified LU factorization for an
    /// invertible indefinite matrix. Singular finite systems receive
    /// progressively larger diagonal jitter and are returned as an explicitly
    /// tagged regularized Cholesky solver. Invalid matrices return a typed
    /// error and are never converted into a solver that silently emits NaN.
    pub fn try_new(gram: &ndarray::Array2<f64>) -> Result<Self, MboostrsError> {
        crate::report::measure_factorization(|| Self::try_new_unmeasured(gram))
    }

    fn try_new_unmeasured(gram: &ndarray::Array2<f64>) -> Result<Self, MboostrsError> {
        let p = gram.nrows();
        if p == 0 || gram.ncols() != p {
            return Err(MboostrsError::Solver(
                "GramSolver: matrix must be non-empty and square".into(),
            ));
        }
        if gram.iter().any(|value| !value.is_finite()) {
            return Err(MboostrsError::Solver(
                "GramSolver: matrix contains NaN or infinite values".into(),
            ));
        }
        // A Cholesky that merely *succeeds* is not enough: a numerically
        // rank-deficient Gram can factorize with a barely-positive pivot and then
        // return coefficients of order 1/pivot. Such a matrix goes straight to
        // regularization and deliberately skips the LU branch below, whose
        // residual check is scaled by the inverse's own magnitude and so accepts
        // a solve that is relatively accurate but absolutely enormous. LU remains
        // the route for matrices that are indefinite rather than near-singular
        // (bmrf, where K contributes negative eigenvalues) — those fail Cholesky
        // outright and never reach this branch.
        let mut cholesky_was_rank_deficient = false;
        match CachedCholesky::new(gram) {
            Ok(chol) if !chol.is_numerically_rank_deficient() => {
                return Ok(GramSolver::Cholesky(chol));
            }
            Ok(_) => cholesky_was_rank_deficient = true,
            Err(_) => {}
        }
        if !cholesky_was_rank_deficient {
            let lu = CachedLu::new(gram)?;
            if lu.has_acceptable_inverse_residual() {
                return Ok(GramSolver::Lu(lu));
            }
        }

        let scale = gram
            .iter()
            .fold(0.0f64, |current, value| current.max(value.abs()))
            .max(1.0);
        let mut jitter = 1e-12 * scale;
        for _ in 0..14 {
            let mut regularized = gram.clone();
            for i in 0..p {
                regularized[[i, i]] += jitter;
            }
            // Hold the regularized factor to the same standard as the direct one:
            // enough jitter to factorize is not enough jitter to be solvable, and
            // accepting the first success would hand back exactly the tiny-pivot
            // solver this route exists to avoid.
            if let Ok(factor) = CachedCholesky::new(&regularized) {
                if !factor.is_numerically_rank_deficient() {
                    return Ok(GramSolver::RegularizedCholesky { factor, jitter });
                }
            }
            jitter *= 10.0;
        }
        Err(MboostrsError::Solver(
            "GramSolver: matrix is singular and diagonal regularization failed".into(),
        ))
    }

    /// Compatibility constructor that panics on invalid matrices.
    /// Prefer [`GramSolver::try_new`] in new fallible code.
    pub fn new(gram: &ndarray::Array2<f64>) -> Self {
        Self::try_new(gram).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Solve A·x = b using the cached factorization.
    pub fn solve(&self, b: &ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        match self {
            GramSolver::Cholesky(chol) => chol.solve(b),
            GramSolver::Lu(lu) => lu.solve(b),
            GramSolver::RegularizedCholesky { factor, .. } => factor.solve(b),
        }
    }

    /// Solve `A·x = b` writing the result into `x_out`.
    pub fn solve_into_slice(&self, b: &[f64], x_out: &mut [f64]) {
        match self {
            GramSolver::Cholesky(chol) => chol.solve_into_slice(b, x_out),
            GramSolver::Lu(lu) => lu.solve_into_slice(b, x_out),
            GramSolver::RegularizedCholesky { factor, .. } => factor.solve_into_slice(b, x_out),
        }
    }

    /// Solve `A·X = B` for a matrix of right-hand sides.
    pub fn solve_mat(&self, b: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        match self {
            GramSolver::Cholesky(chol) => chol.solve_mat(b),
            GramSolver::Lu(lu) => lu.solve_mat(b),
            GramSolver::RegularizedCholesky { factor, .. } => factor.solve_mat(b),
        }
    }

    /// Return the cached lower-triangular Cholesky factor as `(L_packed, p)` when
    /// the solver is (regularized) Cholesky-based, `None` for LU.
    pub fn cholesky_l_slice(&self) -> Option<(&[f64], usize)> {
        match self {
            GramSolver::Cholesky(chol) => Some((chol.l_slice(), chol.dim())),
            GramSolver::Lu(_) => None,
            GramSolver::RegularizedCholesky { factor, .. } => {
                Some((factor.l_slice(), factor.dim()))
            }
        }
    }

    /// Whether this prepared system uses the preferred Cholesky route.
    pub fn is_cholesky(&self) -> bool {
        matches!(self, GramSolver::Cholesky(_))
    }

    /// Diagonal jitter used to regularize a singular system, if any.
    pub fn regularization_jitter(&self) -> Option<f64> {
        match self {
            GramSolver::RegularizedCholesky { jitter, .. } => Some(*jitter),
            _ => None,
        }
    }

    /// Bytes retained by the cached numerical factorization.
    ///
    /// Cholesky stores one dense factor. LU retains the source gram matrix and
    /// a dense factorization, so it accounts for two `p x p` buffers.
    pub fn cached_factor_bytes(&self) -> usize {
        let p = match self {
            GramSolver::Cholesky(chol) => chol.dim(),
            GramSolver::Lu(lu) => lu.dim(),
            GramSolver::RegularizedCholesky { factor, .. } => factor.dim(),
        };
        let buffers = if matches!(self, GramSolver::Lu(_)) {
            2
        } else {
            1
        };
        buffers * p * p * std::mem::size_of::<f64>()
    }

    /// Compute the full inverse matrix S⁻¹.
    ///
    /// Solves S·X = I for X (p right-hand sides). O(p³) for a single
    /// call, but p ≤ ~10 in practice so this is negligible.
    pub fn inverse(&self) -> ndarray::Array2<f64> {
        let p = match self {
            GramSolver::Cholesky(chol) => chol.dim(),
            GramSolver::Lu(lu) => lu.dim(),
            GramSolver::RegularizedCholesky { factor, .. } => factor.dim(),
        };
        let eye = ndarray::Array2::<f64>::eye(p);
        self.solve_mat(&eye)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn csr_round_trip_is_exact_and_thresholds_hold() {
        let mut matrix = Array2::<f64>::zeros((80, 90));
        for i in 0..79 {
            matrix[[i, i]] = 2.0;
            matrix[[i, i + 3]] = -0.25;
        }
        // A completely empty last row and a tiny remote entry: the offsets
        // must keep the empty row, and no numerical sparsification may drop
        // the small entry.
        matrix[[0, 89]] = 1e-100;
        let csr = CsrMatrix::try_from_dense(matrix.view()).unwrap();
        // Exact round trip, including the 1e-100 entry.
        assert_eq!(csr.to_dense(), matrix);
        // The empty row keeps a zero-width span.
        assert_eq!(
            csr.offsets[csr.offsets.len() - 1],
            csr.offsets[csr.offsets.len() - 2]
        );
        assert!(csr.byte_size() > 0);
        // Dense and small matrices keep the dense route.
        let dense = Array2::from_elem((80, 90), 1.0);
        assert!(CsrMatrix::try_from_dense(dense.view()).is_none());
        assert!(CsrMatrix::try_from_dense(Array2::eye(24).view()).is_none());
    }

    #[test]
    fn banded_solve_matches_dense_substitution_and_matrix_rhs() {
        for bandwidth in [0, 1, 7, 31] {
            let p = 32;
            let a = Array2::from_shape_fn((p, p), |(i, j)| {
                if i == j {
                    40.0
                } else if i.abs_diff(j) <= bandwidth {
                    ((i + j + 1) as f64).sin()
                } else {
                    0.0
                }
            });
            let factor = CachedCholesky::new(&a).unwrap();
            assert_eq!(factor.half_bandwidth, bandwidth);
            let rhs = Array2::from_shape_fn((p, 3), |(i, j)| ((i * 3 + j) as f64).cos());
            // solve_mat deliberately retains the dense reference algorithm.
            let expected = factor.solve_mat(&rhs);
            for j in 0..rhs.ncols() {
                let b = rhs.column(j).to_owned();
                let actual = factor.solve(&b.view());
                for i in 0..p {
                    assert!((actual[i] - expected[[i, j]]).abs() < 1e-14);
                }
                assert!((&a.dot(&actual) - &b).iter().all(|v| v.abs() < 1e-12));
            }
        }
    }

    #[test]
    fn factor_band_keeps_tiny_off_band_entries() {
        let mut a = Array2::eye(8);
        a[[0, 7]] = 1e-100;
        a[[7, 0]] = 1e-100;
        let factor = CachedCholesky::new(&a).unwrap();
        assert_eq!(factor.half_bandwidth, 7);
        let mut b = ndarray::Array1::zeros(8);
        b[0] = 1.0;
        let x = factor.solve(&b.view());
        assert_eq!(x[7], -1e-100);
    }

    #[test]
    fn cho_solve_identity() {
        let a = Array2::eye(3);
        let b = array![1.0, 2.0, 3.0];
        let x = cho_solve(&a, &b.view()).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn cho_solve_2x2() {
        // A = [[4, 2], [2, 3]], b = [1, 2] → x = [−0.125, 0.75]
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let b = array![1.0, 2.0];
        let x = cho_solve(&a, &b.view()).unwrap();
        assert!((x[0] - (-0.125)).abs() < 1e-12);
        assert!((x[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn cho_solve_non_pd_errors() {
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let b = array![1.0, 2.0];
        assert!(cho_solve(&a, &b.view()).is_err());
    }

    #[test]
    fn singular_system_is_explicitly_regularized() {
        let a = array![[1.0, 1.0], [1.0, 1.0]];
        let b = array![2.0, 2.0];
        let solver = GramSolver::try_new(&a).unwrap();
        assert!(matches!(solver, GramSolver::RegularizedCholesky { .. }));
        let jitter = solver.regularization_jitter().unwrap();
        assert!(jitter > 0.0 && jitter.is_finite());
        let x = solver.solve(&b.view());
        assert!(x[0].is_finite());
        assert!(x[1].is_finite());
        let mut regularized = a;
        regularized[[0, 0]] += jitter;
        regularized[[1, 1]] += jitter;
        let residual = &regularized.dot(&x) - &b;
        assert!(residual.iter().all(|value| value.abs() < 1e-8));
    }

    #[test]
    fn solve_bmrf_like_positive_definite_gram() {
        // Simulate a bmrf-like gram: X'WX + λK where K has neg eigenvalues
        // X'WX = diag([40, 40, 40, 40, 40])  (n=200, 5 levels, ~40 each)
        // K = chain adjacency (neg eigenvalues)
        // λ = 1.0
        let mut k = Array2::zeros((5, 5));
        for i in 0..4 {
            k[[i, i + 1]] = 1.0;
            k[[i + 1, i]] = 1.0;
        }
        let mut gram = Array2::eye(5) * 40.0;
        for i in 0..5 {
            for j in 0..5 {
                gram[[i, j]] += k[[i, j]];
            }
        }
        // K has negative eigenvalues, but X'WX is dominant, so the combined
        // matrix remains positive definite and should be solved accurately.
        let b = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let solver = GramSolver::new(&gram);
        let x = solver.solve(&b.view());
        // Verify: gram * x ≈ b
        let ax = gram.dot(&x);
        for i in 0..5 {
            assert!((ax[i] - b[i]).abs() < 1e-10, "residual at {}", i);
        }
    }

    #[test]
    fn gram_solver_prefers_cholesky() {
        // For a PD matrix, GramSolver should use Cholesky (faster)
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let solver = GramSolver::new(&a);
        assert!(matches!(solver, GramSolver::Cholesky(_)));
        assert!(solver.is_cholesky());
        assert_eq!(solver.cached_factor_bytes(), 2 * 2 * 8);
    }

    #[test]
    fn gram_solver_reports_lu_cache_for_indefinite_system() {
        let a = array![[0.0, 1.0], [1.0, 0.0]];
        let solver = GramSolver::new(&a);
        assert!(matches!(solver, GramSolver::Lu(_)));
        assert!(!solver.is_cholesky());
        let b = array![1.0, -2.0];
        let residual = &a.dot(&solver.solve(&b.view())) - &b;
        assert!(residual.iter().all(|value| value.abs() < 1e-10));
        assert_eq!(solver.cached_factor_bytes(), 2 * 2 * 2 * 8);
    }

    #[test]
    fn cached_lu_repeated_slice_and_matrix_solves_match() {
        // Zero diagonal forces row pivoting; the matrix is invertible and
        // indefinite, so this directly exercises the packed LU solve path.
        let a = array![[0.0, 2.0, 1.0], [2.0, -1.0, 0.0], [1.0, 0.0, 3.0]];
        let lu = CachedLu::new(&a).unwrap();
        let rhs = [array![1.0, -2.0, 0.5], array![-4.0, 3.0, 2.0]];
        let mut x = [f64::NAN; 3];
        for b in &rhs {
            lu.solve_into_slice(b.as_slice().unwrap(), &mut x);
            let residual = &a.dot(&ndarray::ArrayView1::from(&x)) - b;
            assert!(residual.iter().all(|value| value.abs() < 1e-11));
        }

        let b_mat = array![[1.0, -4.0], [-2.0, 3.0], [0.5, 2.0]];
        let x_mat = lu.solve_mat(&b_mat);
        let residual = a.dot(&x_mat) - b_mat;
        assert!(residual.iter().all(|value| value.abs() < 1e-11));
    }

    #[test]
    fn gram_solver_try_new_rejects_invalid_matrices() {
        assert!(GramSolver::try_new(&Array2::zeros((2, 3))).is_err());
        assert!(GramSolver::try_new(&Array2::zeros((0, 0))).is_err());
        assert!(GramSolver::try_new(&array![[1.0, f64::NAN], [0.0, 1.0]]).is_err());
        assert!(GramSolver::try_new(&array![[1.0, f64::INFINITY], [0.0, 1.0]]).is_err());
    }

    #[test]
    fn near_singular_gram_is_regularized_even_though_cholesky_succeeds() {
        // diag(1, 1e-20) is positive definite, so Cholesky factorizes it without
        // complaint — and then solves it to x[1] = 1e20. Condition number is 1e20:
        // there is not a single reliable digit here. The tiny pivot must be caught
        // and the system regularized rather than solved.
        let a = array![[1.0, 0.0], [0.0, 1e-20]];
        assert!(
            CachedCholesky::new(&a).is_ok(),
            "premise: plain Cholesky accepts this matrix"
        );

        let solver = GramSolver::try_new(&a).unwrap();
        assert!(
            matches!(solver, GramSolver::RegularizedCholesky { .. }),
            "near-singular gram must not be handed to plain Cholesky or LU"
        );

        let x = solver.solve(&array![1.0, 1.0].view());
        assert!(x.iter().all(|value| value.is_finite()));
        // The regularized answer is bounded by 1/jitter, not by 1/1e-20.
        let jitter = solver.regularization_jitter().unwrap();
        assert!(x.iter().all(|value| value.abs() <= 1.0 / jitter));
        assert!(
            x[1] < 1e15,
            "expected the tiny direction to be damped, got {:e}",
            x[1]
        );
    }

    #[test]
    fn rank_deficiency_flag_tracks_the_pivot_ratio_not_mere_factorizability() {
        // Well conditioned: not flagged, and still routed to plain Cholesky.
        let healthy = array![[4.0, 2.0], [2.0, 3.0]];
        assert!(!CachedCholesky::new(&healthy)
            .unwrap()
            .is_numerically_rank_deficient());
        assert!(matches!(
            GramSolver::try_new(&healthy).unwrap(),
            GramSolver::Cholesky(_)
        ));

        // Ill conditioned but still usable (κ ≈ 1e10) must stay on the fast path;
        // the guard is for numerically dead systems, not merely awkward ones.
        let awkward = array![[1.0, 0.0], [0.0, 1e-10]];
        assert!(!CachedCholesky::new(&awkward)
            .unwrap()
            .is_numerically_rank_deficient());
        assert!(matches!(
            GramSolver::try_new(&awkward).unwrap(),
            GramSolver::Cholesky(_)
        ));

        // Past the sqrt(EPSILON) line on the L diagonal.
        let dead = array![[1.0, 0.0], [0.0, 1e-20]];
        assert!(CachedCholesky::new(&dead)
            .unwrap()
            .is_numerically_rank_deficient());
    }

    #[test]
    fn indefinite_gram_still_routes_to_lu_not_regularization() {
        // The near-singularity guard must not steal the bmrf case: an indefinite
        // but well-conditioned matrix fails Cholesky outright and belongs to LU.
        let a = array![[0.0, 1.0], [1.0, 0.0]];
        assert!(CachedCholesky::new(&a).is_err());
        assert!(matches!(
            GramSolver::try_new(&a).unwrap(),
            GramSolver::Lu(_)
        ));
    }

    #[test]
    fn gram_solver_zero_matrix_is_regularized_and_finite() {
        let a = Array2::zeros((3, 3));
        let solver = GramSolver::try_new(&a).unwrap();
        assert!(matches!(solver, GramSolver::RegularizedCholesky { .. }));
        assert!(solver.regularization_jitter().unwrap() > 0.0);
        let b = array![1.0, 2.0, 3.0];
        let x = solver.solve(&b.view());
        assert!(x.iter().all(|value| value.is_finite()));
    }
}
