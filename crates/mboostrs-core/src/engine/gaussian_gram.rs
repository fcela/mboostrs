#![allow(unused_imports)]
use super::*;

/// Gaussian-only covariance/Gram-update fast path (glmnet-style).
///
/// For squared-error loss the residual updates linearly with the fitted
/// increment (`u ← u − ν·X_w·β_w`), so each learner's gradient projection
/// `rhsⱼ = Xⱼ'W u` can be maintained by a cheap `pⱼ×p_w` cross-Gram update
/// instead of re-projecting against the length-n residual every iteration.
/// Per-iteration cost drops from O(L·p·n) to O(L·p²); the length-n vectors
/// are touched only at setup (build cross-Grams + initial projections) and
/// once at the end to reconstruct the fitted values. Selection-only callers
/// skip coefficient/risk retention and final fitted-value reconstruction.
///
/// Returns `None` (caller falls back to the general path) when there is no
/// block learner, a learner is not one of scalar `bols`, block `bols`, or
/// `bmono`, or a candidate solve becomes non-finite. There is deliberately no
/// width-versus-`mstop` crossover guard: for realistic boosting paths the
/// O(n·P²) setup is amortized by avoiding O(n) projection work per candidate
/// and iteration.
///
/// Squared-error semantics are assumed (offset = weighted mean, risk = Σw·u²);
/// the caller must only invoke this when `family.supports_gram_update()`.
/// `Bⱼ'W B_w`, accumulated over blocks of rows so that only a block of the
/// weighted design is ever formed.
pub(super) fn weighted_cross_gram(
    left: ndarray::ArrayView2<f64>,
    weights: ArrayView1<f64>,
    right: ndarray::ArrayView2<f64>,
) -> Array2<f64> {
    use ndarray::s;
    const ROWS: usize = 4096;
    let n = left.nrows();
    let mut acc = Array2::<f64>::zeros((left.ncols(), right.ncols()));
    let mut scaled = Array2::<f64>::zeros((ROWS.min(n.max(1)), left.ncols()));
    let mut start = 0;
    while start < n {
        let end = (start + ROWS).min(n);
        let mut block = scaled.slice_mut(s![..end - start, ..]);
        block.assign(&left.slice(s![start..end, ..]));
        block *= &weights.slice(s![start..end]).insert_axis(ndarray::Axis(1));
        ndarray::linalg::general_mat_mul(
            1.0,
            &block.t(),
            &right.slice(s![start..end, ..]),
            1.0,
            &mut acc,
        );
        start = end;
    }
    acc
}

/// A repeatedly used matrix-vector product. Large sparse matrices retain an
/// exact CSR cache; small or dense matrices keep ndarray's dense kernel. This
/// discovers support from exact zeros, including any fill introduced by
/// spectral centering, rather than assuming every spatial basis is banded.
pub(super) struct GramMatVec<'a> {
    pub(super) matrix: &'a Array2<f64>,
    pub(super) sparse: Option<crate::linalg::CsrMatrix>,
}

impl<'a> GramMatVec<'a> {
    pub(super) fn new(matrix: &'a Array2<f64>) -> Self {
        Self {
            matrix,
            sparse: crate::linalg::CsrMatrix::try_from_dense(matrix.view()),
        }
    }

    pub(super) fn mul_into(&self, rhs: &Array1<f64>, mut out: ndarray::ArrayViewMut1<f64>) {
        debug_assert_eq!(rhs.len(), self.matrix.ncols());
        debug_assert_eq!(out.len(), self.matrix.nrows());
        if let Some(rows) = &self.sparse {
            for row in 0..out.len() {
                let mut sum = 0.0;
                for entry in rows.offsets[row]..rows.offsets[row + 1] {
                    sum += rows.values[entry] * rhs[rows.columns[entry]];
                }
                out[row] = sum;
            }
        } else {
            ndarray::linalg::general_mat_vec_mul(1.0, self.matrix, rhs, 0.0, &mut out);
        }
    }

    fn cache_bytes(&self) -> usize {
        self.sparse
            .as_ref()
            .map_or(0, crate::linalg::CsrMatrix::byte_size)
    }
}

pub(crate) fn try_fit_gaussian_gram(
    learners: &mut [PreparedLearner],
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    control: &BoostControl,
    offset_override: Option<f64>,
    collect_report: bool,
    store_coefs: bool,
    store_risk: bool,
) -> Option<BoostingPath> {
    use crate::learners::PreparedLearner as PL;
    let timing_enabled = collect_report && control.instrumentation != InstrumentationLevel::Off;
    let gram_started = timing_enabled.then(Instant::now);
    let n = y.len(); // Gaussian gram path only called for Gaussian family (n == y.len())

    enum Solv<'a> {
        Scalar(f64),
        Block(&'a crate::linalg::GramSolver),
        // BmonoBbs: constrained solve, handled specially in the selection
        // loop below (never through `solve_into`/`Pen::quad` — a constrained
        // beta does not satisfy the ridge stationarity those rely on).
        Mono {
            solver_a: &'a crate::linalg::GramSolver,
            penalised_gram: &'a Array2<f64>,
            d_mat: &'a Array2<f64>,
            kind: BmonoSolver,
            lambda2: f64,
            niter: usize,
        },
    }
    impl Solv<'_> {
        /// Solve into a caller-owned buffer (zero-alloc hot path). Not called
        /// for `Mono` — see the selection loop.
        fn solve_into(&self, rhs: &Array1<f64>, out: &mut Array1<f64>) {
            match self {
                Solv::Scalar(d) => out[0] = if *d > 0.0 { rhs[0] / d } else { 0.0 },
                Solv::Block(s) => s.solve_into_slice(
                    rhs.as_slice().expect("rhs contiguous"),
                    out.as_slice_mut().expect("beta buffer contiguous"),
                ),
                Solv::Mono { .. } => unreachable!("Mono handled specially in the selection loop"),
            }
        }
    }
    enum Pen<'a> {
        Scalar(f64),
        Block(GramMatVec<'a>),
        // BmonoBbs never goes through `quad` — the selection loop computes
        // its SSE via the full-form identity directly. Kept so `pens` stays
        // one Vec<Pen> indexed in parallel with `solvers`/`designs`.
        Mono,
    }
    impl Pen<'_> {
        /// β'Pβ — the penalty term of the residual-SSE identity.
        fn quad(&self, beta: &Array1<f64>, scratch: &mut Array1<f64>) -> f64 {
            match self {
                Pen::Scalar(lam) => lam * beta[0] * beta[0],
                Pen::Block(p) => {
                    let mut out = scratch.slice_mut(ndarray::s![..beta.len()]);
                    p.mul_into(beta, out.view_mut());
                    beta.dot(&out)
                }
                Pen::Mono => unreachable!("Mono handled specially in the selection loop"),
            }
        }
    }

    // Cheap pre-scan (no cloning): confirm all-linear, ≥1 block learner, and
    // the crossover guard, before paying for any extraction. This keeps the
    // pure-scalar glmboost path (which falls through to the packed path) free
    // of overhead.
    let mstop = control.mstop as usize;
    let mut total_p = 0usize;
    let mut has_block = false;
    for lr in learners.iter() {
        match lr {
            PL::ScalarBols { .. } => total_p += 1,
            PL::BlockBols { design, .. } => {
                has_block = true;
                total_p += design.ncols();
            }
            PL::BmonoBbs { design, .. } => {
                has_block = true;
                total_p += design.ncols();
            }
            // A tree is fitted to an explicit residual kept beside the
            // projections; see `Candidate::Tree`.
            PL::Btree { .. } => {}
            _ => return None, // KroneckerBols and the rest are not handled here
        }
    }
    // Pure-scalar models keep the packed-scalar path. Gate the cross-Gram
    // setup on whether the per-iteration savings amortise the O(L²·p²·n)
    // setup cost. The general path costs O(L·p·n) per iteration; the Gram
    // path costs O(L·p²) per iteration with an O(n·P²) setup (P = Σpⱼ).
    // Break-even: n·P² / (L·p·(n − p̄)) ≈ P² / (L·p̄·(1 − p̄/n)) iterations,
    // simplified to P / L iterations when p̄ ≪ n. Conservative: always
    // try Gram when has_block, bail only for pure-scalar or zero-p models.
    if !has_block || total_p == 0 {
        return None;
    }

    // Split the learners: the linear ones are borrowed for their designs, the
    // trees mutably, since fitting one updates its internal buffers. The
    // candidate order is the learners' order, so ties still go to the
    // lowest index.
    enum Candidate {
        Linear(usize),
        Tree(usize),
    }
    let mut candidates: Vec<Candidate> = Vec::with_capacity(learners.len());
    let mut linear_learners: Vec<&PreparedLearner> = Vec::new();
    let mut trees: Vec<&mut PreparedLearner> = Vec::new();
    for lr in learners.iter_mut() {
        if matches!(lr, PL::Btree { .. }) {
            candidates.push(Candidate::Tree(trees.len()));
            trees.push(lr);
        } else {
            candidates.push(Candidate::Linear(linear_learners.len()));
            linear_learners.push(lr);
        }
    }
    let l = linear_learners.len();
    let has_trees = !trees.is_empty();

    // Extract linear pieces. The designs are borrowed from the learners, and
    // each learner's weights stand in for a stored `Bⱼ'W`: the cross-Grams
    // are accumulated over row blocks (`weighted_cross_gram`), so this route
    // adds no n-row copy of any design.
    let fit_weights = weights.to_owned();
    let mut designs: Vec<ndarray::ArrayView2<f64>> = Vec::with_capacity(l); // (n, pⱼ)
    let mut row_weights: Vec<ArrayView1<f64>> = Vec::with_capacity(l);
    let mut solvers: Vec<Solv> = Vec::with_capacity(l);
    let mut pens: Vec<Pen> = Vec::with_capacity(l);
    // Preparation's unpenalized X'WX for block learners that computed one
    // (the tensor spatial learner). The route's self-Gram block is the
    // largest single setup product at tensor sizes; reusing the stash removes
    // it from the critical path (#276). The stash is an exact CSR copy (the
    // spatial Gram is banded, so a tenth of the dense bytes), cleared by
    // weight-refitting, so a present stash always matches the learner's
    // weights.
    let mut self_grams: Vec<Option<&crate::linalg::CsrMatrix>> = Vec::with_capacity(l);
    for lr in linear_learners.iter().copied() {
        match lr {
            PL::ScalarBols {
                design,
                weights: learner_weights,
                scalar_denom,
                penalty,
                ..
            } => {
                designs.push(design.view().insert_axis(ndarray::Axis(1))); // (n,1)
                row_weights.push(learner_weights.view());
                solvers.push(Solv::Scalar(*scalar_denom));
                pens.push(Pen::Scalar(*penalty));
                self_grams.push(None);
            }
            PL::BlockBols {
                design,
                weights: learner_weights,
                solver,
                penalty,
                weighted_gram,
                ..
            } => {
                designs.push(design.view());
                row_weights.push(learner_weights.view());
                solvers.push(Solv::Block(solver));
                pens.push(Pen::Block(GramMatVec::new(penalty)));
                self_grams.push(weighted_gram.as_ref());
            }
            PL::BmonoBbs {
                design,
                solver_a,
                penalised_gram,
                d_mat,
                solver,
                lambda2,
                niter,
                ..
            } => {
                designs.push(design.view());
                // BmonoBbs keeps no weight vector; it was prepared with the
                // fit's weights, which its `X'W` was built from.
                row_weights.push(fit_weights.view());
                solvers.push(Solv::Mono {
                    solver_a,
                    penalised_gram,
                    d_mat,
                    kind: *solver,
                    lambda2: *lambda2,
                    niter: *niter,
                });
                pens.push(Pen::Mono);
                self_grams.push(None);
            }
            _ => unreachable!("non-linear learner rejected in pre-scan"),
        }
    }

    // Offset (Gaussian = weighted mean) and initial residual u₀ = y − offset.
    let sw: f64 = weights.sum();
    let offset =
        offset_override.unwrap_or_else(|| if sw > 0.0 { y.dot(&weights) / sw } else { 0.0 });
    let u0: Array1<f64> = &y.to_owned() - offset;
    // The explicit residual, kept only when a tree competes: a tree is
    // fitted to it, and it cannot be recovered from the projections.
    let mut u: Array1<f64> = if has_trees {
        u0.clone()
    } else {
        Array1::zeros(0)
    };
    let mut wu2: f64 = weights
        .iter()
        .zip(u0.iter())
        .map(|(&w, &u)| w * u * u)
        .sum();

    // Cross-Grams G[j][w] = Bⱼ'W·B_w (pⱼ × p_w); initial projections rhsⱼ.
    // Compute only the upper triangle (j ≤ w) and fill the lower triangle by
    // transposition: G[j][w] = G[w][j]ᵀ for j > w. For J ≥ 4 learners the
    // upper-triangle matmuls are independent and CPU-bound, so dispatch them
    // via rayon.
    // Exact-duplicate designs (#301): when two learners' designs and row
    // weights are bit-identical, every Gram block in the duplicate's
    // column equals the matching block in the source's column, so the
    // pair's cross block and the duplicate's diagonal are copies of
    // those blocks rather than fresh O(n·p²) matmuls. The
    // copies are exact — the same values in the same standard layout the
    // matmul produces — so duplicate learners keep the bit-identical blocks
    // the R-parity tie contract requires. The check is O(n·p) per pair
    // (pointer equality first for designs sharing one buffer).
    fn designs_bit_identical(a: ndarray::ArrayView2<f64>, b: ndarray::ArrayView2<f64>) -> bool {
        if a.dim() != b.dim() {
            return false;
        }
        if std::ptr::eq(a.as_ptr(), b.as_ptr()) {
            return true;
        }
        match (a.as_slice(), b.as_slice()) {
            (Some(x), Some(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|(u, v)| u.to_bits() == v.to_bits())
            }
            _ => a
                .iter()
                .zip(b.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
        }
    }
    fn weights_bit_identical(a: ArrayView1<f64>, b: ArrayView1<f64>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        if std::ptr::eq(a.as_ptr(), b.as_ptr()) {
            return true;
        }
        match (a.as_slice(), b.as_slice()) {
            (Some(x), Some(y)) => x
                .iter()
                .zip(y.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
            _ => a
                .iter()
                .zip(b.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
        }
    }
    // dup_src[j] = the first earlier learner whose design and row weights
    // are bit-identical to learner j's, if any.
    let mut dup_src: Vec<Option<usize>> = vec![None; l];
    for j in 1..l {
        for w in 0..j {
            if weights_bit_identical(row_weights[j], row_weights[w])
                && designs_bit_identical(designs[j], designs[w])
            {
                dup_src[j] = Some(w);
                break;
            }
        }
    }

    let cross_gram_started = timing_enabled.then(Instant::now);
    let mut g: Vec<Vec<Array2<f64>>> = (0..l)
        .map(|_| (0..l).map(|_| Array2::zeros((0, 0))).collect())
        .collect();
    let pairs: Vec<(usize, usize)> = (0..l)
        .flat_map(|j| (j..l).map(move |w| (j, w)))
        .filter(|&(j, w)| {
            // A pair whose later member duplicates an earlier learner is
            // filled from that learner's blocks below, not computed.
            dup_src[w].is_none() || j == w
        })
        .collect();
    let upper_count = pairs.len();
    // Diagonal blocks prefer the stashed X'WX; cross blocks are always
    // computed. The materialized stash carries the same standard layout the
    // cross-Gram accumulator produces, so the mat-vec kernels treat them
    // identically.
    let compute_pair = |&(j, w): &(usize, usize)| {
        if j == w {
            self_grams[j].map_or_else(
                || weighted_cross_gram(designs[j], row_weights[j], designs[j]),
                |csr| csr.to_dense(),
            )
        } else {
            weighted_cross_gram(designs[j], row_weights[j], designs[w])
        }
    };
    let results: Vec<Array2<f64>> = if upper_count >= 10 {
        use rayon::prelude::*;
        pairs.par_iter().map(compute_pair).collect()
    } else {
        pairs.iter().map(compute_pair).collect()
    };
    for ((j, w), mat) in pairs.into_iter().zip(results) {
        g[j][w] = mat;
    }
    // Fill the pairs skipped above from the source learner's blocks: for
    // a duplicate w of source s, the block g[j][w] equals the block
    // between learners j and s. For j ≤ s that is the upper-triangle
    // block g[j][s] itself (the shared diagonal when j = s), which was
    // computed. For s < j < w it is the transpose of the computed upper
    // block g[s][j] — g[j][s] is a lower-triangle block and still empty
    // at this point — materialised in standard layout exactly as the
    // lower-triangle fill below does. The diagonal g[w][w] is the
    // source's diagonal g[s][s]. Duplicates are processed in increasing
    // w, so when a block g[s][j] was itself filled (j a duplicate of s),
    // that fill has already happened.
    for w in 0..l {
        if let Some(s) = dup_src[w] {
            for j in 0..=w {
                g[j][w] = if j <= s {
                    g[j][s].clone()
                } else if j < w {
                    let (rows, cols) = g[s][j].dim(); // (p_s, p_j)
                    let mut block = Array2::<f64>::zeros((cols, rows)); // (p_j, p_s)
                    block.assign(&g[s][j].t());
                    block
                } else {
                    g[s][s].clone()
                };
            }
        }
    }
    // Lower triangle: transpose the upper. O(p²) memcpy per matrix — trivial
    // compared to the O(p²·n) matmul saved. The copy is materialised in
    // standard (row-major) layout, matching the upper triangle's matmul
    // results: `.t().to_owned()` would keep the transposed layout, and the
    // mat-vec kernels accumulate in a layout-dependent order, so equal-valued
    // blocks would update bit-identical learners differently — breaking the
    // exact selection ties R preserves for duplicate-column learners
    // (gaussian_gram_twin_learners_stay_tied).
    for j in 1..l {
        for w in 0..j {
            let (rows, cols) = g[w][j].dim(); // (p_w, p_j)
            let mut block = Array2::<f64>::zeros((cols, rows)); // (p_j, p_w)
            block.assign(&g[w][j].t());
            g[j][w] = block;
        }
    }
    let products: Vec<Vec<GramMatVec<'_>>> = g
        .iter()
        .map(|row| row.iter().map(GramMatVec::new).collect())
        .collect();
    let sparse_cache_bytes: usize = products
        .iter()
        .flatten()
        .map(GramMatVec::cache_bytes)
        .sum::<usize>()
        + pens
            .iter()
            .map(|pen| match pen {
                Pen::Block(matrix) => matrix.cache_bytes(),
                _ => 0,
            })
            .sum::<usize>();
    let mut rhs: Vec<Array1<f64>> = (0..l)
        .map(|j| designs[j].t().dot(&(&row_weights[j] * &u0)))
        .collect();
    let cross_gram_ns = cross_gram_started.map(|started| started.elapsed().as_nanos() as u64);

    let nu = control.nu;
    let mut selected = Vec::with_capacity(mstop);
    let mut coefs: Vec<Vec<f64>> = if store_coefs {
        Vec::with_capacity(mstop)
    } else {
        Vec::new()
    };
    let mut risk = if store_risk {
        Vec::with_capacity(mstop + 1)
    } else {
        Vec::new()
    };
    if store_risk {
        risk.push(wu2);
    }

    // Per-learner beta buffers and a shared matvec scratch — the inner loop
    // below is then allocation-free (GRAM2).
    let p_js: Vec<usize> = designs.iter().map(|d| d.ncols()).collect();
    let max_p = p_js.iter().copied().max().unwrap_or(0);
    let mut betas: Vec<Array1<f64>> = p_js.iter().map(|&p| Array1::zeros(p)).collect();
    let mut scratch = Array1::<f64>::zeros(max_p);
    let mut tie_selection_count = 0usize;
    let setup_ns = gram_started.map(|started| started.elapsed().as_nanos() as u64);
    let iteration_started = timing_enabled.then(Instant::now);
    for _ in 0..mstop {
        let mut best = 0usize;
        let mut best_sse = f64::INFINITY;
        let mut best_multiplicity = 0usize;
        for (global, candidate) in candidates.iter().enumerate() {
            let j = match candidate {
                Candidate::Linear(j) => *j,
                Candidate::Tree(t) => {
                    let sse = trees[*t].fit_step_select(u.view(), weights, wu2).1;
                    observe_candidate(
                        global,
                        sse,
                        &mut best,
                        &mut best_sse,
                        &mut best_multiplicity,
                    );
                    continue;
                }
            };
            let sse = match &solvers[j] {
                Solv::Mono {
                    solver_a,
                    penalised_gram,
                    d_mat,
                    kind,
                    lambda2,
                    niter,
                } => {
                    let beta = match kind {
                        BmonoSolver::Iterative => solve_bmono_iterative(
                            solver_a,
                            penalised_gram.view(),
                            d_mat.view(),
                            rhs[j].view(),
                            *lambda2,
                            *niter,
                        ),
                        BmonoSolver::QuadProg => solve_bmono_quadprog(
                            solver_a,
                            penalised_gram.view(),
                            d_mat.view(),
                            rhs[j].view(),
                            1e-10,
                            4 * d_mat.nrows().max(8),
                        ),
                    };
                    betas[j].assign(&beta);
                    // Full-form SSE with the unpenalized cross-Gram
                    // g[j][j] = Xⱼ'W Xⱼ (a constrained beta does not satisfy
                    // the ridge stationarity the shortcut above relies on).
                    let mut t = scratch.slice_mut(ndarray::s![..p_js[j]]);
                    products[j][j].mul_into(&betas[j], t.view_mut());
                    let t_view = scratch.slice(ndarray::s![..p_js[j]]);
                    wu2 - 2.0 * betas[j].dot(&rhs[j]) + betas[j].dot(&t_view)
                }
                _ => {
                    solvers[j].solve_into(&rhs[j], &mut betas[j]);
                    // residual SSE identity: Σw(u−Xβ)² = wu2 − β'rhs − β'Pβ
                    wu2 - betas[j].dot(&rhs[j]) - pens[j].quad(&betas[j], &mut scratch)
                }
            };
            observe_candidate(
                global,
                sse,
                &mut best,
                &mut best_sse,
                &mut best_multiplicity,
            );
        }
        // No learner produced a finite SSE — the design is singular/degenerate
        // (e.g. a rank-deficient centered learner with λ=0, where the solver
        // returns NaN). Bail to the general path, which tolerates this (a
        // NaN SSE fails its `<` selection test, yielding a harmless no-op
        // step) rather than propagating NaN through the incremental updates.
        if !best_sse.is_finite() {
            return None;
        }
        if collect_report {
            tie_selection_count += usize::from(best_multiplicity > 1);
        }
        let w = match candidates[best] {
            Candidate::Tree(t) => {
                // A tree step: move the explicit residual, and each linear
                // learner's projection by the tree's weighted fit.
                let tree_pred = trees[t].pred_buf();
                for i in 0..n {
                    u[i] -= nu * tree_pred[i];
                }
                let w_tree_pred: Array1<f64> = Array1::from_shape_fn(n, |i| tree_pred[i]);
                for j in 0..l {
                    let projected = designs[j].t().dot(&(&row_weights[j] * &w_tree_pred));
                    rhs[j].scaled_add(-nu, &projected);
                }
                wu2 = weights
                    .iter()
                    .zip(u.iter())
                    .map(|(&wt, &ui)| wt * ui * ui)
                    .sum();
                selected.push(best);
                if store_coefs {
                    coefs.push(trees[t].selection_beta().to_vec());
                }
                None
            }
            Candidate::Linear(w) => Some(w),
        };
        if let Some(w) = w {
            // wu2 ← wu2 − 2ν·β_w'rhs_w + ν²·β_w'(B_w'W B_w)β_w (M_w = G[w][w])
            {
                let mut t = scratch.slice_mut(ndarray::s![..p_js[w]]);
                products[w][w].mul_into(&betas[w], t.view_mut());
            }
            let mw_beta = scratch.slice(ndarray::s![..p_js[w]]);
            wu2 = wu2 - 2.0 * nu * betas[w].dot(&rhs[w]) + nu * nu * betas[w].dot(&mw_beta);
            // Reuse the self-product for the winner's RHS. Other learners
            // use the same product-then-update arithmetic so duplicate designs
            // stay tied even when one of them is the winner.
            rhs[w].scaled_add(-nu, &mw_beta);
            for j in 0..l {
                if j == w {
                    continue;
                }
                let mut out = scratch.slice_mut(ndarray::s![..p_js[j]]);
                products[j][w].mul_into(&betas[w], out.view_mut());
                rhs[j].scaled_add(-nu, &out);
            }
            if has_trees {
                let step = designs[w].dot(&betas[w]);
                u.scaled_add(-nu, &step);
                wu2 = weights
                    .iter()
                    .zip(u.iter())
                    .map(|(&wt, &ui)| wt * ui * ui)
                    .sum();
            }
            selected.push(best);
            if store_coefs {
                coefs.push(betas[w].to_vec());
            }
        }
        if store_risk {
            if !wu2.is_finite() {
                // The Gram path returns `Option` — `None` means "decline, use
                // the general path". Declining here routes this fit to the
                // engine's own non-finite-risk check, which reports the
                // iteration in a typed error instead of publishing a path of
                // meaningless risks.
                return None;
            }
            risk.push(wu2);
        }
    }
    let iteration_ns = iteration_started.map(|started| started.elapsed().as_nanos() as u64);

    // Reconstruct fitted: f = offset + Σⱼ Bⱼ·(ν·Σ_{m:sel=j} coefs[m]).
    let reconstruction_started = (timing_enabled && store_coefs).then(Instant::now);
    let fitted = if store_coefs && has_trees {
        // `u` is y minus the fit, kept exactly alongside the projections.
        &y.to_owned() - &u
    } else if store_coefs {
        // Without trees the selected indices are the linear learners' own.
        let mut totals: Vec<Vec<f64>> = designs.iter().map(|d| vec![0.0; d.ncols()]).collect();
        for m in 0..mstop {
            let t = &mut totals[selected[m]];
            let c = &coefs[m];
            for k in 0..c.len().min(t.len()) {
                t[k] += nu * c[k];
            }
        }
        let mut fitted = Array1::from_elem(n, offset);
        for j in 0..l {
            let t_arr = Array1::from_vec(totals[j].clone());
            fitted += &designs[j].dot(&t_arr);
        }
        fitted
    } else {
        Array1::zeros(0)
    };
    let reconstruction_ns =
        reconstruction_started.map(|started| started.elapsed().as_nanos() as u64);

    Some(BoostingPath {
        report: if collect_report {
            FitReport {
                tie_selection_count,
                setup_ns,
                cross_gram_ns,
                iteration_ns,
                reconstruction_ns,
                gram_cache_bytes: total_p
                    .saturating_mul(total_p)
                    .saturating_mul(std::mem::size_of::<f64>())
                    .saturating_add(sparse_cache_bytes),
                ..FitReport::default()
            }
        } else {
            FitReport::default()
        },
        offset,
        selected,
        coefs,
        risk,
        // The Gram route maintains per-learner projections instead of a
        // length-n residual, so there is none to retain; a continuation of
        // such a fit takes the general route.
        residual: Array1::zeros(0),
        fitted,
        increments: Vec::new(),
        // Gaussian fast path: no scalar nuisance.
        nuisance_path: Vec::new(),
        nuisance_vec_path: Vec::new(),
        nuisance_vec: Vec::new(),
    })
}
