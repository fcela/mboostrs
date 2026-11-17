//! Concurvity of each term of a refit, as mgcv's `concurvity(full = TRUE)`
//! defines it (Wood, *Generalized Additive Models*, 2nd ed., section 5.6.2).
//!
//! For a term with design columns `X_j` and the rest of the model matrix
//! `X_o` (the intercept and every other term), take the triangular factor
//! `R` of `[X_o, X_j]` and keep its last `k` columns, those of `X_j`. Their
//! first `r = ncol(X_o)` rows, `R_1`, are the part of the term the other
//! terms can reproduce; `R_t` is the triangular factor of all `k` columns.
//! Then
//!
//! ```text
//! worst    = largest squared singular value of R_t^{-T} R_1^T
//! observed = ||R_1 beta||^2 / ||R_t beta||^2
//! estimate = sum(R_1^2) / sum(R_j^2)
//! ```
//!
//! all between 0 (the term is orthogonal to the rest) and 1 (it lies in
//! their span). `worst` is the worst case over every coefficient vector,
//! `observed` uses the fitted coefficients, and `estimate` is an average
//! over coefficient directions.
//!
//! One QR of the full matrix serves every term: if `X = Q R0`, then a column
//! permutation `X P = Q (R0 P)`, so the factor of `[X_o, X_j]` is the
//! factor of the `p x p` matrix `R0 P`. The same holds for a subset of the
//! columns: the factor of `X_S` is the factor of `R0` restricted to `S`, so
//! one QR serves every term of the summary and every pair of the pairwise
//! measures (mgcv's `full = TRUE` and `full = FALSE` respectively).

use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::learners::PreparedLearner;
use crate::MboostrsError;

/// mgcv's three concurvity measures for one term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Concurvity {
    /// Worst case over all coefficient vectors of the term.
    pub worst: f64,
    /// At the fitted coefficients.
    pub observed: f64,
    /// Averaged over coefficient directions.
    pub estimate: f64,
}

/// Upper triangular factor of `m` (rows >= columns), `ncols x ncols`.
fn r_factor(m: &Mat<f64>) -> Mat<f64> {
    let p = m.ncols();
    let r = m.qr().compute_thin_r();
    if r.nrows() == p {
        return r;
    }
    let mut out = Mat::<f64>::zeros(p, p);
    for i in 0..r.nrows() {
        for j in 0..p {
            out.write(i, j, r.read(i, j));
        }
    }
    out
}

/// [`concurvity`] from the Gram matrix `X'X` of the model's design, in place
/// of a QR of `X`.
///
/// The measures use `X` only through the triangular factor `R` of `X`. A
/// Cholesky factor `R'R = X'X` is `R` up to the signs of its rows, and those
/// signs cancel in every measure (`R_t^-T R_1^T` changes by orthogonal
/// diagonal scalings on each side, and the norms are unchanged). The Gram is
/// O(n p^2) like the QR but runs in parallel and needs no pass over a serial
/// factorization of the n-row design. Returns `None` when `X'X` is not
/// positive definite (a collinear design), where the caller uses [`concurvity`].
pub fn concurvity_from_gram(
    xtx: ArrayView2<f64>,
    spans: &[(usize, usize)],
    beta: ArrayView1<f64>,
) -> Option<Vec<Concurvity>> {
    use faer::linalg::solvers::Cholesky;
    use faer::Side;

    let p = xtx.nrows();
    let mut gram = Mat::<f64>::from_fn(p, p, |i, j| xtx[[i, j]]);
    let chol = match Cholesky::try_new(gram.as_ref(), Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            // A collinear design (partitions of unity across blocks, say) has a
            // singular X'X. A ridge of 1e-12 of the mean diagonal makes its
            // factor finite; the null directions then carry the ridge's share
            // of the norm, which is what the measures report as full overlap.
            let mean = (0..p).map(|i| xtx[[i, i]]).sum::<f64>() / p.max(1) as f64;
            let ridge = 1e-12 * mean.max(f64::MIN_POSITIVE);
            for i in 0..p {
                gram.write(i, i, gram.read(i, i) + ridge);
            }
            Cholesky::try_new(gram.as_ref(), Side::Lower).ok()?
        }
    };
    let l = chol.compute_l();
    let r0 = Mat::<f64>::from_fn(p, p, |i, j| l.read(j, i));
    Some(
        spans
            .iter()
            .map(|&(start, end)| {
                let rest: Vec<usize> = (0..p).filter(|c| *c < start || *c >= end).collect();
                let term: Vec<usize> = (start..end).collect();
                let beta_term: Vec<f64> =
                    beta.iter().skip(start).take(end - start).cloned().collect();
                measures_from_factor(&r0, &rest, &term, &beta_term)
            })
            .collect(),
    )
}

/// Triangular factor of `x`, reduced over blocks of 4,096 rows.
fn design_factor(x: ArrayView2<f64>) -> Mat<f64> {
    const ROWS: usize = 4096;
    let (n, p) = x.dim();
    let mut acc = Mat::<f64>::zeros(0, p);
    let mut start = 0;
    while start < n {
        let end = (start + ROWS).min(n);
        let top = acc.nrows();
        let block = Mat::from_fn(top + end - start, p, |i, j| {
            if i < top {
                acc.read(i, j)
            } else {
                x[[start + i - top, j]]
            }
        });
        acc = block.qr().compute_thin_r();
        start = end;
    }
    r_factor(&acc)
}

/// Largest eigenvalue of a symmetric positive semidefinite matrix.
fn largest_eigenvalue(m: &Array2<f64>) -> f64 {
    let (vals, _) = super::reml::symmetric_eigen(m);
    vals.iter().cloned().fold(0.0f64, f64::max)
}

/// The three measures of the columns `term` of a design against the
/// columns `rest`, computed from `r0`, the triangular factor of a design
/// that contains both: the factor of `[X_rest, X_term]` is the factor of
/// `r0` restricted to those columns (in that order), because `X = Q R0`
/// gives `X_S = Q R0_S`. `beta` holds the fitted coefficients of the
/// term's columns and gives `observed`.
fn measures_from_factor(r0: &Mat<f64>, rest: &[usize], term: &[usize], beta: &[f64]) -> Concurvity {
    let r = rest.len();
    let k = term.len();
    let nan = Concurvity {
        worst: f64::NAN,
        observed: f64::NAN,
        estimate: f64::NAN,
    };
    if k == 0 || r == 0 || beta.len() != k {
        return nan;
    }
    let m = Mat::from_fn(r0.nrows(), r + k, |i, j| {
        let col = if j < r { rest[j] } else { term[j - r] };
        r0.read(i, col)
    });
    let full = r_factor(&m);
    // The term's columns of the factor: (r + k) x k.
    let rj = Mat::from_fn(r + k, k, |i, j| full.read(i, r + j));
    let rt = r_factor(&rj);
    // R1 is the first r rows of rj.
    let r1 = Array2::from_shape_fn((r, k), |(i, j)| rj.read(i, j));
    let total: f64 = (0..r + k)
        .flat_map(|i| (0..k).map(move |j| (i, j)))
        .map(|(i, j)| rj.read(i, j).powi(2))
        .sum();
    let shared: f64 = r1.iter().map(|v| v * v).sum();
    let estimate = if total > 0.0 {
        shared / total
    } else {
        f64::NAN
    };
    // Y = Rt^{-T} R1^T (k x r): solve Rt^T Y = R1^T.
    let mut y = Mat::from_fn(k, r, |i, j| r1[[j, i]]);
    let diag_ok = (0..k).all(|i| rt.read(i, i).abs() > 0.0);
    let worst = if diag_ok {
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            rt.transpose(),
            y.as_mut(),
            faer::Parallelism::None,
        );
        let yyt = Array2::from_shape_fn((k, k), |(a, b)| {
            (0..r).map(|c| y.read(a, c) * y.read(b, c)).sum::<f64>()
        });
        largest_eigenvalue(&yyt)
    } else {
        f64::NAN
    };
    let coef = Array1::from_shape_vec(k, beta.to_vec()).expect("beta has length k");
    let r1b = r1.dot(&coef);
    let rtb: Array1<f64> =
        Array1::from_shape_fn(k, |i| (0..k).map(|j| rt.read(i, j) * coef[j]).sum::<f64>());
    let denominator = rtb.dot(&rtb);
    let observed = if denominator > 0.0 {
        r1b.dot(&r1b) / denominator
    } else {
        f64::NAN
    };
    Concurvity {
        worst,
        observed,
        estimate,
    }
}

/// Concurvity of each term `spans[i] = (start, end)` (columns of `x`), at
/// the coefficients `beta` (same columns as `x`). `x` should hold every
/// column of the model, the intercept included.
pub fn concurvity(
    x: ArrayView2<f64>,
    spans: &[(usize, usize)],
    beta: ArrayView1<f64>,
) -> Vec<Concurvity> {
    let p = x.ncols();
    let r0 = design_factor(x);
    spans
        .iter()
        .map(|&(start, end)| {
            let rest: Vec<usize> = (0..p).filter(|c| *c < start || *c >= end).collect();
            let term: Vec<usize> = (start..end).collect();
            let beta_term: Vec<f64> = beta.iter().skip(start).take(end - start).cloned().collect();
            measures_from_factor(&r0, &rest, &term, &beta_term)
        })
        .collect()
}

/// Concurvity of a boosted additive model's terms: each term against the
/// rest and every ordered pair.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConcurvity {
    /// Term `i` against the intercept offset and every other term, mgcv's
    /// `concurvity(full = TRUE)` summary.
    pub summary: Vec<Concurvity>,
    /// `pairwise[i][j]` measures term `j` against term `i`, mgcv's
    /// `concurvity(full = FALSE)` matrices: the largest fraction of a
    /// term-`j` function's squared norm that lies in term `i`'s span, the
    /// same fraction at term `j`'s fitted coefficients, and the averaged
    /// `estimate`. The diagonal is 1 and no intercept is involved.
    pub pairwise: Vec<Vec<Concurvity>>,
}

/// Concurvity of the terms given by their design matrices: `designs[i]` is
/// term `i`'s `(n, p_i)` design and `coefs[i]` its fitted coefficients on
/// those columns (zero for a term that was never fitted, whose `observed`
/// is NaN). One QR of `[1, X_1, ..., X_L]` serves every measure.
pub fn model_concurvity(designs: &[Array2<f64>], coefs: &[Array1<f64>]) -> ModelConcurvity {
    let l = designs.len();
    let ones = Concurvity {
        worst: 1.0,
        observed: 1.0,
        estimate: 1.0,
    };
    let mut summary = vec![ones; l];
    let mut pairwise = vec![vec![ones; l]; l];
    if l == 0 {
        return ModelConcurvity { summary, pairwise };
    }
    let n = designs[0].nrows();
    let p_total = 1 + designs.iter().map(|d| d.ncols()).sum::<usize>();
    let mut x = Array2::<f64>::zeros((n, p_total));
    for i in 0..n {
        x[[i, 0]] = 1.0;
    }
    // Column ranges of each term inside x, past the intercept column 0.
    let spans: Vec<(usize, usize)> = {
        let mut spans = Vec::with_capacity(l);
        let mut col = 1;
        for d in designs {
            let p = d.ncols();
            x.slice_mut(ndarray::s![.., col..col + p]).assign(d);
            spans.push((col, col + p));
            col += p;
        }
        spans
    };
    let r0 = design_factor(x.view());
    let cols: Vec<Vec<usize>> = spans.iter().map(|&(s, e)| (s..e).collect()).collect();
    for i in 0..l {
        // The rest of the summary: the intercept column and every other
        // term's columns.
        let mut rest = vec![0];
        for (j, cj) in cols.iter().enumerate() {
            if j != i {
                rest.extend_from_slice(cj);
            }
        }
        let beta_i = coefs.get(i).map(|c| c.to_vec()).unwrap_or_default();
        summary[i] = measures_from_factor(&r0, &rest, &cols[i], &beta_i);
        for j in 0..l {
            if i == j {
                continue;
            }
            let beta_j = coefs.get(j).map(|c| c.to_vec()).unwrap_or_default();
            pairwise[i][j] = measures_from_factor(&r0, &cols[i], &cols[j], &beta_j);
        }
    }
    ModelConcurvity { summary, pairwise }
}

/// A term's design with the constant direction removed and the columns it
/// made dependent dropped, returning the kept column indices.
///
/// mgcv's smooths carry sum-to-zero constraints that exclude the constant
/// from each smooth's span before `concurvity` measures them. Boosted
/// bases carry no such constraint -- a B-spline basis partitions unity, so
/// every `bbs` span contains the constant and would otherwise reach
/// worst-case concurvity 1 against any other term. Each column first has
/// its mean subtracted; a basis that contained the constant is then
/// internally rank-deficient by exactly that direction (the centered
/// columns sum to zero up to rounding), and leaving the dependency in
/// would let the triangular solves below amplify rounding noise into the
/// measures. A pivoted Gram-Schmidt pass therefore keeps only the
/// numerically independent columns, largest residual first -- the same
/// reduction the refit pipeline applies through `constrain_term`, in the
/// unweighted form that matches the unweighted geometry of the measures.
fn reduce_constant(design: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
    let n = design.nrows();
    let q = design.ncols();
    let centered: Vec<Array1<f64>> = (0..q)
        .map(|j| {
            let col = design.column(j);
            let mean = col.sum() / n as f64;
            col.to_owned() - mean
        })
        .collect();
    // Relative tolerance for "independent": an exact dependency of a
    // partition-of-unity basis leaves residuals around 1e-15 of the column
    // scale, a genuinely varying direction far more than 1e-10 of it.
    let scale = centered
        .iter()
        .map(|c| c.dot(c))
        .fold(0.0f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let tol2 = 1e-20 * scale;
    let mut residual: Vec<Option<Array1<f64>>> = centered.iter().cloned().map(Some).collect();
    let mut kept: Vec<usize> = Vec::new();
    loop {
        let mut best: Option<(usize, f64)> = None;
        for j in 0..q {
            if let Some(r) = &residual[j] {
                let norm2 = r.dot(r);
                if best.is_none_or(|(_, b)| norm2 > b) {
                    best = Some((j, norm2));
                }
            }
        }
        let (j, norm2) = match best {
            Some((j, norm2)) if norm2 > tol2 => (j, norm2),
            _ => break,
        };
        let rj = residual[j].take().expect("just checked Some");
        for r in residual.iter_mut().flatten() {
            let coef = rj.dot(r) / norm2;
            r.scaled_add(-coef, &rj);
        }
        kept.push(j);
    }
    let p = kept.len();
    let mut reduced = Array2::<f64>::zeros((n, p));
    for (jj, &j) in kept.iter().enumerate() {
        reduced.slice_mut(ndarray::s![.., jj]).assign(&centered[j]);
    }
    (reduced, kept)
}

/// Concurvity of a fitted boosted model, from its prepared learners and its
/// boosting path. `selected[s]` names the store learner boosted at step `s`
/// and `coefs_vecs[s]` its unshrunken update (the `selected`/`coefs_vecs`
/// the fit returns); the fitted coefficients of learner `j` are `nu` times
/// the sum of its steps' updates. Learners without a function-space basis
/// are excluded -- trees, whose stored design is their raw covariates, and
/// `%O%` Kronecker learners, whose design is a placeholder -- and so are
/// their steps. Every measured term is first reduced by the constant
/// direction (see `reduce_constant`), as mgcv measures its constrained
/// smooths; a term that is constant after the reduction (an intercept
/// column) has no function space left and reports NaN. Returns the store
/// indices that were measured alongside the measures, in store order.
pub fn boosted_concurvity(
    learners: &[PreparedLearner],
    nu: f64,
    selected: &[usize],
    coefs_vecs: &[Vec<f64>],
) -> Result<(Vec<usize>, ModelConcurvity), MboostrsError> {
    let has_basis = |l: &PreparedLearner| {
        !matches!(
            l,
            PreparedLearner::Btree { .. } | PreparedLearner::KroneckerBols { .. }
        )
    };
    // Fitted coefficients accumulated per learner; None marks an excluded
    // learner, whose steps are skipped.
    let mut acc: Vec<Option<Vec<f64>>> = learners
        .iter()
        .map(|l| has_basis(l).then(|| vec![0.0; l.n_parameters()]))
        .collect();
    if selected.len() != coefs_vecs.len() {
        return Err(MboostrsError::Shape(format!(
            "concurvity: selected has {} steps but coefs_vecs has {} entries",
            selected.len(),
            coefs_vecs.len()
        )));
    }
    for (&sel, v) in selected.iter().zip(coefs_vecs) {
        let slot = match acc.get_mut(sel) {
            Some(slot @ Some(_)) => slot,
            Some(None) => continue,
            None => {
                return Err(MboostrsError::Shape(format!(
                    "concurvity: step selects learner {sel} of {n} in the store",
                    n = learners.len()
                )))
            }
        };
        let slot = slot.as_mut().expect("checked Some above");
        if v.len() != slot.len() {
            return Err(MboostrsError::Shape(format!(
                "concurvity: step updates learner {sel} with {} coefficients, its design has {} columns",
                v.len(),
                slot.len()
            )));
        }
        for (a, b) in slot.iter_mut().zip(v) {
            *a += nu * b;
        }
    }
    let included: Vec<usize> = (0..learners.len()).filter(|&i| acc[i].is_some()).collect();
    let mut designs: Vec<Array2<f64>> = Vec::with_capacity(included.len());
    let mut coefs: Vec<Array1<f64>> = Vec::with_capacity(included.len());
    for &i in &included {
        let design = learners[i]
            .design_matrix()
            .expect("every learner with a basis exposes a design");
        let (reduced, kept) = reduce_constant(&design);
        let beta = acc[i]
            .as_ref()
            .expect("every included learner accumulated coefficients");
        designs.push(reduced);
        coefs.push(Array1::from(
            kept.iter().map(|&j| beta[j]).collect::<Vec<f64>>(),
        ));
    }
    Ok((included, model_concurvity(&designs, &coefs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_terms_have_zero_and_a_duplicate_has_one() {
        let n = 200;
        let x1: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.91).cos()).collect();
        // Columns: intercept, x1, x1 again (a duplicated term), x2.
        let x = Array2::from_shape_fn((n, 4), |(i, j)| match j {
            0 => 1.0,
            1 | 2 => x1[i],
            _ => x2[i],
        });
        let beta = Array1::from(vec![0.5, 1.0, 1.0, 1.0]);
        let c = concurvity(x.view(), &[(1, 2), (2, 3), (3, 4)], beta.view());
        assert!((c[0].worst - 1.0).abs() < 1e-8, "{:?}", c[0]);
        assert!((c[0].observed - 1.0).abs() < 1e-8);
        assert!(c[2].worst < 0.2, "{:?}", c[2]);
        assert!(c[2].estimate <= c[2].worst + 1e-12);
    }

    fn smooth(n: usize, shift: f64) -> Array2<f64> {
        // A small quadratic-polynomial "basis" over an index vector.
        let t: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
        Array2::from_shape_fn((n, 3), |(i, j)| {
            let x = t[i] + shift;
            match j {
                0 => 1.0,
                1 => x,
                _ => x * x,
            }
        })
    }

    #[test]
    fn model_concurvity_summary_matches_the_term_function() {
        let n = 120;
        let d1 = smooth(n, 0.0);
        let d2 = smooth(n, 0.5);
        let d3 = smooth(n, 11.0);
        let designs = [d1.clone(), d2.clone(), d3.clone()];
        let coefs: Vec<Array1<f64>> = vec![
            Array1::from(vec![1.0, -2.0, 0.5]),
            Array1::from(vec![0.3, 1.0, 1.0]),
            Array1::from(vec![0.0, 0.0, 0.0]),
        ];
        // The same model matrix assembled flat, intercept first.
        let p_total = 1 + 3 * 3;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
        }
        let mut col = 1;
        for d in &designs {
            x.slice_mut(ndarray::s![.., col..col + d.ncols()]).assign(d);
            col += d.ncols();
        }
        let mut beta_flat = Array1::<f64>::zeros(p_total);
        let mut c = 1;
        for b in &coefs {
            for j in 0..b.len() {
                beta_flat[c + j] = b[j];
            }
            c += b.len();
        }
        let spans = [(1, 4), (4, 7), (7, 10)];
        let reference = concurvity(x.view(), &spans, beta_flat.view());
        let ours = model_concurvity(&designs, &coefs);
        let close = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-12;
        for i in 0..3 {
            assert!(
                close(ours.summary[i].worst, reference[i].worst),
                "term {i}: {:?} vs {:?}",
                ours.summary[i],
                reference[i]
            );
            assert!(close(ours.summary[i].observed, reference[i].observed));
            assert!(close(ours.summary[i].estimate, reference[i].estimate));
        }
        // A never-fitted term has no observed value.
        assert!(ours.summary[2].observed.is_nan());
    }

    #[test]
    fn model_concurvity_pairwise_is_one_for_duplicates_and_excludes_the_intercept() {
        let n = 150;
        let d1 = smooth(n, 0.0);
        let d2 = d1.clone();
        let ours = model_concurvity(
            &[d1, d2],
            &[Array1::ones(3), Array1::from(vec![0.5, 1.0, 0.0])],
        );
        // Identical spans: every measure is 1 in both directions.
        for i in 0..2 {
            for j in 0..2 {
                let c = &ours.pairwise[i][j];
                assert!((c.worst - 1.0).abs() < 1e-8, "[{i}][{j}] = {c:?}");
                assert!((c.estimate - 1.0).abs() < 1e-8);
                if i != j {
                    assert!((c.observed - 1.0).abs() < 1e-8);
                } else {
                    assert_eq!(c.worst, 1.0);
                }
            }
        }
        // mgcv's pairwise measures exclude the intercept, while the summary
        // counts it among the rest. Term `u = x^2 - proj_x(x^2)` (x exactly
        // zero-mean) is orthogonal to x, so the pairwise worst against [x]
        // is ~0; its mean part, about 0.56 of its norm, is constant, which
        // only the intercept among the rest of the summary can reproduce.
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n as f64) - (n as f64 - 1.0) / (2.0 * n as f64))
            .collect();
        let x2: Vec<f64> = x.iter().map(|v| v * v).collect();
        let alpha = {
            let cross: f64 = x.iter().zip(&x2).map(|(a, b)| a * b).sum();
            let norm: f64 = x.iter().map(|v| v * v).sum();
            cross / norm
        };
        let u: Vec<f64> = x2.iter().zip(&x).map(|(b, a)| b - alpha * a).collect();
        let d1 = Array2::from_shape_fn((n, 1), |(i, _)| x[i]);
        let d2 = Array2::from_shape_fn((n, 1), |(i, _)| u[i]);
        let ours = model_concurvity(&[d1, d2], &[Array1::ones(1), Array1::ones(1)]);
        assert!(
            ours.pairwise[0][1].worst < 0.05,
            "{:?}",
            ours.pairwise[0][1]
        );
        assert!(
            ours.summary[1].worst > 0.4,
            "the intercept among the rest must capture the mean part: {:?}",
            ours.summary[1]
        );
    }

    #[test]
    fn boosted_concurvity_accumulates_coefficients_and_excludes_trees() {
        let n = 60;
        let x1 = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));
        let x2 = Array1::from_shape_fn(n, |i| ((i as f64) * 0.7).cos());
        let w = Array1::ones(n);
        let bbs1 =
            crate::learners::prepare_bbs(x1.view(), w.view(), 10, 3, 2, 4.0, false, "bbs(x1)");
        let bbs1_again = crate::learners::prepare_bbs(
            x1.view(),
            w.view(),
            10,
            3,
            2,
            4.0,
            false,
            "bbs(x1) again",
        );
        let bbs2 =
            crate::learners::prepare_bbs(x2.view(), w.view(), 10, 3, 2, 4.0, false, "bbs(x2)");
        let tree = crate::learners::prepare_btree(
            ndarray::Array2::from_shape_fn((n, 1), |(i, _)| x1[i]).view(),
            w.view(),
            20,
            10,
            4,
            None,
            "btree(x1)",
        );
        let learners = [bbs1, bbs2, bbs1_again, tree];
        // Two steps on learner 0 with updates of all ones; the fitted
        // coefficients of learner 0 are nu * 2 on every column.
        let p0 = learners[0].n_parameters();
        let sel = [0usize, 0usize];
        let coefs = [vec![1.0; p0], vec![1.0; p0]];
        let (included, mc) = boosted_concurvity(&learners, 0.1, &sel, &coefs).expect("valid path");
        assert_eq!(included, vec![0, 1, 2], "the tree must be excluded");
        // observed of term 0 uses beta = 0.2 * ones on the constant-reduced
        // columns: cross-check against a direct call on the same reduced
        // designs.
        let (r0, kept0) = reduce_constant(&learners[0].design_matrix().unwrap());
        let (r1, _) = reduce_constant(&learners[1].design_matrix().unwrap());
        let (r2, _) = reduce_constant(&learners[2].design_matrix().unwrap());
        let coefs = [
            Array1::from(vec![0.2; kept0.len()]),
            Array1::zeros(r1.ncols()),
            Array1::zeros(r2.ncols()),
        ];
        let direct = model_concurvity(&[r0, r1, r2], &coefs);
        assert!((mc.summary[0].observed - direct.summary[0].observed).abs() < 1e-12);
        assert!(mc.summary[1].observed.is_nan(), "learner 1 never selected");
        assert!(!mc.summary[0].worst.is_nan());
        // Identical bases are fully concurved in both directions; that the
        // duplicate pair measures about 1, not every pair, is also what
        // shows the reduction removed the constant every bbs span contains
        // (B-splines partition unity), which would otherwise force 1 for
        // all pairs.
        assert!(mc.pairwise[0][2].worst > 0.99, "{:?}", mc.pairwise[0][2]);
        assert!(mc.pairwise[2][0].worst > 0.99, "{:?}", mc.pairwise[2][0]);
        assert!(
            mc.pairwise[0][1].worst < mc.pairwise[0][2].worst,
            "distinct columns concurve less than identical ones: {:?} vs {:?}",
            mc.pairwise[0][1],
            mc.pairwise[0][2]
        );
    }

    #[test]
    fn reduce_constant_drops_intercept_columns_and_centers_the_rest() {
        let n = 12;
        // Columns: a constant, and 1..=n.
        let mut design = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = 3.0;
            design[[i, 1]] = (i + 1) as f64;
        }
        let (reduced, kept) = reduce_constant(&design);
        assert_eq!(kept, vec![1], "the constant column is dropped");
        assert_eq!(reduced.ncols(), 1);
        let mean: f64 = reduced.column(0).sum() / n as f64;
        assert!(mean.abs() < 1e-12, "the kept column is centered: {mean}");
    }

    #[test]
    fn reduce_constant_drops_the_dependency_a_partition_of_unity_leaves() {
        // Three hat-like columns that sum to 1, as a B-spline basis does:
        // centering makes them sum to (numerically) zero, and the pass must
        // drop one of the three instead of leaving a 1e-15-deep dependency.
        let n = 30;
        let mut design = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = i as f64 / n as f64;
            let (a, b) = (t.min(0.5), (t - 0.5).max(0.0));
            design[[i, 0]] = 1.0 - 2.0 * a;
            design[[i, 1]] = 2.0 * (a - b);
            design[[i, 2]] = 2.0 * b;
        }
        let (reduced, kept) = reduce_constant(&design);
        assert_eq!(kept.len(), 2, "one of the three columns is dropped");
        assert_eq!(reduced.ncols(), 2);
        // Every kept column is centered and the design has full rank.
        for j in 0..2 {
            let mean: f64 = reduced.column(j).sum() / n as f64;
            assert!(mean.abs() < 1e-12);
        }
        let r = r_factor(&Mat::from_fn(n, 2, |i, j| reduced[[i, j]]));
        assert!((0..2).all(|j| r.read(j, j).abs() > 1e-6));
    }

    #[test]
    fn boosted_concurvity_rejects_mismatched_paths() {
        let n = 10;
        let x = Array1::from_shape_fn(n, |i| i as f64);
        let w = Array1::ones(n);
        let learners = [crate::learners::prepare_scalar_bols(
            x.view(),
            w.view(),
            4.0,
            true,
            "bols(x)",
            None,
        )];
        assert!(boosted_concurvity(&learners, 0.1, &[0], &[]).is_err());
        assert!(boosted_concurvity(&learners, 0.1, &[7], &[vec![1.0]]).is_err());
        assert!(boosted_concurvity(&learners, 0.1, &[0], &[vec![1.0, 2.0]]).is_err());
        let (included, mc) = boosted_concurvity(&learners, 0.1, &[0], &[vec![1.0]]).expect("valid");
        assert_eq!(included, vec![0]);
        assert_eq!(mc.summary.len(), 1);
    }
}
