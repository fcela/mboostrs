//! Stability selection parameter computation.
//!
//! Reimplements R's `stabs::stabsel_parameters` logic in pure Rust, including
//! the r-concave tail-bound functions `_D_rc` and `_minD` that were previously
//! implemented in Python using scipy.
//!
//! The core numerical routines are:
//! - `brentq`: root-finding via Brent's method (replaces scipy.optimize.brentq)
//! - `minimize_bounded`: bounded scalar minimization (replaces scipy.optimize.minimize_scalar)
//! - `d_rc`: r-concave tail bound P(V >= which) over r-concave distributions
//! - `min_d`: minimum of two D-based PFER upper bounds
//!
//! All parameter combinations (cutoff/q/PFER with unimodal/r-concave/none assumptions)
//! are handled by `stabsel_parameters`.

use crate::MboostrsError;

/// Result of stability selection parameter computation.
#[derive(Debug, Clone)]
pub struct StabSelParams {
    /// Selection threshold on the inclusion frequency.
    pub cutoff: f64,
    /// Number of selection *cutoff-steps* used in the error bound.
    pub q: usize,
    /// Per-family error rate (PFER) target.
    pub pfer: f64,
    /// Number of candidate predictors.
    pub p: usize,
    /// Number of resampling replications per fold.
    pub b: usize,
    /// Resampling scheme: "subsampling", "bootstrap", or "SS".
    pub sampling_type: String,
    /// Error-bound assumption: "unimodal", "r-concave", or "none".
    pub assumption: String,
}

/// Brent's method for root-finding on a bracketed interval.
///
/// Finds a root of `f` in `[a, b]` assuming `f(a)` and `f(b)` have opposite signs.
/// Returns the root or `None` if the bracket is invalid or the iteration did not
/// converge within `maxiter` iterations.
///
/// Implements the algorithm from the SciPy/Netlib zeroin routine, which is the
/// same routine R's `uniroot` uses. It lives here because the r-concave tail
/// bounds needed it first; `learners::df_to_lambda` is the other caller.
pub(crate) fn brentq<F>(f: F, a: f64, b: f64, xtol: f64, rtol: f64) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    let maxiter = 500;
    let xpre = a;
    let xcur = b;
    let fpre = f(xpre);
    let fcur = f(xcur);

    if fpre * fcur > 0.0 {
        return None;
    }
    if fpre.abs() < f64::EPSILON {
        return Some(xpre);
    }
    if fcur.abs() < f64::EPSILON {
        return Some(xcur);
    }

    let mut xpre = xpre;
    let mut xcur = xcur;
    let mut fpre = fpre;
    let mut fcur = fcur;
    let mut xblk = 0.0f64;
    let mut fblk = 0.0f64;
    let mut spre = 0.0f64;
    let mut scur = 0.0f64;
    let mut sbis;

    for _ in 0..maxiter {
        if fpre * fcur < 0.0 {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        // scipy's zeroin convergence half-width: an absolute floor `xtol`
        // plus a relative term in the *current iterate*. This was previously
        // `(xtol + 2 * EPSILON) * |xcur - xblk|` — proportional to the bracket
        // width rather than to `xcur` — which makes the test below,
        // `|sbis| <= delta` with `|sbis| = 0.5 * |xblk - xcur|`, reduce to
        // `0.5 <= xtol`, i.e. never true. The routine then converged only when
        // `f(xcur)` hit exactly zero and otherwise ran to `maxiter`; on
        // `d_rc(0.09, 10, 50, -0.5)` six of the fifty roots exhausted the
        // budget and fell back to the bracket endpoint, costing ~1.5e-4.
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis_prev = 0.5 * (xblk - xcur);
        sbis = sbis_prev;

        if fcur == 0.0 || sbis.abs() <= delta {
            return Some(xcur);
        }

        if spre.abs() >= delta && fcur.abs() <= fpre.abs() {
            let stry = if xpre == xblk {
                // secant
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // Illinois / Pegasus
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };

            if 2.0 * stry.abs() < spre.abs().min((xblk - xcur).abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;

        if scur.abs() > delta {
            xcur += scur;
        } else if sbis > 0.0 {
            xcur += delta;
        } else {
            xcur -= delta;
        }

        fcur = f(xcur);
    }

    None
}

/// Bounded scalar minimization using Brent's method (golden-section + parabolic).
///
/// Finds the minimum of `f` in `[lo, hi]`. Returns the minimizer or `None`
/// if the interval is degenerate or the iteration does not converge within
/// `MAXITER` iterations.  Direct port of scipy's `_minimize_scalar_bounded` /
/// Numerical Recipes `brent`.
fn minimize_bounded<F>(f: F, lo: f64, hi: f64) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    if lo >= hi {
        return None;
    }
    const CGOLD: f64 = 0.3819660112501051;
    const MAXITER: usize = 500;
    const TOL: f64 = 1.4823223288210188e-8;

    let mut ax = lo;
    let mut bx = hi;
    let mut x = lo + CGOLD * (hi - lo);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0f64;
    let mut e = 0.0f64;

    for _ in 0..MAXITER {
        let xm = 0.5 * (ax + bx);
        let tol1 = TOL * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (bx - ax) {
            return Some(x);
        }
        if e.abs() >= tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            let etemp = e;
            e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (ax - x) && p < q * (bx - x) {
                d = p / q;
                let u = x + d;
                if (u - ax) < tol2 || (bx - u) < tol2 {
                    d = if xm > x { tol1 } else { -tol1 };
                }
            } else {
                e = if x >= xm { ax - x } else { bx - x };
                d = CGOLD * e;
            }
        } else {
            e = if x >= xm { ax - x } else { bx - x };
            d = CGOLD * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d > 0.0 { tol1 } else { -tol1 }
        };
        let fu = f(u);
        if fu <= fx {
            if u < x {
                bx = x;
            } else {
                ax = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                ax = u;
            } else {
                bx = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    None
}

/// Maximum number of base-learners that can be selected (q) for given p and B.
///
/// Reimplements R's `stabs::maxQ`.
pub fn max_q(p: usize, b: usize) -> usize {
    try_max_q(p, b).unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible variant of [`max_q`] for language bindings and user input.
pub fn try_max_q(p: usize, b: usize) -> Result<usize, MboostrsError> {
    if p == 0 {
        return Err(MboostrsError::StabSel("p must be greater than 0".into()));
    }
    if b <= 1 {
        return Err(MboostrsError::StabSel("B must be at least 2".into()));
    }
    let fact_1 = 4.0 * b as f64 / p as f64;
    let mut count = 0usize;
    for q in 1..=p {
        if (q as f64 * fact_1).ceil() as isize + 1 - 2 * (b as isize) < 0 {
            count += 1;
        }
    }
    Ok(count)
}

/// Constant used in unimodal PFER bound.
///
/// Reimplements R's `stabs::um_const`.
pub fn um_const(cutoff: f64, b: usize, theta: f64) -> f64 {
    try_um_const(cutoff, b, theta).unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible variant of [`um_const`] for language bindings and user input.
pub fn try_um_const(cutoff: f64, b: usize, theta: f64) -> Result<f64, MboostrsError> {
    if b == 0 {
        return Err(MboostrsError::StabSel("B must be greater than 0".into()));
    }
    if !cutoff.is_finite() || !theta.is_finite() {
        return Err(MboostrsError::StabSel(
            "cutoff and theta must be finite".into(),
        ));
    }
    let b = b as f64;
    if cutoff <= 0.75 {
        if cutoff < 0.5 + theta.powi(2).min(1.0 / (2.0 * b) + 0.75 * theta.powi(2)) {
            return Err(MboostrsError::StabSel("cutoff out of bounds".into()));
        }
        Ok(2.0 * (2.0 * cutoff - 1.0 - 1.0 / (2.0 * b)))
    } else {
        if cutoff > 1.0 {
            return Err(MboostrsError::StabSel("cutoff out of bounds".into()));
        }
        Ok((1.0 + 1.0 / b) / (4.0 * (1.0 - cutoff + 1.0 / (2.0 * b))))
    }
}

/// R's stabs::D function for r-concave tail bound.
///
/// Computes the worst-case P(V >= which) over all r-concave discrete
/// distributions on `{0, ..., B}` with `E[V] = theta * B`.
pub fn d_rc(theta: f64, which: usize, b: usize, r: f64) -> f64 {
    try_d_rc(theta, which, b, r).unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible variant of [`d_rc`] for language bindings and user input.
pub fn try_d_rc(theta: f64, which: usize, b: usize, r: f64) -> Result<f64, MboostrsError> {
    if b == 0 {
        return Err(MboostrsError::StabSel("B must be greater than 0".into()));
    }
    if !theta.is_finite() || theta < 0.0 {
        return Err(MboostrsError::StabSel(
            "theta must be finite and non-negative".into(),
        ));
    }
    if !r.is_finite() || r >= 0.0 {
        return Err(MboostrsError::StabSel(
            "r must be finite and negative".into(),
        ));
    }
    let s = 1.0 / r;
    let theta_b = theta * b as f64;

    let k_start = (2.0 * theta_b).ceil() as usize + 1;
    if which < k_start {
        return Ok(1.0);
    }
    if k_start > b {
        return Err(MboostrsError::StabSel("theta too large".into()));
    }

    // Find.a: compute a_k values using brentq root-finding
    let mut a_vec = vec![0.0f64; b + 1];
    a_vec[k_start - 1] = 1e5;

    for k in k_start..=b {
        let lo = 1e-5;
        let hi = a_vec[k - 1];
        if hi <= lo {
            a_vec[k] = lo;
            continue;
        }
        let f_lo = calc_a(lo, k, s, theta_b);
        let f_hi = calc_a(hi, k, s, theta_b);
        if f_lo * f_hi > 0.0 {
            a_vec[k] = hi;
            continue;
        }
        if let Some(root) = brentq(|a| calc_a(a, k, s, theta_b), lo, hi, 1e-12, 1e-12) {
            a_vec[k] = root;
        } else {
            a_vec[k] = hi;
        }
    }

    // Find maximum of optim_int over k from k_start to b-1
    let mut best = 0.0f64;
    for k in k_start..b {
        let lo_a = a_vec[k + 1];
        let hi_a = a_vec[k];
        if lo_a >= hi_a {
            continue;
        }
        if let Some(minimizer) =
            minimize_bounded(|a| -optim_int(a, which, k, s, theta_b), lo_a, hi_a)
        {
            // minimize_bounded minimizes -optim_int, so at the minimizer,
            // optim_int is maximized. The value we want is optim_int(minimizer).
            let oi_val = optim_int(minimizer, which, k, s, theta_b);
            let val = oi_val.max(0.0);
            if val > best {
                best = val;
            }
        }
    }

    Ok(best)
}

/// Helper for d_rc: solves for a in the Find.a routine.
///
/// R: denom <- sum((a + 0:k)^s); num <- sum((0:k) * (a + 0:k)^s)
/// Returns num/denom - thetaB
fn calc_a(a: f64, k: usize, s: f64, theta_b: f64) -> f64 {
    let mut numer = 0.0f64;
    let mut denom = 0.0f64;
    for i in 0..=k {
        let term = (a + i as f64).powf(s);
        numer += i as f64 * term;
        denom += term;
    }
    numer / denom - theta_b
}

/// Helper for d_rc: the OptimInt function.
///
/// R: 1 - (k+1-thetaB)*sum((a+0:(t-1))^s) / sum((k+1-0:k)*(a+0:k)^s)
fn optim_int(a: f64, t: usize, k: usize, s: f64, theta_b: f64) -> f64 {
    let mut sum_a = 0.0f64;
    for i in 0..t {
        sum_a += (a + i as f64).powf(s);
    }
    let mut sum_k = 0.0f64;
    for i in 0..=k {
        let term = (a + i as f64).powf(s);
        sum_k += (k as f64 + 1.0 - i as f64) * term;
    }
    1.0 - (k as f64 + 1.0 - theta_b) * sum_a / sum_k
}

/// R's stabs::minD function for r-concave PFER bound.
///
/// Returns the minimum of two D-based upper bounds using r-concavity
/// orders r = -1/2 and r = -1/4.
pub fn min_d(q: usize, p: usize, cutoff: f64, b: usize) -> f64 {
    try_min_d(q, p, cutoff, b).unwrap_or_else(|e| panic!("{e}"))
}

/// R's `signif(x, 10)`: round to 10 significant digits.
fn signif10(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let digits = 9 - x.abs().log10().floor() as i32;
    let m = 10f64.powi(digits);
    (x * m).round() / m
}

/// Fallible variant of [`min_d`] for language bindings and user input.
pub fn try_min_d(q: usize, p: usize, cutoff: f64, b: usize) -> Result<f64, MboostrsError> {
    if !cutoff.is_finite() || !(0.0..=1.0).contains(&cutoff) {
        return Err(MboostrsError::StabSel(
            "cutoff must be finite and between 0 and 1".into(),
        ));
    }
    if q == 0 {
        return Err(MboostrsError::StabSel("q must be greater than 0".into()));
    }
    let max_q_val = try_max_q(p, b)?;
    if q > max_q_val {
        return Err(MboostrsError::StabSel(format!(
            "q must be <= maxQ = {max_q_val}"
        )));
    }
    // R: ceiling(signif(cutoff / (1 / (2 * B)), 10)); rounding to 10 digits
    // removes floating-point noise such as 56.00000000000001.
    let which = signif10(cutoff / (1.0 / (2.0 * b as f64))).ceil() as usize;
    let mut vals = vec![1.0f64];
    let which1 = which as isize - b as isize;
    if which1 > 0 {
        if let Ok(val) = try_d_rc(
            (q as f64).powi(2) / (p as f64).powi(2),
            which1 as usize,
            b,
            -0.5,
        ) {
            vals.push(val);
        }
    }
    if let Ok(val) = try_d_rc(q as f64 / p as f64, which, 2 * b, -0.25) {
        vals.push(val);
    }
    Ok(vals.into_iter().fold(f64::INFINITY, f64::min))
}

/// Compute optimal cutoff for unimodal assumption.
fn optimal_cutoff_unimodal(p: usize, q: usize, pfer: f64, b: usize) -> Result<f64, MboostrsError> {
    let mut cutoffgrid: Vec<f64> = (2..=b).map(|k| 0.5 + k as f64 / (2.0 * b as f64)).collect();
    let c_min = ((q as f64 / p as f64).powi(2) + 0.5)
        .min(0.5 + 1.0 / (2.0 * b as f64) + 0.75 * (q as f64 / p as f64).powi(2));
    cutoffgrid.retain(|&c| c > c_min);
    if cutoffgrid.is_empty() {
        let max_q_val = ((p as f64 * 0.5f64.sqrt()).floor() as usize)
            .max((p as f64 * ((0.5 - 1.0 / (2.0 * b as f64)) * 4.0 / 3.0).sqrt()).floor() as usize);
        return Err(MboostrsError::StabSel(format!("q must be <= {max_q_val}")));
    }
    for &cutoff in &cutoffgrid {
        let ub = (q as f64).powi(2) / p as f64 / try_um_const(cutoff, b, q as f64 / p as f64)?;
        if ub < pfer {
            return Ok(cutoff);
        }
    }
    Ok(cutoffgrid[cutoffgrid.len() - 1])
}

/// Compute optimal q for unimodal assumption.
fn optimal_q_unimodal(p: usize, cutoff: f64, pfer: f64, b: usize) -> usize {
    let upper_q = if cutoff <= 0.75 {
        let v1 = p as f64 * (cutoff - 0.5).max(0.0).sqrt();
        let v2 = p as f64
            * (4.0 / 3.0 * (cutoff - 0.5 - 1.0 / (2.0 * b as f64)))
                .max(0.0)
                .sqrt();
        (v1.max(v2).ceil() as usize).saturating_sub(1)
    } else {
        p
    };
    let f = |q_val: f64| -> f64 {
        q_val.powi(2) / p as f64 / um_const(cutoff, b, q_val / p as f64) - pfer
    };
    let lo = 1.0f64;
    let hi = upper_q as f64;
    if f(lo) > 0.0 {
        return 1;
    }
    if f(hi) < 0.0 {
        return hi as usize;
    }
    let mut lo = lo;
    let mut hi = hi;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    ((lo + hi) / 2.0).floor() as usize
}

/// Compute optimal cutoff for r-concave assumption.
fn optimal_cutoff_rconcave(p: usize, q: usize, pfer: f64, b: usize) -> Result<f64, MboostrsError> {
    let mut cutoff_candidates: Vec<f64> = (0..2 * b)
        .map(|k| (2 * b - k) as f64 / (2 * b) as f64)
        .collect();
    cutoff_candidates.retain(|&c| c > q as f64 / p as f64);
    if cutoff_candidates.is_empty() {
        return Err(MboostrsError::StabSel(format!(
            "cutoff must be larger than q/p = {}",
            q as f64 / p as f64
        )));
    }
    for cutoff in cutoff_candidates.iter().rev() {
        if let Ok(bound) = try_min_d(q, p, *cutoff, b).map(|value| value * p as f64) {
            if bound <= pfer {
                return Ok(*cutoff);
            }
        }
    }
    Ok(cutoff_candidates[0])
}

/// Compute optimal q for r-concave assumption.
fn optimal_q_rconcave(p: usize, cutoff: f64, pfer: f64, b: usize) -> usize {
    let mq = try_max_q(p, b).unwrap_or(0);
    let mut best_q = 1usize;
    for q in 1..=mq {
        if cutoff <= q as f64 / p as f64 {
            break;
        }
        if let Ok(bound) = try_min_d(q, p, cutoff, b).map(|value| value * p as f64) {
            if bound <= pfer {
                best_q = q;
            } else {
                break;
            }
        }
    }
    best_q
}

/// Compute stability selection parameters.
///
/// Exactly two of `cutoff`, `q`, and `pfer` must be `Some`; the third is computed.
///
/// Reimplements R's `stabs::stabsel_parameters`.
pub fn stabsel_parameters(
    p: usize,
    cutoff: Option<f64>,
    q: Option<usize>,
    pfer: Option<f64>,
    b: usize,
    assumption: &str,
    sampling_type: &str,
) -> Result<StabSelParams, MboostrsError> {
    if p == 0 {
        return Err(MboostrsError::StabSel("p must be greater than 0".into()));
    }
    if b <= 1 {
        return Err(MboostrsError::StabSel("B must be at least 2".into()));
    }
    if !matches!(sampling_type, "SS" | "MB") {
        return Err(MboostrsError::StabSel(
            "sampling_type must be 'SS' or 'MB'".into(),
        ));
    }
    if !matches!(assumption, "none" | "unimodal" | "r-concave") {
        return Err(MboostrsError::StabSel(
            "assumption must be 'none', 'unimodal', or 'r-concave'".into(),
        ));
    }
    let n_some = [cutoff.is_some(), q.is_some(), pfer.is_some()]
        .iter()
        .filter(|&&x| x)
        .count();
    if n_some != 2 {
        if n_some > 2 {
            return Err(MboostrsError::StabSel(
                "Only two of cutoff, q, and PFER can be specified at the same time".into(),
            ));
        }
        return Err(MboostrsError::StabSel(
            "Two of cutoff, q, and PFER must be specified".into(),
        ));
    }

    if let Some(pf) = pfer {
        if !pf.is_finite() || pf <= 0.0 {
            return Err(MboostrsError::StabSel("PFER must be greater than 0".into()));
        }
    }
    if let Some(c) = cutoff {
        if !c.is_finite() {
            return Err(MboostrsError::StabSel("cutoff must be finite".into()));
        }
        if assumption != "r-concave" && !(0.5..=1.0).contains(&c) {
            return Err(MboostrsError::StabSel(
                "cutoff must be between 0.5 and 1".into(),
            ));
        }
        if assumption == "r-concave" && (c < 0.0 || c > 1.0) {
            return Err(MboostrsError::StabSel(
                "cutoff must be between 0 and 1".into(),
            ));
        }
    }
    if let Some(qv) = q {
        if p < qv {
            return Err(MboostrsError::StabSel(format!(
                "q ({qv}) must be smaller than the number of effects ({p})"
            )));
        }
        if qv == 0 {
            return Err(MboostrsError::StabSel("q must be greater than 0".into()));
        }
    }

    let assumption = if sampling_type == "MB" {
        "none"
    } else {
        assumption
    };

    let (cutoff_out, q_out, upperbound) = match (cutoff, q, pfer) {
        (None, Some(q_val), Some(pf)) => {
            if assumption == "none" {
                let c = ((q_val as f64).powi(2) / (pf * p as f64) + 1.0) / 2.0;
                let c = c.min(1.0);
                let ub = (q_val as f64).powi(2) / p as f64 / (2.0 * c - 1.0);
                (c, q_val, ub)
            } else if assumption == "unimodal" {
                let c = optimal_cutoff_unimodal(p, q_val, pf, b)?;
                let ub = (q_val as f64).powi(2)
                    / p as f64
                    / try_um_const(c, b, q_val as f64 / p as f64)?;
                (c, q_val, ub)
            } else {
                let c = optimal_cutoff_rconcave(p, q_val, pf, b)?;
                let ub = try_min_d(q_val, p, c, b)? * p as f64;
                (c, q_val, ub)
            }
        }
        (Some(c), None, Some(pf)) => {
            if assumption == "none" {
                let q_val = (pf * (2.0 * c - 1.0) * p as f64).max(0.0).sqrt().floor() as usize;
                let ub = (q_val as f64).powi(2) / p as f64 / (2.0 * c - 1.0);
                (c, q_val, ub)
            } else if assumption == "unimodal" {
                let q_val = optimal_q_unimodal(p, c, pf, b);
                let ub = (q_val as f64).powi(2)
                    / p as f64
                    / try_um_const(c, b, q_val as f64 / p as f64)?;
                (c, q_val, ub)
            } else {
                let q_val = optimal_q_rconcave(p, c, pf, b);
                let ub = try_min_d(q_val, p, c, b)? * p as f64;
                (c, q_val, ub)
            }
        }
        (Some(c), Some(q_val), None) => {
            if assumption == "none" {
                let ub = (q_val as f64).powi(2) / p as f64 / (2.0 * c - 1.0);
                (c, q_val, ub)
            } else if assumption == "unimodal" {
                let ub = (q_val as f64).powi(2)
                    / p as f64
                    / try_um_const(c, b, q_val as f64 / p as f64)?;
                (c, q_val, ub)
            } else {
                let ub = try_min_d(q_val, p, c, b)? * p as f64;
                (c, q_val, ub)
            }
        }
        _ => unreachable!("exactly two of cutoff, q, pfer must be Some"),
    };

    if let Some(c) = cutoff {
        if assumption == "r-concave" && c <= q_out as f64 / p as f64 {
            return Err(MboostrsError::StabSel(
                "cutoff must be larger than q/p".into(),
            ));
        }
    }

    let upperbound = if upperbound == 0.0 || !upperbound.is_finite() {
        // cutoff=0.5 under the MB/no-assumption bound has a zero
        // denominator and therefore an explicitly unbounded PFER.
        // Preserve infinity instead of attempting decimal rounding.
        upperbound
    } else {
        let d = 3 - upperbound.abs().log10().floor() as i32 - 1;
        let factor = 10f64.powi(d);
        (upperbound * factor).round() / factor
    };

    Ok(StabSelParams {
        cutoff: cutoff_out,
        q: q_out,
        pfer: upperbound,
        p,
        b,
        sampling_type: sampling_type.to_string(),
        assumption: assumption.to_string(),
    })
}

/// Selection frequencies of a stability-selection run.
///
/// `paths[b]` is subsample `b`'s selection sequence: the 0-based learner
/// index chosen at each boosting iteration. An empty path is a subsample
/// whose fit failed; it selects nothing but still counts in the
/// denominator. Only the first `q` distinct learners of a path count, as in
/// `stabs`: a learner that first appears after `q` others have appeared is
/// treated as never selected. (`stabs` rewrites every entry from that point
/// on to the path's first learner, which has the same effect.) The first
/// learner of a non-empty path always counts, so `q = 0` behaves as `q = 1`.
///
/// Returns the `n_learners x mstop` matrix whose entry `[l, m]` is the
/// fraction of subsamples in which learner `l` was among those counted and
/// first appeared at iteration `m + 1` or earlier. Each row is
/// non-decreasing in `m`.
pub fn try_selection_frequencies(
    paths: &[Vec<usize>],
    n_learners: usize,
    mstop: usize,
    q: usize,
) -> Result<ndarray::Array2<f64>, MboostrsError> {
    if paths.is_empty() {
        return Err(MboostrsError::Control(
            "selection frequencies need at least one subsample path".into(),
        ));
    }
    let keep = q.max(1);
    let mut counts = ndarray::Array2::<f64>::zeros((n_learners, mstop));
    let mut seen = vec![false; n_learners];
    for (b, path) in paths.iter().enumerate() {
        seen.iter_mut().for_each(|s| *s = false);
        let mut distinct = 0usize;
        for (m, &learner) in path.iter().enumerate() {
            if learner >= n_learners {
                return Err(MboostrsError::Shape(format!(
                    "subsample {} selects learner {learner}, outside 0..{n_learners}",
                    b + 1
                )));
            }
            if seen[learner] {
                continue;
            }
            seen[learner] = true;
            distinct += 1;
            if distinct > keep {
                break;
            }
            for cell in counts.row_mut(learner).iter_mut().skip(m) {
                *cell += 1.0;
            }
        }
    }
    counts.mapv_inplace(|c| c / paths.len() as f64);
    Ok(counts)
}

/// Selection frequencies and fit-failure count of a stability-selection run.
#[derive(Clone, Debug)]
pub struct StabilityFrequencies {
    /// `n_learners x mstop` selection frequencies; see
    /// [`try_selection_frequencies`].
    pub phat: ndarray::Array2<f64>,
    /// Each learner's largest frequency over iterations (its last column).
    pub max: Vec<f64>,
    /// Number of subsample fits that failed and selected nothing.
    pub failed: usize,
}

/// Stability selection over the in-bag weight columns of `folds`: fits the
/// model once per column, keeps each fit's selection path, and returns the
/// selection frequencies of [`try_selection_frequencies`]. A fit that fails
/// counts as a subsample that selected nothing and is reported in `failed`.
/// The offset arguments are those of [`crate::cv::try_stability_paths_with`].
#[allow(clippy::too_many_arguments)]
pub fn try_stability_selection_with<F: crate::families::Family + Clone>(
    preparer: &dyn crate::cv::LearnerPreparer,
    y: ndarray::ArrayView1<f64>,
    weights: ndarray::ArrayView1<f64>,
    control: &crate::control::BoostControl,
    folds: &ndarray::Array2<f64>,
    offset_override: Option<f64>,
    offset_vec: Option<ndarray::ArrayView1<f64>>,
    n_learners: usize,
    q: usize,
    family_template: F,
) -> Result<StabilityFrequencies, MboostrsError> {
    let per_fold = crate::cv::try_stability_paths_with(
        preparer,
        y,
        weights,
        control,
        folds,
        offset_override,
        offset_vec,
        family_template,
    )?;
    let failed = per_fold.iter().filter(|p| p.is_none()).count();
    let paths: Vec<Vec<usize>> = per_fold
        .into_iter()
        .map(Option::unwrap_or_default)
        .collect();
    let phat = try_selection_frequencies(&paths, n_learners, control.mstop as usize, q)?;
    let max = phat
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().fold(0.0, f64::max))
        .collect();
    Ok(StabilityFrequencies { phat, max, failed })
}

#[cfg(test)]
mod tests {
    #[test]
    fn min_d_cutoff_uses_signif_10_like_r() {
        // R: ceiling(signif(0.56 / (1 / (2 * 50)), 10)) = 56, while the
        // unrounded product 56.00000000000001 ceils to 57.
        assert_eq!(signif10(0.56 / (1.0 / (2.0 * 50.0))), 56.0);
        assert_eq!(signif10(0.56 / (1.0 / (2.0 * 50.0))).ceil() as usize, 56);
        assert!(try_min_d(2, 20, 0.56, 50).is_ok());
    }
    use super::*;

    #[test]
    fn test_brentq_simple() {
        let root = brentq(|x| x.powi(3) - 1.0, 0.0, 2.0, 1e-12, 1e-12);
        assert!(root.is_some());
        assert!((root.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_brentq_no_bracket() {
        let root = brentq(|x| x.powi(2) + 1.0, -1.0, 1.0, 1e-12, 1e-12);
        assert!(root.is_none());
    }

    #[test]
    fn test_max_q() {
        assert_eq!(max_q(3, 50), 1);
        assert_eq!(max_q(10, 100), 4);
        assert_eq!(max_q(5, 20), 2);
    }

    #[test]
    fn test_um_const() {
        // cutoff <= 0.75
        let val = um_const(0.6, 50, 0.1);
        let expected = 2.0 * (2.0 * 0.6 - 1.0 - 1.0 / 100.0);
        assert!((val - expected).abs() < 1e-10);
        // cutoff > 0.75
        let val = um_const(0.9, 50, 0.3);
        let expected = (1.0 + 1.0 / 50.0) / (4.0 * (1.0 - 0.9 + 1.0 / 100.0));
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_d_rc_basic() {
        // which < k_start => return 1.0
        let val = d_rc(0.09, 2, 50, -0.5);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minimize_bounded_parabola() {
        let min = minimize_bounded(|x| (x - 3.0).powi(2), 0.0, 10.0);
        assert!(min.is_some());
        assert!((min.unwrap() - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_d_rc_vs_python() {
        assert!((d_rc(0.09, 10, 50, -0.5) - 0.162497501872).abs() < 1e-4);
        assert!((d_rc(0.09, 10, 50, -0.25) - 0.151660893460).abs() < 1e-4);
        assert!((d_rc(0.05, 5, 20, -0.5) - 0.066356632774).abs() < 1e-4);
    }

    #[test]
    fn test_d_rc_rconcave_quarter() {
        let val = d_rc(0.09, 10, 50, -0.25);
        assert!(val >= 0.0, "d_rc should be non-negative, got {val}");
        assert!(val <= 1.0, "d_rc should be <= 1, got {val}");
    }

    #[test]
    fn test_min_d_basic() {
        let val = min_d(3, 10, 0.6, 50);
        assert!(
            (val - 0.162497501872).abs() < 1e-4,
            "min_d(3,10,0.6,50) should be ~0.1625, got {val}"
        );
        let val2 = min_d(3, 10, 0.7, 50);
        assert!(
            (val2 - 0.057560316296).abs() < 1e-3,
            "min_d(3,10,0.7,50) should be ~0.0576, got {val2}"
        );
    }

    #[test]
    fn test_stabsel_parameters_unimodal_cutoff_none() {
        let result = stabsel_parameters(3, None, Some(1), Some(1.0), 50, "unimodal", "SS").unwrap();
        assert!(result.cutoff > 0.5);
        assert_eq!(result.q, 1);
    }

    #[test]
    fn test_stabsel_parameters_unimodal_q_none() {
        let result =
            stabsel_parameters(10, Some(0.8), None, Some(2.0), 50, "unimodal", "SS").unwrap();
        assert!((result.cutoff - 0.8).abs() < 1e-10);
        assert!(result.q > 0);
    }

    #[test]
    fn test_stabsel_parameters_none_assumption() {
        let result = stabsel_parameters(10, None, Some(3), Some(2.0), 50, "none", "SS").unwrap();
        assert!(result.cutoff > 0.5);
    }

    #[test]
    fn test_stabsel_parameters_mb_sampling() {
        let result =
            stabsel_parameters(10, None, Some(3), Some(2.0), 100, "unimodal", "MB").unwrap();
        assert_eq!(result.assumption, "none");
    }

    #[test]
    fn test_stabsel_parameters_error_two_missing() {
        let result = stabsel_parameters(10, None, None, Some(1.0), 50, "unimodal", "SS");
        assert!(result.is_err());
    }

    #[test]
    fn test_stabsel_parameters_error_all_three() {
        let result = stabsel_parameters(10, Some(0.8), Some(2), Some(1.0), 50, "unimodal", "SS");
        assert!(result.is_err());
    }

    #[test]
    fn test_stabsel_parameters_error_negative_pfer() {
        let result = stabsel_parameters(10, None, Some(2), Some(-1.0), 50, "unimodal", "SS");
        assert!(result.is_err());
    }

    #[test]
    fn test_stabsel_parameters_rconcave_cutoff_none() {
        let result =
            stabsel_parameters(10, None, Some(3), Some(2.0), 50, "r-concave", "SS").unwrap();
        assert!(result.cutoff > 3.0 / 10.0);
    }

    #[test]
    fn test_stabsel_parameters_rconcave_q_none() {
        let result =
            stabsel_parameters(10, Some(0.8), None, Some(2.0), 50, "r-concave", "SS").unwrap();
        assert!((result.cutoff - 0.8).abs() < 1e-10);
        assert!(result.q > 0);
    }

    #[test]
    fn test_stabsel_parameters_rconcave_pfer_none() {
        let result =
            stabsel_parameters(10, Some(0.8), Some(3), None, 50, "r-concave", "SS").unwrap();
        assert!((result.cutoff - 0.8).abs() < 1e-10);
        assert_eq!(result.q, 3);
        assert!(result.pfer >= 0.0);
    }

    #[test]
    fn test_q_truncation_after_q_unique() {
        let sel = vec![2, 3, 1, 2, 3, 1, 0, 4];
        let q_val = 2;
        let mut truncated = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut threshold_reached = false;
        for &idx in &sel {
            if threshold_reached {
                truncated.push(sel[0]);
            } else {
                if seen.insert(idx) {
                    if seen.len() > q_val {
                        threshold_reached = true;
                        truncated.push(sel[0]);
                    } else {
                        truncated.push(idx);
                    }
                } else {
                    truncated.push(idx);
                }
            }
        }
        assert_eq!(truncated, vec![2, 3, 2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_q_truncation_simple() {
        let sel = vec![0, 1, 2, 3];
        let q_val = 2;
        let mut truncated = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut threshold_reached = false;
        for &idx in &sel {
            if threshold_reached {
                truncated.push(sel[0]);
            } else {
                if seen.insert(idx) {
                    if seen.len() > q_val {
                        threshold_reached = true;
                        truncated.push(sel[0]);
                    } else {
                        truncated.push(idx);
                    }
                } else {
                    truncated.push(idx);
                }
            }
        }
        assert_eq!(truncated, vec![0, 1, 0, 0]);
    }

    #[test]
    fn test_q_truncation_at_threshold() {
        let sel = vec![0, 1, 0, 1, 0];
        let q_val = 2;
        let mut truncated = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut threshold_reached = false;
        for &idx in &sel {
            if threshold_reached {
                truncated.push(sel[0]);
            } else {
                if seen.insert(idx) {
                    if seen.len() > q_val {
                        threshold_reached = true;
                        truncated.push(sel[0]);
                    } else {
                        truncated.push(idx);
                    }
                } else {
                    truncated.push(idx);
                }
            }
        }
        assert_eq!(truncated, vec![0, 1, 0, 1, 0]);
    }

    /// The R `.lss_stabsel_frequencies` loop as it was written before the
    /// port: truncate each path after `q` distinct learners by rewriting the
    /// rest to its first entry, then add one from each learner's first
    /// appearance onward.
    fn frequencies_by_truncation(paths: &[Vec<usize>], p: usize, m: usize, q: usize) -> Vec<f64> {
        let mut phat = vec![0.0; p * m];
        for sel in paths {
            if sel.is_empty() {
                continue;
            }
            let mut seen: Vec<usize> = Vec::new();
            let mut trunc = Vec::with_capacity(sel.len());
            let mut stop = false;
            for &idx in sel {
                if stop {
                    trunc.push(sel[0]);
                } else if !seen.contains(&idx) {
                    seen.push(idx);
                    if seen.len() > q {
                        stop = true;
                        trunc.push(sel[0]);
                    } else {
                        trunc.push(idx);
                    }
                } else {
                    trunc.push(idx);
                }
            }
            for learner in 0..p {
                if let Some(first) = trunc.iter().position(|&s| s == learner) {
                    for col in first..m {
                        phat[learner * m + col] += 1.0;
                    }
                }
            }
        }
        phat.iter().map(|v| v / paths.len() as f64).collect()
    }

    #[test]
    fn selection_frequencies_match_truncation_rule() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(848);
        let (p, m) = (7usize, 12usize);
        for q in 0..=p {
            let paths: Vec<Vec<usize>> = (0..25)
                .map(|b| {
                    if b % 9 == 4 {
                        Vec::new() // a failed fit
                    } else {
                        (0..m).map(|_| rng.random_range(0..p)).collect()
                    }
                })
                .collect();
            let got = try_selection_frequencies(&paths, p, m, q).unwrap();
            let want = frequencies_by_truncation(&paths, p, m, q);
            assert_eq!(got.as_slice().unwrap(), want.as_slice(), "q = {q}");
        }
    }

    #[test]
    fn selection_frequencies_hand_example() {
        // Path 1 counts learners 2 (from step 1) and 0 (from step 2); learner
        // 1 is its third distinct learner and q = 2 drops it. Path 2 counts 1
        // from step 1. Path 3 failed.
        let paths = vec![vec![2, 0, 2, 1], vec![1, 1, 1, 1], vec![]];
        let phat = try_selection_frequencies(&paths, 3, 4, 2).unwrap();
        let third = 1.0 / 3.0;
        assert_eq!(phat.row(0).to_vec(), vec![0.0, third, third, third]);
        assert_eq!(phat.row(1).to_vec(), vec![third; 4]);
        assert_eq!(phat.row(2).to_vec(), vec![third; 4]);
        assert!(try_selection_frequencies(&paths, 2, 4, 2).is_err());
        assert!(try_selection_frequencies(&[], 2, 4, 2).is_err());
    }

    #[test]
    fn stability_selection_finds_the_signal_column() {
        use crate::control::BoostControl;
        use crate::cv::LearnerPreparer;
        use crate::families::{Family, Gaussian};
        use crate::learners::{prepare_scalar_bols, PreparedLearner};
        use ndarray::{Array1, Array2, ArrayView1};
        use rand::{RngExt, SeedableRng};

        struct Columns(Vec<Array1<f64>>);
        impl LearnerPreparer for Columns {
            fn prepare(&self, weights: ArrayView1<f64>) -> Vec<PreparedLearner> {
                self.0
                    .iter()
                    .map(|c| prepare_scalar_bols(c.view(), weights, 0.0, true, "x", None))
                    .collect()
            }
            fn family(&self) -> Box<dyn Family> {
                Box::new(Gaussian)
            }
        }

        let n = 60;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let cols: Vec<Array1<f64>> = (0..4)
            .map(|_| Array1::from_shape_fn(n, |_| rng.random::<f64>() - 0.5))
            .collect();
        let y = Array1::from_shape_fn(n, |i| 3.0 * cols[2][i] + 0.1 * (rng.random::<f64>() - 0.5));
        // Two half-samples and their complements.
        let mut folds = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            folds[[i, 0]] = (i % 2) as f64;
            folds[[i, 1]] = (i < n / 2) as u8 as f64;
            folds[[i, 2]] = 1.0 - folds[[i, 0]];
            folds[[i, 3]] = 1.0 - folds[[i, 1]];
        }
        let control = BoostControl::new(20, 0.1).unwrap();
        let preparer = Columns(cols);
        let ones = Array1::ones(n);
        let result = try_stability_selection_with(
            &preparer,
            y.view(),
            ones.view(),
            &control,
            &folds,
            None,
            None,
            4,
            1,
            Gaussian,
        )
        .unwrap();
        assert_eq!(result.failed, 0);
        assert_eq!(result.phat.dim(), (4, 20));
        assert_eq!(result.max, vec![0.0, 0.0, 1.0, 0.0]);
        // The same paths, taken directly, give the same frequencies.
        let paths = crate::cv::try_stability_paths_with(
            &preparer,
            y.view(),
            ones.view(),
            &control,
            &folds,
            None,
            None,
            Gaussian,
        )
        .unwrap();
        let paths: Vec<Vec<usize>> = paths.into_iter().map(Option::unwrap).collect();
        assert_eq!(
            try_selection_frequencies(&paths, 4, 20, 1).unwrap(),
            result.phat
        );
    }
}
