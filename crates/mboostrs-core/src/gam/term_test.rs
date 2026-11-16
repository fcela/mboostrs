//! Approximate p-values for the terms of a penalized refit (Wood 2013).
//!
//! A smooth term's coefficients `b` have Bayesian covariance `V`. Testing
//! `f = 0` with the Wald statistic `b' V^- b` needs a rank for the
//! pseudo-inverse; using the full rank overstates the evidence of a heavily
//! penalized term and using a rounded edf discards information. Wood (2013,
//! Biometrika 100) truncates the pseudo-inverse at the term's alternative
//! effective degrees of freedom `edf1 = 2 edf - tr(F F)` (F the influence
//! matrix restricted to the term), with a fractional rank handled by
//! blending the last two eigen-directions, and refers the statistic to a
//! weighted sum of chi-square variables (or of F variables when the scale is
//! estimated). This module follows mgcv's `testStat` (mgcv 1.9, `type = 0`),
//! including its averaging of the two equivalent blended statistics, so the
//! p-values agree with `summary.gam` up to the tail approximations.
//!
//! The term's design enters through `R`, any factor with `R'R = X_j' W X_j`:
//! mgcv takes `R` from a QR decomposition, this module from the
//! eigendecomposition of `X_j' W X_j`. The two differ by an orthogonal
//! transformation, which rotates the eigenvectors of `R V R'` and leaves the
//! plain statistic unchanged; with a fractional rank it can exchange the two
//! blended statistics, whose p-values mgcv averages, so the averaged p-value
//! is unchanged too.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// The test of one term.
#[derive(Clone, Debug, PartialEq)]
pub struct TermTest {
    /// Rank the statistic was computed at (`min(columns, edf1)`), mgcv's
    /// `Ref.df`.
    pub ref_df: f64,
    /// The Wald-type statistic `b' V_r^- b`; reported as `F = stat / ref_df`
    /// when the scale is estimated and as a chi-square otherwise.
    pub statistic: f64,
    /// Approximate p-value for `f = 0`.
    pub p_value: f64,
}

/// Wood's (2013) test of `b = 0` for a term with coefficients `b`,
/// covariance `v`, weighted cross-product `xtwx = X'WX` restricted to the
/// term, at rank `rank`. `res_df` is the residual degrees of freedom when the
/// scale was estimated, or `None` for a known scale.
pub fn term_test(
    b: ArrayView1<f64>,
    v: ArrayView2<f64>,
    xtwx: ArrayView2<f64>,
    rank: f64,
    res_df: Option<f64>,
) -> TermTest {
    let q = b.len();
    let r = design_factor(xtwx);
    // V in the space of fitted values: R V R'.
    let mut m = r.dot(&v).dot(&r.t());
    for i in 0..q {
        for j in 0..i {
            let s = 0.5 * (m[[i, j]] + m[[j, i]]);
            m[[i, j]] = s;
            m[[j, i]] = s;
        }
    }
    let (values, mut vectors) = symmetric_eigen_desc(&m);
    // Fix each eigenvector's sign by its first entry, as mgcv does.
    for c in 0..q {
        if vectors[[0, c]] < 0.0 {
            vectors.column_mut(c).mapv_inplace(|x| -x);
        }
    }

    let mut k = rank.floor().max(0.0) as usize;
    let mut nu = (rank - k as f64).abs();
    let mut rank = rank;
    let mut k1 = if nu > 0.0 { k + 1 } else { k };
    let top = values.first().copied().unwrap_or(0.0);
    let r_est = values
        .iter()
        .filter(|&&e| e > top * f64::EPSILON.powf(0.9))
        .count();
    if r_est < k1 {
        k1 = r_est;
        k = r_est;
        nu = 0.0;
        rank = r_est as f64;
    }
    if k1 == 0 {
        return TermTest {
            ref_df: 0.0,
            statistic: 0.0,
            p_value: 1.0,
        };
    }
    let k1 = k1.min(q);
    let mut vec = vectors.slice(ndarray::s![.., ..k1]).to_owned();
    let mut vec1;
    if nu > 0.0 && k > 0 {
        for c in 0..k - 1 {
            let s = values[c].sqrt();
            vec.column_mut(c).mapv_inplace(|x| x / s);
        }
        let b12 = (0.5 * nu * (1.0 - nu)).max(0.0).sqrt();
        let e0 = values[k - 1].powf(-0.5);
        let e1 = values[k1 - 1].powf(-0.5);
        // B = ev [[1, b12], [b12, nu]] ev, and rB its symmetric square root.
        let bm = [[e0 * e0, e0 * b12 * e1], [e1 * b12 * e0, e1 * nu * e1]];
        let rb = sqrt_sym2(bm);
        vec1 = vec.clone();
        for row in 0..q {
            let a = vec[[row, k - 1]];
            let c = vec[[row, k1 - 1]];
            // vec1[, k:k1] = t(rB %*% diag(c(-1, 1)) %*% t(vec[, k:k1]))
            vec1[[row, k - 1]] = rb[0][0] * (-a) + rb[0][1] * c;
            vec1[[row, k1 - 1]] = rb[1][0] * (-a) + rb[1][1] * c;
            // vec[, k:k1] = t(rB %*% t(vec[, k:k1]))
            vec[[row, k - 1]] = rb[0][0] * a + rb[0][1] * c;
            vec[[row, k1 - 1]] = rb[1][0] * a + rb[1][1] * c;
        }
    } else {
        if k == 0 {
            let s = values[0].sqrt();
            vec.mapv_inplace(|x| x / s);
        } else {
            for c in 0..k.min(k1) {
                let s = values[c].sqrt();
                vec.column_mut(c).mapv_inplace(|x| x / s);
            }
        }
        vec1 = vec.clone();
        if k == 1 {
            rank = 1.0;
        }
    }
    let rb_coef = r.dot(&b);
    let d: f64 = vec.t().dot(&rb_coef).iter().map(|x| x * x).sum();
    let d1: f64 = vec1.t().dot(&rb_coef).iter().map(|x| x * x).sum();

    let mut rank1 = rank;
    let mut pval = 2.0;
    if nu > 0.0 {
        let weights: Vec<f64> = if k1 == 1 {
            rank1 = 1.0;
            vec![1.0]
        } else {
            let mut w = vec![1.0; k1];
            let rp = nu + 1.0;
            w[k - 1] = (rp + (rp * (2.0 - rp)).sqrt()) / 2.0;
            w[k1 - 1] = rp - w[k - 1];
            w
        };
        let h = vec![1.0; weights.len()];
        let tail = |stat: f64| -> f64 {
            match res_df {
                None => imhof_upper(stat, &weights, &h).unwrap_or_else(|| liu2(stat, &weights)),
                Some(df) => {
                    // P(sum w_i chi2_1 > stat chi2_k0 / k0): a quadratic form
                    // with one negative weight, evaluated at zero.
                    let k0 = df.round().max(1.0);
                    let mut w = weights.clone();
                    w.push(-stat / k0);
                    let mut hk = h.clone();
                    hk.push(k0);
                    imhof_upper(0.0, &w, &hk).unwrap_or_else(|| simf(stat, &weights, df))
                }
            }
        };
        pval = 0.5 * (tail(d) + tail(d1));
    }
    if pval > 1.0 {
        pval = match res_df {
            None => 0.5 * (chisq_upper(d, rank1) + chisq_upper(d1, rank1)),
            Some(df) => 0.5 * (f_upper(d / rank1, rank1, df) + f_upper(d1 / rank1, rank1, df)),
        };
    }
    TermTest {
        ref_df: rank,
        statistic: d,
        p_value: pval.min(1.0),
    }
}

/// A factor `R` with `R'R = a` for a symmetric positive semi-definite `a`:
/// `R = L^1/2 U'` from `a = U L U'`. A basis function with no data under it
/// makes `a` singular; its direction gets a zero row, as it gets a zero row
/// in mgcv's QR factor.
fn design_factor(a: ArrayView2<f64>) -> Array2<f64> {
    let (values, vectors) = symmetric_eigen_desc(&a.to_owned());
    let n = values.len();
    Array2::from_shape_fn((n, n), |(i, j)| values[i].max(0.0).sqrt() * vectors[[j, i]])
}

/// Eigenvalues (descending) and eigenvectors (columns) of a symmetric matrix.
fn symmetric_eigen_desc(m: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = m.nrows();
    let mat = faer::Mat::<f64>::from_fn(n, n, |i, j| m[[i, j]]);
    let evd = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = evd.s();
    let u = evd.u();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        s.column_vector()
            .read(b)
            .total_cmp(&s.column_vector().read(a))
    });
    let values = order.iter().map(|&i| s.column_vector().read(i)).collect();
    let vectors = Array2::from_shape_fn((n, n), |(i, c)| u.read(i, order[c]));
    (values, vectors)
}

/// Symmetric square root of a 2 x 2 symmetric positive semi-definite matrix.
fn sqrt_sym2(a: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let (p, q, r) = (a[0][0], a[0][1], a[1][1]);
    let mean = 0.5 * (p + r);
    let rad = (0.25 * (p - r) * (p - r) + q * q).sqrt();
    let (l1, l2) = (mean + rad, (mean - rad).max(0.0));
    // Eigenvector of l1.
    let (vx, vy) = if q.abs() > 1e-300 {
        let (x, y) = (l1 - r, q);
        let norm = (x * x + y * y).sqrt();
        (x / norm, y / norm)
    } else if p >= r {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let (s1, s2) = (l1.sqrt(), l2.sqrt());
    // sqrt(A) = s1 v v' + s2 w w', w orthogonal to v.
    [
        [s1 * vx * vx + s2 * vy * vy, (s1 - s2) * vx * vy],
        [(s1 - s2) * vx * vy, s1 * vy * vy + s2 * vx * vx],
    ]
}

/// `P(sum_j lambda_j chi^2_{h_j} > x)` by Imhof's (1961) inversion of the
/// characteristic function,
/// `1/2 + (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du` with
/// `theta(u) = 1/2 sum_j h_j atan(lambda_j u) - x u / 2` and
/// `rho(u) = prod_j (1 + lambda_j^2 u^2)^(h_j / 4)`. Weights may be negative.
/// mgcv evaluates the same probability with Davies' (1980) algorithm at an
/// absolute tolerance of 2e-5; the integral here is truncated where Imhof's
/// bound puts the remainder below 1e-10 and integrated adaptively. When
/// that truncation point is too far out to integrate (few degrees of
/// freedom with a very small weight, so the integrand decays slowly while
/// it oscillates), `None` is returned and the caller uses the Liu
/// approximation.
pub fn imhof_upper(x: f64, lambda: &[f64], h: &[f64]) -> Option<f64> {
    // Truncation (Imhof 1961, eq. 3.4): the integral beyond U is at most
    // 2 / (pi k U^k prod |lambda_j|^(h_j / 2)) with k = sum h_j / 2.
    let k = 0.5 * h.iter().sum::<f64>();
    let log_prod: f64 = lambda
        .iter()
        .zip(h)
        .map(|(l, hj)| 0.5 * hj * l.abs().ln())
        .sum();
    let tol = 1e-10;
    let log_u = ((2.0 / (std::f64::consts::PI * k * tol)).ln() - log_prod) / k;
    let upper = log_u.exp().max(1.0);
    // Oscillation periods of sin(x u / 2 + ...) over [0, U].
    let periods = upper * (x.abs() + lambda.iter().map(|l| l.abs()).sum::<f64>())
        / (4.0 * std::f64::consts::PI);
    if !upper.is_finite() || periods > 2.0e4 {
        return None;
    }
    let slope0 = 0.5 * lambda.iter().zip(h).map(|(l, hj)| l * hj).sum::<f64>() - 0.5 * x;
    let integrand = |u: f64| -> f64 {
        if u <= 0.0 {
            return slope0;
        }
        let mut theta = -0.5 * x * u;
        let mut log_rho = 0.0;
        for (l, hj) in lambda.iter().zip(h) {
            theta += 0.5 * hj * (l * u).atan();
            log_rho += 0.25 * hj * (l * l * u * u).ln_1p();
        }
        theta.sin() / (u * log_rho.exp())
    };
    // Integrate over [0, U] in pieces of about one oscillation period each,
    // so every Gauss-Kronrod rule sees a smooth, slowly varying integrand.
    let pieces = (periods.ceil() as usize).clamp(8, 40_000);
    let width = upper / pieces as f64;
    let mut integral = 0.0;
    for i in 0..pieces {
        let a = i as f64 * width;
        integral += adaptive_gk(&integrand, a, a + width, tol / pieces as f64, 50);
    }
    Some((0.5 + integral / std::f64::consts::PI).clamp(0.0, 1.0))
}

/// One 7-15 point Gauss-Kronrod rule on `[a, b]`: `(estimate, error)`.
fn gk15(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> (f64, f64) {
    const XK: [f64; 8] = [
        0.991_455_371_120_812_6,
        0.949_107_912_342_758_5,
        0.864_864_423_359_769_1,
        0.741_531_185_599_394_4,
        0.586_087_235_467_691_1,
        0.405_845_151_377_397_2,
        0.207_784_955_007_898_5,
        0.0,
    ];
    const WK: [f64; 8] = [
        0.022_935_322_010_529_22,
        0.063_092_092_629_978_55,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_728,
    ];
    const WG: [f64; 4] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
    ];
    let c = 0.5 * (a + b);
    let hw = 0.5 * (b - a);
    let fc = f(c);
    let mut kron = WK[7] * fc;
    let mut gauss = WG[3] * fc;
    for i in 0..7 {
        let dx = hw * XK[i];
        let (f1, f2) = (f(c - dx), f(c + dx));
        kron += WK[i] * (f1 + f2);
        if i % 2 == 1 {
            gauss += WG[i / 2] * (f1 + f2);
        }
    }
    (kron * hw, ((kron - gauss) * hw).abs())
}

/// Globally adaptive Gauss-Kronrod quadrature over `[a, b]`: the interval
/// with the largest error estimate is bisected until the summed error falls
/// below `tol` or `max_intervals` is reached (the QUADPACK `qag` strategy).
fn adaptive_gk(f: &dyn Fn(f64) -> f64, a: f64, b: f64, tol: f64, max_intervals: usize) -> f64 {
    let (value, error) = gk15(f, a, b);
    let mut parts = vec![(a, b, value, error)];
    loop {
        let total_error: f64 = parts.iter().map(|p| p.3).sum();
        if total_error <= tol || parts.len() >= max_intervals {
            break;
        }
        let worst = parts
            .iter()
            .enumerate()
            .max_by(|x, y| x.1 .3.total_cmp(&y.1 .3))
            .map(|(i, _)| i)
            .expect("at least one interval");
        let (lo, hi, _, _) = parts.swap_remove(worst);
        let mid = 0.5 * (lo + hi);
        let (v1, e1) = gk15(f, lo, mid);
        let (v2, e2) = gk15(f, mid, hi);
        parts.push((lo, mid, v1, e1));
        parts.push((mid, hi, v2, e2));
    }
    parts.iter().map(|p| p.2).sum()
}

/// `P(sum_i lambda_i chi^2_1 > x)` by the Liu, Tang and Zhang (2009)
/// four-moment approximation, as mgcv's `liu2`.
pub fn liu2(x: f64, lambda: &[f64]) -> f64 {
    let c1: f64 = lambda.iter().sum();
    let c2: f64 = lambda.iter().map(|l| l * l).sum();
    let c3: f64 = lambda.iter().map(|l| l * l * l).sum();
    let c4: f64 = lambda.iter().map(|l| l * l * l * l).sum();
    if x <= 0.0 || c2 <= 0.0 {
        return 1.0;
    }
    let s1 = c3 / c2.powf(1.5);
    let s2 = c4 / (c2 * c2);
    let sig_q = (2.0 * c2).sqrt();
    let t = (x - c1) / sig_q;
    let (a, delta, l) = if s1 * s1 > s2 {
        let a = 1.0 / (s1 - (s1 * s1 - s2).sqrt());
        let delta = s1 * a * a * a - a * a;
        (a, delta, a * a - 2.0 * delta)
    } else {
        let a = 1.0 / s1;
        let l = if c3 == 0.0 {
            1.0
        } else {
            c2 * c2 * c2 / (c3 * c3)
        };
        (a, 0.0, l)
    };
    let mu_x = l + delta;
    let sig_x = std::f64::consts::SQRT_2 * a;
    noncentral_chisq_upper(t * sig_x + mu_x, l, delta)
}

/// `P(sum_i a_i chi^2_1 / (chi^2_df / df) > x)` by 50-point quadrature over
/// the denominator, as mgcv's `simf`.
pub fn simf(x: f64, a: &[f64], df: f64) -> f64 {
    let nq = 50;
    let mut total = 0.0;
    for i in 0..nq {
        let p = (i as f64 + 0.5) / nq as f64;
        let q = chisq_quantile(p, df);
        total += liu2(x * q / df, a);
    }
    total / nq as f64
}

/// Upper tail of a chi-square with `df` degrees of freedom.
pub fn chisq_upper(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(0.5 * df, 0.5 * x)
}

/// Upper tail of a noncentral chi-square: a Poisson(delta / 2) mixture of
/// central chi-squares with `df + 2j` degrees of freedom.
fn noncentral_chisq_upper(x: f64, df: f64, delta: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if delta <= 0.0 {
        return chisq_upper(x, df);
    }
    let lam = 0.5 * delta;
    let centre = lam.floor() as i64;
    let log_weight = |j: i64| -lam + (j as f64) * lam.ln() - libm::lgamma(j as f64 + 1.0);
    let mut total = 0.0;
    // Sum outward from the Poisson mode until the weights are negligible.
    for dir in [1i64, -1] {
        let mut j = if dir == 1 { centre } else { centre - 1 };
        while j >= 0 {
            let w = log_weight(j).exp();
            total += w * chisq_upper(x, df + 2.0 * j as f64);
            if w < 1e-16 && (j - centre).abs() > 10 {
                break;
            }
            j += dir;
        }
    }
    total.clamp(0.0, 1.0)
}

/// Upper tail of an F distribution with `d1` and `d2` degrees of freedom.
pub fn f_upper(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    beta_inc(0.5 * d2, 0.5 * d1, d2 / (d2 + d1 * f))
}

/// Quantile of a chi-square with `df` degrees of freedom, by bisection on
/// the regularized lower incomplete gamma function.
fn chisq_quantile(p: f64, df: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, df.max(1.0));
    while gamma_p(0.5 * df, 0.5 * hi) < p {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_p(0.5 * df, 0.5 * mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 * hi.max(1.0) {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Regularized lower incomplete gamma function `P(a, x)`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`,
/// computed directly in the upper tail so small values keep their accuracy.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = sum;
    let mut ap = a;
    for _ in 0..10_000 {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    (sum.ln() - x + a * x.ln() - libm::lgamma(a)).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..10_000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - libm::lgamma(a)).exp() * h
}

/// Regularized incomplete beta function `I_x(a, b)`.
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front =
        libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front.exp() * beta_cf(a, b, x) / a
    } else {
        1.0 - ln_front.exp() * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..10_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Every term's test from the fitted quantities: `ainv = A^-1`, `xtwx =
/// X'WX` at the final working weights, `cov` the Bayesian covariance and
/// `beta` the coefficients (intercept first). `res_df` is `Some(n -
/// edf_total)` when the scale was estimated.
pub(crate) fn block_tests(
    blocks: &[(usize, usize)],
    beta: &Array1<f64>,
    ainv: &Array2<f64>,
    xtwx: &Array2<f64>,
    cov: &Array2<f64>,
    res_df: Option<f64>,
) -> Vec<(f64, TermTest)> {
    let f = ainv.dot(xtwx);
    blocks
        .iter()
        .map(|&(start, end)| {
            // edf1 = 2 edf - tr(F F) over the term's columns (mgcv's `edf1`).
            let mut edf = 0.0;
            let mut ff = 0.0;
            for i in start..end {
                edf += f[[i, i]];
                for k in 0..f.ncols() {
                    ff += f[[i, k]] * f[[k, i]];
                }
            }
            let edf1 = 2.0 * edf - ff;
            let rank = ((end - start) as f64).min(edf1);
            let s = ndarray::s![start..end, start..end];
            let test = term_test(
                beta.slice(ndarray::s![start..end]),
                cov.slice(s),
                xtwx.slice(s),
                rank,
                res_df,
            );
            (edf1, test)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tails_match_reference_values() {
        // scipy.stats: chi2.sf(3.84, 1) = 0.050043521248705085
        assert!((chisq_upper(3.84, 1.0) - 0.050043521248705).abs() < 1e-12);
        // chi2.sf(20, 7.5) = 0.007657674067298914
        assert!((chisq_upper(20.0, 7.5) - 0.007657674067298914).abs() < 1e-12);
        // f.sf(4, 3, 40) = 0.0139586945050814
        assert!((f_upper(4.0, 3.0, 40.0) - 0.0139586945050814).abs() < 1e-12);
        // ncx2.sf(10, 3, 2) = 0.10143503648600136
        assert!((noncentral_chisq_upper(10.0, 3.0, 2.0) - 0.10143503648600136).abs() < 1e-10);
        // chi2.ppf(0.3, 4) = 2.1946984214069833
        assert!((chisq_quantile(0.3, 4.0) - 2.1946984214069833).abs() < 1e-8);
    }

    #[test]
    fn imhof_matches_the_chi_square_and_f_tails() {
        // Few degrees of freedom decay too slowly to truncate cheaply.
        assert!(imhof_upper(9.0, &[1.0; 3], &[1.0; 3]).is_none());
        let p = imhof_upper(20.0, &[1.0], &[7.0]).unwrap();
        assert!((p - chisq_upper(20.0, 7.0)).abs() < 1e-9);
        // P(chi2_3 / 3 > F chi2_40 / 40) is the F(3, 40) tail at F.
        let f = 4.0;
        let p = imhof_upper(
            0.0,
            &[1.0, 1.0, 1.0, -3.0 * f / 40.0],
            &[1.0, 1.0, 1.0, 40.0],
        )
        .unwrap();
        assert!((p - f_upper(f, 3.0, 40.0)).abs() < 1e-9);
        // A very small weight with few degrees of freedom is left to Liu.
        assert!(imhof_upper(9.0, &[1.0, 1e-9], &[1.0, 1.0]).is_none());
    }

    #[test]
    fn liu2_is_exact_for_equal_weights() {
        // With equal weights the sum is a scaled central chi-square.
        let p = liu2(9.0, &[1.0, 1.0, 1.0]);
        assert!((p - chisq_upper(9.0, 3.0)).abs() < 1e-10);
        // simf with equal weights is the F tail (mgcv's own check:
        // 1 - pf(4/3, 3, 40) against simf(4, rep(1, 3), 40)).
        let s = simf(4.0, &[1.0, 1.0, 1.0], 40.0);
        assert!((s - f_upper(4.0 / 3.0, 3.0, 40.0)).abs() < 5e-3);
    }

    #[test]
    fn an_unpenalized_term_gives_the_wald_test() {
        // One coefficient with known variance: the statistic is b^2 / v and
        // the p-value the chi-square(1) tail.
        let b = Array1::from_vec(vec![0.5]);
        let v = Array2::from_elem((1, 1), 0.04);
        let x = Array2::from_elem((1, 1), 30.0);
        let t = term_test(b.view(), v.view(), x.view(), 1.0, None);
        assert!((t.statistic - 6.25).abs() < 1e-10);
        assert!((t.p_value - chisq_upper(6.25, 1.0)).abs() < 1e-12);
    }
}
