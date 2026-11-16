//! Special functions the distribution helpers need: the regularized incomplete
//! gamma and beta functions, the Student-t, gamma and beta CDFs and their
//! inverses, and the normal CDF, quantile and log-CDF.
//!
//! Every series, continued fraction and root search here stops after a fixed
//! number of iterations. Each function returns a [`Special`], whose
//! `converged` is false when that limit was reached before the tolerance, so a
//! caller can report it instead of returning a value of unknown accuracy.

// AS 241's coefficients are kept exactly as published, and the argument
// checks are written `!(a > 0.0)` so that NaN fails them.
#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::neg_cmp_op_on_partial_ord
)]

use libm::{erfc, lgamma};

/// Relative tolerance of the series and continued fractions.
const EPS: f64 = 3.0e-16;
/// Floor that keeps a continued fraction's denominators away from zero.
const TINY: f64 = 1.0e-300;
/// Iteration limit of the series and continued fractions.
const MAX_ITER: usize = 300;
/// Iteration limit of the series and continued fractions for shape `a`. Both
/// need on the order of `sqrt(a)` terms near the mode (a Poisson CDF at a mean
/// of 1e6 is an incomplete gamma with `a` and `x` near 1e6), so the fixed
/// limit grows with it.
/// The growth is capped at `MAX_ITER_SCALED`, past which the value is
/// reported as not converged.
fn iteration_limit(a: f64) -> usize {
    if !a.is_finite() {
        return MAX_ITER;
    }
    (20.0 * a.sqrt()).min(MAX_ITER_SCALED as f64) as usize + MAX_ITER
}

/// Upper bound of `iteration_limit`'s growth: shapes up to about 2.5e9.
const MAX_ITER_SCALED: usize = 1_000_000;
/// Iteration limit of the bisection in the quantile functions.
const MAX_BISECT: usize = 200;

/// A function value and whether its computation met its tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Special {
    /// The function value (NaN for an argument outside the domain).
    pub value: f64,
    /// Whether the iteration met its tolerance before its limit.
    pub converged: bool,
}

impl Special {
    fn exact(value: f64) -> Self {
        Self {
            value,
            converged: true,
        }
    }
}

/// `t - log(1 + t)`, without the cancellation of the two terms for small `t`.
fn t_minus_log1p(t: f64) -> f64 {
    if t.abs() < 0.25 {
        // sum_{k>=2} (-t)^k / k, alternating with ratio |t| < 1/4.
        let mut term = t * t;
        let mut total = 0.0;
        let mut k = 2.0;
        loop {
            let next = term / k;
            total += next;
            if next.abs() <= total.abs() * 1e-17 {
                break;
            }
            term *= -t;
            k += 1.0;
            if k > 200.0 {
                break;
            }
        }
        total
    } else {
        t - t.ln_1p()
    }
}

/// Stirling's error `lgamma(a) - ((a - 0.5) ln a - a + 0.5 ln(2 pi))`, for
/// `a >= STIRLING_FROM`, from its asymptotic series. The first omitted term
/// is about `1 / (1188 a^9)`: 4e-19 at a = 50 (8e-13 at a = 10, too large).
fn stirling_error(a: f64) -> f64 {
    let a2 = a * a;
    (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - 1.0 / (1680.0 * a2)) / a2) / a2) / a
}

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

/// Shape from which the prefactors use Stirling's series. Below it the direct
/// form loses at most a few units of 1e-14 (its terms are near a ln a).
const STIRLING_FROM: f64 = 50.0;

/// `log(x^a e^{-x} / Gamma(a))`, the prefactor of the incomplete gamma. For a
/// large shape it is written as `-a (t - log1p(t)) + 0.5 ln a - 0.5 ln(2 pi)
/// - stirling_error(a)` with `t = (x - a) / a`, which avoids subtracting
/// terms of size `a ln a` (at a = 1e9 the direct form loses six digits).
fn log_gamma_front(a: f64, x: f64) -> f64 {
    if a < STIRLING_FROM {
        return -x + a * x.ln() - lgamma(a);
    }
    let t = (x - a) / a;
    -a * t_minus_log1p(t) + 0.5 * a.ln() - HALF_LN_2PI - stirling_error(a)
}

/// `log(x^a (1 - x)^b / B(a, b))`, the prefactor of the incomplete beta. For
/// large shapes it is written around the mean `x0 = a / (a + b)` as
/// `-a d(x / x0) - b d((1 - x) / (1 - x0)) + 0.5 ln(a b / (a + b))
/// - 0.5 ln(2 pi) + stirling terms`, with `d(r) = r - 1 - ln r`, for the same
/// reason as `log_gamma_front`.
fn log_beta_front(a: f64, b: f64, x: f64) -> f64 {
    if a < STIRLING_FROM && b < STIRLING_FROM {
        return lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (-x).ln_1p();
    }
    if a < STIRLING_FROM || b < STIRLING_FROM {
        // One small shape s and one large l: lgamma(l + s) - lgamma(l) by
        // Stirling, (l - 0.5) log1p(s / l) + s ln(l + s) - s plus the error
        // terms, instead of a difference of two values near l ln l (at
        // l = 1e7 the direct form loses eight digits).
        let (small, large) = if a < b { (a, b) } else { (b, a) };
        let lgamma_ratio = (large - 0.5) * (small / large).ln_1p() + small * (large + small).ln()
            - small
            + stirling_error(large + small)
            - stirling_error(large);
        return lgamma_ratio - lgamma(small) + a * x.ln() + b * (-x).ln_1p();
    }
    let n = a + b;
    let t1 = (x * n - a) / a;
    let t2 = ((1.0 - x) * n - b) / b;
    -a * t_minus_log1p(t1) - b * t_minus_log1p(t2) + 0.5 * (a * b / n).ln() - HALF_LN_2PI
        + stirling_error(n)
        - stirling_error(a)
        - stirling_error(b)
}

/// Regularized lower incomplete gamma `P(a, x)`: the series for `x < a + 1`,
/// otherwise the Lentz continued fraction for `Q(a, x)` and its complement.
/// NaN for `a <= 0`, `x < 0` or a non-finite argument.
///
/// ```
/// use mboostrs_core::special::gamma_p;
///
/// // R: pgamma(2, shape = 3) = 0.3233235838169366
/// let p = gamma_p(3.0, 2.0);
/// assert!(p.converged);
/// assert!((p.value - 0.3233235838169366).abs() < 1e-14);
/// ```
pub fn gamma_p(a: f64, x: f64) -> Special {
    if !(a > 0.0 && x >= 0.0 && a.is_finite()) {
        return Special::exact(f64::NAN);
    }
    if x == 0.0 {
        return Special::exact(0.0);
    }
    if x.is_infinite() {
        return Special::exact(1.0);
    }
    let log_front = log_gamma_front(a, x);
    if x < a + 1.0 {
        let mut term = 1.0 / a;
        let mut total = term;
        let mut n = a;
        let mut converged = false;
        for _ in 0..iteration_limit(a) {
            n += 1.0;
            term *= x / n;
            total += term;
            if term.abs() < total.abs() * EPS {
                converged = true;
                break;
            }
        }
        return Special {
            value: total * log_front.exp(),
            converged,
        };
    }
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut converged = false;
    for i in 1..=iteration_limit(a) {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            converged = true;
            break;
        }
    }
    Special {
        value: 1.0 - log_front.exp() * h,
        converged,
    }
}

/// The continued fraction of the incomplete beta (Lentz), and whether it met
/// its tolerance.
fn beta_cf(a: f64, b: f64, x: f64) -> (f64, bool) {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=iteration_limit(a.max(b)) {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            return (h, true);
        }
    }
    (h, false)
}

/// Regularized incomplete beta `I_x(a, b)`, through the continued fraction on
/// whichever side of `(a + 1) / (a + b + 2)` it converges fastest. NaN for a
/// non-positive shape or a NaN argument.
pub fn beta_i(a: f64, b: f64, x: f64) -> Special {
    if !(a > 0.0 && b > 0.0) || x.is_nan() {
        return Special::exact(f64::NAN);
    }
    if x <= 0.0 {
        return Special::exact(0.0);
    }
    if x >= 1.0 {
        return Special::exact(1.0);
    }
    let front = log_beta_front(a, b, x).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        let (cf, converged) = beta_cf(a, b, x);
        Special {
            value: front * cf / a,
            converged,
        }
    } else {
        let (cf, converged) = beta_cf(b, a, 1.0 - x);
        Special {
            value: 1.0 - front * cf / b,
            converged,
        }
    }
}

/// Student-t CDF with `df` degrees of freedom, through
/// `I_{df/(df+t^2)}(df/2, 1/2)`.
pub fn student_t_cdf(t: f64, df: f64) -> Special {
    if !(df > 0.0) || t.is_nan() {
        return Special::exact(f64::NAN);
    }
    if t.is_infinite() {
        return Special::exact(if t > 0.0 { 1.0 } else { 0.0 });
    }
    let x = df / (df + t * t);
    let ib = beta_i(0.5 * df, 0.5, x);
    let tail = 0.5 * ib.value;
    Special {
        value: if t > 0.0 { 1.0 - tail } else { tail },
        converged: ib.converged,
    }
}

/// Invert a monotone increasing CDF by bracket expansion and bisection,
/// to `1e-12` relative to the bracket's midpoint. `converged` is false when
/// the bisection, or the CDF at any probe, did not converge.
fn invert(cdf: impl Fn(f64) -> Special, p: f64, mut lo: f64, mut hi: f64) -> Special {
    let mut converged = true;
    let mut eval = |x: f64| {
        let s = cdf(x);
        converged &= s.converged;
        s.value
    };
    let mut expansions = 0;
    while eval(lo) > p && expansions < 2000 {
        lo -= (hi - lo).max(1.0);
        expansions += 1;
    }
    while eval(hi) < p && expansions < 2000 {
        hi += (hi - lo).max(1.0);
        expansions += 1;
    }
    if expansions >= 2000 {
        return Special {
            value: f64::NAN,
            converged: false,
        };
    }
    // A lower bound of zero on a positive support is replaced by a positive
    // one, so the search runs on the log scale and resolves a quantile of any
    // magnitude: a gamma quantile at shape 0.19 and p = 0.005 is near 1e-13,
    // which a stopping width of 1e-12 left undetermined.
    if lo == 0.0 && hi > 0.0 {
        let mut x = hi;
        let mut steps = 0;
        while eval(x) >= p && steps < 400 {
            x *= 0.125;
            steps += 1;
        }
        if steps == 400 {
            // The quantile is below about 1e-361 of `hi`: zero in f64.
            return Special {
                value: 0.0,
                converged,
            };
        }
        lo = x;
        hi = hi.min(x * 8.0);
    }
    let mut bisected = false;
    for _ in 0..MAX_BISECT {
        let mid = if lo > 0.0 {
            // sqrt(lo) * sqrt(hi): the product lo * hi underflows for
            // quantiles near 1e-160.
            lo.sqrt() * hi.sqrt()
        } else {
            0.5 * (lo + hi)
        };
        if mid <= lo || mid >= hi {
            bisected = true;
            break;
        }
        if eval(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 4.0 * f64::EPSILON * hi.abs().max(lo.abs()) {
            bisected = true;
            break;
        }
    }
    Special {
        value: 0.5 * (lo + hi),
        converged: converged && bisected,
    }
}

/// Student-t quantile. The search runs on the upper half, where the CDF rises
/// away from 1/2: below `|t| ~ sqrt(df * eps)` the incomplete-beta argument
/// rounds to one and the CDF is flat at 1/2, so the median is returned
/// exactly and the lower half by symmetry.
pub fn student_t_quantile(p: f64, df: f64) -> Special {
    if !(df > 0.0 && p > 0.0 && p < 1.0) {
        return Special::exact(f64::NAN);
    }
    if p == 0.5 {
        return Special::exact(0.0);
    }
    let upper = p > 0.5;
    let target = if upper { p } else { 1.0 - p };
    let z = invert(|t| student_t_cdf(t, df), target, 0.0, 10.0);
    Special {
        value: if upper { z.value } else { -z.value },
        converged: z.converged,
    }
}

/// Quantile of the gamma distribution with shape `a` and unit scale.
pub fn gamma_quantile(p: f64, a: f64) -> Special {
    if !(a > 0.0 && p > 0.0 && p < 1.0) {
        return Special::exact(f64::NAN);
    }
    invert(|x| gamma_p(a, x.max(0.0)), p, 0.0, a.max(1.0) * 10.0)
}

/// Quantile of the beta distribution with shapes `a` and `b`, by bisection on
/// `[0, 1]` (on the log scale, so a small quantile keeps its precision).
pub fn beta_quantile(p: f64, a: f64, b: f64) -> Special {
    if !(a > 0.0 && b > 0.0 && p > 0.0 && p < 1.0) {
        return Special::exact(f64::NAN);
    }
    invert(|x| beta_i(a, b, x.min(1.0)), p, 0.0, 1.0)
}

/// Standard normal CDF, `erfc(-z / sqrt(2)) / 2`, accurate in both tails.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// `log(Phi(z))`, as `log1p(-Phi(-z))` for positive `z` and `log(Phi(z))`
/// down to `z = -37`, where `Phi` is still a normal double. Below that the
/// CDF underflows, so the asymptotic expansion of the Mills ratio is used,
/// `log phi(z) - log(-z) + log(sum_k (-1)^k (2k - 1)!! / z^(2k))`, whose terms
/// at `|z| >= 37` fall below 1e-16 relative within a few steps. (It was used
/// from `z = -8` with three terms, which left a relative error of `105 / z^8`,
/// 6e-6 at `z = -8`.)
pub fn normal_log_cdf(z: f64) -> f64 {
    if z < -37.0 {
        let inv = 1.0 / (z * z);
        let mut term = 1.0;
        let mut series = 1.0;
        for k in 1..12 {
            term *= -((2 * k - 1) as f64) * inv;
            series += term;
            if term.abs() < 1e-17 {
                break;
            }
        }
        -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln() - (-z).ln() + series.ln()
    } else if z > 0.0 {
        // Phi(z) = 1 - Phi(-z) is within rounding of one here; log1p keeps
        // the small Phi(-z).
        (-normal_cdf(-z)).ln_1p()
    } else {
        normal_cdf(z).ln()
    }
}

/// Standard normal quantile, Wichura's AS 241 (`PPND16`), the algorithm R's
/// `qnorm` uses; accurate to about 1e-16. NaN outside `(0, 1)`.
///
/// ```
/// use mboostrs_core::special::normal_quantile;
///
/// assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-15);
/// assert!(normal_quantile(1.0).is_nan());
/// ```
pub fn normal_quantile(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                + 67265.770927008700853)
                * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((r * 5226.495278852545925 + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
            + 0.24178072517745061177)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                + 0.0151986665636164571966)
                * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0)
    } else {
        let r = r - 5.0;
        (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
            + 0.0012426609473880784386)
            * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                + 1.8463183175100546818e-5)
                * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0)
    };
    if q < 0.0 {
        -val
    } else {
        val
    }
}

/// Inverse-Gaussian CDF with mean `mu` and `sigma` (variance `sigma^2 mu^3`,
/// the gamlss.dist `IG` parameterization): `Phi(a) + exp(2 / (mu sigma^2))
/// Phi(b)` with `a = (y / mu - 1) / (sigma sqrt(y))` and
/// `b = -(y / mu + 1) / (sigma sqrt(y))`. The exponential factor overflows for
/// a small `mu sigma^2`, so the second term is formed in logs. 0 for `y <= 0`.
pub fn inverse_gaussian_cdf(y: f64, mu: f64, sigma: f64) -> f64 {
    if !(mu > 0.0 && sigma > 0.0) || y.is_nan() {
        return f64::NAN;
    }
    if y <= 0.0 {
        return 0.0;
    }
    if y.is_infinite() {
        return 1.0;
    }
    let root = sigma * y.sqrt();
    let first = normal_cdf((y / mu - 1.0) / root);
    let log_second = 2.0 / (mu * sigma * sigma) + normal_log_cdf(-(y / mu + 1.0) / root);
    (first + log_second.exp()).clamp(0.0, 1.0)
}

/// Inverse-Gaussian quantile, by bracketing and bisection on
/// [`inverse_gaussian_cdf`] from `[0, mu (1 + 40 sigma^2) + 10 mu]`.
pub fn inverse_gaussian_quantile(p: f64, mu: f64, sigma: f64) -> Special {
    if !(mu > 0.0 && sigma > 0.0 && p > 0.0 && p < 1.0) {
        return Special::exact(f64::NAN);
    }
    let upper = mu * (1.0 + 40.0 * sigma * sigma) + 10.0 * mu;
    invert(
        |x| Special::exact(inverse_gaussian_cdf(x.max(0.0), mu, sigma)),
        p,
        0.0,
        upper,
    )
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)`, computed
/// directly on the continued-fraction side so a small upper tail keeps its
/// precision. NaN under the same conditions as [`gamma_p`].
pub fn gamma_q(a: f64, x: f64) -> Special {
    if !(a > 0.0 && x >= 0.0 && a.is_finite()) {
        return Special::exact(f64::NAN);
    }
    if x == 0.0 {
        return Special::exact(1.0);
    }
    if x.is_infinite() {
        return Special::exact(0.0);
    }
    if x < a + 1.0 {
        let p = gamma_p(a, x);
        return Special {
            value: 1.0 - p.value,
            converged: p.converged,
        };
    }
    let log_front = log_gamma_front(a, x);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut converged = false;
    for i in 1..=iteration_limit(a) {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            converged = true;
            break;
        }
    }
    Special {
        value: log_front.exp() * h,
        converged,
    }
}

/// Poisson CDF `P(X <= q)` for mean `mu`: `Q(floor(q) + 1, mu)`, as R's
/// `ppois`. 0 below zero; NaN for a negative or non-finite `mu`.
///
/// ```
/// use mboostrs_core::special::poisson_cdf;
///
/// // R: ppois(3, 2.5) = 0.7575761331330658
/// assert!((poisson_cdf(3.0, 2.5).value - 0.7575761331330658).abs() < 1e-14);
/// ```
pub fn poisson_cdf(q: f64, mu: f64) -> Special {
    if !(mu >= 0.0 && mu.is_finite()) || q.is_nan() {
        return Special::exact(f64::NAN);
    }
    if q < 0.0 {
        return Special::exact(0.0);
    }
    if mu == 0.0 || q.is_infinite() {
        return Special::exact(1.0);
    }
    gamma_q(q.floor() + 1.0, mu)
}

/// Negative binomial CDF `P(X <= q)` for mean `mu` and size `size` (success
/// probability `size / (size + mu)`): `I_p(size, floor(q) + 1)`, as R's
/// `pnbinom(q, size = size, mu = mu)`.
///
/// ```
/// use mboostrs_core::special::nbinom_cdf;
///
/// // R: pnbinom(4, size = 2, mu = 3) = 0.76672
/// assert!((nbinom_cdf(4.0, 3.0, 2.0).value - 0.76672).abs() < 1e-13);
/// ```
pub fn nbinom_cdf(q: f64, mu: f64, size: f64) -> Special {
    if !(mu >= 0.0 && mu.is_finite() && size > 0.0) || q.is_nan() {
        return Special::exact(f64::NAN);
    }
    if q < 0.0 {
        return Special::exact(0.0);
    }
    if mu == 0.0 || q.is_infinite() {
        return Special::exact(1.0);
    }
    if size.is_infinite() {
        return poisson_cdf(q, mu);
    }
    beta_i(size, q.floor() + 1.0, size / (size + mu))
}

/// Binomial CDF `P(X <= q)` for `n` trials and success probability `prob`:
/// `I_{1-prob}(n - k, k + 1)` with `k = floor(q)`, as R's `pbinom`.
pub fn binom_cdf(q: f64, n: f64, prob: f64) -> Special {
    if !(n >= 0.0 && n.fract() == 0.0 && (0.0..=1.0).contains(&prob)) || q.is_nan() {
        return Special::exact(f64::NAN);
    }
    if q < 0.0 {
        return Special::exact(0.0);
    }
    let k = q.floor();
    if k >= n {
        return Special::exact(1.0);
    }
    if prob == 0.0 {
        return Special::exact(1.0);
    }
    if prob == 1.0 {
        return Special::exact(0.0);
    }
    beta_i(n - k, k + 1.0, 1.0 - prob)
}

/// Smallest integer `k >= 0` with `a + b * cdf(k) >= p`, for a nondecreasing
/// `cdf` on the integers. `a = 0, b = 1` is the ordinary quantile; a
/// zero-inflated count (`pi` extra mass at zero) has `a = pi, b = 1 - pi`; a
/// zero-adjusted one has `a = s - (1 - s) F(0) / (1 - F(0))`,
/// `b = (1 - s) / (1 - F(0))`. The search brackets from `start` by doubling
/// and then bisects on the integers; `upper` caps it (support maximum or a
/// size limit). Not converged when the cap is reached below `p`.
fn affine_count_quantile(
    p: f64,
    a: f64,
    b: f64,
    start: f64,
    upper: f64,
    cdf: impl Fn(f64) -> Special,
) -> Special {
    if p.is_nan() || a.is_nan() || b.is_nan() {
        return Special::exact(f64::NAN);
    }
    let mut converged = true;
    let mut reaches = |k: f64| {
        let f = cdf(k);
        converged &= f.converged;
        a + b * f.value >= p
    };
    if reaches(0.0) {
        return Special {
            value: 0.0,
            converged,
        };
    }
    // Bracket: lo fails, hi reaches.
    let mut lo = 0.0_f64;
    let mut hi = start.max(1.0).floor().min(upper);
    while !reaches(hi) {
        if hi >= upper {
            return Special {
                value: upper,
                converged: false,
            };
        }
        lo = hi;
        hi = (hi * 2.0).min(upper);
    }
    while hi - lo > 1.0 {
        let mid = ((lo + hi) / 2.0).floor();
        if reaches(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Special {
        value: hi,
        converged,
    }
}

/// Largest count the quantile searches consider (2^53, beyond which integers
/// are not exact in `f64`).
const COUNT_CAP: f64 = 9_007_199_254_740_992.0;

/// Smallest `k` with `a + b * ppois(k, mu) >= p`; see the ordinary,
/// zero-inflated and zero-adjusted forms in `affine_count_quantile`.
pub fn poisson_quantile_affine(p: f64, mu: f64, a: f64, b: f64) -> Special {
    if !(mu >= 0.0 && mu.is_finite()) {
        return Special::exact(f64::NAN);
    }
    let start = mu + 4.0 * mu.sqrt() + 1.0;
    affine_count_quantile(p, a, b, start, COUNT_CAP, |k| poisson_cdf(k, mu))
}

/// Smallest `k` with `a + b * pnbinom(k, size, mu) >= p`.
pub fn nbinom_quantile_affine(p: f64, mu: f64, size: f64, a: f64, b: f64) -> Special {
    if !(mu >= 0.0 && mu.is_finite() && size > 0.0) {
        return Special::exact(f64::NAN);
    }
    let sd = (mu + mu * mu / size).sqrt();
    affine_count_quantile(p, a, b, mu + 4.0 * sd + 1.0, COUNT_CAP, |k| {
        nbinom_cdf(k, mu, size)
    })
}

/// Smallest `k` with `pbinom(k, n, prob) >= p`.
pub fn binom_quantile(p: f64, n: f64, prob: f64) -> Special {
    if !(n >= 0.0 && n.fract() == 0.0 && (0.0..=1.0).contains(&prob)) {
        return Special::exact(f64::NAN);
    }
    affine_count_quantile(p, 0.0, 1.0, n * prob + 1.0, n, |k| binom_cdf(k, n, prob))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // R: pgamma(2, 3), pbeta(0.3, 2, 5), pt(1.5, 4), qnorm(0.975),
        // qnorm(1e-300).
        assert!((gamma_p(3.0, 2.0).value - 0.3233235838169366).abs() < 1e-14);
        assert!((beta_i(2.0, 5.0, 0.3).value - 0.579825).abs() < 1e-12);
        assert!((student_t_cdf(1.5, 4.0).value - 0.896).abs() < 1e-14);
        assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-15);
        assert!((normal_quantile(1e-300) + 37.0470962993612).abs() < 1e-10);
        let q = student_t_quantile(0.9, 7.0);
        assert!(q.converged && (student_t_cdf(q.value, 7.0).value - 0.9).abs() < 1e-11);
        let g = gamma_quantile(0.3, 2.5);
        assert!((gamma_p(2.5, g.value).value - 0.3).abs() < 1e-11);
        let b = beta_quantile(0.7, 2.0, 3.0);
        assert!((beta_i(2.0, 3.0, b.value).value - 0.7).abs() < 1e-12);
        // R: pnorm(z, log.p = TRUE). The expansion's first omitted term is
        // about 105/z^8 relative, 2e-6 at z = -9.
        assert!((normal_log_cdf(-40.0) + 804.6084420137538).abs() < 1e-9);
        assert!((normal_log_cdf(-9.0) + 43.628149113332114).abs() < 1e-5);
        assert!((normal_log_cdf(-3.0) + 6.607726221510349).abs() < 1e-13);
    }

    #[test]
    fn an_iteration_limit_is_reported() {
        // A shape of 1e14 needs about 1e7 series terms at x just below a + 1,
        // past the scaled limit.
        let s = gamma_p(1.0e14, 1.0e14 - 1.0);
        assert!(!s.converged, "{s:?}");
        assert!(gamma_p(3.0, 2.0).converged);
        // R: pgamma(1e9 - 1, 1e9) = 0.4999915895582568
        let s = gamma_p(1.0e9, 1.0e9 - 1.0);
        assert!(
            s.converged && (s.value - 0.4999915895582568).abs() < 1e-9,
            "{s:?}"
        );
    }

    #[test]
    fn invalid_arguments_are_nan() {
        assert!(gamma_p(-1.0, 1.0).value.is_nan());
        assert!(beta_i(1.0, 0.0, 0.5).value.is_nan());
        assert!(student_t_quantile(1.0, 3.0).value.is_nan());
        assert!(normal_quantile(0.0).is_nan());
    }
    #[test]
    fn count_cdfs_match_r() {
        // R: ppois(c(0, 40), 2.5); ppois(150, 100)
        assert!((poisson_cdf(0.0, 2.5).value - 0.0820849986238988).abs() < 1e-15);
        assert!((poisson_cdf(40.0, 2.5).value - 1.0).abs() < 1e-15);
        assert!((poisson_cdf(150.0, 100.0).value - 0.9999987669055809).abs() < 1e-13);
        // R: pnbinom(c(0, 10), size = 0.5, mu = 4)
        assert!((nbinom_cdf(0.0, 4.0, 0.5).value - 0.3333333333333333).abs() < 1e-14);
        assert!((nbinom_cdf(10.0, 4.0, 0.5).value - 0.8885540555489406).abs() < 1e-13);
        // R: pbinom(2, 5, 0.3) = 0.83692
        assert!((binom_cdf(2.0, 5.0, 0.3).value - 0.83692).abs() < 1e-14);
        assert_eq!(binom_cdf(5.0, 5.0, 0.3).value, 1.0);
        assert_eq!(poisson_cdf(-0.5, 2.0).value, 0.0);
    }

    #[test]
    fn count_quantiles_are_the_smallest_k_reaching_p() {
        for &(mu, size) in &[(0.3, 2.0), (4.0, 0.5), (250.0, 3.0), (1.0e6, 10.0)] {
            for &p in &[1e-6, 0.01, 0.3, 0.5, 0.9, 0.999999] {
                for &(a, b) in &[(0.0, 1.0), (0.2, 0.8)] {
                    let k = poisson_quantile_affine(p, mu, a, b);
                    assert!(k.converged);
                    assert!(a + b * poisson_cdf(k.value, mu).value >= p);
                    if k.value > 0.0 {
                        assert!(a + b * poisson_cdf(k.value - 1.0, mu).value < p);
                    }
                    let k = nbinom_quantile_affine(p, mu, size, a, b);
                    assert!(k.converged);
                    assert!(a + b * nbinom_cdf(k.value, mu, size).value >= p);
                    if k.value > 0.0 {
                        assert!(a + b * nbinom_cdf(k.value - 1.0, mu, size).value < p);
                    }
                }
            }
        }
        // R: qbinom(c(0.1, 0.5, 0.99), 10, 0.3) = 1 3 7
        assert_eq!(binom_quantile(0.1, 10.0, 0.3).value, 1.0);
        assert_eq!(binom_quantile(0.5, 10.0, 0.3).value, 3.0);
        assert_eq!(binom_quantile(0.99, 10.0, 0.3).value, 7.0);
        // R: qpois(0.5, 2.5) = 2; qnbinom(0.9, size = 0.5, mu = 4) = 11
        assert_eq!(poisson_quantile_affine(0.5, 2.5, 0.0, 1.0).value, 2.0);
        assert_eq!(nbinom_quantile_affine(0.9, 4.0, 0.5, 0.0, 1.0).value, 11.0);
    }

    #[test]
    fn inverse_gaussian_matches_gamlss_dist() {
        // gamlss.dist: pIG(c(0.5, 2, 6), mu = 2, sigma = 0.7)
        let expected = [0.1093347879359315, 0.6668368262707303, 0.9539240451283723];
        for (y, e) in [0.5, 2.0, 6.0].iter().zip(expected) {
            assert!((inverse_gaussian_cdf(*y, 2.0, 0.7) - e).abs() < 1e-12);
        }
        let q = inverse_gaussian_quantile(0.6668368262707303, 2.0, 0.7);
        assert!(q.converged && (q.value - 2.0).abs() < 1e-9, "{q:?}");
    }

    #[test]
    fn normal_log_cdf_is_accurate_in_the_left_tail() {
        // R: pnorm(c(-8.04, -20, -36.9, -40), log.p = TRUE)
        let expected = [
            (-8.04, -35.33908045949309),
            (-20.0, -203.917_155_371_097_27),
            (-36.9, -685.332_883_165_350_6),
            (-40.0, -804.608_442_013_753_8),
        ];
        for (z, e) in expected {
            let got = normal_log_cdf(z);
            assert!(((got - e) / e).abs() < 1e-14, "z = {z}: {got} vs {e}");
        }
    }

    #[test]
    fn small_quantiles_keep_their_precision() {
        // R: qgamma(0.005, shape = 0.187) and qgamma(1.0244131227763697e-4,
        // shape = 0.02379473363386335), the second near 1e-168, where the
        // product of the two bisection bounds underflows.
        let q = gamma_quantile(0.005, 0.187);
        assert!(
            q.converged
                && ((q.value - 3.204_261_405_428_178e-13) / 3.204_261_405_428_178e-13).abs() < 1e-9,
            "{q:?}"
        );
        let q = gamma_quantile(1.024_413_122_776_369_7e-4, 0.023_794_733_633_863_35);
        assert!(
            ((q.value - 1.240_296_367_459_462e-168) / 1.240_296_367_459_462e-168).abs() < 1e-9,
            "{q:?}"
        );
    }
}
