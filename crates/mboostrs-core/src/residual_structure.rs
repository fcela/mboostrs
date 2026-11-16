//! Residual structure that points to a revision of the model.
//!
//! Three checks run on the residuals of a fitted mean model:
//!
//! * [`cluster_icc`]: whether residuals share a value within a cluster. A
//!   positive intra-cluster correlation means the rows of one cluster carry a
//!   common effect the model does not contain, which a cluster intercept
//!   (`brandom(cluster)`) would absorb.
//! * [`serial_dependence`]: whether residuals, ordered by time, are
//!   correlated with their neighbours. The Ljung-Box statistic sums the
//!   squared autocorrelations over the first `h` lags; a time-varying effect
//!   the model lacks (`bbs(time)`) or an autoregressive error both produce it.
//! * [`gaussian_influence`]: how far each row moves the fitted values. Cook's
//!   distance and the leave-one-out residual are computed from the hat-matrix
//!   diagonal without refitting.
//!
//! Each check has a matching `*_false_alarm_rate` function. It draws residuals
//! from the null of no structure, applies the same decision rule as the check,
//! and returns the fraction of draws the rule flags. A check's reported rate
//! is therefore the rule's behaviour on data with nothing to find, for the same
//! design (cluster sizes, time points, leverages).
//!
//! The cluster and time checks use permutation reference distributions, so no
//! normality of the residuals is assumed for them. The influence check uses an
//! F reference from the Gaussian linear-model theory, which is exact for a
//! linear smoother with independent normal errors and approximate for a
//! boosting fit; its false-alarm rate is the number to read.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::special::{beta_i, gamma_q, student_t_cdf};
use crate::MboostrsError;

/// Intra-cluster correlation of residuals with its reference distributions.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterIcc {
    /// One-way ANOVA intra-cluster correlation, `(MSB - MSW) / (MSB + (k0 - 1) MSW)`
    /// with `k0` the average cluster size. Zero when the denominator is not positive.
    pub icc: f64,
    /// `MSB / MSW`, the one-way ANOVA F statistic.
    pub f_statistic: f64,
    /// Between-cluster degrees of freedom, `G - 1`.
    pub df_between: f64,
    /// Within-cluster degrees of freedom, `n - G`.
    pub df_within: f64,
    /// Upper tail of the F distribution at `f_statistic`.
    pub f_p_value: f64,
    /// Share of label-free permutations (residuals shuffled across rows,
    /// cluster sizes kept) whose ICC is at least the observed ICC, with the
    /// observed value counted once: `(1 + count) / (permutations + 1)`.
    pub permutation_p_value: f64,
    /// Number of permutations behind `permutation_p_value`.
    pub permutations: usize,
    /// Number of clusters `G`.
    pub clusters: usize,
    /// Number of rows in each cluster, by cluster code.
    pub cluster_sizes: Vec<usize>,
    /// Mean residual of each cluster, by cluster code.
    pub cluster_means: Vec<f64>,
}

/// Autocorrelation of residuals ordered by time, with the Ljung-Box test.
#[derive(Clone, Debug, PartialEq)]
pub struct SerialDependence {
    /// Sample autocorrelations at lags `1..=lags`; element `k` is lag `k + 1`.
    pub autocorrelation: Vec<f64>,
    /// Ljung-Box statistic over lags `1..=lags`.
    pub ljung_box: f64,
    /// Number of lags in the Ljung-Box statistic.
    pub lags: usize,
    /// Upper tail of chi-squared with `lags` degrees of freedom at `ljung_box`.
    pub chisq_p_value: f64,
    /// Share of permutations of the series whose Ljung-Box statistic is at
    /// least the observed one, `(1 + count) / (permutations + 1)`.
    pub permutation_p_value: f64,
    /// Number of permutations behind `permutation_p_value`.
    pub permutations: usize,
    /// Length of the series: the number of distinct time values.
    pub series_length: usize,
    /// Two-sided 95% white-noise band for each autocorrelation, `1.96 / sqrt(m)`.
    pub bound: f64,
    /// The distinct time values, ascending; the series is ordered by them.
    pub times: Vec<f64>,
}

/// Leave-one-out and influence measures of a Gaussian linear smoother.
#[derive(Clone, Debug, PartialEq)]
pub struct Influence {
    /// Degrees of freedom of the fit, the trace of the hat matrix.
    pub df: f64,
    /// Residual standard deviation, `sqrt(SSE / (n - df))`.
    pub sigma: f64,
    /// Leave-one-out residual of each row, `r_i / (1 - h_i)`.
    pub loo_residual: Vec<f64>,
    /// Cook's distance of each row: the size of the change in all fitted
    /// values when the row is deleted, in units of the residual variance.
    pub cooks: Vec<f64>,
    /// Externally studentized deleted residual of each row, on `n - df - 1`
    /// degrees of freedom. Under the Gaussian linear model it is exactly
    /// Student-t, which is why the flag is judged on it.
    pub studentized: Vec<f64>,
    /// Two-sided Student-t p-value of each row's studentized deleted residual.
    pub p_value: Vec<f64>,
    /// Rows whose Bonferroni-adjusted p-value, `min(1, n * p_i)`, is below
    /// `alpha`, in ascending order.
    pub flagged: Vec<usize>,
}

fn require_same_length(first: usize, second: usize, what: &str) -> Result<(), MboostrsError> {
    if first != second {
        return Err(MboostrsError::Shape(format!(
            "residuals ({first}) and {what} ({second}) must have one entry per row"
        )));
    }
    Ok(())
}

fn require_finite(values: &[f64], what: &str) -> Result<(), MboostrsError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(MboostrsError::Control(format!(
            "{what} must be finite; remove or impute the missing rows before the diagnostic"
        )));
    }
    Ok(())
}

/// Dense cluster codes `0..G` with every code used, and the size of each.
fn dense_codes(codes: &[usize]) -> Result<(usize, Vec<usize>), MboostrsError> {
    let groups = codes.iter().copied().max().map_or(0, |m| m + 1);
    let mut sizes = vec![0usize; groups];
    for &c in codes {
        sizes[c] += 1;
    }
    if sizes.contains(&0) {
        return Err(MboostrsError::Shape(
            "cluster codes must be 0..G-1 with every code used by at least one row; \
             recode the cluster labels with a factorize step"
                .into(),
        ));
    }
    Ok((groups, sizes))
}

fn check_permutations(permutations: usize) -> Result<(), MboostrsError> {
    if permutations < 1 {
        return Err(MboostrsError::Control(
            "permutations must be at least 1".into(),
        ));
    }
    Ok(())
}

/// Upper tail `P(F > f)` of the F distribution with `d1` and `d2` degrees of freedom.
fn f_upper_tail(f: f64, d1: f64, d2: f64) -> f64 {
    if f.is_nan() {
        return 1.0;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if f.is_infinite() {
        return 0.0;
    }
    beta_i(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f)).value
}

/// Fisher-Yates shuffle with the generator the crate uses everywhere else.
fn shuffle(values: &mut [f64], rng: &mut ChaCha8Rng) {
    for i in (1..values.len()).rev() {
        let j = rng.random_range(0..=i);
        values.swap(i, j);
    }
}

/// One-way ANOVA parts for `values` grouped by `codes`: `(icc, msb, msw)`.
fn anova_icc(values: &[f64], codes: &[usize], sizes: &[usize]) -> (f64, f64, f64) {
    let n = values.len() as f64;
    let groups = sizes.len();
    let grand = values.iter().sum::<f64>() / n;
    let mut sums = vec![0.0f64; groups];
    for (&v, &c) in values.iter().zip(codes) {
        sums[c] += v;
    }
    let means: Vec<f64> = sums
        .iter()
        .zip(sizes)
        .map(|(&s, &k)| s / k as f64)
        .collect();
    let ssb: f64 = means
        .iter()
        .zip(sizes)
        .map(|(&m, &k)| k as f64 * (m - grand).powi(2))
        .sum();
    let ssw: f64 = values
        .iter()
        .zip(codes)
        .map(|(&v, &c)| (v - means[c]).powi(2))
        .sum();
    let g = groups as f64;
    let msb = ssb / (g - 1.0);
    let msw = ssw / (n - g);
    let size_squares: f64 = sizes.iter().map(|&k| (k * k) as f64).sum();
    let k0 = (n - size_squares / n) / (g - 1.0);
    let denominator = msb + (k0 - 1.0) * msw;
    let icc = if denominator > 0.0 {
        (msb - msw) / denominator
    } else {
        0.0
    };
    (icc, msb, msw)
}

/// The cluster test on given values: ICC, its permutation p-value, and the
/// one-sided decision (`icc > 0 && p < alpha`) that the false-alarm rate uses.
fn cluster_test(
    values: &[f64],
    codes: &[usize],
    sizes: &[usize],
    permutations: usize,
    rng: &mut ChaCha8Rng,
) -> (f64, f64) {
    let (icc, _, _) = anova_icc(values, codes, sizes);
    let mut shuffled = values.to_vec();
    let mut at_least = 0usize;
    for _ in 0..permutations {
        shuffle(&mut shuffled, rng);
        let (icc_b, _, _) = anova_icc(&shuffled, codes, sizes);
        if icc_b >= icc {
            at_least += 1;
        }
    }
    let p = (1 + at_least) as f64 / (permutations + 1) as f64;
    (icc, p)
}

/// Intra-cluster correlation of `residuals` by `clusters`, with an F reference
/// and a permutation reference.
///
/// `clusters[i]` is the cluster of row `i`, as a code `0..G`. At least two
/// clusters are needed and at least one cluster must hold two or more rows,
/// so that the within-cluster variance exists.
pub fn cluster_icc(
    residuals: &[f64],
    clusters: &[usize],
    permutations: usize,
    seed: u64,
) -> Result<ClusterIcc, MboostrsError> {
    require_same_length(residuals.len(), clusters.len(), "clusters")?;
    require_finite(residuals, "residuals")?;
    check_permutations(permutations)?;
    let (groups, sizes) = dense_codes(clusters)?;
    let n = residuals.len();
    if groups < 2 {
        return Err(MboostrsError::Control(
            "the cluster check needs at least two clusters; this cluster column has one, \
             so there is no between-cluster variance to compare with"
                .into(),
        ));
    }
    if n <= groups {
        return Err(MboostrsError::Control(
            "the cluster check needs clusters with more than one row each; every cluster \
             here has one row, so the within-cluster variance is undefined"
                .into(),
        ));
    }
    let (icc, msb, msw) = anova_icc(residuals, clusters, &sizes);
    let df_between = (groups - 1) as f64;
    let df_within = (n - groups) as f64;
    let f_statistic = if msw > 0.0 {
        msb / msw
    } else if msb > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let f_p_value = f_upper_tail(f_statistic, df_between, df_within);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (_, permutation_p_value) =
        cluster_test(residuals, clusters, &sizes, permutations, &mut rng);

    let mut sums = vec![0.0f64; groups];
    for (&v, &c) in residuals.iter().zip(clusters) {
        sums[c] += v;
    }
    let cluster_means = sums
        .iter()
        .zip(&sizes)
        .map(|(&s, &k)| s / k as f64)
        .collect();

    Ok(ClusterIcc {
        icc,
        f_statistic,
        df_between,
        df_within,
        f_p_value,
        permutation_p_value,
        permutations,
        clusters: groups,
        cluster_sizes: sizes,
        cluster_means,
    })
}

/// Residual means per distinct time value, in ascending time order.
///
/// `group_of_row[i]` is the rank of row `i`'s time among the distinct times.
fn series_from_groups(residuals: &[f64], group_of_row: &[usize], groups: usize) -> Vec<f64> {
    let mut sums = vec![0.0f64; groups];
    let mut counts = vec![0usize; groups];
    for (&r, &g) in residuals.iter().zip(group_of_row) {
        sums[g] += r;
        counts[g] += 1;
    }
    sums.iter()
        .zip(&counts)
        .map(|(&s, &k)| s / k as f64)
        .collect()
}

/// Ljung-Box statistic and sample autocorrelations of a centred series.
fn ljung_box(series: &[f64], lags: usize) -> Result<(Vec<f64>, f64), MboostrsError> {
    let m = series.len();
    let mean = series.iter().sum::<f64>() / m as f64;
    let centred: Vec<f64> = series.iter().map(|v| v - mean).collect();
    let denominator: f64 = centred.iter().map(|c| c * c).sum();
    if denominator.is_nan() || denominator <= 0.0 {
        return Err(MboostrsError::Control(
            "the residuals are the same at every time point, so there is no serial \
             dependence to measure; the serial check needs residuals that vary over time"
                .into(),
        ));
    }
    let mut acf = Vec::with_capacity(lags);
    let mut q = 0.0f64;
    for k in 1..=lags {
        let numerator: f64 = (0..m - k).map(|t| centred[t] * centred[t + k]).sum();
        let rho = numerator / denominator;
        acf.push(rho);
        q += rho * rho / (m - k) as f64;
    }
    q *= (m * (m + 2)) as f64;
    Ok((acf, q))
}

/// Sorted distinct times, and for each row the rank of its time.
fn time_groups(times: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut distinct = times.to_vec();
    distinct.sort_by(|a, b| a.partial_cmp(b).expect("times are finite"));
    distinct.dedup();
    let group_of_row = times
        .iter()
        .map(|t| {
            distinct
                .binary_search_by(|d| d.partial_cmp(t).expect("times are finite"))
                .expect("every time is in the distinct list")
        })
        .collect();
    (distinct, group_of_row)
}

/// Autocorrelation of `residuals` ordered by `times`, with the Ljung-Box test.
///
/// Rows that share a time value are averaged into one point of the series,
/// so the lag is a step between distinct time values, not between rows. The
/// series needs at least `lags + 2` distinct times.
pub fn serial_dependence(
    residuals: &[f64],
    times: &[f64],
    lags: usize,
    permutations: usize,
    seed: u64,
) -> Result<SerialDependence, MboostrsError> {
    require_same_length(residuals.len(), times.len(), "times")?;
    require_finite(residuals, "residuals")?;
    require_finite(times, "times")?;
    check_permutations(permutations)?;
    if lags < 1 {
        return Err(MboostrsError::Control("lags must be at least 1".into()));
    }
    let (distinct, group_of_row) = time_groups(times);
    let m = distinct.len();
    if m < lags + 2 {
        return Err(MboostrsError::Control(format!(
            "the serial check needs at least {} distinct time values for {lags} lags; \
             this series has {m}. Lower max_lag",
            lags + 2
        )));
    }
    let series = series_from_groups(residuals, &group_of_row, m);
    let (acf, q) = ljung_box(&series, lags)?;
    let chisq_p_value = gamma_q(lags as f64 / 2.0, q / 2.0).value;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shuffled = series.clone();
    let mut at_least = 0usize;
    for _ in 0..permutations {
        shuffle(&mut shuffled, &mut rng);
        let (_, q_b) = ljung_box(&shuffled, lags)?;
        if q_b >= q {
            at_least += 1;
        }
    }
    let permutation_p_value = (1 + at_least) as f64 / (permutations + 1) as f64;

    Ok(SerialDependence {
        autocorrelation: acf,
        ljung_box: q,
        lags,
        chisq_p_value,
        permutation_p_value,
        permutations,
        series_length: m,
        bound: 1.96 / (m as f64).sqrt(),
        times: distinct,
    })
}

/// Leave-one-out and influence measures of a Gaussian linear smoother.
///
/// `residuals` are `y - fitted` and `hat` the hat-matrix diagonal, both at the
/// fit being checked. The degrees of freedom are the sum of `hat`. Each row's
/// Cook's distance is `r_i^2 h_i / (df * sigma^2 * (1 - h_i)^2)`. The flag is
/// judged on the externally studentized deleted residual `t_i`, referred to
/// Student-t on `n - df - 1` degrees of freedom: a row is flagged when
/// `min(1, n * p_i) < alpha`, with `p_i` its two-sided p-value. A Cook's
/// distance reference would not be calibrated here, because its F reference
/// is only approximate for rows of high leverage.
pub fn gaussian_influence(
    residuals: &[f64],
    hat: &[f64],
    alpha: f64,
) -> Result<Influence, MboostrsError> {
    require_same_length(residuals.len(), hat.len(), "hat-matrix diagonal")?;
    require_finite(residuals, "residuals")?;
    require_finite(hat, "hat-matrix diagonal")?;
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(MboostrsError::Control("alpha must lie in (0, 1)".into()));
    }
    let n = residuals.len();
    if let Some(i) = hat.iter().position(|&h| h >= 1.0) {
        return Err(MboostrsError::Control(format!(
            "row {i} has leverage {} (its fitted value is its response); leave-one-out \
             measures are undefined there. Refit without that row or check its predictors",
            hat[i]
        )));
    }
    let df: f64 = hat.iter().sum();
    let df_residual = n as f64 - df;
    if df_residual < 2.0 {
        return Err(MboostrsError::Control(format!(
            "the fit has {df:.3} degrees of freedom for {n} rows, leaving fewer than two \
             residual degrees of freedom; the influence measures need more rows than \
             effective parameters"
        )));
    }
    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = sse / df_residual;
    let t_df = df_residual - 1.0;
    let mut loo_residual = Vec::with_capacity(n);
    let mut cooks = Vec::with_capacity(n);
    let mut studentized = Vec::with_capacity(n);
    let mut p_value = Vec::with_capacity(n);
    let mut flagged = Vec::new();
    for i in 0..n {
        let r = residuals[i];
        let h = hat[i];
        let one_minus = 1.0 - h;
        loo_residual.push(r / one_minus);
        let d = if df > 0.0 && sigma2 > 0.0 {
            r * r * h / (df * sigma2 * one_minus * one_minus)
        } else {
            0.0
        };
        cooks.push(d);
        // Residual variance with row i left out: SSE_(i) = SSE - r_i^2 / (1 - h_i).
        let sse_i = (sse - r * r / one_minus).max(0.0);
        let sigma_i = (sse_i / t_df).sqrt();
        let (t, p) = if sigma_i > 0.0 {
            let t = r / (sigma_i * one_minus.sqrt());
            let tail = student_t_cdf(-t.abs(), t_df).value;
            (t, (2.0 * tail).min(1.0))
        } else if r == 0.0 {
            (0.0, 1.0)
        } else {
            // Every other residual is zero: row i alone carries the misfit.
            (f64::INFINITY.copysign(r), 0.0)
        };
        studentized.push(t);
        p_value.push(p);
        if (n as f64 * p).min(1.0) < alpha {
            flagged.push(i);
        }
    }
    Ok(Influence {
        df,
        sigma: sigma2.sqrt(),
        loo_residual,
        cooks,
        studentized,
        p_value,
        flagged,
    })
}

/// Standard normal deviate by the Box-Muller transform.
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Share of `draws` null datasets for which the cluster rule fires.
///
/// Each draw is independent standard normal residuals on the given cluster
/// layout; the rule is `icc > 0 && permutation_p_value < alpha`, as in
/// [`cluster_icc`]. The value is the empirical false-alarm rate.
pub fn cluster_false_alarm_rate(
    clusters: &[usize],
    draws: usize,
    permutations: usize,
    seed: u64,
    alpha: f64,
) -> Result<f64, MboostrsError> {
    check_permutations(permutations)?;
    if draws < 1 {
        return Err(MboostrsError::Control("draws must be at least 1".into()));
    }
    let (groups, sizes) = dense_codes(clusters)?;
    if groups < 2 || clusters.len() <= groups {
        return Err(MboostrsError::Control(
            "the cluster false-alarm rate needs at least two clusters and some cluster \
             with two or more rows"
                .into(),
        ));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut flagged = 0usize;
    for _ in 0..draws {
        let values: Vec<f64> = (0..clusters.len())
            .map(|_| standard_normal(&mut rng))
            .collect();
        let (icc, p) = cluster_test(&values, clusters, &sizes, permutations, &mut rng);
        if icc > 0.0 && p < alpha {
            flagged += 1;
        }
    }
    Ok(flagged as f64 / draws as f64)
}

/// Share of `draws` null datasets for which the serial rule fires.
///
/// The rule is `permutation_p_value < alpha` on the Ljung-Box statistic, as in
/// [`serial_dependence`], with the same time layout (rows sharing a time are
/// averaged).
pub fn serial_false_alarm_rate(
    times: &[f64],
    lags: usize,
    draws: usize,
    permutations: usize,
    seed: u64,
    alpha: f64,
) -> Result<f64, MboostrsError> {
    check_permutations(permutations)?;
    if draws < 1 {
        return Err(MboostrsError::Control("draws must be at least 1".into()));
    }
    require_finite(times, "times")?;
    let (distinct, group_of_row) = time_groups(times);
    let m = distinct.len();
    if lags < 1 || m < lags + 2 {
        return Err(MboostrsError::Control(format!(
            "the serial false-alarm rate needs at least {} distinct time values for {lags} lags",
            lags + 2
        )));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut flagged = 0usize;
    for _ in 0..draws {
        let residuals: Vec<f64> = (0..times.len())
            .map(|_| standard_normal(&mut rng))
            .collect();
        let series = series_from_groups(&residuals, &group_of_row, m);
        let (_, q) = ljung_box(&series, lags)?;
        let mut shuffled = series.clone();
        let mut at_least = 0usize;
        for _ in 0..permutations {
            shuffle(&mut shuffled, &mut rng);
            let (_, q_b) = ljung_box(&shuffled, lags)?;
            if q_b >= q {
                at_least += 1;
            }
        }
        let p = (1 + at_least) as f64 / (permutations + 1) as f64;
        if p < alpha {
            flagged += 1;
        }
    }
    Ok(flagged as f64 / draws as f64)
}

/// Share of `draws` null datasets in which at least one row is flagged.
///
/// Each draw has independent normal residuals with standard deviation
/// `sigma * sqrt(1 - h_i)` for row `i`, the marginal variance a linear smoother
/// gives its residuals. The residuals are drawn independently of one another,
/// which ignores the off-diagonal part of `(I - H)`, because only the
/// hat-matrix diagonal is available here. The rule is the one in
/// [`gaussian_influence`]. Under the null, each draw's residuals are
/// evaluated with that rule's own `sigma`, so a draw with a large residual
/// does not inflate the scale of the others.
pub fn influence_false_alarm_rate(
    hat: &[f64],
    sigma: f64,
    draws: usize,
    seed: u64,
    alpha: f64,
) -> Result<f64, MboostrsError> {
    if draws < 1 {
        return Err(MboostrsError::Control("draws must be at least 1".into()));
    }
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(MboostrsError::Control(
            "sigma must be positive and finite".into(),
        ));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut flagged = 0usize;
    for _ in 0..draws {
        let residuals: Vec<f64> = hat
            .iter()
            .map(|&h| sigma * (1.0 - h).max(0.0).sqrt() * standard_normal(&mut rng))
            .collect();
        let influence = gaussian_influence(&residuals, hat, alpha)?;
        if !influence.flagged.is_empty() {
            flagged += 1;
        }
    }
    Ok(flagged as f64 / draws as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clustered_residuals(groups: usize, per_group: usize, shift: f64) -> (Vec<f64>, Vec<usize>) {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut values = Vec::new();
        let mut codes = Vec::new();
        for g in 0..groups {
            let effect = shift * standard_normal(&mut rng);
            for _ in 0..per_group {
                values.push(effect + standard_normal(&mut rng));
                codes.push(g);
            }
        }
        (values, codes)
    }

    #[test]
    fn a_shared_cluster_effect_is_detected_and_independent_residuals_are_not() {
        let (values, codes) = clustered_residuals(20, 10, 1.5);
        let planted = cluster_icc(&values, &codes, 499, 1).unwrap();
        assert!(planted.icc > 0.3, "icc {}", planted.icc);
        assert!(planted.permutation_p_value < 0.01);
        assert!(planted.f_p_value < 1e-6);

        let (values, codes) = clustered_residuals(20, 10, 0.0);
        let null = cluster_icc(&values, &codes, 499, 1).unwrap();
        assert!(null.icc.abs() < 0.15, "icc {}", null.icc);
        assert!(null.permutation_p_value > 0.05);
    }

    #[test]
    fn cluster_means_and_sizes_follow_the_codes() {
        let values = [1.0, 3.0, 5.0, 7.0, 9.0];
        let codes = [0, 0, 1, 1, 2];
        let out = cluster_icc(&values, &codes, 9, 0).unwrap();
        assert_eq!(out.cluster_sizes, vec![2, 2, 1]);
        assert_eq!(out.cluster_means, vec![2.0, 6.0, 9.0]);
        assert_eq!(out.df_between, 2.0);
        assert_eq!(out.df_within, 2.0);
    }

    #[test]
    fn cluster_codes_with_a_gap_are_refused() {
        let err = cluster_icc(&[1.0, 2.0, 3.0, 4.0], &[0, 0, 2, 2], 9, 0).unwrap_err();
        assert!(err.to_string().contains("every code used"));
    }

    #[test]
    fn ar1_residuals_over_time_are_detected() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n = 300;
        let mut e = 0.0f64;
        let mut residuals = Vec::with_capacity(n);
        for _ in 0..n {
            e = 0.7 * e + standard_normal(&mut rng);
            residuals.push(e);
        }
        let times: Vec<f64> = (0..n).map(|t| t as f64).collect();
        let out = serial_dependence(&residuals, &times, 10, 499, 2).unwrap();
        assert!(
            out.autocorrelation[0] > 0.5,
            "lag 1 {}",
            out.autocorrelation[0]
        );
        assert!(out.permutation_p_value < 0.01);
        assert!(out.chisq_p_value < 1e-6);
    }

    #[test]
    fn white_noise_over_time_is_not_flagged() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let residuals: Vec<f64> = (0..300).map(|_| standard_normal(&mut rng)).collect();
        let times: Vec<f64> = (0..300).map(|t| t as f64).collect();
        let out = serial_dependence(&residuals, &times, 10, 499, 2).unwrap();
        assert!(
            out.permutation_p_value > 0.05,
            "p {}",
            out.permutation_p_value
        );
        assert!((out.bound - 1.96 / 300f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn rows_sharing_a_time_are_averaged_into_one_point() {
        let residuals = [1.0, 3.0, 2.0, 4.0, 6.0, 8.0];
        let times = [2.0, 2.0, 1.0, 3.0, 3.0, 4.0];
        let out = serial_dependence(&residuals, &times, 1, 9, 0).unwrap();
        assert_eq!(out.times, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.series_length, 4);
    }

    #[test]
    fn a_planted_high_influence_row_is_flagged_and_the_rest_are_not() {
        let n = 200;
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let hat: Vec<f64> = (0..n).map(|i| if i == 0 { 0.6 } else { 0.05 }).collect();
        let mut residuals: Vec<f64> = hat
            .iter()
            .map(|&h| (1.0 - h).sqrt() * standard_normal(&mut rng))
            .collect();
        residuals[0] = 12.0;
        let out = gaussian_influence(&residuals, &hat, 0.05).unwrap();
        assert_eq!(out.flagged, vec![0]);
        assert!(out.cooks[0] > 10.0 * out.cooks[1]);
        assert!((out.loo_residual[0] - 12.0 / 0.4).abs() < 1e-12);
    }

    #[test]
    fn influence_refuses_a_row_with_leverage_one() {
        let err =
            gaussian_influence(&[0.0, 1.0, -1.0, 0.5], &[1.0, 0.2, 0.2, 0.2], 0.05).unwrap_err();
        assert!(err.to_string().contains("leverage"));
    }

    #[test]
    fn false_alarm_rates_stay_near_alpha_for_clean_residuals() {
        let (_, codes) = clustered_residuals(15, 8, 0.0);
        let rate = cluster_false_alarm_rate(&codes, 200, 199, 11, 0.05).unwrap();
        assert!(rate <= 0.12, "cluster rate {rate}");

        let times: Vec<f64> = (0..120).map(|t| t as f64).collect();
        let rate = serial_false_alarm_rate(&times, 5, 200, 199, 11, 0.05).unwrap();
        assert!(rate <= 0.12, "serial rate {rate}");

        let hat = vec![0.1; 150];
        let rate = influence_false_alarm_rate(&hat, 1.0, 200, 11, 0.05).unwrap();
        assert!(rate <= 0.12, "influence rate {rate}");
    }
}
