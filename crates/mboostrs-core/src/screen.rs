//! Fast screening diagnostics: statistics that say, before any model
//! comparison, which shapes and which pairs are worth a cross-validated test.
//!
//! [`column_diagnostics`] describes how a response (or a partial residual)
//! depends on one column: how much a straight line, a flexible curve and a
//! single step each explain, whether the dependence is monotone, and the
//! column's own skewness and number of distinct values. These are
//! scatterplot diagnostics in the spirit of Tukey's and Wilkinson's
//! "scagnostics", measured on the quantities a shape choice needs.
//!
//! [`h_statistic`] is Friedman and Popescu's (2008) H statistic for a pair:
//! the share of the pair's joint partial dependence that the sum of the two
//! one-column partial dependences does not explain. It is zero when the two
//! columns act additively in the fitted model.
//!
//! [`screen_pairs`] and [`screen_main_effects`] (added for #298 phase 1,
//! 2026-10-08) rank candidate pairs and columns by the held-out residual
//! mean squared error their binned cell means remove — the same currency
//! autospec's interaction floor and suggestion ranking price in. The
//! half-split is a ChaCha8(seed) permutation and the accumulation order is
//! fixed, so the output is bit-identical for a given seed across thread
//! counts.

use crate::learners::quantile_bin_column;
use crate::MboostrsError;
use ndarray::{ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// How one column relates to a response.
#[derive(Debug, Clone)]
pub struct ColumnDiagnostics {
    /// Distinct values of the column.
    pub distinct: usize,
    /// Whether every value is a whole number, useful for count-like thresholds.
    pub integer_like: bool,
    /// Sample skewness of the column.
    pub skewness: f64,
    /// Pearson correlation with the response.
    pub pearson: f64,
    /// Spearman rank correlation with the response.
    pub spearman: f64,
    /// Share of the response variance a straight line explains.
    pub linear_r2: f64,
    /// Share a curve explains: the means of the response in up to 20 bins of
    /// equal counts of the column, adjusted for the bins' degrees of freedom
    /// (`1 - (1 - R^2)(n - 1)/(n - bins)`), so noise scores near zero.
    pub curve_r2: f64,
    /// Share the best single step (one split of the column) explains.
    pub step_r2: f64,
}

impl ColumnDiagnostics {
    /// What the curve adds over the line: the evidence for a smooth.
    pub fn curvature(&self) -> f64 {
        (self.curve_r2 - self.linear_r2).max(0.0)
    }

    /// What a single step explains beyond the line, relative to what the
    /// curve adds: near or above one when a jump carries the curve.
    pub fn step_share(&self) -> f64 {
        let over_line = (self.step_r2 - self.linear_r2).max(0.0);
        let curve = self.curvature();
        if curve <= 0.0 {
            0.0
        } else {
            over_line / curve
        }
    }
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let average = (i + j) as f64 / 2.0;
        for &k in &order[i..=j] {
            out[k] = average;
        }
        i = j + 1;
    }
    out
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    if saa > 0.0 && sbb > 0.0 {
        sab / (saa * sbb).sqrt()
    } else {
        0.0
    }
}

/// Diagnostics of how `y` depends on the column `x`.
pub fn column_diagnostics(x: &[f64], y: &[f64]) -> Result<ColumnDiagnostics, MboostrsError> {
    let n = x.len();
    if y.len() != n {
        return Err(MboostrsError::Shape(format!(
            "y has {} rows but x has {n}",
            y.len()
        )));
    }
    if n < 10 {
        return Err(MboostrsError::Shape(
            "diagnostics need at least ten rows".to_string(),
        ));
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "screen".to_string(),
            "values must be finite".to_string(),
        ));
    }
    let nf = n as f64;
    let mean_x = x.iter().sum::<f64>() / nf;
    let sd_x = (x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / nf).sqrt();
    let skewness = if sd_x > 0.0 {
        x.iter().map(|v| ((v - mean_x) / sd_x).powi(3)).sum::<f64>() / nf
    } else {
        0.0
    };
    let mean_y = y.iter().sum::<f64>() / nf;
    let tss: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
    let pearson = correlation(x, y);
    let spearman = correlation(&ranks(x), &ranks(y));
    let linear_r2 = pearson * pearson;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut distinct = 0;
    for k in 0..n {
        if k == 0 || x[order[k]] != x[order[k - 1]] {
            distinct += 1;
        }
    }

    // Curve: means in up to 20 bins of equal counts, never splitting a tie.
    let bins = distinct.clamp(1, 20);
    let mut rss_curve = 0.0;
    let mut start = 0;
    for b in 0..bins {
        let mut end = ((b + 1) * n) / bins;
        if b + 1 == bins {
            end = n;
        }
        while end < n && end > start && x[order[end]] == x[order[end - 1]] {
            end += 1;
        }
        if end <= start {
            continue;
        }
        let m = order[start..end].iter().map(|&k| y[k]).sum::<f64>() / (end - start) as f64;
        rss_curve += order[start..end]
            .iter()
            .map(|&k| (y[k] - m).powi(2))
            .sum::<f64>();
        start = end;
        if start >= n {
            break;
        }
    }
    // Adjusted for the bins' degrees of freedom: 20 bin means explain about
    // 19/(n-1) of pure noise.
    let curve_r2 = if tss > 0.0 && n > bins {
        (1.0 - (rss_curve / tss) * (nf - 1.0) / (nf - bins as f64)).max(0.0)
    } else {
        0.0
    };

    // Best single split, scanning the sorted column.
    let total: f64 = y.iter().sum();
    let (mut left_sum, mut best_between) = (0.0, 0.0f64);
    for k in 0..n - 1 {
        left_sum += y[order[k]];
        if x[order[k]] == x[order[k + 1]] {
            continue;
        }
        let nl = (k + 1) as f64;
        let nr = nf - nl;
        let between =
            left_sum * left_sum / nl + (total - left_sum).powi(2) / nr - total * total / nf;
        best_between = best_between.max(between);
    }
    let step_r2 = if tss > 0.0 { best_between / tss } else { 0.0 };

    Ok(ColumnDiagnostics {
        distinct,
        integer_like: x.iter().all(|value| value.fract() == 0.0),
        skewness,
        pearson,
        spearman,
        linear_r2,
        curve_r2,
        step_r2,
    })
}

/// Friedman's H^2 for a pair, `sum (f_jk - f_j - f_k)^2 / sum f_jk^2` over
/// the evaluation rows with each partial dependence centred first, and the
/// interaction variance `mean (f_jk - f_j - f_k)^2` in the response's squared
/// units. H^2 is a share and can be large for a pair whose joint effect is
/// negligible; the variance says how much prediction the interaction carries.
pub fn h_statistic_with_variance(
    pd_jk: &[f64],
    pd_j: &[f64],
    pd_k: &[f64],
) -> Result<(f64, f64), MboostrsError> {
    let h = h_statistic(pd_jk, pd_j, pd_k)?;
    let n = pd_jk.len() as f64;
    let centre = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| x - m).collect::<Vec<f64>>()
    };
    let (jk, j, k) = (centre(pd_jk), centre(pd_j), centre(pd_k));
    let variance = (0..pd_jk.len())
        .map(|i| (jk[i] - j[i] - k[i]).powi(2))
        .sum::<f64>()
        / n;
    Ok((h, variance))
}

/// Friedman's H^2 for a pair: `sum (f_jk - f_j - f_k)^2 / sum f_jk^2` over
/// the evaluation rows, with each partial dependence centred first.
pub fn h_statistic(pd_jk: &[f64], pd_j: &[f64], pd_k: &[f64]) -> Result<f64, MboostrsError> {
    let n = pd_jk.len();
    if pd_j.len() != n || pd_k.len() != n || n == 0 {
        return Err(MboostrsError::Shape(
            "the three partial dependences must have the same, non-zero length".to_string(),
        ));
    }
    let centre = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| x - m).collect::<Vec<f64>>()
    };
    let (jk, j, k) = (centre(pd_jk), centre(pd_j), centre(pd_k));
    let den: f64 = jk.iter().map(|v| v * v).sum();
    if den <= 0.0 {
        return Ok(0.0);
    }
    let num: f64 = (0..n).map(|i| (jk[i] - j[i] - k[i]).powi(2)).sum();
    Ok((num / den).clamp(0.0, 1.0))
}

/// One candidate interaction: the held-out residual mean squared error the
/// pair's binned cell means remove. Deterministic: the half-split is a
/// ChaCha8(seed) permutation, the bins are sort-based quantile bins learned
/// on the fit half, and ties break by `(left, right)` ascending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenedPair {
    /// Index of the pair's left column in `x`.
    pub left: usize,
    /// Index of the pair's right column in `x`.
    pub right: usize,
    /// Held-out residual MSE the pair's cell means remove, per held-out
    /// observation. At or below zero is evidence against the pair, not
    /// merely a weak ranking.
    pub removed: f64,
}

/// One candidate main effect: the held-out residual mean squared error the
/// column's binned means remove (the residual-aware counterpart of
/// [`column_diagnostics`]' marginal screen).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenedMain {
    /// Index of the column in `x`.
    pub column: usize,
    /// Held-out residual MSE the column's bin means remove, per held-out
    /// observation.
    pub removed: f64,
}

/// Binned cells shared by the pair and main-effect screens: per-column cell
/// codes for every row, the per-column cell counts, the ChaCha8 half-split
/// row indices, and the held-out residual variance both screens score
/// against.
struct Screening {
    codes: Vec<Vec<u32>>,
    cells: Vec<usize>,
    fit: Vec<usize>,
    held: Vec<usize>,
    baseline: f64,
}

/// Validate the frame, split the rows, and bin every column once.
///
/// Mirrors the Python screen's rules: the fit half is the first `n / 2`
/// entries of a ChaCha8(seed) shuffle of the row indices (a different stream
/// than numpy's `default_rng(seed)`, so scores move within noise); numeric
/// columns learn quantile edges on the fit half only and then apply them to
/// every row, so held-out rows occupy the same cells a fit-half row would;
/// factor columns (`factor_levels[j] > 0`) use their values as level codes
/// directly.
fn prepare_screening(
    x: ArrayView2<f64>,
    residual: &[f64],
    bins: usize,
    seed: u64,
    factor_levels: &[usize],
) -> Result<Screening, MboostrsError> {
    let (n, p) = (x.nrows(), x.ncols());
    if factor_levels.len() != p {
        return Err(MboostrsError::Shape(format!(
            "factor_levels has {} entries for {p} columns",
            factor_levels.len()
        )));
    }
    if residual.len() != n {
        return Err(MboostrsError::Shape(format!(
            "residual has {} rows but x has {n}",
            residual.len()
        )));
    }
    if n < 2 {
        return Err(MboostrsError::Shape(
            "screening needs at least two rows".to_string(),
        ));
    }
    if bins == 0 {
        return Err(MboostrsError::LearnerSetup(
            "screen".to_string(),
            "bins must be a positive integer".to_string(),
        ));
    }
    if x.iter().chain(residual.iter()).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "screen".to_string(),
            "values must be finite".to_string(),
        ));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let split = n / 2;
    let (fit, held) = order.split_at(split);
    let fit = fit.to_vec();
    let held = held.to_vec();

    let mut codes = Vec::with_capacity(p);
    let mut cells = Vec::with_capacity(p);
    for j in 0..p {
        if factor_levels[j] > 0 {
            let levels = factor_levels[j];
            for i in 0..n {
                let value = x[[i, j]];
                if value < 0.0 || value.fract() != 0.0 || value as usize >= levels {
                    return Err(MboostrsError::Shape(format!(
                        "factor column {j} has value {value}, outside its 0..{levels} level codes"
                    )));
                }
            }
            codes.push((0..n).map(|i| x[[i, j]] as u32).collect());
            cells.push(levels);
        } else {
            // Edges learned on the fit half only, then applied to every row:
            // a held-out row lands in the cell a fit-half row with the same
            // value would occupy. `partition_point` counts the edges below
            // the value, numpy `searchsorted`'s side='left' rule.
            let fit_values: Vec<f64> = fit.iter().map(|&i| x[[i, j]]).collect();
            let binned = quantile_bin_column(ArrayView1::from(&fit_values), bins)?;
            let cutpoints = binned.cutpoints;
            codes.push(
                (0..n)
                    .map(|i| cutpoints.partition_point(|&e| e < x[[i, j]]) as u32)
                    .collect(),
            );
            cells.push(binned.effective);
        }
    }

    let held_mean = held.iter().map(|&i| residual[i]).sum::<f64>() / held.len() as f64;
    let baseline = held
        .iter()
        .map(|&i| (residual[i] - held_mean).powi(2))
        .sum::<f64>()
        / held.len() as f64;
    Ok(Screening {
        codes,
        cells,
        fit,
        held,
        baseline,
    })
}

/// Held-out residual MSE the pair's cell means remove: accumulate
/// `(count, sum of residual)` per cell on the fit half, take means, then
/// measure the squared error they leave on the held half. Two linear passes
/// over the rows, sequential inside the pair so rayon across pairs cannot
/// perturb the arithmetic.
fn removed_pair(screening: &Screening, residual: &[f64], left: usize, right: usize) -> f64 {
    let (cells_left, cells_right) = (screening.cells[left], screening.cells[right]);
    let size = cells_left * cells_right;
    let mut count = vec![0u64; size];
    let mut sum = vec![0.0f64; size];
    for &i in &screening.fit {
        let cell =
            screening.codes[left][i] as usize * cells_right + screening.codes[right][i] as usize;
        count[cell] += 1;
        sum[cell] += residual[i];
    }
    let means: Vec<f64> = sum
        .iter()
        .zip(&count)
        .map(|(&total, &seen)| if seen > 0 { total / seen as f64 } else { 0.0 })
        .collect();
    let mut sse = 0.0;
    for &i in &screening.held {
        let cell =
            screening.codes[left][i] as usize * cells_right + screening.codes[right][i] as usize;
        let error = residual[i] - means[cell];
        sse += error * error;
    }
    screening.baseline - sse / screening.held.len() as f64
}

/// Held-out residual MSE one column's bin means remove; same two-pass
/// pattern as [`removed_pair`] with a single code per row.
fn removed_main(screening: &Screening, residual: &[f64], column: usize) -> f64 {
    let cells = screening.cells[column];
    let mut count = vec![0u64; cells];
    let mut sum = vec![0.0f64; cells];
    for &i in &screening.fit {
        let cell = screening.codes[column][i] as usize;
        count[cell] += 1;
        sum[cell] += residual[i];
    }
    let means: Vec<f64> = sum
        .iter()
        .zip(&count)
        .map(|(&total, &seen)| if seen > 0 { total / seen as f64 } else { 0.0 })
        .collect();
    let mut sse = 0.0;
    for &i in &screening.held {
        let cell = screening.codes[column][i] as usize;
        let error = residual[i] - means[cell];
        sse += error * error;
    }
    screening.baseline - sse / screening.held.len() as f64
}

/// Rank candidate pairs by the held-out residual MSE their binned cell means
/// remove, mirroring Python's `terms.screen_interactions` semantics.
///
/// Both columns of a pair are quantile-binned into `bins` cells (edges
/// learned on the fit half only, via [`crate::learners::quantile_bin_column`]),
/// with factor columns — `factor_levels[j] > 0` — using their values as that
/// many level cells instead. Cell means of `residual` are fit on one half of
/// the rows and the MSE they remove is measured on the other half, so a pair
/// that fits only noise removes nothing out of sample and scores at or below
/// zero. No model is fitted.
///
/// `removed` is per held-out observation, on the same scale as the difference
/// between two out-of-fold risks. An empty `pairs` list means all column
/// pairs. Sorting is `removed` descending with ties by `(left, right)`
/// ascending; `top_k` truncates after the sort. rayon runs across pairs
/// only, and each pair's accumulation is sequential, so the result is
/// bit-identical for a given seed across thread counts.
pub fn screen_pairs(
    x: ArrayView2<f64>,
    residual: &[f64],
    bins: usize,
    seed: u64,
    factor_levels: &[usize],
    pairs: &[(usize, usize)],
    top_k: Option<usize>,
) -> Result<Vec<ScreenedPair>, MboostrsError> {
    let screening = prepare_screening(x, residual, bins, seed, factor_levels)?;
    let p = x.ncols();
    let list: Vec<(usize, usize)> = if pairs.is_empty() {
        (0..p)
            .flat_map(|left| (left + 1..p).map(move |right| (left, right)))
            .collect()
    } else {
        pairs.to_vec()
    };
    for &(left, right) in &list {
        if left >= p || right >= p {
            return Err(MboostrsError::Shape(format!(
                "pair ({left}, {right}) names a column outside x's {p} columns"
            )));
        }
        if screening.cells[left]
            .checked_mul(screening.cells[right])
            .is_none()
        {
            return Err(MboostrsError::Shape(format!(
                "pair ({left}, {right}) has {} x {} cells, which overflows the cell index",
                screening.cells[left], screening.cells[right]
            )));
        }
    }
    use rayon::prelude::*;
    let mut scored: Vec<ScreenedPair> = list
        .par_iter()
        .map(|&(left, right)| ScreenedPair {
            left,
            right,
            removed: removed_pair(&screening, residual, left, right),
        })
        .collect();
    scored.sort_by(|a, b| {
        b.removed
            .total_cmp(&a.removed)
            .then_with(|| (a.left, a.right).cmp(&(b.left, b.right)))
    });
    if let Some(k) = top_k {
        scored.truncate(k);
    }
    Ok(scored)
}

/// Rank columns by the held-out residual MSE their 1-D bin means remove.
///
/// The residual-aware counterpart of [`column_diagnostics`]: it prices each
/// column in the same per-held-out-observation MSE currency as
/// [`screen_pairs`], so main effects and interactions can be ranked side by
/// side. The same binning, half-split and tie rules apply, sorting `removed`
/// descending with ties by column ascending; `top_k` truncates after the
/// sort. Column scoring is sequential (there are only as many columns as
/// features), so the result is bit-identical for a given seed.
pub fn screen_main_effects(
    x: ArrayView2<f64>,
    residual: &[f64],
    bins: usize,
    seed: u64,
    factor_levels: &[usize],
    top_k: Option<usize>,
) -> Result<Vec<ScreenedMain>, MboostrsError> {
    let screening = prepare_screening(x, residual, bins, seed, factor_levels)?;
    let mut scored: Vec<ScreenedMain> = (0..x.ncols())
        .map(|column| ScreenedMain {
            column,
            removed: removed_main(&screening, residual, column),
        })
        .collect();
    scored.sort_by(|a, b| {
        b.removed
            .total_cmp(&a.removed)
            .then_with(|| a.column.cmp(&b.column))
    });
    if let Some(k) = top_k {
        scored.truncate(k);
    }
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
            .collect()
    }

    #[test]
    fn a_line_a_curve_and_a_step_are_told_apart() {
        let x = grid(400);
        let line: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        let curve: Vec<f64> = x.iter().map(|v| v * v).collect();
        let step: Vec<f64> = x.iter().map(|v| if *v > 0.2 { 1.0 } else { 0.0 }).collect();
        let dl = column_diagnostics(&x, &line).unwrap();
        let dc = column_diagnostics(&x, &curve).unwrap();
        let ds = column_diagnostics(&x, &step).unwrap();
        assert!(dl.linear_r2 > 0.999 && dl.curvature() < 0.01);
        assert!(dc.linear_r2 < 0.01 && dc.curve_r2 > 0.95);
        assert!(ds.step_r2 > 0.99 && ds.step_share() > 0.9);
        assert!((dc.spearman).abs() < 0.05);
        assert!(!dc.integer_like);
        let counts: Vec<f64> = (0..400).map(|i| (i % 7) as f64).collect();
        assert!(column_diagnostics(&counts, &line).unwrap().integer_like);
    }

    #[test]
    fn h_is_zero_for_additive_and_positive_for_a_product() {
        let a = grid(21);
        let mut jk_add = vec![];
        let mut jk_prod = vec![];
        let mut j = vec![];
        let mut k = vec![];
        for &x in &a {
            for &z in &a {
                jk_add.push(x + z);
                jk_prod.push(x * z);
                j.push(x);
                k.push(z);
            }
        }
        assert!(h_statistic(&jk_add, &j, &k).unwrap() < 1e-12);
        let zeros = vec![0.0; j.len()];
        assert!(h_statistic(&jk_prod, &zeros, &zeros).unwrap() > 0.99);
    }

    // The #298 phase-1 screening regression suite (EVAL_298.md "Regression
    // tests", 2026-10-08): planted effects recovered in the top ranks,
    // factor level cells, bit-identical determinism across thread counts,
    // and the documented error shapes and degenerations.

    /// An n-rows × p-columns frame of iid uniform(-1, 1) values plus an
    /// independent uniform(-0.05, 0.05) noise draw per row, both from one
    /// ChaCha8 stream so every fixture is reproducible from its seed.
    fn uniform_frame(seed: u64, n: usize, p: usize) -> (ndarray::Array2<f64>, Vec<f64>) {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let x = ndarray::Array2::from_shape_fn((n, p), |(_, _)| rng.random::<f64>() * 2.0 - 1.0);
        let noise: Vec<f64> = (0..n).map(|_| (rng.random::<f64>() - 0.5) * 0.1).collect();
        (x, noise)
    }

    // Test 1: a planted product on iid uniform columns is rank 1 and the
    // runner-up noise pair is far below it. The response is its own
    // residual: no additive fit has removed any mains yet.
    #[test]
    fn a_planted_interaction_is_rank_one_and_far_above_the_runner_up() {
        let (x, noise) = uniform_frame(11, 4000, 6);
        let residual: Vec<f64> = (0..4000)
            .map(|i| x[[i, 0]] * x[[i, 1]] + noise[i])
            .collect();
        let scored = screen_pairs(x.view(), &residual, 4, 7, &[0; 6], &[], None).unwrap();
        assert_eq!(scored.first().map(|s| (s.left, s.right)), Some((0, 1)));
        assert!(scored[0].removed > 0.0);
        assert!(scored[1].removed < scored[0].removed / 4.0);
        let top_three = screen_pairs(x.view(), &residual, 4, 7, &[0; 6], &[], Some(3)).unwrap();
        assert_eq!(top_three.len(), 3);
        assert_eq!(top_three, scored[..3].to_vec());
    }

    // Test 2: a planted main effect ranks first, and a column carrying only
    // noise fits its own half's noise, removing nothing out of sample.
    #[test]
    fn a_planted_main_effect_ranks_first_and_noise_columns_remove_nothing() {
        let (x, noise) = uniform_frame(23, 4000, 5);
        let residual: Vec<f64> = (0..4000)
            .map(|i| x[[i, 0]] * x[[i, 0]] + noise[i])
            .collect();
        let scored = screen_main_effects(x.view(), &residual, 4, 5, &[0; 5], None).unwrap();
        assert_eq!(scored[0].column, 0);
        assert!(scored[0].removed > 0.0);
        assert!(scored[1..].iter().all(|s| s.removed <= 0.0), "{scored:?}");
    }

    // Test 3: on a noise-only frame every pair's cell means fit only the
    // fit half's noise, so `removed` stays at or below zero — the
    // out-of-sample property that makes autospec's 1%-of-variance floor
    // safe.
    #[test]
    fn a_noise_only_frame_scores_every_pair_at_or_below_zero() {
        let (x, noise) = uniform_frame(37, 4000, 5);
        let scored = screen_pairs(x.view(), &noise, 4, 3, &[0; 5], &[], None).unwrap();
        assert_eq!(scored.len(), 10);
        assert!(scored.iter().all(|s| s.removed <= 0.0), "{scored:?}");
    }

    // Test 4: a group column's levels are cells, not quantile bins. A slope
    // that alternates by level is plain to see when each level is its own
    // cell, but four quantile bins merge adjacent levels with opposite
    // slopes into one cell whose mean is zero and the same structure goes
    // undetected — the `groups` rationale of `terms.screen_interactions`.
    #[test]
    fn factor_levels_are_cells_not_quantile_bins() {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let n = 4000;
        let mut rng = ChaCha8Rng::seed_from_u64(41);
        let mut x = ndarray::Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            x[[i, 0]] = (rng.random::<f64>() * 8.0).floor();
            x[[i, 1]] = rng.random::<f64>() * 2.0 - 1.0;
            x[[i, 2]] = rng.random::<f64>() * 2.0 - 1.0;
        }
        let residual: Vec<f64> = (0..n)
            .map(|i| {
                let slope = if x[[i, 0]] as i64 % 2 == 0 { 1.0 } else { -1.0 };
                slope * x[[i, 1]] + (rng.random::<f64>() - 0.5) * 0.1
            })
            .collect();
        let variance = residual.iter().map(|v| v * v).sum::<f64>() / n as f64;

        let by_levels =
            screen_pairs(x.view(), &residual, 4, 3, &[8, 0, 0], &[(0, 1)], None).unwrap();
        assert_eq!(by_levels.len(), 1);
        assert!(by_levels[0].removed > 0.2, "{:?}", by_levels[0].removed);

        let by_bins = screen_pairs(x.view(), &residual, 4, 3, &[0; 3], &[(0, 1)], None).unwrap();
        assert!(
            by_bins[0].removed < 0.01 * variance,
            "{:?} vs variance {variance}",
            by_bins[0].removed
        );
    }

    // Test 5: the same seed produces bit-identical scores, and each pair's
    // accumulation is sequential, so rayon across pairs cannot change the
    // arithmetic: one thread and four produce the same bits. A different
    // seed moves the half-split and with it the scores.
    #[test]
    fn screening_is_bit_identical_across_calls_and_thread_counts() {
        let (x, noise) = uniform_frame(59, 2000, 7);
        let residual: Vec<f64> = (0..2000)
            .map(|i| x[[i, 0]] * x[[i, 1]] + noise[i])
            .collect();
        let run = || screen_pairs(x.view(), &residual, 4, 13, &[0; 7], &[], None).unwrap();
        let first = run();
        assert_eq!(first, run());
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(run);
        assert!(serial.iter().zip(&parallel).all(|(a, b)| {
            (a.left, a.right) == (b.left, b.right) && a.removed.to_bits() == b.removed.to_bits()
        }));
        let reseeded = screen_pairs(x.view(), &residual, 4, 14, &[0; 7], &[], None).unwrap();
        assert!(reseeded
            .iter()
            .zip(&first)
            .any(|(a, b)| a.removed != b.removed));
    }

    // Test 6: the documented error shapes and graceful degenerations —
    // mismatched lengths, non-finite input, a zero bin count, an
    // out-of-range pair index and a factor value outside its level codes
    // are errors; a constant column collapses to a single cell, so the
    // pair (constant, column) scores exactly the column's own main screen;
    // a two-column frame screens exactly its one pair.
    #[test]
    fn screening_edge_cases_error_or_degenerate_gracefully() {
        let (x, noise) = uniform_frame(71, 600, 3);
        let short: Vec<f64> = noise[..599].to_vec();
        assert!(matches!(
            screen_pairs(x.view(), &short, 4, 1, &[0; 3], &[], None),
            Err(MboostrsError::Shape(_))
        ));
        assert!(matches!(
            screen_main_effects(x.view(), &short, 4, 1, &[0; 3], None),
            Err(MboostrsError::Shape(_))
        ));
        let mut nan_x = x.clone();
        nan_x[[0, 0]] = f64::NAN;
        assert!(matches!(
            screen_pairs(nan_x.view(), &noise, 4, 1, &[0; 3], &[], None),
            Err(MboostrsError::BadResponse(_, _))
        ));
        assert!(matches!(
            screen_pairs(x.view(), &noise, 0, 1, &[0; 3], &[], None),
            Err(MboostrsError::LearnerSetup(_, _))
        ));
        assert!(matches!(
            screen_pairs(x.view(), &noise, 4, 1, &[0; 3], &[(0, 3)], None),
            Err(MboostrsError::Shape(_))
        ));
        assert!(matches!(
            screen_pairs(x.view(), &noise, 4, 1, &[0; 2], &[], None),
            Err(MboostrsError::Shape(_))
        ));
        assert!(matches!(
            screen_pairs(x.view(), &noise, 4, 1, &[2, 0, 0], &[(0, 1)], None),
            Err(MboostrsError::Shape(_))
        ));

        let mut constant = ndarray::Array2::<f64>::zeros((600, 2));
        for i in 0..600 {
            constant[[i, 0]] = 5.0;
            constant[[i, 1]] = x[[i, 1]];
        }
        let pairs = screen_pairs(constant.view(), &noise, 4, 2, &[0; 2], &[], None).unwrap();
        let mains = screen_main_effects(constant.view(), &noise, 4, 2, &[0; 2], None).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(mains.len(), 2);
        assert_eq!(pairs[0].removed.to_bits(), mains[1].removed.to_bits());
    }
}
