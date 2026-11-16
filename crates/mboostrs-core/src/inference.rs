//! Variable importance, bootstrap bookkeeping and percentile intervals.
//!
//! These are the numeric steps behind `varimp()`, `confint()` and the
//! bootstrap refits. The refits themselves are orchestrated by the Python and
//! R front ends, which build each replicate's model and report, per replicate,
//! whether it failed; the functions here turn those reports into counts,
//! intervals and resampling weights.

use ndarray::{Array2, ArrayView2};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::MboostrsError;

/// Risk reduction and selection frequency for each group of updates.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribution {
    /// Sum of the change credited to each group. Shares of the total absolute
    /// reduction when `percent` was requested, otherwise the raw sums.
    pub reduction: Vec<f64>,
    /// Number of updates in each group divided by the number of updates.
    pub selfreq: Vec<f64>,
}

/// Credit each update's risk change to the group it belongs to.
///
/// `changes[i]` is the risk reduction of update `i`, already divided by the
/// number of observations, and `groups[i]` is the group (a learner, or a
/// learner name shared by several learners) that update `i` selected. With
/// `percent = true` each group's reduction is divided by the sum of the
/// absolute group reductions, so the shares add to one in absolute value.
///
/// Updates are summed in order, so the result does not depend on how the
/// groups were formed.
pub fn varimp_attribution(
    changes: &[f64],
    groups: &[usize],
    n_groups: usize,
    percent: bool,
) -> Result<Attribution, MboostrsError> {
    if changes.len() != groups.len() {
        return Err(MboostrsError::Shape(format!(
            "varimp: {} risk changes but {} selected updates; they must pair up one to one",
            changes.len(),
            groups.len()
        )));
    }
    if let Some(&bad) = groups.iter().find(|&&g| g >= n_groups) {
        return Err(MboostrsError::Shape(format!(
            "varimp: group index {bad} is outside the {n_groups} groups"
        )));
    }
    let mut reduction = vec![0.0f64; n_groups];
    let mut counts = vec![0usize; n_groups];
    for (&change, &group) in changes.iter().zip(groups) {
        reduction[group] += change;
        counts[group] += 1;
    }
    let total = groups.len();
    let selfreq = counts
        .iter()
        .map(|&c| {
            if total > 0 {
                c as f64 / total as f64
            } else {
                0.0
            }
        })
        .collect();
    if percent {
        let total_abs: f64 = reduction.iter().map(|v| v.abs()).sum();
        if total_abs > 0.0 {
            for value in reduction.iter_mut() {
                *value /= total_abs;
            }
        }
    }
    Ok(Attribution { reduction, selfreq })
}

/// Row multiplicities for `b` bootstrap samples drawn over whole groups.
///
/// `codes[i]` is the group of row `i`, as a dense index, and `eligible[i]`
/// says whether row `i` may be drawn. Every group with at least one eligible
/// row is a unit. Each sample draws as many units as there are, uniformly and
/// with replacement, and an eligible row gets the number of times its unit
/// was drawn; an ineligible row gets zero. Column `j` of the result is sample
/// `j`, so the rows of one group enter or leave a sample together.
///
/// The draws come from a ChaCha8 stream seeded with `seed`, so a seed fixes
/// the samples on every platform and in every thread count.
pub fn group_bootstrap_counts(
    codes: &[usize],
    eligible: &[bool],
    b: usize,
    seed: u64,
) -> Result<Array2<f64>, MboostrsError> {
    if codes.len() != eligible.len() {
        return Err(MboostrsError::Shape(format!(
            "groups= needs one label per row: got {} labels for {} rows",
            codes.len(),
            eligible.len()
        )));
    }
    let n_units = codes.iter().copied().max().map_or(0, |m| m + 1);
    let mut has_eligible = vec![false; n_units];
    for (&code, &ok) in codes.iter().zip(eligible) {
        if ok {
            has_eligible[code] = true;
        }
    }
    let pool: Vec<usize> = (0..n_units).filter(|&u| has_eligible[u]).collect();
    if pool.len() < 2 {
        return Err(MboostrsError::Control(format!(
            "groups= needs at least two groups with positive weight to resample; found {}",
            pool.len()
        )));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut counts = Array2::<f64>::zeros((codes.len(), b));
    let mut drawn = vec![0usize; n_units];
    for column in 0..b {
        drawn.iter_mut().for_each(|d| *d = 0);
        for _ in 0..pool.len() {
            let unit = pool[rng.random_range(0..pool.len())];
            drawn[unit] += 1;
        }
        for (row, (&code, &ok)) in codes.iter().zip(eligible).enumerate() {
            if ok {
                counts[[row, column]] = drawn[code] as f64;
            }
        }
    }
    Ok(counts)
}

/// Per-replicate outcome bit: the replicate produced no estimate.
pub const REPLICATE_FAILED: u8 = 1;
/// Per-replicate outcome bit: the inner `mstop` cross-validation failed, so
/// the replicate was fitted at the outer model's `mstop`.
pub const REPLICATE_UNTUNED: u8 = 2;
/// Per-replicate outcome bit: the inner cross-validation chose the largest
/// `mstop` it was allowed.
pub const REPLICATE_AT_CAP: u8 = 4;

/// How many replicates had each outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BootstrapCounts {
    /// Replicates with `REPLICATE_FAILED` set.
    pub failed: usize,
    /// Replicates with `REPLICATE_UNTUNED` set.
    pub untuned: usize,
    /// Replicates with `REPLICATE_AT_CAP` set.
    pub at_cap: usize,
}

/// Count the outcome bits of every replicate.
///
/// A replicate can carry more than one bit: an inner `mstop` search that
/// failed and a later failure of the replicate's own fit both count.
pub fn bootstrap_counts(flags: &[u8]) -> BootstrapCounts {
    let mut counts = BootstrapCounts::default();
    for &flag in flags {
        if flag & REPLICATE_FAILED != 0 {
            counts.failed += 1;
        }
        if flag & REPLICATE_UNTUNED != 0 {
            counts.untuned += 1;
        }
        if flag & REPLICATE_AT_CAP != 0 {
            counts.at_cap += 1;
        }
    }
    counts
}

/// Percentile interval of each column of a bootstrap matrix.
///
/// `values` holds one row per replicate and one column per quantity (a
/// coefficient, or an effect at one evaluation point). The interval for
/// column `j` is the pair of type-7 quantiles (R's default, the one numpy
/// calls `linear`) at `(1 - level) / 2` and `1 - (1 - level) / 2` of the
/// non-missing values in that column. A column with no non-missing value
/// gets `NaN` in both bounds.
///
/// With `centre_rows = true` each replicate is first shifted by its own mean
/// over the columns, so the interval is that of the centred curve, not of the
/// curve as fitted. Every value must be finite for that to be defined.
///
/// Returns `(lower, upper)`, one entry per column.
pub fn percentile_interval(
    values: ArrayView2<'_, f64>,
    level: f64,
    centre_rows: bool,
) -> Result<(Vec<f64>, Vec<f64>), MboostrsError> {
    let q_low = (1.0 - level) / 2.0;
    let q_high = 1.0 - (1.0 - level) / 2.0;
    if !(0.0..=1.0).contains(&q_low) || !(0.0..=1.0).contains(&q_high) {
        return Err(MboostrsError::Control(format!(
            "interval level must lie in [0, 1], got {level}"
        )));
    }
    let n_col = values.ncols();
    let mut matrix = values.to_owned();
    if centre_rows && n_col > 0 {
        for mut row in matrix.rows_mut() {
            let mean = row.iter().sum::<f64>() / n_col as f64;
            row.mapv_inplace(|v| v - mean);
        }
    }
    let mut lower = Vec::with_capacity(n_col);
    let mut upper = Vec::with_capacity(n_col);
    for column in matrix.columns() {
        let mut present: Vec<f64> = column.iter().copied().filter(|v| !v.is_nan()).collect();
        present.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lower.push(type7_sorted(&present, q_low));
        upper.push(type7_sorted(&present, q_high));
    }
    Ok((lower, upper))
}

/// Linear-interpolation quantile of sorted values, numpy's `linear` method.
fn type7_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let virtual_index = q * (n as f64 - 1.0);
    let lo = virtual_index.floor();
    let gamma = virtual_index - lo;
    let lo = lo as usize;
    let hi = (lo + 1).min(n - 1);
    let a = sorted[lo];
    let b = sorted[hi];
    // numpy's _lerp: interpolate from the nearer end so the result is exact at
    // both ends of the interval.
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn varimp_splits_reduction_by_group_and_normalizes() {
        // Risk path 10, 8, 7.5, 7.5 over n = 2 observations: changes are
        // 1, 0.25 and 0 after the division by n.
        let changes = [1.0, 0.25, 0.0];
        let groups = [0usize, 1, 0];
        let raw = varimp_attribution(&changes, &groups, 2, false).unwrap();
        assert_eq!(raw.reduction, vec![1.0, 0.25]);
        assert_eq!(raw.selfreq, vec![2.0 / 3.0, 1.0 / 3.0]);
        let shares = varimp_attribution(&changes, &groups, 2, true).unwrap();
        assert!((shares.reduction[0] - 0.8).abs() < 1e-15);
        assert!((shares.reduction[1] - 0.2).abs() < 1e-15);
    }

    #[test]
    fn varimp_rejects_mismatched_lengths_and_bad_groups() {
        assert!(varimp_attribution(&[1.0], &[0, 1], 2, false).is_err());
        assert!(varimp_attribution(&[1.0], &[3], 2, false).is_err());
    }

    #[test]
    fn varimp_with_no_updates_gives_zero_frequencies() {
        let out = varimp_attribution(&[], &[], 2, true).unwrap();
        assert_eq!(out.reduction, vec![0.0, 0.0]);
        assert_eq!(out.selfreq, vec![0.0, 0.0]);
    }

    #[test]
    fn group_draws_are_whole_groups_and_respect_eligibility() {
        let codes = [0usize, 0, 1, 1, 2, 2];
        let eligible = [true, true, true, true, false, false];
        let counts = group_bootstrap_counts(&codes, &eligible, 20, 7).unwrap();
        assert_eq!(counts.dim(), (6, 20));
        for j in 0..20 {
            assert_eq!(
                counts[[0, j]],
                counts[[1, j]],
                "rows of group 0 move together"
            );
            assert_eq!(
                counts[[2, j]],
                counts[[3, j]],
                "rows of group 1 move together"
            );
            assert_eq!(counts[[4, j]], 0.0);
            assert_eq!(counts[[5, j]], 0.0);
            let total: f64 = counts.column(j).sum();
            assert_eq!(total, 2.0 * 2.0, "two units of two rows each, drawn twice");
        }
    }

    #[test]
    fn group_draws_are_reproducible_for_a_seed() {
        let codes = [0usize, 1, 2, 3, 3];
        let eligible = [true; 5];
        let a = group_bootstrap_counts(&codes, &eligible, 8, 99).unwrap();
        let b = group_bootstrap_counts(&codes, &eligible, 8, 99).unwrap();
        let c = group_bootstrap_counts(&codes, &eligible, 8, 100).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn group_draws_need_two_eligible_groups() {
        let codes = [0usize, 0, 1];
        let eligible = [true, true, false];
        assert!(group_bootstrap_counts(&codes, &eligible, 3, 1).is_err());
    }

    #[test]
    fn tally_counts_each_bit_once_per_replicate() {
        let flags = [
            0,
            REPLICATE_FAILED,
            REPLICATE_UNTUNED | REPLICATE_FAILED,
            REPLICATE_AT_CAP,
        ];
        assert_eq!(
            bootstrap_counts(&flags),
            BootstrapCounts {
                failed: 2,
                untuned: 1,
                at_cap: 1
            }
        );
    }

    #[test]
    fn percentile_matches_numpy_linear_quantiles() {
        // numpy.quantile([1, 2, 3, 4, 10], [0.025, 0.975]) = [1.1, 9.4]: the
        // indices are 0.1 and 3.9 on the sorted values.
        let values = array![[1.0], [2.0], [3.0], [4.0], [10.0]];
        let (lo, hi) = percentile_interval(values.view(), 0.95, false).unwrap();
        assert!((lo[0] - 1.1).abs() < 1e-12);
        assert!((hi[0] - 9.4).abs() < 1e-12);
    }

    #[test]
    fn percentile_skips_missing_values_and_reports_all_missing_as_nan() {
        let values = array![[1.0, f64::NAN], [3.0, f64::NAN], [5.0, f64::NAN]];
        // Present values 1, 3, 5 at level 0.5: quantiles 0.25 and 0.75 are
        // at indices 0.5 and 1.5, giving 2 and 4.
        let (lo, hi) = percentile_interval(values.view(), 0.5, false).unwrap();
        assert!((lo[0] - 2.0).abs() < 1e-12);
        assert!((hi[0] - 4.0).abs() < 1e-12);
        assert!(lo[1].is_nan() && hi[1].is_nan());
    }

    #[test]
    fn centring_subtracts_each_replicates_own_mean() {
        // Row means are 2 and 15, so the centred rows are [-1, 1] and [-5, 5].
        // At level 1 the interval is the full range of each column.
        let values = array![[1.0, 3.0], [10.0, 20.0]];
        let (lo, hi) = percentile_interval(values.view(), 1.0, true).unwrap();
        assert_eq!(lo, vec![-5.0, 1.0]);
        assert_eq!(hi, vec![-1.0, 5.0]);
    }

    #[test]
    fn level_outside_the_unit_interval_is_refused() {
        let values = array![[1.0], [2.0]];
        assert!(percentile_interval(values.view(), 1.5, false).is_err());
    }
}
