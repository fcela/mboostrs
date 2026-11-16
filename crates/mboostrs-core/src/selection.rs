//! Paired per-fold comparison of two specifications on the same folds.
//!
//! Model search decides whether a more complex specification is worth its
//! extra terms by comparing out-of-fold risk across folds, and the two
//! candidates share their folds, so the comparison is paired: the quantity
//! that carries the uncertainty is the per-fold difference, not either
//! candidate's spread. The rule used throughout the search layer is a
//! multiple of that difference's standard error ("two-fold-SE" in the
//! specification-search literature), which is a search heuristic, not a
//! hypothesis test: it is a threshold on how consistently the folds agree
//! that the extra complexity pays, not a statement about a population
//! parameter at a nominal level.

use crate::MboostrsError;

/// The paired comparison of a candidate against the incumbent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedGain {
    /// Mean risk of the incumbent, across all fold scores.
    pub baseline: f64,
    /// Mean risk of the candidate, across all fold scores.
    pub candidate: f64,
    /// `baseline - candidate`: the risk reduction, positive when the
    /// candidate is better on the same folds.
    pub gain: f64,
    /// Standard error of the mean per-fold difference. With more than one
    /// partition, folds in different partitions reuse rows, so the variance
    /// is computed within each partition and averaged, as for one split.
    pub se: f64,
    /// The gain in units of its standard error.
    pub t: f64,
    /// Whether `gain > threshold_se * se`.
    pub passed: bool,
    /// Folds whose risk rose under the candidate.
    pub worse_folds: usize,
    /// Fold scores the comparison rests on.
    pub n_folds: usize,
}

/// The standard error of the mean of a paired difference vector.
///
/// With `partitions` greater than one, `values` holds that many partitions
/// of folds stacked in order. Folds in different partitions reuse the same
/// rows, so an ordinary standard error over the whole stack understates the
/// uncertainty about new data; the variance is therefore computed within
/// each partition and averaged across partitions.
pub fn paired_se(difference: &[f64], partitions: usize) -> f64 {
    let n = difference.len();
    if n < 2 {
        return f64::NAN;
    }
    let partitions = partitions.max(1);
    if partitions > 1 && n % partitions == 0 && n / partitions >= 2 {
        let per = n / partitions;
        let mut mean_variance = 0.0;
        for p in 0..partitions {
            let slice = &difference[p * per..(p + 1) * per];
            let mean = slice.iter().sum::<f64>() / per as f64;
            let variance = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (per - 1) as f64;
            mean_variance += variance;
        }
        mean_variance /= partitions as f64;
        return (mean_variance / per as f64).sqrt();
    }
    let mean = difference.iter().sum::<f64>() / n as f64;
    let variance = difference.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (variance / n as f64).sqrt()
}

/// Compare a candidate's fold scores with the incumbent's on the same folds.
///
/// `before` and `after` must be the same length and hold at least two
/// scores; `partitions` is how many stacked fold partitions they contain
/// (one for a plain k-fold), and `threshold_se` the number of standard
/// errors the gain must clear. A degenerate comparison -- two folds, or
/// candidates whose fold scores agree exactly -- cannot clear any positive
/// threshold, and reports `se` as it stands so the caller can say why.
pub fn try_paired_gain(
    before: &[f64],
    after: &[f64],
    partitions: usize,
    threshold_se: f64,
) -> Result<PairedGain, MboostrsError> {
    if before.len() != after.len() {
        return Err(MboostrsError::Shape(format!(
            "fold score vectors differ in length: {} vs {}",
            before.len(),
            after.len()
        )));
    }
    if before.len() < 2 {
        return Err(MboostrsError::Shape(format!(
            "a paired comparison needs at least 2 folds, got {}",
            before.len()
        )));
    }
    if before.iter().chain(after.iter()).any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "paired_gain".to_string(),
            "fold scores must all be finite".to_string(),
        ));
    }
    if !(threshold_se.is_finite() && threshold_se >= 0.0) {
        return Err(MboostrsError::Control(format!(
            "threshold_se must be a non-negative finite number, got {threshold_se}"
        )));
    }
    let n = before.len();
    let baseline = before.iter().sum::<f64>() / n as f64;
    let candidate = after.iter().sum::<f64>() / n as f64;
    let difference: Vec<f64> = after
        .iter()
        .zip(before.iter())
        .map(|(a, b)| a - b)
        .collect();
    let se = paired_se(&difference, partitions);
    let gain = baseline - candidate;
    let worse_folds = difference.iter().filter(|d| **d > 0.0).count();
    let passed = se.is_finite() && gain > threshold_se * se;
    Ok(PairedGain {
        baseline,
        candidate,
        gain,
        se,
        t: if se > 0.0 { gain / se } else { f64::INFINITY },
        passed,
        worse_folds,
        n_folds: n,
    })
}

/// The comparison, panicking on invalid input; see [`try_paired_gain`].
pub fn paired_gain(
    before: &[f64],
    after: &[f64],
    partitions: usize,
    threshold_se: f64,
) -> PairedGain {
    try_paired_gain(before, after, partitions, threshold_se).expect("paired gain inputs are valid")
}

/// The sign-flip test of the selected specification against the curve median.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveSignTest {
    /// Mean over folds of `curve median - selected`, per fold: positive when
    /// the selected specification has lower out-of-fold risk than the median
    /// specification on the same folds.
    pub statistic: f64,
    /// One-sided sign-flip p-value: the share of sign patterns whose
    /// statistic is at least the observed one (add-one corrected when the
    /// patterns are sampled rather than enumerated).
    pub p_value: f64,
    /// Sign-flip units: folds for one partition, partitions for stacked ones.
    pub units: usize,
    /// Whether every sign pattern was enumerated.
    pub exact: bool,
    /// Sign patterns evaluated: `2^units` when exact, else `max_draws`.
    pub draws: usize,
}

/// Largest unit count enumerated exhaustively (2^16 patterns).
const CURVE_TEST_EXACT_UNITS: usize = 16;

/// Test whether the selected specification beats the curve's median on the
/// shared folds by more than the signs of the paired differences allow.
///
/// `scores[j]` holds specification `j`'s out-of-fold fold scores (lower is
/// better), every row over the same folds. The curve median of fold `f` is
/// the median over all rows, the selected row included; `d_f` is that median
/// minus the selected row's score. Under the null that the selected row is
/// no better than a typical member of the curve, each `d_f` is equally likely
/// to have either sign, so the statistic is compared with its distribution
/// over random sign flips of the paired differences.
///
/// With `partitions` greater than one, the folds are stacked partitions and
/// the flip unit is a whole partition: folds of one partition share rows with
/// each other, so their signs are not flipped independently of one another.
///
/// The selected folds are reused here, because the curve and the selection
/// were both computed on them. The p-value describes the curve; it is not a
/// confirmatory test on new data.
pub fn try_curve_sign_test(
    scores: &[Vec<f64>],
    selected: usize,
    partitions: usize,
    max_draws: usize,
    seed: u64,
) -> Result<CurveSignTest, MboostrsError> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    if scores.len() < 2 {
        return Err(MboostrsError::Shape(format!(
            "a specification curve test needs at least 2 specifications, got {}",
            scores.len()
        )));
    }
    if selected >= scores.len() {
        return Err(MboostrsError::Shape(format!(
            "selected row {selected} is outside the {} specifications",
            scores.len()
        )));
    }
    let folds = scores[0].len();
    if folds < 2 || scores.iter().any(|row| row.len() != folds) {
        return Err(MboostrsError::Shape(format!(
            "every specification needs the same {folds} fold scores, at least 2"
        )));
    }
    if scores.iter().flatten().any(|v| !v.is_finite()) {
        return Err(MboostrsError::BadResponse(
            "curve_sign_test".to_string(),
            "fold scores must all be finite".to_string(),
        ));
    }
    if max_draws == 0 {
        return Err(MboostrsError::Control(
            "max_draws must be at least 1".to_string(),
        ));
    }
    let mut difference = Vec::with_capacity(folds);
    for f in 0..folds {
        let mut column: Vec<f64> = scores.iter().map(|row| row[f]).collect();
        column.sort_by(|a, b| a.total_cmp(b));
        let k = column.len();
        let median = if k % 2 == 1 {
            column[k / 2]
        } else {
            0.5 * (column[k / 2 - 1] + column[k / 2])
        };
        difference.push(median - scores[selected][f]);
    }
    let total = folds as f64;
    let statistic = difference.iter().sum::<f64>() / total;

    // Flip units: one per fold, or one per stacked partition.
    let partitions = partitions.max(1);
    let unit_sums: Vec<f64> =
        if partitions > 1 && folds % partitions == 0 && folds / partitions >= 2 {
            let per = folds / partitions;
            difference
                .chunks(per)
                .map(|c| c.iter().sum::<f64>())
                .collect()
        } else {
            difference.clone()
        };
    let weights: Vec<f64> = unit_sums.iter().map(|s| s / total).collect();
    let units = weights.len();
    let threshold = statistic - 1e-12 * statistic.abs().max(1.0);

    let (count, draws, exact) = if units <= CURVE_TEST_EXACT_UNITS {
        let patterns = 1usize << units;
        let mut count = 0usize;
        for mask in 0..patterns {
            let flipped: f64 = weights
                .iter()
                .enumerate()
                .map(|(u, w)| if (mask >> u) & 1 == 1 { -w } else { *w })
                .sum();
            if flipped >= threshold {
                count += 1;
            }
        }
        (count, patterns, true)
    } else {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut count = 0usize;
        for _ in 0..max_draws {
            let flipped: f64 = weights
                .iter()
                .map(|w| if rng.random::<bool>() { *w } else { -*w })
                .sum();
            if flipped >= threshold {
                count += 1;
            }
        }
        (count, max_draws, false)
    };
    let p_value = if exact {
        count as f64 / draws as f64
    } else {
        (1 + count) as f64 / (1 + draws) as f64
    };
    Ok(CurveSignTest {
        statistic,
        p_value,
        units,
        exact,
        draws,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_consistent_gain_clears_two_standard_errors() {
        let before = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        let after: Vec<f64> = before.iter().map(|v| v - 0.1).collect();
        let gain = paired_gain(&before, &after, 1, 2.0);
        assert!(gain.passed);
        assert!((gain.gain - 0.1).abs() < 1e-12);
        assert!(gain.se < 1e-12);
        assert_eq!(gain.worse_folds, 0);
    }

    #[test]
    fn a_gain_inside_the_noise_does_not_clear() {
        // One fold disagrees: the mean improves, the spread does not.
        let before = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let after = vec![0.9, 0.9, 0.9, 0.9, 1.3];
        let gain = paired_gain(&before, &after, 1, 2.0);
        assert!(!gain.passed);
        assert!(gain.se > 0.0);
        assert_eq!(gain.worse_folds, 1);
    }

    #[test]
    fn identical_scores_never_clear_a_positive_threshold() {
        let scores = vec![2.0; 5];
        let gain = paired_gain(&scores, &scores, 1, 2.0);
        assert!(!gain.passed);
        assert_eq!(gain.gain, 0.0);
    }

    #[test]
    fn a_higher_threshold_demands_more_evidence() {
        // The folds mostly agree, two disagree: the mean gain is 0.01 with
        // se 0.0053, so a one-SE rule adopts and a two-SE rule does not.
        let before = vec![1.0; 8];
        let after = vec![0.98, 0.98, 0.98, 0.98, 0.98, 1.0, 1.0, 1.02];
        let one = paired_gain(&before, &after, 1, 1.0);
        assert!(one.passed);
        assert!((one.t - 1.87).abs() < 0.02);
        assert!(!paired_gain(&before, &after, 1, 2.0).passed);
    }

    #[test]
    fn a_gain_the_folds_agree_on_exactly_clears_any_threshold() {
        // Every fold improves by the same amount: there is no spread to
        // argue about, so the rule admits it whatever threshold is stated.
        let before = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let after: Vec<f64> = before.iter().map(|v| v - 0.25).collect();
        let gain = paired_gain(&before, &after, 1, 2.0);
        assert!(gain.passed);
        assert_eq!(gain.se, 0.0);
        assert!(gain.t.is_infinite());
    }

    #[test]
    fn partitions_are_averaged_within_not_pooled() {
        // Two partitions of the same two folds, the second shifted: pooling
        // would see the shift as variance, averaging within does not.
        let before = [1.0, 1.0, 2.0, 2.0];
        let after = [0.9, 0.9, 1.9, 1.9];
        let difference: Vec<f64> = after
            .iter()
            .zip(before.iter())
            .map(|(a, b)| a - b)
            .collect();
        let pooled = paired_se(&difference, 1);
        let within = paired_se(&difference, 2);
        assert!(within < pooled);
        assert!(within < 1e-12);
    }

    #[test]
    fn mismatched_lengths_are_refused() {
        let err = try_paired_gain(&[1.0, 2.0], &[1.0], 1, 2.0).unwrap_err();
        assert!(matches!(err, MboostrsError::Shape(_)));
    }

    #[test]
    fn one_fold_is_not_a_comparison() {
        let err = try_paired_gain(&[1.0], &[0.9], 1, 2.0).unwrap_err();
        assert!(matches!(err, MboostrsError::Shape(_)));
    }

    #[test]
    fn non_finite_scores_are_refused() {
        let err = try_paired_gain(&[1.0, f64::NAN], &[0.9, 0.8], 1, 2.0).unwrap_err();
        assert!(matches!(err, MboostrsError::BadResponse(_, _)));
    }

    #[test]
    fn a_negative_threshold_is_refused() {
        let err = try_paired_gain(&[1.0, 1.1], &[0.9, 1.0], 1, -1.0).unwrap_err();
        assert!(matches!(err, MboostrsError::Control(_)));
    }

    #[test]
    fn a_selected_row_below_the_median_on_every_fold_is_enumerated() {
        // Three rows; the selected one (index 1) is lowest on all five folds.
        let scores = vec![
            vec![1.0, 1.1, 0.9, 1.05, 0.95],
            vec![0.5, 0.5, 0.5, 0.5, 0.5],
            vec![1.2, 1.3, 1.1, 1.25, 1.15],
        ];
        let test = try_curve_sign_test(&scores, 1, 1, 1000, 0).unwrap();
        assert!(test.exact);
        assert_eq!(test.units, 5);
        assert_eq!(test.draws, 32);
        // Every fold points the same way, so only the all-positive pattern
        // reaches the observed statistic: 1 of 32.
        assert!((test.p_value - 1.0 / 32.0).abs() < 1e-12);
        assert!(test.statistic > 0.0);
    }

    #[test]
    fn identical_rows_give_no_evidence() {
        let scores = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let test = try_curve_sign_test(&scores, 0, 1, 1000, 0).unwrap();
        assert_eq!(test.statistic, 0.0);
        assert_eq!(test.p_value, 1.0);
    }

    #[test]
    fn sampled_patterns_are_reproducible_and_add_one_corrected() {
        let scores: Vec<Vec<f64>> = (0..4)
            .map(|j| {
                (0..20)
                    .map(|f| (j as f64) * 0.1 + (f as f64).sin() * 0.01)
                    .collect()
            })
            .collect();
        let first = try_curve_sign_test(&scores, 0, 1, 500, 7).unwrap();
        let again = try_curve_sign_test(&scores, 0, 1, 500, 7).unwrap();
        assert!(!first.exact);
        assert_eq!(first, again);
        assert!(first.p_value >= 1.0 / 501.0);
        assert!(first.p_value < 0.05);
    }

    #[test]
    fn partitions_are_the_flip_unit_for_stacked_folds() {
        // Two partitions of three folds: two flip units, so four patterns.
        let scores = vec![vec![1.0; 6], vec![0.0; 6]];
        let test = try_curve_sign_test(&scores, 1, 2, 100, 0).unwrap();
        assert_eq!(test.units, 2);
        assert_eq!(test.draws, 4);
    }

    #[test]
    fn invalid_curves_are_refused() {
        assert!(try_curve_sign_test(&[vec![1.0, 2.0]], 0, 1, 10, 0).is_err());
        assert!(try_curve_sign_test(&[vec![1.0, 2.0], vec![1.0]], 0, 1, 10, 0).is_err());
        assert!(try_curve_sign_test(&[vec![1.0, 2.0], vec![1.0, 2.0]], 2, 1, 10, 0).is_err());
        assert!(try_curve_sign_test(&[vec![1.0, f64::NAN], vec![1.0, 2.0]], 0, 1, 10, 0).is_err());
        assert!(try_curve_sign_test(&[vec![1.0, 2.0], vec![1.0, 2.0]], 0, 1, 0, 0).is_err());
    }
}

// ---------------------------------------------------------------- deselect

/// A learner's share of the distributional model's total risk reduction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnerShare {
    /// Index of the distribution parameter.
    pub parameter: usize,
    /// Index of the learner within that parameter.
    pub learner: usize,
    /// `|accumulated risk change| / sum over learners of |accumulated risk change|`.
    pub share: f64,
}

/// Shares of the total risk reduction per `(parameter, learner)` in a
/// non-cyclic distributional fit.
///
/// Row `i` of the path updated learner `learner[i]` of parameter
/// `parameter[i]` and changed the risk by `risk_change[i]`. Changes are
/// accumulated per pair, and each pair's share is the absolute accumulated
/// change over the sum of the absolute accumulated changes (taken as 1 when
/// that sum is 0, so every share is then 0). Pairs are returned sorted by
/// parameter, then learner.
pub fn lss_learner_shares(
    parameter: &[usize],
    learner: &[usize],
    risk_change: &[f64],
) -> Result<Vec<LearnerShare>, MboostrsError> {
    if parameter.len() != learner.len() || parameter.len() != risk_change.len() {
        return Err(MboostrsError::Shape(
            "parameter, learner and risk_change must have one entry per path row".into(),
        ));
    }
    let mut totals: std::collections::BTreeMap<(usize, usize), f64> = Default::default();
    for ((&p, &l), &r) in parameter.iter().zip(learner).zip(risk_change) {
        let r = if r.is_nan() { 0.0 } else { r };
        *totals.entry((p, l)).or_insert(0.0) += r;
    }
    let sum: f64 = totals.values().map(|v| v.abs()).sum();
    let total = if sum == 0.0 { 1.0 } else { sum };
    Ok(totals
        .into_iter()
        .map(|((parameter, learner), v)| LearnerShare {
            parameter,
            learner,
            share: v.abs() / total,
        })
        .collect())
}

/// Sums of consecutive groups of `values`: group `g` has `lengths[g]` values.
///
/// A formula term's share of the risk reduction is the sum over the learners
/// it produced.
pub fn group_sums(values: &[f64], lengths: &[usize]) -> Result<Vec<f64>, MboostrsError> {
    if lengths.iter().sum::<usize>() != values.len() {
        return Err(MboostrsError::Shape(
            "group lengths must add up to the number of values".into(),
        ));
    }
    let mut start = 0;
    Ok(lengths
        .iter()
        .map(|&len| {
            let mut total = 0.0;
            for v in &values[start..start + len] {
                total += v;
            }
            start += len;
            total
        })
        .collect())
}

/// Which terms deselection keeps: those with `share >= threshold`.
pub fn deselect_keep(shares: &[f64], threshold: f64) -> Result<Vec<bool>, MboostrsError> {
    if !(threshold > 0.0 && threshold < 1.0) {
        return Err(MboostrsError::Control(
            "threshold must lie in (0, 1)".into(),
        ));
    }
    Ok(shares.iter().map(|&s| s >= threshold).collect())
}

// ------------------------------------------------------------------- probe

/// Permutations of `0..n` for the shadow columns of probing, one per shadow.
///
/// A seeded ChaCha8 generator drives a Fisher-Yates shuffle; `count`
/// permutations are drawn in sequence from the one generator. For a given
/// seed the permutations are not those of NumPy's `Generator.permutation`.
pub fn probe_permutations(n: usize, count: usize, seed: u64) -> Vec<Vec<usize>> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                order.swap(i, j);
            }
            order
        })
        .collect()
}

/// Where the first shadow enters a boosting path.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeStop {
    /// 1-based iteration that first selected a shadow, if any did.
    pub stop: Option<usize>,
    /// Learner index of that shadow.
    pub first_shadow: Option<usize>,
    /// Distinct real learners selected before the first shadow, in order of
    /// first selection.
    pub selected: Vec<usize>,
}

/// Read the stopping iteration off the sequence of selected learners.
///
/// `selected[t]` is the learner chosen at iteration `t + 1` and
/// `is_shadow[j]` flags learner `j` as a shadow column.
pub fn probe_stop(selected: &[usize], is_shadow: &[bool]) -> Result<ProbeStop, MboostrsError> {
    let mut out = ProbeStop {
        stop: None,
        first_shadow: None,
        selected: Vec::new(),
    };
    for (t, &j) in selected.iter().enumerate() {
        let shadow = *is_shadow
            .get(j)
            .ok_or_else(|| MboostrsError::Shape(format!("learner index {j} out of range")))?;
        if shadow {
            out.stop = Some(t + 1);
            out.first_shadow = Some(j);
            break;
        }
        if !out.selected.contains(&j) {
            out.selected.push(j);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod deselect_probe_tests {
    use super::*;

    #[test]
    fn shares_accumulate_per_pair_and_sum_to_one() {
        let s = lss_learner_shares(&[0, 1, 0, 1], &[2, 0, 2, 0], &[-1.0, -2.0, -1.0, f64::NAN])
            .unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].parameter, s[0].learner), (0, 2));
        assert!((s[0].share - 0.5).abs() < 1e-15);
        assert!((s[1].share - 0.5).abs() < 1e-15);
    }

    #[test]
    fn zero_total_gives_zero_shares() {
        let s = lss_learner_shares(&[0], &[0], &[0.0]).unwrap();
        assert_eq!(s[0].share, 0.0);
    }

    #[test]
    fn group_sums_and_threshold() {
        let sums = group_sums(&[0.1, 0.2, 0.005, 0.0], &[2, 1, 0, 1]).unwrap();
        assert!((sums[0] - 0.3).abs() < 1e-15);
        assert_eq!(sums[2], 0.0);
        assert_eq!(
            deselect_keep(&[0.3, 0.005, 0.01], 0.01).unwrap(),
            vec![true, false, true]
        );
        assert!(deselect_keep(&[0.3], 1.0).is_err());
        assert!(group_sums(&[1.0], &[2]).is_err());
    }

    #[test]
    fn permutations_are_permutations_and_depend_on_seed() {
        let a = probe_permutations(50, 3, 7);
        assert_eq!(a, probe_permutations(50, 3, 7));
        assert_ne!(a, probe_permutations(50, 3, 8));
        for p in &a {
            let mut s = p.clone();
            s.sort();
            assert_eq!(s, (0..50).collect::<Vec<_>>());
        }
        assert_ne!(a[0], a[1]);
    }

    #[test]
    fn stop_is_the_first_shadow_in_the_sequence() {
        let shadow = [false, false, true, true];
        let r = probe_stop(&[1, 0, 1, 3, 2, 0], &shadow).unwrap();
        assert_eq!(r.stop, Some(4));
        assert_eq!(r.first_shadow, Some(3));
        assert_eq!(r.selected, vec![1, 0]);
        let r = probe_stop(&[0, 1, 0], &shadow).unwrap();
        assert_eq!(r.stop, None);
        assert_eq!(r.selected, vec![0, 1]);
        assert!(probe_stop(&[9], &shadow).is_err());
    }
}
