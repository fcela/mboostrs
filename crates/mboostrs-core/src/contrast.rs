//! Dependence-aware uncertainty for a weighted average of per-row contrasts.
//!
//! A standardized contrast or average marginal effect is a weighted mean of
//! row-level values `v_i` (the row's contrast or derivative). When the rows
//! fall in clusters (repeated measures of one subject, pupils in one school),
//! the rows of a cluster share unmeasured influences, so the naive standard
//! error of the mean, which treats every row as independent, is too small.
//! This module treats the clusters as the sampling units and reports two
//! measures of the uncertainty of the weighted mean:
//!
//! * a cluster-robust (sandwich) standard error, `sqrt(G/(G-1) * sum_g s_g^2)`
//!   over `W^2`, with `s_g` the weighted deviations summed within cluster `g`;
//! * a cluster bootstrap: clusters are drawn with replacement, the weighted
//!   mean is recomputed from the drawn clusters, and the percentile interval
//!   of those means is reported.
//!
//! Both describe the averaging population as a sample of clusters and the
//! fitted model as fixed. They do not include the variability of the model
//! fit itself.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::MboostrsError;

/// The weighted mean of the per-row values and its cluster-level uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterAverage {
    /// Weighted mean of the row values, `sum w v / sum w`.
    pub estimate: f64,
    /// Cluster-robust standard error of the mean (sandwich form, with the
    /// `G/(G-1)` small-sample factor).
    pub se_sandwich: f64,
    /// Standard deviation of the cluster-bootstrap means.
    pub se_bootstrap: f64,
    /// Lower end of the percentile interval of the bootstrap means.
    pub lower: f64,
    /// Upper end of the percentile interval of the bootstrap means.
    pub upper: f64,
    /// Number of distinct clusters.
    pub clusters: usize,
    /// Number of bootstrap replicates behind `se_bootstrap` and the interval.
    pub draws: usize,
}

/// Weighted mean of `values` with cluster-level uncertainty.
///
/// `clusters[i]` is the cluster of row `i`, given as a code in `0..G`; every
/// code in `0..G` must be used by at least one row. Rows with zero weight
/// contribute nothing to the mean and to the cluster sums.
///
/// `level` is the coverage of the percentile interval (for example 0.95).
/// `draws` is the number of bootstrap replicates, and `seed` makes the
/// replicates reproducible.
pub fn cluster_average(
    values: &[f64],
    weights: &[f64],
    clusters: &[usize],
    draws: usize,
    seed: u64,
    level: f64,
) -> Result<ClusterAverage, MboostrsError> {
    let n = values.len();
    if weights.len() != n || clusters.len() != n {
        return Err(MboostrsError::Shape(format!(
            "values ({n}), weights ({}) and clusters ({}) must have one entry per row",
            weights.len(),
            clusters.len()
        )));
    }
    if n == 0 {
        return Err(MboostrsError::Control(
            "cluster_average needs at least one row".into(),
        ));
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(MboostrsError::Control(format!(
            "level must be in (0, 1), got {level}"
        )));
    }
    if draws < 2 {
        return Err(MboostrsError::Control("draws must be at least 2".into()));
    }
    if values.iter().any(|v| !v.is_finite()) || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(MboostrsError::Control(
            "values must be finite and weights finite and non-negative".into(),
        ));
    }
    let groups = clusters.iter().copied().max().map_or(0, |m| m + 1);
    let mut weight_by_cluster = vec![0.0f64; groups];
    let mut total_by_cluster = vec![0.0f64; groups];
    let mut used = vec![false; groups];
    for ((&v, &w), &g) in values.iter().zip(weights).zip(clusters) {
        weight_by_cluster[g] += w;
        total_by_cluster[g] += w * v;
        used[g] = true;
    }
    if used.iter().any(|u| !u) {
        return Err(MboostrsError::Shape(
            "cluster codes must be 0..G-1 with every code used by at least one row".into(),
        ));
    }
    let total_weight: f64 = weight_by_cluster.iter().sum();
    if total_weight <= 0.0 {
        return Err(MboostrsError::Control("the weights sum to zero".into()));
    }
    let estimate = total_by_cluster.iter().sum::<f64>() / total_weight;
    if groups < 2 {
        return Err(MboostrsError::Control(
            "cluster uncertainty needs at least two clusters; with one cluster the \
             average has no cluster-level variability to estimate"
                .into(),
        ));
    }

    // Sandwich: the score of cluster g is the sum over its rows of w (v - mean).
    let mut score_sq = 0.0;
    let mut scores = vec![0.0f64; groups];
    for ((&v, &w), &g) in values.iter().zip(weights).zip(clusters) {
        scores[g] += w * (v - estimate);
    }
    for s in &scores {
        score_sq += s * s;
    }
    let g_f = groups as f64;
    let se_sandwich = (g_f / (g_f - 1.0) * score_sq).sqrt() / total_weight;

    // Bootstrap: resample G clusters with replacement and recompute the
    // ratio sum(w v) / sum(w) from the cluster totals of the draw.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for _ in 0..groups {
            let g = ((rng.random::<f64>() * g_f) as usize).min(groups - 1);
            numerator += total_by_cluster[g];
            denominator += weight_by_cluster[g];
        }
        // A draw whose clusters all carry zero weight has no mean; such
        // draws are dropped, and their count is part of the result.
        if denominator > 0.0 {
            means.push(numerator / denominator);
        }
    }
    if means.len() < 2 {
        return Err(MboostrsError::Control(
            "too few bootstrap replicates had positive weight; the weights are concentrated \
             on one cluster"
                .into(),
        ));
    }
    let count = means.len() as f64;
    let center = means.iter().sum::<f64>() / count;
    let se_bootstrap =
        (means.iter().map(|m| (m - center).powi(2)).sum::<f64>() / (count - 1.0)).sqrt();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tail = (1.0 - level) / 2.0;
    Ok(ClusterAverage {
        estimate,
        se_sandwich,
        se_bootstrap,
        lower: quantile_sorted(&means, tail),
        upper: quantile_sorted(&means, 1.0 - tail),
        clusters: groups,
        draws: means.len(),
    })
}

/// Linear-interpolation quantile of sorted values (type 7, as R's default).
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let position = p * (sorted.len() as f64 - 1.0);
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let fraction = position - low as f64;
    sorted[low] + fraction * (sorted[high] - sorted[low])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandwich_matches_a_hand_computation() {
        // Cluster 0 holds rows valued 1 and 1 (weights 1, 1); cluster 1 holds
        // one row valued 4 (weight 2); cluster 2 holds one row valued 7
        // (weight 1). Weighted mean = (1 + 1 + 8 + 7) / 5 = 3.4.
        // Cluster scores: s0 = 1*(1-3.4) + 1*(1-3.4) = -4.8,
        // s1 = 2*(4-3.4) = 1.2, s2 = 1*(7-3.4) = 3.6.
        // Sum of squares = 23.04 + 1.44 + 12.96 = 37.44; G/(G-1) = 1.5;
        // se = sqrt(1.5 * 37.44) / 5 = sqrt(56.16) / 5.
        let out = cluster_average(
            &[1.0, 1.0, 4.0, 7.0],
            &[1.0, 1.0, 2.0, 1.0],
            &[0, 0, 1, 2],
            200,
            1,
            0.9,
        )
        .unwrap();
        assert!((out.estimate - 3.4).abs() < 1e-12);
        assert!((out.se_sandwich - (56.16f64).sqrt() / 5.0).abs() < 1e-12);
        assert_eq!(out.clusters, 3);
        assert_eq!(out.draws, 200);
    }

    #[test]
    fn identical_values_within_clusters_give_a_bootstrap_spread_from_cluster_means() {
        // Each row equals its cluster's constant; the sandwich and the
        // bootstrap both reflect between-cluster variation only.
        let values = [0.0, 0.0, 10.0, 10.0, 10.0, 20.0];
        let weights = [1.0; 6];
        let clusters = [0, 0, 1, 1, 1, 2];
        let out = cluster_average(&values, &weights, &clusters, 500, 3, 0.95).unwrap();
        assert!((out.estimate - 50.0 / 6.0).abs() < 1e-12);
        assert!(out.se_sandwich > 0.0);
        assert!(out.lower < out.estimate && out.estimate < out.upper);
        let again = cluster_average(&values, &weights, &clusters, 500, 3, 0.95).unwrap();
        assert_eq!(out, again, "the same seed must reproduce the bootstrap");
    }

    #[test]
    fn a_single_cluster_is_refused_by_name() {
        let err = cluster_average(&[1.0, 2.0], &[1.0, 1.0], &[0, 0], 100, 0, 0.9).unwrap_err();
        assert!(err.to_string().contains("at least two clusters"), "{err}");
    }

    #[test]
    fn unused_cluster_codes_are_refused() {
        let err = cluster_average(&[1.0, 2.0], &[1.0, 1.0], &[0, 2], 100, 0, 0.9).unwrap_err();
        assert!(err.to_string().contains("every code"), "{err}");
    }
}
