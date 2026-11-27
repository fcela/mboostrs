#![allow(unused_imports)]
use super::*;

/// How a tree node chooses the column to split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SplitSelection {
    /// The split, over every column, that most reduces the weighted squared
    /// error (CART). Favours columns with many candidate split points.
    #[default]
    Sse,
    /// partykit's `ctree` as mboost's `btree` runs it (`teststat = "quad"`,
    /// `testtype = "Teststatistic"`): the column with the largest quadratic
    /// statistic of association with the gradient, `n r^2` for a number and
    /// the between-level sum of squares over the variance for a factor.
    CtreeStatistic,
    /// The same statistic referred to its chi-squared distribution (1 degree
    /// of freedom for a number, `L - 1` for a factor with `L` levels at the
    /// node): the smallest p-value, which removes the preference for factors
    /// with many levels.
    CtreePvalue,
}

/// How a tree finds splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SplitMethod {
    /// Exhaustive search over sorted unique values (exact).
    #[default]
    Exact,
    /// Histogram search over binned values.
    Histogram,
}

/// The fixed inputs of one tree fit.
pub(super) struct TreeSpec<'a> {
    pub(super) design: ArrayView2<'a, f64>,
    pub(super) u: ArrayView1<'a, f64>,
    pub(super) weights: ArrayView1<'a, f64>,
    /// Level count of each categorical column (codes 0..L-1); 0, or an empty
    /// slice, for a numeric column.
    pub(super) categorical: &'a [usize],
    pub(super) selection: SplitSelection,
    pub(super) split_method: SplitMethod,
    pub(super) minbucket: usize,
    pub(super) minsplit: usize,
    pub(super) maxdepth: usize,
    /// Precomputed quantile binning for numeric columns when using histogram method.
    pub(super) binning: &'a [Option<QuantileBinning>],
    /// L2 leaf shrinkage penalty (`0.0` = unregularized).
    pub(super) lambda_leaf: f64,
    /// Optional per-step column activity mask (`None` = all columns active).
    pub(super) active_cols: Option<&'a [bool]>,
}

/// The best split found at a node.
pub(super) enum BestSplit {
    /// `x <= point`, with the last left position in the column's order at
    /// the node.
    Numeric {
        var: usize,
        point: f64,
        last_left: usize,
        /// The highest bin sent left, in histogram mode; `None` on the exact
        /// path, where the split is reconstructed from `work.sorted`.
        bin: Option<usize>,
    },
    /// Level codes sent left.
    Levels { var: usize, left_levels: Vec<usize> },
}

/// The column ctree splits at this node: the largest quadratic association
/// statistic between the column and the gradient `u` (or the smallest
/// chi-squared p-value of it), `None` when no column varies.
pub(super) fn ctree_column(
    spec: &TreeSpec<'_>,
    indices: &[usize],
    rule: SplitSelection,
    total_w: f64,
    total_wsum: f64,
    total_sqsum: f64,
) -> Option<usize> {
    let (design, u, weights) = (spec.design, spec.u, spec.weights);
    let u_mean = total_wsum / total_w;
    let v_u = (total_sqsum / total_w - u_mean * u_mean).max(0.0);
    if v_u <= 0.0 || total_w <= 1.0 {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for j in 0..design.ncols() {
        if spec.active_cols.is_some_and(|active| !active[j]) {
            continue;
        }
        let col = design.column(j);
        let n_levels = spec.categorical.get(j).copied().unwrap_or(0);
        let (statistic, df) = if n_levels > 0 {
            let mut w_l = vec![0.0f64; n_levels];
            let mut s_l = vec![0.0f64; n_levels];
            for &ii in indices {
                let code = col[ii];
                if code >= 0.0 && code < n_levels as f64 {
                    w_l[code as usize] += weights[ii];
                    s_l[code as usize] += weights[ii] * u[ii];
                }
            }
            let present = w_l.iter().filter(|&&w| w > 0.0).count();
            if present < 2 {
                continue;
            }
            // Between-level sum of squares over the permutation variance.
            let between: f64 = (0..n_levels)
                .filter(|&l| w_l[l] > 0.0)
                .map(|l| (s_l[l] - w_l[l] * u_mean).powi(2) / w_l[l])
                .sum();
            (
                between / v_u * (total_w - 1.0) / total_w,
                (present - 1) as f64,
            )
        } else {
            let (mut sw_x, mut sw_xx, mut t) = (0.0, 0.0, 0.0);
            for &ii in indices {
                let (wi, xi) = (weights[ii], col[ii]);
                sw_x += wi * xi;
                sw_xx += wi * xi * xi;
                t += wi * xi * u[ii];
            }
            let var_x = sw_xx - sw_x * sw_x / total_w;
            if var_x <= 1e-12 * sw_xx.abs().max(1.0) {
                continue;
            }
            // Strasser-Weber: E(T) = sum(w x) mean(u), Var(T) =
            // W/(W-1) v_u (sum(w x^2) - sum(w x)^2 / W).
            let mu = sw_x * u_mean;
            let var_t = total_w / (total_w - 1.0) * v_u * var_x;
            ((t - mu).powi(2) / var_t, 1.0)
        };
        let score = match rule {
            SplitSelection::CtreePvalue => -crate::causal::chi_square_sf(statistic, df),
            _ => statistic,
        };
        if best.is_none_or(|(_, b)| score > b) {
            best = Some((j, score));
        }
    }
    best.map(|(j, _)| j)
}

/// Buffers one tree fit works in, allocated once per tree.
///
/// A node is a range `lo..hi`: `rows[lo..hi]` are its rows, and for each
/// numeric column `j`, `sorted[j * n + lo..j * n + hi]` are the same rows in
/// the column's order. A split partitions every range in place (stably, so
/// each column stays sorted) and the children take the two halves.
pub(super) struct TreeWork {
    pub(super) n: usize,
    pub(super) rows: Vec<usize>,
    pub(super) sorted: Vec<usize>,
    pub(super) scratch: Vec<usize>,
    pub(super) goes_left: Vec<bool>,
    /// Precomputed `weights[i] * u[i]` for fast histogram accumulation.
    pub(super) wu: Vec<f64>,
    /// Histogram accumulation buffers, one entry per bin of the current
    /// numeric column: `(Σw, Σw·u, count)`. Reused across nodes and columns.
    pub(super) hist: Vec<(f64, f64, usize)>,
    /// Per-level `(Σw, Σw·u, Σw·u²)` buffers for the categorical path.
    pub(super) level_stats: Vec<(f64, f64, f64)>,
    /// Level or bin indices with positive count at the current node.
    pub(super) level_present: Vec<usize>,
}

/// Fill per-column histograms `(Σw, Σw·u, count)` for a slice of row indices.
pub(super) fn build_node_hists(
    spec: &TreeSpec<'_>,
    rows: &[usize],
    wu: &[f64],
    col_offsets: &[usize],
    out: &mut [(f64, f64, usize)],
) {
    let weights = spec.weights;
    let p = spec.design.ncols();
    for j in 0..p {
        if spec.active_cols.is_some_and(|active| !active[j]) {
            continue;
        }
        let start = col_offsets[j];
        let end = col_offsets[j + 1];
        let h = &mut out[start..end];
        h.fill((0.0, 0.0, 0));
        let n_levels = spec.categorical.get(j).copied().unwrap_or(0);
        if n_levels > 0 {
            let col = spec.design.column(j);
            for &ii in rows {
                let code = col[ii];
                if code >= 0.0 && code < n_levels as f64 {
                    let entry = &mut h[code as usize];
                    entry.0 += weights[ii];
                    entry.1 += wu[ii];
                    entry.2 += 1;
                }
            }
        } else if let Some(binning) = spec.binning[j].as_ref() {
            let index = &binning.index;
            for &ii in rows {
                let entry = &mut h[index[ii] as usize];
                entry.0 += weights[ii];
                entry.1 += wu[ii];
                entry.2 += 1;
            }
        }
    }
}

pub(super) enum BestHistSplit {
    Numeric {
        var: usize,
        point: f64,
        last_left: usize,
        bin: usize,
    },
    Levels {
        var: usize,
        left_levels: Vec<usize>,
    },
}

/// Fast histogram recursive builder using the parent-minus-small-child
/// histogram subtraction trick for both binned numeric and categorical columns:
/// only the smaller child scans rows, and the larger child derives its
/// histograms in O(total_slots) by subtraction.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_tree_hist_subtract(
    spec: &TreeSpec<'_>,
    work: &mut TreeWork,
    lo: usize,
    hi: usize,
    depth: usize,
    col_offsets: &[usize],
    total_w: f64,
    total_wsum: f64,
    mut hists: Vec<(f64, f64, usize)>,
    nodes: &mut Vec<TreeNode>,
) -> usize {
    let n = hi - lo;
    let p = spec.design.ncols();

    let make_leaf = |nodes: &mut Vec<TreeNode>| {
        let mean = if total_w > 0.0 {
            if spec.lambda_leaf > 0.0 {
                total_wsum / (total_w + spec.lambda_leaf)
            } else {
                total_wsum / total_w
            }
        } else {
            0.0
        };
        nodes.push(TreeNode::Leaf { mean });
        nodes.len() - 1
    };

    if total_w < spec.minsplit as f64 || n < spec.minsplit || depth >= spec.maxdepth {
        return make_leaf(nodes);
    }

    let minbucket = spec.minbucket as f64;
    let lambda_leaf = spec.lambda_leaf;
    let parent_score = if lambda_leaf > 0.0 {
        total_wsum * total_wsum / (total_w + lambda_leaf)
    } else {
        0.0
    };
    let mut best_gain = if lambda_leaf > 0.0 {
        0.0
    } else {
        f64::NEG_INFINITY
    };
    let mut best: Option<BestHistSplit> = None;

    for j in 0..p {
        if spec.active_cols.is_some_and(|active| !active[j]) {
            continue;
        }
        let start = col_offsets[j];
        let end = col_offsets[j + 1];
        let h = &hists[start..end];
        let n_levels = spec.categorical.get(j).copied().unwrap_or(0);
        let present = &mut work.level_present;
        present.clear();
        if n_levels > 0 {
            for (l, &(w, _, cnt)) in h.iter().enumerate() {
                if cnt > 0 && w > 0.0 {
                    present.push(l);
                }
            }
            if present.len() < 2 {
                continue;
            }
            present.sort_by(|&a, &b| {
                (h[a].1 / h[a].0)
                    .partial_cmp(&(h[b].1 / h[b].0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let (mut lw, mut lws) = (0.0f64, 0.0f64);
            let mut col_best_k: Option<usize> = None;
            let mut col_best_gain = best_gain;
            for k in 0..present.len() - 1 {
                let (bw, bws, _) = h[present[k]];
                lw += bw;
                lws += bws;
                let rw = total_w - lw;
                if lw < minbucket || rw < minbucket {
                    continue;
                }
                let rws = total_wsum - lws;
                let gain = if lambda_leaf > 0.0 {
                    lws * lws / (lw + lambda_leaf) + rws * rws / (rw + lambda_leaf) - parent_score
                } else {
                    lws * lws / lw + rws * rws / rw
                };
                if gain > col_best_gain {
                    col_best_gain = gain;
                    col_best_k = Some(k);
                }
            }
            if let Some(k) = col_best_k {
                best_gain = col_best_gain;
                let mut left_levels = present[..=k].to_vec();
                left_levels.sort_unstable();
                best = Some(BestHistSplit::Levels {
                    var: j,
                    left_levels,
                });
            }
        } else {
            let binning = spec.binning[j].as_ref().unwrap();
            for (b, &(_, _, cnt)) in h.iter().enumerate() {
                if cnt > 0 {
                    present.push(b);
                }
            }
            let (mut lw, mut lws) = (0.0f64, 0.0f64);
            let mut lcount = 0usize;
            for idx in 0..present.len().saturating_sub(1) {
                let b = present[idx];
                let (bw, bws, bcnt) = h[b];
                lw += bw;
                lws += bws;
                lcount += bcnt;
                let rw = total_w - lw;
                if lw < minbucket || rw < minbucket {
                    continue;
                }
                let rws = total_wsum - lws;
                let gain = if lambda_leaf > 0.0 {
                    lws * lws / (lw + lambda_leaf) + rws * rws / (rw + lambda_leaf) - parent_score
                } else {
                    lws * lws / lw + rws * rws / rw
                };
                if gain > best_gain {
                    best_gain = gain;
                    let next = present[idx + 1];
                    let point = (binning.values[b] + binning.values[next]) * 0.5;
                    best = Some(BestHistSplit::Numeric {
                        var: j,
                        point,
                        last_left: lcount - 1,
                        bin: b,
                    });
                }
            }
        }
    }

    let Some(chosen) = best else {
        return make_leaf(nodes);
    };

    let (split_idx, n_left, left_w, left_wu) = match chosen {
        BestHistSplit::Numeric {
            var,
            point,
            last_left,
            bin: best_bin,
        } => {
            let idx = nodes.len();
            nodes.push(TreeNode::Split {
                var,
                point,
                left: 0,
                right: 0,
            });
            let index = &spec.binning[var].as_ref().unwrap().index;
            let TreeWork {
                rows,
                scratch,
                goes_left,
                wu,
                ..
            } = work;
            let (mut lw, mut lwu) = (0.0f64, 0.0f64);
            for &ii in &rows[lo..hi] {
                let left = (index[ii] as usize) <= best_bin;
                goes_left[ii] = left;
                if left {
                    lw += spec.weights[ii];
                    lwu += wu[ii];
                }
            }
            TreeWork::partition(&mut rows[lo..hi], scratch, goes_left);
            (idx, last_left + 1, lw, lwu)
        }
        BestHistSplit::Levels {
            var,
            mut left_levels,
        } => {
            let col = spec.design.column(var);
            let in_set = |levels: &[usize], ii: usize| {
                let v = col[ii];
                v >= 0.0 && levels.binary_search(&(v as usize)).is_ok()
            };
            let listed_weight: f64 = work.rows[lo..hi]
                .iter()
                .filter(|&&ii| in_set(&left_levels, ii))
                .map(|&ii| spec.weights[ii])
                .sum();
            if listed_weight > total_w - listed_weight {
                let mut other: Vec<usize> = work.rows[lo..hi]
                    .iter()
                    .filter(|&&ii| !in_set(&left_levels, ii))
                    .filter_map(|&ii| {
                        let v = col[ii];
                        (v >= 0.0).then_some(v as usize)
                    })
                    .collect();
                other.sort_unstable();
                other.dedup();
                left_levels = other;
            }
            let TreeWork {
                rows,
                scratch,
                goes_left,
                wu,
                ..
            } = work;
            let mut n_l = 0usize;
            let (mut lw, mut lwu) = (0.0f64, 0.0f64);
            for &ii in &rows[lo..hi] {
                let left = in_set(&left_levels, ii);
                goes_left[ii] = left;
                if left {
                    n_l += 1;
                    lw += spec.weights[ii];
                    lwu += wu[ii];
                }
            }
            TreeWork::partition(&mut rows[lo..hi], scratch, goes_left);
            let idx = nodes.len();
            nodes.push(TreeNode::SplitSet {
                var,
                left_levels,
                left: 0,
                right: 0,
            });
            (idx, n_l, lw, lwu)
        }
    };

    let n_right = n - n_left;
    let right_w = (total_w - left_w).max(0.0);
    let right_wu = total_wsum - left_wu;
    let mid = lo + n_left;

    if depth + 1 >= spec.maxdepth {
        let left_child = make_leaf_from_rows(spec, &work.rows[lo..mid], &work.wu, nodes);
        let right_child = make_leaf_from_rows(spec, &work.rows[mid..hi], &work.wu, nodes);
        match &mut nodes[split_idx] {
            TreeNode::Split { left, right, .. } | TreeNode::SplitSet { left, right, .. } => {
                *left = left_child;
                *right = right_child;
            }
            TreeNode::Leaf { .. } => {}
        }
        return split_idx;
    }

    let total_slots = *col_offsets.last().unwrap_or(&0);
    let mut small_hists = vec![(0.0f64, 0.0f64, 0usize); total_slots];
    let left_is_smaller = n_left <= n_right;
    if left_is_smaller {
        build_node_hists(
            spec,
            &work.rows[lo..mid],
            &work.wu,
            col_offsets,
            &mut small_hists,
        );
    } else {
        build_node_hists(
            spec,
            &work.rows[mid..hi],
            &work.wu,
            col_offsets,
            &mut small_hists,
        );
    }
    for (parent_e, &(sw, swu, sc)) in hists.iter_mut().zip(small_hists.iter()) {
        parent_e.0 = (parent_e.0 - sw).max(0.0);
        parent_e.1 -= swu;
        parent_e.2 = parent_e.2.saturating_sub(sc);
    }

    let (left_hists, right_hists) = if left_is_smaller {
        (small_hists, hists)
    } else {
        (hists, small_hists)
    };
    let left_child = build_tree_hist_subtract(
        spec,
        work,
        lo,
        mid,
        depth + 1,
        col_offsets,
        left_w,
        left_wu,
        left_hists,
        nodes,
    );
    let right_child = build_tree_hist_subtract(
        spec,
        work,
        mid,
        hi,
        depth + 1,
        col_offsets,
        right_w,
        right_wu,
        right_hists,
        nodes,
    );
    match &mut nodes[split_idx] {
        TreeNode::Split { left, right, .. } | TreeNode::SplitSet { left, right, .. } => {
            *left = left_child;
            *right = right_child;
        }
        TreeNode::Leaf { .. } => {}
    }
    split_idx
}

pub(super) fn make_leaf_from_rows(
    spec: &TreeSpec<'_>,
    rows: &[usize],
    wu: &[f64],
    nodes: &mut Vec<TreeNode>,
) -> usize {
    let mut tw = 0.0f64;
    let mut twu = 0.0f64;
    for &ii in rows {
        tw += spec.weights[ii];
        twu += wu[ii];
    }
    let mean = if tw > 0.0 {
        if spec.lambda_leaf > 0.0 {
            twu / (tw + spec.lambda_leaf)
        } else {
            twu / tw
        }
    } else {
        0.0
    };
    nodes.push(TreeNode::Leaf { mean });
    nodes.len() - 1
}

impl TreeWork {
    /// Stably move the rows marked `goes_left` to the front of `range`.
    pub(super) fn partition(buf: &mut [usize], scratch: &mut [usize], goes_left: &[bool]) {
        let mut front = 0;
        let mut back = 0;
        for k in 0..buf.len() {
            let ii = buf[k];
            if goes_left[ii] {
                buf[front] = ii;
                front += 1;
            } else {
                scratch[back] = ii;
                back += 1;
            }
        }
        buf[front..].copy_from_slice(&scratch[..back]);
    }
}

/// Recursively build a CART regression tree over the node `lo..hi` of
/// `work`.
///
/// A numeric column is scanned in its presorted order. A categorical column
/// (codes 0..L-1) has its levels ordered by the mean of `u` at this node and
/// scanned in that order: for squared error, the best split of the levels
/// into two groups is one of these L-1 prefixes (Breiman et al. 1984,
/// section 9.4), so a level set is found without trying all 2^(L-1)
/// partitions.
///
/// Returns the index of this node in `nodes`.
pub(super) fn build_tree_recursive(
    spec: &TreeSpec<'_>,
    work: &mut TreeWork,
    lo: usize,
    hi: usize,
    depth: usize,
    nodes: &mut Vec<TreeNode>,
) -> usize {
    let (design, u, weights) = (spec.design, spec.u, spec.weights);
    let n = hi - lo;
    let p = design.ncols();
    let stride = work.n;

    let mut total_w = 0.0f64;
    let mut total_wsum = 0.0f64;
    let mut total_sqsum = 0.0f64;
    for &ii in &work.rows[lo..hi] {
        let wi = weights[ii];
        let ui = u[ii];
        total_w += wi;
        total_wsum += wi * ui;
        total_sqsum += wi * ui * ui;
    }
    let leaf = |nodes: &mut Vec<TreeNode>| {
        let mean = if total_w > 0.0 {
            if spec.lambda_leaf > 0.0 {
                total_wsum / (total_w + spec.lambda_leaf)
            } else {
                total_wsum / total_w
            }
        } else {
            0.0
        };
        nodes.push(TreeNode::Leaf { mean });
        nodes.len() - 1
    };

    if total_w < spec.minsplit as f64 || n < spec.minsplit || depth >= spec.maxdepth {
        return leaf(nodes);
    }

    let minbucket = spec.minbucket as f64;
    let lambda_leaf = spec.lambda_leaf;
    let split_sse = |lw: f64, lws: f64, lsq: f64| {
        let rw = total_w - lw;
        let rws = total_wsum - lws;
        if lambda_leaf > 0.0 {
            (total_sqsum - lws * lws / (lw + lambda_leaf) - rws * rws / (rw + lambda_leaf)).max(0.0)
        } else {
            let lsse = lsq - lws * lws / lw;
            let rsse = (total_sqsum - lsq) - rws * rws / rw;
            (lsse + rsse).max(0.0)
        }
    };

    let mut best_sse = if lambda_leaf > 0.0 {
        (total_sqsum - total_wsum * total_wsum / (total_w + lambda_leaf)).max(0.0)
    } else {
        f64::INFINITY
    };
    let mut best: Option<BestSplit> = None;
    // ctree chooses the column first, by its association with the gradient;
    // only that column's split points are then searched.
    let only = match spec.selection {
        SplitSelection::Sse => None,
        rule => ctree_column(
            spec,
            &work.rows[lo..hi],
            rule,
            total_w,
            total_wsum,
            total_sqsum,
        ),
    };
    if spec.selection != SplitSelection::Sse && only.is_none() {
        return leaf(nodes);
    }

    for j in 0..p {
        if spec.active_cols.is_some_and(|active| !active[j]) {
            continue;
        }
        if only.is_some_and(|chosen| chosen != j) {
            continue;
        }
        let col = design.column(j);
        let n_levels = spec.categorical.get(j).copied().unwrap_or(0);
        if n_levels > 0 {
            // Per-level weight, weighted sum and weighted sum of squares,
            // accumulated in reusable per-node buffers (bins = levels).
            let TreeWork {
                rows,
                level_stats: stats,
                level_present: present,
                ..
            } = &mut *work;
            for entry in stats[..n_levels].iter_mut() {
                *entry = (0.0, 0.0, 0.0);
            }
            for &ii in &rows[lo..hi] {
                let code = col[ii];
                if code < 0.0 || code >= n_levels as f64 {
                    continue;
                }
                let (w, ws, sq) = &mut stats[code as usize];
                let (wi, ui) = (weights[ii], u[ii]);
                *w += wi;
                *ws += wi * ui;
                *sq += wi * ui * ui;
            }
            present.clear();
            for (l, entry) in stats[..n_levels].iter().enumerate() {
                if entry.0 > 0.0 {
                    present.push(l);
                }
            }
            if present.len() < 2 {
                continue;
            }
            present.sort_by(|&a, &b| {
                (stats[a].1 / stats[a].0)
                    .partial_cmp(&(stats[b].1 / stats[b].0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let (mut lw, mut lws, mut lsq) = (0.0, 0.0, 0.0);
            let mut col_best_k: Option<usize> = None;
            let mut col_best_sse = best_sse;
            for k in 0..present.len() - 1 {
                let (w, ws, sq) = stats[present[k]];
                lw += w;
                lws += ws;
                lsq += sq;
                if lw < minbucket || total_w - lw < minbucket {
                    continue;
                }
                let sse = split_sse(lw, lws, lsq);
                if sse < col_best_sse {
                    col_best_sse = sse;
                    col_best_k = Some(k);
                }
            }
            if let Some(k) = col_best_k {
                best_sse = col_best_sse;
                // Build the winning level set once per column, not once per
                // improving candidate.
                let mut left_levels = present[..=k].to_vec();
                left_levels.sort_unstable();
                best = Some(BestSplit::Levels {
                    var: j,
                    left_levels,
                });
            }
            continue;
        }

        // This node's rows in the column's order.
        if spec.split_method == SplitMethod::Histogram {
            if let Some(binning) = spec.binning.get(j).and_then(|b| b.as_ref()) {
                let effective = binning.effective;
                let TreeWork {
                    rows,
                    wu,
                    hist,
                    level_present: present,
                    ..
                } = &mut *work;
                hist[..effective].fill((0.0, 0.0, 0));
                for &ii in &rows[lo..hi] {
                    let entry = &mut hist[binning.index[ii] as usize];
                    entry.0 += weights[ii];
                    entry.1 += wu[ii];
                    entry.2 += 1;
                }
                let present = &mut *present;
                present.clear();
                for (b, &(_, _, count)) in hist[..effective].iter().enumerate() {
                    if count > 0 {
                        present.push(b);
                    }
                }
                let (mut lw, mut lws) = (0.0f64, 0.0f64);
                let mut lcount = 0usize;
                for idx in 0..present.len().saturating_sub(1) {
                    let b = present[idx];
                    lw += hist[b].0;
                    lws += hist[b].1;
                    lcount += hist[b].2;
                    let rw = total_w - lw;
                    if lw < minbucket || rw < minbucket {
                        continue;
                    }
                    let rws = total_wsum - lws;
                    let sse = if lambda_leaf > 0.0 {
                        (total_sqsum
                            - lws * lws / (lw + lambda_leaf)
                            - rws * rws / (rw + lambda_leaf))
                            .max(0.0)
                    } else {
                        (total_sqsum - lws * lws / lw - rws * rws / rw).max(0.0)
                    };
                    if sse < best_sse {
                        best_sse = sse;
                        let next = present[idx + 1];
                        best = Some(BestSplit::Numeric {
                            var: j,
                            point: (binning.values[b] + binning.values[next]) / 2.0,
                            last_left: lcount - 1,
                            bin: Some(b),
                        });
                    }
                }
                continue;
            }
        }

        let idx = &work.sorted[j * stride + lo..j * stride + hi];
        let (mut lw, mut lws, mut lsq) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..n - 1 {
            let ii = idx[k];
            let ii_next = idx[k + 1];
            let wi = weights[ii];
            let ui = u[ii];
            lws += wi * ui;
            lw += wi;
            lsq += wi * ui * ui;
            if col[ii] == col[ii_next] {
                continue;
            }
            if lw < minbucket || total_w - lw < minbucket {
                continue;
            }
            let sse = split_sse(lw, lws, lsq);
            if sse < best_sse {
                best_sse = sse;
                best = Some(BestSplit::Numeric {
                    var: j,
                    point: (col[ii] + col[ii_next]) / 2.0,
                    last_left: k,
                    bin: None,
                });
            }
        }
    }

    // Mark the rows that go left, and fix the node.
    let (node, n_left, numeric_var) = match best {
        None => return leaf(nodes),
        Some(BestSplit::Numeric {
            var,
            point,
            last_left,
            bin,
        }) => {
            {
                let TreeWork {
                    rows,
                    goes_left,
                    sorted,
                    ..
                } = &mut *work;
                match bin {
                    Some(b) => {
                        // Histogram mode: the marker comes from the binned
                        // code, `work.sorted` is not maintained.
                        let index = &spec.binning[var]
                            .as_ref()
                            .expect("histogram split requires a binning")
                            .index;
                        for &ii in &rows[lo..hi] {
                            goes_left[ii] = (index[ii] as usize) <= b;
                        }
                    }
                    None => {
                        let order = &sorted[var * stride + lo..var * stride + hi];
                        for (k, &ii) in order.iter().enumerate() {
                            goes_left[ii] = k <= last_left;
                        }
                    }
                }
            }
            (
                TreeNode::Split {
                    var,
                    point,
                    left: 0,
                    right: 0,
                },
                last_left + 1,
                Some(var),
            )
        }
        Some(BestSplit::Levels {
            var,
            mut left_levels,
        }) => {
            let col = design.column(var);
            let in_set = |levels: &[usize], ii: usize| {
                let v = col[ii];
                v >= 0.0 && levels.binary_search(&(v as usize)).is_ok()
            };
            let rows = &work.rows[lo..hi];
            // Store the lighter child's levels: a level the fit never saw is
            // routed away from the stored set, so it follows the child that
            // held more of the training weight.
            let listed_weight: f64 = rows
                .iter()
                .filter(|&&ii| in_set(&left_levels, ii))
                .map(|&ii| weights[ii])
                .sum();
            if listed_weight > total_w - listed_weight {
                let mut other: Vec<usize> = rows
                    .iter()
                    .filter(|&&ii| !in_set(&left_levels, ii))
                    .map(|&ii| col[ii] as usize)
                    .collect();
                other.sort_unstable();
                other.dedup();
                left_levels = other;
            }
            let mut n_left = 0;
            for &ii in rows {
                let left = in_set(&left_levels, ii);
                work.goes_left[ii] = left;
                n_left += usize::from(left);
            }
            (
                TreeNode::SplitSet {
                    var,
                    left_levels,
                    left: 0,
                    right: 0,
                },
                n_left,
                None,
            )
        }
    };

    let split_idx = nodes.len();
    nodes.push(node);
    // Hand each child its rows in every column's order: a stable partition of
    // this node's ranges, so each level of the tree costs O(n p) and no
    // buffer is allocated per node.
    let TreeWork {
        rows,
        sorted,
        scratch,
        goes_left,
        ..
    } = work;
    if spec.split_method == SplitMethod::Exact {
        for j in 0..p {
            if spec.categorical.get(j).copied().unwrap_or(0) == 0 {
                TreeWork::partition(
                    &mut sorted[j * stride + lo..j * stride + hi],
                    scratch,
                    goes_left,
                );
            }
        }
    }
    match numeric_var {
        // A numeric split's children list their rows in the split column's
        // order, the order their totals are summed in. The histogram path
        // keeps only `rows`, which it partitions directly.
        Some(var) if spec.split_method == SplitMethod::Exact => {
            rows[lo..hi].copy_from_slice(&sorted[var * stride + lo..var * stride + hi])
        }
        _ => TreeWork::partition(&mut rows[lo..hi], scratch, goes_left),
    }
    let left_child = build_tree_recursive(spec, work, lo, lo + n_left, depth + 1, nodes);
    let right_child = build_tree_recursive(spec, work, lo + n_left, hi, depth + 1, nodes);
    match &mut nodes[split_idx] {
        TreeNode::Split { left, right, .. } | TreeNode::SplitSet { left, right, .. } => {
            *left = left_child;
            *right = right_child;
        }
        TreeNode::Leaf { .. } => unreachable!("a split node was pushed at split_idx"),
    }
    split_idx
}
