use ndarray::{Array1, ArrayView1, ArrayViewMut1};

use crate::control::ResponseScale;
use crate::MboostrsError;
use libm::lgamma;

pub(crate) const R_OPTIMIZE_TOL: f64 = 1.220703125e-4;

/// `uniroot()`'s default tolerance, `sqrt(.Machine$double.eps)`.
pub(crate) const R_UNIROOT_TOL: f64 = 1.4901161193847656e-8;

/// Root of `f` on `[ax, bx]`, following R's `zeroin` exactly.
///
/// This is the algorithm behind `uniroot()`: R's `R_zeroin2` in
/// `src/library/stats/src/zeroin.c`, itself Brent's method from Forsythe,
/// Malcolm and Moler. Ported statement by statement rather than reimplemented,
/// because the value it returns is not determined by the tolerance alone.
///
/// On a function with a well-separated crossing, any bracketing method lands
/// within `tol` of the same point and which one it is does not matter. The
/// caller here is [`crate::learners::df_to_lambda`], which inverts an
/// effective-degrees-of-freedom curve that flattens as `lambda` grows: once
/// `df` has fallen to the dimension of the penalty's null space it stops
/// moving, so for a `df` target at that floor the "root" is a plateau
/// thousands of decades wide and the answer is decided entirely by which
/// iterates the search visits. Reproducing R's `lambda` there means
/// reproducing R's iterate sequence, including its convergence test
/// (`2 * DBL_EPSILON * |b| + tol / 2`), its interpolation acceptance rule, and
/// its habit of stepping by `tol_act` when interpolation proposes less.
///
/// `f(ax)` and `f(bx)` must straddle zero; the caller establishes that.
/// Returns the last iterate when `max_iter` is exhausted, as R does.
pub(crate) fn r_zeroin(f: &dyn Fn(f64) -> f64, ax: f64, bx: f64, tol: f64, max_iter: usize) -> f64 {
    let (mut a, mut b) = (ax, bx);
    let (mut fa, mut fb) = (f(ax), f(bx));

    // R tests the endpoints before iterating.
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }

    let mut c = a;
    let mut fc = fa;

    // R's loop counter is `*Maxit + 1`.
    for _ in 0..=max_iter {
        let prev_step = b - a;

        // Keep the smaller |f| in b, and c on the opposite side of the root.
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol_act = 2.0 * f64::EPSILON * b.abs() + tol / 2.0;
        let mut new_step = (c - b) / 2.0;

        if new_step.abs() <= tol_act || fb == 0.0 {
            return b;
        }

        // Interpolate only when the previous step was large enough for the
        // extra accuracy to be meaningful and b really is the better iterate.
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q) = if a == c {
                // Two distinct points only: linear interpolation.
                let t1 = fb / fa;
                (cb * t1, 1.0 - t1)
            } else {
                // Three points: inverse quadratic interpolation.
                let q0 = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                (
                    t2 * (cb * q0 * (q0 - t1) - (b - a) * (t1 - 1.0)),
                    (q0 - 1.0) * (t1 - 1.0) * (t2 - 1.0),
                )
            };

            // Carry the sign in p, so the acceptance test below reads as
            // written in the C source.
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }

            if p < (0.75 * cb * q - (tol_act * q).abs() / 2.0) && p < (prev_step * q / 2.0).abs() {
                new_step = p / q;
            }
        }

        // Never step by less than the tolerance.
        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }

        a = b;
        fa = fb;
        b += new_step;
        fb = f(b);
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }

    b
}

/// Outcome reported by family calibrators with an explicit iterative stopping
/// rule. Most scalar calibrators mirror R's opaque `optimize()` call and leave
/// this as `NotTracked`; PropOdds coordinate descent reports its actual stop.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CalibrationStatus {
    #[default]
    NotTracked,
    Converged,
    LimitReached,
    Failed,
}

/// Inline-expansion of the "view → slice + length asserts" prelude that every
/// family inner loop needs. Asserts pin lengths so LLVM can elide per-element
/// bounds checks inside the loop. Panics on non-contiguous inputs.
macro_rules! s4 {
    ($y:ident, $f:ident, $w:ident, $out:ident) => {{
        let ys: &[f64] = $y.as_slice().expect("y must be contiguous");
        let fs: &[f64] = $f.as_slice().expect("f must be contiguous");
        let ws: &[f64] = $w.as_slice().expect("w must be contiguous");
        let os: &mut [f64] = $out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len();
        assert_eq!(fs.len(), n);
        assert_eq!(ws.len(), n);
        assert_eq!(os.len(), n);
        (ys, fs, ws, os, n)
    }};
}
macro_rules! s3 {
    ($y:ident, $f:ident, $w:ident) => {{
        let ys: &[f64] = $y.as_slice().expect("y must be contiguous");
        let fs: &[f64] = $f.as_slice().expect("f must be contiguous");
        let ws: &[f64] = $w.as_slice().expect("w must be contiguous");
        let n = ys.len();
        assert_eq!(fs.len(), n);
        assert_eq!(ws.len(), n);
        (ys, fs, ws, n)
    }};
}

/// A boosting family: the loss (via its risk and negative gradient), the link
/// (via `response`), and any nuisance parameters the loss calibrates.
pub trait Family: Sync + Send {
    /// Family name, matching R's family naming (e.g. `"gaussian"`).
    fn name(&self) -> &'static str;

    /// Initial predictor value (`eta` at iteration 0), on the link scale.
    ///
    /// R's `mboost` uses a weighted response offset, family-specific; see
    /// each family's doc comment for its formula.
    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64;

    /// Write the length-`n` negative gradient of the risk at fitted value `f`
    /// into `out`.
    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        out: ArrayViewMut1<f64>,
    );

    /// Weighted empirical risk at fitted value `f`.
    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64;

    /// A stable change in training risk along a direction, with nuisance
    /// parameters held fixed. `None` uses the ordinary risk evaluation.
    fn risk_change(
        &self,
        _y: ArrayView1<f64>,
        _f: &[f64],
        _h: &[f64],
        _w: &[f64],
        _step: f64,
    ) -> Option<f64> {
        None
    }

    /// Relative training-risk increase worth reporting for the fixed-step
    /// path. Adaptive losses are omitted unless their recalibration minimizes
    /// the same objective, so consecutive risks remain comparable.
    fn risk_rise_threshold(&self) -> Option<f64> {
        if self.calibrates() {
            None
        } else {
            Some(0.01)
        }
    }

    /// The risk cross-validation scores held-out rows with. It is `risk` for
    /// every family whose loss is its negative log-likelihood (or the loss
    /// being minimised); a family whose upstream `risk` is a different
    /// quantity overrides it (see `Multinomial`).
    fn held_out_risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        self.risk(y, f, w)
    }

    /// Recalibrate nuisance parameters (e.g. Huber's scale, survival sigmas,
    /// NBinomial dispersion) from the current residuals.
    fn calibrate(&mut self, _y: ArrayView1<f64>, _f: ArrayView1<f64>, _w: ArrayView1<f64>) {}

    /// Calibrate and return an explicit convergence event when the family has
    /// a trackable iterative stopping rule.
    fn calibrate_with_status(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
    ) -> CalibrationStatus {
        self.calibrate(y, f, w);
        CalibrationStatus::NotTracked
    }

    /// Whether `calibrate` depends on the state it starts from.
    ///
    /// Almost every family calibrates by a pure function of `(y, f, w)`:
    /// Huber takes the median absolute residual, the survival families and
    /// `NBinomial` run a Brent search over a fixed interval. Re-running such a
    /// calibration at a given `f` gives the same answer whatever the family
    /// held before, which is what lets a boosting path be *resumed* from its
    /// retained fitted values rather than replayed from zero.
    ///
    /// `PropOdds` is the exception: its thresholds come from coordinate
    /// descent seeded from the current thresholds, so the result depends on
    /// where it started. A resumed fit restores them from the retained path
    /// instead of recalibrating, because recalibrating would run sweeps the
    /// original path never ran.
    fn calibration_is_path_dependent(&self) -> bool {
        false
    }

    /// Whether `calibrate` performs family-specific work for this instance.
    /// Used to report exact call counts without timing or branching inside the
    /// candidate-selection loop.
    fn calibrates(&self) -> bool {
        false
    }

    /// Whether the pseudo-residual updates *linearly* with the fitted
    /// increment, i.e. `u_{m+1} = u_m - nu * pred_w`. True only for
    /// squared-error (Gaussian) loss. Enables the engine's covariance/Gram
    /// fast path, which maintains the per-learner gradient projections
    /// incrementally instead of re-projecting against the length-n residual.
    fn supports_gram_update(&self) -> bool {
        false
    }

    /// Whether this family requires unit (w ≡ 1) case weights. Non-unit
    /// weights are otherwise silently accepted by `fit`/`cvrisk`; a family
    /// that hasn't implemented a well-defined non-unit-weight semantics sets
    /// this to `true` so callers get a clear rejection instead of a quietly
    /// wrong fit. No family currently sets this -- [`Auc`] and [`Gehan`]
    /// used to (see their doc comments), but both now implement a
    /// principled weighted gradient (see `auc_ngradient_core`/
    /// `gehan_ngradient_core`).
    fn requires_unit_weights(&self) -> bool {
        false
    }

    /// Whether this family requires case weights in `{0, 1}` (R: `weights =
    /// "zeroone"`). Unlike [`Family::requires_unit_weights`], `0` is a valid
    /// value here -- it is the standard subsampling/k-fold cvrisk domain, so
    /// families reporting this are *not* excluded from `cvrisk`/`stabsel`;
    /// only bootstrap-style integer-count folds (which produce values > 1)
    /// are rejected. Currently only [`Cindex`] sets this.
    fn requires_binary_weights(&self) -> bool {
        false
    }

    /// Update the risk from the previous step's risk and the increment, if the
    /// loss admits it.
    ///
    /// `None` — the default — means the family has no such shortcut and the
    /// engine must recompute the risk from scratch. Only squared-error losses
    /// override it: for `L(y, f) = sum w (y - f)^2`, adding `nu * pred` to `f`
    /// changes the risk by exactly `-2*nu*<wu, pred> + nu^2 * <w*pred, pred>`,
    /// which is `O(n)` from quantities the fit loop already has.
    ///
    /// This used to be a `bool` capability flag beside a method that was
    /// `unreachable!()` by default, so a family that set the flag and forgot
    /// the method panicked at run time rather than failing to compile. Encoding
    /// "no shortcut" as `None` makes the pair unrepresentable apart: there is
    /// one method, and overriding it *is* declaring support.
    fn risk_from_increment(
        &self,
        _risk_old: f64,
        _wu_dot_pred: f64,
        _w_pred_sq: f64,
        _nu: f64,
    ) -> Option<f64> {
        None
    }

    /// Compute the negative gradient and the risk together, defaults to calling
    /// each separately.
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        self.negative_gradient(y, f, w, out.view_mut());
        self.risk(y, f, w)
    }

    /// Inverse link: write the natural-scale prediction for link-scale `f`
    /// into `out`. Defaults to the identity link.
    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        out.assign(&f);
    }

    /// Zero-based class indices for a binary loss, or `None` when this
    /// family has no binary classification rule. A score on the decision
    /// boundary belongs to the first class, as in mboost's `rclass`.
    fn class(&self, _f: ArrayView1<f64>) -> Option<Vec<usize>> {
        None
    }

    /// Optionally re-encode the response before it is consumed by fitting,
    /// risk, or cross-validation. Most families use the raw `y` and return
    /// `None` (no allocation). Binomial overrides this to map a {0,1} response
    /// to its internal {-1,+1} coding. Centralising the transform here keeps
    /// every entry point — `fit`, `cvrisk`, and all language bindings —
    /// consistent (previously each binding recoded y itself, and the cvrisk
    /// path was missed, giving wrong cross-validated risk for Binomial).
    fn encode_response(&self, _y: ArrayView1<f64>) -> Option<Array1<f64>> {
        None
    }

    /// The family's current scalar nuisance parameter (e.g. NBinomial /
    /// Weibull / Loglog / GaussReg sigma), or `None` for families without one
    /// (or with a non-scalar nuisance, like PropOdds thresholds). Used by
    /// cross-validation to capture the per-step INBAG nuisance and reuse it for
    /// the out-of-bag risk, matching R's cvrisk.
    fn nuisance(&self) -> Option<f64> {
        None
    }

    /// Set the scalar nuisance parameter. No-op for families without one.
    fn set_nuisance(&mut self, _value: f64) {}

    /// The family's vector nuisance parameter (e.g. PropOdds K-1 sigma
    /// thresholds), or `None` for families without a vector nuisance.
    /// Stored in BoostingPath for response prediction.
    fn nuisance_vec(&self) -> Option<Vec<f64>> {
        None
    }

    /// Set the family's vector nuisance parameter. No-op for families without
    /// one. Used by cross-validation to restore per-step vector nuisance
    /// (e.g. PropOdds thresholds) for OOB risk evaluation.
    fn set_nuisance_vec(&mut self, _value: Vec<f64>) {}

    /// Validate that `y` lies in this family's domain (e.g. positive values for
    /// Poisson/Gamma). Errors carry the family name and a stable message.
    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError>;

    /// Number of observations from the response vector. For standard families
    /// y has length n; for survival families y is concatenated [time, event]
    /// with length 2n, so this returns y.len() / 2.
    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        y.len()
    }

    /// How many flat entries the response stores per observation.
    ///
    /// Counts entries of the fitted predictor and the weight vector, not of
    /// `y`. Standard and survival families hold one predictor entry and one
    /// weight per observation (span 1; a survival `y` concatenates
    /// `[time, event]` but its predictor does not). `Multinomial` holds the
    /// `K-1` non-reference class coordinates per observation, each with a copy
    /// of that observation's weight (span `n_resp`). Cross-validation divides
    /// the summed out-of-bag risk by the *observation* weight total, so it
    /// must undo this repetition to normalize on the same scale as an in-bag
    /// fit. Defaults to 1.
    fn flat_response_span(&self) -> usize {
        1
    }

    /// Whether the held-out loss is a sum of one term per observation.
    /// False for the Cox partial likelihood and the ranking losses (AUC,
    /// Cindex, Gehan), whose loss compares observations with each other, so
    /// no observation has a loss of its own.
    fn loss_is_row_separable(&self) -> bool {
        true
    }

    /// Heap-allocated clone of this family as a trait object.
    fn clone_family(&self) -> Box<dyn Family>;
}

/// Each observation's held-out loss, unweighted: the term it contributes to
/// [`Family::held_out_risk`] at the family's current nuisance value. The
/// weighted sum of these terms is `held_out_risk(y, f, w)`. `None` when the
/// loss couples observations ([`Family::loss_is_row_separable`]).
///
/// `y` is the family's internal coding (after `encode_response`). A survival
/// `y` is `[time, event]` of length `2 * f.len()`; a Multinomial `f` holds
/// `flat_response_span()` entries per observation.
pub fn held_out_row_losses<F: Family + ?Sized>(
    family: &F,
    y: ArrayView1<f64>,
    f: ArrayView1<f64>,
) -> Option<Vec<f64>> {
    if !family.loss_is_row_separable() {
        return None;
    }
    let span = family.flat_response_span().max(1);
    let survival = y.len() == 2 * f.len() && span == 1;
    let rows = f.len() / span;
    let ones = Array1::<f64>::ones(span);
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let f_row = f.slice(ndarray::s![i * span..(i + 1) * span]).to_owned();
        let y_row = if survival {
            Array1::from(vec![y[i], y[rows + i]])
        } else {
            y.slice(ndarray::s![i * span..(i + 1) * span]).to_owned()
        };
        out.push(family.held_out_risk(y_row.view(), f_row.view(), ones.view()));
    }
    Some(out)
}

fn weighted_mean(y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
    let sw: f64 = w.sum();
    if sw <= 0.0 {
        return 0.0;
    }
    y.dot(&w) / sw
}

/// R's type-7 quantile (default), UNWEIGHTED. Uses linear interpolation:
/// p(k) = (k - 1) / (n - 1), so for quantile q the index is
/// h = q * (n - 1) + 1, and the result interpolates between
/// sorted_y[floor(h)-1] and sorted_y[ceil(h)-1].
///
/// This matches R's `median(...)` / `quantile(..., type = 7)` with no weights,
/// which is what R's Huber family uses for its adaptive delta
/// (`d <- median(abs(y - fit))`). For the weight-aware QuantReg offset use
/// `weighted_quantile`.
pub fn type7_quantile(y: ArrayView1<f64>, q: f64) -> f64 {
    let mut sorted: Vec<f64> = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    type7_from_sorted(&sorted, q)
}

/// Type-7 quantile from an already-sorted slice.
fn type7_from_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    // R's type-7: h = q * (n - 1) + 1
    let h = q * (n as f64 - 1.0) + 1.0;
    let lo = (h.floor() as usize).saturating_sub(1).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let frac = h - h.floor();
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// R's QuantReg offset convention: `quantile(y[rep(1:n, w)], q)`. This computes
/// the same integer-frequency type-7 quantile from sorted `(value, count)`
/// pairs without materializing the expanded sample. Memory is therefore O(n),
/// not O(sum(w)). With `w ≡ 1` this equals the unweighted type-7 quantile.
pub fn weighted_quantile(y: ArrayView1<f64>, w: ArrayView1<f64>, q: f64) -> f64 {
    assert_eq!(
        y.len(),
        w.len(),
        "weighted_quantile: y/weights length mismatch"
    );
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let mut frequencies: Vec<(f64, u128)> = y
        .iter()
        .zip(w.iter())
        .filter_map(|(&value, &weight)| {
            let count = if weight > 0.0 {
                // Rust's float-to-integer cast truncates finite fractional
                // values and saturates huge values, matching the prior
                // expansion semantics without a potentially enormous loop.
                weight as usize as u128
            } else {
                0
            };
            (count > 0).then_some((value, count))
        })
        .collect();
    if frequencies.is_empty() {
        return type7_quantile(y, q);
    }
    frequencies.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let total = frequencies
        .iter()
        .fold(0_u128, |sum, (_, count)| sum.saturating_add(*count));
    debug_assert!(total > 0);

    // Keep the same arithmetic order as `type7_from_sorted` so ordinary
    // frequency weights remain bit-for-bit compatible with the former
    // expansion implementation.
    let h = q * (total - 1) as f64 + 1.0;
    let h_floor = h.floor();
    let lower_rank = (h_floor as u128).saturating_sub(1).min(total - 1);
    let upper_rank = lower_rank.saturating_add(1).min(total - 1);
    let fraction = h - h_floor;
    let value_at = |rank: u128| {
        let mut cumulative = 0_u128;
        for &(value, count) in &frequencies {
            cumulative = cumulative.saturating_add(count);
            if rank < cumulative {
                return value;
            }
        }
        frequencies.last().map_or(0.0, |(value, _)| *value)
    };
    let lower = value_at(lower_rank);
    let upper = value_at(upper_rank);
    lower * (1.0 - fraction) + upper * fraction
}

/// Brent's 1D minimizer matching R's optimize().
/// Finds the minimum of f(x) on [a, b] to relative tolerance ~1e-10.
/// Uses the algorithm from R's C source (src/library/stats/src/optimize.c).
/// 1-D Brent minimizer matching R's `Brent_fmin` from
/// `src/library/stats/src/optimize.c` line-by-line.
/// Bracket a one-dimensional minimum on a coarse grid and refine it by
/// golden section — the shared core of the `LineSearch` step rule.
///
/// `evaluate` maps a candidate multiple `rho` of a step direction to the
/// training loss at `f + rho·h`, with non-finite values counted as positive
/// infinity so the search backs away from overflow instead of chasing it.
/// The grid 0 … 4 is dense near the classical unit step (a least-squares
/// fit to the negative gradient undershoots the loss-optimal multiple for
/// low-curvature families such as Poisson with low counts, and overshoots
/// for large counts), doubles its upper end while the loss still falls,
/// halves its lower end while no grid point lowers the loss, and
/// `max_iter` golden sections then sharpen the bracket. The answer is
/// multiplied by `nu` by the caller, so a handful of sections suffice.
pub(crate) fn golden_grid_min(evaluate: impl FnMut(f64) -> f64, max_iter: usize) -> f64 {
    let mut evaluate = evaluate;
    let grid = [0.0f64, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
    let mut best = 0.0f64;
    let mut best_r = evaluate(0.0);
    let mut best_index = 0usize;
    for (index, &rho) in grid.iter().enumerate().skip(1) {
        let risk = evaluate(rho);
        if risk < best_r {
            best_r = risk;
            best = rho;
            best_index = index;
        }
    }
    let (lo, hi) = if best_index == 0 {
        // No grid point lowers the loss: the direction's scale overshoots by
        // orders of magnitude (Poisson on counts near 1e5, where the
        // least-squares direction is about y and the loss-optimal multiple
        // about 1/mu). Halve until the loss falls; the minimum then lies
        // below twice that multiple.
        let mut upper = grid[1];
        let mut found = false;
        for _ in 0..60 {
            upper *= 0.5;
            let risk = evaluate(upper);
            if risk < best_r {
                best_r = risk;
                best = upper;
                found = true;
                break;
            }
        }
        if !found {
            return 0.0;
        }
        (0.0, 2.0 * upper)
    } else if best_index == grid.len() - 1 {
        // Still falling at the last grid point: keep doubling while it does.
        let mut lower = grid[grid.len() - 2];
        let mut upper = grid[grid.len() - 1];
        while upper < 64.0 {
            let risk = evaluate(upper * 2.0);
            if risk >= best_r {
                upper *= 2.0;
                break;
            }
            lower = upper;
            upper *= 2.0;
            best_r = risk;
            best = upper;
        }
        (lower, upper)
    } else {
        (grid[best_index - 1], grid[best_index + 1])
    };
    if max_iter == 0 || hi <= lo {
        return best;
    }
    let section = 0.618_033_988_749_989;
    let mut a = lo;
    let mut b = hi;
    let mut x1 = b - section * (b - a);
    let mut x2 = a + section * (b - a);
    let mut f1 = evaluate(x1);
    let mut f2 = evaluate(x2);
    for _ in 0..max_iter {
        if f1 <= f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - section * (b - a);
            f1 = evaluate(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + section * (b - a);
            f2 = evaluate(x2);
        }
    }
    if f1.min(f2) >= best_r {
        best
    } else if f1 < f2 {
        x1
    } else {
        x2
    }
}

/// `a * b + c` as R's compiled `Brent_fmin` evaluates it.
///
/// R's `optimize()` is C, and on aarch64 both clang and GCC contract
/// `a * b + c` into one fused multiply-add by default (x86-64 builds without
/// FMA flags do not). The single rounding moves a probe by an ulp, and on a
/// flat objective (the Laplace risk between the two middle order statistics)
/// that is enough to send the search to another point of the flat interval:
/// 0.781342 against R's 0.782023 on one 200-row sample (issue #161).
#[inline]
fn fused(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

pub(crate) fn minimize_brent(
    f: &dyn Fn(f64) -> f64,
    ax: f64,
    bx: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    // c is the squared inverse of the golden ratio
    let c = (3.0 - 5.0f64.sqrt()) * 0.5;

    // eps is approximately the square root of the relative machine precision
    let eps = f64::EPSILON.sqrt();
    let tol3 = tol / 3.0;

    let mut a = ax;
    let mut b = bx;
    let mut v = fused(c, b - a, a);
    let mut w = v;
    let mut x = v;
    let mut d = 0.0f64;
    let mut e = 0.0f64;
    let mut fx = f(x);
    let mut fv = fx;
    let mut fw = fx;

    for _ in 0..max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = fused(eps, x.abs(), tol3);
        let t2 = 2.0 * tol1;

        // check stopping criterion
        if (x - xm).abs() <= t2 - (b - a) * 0.5 {
            return x;
        }

        let mut p = 0.0f64;
        let mut q = 0.0f64;
        let mut r = 0.0f64;

        if e.abs() > tol1 {
            // fit parabola
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            // clang fuses the left product: fma(x - v, q, -((x - w) * r)).
            p = fused(x - v, q, -((x - w) * r));
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }

        if p.abs() >= (q * 0.5 * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            // a golden-section step
            if x < xm {
                e = b - x;
            } else {
                e = a - x;
            }
            d = c * e;
        } else {
            // a parabolic-interpolation step
            d = p / q;
            let u = x + d;
            // f must not be evaluated too close to ax or bx
            if u - a < t2 || b - u < t2 {
                d = tol1;
                if x >= xm {
                    d = -d;
                }
            }
        }

        // f must not be evaluated too close to x
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);

        // update a, b, v, w, and x
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
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

    x
}

// ─── Gaussian ────────────────────────────────────────────────────────

/// Squared-error loss for continuous responses (`L(y, f) = (y - f)^2`).
#[derive(Clone, Copy)]
pub struct Gaussian;

impl Family for Gaussian {
    #[inline(always)]
    fn name(&self) -> &'static str {
        "Gaussian"
    }

    #[inline(always)]
    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        weighted_mean(y, w)
    }

    #[inline(always)]
    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        for i in 0..n {
            os[i] = ys[i] - fs[i];
        }
    }

    #[inline(always)]
    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            risk += ws[i] * r * r;
        }
        risk
    }

    #[inline(always)]
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            os[i] = r;
            risk += ws[i] * r * r;
        }
        risk
    }

    #[inline(always)]
    fn check_y(&self, _y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        Ok(())
    }

    fn supports_gram_update(&self) -> bool {
        true
    }

    fn risk_from_increment(
        &self,
        risk_old: f64,
        wu_dot_pred: f64,
        w_pred_sq: f64,
        nu: f64,
    ) -> Option<f64> {
        Some(risk_old - 2.0 * nu * wu_dot_pred + nu * nu * w_pred_sq)
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(Gaussian)
    }
}

// ─── GaussReg (Gaussian for regression) ─────────────────────────────

/// Gaussian regression family — identical to [`Gaussian`] but with a
/// distinct name for semantic clarity (regression with squared-error loss).
///
/// In R's `mboost`, `GaussReg()` is the same as `Gaussian()` but
/// explicitly marks the model for regression rather than classification.
#[derive(Clone, Copy)]
pub struct GaussReg;

impl Family for GaussReg {
    #[inline(always)]
    fn name(&self) -> &'static str {
        "GaussReg"
    }

    #[inline(always)]
    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        weighted_mean(y, w)
    }

    #[inline(always)]
    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        for i in 0..n {
            os[i] = ys[i] - fs[i];
        }
    }

    #[inline(always)]
    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            risk += ws[i] * r * r;
        }
        risk
    }

    #[inline(always)]
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            os[i] = r;
            risk += ws[i] * r * r;
        }
        risk
    }

    #[inline(always)]
    fn check_y(&self, _y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        Ok(())
    }

    fn supports_gram_update(&self) -> bool {
        true
    }

    fn risk_from_increment(
        &self,
        risk_old: f64,
        wu_dot_pred: f64,
        w_pred_sq: f64,
        nu: f64,
    ) -> Option<f64> {
        Some(risk_old - 2.0 * nu * wu_dot_pred + nu * nu * w_pred_sq)
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(GaussReg)
    }
}

// ─── Binomial ────────────────────────────────────────────────────────

/// Logistic loss for a binary or binomial response, on mboost's half-logit
/// link: `f = log(pi / (1 - pi)) / 2`, so `pi = 1 / (1 + exp(-2 f))`.
///
/// Accepted responses:
///
/// - binary labels, coded `{0, 1}` or `{-1, +1}`;
/// - a success proportion `p` in `[0, 1]`, with the prior weight of the row
///   set to its number of trials (a `cbind(success, failure)` response is
///   `p = success / (success + failure)` with weight `success + failure`).
///
/// A row with proportion `p` and weight `n` contributes the same loss,
/// gradient and offset as `n` Bernoulli rows of which a share `p` are
/// successes, so the fit equals the fit on the expanded data. The per-row
/// loss is the binomial negative log-likelihood per trial,
///
/// `L(p, f) = p log(1 + exp(-2 f)) + (1 - p) log(1 + exp(2 f))`,
///
/// which is mboost's `log(1 + exp(-2 y f))` when `y = 2p - 1` is a label in
/// `{-1, +1}`. The negative gradient is `-(1 / log 2) dL/df =
/// 2 (p - pi) / log 2`; the `1 / log 2` factor is mboost's and rescales every
/// step's gradient by the same constant. `f` is clamped to `[-36, 36]` in the
/// loss and the gradient, as mboost does.
///
/// [`Family::encode_response`] maps the labels to `{-1, +1}` and leaves a
/// proportion strictly between 0 and 1 unchanged. The encoding is therefore
/// idempotent, which matters because cross-validation encodes the response
/// and then passes it to a fit that encodes it again; an affine map
/// `p -> 2p - 1` would turn `0.7` into `0.4` and then `-0.2`. The loss, the
/// gradient and the offset read the success share as `y` clamped to
/// `[0, 1]`, so `-1` and `0` both count as a failure.
#[derive(Clone, Copy)]
pub struct Binomial;

/// Negative gradient and loss of one Binomial row at fitted value `f`; see
/// [`Binomial`] for the formulas. A label (`p` exactly 0 or 1) takes mboost's
/// one-exponential form, so label fits keep their previous arithmetic.
#[inline(always)]
fn binomial_row(y: f64, f: f64) -> (f64, f64) {
    let ln2 = std::f64::consts::LN_2;
    // Clamp f: an unclamped exp overflows to +inf for strongly misclassified
    // points (|f| large), giving inf/inf = NaN in the gradient.
    let fc = f.clamp(-36.0, 36.0);
    let p = y.clamp(0.0, 1.0);
    if p == 1.0 || p == 0.0 {
        let s = if p == 1.0 { 1.0 } else { -1.0 };
        let ef = (-2.0 * s * fc).exp();
        ((2.0 * s * ef) / (ln2 * (1.0 + ef)), (1.0 + ef).ln())
    } else {
        // ef = exp(-2f); log(1 + exp(2f)) = log(1 + 1/ef). Both stay finite
        // under the clamp.
        let ef = (-2.0 * fc).exp();
        let loss = p * (1.0 + ef).ln() + (1.0 - p) * (1.0 + 1.0 / ef).ln();
        let ngrad = (2.0 * (p * ef - (1.0 - p))) / (ln2 * (1.0 + ef));
        (ngrad, loss)
    }
}

impl Family for Binomial {
    fn class(&self, f: ArrayView1<f64>) -> Option<Vec<usize>> {
        Some(f.iter().map(|&v| usize::from(v > 0.0)).collect())
    }

    fn name(&self) -> &'static str {
        "Binomial"
    }

    fn encode_response(&self, y: ArrayView1<f64>) -> Option<Array1<f64>> {
        // Labels go to {-1, +1}: 0 -> -1, 1 -> +1, -1 stays. A proportion
        // strictly between 0 and 1 is left as it is, so encoding an encoded
        // response returns it unchanged (see the `Binomial` doc comment).
        // Values outside [-1, 1] pass through; `check_y` rejects them.
        Some(y.mapv(|yi| {
            if (yi + 1.0).abs() <= 1e-10 || yi.abs() <= 1e-10 {
                -1.0
            } else if (yi - 1.0).abs() <= 1e-10 {
                1.0
            } else {
                yi
            }
        }))
    }

    /// `f0 = log(pbar / (1 - pbar)) / 2`, where `pbar = sum(w p) / sum(w)` is
    /// the weighted success share, clamped to `[1e-15, 1 - 1e-15]`.
    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let n = y.len();
        let mut pos_weight = 0.0f64;
        for i in 0..n {
            pos_weight += w[i] * y[i].clamp(0.0, 1.0);
        }
        let sw: f64 = w.sum();
        let p = if sw > 0.0 {
            let raw = pos_weight / sw;
            raw.clamp(1e-15, 1.0 - 1e-15)
        } else {
            0.5
        };
        0.5 * (p / (1.0 - p)).ln()
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        for i in 0..n {
            os[i] = binomial_row(ys[i], fs[i]).0;
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * binomial_row(ys[i], fs[i]).1;
        }
        risk
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let mut risk = 0.0f64;
        for i in 0..n {
            let (ngrad, loss) = binomial_row(ys[i], fs[i]);
            os[i] = ngrad;
            risk += ws[i] * loss;
        }
        risk
    }

    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = fs.len();
        assert_eq!(os.len(), n);
        for i in 0..n {
            os[i] = 1.0 / (1.0 + (-2.0 * fs[i]).exp());
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            // Accept {0,1}, {-1,+1}, and proportions in [0,1] (cbind response
            // converted to proportions, with trial counts passed as weights).
            let minus_one = (yi + 1.0).abs() <= 1e-10;
            let proportion = (-1e-10..=1.0 + 1e-10).contains(&yi);
            if !minus_one && !proportion {
                return Err(MboostrsError::BadResponse(
                    "Binomial".to_string(),
                    format!("y must be in {{-1, +1}}, {{0, 1}}, or [0, 1] proportions, got {yi}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(Binomial)
    }
}

// ─── Poisson ─────────────────────────────────────────────────────────

/// Poisson deviance for count responses; log link with default offset
/// `log(weighted_mean(y, w))`.
#[derive(Clone, Copy)]
pub struct Poisson;

impl Family for Poisson {
    fn name(&self) -> &'static str {
        "Poisson"
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let mu = weighted_mean(y, w);
        if mu > 0.0 {
            mu.ln()
        } else {
            0.0
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        for i in 0..n {
            os[i] = ys[i] - fs[i].exp();
        }
    }

    // R's Poisson risk: sum(w * -dpois(y, exp(f), log=TRUE))
    // = sum(w * [exp(f) - y*f - log(y!)]) where log(y!) = lgamma(y+1)
    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let mut risk = 0.0f64;
        for i in 0..n {
            let log_y_fact = if ys[i] <= 0.0 {
                0.0
            } else {
                lgamma(ys[i] + 1.0)
            };
            risk += ws[i] * (fs[i].exp() - ys[i] * fs[i] + log_y_fact);
        }
        risk
    }

    /// Fused ngradient + risk in a single pass for the boosting hot loop.
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let mut risk = 0.0f64;
        for i in 0..n {
            let ef = fs[i].exp();
            let log_y_fact = if ys[i] <= 0.0 {
                0.0
            } else {
                lgamma(ys[i] + 1.0)
            };
            os[i] = ys[i] - ef;
            risk += ws[i] * (ef - ys[i] * fs[i] + log_y_fact);
        }
        risk
    }

    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = fs.len();
        assert_eq!(os.len(), n);
        for i in 0..n {
            os[i] = fs[i].exp();
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            if yi < 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Poisson".to_string(),
                    format!("y must be >= 0, got {yi}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(Poisson)
    }
}

// ─── Laplace ─────────────────────────────────────────────────────────

/// `L1` loss for robust regression; the offset is the L1 risk minimizer found
/// by Brent search, matching R.
#[derive(Clone, Copy, Debug, Default)]
pub struct Laplace {
    /// The scale the response is divided by before fitting (see
    /// [`gradient_scale`]); [`ResponseScale::Raw`] is R mboost's path.
    pub response_scale: ResponseScale,
}

impl Laplace {
    /// A Laplace family fitted on the response divided by `response_scale`.
    pub fn with_response_scale(response_scale: ResponseScale) -> Self {
        Self { response_scale }
    }
}

/// The multiplier a family whose gradient does not grow with the response
/// (absolute and check loss: `sign(y - f)`, `tau - 1{y < f}`) applies to it
/// to fit `y / s` and map the model back.
///
/// Boosting `y / s` takes the steps `nu * fit(g)` on `f / s` with the same
/// `g`, because the sign of a residual does not change when the response is
/// divided by `s`; on `f` that is `nu * s * fit(g)`, since a base learner's fit
/// is linear in its target. The offset (a weighted quantile) and the risk
/// already scale with the units. So multiplying the gradient by `s` is the
/// scaled fit, mapped back exactly. Without it each step moves the fit by
/// about `nu * tau` in response units, and a response in dollars needs millions
/// of steps. [`ResponseScale::Auto`] takes `s` as the weighted SD of the
/// response the fit sees, so a cross-validation fold and a fit on the same
/// rows take the same steps.
pub fn gradient_scale(scale: ResponseScale, y: &[f64], w: &[f64]) -> f64 {
    match scale {
        ResponseScale::Raw => 1.0,
        ResponseScale::Fixed(s) => s,
        ResponseScale::Auto => {
            let (mut sw, mut swy) = (0.0f64, 0.0f64);
            for (&v, &wi) in y.iter().zip(w) {
                if wi > 0.0 && v.is_finite() {
                    sw += wi;
                    swy += wi * v;
                }
            }
            if sw <= 0.0 {
                return 1.0;
            }
            let mean = swy / sw;
            let mut ss = 0.0f64;
            for (&v, &wi) in y.iter().zip(w) {
                if wi > 0.0 && v.is_finite() {
                    ss += wi * (v - mean) * (v - mean);
                }
            }
            let sd = (ss / sw).sqrt();
            if sd.is_finite() && sd > 0.0 {
                sd
            } else {
                1.0
            }
        }
    }
}

impl Family for Laplace {
    fn name(&self) -> &'static str {
        "Laplace"
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        // R: optimize(risk, interval = range(y))$minimum with risk =
        // Σ w_i·|y_i − f|. Replicate the Brent search exactly (same interval,
        // tolerance, and probe sequence) rather than returning the exact
        // weighted median: Brent stops within ~1.2e-4·|x| of the L1 kink, so
        // the exact median differs from R by that much on every fit and broke
        // offset/risk/coef parity at 1e-10 (R-quirks D1, LAP1).
        let ys = y.as_slice().expect("y contiguous");
        let ws = w.as_slice().expect("w contiguous");
        let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !y_min.is_finite() || !y_max.is_finite() {
            return 0.0;
        }
        // Brent's tolerance is absolute, so R's offset is found more finely,
        // relative to the data, in larger units. A scaled fit runs the same
        // search on `y / s` and scales the result back, so the offset is the
        // same in any units; the raw response keeps R's search.
        let scale = gradient_scale(self.response_scale, ys, ws);
        let risk = |f: f64| -> f64 {
            ys.iter()
                .zip(ws)
                .map(|(yi, wi)| wi * (yi / scale - f).abs())
                .sum()
        };
        scale * minimize_brent(&risk, y_min / scale, y_max / scale, R_OPTIMIZE_TOL, 500)
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let scale = gradient_scale(self.response_scale, ys, ws);
        for i in 0..n {
            os[i] = scale * (ys[i] - fs[i]).signum();
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * (ys[i] - fs[i]).abs();
        }
        risk
    }

    /// Fused ngradient + risk in a single pass.
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let scale = gradient_scale(self.response_scale, ys, ws);
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            os[i] = scale * r.signum();
            risk += ws[i] * r.abs();
        }
        risk
    }

    fn check_y(&self, _y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(*self)
    }
}

// ─── QuantReg ────────────────────────────────────────────────────────

/// Quantile regression loss at level `tau` (`rho_tau` pinball loss).
#[derive(Debug, Clone)]
pub struct QuantReg {
    /// The target quantile level, in `(0, 1)`.
    pub tau: f64,
    /// Additive shift applied to the fitted quantile line (R's `qoffset`).
    pub qoffset: f64,
    /// The scale the response is divided by before fitting (see
    /// [`gradient_scale`]); [`ResponseScale::Raw`] is R mboost's path.
    pub response_scale: ResponseScale,
}

/// `value` strictly inside `(0, 1)`, and finite.
fn check_open_unit(family: &str, name: &str, value: f64) -> Result<(), MboostrsError> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        return Err(MboostrsError::BadFamilyParameter(
            family.to_string(),
            format!("{name} must be in (0, 1), got {value}"),
        ));
    }
    Ok(())
}

/// `value` strictly positive, and finite.
fn check_positive(family: &str, name: &str, value: f64) -> Result<(), MboostrsError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(MboostrsError::BadFamilyParameter(
            family.to_string(),
            format!("{name} must be positive, got {value}"),
        ));
    }
    Ok(())
}

/// `value` at least `minimum`.
fn check_at_least(
    family: &str,
    name: &str,
    value: usize,
    minimum: usize,
) -> Result<(), MboostrsError> {
    if value < minimum {
        return Err(MboostrsError::BadFamilyParameter(
            family.to_string(),
            format!("{name} must be at least {minimum}, got {value}"),
        ));
    }
    Ok(())
}

impl QuantReg {
    /// Unchecked. Kept for callers that already validate; prefer
    /// [`QuantReg::try_new`], which rejects a `tau` outside `(0, 1)` instead
    /// of producing a check loss that is minimised at an endpoint.
    pub fn new(tau: f64) -> Self {
        Self {
            tau,
            qoffset: 0.5,
            response_scale: ResponseScale::Raw,
        }
    }

    /// Unchecked. Prefer [`QuantReg::try_with_qoffset`].
    pub fn with_qoffset(tau: f64, qoffset: f64) -> Self {
        Self {
            tau,
            qoffset,
            response_scale: ResponseScale::Raw,
        }
    }

    /// The same family fitted on the response divided by `response_scale`.
    pub fn with_response_scale(mut self, response_scale: ResponseScale) -> Self {
        self.response_scale = response_scale;
        self
    }

    /// `tau` in `(0, 1)`.
    pub fn try_new(tau: f64) -> Result<Self, MboostrsError> {
        Self::try_with_qoffset(tau, 0.5)
    }

    /// `tau` and `qoffset` in `(0, 1)`.
    pub fn try_with_qoffset(tau: f64, qoffset: f64) -> Result<Self, MboostrsError> {
        check_open_unit("QuantReg", "tau", tau)?;
        check_open_unit("QuantReg", "qoffset", qoffset)?;
        Ok(Self::with_qoffset(tau, qoffset))
    }
}

impl Family for QuantReg {
    fn name(&self) -> &'static str {
        "QuantReg"
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        weighted_quantile(y, w, self.qoffset)
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let scale = gradient_scale(self.response_scale, ys, ws);
        let tau = scale * self.tau;
        let neg = -scale * (1.0 - self.tau);
        for i in 0..n {
            os[i] = if ys[i] >= fs[i] { tau } else { neg };
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let tau = self.tau;
        let one_minus_tau = 1.0 - tau;
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            if r >= 0.0 {
                risk += ws[i] * tau * r;
            } else {
                risk += ws[i] * one_minus_tau * (-r);
            }
        }
        risk
    }

    fn check_y(&self, _y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(self.clone())
    }
}

// ─── ExpectReg ───────────────────────────────────────────────────────

/// Expectile regression loss at asymmetry level `tau` (asymmetric squares).
#[derive(Clone)]
pub struct ExpectReg {
    /// The asymmetry level, in `(0, 1)`.
    pub tau: f64,
}

impl ExpectReg {
    /// Construct an expectile family at the given asymmetry `tau`.
    pub fn new(tau: f64) -> Self {
        Self { tau }
    }
}

impl Family for ExpectReg {
    fn name(&self) -> &'static str {
        "ExpectReg"
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        // R's ExpectReg offset is `mean(y[w == 1])` -- the mean over
        // unit-weight observations only, and INDEPENDENT of tau. With w ≡ 1
        // this is the plain mean, so default (unweighted) fits are unchanged;
        // it only differs under non-uniform case weights. Fall back to the
        // weighted mean if no observation has weight exactly 1 (R would yield
        // NaN there; this keeps the fit well-defined in that degenerate case).
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for (&yi, &wi) in y.iter().zip(w.iter()) {
            if wi == 1.0 {
                sum += yi;
                cnt += 1;
            }
        }
        if cnt > 0 {
            sum / cnt as f64
        } else {
            weighted_mean(y, w)
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        let pos_coef = 2.0 * self.tau;
        let neg_coef = 2.0 * (1.0 - self.tau);
        for i in 0..n {
            let r = ys[i] - fs[i];
            os[i] = if r > 0.0 {
                pos_coef * r
            } else if r < 0.0 {
                neg_coef * r
            } else {
                0.0
            };
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let tau = self.tau;
        let one_minus_tau = 1.0 - tau;
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            let coef = if r >= 0.0 { tau } else { one_minus_tau };
            risk += ws[i] * coef * r * r;
        }
        risk
    }

    fn check_y(&self, _y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(ExpectReg { tau: self.tau })
    }
}

// ─── Huber ───────────────────────────────────────────────────────────

/// Huber loss with either a fixed or adaptive (median-absolute-residual)
/// scale parameter `delta`.
#[derive(Debug, Clone)]
pub struct Huber {
    /// Fixed `delta` for the loss, or `None` for the adaptive default.
    pub delta: Option<f64>,
    /// Whether `delta` was supplied by the caller rather than estimated.
    pub user_supplied_delta: bool,
    /// Scratch buffer for quartic absolute residuals.
    pub abs_buf: Vec<f64>,
    /// Previous fit values (for delta lag in ngradient). None = use zeros
    /// (matching R's closure where `fit` starts at 0).
    pub prev_f: Option<Array1<f64>>,
}

impl Huber {
    /// Unchecked. Prefer [`Huber::try_new`]: a non-positive `delta` makes the
    /// loss decrease with the residual and reports a negative risk.
    pub fn new(delta: Option<f64>) -> Self {
        Self {
            delta,
            user_supplied_delta: delta.is_some(),
            abs_buf: Vec::new(),
            prev_f: None,
        }
    }

    /// `delta` positive and finite, or `None` for the adaptive default.
    pub fn try_new(delta: Option<f64>) -> Result<Self, MboostrsError> {
        if let Some(value) = delta {
            check_positive("Huber", "delta", value)?;
        }
        Ok(Self::new(delta))
    }
}

fn median_abs_slice(buf: &mut Vec<f64>, y: &[f64], f: &[f64]) -> f64 {
    let n = y.len();
    buf.clear();
    buf.extend(y.iter().zip(f.iter()).map(|(y, f)| (y - f).abs()));
    median_buf(buf, n)
}

fn median_abs_slice_zero(buf: &mut Vec<f64>, y: &[f64]) -> f64 {
    let n = y.len();
    buf.clear();
    buf.extend(y.iter().map(|v| v.abs()));
    median_buf(buf, n)
}

fn median_buf(buf: &mut [f64], n: usize) -> f64 {
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    if n % 2 == 0 {
        buf.select_nth_unstable_by(n / 2 - 1, cmp);
        let lower = buf[n / 2 - 1];
        let upper = buf[n / 2..]
            .iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(lower);
        if upper == lower {
            lower
        } else {
            (lower + upper) / 2.0
        }
    } else {
        buf.select_nth_unstable_by(n / 2, cmp);
        buf[n / 2]
    }
}

impl Family for Huber {
    fn name(&self) -> &'static str {
        "Huber"
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let y_owned = y.to_owned();
        let w_owned = w.to_owned();

        // The offset computation is one-time, so we don't need the quickselect
        // optimisation here — the full sort is fine. We DO need self for the
        // quickselect, but offset is &self not &mut self.
        let delta_init = if self.user_supplied_delta {
            self.delta.unwrap_or(1.0)
        } else {
            let mut abs_y: Vec<f64> = y.iter().map(|v| v.abs()).collect();
            abs_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = abs_y.len();
            let d = if n % 2 == 0 {
                (abs_y[n / 2 - 1] + abs_y[n / 2]) / 2.0
            } else {
                abs_y[n / 2]
            };
            if d > 0.0 {
                d
            } else {
                1e-10
            }
        };

        let risk_fn = |f_val: f64| -> f64 {
            let mut s = 0.0f64;
            for i in 0..y_owned.len() {
                let a = (y_owned[i] - f_val).abs();
                s += w_owned[i]
                    * if a < delta_init {
                        a * a / 2.0
                    } else {
                        delta_init * (a - delta_init / 2.0)
                    };
            }
            s
        };

        minimize_brent(&risk_fn, y_min, y_max, R_OPTIMIZE_TOL, 500)
    }

    fn calibrates(&self) -> bool {
        !self.user_supplied_delta
    }

    // The estimated `delta` is the state the risk depends on, so a fold's
    // held-out risk must be computed at the fold fit's own `delta`; exposing
    // it records it on the fitted path and lets held-out scoring restore it.
    // A caller-supplied `delta` is a constant, not a nuisance.
    fn nuisance(&self) -> Option<f64> {
        if self.user_supplied_delta {
            None
        } else {
            self.delta
        }
    }

    fn set_nuisance(&mut self, value: f64) {
        if !self.user_supplied_delta {
            self.delta = Some(value);
        }
    }

    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, _w: ArrayView1<f64>) {
        if !self.user_supplied_delta {
            let ys = y.as_slice().expect("y contiguous");
            let fs = f.as_slice().expect("f contiguous");
            self.delta = Some({
                let d = median_abs_slice(&mut self.abs_buf, ys, fs);
                if d > 0.0 {
                    d
                } else {
                    1e-10
                }
            });
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        _w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        // R's Huber closure keeps a `fit` variable that lags one step behind:
        // ngradient computes d = median(|y - fit_closure|) where fit_closure
        // was set by the PREVIOUS ngradient call's `fit <<- f`. On the very
        // first call, fit_closure = 0, so d = median(|y|).
        let delta = {
            let ys = y.as_slice().expect("y contiguous");
            let d = if let Some(pf) = self.prev_f.as_ref() {
                median_abs_slice(
                    &mut self.abs_buf,
                    ys,
                    pf.as_slice().expect("prev_f contiguous"),
                )
            } else {
                // First call: R's closure fit = 0 → d = median(|y|)
                median_abs_slice_zero(&mut self.abs_buf, ys)
            };
            if d > 0.0 {
                d
            } else {
                1e-10
            }
        };
        // Save current f for next call's delta (reuse allocation).
        match self.prev_f.as_mut() {
            Some(buf) if buf.len() == f.len() => {
                buf.as_slice_mut()
                    .expect("prev_f contiguous")
                    .copy_from_slice(f.as_slice().expect("f contiguous"));
            }
            _ => self.prev_f = Some(f.to_owned()),
        }
        let ys = y.as_slice().expect("y contiguous");
        let fs = f.as_slice().expect("f contiguous");
        let os = out.as_slice_mut().expect("out contiguous");
        let n = y.len();
        for i in 0..n {
            let r = ys[i] - fs[i];
            os[i] = if r.abs() < delta {
                r
            } else {
                delta * r.signum()
            };
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let delta = self.delta.unwrap_or(1.0);
        let (ys, fs, ws, n) = s3!(y, f, w);
        let half_delta = delta / 2.0;
        let mut risk = 0.0f64;
        for i in 0..n {
            let r = ys[i] - fs[i];
            let ar = r.abs();
            if ar < delta {
                risk += ws[i] * r * r * 0.5;
            } else {
                risk += ws[i] * delta * (ar - half_delta);
            }
        }
        risk
    }

    fn check_y(&self, _y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(Huber {
            delta: self.delta,
            user_supplied_delta: self.user_supplied_delta,
            abs_buf: Vec::new(),
            prev_f: None,
        })
    }
}

// ─── GammaReg ────────────────────────────────────────────────────────

/// Gamma regression family with a scalar shape parameter `sigma` calibrated
/// from the data.
#[derive(Clone)]
pub struct GammaReg {
    /// Shape parameter of the gamma response distribution.
    pub sigma: f64,
}

impl Default for GammaReg {
    fn default() -> Self {
        Self { sigma: 1.0 }
    }
}

impl GammaReg {
    /// Construct a Gamma regression family with the default shape (`sigma =
    /// 1.0`).
    pub fn new() -> Self {
        Self::default()
    }
}

/// R's plloss (partial likelihood loss) for Gamma regression.
/// plloss(sigma, y, f) = lgamma(sigma) + sigma*y*exp(-f) - sigma*log(y) - sigma*log(sigma) + sigma*f
fn plloss(sigma: f64, y: f64, f_val: f64) -> f64 {
    lgamma(sigma) + sigma * y * (-f_val).exp() - sigma * y.ln() - sigma * sigma.ln() + sigma * f_val
}

impl Family for GammaReg {
    fn name(&self) -> &'static str {
        "GammaReg"
    }

    fn risk_rise_threshold(&self) -> Option<f64> {
        // Shape calibration minimizes the same partial negative log-likelihood.
        // A material rise therefore reflects the step, not a changing loss.
        Some(1e-6)
    }

    fn risk_change(
        &self,
        y: ArrayView1<f64>,
        f: &[f64],
        h: &[f64],
        w: &[f64],
        step: f64,
    ) -> Option<f64> {
        if step == 0.0 {
            return Some(0.0);
        }
        let mut change = 0.0;
        for i in 0..f.len() {
            if w[i] == 0.0 {
                continue;
            }
            let increment = step * h[i];
            let z = -increment;
            // expm1(z) - z loses its quadratic term near zero. Retaining it
            // avoids a flat floating-point loss when the fit is near the MLE.
            let remainder = if z.abs() < 1e-3 {
                z * z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z * (1.0 / 120.0 + z / 720.0))))
            } else {
                z.exp_m1() - z
            };
            let ratio = y[i] * (-f[i]).exp();
            change += w[i] * (increment * (1.0 - ratio) + ratio * remainder);
        }
        Some(self.sigma * change)
    }

    // The shape, as R's `nuisance()` reports it for `GammaReg()`.
    fn nuisance(&self) -> Option<f64> {
        Some(self.sigma)
    }

    fn set_nuisance(&mut self, value: f64) {
        self.sigma = value;
    }

    // R uses optimize(risk, interval = c(0, max(y^2, na.rm=TRUE)), y = y, w = w)$minimum
    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let y_max_sq = y.iter().map(|yi| yi * yi).fold(0.0f64, f64::max);

        // The risk at a constant f is `W (lgamma(s) - s ln s) - s Σw ln y
        // + s e^{-f} Σwy + s f W`: three sums, then O(1) per evaluation.
        let sigma = self.sigma;
        let (mut total_w, mut sum_wy, mut sum_wlny) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..y.len() {
            total_w += w[i];
            sum_wy += w[i] * y[i];
            sum_wlny += w[i] * y[i].ln();
        }
        let base = total_w * (lgamma(sigma) - sigma * sigma.ln()) - sigma * sum_wlny;
        let risk_fn = |f_val: f64| -> f64 {
            base + sigma * (-f_val).exp() * sum_wy + sigma * f_val * total_w
        };

        minimize_brent(&risk_fn, 0.0, y_max_sq, R_OPTIMIZE_TOL, 500)
    }

    fn calibrates(&self) -> bool {
        true
    }

    // R uses optimize(riskS, interval = nuirange, ...) for sigma calibration
    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) {
        // The risk is `W (lgamma(s) - s ln s) + s Σw (y e^{-f} - ln y + f)`,
        // linear in one sum once `s` is fixed. The search is still R's
        // `optimize`, step for step; each evaluation is O(1) instead of a
        // pass over the rows, which made GammaReg 35x slower than Gaussian.
        let (mut total_w, mut linear) = (0.0f64, 0.0f64);
        for i in 0..y.len() {
            total_w += w[i];
            linear += w[i] * (y[i] * (-f[i]).exp() - y[i].ln() + f[i]);
        }
        let risk_s =
            |sigma: f64| -> f64 { total_w * (lgamma(sigma) - sigma * sigma.ln()) + sigma * linear };

        let sigma_opt = minimize_brent(&risk_s, 0.0, 100.0, R_OPTIMIZE_TOL, 500);
        if sigma_opt > 0.0 && sigma_opt.is_finite() {
            self.sigma = sigma_opt;
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        let sigma = self.sigma;
        for i in 0..n {
            os[i] = sigma * (ys[i] * (-fs[i]).exp() - 1.0);
        }
    }

    // R's GammaReg risk: sum(w * plloss(sigma, y, f))
    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let sigma = self.sigma;
        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * plloss(sigma, ys[i], fs[i]);
        }
        risk
    }

    /// Fused ngradient + risk in a single pass.
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let sigma = self.sigma;
        let mut risk = 0.0f64;
        for i in 0..n {
            let exp_f = (-fs[i]).exp();
            os[i] = sigma * (ys[i] * exp_f - 1.0);
            risk += ws[i] * plloss(sigma, ys[i], fs[i]);
        }
        risk
    }

    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = fs.len();
        assert_eq!(os.len(), n);
        for i in 0..n {
            os[i] = fs[i].exp();
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            if yi <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "GammaReg".to_string(),
                    format!("y must be > 0, got {yi}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(GammaReg { sigma: self.sigma })
    }
}

/// Adaboost exponential loss (AdaExp). Response is coded as -1/+1.
///
/// ngradient = y * exp(-y * f)
/// risk = Σ w_i * exp(-y_i * f_i)
/// offset = 0.5 * log(p / (1 - p)) where p = weighted mean(y > 0)
#[derive(Clone, Copy)]
pub struct AdaExp;

impl Family for AdaExp {
    fn class(&self, f: ArrayView1<f64>) -> Option<Vec<usize>> {
        Binomial.class(f)
    }

    fn response(&self, f: ArrayView1<f64>, out: ArrayViewMut1<f64>) {
        Binomial.response(f, out);
    }
    #[inline(always)]
    fn name(&self) -> &'static str {
        "AdaExp"
    }

    fn encode_response(&self, y: ArrayView1<f64>) -> Option<Array1<f64>> {
        // The public frontends accept the usual binary {0, 1} coding while
        // the exponential loss is written for {-1, +1}. Keep this conversion
        // in the family so pure-Rust, Python, R, fit, and resampling callers
        // all execute the same rule.
        Some(y.mapv(|yi| if yi.abs() < 1e-10 { -1.0 } else { yi }))
    }

    #[inline(always)]
    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().unwrap();
        let ws = w.as_slice().unwrap();
        let n = ys.len();
        assert_eq!(n, ws.len());
        let mut sum_w_pos = 0.0f64;
        let mut sum_w = 0.0f64;
        for i in 0..n {
            if ys[i] > 0.0 {
                sum_w_pos += ws[i];
            }
            sum_w += ws[i];
        }
        let p = if sum_w > 0.0 { sum_w_pos / sum_w } else { 0.5 };
        // Clamp as Binomial does: a one-class training fold would otherwise
        // give an offset of +-inf.
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        0.5 * (p / (1.0 - p)).ln()
    }

    #[inline(always)]
    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        // The gradient is y * exp(-y*f), WITHOUT the weight factor: weights enter
        // only through the base-learner's weighted least squares fit (and the
        // risk), matching R's AdaExp@ngradient. Multiplying by w here would
        // double-count weights for any non-uniform-weight fit (bootstrap, CV
        // folds, case weights).
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        for i in 0..n {
            os[i] = ys[i] * (-ys[i] * fs[i]).exp();
        }
    }

    #[inline(always)]
    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * (-ys[i] * fs[i]).exp();
        }
        risk
    }

    #[inline(always)]
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let mut risk = 0.0f64;
        for i in 0..n {
            let e = (-ys[i] * fs[i]).exp();
            // Gradient excludes the weight factor (see negative_gradient); the
            // risk applies it.
            os[i] = ys[i] * e;
            risk += ws[i] * e;
        }
        risk
    }

    fn supports_gram_update(&self) -> bool {
        false
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            let binary01 = yi.abs() <= 1e-10 || (yi - 1.0).abs() <= 1e-10;
            let binary_pm = (yi + 1.0).abs() <= 1e-10 || (yi - 1.0).abs() <= 1e-10;
            if !binary01 && !binary_pm {
                return Err(MboostrsError::BadResponse(
                    "AdaExp".to_string(),
                    format!("y must be in {{-1, +1}} or {{0, 1}}, got {yi}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(AdaExp)
    }
}

// ─── HingeLoss (SVM-like hinge loss) ──────────────────────────────

/// Hinge loss for binary classification with {-1, +1} response coding.
///
/// **loss_i = max(0, 1 - y_i * f_i)**
///
/// `ngradient_i = [y*f < 1] * ((1-alphaCost)*y if y>0 else alphaCost*y)`
///
/// In R's mboost, `HingeLoss()` has an `alphaCost` parameter (default 0.5)
/// controlling asymmetric misclassification costs. The default 0.5 gives
/// the standard (symmetric) hinge loss.
///
/// **risk = sum(w * max(0, 1 - y*f))**
///
/// **Offset:** 0.5 * log(p / (1-p)) where p = weighted.mean(y > 0, w)
#[derive(Clone, Copy)]
pub struct HingeLoss {
    /// Cost ratio for misclassifying a negative as positive; 0.5 is the
    /// symmetric case.
    pub alpha_cost: f64,
}

impl Default for HingeLoss {
    fn default() -> Self {
        Self { alpha_cost: 0.5 }
    }
}

impl HingeLoss {
    /// Construct a hinge-loss family with the given asymmetry cost.
    pub fn new(alpha_cost: f64) -> Self {
        Self { alpha_cost }
    }
}

impl Family for HingeLoss {
    fn class(&self, f: ArrayView1<f64>) -> Option<Vec<usize>> {
        let threshold = 2.0 * self.alpha_cost - 1.0;
        Some(f.iter().map(|&v| usize::from(v > threshold)).collect())
    }

    fn response(&self, f: ArrayView1<f64>, out: ArrayViewMut1<f64>) {
        Binomial.response(f, out);
    }
    #[inline(always)]
    fn name(&self) -> &'static str {
        "HingeLoss"
    }

    fn encode_response(&self, y: ArrayView1<f64>) -> Option<Array1<f64>> {
        // Like Binomial and AdaExp, accept the frontend's conventional {0, 1}
        // labels while keeping the loss itself on the {-1, +1} margin scale.
        Some(y.mapv(|yi| if yi.abs() < 1e-10 { -1.0 } else { yi }))
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y contiguous");
        let ws = w.as_slice().expect("w contiguous");
        let n = ys.len();
        let mut sum_w_pos = 0.0f64;
        let mut sum_w = 0.0f64;
        for i in 0..n {
            if ys[i] > 0.0 {
                sum_w_pos += ws[i];
            }
            sum_w += ws[i];
        }
        let p = if sum_w > 0.0 { sum_w_pos / sum_w } else { 0.5 };
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        0.5 * (p / (1.0 - p)).ln()
    }

    #[inline(always)]
    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        _w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, os, n) = {
            let ys = y.as_slice().unwrap();
            let fs = f.as_slice().unwrap();
            let os = out.as_slice_mut().unwrap();
            (ys, fs, os, ys.len())
        };
        let ac = self.alpha_cost;
        for i in 0..n {
            let margin = ys[i] * fs[i];
            os[i] = if margin < 1.0 {
                if ys[i] > 0.0 {
                    (1.0 - ac) * ys[i]
                } else {
                    ac * ys[i]
                }
            } else {
                0.0
            };
        }
    }

    #[inline(always)]
    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = {
            let ys = y.as_slice().unwrap();
            let fs = f.as_slice().unwrap();
            let ws = w.as_slice().unwrap();
            (ys, fs, ws, ys.len())
        };
        let ac = self.alpha_cost;
        let mut risk = 0.0f64;
        for i in 0..n {
            let margin = ys[i] * fs[i];
            let class_weight = if ys[i] > 0.0 { 1.0 - ac } else { ac };
            risk += ws[i] * class_weight * if margin < 1.0 { 1.0 - margin } else { 0.0 };
        }
        risk
    }

    #[inline(always)]
    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = {
            let ys = y.as_slice().unwrap();
            let fs = f.as_slice().unwrap();
            let ws = w.as_slice().unwrap();
            let os = out.as_slice_mut().unwrap();
            (ys, fs, ws, os, ys.len())
        };
        let ac = self.alpha_cost;
        let mut risk = 0.0f64;
        for i in 0..n {
            let margin = ys[i] * fs[i];
            let class_weight = if ys[i] > 0.0 { 1.0 - ac } else { ac };
            if margin < 1.0 {
                os[i] = if ys[i] > 0.0 {
                    (1.0 - ac) * ys[i]
                } else {
                    ac * ys[i]
                };
                risk += ws[i] * class_weight * (1.0 - margin);
            } else {
                os[i] = 0.0;
            }
        }
        risk
    }

    fn supports_gram_update(&self) -> bool {
        false
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            let binary01 = yi.abs() <= 1e-10 || (yi - 1.0).abs() <= 1e-10;
            let binary_pm = (yi + 1.0).abs() <= 1e-10 || (yi - 1.0).abs() <= 1e-10;
            if !binary01 && !binary_pm {
                return Err(MboostrsError::BadResponse(
                    "HingeLoss".to_string(),
                    format!("y must be in {{-1, +1}} or {{0, 1}}, got {yi}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(*self)
    }
}

// ─── AUC (smoothed 1 − AUC ranking loss) ────────────────────────────

/// Smoothed 1 − AUC ranking loss, matching R mboost's `AUC()` family.
///
/// The response is a 2-level factor. mboostrs represents it as any raw
/// numeric column with exactly two distinct values; [`Family::encode_response`]
/// maps the larger raw value to +1 (`ind1`, the "positive" class,
/// mirroring R's `levels(y)[2]`) and the smaller to −1 (`ind0`,
/// `levels(y)[1]`).
///
/// **negative_gradient:** each call rescales `f` to `f / sd(f)` (R's n−1
/// sample standard deviation), unless every entry of `f` is already
/// identical (`length(unique(f)) == 1` in R -- true on the very first
/// boosting iteration, where `f ≡ offset ≡ 0`). Build the pairwise
/// score-difference matrix `M0[k, j] = f_std[ind1[j]] − f_std[ind0[k]]`
/// (n0 × n1) and pass it through
/// `approxGrad(x) = 0` for `|x| ≥ 1`, `1 − x` for `0 ≤ x < 1`, `x + 1` for
/// `−1 < x < 0`. The pseudo-residual at a positive-class observation is the
/// column sum of `approxGrad(M0)`; at a negative-class observation it is
/// minus the row sum.
///
/// **risk:** the TRUE (non-smooth) `1 − AUC = 1 − #{M0 > 0} / (n0·n1)`,
/// recomputed from the same sd-standardized `f` used by `negative_gradient`.
/// (R's own `risk` closure reuses the `M0` left over from the last
/// `ngradient` call when `w ≡ 1`, so it is implicitly consistent with
/// whatever standardization `ngradient` last applied; mboostrs recomputes
/// `M0` directly from the `f` it is given instead, so `risk` never depends
/// on call order or on `negative_gradient` having run first.)
///
/// **offset:** 0 (no fit).
///
/// **Weights:** a principled mboostrs *extension*, not R parity. R's
/// `AUC()` treats non-unit weights as integer "case" weights via a
/// `rep(y, w)` / `rep(1:n, w)` expansion, but that expansion indexes the
/// *replicated* vector positionally while `f` keeps its original
/// (unreplicated) length whenever `f` isn't a bare scalar -- a mismatch in
/// R's own closure (verified: R's weighted `ngradient` returns a vector of
/// length `sum(w)`, not `n`, so it cannot even be assigned back against the
/// unreplicated `f`). Rather than reproduce that broken R corner case,
/// `Auc` instead implements a directly-derived weighted gradient: each
/// observation `i` participates with weight `w_i` (`w_i == 0` observations
/// are excluded from the ranking pairs and get gradient `0`), the
/// pre-partition standardization `f <- f / sd(f)` becomes a *weighted*
/// sample sd over the participating (`w > 0`) observations, and every
/// pairwise `approxGrad`/`1{f_i>f_k}` term picks up a `w_i * w_k` factor.
/// This is exactly `ng = -d(risk)/df` for the weighted risk (see
/// `auc_ngradient_core`, `auc_standardize`); it reduces *exactly* --
/// bit-for-bit -- to the unit-weight formulas above when `w ≡ 1`, since
/// every weight factor is then a multiplication by `1.0`. `w` must be
/// non-negative and finite (enforced generically by `fit`/`cvrisk`, not
/// re-checked here); any non-negative `w` is accepted, so bootstrap
/// (integer-count) and subsampling ({0,1}) `cvrisk`/`stabsel` folds both
/// work, unlike R's.
#[derive(Clone, Copy)]
pub struct Auc;

/// Split observation indices into the negative (`ind0`, encoded −1) and
/// positive (`ind1`, encoded +1) classes, matching R's `which(y == ...)`.
/// Not filtered by weight: `w_i == 0` observations stay in their class
/// bucket here, but every downstream sum multiplies through by `w_i`, so
/// they contribute exactly `0` wherever they appear -- see
/// `auc_ngradient_core`/`Auc::risk`. Keeping the partition itself
/// weight-agnostic (rather than filtering to `w > 0`) is what makes the
/// unit-weight (`w ≡ 1`) path bit-identical to before: the class buckets,
/// iteration order, and every intermediate sum are unchanged, only now with
/// an inert `* 1.0` folded in.
fn auc_partition(ys: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut ind1 = Vec::new();
    let mut ind0 = Vec::new();
    for (i, &yi) in ys.iter().enumerate() {
        if yi > 0.0 {
            ind1.push(i);
        } else {
            ind0.push(i);
        }
    }
    (ind1, ind0)
}

/// R's per-call `f <- f / sd(f)` rescale, generalized to a **weighted**
/// sample sd over the participating (`w_i > 0`) observations: `mean_w =
/// sum(w*f)/W`, `var_w = sum(w*(f-mean_w)^2)/(W-1)`, `W = sum(w)`. Skipped
/// (returns `f` unchanged) when `W <= 1` or every participating `f_i` is
/// identical -- the weighted generalization of R's `length(unique(f)) ==
/// 1` guard and the `n < 2` guard both collapse into `W <= 1` when `w ≡ 1`
/// (`W` is then exactly `n`).
///
/// For `w ≡ 1` this is bit-identical to the previous unweighted
/// implementation: every `w_i` factor is `1.0` (an exact IEEE-754
/// no-op under multiplication), `W` sums to exactly `n as f64`, and the
/// summation order (`0..n`) is unchanged.
fn auc_standardize(fs: &[f64], ws: &[f64]) -> Vec<f64> {
    let n = fs.len();
    let w_sum: f64 = ws.iter().sum();
    if w_sum <= 1.0 {
        return fs.to_vec();
    }
    let first_participating = fs
        .iter()
        .zip(ws.iter())
        .find(|&(_, &w)| w > 0.0)
        .map(|(&f, _)| f);
    let all_equal = match first_participating {
        None => true,
        Some(v0) => fs.iter().zip(ws.iter()).all(|(&f, &w)| w <= 0.0 || f == v0),
    };
    if all_equal {
        return fs.to_vec();
    }
    let mut mean = 0.0f64;
    for i in 0..n {
        mean += ws[i] * fs[i];
    }
    mean /= w_sum;
    let mut var = 0.0f64;
    for i in 0..n {
        var += ws[i] * (fs[i] - mean) * (fs[i] - mean);
    }
    var /= w_sum - 1.0;
    let sd = var.sqrt();
    if sd <= 0.0 {
        return fs.to_vec();
    }
    fs.iter().map(|&v| v / sd).collect()
}

/// `approxGrad` from R's `AUC()` closure: the (sub)gradient of `approxLoss`.
#[inline]
fn auc_approx_grad(x: f64) -> f64 {
    if x.abs() >= 1.0 {
        0.0
    } else if x >= 0.0 {
        1.0 - x
    } else {
        x + 1.0
    }
}

/// `approxLoss` from R's `AUC()` closure: a smooth surrogate for
/// `1{x > 0}`, used only by the Gate-2 finite-difference gradient test
/// below (R's own `loss` -- as opposed to `risk` -- is otherwise unused by
/// mboostrs, which tracks the true, non-smooth `risk` for reporting/CV).
#[inline]
#[cfg(test)]
fn auc_approx_loss(x: f64) -> f64 {
    if x < -1.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else if x >= 0.0 {
        1.0 - (1.0 - x) * (1.0 - x) / 2.0
    } else {
        (x + 1.0) * (x + 1.0) / 2.0
    }
}

/// Shared core: fills `os` with the (weighted) AUC pseudo-residuals given
/// already-partitioned class indices and an already-standardized `f`. Split
/// out from `negative_gradient` so the Gate-2 finite-difference unit test
/// can call it with standardization disabled -- see that test for why.
///
/// Weighted derivation: `ng_j = w_j * sum_k(w_k * approxGrad(fj-fk))` for
/// `j` in the positive class, `ng_k = -w_k * sum_j(w_j * approxGrad(fj-fk))`
/// for `k` in the negative class -- exactly `-d(risk)/df` for the weighted
/// risk (`w_i * w_k` pairwise weight on every ranking pair). An observation
/// with `w_i == 0` gets `ng_i == 0` automatically here, with no extra
/// filtering needed: it zeroes out both its own outer factor and every
/// pairwise term it appears in on the other side.
///
/// For `w ≡ 1` this is bit-identical to the previous unweighted
/// implementation: every `ws[i]` is `1.0`, an exact IEEE-754 no-op under
/// multiplication, and the loop structure/order is otherwise unchanged.
fn auc_ngradient_core(ind1: &[usize], ind0: &[usize], fstd: &[f64], ws: &[f64], os: &mut [f64]) {
    // Every pair is `O(n0 * n1)`, and the ranked form below is
    // `O((n0 + n1) log(n0 + n1))` for the same numbers. The crossover is
    // measured rather than assumed: on the sizes a ranking fit actually runs
    // -- thousands of observations, `mstop` gradient evaluations each -- the
    // pairwise loop is the whole cost, and below a few hundred the sort is
    // not worth its own overhead.
    if ind0.len().saturating_mul(ind1.len()) >= AUC_RANKED_CROSSOVER {
        auc_ngradient_ranked(ind1, ind0, fstd, ws, os);
        return;
    }
    auc_ngradient_pairwise(ind1, ind0, fstd, ws, os);
}

/// Above this many pairs, the ranked kernel is used. Below it the pairwise
/// one is, because a sort and two binary searches per observation cost more
/// than a short double loop.
const AUC_RANKED_CROSSOVER: usize = 4096;

/// The definition, evaluated pair by pair.
///
/// Kept as the reference the ranked kernel is checked against, and used below
/// the crossover.
fn auc_ngradient_pairwise(
    ind1: &[usize],
    ind0: &[usize],
    fstd: &[f64],
    ws: &[f64],
    os: &mut [f64],
) {
    for &k in ind0 {
        os[k] = 0.0;
    }
    for &j in ind1 {
        let fj = fstd[j];
        let wj = ws[j];
        let mut col_sum = 0.0;
        for &k in ind0 {
            let g = auc_approx_grad(fj - fstd[k]);
            let wg = ws[k] * g;
            col_sum += wg;
            os[k] -= wj * wg;
        }
        os[j] = wj * col_sum;
    }
}

/// The same sums, in `O((n0 + n1) log(n0 + n1))`.
///
/// `approxGrad` is a tent: `1 - |x|` on `|x| < 1` and zero outside. So for a
/// fixed `f_j` the inner sum splits into two windows that are *linear* in
/// `f_k`:
///
/// * `f_j - 1 < f_k <= f_j` contributes `w_k (1 - f_j + f_k)`;
/// * `f_j < f_k < f_j + 1` contributes `w_k (1 + f_j - f_k)`.
///
/// Sorting one class and taking prefix sums of `w_k` and `w_k f_k` turns each
/// window into two subtractions, and two binary searches find its ends. The
/// half-open boundaries are the tent's own: `x = 0` is in the first window
/// (`approxGrad(0) = 1`) and `|x| = 1` is in neither (`approxGrad(±1) = 0`),
/// which is why the searches are `<=` on one side and `<` on the other.
///
/// The result is the same arithmetic in a different order, so it agrees with
/// [`auc_ngradient_pairwise`] to rounding rather than bit-for-bit.
fn auc_ngradient_ranked(ind1: &[usize], ind0: &[usize], fstd: &[f64], ws: &[f64], os: &mut [f64]) {
    let negatives = SortedClass::new(ind0, fstd, ws);
    let positives = SortedClass::new(ind1, fstd, ws);

    for &j in ind1 {
        os[j] = ws[j] * negatives.tent_sum(fstd[j]);
    }
    for &k in ind0 {
        // `approxGrad` is even, so the sum over `j` of `g(f_j - f_k)` is the
        // same windowed quantity read from the positive class.
        os[k] = -ws[k] * positives.tent_sum(fstd[k]);
    }
}

/// One class sorted by `f`, with prefix sums of `w` and `w * f`.
struct SortedClass {
    f: Vec<f64>,
    weight_prefix: Vec<f64>,
    weighted_f_prefix: Vec<f64>,
}

impl SortedClass {
    fn new(indices: &[usize], fstd: &[f64], ws: &[f64]) -> Self {
        let mut order: Vec<usize> = indices.to_vec();
        order.sort_by(|&a, &b| {
            fstd[a]
                .partial_cmp(&fstd[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut f = Vec::with_capacity(order.len());
        let mut weight_prefix = Vec::with_capacity(order.len() + 1);
        let mut weighted_f_prefix = Vec::with_capacity(order.len() + 1);
        weight_prefix.push(0.0);
        weighted_f_prefix.push(0.0);
        for &index in &order {
            let value = fstd[index];
            let weight = ws[index];
            f.push(value);
            weight_prefix.push(weight_prefix.last().unwrap() + weight);
            weighted_f_prefix.push(weighted_f_prefix.last().unwrap() + weight * value);
        }
        Self {
            f,
            weight_prefix,
            weighted_f_prefix,
        }
    }

    /// `sum_k w_k * approxGrad(centre - f_k)` over this class.
    fn tent_sum(&self, centre: f64) -> f64 {
        // Window 1: `centre - 1 < f_k <= centre`, weight `1 - centre + f_k`.
        let lower_start = self.upper_bound(centre - 1.0);
        let lower_end = self.upper_bound_inclusive(centre);
        let (weight_lower, weighted_f_lower) = self.range(lower_start, lower_end);
        // Window 2: `centre < f_k < centre + 1`, weight `1 + centre - f_k`.
        let upper_start = lower_end;
        let upper_end = self.lower_bound(centre + 1.0);
        let (weight_upper, weighted_f_upper) = self.range(upper_start, upper_end.max(upper_start));

        (1.0 - centre) * weight_lower + weighted_f_lower + (1.0 + centre) * weight_upper
            - weighted_f_upper
    }

    fn range(&self, start: usize, end: usize) -> (f64, f64) {
        if end <= start {
            return (0.0, 0.0);
        }
        (
            self.weight_prefix[end] - self.weight_prefix[start],
            self.weighted_f_prefix[end] - self.weighted_f_prefix[start],
        )
    }

    /// First index with `f > value`.
    fn upper_bound(&self, value: f64) -> usize {
        self.f.partition_point(|&candidate| candidate <= value)
    }

    /// First index with `f > value`, spelled apart from [`Self::upper_bound`]
    /// only for readability at the call site: the lower window ends *after*
    /// the observations equal to the centre, because `approxGrad(0) = 1`.
    fn upper_bound_inclusive(&self, value: f64) -> usize {
        self.f.partition_point(|&candidate| candidate <= value)
    }

    /// First index with `f >= value`.
    fn lower_bound(&self, value: f64) -> usize {
        self.f.partition_point(|&candidate| candidate < value)
    }

    /// Total weight of the observations strictly below `value`.
    fn weight_below(&self, value: f64) -> f64 {
        self.weight_prefix[self.lower_bound(value)]
    }
}

impl Family for Auc {
    fn class(&self, f: ArrayView1<f64>) -> Option<Vec<usize>> {
        Binomial.class(f)
    }

    fn loss_is_row_separable(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "AUC"
    }

    fn encode_response(&self, y: ArrayView1<f64>) -> Option<Array1<f64>> {
        if y.is_empty() {
            return Some(y.to_owned());
        }
        let min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if min == max {
            // Degenerate (single-class) response; check_y rejects this
            // before it reaches negative_gradient/risk in the normal fit
            // path. Map defensively rather than divide by a zero range.
            return Some(Array1::from_elem(y.len(), -1.0));
        }
        let mid = 0.5 * (min + max);
        Some(y.mapv(|yi| if yi <= mid { -1.0 } else { 1.0 }))
    }

    fn offset(&self, _y: ArrayView1<f64>, _w: ArrayView1<f64>) -> f64 {
        0.0
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, ws, os, _n) = s4!(y, f, w, out);
        let (ind1, ind0) = auc_partition(ys);
        let w_pos: f64 = ind1.iter().map(|&i| ws[i]).sum();
        let w_neg: f64 = ind0.iter().map(|&i| ws[i]).sum();
        if ind1.is_empty() || ind0.is_empty() || w_pos <= 0.0 || w_neg <= 0.0 {
            // R warns "response is constant" and returns risk 0; there is
            // no gradient direction to report either. The `w_pos`/`w_neg`
            // checks extend this to the weighted case: if every observation
            // in one class has weight 0, that class has no participating
            // members either, same degenerate outcome.
            for v in os.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        let fstd = auc_standardize(fs, ws);
        auc_ngradient_core(&ind1, &ind0, &fstd, ws, os);
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, _n) = s3!(y, f, w);
        let (ind1, ind0) = auc_partition(ys);
        let w_pos: f64 = ind1.iter().map(|&i| ws[i]).sum();
        let w_neg: f64 = ind0.iter().map(|&i| ws[i]).sum();
        if ind1.is_empty() || ind0.is_empty() || w_pos <= 0.0 || w_neg <= 0.0 {
            return 0.0;
        }
        let fstd = auc_standardize(fs, ws);
        // The concordant-pair weight, `sum_{j,k} w_j w_k 1{f_j > f_k}`, is a
        // rank statistic: sorting the negatives and taking prefix sums of
        // their weights turns the inner loop into one binary search. `>` is
        // strict, so a tie contributes nothing, which is `lower_bound` --
        // the first index whose score is *not* below `f_j`.
        let above = if ind0.len().saturating_mul(ind1.len()) >= AUC_RANKED_CROSSOVER {
            let negatives = SortedClass::new(&ind0, &fstd, ws);
            ind1.iter()
                .map(|&j| ws[j] * negatives.weight_below(fstd[j]))
                .sum()
        } else {
            let mut total = 0.0f64;
            for &j in &ind1 {
                for &k in &ind0 {
                    if fstd[j] - fstd[k] > 0.0 {
                        total += ws[j] * ws[k];
                    }
                }
            }
            total
        };
        1.0 - above / (w_pos * w_neg)
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        if y.is_empty() {
            return Err(MboostrsError::BadResponse(
                "AUC".to_string(),
                "response is empty".to_string(),
            ));
        }
        let mut values: Vec<f64> = y.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        if values.len() != 2 {
            return Err(MboostrsError::BadResponse(
                "AUC".to_string(),
                format!(
                    "response is not a two-level factor: expected exactly 2 distinct values, got {}",
                    values.len()
                ),
            ));
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(*self)
    }
}

// ─── CoxPH (Cox Proportional Hazards) ───────────────────────────────

/// Cox proportional hazards model with Breslow partial likelihood.
///
/// Response y is concatenated [time, event] with length 2n.
/// event = 1 means observed (uncensored), event = 0 means censored.
///
/// **Offset:** 0 (no intercept in Cox model)
///
/// **Risk:** negative log partial likelihood (Breslow):
///   -sum_{i: δ_i=1} [f_i - log(sum_{j∈R(t_i)} exp(f_j) * w_j)]
///   where R(t_i) = {j : t_j >= t_i}
///
/// **ngradient (Breslow score):**
/// score_i = δ_i - sum_{k: δ_k=1, t_i>=t_k} exp(f_i)*w_i / sum_{j∈R(t_k)} exp(f_j)*w_j
#[derive(Clone, Default)]
pub struct CoxPH {
    /// Ascending-time sort order, cached across boosting iterations: the time
    /// vector is fit-invariant, so the O(n log n) sort only needs to rerun
    /// when the response changes (e.g. a new CV fold). Validated by O(n)
    /// equality against the stored times — much cheaper than re-sorting.
    order_cache: Option<(Vec<f64>, Vec<usize>)>,
    /// Reusable scratch buffers for the `&mut self` hot path
    /// (`ngradient_and_risk` / `negative_gradient`). Avoids 4× O(n)
    /// allocation per iteration. Grown as needed; never shrunk.
    buf_ef: Vec<f64>,
    buf_wef: Vec<f64>,
    buf_suf: Vec<f64>,
    buf_c_at: Vec<f64>,
}

impl CoxPH {
    /// Construct a Cox proportional-hazards family with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    fn cached_order(&mut self, times: &[f64]) -> &[usize] {
        let valid = self
            .order_cache
            .as_ref()
            .is_some_and(|(t, _)| t.as_slice() == times);
        if !valid {
            self.order_cache = Some((times.to_vec(), cox_order(times)));
        }
        &self.order_cache.as_ref().unwrap().1
    }
}

/// Ascending-time permutation for the Breslow risk-set walk.
fn cox_order(time: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).expect("time must not be NaN"));
    order
}

/// O(n log n) Breslow gradient + risk via sort-once + suffix sums.
///
/// Replaces the naive O(n²) double loop. Mathematically identical:
/// - `denom(t) = Σ_{j: time_j ≥ t} w_j·exp(f_j)` is a suffix sum over
///   time-ascending order (tied times share the suffix start, matching the
///   original `>=` risk set exactly).
/// - The martingale residual `out_i = δ_i − w_i·exp(f_i)·Σ_{k: δ_k=1, t_k ≤ t_i} 1/denom_k`
///   uses a forward cumulative of `1/denom` over event groups; the constant
///   `exp(−max_f)` shift cancels in every `exp(f_i)/denom` ratio.
/// - `risk = Σ_{k: δ_k=1, w_k>0} (log_denom_k − f_k)` with
///   `log_denom_k = max_f + ln(denom_shifted_k)` — the same log-sum-exp the
///   naive version computes, just shared with the gradient's suffix sum.
///
/// Gradient counts every event (weights enter only via `w_i·exp(f_i)`); risk
/// is gated on `w_k>0`. This preserves the previous behaviour bit-for-bit
/// (verified against R to ~1e-15), including out-of-bag (w=0) handling in CV.
fn cox_compute(
    time: &[f64],
    event: &[f64],
    fs: &[f64],
    ws: &[f64],
    out: Option<&mut [f64]>,
    order: &[usize],
) -> f64 {
    let n = time.len();
    let mut ef = vec![0.0f64; n];
    let mut wef = vec![0.0f64; n];
    let mut suf = vec![0.0f64; n + 1];
    let mut c_at = vec![0.0f64; n];
    cox_compute_into(
        time, event, fs, ws, out, order, &mut ef, &mut wef, &mut suf, &mut c_at,
    )
}

/// Buffer-reusing core: identical math to `cox_compute`, but writes into
/// caller-owned scratch vectors (which are grown as needed but never shrunk).
fn cox_compute_into(
    time: &[f64],
    event: &[f64],
    fs: &[f64],
    ws: &[f64],
    out: Option<&mut [f64]>,
    order: &[usize],
    ef: &mut Vec<f64>,
    wef: &mut Vec<f64>,
    suf: &mut Vec<f64>,
    c_at: &mut Vec<f64>,
) -> f64 {
    let n = time.len();
    ef.resize(n, 0.0);
    wef.resize(n, 0.0);
    suf.resize(n + 1, 0.0);
    c_at.resize(n, 0.0);
    let ef = &mut ef[..];
    let wef = &mut wef[..];
    let suf = &mut suf[..];
    let c_at = &mut c_at[..];

    let max_f = fs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shift = if max_f.is_finite() { max_f } else { 0.0 };
    for i in 0..n {
        ef[i] = (fs[i] - shift).exp();
        wef[i] = ws[i] * ef[i];
    }
    for v in c_at.iter_mut() {
        *v = 0.0;
    }

    debug_assert_eq!(order.len(), n);

    for p in (0..n).rev() {
        suf[p] = suf[p + 1] + wef[order[p]];
    }

    let mut risk = 0.0f64;
    let mut c_acc = 0.0f64;
    let mut p = 0usize;
    while p < n {
        let t = time[order[p]];
        let start = p;
        let mut end = p;
        while end < n && time[order[end]] == t {
            end += 1;
        }
        let denom = suf[start];
        let mut dn = 0.0f64;
        for &i in &order[start..end] {
            if event[i] > 0.5 {
                dn += ws[i];
            }
        }
        if denom > 0.0 && dn > 0.0 {
            c_acc += dn / denom;
            let log_denom = shift + denom.ln();
            for &i in &order[start..end] {
                if event[i] > 0.5 && ws[i] > 0.0 {
                    risk += ws[i] * (log_denom - fs[i]);
                }
            }
        }
        for &i in &order[start..end] {
            c_at[i] = c_acc;
        }
        p = end;
    }

    if let Some(out) = out {
        for i in 0..n {
            out[i] = event[i] - ef[i] * c_at[i];
        }
    }
    risk
}

impl Family for CoxPH {
    fn loss_is_row_separable(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "CoxPH"
    }

    fn offset(&self, _y: ArrayView1<f64>, _w: ArrayView1<f64>) -> f64 {
        0.0
    }

    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        y.len() / 2
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(os.len(), n);
        let times = &ys[..n];
        let events = &ys[n..];
        let order = self.cached_order(times).to_vec();
        cox_compute_into(
            times,
            events,
            fs,
            ws,
            Some(os),
            &order,
            &mut self.buf_ef,
            &mut self.buf_wef,
            &mut self.buf_suf,
            &mut self.buf_c_at,
        );
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        let times = &ys[..n];
        let events = &ys[n..];
        let cached = self
            .order_cache
            .as_ref()
            .filter(|(t, _)| t.as_slice() == times)
            .map(|(_, o)| o.as_slice());
        match cached {
            Some(order) => cox_compute(times, events, fs, ws, None, order),
            None => cox_compute(times, events, fs, ws, None, &cox_order(times)),
        }
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(os.len(), n);
        let times = &ys[..n];
        let events = &ys[n..];
        let order = self.cached_order(times).to_vec();
        cox_compute_into(
            times,
            events,
            fs,
            ws,
            Some(os),
            &order,
            &mut self.buf_ef,
            &mut self.buf_wef,
            &mut self.buf_suf,
            &mut self.buf_c_at,
        )
    }

    fn supports_gram_update(&self) -> bool {
        false
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let ys = y.as_slice().expect("y must be contiguous");
        let len = ys.len();
        if len % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                "CoxPH".to_string(),
                format!("y must have even length (concatenated [time, event]), got {len}"),
            ));
        }
        let n = len / 2;
        for i in 0..n {
            // Survival times must be positive (like Weibull/Loglog); a
            // non-positive time would otherwise reach the risk-set sort with a
            // NaN/degenerate value.
            if ys[i] <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "CoxPH".to_string(),
                    format!("survival time must be > 0, got {}", ys[i]),
                ));
            }
            if ys[n + i] != 0.0 && ys[n + i] != 1.0 {
                return Err(MboostrsError::BadResponse(
                    "CoxPH".to_string(),
                    format!("event indicator must be 0 or 1, got {}", ys[n + i]),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(self.clone())
    }
}

// ─── Gehan (Gehan rank loss for AFT survival models) ────────────────

/// Gehan generalized-Wilcoxon rank loss for accelerated failure time (AFT)
/// survival models, matching R's `mboost::Gehan()`.
///
/// Response `y` is concatenated `[time, event]` with length `2n` (same
/// layout as [`CoxPH`]). Let `Z_i = ln(time_i)`, `Del_i = event_i`, and
/// `res_i = Z_i - f_i`.
///
/// **risk:** `(1/n) * sum_i loss_i` where
/// `loss_i = Del_i * sum_{j: res_j >= res_i} (res_j - res_i)`.
///
/// **negative_gradient:** for each `i`,
/// `arg2_i = #{j: Del_j = 1 and res_j <= res_i}`,
/// `arg1_i = Del_i * #{j: res_j >= res_i}`,
/// `ng_i = -(arg1_i - arg2_i) / n`.
/// Both counts are rank statistics over the residuals, so once
/// `n^2 >= GEHAN_RANKED_CROSSOVER` they are computed in `O(n log n)` by
/// [`gehan_ngradient_ranked`]/[`gehan_risk_ranked`] (one sort, prefix
/// sums) instead of the `O(n^2)` pairwise [`gehan_ngradient_core`]/
/// [`gehan_risk_core`], which stay as the reference below the crossover.
/// The ranked sums run over the same weights in a different order, so they
/// agree to rounding rather than bit-for-bit; the exact tie boundaries are
/// the same non-strict `>=`/`<=` R uses. This is O(n^2); both counts use R's
/// non-strict `>=`/`<=` boundaries, so
/// ties and the diagonal `j == i` are counted exactly as R counts them.
///
/// **offset:** the Gehan loss depends only on pairwise residual differences
/// `res_j - res_i`, so it is invariant to a constant shift of every `f_i` --
/// the risk is *flat* in a constant offset `c` (verified against R: the risk
/// is byte-identical at several test values of `c`). R's own `offset()`
/// calls `optimize()` on this flat objective and returns an arbitrary point
/// near the upper end of its search bracket (`max(log(time))`), not a
/// meaningful minimizer -- there is no minimizer, every point ties. mboostrs
/// instead returns `0.0`, a defined, principled choice for an unidentifiable
/// offset; this is an intentional deviation from R's arbitrary value (see
/// the LogLogLSS/WeibullLSS offset deviations in the capability manifest for
/// the same pattern). Everything downstream of the offset that boosting
/// actually needs -- the gradient, coefficient increments, `selected`, and
/// risk -- is unaffected by this choice and matches R exactly; only the
/// additive constant baked into `fitted` differs from R's (by the arbitrary
/// `optimize()` value R happened to return).
///
/// **Weights:** a principled mboostrs *extension*, not R parity. R's
/// `Gehan()` is `weights = "case"`, treating non-unit weights via an
/// integer `rep()` expansion with the same positional mismatch bug as
/// [`Auc`] (verified: R's weighted `ngradient` returns a vector of length
/// `sum(w)`, not `n`). Rather than reproduce that, `Gehan` implements a
/// directly-derived weighted gradient: each observation `i` participates
/// with weight `w_i` (`w_i == 0` gets gradient `0` and contributes nothing
/// to any other observation's pairwise sum), `W = sum(w)` replaces the
/// unweighted `n` normalizer, and every pairwise indicator sum
/// (`arg1`/`arg2`) picks up a `w_j` factor. This is exactly
/// `ng = -d(risk)/df` for the weighted risk (see `gehan_ngradient_core`,
/// `gehan_risk_core`); it reduces *exactly* -- bit-for-bit -- to the
/// unit-weight formulas above when `w ≡ 1`, since every weight factor is
/// then a multiplication by `1.0`. `w` must be non-negative and finite
/// (enforced generically by `fit`/`cvrisk`, not re-checked here); any
/// non-negative `w` is accepted, so bootstrap (integer-count) and
/// subsampling ({0,1}) `cvrisk`/`stabsel` folds both work, unlike R's.
#[derive(Clone, Copy)]
pub struct Gehan;

/// Core O(n^2) Gehan gradient computation, matching R's `tmpg.fun` closure
/// (see `scratchpad/ranking_families.R`) term-for-term, including its
/// non-strict `>=`/`<=` boundaries, generalized to case weights `ws`:
/// `arg2_i = sum_j(w_j * Del_j * 1{res_j <= res_i})`,
/// `arg1_i = Del_i * sum_j(w_j * 1{res_j >= res_i})`,
/// `ng_i = -w_i * (arg1_i - arg2_i) / W` with `W = sum(ws)`.
///
/// For `w ≡ 1` this is bit-identical to the previous unweighted
/// implementation: every `ws[i]` is `1.0` (an exact IEEE-754 no-op under
/// multiplication), `W` sums to exactly `n as f64`, and the loop
/// structure/order is otherwise unchanged.
pub fn gehan_ngradient_core(res: &[f64], del: &[f64], ws: &[f64], out: &mut [f64]) {
    let n = res.len();
    let w_sum: f64 = ws.iter().sum();
    for i in 0..n {
        let mut arg2 = 0.0f64;
        for j in 0..n {
            if del[j] > 0.5 && res[i] - res[j] >= 0.0 {
                arg2 += ws[j];
            }
        }
        let mut arg1 = 0.0f64;
        if del[i] > 0.5 {
            for j in 0..n {
                if res[i] - res[j] <= 0.0 {
                    arg1 += ws[j];
                }
            }
        }
        let raw = -(arg1 - arg2);
        out[i] = if w_sum > 0.0 {
            ws[i] * raw / w_sum
        } else {
            0.0
        };
    }
}

/// Core Gehan risk computation, matching R's `tmpl.fun`/`gehan.loss`/`risk`
/// chain (see `scratchpad/ranking_families.R`), generalized to case weights
/// `ws`: `loss_i = Del_i * sum_{j: res_j>=res_i}(w_j * (res_j - res_i))`,
/// `risk = sum(w_i * loss_i) / W` with `W = sum(ws)` -- the weighted mean
/// per-observation loss, zero for censored observations.
///
/// For `w ≡ 1` this is bit-identical to the previous unweighted
/// implementation, for the same reason as `gehan_ngradient_core` above.
pub fn gehan_risk_core(res: &[f64], del: &[f64], ws: &[f64]) -> f64 {
    let n = res.len();
    let w_sum: f64 = ws.iter().sum();
    if w_sum <= 0.0 {
        return 0.0;
    }
    let mut total = 0.0f64;
    for i in 0..n {
        if del[i] > 0.5 {
            let mut loss_i = 0.0f64;
            for j in 0..n {
                if res[i] - res[j] <= 0.0 {
                    loss_i += ws[j] * (res[j] - res[i]);
                }
            }
            total += ws[i] * loss_i;
        }
    }
    total / w_sum
}

/// Number of residual pairs above which `Gehan` uses the ranked
/// `O(n log n)` kernels rather than the pairwise `O(n^2)` reference.
///
/// The pairwise gradient evaluates about `2 n` comparisons per observation,
/// so the sensible crossover is on `n^2`, the same pair count
/// `AUC_RANKED_CROSSOVER` keys on: below `n = 64` the sort and prefix
/// arrays in [`gehan_ngradient_ranked`] cost more than the double loop, and
/// above it the ranked forms dominate. Chosen to match the AUC crossover
/// rather than re-measured, because the two kernels have the same shape of
/// cost (quadratic to linearithmic) and the AUC number already stands.
pub const GEHAN_RANKED_CROSSOVER: usize = 4096;

/// Ranked `O(n log n)` re-write of [`gehan_ngradient_core`] (P3.4 item 5).
///
/// Both quantities are rank statistics over the residuals:
///
/// * `arg2_i = sum_j w_j Del_j 1{res_j <= res_i}` -- a weighted count of
///   earlier-or-equal *events*, i.e. the prefix sum of `w * Del` through the
///   end of `res_i`'s tie block (the boundary is non-strict, `<=`);
/// * `arg1_i = Del_i sum_j w_j 1{res_j >= res_i}` -- the weighted count of
///   *all* observations at-or-after `res_i`, i.e. a suffix sum of `w` from
///   the start of `res_i`'s tie block (non-strict again, so the diagonal
///   and every tie count, exactly as the pairwise `>=`/`<=` boundaries do).
///
/// One sort and two prefix passes replace the two nested loops. The sums
/// run over the same weights in a different order, so the result agrees with
/// [`gehan_ngradient_core`] to rounding rather than bit-for-bit; at unit
/// weights both are exactly the integer counts, since every `1.0` sums
/// exactly regardless of order.
pub fn gehan_ngradient_ranked(res: &[f64], del: &[f64], ws: &[f64], out: &mut [f64]) {
    let n = res.len();
    let w_sum: f64 = ws.iter().sum();
    if w_sum <= 0.0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        res[a]
            .partial_cmp(&res[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut w_prefix = vec![0.0f64; n + 1];
    let mut weighted_event_prefix = vec![0.0f64; n + 1];
    for (k, &idx) in order.iter().enumerate() {
        w_prefix[k + 1] = w_prefix[k] + ws[idx];
        weighted_event_prefix[k + 1] = weighted_event_prefix[k] + ws[idx] * del[idx];
    }
    let mut block_start = 0usize;
    while block_start < n {
        let r = res[order[block_start]];
        let mut block_end = block_start + 1;
        while block_end < n && res[order[block_end]] == r {
            block_end += 1;
        }
        // `arg2` counts events with residual at-or-below `res_i`, so every
        // observation in the block takes the prefix through `block_end`.
        let arg2 = weighted_event_prefix[block_end];
        // `arg1` counts all observations at-or-above `res_i`, so events in
        // the block take the suffix from `block_start` (the diagonal and
        // every tie are inside it).
        let arg1_suffix = w_prefix[n] - w_prefix[block_start];
        for &idx in &order[block_start..block_end] {
            // `ng_i = -w_i (arg1_i - arg2_i) / W = w_i (arg2_i - arg1_i) / W`
            // with `arg1_i == 0` for censored observations.
            let raw = if del[idx] > 0.5 {
                arg2 - arg1_suffix
            } else {
                arg2
            };
            out[idx] = ws[idx] * raw / w_sum;
        }
        block_start = block_end;
    }
}

/// Ranked `O(n log n)` re-write of [`gehan_risk_core`] (P3.4 item 5).
///
/// `loss_i = sum_{j: res_j >= res_i} w_j (res_j - res_i)` is the suffix of
/// a sorted residual column, so two more prefix arrays (`w`, `w * res`)
/// turn the inner loop into two suffixes: `(sum w_j res_j) - res_i (sum w_j)`
/// over the block's suffix. The `>=` boundary is non-strict, and the block
/// starts at `res_i`'s own tie block, so ties and the diagonal are counted
/// exactly as the pairwise form counts them. Rounding differs from
/// [`gehan_risk_core`] because the subtraction `res_j - res_i` is moved
/// outside the per-`j` sum.
pub fn gehan_risk_ranked(res: &[f64], del: &[f64], ws: &[f64]) -> f64 {
    let n = res.len();
    let w_sum: f64 = ws.iter().sum();
    if w_sum <= 0.0 {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        res[a]
            .partial_cmp(&res[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut w_prefix = vec![0.0f64; n + 1];
    let mut weighted_res_prefix = vec![0.0f64; n + 1];
    for (k, &idx) in order.iter().enumerate() {
        w_prefix[k + 1] = w_prefix[k] + ws[idx];
        weighted_res_prefix[k + 1] = weighted_res_prefix[k] + ws[idx] * res[idx];
    }
    let mut total = 0.0f64;
    let mut block_start = 0usize;
    while block_start < n {
        let r = res[order[block_start]];
        let mut block_end = block_start + 1;
        while block_end < n && res[order[block_end]] == r {
            block_end += 1;
        }
        let sum_w = w_prefix[n] - w_prefix[block_start];
        let sum_wr = weighted_res_prefix[n] - weighted_res_prefix[block_start];
        for &idx in &order[block_start..block_end] {
            if del[idx] > 0.5 {
                total += ws[idx] * (sum_wr - r * sum_w);
            }
        }
        block_start = block_end;
    }
    total / w_sum
}

impl Family for Gehan {
    fn loss_is_row_separable(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "Gehan"
    }

    fn offset(&self, _y: ArrayView1<f64>, _w: ArrayView1<f64>) -> f64 {
        0.0
    }

    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        y.len() / 2
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(ws.len(), n);
        assert_eq!(os.len(), n);
        let times = &ys[..n];
        let del = &ys[n..];
        let res: Vec<f64> = (0..n).map(|i| times[i].ln() - fs[i]).collect();
        if n * n >= GEHAN_RANKED_CROSSOVER {
            gehan_ngradient_ranked(&res, del, ws, os);
        } else {
            gehan_ngradient_core(&res, del, ws, os);
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(ws.len(), n);
        let times = &ys[..n];
        let del = &ys[n..];
        let res: Vec<f64> = (0..n).map(|i| times[i].ln() - fs[i]).collect();
        if n * n >= GEHAN_RANKED_CROSSOVER {
            gehan_risk_ranked(&res, del, ws)
        } else {
            gehan_risk_core(&res, del, ws)
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let ys = y.as_slice().expect("y must be contiguous");
        let len = ys.len();
        if len % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                "Gehan".to_string(),
                format!("y must have even length (concatenated [time, event]), got {len}"),
            ));
        }
        let n = len / 2;
        for i in 0..n {
            if !ys[i].is_finite() || ys[i] <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Gehan".to_string(),
                    format!("survival time must be finite and > 0, got {}", ys[i]),
                ));
            }
            if ys[n + i] != 0.0 && ys[n + i] != 1.0 {
                return Err(MboostrsError::BadResponse(
                    "Gehan".to_string(),
                    format!("event indicator must be 0 or 1, got {}", ys[n + i]),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(*self)
    }
}

// ─── Cindex (Uno's IPCW-weighted concordance, mboost::Cindex/UnoC) ──

/// Numerically stable logistic sigmoid: `1/(1+exp(-z))` for `z >= 0`,
/// `exp(z)/(1+exp(z))` for `z < 0`. Both branches only ever call `exp` on a
/// non-positive argument, so this never overflows regardless of `|z|` --
/// needed because `Cindex`'s default `sigma = 0.1` makes `z = x/sigma` swing
/// large even for modest fitted-value differences `x`.
#[inline]
fn stable_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `approxGrad` from R's `Cindex()` closure: `exp(z)/(sigma*(1+exp(z))^2)`
/// with `z = x/sigma`, rewritten as `sigmoid(z)*(1-sigmoid(z))/sigma` (the
/// logistic density) via [`stable_sigmoid`] to avoid overflowing `exp(z)`.
#[inline]
fn cindex_approx_grad(x: f64, sigma: f64) -> f64 {
    let s = stable_sigmoid(x / sigma);
    s * (1.0 - s) / sigma
}

/// `approxLoss` from R's `Cindex()` closure: `1/(1+exp(z))` with `z =
/// x/sigma`, rewritten as `1 - sigmoid(z)` via [`stable_sigmoid`] to avoid
/// overflowing `exp(z)`.
#[inline]
fn cindex_approx_loss(x: f64, sigma: f64) -> f64 {
    1.0 - stable_sigmoid(x / sigma)
}

/// Reverse-Kaplan-Meier estimate of the *censoring* distribution, evaluated
/// at each subject's own observed time. Mirrors
/// `python/mboostrs/diagnostics.py::_reverse_km_survival` (verified against
/// R's `survival::survfit` to 1e-15): standard right-continuous KM with
/// `flip_event = 1 - status` in place of the event indicator, so a
/// same-time censored subject stays in the risk set for that step (R's
/// `survfit` tie convention). `flip_event` and `time` must have equal
/// length; `time` must not contain NaN.
fn reverse_km_ghat(time: &[f64], flip_event: &[f64]) -> Vec<f64> {
    let n = time.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).expect("time must not be NaN"));
    let mut ghat = vec![0.0f64; n];
    let mut s = 1.0f64;
    let mut i = 0usize;
    while i < n {
        let t = time[order[i]];
        let mut j = i;
        let mut deaths = 0.0f64;
        while j < n && time[order[j]] == t {
            if flip_event[order[j]] > 0.5 {
                deaths += 1.0;
            }
            j += 1;
        }
        let at_risk = (n - i) as f64;
        if deaths > 0.0 {
            s *= 1.0 - deaths / at_risk;
        }
        for &idx in &order[i..j] {
            ghat[idx] = s;
        }
        i = j;
    }
    ghat
}

/// Inverse probability of censoring weights, matching
/// `python/mboostrs/diagnostics.py::ipc_weights` / R's
/// `mboost::IPCweights`: reverse-KM of the censoring distribution
/// (event/censoring flipped) evaluated at each subject's own time, `status /
/// Ghat` clipped to `maxweight`, with `Ghat` forced to `1` for censored
/// subjects (so their weight is exactly `0` via the `status` numerator) and
/// event subjects' weight clipped to `maxweight` (default 5). `time` and
/// `status` must have equal length; `status` must be `0`/`1`.
fn ipc_weights_core(time: &[f64], status: &[f64], maxweight: f64) -> Vec<f64> {
    let n = time.len();
    let flip_event: Vec<f64> = status.iter().map(|&s| 1.0 - s).collect();
    let mut ghat = reverse_km_ghat(time, &flip_event);
    for i in 0..n {
        if status[i] == 0.0 {
            ghat[i] = 1.0;
        }
    }
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        w[i] = status[i] / ghat[i];
        if w[i] > maxweight {
            w[i] = maxweight;
        }
    }
    w
}

/// Full-length IPCW vector for `Cindex::compute_weights`, matching R's
/// `ipcw_wow <- IPCweights(y[w != 0, ]); ipcw <- numeric(nrow(y));
/// ipcw[w != 0] <- ipcw_wow`: the censoring-distribution KM is fit on the
/// case-weight-positive (`w != 0`) subset only, then scattered back into a
/// full-length vector with `0` at the out-of-bag (`w == 0`) positions.
fn cindex_full_ipcw(time: &[f64], status: &[f64], w: &[f64]) -> Vec<f64> {
    let n = time.len();
    let idx: Vec<usize> = (0..n).filter(|&i| w[i] != 0.0).collect();
    let sub_time: Vec<f64> = idx.iter().map(|&i| time[i]).collect();
    let sub_status: Vec<f64> = idx.iter().map(|&i| status[i]).collect();
    let sub_ipcw = ipc_weights_core(&sub_time, &sub_status, 5.0);
    let mut ipcw = vec![0.0f64; n];
    for (k, &i) in idx.iter().enumerate() {
        ipcw[i] = sub_ipcw[k];
    }
    ipcw
}

/// Shared O(n^2) core for `Cindex::negative_gradient`/`risk`: builds the
/// (implicit, never fully materialized) IPCW-weighted concordance matrix `W`
/// from R's `compute_weights` and folds it directly into the gradient and
/// risk accumulators, matching `ranking_families.R`'s `Cindex()` term for
/// term. `W[i,j] = ipcw[i]^2 * weightsI[i,j] * w[i] * w[j]`, where
/// `weightsI[i,j]` is `1` if `time[i] < time[j]`, `0.5` if `time[i] ==
/// time[j]` (`i != j`), else `0` (diagonal `0`), normalized so the full
/// matrix sums to `1`. Because gradient and risk are both linear in `W`,
/// the normalization is applied once at the end (dividing the raw
/// accumulators by the raw matrix sum) rather than per pair.
///
/// `etaMat[i,j] = f[i] - f[j]`; `M1[i,j] = approxGrad(etaMat[i,j]) *
/// W[i,j]`; `ng[k] = colSums(M1)[k] - rowSums(M1)[k]`, i.e. pair `(i,j)`
/// contributes `+M1[i,j]` to `ng[j]` (colSums) and `-M1[i,j]` to `ng[i]`
/// (rowSums). `risk = -sum(approxLoss(etaMat) * W)`.
fn cindex_ngradient_and_risk(
    time: &[f64],
    status: &[f64],
    w: &[f64],
    f: &[f64],
    sigma: f64,
    ng_out: Option<&mut [f64]>,
) -> f64 {
    let n = time.len();
    let ipcw = cindex_full_ipcw(time, status, w);
    let mut grad_raw = vec![0.0f64; n];
    let mut risk_raw = 0.0f64;
    let mut sum_w_raw = 0.0f64;
    for i in 0..n {
        let ipcw_i2 = ipcw[i] * ipcw[i];
        let wi = w[i];
        if ipcw_i2 == 0.0 || wi == 0.0 {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            let weights_i = if time[i] < time[j] {
                1.0
            } else if time[i] == time[j] {
                0.5
            } else {
                0.0
            };
            if weights_i == 0.0 {
                continue;
            }
            let wij = ipcw_i2 * weights_i * wi * w[j];
            if wij == 0.0 {
                continue;
            }
            sum_w_raw += wij;
            let eta_ij = f[i] - f[j];
            risk_raw += cindex_approx_loss(eta_ij, sigma) * wij;
            let m1 = cindex_approx_grad(eta_ij, sigma) * wij;
            grad_raw[j] += m1;
            grad_raw[i] -= m1;
        }
    }
    if sum_w_raw > 0.0 {
        if let Some(out) = ng_out {
            for k in 0..n {
                out[k] = grad_raw[k] / sum_w_raw;
            }
        }
        -risk_raw / sum_w_raw
    } else {
        if let Some(out) = ng_out {
            for v in out.iter_mut() {
                *v = 0.0;
            }
        }
        0.0
    }
}

/// Uno's IPCW-weighted concordance ("C-index") loss for survival models,
/// matching R's `mboost::Cindex(sigma, ipcw = 1)` (the case with the default
/// internally-estimated `ipcw`; mboostrs does not expose R's rarely-used
/// externally-supplied `ipcw` argument).
///
/// Response `y` is concatenated `[time, event]` with length `2n` (same
/// layout as [`Gehan`]/[`CoxPH`]).
///
/// **The core novelty vs. [`Gehan`]/[`Auc`]** is the `n x n` IPCW-weighted
/// concordance weight matrix `W` (see `cindex_ngradient_and_risk`):
/// `W[i,j] = ipcw[i]^2 * weightsI[i,j] * w[i] * w[j]`, normalized to sum to
/// `1`, where `weightsI[i,j]` is `1`/`0.5`/`0` according to whether
/// `time[i]` is less than / equal to / greater than `time[j]` (`0` on the
/// diagonal), and `ipcw` is the inverse-probability-of-censoring weight
/// (see `ipc_weights_core`, a Rust port of
/// `python/mboostrs/diagnostics.py::ipc_weights`, itself verified against
/// R's `mboost::IPCweights` to 1e-15) fit on the case-weight-positive subset
/// and scattered back with `0` at out-of-bag positions.
///
/// **negative_gradient/risk:** with `etaMat[i,j] = f[i] - f[j]`,
/// `approxGrad(x) = sigmoid(x/sigma)*(1-sigmoid(x/sigma))/sigma`,
/// `approxLoss(x) = 1 - sigmoid(x/sigma)` (both rewritten from R's raw
/// `exp(x/sigma)` form via a numerically stable sigmoid -- `sigma = 0.1`
/// makes `x/sigma` swing large enough to overflow a direct `exp` -- see
/// `stable_sigmoid`):
/// `ng[k] = sum_i (approxGrad(etaMat[i,k]) - approxGrad(etaMat[k,i])) *
/// (their respective W entries)` (colSums − rowSums of `approxGrad(etaMat) *
/// W`), `risk = -sum(approxLoss(etaMat) * W)`.
///
/// **offset:** `0`, identical to R (`Cindex()@offset` is the constant
/// function `0`; the risk is offset-invariant regardless, since `etaMat`
/// only ever sees differences `f[i] - f[j]`). Unlike [`Gehan`], this is
/// *not* an intentional deviation -- mboostrs' raw fitted values match R's
/// directly, no centering needed.
///
/// **Weights:** R requires `w` in `{0, 1}` (`stop()`s otherwise) and uses
/// `w == 0` positions meaningfully (excluded from the IPCW fit and zeroed
/// out of `W`). `Auc` and `Gehan` take any non-negative weight through
/// their weighted pairwise gradients; `Cindex` does not, because the IPCW
/// weights it carries are already a per-observation quantity and a second
/// count on top of them has no definition here.
/// `Cindex` reports [`Family::requires_binary_weights`]
/// (not [`Family::requires_unit_weights`]), so `fit`/`cvrisk` reject any
/// weight value outside `{0, 1}` with a clear error, but subsampling-style
/// `cvrisk`/`stabsel` folds (which are exactly `{0, 1}`) are supported;
/// only bootstrap-style integer-count folds are rejected.
#[derive(Clone, Copy, Debug)]
pub struct Cindex {
    /// Smoothing scale of the sigmoid concordance kernel.
    pub sigma: f64,
}

impl Cindex {
    /// Unchecked. Prefer [`Cindex::try_new`]: a non-positive `sigma` inverts
    /// the smoothing sigmoid, so the loss rewards discordant pairs.
    pub fn new(sigma: f64) -> Self {
        Cindex { sigma }
    }

    /// `sigma` positive and finite.
    pub fn try_new(sigma: f64) -> Result<Self, MboostrsError> {
        check_positive("Cindex", "sigma", sigma)?;
        Ok(Cindex { sigma })
    }
}

impl Default for Cindex {
    fn default() -> Self {
        Cindex { sigma: 0.1 }
    }
}

impl Family for Cindex {
    fn loss_is_row_separable(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "Cindex"
    }

    fn offset(&self, _y: ArrayView1<f64>, _w: ArrayView1<f64>) -> f64 {
        0.0
    }

    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        y.len() / 2
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(ws.len(), n);
        assert_eq!(os.len(), n);
        let times = &ys[..n];
        let status = &ys[n..];
        cindex_ngradient_and_risk(times, status, ws, fs, self.sigma, Some(os));
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(ws.len(), n);
        let times = &ys[..n];
        let status = &ys[n..];
        cindex_ngradient_and_risk(times, status, ws, fs, self.sigma, None)
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let ys = y.as_slice().expect("y must be contiguous");
        let len = ys.len();
        if len % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                "Cindex".to_string(),
                format!("y must have even length (concatenated [time, event]), got {len}"),
            ));
        }
        let n = len / 2;
        for i in 0..n {
            if !ys[i].is_finite() || ys[i] <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Cindex".to_string(),
                    format!("survival time must be finite and > 0, got {}", ys[i]),
                ));
            }
            if ys[n + i] != 0.0 && ys[n + i] != 1.0 {
                return Err(MboostrsError::BadResponse(
                    "Cindex".to_string(),
                    format!("event indicator must be 0 or 1, got {}", ys[n + i]),
                ));
            }
        }
        Ok(())
    }

    fn requires_binary_weights(&self) -> bool {
        true
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(*self)
    }
}

// ─── Weibull (Weibull AFT) ────────────────────────────────────────────

/// Weibull accelerated failure time model with scale (sigma) nuisance.
///
/// Response y is concatenated [time, event] with length 2n.
/// **f = log(scale)** parameterization matching R's mboost.
///
/// **Risk:** sum(w * plloss(sigma, y, f)) where, with eta = (log(time)-f)/sigma,
///   plloss = event*(exp(eta) - eta + log(sigma)) + (1 - event)*exp(eta)
///
/// **ngradient:** (event * (exp(eta) - 1) + (1 - event) * exp(eta)) / sigma
///
/// **Calibrate:** optimize riskS over sigma in nuirange (default [0, 100])
#[derive(Clone)]
pub struct Weibull {
    /// Scale parameter of the accelerated-failure-time model.
    pub sigma: f64,
    nuirange: (f64, f64),
}

impl Weibull {
    /// Construct a Weibull survival family with the default `sigma = 1.0`.
    pub fn new() -> Self {
        Weibull {
            sigma: 1.0,
            // R's Weibull nuirange is [0, 100]; matching the exact interval keeps
            // Brent's interior-probe sequence identical to R's optimize() so the
            // calibrated sigma agrees (a [0.01, 100] bracket diverged slightly,
            // visible as a ~1e-2 per-fold cvrisk difference).
            nuirange: (0.0, 100.0),
        }
    }

    fn plloss(sigma: f64, time: f64, event: f64, f_val: f64) -> f64 {
        let eta = (time.ln() - f_val) / sigma;
        let exp_eta = eta.exp();
        // Weibull AFT negative log-likelihood (matches R's Weibull plloss):
        //   event*(exp(eta) - eta + ln(sigma)) + (1-event)*exp(eta)
        event * (exp_eta - eta + sigma.ln()) + (1.0 - event) * exp_eta
    }

    fn risk_s(sigma: f64, times: &[f64], events: &[f64], fs: &[f64], ws: &[f64]) -> f64 {
        let mut s = 0.0f64;
        for i in 0..times.len() {
            s += ws[i] * Self::plloss(sigma, times[i], events[i], fs[i]);
        }
        s
    }
}

impl Default for Weibull {
    fn default() -> Self {
        Self::new()
    }
}

impl Family for Weibull {
    fn risk_rise_threshold(&self) -> Option<f64> {
        // Nuisance calibration minimizes the same likelihood along the path.
        Some(0.01)
    }

    fn name(&self) -> &'static str {
        "Weibull"
    }

    fn nuisance(&self) -> Option<f64> {
        Some(self.sigma)
    }

    fn set_nuisance(&mut self, value: f64) {
        self.sigma = value;
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        let times = &ys[..n];
        let events = &ys[n..];
        let max_log_time = times
            .iter()
            .zip(ws.iter())
            .filter(|(_, &w)| w > 0.0)
            .map(|(t, _)| t.ln())
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log_time.is_finite() || max_log_time <= 0.0 {
            return 0.0;
        }
        let times_owned = times.to_vec();
        let events_owned = events.to_vec();
        let ws_owned = ws.to_vec();
        let offset_fn = |f: f64| -> f64 {
            let mut s = 0.0f64;
            for i in 0..n {
                s += ws_owned[i] * Self::plloss(1.0, times_owned[i], events_owned[i], f);
            }
            s
        };
        // R: optimize(risk, interval = c(0, max(log(time)))) -- lower bound is 0,
        // not 0.01 (a nonzero lower bound shifts the minimum by ~1e-5).
        minimize_brent(&offset_fn, 0.0, max_log_time, R_OPTIMIZE_TOL, 500)
    }

    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        y.len() / 2
    }

    fn calibrates(&self) -> bool {
        true
    }

    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        let times = ys[..n].to_vec();
        let events = ys[n..].to_vec();
        let fs_owned = fs.to_vec();
        let ws_owned = ws.to_vec();

        let risk_fn = |s: f64| Self::risk_s(s, &times, &events, &fs_owned, &ws_owned);
        let lo = self.nuirange.0;
        let hi = self.nuirange.1;
        let sigma_opt = minimize_brent(&risk_fn, lo, hi, R_OPTIMIZE_TOL, 500);
        if sigma_opt > 0.0 && sigma_opt.is_finite() {
            self.sigma = sigma_opt;
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let _ws = w.as_slice().expect("w must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(os.len(), n);
        let times = &ys[..n];
        let events = &ys[n..];
        let sigma = self.sigma;
        for i in 0..n {
            let eta = (times[i].ln() - fs[i]) / sigma;
            let exp_eta = eta.exp();
            os[i] = (events[i] * (exp_eta - 1.0) + (1.0 - events[i]) * exp_eta) / sigma;
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        let times = &ys[..n];
        let events = &ys[n..];
        let sigma = self.sigma;
        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * Self::plloss(sigma, times[i], events[i], fs[i]);
        }
        risk
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        self.negative_gradient(y, f, w, out.view_mut());
        self.risk(y, f, w)
    }

    fn supports_gram_update(&self) -> bool {
        false
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let ys = y.as_slice().expect("y must be contiguous");
        let len = ys.len();
        if len % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                "Weibull".to_string(),
                format!("y must have even length (concatenated [time, event]), got {len}"),
            ));
        }
        let n = len / 2;
        for i in 0..n {
            if ys[i] <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Weibull".to_string(),
                    format!("survival times must be > 0, got {}", ys[i]),
                ));
            }
            if ys[n + i] != 0.0 && ys[n + i] != 1.0 {
                return Err(MboostrsError::BadResponse(
                    "Weibull".to_string(),
                    format!("event indicator must be 0 or 1, got {}", ys[n + i]),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(Weibull {
            sigma: self.sigma,
            nuirange: self.nuirange,
        })
    }
}

// ─── Loglog (Log-Logistic AFT) ─────────────────────────────────────

/// Log-logistic accelerated failure time model with scale (sigma) nuisance.
///
/// Response y is concatenated [time, event] with length 2n.
///
/// **eta_i = (log(time_i) - f_i) / sigma**
/// **nom_i = exp(-eta_i) + 1**
///
/// **ngradient_i:** (-∂risk/∂f)
///   [event_i * (2/nom_i - 1) - (1 - event_i) / nom_i] / sigma
///
/// **Risk:** sum(w * plloss(sigma, y, f)) where
///   plloss = event * (eta + sigma.ln() + 2*nom.ln())
///          + (1 - event) * (eta + nom.ln())
///   eta = (ln(time) - f) / sigma,  nom = exp(-eta) + 1
///
/// **Offset:** optimize(risk, interval=c(0, max(log(time))))
///
/// **Calibrate:** optimize riskS over sigma in nuirange (default [0, 100])
#[derive(Clone)]
pub struct Loglog {
    /// Scale parameter of the accelerated-failure-time model.
    pub sigma: f64,
    nuirange: (f64, f64),
}

impl Default for Loglog {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            // R's Loglog nuirange is [0, 100]; match it exactly (see Weibull).
            nuirange: (0.0, 100.0),
        }
    }
}

impl Loglog {
    /// Construct a log-logistic survival family with the default
    /// `sigma = 1.0`.
    pub fn new() -> Self {
        Self::default()
    }

    fn plloss(sigma: f64, time: f64, event: f64, f_val: f64) -> f64 {
        let eta = (time.ln() - f_val) / sigma;
        let nom = (-eta).exp() + 1.0;
        event * (eta + 2.0 * nom.ln() + sigma.ln()) + (1.0 - event) * (eta + nom.ln())
    }

    fn risk_s(sigma: f64, times: &[f64], events: &[f64], fs: &[f64], ws: &[f64]) -> f64 {
        let n = times.len();
        let mut s = 0.0f64;
        for i in 0..n {
            s += ws[i] * Self::plloss(sigma, times[i], events[i], fs[i]);
        }
        s
    }
}

impl Family for Loglog {
    fn risk_rise_threshold(&self) -> Option<f64> {
        // Nuisance calibration minimizes the same likelihood along the path.
        Some(0.01)
    }

    fn name(&self) -> &'static str {
        "Loglog"
    }

    fn nuisance(&self) -> Option<f64> {
        Some(self.sigma)
    }

    fn set_nuisance(&mut self, value: f64) {
        self.sigma = value;
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        let times = &ys[..n];
        let events = &ys[n..];

        // Match Weibull::offset: restrict to positive-weight observations and
        // start the fold from -inf, so the upper Brent bound is a real data
        // value. Folding from 0.0 would (when every log-time is < 0, i.e. all
        // survival times < 1) return 0.0 — a value no observation has — and a
        // degenerate bracket [0, 0].
        let max_log_time = times
            .iter()
            .zip(ws.iter())
            .filter(|(_, &w)| w > 0.0)
            .map(|(t, _)| t.ln())
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log_time.is_finite() || max_log_time <= 0.0 {
            return 0.0;
        }

        let times_owned = times.to_vec();
        let events_owned = events.to_vec();
        let ws_owned = ws.to_vec();
        let sigma = self.sigma;

        let risk_fn = |f_val: f64| -> f64 {
            let mut s = 0.0f64;
            for i in 0..times_owned.len() {
                s += ws_owned[i] * Self::plloss(sigma, times_owned[i], events_owned[i], f_val);
            }
            s
        };

        minimize_brent(&risk_fn, 0.0, max_log_time, R_OPTIMIZE_TOL, 500)
    }

    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        y.len() / 2
    }

    fn calibrates(&self) -> bool {
        true
    }

    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        let times = ys[..n].to_vec();
        let events = ys[n..].to_vec();
        let fs_owned = fs.to_vec();
        let ws_owned = ws.to_vec();

        let risk_fn = |sigma: f64| Self::risk_s(sigma, &times, &events, &fs_owned, &ws_owned);
        let sigma_opt = minimize_brent(
            &risk_fn,
            self.nuirange.0,
            self.nuirange.1,
            R_OPTIMIZE_TOL,
            500,
        );
        if sigma_opt > 0.0 && sigma_opt.is_finite() {
            self.sigma = sigma_opt;
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let _ws = w.as_slice().expect("w must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(os.len(), n);

        let times = &ys[..n];
        let events = &ys[n..];
        let sigma = self.sigma;

        for i in 0..n {
            let eta = (times[i].ln() - fs[i]) / sigma;
            let nom = (-eta).exp() + 1.0;
            os[i] = (events[i] * (2.0 / nom - 1.0) + (1.0 - events[i]) / nom) / sigma;
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);

        let times = &ys[..n];
        let events = &ys[n..];
        let sigma = self.sigma;

        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * Self::plloss(sigma, times[i], events[i], fs[i]);
        }
        risk
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        self.negative_gradient(y, f, w, out.view_mut());
        self.risk(y, f, w)
    }

    fn supports_gram_update(&self) -> bool {
        false
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let ys = y.as_slice().expect("y must be contiguous");
        let len = ys.len();
        if len % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                "Loglog".to_string(),
                format!("y must have even length (concatenated [time, event]), got {len}"),
            ));
        }
        let n = len / 2;
        for i in 0..n {
            if ys[i] <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Loglog".to_string(),
                    format!("survival times must be > 0, got {}", ys[i]),
                ));
            }
            if ys[n + i] != 0.0 && ys[n + i] != 1.0 {
                return Err(MboostrsError::BadResponse(
                    "Loglog".to_string(),
                    format!("event indicator must be 0 or 1, got {}", ys[n + i]),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(Loglog {
            sigma: self.sigma,
            nuirange: self.nuirange,
        })
    }
}

// ─── Lognormal (Log-Normal AFT) ────────────────────────────────────

/// Log-normal accelerated failure time model with scale (sigma) nuisance.
///
/// Response y is concatenated [time, event] with length 2n.
///
/// **eta_i = (log(time_i) - f_i) / sigma**
///
/// **plloss(sigma, time, event, f):**
///   event * (eta^2/2 + log(sigma)) + (1-event) * (-eta + log(sigma) + log(Phi(-eta)))
/// where Phi is the standard normal CDF.
///
/// **ngradient_i:** -∂risk/∂f_i
///   event * eta / sigma + (1-event) / sigma * (phi(-eta) / Phi(-eta))
///
/// **Offset:** optimize(risk, interval=c(0, max(log(time))))
///
/// **Calibrate:** optimize risk over sigma in nuirange (default [0.01, 100])
#[derive(Clone)]
pub struct Lognormal {
    /// Scale parameter of the accelerated-failure-time model.
    pub sigma: f64,
    nuirange: (f64, f64),
}

impl Default for Lognormal {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            nuirange: (0.0, 100.0),
        }
    }
}

impl Lognormal {
    /// Construct a log-normal survival family with the default `sigma = 1.0`.
    pub fn new() -> Self {
        Self::default()
    }

    fn phi(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    fn plloss(sigma: f64, time: f64, event: f64, f_val: f64) -> f64 {
        let eta = (time.ln() - f_val) / sigma;
        if event > 0.5 {
            // -log(dnorm(eta, 0, sigma)) = 0.5*eta^2 + log(sigma) + 0.5*log(2*pi)
            0.5 * eta * eta + sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln()
        } else {
            // -log(S(t)) = -log(1 - pnorm(eta)) = -log(Phi(-eta))
            // Phi(x) = 0.5 * erfc(-x/sqrt(2))
            // Phi(-eta) = 0.5 * erfc(eta/sqrt(2))
            let phi_neg = 0.5 * libm::erfc(eta / std::f64::consts::SQRT_2);
            if phi_neg > 0.0 {
                -phi_neg.ln()
            } else {
                1e30
            }
        }
    }

    fn risk_s(sigma: f64, times: &[f64], events: &[f64], fs: &[f64], ws: &[f64]) -> f64 {
        let mut s = 0.0f64;
        for i in 0..times.len() {
            s += ws[i] * Self::plloss(sigma, times[i], events[i], fs[i]);
        }
        s
    }
}

impl Family for Lognormal {
    fn risk_rise_threshold(&self) -> Option<f64> {
        // Nuisance calibration minimizes the same likelihood along the path.
        Some(0.01)
    }

    fn name(&self) -> &'static str {
        "Lognormal"
    }

    fn nuisance(&self) -> Option<f64> {
        Some(self.sigma)
    }

    fn set_nuisance(&mut self, value: f64) {
        self.sigma = value;
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        let times = &ys[..n];
        let events = &ys[n..];
        let max_log_time = times
            .iter()
            .zip(ws.iter())
            .filter(|(_, &w)| w > 0.0)
            .map(|(t, _)| t.ln())
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log_time.is_finite() || max_log_time <= 0.0 {
            return 0.0;
        }
        let times_owned = times.to_vec();
        let events_owned = events.to_vec();
        let ws_owned = ws.to_vec();
        let offset_fn = |f: f64| -> f64 {
            let mut s = 0.0f64;
            for i in 0..n {
                s += ws_owned[i] * Self::plloss(1.0, times_owned[i], events_owned[i], f);
            }
            s
        };
        minimize_brent(&offset_fn, 0.0, max_log_time, R_OPTIMIZE_TOL, 500)
    }

    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        y.len() / 2
    }

    fn calibrates(&self) -> bool {
        true
    }

    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        let times = ys[..n].to_vec();
        let events = ys[n..].to_vec();
        let fs_owned = fs.to_vec();
        let ws_owned = ws.to_vec();

        let risk_fn = |s: f64| Self::risk_s(s, &times, &events, &fs_owned, &ws_owned);
        let sigma_opt = minimize_brent(
            &risk_fn,
            self.nuirange.0,
            self.nuirange.1,
            R_OPTIMIZE_TOL,
            500,
        );
        if sigma_opt > 0.0 && sigma_opt.is_finite() {
            self.sigma = sigma_opt;
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let _ws = w.as_slice().expect("w must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        assert_eq!(os.len(), n);
        let times = &ys[..n];
        let events = &ys[n..];
        let sigma = self.sigma;
        for i in 0..n {
            let eta = (times[i].ln() - fs[i]) / sigma;
            if events[i] > 0.5 {
                // Event: -d(plloss)/d(f) = eta/sigma
                os[i] = eta / sigma;
            } else {
                // Censored: dnorm(eta) / (sigma * (1 - pnorm(eta)))
                //          = phi(eta) / (sigma * Phi(-eta))
                let phi_neg = 0.5 * libm::erfc(eta / std::f64::consts::SQRT_2);
                if phi_neg > 1e-30 {
                    os[i] = Self::phi(eta) / (sigma * phi_neg);
                } else {
                    os[i] = 0.0;
                }
            }
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let n = ys.len() / 2;
        assert_eq!(fs.len(), n);
        let times = &ys[..n];
        let events = &ys[n..];
        let sigma = self.sigma;
        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * Self::plloss(sigma, times[i], events[i], fs[i]);
        }
        risk
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        self.negative_gradient(y, f, w, out.view_mut());
        self.risk(y, f, w)
    }

    fn supports_gram_update(&self) -> bool {
        false
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let ys = y.as_slice().expect("y must be contiguous");
        let len = ys.len();
        if len % 2 != 0 {
            return Err(MboostrsError::BadResponse(
                "Lognormal".to_string(),
                format!("y must have even length (concatenated [time, event]), got {len}"),
            ));
        }
        let n = len / 2;
        for i in 0..n {
            if ys[i] <= 0.0 {
                return Err(MboostrsError::BadResponse(
                    "Lognormal".to_string(),
                    format!("survival times must be > 0, got {}", ys[i]),
                ));
            }
            if ys[n + i] != 0.0 && ys[n + i] != 1.0 {
                return Err(MboostrsError::BadResponse(
                    "Lognormal".to_string(),
                    format!("event indicator must be 0 or 1, got {}", ys[n + i]),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(Lognormal {
            sigma: self.sigma,
            nuirange: self.nuirange,
        })
    }
}

// ─── NBinomial (Negative Binomial) ──────────────────────────────────

/// Negative binomial family with nuisance size `sigma`.
///
/// Sigma is calibrated via Brent minimization of the risk at each boosting
/// step, matching R's `mboost::NBinomial()`.
/// The variance is `mu + mu^2 / sigma`; statsmodels NB2's `alpha` is
/// `1 / sigma`. [`Self::size`] names the calibrated parameter explicitly.
///
/// Risk (negative log-likelihood):
///   plloss(sigma, y, f) = lgamma(sigma) - lgamma(y+sigma) + lgamma(y+1)
///                         + y*log(1+sigma/exp(f)) + sigma*log(1+exp(f)/sigma)
/// Offset: log(weighted_mean(y, w))
/// ngradient: y - exp(f) * (y + sigma) / (exp(f) + sigma)
#[derive(Clone)]
pub struct NBinomial {
    /// Size parameter (`sigma` in R's NBinomial), with variance
    /// `mu + mu^2 / sigma`.
    pub sigma: f64,
}

impl Default for NBinomial {
    fn default() -> Self {
        Self { sigma: 1.0 }
    }
}

impl NBinomial {
    /// Construct a negative-binomial family with the initial size
    /// `sigma = 1.0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calibrated negative-binomial size. statsmodels NB2's `alpha` is its
    /// reciprocal; [`Family::nuisance`] returns this same size.
    pub fn size(&self) -> f64 {
        self.sigma
    }
}

fn nb_plloss(sigma: f64, y: f64, f_val: f64) -> f64 {
    let mu = f_val.exp();
    // The `lgamma(y + 1)` term is the count normalising constant log(y!). It
    // depends only on y (not sigma or f), so it leaves the ngradient and the
    // sigma calibration (argmin over sigma) unchanged, but R's NBinomial
    // plloss includes it, so omitting it made the absolute risk wrong by
    // sum(w * lgamma(y+1)) -- visible in AIC and any cross-library risk check.
    lgamma(sigma) - lgamma(y + sigma)
        + lgamma(y + 1.0)
        + y * (1.0 + sigma / mu).ln()
        + sigma * (1.0 + mu / sigma).ln()
}

impl Family for NBinomial {
    fn risk_rise_threshold(&self) -> Option<f64> {
        // Nuisance calibration minimizes the same likelihood along the path.
        Some(0.01)
    }

    fn name(&self) -> &'static str {
        "NBinomial"
    }

    fn nuisance(&self) -> Option<f64> {
        Some(self.sigma)
    }

    fn set_nuisance(&mut self, value: f64) {
        self.sigma = value;
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let mu = weighted_mean(y, w);
        if mu > 0.0 {
            mu.ln()
        } else {
            0.0
        }
    }

    fn calibrates(&self) -> bool {
        true
    }

    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) {
        // The search is R's `optimize`, step for step, so the chosen sigma
        // matches mboost; each of its ~30 evaluations is made cheap instead.
        // `mu` and the sigma-free `lgamma(y + 1)` term are computed once, and
        // `lgamma(y + sigma)` once per distinct count rather than once per
        // row: counts take few distinct values, and this was 90% of an
        // NBinomial fit's time (20x a Poisson fit on 2,000 rows).
        let n = y.len();
        let mut mu = Vec::with_capacity(n);
        let mut constant = 0.0f64;
        let mut total_weight = 0.0f64;
        let mut by_count: Vec<(f64, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            mu.push(f[i].exp());
            constant += w[i] * lgamma(y[i] + 1.0);
            total_weight += w[i];
            by_count.push((y[i], w[i]));
        }
        by_count.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut counts: Vec<(f64, f64)> = Vec::new();
        for (value, weight) in by_count {
            match counts.last_mut() {
                Some(last) if last.0 == value => last.1 += weight,
                _ => counts.push((value, weight)),
            }
        }
        let risk_fn = |s: f64| -> f64 {
            let mut risk = constant + total_weight * lgamma(s);
            for &(value, weight) in &counts {
                risk -= weight * lgamma(value + s);
            }
            for i in 0..n {
                risk += w[i] * (y[i] * (1.0 + s / mu[i]).ln() + s * (1.0 + mu[i] / s).ln());
            }
            risk
        };
        let sigma_opt = minimize_brent(&risk_fn, 0.0, 100.0, R_OPTIMIZE_TOL, 500);
        if sigma_opt > 0.0 && sigma_opt.is_finite() {
            self.sigma = sigma_opt;
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        let sigma = self.sigma;
        for i in 0..n {
            let ef = fs[i].exp();
            os[i] = ys[i] - ef * (ys[i] + sigma) / (ef + sigma);
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let sigma = self.sigma;
        let mut risk = 0.0f64;
        for i in 0..n {
            risk += ws[i] * nb_plloss(sigma, ys[i], fs[i]);
        }
        risk
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let sigma = self.sigma;
        let mut risk = 0.0f64;
        for i in 0..n {
            let ef = fs[i].exp();
            os[i] = ys[i] - ef * (ys[i] + sigma) / (ef + sigma);
            risk += ws[i] * nb_plloss(sigma, ys[i], fs[i]);
        }
        risk
    }

    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = fs.len();
        assert_eq!(os.len(), n);
        for i in 0..n {
            os[i] = fs[i].exp();
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            if yi < 0.0 {
                return Err(MboostrsError::BadResponse(
                    "NBinomial".to_string(),
                    format!("y must be >= 0, got {yi}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(NBinomial { sigma: self.sigma })
    }
}

// ─── Hurdle (zero-truncated negative binomial) ───────────────────────

/// The non-zero part of a hurdle model, matching R's `mboost::Hurdle()`: a
/// negative binomial truncated at zero, with mean parameter `mu = exp(f)` and
/// dispersion `sigma` re-estimated at every step by Brent's method on
/// `(0, 100)`, as `optimize` does in mboost.
///
/// With `a = 1 + mu * sigma` and `p0 = a^(-1/sigma)`, the negative
/// binomial's probability of a zero, the loss of one count `y` is
///
/// ```text
/// -[ y (f - log(1/sigma + mu)) - (log(1/sigma + mu) + log sigma) / sigma
///    + lgamma(y + 1/sigma) - lgamma(y + 1) - lgamma(1/sigma) - log(1 - p0) ]
/// ```
///
/// the negative log of the negative binomial mass divided by `1 - p0`. The
/// zero part of the hurdle is a separate binary model; this family is fitted
/// to the positive counts.
#[derive(Clone)]
pub struct Hurdle {
    /// Dispersion of the negative binomial (`sigma` in R's Hurdle).
    pub sigma: f64,
}

impl Default for Hurdle {
    fn default() -> Self {
        Self { sigma: 1.0 }
    }
}

impl Hurdle {
    /// A hurdle family with the starting dispersion `sigma = 1`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// `log(1 - p0)` and `p0` for `p0 = (1 + mu sigma)^(-1/sigma)`, without
/// cancellation when `p0` is close to one.
fn hurdle_zero_mass(mu: f64, sigma: f64) -> (f64, f64) {
    let log_p0 = -(mu * sigma).ln_1p() / sigma;
    let one_minus = -log_p0.exp_m1();
    (one_minus.ln(), log_p0.exp())
}

fn hurdle_plloss(sigma: f64, y: f64, f: f64) -> f64 {
    let mu = f.exp();
    let inv = 1.0 / sigma;
    let log_rate = (inv + mu).ln();
    let (log_nonzero, _) = hurdle_zero_mass(mu, sigma);
    -(y * (f - log_rate) - (log_rate + sigma.ln()) / sigma + lgamma(y + inv)
        - lgamma(y + 1.0)
        - lgamma(inv)
        - log_nonzero)
}

fn hurdle_ngradient(sigma: f64, y: f64, f: f64) -> f64 {
    let mu = f.exp();
    let a = 1.0 + mu * sigma;
    let (log_nonzero, p0) = hurdle_zero_mass(mu, sigma);
    y / sigma / (mu + 1.0 / sigma)
        - 1.0 / sigma / ((-f).exp() / sigma + 1.0)
        - mu * p0 / a / log_nonzero.exp()
}

impl Family for Hurdle {
    fn risk_rise_threshold(&self) -> Option<f64> {
        // Nuisance calibration minimizes the same likelihood along the path.
        Some(0.01)
    }

    fn name(&self) -> &'static str {
        "Hurdle"
    }

    fn nuisance(&self) -> Option<f64> {
        Some(self.sigma)
    }

    fn set_nuisance(&mut self, value: f64) {
        self.sigma = value;
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let mu = weighted_mean(y, w);
        if mu > 0.0 {
            mu.ln()
        } else {
            0.0
        }
    }

    fn calibrates(&self) -> bool {
        true
    }

    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) {
        let risk_fn = |s: f64| -> f64 {
            let mut risk = 0.0f64;
            for i in 0..y.len() {
                risk += w[i] * hurdle_plloss(s, y[i], f[i]);
            }
            risk
        };
        let sigma_opt = minimize_brent(&risk_fn, 0.0, 100.0, R_OPTIMIZE_TOL, 500);
        if sigma_opt > 0.0 && sigma_opt.is_finite() {
            self.sigma = sigma_opt;
        }
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        for i in 0..n {
            os[i] = hurdle_ngradient(self.sigma, ys[i], fs[i]);
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        (0..n)
            .map(|i| ws[i] * hurdle_plloss(self.sigma, ys[i], fs[i]))
            .sum()
    }

    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        assert_eq!(os.len(), fs.len());
        for (o, &fi) in os.iter_mut().zip(fs) {
            *o = fi.exp();
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            if !yi.is_finite() || yi != yi.floor() {
                return Err(MboostrsError::BadResponse(
                    "Hurdle".to_string(),
                    format!("y must be whole counts, got {yi}"),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(self.clone())
    }
}

// ─── PropOdds (Proportional Odds / Ordered Logit) ────────────────────

/// R's `d2s`: increasing thresholds from unconstrained parameters.
pub(crate) fn delta_to_sigma(delta: &[f64]) -> Vec<f64> {
    let mut sigma = Vec::with_capacity(delta.len());
    let mut current = delta.first().copied().unwrap_or(0.0);
    for (j, &d) in delta.iter().enumerate() {
        if j > 0 {
            current += d.exp();
        }
        sigma.push(current);
    }
    sigma
}

/// The inverse of [`delta_to_sigma`]; gaps that are not positive (a start of
/// equal thresholds) are floored so the logarithm exists.
pub(crate) fn sigma_to_delta(sigma: &[f64]) -> Vec<f64> {
    sigma
        .iter()
        .enumerate()
        .map(|(j, &s)| {
            if j == 0 {
                s
            } else {
                (s - sigma[j - 1]).max(1e-8).ln()
            }
        })
        .collect()
}

/// Proportional odds model (ordered logit) with K classes.
///
/// Response `y` is integer-coded 1..K. The family maintains K-1
/// threshold parameters `sigma` that are calibrated at each boosting
/// step via coordinate descent (Brent on each sigma_k).
///
/// **Cumulative probabilities:**
///   P(Y <= k | f) = 1 / (1 + exp(-(sigma_k - f)))   for k = 1..K-1
///
///   This matches R's mboost convention: higher f → higher category.
///
/// **ngradient(y_i, f_i):**
///   class 1:  -1 / (1 + exp(sigma_1 - f_i))
///   class K:   1 / (1 + exp(f_i - sigma_{K-1}))
///   middle k:  (1 - exp(2*f_i - sigma_{k-1} - sigma_k))
///               / (1 + exp(f_i - sigma_{k-1}) + exp(f_i - sigma_k)
///                  + exp(2*f_i - sigma_{k-1} - sigma_k))
///
/// **Risk:** -sum(w * log P(Y = y_i | f_i))
#[derive(Debug, Clone)]
pub struct PropOdds {
    /// K-1 ordered threshold parameters. Calibrated at each step.
    pub sigma: Vec<f64>,
}

impl PropOdds {
    /// Unchecked. Prefer [`PropOdds::try_new`]: `saturating_sub` turns
    /// `n_levels = 0` into an empty threshold vector, which describes an
    /// ordinal response with no categories.
    pub fn new(n_levels: usize) -> Self {
        let k = n_levels.saturating_sub(1);
        // R's start: delta = seq(-0.5, -1, length = K - 1), sigma = d2s(delta).
        let delta: Vec<f64> = (0..k)
            .map(|j| {
                if k > 1 {
                    -0.5 - 0.5 * j as f64 / (k - 1) as f64
                } else {
                    -0.5
                }
            })
            .collect();
        Self {
            sigma: delta_to_sigma(&delta),
        }
    }

    /// `n_levels` at least 3. Two ordered categories are a binary response,
    /// which `Binomial` fits; one is not a classification problem.
    pub fn try_new(n_levels: usize) -> Result<Self, MboostrsError> {
        check_at_least("PropOdds", "n_levels", n_levels, 3)?;
        Ok(Self::new(n_levels))
    }

    fn calibrate_thresholds(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
    ) -> CalibrationStatus {
        let n_levels = self.sigma.len() + 1;
        if n_levels < 3 {
            return CalibrationStatus::NotTracked;
        }

        // Per row: its class and exp(f). The cumulative probability
        // P(y <= k) = 1 / (1 + exp(f - sigma_k)) is then exp(f) * exp(-sigma_k)
        // away, so a risk evaluation computes one exponential per threshold
        // rather than one per threshold per row, and each row reads only the
        // two thresholds around its own class. This search runs about 30
        // evaluations per threshold at every boosting step, which made
        // PropOdds 25 times slower than a proportional-odds GLM.
        let classes: Vec<usize> = y.iter().map(|&v| v as usize).collect();
        let exp_f: Vec<f64> = f.iter().map(|&v| v.exp()).collect();
        let w_owned = w.to_owned();
        let mut saw_nonfinite = false;

        // Coordinate descent on R's unconstrained parameters `delta`, where
        // sigma = delta[0] + cumsum(0, exp(delta[1..])): every candidate is an
        // increasing threshold vector. Searching sigma directly let the
        // thresholds cross; a middle class's probability then went negative,
        // its log hit the floor, and with four levels the first threshold
        // climbed without bound (log-loss 5.9 on data where 1.28 is
        // attainable).
        let mut delta = sigma_to_delta(&self.sigma);
        for _iter in 0..5 {
            let mut max_change = 0.0f64;
            for j in 0..delta.len() {
                let delta_snapshot = delta.clone();
                let compute_risk = |d_j: f64| -> f64 {
                    let mut candidate = delta_snapshot.clone();
                    candidate[j] = d_j;
                    let sigma = delta_to_sigma(&candidate);
                    let exp_neg_sigma: Vec<f64> = sigma.iter().map(|&s| (-s).exp()).collect();
                    let top = sigma.len() + 1;
                    let mut risk = 0.0f64;
                    for i in 0..classes.len() {
                        let yi = classes[i];
                        if yi == 0 || yi > top {
                            // Outside 1..K: contributes the floor, as before.
                            risk -= w_owned[i] * -690.0;
                            continue;
                        }
                        let cum = |k: usize| 1.0 / (1.0 + exp_f[i] * exp_neg_sigma[k]);
                        let p_k = if yi == 1 {
                            cum(0)
                        } else if yi == top {
                            1.0 - cum(top - 2)
                        } else {
                            cum(yi - 1) - cum(yi - 2)
                        };
                        let log_p = if p_k > 1e-300 { p_k.ln() } else { -690.0 };
                        risk -= w_owned[i] * log_p;
                    }
                    risk
                };

                let lo = delta[j] - 5.0;
                let hi = delta[j] + 5.0;
                let opt = minimize_brent(&compute_risk, lo, hi, R_OPTIMIZE_TOL, 200);
                if opt.is_finite() {
                    let change = (opt - delta[j]).abs();
                    max_change = max_change.max(change);
                    delta[j] = opt;
                } else {
                    saw_nonfinite = true;
                }
            }
            self.sigma = delta_to_sigma(&delta);
            if max_change < 1e-6 {
                return if saw_nonfinite {
                    CalibrationStatus::Failed
                } else {
                    CalibrationStatus::Converged
                };
            }
        }
        if saw_nonfinite {
            CalibrationStatus::Failed
        } else {
            CalibrationStatus::LimitReached
        }
    }
}

impl Family for PropOdds {
    fn risk_rise_threshold(&self) -> Option<f64> {
        // Nuisance calibration minimizes the same likelihood along the path.
        Some(0.01)
    }

    fn name(&self) -> &'static str {
        "PropOdds"
    }

    fn offset(&self, y: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let y_max = y.iter().cloned().fold(0.0f64, f64::max);

        let y_owned = y.to_owned();
        let w_owned = w.to_owned();
        let sigma_owned = self.sigma.clone();
        let risk_fn = |f_val: f64| -> f64 {
            let mut s = 0.0f64;
            for i in 0..y_owned.len() {
                let yi = y_owned[i] as usize;
                let mut log_p = 0.0f64;
                let mut cum = 0.0f64;
                let mut prev_cum = 0.0f64;
                for j in 0..sigma_owned.len() {
                    cum = 1.0 / (1.0 + (-(sigma_owned[j] - f_val)).exp());
                    if yi == j + 1 {
                        let p_k = cum - prev_cum;
                        log_p = if p_k > 1e-300 { p_k.ln() } else { -690.0 };
                    }
                    prev_cum = cum;
                }
                if yi == sigma_owned.len() + 1 {
                    let p_k = 1.0 - cum;
                    log_p = if p_k > 1e-300 { p_k.ln() } else { -690.0 };
                }
                s -= w_owned[i] * log_p;
            }
            s
        };

        let lo = -y_max.abs().max(1.0) - 5.0;
        let hi = y_max.abs().max(1.0) + 5.0;
        minimize_brent(&risk_fn, lo, hi, R_OPTIMIZE_TOL, 500)
    }

    fn calibrates(&self) -> bool {
        self.sigma.len() >= 2
    }

    // Coordinate descent seeded from the current thresholds: where it starts
    // decides where it stops, so a resumed fit restores them rather than
    // recalibrating.
    fn calibration_is_path_dependent(&self) -> bool {
        true
    }

    fn calibrate(&mut self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) {
        let _ = self.calibrate_thresholds(y, f, w);
    }

    fn calibrate_with_status(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
    ) -> CalibrationStatus {
        self.calibrate_thresholds(y, f, w)
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let (ys, fs, _ws, os, n) = s4!(y, f, w, out);
        let sigma = &self.sigma;
        let k = sigma.len();
        if k == 0 {
            for i in 0..n {
                os[i] = 0.0;
            }
            return;
        }
        for i in 0..n {
            let yi = ys[i] as usize;
            let fi = fs[i];
            if yi == 0 || yi > k + 1 {
                os[i] = 0.0;
            } else if yi == 1 {
                os[i] = -1.0 / (1.0 + (sigma[0] - fi).exp());
            } else if yi == k + 1 {
                os[i] = 1.0 / (1.0 + (fi - sigma[k - 1]).exp());
            } else {
                let sp = sigma[yi - 2];
                let sc = sigma[yi - 1];
                let ef2 = (2.0 * fi - sp - sc).exp();
                let denom = 1.0 + (fi - sp).exp() + (fi - sc).exp() + ef2;
                os[i] = (1.0 - ef2) / denom;
            }
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let (ys, fs, ws, n) = s3!(y, f, w);
        let sigma = &self.sigma;
        let k = sigma.len();
        if k == 0 {
            return 0.0;
        }
        let mut risk = 0.0f64;
        for i in 0..n {
            let yi = ys[i] as usize;
            let fi = fs[i];
            let mut cum = 0.0f64;
            let mut prev_cum = 0.0f64;
            let mut log_p = -690.0f64;
            for j in 0..k {
                cum = 1.0 / (1.0 + (-(sigma[j] - fi)).exp());
                if yi == j + 1 {
                    let p_k = cum - prev_cum;
                    log_p = if p_k > 1e-300 { p_k.ln() } else { -690.0 };
                }
                prev_cum = cum;
            }
            if yi == k + 1 {
                let p_k = 1.0 - cum;
                log_p = if p_k > 1e-300 { p_k.ln() } else { -690.0 };
            }
            risk -= ws[i] * log_p;
        }
        risk
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let (ys, fs, ws, os, n) = s4!(y, f, w, out);
        let sigma = &self.sigma;
        let k = sigma.len();
        let mut risk = 0.0f64;

        if k == 0 {
            for i in 0..n {
                os[i] = 0.0;
            }
            return 0.0;
        }

        for i in 0..n {
            let yi = ys[i] as usize;
            let fi = fs[i];

            let mut cum = 0.0f64;
            let mut prev_cum = 0.0f64;
            let mut log_p = -690.0f64;
            for j in 0..k {
                cum = 1.0 / (1.0 + (-(sigma[j] - fi)).exp());
                if yi == j + 1 {
                    let p_k = cum - prev_cum;
                    log_p = if p_k > 1e-300 { p_k.ln() } else { -690.0 };
                }
                prev_cum = cum;
            }
            if yi == k + 1 {
                let p_k = 1.0 - cum;
                log_p = if p_k > 1e-300 { p_k.ln() } else { -690.0 };
            }
            risk -= ws[i] * log_p;

            if yi == 0 || yi > k + 1 {
                os[i] = 0.0;
            } else if yi == 1 {
                os[i] = -1.0 / (1.0 + (sigma[0] - fi).exp());
            } else if yi == k + 1 {
                os[i] = 1.0 / (1.0 + (fi - sigma[k - 1]).exp());
            } else {
                let sp = sigma[yi - 2];
                let sc = sigma[yi - 1];
                let ef2 = (2.0 * fi - sp - sc).exp();
                let denom = 1.0 + (fi - sp).exp() + (fi - sc).exp() + ef2;
                os[i] = (1.0 - ef2) / denom;
            }
        }
        risk
    }

    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let n = fs.len();
        assert_eq!(os.len(), n);
        let sigma = &self.sigma;
        for i in 0..n {
            if sigma.is_empty() {
                os[i] = fs[i];
            } else {
                os[i] = 1.0 / (1.0 + (-(sigma[0] - fs[i])).exp());
            }
        }
    }

    fn nuisance_vec(&self) -> Option<Vec<f64>> {
        Some(self.sigma.clone())
    }

    fn set_nuisance_vec(&mut self, value: Vec<f64>) {
        self.sigma = value;
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        for &yi in y.iter() {
            let yi_int = yi as usize;
            if yi != yi_int as f64 || yi_int < 1 || yi_int > self.sigma.len() + 1 {
                return Err(MboostrsError::BadResponse(
                    "PropOdds".to_string(),
                    format!(
                        "y must be integers in 1..{}, got {yi}",
                        self.sigma.len() + 1
                    ),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(PropOdds {
            sigma: self.sigma.clone(),
        })
    }
}

/// Per-category response probabilities for PropOdds, matching R's
/// `PropOdds()@response`. Returns a row-major `(n x K)` matrix flattened to a
/// `Vec`, where `K = sigma.len() + 1`:
///   `C_k = 1 / (1 + exp(f - sigma_k))`,
///   `P(Y=1) = C_1`, `P(Y=k) = C_k - C_{k-1}`, `P(Y=K) = 1 - C_{K-1}`.
///
/// The probabilities depend only on `sigma_k - f`, so they are invariant to the
/// PropOdds location shift; this matches R's per-category probabilities even
/// when the raw linear predictor differs from R by a constant. Centralised here
/// so any binding computes the response identically. With no thresholds
/// (degenerate), returns `f` unchanged.
pub fn propodds_response(f: &[f64], sigma: &[f64]) -> Vec<f64> {
    let n = f.len();
    if sigma.is_empty() {
        return f.to_vec();
    }
    let k = sigma.len() + 1; // number of categories
    let mut out = vec![0.0f64; n * k];
    for i in 0..n {
        let fi = f[i];
        let mut prev_cum = 0.0f64;
        for (j, &s_j) in sigma.iter().enumerate() {
            let cum = 1.0 / (1.0 + (fi - s_j).exp());
            out[i * k + j] = cum - prev_cum;
            prev_cum = cum;
        }
        out[i * k + (k - 1)] = 1.0 - prev_cum;
    }
    out
}

// ─── Multinomial (Multinomial Logit) ─────────────────────────────────

/// Multinomial logit family for multi-class classification.
///
/// Fits a linear array model via negative multinomial log-likelihood.
/// The response `y` is a dummy-coded factor vector of length `n_obs * (K-1)`,
/// where K is the number of classes and the last class is the reference.
///
/// Requires Kronecker-product base-learners (see R's `%O%` operator).
///
/// **Offset:** Zero vector of length `n_obs * (K-1)`.
///
/// **Negative gradient (softmax residual):**
///   p_k = exp(f_k) / (1 + sum_j exp(f_j))  for class k = 1..K-1
///   u_k = y_k - p_k
///
/// **Risk (negative log-likelihood):**
///   -sum_k y_k * log(p_k)
#[derive(Clone, Debug)]
pub struct Multinomial {
    /// Total number of classes (K). The last class is the reference category.
    pub n_classes: usize,
}

impl Multinomial {
    /// Unchecked in the sense that it **panics** rather than returning an
    /// error. Prefer [`Multinomial::try_new`] anywhere the count comes from
    /// outside the caller: a library that aborts the process on a bad
    /// argument gives its caller nothing to handle.
    pub fn new(n_classes: usize) -> Self {
        Self::try_new(n_classes).expect("Multinomial::new given an invalid class count")
    }

    /// `n_classes` at least 3; two classes are a binary response, which
    /// `Binomial` fits.
    pub fn try_new(n_classes: usize) -> Result<Self, MboostrsError> {
        check_at_least("Multinomial", "n_classes", n_classes, 3)?;
        Ok(Self { n_classes })
    }

    /// Number of non-reference classes (K-1).
    #[inline]
    fn n_resp(&self) -> usize {
        self.n_classes - 1
    }
}

/// One observation's weighted multinomial loss.
///
/// With `include_reference` false this is R's `Multinomial()` loss, `-y *
/// log(p)` over the K-1 non-reference indicator columns: an observation in
/// the reference class contributes nothing. `risk` keeps it, so the in-bag
/// risk path matches mboost. It is not the log-likelihood, and cross-validation
/// that minimises it stops too early: on a three-class simulation `cvrisk`
/// chose 257 steps where held-out log-loss was lowest at about 1,000 (test
/// log-loss 1.036 against 1.026). `held_out_risk` sets `include_reference` and
/// adds the reference class's `-log p_ref = log(1 + Σ exp f_k)`, which is the
/// negative log-likelihood; this is a deliberate deviation from mboost's
/// `cvrisk`.
fn multinomial_row_nll(
    y: &[f64],
    f: &[f64],
    log_denom: f64,
    weight: f64,
    include_reference: bool,
) -> f64 {
    let mut nll = 0.0f64;
    let mut y_ref = 1.0f64;
    for (&yk, &fk) in y.iter().zip(f) {
        if yk > 0.0 {
            nll -= yk * (fk - log_denom);
        }
        y_ref -= yk;
    }
    if include_reference && y_ref > 0.0 {
        nll += y_ref * log_denom;
    }
    weight * nll
}

impl Family for Multinomial {
    fn name(&self) -> &'static str {
        "Negative Multinomial Likelihood"
    }

    fn offset(&self, y: ArrayView1<f64>, _w: ArrayView1<f64>) -> f64 {
        // Scalar 0 — the engine broadcasts to length n via Array1::from_elem(n, 0.0).
        // R returns rep(0, length(y)), which is equivalent since all entries are 0.
        let _ = y;
        0.0
    }

    fn n_from_y(&self, y: ArrayView1<f64>) -> usize {
        // y has length n_obs * (K-1); the engine should use this as the full length
        // for f, u, and best_pred_buf.
        y.len()
    }

    fn flat_response_span(&self) -> usize {
        self.n_resp()
    }

    fn negative_gradient(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        _w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let nr = self.n_resp();
        let n_obs = ys.len() / nr;
        // Clamp f like R does: pmin(abs(f), 36) * sign(f)
        let f_clamped: Vec<f64> = fs.iter().map(|&fi| fi.clamp(-36.0, 36.0)).collect();

        let mut denom = vec![0.0f64; n_obs];
        let mut max_f = vec![0.0f64; n_obs];
        for i in 0..n_obs {
            let mut mf = 0.0f64;
            for k in 0..nr {
                mf = mf.max(f_clamped[i * nr + k]);
            }
            max_f[i] = mf;
            // Reference class logit is 0, so its stabilized term is exp(0 - mf);
            // when mf > 0 this is < 1 (using a literal 1.0 here overcounts the
            // denominator and is only correct at mf == 0, e.g. the offset f = 0).
            let mut d = (-mf).exp();
            for k in 0..nr {
                d += (f_clamped[i * nr + k] - mf).exp();
            }
            denom[i] = d;
        }
        for i in 0..n_obs {
            for k in 0..nr {
                let idx = i * nr + k;
                let p_ik = (f_clamped[idx] - max_f[i]).exp() / denom[i];
                os[idx] = ys[idx] - p_ik;
            }
        }
    }

    fn risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        // Weights are flattened like the response: entry `i * nr + k` carries
        // observation i's weight for every class k, as in upstream's
        // `sum(w * loss(y, f))`. Indexing them by observation (`ws[i]`) reads
        // the weight of observation `i / nr` instead.
        debug_assert_eq!(ws.len(), ys.len());
        let nr = self.n_resp();
        let n_obs = ys.len() / nr;
        let f_clamped: Vec<f64> = fs.iter().map(|&fi| fi.clamp(-36.0, 36.0)).collect();

        let mut risk = 0.0f64;
        for i in 0..n_obs {
            let mut mf = 0.0f64;
            for k in 0..nr {
                mf = mf.max(f_clamped[i * nr + k]);
            }
            // Reference-class term is exp(0 - mf), not a literal 1.0 (see
            // negative_gradient); they coincide only when mf == 0.
            let mut denom = (-mf).exp();
            for k in 0..nr {
                denom += (f_clamped[i * nr + k] - mf).exp();
            }
            let log_denom = mf + denom.ln();
            risk += multinomial_row_nll(
                &ys[i * nr..(i + 1) * nr],
                &f_clamped[i * nr..(i + 1) * nr],
                log_denom,
                ws[i * nr],
                false,
            );
        }
        risk
    }

    fn held_out_risk(&self, y: ArrayView1<f64>, f: ArrayView1<f64>, w: ArrayView1<f64>) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        let nr = self.n_resp();
        let f_clamped: Vec<f64> = fs.iter().map(|&fi| fi.clamp(-36.0, 36.0)).collect();
        let mut risk = 0.0f64;
        for i in 0..ys.len() / nr {
            let row = &f_clamped[i * nr..(i + 1) * nr];
            let mf = row.iter().fold(0.0f64, |acc, &v| acc.max(v));
            let denom = (-mf).exp() + row.iter().map(|&v| (v - mf).exp()).sum::<f64>();
            risk += multinomial_row_nll(
                &ys[i * nr..(i + 1) * nr],
                row,
                mf + denom.ln(),
                ws[i * nr],
                true,
            );
        }
        risk
    }

    fn ngradient_and_risk(
        &mut self,
        y: ArrayView1<f64>,
        f: ArrayView1<f64>,
        w: ArrayView1<f64>,
        mut out: ArrayViewMut1<f64>,
    ) -> f64 {
        let ys = y.as_slice().expect("y must be contiguous");
        let fs = f.as_slice().expect("f must be contiguous");
        let ws = w.as_slice().expect("w must be contiguous");
        // Flattened weights, indexed like the response (see `risk`).
        debug_assert_eq!(ws.len(), ys.len());
        let os = out.as_slice_mut().expect("out must be contiguous");
        let nr = self.n_resp();
        let n_obs = ys.len() / nr;
        let f_clamped: Vec<f64> = fs.iter().map(|&fi| fi.clamp(-36.0, 36.0)).collect();

        let mut denom = vec![0.0f64; n_obs];
        let mut max_f = vec![0.0f64; n_obs];
        for i in 0..n_obs {
            let mut mf = 0.0f64;
            for k in 0..nr {
                mf = mf.max(f_clamped[i * nr + k]);
            }
            max_f[i] = mf;
            // Reference-class term exp(0 - mf) (see negative_gradient).
            let mut d = (-mf).exp();
            for k in 0..nr {
                d += (f_clamped[i * nr + k] - mf).exp();
            }
            denom[i] = d;
        }

        let mut risk = 0.0f64;
        for i in 0..n_obs {
            let log_denom = max_f[i] + denom[i].ln();
            for k in 0..nr {
                let idx = i * nr + k;
                let p_ik = (f_clamped[idx] - max_f[i]).exp() / denom[i];
                os[idx] = ys[idx] - p_ik;
            }
            risk += multinomial_row_nll(
                &ys[i * nr..(i + 1) * nr],
                &f_clamped[i * nr..(i + 1) * nr],
                log_denom,
                ws[i * nr],
                false,
            );
        }
        risk
    }

    fn response(&self, f: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
        let fs = f.as_slice().expect("f must be contiguous");
        let os = out.as_slice_mut().expect("out must be contiguous");
        let nr = self.n_resp();
        let n_obs = fs.len() / nr;
        let f_clamped: Vec<f64> = fs.iter().map(|&fi| fi.clamp(-36.0, 36.0)).collect();

        // R's response: cbind(exp(f), 1) / (1 + rowSums(exp(f)))
        // Output has length n_obs * K (includes reference class)
        // But our out has same length as f = n_obs*(K-1), so we only write non-ref probs.
        // Actually, R returns an n_obs × K matrix. Since out has length n_obs*(K-1),
        // we just write the non-reference probabilities (same as softmax_no_ref).
        for i in 0..n_obs {
            let mut mf = 0.0f64;
            for k in 0..nr {
                mf = mf.max(f_clamped[i * nr + k]);
            }
            // Reference-class term exp(0 - mf) (see negative_gradient).
            let mut denom = (-mf).exp();
            for k in 0..nr {
                denom += (f_clamped[i * nr + k] - mf).exp();
            }
            for k in 0..nr {
                let idx = i * nr + k;
                os[idx] = (f_clamped[idx] - mf).exp() / denom;
            }
        }
    }

    fn check_y(&self, y: ArrayView1<f64>) -> Result<(), MboostrsError> {
        let ys = y.as_slice().expect("y must be contiguous");
        let nr = self.n_resp();
        if ys.len() % nr != 0 {
            return Err(MboostrsError::BadResponse(
                "Multinomial".to_string(),
                format!(
                    "y length {} is not a multiple of n_classes-1={}; expected n_obs * {}",
                    ys.len(),
                    nr,
                    nr
                ),
            ));
        }
        for (i, &yi) in ys.iter().enumerate() {
            if yi != 0.0 && yi != 1.0 {
                return Err(MboostrsError::BadResponse(
                    "Multinomial".to_string(),
                    format!("y must be 0 or 1 (dummy-coded), got {} at index {}", yi, i),
                ));
            }
        }
        // Check that each observation has at most one 1
        let n_obs = ys.len() / nr;
        for i in 0..n_obs {
            let mut count = 0;
            for k in 0..nr {
                if ys[i * nr + k] == 1.0 {
                    count += 1;
                }
            }
            if count > 1 {
                return Err(MboostrsError::BadResponse(
                    "Multinomial".to_string(),
                    format!("observation {} has {} active classes (max 1)", i, count),
                ));
            }
        }
        Ok(())
    }

    fn clone_family(&self) -> Box<dyn Family> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod checked_construction {
    use super::*;

    fn message(error: MboostrsError) -> String {
        match error {
            MboostrsError::BadFamilyParameter(family, detail) => {
                format!("{family}: {detail}")
            }
            other => panic!("expected BadFamilyParameter, got {other:?}"),
        }
    }

    #[test]
    fn quantreg_rejects_tau_outside_the_open_unit_interval() {
        for tau in [0.0, 1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            let error = QuantReg::try_new(tau).unwrap_err();
            assert!(message(error).contains("tau must be in (0, 1)"));
        }
        // The endpoints are the interesting ones: `tau = 0` and `tau = 1`
        // give a check loss minimised by any fit below or above every
        // observation, so the path runs and the answer is unbounded.
        assert!(QuantReg::try_new(1e-6).is_ok());
        assert!(QuantReg::try_new(0.5).is_ok());
        assert!(QuantReg::try_new(1.0 - 1e-6).is_ok());
    }

    #[test]
    fn quantreg_checks_the_offset_quantile_too() {
        assert!(QuantReg::try_with_qoffset(0.5, 0.0).is_err());
        assert!(QuantReg::try_with_qoffset(0.5, 1.0).is_err());
        let error = QuantReg::try_with_qoffset(0.5, 2.0).unwrap_err();
        assert!(message(error).contains("qoffset must be in (0, 1)"));
        assert!(QuantReg::try_with_qoffset(0.9, 0.9).is_ok());
    }

    #[test]
    fn huber_rejects_a_non_positive_delta() {
        for delta in [0.0, -1.0, f64::NAN, f64::NEG_INFINITY] {
            let error = Huber::try_new(Some(delta)).unwrap_err();
            assert!(message(error).contains("delta must be positive"));
        }
        assert!(Huber::try_new(Some(1e-12)).is_ok());
        // `None` is the adaptive default, not a missing value.
        assert!(Huber::try_new(None).is_ok());
    }

    #[test]
    fn cindex_rejects_a_non_positive_sigma() {
        for sigma in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let error = Cindex::try_new(sigma).unwrap_err();
            assert!(message(error).contains("sigma must be positive"));
        }
        assert!(Cindex::try_new(0.1).is_ok());
    }

    #[test]
    fn propodds_rejects_fewer_than_three_levels() {
        for n_levels in [0, 1, 2] {
            let error = PropOdds::try_new(n_levels).unwrap_err();
            assert!(message(error).contains("n_levels must be at least 3"));
        }
        assert_eq!(PropOdds::try_new(3).unwrap().sigma.len(), 2);
    }

    #[test]
    fn propodds_zero_used_to_produce_an_empty_threshold_vector() {
        // `saturating_sub` turned 0 into 0 thresholds rather than failing,
        // which describes an ordinal response with no categories.
        assert_eq!(PropOdds::new(0).sigma.len(), 0);
        assert!(PropOdds::try_new(0).is_err());
    }

    #[test]
    fn multinomial_rejects_fewer_than_three_classes() {
        for n_classes in [0, 1, 2] {
            let error = Multinomial::try_new(n_classes).unwrap_err();
            assert!(message(error).contains("n_classes must be at least 3"));
        }
        assert_eq!(Multinomial::try_new(3).unwrap().n_classes, 3);
    }

    #[test]
    #[should_panic(expected = "invalid class count")]
    fn multinomial_new_still_panics_and_says_so_in_its_docs() {
        // Kept, because deprecating it would break callers that already
        // validate. `try_new` is the route for a count from outside.
        let _ = Multinomial::new(2);
    }

    #[test]
    fn the_checked_constructors_agree_with_the_unchecked_ones_when_valid() {
        assert_eq!(
            QuantReg::try_new(0.25).unwrap().tau,
            QuantReg::new(0.25).tau
        );
        assert_eq!(Cindex::try_new(0.3).unwrap().sigma, Cindex::new(0.3).sigma);
        assert_eq!(
            PropOdds::try_new(4).unwrap().sigma.len(),
            PropOdds::new(4).sigma.len()
        );
        assert_eq!(
            Multinomial::try_new(5).unwrap().n_classes,
            Multinomial::new(5).n_classes
        );
    }
}

#[cfg(test)]
mod test_suite;
