//! The doubling grid search behind `tune_mstop`.
//!
//! A search cross-validates the boosting step count on a sequence of grids.
//! Each grid ends at a larger iteration than the one before (the end doubles,
//! capped at `max_mstop`), and the search stops at the first grid whose best
//! step count is not in the boundary band at its end, or at the last grid of
//! the schedule. The folds are advanced by a [`FoldRisks`] implementation, so
//! each fold's path is resumed from where the previous grid stopped rather
//! than refitted from zero.

use crate::MboostrsError;

/// One grid of a search: the iteration the grid ends at, and the iterations
/// at which the held-out risk is read (ascending, each at least 1 for the
/// optimum, each at most `end`).
#[derive(Clone, Debug)]
pub struct GridStep {
    /// The grid's largest iteration.
    pub end: usize,
    /// The iterations whose aggregated risk the grid compares.
    pub grid: Vec<usize>,
}

/// The per-fold risk paths a search grows.
pub trait FoldRisks {
    /// Advance every fold's path to `end` iterations.
    ///
    /// Returns, for each fold, the held-out risk at the iterations after the
    /// previous call's `end` up to this `end` (iteration 0 on the first call).
    /// Iterations no grid reads may hold `NaN`. Every fold returns the same
    /// number of values.
    fn advance(&mut self, end: usize) -> Result<Vec<Vec<f64>>, MboostrsError>;
}

/// The result of a search.
#[derive(Clone, Debug)]
pub struct GridSearch {
    /// The `end` of every grid the search evaluated, in order.
    pub ends: Vec<usize>,
    /// The iterations of the final grid.
    pub grid: Vec<usize>,
    /// Per fold, the held-out risk at each iteration of the final grid.
    pub risk: Vec<Vec<f64>>,
    /// The aggregated risk at each iteration of the final grid: the mean over
    /// folds, summed as `sum(risk / folds)` in fold order.
    pub curve: Vec<f64>,
    /// The iteration of the lowest finite aggregated risk on the final grid
    /// (ties go to the lower iteration).
    pub best: usize,
    /// True when `best` lies in the boundary band at the final grid's end.
    pub at_boundary: bool,
}

/// Whether an optimum at `best` lies in the boundary band of a grid that ends
/// at `end`: the band is `boundary_share` of the end, rounded half up and at
/// least one step, and `best` is inside it when it is that close to `end`.
pub fn is_at_boundary(best: usize, end: usize, boundary_share: f64) -> bool {
    let tolerance = ((boundary_share * end as f64 + 0.5).floor().max(1.0)) as usize;
    best + tolerance >= end
}

/// Run the doubling search over `schedule`.
///
/// The first grid is evaluated after `folds.advance(schedule[0].end)`. While
/// its optimum is in the boundary band and a later grid exists, the folds are
/// advanced to the next grid's end and that grid is evaluated. The schedule is
/// fixed up front, so the search reads no more of it than the data requires.
pub fn search_grid(
    folds: &mut dyn FoldRisks,
    schedule: &[GridStep],
    boundary_share: f64,
) -> Result<GridSearch, MboostrsError> {
    if schedule.is_empty() {
        return Err(MboostrsError::Control(
            "a grid search needs at least one grid".to_string(),
        ));
    }
    if !(boundary_share.is_finite() && (0.0..1.0).contains(&boundary_share)) {
        return Err(MboostrsError::Control(format!(
            "boundary_share must lie in [0, 1), got {boundary_share}"
        )));
    }
    for (k, step) in schedule.iter().enumerate() {
        if step.end == 0 {
            return Err(MboostrsError::Control(
                "every grid must end at 1 iteration or more".to_string(),
            ));
        }
        if k > 0 && step.end <= schedule[k - 1].end {
            return Err(MboostrsError::Control(
                "grid ends must increase along the schedule".to_string(),
            ));
        }
        if step.grid.iter().any(|&g| g > step.end) {
            return Err(MboostrsError::Control(format!(
                "a grid point lies beyond its grid's end of {}",
                step.end
            )));
        }
    }

    // history[fold][iteration] holds every risk value the folds have produced.
    let mut history: Vec<Vec<f64>> = Vec::new();
    for (k, step) in schedule.iter().enumerate() {
        let fresh = folds.advance(step.end)?;
        if k == 0 {
            if fresh.is_empty() {
                return Err(MboostrsError::Solver(
                    "a grid search needs at least one fold".to_string(),
                ));
            }
            history = vec![Vec::new(); fresh.len()];
        }
        if fresh.len() != history.len() {
            return Err(MboostrsError::Solver(format!(
                "the folds returned {} risk paths, expected {}",
                fresh.len(),
                history.len()
            )));
        }
        for (path, values) in history.iter_mut().zip(fresh) {
            path.extend(values);
        }
        for path in &history {
            if path.len() != step.end + 1 {
                return Err(MboostrsError::Solver(format!(
                    "a fold's risk path has {} iterations, expected {}",
                    path.len(),
                    step.end + 1
                )));
            }
        }

        let folds_count = history.len() as f64;
        let curve: Vec<f64> = step
            .grid
            .iter()
            .map(|&g| {
                if history.iter().all(|path| path[g].is_finite()) {
                    history.iter().map(|path| path[g] / folds_count).sum()
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        let mut best: Option<(usize, f64)> = None;
        for (j, &g) in step.grid.iter().enumerate() {
            if g == 0 || !curve[j].is_finite() {
                continue;
            }
            match best {
                Some((_, value)) if curve[j] >= value => {}
                _ => best = Some((g, curve[j])),
            }
        }
        let (best, _) = best.ok_or_else(|| {
            MboostrsError::Solver(format!(
                "no finite held-out risk at a positive iteration on the grid ending at {}",
                step.end
            ))
        })?;
        let at_boundary = is_at_boundary(best, step.end, boundary_share);
        if !at_boundary || k + 1 == schedule.len() {
            let risk = history
                .iter()
                .map(|path| step.grid.iter().map(|&g| path[g]).collect())
                .collect();
            return Ok(GridSearch {
                ends: schedule[..=k].iter().map(|s| s.end).collect(),
                grid: step.grid.clone(),
                risk,
                curve,
                best,
                at_boundary,
            });
        }
    }
    Err(MboostrsError::Solver(
        "the grid search ended without evaluating a grid".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Folds whose risk at iteration m is `curve[m]` for every fold.
    struct Constant {
        curve: Vec<f64>,
        done: usize,
        folds: usize,
    }

    impl FoldRisks for Constant {
        fn advance(&mut self, end: usize) -> Result<Vec<Vec<f64>>, MboostrsError> {
            let fresh: Vec<f64> = (self.done..=end).map(|m| self.curve[m]).collect();
            self.done = end + 1;
            Ok(vec![fresh; self.folds])
        }
    }

    fn all_steps(end: usize) -> GridStep {
        GridStep {
            end,
            grid: (0..=end).collect(),
        }
    }

    #[test]
    fn boundary_band_follows_the_five_percent_rule() {
        // 5% of 100 is 5 steps: 95 is in the band, 94 is not.
        assert!(is_at_boundary(95, 100, 0.05));
        assert!(!is_at_boundary(94, 100, 0.05));
        // At least one step, even for a short grid.
        assert!(is_at_boundary(9, 10, 0.05));
        assert!(!is_at_boundary(8, 10, 0.05));
    }

    #[test]
    fn stops_at_the_first_grid_whose_optimum_is_inside() {
        // Minimum at 30 on the grid ending at 40. 5% of 40 is 2 steps, and
        // 30 + 2 < 40, so the optimum is not in the band and the search stops.
        let curve: Vec<f64> = (0..=200).map(|m| ((m as f64) - 30.0).powi(2)).collect();
        let mut folds = Constant {
            curve,
            done: 0,
            folds: 3,
        };
        let schedule = vec![all_steps(40), all_steps(80)];
        let found = search_grid(&mut folds, &schedule, 0.05).unwrap();
        assert_eq!(found.best, 30);
        assert_eq!(found.ends, vec![40]);
        assert!(!found.at_boundary);
        assert_eq!(folds.done, 41, "the second grid must not be evaluated");
    }

    #[test]
    fn doubles_while_the_optimum_is_at_the_end() {
        // Decreasing risk: the optimum is always the grid's end.
        let curve: Vec<f64> = (0..=200).map(|m| -(m as f64)).collect();
        let mut folds = Constant {
            curve,
            done: 0,
            folds: 2,
        };
        let schedule = vec![all_steps(10), all_steps(20), all_steps(40)];
        let found = search_grid(&mut folds, &schedule, 0.05).unwrap();
        assert_eq!(found.ends, vec![10, 20, 40]);
        assert_eq!(found.best, 40);
        assert!(found.at_boundary);
        assert_eq!(found.risk.len(), 2);
        assert_eq!(found.risk[0].len(), 41);
    }

    #[test]
    fn ties_go_to_the_lower_iteration_and_nonfinite_values_are_skipped() {
        let mut curve = vec![1.0; 21];
        curve[0] = 0.0; // offset-only: never eligible
        curve[5] = f64::NAN;
        curve[7] = 0.5;
        curve[9] = 0.5;
        let mut folds = Constant {
            curve,
            done: 0,
            folds: 1,
        };
        let found = search_grid(&mut folds, &[all_steps(20)], 0.05).unwrap();
        assert_eq!(found.best, 7);
    }

    #[test]
    fn rejects_a_schedule_that_does_not_increase() {
        let mut folds = Constant {
            curve: vec![0.0; 50],
            done: 0,
            folds: 1,
        };
        let err = search_grid(&mut folds, &[all_steps(20), all_steps(20)], 0.05);
        assert!(err.is_err());
    }
}
