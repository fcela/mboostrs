# The R bootstrap confidence intervals, and the four defects that made them
# report something other than the fit they came from.
#
# * `set.seed(seed)` reset the session's stream instead of scoping the seed to
#   the call, so a script that seeded before `confint()` and drew after it got
#   numbers it did not ask for.
# * The refits dropped the fit's observation weights, so a weighted fit was
#   resampled as if every row weighed one.
# * A replicate that never selected a learner contributed `numeric(0)`, and
#   `rbind` with that dropped the whole replicate -- removing exactly the draws
#   where the effect is zero and pushing the interval away from it.
# * A failed refit was skipped without a warning, so the bounds could come from
#   far fewer draws than `B` and say nothing about it.
#
# The percentile quantiles now come from the Rust core
# (`mboostrs_core::inference::percentile_interval`), the function the Python
# frontend uses.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(11)
n <- 120
d <- data.frame(x1 = rnorm(n), x2 = runif(n))
d$y <- 2 * sin(2 * d$x1) + 0.5 * d$x2 + rnorm(n, sd = 0.3)
w <- rep(1, n)
w[seq_len(20)] <- 5

fit_u <- glmboost(y ~ x1 + x2, data = d, control = boost_control(mstop = 30))
fit_w <- glmboost(y ~ x1 + x2, data = d, control = boost_control(mstop = 30),
                  weights = w)

# ─── the Rust quantile is R's type-7 quantile ───────────────────────

draws <- matrix(c(1, 2, 3, 4, 5, 10, 20, 30, 40, 50), nrow = 5, ncol = 2)
bounds <- mboostrs:::.percentile_bounds(draws, 0.9)
ok(isTRUE(all.equal(unname(bounds[1, 1]), unname(quantile(draws[, 1], 0.05)))),
   "lower bound should be R's type-7 5% quantile")
ok(isTRUE(all.equal(unname(bounds[1, 2]), unname(quantile(draws[, 1], 0.95)))),
   "upper bound should be R's type-7 95% quantile")
ok(isTRUE(all.equal(unname(bounds[2, 1]), unname(quantile(draws[, 2], 0.05)))),
   "the second coefficient gets its own lower bound")

# A missing draw is excluded from the column, as `quantile(na.rm = TRUE)` did.
with_na <- rbind(draws, c(NA_real_, NA_real_))
bounds_na <- mboostrs:::.percentile_bounds(with_na, 0.9)
ok(isTRUE(all.equal(unname(bounds_na), unname(bounds))),
   "missing draws should be excluded, not propagated")

# ─── the seed is scoped to the call ─────────────────────────────────

set.seed(99)
before <- runif(1)
set.seed(99)
invisible(confint(fit_u, B = 3))
after <- runif(1)
ok(identical(before, after),
   "confint() must not advance the caller's random stream")

set.seed(1)
a <- confint(fit_u, B = 3, seed = 7)
set.seed(2)
b <- confint(fit_u, B = 3, seed = 7)
ok(identical(unclass(a), unclass(b)),
   "the same seed should give the same interval whatever the ambient stream")

# ─── the refit carries the fit's weights ────────────────────────────

ci_u <- confint(fit_u, B = 8, seed = 3)
ci_w <- confint(fit_w, B = 8, seed = 3)
ok(attr(ci_u, "n_used") == 8L && attr(ci_w, "n_used") == 8L,
   "every unweighted and weighted refit should have succeeded")
# The refit reads the formula, data and control only. If it ignored the weights
# these two intervals would be identical, because the original coefficient
# vectors play no part in the resampling.
ok(!isTRUE(all.equal(unclass(ci_u), unclass(ci_w))),
   "a weighted fit's interval must come from weighted refits")
ok(isTRUE(all.equal(attr(ci_w, "n_failed"), 0L)),
   "no weighted refit should have failed")

# ─── an unselected learner contributes zero, not a missing draw ─────

# With one update, `bbs(x1)` wins every draw and `bbs(x2)` is never selected.
# `coef()` returns `numeric(0)` for it, and the old loop's `rbind` with that
# dropped the replicate; the interval came back as an all-NA matrix. An
# unselected learner has no contribution, so every draw is zero.
fit0 <- gamboost(y ~ bbs(x1, df = 4) + bbs(x2, df = 4), data = d,
                 control = boost_control(mstop = 1))
ok(identical(selected(fit0), "x1"), "the fixture should select x1 only")
ci0 <- confint(fit0, B = 5, seed = 5)
ok(length(ci0) == 2L, "one interval block per learner")
ok(all(!is.na(ci0$x2)), "a zero effect must not be NA")
ok(all(ci0$x2 == 0), "an unselected learner's interval is [0, 0]")
ok(nrow(ci0$x2) == fit0$learner_ncol[2],
   "the block keeps the learner's coefficient count")
ok(attr(ci0, "n_used") == 5L, "all five refits should count")

# ─── a failed refit is reported ────────────────────────────────────

broken <- fit_u
broken$data$x1 <- NULL
warned <- FALSE
ci_broken <- withCallingHandlers(
  confint(broken, B = 2, seed = 1),
  warning = function(w) {
    if (grepl("bootstrap refits failed", conditionMessage(w))) warned <<- TRUE
    invokeRestart("muffleWarning")
  }
)
ok(warned, "a failed refit must produce a warning")
ok(all(is.na(ci_broken)), "with no successful refit the bounds are NA")
ok(attr(ci_broken, "n_failed") == 2L, "the failure count should be reported")

# ─── gamboost keeps the same guarantees ────────────────────────────

fit_g <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d,
                  control = boost_control(mstop = 20))
set.seed(123)
before <- runif(1)
set.seed(123)
ci_g <- confint(fit_g, B = 5, seed = 4)
after <- runif(1)
ok(identical(before, after),
   "gamboost confint() must not advance the caller's random stream")
ok(attr(ci_g, "n_used") == 5L, "all five gamboost refits should count")
for (nm in names(ci_g)) {
  ok(ncol(ci_g[[nm]]) == 2L, sprintf("%s: two bounds", nm))
  ok(all(ci_g[[nm]][, 1] <= ci_g[[nm]][, 2]), sprintf("%s: lower <= upper", nm))
}
