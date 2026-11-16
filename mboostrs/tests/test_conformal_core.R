# conformal_interval() in R. The numerics are the Rust core's
# (mboostrs_core::conformal); the checks below compare each core call with a
# plain-R version of the same rule, then run the three methods end to end.
suppressMessages(library(mboostrs))
ns <- asNamespace("mboostrs")

# Plain-R versions of the rules the core implements.
oracle_quantile <- function(scores, level) {
  scores <- sort(scores)
  n <- length(scores)
  k <- ceiling((n + 1) * level)
  if (k > n) return(Inf)
  scores[k]
}
oracle_cv_plus_bounds <- function(residuals, fold_of, preds, level) {
  n <- length(residuals)
  alpha <- 1 - level
  k_lo <- floor(alpha * (n + 1))
  k_hi <- ceiling(level * (n + 1))
  block <- preds[fold_of + 1L, , drop = FALSE]
  lower <- rep(-Inf, ncol(preds))
  upper <- rep(Inf, ncol(preds))
  if (k_lo >= 1) lower <- apply(block - residuals, 2, function(v) sort(v)[k_lo])
  if (k_hi <= n) upper <- apply(block + residuals, 2, function(v) sort(v)[k_hi])
  list(lower = lower, upper = upper)
}

# Core calls against the plain-R rules
set.seed(1)
s <- rnorm(19)
for (level in c(0.5, 0.8, 0.9, 0.95)) {
  stopifnot(isTRUE(all.equal(ns$.conformal_quantile(s, level), oracle_quantile(s, level),
                             tolerance = 1e-12)))
}
stopifnot(is.infinite(ns$.conformal_quantile(c(1, 2, 3), 0.95)))
stopifnot(ns$.conformal_min_size(0.5) == 1L)

# Inverted band: q = -1 turns [0, 1] into [1, 0]; both ends become 0.5.
w <- ns$.conformal_widen(c(0, 0), c(1, 4), -1)
stopifnot(all.equal(w$lower, c(0.5, 1)), all.equal(w$upper, c(0.5, 3)))
stopifnot(all.equal(ns$.conformal_midpoint(w$lower, w$upper), c(0.5, 2)))

# CV+ endpoints: folds 0 and 1 hold rows 1-2 and 3-4; two target columns.
fold_of <- c(0L, 0L, 1L, 1L)
residuals <- c(1, 2, 3, 4)
preds <- rbind(c(10, 0), c(20, 0))
got <- ns$.conformal_cv_plus_bounds(residuals, fold_of, preds, 2L, 0.5)
want <- oracle_cv_plus_bounds(residuals, fold_of, preds, 0.5)
stopifnot(isTRUE(all.equal(got$lower, want$lower, tolerance = 1e-12)),
          isTRUE(all.equal(got$upper, want$upper, tolerance = 1e-12)))
stopifnot(got$lower[1] == 9, got$upper[1] == 23, got$lower[2] == -3, got$upper[2] == 3)

# The guarantee is NULL when n is not a multiple of the fold count.
stopifnot(is.null(ns$.conformal_guarantee(101L, 5L, 0.9)))
stopifnot(abs(ns$.conformal_guarantee(100L, 10L, 0.9) -
                (1 - 0.2 - min(2 * 0.9 / 11, 0.9 / 11))) < 1e-12)

# Tuned step grid: start at the floor of 1600, double to the cap.
stopifnot(ns$.conformal_mstop_start(50L) == 1600, ns$.conformal_mstop_next(1600, 10000) == 3200,
          ns$.conformal_mstop_next(8000, 10000) == 10000, ns$.conformal_mstop_choice(0) == 1)

# Bad inputs are refused with a message.
bad <- glmboost(y ~ x, data = data.frame(x = runif(40), y = rnorm(40)),
                control = boost_control(mstop = 10))
err <- tryCatch(conformal_interval(bad, level = 1.2), error = function(e) e)
stopifnot(inherits(err, "error"), grepl("level must be in", conditionMessage(err)))

# End to end on a glmboost fit
set.seed(7)
n <- 240
d <- data.frame(x1 = runif(n, -2, 2), x2 = rnorm(n))
d$y <- sin(d$x1) + 0.5 * d$x2 + rt(n, df = 4) * (0.2 + 0.4 * (d$x1 > 0))
train <- d[1:180, ]
test <- d[181:240, ]
fit <- glmboost(y ~ x1 + x2, data = train, control = boost_control(mstop = 40, nu = 0.1))

for (method in c("split", "cv+", "cqr")) {
  ci <- conformal_interval(fit, newdata = test[, c("x1", "x2")], level = 0.9,
                           method = method, seed = 3, n_folds = 5L, mstop = 25L)
  stopifnot(inherits(ci, "mboostrs_conformal"), identical(ci$method, method),
            length(ci$lower) == nrow(test),
            all(ci$lower <= ci$upper + 1e-12),
            all(ci$predicted >= ci$lower - 1e-12), all(ci$predicted <= ci$upper + 1e-12),
            length(ci$fitted_mean) == nrow(test))
  covered <- mean(test$y >= ci$lower & test$y <= ci$upper)
  stopifnot(covered >= 0.75)
}

# Split: constant width, and the quantile is the rule applied to the residuals.
split_ci <- conformal_interval(fit, level = 0.8, method = "split", seed = 2, mstop = 25L)
stopifnot(length(unique(round(split_ci$upper - split_ci$lower, 10))) == 1L,
          split_ci$guaranteed_coverage == 0.8, split_ci$n_calibration == 90L)

# CV+ records the equal-fold set aside and the guarantee of the balanced bound.
cvp <- conformal_interval(fit, level = 0.9, method = "cv+", seed = 4, n_folds = 7L, mstop = 25L)
stopifnot(cvp$n_set_aside == 180 %% 7, cvp$n_folds == 7L)

# A held-out calibration set is used as fitted, with no refit.
cal <- d[181:240, ]
held <- conformal_interval(fit, newdata = d[1:20, ], level = 0.9, calibration = cal)
stopifnot(held$n_calibration == 60L, grepl("supplied by the caller", held$calibration_source))

# The same seed gives the same split.
a <- conformal_interval(fit, level = 0.9, method = "cqr", seed = 5, mstop = 25L)
b <- conformal_interval(fit, level = 0.9, method = "cqr", seed = 5, mstop = 25L)
stopifnot(identical(a$lower, b$lower), identical(a$upper, b$upper))

# Distributional fits and mstop = "tune" are not in R
stopifnot(inherits(tryCatch(conformal_interval(fit, mstop = "tune"), error = identity), "error"))

cat("test_conformal_core.R: all checks passed\n")
