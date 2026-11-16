# Penalization drift: a long path warns, a short one is quiet (issue #94).
suppressMessages(library(mboostrs))
set.seed(1); n <- 400
d <- data.frame(x0 = runif(n, -2, 2), x1 = runif(n, -2, 2))
d$y <- sin(2 * d$x0) + 0.5 * d$x1^2 + rnorm(n, sd = 0.3)

quiet <- withCallingHandlers(
  gamboost(y ~ bbs(x0) + bbs(x1), data = d, control = boost_control(mstop = 200)),
  warning = function(w) stop("unexpected warning: ", conditionMessage(w)))
stopifnot(is.null(quiet$penalty_drift))

msg <- NULL
long <- withCallingHandlers(
  gamboost(y ~ bbs(x0) + bbs(x1) + bols(x1), data = d,
           control = boost_control(mstop = 10000)),
  warning = function(w) { msg <<- conditionMessage(w); invokeRestart("muffleWarning") })
stopifnot(grepl("undone much of the penalty", msg), grepl("bbs\\(x0\\)", msg))
pd <- long$penalty_drift
stopifnot(is.data.frame(pd), sum(pd$selections) == 10000,
          all(pd$boosted_df[pd$selections > 0] > pd$single_step_df[pd$selections > 0]),
          all(pd$unpenalized_df[pd$learner != "bols(x1)"] == 24),
          max(pd$drift_share) >= 1 / 3)
cat("test_penalty_drift.R: OK\n")
