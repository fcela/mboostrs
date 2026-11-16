# Factor by= on bbs() and bols() against mboost (issue #164).
suppressMessages(library(mboostrs))
have_mboost <- requireNamespace("mboost", quietly = TRUE)
# mboost resolves its base-learners from the formula's environment, so it is
# attached; mboostrs functions are called by their namespace below.
if (have_mboost) suppressMessages(library(mboost))
set.seed(4); n <- 300
d <- data.frame(x = runif(n), w = rnorm(n), g = factor(sample(c("a", "b", "c"), n, TRUE)))
d$y <- sin(3 * d$x) * (d$g == "b") + d$x * (d$g == "c") + 0.3 * d$w + rnorm(n, sd = 0.2)
nd <- d[1:20, ]

cases <- list(
  # df = 4 equals the stacked null space (2 levels x 2): lambda is unbounded,
  # mboost warns, and the two searches stop at different large lambdas.
  list(f = y ~ bbs(x, by = g) + bols(w), tol = 1e-4),
  list(f = y ~ bbs(x) + bbs(x, by = g, df = 6), tol = 1e-8),
  list(f = y ~ bols(x, by = g) + bols(w), tol = 1e-10),
  list(f = y ~ bols(x, by = g, df = 2) + bbs(w), tol = 1e-8),
  list(f = y ~ bols(x, by = g, lambda = 5) + bols(w), tol = 1e-10)
)
for (case in cases) {
  ours <- suppressWarnings(mboostrs::gamboost(case$f, data = d,
                                              control = mboostrs::boost_control(mstop = 100)))
  # Prediction on the training rows reproduces the fit.
  stopifnot(max(abs(predict(ours, newdata = d) - ours$fitted)) < 1e-10)
  if (have_mboost) {
    up <- suppressWarnings(mboost::gamboost(case$f, data = d,
                                            control = mboost::boost_control(mstop = 100)))
    gap <- max(abs(fitted(up) - ours$fitted))
    pgap <- max(abs(predict(up, newdata = nd) - predict(ours, newdata = nd)))
    if (gap > case$tol || pgap > case$tol)
      stop(deparse(case$f), ": fitted gap ", gap, ", predict gap ", pgap)
  }
}

# An unseen level is refused, and df below the stacked null space is refused.
fit <- mboostrs::gamboost(y ~ bols(x, by = g), data = d,
                          control = mboostrs::boost_control(mstop = 20))
bad <- nd; bad$g <- factor("z")
msg <- tryCatch(predict(fit, newdata = bad), error = conditionMessage)
stopifnot(grepl("not present when the model was fitted", msg))
msg <- tryCatch(mboostrs::gamboost(y ~ bbs(x, by = g, df = 3), data = d),
                error = conditionMessage)
stopifnot(grepl("below the 4-dimensional", msg))
cat("test_factor_by_parity.R: OK\n")
