# R `predict(newdata=)` works for centered smooths, and reproduces the fit.
#
# `bbs(center = TRUE)` removes the penalty's null space, so the fitted design
# has fewer columns than the raw basis and the coefficients live in that
# reduced space. The R prediction bridge rebuilt the raw basis, found the
# counts disagreed, and refused -- correctly, because multiplying them would
# have been silently wrong, but it left `predict(newdata=)` unavailable for a
# default `bbs(center = TRUE)` and for every `bspatial`, whose `center`
# defaults to TRUE.
#
# The transform depends only on the penalty, not on the data, so applying it
# to a basis built at new x values lands in the same reduced space. Two
# different transforms are in play and using the wrong one is the trap: a
# one-dimensional `bbs` centers by the difference matrix `D'(DD')^-1`, while
# `bspatial` centers by the spectral decomposition of the tensor penalty.
# Rebuilding a `bbs` with the spectral transform *runs* and is wrong by an
# amount that reads as a poor fit -- measured at 0.80 on an in-sample replay
# where the correct transform gives 4e-16.
#
# So the check here is not "prediction succeeds" but "prediction on the
# training data reproduces `fitted()`". A wrong transform passes the first and
# fails the second.
#@validates r.predict.centered_newdata
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(1)
n <- 150
d <- data.frame(x = rnorm(n), x1 = runif(n), x2 = runif(n), g = rnorm(n))
d$y <- sin(d$x) + 2 * d$x1 + rnorm(n, sd = 0.3)

replay_error <- function(formula, label, mstop = 40) {
  fit <- gamboost(formula, data = d, control = boost_control(mstop = mstop, nu = 0.1))
  predicted <- predict(fit, newdata = d)
  error <- max(abs(as.numeric(predicted) - as.numeric(fitted(fit))))
  ok(error < 1e-10,
     sprintf("%s: predicting on the training data is off from fitted() by %.3e",
             label, error))
  cat(sprintf("  %-44s replay error %.2e\n", label, error))
  invisible(fit)
}

cat("centered new-data prediction:\n")

# The cases that used to refuse.
replay_error(y ~ bbs(x, center = TRUE, df = 3), "bbs(center=TRUE)")
replay_error(y ~ bspatial(x1, x2, df = 5), "bspatial (center defaults to TRUE)")
replay_error(y ~ bbs(x, center = TRUE, df = 3) + bols(x1),
             "bbs(center=TRUE) + bols")
replay_error(y ~ bbs(x, center = TRUE, df = 3) + bbs(x1, center = TRUE, df = 3),
             "two centered smooths")

# The cases that already worked, kept so the fix cannot break them.
replay_error(y ~ bbs(x, center = FALSE, df = 3), "bbs(center=FALSE)")
replay_error(y ~ bspatial(x1, x2, df = 5, center = FALSE), "bspatial(center=FALSE)")
replay_error(y ~ bols(x) + bols(x1), "bols menu")
replay_error(y ~ bmono(x, df = 3), "bmono")
replay_error(y ~ bbs(x, df = 3) + bspatial(x1, x2, df = 5), "bbs + bspatial")

# Genuine new data, not just the training rows: the values must be finite and
# must lie in a plausible range for a fit of this response.
fit <- gamboost(y ~ bbs(x, center = TRUE, df = 3), data = d,
                control = boost_control(mstop = 40, nu = 0.1))
grid <- data.frame(x = seq(min(d$x), max(d$x), length.out = 25))
predicted <- predict(fit, newdata = grid)
ok(length(predicted) == 25L, "one prediction per new row")
ok(all(is.finite(predicted)), "new-data predictions must be finite")
ok(max(abs(predicted)) < 10 * max(abs(d$y)),
   "new-data predictions should be on the response's scale")

# A smooth predicted over a grid should vary: a transform that collapsed the
# basis would give a constant, which the finiteness check alone would pass.
ok(diff(range(predicted)) > 0.1,
   sprintf("a fitted smooth should vary over its range; got a span of %.4f",
           diff(range(predicted))))

cat("centered new-data prediction (R): PASS\n")
