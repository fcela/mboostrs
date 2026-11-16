# The R frontend fitted bols(f) on a factor as a slope on its integer codes,
# and bspatial with knots = 10, center = TRUE instead of mboost's 20 / FALSE.
library(mboostrs)
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping"); quit(save = "no")
}
# mboost's formula interface finds bbs() etc. on the search path; mboostrs
# calls below are namespace-qualified.
suppressMessages(library(mboost))
set.seed(3); n <- 200
d <- data.frame(x = runif(n), z = rnorm(n), g = factor(sample(1:4, n, TRUE)))
d$y <- sin(3 * d$x) + 0.5 * d$z + as.integer(d$g) / 4 + rnorm(n, 0, 0.3)
d$gc <- as.character(d$g)
nd <- d[1:25, ]
check <- function(ours, theirs, label, tol = 1e-8) {
  gap <- max(abs(fitted(ours) - fitted(theirs)))
  pgap <- max(abs(predict(ours, newdata = nd) - predict(theirs, newdata = nd)))
  if (gap > tol || pgap > tol) stop(label, ": fitted gap ", gap, ", predict gap ", pgap)
}
O <- function(f) mboostrs::gamboost(f, data = d, control = mboostrs::boost_control(mstop = 80))
B <- function(f) mboost::gamboost(f, data = d, control = mboost::boost_control(mstop = 80))
check(O(y ~ bols(g) + bbs(x)), B(y ~ bols(g) + bbs(x)), "bols(factor)")
check(O(y ~ bols(gc) + bbs(x)), B(y ~ bols(g) + bbs(x)), "bols(character)")
check(O(y ~ bols(g, intercept = FALSE) + bbs(x)), B(y ~ bols(g, intercept = FALSE) + bbs(x)),
      "bols(factor, intercept = FALSE)")
check(O(y ~ bols(g, lambda = 2) + bbs(x)), B(y ~ bols(g, lambda = 2) + bbs(x)), "bols(factor, lambda)")
check(O(y ~ bspatial(x, z)), B(y ~ bspatial(x, z)), "bspatial defaults", tol = 1e-6)
bad <- nd; bad$g <- factor(rep("9", nrow(bad)))
msg <- tryCatch({ predict(O(y ~ bols(g) + bbs(x)), newdata = bad); "" },
                error = function(e) conditionMessage(e))
stopifnot(grepl("levels the fit did not see", msg))
cat("test_factor_bols_bspatial_parity.R: OK\n")
