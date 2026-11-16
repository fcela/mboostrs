# Numeric by= in the R frontend (issue #160): bbs(x, by = z) and
# bols(x, by = z) against mboost, fitted values and new-data predictions,
# including the menu rule that gives every bols its own intercept once a
# learner without the constant (a by= term) is present.
library(mboostrs)
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping"); quit(save = "no")
}
suppressMessages(library(mboost))
set.seed(6); n <- 200
d <- data.frame(x = runif(n), z = rnorm(n), w = runif(n),
                f = factor(sample(letters[1:3], n, TRUE)))
d$y <- sin(3 * d$x) * d$z + d$w + rnorm(n, 0, 0.3)
nd <- d[1:30, ]
for (fo in c("y ~ bbs(x, by = z) + bols(w)", "y ~ bols(x, by = z) + bols(w)",
             "y ~ bbs(x, by = z) + bbs(w) + bols(z)",
             "y ~ bbs(x, by = z, lambda = 5) + bols(w)")) {
  o <- mboostrs::gamboost(as.formula(fo), data = d, control = mboostrs::boost_control(mstop = 100))
  u <- mboost::gamboost(as.formula(fo), data = d, control = mboost::boost_control(mstop = 100))
  gap <- max(abs(fitted(o) - fitted(u)), abs(predict(o, newdata = nd) - predict(u, newdata = nd)))
  if (gap > 1e-8) stop(fo, ": gap ", gap)
}
# A factor by= is fitted (test_factor_by_parity.R), and a numeric by= on
# bmono, bns, bspatial, brandom, bmrf and btree (test_by_other_learners.R).
# mboost's btree takes only a 0/1 or two-level modifier; a continuous one
# fits f(x) * z by least squares here (tests/test_r_by_other_learners.py
# compares it with Python).
# gamboostLSS fits by= too; test_lss_by_parity.R compares it with gamboostLSS.
cat("test_by_parity.R: OK\n")
