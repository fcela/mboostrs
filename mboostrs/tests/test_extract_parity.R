# extract() reports mboost's design, penalty, lambda and df for each selected
# learner. It used to return flattened unnamed designs, -1 placeholders for
# the penalty and lambda, the column count as the df, unit weights for a
# weighted fit, a weighted learner's df from tr(2H - H'H) instead of
# 2 tr(H) - tr(H^2), and every learner rather than the selected ones.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping extract parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(mboost))

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(7); n <- 200
d <- data.frame(x = runif(n, -1, 1), z = rnorm(n), w = runif(n))
d$y <- sin(2 * d$x) + 0.5 * d$z + rnorm(n, sd = .4)
wt <- runif(n, 0.5, 2)
up <- mboost::gamboost(y ~ bbs(x) + bols(z) + bbs(w, df = 3), data = d, weights = wt,
                       control = mboost::boost_control(mstop = 80))
ours <- mboostrs::gamboost(y ~ bbs(x) + bols(z) + bbs(w, df = 3), data = d, weights = wt,
                           control = mboostrs::boost_control(mstop = 80))
for (what in c("design", "penalty", "lambda", "df")) {
  a <- extract(up, what = what); b <- extract(ours, what = what)
  ok(length(a) == length(b), paste(what, ": a different set of learners"))
  for (i in seq_along(a)) {
    diff <- max(abs(as.matrix(unclass(a[[i]])) - as.matrix(unclass(b[[i]]))))
    tol <- if (what == "lambda") 1e-7 * max(1, abs(as.numeric(a[[i]]))) else 1e-9
    ok(diff < tol, sprintf("%s of learner %d differs from mboost by %.1e", what, i, diff))
  }
}
# Both packages report the weights the fit used (rescaled to mean 1).
ok(max(abs(extract(ours, what = "weights") - extract(up, what = "weights"))) < 1e-14,
   "extract(weights) differs from mboost")
ok(identical(unname(extract(ours, what = "variable.names")), c("x", "z", "w")), "variable names")
cat("extract parity (R): PASS\n")
