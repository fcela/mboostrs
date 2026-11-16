# A multi-column btree chooses its split column as partykit's ctree does in
# mboost (selection = "ctree", the default), so fitted values agree with
# mboost::mboost(y ~ btree(...)).
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping btree ctree parity")
  quit(save = "no")
}
suppressWarnings(suppressMessages({library(mboost); library(mboostrs)}))
set.seed(1); n <- 400
d <- data.frame(x1 = runif(n), x2 = runif(n), x3 = sample(1:40, n, TRUE) / 40)
d$y <- sin(4 * d$x1) + (d$x2 > 0.5) + rnorm(n, 0, 0.3)
r <- mboost::mboost(y ~ btree(x1, x2, x3), data = d,
                    control = mboost::boost_control(mstop = 50))
m <- mboostrs::gamboost(y ~ btree(x1, x2, x3), data = d,
                        control = mboostrs::boost_control(mstop = 50))
stopifnot(max(abs(fitted(m) - fitted(r))) < 1e-10)
m2 <- mboostrs::gamboost(y ~ btree(x1, x2, x3, maxdepth = 2), data = d,
                         control = mboostrs::boost_control(mstop = 30))
r2 <- mboost::mboost(y ~ btree(x1, x2, x3, tree_controls = partykit::ctree_control(
  teststat = "quad", testtype = "Teststatistic", mincriterion = 0, minsplit = 10,
  minbucket = 4, maxdepth = 2, saveinfo = FALSE)), data = d,
  control = mboost::boost_control(mstop = 30))
stopifnot(max(abs(fitted(m2) - fitted(r2))) < 1e-10)
cat("btree ctree parity (R): PASS\n")
