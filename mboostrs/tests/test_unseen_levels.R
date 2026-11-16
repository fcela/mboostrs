# predict(unseen_levels = ) and factor coding in R gamboost and gamboostLSS
# (issue #184). New data are coded against the training labels: brandom
# read a factor's own integer codes, so new data holding levels "b" and "e"
# was predicted as "a" and "b", and a level the fit never saw was predicted
# as some other level. gamboostLSS could not predict new data with a btree,
# and its trees split factor codes as numbers.
suppressMessages(library(mboostrs))
set.seed(1); n <- 300
d <- data.frame(g = factor(sample(letters[1:5], n, TRUE)), x = runif(n), z = rnorm(n))
d$y <- c(a = 0, b = 1, c = -1, d = 0.5, e = 2)[as.character(d$g)] + sin(3 * d$x) +
  0.5 * d$z * (d$g == "b") + rnorm(n, sd = 0.3)
formulas <- list(y ~ brandom(g) + bbs(x), y ~ bols(g) + bbs(x), y ~ bols(z, by = g) + bbs(x),
                 y ~ btree(g, x, maxdepth = 3) + bbs(x), y ~ brandom(g, by = z) + bbs(x))
check <- function(P, label) {
  full <- P(data.frame(g = factor(c("b", "e"), levels = letters[1:5]), x = 0.3, z = 1))
  subset <- P(data.frame(g = factor(c("b", "e")), x = 0.3, z = 1))
  if (!isTRUE(all.equal(full, subset))) stop(label, ": a factor holding a subset of the levels")
  per <- sapply(letters[1:5], function(l) P(data.frame(g = l, x = 0.3, z = 1)))
  average <- P(data.frame(g = "zz", x = 0.3, z = 1), unseen_levels = "average")
  if (abs(average - mean(per)) > 1e-9) stop(label, ": average ", average, " vs mean ", mean(per))
  zero <- P(data.frame(g = "zz", x = 0.3, z = 1), unseen_levels = "zero_effect")
  stopifnot(is.finite(zero))
  refused <- tryCatch({ P(data.frame(g = "zz", x = 0.3, z = 1)); FALSE },
                      error = function(e) grepl("zz|did not see", conditionMessage(e)))
  if (!refused) stop(label, ": an unseen level should be refused by default")
}
for (f in formulas) {
  m <- gamboost(f, data = d, control = boost_control(mstop = 100))
  check(function(nd, ...) predict(m, newdata = nd, ...), paste("gamboost", deparse(f)))
  lss <- gamboostLSS(list(mu = f, sigma = y ~ bols(x)), data = d, families = GaussianLSS(),
                     control = boost_control(mstop = 60))
  check(function(nd, ...) predict(lss, newdata = nd, parameter = "mu", ...),
        paste("gamboostLSS", deparse(f)))
}
cat("test_unseen_levels.R: OK\n")
