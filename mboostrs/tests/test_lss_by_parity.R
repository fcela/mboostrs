# by= in R gamboostLSS against gamboostLSS (issue #164): a numeric by= on
# bbs() and bols(), a factor by= on bbs() and bols(), with in-sample and
# new-data link predictions.
suppressMessages(library(mboostrs))
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  cat("test_lss_by_parity.R: skipped (gamboostLSS not installed)\n")
  quit(save = "no")
}
# gamboostLSS resolves its base-learners from the formula's environment.
suppressMessages(library(gamboostLSS))
set.seed(2); n <- 400
d <- data.frame(x = runif(n), z = rnorm(n), w = runif(n),
                g = factor(sample(c("a", "b", "c"), n, TRUE)))
d$y <- rnorm(n, mean = sin(3 * d$x) * d$z + (d$g == "b"), sd = exp(0.3 * d$w))
# Rows from the middle and end, with the by= modifier changed: new data that
# reused the training rows' z would differ (it did, by 8.5, for bbs by=).
L <- 8
K <- matrix(0, L, L); for (i in 1:(L - 1)) K[i, i + 1] <- K[i + 1, i] <- -1
diag(K) <- -rowSums(K); dimnames(K) <- list(as.character(1:L), as.character(1:L))
d$r <- factor(sample(1:L, n, TRUE), levels = 1:L)
d$b <- rbinom(n, 1, 0.5)
nd <- d[c(200:210, 390:399), ]
nd$z <- 2 * nd$z
has_bayesx <- requireNamespace("BayesX", quietly = TRUE) &&
  requireNamespace("foreign", quietly = TRUE)
cases <- list(
  list(mu = y ~ bbs(x, by = z) + bols(w), sigma = y ~ bols(w)),
  list(mu = y ~ bols(x, by = z) + bbs(w), sigma = y ~ bbs(w)),
  list(mu = y ~ bbs(x, by = g, df = 6) + bols(w), sigma = y ~ bols(w)),
  list(mu = y ~ bols(x, by = g) + bols(w), sigma = y ~ bols(x, by = g, df = 2)),
  list(mu = y ~ brandom(g, by = z) + bols(w), sigma = y ~ bols(w)),
  # The LSS binding dropped the modifier for these (2.5 and 1.2 from
  # gamboostLSS) until their arms read by_values.
  list(mu = y ~ bmono(x, by = z) + bols(w), sigma = y ~ bols(w)),
  list(mu = y ~ bspatial(x, w, by = z, df = 6) + bols(z), sigma = y ~ bols(w)),
  # A bmrf or btree by= was refused at new-data prediction by a guard on the
  # old $by field.
  # bmrf needs BayesX and BayesX needs foreign; the case is dropped without them.
  if (has_bayesx) list(mu = y ~ bmrf(r, bnd = K, by = z) + bols(w), sigma = y ~ bols(w)),
  list(mu = y ~ btree(x, by = b) + bols(w), sigma = y ~ bols(w)))
cases <- Filter(Negate(is.null), cases)
for (f in cases) {
  ours <- mboostrs::gamboostLSS(f, data = d, families = mboostrs::GaussianLSS(),
                                control = mboostrs::boost_control(mstop = 100))
  up <- gamboostLSS::gamboostLSS(f, data = d, families = gamboostLSS::GaussianLSS(),
                                 control = mboost::boost_control(mstop = 100))
  gap <- max(abs(unlist(predict(up, type = "link")) - unlist(predict(ours, type = "link"))))
  ngap <- max(abs(unlist(predict(up, newdata = nd, type = "link")) -
                  unlist(predict(ours, newdata = nd, type = "link"))))
  # bspatial differs from upstream at the 1e-6 scale without by= as well
  # (its df calibration; test_factor_bols_bspatial_parity.R).
  tol <- if (any(grepl("bspatial", deparse(f$mu)))) 5e-6 else 1e-10
  if (gap > tol || ngap > tol)
    stop(deparse(f$mu), " | ", deparse(f$sigma), ": gap ", gap, ", new-data gap ", ngap)
}
cat("test_lss_by_parity.R: OK\n")
