# Parity check: our stabsel_glmboostLSS vs R's gamboostLSS::stabsel (non-cyclic).
# Shared subsampling folds; selected set matches exactly, max selection
# frequencies match within a near-tie tolerance (non-cyclic flat-objective
# ambiguity affects a couple of subsamples).
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping stabsel LSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(1); n <- 100; p <- 8
X <- matrix(rnorm(n * p), n, p); colnames(X) <- paste0("x", 1:p)
y <- rnorm(n, 1 + 0.8 * X[, 1], exp(0.3 * X[, 2])); d <- data.frame(y = y, X)
ref <- gamboostLSS::glmboostLSS(y ~ ., data = d, families = gamboostLSS::GaussianLSS(),
                               control = mboost::boost_control(mstop = 50, nu = 0.1),
                               method = "noncyclic")
set.seed(7); folds <- (cv(model.weights(ref), type = "subsampling", B = 20) > 0) * 1
s <- stabsel(ref, folds = folds, q = 4, PFER = 2, sampling.type = "MB", B = 20)
mine <- mboostrs::stabsel_glmboostLSS(
  y ~ ., data = d, families = mboostrs::GaussianLSS(),
  control = mboostrs::boost_control(mstop = 50, nu = 0.1),
  folds = folds, q = 4, PFER = 2, sampling_type = "MB")

stopifnot(setequal(names(s$selected), mine$selected))
md <- max(abs(as.numeric(s$max) - mine$max[names(s$max)]))
if (md >= 0.15) stop(sprintf("stabsel max_phat mismatch: maxdiff=%.3f", md))
cat(sprintf("OK stabsel selected set matches; max_phat maxdiff=%.3f\n", md))
cat("stabsel glmboostLSS R-binding parity: PASS\n")
