# Parity check: our cvrisk.mboostrs_glmboost vs R mboost::cvrisk, with shared
# folds. Regression for the R binding previously recording the per-fold *inbag*
# risk path (always selecting near-max mstop) and reusing the full-sample offset
# across folds.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping cvrisk glmboost parity check")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost)))

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-28s maxdiff=%.2e\n", what, md))
}

set.seed(1)
n <- 120
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
y <- 1 + 2 * x1 + rnorm(n)               # x2, x3 irrelevant
d <- data.frame(x1 = x1, x2 = x2, x3 = x3, y = y)
mstop <- 150; nu <- 0.1

# Shared fold matrix.
set.seed(99)
F <- mboost::cv(rep(1, n), type = "kfold", B = 10)

# R mboost reference.
mr <- mboost::glmboost(y ~ x1 + x2 + x3, data = d, family = mboost::Gaussian(),
                       center = TRUE,
                       control = mboost::boost_control(mstop = mstop, nu = nu))
cvr <- mboost::cvrisk(mr, folds = F, papply = lapply)
Rmat <- as.matrix(cvr)                   # (B, mstop+1), normalized OOB risk

# Ours, same folds.
m <- mboostrs::glmboost(y ~ x1 + x2 + x3, data = d, family = mboostrs::Gaussian(),
                        control = mboostrs::boost_control(mstop = mstop, nu = nu))
cv <- mboostrs::cvrisk(m, folds = F)

chk(cv$risk_matrix, Rmat, "per-fold OOB risk matrix")
chk(cv$aggregate, colMeans(Rmat), "aggregate (colMeans) risk path")
if (cv$optimal_mstop != mboost::mstop(cvr)) {
  stop(sprintf("optimal mstop mismatch: %d vs R %d", cv$optimal_mstop, mboost::mstop(cvr)))
}
cat(sprintf("OK %-28s %d\n", "optimal mstop", cv$optimal_mstop))

# Sanity: the optimum is interior, not pinned at max mstop (the old inbag bug
# always selected near the maximum).
if (cv$optimal_mstop >= mstop) stop("optimal mstop pinned at maximum (inbag-risk regression)")

# R1: negative / zero mstop must error, not wrap to ~4e9 iterations.
err <- tryCatch({
  mboostrs::glmboost(y ~ x1, data = d, family = mboostrs::Gaussian(),
                     control = mboostrs::boost_control(mstop = -5, nu = nu))
  FALSE
}, error = function(e) TRUE)
if (!err) stop("negative mstop did not error")
cat("OK negative mstop rejected\n")

cat("cvrisk glmboost R-binding parity: PASS\n")
