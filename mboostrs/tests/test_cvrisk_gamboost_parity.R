# Parity check: our cvrisk.mboostrs_gamboost vs R mboost::cvrisk, with shared
# folds. Regression for the R binding computing the out-of-bag risk from its
# own per-family loss table over row-subset refits: the losses differed from
# the fitted family's risk (squared error stood in for every survival and
# Multinomial loss), the per-row mean ignored the out-of-bag weight total, and
# the grid started at mstop 1 rather than the offset state.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping cvrisk gamboost parity check")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost)))

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-34s maxdiff=%.2e\n", what, md))
}

set.seed(7)
n <- 120
x1 <- runif(n, -2, 2); x2 <- rnorm(n)
d <- data.frame(x1 = x1, x2 = x2,
                y = sin(x1) + 0.5 * x2 + rnorm(n, sd = 0.3),
                count = rpois(n, lambda = exp(0.4 + 0.3 * x1)))
mstop <- 60; nu <- 0.1
ctrl  <- mboostrs::boost_control(mstop = mstop, nu = nu)
rctrl <- mboost::boost_control(mstop = mstop, nu = nu)

set.seed(99)
F <- mboost::cv(rep(1, n), type = "kfold", B = 5)

# ── Gaussian: bbs + bols ─────────────────────────────────────────────
mg <- mboostrs::gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d,
                         family = mboostrs::Gaussian(), control = ctrl)
rg <- mboost::gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d,
                       control = rctrl)
cv_ours <- mboostrs::cvrisk(mg, folds = F)
cv_r <- mboost::cvrisk(rg, folds = F, papply = lapply)
r_matrix <- as.matrix(cv_r)

chk(cv_ours$risk_matrix, r_matrix, "gaussian per-fold OOB risk")
chk(cv_ours$aggregate, colMeans(r_matrix), "gaussian aggregate risk path")
if (cv_ours$optimal_mstop != mboost::mstop(cv_r)) {
  stop(sprintf("gaussian optimal mstop mismatch: %d vs R %d",
               cv_ours$optimal_mstop, mboost::mstop(cv_r)))
}
cat(sprintf("OK %-34s %d\n", "gaussian optimal mstop", cv_ours$optimal_mstop))

# ── Poisson: the old loss table dropped log(y!) and the weight total ──
mp <- mboostrs::gamboost(count ~ bbs(x1, df = 4), data = d,
                         family = mboostrs::Poisson(), control = ctrl)
rp <- mboost::gamboost(count ~ bbs(x1, df = 4), data = d,
                       family = mboost::Poisson(), control = rctrl)
cvp_ours <- mboostrs::cvrisk(mp, folds = F)
cvp_r <- mboost::cvrisk(rp, folds = F, papply = lapply)
chk(cvp_ours$risk_matrix, as.matrix(cvp_r), "poisson per-fold OOB risk")
cat(sprintf("OK %-34s %d\n", "poisson optimal mstop", cvp_ours$optimal_mstop))

# ── The risk path starts at the offset state ─────────────────────────
stopifnot(ncol(cv_ours$risk_matrix) == mstop + 1L)
stopifnot(length(cv_ours$aggregate) == mstop + 1L)

# ── Refusals ─────────────────────────────────────────────────────────
expect_error <- function(expr, pattern) {
  err <- tryCatch({ force(expr); NULL }, error = identity)
  if (!inherits(err, "error")) stop("expected an error matching: ", pattern)
  if (!grepl(pattern, conditionMessage(err), ignore.case = TRUE)) {
    stop("error message did not match ", pattern, ": ", conditionMessage(err))
  }
  cat(sprintf("OK %-34s\n", paste("refused:", pattern)))
}

# Every row in bag leaves nothing to evaluate.
expect_error(mboostrs::cvrisk(mg, folds = matrix(1, nrow = n, ncol = 2)), "OOB")
# A fold matrix for a different sample size.
expect_error(mboostrs::cvrisk(mg, folds = matrix(1, nrow = n - 1, ncol = 2)),
             "rows but the model has")
# ── Non-constant model weights ───────────────────────────────────────
# Upstream refits each fold on the fold column as given and scores held-out
# rows with the model's weights. A 0/1 mask here is multiplied by the
# weights, so the result equals upstream's on folds that carry the weights.
wv <- rep(c(1, 2), length.out = n)
mw <- mboostrs::gamboost(y ~ bbs(x1, df = 4), data = d,
                         family = mboostrs::Gaussian(), control = ctrl,
                         weights = wv)
rw <- mboost::gamboost(y ~ bbs(x1, df = 4), data = d, weights = wv,
                       control = rctrl)
chk(mboostrs::cvrisk(mw, folds = F)$risk_matrix,
    as.matrix(mboost::cvrisk(rw, folds = F * wv, papply = lapply)),
    "weighted per-fold OOB risk")

cat("cvrisk gamboost parity: PASS\n")
