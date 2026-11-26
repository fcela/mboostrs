# Parity check: our cvrisk_glmboostLSS vs R's gamboostLSS::cvrisk (cvrisk.mboostLSS).
# NB: gamboostLSS::glmboostLSS re-evaluates its `families` argument from the call,
# so the family constructors are inlined (not passed via a variable).
# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping cvrisk LSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

cmp <- function(cvr, mine, what) {
  md <- max(abs(as.matrix(cvr) - mine))
  if (md >= 1e-6) stop(sprintf("%s cvrisk mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-12s maxdiff=%.2e\n", what, md))
}
grid_of <- function(cvr) do.call(rbind, lapply(strsplit(colnames(cvr), ","), as.integer))

# --- GaussianLSS ---
set.seed(1); n <- 120; x1 <- rnorm(n); x2 <- rnorm(n)
y <- rnorm(n, 1 + 0.8 * x1, exp(0.3 * x2))
d <- data.frame(y = y, x1 = x1, x2 = x2)
m <- gamboostLSS::glmboostLSS(y ~ x1 + x2, data = d, families = gamboostLSS::GaussianLSS(),
                             control = mboost::boost_control(mstop = 30, nu = 0.1))
folds <- cv(model.weights(m), type = "kfold", B = 5)
cvr <- cvrisk(m, folds = folds, trace = FALSE)
mine <- mboostrs::cvrisk_glmboostLSS(y ~ x1 + x2, data = d, families = mboostrs::GaussianLSS(),
                                    control = mboostrs::boost_control(mstop = 30, nu = 0.1),
                                    folds = as.matrix(folds), grid = grid_of(cvr))
cmp(cvr, mine, "GaussianLSS")

# --- GammaLSS ---
set.seed(1); x1 <- rnorm(n); x2 <- rnorm(n)
mu <- exp(0.5 + 0.3 * x1); sig <- exp(0.2 + 0.2 * x2)
y <- rgamma(n, shape = sig, scale = mu / sig)
d <- data.frame(y = y, x1 = x1, x2 = x2)
m <- gamboostLSS::glmboostLSS(y ~ x1 + x2, data = d, families = gamboostLSS::GammaLSS(),
                             control = mboost::boost_control(mstop = 30, nu = 0.1))
folds <- cv(model.weights(m), type = "kfold", B = 5)
cvr <- cvrisk(m, folds = folds, trace = FALSE)
mine <- mboostrs::cvrisk_glmboostLSS(y ~ x1 + x2, data = d, families = mboostrs::GammaLSS(),
                                    control = mboostrs::boost_control(mstop = 30, nu = 0.1),
                                    folds = as.matrix(folds), grid = grid_of(cvr))
cmp(cvr, mine, "GammaLSS")

cat("cvrisk glmboostLSS R-binding parity: PASS\n")
