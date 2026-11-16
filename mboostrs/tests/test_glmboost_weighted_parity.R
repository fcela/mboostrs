#@validates parity.r.gaussian
# Parity check: weighted glmboost fits vs R mboost. Regression for
#  - AdaExp gradient incorrectly multiplying by the weight (B1), and
#  - unweighted column centering at fit time vs weighted at predict time (B2).
# Both only manifest with non-uniform weights.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping weighted glmboost parity check")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost)))

chk <- function(a, b, what, tol = 1e-8) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-32s maxdiff=%.2e\n", what, md))
}

set.seed(7)
n <- 150
x1 <- rnorm(n); x2 <- rnorm(n)
w <- runif(n, 0.2, 4.0)                    # non-uniform weights
ctrl_m <- mboostrs::boost_control(mstop = 80, nu = 0.1)
ctrl_r <- mboost::boost_control(mstop = 80, nu = 0.1)

# --- Gaussian, weighted (B2: weighted centering) ---
yg <- 1 + 2 * x1 - 0.5 * x2 + rnorm(n)
dg <- data.frame(x1 = x1, x2 = x2, y = yg)
rg <- mboost::glmboost(y ~ x1 + x2, data = dg, family = mboost::Gaussian(),
                       weights = w, center = TRUE, control = ctrl_r)
mg <- mboostrs::glmboost(y ~ x1 + x2, data = dg, family = mboostrs::Gaussian(),
                         weights = w, control = ctrl_m)
chk(fitted(mg), as.numeric(fitted(rg)), "Gaussian weighted fitted")

# --- AdaExp, weighted (B1: gradient excludes weight) ---
lin <- 0.8 * x1 - 0.5 * x2
y01 <- rbinom(n, 1, plogis(2 * lin))
dR <- data.frame(x1 = x1, x2 = x2, y = factor(y01))   # R AdaExp wants a factor
dM <- data.frame(x1 = x1, x2 = x2, y = 2 * y01 - 1)    # ours uses -1/+1
ra <- mboost::glmboost(y ~ x1 + x2, data = dR, family = mboost::AdaExp(),
                       weights = w, control = ctrl_r)
ma <- mboostrs::glmboost(y ~ x1 + x2, data = dM, family = mboostrs::AdaExp(),
                         weights = w, control = ctrl_m)
chk(fitted(ma), as.numeric(fitted(ra)), "AdaExp weighted fitted")

cat("weighted glmboost R-binding parity: PASS\n")
