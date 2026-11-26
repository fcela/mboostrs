# Parity check: upstream mboost glmboost() examples from glmboost.Rd.
# Tests cars data with Gaussian (default), Laplace, and Huber families.
# Run after `R CMD INSTALL mboostrs` (needs mboost installed). Skips cleanly
# if mboost is unavailable.
# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping glmboost example parity check")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost)))

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-34s maxdiff=%.2e\n", what, md))
}

data(cars, package = "datasets")

# ── Gaussian (centered, default family) ───────────────────────────────
# glmboost(dist ~ speed, data = cars, control = boost_control(mstop = 2000))
ours_g <- mboostrs::glmboost(dist ~ speed, data = cars,
                     control = mboostrs::boost_control(mstop = 2000, nu = 0.1),
                     family = mboostrs::Gaussian())
ref_g  <- mboost::glmboost(dist ~ speed, data = cars,
                    control = mboost::boost_control(mstop = 2000), center = TRUE)
chk(fitted(ours_g), fitted(ref_g),  "cars Gaussian fitted")
chk(predict(ours_g), predict(ref_g), "cars Gaussian predict")

# ── Laplace ───────────────────────────────────────────────────────────
# glmboost(dist ~ speed, data = cars, family = Laplace(), mstop = 1000)
ours_l <- mboostrs::glmboost(dist ~ speed, data = cars,
                     control = mboostrs::boost_control(mstop = 1000, nu = 0.1),
                     family = mboostrs::Laplace())
ref_l  <- mboost::glmboost(dist ~ speed, data = cars,
                    control = mboost::boost_control(mstop = 1000),
                    family = mboost::Laplace(), center = TRUE)
chk(fitted(ours_l), fitted(ref_l), "cars Laplace fitted", tol = 1e-5)

# ── Huber ─────────────────────────────────────────────────────────────
# glmboost(dist ~ speed, data = cars, family = Huber(), mstop = 1000)
ours_h <- mboostrs::glmboost(dist ~ speed, data = cars,
                     control = mboostrs::boost_control(mstop = 1000, nu = 0.1),
                     family = mboostrs::Huber())
ref_h  <- mboost::glmboost(dist ~ speed, data = cars,
                    control = mboost::boost_control(mstop = 1000),
                    family = mboost::Huber(), center = TRUE)
chk(fitted(ours_h), fitted(ref_h), "cars Huber fitted", tol = 1e-4)

cat("glmboost R-example parity: PASS\n")
