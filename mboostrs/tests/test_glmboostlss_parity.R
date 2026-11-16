#@validates parity.r.gaussianlss
# Parity check: our glmboostLSS (GaussianLSS) vs R's gamboostLSS::glmboostLSS.
# Run after `R CMD INSTALL mboostrs` (needs gamboostLSS installed). Skips cleanly
# if gamboostLSS is unavailable.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping LSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(1); n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
y <- rnorm(n, mean = 1 + 0.8 * x1, sd = exp(0.3 * x2))
d <- data.frame(y = y, x1 = x1, x2 = x2)

# NB: gamboostLSS is attached after mboostrs and also exports glmboostLSS /
# GaussianLSS, so "ours" must be namespace-qualified or it silently resolves to
# gamboostLSS (making the comparison trivially pass).
ours <- mboostrs::glmboostLSS(y ~ x1 + x2, data = d,
                    families = mboostrs::GaussianLSS(),
                    control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref <- gamboostLSS::glmboostLSS(
  y ~ x1 + x2, data = d,
  families = gamboostLSS::GaussianLSS(),
  control = mboost::boost_control(mstop = 100, nu = 0.1))

chk <- function(a, b, what, tol = 1e-7) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-22s maxdiff=%.2e\n", what, md))
}
chk(predict(ours, parameter = "mu", type = "link"),
    as.numeric(predict(ref, parameter = "mu", type = "link")), "mu (link)")
chk(predict(ours, parameter = "sigma", type = "link"),
    as.numeric(predict(ref, parameter = "sigma", type = "link")), "sigma (link)")
chk(predict(ours, parameter = "sigma", type = "response"),
    as.numeric(predict(ref, parameter = "sigma", type = "response")), "sigma (response)")

# Per-parameter mstop (R's mstop<- vector).
ours2 <- mboostrs::glmboostLSS(y ~ x1 + x2, data = d,
                     families = mboostrs::GaussianLSS(),
                     control = mboostrs::boost_control(mstop = 60, nu = 0.1),
                     mstop = c(mu = 50, sigma = 20))
ref2 <- gamboostLSS::glmboostLSS(
  y ~ x1 + x2, data = d, families = gamboostLSS::GaussianLSS(),
  control = mboost::boost_control(mstop = 60, nu = 0.1))
mstop(ref2) <- c(mu = 50, sigma = 20)  # gamboostLSS's mstop<- (attached)
chk(predict(ours2, parameter = "mu", type = "link"),
    as.numeric(predict(ref2, parameter = "mu", type = "link")), "mstop-vec mu")
chk(predict(ours2, parameter = "sigma", type = "link"),
    as.numeric(predict(ref2, parameter = "sigma", type = "link")), "mstop-vec sigma")

# Coefficients, not only predictions. `glmboostLSS` centers its predictors, so
# a slope fitted on the centered column implies an intercept shift of
# `-sum(beta_j * mean_j)` when the coefficients are reported on the original
# scale. This frontend omitted that correction and reported the intercept as 0
# whenever the intercept learner itself was never selected -- which is the
# usual case, because the offset already carries the level. Predictions were
# unaffected, so every check above passed while `coef()` disagreed with
# upstream by exactly `mean(x) * slope`.
for (pn in c("mu", "sigma")) {
  ours_coef <- as.numeric(coef(ours, parameter = pn))
  ref_coef <- as.numeric(coef(ref)[[pn]])
  chk(ours_coef, ref_coef, paste0(pn, " coefficients"))
}

# And the intercept specifically, so a future change that zeroes it again
# fails on the name rather than on a vector comparison.
intercept <- coef(ours, parameter = "mu")[["(Intercept)"]]
ref_intercept <- as.numeric(coef(ref)$mu)[1]
if (abs(intercept - ref_intercept) > 1e-8) {
  stop(sprintf("mu intercept %.8f differs from gamboostLSS's %.8f",
               intercept, ref_intercept))
}

cat("glmboostLSS GaussianLSS R-binding parity: PASS\n")
