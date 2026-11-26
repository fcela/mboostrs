# Parity check: our gamboostLSS (GaussianLSS, bbs) vs R's gamboostLSS::gamboostLSS.
# Run after `R CMD INSTALL mboostrs` (needs gamboostLSS installed).
# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping LSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(1); n <- 200; x <- runif(n)
y <- rnorm(n, mean = 2 + sin(2 * pi * x), sd = exp(0.3 + 0.5 * x))
d <- data.frame(y = y, x = x)

# NB: gamboostLSS is attached after mboostrs and also exports gamboostLSS /
# GaussianLSS, so "ours" must be namespace-qualified.
ours <- mboostrs::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = d,
                              families = mboostrs::GaussianLSS(),
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref <- gamboostLSS::gamboostLSS(
  y ~ bbs(x, knots = 20, df = 4), data = d,
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
ours2 <- mboostrs::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = d,
                              families = mboostrs::GaussianLSS(),
                              control = mboostrs::boost_control(mstop = 80, nu = 0.1),
                              mstop = c(mu = 60, sigma = 30))
ref2 <- gamboostLSS::gamboostLSS(
  y ~ bbs(x, knots = 20, df = 4), data = d,
  families = gamboostLSS::GaussianLSS(),
  control = mboost::boost_control(mstop = 80, nu = 0.1))
mstop(ref2) <- c(mu = 60, sigma = 30)  # gamboostLSS's mstop<- (attached)
chk(predict(ours2, parameter = "mu", type = "link"),
    as.numeric(predict(ref2, parameter = "mu", type = "link")), "mstop-vec mu")
chk(predict(ours2, parameter = "sigma", type = "link"),
    as.numeric(predict(ref2, parameter = "sigma", type = "link")), "mstop-vec sigma")

cat("gamboostLSS GaussianLSS R-binding parity: PASS\n")
