#@validates parity.r.nbinomiallss
# Parity check: our NBinomialLSS (glmboostLSS + gamboostLSS) vs R's gamboostLSS.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping NBinomialLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(7); n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
y <- rnbinom(n, mu = exp(1 + 0.3 * x1), size = exp(0.5 + 0.2 * x2))
d <- data.frame(y = y, x1 = x1, x2 = x2)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-22s maxdiff=%.2e\n", what, md))
}

ours <- mboostrs::glmboostLSS(y ~ x1 + x2, data = d,
                             families = mboostrs::NBinomialLSS(),
                             control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref <- gamboostLSS::glmboostLSS(y ~ x1 + x2, data = d,
                               families = gamboostLSS::NBinomialLSS(),
                               control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(ours, parameter = p, type = "link"),
      as.numeric(predict(ref, parameter = p, type = "link")), paste("glm", p))

xg <- runif(n, -1, 1)
yg <- rnbinom(n, mu = exp(1 + sin(2 * xg)), size = exp(0.5 + 0.3 * xg))
dg <- data.frame(y = yg, x = xg)
oursg <- mboostrs::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                              families = mboostrs::NBinomialLSS(),
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
refg <- gamboostLSS::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                                families = gamboostLSS::NBinomialLSS(),
                                control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(oursg, parameter = p, type = "link"),
      as.numeric(predict(refg, parameter = p, type = "link")), paste("gam", p))

cat("NBinomialLSS R-binding parity: PASS\n")
