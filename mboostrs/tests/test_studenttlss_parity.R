#@validates parity.r.studenttlss
# Parity check: our StudentTLSS (glmboostLSS + gamboostLSS) vs R's gamboostLSS.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping StudentTLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(3); n <- 200; x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
y <- (1 + 0.5 * x1) + exp(0.2 * x2) * rt(n, df = exp(1 + 0.3 * x3))
d <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-22s maxdiff=%.2e\n", what, md))
}

# glmboostLSS (linear, 3 parameters). Namespace-qualify "ours".
ours <- mboostrs::glmboostLSS(y ~ x1 + x2 + x3, data = d,
                             families = mboostrs::StudentTLSS(),
                             control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref <- gamboostLSS::glmboostLSS(y ~ x1 + x2 + x3, data = d,
                               families = gamboostLSS::StudentTLSS(),
                               control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma", "df"))
  chk(predict(ours, parameter = p, type = "link"),
      as.numeric(predict(ref, parameter = p, type = "link")), paste("glm", p))

# gamboostLSS (smooth, bbs).
xg <- runif(n, -1, 1)
yg <- (1 + sin(2 * xg)) + exp(0.2 * xg) * rt(n, df = 5)
dg <- data.frame(y = yg, x = xg)
oursg <- mboostrs::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                              families = mboostrs::StudentTLSS(),
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
refg <- gamboostLSS::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                                families = gamboostLSS::StudentTLSS(),
                                control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma", "df"))
  chk(predict(oursg, parameter = p, type = "link"),
      as.numeric(predict(refg, parameter = p, type = "link")), paste("gam", p))

cat("StudentTLSS R-binding parity: PASS\n")
