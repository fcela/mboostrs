#@validates parity.r.lognormallss
# Parity check: our LogNormalLSS (glmboostLSS + gamboostLSS) vs R's gamboostLSS.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping LogNormalLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS)); suppressMessages(library(survival))

set.seed(9); n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
t <- exp(1 + 0.5 * x1 + exp(0.2 * x2) * rnorm(n)); st <- rbinom(n, 1, 0.7)
d <- data.frame(t = t, st = st, x1 = x1, x2 = x2)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-22s maxdiff=%.2e\n", what, md))
}

ours <- mboostrs::glmboostLSS(Surv(t, st) ~ x1 + x2, data = d,
                             families = mboostrs::LogNormalLSS(),
                             control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref <- gamboostLSS::glmboostLSS(Surv(t, st) ~ x1 + x2, data = d,
                               families = gamboostLSS::LogNormalLSS(),
                               control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(ours, parameter = p, type = "link"),
      as.numeric(predict(ref, parameter = p, type = "link")), paste("glm", p))

xg <- runif(n, -1, 1)
tg <- exp(1 + sin(2 * xg) + exp(0.2 * xg) * rnorm(n)); stg <- rbinom(n, 1, 0.7)
dg <- data.frame(t = tg, st = stg, x = xg)
oursg <- mboostrs::gamboostLSS(Surv(t, st) ~ bbs(x, knots = 20, df = 4), data = dg,
                              families = mboostrs::LogNormalLSS(),
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
refg <- gamboostLSS::gamboostLSS(Surv(t, st) ~ bbs(x, knots = 20, df = 4), data = dg,
                                families = gamboostLSS::LogNormalLSS(),
                                control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(oursg, parameter = p, type = "link"),
      as.numeric(predict(refg, parameter = p, type = "link")), paste("gam", p))

cat("LogNormalLSS R-binding parity: PASS\n")
