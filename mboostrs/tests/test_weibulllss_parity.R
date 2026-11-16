# Correctness + boosting-parity check for our WeibullLSS vs R's gamboostLSS.
#
# mboostrs does NOT replicate gamboostLSS' WeibullLSS mu-offset bug (it stores
# log() of the optimized location even though mu's link is the identity; see
# .remember/gamboostlss-survival-offset-bugs.md). So we validate:
#   (1) our auto mu/sigma offset minimizes the *correct* Weibull NLL;
#   (2) fed R's own offset, our boosting matches R's to machine precision.
#
# The response scale is kept gentle (sd ~ 0.5 on the log-time scale): R's
# WeibullLSS boosting gradient contains exp(eta) and aborts on heavy-tailed draws.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping WeibullLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS)); suppressMessages(library(survival))

set.seed(7); n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
t <- exp(1 + 0.3 * x1 + 0.5 * rnorm(n)); st <- rbinom(n, 1, 0.7)
d <- data.frame(t = t, st = st, x1 = x1, x2 = x2)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-26s maxdiff=%.2e\n", what, md))
}

# Correct Weibull NLL of a (mu, eta_sigma) offset pair.
nll <- function(mu, es) {
  e <- (log(t) - mu) / exp(es)
  logf <- e - exp(e); logs <- -exp(e)
  sum(-st * (logf - es) - (1 - st) * logs)
}

ours <- mboostrs::glmboostLSS(Surv(t, st) ~ x1 + x2, data = d,
                             families = mboostrs::WeibullLSS(),
                             control = mboostrs::boost_control(mstop = 1, nu = 0.1))
mu_off <- ours$params$mu$offset; sig_off <- ours$params$sigma$offset
h <- 1e-3
stopifnot(nll(mu_off, 0) <= nll(mu_off + h, 0), nll(mu_off, 0) <= nll(mu_off - h, 0))
stopifnot(nll(mu_off, sig_off) <= nll(mu_off, sig_off + h),
          nll(mu_off, sig_off) <= nll(mu_off, sig_off - h))
cat("OK offset minimizes correct Weibull NLL\n")

# Boosting parity at equal (R's) offset.
ref <- gamboostLSS::glmboostLSS(Surv(t, st) ~ x1 + x2, data = d,
                               families = gamboostLSS::WeibullLSS(),
                               control = mboost::boost_control(mstop = 100, nu = 0.1))
roff <- c(ref$mu$offset, ref$sigma$offset)
oursf <- mboostrs::glmboostLSS(Surv(t, st) ~ x1 + x2, data = d,
                              families = mboostrs::WeibullLSS(), offset = roff,
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(oursf, parameter = p, type = "link"),
      as.numeric(predict(ref, parameter = p, type = "link")), paste("glm", p))

xg <- runif(n, -1, 1)
tg <- exp(1 + sin(2 * xg) + 0.4 * rnorm(n)); stg <- rbinom(n, 1, 0.7)
dg <- data.frame(t = tg, st = stg, x = xg)
refg <- gamboostLSS::gamboostLSS(Surv(t, st) ~ bbs(x, knots = 20, df = 4), data = dg,
                                families = gamboostLSS::WeibullLSS(),
                                control = mboost::boost_control(mstop = 100, nu = 0.1))
roffg <- c(refg$mu$offset, refg$sigma$offset)
oursg <- mboostrs::gamboostLSS(Surv(t, st) ~ bbs(x, knots = 20, df = 4), data = dg,
                              families = mboostrs::WeibullLSS(), offset = roffg,
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(oursg, parameter = p, type = "link"),
      as.numeric(predict(refg, parameter = p, type = "link")), paste("gam", p))

cat("WeibullLSS R-binding correctness + boosting parity: PASS\n")
