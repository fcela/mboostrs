#@validates parity.r.zipolss
# Parity check: our ZIPoLSS (glmboostLSS + gamboostLSS) vs R's gamboostLSS.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping ZIPoLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(3); n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
mu <- exp(0.5 + 0.4 * x1); pz <- plogis(-1 + 0.5 * x2)
z <- rbinom(n, 1, pz); y <- ifelse(z == 1, 0, rpois(n, mu))
d <- data.frame(y = y, x1 = x1, x2 = x2)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-22s maxdiff=%.2e\n", what, md))
}

ours <- mboostrs::glmboostLSS(y ~ x1 + x2, data = d, families = mboostrs::ZIPoLSS(),
                             control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref <- gamboostLSS::glmboostLSS(y ~ x1 + x2, data = d, families = gamboostLSS::ZIPoLSS(),
                               control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(ours, parameter = p, type = "link"),
      as.numeric(predict(ref, parameter = p, type = "link")), paste("glm", p))

xg <- runif(n, -1, 1)
mug <- exp(0.5 + sin(2 * xg)); pzg <- plogis(-1 + xg)
zg <- rbinom(n, 1, pzg); yg <- ifelse(zg == 1, 0, rpois(n, mug))
dg <- data.frame(y = yg, x = xg)
oursg <- mboostrs::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                              families = mboostrs::ZIPoLSS(),
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
refg <- gamboostLSS::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                                families = gamboostLSS::ZIPoLSS(),
                                control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("mu", "sigma"))
  chk(predict(oursg, parameter = p, type = "link"),
      as.numeric(predict(refg, parameter = p, type = "link")), paste("gam", p))

cat("ZIPoLSS R-binding parity: PASS\n")
