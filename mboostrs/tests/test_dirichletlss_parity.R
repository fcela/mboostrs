#@validates parity.r.dirichletlss
# Parity check: our DirichletLSS (glmboostLSS + gamboostLSS) vs R's gamboostLSS.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping DirichletLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

rdir <- function(a) { g <- rgamma(length(a), a); g / sum(g) }

set.seed(2); n <- 150; x <- rnorm(n)
a1 <- exp(0.5 + 0.3 * x); a2 <- exp(0.2); a3 <- exp(0.4 - 0.2 * x)
Y <- t(apply(cbind(a1, a2, a3), 1, rdir)); colnames(Y) <- c("y1", "y2", "y3")
d <- data.frame(Y, x = x)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-22s maxdiff=%.2e\n", what, md))
}

ours <- mboostrs::glmboostLSS(cbind(y1, y2, y3) ~ x, data = d,
                             families = mboostrs::DirichletLSS(K = 3),
                             control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref <- gamboostLSS::glmboostLSS(cbind(y1, y2, y3) ~ x, data = d,
                               families = gamboostLSS::DirichletLSS(K = 3),
                               control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("a1", "a2", "a3"))
  chk(predict(ours, parameter = p, type = "link"),
      as.numeric(predict(ref, parameter = p, type = "link")), paste("glm", p))

xg <- runif(n, -1, 1)
a1g <- exp(0.5 + sin(2 * xg)); a2g <- exp(0.2); a3g <- exp(0.4 - 0.3 * xg)
Yg <- t(apply(cbind(a1g, a2g, a3g), 1, rdir)); colnames(Yg) <- c("y1", "y2", "y3")
dg <- data.frame(Yg, x = xg)
oursg <- mboostrs::gamboostLSS(cbind(y1, y2, y3) ~ bbs(x, knots = 20, df = 4), data = dg,
                              families = mboostrs::DirichletLSS(K = 3),
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1))
refg <- gamboostLSS::gamboostLSS(cbind(y1, y2, y3) ~ bbs(x, knots = 20, df = 4), data = dg,
                                families = gamboostLSS::DirichletLSS(K = 3),
                                control = mboost::boost_control(mstop = 100, nu = 0.1))
for (p in c("a1", "a2", "a3"))
  chk(predict(oursg, parameter = p, type = "link"),
      as.numeric(predict(refg, parameter = p, type = "link")), paste("gam", p))

cat("DirichletLSS R-binding parity: PASS\n")
