#@validates parity.r.zinblss
# Correctness check for our ZINBLSS vs R's gamboostLSS.
#
# mboostrs uses the mathematically correct ZINBI gradient; R's gamboostLSS does
# not (gamlss.dist's dNBI(0, mu, sigma) collapses to the first observation's
# zero-density and recycles it across all rows — see
# .remember/gamboostlss-survival-offset-bugs.md). So full-trajectory parity is
# not asserted. We check that the offsets (gamlss initials) and the ZINBI
# negative log-likelihood at the offset (where the bug is invisible) match R.
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping ZINBLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(4); n <- 250; x1 <- rnorm(n); x2 <- rnorm(n)
mu <- exp(0.7 + 0.3 * x1); pz <- plogis(-1.2 + 0.5 * x2)
z <- rbinom(n, 1, pz); y <- ifelse(z == 1, 0, rnbinom(n, size = 2, mu = mu))
d <- data.frame(y = y, x1 = x1, x2 = x2)

chk <- function(a, b, what, tol = 1e-8) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-26s maxdiff=%.2e\n", what, md))
}

ours <- mboostrs::glmboostLSS(y ~ x1 + x2, data = d, families = mboostrs::ZINBLSS(),
                             control = mboostrs::boost_control(mstop = 1, nu = 0.1))
ref <- gamboostLSS::glmboostLSS(y ~ x1 + x2, data = d, families = gamboostLSS::ZINBLSS(),
                               control = mboost::boost_control(mstop = 1, nu = 0.1))

chk(ours$params$mu$offset, ref$mu$offset, "mu offset")
chk(ours$params$sigma$offset, ref$sigma$offset, "sigma offset")
chk(ours$params$nu$offset, ref$nu$offset, "nu offset")
chk(ours$risk[1], as.numeric(risk(ref, merge = TRUE))[1], "risk at offset")

cat("ZINBLSS R-binding correctness (offsets + density): PASS\n")
