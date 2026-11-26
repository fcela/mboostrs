# Parity check: upstream gamboostLSS examples from glmboostLSS/gamboostLSS.
# Tests NBinomialLSS (from example("glmboostLSS")), GaussianLSS toy data
# (from the gamboostLSS Tutorial vignette), and per-parameter formulas.
# Run after `R CMD INSTALL mboostrs` (needs gamboostLSS installed). Skips
# cleanly if gamboostLSS is unavailable.
# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping gamboostLSS example parity")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-34s maxdiff=%.2e\n", what, md))
}

# ── NBinomialLSS (from example("glmboostLSS")) ────────────────────────
set.seed(1907)
x1 <- rnorm(1000); x2 <- rnorm(1000)
x3 <- rnorm(1000); x4 <- rnorm(1000)
x5 <- rnorm(1000); x6 <- rnorm(1000)
mu    <- exp(1.5 + 1 * x1 + 0.5 * x2 - 0.5 * x3 - 1 * x4)
sigma <- exp(-0.4 * x3 - 0.2 * x4 + 0.2 * x5 + 0.4 * x6)
y <- numeric(1000)
for (i in 1:1000)
    y[i] <- rnbinom(1, size = sigma[i], mu = mu[i])
dat <- data.frame(x1, x2, x3, x4, x5, x6, y)

ours_nb <- mboostrs::glmboostLSS(y ~ ., data = dat,
                      families = mboostrs::NBinomialLSS(),
                      control = mboostrs::boost_control(mstop = 200, nu = 0.1))
ref_nb <- gamboostLSS::glmboostLSS(y ~ ., data = dat,
                       families = gamboostLSS::NBinomialLSS(),
                       control = mboost::boost_control(mstop = 200),
                       center = TRUE)
chk(predict(ours_nb, parameter = "mu", type = "link"),
    as.numeric(predict(ref_nb, parameter = "mu", type = "link")),
    "NBinomialLSS mu (link)")
chk(predict(ours_nb, parameter = "sigma", type = "link"),
    as.numeric(predict(ref_nb, parameter = "sigma", type = "link")),
    "NBinomialLSS sigma (link)")

# ── GaussianLSS toy data (from Tutorial vignette) ─────────────────────
set.seed(1907)
n <- 150
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
toy <- data.frame(x1 = x1, x2 = x2, x3 = x3)
toy$y <- rnorm(n, mean = 1 + 2 * x1 - x2,
                     sd = exp(0.5 - 0.25 * x1 + 0.5 * x3))

ours_g <- mboostrs::glmboostLSS(y ~ x1 + x2 + x3, data = toy,
                     families = mboostrs::GaussianLSS(),
                     control = mboostrs::boost_control(mstop = 100, nu = 0.1))
ref_g <- gamboostLSS::glmboostLSS(y ~ x1 + x2 + x3, data = toy,
                      families = gamboostLSS::GaussianLSS(),
                      control = mboost::boost_control(mstop = 100))
chk(predict(ours_g, parameter = "mu", type = "link"),
    as.numeric(predict(ref_g, parameter = "mu", type = "link")),
    "GaussianLSS toy mu (link)", tol = 1e-7)
chk(predict(ours_g, parameter = "sigma", type = "link"),
    as.numeric(predict(ref_g, parameter = "sigma", type = "link")),
    "GaussianLSS toy sigma (link)", tol = 1e-7)

# ── Per-parameter formulas (from glmboostLSS example) ─────────────────
ours_pp <- mboostrs::glmboostLSS(
    formula = list(mu = y ~ x1 + x2 + x3 + x4,
                   sigma = y ~ x3 + x4 + x5 + x6),
    families = mboostrs::NBinomialLSS(), data = dat,
    control = mboostrs::boost_control(mstop = 200, nu = 0.1),
    mstop = c(mu = 200, sigma = 150))
ref_pp <- gamboostLSS::glmboostLSS(
    formula = list(mu = y ~ x1 + x2 + x3 + x4,
                   sigma = y ~ x3 + x4 + x5 + x6),
    families = gamboostLSS::NBinomialLSS(), data = dat,
    control = mboost::boost_control(mstop = 200), center = TRUE)
mstop(ref_pp) <- c(mu = 200, sigma = 150)
chk(predict(ours_pp, parameter = "mu", type = "link"),
    as.numeric(predict(ref_pp, parameter = "mu", type = "link")),
    "NBinomialLSS per-param mu (link)")
chk(predict(ours_pp, parameter = "sigma", type = "link"),
    as.numeric(predict(ref_pp, parameter = "sigma", type = "link")),
    "NBinomialLSS per-param sigma (link)")

cat("gamboostLSS R-example parity: PASS\n")
