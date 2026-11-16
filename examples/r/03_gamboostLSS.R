# ============================================================================
# mboostrs R examples — 03: gamboostLSS (distributional regression)
# ============================================================================
# Run with:
#   Rscript examples/r/03_gamboostLSS.R
# ============================================================================

library(mboostrs)

# ---- 1. Basic GaussianLSS: joint mean + variance ----
cat("\n=== 1. GaussianLSS location-scale model ===\n")

set.seed(42)
n <- 400
x <- runif(n, -2, 2)
mu <- 1.5 * x
log_sigma <- 0.3 + 0.4 * abs(x)
y <- rnorm(n, mu, exp(log_sigma))
data <- data.frame(x, y)

m_lss <- gamboostLSS(y ~ bbs(x, df = 4), data = data,
                     families = GaussianLSS(),
                     control = boost_control(mstop = 300, nu = 0.1))
cat("Parameters:", paste(m_lss$param_names, collapse = ", "), "\n")

mu_pred <- predict(m_lss, parameter = "mu", type = "response")
sigma_pred <- predict(m_lss, parameter = "sigma", type = "response")
cat("mu range:", round(range(mu_pred), 4), "\n")
cat("sigma range:", round(range(sigma_pred), 4), "\n")

# ---- 2. Per-parameter formulas (bbs only) ----
cat("\n=== 2. Per-parameter formulas ===\n")

set.seed(42)
n <- 400
x1 <- rnorm(n)
x2 <- rnorm(n)
mu <- 0.5 * x1
log_sigma <- 0.2 + 0.1 * x2
y <- rnorm(n, mu, exp(log_sigma))
data <- data.frame(x1, x2, y)

m_asy <- gamboostLSS(
  list(mu = y ~ bbs(x1, df = 4),
       sigma = y ~ bbs(x2, df = 3)),
  data = data, families = GaussianLSS(),
  control = boost_control(mstop = 200, nu = 0.1))
cat("Per-parameter mstop:", m_asy$params$mu$mstop, "/", m_asy$params$sigma$mstop, "\n")
cat("mu selected terms:", sum(m_asy$params$mu$selected > 0), "\n")
cat("sigma selected terms:", sum(m_asy$params$sigma$selected > 0), "\n")

# ---- 3. GammaLSS ----
cat("\n=== 3. GammaLSS ===\n")

set.seed(42)
n <- 400
x <- runif(n, 0.5, 2)
mu <- 0.5 + 1.5 * x
shape <- 3.0
scale <- mu / shape
y <- rgamma(n, shape = shape, scale = scale)
data <- data.frame(x, y)

m_gamma <- gamboostLSS(y ~ bbs(x, df = 4), data = data,
                       families = GammaLSS(),
                       control = boost_control(mstop = 200, nu = 0.1))
cat("GammaLSS parameters:", paste(m_gamma$param_names, collapse = ", "), "\n")

# ---- 4. Prediction (in-sample only) ----
cat("\n=== 4. Prediction (in-sample) ===\n")

pred_mu <- predict(m_lss, parameter = "mu", type = "response")
pred_sigma <- predict(m_lss, parameter = "sigma", type = "response")
cat("mu range:", round(range(pred_mu), 4), "\n")
cat("sigma range:", round(range(pred_sigma), 4), "\n")

# ---- 5. Log-likelihood ----
cat("\n=== 5. Log-likelihood ===\n")
ll <- logLik(m_lss)
cat("logLik:", round(ll, 2), "(df:", attr(ll, "df"), ")\n")

cat("\nAll gamboostLSS examples completed.\n")
