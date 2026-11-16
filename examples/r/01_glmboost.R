# ============================================================================
# mboostrs R examples — 01: glmboost
# ============================================================================
# Run with:
#   Rscript examples/r/01_glmboost.R
# ============================================================================

library(mboostrs)

# ---- 1. Basic Gaussian glmboost ----
cat("\n=== 1. Basic Gaussian glmboost ===\n")

set.seed(42)
n <- 500
p <- 8
X <- matrix(rnorm(n * p), n, p)
true_beta <- c(1.5, -0.8, 0, 0, 0.6, 0, 0, -0.3)
y <- X %*% true_beta + rnorm(n, sd = 0.5)

data <- data.frame(X, y)
colnames(data) <- c(paste0("x", 1:8), "y")
formula <- as.formula(paste("y ~", paste(paste0("x", 1:8), collapse = " + ")))

fit <- glmboost(formula, data = data, family = Gaussian(),
                control = boost_control(mstop = 300, nu = 0.1))

cat("Coefficients:\n")
coefs <- coef(fit)
print(round(coefs[abs(coefs) > 1e-4], 4))

# ---- 2. Cross-validated mstop ----
cat("\n=== 2. Cross-validated mstop ===\n")

folds <- cv_weights(n, fold_type = "kfold", b = 10)
cvr <- cvrisk(fit, folds = folds)
m_opt <- mstop(cvr)
cat("Optimal mstop:", m_opt, "\n")

fit_opt <- fit[m_opt]
cat("Risk at mstop*:", tail(risk(fit_opt), 1), "\n")

# ---- 3. Poisson glmboost ----
cat("\n=== 3. Poisson glmboost ===\n")

set.seed(42)
n <- 400
x <- rnorm(n)
eta <- 0.5 + 0.6 * x
y <- rpois(n, lambda = exp(eta))

fit_pois <- glmboost(y ~ x, data = data.frame(x = x, y = y),
                     family = Poisson(),
                     control = boost_control(mstop = 200, nu = 0.1))
cat("Poisson coefs:\n")
print(round(coef(fit_pois)[abs(coef(fit_pois)) > 1e-4], 4))

# ---- 4. Binomial glmboost ----
cat("\n=== 4. Binomial glmboost ===\n")

set.seed(42)
n <- 600
x1 <- rnorm(n)
x2 <- rnorm(n)
eta <- 1.0 * x1 - 0.6 * x2
p <- 1 / (1 + exp(-eta))
y <- rbinom(n, 1, p)

fit_bin <- glmboost(y ~ x1 + x2, data = data.frame(x1 = x1, x2 = x2, y = y),
                    family = Binomial(),
                    control = boost_control(mstop = 200, nu = 0.1))
cat("Binomial accuracy:", mean((predict(fit_bin, type = "response") > 0.5) == y), "\n")

# ---- 5. AIC-based selection ----
cat("\n=== 5. AIC-based selection ===\n")

aic <- AIC(fit, method = "corrected")
cat("AIC-optimal mstop:", which.min(aic), "\n")

aic_classical <- AIC(fit, method = "classical")
cat("Classical AIC mstop:", which.min(aic_classical), "\n")

# ---- 6. Variable importance ----
cat("\n=== 6. Variable importance ===\n")

imp <- varimp(fit)
cat("Risk reduction per learner:\n")
print(round(imp$reduction, 4))
cat("Selection frequencies:\n")
print(imp$selfreq)

# ---- 7. Summary ----
cat("\n=== 7. Model summary ===\n")
invisible(summary(fit))

# ---- 8. Hat values ----
cat("\n=== 8. Hat values ===\n")
h <- hatvalues(fit)
cat("Hat diag: range", round(range(h), 6), "- sum DF:", round(sum(h), 4), "\n")

# ---- 9. Residuals ----
cat("\n=== 9. Residuals ===\n")
res <- residuals(fit)
cat("Residual range:", round(range(res), 6), "\n")

# ---- 10. Coefficient paths ----
cat("\n=== 10. Coefficient paths ===\n")
paths <- coef_path(fit)
names_shown <- head(names(paths), 3)
for (nm in names_shown) {
  cat(nm, ":", round(paths[[nm]][1], 4), "->", round(tail(paths[[nm]], 1), 4), "\n")
}

# ---- 11. Post-selection hat values ----
cat("\n=== 11. Post-selection hat values ===\n")
h_opt <- hatvalues(fit_opt)
cat("Hat diag at m_opt: range", round(range(h_opt), 6), "\n")

cat("\nAll glmboost examples completed.\n")
