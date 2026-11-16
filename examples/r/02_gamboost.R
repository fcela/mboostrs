# ============================================================================
# mboostrs R examples — 02: gamboost (gradient boosting with base-learners)
# ============================================================================
# Run with:
#   Rscript examples/r/02_gamboost.R
# ============================================================================

library(mboostrs)

# ---- 1. gamboost with bbs P-spline ----
cat("\n=== 1. gamboost with bbs P-spline ===\n")

set.seed(42)
n <- 300
x <- seq(0, 5, length.out = n)
y <- sin(x) + rnorm(n, sd = 0.2)

fit <- gamboost(y ~ bbs(x, df = 4), data = data.frame(x = x, y = y),
                family = Gaussian(),
                control = boost_control(mstop = 200, nu = 0.1))

cat("fit$mstop:", fit$mstop, "\n")
cat("Risk range:", round(range(risk(fit)), 6), "\n")

# ---- 2. Variable importance ----
cat("\n=== 2. Variable importance ===\n")

imp <- varimp(fit)
cat("Risk reduction per learner:\n")
print(round(imp$reduction, 4))

# ---- 3. AIC-based selection ----
cat("\n=== 3. AIC-based selection ===\n")

aic <- AIC(fit, method = "corrected")
m_opt <- which.min(aic)
cat("Optimal mstop (AICc):", m_opt, "\n")

fit_opt <- fit[m_opt]
cat("Risk at mopt:", round(tail(risk(fit_opt), 1), 6), "\n")

# ---- 4. Hat values (in-sample) ----
cat("\n=== 4. Hat values ===\n")

h <- hatvalues(fit)
cat("Full model, hat diag: range", round(range(h), 6), "\n")
h_opt <- hatvalues(fit_opt)
cat("At mopt, hat diag: range", round(range(h_opt), 6), "\n")

# ---- 5. Residuals at m_opt ----
cat("\n=== 5. Residuals ===\n")

res <- residuals(fit_opt)
cat("Residual range:", round(range(res), 6), "\n")

# ---- 6. Summary ----
cat("\n=== 6. Model summary ===\n")
invisible(summary(fit_opt))

# ---- 7. Predict on new data ----
cat("\n=== 7. Predict on new data ===\n")

x_grid <- seq(0, 5, length.out = 100)
pred <- predict(fit_opt, newdata = data.frame(x = x_grid))
cat("Predictions: length", length(pred), ", range", round(range(pred), 4), "\n")

# ---- 8. Coef (smooth) ----
cat("\n=== 8. Fitted smooth term ===\n")

coef_smooth <- coef(fit_opt, which = 1)
cat("Smooth term coefs:", length(coef_smooth[[1]]), "points\n")
print(head(round(coef_smooth[[1]], 4), 6))

# ---- 9. Update (reduce mstop) ----
cat("\n=== 9. Update (reduce mstop) ===\n")

fit2 <- update(fit, mstop = 50)
cat("Updated mstop:", fit2$mstop, "\n")
cat("Risk at mstop=50:", round(tail(risk(fit2), 1), 6), "\n")

cat("\nAll gamboost examples completed.\n")
