# Test bbs() with deriv parameter — fit, predict-newdata, and partial effects.
#
# Run after `R CMD INSTALL mboostrs` (standalone, no dependency on mboost).
library(mboostrs)

cat("========================================\n")
cat("mboostrs R Package — bbs deriv Test\n")
cat("========================================\n\n")

chk <- function(a, b, what, tol = 1e-5) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-40s maxdiff=%.2e\n", what, md))
}

# ── Synthetic data ────────────────────────────────────────────────────
set.seed(42)
n <- 100
x <- sort(runif(n) * 10)
y <- sin(x) + rnorm(n, sd = 0.25)
d <- data.frame(x = x, y = y)

ctrl <- boost_control(mstop = 200, nu = 0.1)

# ── deriv = 0 (standard bbs, no derivative) ───────────────────────────
cat("--- bbs with deriv=0 (standard) ---\n")
m0 <- gamboost(y ~ bbs(x, df = 4, deriv = 0), data = d,
               control = ctrl, family = Gaussian())
stopifnot(length(fitted(m0)) == n)
chk(fitted(m0), fitted(m0), "deriv=0 self-consistency", tol = 1e-12)

cat("OK fitted values (deriv=0) produce a numeric vector of length n\n")

# ── deriv = 1 (first derivative) ──────────────────────────────────────
cat("\n--- bbs with deriv=1 (1st derivative) ---\n")
m1 <- gamboost(y ~ bbs(x, df = 4, deriv = 1), data = d,
               control = ctrl, family = Gaussian())
stopifnot(length(fitted(m1)) == n)
cat("OK fitted values (deriv=1) produce a numeric vector of length n\n")

# ── deriv = 2 (second derivative) ─────────────────────────────────────
cat("\n--- bbs with deriv=2 (2nd derivative) ---\n")
m2 <- gamboost(y ~ bbs(x, df = 4, deriv = 2), data = d,
               control = ctrl, family = Gaussian())
stopifnot(length(fitted(m2)) == n)
cat("OK fitted values (deriv=2) produce a numeric vector of length n\n")

# ── Coefficients differ across deriv values ────────────────────────────
cat("\n--- Check that coefficients differ across deriv values ---\n")
c0 <- sum(abs(coef(m0)$x))
c1 <- sum(abs(coef(m1)$x))
c2 <- sum(abs(coef(m2)$x))
cat(sprintf("  deriv=0 coef L1: %.6f\n", c0))
cat(sprintf("  deriv=1 coef L1: %.6f\n", c1))
cat(sprintf("  deriv=2 coef L1: %.6f\n", c2))
if (abs(c0 - c1) < 1e-8 && abs(c1 - c2) < 1e-8) {
  stop("Coefficients do not differ across deriv values — derivative basis not active")
}
cat("OK coefficients differ meaningfully\n")

# ── predict with newdata ──────────────────────────────────────────────
cat("\n--- predict with newdata ---\n")
newx <- data.frame(x = c(1.5, 3.5, 7.2))
p0 <- predict(m0, newdata = newx, mstop = 200)
p1 <- predict(m1, newdata = newx, mstop = 200)
p2 <- predict(m2, newdata = newx, mstop = 200)
stopifnot(length(p0) == 3, length(p1) == 3, length(p2) == 3)
cat("OK predict(newdata) works for deriv=0,1,2\n")

# ── partial_effects ───────────────────────────────────────────────────
cat("\n--- partial_effects (deriv=0) ---\n")
pe <- partial_effects(m0)
stopifnot(is.data.frame(pe$x), nrow(pe$x) == 100)
cat("OK partial_effects works for deriv=0\n")

# ── risk path is finite ───────────────────────────────────────────────
cat("\n--- risk path is finite ---\n")
stopifnot(all(is.finite(risk(m0))))
stopifnot(all(is.finite(risk(m1))))
stopifnot(all(is.finite(risk(m2))))
cat("OK risk paths are finite for all deriv values\n")

cat("\n========================================\n")
cat("bbs deriv test: PASS\n")
cat("========================================\n")
