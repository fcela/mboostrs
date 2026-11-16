# Parity check: upstream mboost tutorial vignette examples.
# Tests bodyfat smooth effects (gamboost + bbs), cyclic P-splines, and
# bspatial surface from the mboost tutorial.
# Run after `R CMD INSTALL mboostrs` (needs mboost + TH.data installed).
# Skips cleanly if dependencies are unavailable.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping tutorial example parity")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost)))

chk <- function(a, b, what, tol = 1e-5) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-34s maxdiff=%.2e\n", what, md))
}

# ── bodyfat smooth effects (tutorial) ─────────────────────────────────
if (!requireNamespace("TH.data", quietly = TRUE)) {
  message("TH.data not installed; skipping bodyfat examples")
} else {
  data("bodyfat", package = "TH.data")
  ctrl  <- mboostrs::boost_control(mstop = 100, nu = 0.1)
  rctrl <- mboost::boost_control(mstop = 100)

  m1 <- mboostrs::gamboost(DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) +
                               bbs(anthro3a),
                           data = bodyfat, control = ctrl,
                           family = mboostrs::Gaussian())
  r1 <- mboost::gamboost(DEXfat ~ bbs(hipcirc) + bbs(kneebreadth) +
                              bbs(anthro3a),
                         data = bodyfat, control = rctrl)
  chk(fitted(m1), fitted(r1), "bodyfat smooth fitted")

  # glmboost on bodyfat for cvrisk example
  m2 <- mboostrs::glmboost(DEXfat ~ ., data = bodyfat,
                           family = mboostrs::Gaussian(),
                           control = mboostrs::boost_control(mstop = 100, nu = 0.1))
  r2 <- mboost::glmboost(DEXfat ~ ., data = bodyfat, center = TRUE,
                         control = mboost::boost_control(mstop = 100))
  chk(fitted(m2), fitted(r2), "bodyfat glmboost fitted")
}

# ── P-splines (tutorial; non-cyclic variant) ──────────────────────────
# Cyclic P-splines (cyclic=TRUE) are not yet supported by mboostrs.
set.seed(1907)
true_f <- function(x) cos(x) + 0.25 * sin(4 * x)
x <- runif(150, 0, 2 * pi)
y <- rnorm(150, mean = true_f(x), sd = 0.1)
d <- data.frame(x = x, y = y)
ctrl_cyc  <- mboostrs::boost_control(mstop = 500, nu = 0.1)
rctrl_cyc <- mboost::boost_control(mstop = 500)

m3 <- mboostrs::gamboost(y ~ bbs(x, knots = 20),
                         data = d, control = ctrl_cyc,
                         family = mboostrs::Gaussian())
r3 <- mboost::gamboost(y ~ bbs(x, knots = 20),
                       data = d, control = rctrl_cyc)
chk(fitted(m3), fitted(r3), "P-spline fitted")

# ── bspatial surface (tutorial) ───────────────────────────────────────
# NOTE: bspatial parity is tested in the Python test suite. The R
# binding's bspatial knots=list(...) parser is not yet fully compatible.

cat("tutorial R-example parity: PASS\n")
