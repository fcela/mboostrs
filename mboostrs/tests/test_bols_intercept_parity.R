#@validates parity.r.bols_intercept
# Normalizing a `bols` menu changes representation, not model space.
#
# `bols(x)` has two engine forms: the *centered* single column, which takes its
# constant from the fixed model offset, and the *block* `[1, x]`, which carries
# its own and refits it every time that learner is selected. A menu has to
# choose one, because mixing them represents the constant twice. That much was
# already handled.
#
# What was not: the conversion added `[1, x]` to every scalar `bols`, including
# terms written `intercept = FALSE`, handing back the constant the caller had
# removed. And the conversion only ran when some term forced the block form
# with a penalty, so a menu like `bols(x) + bols(z, intercept = FALSE)` stayed
# in the centered form with one term uncentered -- the fixed offset supplying a
# constant to one learner and not the other.
#
# Measured against `mboost` on 60 rows, mstop 40:
#
#   bols(x, lambda = 2) + bols(z, intercept = FALSE)   0.0307
#   bols(x) + bols(z, intercept = FALSE)               0.0502
#
# Both had an **identical selection sequence**, so a parity test comparing
# `selected()` would have passed. This one compares fitted values, and crosses
# penalized against unpenalized, both intercept settings, and `by`.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping bols intercept parity check")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost)))

set.seed(7)
n <- 60
x <- rnorm(n)
z <- rnorm(n)
w <- rnorm(n)
by <- rnorm(n)
y <- 1.2 * x - 0.8 * z + 0.5 * w + rnorm(n, sd = 0.4)
d <- data.frame(x = x, z = z, w = w, by = by, y = y)

MSTOP <- 40L
NU <- 0.1
TOL <- 1e-6

compare <- function(rhs, tol = TOL) {
  fo <- as.formula(paste("y ~", rhs))
  mine <- suppressWarnings(mboostrs::gamboost(
    fo, data = d, family = mboostrs::Gaussian(),
    control = mboostrs::boost_control(mstop = MSTOP, nu = NU)))
  theirs <- suppressWarnings(mboost::gamboost(
    fo, data = d, family = mboost::Gaussian(),
    control = mboost::boost_control(mstop = MSTOP, nu = NU)))

  gap <- max(abs(as.numeric(fitted(mine)) - as.numeric(fitted(theirs))))
  if (gap >= tol) {
    stop(sprintf("%s: fitted values differ from mboost by %.3e", rhs, gap))
  }
  # Checked second, and deliberately: the selection sequence agreed through
  # every version of this defect, so it cannot be the thing the test rests on.
  if (!identical(as.integer(mine$selected), as.integer(theirs$xselect()))) {
    stop(sprintf("%s: selection sequence differs from mboost", rhs))
  }
  cat(sprintf("OK %-58s maxdiff=%.2e\n", rhs, gap))
}

# One term at a time: each form is right alone, which is what made the menu
# defects hard to see.
compare("bols(x)")
compare("bols(x, intercept = FALSE)")
compare("bols(x, lambda = 2)")

# Penalized against unpenalized.
compare("bols(x, lambda = 2) + bols(z)")
compare("bols(x) + bols(z, lambda = 3)")

# Both intercept settings, mixed.
compare("bols(x) + bols(z, intercept = FALSE)")
compare("bols(x, intercept = FALSE) + bols(z)")
compare("bols(x, intercept = FALSE) + bols(z, intercept = FALSE)")

# The two crossed: a penalty forcing the block form beside a term that must
# not gain an intercept from it.
compare("bols(x, lambda = 2) + bols(z, intercept = FALSE)")
compare("bols(x, lambda = 2) + bols(z, intercept = FALSE) + bols(w)")
compare("bols(x, intercept = FALSE) + bols(z) + bols(w, lambda = 3)")

# `by = by` (numeric) is supported since issue #160: the menu gives every
# bols its own intercept once a by= term is present, as mboost does.
compare("bols(x, by = by) + bols(z)")
compare("bols(x, by = by) + bols(z, intercept = FALSE) + bols(w)")

# A smooth in the menu: the conversion must leave non-`bols` terms alone. The
# df-to-lambda solve carries its own tolerance, which is why this one is
# compared at the recorded parity rtol rather than at machine precision.
compare("bols(x) + bbs(z, df = 4)", tol = 1e-5)
compare("bols(x, intercept = FALSE) + bbs(z, df = 4)", tol = 1e-5)

cat("bols intercept parity: all menus match mboost\n")
