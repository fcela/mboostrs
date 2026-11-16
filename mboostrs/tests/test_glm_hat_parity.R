# Hat values and classical AIC for GLM families match mboost.
#
# mboost weights each step of a Binomial or Poisson hat matrix by the IRLS
# weight fW at the fit after that step. This frontend used the unweighted
# matrix, ignored the bbs penalty in the exact gamboost hat values (0.7 off
# for Gaussian), took gamboost AIC degrees of freedom from a per-learner df
# sum instead of the trace, and refused a factor Binomial response with
# "Must not be NA". The hat matrix now comes from the Rust core.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping GLM hat parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(mboost))

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(42); n <- 300
d <- data.frame(x = runif(n, -1, 1), z = rnorm(n), w = runif(n))
eta <- sin(2 * d$x) + 0.5 * d$z
d$yg <- eta + rnorm(n, sd = .4)
d$yb <- factor(rbinom(n, 1, plogis(eta)))
d$yp <- rpois(n, exp(0.3 * eta))

cat("GLM hat parity against mboost:\n")
for (fn in c("Gaussian", "Binomial", "Poisson")) {
  resp <- c(Gaussian = "yg", Binomial = "yb", Poisson = "yp")[[fn]]
  meth <- if (fn == "Gaussian") "corrected" else "classical"
  for (kind in c("gam", "glm")) {
    f <- as.formula(paste(resp, if (kind == "gam") "~ bbs(x) + bols(z)" else "~ x + z + w"))
    up_fit <- if (kind == "gam") mboost::gamboost else mboost::glmboost
    our_fit <- if (kind == "gam") mboostrs::gamboost else mboostrs::glmboost
    up <- up_fit(f, data = d, family = get(fn, asNamespace("mboost"))(),
                 control = mboost::boost_control(mstop = 60))
    ours <- our_fit(f, data = d, family = get(fn, asNamespace("mboostrs"))(),
                    control = mboostrs::boost_control(mstop = 60))
    # mboost's cumsum prediction for a Binomial glmboost is a constant 1.6e-4
    # off its own fitted values here; fW carries that into the hat matrix.
    tol <- if (fn == "Binomial" && kind == "glm") c(1e-5, 1e-7) else c(1e-10, 1e-11)
    da <- max(abs(aic_path(AIC(ours, method = meth)) - attr(AIC(up, method = meth), "AIC")))
    dh <- max(abs(hatvalues(ours) - hatvalues(up)))
    cat(sprintf("  %-9s %s  AIC %.1e  hat %.1e\n", fn, kind, da, dh))
    ok(da < tol[1], paste(fn, kind, "AIC path differs from mboost by", da))
    ok(dh < tol[2], paste(fn, kind, "hat values differ from mboost by", dh))
  }
}
cat("GLM hat parity (R): PASS\n")
