#@validates parity.r.hurdle
# Hurdle() matches mboost::Hurdle(): the count part of a hurdle model, a
# zero-truncated negative binomial whose dispersion is re-estimated at every
# step on (0, 100).
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping Hurdle parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(mboost))

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(3); n <- 600
d <- data.frame(x = runif(n, -1, 1), z = rnorm(n))
d$y <- rnbinom(n, mu = exp(1 + sin(2 * d$x) + 0.3 * d$z), size = 1.5)
d <- d[d$y > 0, ]

cat("Hurdle parity against mboost:\n")
for (kind in c("glm", "gam")) {
  f <- if (kind == "glm") y ~ x + z else y ~ bbs(x) + bols(z)
  up_fit <- if (kind == "glm") mboost::glmboost else mboost::gamboost
  our_fit <- if (kind == "glm") mboostrs::glmboost else mboostrs::gamboost
  up <- up_fit(f, data = d, family = mboost::Hurdle(),
               control = mboost::boost_control(mstop = 100))
  ours <- our_fit(f, data = d, family = mboostrs::Hurdle(),
                  control = mboostrs::boost_control(mstop = 100))
  dfit <- max(abs(as.numeric(fitted(up)) - as.numeric(ours$fitted)))
  drisk <- max(abs(up$risk() - ours$risk))
  dsig <- abs(mboost::nuisance(up) - mboostrs::nuisance(ours))
  cat(sprintf("  %s  fitted %.1e  risk %.1e  sigma %.1e\n", kind, dfit, drisk, dsig))
  ok(dfit < 1e-9, paste(kind, "fitted values differ from mboost by", dfit))
  ok(drisk < 1e-8, paste(kind, "risk path differs from mboost by", drisk))
  ok(dsig < 1e-9, paste(kind, "dispersion differs from mboost by", dsig))
}
ok(inherits(try(glmboost(y ~ x, data = transform(d, y = y + 0.5), family = Hurdle()),
                silent = TRUE), "try-error"), "a non-integer response should be refused")
cat("Hurdle parity (R): PASS\n")
