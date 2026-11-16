# coef() of a centered bols(x) reports mboost's [(Intercept), x], with and
# without observation weights.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping bols coef parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(mboost))

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(7); n <- 250
d <- data.frame(x = runif(n, -1, 1), z = rnorm(n))
d$y <- sin(2 * d$x) + 0.5 * d$z + rnorm(n, sd = .4)
w <- runif(n, 0.5, 2)
for (weighted in c(FALSE, TRUE)) {
  ww <- if (weighted) w else NULL
  up <- mboost::gamboost(y ~ bbs(x) + bols(z), data = d, weights = ww,
                         control = mboost::boost_control(mstop = 80))
  ours <- mboostrs::gamboost(y ~ bbs(x) + bols(z), data = d, weights = ww,
                             control = mboostrs::boost_control(mstop = 80))
  diff <- max(abs(coef(up)[[2]] - coef(ours)[[2]]))
  cat(sprintf("bols coef (weighted = %s): %.1e\n", weighted, diff))
  ok(diff < 1e-12, paste("bols coef differs from mboost by", diff))
  ok(identical(names(coef(ours)[[2]]), c("(Intercept)", "z")), "coef names differ from mboost")
}
# The term's curve from partial_effects() is mboost's term prediction.
up <- mboost::gamboost(y ~ bbs(x) + bols(z), data = d, weights = w,
                       control = mboost::boost_control(mstop = 80))
ours <- mboostrs::gamboost(y ~ bbs(x) + bols(z), data = d, weights = w,
                           control = mboostrs::boost_control(mstop = 80))
pe <- partial_effects(ours, which = "z")[[1]]
expected <- as.numeric(predict(up, newdata = data.frame(x = 0, z = pe$x), which = 2))
diff <- max(abs(pe$effect - expected))
cat(sprintf("weighted bols partial effect: %.1e\n", diff))
ok(diff < 1e-12, paste("weighted bols partial effect differs from mboost by", diff))
cat("bols coef parity (R): PASS\n")
