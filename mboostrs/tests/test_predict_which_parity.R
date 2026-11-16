# predict(which = ) returns each learner's link-scale contribution, as mboost
# does, and coef() of a glmboost corrects the intercept for the centered
# predictors as mboost's coef.glmboost does. Before, R's gamboost predict
# ignored `which` (it went into `...`) and returned the whole prediction, and
# glmboost coef() reported the intercept learner's raw coefficient.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping predict(which) parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(mboost))

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(7); n <- 250
d <- data.frame(x = runif(n, -1, 1), z = rnorm(n), w = runif(n))
d$y <- sin(2 * d$x) + 0.5 * d$z + rnorm(n, sd = .4)
nd <- data.frame(x = c(-0.5, 0.1, 0.7), z = c(0.3, -1, 0.2), w = c(0.2, 0.5, 0.9))

up <- mboost::gamboost(y ~ bbs(x) + bols(z) + bbs(w, df = 3), data = d,
                       control = mboost::boost_control(mstop = 80))
ours <- mboostrs::gamboost(y ~ bbs(x) + bols(z) + bbs(w, df = 3), data = d,
                           control = mboostrs::boost_control(mstop = 80))
for (args in list(list(which = 1), list(which = c(1, 3)), list(which = 2, mstop = 30))) {
  for (data in list(NULL, nd)) {
    a <- do.call(predict, c(list(ours, newdata = data), args))
    b <- if (is.null(args$mstop)) do.call(predict, c(list(up, newdata = data), args["which"]))
         else predict(up[args$mstop], newdata = data, which = args$which)
    diff <- max(abs(unclass(a) - unclass(b)))
    ok(diff < 1e-10, paste("predict(which) differs from mboost by", diff))
    ok(ncol(a) == length(args$which), "one column per learner")
  }
}
ok(isTRUE(all.equal(predict(ours, newdata = nd, which = "z"),
                    predict(ours, newdata = nd, which = 2))), "which by name")
ok(inherits(try(predict(ours, which = 1, type = "response"), silent = TRUE),
            "try-error"), "a response-scale contribution should be refused")

upg <- mboost::glmboost(y ~ x + z + w, data = d, control = mboost::boost_control(mstop = 80))
og <- mboostrs::glmboost(y ~ x + z + w, data = d, control = mboostrs::boost_control(mstop = 80))
for (o2i in c(FALSE, TRUE)) {
  diff <- max(abs(coef(og, off2int = o2i) - coef(upg, off2int = o2i)))
  ok(diff < 1e-12, paste("glmboost coef differs from mboost by", diff))
}
for (args in list(list(which = 2), list(which = c(1, 3, 4)), list(which = "z", mstop = 30))) {
  for (data in list(NULL, nd)) {
    a <- do.call(predict, c(list(og, newdata = data), args))
    b <- if (is.null(args$mstop)) do.call(predict, c(list(upg, newdata = data), args["which"]))
         else predict(upg[args$mstop], newdata = data, which = args$which)
    diff <- max(abs(unclass(a) - unclass(b)))
    ok(diff < 1e-10, paste("glmboost predict(which) differs from mboost by", diff))
  }
}
# An argument mboost takes and these methods do not is refused, not dropped.
for (call in list(quote(predict(ours, aggregate = "cumsum")), quote(predict(og, whch = 1)),
                  quote(coef(og, aggregate = "none")))) {
  ok(inherits(try(eval(call), silent = TRUE), "try-error"),
     paste("an unknown argument was accepted:", deparse(call)))
}
cat("predict(which) and glmboost coef parity (R): PASS\n")
