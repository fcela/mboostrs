# predict.mboostrs_gamboost answers an in-sample, full-path request from the
# stored fitted values. That shortcut applied the plain inverse link before
# the PropOdds branch was reached, so predict(fit, type = "response") on a
# three-category PropOdds fit returned a vector of 150 sigmoids where
# predict(newdata =) returned the 150 x 3 matrix of category probabilities.
# The same shortcut ignored offset =, returning the fit's own offset: with
# offset = 5 the in-sample and newdata predictions differed by 3.97 at every
# row (issue #823, item 2).
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(2)
n <- 150
d <- data.frame(x = rnorm(n), z = runif(n))
d$y <- cut(d$x + rnorm(n), c(-Inf, -0.5, 0.5, Inf), labels = c("lo", "mid", "hi"))

fit <- gamboost(y ~ bbs(x) + bols(z), data = d, family = PropOdds(n_levels = 3),
                control = boost_control(mstop = 30))
p_in <- predict(fit, type = "response")
p_new <- predict(fit, newdata = d, type = "response")
ok(is.matrix(p_in) && identical(dim(p_in), c(150L, 3L)),
   "in-sample PropOdds response is not an n x 3 probability matrix")
ok(max(abs(rowSums(p_in) - 1)) < 1e-10, "PropOdds probabilities do not sum to one")
ok(max(abs(p_in - p_new)) < 1e-10,
   "in-sample and newdata PropOdds response predictions differ")
p10 <- predict(fit, mstop = 10, type = "response")
ok(identical(dim(p10), c(150L, 3L)),"PropOdds response at mstop = 10 is not n x 3")

d$yg <- 1 + d$x + rnorm(n)
g <- gamboost(yg ~ bbs(x) + bols(z), data = d, control = boost_control(mstop = 30))
o <- rep(5, n)
a <- predict(g, offset = o)
b <- predict(g, newdata = d, offset = o)
ok(max(abs(a - b)) < 1e-10, "in-sample predict(offset =) ignores the offset")
ok(max(abs(a - (fitted(g) - g$offset + 5))) < 1e-10,
   "predict(offset =) does not replace the fitted offset")
ok(max(abs(predict(g) - fitted(g))) < 1e-12, "in-sample predict() changed")

cat("test_gamboost_predict_fast_path: OK\n")
