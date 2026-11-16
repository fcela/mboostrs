# A weighted gamboost fit centres each scalar bols column at the weighted mean
# (prepare_scalar_bols in the core). The R frontend stored the unweighted
# mean(x) as the term's centring, so predict(newdata =) subtracted a different
# constant from the one the fit used: on this data the prediction on the
# training rows differed from fitted() by 0.036 at every row (issue #823,
# item 1). The binding now returns the constant the core used.
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(1)
n <- 200
d <- data.frame(x = rnorm(n), z = runif(n))
d$y <- 1 + 2 * d$x + sin(3 * d$z) + rnorm(n, sd = 0.3)
w <- rep(c(1, 3), length.out = n)

fit <- gamboost(y ~ x + bols(z) + bbs(z), data = d, weights = w,
                control = boost_control(mstop = 50))

ok(abs(fit$learner_centering[[1]] - weighted.mean(d$x, w)) < 1e-12,
   "scalar bols centring is not the weighted mean of x")
ok(abs(fit$learner_centering[[2]] - weighted.mean(d$z, w)) < 1e-12,
   "bols(z) centring is not the weighted mean of z")

gap <- max(abs(predict(fit, newdata = d) - fitted(fit)))
ok(gap < 1e-10,
   sprintf("predict(newdata = training data) differs from fitted() by %.3g", gap))

gap20 <- max(abs(predict(fit, newdata = d, mstop = 20) - fitted(fit[20])))
ok(gap20 < 1e-10,
   sprintf("predict(newdata =, mstop = 20) differs from fitted(fit[20]) by %.3g", gap20))

# An unweighted fit is unchanged: the weighted mean is the plain mean.
fit1 <- gamboost(y ~ x + bols(z), data = d, control = boost_control(mstop = 50))
ok(abs(fit1$learner_centering[[1]] - mean(d$x)) < 1e-12,
   "unweighted scalar bols centring changed")

cat("test_weighted_scalar_centering: OK\n")
