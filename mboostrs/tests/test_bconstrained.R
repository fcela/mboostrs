# bconstrained() in R gamboost (issue #69). gamboost refused every
# bconstrained() term as an unknown type, although the binding fitted it.
# Each step now solves the constrained problem exactly by default.
suppressMessages(library(mboostrs))
set.seed(3); n <- 150
d <- data.frame(x = runif(n))
d$y <- sqrt(d$x) + rnorm(n, sd = 0.15)
p <- 20 + 3 + 1
C <- diag(p)                      # every B-spline coefficient non-negative
fit <- gamboost(y ~ bconstrained(x, C = C), data = d, control = boost_control(mstop = 40))
stopifnot(min(unlist(fit$coefs)) > -1e-12)
stopifnot(max(abs(predict(fit, newdata = d) - fit$fitted)) < 1e-10)

approx <- gamboost(y ~ bconstrained(x, C = C, type = "iterative"), data = d,
                   control = boost_control(mstop = 40))
stopifnot(min(unlist(approx$coefs)) < -1e-8)   # the approximate solver's violation

D <- diff(diag(p))                # increasing
inc <- gamboost(y ~ bconstrained(x, C = D), data = d, control = boost_control(mstop = 100))
grid <- data.frame(x = seq(min(d$x), max(d$x), length.out = 500))
stopifnot(min(diff(predict(inc, newdata = grid))) > -1e-10)
cat("test_bconstrained.R: OK\n")
