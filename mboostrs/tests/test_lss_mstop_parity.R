# mstop() of an LSS model matches upstream gamboostLSS: a named per-parameter
# vector for a cyclic fit; for a non-cyclic fit the total number of updates
# with their division between parameters in attr(, "partitions").
library(mboostrs)
set.seed(1); n <- 300; x <- runif(n, -1, 1)
y <- sin(2 * x) + exp(0.5 * x) * rnorm(n)
d <- data.frame(x = x, y = y)
f <- list(mu = y ~ bbs(x), sigma = y ~ bbs(x))

cyc <- mboostrs::gamboostLSS(f, data = d, families = mboostrs::GaussianLSS(),
                             mstop = c(mu = 60, sigma = 30))
stopifnot(identical(mstop(cyc), c(mu = 60L, sigma = 30L)))

non <- mboostrs::gamboostLSS(f, data = d, families = mboostrs::GaussianLSS(),
                             control = boost_control(mstop = 90),
                             method = "noncyclic")
stopifnot(as.integer(mstop(non)) == 90L,
          sum(attr(mstop(non), "partitions")) == 90L,
          identical(names(attr(mstop(non), "partitions")), c("mu", "sigma")))

glm <- mboostrs::glmboostLSS(list(mu = y ~ x, sigma = y ~ x), data = d,
                             families = mboostrs::GaussianLSS(),
                             mstop = c(mu = 40, sigma = 20))
stopifnot(identical(mstop(glm), c(mu = 40L, sigma = 20L)))

if (requireNamespace("gamboostLSS", quietly = TRUE)) {
  up <- gamboostLSS::gamboostLSS(y ~ mboost::bbs(x), data = d,
                                 families = gamboostLSS::GaussianLSS(),
                                 control = mboost::boost_control(mstop = 90),
                                 method = "noncyclic")
  stopifnot(as.integer(mstop(non)) == as.integer(mboost::mstop(up)))
  stopifnot(identical(names(attr(mboost::mstop(up), "partitions")),
                      names(attr(mstop(non), "partitions"))))
}
cat("test_lss_mstop_parity.R: OK\n")
