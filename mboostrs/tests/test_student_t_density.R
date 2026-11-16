library(mboostrs)

# The old gamma ratio overflowed at df around 340, returning NaN.
y <- c(-30, -2, 0, 1, 12)
for (df in c(0.5, 1, 4, 340, 400, 1000, 100000)) {
  params <- list(mu = 1, sigma = 2, df = df)
  density <- lss_pdf(StudentTLSS(), y, params)
  expected <- dt((y - 1) / 2, df) / 2
  stopifnot(all(is.finite(density)))
  stopifnot(max(abs(density / expected - 1)) < 2e-9)
  deviance <- lss_deviance_increment(StudentTLSS(), y, params)
  stopifnot(max(abs(deviance + 2 * log(expected))) < 1e-8)
}
params <- list(mu = c(0, 1), sigma = 2, df = c(400, 1000))
stopifnot(isTRUE(all.equal(lss_pdf(StudentTLSS(), 0, params),
                         dt((0 - params$mu) / 2, params$df) / 2)))
stopifnot(length(lss_pdf(StudentTLSS(), numeric(), params)) == 0)
