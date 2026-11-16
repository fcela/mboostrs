# predict(newdata=, extrapolation=) on spline learners (issue #67).
suppressMessages(library(mboostrs))

set.seed(1)
d <- data.frame(x = runif(200), z = runif(200))
d$y <- sin(2 * pi * d$x) + d$z + rnorm(200, sd = 0.2)
fit <- gamboost(y ~ bbs(x) + bols(z), data = d,
                control = boost_control(mstop = 50, nu = 0.1))

inside <- data.frame(x = c(0.2, 0.5), z = c(0.5, 0.5))
far <- data.frame(x = c(0.5, 1.5), z = c(0.5, 0.5))
near <- data.frame(x = c(0.5, max(d$x) + 1e-3), z = c(0.5, 0.5))
edge <- data.frame(x = c(0.5, max(d$x)), z = c(0.5, 0.5))

expect_no_warning <- function(expr) {
  w <- NULL
  val <- withCallingHandlers(expr, warning = function(c) {
    w <<- c(w, conditionMessage(c)); invokeRestart("muffleWarning")
  })
  if (length(w)) stop("unexpected warning: ", w[[1]])
  val
}

# Inside the range: no message under any policy.
for (p in c("warn", "error", "clip", "allow")) {
  expect_no_warning(predict(fit, newdata = inside, extrapolation = p))
}

# warn: one warning naming the variable and the count.
msg <- tryCatch(predict(fit, newdata = far), warning = conditionMessage)
stopifnot(is.character(msg), grepl("x: 1 above", msg), grepl("training ranges out", msg))

# A small excursion still extrapolates but is below the warning threshold.
invisible(expect_no_warning(predict(fit, newdata = near)))

# error refuses every value strictly outside, however small.
err <- tryCatch(predict(fit, newdata = near, extrapolation = "error"),
                error = conditionMessage)
stopifnot(is.character(err), grepl("outside the training range", err))

# clip evaluates at the training boundary.
clipped <- suppressWarnings(predict(fit, newdata = far, extrapolation = "clip"))
at_edge <- predict(fit, newdata = edge)
stopifnot(isTRUE(all.equal(clipped, at_edge, tolerance = 1e-12)))

# allow predicts silently and continues the edge slope (differs from clip).
allowed <- expect_no_warning(predict(fit, newdata = far, extrapolation = "allow"))
stopifnot(abs(allowed[2] - clipped[2]) > 1e-6)
stopifnot(isTRUE(all.equal(allowed, suppressWarnings(predict(fit, newdata = far)))))

# bspatial checks each coordinate.
d2 <- data.frame(a = runif(200), b = runif(200))
d2$y <- d2$a * d2$b + rnorm(200, sd = 0.1)
sfit <- gamboost(y ~ bspatial(a, b, df = 4), data = d2,
                 control = boost_control(mstop = 20))
serr <- tryCatch(predict(sfit, newdata = data.frame(a = 0.5, b = 2),
                         extrapolation = "error"), error = conditionMessage)
stopifnot(is.character(serr), grepl("bspatial\\(a, b\\) \\(b\\)", serr))

cat("test_extrapolation_policy.R: OK\n")

# gamboostLSS applies the same policy to every parameter's spline learners.
set.seed(5); n <- 200
dl <- data.frame(x = runif(n), w = runif(n))
dl$y <- rnorm(n, sin(3 * dl$x), exp(0.5 * dl$w))
lss <- gamboostLSS(list(mu = y ~ bbs(x), sigma = y ~ bbs(w)), data = dl,
                   families = GaussianLSS(), control = boost_control(mstop = 40))
out <- data.frame(x = c(0.5, 1.6), w = c(0.5, 0.5))
msg <- tryCatch(predict(lss, newdata = out), warning = conditionMessage)
stopifnot(grepl("mu: x", msg))
err <- tryCatch(predict(lss, newdata = data.frame(x = 0.5, w = 1.01), extrapolation = "error"),
                error = conditionMessage)
stopifnot(grepl("sigma: w", err))
edge <- data.frame(x = c(0.5, max(dl$x)), w = c(0.5, 0.5))
clipped <- suppressWarnings(predict(lss, newdata = out, extrapolation = "clip"))
stopifnot(isTRUE(all.equal(clipped, predict(lss, newdata = edge), tolerance = 1e-12)))
cat("test_extrapolation_policy.R (LSS): OK\n")
