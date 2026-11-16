# Split conformal must report a covariate whose third is under-covered, and must
# stay silent under constant noise.
suppressMessages(library(mboostrs))
ns <- asNamespace("mboostrs")

sim <- function(seed, hetero, n = 1000) {
  set.seed(seed)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  scale <- if (hetero) 0.3 + 1.5 * abs(x1) else rep(1, n)
  data.frame(y = sin(2 * pi * x1) + 0.5 * x2 + scale * rnorm(n), x1 = x1, x2 = x2)
}
collect_warnings <- function(expr) {
  msgs <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, messages = msgs)
}

# 1. Uneven coverage: noise that grows toward both tails is flagged on |x1 - median|.
d <- sim(0, hetero = TRUE)
fit <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d, family = Gaussian())
out <- collect_warnings(conformal_interval(fit, level = 0.9, seed = 1, method = "split"))
stopifnot(length(out$messages) == 1L)
stopifnot(grepl("|x1 - median|", out$messages[1], fixed = TRUE))
stopifnot(grepl("method='cqr'", out$messages[1], fixed = TRUE))

# 2. Constant noise is silent.
d0 <- sim(0, hetero = FALSE)
fit0 <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d0, family = Gaussian())
out0 <- collect_warnings(conformal_interval(fit0, level = 0.9, seed = 1, method = "split"))
stopifnot(length(out0$messages) == 0L)

cat("test_conformal_uneven_coverage.R: all checks passed\n")
