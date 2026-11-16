# logLik.mboostrs_gamboost returned -risk for every family. The Gaussian risk
# is the residual sum of squares, so for Gaussian that is not a
# log-likelihood: on this fit it gave -130.71 where glmboost, the Python
# frontend and the Gaussian density at the MLE sigma^2 = RSS / n give
# -202.52 for the same fitted values. A robust or ranking loss (Laplace here)
# returned its negated loss (-114.09) where the Python frontend refuses
# (issue #823, item 4).
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)
refuses <- function(expr, pattern, label) {
  msg <- tryCatch({ force(expr); NULL }, error = conditionMessage)
  ok(!is.null(msg) && grepl(pattern, msg), paste(label, "did not refuse:", msg))
}

set.seed(4)
n <- 150
d <- data.frame(x = rnorm(n), z = runif(n))
d$y <- 1 + d$x + rnorm(n)

g <- gamboost(y ~ bols(x) + bols(z), data = d, control = boost_control(mstop = 50))
gl <- glmboost(y ~ x + z, data = d, control = boost_control(mstop = 50))
rss <- sum((d$y - fitted(g))^2)
reference <- -n / 2 * log(2 * pi * rss / n) - n / 2
ok(abs(as.numeric(logLik(g)) - reference) < 1e-8,
   sprintf("gamboost Gaussian logLik %.4f is not the Gaussian log-likelihood %.4f",
           as.numeric(logLik(g)), reference))
ok(abs(as.numeric(logLik(g)) - as.numeric(logLik(gl))) < 1e-6,
   "gamboost and glmboost Gaussian logLik differ on the same linear model")
ok(inherits(logLik(g), "logLik") && attr(logLik(g), "nobs") == n,
   "gamboost logLik lost its class or nobs")

# A likelihood family still returns -risk.
d$c <- rpois(n, exp(0.2 + 0.3 * d$x))
p <- gamboost(c ~ bbs(x), data = d, family = Poisson(), control = boost_control(mstop = 30))
ok(abs(as.numeric(logLik(p)) + p$risk[31]) < 1e-12, "Poisson logLik is not -risk")

lp <- gamboost(y ~ bbs(x), data = d, family = Laplace(), control = boost_control(mstop = 20))
refuses(logLik(lp), "not a log-likelihood", "gamboost Laplace logLik")
qg <- glmboost(y ~ x, data = d, family = Quantile(tau = 0.3), control = boost_control(mstop = 20))
refuses(logLik(qg), "not a log-likelihood", "glmboost Quantile logLik")

cat("test_gamboost_loglik: OK\n")
