# residuals.mboostrs_gamboost read the response as as.double(data[[name]]).
# A two-class factor then became its integer codes 1/2 while mu is the
# probability of the second class, so every residual was off by one (range
# 0.11 to 1.84 on this fit), and a transformed response such as log(y) was
# replaced by the untransformed column (residuals off by up to 7.26). The fit
# and glmboost both read the response with the left-hand side evaluated and a
# two-class factor coded 0/1 (issue #823, item 3).
library(mboostrs)

ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(3)
n <- 150
d <- data.frame(x = rnorm(n), z = runif(n))
d$yb <- factor(ifelse(d$x + rnorm(n) > 0, "yes", "no"))

fb <- gamboost(yb ~ bbs(x) + bols(z), data = d, family = Binomial(),
               control = boost_control(mstop = 30))
y01 <- as.double(d$yb == "yes")
r <- residuals(fb)
ok(max(abs(r - (y01 - predict(fb, type = "response")))) < 1e-12,
   "binomial factor residuals are not y01 - mu")
ok(all(abs(r) < 1), "binomial residuals fall outside (-1, 1)")
rp <- residuals(fb, type = "pearson")
mu <- predict(fb, type = "response")
ok(max(abs(rp - (y01 - mu) / sqrt(mu * (1 - mu)))) < 1e-10,
   "binomial factor Pearson residuals use the wrong response coding")

d$yp <- exp(1 + 0.5 * d$x + rnorm(n, sd = 0.2))
fl <- gamboost(log(yp) ~ bbs(x) + bols(z), data = d,
               control = boost_control(mstop = 30))
ok(max(abs(residuals(fl) - (log(d$yp) - fitted(fl)))) < 1e-12,
   "residuals drop the log() transformation of the response")

# A plain numeric response is unchanged.
d$yg <- 1 + d$x + rnorm(n)
fg <- gamboost(yg ~ bbs(x) + bols(z), data = d, control = boost_control(mstop = 30))
ok(max(abs(residuals(fg) - (d$yg - fitted(fg)))) < 1e-12,
   "numeric-response residuals changed")

cat("test_gamboost_residual_response: OK\n")
