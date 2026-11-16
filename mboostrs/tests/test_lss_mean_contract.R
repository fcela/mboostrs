#@validates contract.r.lss_mean

library(mboostrs)

stopifnot(isTRUE(all.equal(
  lss_mean(ZIPoLSS(), list(mu = c(2, 4), sigma = c(0.25, 0.5))),
  c(1.5, 2)
)))
stopifnot(isTRUE(all.equal(
  lss_mean(ZINBLSS(), list(mu = c(2, 4), sigma = c(1, 2), nu = c(0.25, 0.5))),
  c(1.5, 2)
)))

mu <- c(0.5, 2)
zero <- c(0.2, 0.6)
stopifnot(isTRUE(all.equal(
  lss_mean(ZeroAdjustedPoissonLSS(), list(mu = mu, sigma = zero)),
  (1 - zero) * mu / (1 - exp(-mu))
)))

stopifnot(isTRUE(all.equal(
  lss_mean(Pareto2LSS(), list(mu = 2, sigma = 3)), 1
)))
stopifnot(inherits(try(
  lss_mean(Pareto2LSS(), list(mu = 2, sigma = 1)), silent = TRUE
), "try-error"))

mu_aft <- c(0.2, 0.5)
sigma_aft <- c(0.4, 0.8)
stopifnot(isTRUE(all.equal(
  lss_mean(LogNormalLSS(), list(mu = mu_aft, sigma = sigma_aft)),
  exp(mu_aft + sigma_aft^2 / 2)
)))
stopifnot(isTRUE(all.equal(
  lss_mean(WeibullLSS(), list(mu = mu_aft, sigma = sigma_aft)),
  exp(mu_aft + lgamma(1 + sigma_aft))
)))

stopifnot(inherits(try(
  lss_mean(BCCGLSS(), list(mu = 2, sigma = 0.2, nu = 1)), silent = TRUE
), "try-error"))

cat("R LSS mean contract passed\n")
