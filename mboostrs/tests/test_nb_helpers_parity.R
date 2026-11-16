# NegativeBinomialLSS and ZINBLSS fit gamlss.dist NBI / ZINBI, whose sigma is
# the dispersion (size 1/sigma). The R density, CDF, quantile and simulation
# helpers used size = sigma.
library(mboostrs)
k <- 0:12
p <- list(mu = 4, sigma = 0.5)
size <- 1 / 0.5
stopifnot(isTRUE(all.equal(lss_pdf(NegativeBinomialLSS(), k, p),
                           stats::dnbinom(k, mu = 4, size = size))))
stopifnot(isTRUE(all.equal(lss_cdf(NegativeBinomialLSS(), k, p),
                           stats::pnbinom(k, mu = 4, size = size))))
p3 <- list(mu = 4, sigma = 0.5, nu = 0.3)
zinb <- 0.7 * stats::dnbinom(k, mu = 4, size = size) + ifelse(k == 0, 0.3, 0)
stopifnot(isTRUE(all.equal(lss_pdf(ZINBLSS(), k, p3), zinb)))
stopifnot(isTRUE(all.equal(lss_cdf(ZINBLSS(), k, p3), cumsum(zinb))))
if (requireNamespace("gamlss.dist", quietly = TRUE)) {
  stopifnot(isTRUE(all.equal(lss_cdf(NegativeBinomialLSS(), k, p),
                             gamlss.dist::pNBI(k, mu = 4, sigma = 0.5))))
  stopifnot(isTRUE(all.equal(lss_pdf(ZINBLSS(), k, p3),
                             gamlss.dist::dZINBI(k, mu = 4, sigma = 0.5, nu = 0.3))))
}
cat("test_nb_helpers_parity.R: OK\n")
