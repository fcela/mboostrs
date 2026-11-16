library(mboostrs)

family <- ZeroAdjustedGammaLSS()
params <- list(mu = rep(2, 3), sigma = rep(0.5, 3), nu = rep(0.3, 3))
y <- c(-1, -.Machine$double.xmin, 0)
stopifnot(identical(lss_pdf(family, y, params), c(0, 0, .3)))
stopifnot(identical(lss_cdf(family, y, params), c(0, 0, .3)))
p <- c(.1, .3, .8)
q <- lss_quantile(family, p, params)
stopifnot(q[1] == 0, q[2] == 0, q[3] > 0)
stopifnot(isTRUE(all.equal(lss_cdf(family, q[3], lapply(params, `[`, 3)), p[3])))

set.seed(9)
n <- 500
x <- rnorm(n)
zero <- runif(n) < .35
response <- ifelse(zero, 0, rgamma(n, shape = 4, scale = .5))
fit <- glmboostLSS(
  list(mu = response ~ x, sigma = response ~ x, nu = response ~ x),
  data.frame(x, response), family = family, control = boost_control(mstop = 20)
)
set.seed(1); first <- pit(fit)
set.seed(2); second <- pit(fit)
stopifnot(max(abs(first[zero] - second[zero])) > .05)
stopifnot(identical(first[!zero], second[!zero]))
deterministic <- pit(fit, randomized = FALSE)
fitted_params <- lss_params(fit, type = "response")
stopifnot(isTRUE(all.equal(deterministic[zero], .5 * fitted_params$nu[zero])))

# Every lattice family uses F(y-) = F(y-1), including its mass at zero.
count_families <- list(
  PoissonLSS(), NBinomialLSS(), NegativeBinomialLSS(), GeometricLSS(),
  BinomialLSS(), ZIPoLSS(), ZINBLSS()
)
for (candidate in count_families) {
  stopifnot(lss_cdf(candidate, -1, switch(candidate$rust_name,
    poissonlss = list(mu = 2),
    nbinomiallss = list(mu = 2, sigma = 1),
    negativebinomiallss = list(mu = 2, sigma = 1),
    geometriclss = list(mu = 2),
    binomiallss = list(mu = .4),
    zipolss = list(mu = 2, sigma = .2),
    zinblss = list(mu = 2, sigma = 1, nu = .2)
  )) == 0)
}
