#@validates parity.r.bct_bcpe_distribution

library(mboostrs)

# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
if (!requireNamespace("gamlss.dist", quietly = TRUE)) {
  cat("gamlss.dist unavailable; BCT/BCPE R oracle skipped\n")
  quit(status = 0L)
}

y <- c(0.15, 0.5, 1.2, 2.5, 6)
p <- c(0.01, 0.1, 0.5, 0.9, 0.99)
settings <- list(
  c(mu = 1.7, sigma = 0.4, nu = 0.3, tau = 5),
  c(mu = 2.4, sigma = 0.7, nu = -0.5, tau = 12),
  c(mu = 0.9, sigma = 0.25, nu = 0.01, tau = 2.2),
  c(mu = 1.2, sigma = 0.6, nu = 0, tau = 4)
)

for (setting in settings) {
  params <- as.list(setting)
  stopifnot(isTRUE(all.equal(
    lss_pdf(BCTLSS(), y, params),
    gamlss.dist::dBCT(y, setting["mu"], setting["sigma"],
                      setting["nu"], setting["tau"]),
    tolerance = 2e-11
  )))
  stopifnot(isTRUE(all.equal(
    lss_cdf(BCTLSS(), y, params),
    gamlss.dist::pBCT(y, setting["mu"], setting["sigma"],
                      setting["nu"], setting["tau"]),
    tolerance = 2e-11
  )))
  stopifnot(isTRUE(all.equal(
    lss_quantile(BCTLSS(), p, params),
    gamlss.dist::qBCT(p, setting["mu"], setting["sigma"],
                      setting["nu"], setting["tau"]),
    tolerance = 2e-9
  )))

  stopifnot(isTRUE(all.equal(
    lss_pdf(BCPELSS(), y, params),
    gamlss.dist::dBCPE(y, setting["mu"], setting["sigma"],
                       setting["nu"], setting["tau"]),
    tolerance = 2e-11
  )))
  stopifnot(isTRUE(all.equal(
    lss_cdf(BCPELSS(), y, params),
    gamlss.dist::pBCPE(y, setting["mu"], setting["sigma"],
                       setting["nu"], setting["tau"]),
    tolerance = 2e-11
  )))
  stopifnot(isTRUE(all.equal(
    lss_quantile(BCPELSS(), p, params),
    gamlss.dist::qBCPE(p, setting["mu"], setting["sigma"],
                       setting["nu"], setting["tau"]),
    tolerance = 2e-9
  )))
}

set.seed(91)
bct_draw <- lss_simulate(
  BCTLSS(),
  list(mu = rep(2, 40), sigma = rep(0.5, 40),
       nu = rep(0.3, 40), tau = rep(5, 40)),
  n = 40
)
set.seed(91)
bct_oracle <- gamlss.dist::qBCT(runif(40), 2, 0.5, 0.3, 5)
stopifnot(isTRUE(all.equal(bct_draw, bct_oracle, tolerance = 2e-9)))

set.seed(92)
bcpe_draw <- lss_simulate(
  BCPELSS(),
  list(mu = rep(2, 40), sigma = rep(0.5, 40),
       nu = rep(0.3, 40), tau = rep(2, 40)),
  n = 40
)
set.seed(92)
bcpe_oracle <- gamlss.dist::qBCPE(runif(40), 2, 0.5, 0.3, 2)
stopifnot(isTRUE(all.equal(bcpe_draw, bcpe_oracle, tolerance = 2e-9)))

stopifnot(inherits(try(
  lss_mean(BCTLSS(), list(mu = 2, sigma = 0.5, nu = 0.3, tau = 5)),
  silent = TRUE
), "try-error"))
stopifnot(inherits(try(
  lss_mean(BCPELSS(), list(mu = 2, sigma = 0.5, nu = 0.3, tau = 2)),
  silent = TRUE
), "try-error"))

#@validates parity.r.bct_bcpe_fit
set.seed(73)
n <- 120L
x1 <- rnorm(n)
x2 <- runif(n, -1, 1)
mu <- 2 + 0.25 * x1
u <- runif(n, 0.01, 0.99)
for (family_name in c("BCT", "BCPE")) {
  nu <- if (family_name == "BCT") 0.4 else 0.7
  tau <- if (family_name == "BCT") 6 else 2.5
  y <- if (family_name == "BCT") {
    gamlss.dist::qBCT(u, mu, 0.18, nu, tau)
  } else {
    gamlss.dist::qBCPE(u, mu, 0.18, nu, tau)
  }
  dat <- data.frame(y = y, x1 = x1, x2 = x2)
  local_family <- if (family_name == "BCT") BCTLSS() else BCPELSS()
  ours <- mboostrs::glmboostLSS(
    y ~ x1 + x2, data = dat, families = local_family,
    control = mboostrs::boost_control(mstop = 20, nu = 0.05)
  )
  reference <- gamboostLSS::glmboostLSS(
    y ~ x1 + x2, data = dat,
    families = gamboostLSS::as.families(family_name),
    control = mboost::boost_control(mstop = 20, nu = 0.05)
  )
  for (parameter in c("mu", "sigma", "nu", "tau")) {
    stopifnot(isTRUE(all.equal(
      as.numeric(predict(ours, parameter = parameter, type = "link")),
      as.numeric(predict(reference, parameter = parameter, type = "link")),
      tolerance = 2e-10
    )))
  }
  fitted_centiles <- centiles(ours, probs = c(0.1, 0.5, 0.9))
  fitted_interval <- PI(ours, level = 0.8)
  stopifnot(nrow(fitted_centiles) == n, nrow(fitted_interval) == n)
  stopifnot(all(is.finite(unlist(fitted_centiles))))
  stopifnot(all(is.finite(unlist(fitted_interval))))
}

cat("R BCT/BCPE distribution contract passed\n")
