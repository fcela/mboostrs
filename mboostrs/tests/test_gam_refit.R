# REML refit (gam_refit): the numerics are in the Rust core; these checks
# compare the R binding with mgcv, which fits the same penalized model, and
# check the R-side contract (new groups, refusals, saveRDS).
library(mboostrs)

close <- function(a, b, tol, what) {
  if (!isTRUE(all(abs(a - b) <= tol))) {
    stop(what, ": ", paste(format(a), collapse = " "), " vs ",
         paste(format(b), collapse = " "), " (tolerance ", tol, ")", call. = FALSE)
  }
}

# Contract checks that need no reference package.
set.seed(3)
b <- rnorm(30)
d <- data.frame(g = rep(1:30, each = 20), x = runif(600, -1, 1))
d$y <- d$x + b[d$g] + rnorm(600, sd = 0.3)
fit <- gamboost(y ~ bbs(x) + brandom(g), data = d,
                control = boost_control(mstop = 2000, nu = 0.1))
r <- gam_refit(fit)
stopifnot(inherits(r, "mboostrs_gam_refit"), isTRUE(r$convergence$converged),
          isTRUE(r$conditional_on_selection), length(r$edf) == 2L)
new <- data.frame(x = c(0, 0), g = c(1, 999))
stopifnot(inherits(try(predict(r, new), silent = TRUE), "try-error"))
e <- effects(r, new, unseen_levels = "zero_effect")[["brandom(g)"]]
stopifnot(e$estimate[2] == 0, e$se[2] == 0, e$estimate[1] != 0)
# Effects sum to zero over the fitting rows (the refit's constraint).
close(sum(effects(r)[["bbs(x)"]]$estimate), 0, 1e-8, "centred effect")
# A saved refit predicts after reloading: it holds plain R data.
path <- tempfile(fileext = ".rds")
saveRDS(r, path)
close(predict(readRDS(path), d[1:5, ]), predict(r, d[1:5, ]), 0, "saveRDS round trip")
stopifnot(inherits(try(gam_refit(gamboost(y ~ bbs(x), data = d, family = Laplace())),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(gam_refit(gamboost(y ~ btree(x) + bbs(x), data = d), which = "all"),
                       silent = TRUE), "try-error"))

if (!requireNamespace("mgcv", quietly = TRUE) ||
    !requireNamespace("TH.data", quietly = TRUE)) {
  message("mgcv or TH.data not installed; skipping the mgcv comparisons")
  quit(save = "no")
}
suppressMessages(library(mgcv))
ps <- function(v) sprintf('s(%s, bs = "ps", k = 24, m = c(2, 2))', v)

# Gaussian: mgcv gives edf 1.000 and 1.749, sigma 3.99913. The small
# differences come from mgcv spreading its knots over a range 0.1% wider.
data("bodyfat", package = "TH.data")
fit <- gamboost(DEXfat ~ bbs(waistcirc) + bbs(hipcirc) + bols(age), data = bodyfat,
                control = boost_control(mstop = 200, nu = 0.1))
r <- gam_refit(fit)
m <- gam(as.formula(paste("DEXfat ~", ps("waistcirc"), "+", ps("hipcirc"), "+ age")),
         data = bodyfat, method = "REML")
close(r$edf[1:2], summary(m)$edf, 0.01, "bodyfat edf")
close(r$sigma, sqrt(m$sig2), 1e-3, "bodyfat sigma")
close(predict(r, bodyfat), predict(m, bodyfat), 1e-3, "bodyfat predictions")
close(as.numeric(logLik(r)), as.numeric(logLik(m)), 1e-3, "bodyfat logLik")
pdf(NULL); drawn <- plot(r); invisible(dev.off())
stopifnot(length(drawn) == 3L)
# Conditional standard errors match mgcv's; the corrected ones are wider.
se <- effects(r, bodyfat[1:6, ])[[2]]$se
close(se, predict(m, bodyfat[1:6, ], type = "terms", se.fit = TRUE)$se.fit[, "s(hipcirc)"], 1e-3,
      "bodyfat standard errors")
stopifnot(all(effects(r, bodyfat[1:6, ], corrected = TRUE)[[2]]$se >= se))

# Poisson and negative binomial (theta from the boosted fit, given to mgcv).
set.seed(11)
d <- data.frame(x = runif(1500, -2, 2))
d$y <- rpois(1500, exp(0.3 + sin(d$x)))
r <- gam_refit(gamboost(y ~ bbs(x), data = d, family = Poisson(),
                        control = boost_control(mstop = 300, nu = 0.1)))
m <- gam(as.formula(paste("y ~", ps("x"))), family = poisson, data = d, method = "REML")
close(r$edf, summary(m)$edf, 0.01, "Poisson edf")
close(predict(r, d), predict(m, d), 1e-3, "Poisson predictions")

set.seed(4)
d <- data.frame(x = runif(1500, -1, 1))
d$y <- rnbinom(1500, size = 2, mu = exp(-0.5 + 0.8 * sin(2 * d$x)))
fit <- gamboost(y ~ bbs(x), data = d, family = NBinomial(),
                control = boost_control(mstop = 1500, nu = 0.1))
r <- gam_refit(fit)
# theta is estimated by the refit's REML criterion, as mgcv's nb() does.
m <- gam(as.formula(paste("y ~", ps("x"))), family = nb(), data = d, method = "REML")
close(r$theta, m$family$getTheta(TRUE), 1e-3, "negative binomial theta")
close(r$edf, summary(m)$edf, 0.01, "negative binomial edf")
close(predict(r, d), predict(m, d), 1e-3, "negative binomial predictions")

# Gamma with a log link: the shape is estimated by the refit's criterion.
set.seed(11)
d <- data.frame(x = runif(2000, -2, 2))
d$y <- rgamma(2000, shape = 5, scale = exp(0.3 + sin(d$x)) / 5)
r <- gam_refit(gamboost(y ~ bbs(x), data = d, family = GammaReg(),
                        control = boost_control(mstop = 300, nu = 0.1)))
stopifnot(identical(r$convergence$method, "outer"))
m <- gam(as.formula(paste("y ~", ps("x"))), family = Gamma(link = "log"), data = d,
         method = "REML")
close(r$edf, summary(m)$edf, 0.01, "Gamma edf")
close(predict(r, d), predict(m, d), 1e-3, "Gamma predictions")

# A random effect: mgcv's s(g, bs = "re").
set.seed(3)
b <- rnorm(30)
d <- data.frame(g = rep(1:30, each = 20), x = runif(600, -1, 1))
d$y <- d$x + b[d$g] + rnorm(600, sd = 0.3)
r <- gam_refit(gamboost(y ~ bbs(x) + brandom(g), data = d,
                        control = boost_control(mstop = 2000, nu = 0.1)))
d$gf <- factor(d$g)
m <- gam(as.formula(paste("y ~", ps("x"), '+ s(gf, bs = "re")')), data = d, method = "REML")
close(r$edf, summary(m)$edf, 0.01, "random effect edf")
close(predict(r, d), predict(m, d), 1e-4, "random effect predictions")

# A spatial surface: mgcv's te() with np = FALSE uses the same Kronecker
# penalties on the B-spline coefficients (np = TRUE reparametrizes the
# margins, which changes the penalty).
set.seed(6)
d <- data.frame(lon = runif(800), lat = runif(800))
d$y <- exp(-((d$lon - 0.3)^2 + (d$lat - 0.7)^2) / 0.05) + rnorm(800, sd = 0.3)
r <- gam_refit(gamboost(y ~ bspatial(lon, lat), data = d,
                        control = boost_control(mstop = 500, nu = 0.1)), which = "all")
stopifnot(length(r$smoothing[[1]]) == 2L)
m <- gam(y ~ te(lon, lat, bs = "ps", k = c(14, 14), m = c(2, 2), np = FALSE),
         data = d, method = "REML")
close(r$edf, sum(summary(m)$edf), 0.1, "spatial edf")
close(predict(r, d), predict(m, d), 2e-3, "spatial predictions")

# Cox: the Poisson form with a Breslow baseline, then the outer LAML
# iteration. The REML criterion has two local minima for x3 on this fixture:
# mgcv's default search stops at edf 2.60 (REML score 3742.703); started
# with x3 fully smoothed, mgcv reaches a lower score (3742.642) with edf
# 5.516 1 1.002 1.782 1.001 1.731 1 1.001, which the refit matches.
if (requireNamespace("survival", quietly = TRUE)) {
  suppressMessages(library(survival))
  cx <- read.csv("fixtures/cox_refit.csv")
  rhs <- paste(sprintf("bbs(x%d, df = 4)", 1:8), collapse = " + ")
  fit <- gamboost(as.formula(paste("Surv(time, status) ~", rhs)), data = cx,
                  family = CoxPH(), control = boost_control(mstop = 300, nu = 0.1))
  r <- gam_refit(fit, which = "all")
  stopifnot(identical(r$convergence$method, "outer"))
  close(r$edf, c(5.516, 1, 1.002, 1.782, 1.001, 1.731, 1, 1.001), 0.02, "Cox edf")
}

# GAMLSS: the joint refit against mgcv's gaulss, compared on the fitted mean
# and standard deviation (gaulss models 1/sigma, so coefficients differ).
set.seed(1)
n <- 600
d <- data.frame(x1 = runif(n), x2 = runif(n))
d$y <- sin(6 * d$x1) + exp(-0.7 + d$x2) * rnorm(n)
fit <- gamboostLSS(list(mu = y ~ bbs(x1) + bbs(x2), sigma = y ~ bbs(x1) + bbs(x2)),
                   data = d, families = GaussianLSS(),
                   control = boost_control(mstop = 500, nu = 0.1))
r <- gamlss_refit(fit, which = "all")
stopifnot(isTRUE(r$convergence$converged), !r$conditional_on_selection)
m <- gam(list(as.formula(paste("y ~", ps("x1"), "+", ps("x2"))),
              as.formula(paste("~", ps("x1"), "+", ps("x2")))),
         family = gaulss(), data = d, method = "REML")
close(c(r$edf$mu, r$edf$sigma), summary(m)$edf, 0.05, "gaulss edf")
pr <- predict(r, d, type = "response")
pm <- predict(m, d, type = "response")
close(pr$mu, pm[, 1], 5e-3, "gaulss mean")
close(pr$sigma, 1 / pm[, 2], 5e-3, "gaulss standard deviation")
e <- effects(r, d[1:10, ], parameter = "sigma")
stopifnot(all(effects(r, d[1:10, ], parameter = "sigma", corrected = TRUE)[[1]]$se >=
                e[[1]]$se - 1e-12))

# Posterior draws: the median draw of the 90th centile is near the point value.
dr <- gamlss_draws(r, d[1:5, ], 2000, seed = 2)
stopifnot(identical(dim(dr$mu), c(2000L, 5L)), all(dr$sigma > 0))
q90 <- dr$mu + qnorm(0.9) * dr$sigma
point <- pr$mu[1:5] + qnorm(0.9) * pr$sigma[1:5]
close(apply(q90, 2, stats::median), point, 0.05, "centile draws")

# glmboostLSS: linear terms refitted unpenalized; mgcv's gaulss links sigma
# through its own shifted log, so the fits agree to about 1e-3, not exactly.
set.seed(1)
n <- 500
d <- data.frame(x1 = runif(n), x2 = runif(n), x3 = runif(n))
d$y <- 1 + 2 * d$x1 + exp(-0.5 + d$x2) * rnorm(n)
g <- glmboostLSS(y ~ x1 + x2 + x3, data = d, families = GaussianLSS(),
                 control = boost_control(mstop = 500, nu = 0.1))
r <- gamlss_refit(g, which = "all")
m <- gam(list(y ~ x1 + x2 + x3, ~ x1 + x2 + x3), family = gaulss(), data = d)
p <- predict(r, d, type = "response")
pm <- predict(m, d, type = "response")
close(p$mu, pm[, 1], 2e-3, "glmboostLSS mean")
close(p$sigma, 1 / pm[, 2], 5e-3, "glmboostLSS standard deviation")

cat("gam_refit checks passed\n")
