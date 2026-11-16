#@validates r.method.contract
library(mboostrs)

caps <- mboostrs_capabilities()
stopifnot(is.list(caps))
stopifnot(all(c("families", "learners", "methods") %in% names(caps)))

families <- family_capabilities()
stopifnot(is.data.frame(families))
stopifnot("Gaussian" %in% families$name)
stopifnot("AUC" %in% families$name)
# AUC gained a working fit path when the ranking families were implemented, and
# the manifest records fit = "yes". This assertion still read "no" from before
# that work, so it was failing against a manifest that had moved on.
stopifnot(families$fit[families$name == "AUC"] == "yes")
stopifnot(families$cdf[families$name == "GaussianLSS"] == "yes")
stopifnot(families$pdf_or_pmf[families$name == "GammaLSS"] == "yes")
stopifnot(families$simulate[families$name == "BetaLSS"] == "yes")
stopifnot(families$deviance_increment[families$name == "NBinomialLSS"] == "yes")
stopifnot(families$deviance_increment[families$name == "PoissonLSS"] == "yes")
stopifnot(families$cdf[families$name == "NBinomialLSS"] == "yes")
stopifnot(families$quantile[families$name == "PoissonLSS"] == "yes")
# GammaLSS, StudentTLSS, BetaLSS, and ZINBLSS gained CDF/quantile once the
# regularized incomplete gamma and beta functions were implemented Python-side
# (see tests/test_lss_special_functions.py). The generated table follows the
# manifest, so these assertions track it.
for (nm in c("GammaLSS", "StudentTLSS", "BetaLSS", "ZINBLSS")) {
  stopifnot(families$cdf[families$name == nm] == "yes")
  stopifnot(families$quantile[families$name == nm] == "yes")
}
stopifnot(families$fit[families$name == "PoissonLSS"] == "yes")

learners <- learner_capabilities()
stopifnot("Bols" %in% learners$name)
stopifnot("blackboost" %in% learners$name)

methods <- method_capabilities()
stopifnot(any(methods$model_class == "mboostrs_glmboostLSS" &
              methods$method == "selected" &
              methods$implemented == "yes"))
stopifnot(any(methods$model_class == "mboostrs_gamboostLSS" &
              methods$method == "model_weights" &
              methods$implemented == "yes"))
stopifnot(any(methods$model_class == "mboostrs_lss_cvrisk" &
              methods$method == "as.data.frame" &
              methods$implemented == "yes"))
stopifnot(any(methods$model_class == "mboostrs_glmboostLSS" &
              methods$method == "prediction_interval" &
              methods$implemented == "yes"))
stopifnot(any(methods$model_class == "mboostrs_glmboostLSS" &
              methods$method == "predictAll" &
              methods$implemented == "yes"))
stopifnot(any(methods$model_class == "mboostrs_gamboostLSS" &
              methods$method == "coefAll" &
              methods$implemented == "yes"))
stopifnot(any(methods$model_class == "mboostrs_glmboostLSS" &
              methods$method == "residual_frame" &
              methods$implemented == "yes"))
stopifnot(any(methods$model_class == "mboostrs_glmboostLSS" &
              methods$method == "q_stats" &
              methods$implemented == "yes"))

validations <- validation_capabilities()
stopifnot(is.data.frame(validations), nrow(validations) >= 40L)
stopifnot(!anyDuplicated(validations$id))
stopifnot(all(nzchar(validations$node_id)), all(nzchar(validations$oracle)))
stopifnot(families$python_fit_status[families$name == "Gaussian"] == "verified")
stopifnot(families$r_fit_status[families$name == "GaussianLSS"] == "verified")
# `unsupported` until P4.1 item 1 replaced a Poisson approximation with the
# exact PIG density and gradients. `smoke` rather than `verified`: the
# numerics are checked against gamlss.dist in Rust, and the frontend's own
# evidence is a simulation recovery against R's maximum likelihood.
stopifnot(families$python_fit_status[families$name == "PIGLSS"] == "smoke")
stopifnot(families$fit[families$name == "PIGLSS"] == "yes")

set.seed(2026)
n <- 36
dat <- data.frame(
  y = rnorm(n),
  x1 = rnorm(n),
  x2 = rnorm(n)
)

mod <- glmboostLSS(y ~ x1 + x2, data = dat, families = GaussianLSS(),
                   control = boost_control(mstop = 5, nu = 0.1))
sel <- selected(mod)
stopifnot(is.list(sel), all(c("mu", "sigma") %in% names(sel)))
stopifnot(identical(selected(mod, parameter = "mu"), sel$mu))
stopifnot(length(selected(mod, merge = TRUE)) == sum(vapply(sel, length, integer(1))))
stopifnot(is.numeric(fitted(mod, parameter = "mu")))
stopifnot(all.equal(fitted(mod, parameter = "mu"),
                    predict(mod, parameter = "mu", type = "link")))
# risk() follows gamboostLSS: one path per parameter, or merged.
stopifnot(identical(names(risk(mod)), c("mu", "sigma")),
          length(risk(mod)$mu) >= 2L,
          length(risk(mod, merge = TRUE)) == sum(lengths(risk(mod))))
stopifnot(length(model_weights(mod)) == n)
stopifnot(all(model_weights(mod) == 1.0))
stopifnot(identical(names(risk(mod, parameter = "mu")), "mu"))
err <- tryCatch(risk(mod, parameter = "nu"), error = function(e) e)
stopifnot(inherits(err, "error"))
params <- lss_params(mod)
stopifnot(all(c("mu", "sigma") %in% names(params)))
stopifnot(all.equal(params$mu, predict(mod, parameter = "mu", type = "response")))
pa <- predictAll(mod)
stopifnot(is.data.frame(pa), nrow(pa) == n)
stopifnot(all(c("y", "mu", "sigma") %in% names(pa)))
pa_list <- predictAll(mod, output = "list")
stopifnot(is.list(pa_list), all(c("mu", "sigma") %in% names(pa_list)))
ca <- coefAll(mod)
stopifnot(is.data.frame(ca), all(c("parameter", "learner", "coefficient") %in% names(ca)))
dens <- lss_pdf(GaussianLSS(), dat$y, params)
stopifnot(all.equal(dens, stats::dnorm(dat$y, mean = params$mu, sd = params$sigma)))
stopifnot(all.equal(lss_cdf(GaussianLSS(), params$mu, params), rep(0.5, n)))
stopifnot(all.equal(lss_quantile(GaussianLSS(), rep(0.5, n), params), params$mu))
stopifnot(all.equal(deviance_increment(mod),
                    lss_deviance_increment(GaussianLSS(), dat$y, params)))
pi80 <- prediction_interval(mod, level = 0.8)
stopifnot(is.data.frame(pi80), nrow(pi80) == n)
stopifnot(all(c("lower", "median", "upper", "level") %in% names(pi80)))
stopifnot(all.equal(predint(mod, level = 0.8), pi80))
stopifnot(all.equal(PI(mod, level = 0.8), pi80))
cent <- centiles(mod, probs = c(0.1, 0.5, 0.9))
stopifnot(is.data.frame(cent), all(c("q10", "q50", "q90") %in% names(cent)))
qr <- quantile_residuals(mod, randomized = FALSE)
stopifnot(length(qr) == n, all(is.finite(qr)))
pit_vals <- pit(mod, randomized = FALSE)
stopifnot(length(pit_vals) == n, all(pit_vals > 0.0 & pit_vals < 1.0))
rf <- residual_frame(mod, kind = "quantile", randomized = FALSE)
stopifnot(is.data.frame(rf), nrow(rf) == n)
stopifnot(all(c("y", "mu", "sigma", "residual", "pit", "deviance") %in% names(rf)))
rf_response <- residual_frame(mod, kind = "response")
stopifnot(is.data.frame(rf_response), all(c("residual_kind", "residual") %in% names(rf_response)))
qs <- q_stats(mod, xvar = "x1", n_intervals = 4, randomized = FALSE)
stopifnot(is.data.frame(qs), nrow(qs) == 4)
stopifnot(all(c("interval", "xvar", "x_min", "x_max", "n", "mean", "variance",
                "skewness", "kurtosis") %in% names(qs)))
set.seed(2026)
stopifnot(length(lss_simulate(GaussianLSS(), params)) == n)
dist_err <- tryCatch(lss_pdf(StudentTLSS(), dat$y, params), error = function(e) e)
stopifnot(inherits(dist_err, "error"))
gamma_params <- list(mu = c(1.0, 1.5, 2.5), sigma = c(2.0, 3.0, 4.0))
y_gamma <- c(0.5, 1.0, 2.0)
stopifnot(all.equal(lss_pdf(GammaLSS(), y_gamma, gamma_params),
                    stats::dgamma(y_gamma, shape = gamma_params$sigma,
                                  scale = gamma_params$mu / gamma_params$sigma)))
beta_params <- list(mu = c(0.3, 0.5, 0.7), phi = c(5.0, 8.0, 10.0))
y_beta <- c(0.2, 0.5, 0.8)
stopifnot(all.equal(lss_pdf(BetaLSS(), y_beta, beta_params),
                    stats::dbeta(y_beta, shape1 = beta_params$mu * beta_params$phi,
                                 shape2 = (1.0 - beta_params$mu) * beta_params$phi)))
nb_params <- list(mu = c(1.0, 3.0, 4.0), sigma = c(2.0, 5.0, 3.0))
y_nb <- c(0, 2, 5)
stopifnot(all.equal(lss_pdf(NBinomialLSS(), y_nb, nb_params),
                    stats::dnbinom(y_nb, mu = nb_params$mu, size = nb_params$sigma)))
stopifnot(all.equal(lss_cdf(NBinomialLSS(), y_nb, nb_params),
                    stats::pnbinom(y_nb, mu = nb_params$mu, size = nb_params$sigma)))
stopifnot(all.equal(lss_quantile(NBinomialLSS(), c(0.25, 0.5, 0.75), nb_params),
                    stats::qnbinom(c(0.25, 0.5, 0.75), mu = nb_params$mu, size = nb_params$sigma)))
pois_params <- list(mu = c(1.0, 3.0, 5.0))
y_pois <- c(0, 2, 4)
stopifnot(all.equal(lss_pdf(PoissonLSS(), y_pois, pois_params),
                    stats::dpois(y_pois, lambda = pois_params$mu)))
stopifnot(all.equal(lss_cdf(PoissonLSS(), y_pois, pois_params),
                    stats::ppois(y_pois, lambda = pois_params$mu)))
stopifnot(all.equal(lss_quantile(PoissonLSS(), c(0.25, 0.5, 0.75), pois_params),
                    stats::qpois(c(0.25, 0.5, 0.75), lambda = pois_params$mu)))
stopifnot(length(lss_simulate(GammaLSS(), gamma_params)) == length(y_gamma))
stopifnot(length(lss_simulate(BetaLSS(), beta_params)) == length(y_beta))
stopifnot(length(lss_simulate(NBinomialLSS(), nb_params)) == length(y_nb))
stopifnot(length(lss_simulate(PoissonLSS(), pois_params)) == length(y_pois))
cdf_err <- tryCatch(lss_cdf(GammaLSS(), y_gamma, gamma_params), error = function(e) e)
stopifnot(inherits(cdf_err, "error"))

pois_dat <- data.frame(y = rpois(n, lambda = 2.5), x1 = dat$x1, x2 = dat$x2)
pois_mod <- glmboostLSS(y ~ x1 + x2, data = pois_dat, families = PoissonLSS(),
                        control = boost_control(mstop = 4, nu = 0.1))
stopifnot(inherits(pois_mod, "mboostrs_glmboostLSS"))
stopifnot(names(lss_params(pois_mod))[1] == "mu")
stopifnot(length(deviance_increment(pois_mod)) == n)
pois_pi <- prediction_interval(pois_mod, level = 0.8)
stopifnot(nrow(pois_pi) == n, all(pois_pi$lower <= pois_pi$upper))
pois_qr <- quantile_residuals(pois_mod, randomized = FALSE)
stopifnot(length(pois_qr) == n, all(is.finite(pois_qr)))
pois_pit <- pit(pois_mod, randomized = FALSE)
stopifnot(length(pois_pit) == n, all(pois_pit > 0.0 & pois_pit < 1.0))
pois_rf <- residual_frame(pois_mod, kind = "quantile", randomized = FALSE)
stopifnot(is.data.frame(pois_rf), all(c("y", "mu", "residual", "pit", "deviance") %in% names(pois_rf)))
pois_qs <- q_stats(pois_mod, xvar = "x1", n_intervals = 4, randomized = FALSE)
stopifnot(is.data.frame(pois_qs), nrow(pois_qs) == 4)

mod3 <- update(mod, mstop = 3)
stopifnot(all(vapply(selected(mod3), length, integer(1)) == 3L))

folds <- matrix(1.0, nrow = n, ncol = 3)
for (j in seq_len(ncol(folds))) {
  folds[seq(j, n, by = ncol(folds)), j] <- 0.0
}
grid <- matrix(c(1, 1, 2, 1, 2, 2), ncol = 2, byrow = TRUE)
colnames(grid) <- c("mu", "sigma")
cv <- cvrisk_glmboostLSS(y ~ x1 + x2, data = dat, families = GaussianLSS(),
                         control = boost_control(mstop = 3, nu = 0.1),
                         folds = folds, grid = grid)
stopifnot(inherits(cv, "mboostrs_lss_cvrisk"))
stopifnot(is.matrix(cv), nrow(cv) == ncol(folds), ncol(cv) == nrow(grid))
stopifnot(length(attr(cv, "aggregate")) == nrow(grid))
stopifnot(length(mstop(cv)) == 2L)
cv_frame <- as.data.frame(cv)
stopifnot(nrow(cv_frame) == ncol(folds) * nrow(grid))
stopifnot(all(c("fold", "grid_index", "risk", "aggregate", "optimal") %in% names(cv_frame)))

ss <- stabsel_glmboostLSS(
  y ~ x1 + x2, data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 5, nu = 0.1),
  folds = folds, cutoff = 0.5, q = 2, sampling_type = "MB"
)
stopifnot(inherits(ss, "mboostrs_lss_stabsel"))
stopifnot(length(ss$names) == length(ss$max))
ss_frame <- as.data.frame(ss)
stopifnot(nrow(ss_frame) == length(ss$max))
stopifnot(all(c("learner", "max", "selected", "cutoff", "q", "PFER") %in% names(ss_frame)))

smooth <- gamboostLSS(y ~ bbs(x1, df = 4), data = dat,
                      families = GaussianLSS(),
                      control = boost_control(mstop = 4, nu = 0.1))
smooth_sel <- selected(smooth)
stopifnot(is.list(smooth_sel), all(c("mu", "sigma") %in% names(smooth_sel)))
stopifnot(length(fitted(smooth, parameter = "mu")) == n)
stopifnot(length(risk(smooth)$mu) >= 2L)
stopifnot(length(model_weights(smooth)) == n)
smooth_pa <- predictAll(smooth)
stopifnot(is.data.frame(smooth_pa), all(c("y", "mu", "sigma") %in% names(smooth_pa)))
smooth_ca <- coefAll(smooth)
stopifnot(is.data.frame(smooth_ca), all(c("parameter", "learner", "coefficient") %in% names(smooth_ca)))
smooth_rf <- residual_frame(smooth, kind = "response")
stopifnot(is.data.frame(smooth_rf), all(c("y", "mu", "sigma", "residual") %in% names(smooth_rf)))
smooth_qs <- q_stats(smooth, n_intervals = 4, randomized = FALSE)
stopifnot(is.data.frame(smooth_qs), nrow(smooth_qs) == 4)

cat("capabilities and LSS accessor smoke: PASS\n")
