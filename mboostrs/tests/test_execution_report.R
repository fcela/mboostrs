# Structured execution reports are stable across the R boundary and model
# lifecycle operations. Run after `R CMD INSTALL mboostrs`.
library(mboostrs)

set.seed(42)
n <- 60L
dat <- data.frame(y = rnorm(n), x = rnorm(n), z = runif(n))

plain <- glmboost(
  y ~ x + z, dat,
  control = boost_control(mstop = 6, nu = 0.1)
)
plain_report <- execution_report(plain)
stopifnot(
  identical(plain_report$schema_version, 1L),
  identical(plain_report$instrumentation, "off"),
  identical(plain_report$route, "packed_scalar"),
  identical(plain_report$requested_mstop, 6L),
  identical(plain_report$completed_mstop, 6L),
  identical(plain_report$n_observations, n),
  identical(plain_report$n_learners, 3L),
  identical(plain_report$scalar_learner_count, 3L),
  identical(plain_report$block_learner_count, 0L),
  identical(plain_report$tree_learner_count, 0L),
  identical(plain_report$cholesky_solver_count, 0L),
  identical(plain_report$lu_solver_count, 0L),
  identical(plain_report$regularized_solver_count, 0L),
  is.na(plain_report$max_solver_jitter),
  identical(plain_report$factor_cache_bytes, 0),
  identical(plain_report$gram_cache_bytes, 0),
  identical(plain_report$calibration_count, 0L),
  identical(plain_report$tie_selection_count, 0L),
  identical(plain_report$calibration_converged_count, 0L),
  identical(plain_report$calibration_limit_count, 0L),
  identical(plain_report$calibration_failure_count, 0L),
  identical(plain_report$fallback_reasons, "pure_scalar_model"),
  is.na(plain_report$total_ns)
)

sliced <- plain[3]
stopifnot(
  identical(execution_report(sliced)$route, plain_report$route),
  identical(execution_report(sliced)$requested_mstop, 6L),
  identical(execution_report(sliced)$completed_mstop, 3L)
)
restored <- unserialize(serialize(plain, NULL))
stopifnot(identical(execution_report(restored), plain_report))

tie_dat <- data.frame(
  y = seq(-1, 1, length.out = 30L),
  x1 = seq(-1, 1, length.out = 30L),
  x2 = seq(-1, 1, length.out = 30L)
)
tied <- glmboost(
  y ~ x1 + x2, tie_dat,
  control = boost_control(mstop = 4L, nu = 0.1)
)
stopifnot(
  identical(length(unique(tied$selected)), 1L),
  identical(execution_report(tied)$tie_selection_count, 4L)
)

plain_path <- path_frame(plain)
stopifnot(
  identical(nrow(plain_path), 6L),
  identical(plain_path$iteration, 1:6),
  all(is.na(plain_path$cycle)),
  all(is.na(plain_path$parameter)),
  identical(plain_path$risk, plain$risk[-1]),
  isTRUE(all.equal(plain_path$risk_change, -diff(plain$risk), tolerance = 0)),
  all(plain_path$learner_index >= 1L),
  identical(nrow(path_frame(sliced)), 3L),
  identical(path_frame(unserialize(serialize(plain, NULL))), plain_path)
)

traced <- suppressMessages(capture.output(
  traced_fit <- glmboost(
    y ~ x + z, dat,
    control = boost_control(mstop = 6, nu = 0.1, trace = TRUE)
  )
))
traced_report <- execution_report(traced_fit)
stopifnot(
  identical(traced_report$instrumentation, "summary"),
  is.numeric(traced_report$total_ns),
  traced_report$total_ns >= 0,
  is.numeric(traced_report$binding_ns),
  traced_report$binding_ns >= 0,
  is.numeric(traced_report$basis_ns),
  traced_report$basis_ns >= 0,
  is.numeric(traced_report$factorization_ns),
  traced_report$factorization_ns >= 0,
  is.na(traced_report$cross_gram_ns),
  is.na(traced_report$reconstruction_ns),
  isTRUE(all.equal(fitted(plain), fitted(traced_fit), tolerance = 0))
)

smooth <- gamboost(
  y ~ bbs(x, df = 4), dat,
  control = boost_control(mstop = 6, nu = 0.1)
)
smooth_report <- execution_report(smooth)
stopifnot(
  identical(smooth_report$route, "gaussian_gram"),
  isTRUE(smooth_report$gram_eligible),
  identical(smooth_report$block_learner_count, 1L),
  identical(smooth_report$cholesky_solver_count, 1L),
  smooth_report$factor_cache_bytes > 0,
  smooth_report$gram_cache_bytes > 0,
  length(smooth_report$fallback_reasons) == 0L
)
stopifnot(
  identical(nrow(path_frame(smooth)), 6L),
  all(path_frame(smooth)$learner == "x")
)

poisson_dat <- transform(dat, y = rpois(n, exp(0.2 + 0.1 * x)))
general <- gamboost(
  y ~ bbs(x, df = 4), poisson_dat, family = Poisson(),
  control = boost_control(mstop = 3)
)
general_report <- execution_report(general)
stopifnot(
  identical(general_report$route, "general"),
  !general_report$gram_eligible,
  identical(general_report$block_learner_count, 1L),
  identical(general_report$cholesky_solver_count, 1L),
  identical(general_report$gram_cache_bytes, 0),
  "family_does_not_support_gram" %in% general_report$fallback_reasons
)

count_dat <- transform(dat, y = pmax(0, round(exp(0.5 + x))))
negative_binomial <- glmboost(
  y ~ x, count_dat, family = NBinomial(),
  control = boost_control(mstop = 3)
)
stopifnot(
  length(negative_binomial$nuisance_vec_path) == 4L,
  identical(execution_report(negative_binomial)$calibration_count, 4L),
  all(lengths(path_frame(negative_binomial)$nuisance) == 1L),
  all(vapply(path_frame(negative_binomial)$nuisance, function(value) {
    is.finite(value[1])
  }, logical(1)))
)

lss <- glmboostLSS(
  y ~ x, dat, families = GaussianLSS(),
  control = boost_control(mstop = 4),
  mstop = c(mu = 4L, sigma = 2L)
)
lss_report <- execution_report(lss)
stopifnot(
  identical(lss_report$route, "lss_cyclic"),
  identical(lss_report$n_observations, n),
  identical(lss_report$n_learners, 4L),
  identical(lss_report$total_basis_columns, 4L),
  identical(lss_report$scalar_learner_count, 4L),
  identical(lss_report$block_learner_count, 0L),
  identical(lss_report$requested_mstop, 6L),
  identical(lss_report$completed_mstop, 6L)
)
lss_path <- path_frame(lss)
stopifnot(
  identical(nrow(lss_path), 6L),
  identical(lss_path$iteration, 1:6),
  identical(lss_path$parameter, c("mu", "sigma", "mu", "sigma", "mu", "mu")),
  identical(lss_path$cycle, c(1L, 1L, 2L, 2L, 3L, 4L)),
  is.na(lss_path$risk[1]),
  all(is.finite(lss_path$risk[c(2, 4, 5, 6)])),
  identical(lss_path$risk[c(2, 4, 5, 6)], lss$risk[-1])
)
lss_sliced <- lss[c(mu = 3L, sigma = 1L)]
stopifnot(
  identical(execution_report(lss_sliced)$requested_mstop, 6L),
  identical(execution_report(lss_sliced)$completed_mstop, 4L),
  identical(path_frame(lss_sliced)$parameter, c("mu", "sigma", "mu", "mu")),
  identical(path_frame(lss_sliced)$cycle, c(1L, 1L, 2L, 3L))
)
stopifnot(identical(
  execution_report(unserialize(serialize(lss, NULL))), lss_report
))

suppressMessages(capture.output(
  lss_noncyclic <- glmboostLSS(
    y ~ x, dat, families = GaussianLSS(), method = "noncyclic",
    control = boost_control(mstop = 5, trace = TRUE)
  )
))
lss_noncyclic_report <- execution_report(lss_noncyclic)
stopifnot(
  identical(lss_noncyclic_report$route, "lss_noncyclic"),
  identical(lss_noncyclic_report$requested_mstop, 5L),
  identical(lss_noncyclic_report$completed_mstop, 5L),
  identical(lss_noncyclic_report$instrumentation, "summary"),
  is.numeric(lss_noncyclic_report$total_ns)
)
lss_noncyclic_path <- path_frame(lss_noncyclic)
stopifnot(
  identical(nrow(lss_noncyclic_path), 5L),
  all(is.na(lss_noncyclic_path$cycle)),
  all(is.finite(lss_noncyclic_path$risk)),
  identical(lss_noncyclic_path$risk, lss_noncyclic$risk[-1])
)

lss_smooth <- gamboostLSS(
  y ~ bbs(x, df = 4), dat, families = GaussianLSS(),
  control = boost_control(mstop = 3)
)
lss_smooth_report <- execution_report(lss_smooth)
stopifnot(
  identical(lss_smooth_report$route, "lss_cyclic"),
  identical(lss_smooth_report$n_learners, 2L),
  identical(lss_smooth_report$block_learner_count, 2L),
  identical(lss_smooth_report$cholesky_solver_count, 2L),
  identical(lss_smooth_report$requested_mstop, 6L),
  identical(lss_smooth_report$completed_mstop, 6L)
)
stopifnot(
  identical(nrow(path_frame(lss_smooth)), 6L),
  all(path_frame(lss_smooth)$learner == "x")
)

# Regression: path_frame() must resolve learner names for every gamboostLSS
# learner type, not just bbs(). Before the fix, mixed formulas hit a legacy
# bbs-only parser and errored (or, for non-bbs learners whose label happens
# to match the bbs() regex, silently mislabeled the path).
mixed_dat <- data.frame(y = rnorm(n), x = rnorm(n), z = rnorm(n))
lss_mixed <- gamboostLSS(
  y ~ bols(z) + bbs(x, df = 4), mixed_dat, families = GaussianLSS(),
  control = boost_control(mstop = 6)
)
lss_mixed_path <- path_frame(lss_mixed)
mixed_selected <- selected(lss_mixed, merge = FALSE)
stopifnot(
  identical(nrow(lss_mixed_path), 12L),
  all(lss_mixed_path$learner %in% c("z", "x")),
  identical(lss_mixed_path$learner[lss_mixed_path$parameter == "mu"], mixed_selected$mu),
  identical(lss_mixed_path$learner[lss_mixed_path$parameter == "sigma"], mixed_selected$sigma)
)

cat("execution report and path frame tests passed\n")
