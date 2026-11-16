library(mboostrs)

# Smooth-LSS CV must accept the ordinary two-parameter GaussianLSS layout and
# preserve fold ordering across R's column-major FFI boundary.
set.seed(42)
n <- 36L
x <- seq(-1, 1, length.out = n)
y <- 0.6 * x + rnorm(n, sd = 0.2)
dat <- data.frame(y = y, x = x)
folds <- matrix(1, nrow = n, ncol = 3)
folds[1:12, 1] <- 0
folds[13:24, 2] <- 0
folds[25:36, 3] <- 0
# Exercise non-uniform in-bag weights, not only binary fold membership.
for (bb in seq_len(ncol(folds))) {
  inbag <- folds[, bb] > 0
  folds[inbag, bb] <- 1 + ((which(inbag) + bb) %% 3)
}
grid <- rbind(c(0L, 0L), c(1L, 1L), c(2L, 2L))

cv1 <- cvrisk_gamboostLSS(
  y ~ bbs(x, df = 4), dat, families = GaussianLSS(),
  control = boost_control(mstop = 2), folds = folds, grid = grid
)
perm <- c(3L, 1L, 2L)
cv2 <- cvrisk_gamboostLSS(
  y ~ bbs(x, df = 4), dat, families = GaussianLSS(),
  control = boost_control(mstop = 2), folds = folds[, perm], grid = grid
)
stopifnot(identical(dim(cv1), c(3L, 3L)))
stopifnot(all(is.finite(cv1)))
stopifnot(isTRUE(all.equal(unclass(cv2), unclass(cv1)[perm, ], tolerance = 1e-12,
                           check.attributes = FALSE)))

# PropOdds iteration k is stored at list index k+1, and newdata controls the
# number of output rows.
set.seed(7)
n <- 80L
x <- rnorm(n)
latent <- 0.8 * x + rlogis(n)
y <- as.integer(cut(latent, c(-Inf, -0.5, 0.5, Inf), labels = FALSE))
dat <- data.frame(y = y, x = x)
long <- glmboost(
  y ~ x, dat, family = PropOdds(n_levels = 3),
  control = boost_control(mstop = 8, nu = 0.1)
)
short <- glmboost(
  y ~ x, dat, family = PropOdds(n_levels = 3),
  control = boost_control(mstop = 3, nu = 0.1)
)
stopifnot(isTRUE(all.equal(
  predict(long, mstop = 3, type = "response"),
  predict(short, type = "response"),
  tolerance = 1e-12
)))
new_pred <- predict(long, newdata = data.frame(x = x[1:5]), type = "response")
stopifnot(identical(dim(new_pred), c(5L, 3L)))
stopifnot(max(abs(rowSums(new_pred) - 1)) < 1e-12)

# by= on a learner the R frontend does not build is refused rather than
# fitted without its modifier; spline, level, tree and bols() learners with a
# numeric by are supported.
by_error <- tryCatch(
  {
    gamboost(y ~ bkernel(x, by = x), dat, control = boost_control(mstop = 2))
    NULL
  },
  error = identity
)
stopifnot(inherits(by_error, "error"))
stopifnot(grepl("by=.*not yet supported", conditionMessage(by_error)))

# LSS residuals are location-only; Gaussian Pearson residuals use the fitted
# scale rather than silently returning raw residuals.
lss <- glmboostLSS(
  y ~ x, dat, families = GaussianLSS(),
  control = boost_control(mstop = 5)
)
raw <- residuals(lss, parameter = "mu", type = "response")
pearson <- residuals(lss, parameter = "mu", type = "pearson")
sigma <- predict(lss, parameter = "sigma", type = "response")
stopifnot(isTRUE(all.equal(pearson, raw / pmax(sigma, 1e-10), tolerance = 1e-12)))
sigma_error <- tryCatch(
  {
    residuals(lss, parameter = "sigma", type = "response")
    NULL
  },
  error = identity
)
stopifnot(inherits(sigma_error, "error"))
stopifnot(grepl("location parameter", conditionMessage(sigma_error)))

# Fallible fold/CV/LSS front doors must report R errors rather than unwind.
expect_error <- function(expr, pattern) {
  err <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(err, "error"))
  stopifnot(grepl(pattern, conditionMessage(err), ignore.case = TRUE))
}
expect_error(cv_weights(0), "positive|observation")
expect_error(cv_weights(5, b = 0), "positive|fold|replicate")
expect_error(cv_weights(5, fold_type = "kfold", b = 6), "k.*n")
expect_error(cvrisk(long, folds = matrix(1, nrow = n, ncol = 2)), "OOB")
expect_error(
  glmboostLSS(
    y ~ x, dat, families = GaussianLSS(), offset = c(NaN, 0),
    control = boost_control(mstop = 5)
  ),
  "offset.*finite|finite.*offset"
)

# Quantile()/Expectile() must reject an out-of-range tau at construction
# time instead of silently fitting a different quantile than requested.
expect_error(Quantile(tau = 1.5), "tau.*\\(0, 1\\)")
expect_error(Quantile(tau = 0), "tau.*\\(0, 1\\)")
expect_error(Quantile(tau = -0.2), "tau.*\\(0, 1\\)")
expect_error(Quantile(tau = NA_real_), "tau.*\\(0, 1\\)")
expect_error(Expectile(tau = 9), "tau.*\\(0, 1\\)")

# A non-finite family parameter that bypasses the R-level constructor (e.g.
# assembled by hand) must be rejected at the Rust FFI boundary rather than
# silently dropped in favour of the family's default value.
bad_huber <- structure(list(name = "huber", params = list(delta = NaN)),
                        class = "mboostrs_family")
expect_error(
  glmboost(y ~ x, dat, family = bad_huber, control = boost_control(mstop = 5)),
  "delta.*finite|finite.*delta"
)
# R's NA_real_ is a distinct sentinel from NaN at the extendr boundary
# (`Robj::as_real()` returns None for it), so it needs its own check or it
# is dropped from the parameter map and the family silently uses its default.
na_huber <- structure(list(name = "huber", params = list(delta = NA_real_)),
                       class = "mboostrs_family")
expect_error(
  glmboost(y ~ x, dat, family = na_huber, control = boost_control(mstop = 5)),
  "delta.*finite|finite.*delta"
)

# plot_varimp() is titled "(% reduction)" and labels its axis "Reduction
# (%)", but varimp(percent = TRUE) returns shares that sum to 1, not
# percentages that sum to 100 (same convention as Python's varimp()). The
# plot must scale for display so a 0.93 share is drawn as 93, not 0.93,
# under a "%" heading. Captured via trace() since barplot() draws rather
# than returning its input.
varimp_dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
varimp_fit <- glmboost(
  y ~ x1 + x2 + x3, varimp_dat, control = boost_control(mstop = 300)
)
captured_heights <- NULL
invisible(suppressMessages(trace(
  what = "barplot.default", where = asNamespace("graphics"),
  tracer = quote(captured_heights <<- height), print = FALSE
)))
on.exit(
  suppressMessages(try(untrace("barplot.default", where = asNamespace("graphics")),
                        silent = TRUE)),
  add = TRUE
)
pdf(NULL)
plot_varimp(varimp_fit)
invisible(dev.off())
stopifnot(!is.null(captured_heights))
stopifnot(isTRUE(all.equal(sum(abs(captured_heights)), 100, tolerance = 1e-8)))

# ── enhancement-hunt 14.1 / 14.2 / 14.3 / 14.4: the R AIC cluster ────────────
# `mboostrs_extra.R` used to be a third AIC implementation beside the Rust
# core's and Python's, and disagreed with both. All four defects are covered
# here; the R frontend now calls the core through `wrap__compute_aic_r`.
aic_dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
aic_dat$y <- 1 + 2 * aic_dat$x1 - aic_dat$x2 + rnorm(n, 0, 0.5)
aic_fit <- glmboost(y ~ x1 + x2 + x3, aic_dat,
                    control = boost_control(mstop = 120, nu = 0.1))

# 14.1: method = "gmdl" returned the corrected AIC, bit-identical.
aic_corrected <- AIC(aic_fit, method = "corrected")
aic_gmdl <- AIC(aic_fit, method = "gmdl")
classical_error <- tryCatch(AIC(aic_fit, method = "classical"), error = identity)
stopifnot(!identical(aic_path(aic_gmdl), aic_path(aic_corrected)))
stopifnot(inherits(classical_error, "error"))

# 14.2: "corrected" is mboost's log-scale Hurvich-Clifford criterion, which is
# on the order of 1, not the textbook AICc on the order of 500.
stopifnot(all(abs(aic_path(aic_corrected)) < 50))

# 14.3: `stats::AIC` returns one value per model; the path is an attribute.
stopifnot(length(aic_corrected) == 1L)
stopifnot(length(aic_path(aic_corrected)) == 120L)
stopifnot(mstop(aic_corrected) >= 1L, mstop(aic_corrected) <= 120L)
stopifnot(isTRUE(all.equal(as.numeric(aic_corrected),
                           aic_path(aic_corrected)[mstop(aic_corrected)])))
stopifnot(inherits(aic_corrected, "mboostrs_gbAIC"))

# 14.4: upstream warns for a family whose loss is not a likelihood, then errors
# on the corrected method. Both used to return a plausible path silently.
aic_quant <- glmboost(y ~ x1 + x2, aic_dat, family = Quantile(tau = 0.5),
                      control = boost_control(mstop = 30, nu = 0.1))
quant_warning <- FALSE
quant_error <- FALSE
withCallingHandlers(
  tryCatch(AIC(aic_quant, method = "corrected"),
           error = function(e) quant_error <<- TRUE),
  warning = function(w) {
    quant_warning <<- grepl("might not be reasonable", conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
stopifnot(quant_error)

# Robust losses have no information criterion under either spelling.
quant_classical <- tryCatch(AIC(aic_quant, method = "classical"), error = identity)
stopifnot(inherits(quant_classical, "error"))

# ── LSS families exported in R but unreachable from it ───────────────────────
# Nine families had an R constructor, a capability-manifest row saying
# `fit = yes`, and no match arm in the Rust R binding — so every one failed at
# fit time with "Unknown LSS family". The arms are now present at all eight
# match sites in `crates/mboostrs-r/src/lib.rs`.
lss_x <- rnorm(n)
lss_real <- data.frame(x = lss_x, y = 1 + 2 * lss_x + rnorm(n))
lss_pos  <- data.frame(x = lss_x, y = abs(rnorm(n, 3, 0.7)) + 0.2)
lss_cnt  <- data.frame(x = lss_x, y = as.numeric(rpois(n, exp(0.5 + 0.3 * lss_x))))
lss_bin  <- data.frame(x = lss_x, y = as.numeric(rbinom(n, 1, 0.4)))
lss_ctl  <- boost_control(mstop = 10, nu = 0.1)

lss_data_for <- function(f) {
  if (f %in% c("Pareto2LSS", "InverseGaussianLSS", "BCCGLSS")) return(lss_pos)
  if (f %in% c("GeometricLSS", "NegativeBinomialLSS", "ZeroAdjustedPoissonLSS")) return(lss_cnt)
  if (identical(f, "BinomialLSS")) return(lss_bin)
  lss_real
}

for (fam_name in c("GumbelLSS", "JSULSS", "BCCGLSS", "Pareto2LSS", "GeometricLSS",
                   "BinomialLSS", "NegativeBinomialLSS", "ZeroAdjustedPoissonLSS",
                   "InverseGaussianLSS", "LogisticLSS")) {
  fam <- get(fam_name)()
  fit <- glmboostLSS(y ~ x, data = lss_data_for(fam_name),
                     families = fam, control = lss_ctl)
  stopifnot(!is.null(fit))
  stopifnot(identical(names(selected(fit)), fam$param_names))
  stopifnot(all(is.finite(unlist(predict(fit, type = "link")))))
}

# ZeroAdjustedGammaLSS: semicontinuous response, nu = P(exact zero).
zaga_x <- rnorm(n)
zaga_p0 <- 1 / (1 + exp(-(-0.5 + 0.8 * zaga_x)))
zaga_y <- ifelse(runif(n) < zaga_p0, 0, rgamma(n, shape = 3, scale = exp(0.4) / 3))
zaga_fit <- glmboostLSS(y ~ x, data = data.frame(x = zaga_x, y = zaga_y),
                        families = ZeroAdjustedGammaLSS(),
                        control = boost_control(mstop = 20, nu = 0.1))
stopifnot(identical(names(selected(zaga_fit)), c("mu", "sigma", "nu")))
zaga_pred <- predict(zaga_fit, type = "response")
stopifnot(all(zaga_pred$mu > 0), all(zaga_pred$sigma > 0))
stopifnot(all(zaga_pred$nu > 0 & zaga_pred$nu < 1))
# An all-zero response leaves the Gamma part unidentified and must be refused.
stopifnot(inherits(try(glmboostLSS(y ~ x, data = data.frame(x = zaga_x, y = rep(0, n)),
                                   families = ZeroAdjustedGammaLSS(),
                                   control = boost_control(mstop = 5)),
                       silent = TRUE), "try-error"))

# ── brad: radial-basis learner, lowered to the generic block spec ────────────
# `brad` is a penalized linear smoother — a Matérn design against space-filling
# centres, whitened so a plain ridge applies — so it needs no engine support.
# R lowers it to `block` (the binding implements `block` but not `buser`).
set.seed(3)
brad_x <- runif(n); brad_z <- runif(n)
brad_truth <- sin(3 * pi * brad_x) * cos(3 * pi * brad_z)
brad_d <- data.frame(x = brad_x, z = brad_z, y = brad_truth + rnorm(n, 0, 0.2))
brad_fit <- gamboost(y ~ brad(x, z, knots = 30, df = 4), data = brad_d,
                     control = boost_control(mstop = 200, nu = 0.1))
stopifnot(tail(risk(brad_fit), 1) < 0.6 * risk(brad_fit)[1])
stopifnot(cor(fitted(brad_fit), brad_truth) > 0.8)
stopifnot(all(is.finite(predict(brad_fit, newdata = brad_d[1:5, ]))))
# one-dimensional smooth
brad_1d <- data.frame(x = brad_x, y = sin(4 * pi * brad_x) + rnorm(n, 0, 0.15))
brad_fit1 <- gamboost(y ~ brad(x, knots = 25), data = brad_1d,
                      control = boost_control(mstop = 150, nu = 0.1))
stopifnot(tail(risk(brad_fit1), 1) < 0.5 * risk(brad_fit1)[1])
