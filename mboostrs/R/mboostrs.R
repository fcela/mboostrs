#' Rust-Powered Component-Wise Gradient Boosting
#'
#' Component-wise gradient boosting with a Rust engine, matching R's mboost
#' semantics. Supports standard families and base-learners including
#' \code{bbs()} (smooth P-splines), \code{bols()} (linear), \code{btree()} (tree stumps),
#' \code{bmono()} (monotone P-splines), \code{bspatial()} (tensor-product 2-D surfaces),
#' \code{brandom()} (random intercepts), \code{bmrf()} (Markov random fields), and
#' \code{bns()} (ridge-penalized B-splines).
#'
#' @details
#' The mboostrs package provides component-wise gradient boosting through a
#' Rust engine. Two main entry points are available:
#' [glmboost()] for generalized linear models and [gamboost()] for
#' structured additive regression models. For distributional modeling,
#' [glmboostLSS()] and [gamboostLSS()] fit all parameters of a
#' distribution simultaneously. Model behavior is governed by
#' [boost_control()], which sets the number of iterations, learning rate,
#' and risk type. A wide range of [Gaussian()], [Binomial()],
#' [Poisson()], and other families are supported — see the family
#' constructor functions below.
#'
#' @seealso [glmboost()], [gamboost()], [glmboostLSS()], [gamboostLSS()],
#'   [boost_control()], [Gaussian()], [Binomial()], [Poisson()],
#'   [cvrisk()], [varimp()]
#'
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = Gaussian(),
#'                 control = boost_control(mstop = 50))
#' coef(fit)
#'
#' @useDynLib mboostrs, .registration = TRUE
"_PACKAGE"

#' Create boost control parameters
#'
#' @param mstop Number of boosting iterations (default: 100)
#' @param nu Learning rate / shrinkage factor (default: 0.1). For
#'   [glmboostLSS()] and [gamboostLSS()] it may be one value per distribution
#'   parameter: a named vector (`c(mu = 0.1, sigma = 0.05)`), reordered to the
#'   family's parameters, or an unnamed one in family order, as in upstream
#'   gamboostLSS. Standard fits take a single value.
#' @param risk Risk type: "inbag", "oobag", or "none" (default: "inbag")
#' @param trace If `TRUE`, print iteration progress and collect summary timing
#'   fields in the execution report.
#' @param subsample Stochastic component subsampling fraction. `NA_real_`
#'   (the default) disables subsampling: every candidate base-learner is
#'   evaluated at each iteration. A value in the open interval `(0, 1)`
#'   evaluates that random fraction of candidates per iteration instead,
#'   which reduces per-iteration cost on wide models and perturbs selection
#'   frequencies.
#' @param subsample_seed Integer seed for component subsampling. A fixed
#'   seed makes the sequence of subsampled candidates, and therefore the
#'   fitted path, reproducible. Ignored when `subsample` is `NA_real_`
#'   (default: `0L`).
#' @param step "fixed" (default) takes mboost's constant step. "line_search"
#'   minimizes the training loss along the selected learner's direction before
#'   multiplying it by `nu`. This also applies to cyclic distributional fits;
#'   non-cyclic distributional selection rejects this option.
#' @param step_iterations Positive number of golden-section refinements for
#'   line search (default: `4L`). The coarse grid first brackets the minimum.
#' @return A list with control parameters.
#' @details
#' The control parameters govern the boosting loop:
#' * `mstop` — total number of boosting iterations.
#' * `nu` — shrinkage parameter applied to each base-learner update.
#'   Smaller values slow learning and typically require more iterations.
#' * `risk` — how the empirical risk is computed. `"inbag"` uses the
#'   training data, `"oobag"` uses out-of-bag samples (requires weights),
#'   and `"none"` skips risk computation.
#' * `trace` — print progress and collect summary phase timings in the
#'   structured execution report.
#' * `subsample` — stochastic component subsampling. `NA` (the default)
#'   evaluates every candidate base-learner at each iteration. A value in
#'   `(0, 1)` evaluates that random fraction of candidates per iteration,
#'   reducing per-iteration cost on wide models and perturbing selection
#'   frequencies. The number of subsampled iterations is recorded in the
#'   execution report.
#' * `subsample_seed` — seed for component subsampling, so a fixed seed gives
#'   a reproducible path. Ignored when `subsample` is `NA`.
#' * `step` — fixed shrinkage or a fraction of the loss-optimal update. For
#'   Gamma regression, the gradient scales with the estimated shape, so a
#'   fixed `nu` that worked early can overshoot as the shape increases.
#' @seealso [glmboost()], [gamboost()]
#' @examples
#' ctrl <- boost_control(mstop = 200, nu = 0.05, risk = "inbag")
#' set.seed(1)
#' df <- data.frame(y = rnorm(50), x = rnorm(50))
#' fit <- glmboost(y ~ x, data = df, control = ctrl)
#' @export
boost_control <- function(mstop = 100L, nu = 0.1, risk = "inbag", trace = FALSE,
                          subsample = NA_real_, subsample_seed = 0L,
                          step = c("fixed", "line_search"), step_iterations = 4L) {
  step <- match.arg(step)
  step_iterations <- as.integer(.check_int_at_least("step_iterations", step_iterations, 1L))
  stopifnot(mstop > 0, length(nu) >= 1L, all(nu > 0 & nu <= 1),
            risk %in% c("inbag", "oobag", "none"))
  if (!is.na(subsample)) {
    stopifnot(subsample > 0 && subsample < 1)
  }
  list(mstop = as.integer(mstop), nu = nu, risk = risk, trace = isTRUE(trace),
       subsample = as.double(subsample), subsample_seed = as.double(subsample_seed),
       step = step, step_iterations = step_iterations)
}

# ─── weight contract ───────────────────────────────────────────────
#
# A boosting fit uses weights in three places at once: the base-learner normal
# equations (X'WX), the df -> lambda calibration that reads the hat matrix of
# those same equations, and the empirical risk. All three scale with the weight
# vector, so the *scale* of the weights, not only their ratios, changes the
# fitted path.
#
# `mboost_fit` settles that scale before anything else runs, with
#
#   rescale_weights <- function(w) {
#       if (max(abs(w - floor(w))) < sqrt(.Machine$double.eps)) return(w)
#       return(w / sum(w) * sum(w > 0))
#   }
#
# Integer weights are read as frequencies -- weight 3 stands for three
# identical rows -- and are left at the scale given. Anything fractional is
# read as an analytic weight, where only the ratios are meaningful, and is
# rescaled to sum to the number of weight-positive observations.

# TRUE when every weight is an integer to upstream's tolerance.
.weights_are_integral <- function(w) {
  if (length(w) == 0L) return(TRUE)
  max(abs(w - floor(w))) < sqrt(.Machine$double.eps)
}

# mboost:::rescale_weights.
.rescale_weights <- function(w) {
  if (.weights_are_integral(w)) return(w)
  total <- sum(w)
  # Unreachable for non-negative weights: a vector summing to zero has every
  # entry zero, which is integral and returned above.
  if (total <= 0) return(w)
  w / total * sum(w > 0)
}

# Validate the caller's weights and apply the named scale policy.
#
# Returns a list with `given` (the caller's vector, or NULL for unit weights),
# `fit` (the vector the engine sees), `policy`, and `rescaled`. `fit` is what
# the model stores as its weights, matching `model.weights()` in `mboost`,
# which returns the rescaled vector, so cross-validation and
# stability-selection folds are drawn on the same scale R draws them on.
.normalize_weights <- function(weights, n, policy = "upstream",
                               allow_all_zero = FALSE) {
  if (!policy %in% c("upstream", "raw")) {
    stop("weight_policy must be \"upstream\" or \"raw\", got \"", policy, "\"",
         call. = FALSE)
  }
  if (is.null(weights)) {
    return(list(given = NULL, fit = rep(1.0, n), policy = policy,
                rescaled = FALSE))
  }
  w <- as.double(weights)
  if (length(w) != n) {
    stop(sprintf("weights length %d != n=%d", length(w), n), call. = FALSE)
  }
  if (any(!is.finite(w))) {
    stop("weights must be finite", call. = FALSE)
  }
  if (any(w < 0)) stop("weights must be non-negative", call. = FALSE)
  if (sum(w > 0) == 0L && !allow_all_zero) {
    stop("weights are all zero, so the fit would have no observations: every ",
         "learner's normal equations are empty and the risk is zero at every ",
         "iteration. Pass positive weights on the rows to fit.", call. = FALSE)
  }
  if (identical(policy, "raw")) {
    return(list(given = w, fit = w, policy = policy, rescaled = FALSE))
  }
  rescaled <- !.weights_are_integral(w)
  list(given = w, fit = .rescale_weights(w), policy = policy,
       rescaled = rescaled)
}

# One line saying how the weight scale was settled, for print/summary methods.
.weights_note <- function(normalized) {
  if (is.null(normalized$given)) return("unit weights")
  if (identical(normalized$policy, "raw")) {
    return("user weights, policy=\"raw\" (no rescaling)")
  }
  if (isTRUE(normalized$rescaled)) {
    return(sprintf(
      "user weights rescaled to sum to %g (policy=\"upstream\"; fractional weights are analytic)",
      sum(normalized$fit)))
  }
  "user weights unchanged (policy=\"upstream\"; integer frequencies)"
}

#' Gaussian family (squared error loss)
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Squared error loss with identity link. Suitable for continuous
#'   unbounded responses.
#' @seealso [glmboost()], [gamboost()], [GaussReg()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = Gaussian(),
#'                 control = boost_control(mstop = 50))
#' coef(fit)
#' @export
Gaussian <- function() structure(list(name = "gaussian"), class = "mboostrs_family")

#' Gaussian family for regression (squared error loss)
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Squared error loss with identity link. Equivalent to Gaussian for
#'   the boosting framework.
#' @seealso [glmboost()], [gamboost()], [Gaussian()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = GaussReg(),
#'                 control = boost_control(mstop = 50))
#' coef(fit)
#' @export
GaussReg <- function() structure(list(name = "gaussreg"), class = "mboostrs_family")

#' Binomial family (logistic regression)
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Logistic loss (negative log-likelihood) with canonical logit link.
#'   The response must be coded as \{0, 1\}. Internally converted to \{-1, +1\}
#'   for the boosting algorithm.
#' @seealso [glmboost()], [gamboost()], [AdaExp()]
#' @examples
#' set.seed(42)
#' n <- 200; x <- rnorm(n)
#' y <- rbinom(n, 1, plogis(0.5 + 1.5 * x))
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = Binomial(),
#'                 control = boost_control(mstop = 100))
#' head(predict(fit, type = "response"))
#' @export
Binomial <- function() structure(list(name = "binomial"), class = "mboostrs_family")

#' Poisson family (log link)
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Poisson deviance with log link. For count data (non-negative
#'   integers).
#' @seealso [glmboost()], [gamboost()], [NBinomial()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- rpois(n, exp(0.5 + 0.8 * x))
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = Poisson(),
#'                 control = boost_control(mstop = 100))
#' head(predict(fit, type = "response"))
#' @export
Poisson <- function() structure(list(name = "poisson"), class = "mboostrs_family")

#' Laplace family (L1 absolute loss)
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details L1 absolute error loss with identity link. Large residuals have
#'   less influence than under squared-error loss.
#'   The negative gradient is sign(y - mu).
#' @seealso [glmboost()], [gamboost()], [Huber()], [Quantile()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- 2 * x + rnorm(n)
#' y[1:5] <- y[1:5] + 10  # add outliers
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = Laplace(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Laplace <- function() structure(list(name = "laplace"), class = "mboostrs_family")

#' Gamma regression family
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Gamma deviance with log link. Requires strictly positive responses
#'   (y > 0).
#' @seealso [glmboost()], [gamboost()], [Poisson()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- rgamma(n, shape = 2, rate = exp(0.5 + 0.8 * x))
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = GammaReg(),
#'                 control = boost_control(mstop = 100))
#' head(predict(fit, type = "response"))
#' @export
GammaReg <- function() structure(list(name = "gammasreg"), class = "mboostrs_family")

# ─── parameter validation ──────────────────────────────────────────
#
# A family constructor is the last place a bad value can be named against the
# argument the caller wrote. Past it the value reaches a numerical routine,
# where it becomes a wrong number rather than an error: a non-positive Huber
# `delta` makes the loss decrease with the residual and reports a negative
# value as risk; a non-positive C-index `sigma` inverts the smoothing sigmoid;
# `n_levels = 1` describes a categorical response with one category, which
# binding code used to repair silently with `.max(2)`.

.check_unit_interval <- function(name, value) {
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) ||
      value <= 0 || value >= 1) {
    stop(sprintf("%s must be in (0, 1), got %s", name,
                 paste(format(value), collapse = ", ")), call. = FALSE)
  }
  as.numeric(value)
}

.check_positive <- function(name, value) {
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) || value <= 0) {
    stop(sprintf("%s must be positive, got %s", name,
                 paste(format(value), collapse = ", ")), call. = FALSE)
  }
  as.numeric(value)
}

.check_int_at_least <- function(name, value, minimum) {
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) ||
      value != trunc(value)) {
    stop(sprintf("%s must be a whole number >= %d, got %s", name, minimum,
                 paste(format(value), collapse = ", ")), call. = FALSE)
  }
  if (value < minimum) {
    stop(sprintf("%s must be at least %d, got %d", name, minimum,
                 as.integer(value)), call. = FALSE)
  }
  as.numeric(value)
}

#' Quantile regression family
#'
#' The check (pinball) loss, which weights positive and negative residuals by
#' `tau` and `1 - tau`. The fit estimates the conditional `tau`-quantile of the
#' response rather than its mean, so a `tau = 0.5` fit is a conditional median
#' and is unaffected by how far the extreme observations lie, only by how many
#' fall on each side.
#'
#' Upstream `mboost` spells this `QuantReg()`; the name here follows the
#' quantity rather than the upstream function.
#'
#' @param tau The quantile to estimate, in (0, 1).
#' @param qoffset The quantile of the response used as the boosting starting
#'   value, in (0, 1). `mboost` uses the same convention.
#' @return A family object for [glmboost()] / [gamboost()].
#' @details The negative gradient is `tau * I(y > mu) + (tau - 1) * I(y < mu)`.
#' @seealso [Expectile()], which minimises an asymmetric squared loss instead
#'   and so estimates an expectile rather than a quantile; [Laplace()],
#'   which is the `tau = 0.5` case written as an absolute-error loss.
#' @examples
#' set.seed(1)
#' n <- 200
#' x <- rnorm(n)
#' y <- 1 + x + rt(n, df = 3)
#' d <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = d, family = Quantile(tau = 0.9),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Quantile <- function(tau = 0.5, qoffset = 0.5) {
  structure(list(name = "quantreg",
                 params = list(tau = .check_unit_interval("tau", tau),
                               qoffset = .check_unit_interval("qoffset", qoffset))),
            class = "mboostrs_family")
}

#' Expectile regression family
#'
#' @param tau Expectile level (default: 0.5), in (0, 1).
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Asymmetric least squares for estimating expectiles. The negative
#'   gradient is 2 * (tau * I(y > mu) + (tau - 1) * I(y < mu)) * (y - mu).
#'   At tau = 0.5 this reduces to the Gaussian family.
#' @seealso [glmboost()], [gamboost()], [Quantile()], [Gaussian()]
#' @examples
#' set.seed(42)
#' n <- 200; x <- rnorm(n)
#' y <- 1 + 2 * x + (1 + 0.5 * x) * rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = Expectile(tau = 0.9),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Expectile <- function(tau = 0.5) {
  structure(list(name = "expectreg",
                 params = list(tau = .check_unit_interval("tau", tau))),
            class = "mboostrs_family")
}

#' Huber loss family
#'
#' @param delta Tuning constant (default: NULL, estimated from data)
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Huber loss combines squared error (small residuals) and absolute
#'   error (large residuals), providing robustness to outliers while
#'   maintaining differentiability near zero. When `delta = NULL` (the
#'   default), delta is estimated from the data as 1.345 * MAD (median
#'   absolute deviation), following the standard robust statistics
#'   convention.
#' @seealso [glmboost()], [gamboost()], [Laplace()], [Gaussian()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- 2 * x + rnorm(n)
#' y[1:5] <- y[1:5] + 10  # add outliers
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = Huber(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Huber <- function(delta = NULL) {
  params <- if (is.null(delta)) list() else list(delta = .check_positive("delta", delta))
  structure(list(name = "huber", params = params), class = "mboostrs_family")
}

#' Negative binomial family
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Negative binomial with log link for overdispersed count data.
#'   Provides a more flexible alternative to the Poisson family when the
#'   variance exceeds the mean.
#'   The fitted nuisance is the size in `Var(Y) = mu + mu^2 / size`,
#'   estimated at each step. `size(fit)` names this value; `nuisance(fit)`
#'   returns the same size. statsmodels NB2 uses `alpha = 1 / size`.
#' @param ... Unsupported constructor arguments. The size is estimated during
#'   fitting; fixed `size`, `alpha` and `dispersion` arguments are rejected.
#' @seealso [glmboost()], [gamboost()], [Poisson()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' mu <- exp(0.5 + 0.8 * x)
#' y <- rnbinom(n, mu = mu, size = 2)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = NBinomial(),
#'                 control = boost_control(mstop = 100))
#' head(predict(fit, type = "response"))
#' @export
NBinomial <- function(...) {
  if (length(list(...))) {
    stop("NBinomial estimates the negative-binomial size during fitting; ",
         "read it with size(fit). statsmodels NB2 alpha is its reciprocal ",
         "(alpha = 1/size); alpha, dispersion and fixed size arguments are ",
         "unsupported.", call. = FALSE)
  }
  structure(list(name = "nbinomial"), class = "mboostrs_family")
}

#' Size of a fitted negative-binomial model
#'
#' The calibrated size at the model's step count, with
#' `Var(Y) = mu + mu^2 / size`. statsmodels NB2 uses `alpha = 1 / size`.
#' `nuisance(object)` returns the same value using mboost's generic accessor.
#'
#' @param object A fitted `mboostrs_glmboost` or `mboostrs_gamboost` model
#'   with `NBinomial()` family. Other families are rejected, including Hurdle,
#'   whose nuisance uses the reciprocal parameterization.
#' @return A numeric scalar.
#' @seealso [NBinomial()], [nuisance()]
#' @export
size <- function(object) UseMethod("size")

#' @rdname size
#' @export
size.mboostrs_glmboost <- function(object) {
  if (!identical(object$family$name, "nbinomial")) {
    stop("size is available for fitted NBinomial models", call. = FALSE)
  }
  nuisance(object)[1L]
}

#' @rdname size
#' @export
size.mboostrs_gamboost <- size.mboostrs_glmboost

#' Nuisance parameter of a fitted model
#'
#' The family parameter estimated alongside the boosting steps, as
#' `mboost::nuisance()` reports it: the size of `NBinomial()`
#' (`Var(Y) = mu + mu^2 / size`), the dispersion of `Hurdle()`,
#' the shape of `GammaReg()`, the scale of the survival families,
#' and the thresholds of `PropOdds()`. It is the value after the model's last
#' step.
#'
#' @param object A fitted `mboostrs_glmboost` or `mboostrs_gamboost` model.
#' @return A numeric vector, empty for a family without a nuisance parameter.
#' @export
nuisance <- function(object) UseMethod("nuisance")

#' @export
nuisance.mboostrs_glmboost <- function(object) {
  path <- object$nuisance_vec_path
  if (!is.null(path) && length(path) >= object$mstop + 1L) {
    return(as.numeric(path[[object$mstop + 1L]]))
  }
  as.numeric(object$nuisance)
}

#' @export
nuisance.mboostrs_gamboost <- nuisance.mboostrs_glmboost

#' Hurdle family (zero-truncated negative binomial)
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details The count part of a hurdle model, as `mboost::Hurdle()`: a
#'   negative binomial with log link truncated at zero, fitted to the
#'   positive counts. The dispersion is re-estimated at every boosting step
#'   on the interval (0, 100). The zero part (whether a count is positive) is
#'   a separate binary model, for example `Binomial()` on `y > 0`.
#' @seealso [NBinomial()], [Binomial()]
#' @examples
#' set.seed(42)
#' n <- 300; x <- rnorm(n)
#' y <- rnbinom(n, mu = exp(1 + 0.5 * x), size = 2)
#' df <- data.frame(y = y, x = x)[y > 0, ]
#' fit <- glmboost(y ~ x, data = df, family = Hurdle(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Hurdle <- function() structure(list(name = "hurdle"), class = "mboostrs_family")

#' Proportional odds family
#'
#' @param n_levels Number of ordered response levels (default: NULL, estimated)
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Proportional odds (cumulative logit) model for ordered categorical
#'   responses. When `n_levels = NULL`, the number of ordered levels is
#'   inferred from the response factor at fit time.
#' @seealso [glmboost()], [gamboost()], [Binomial()], [Multinomial()]
#' @examples
#' set.seed(42)
#' n <- 200; x <- rnorm(n)
#' eta <- 1 + 2 * x
#' y <- cut(eta + rnorm(n), breaks = c(-Inf, -1, 1, Inf),
#'          labels = c("low", "med", "high"))
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = PropOdds(n_levels = 3),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
PropOdds <- function(n_levels = NULL) {
  # Two ordered categories is a binary response, which Binomial fits; one is
  # not a categorical response at all.
  params <- if (is.null(n_levels)) list()
            else list(n_levels = .check_int_at_least("n_levels", n_levels, 3L))
  structure(list(name = "propodds", params = params), class = "mboostrs_family")
}

#' AdaBoost exponential loss family
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details AdaBoost exponential loss. The response must be \{0, 1\} (coded to
#'   \{-1, +1\} internally). This family uses the exponential loss
#'   exp(-2 * y * eta), which is the canonical loss for AdaBoost-style
#'   boosting.
#' @seealso [glmboost()], [gamboost()], [Binomial()], [HingeLoss()]
#' @examples
#' set.seed(42)
#' n <- 200; x <- rnorm(n)
#' y <- rbinom(n, 1, plogis(0.5 + 1.5 * x))
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = AdaExp(),
#'                 control = boost_control(mstop = 100))
#' head(predict(fit, type = "response"))
#' @export
AdaExp <- function() structure(list(name = "adaexp"), class = "mboostrs_family")

#' Cox proportional hazards family
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Cox proportional hazards model for right-censored survival data.
#'   Requires a `Surv(time, event)` response.
#' @seealso [glmboost()], [gamboost()], [Weibull()], [Loglog()], [Lognormal()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' t <- rexp(n, rate = exp(0.5 + 0.8 * x))
#' c <- rexp(n, rate = 0.1)
#' time <- pmin(t, c)
#' event <- as.integer(t <= c)
#' df <- data.frame(time = time, event = event, x = x)
#' library(survival)
#' fit <- glmboost(Surv(time, event) ~ x, data = df, family = CoxPH(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
CoxPH <- function() structure(list(name = "coxph"), class = "mboostrs_family")

#' Weibull AFT family
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Weibull accelerated failure time model for right-censored survival
#'   data. Requires a `Surv(time, event)` response.
#' @seealso [glmboost()], [gamboost()], [CoxPH()], [Loglog()], [Lognormal()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' t <- rweibull(n, shape = 1.5, scale = exp(0.5 + 0.8 * x))
#' c <- rexp(n, rate = 0.1)
#' time <- pmin(t, c)
#' event <- as.integer(t <= c)
#' df <- data.frame(time = time, event = event, x = x)
#' library(survival)
#' fit <- glmboost(Surv(time, event) ~ x, data = df, family = Weibull(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Weibull <- function() structure(list(name = "weibull"), class = "mboostrs_family")

#' Log-logistic AFT family
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Log-logistic accelerated failure time model for right-censored
#'   survival data. Requires a `Surv(time, event)` response.
#' @seealso [glmboost()], [gamboost()], [CoxPH()], [Weibull()], [Lognormal()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' t <- rlnorm(n, meanlog = 0.5 + 0.8 * x, sdlog = 0.5)
#' c <- rexp(n, rate = 0.1)
#' time <- pmin(t, c)
#' event <- as.integer(t <= c)
#' df <- data.frame(time = time, event = event, x = x)
#' library(survival)
#' fit <- glmboost(Surv(time, event) ~ x, data = df, family = Loglog(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Loglog <- function() structure(list(name = "loglog"), class = "mboostrs_family")

#' Log-normal AFT survival family.
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Log-normal accelerated failure time model for right-censored
#'   survival data. Requires a `Surv(time, event)` response.
#' @seealso [glmboost()], [gamboost()], [CoxPH()], [Weibull()], [Loglog()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' t <- rlnorm(n, meanlog = 0.5 + 0.8 * x, sdlog = 0.5)
#' c <- rexp(n, rate = 0.1)
#' time <- pmin(t, c)
#' event <- as.integer(t <= c)
#' df <- data.frame(time = time, event = event, x = x)
#' library(survival)
#' fit <- glmboost(Surv(time, event) ~ x, data = df, family = Lognormal(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Lognormal <- function() structure(list(name = "lognormal"), class = "mboostrs_family")

#' Hinge loss family (SVM-like) for binary classification.
#'
#' Response must be -1 or +1.
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Hinge loss for binary classification, as used in support vector
#'   machines. The loss is max(0, 1 - y * eta) with y in \{-1, +1\}.
#'   The response must be coded as \{-1, +1\}.
#' @seealso [glmboost()], [gamboost()], [Binomial()], [AdaExp()]
#' @examples
#' set.seed(42)
#' n <- 200; x <- rnorm(n)
#' y <- sign(1 + 2 * x + rnorm(n))
#' y[y == 0] <- 1
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, family = HingeLoss(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
HingeLoss <- function() structure(list(name = "hingeloss"), class = "mboostrs_family")

#' Smoothed 1 - AUC ranking loss.
#'
#' Response must be a factor with exactly two levels.
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()] or [gamboost()].
#' @details Smoothed surrogate for \code{1 - AUC} (area under the ROC curve),
#'   matching \code{mboost::AUC()}. The response must be a two-level factor;
#'   the second factor level is the positive class. At every boosting
#'   iteration the current fit is rescaled (\code{f / sd(f)}, using R's n-1
#'   sample standard deviation) before all positive/negative score pairs are
#'   compared, matching R's implementation. Only unit case weights
#'   (\code{weights = NULL}, or all-ones) are currently supported;
#'   \code{cvrisk()} is not yet supported for this family.
#' @seealso [glmboost()], [gamboost()], [Binomial()], [HingeLoss()]
#' @examples
#' set.seed(42)
#' n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
#' eta <- 0.5 * x1 - 0.3 * x2
#' y <- factor(ifelse(runif(n) < 1 / (1 + exp(-eta)), "yes", "no"))
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, family = AUC(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
AUC <- function() structure(list(name = "auc"), class = "mboostrs_family")

#' Gehan rank loss for AFT survival models.
#'
#' Requires a `Surv(time, event)` response, matching \code{mboost::Gehan()}.
#'
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()].
#' @details Gehan generalized-Wilcoxon rank loss for accelerated failure time
#'   (AFT) models. The loss depends only on pairwise residual differences
#'   between observations, so it is invariant to rank-preserving location
#'   shifts of the fitted values -- in particular, the offset is
#'   unidentifiable (the risk is flat in a constant shift). mboostrs defines
#'   \code{offset() == 0} rather than reproducing R's arbitrary
#'   \code{optimize()}-on-a-flat-function value; the gradient, coefficient
#'   increments, `xselect()`, and risk all match R exactly, but fitted values
#'   differ from R's by an additive constant. Only unit case weights
#'   (\code{weights = NULL}, or all-ones) are currently supported;
#'   \code{cvrisk()} is not yet supported for this family.
#' @seealso [glmboost()], [CoxPH()], [AUC()]
#' @examples
#' set.seed(42)
#' n <- 150; x1 <- rnorm(n); x2 <- rnorm(n)
#' eta <- 0.4 * x1 - 0.25 * x2
#' time <- rexp(n, rate = exp(eta))
#' event <- rbinom(n, 1, 0.7)
#' df <- data.frame(time = time, event = event, x1 = x1, x2 = x2)
#' library(survival)
#' fit <- glmboost(Surv(time, event) ~ x1 + x2, data = df, family = Gehan(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Gehan <- function() structure(list(name = "gehan"), class = "mboostrs_family")

#' Uno's IPCW-weighted concordance ("C-index") loss for survival models.
#'
#' Requires a `Surv(time, event)` response, matching \code{mboost::Cindex()}.
#'
#' @param sigma Width of the sigmoid smoothing of the (otherwise
#'   non-differentiable) pairwise concordance indicator (default: 0.1).
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [glmboost()].
#' @details A smoothed, inverse-probability-of-censoring-weighted (IPCW)
#'   concordance loss (Uno's C-index / "UnoC"). Unlike [Gehan()], the offset
#'   is identically 0 in R (verified: \code{Cindex()@@offset} is the constant
#'   function 0, and the risk only ever depends on pairwise differences
#'   `f[i] - f[j]`), so mboostrs' fitted values match R directly -- no
#'   centering deviation. Case weights must be either 0 or 1
#'   (\code{weights = "zeroone"} in R); unlike [Gehan()]/[AUC()]'s
#'   unit-weight-only restriction, `0` is a valid, meaningfully-used value
#'   here (the IPCW fit runs on the weight-positive subset), so
#'   \code{cvrisk()}/\code{stabsel()} with subsampling-style folds ARE
#'   supported; only bootstrap-style folds (integer resample counts) are
#'   rejected.
#' @seealso [glmboost()], [CoxPH()], [Gehan()]
#' @examples
#' set.seed(42)
#' n <- 150; x1 <- rnorm(n); x2 <- rnorm(n)
#' eta <- 0.4 * x1 - 0.25 * x2
#' time <- rexp(n, rate = exp(eta))
#' event <- rbinom(n, 1, 0.7)
#' df <- data.frame(time = time, event = event, x1 = x1, x2 = x2)
#' library(survival)
#' fit <- glmboost(Surv(time, event) ~ x1 + x2, data = df, family = Cindex(),
#'                 control = boost_control(mstop = 100))
#' coef(fit)
#' @export
Cindex <- function(sigma = 0.1) {
  structure(list(name = "cindex",
                 params = list(sigma = .check_positive("sigma", sigma))),
            class = "mboostrs_family")
}

#' Multinomial family (multi-class classification).
#'
#' Fits K-1 class-specific predictors with a Kronecker-product design. Supported
#' base learners: \code{bols} (linear) and \code{bbs} (smooth P-splines). The
#' response must be a factor.
#'
#' @param classes Number of classes; usually left NULL and inferred from the data.
#' @return A family object (list with class `"mboostrs_family"`) for use with
#'   [gamboost()].
#' @details Multi-class classification with a Kronecker-product design matrix.
#'   The response must be a factor with K >= 3 levels. K-1 class-specific
#'   predictors are fitted simultaneously. Only supported by [gamboost()]
#'   with \code{bols} or \code{bbs} base-learners.
#' @seealso [gamboost()], [Binomial()], [PropOdds()]
#' @examples
#' set.seed(42)
#' n <- 150; x <- rnorm(n)
#' eta <- 2 * x
#' y <- factor(cut(eta + rnorm(n), breaks = c(-Inf, -0.5, 0.5, Inf),
#'                 labels = c("A", "B", "C")))
#' df <- data.frame(y = y, x = x)
#' fit <- gamboost(y ~ bols(x), data = df, family = Multinomial(),
#'                 control = boost_control(mstop = 100))
#' @export
Multinomial <- function(classes = NULL) {
  # Two classes is a binary response, which Binomial fits; one is not a
  # classification problem.
  params <- if (is.null(classes)) list()
            else list(n_classes = .check_int_at_least("classes", classes, 3L))
  structure(list(name = "multinomial", params = params), class = "mboostrs_family")
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' Fit a GLMBoost model
#'
#' @param formula A formula object (e.g., \code{y ~ x1 + x2}).
#'   For survival families, use \code{Surv(time, event) ~ x1 + x2}.
#' @param data A data frame containing the variables.
#' @param family A family object created by one of the family functions.
#' @param control A list created by \code{boost_control()}.
#' @param weights Optional observation weights, one per row, non-negative.
#'   Integer weights are frequencies -- weight 3 stands for three identical
#'   rows -- and are used at the scale given; fractional weights are rescaled
#'   (see `weight_policy`). A zero weight excludes a row while keeping its
#'   position. An all-zero vector is rejected: the fit it produces reports a
#'   completed path over an empty sample.
#' @param center Logical; center the covariates before fitting (default
#'   \code{TRUE}), as mboost's \code{glmboost(center = )}. Centering changes
#'   how fast the path converges, not the model it converges to.
#' @param weight_policy How the scale of `weights` is settled before fitting.
#'   `"upstream"` (the default) reproduces `mboost`'s `rescale_weights`:
#'   integer weights are frequencies and pass through, fractional weights are
#'   analytic and are rescaled to sum to the number of weight-positive rows.
#'   `"raw"` passes the vector through unchanged, for a caller whose weight
#'   scale is itself meaningful (survey weights summing to a population total,
#'   for example). The scale is not cosmetic -- it enters `X'WX`, the
#'   `df -> lambda` calibration, and the risk -- so `"raw"` gives a different
#'   path from R's on the same fractional weights.
#' @param offset Optional offset: one number (the starting value of the
#'   linear predictor), or one known contribution per row, such as
#'   `log(exposure)` for a count per unit of exposure. A per-row offset is
#'   never re-estimated, enters every cross-validation fold, and must be given
#'   again as `offset =` to `predict()` for new rows.
#' @return An object of class \code{"mboostrs_glmboost"}.
#' @details
#' Fits a generalized linear model via component-wise gradient boosting.
#' Each boosting iteration selects the base-learner that most reduces the
#' empirical risk and applies a shrinkage factor `nu`. The process continues
#' for `mstop` iterations, producing a coefficient path that can be sliced
#' with `model[k]`. Use [cvrisk()] to determine the optimal stopping
#' iteration.
#' @seealso [gamboost()], [boost_control()], [Gaussian()], [Binomial()],
#'   [predict.mboostrs_glmboost()], [coef.mboostrs_glmboost()],
#'   [cvrisk()], [summary.mboostrs_glmboost()]
#' @examples
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n)
#' y <- 1 + 2 * x1 - 1 * x2 + rnorm(n)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, family = Gaussian(),
#'                 control = boost_control(mstop = 100))
#' print(fit)
#' head(predict(fit))
#' head(predict(fit, type = "response"))
#' coef(fit)
#' @export
glmboost <- function(formula, data, family = Gaussian(), control = boost_control(),
                     weights = NULL, offset = NULL, weight_policy = "upstream",
                     center = TRUE) {
  cl <- match.call()
  .require_scalar_nu(control)

  if (identical(family$name, "multinomial")) {
    # One coefficient vector per non-reference class, which the per-column
    # learners here do not carry; mboost's glmboost refuses it as well.
    stop("glmboost does not fit family = Multinomial(): a multinomial model ",
         "needs one coefficient per column for each non-reference class. Fit ",
         "gamboost with linear terms instead, e.g. ",
         "gamboost(y ~ bols(x1) + bols(x2), data, family = Multinomial()).",
         call. = FALSE)
  }
  has_survival <- family$name %in% c("coxph", "weibull", "loglog", "lognormal", "gehan", "cindex")
  parsed <- .parse_formula(formula, data, has_survival, allow_no_intercept = TRUE)

  y <- parsed$y
  labelled <- .binary_label_response(.raw_response(formula, data), family$name)
  if (!is.null(labelled)) y <- labelled
  n <- parsed$n
  x_matrix <- parsed$x_matrix
  learner_names <- parsed$learner_names
  intercept <- isTRUE(parsed$intercept)
  p <- ncol(x_matrix)
  if (!intercept && isTRUE(center)) {
    # mboost's glmboost warns the same way and centers anyway.
    warning("model with centered covariates does not contain intercept", call. = FALSE)
  }

  normalized <- .normalize_weights(weights, n, weight_policy)
  w <- normalized$fit

  # center = FALSE fits the raw columns, as mboost's glmboost(center = FALSE):
  # the same limit, reached more slowly, because each column stays correlated
  # with the intercept.
  col_means <- if (isTRUE(center)) colMeans(x_matrix) else rep(0, ncol(x_matrix))

  family_name <- family$name
  family_params <- family$params %||% list()
  offset_val <- .offset_arg(offset, n)

  # Validate inputs before passing to Rust
  if (any(!is.finite(y))) stop("y contains NA, NaN, or Inf values")
  if (any(!is.finite(w))) stop("weights contains NA, NaN, or Inf values")
  if (any(!is.finite(x_matrix))) stop("x_matrix contains NA, NaN, or Inf values")

  result <- .Call("wrap__fit_glmboost",
    as.double(t(x_matrix)),  # transpose: R is col-major, Rust reads row-major
    as.integer(n),
    as.integer(p),
    as.double(y),
    as.double(w),
    as.integer(control$mstop),
    as.double(control$nu),
    family_name,
    family_params,
    as.double(col_means),
    offset_val,
    control$risk,
    control$trace %||% FALSE,
    # extendr rejects NA for a scalar double; 0 is the "off" sentinel
    { s <- control$subsample %||% NA_real_; if (is.na(s)) 0 else as.double(s) },
    as.double(control$subsample_seed %||% 0),
    intercept,
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
    PACKAGE = "mboostrs"
  )
  if (isTRUE(control$trace)) .render_execution_report(result$execution_report)

  .warn_if_risk_rose(result$risk, control, family_name, family_params)

  structure(
    list(
      # A per-row offset is kept as given; otherwise the fitted start.
      offset = if (length(offset_val) > 1L) offset_val else result$offset,
      user_offset = offset,
      selected = result$selected,
      coefs = result$coefs,
      risk = result$risk,
      risk_kind = control$risk,
      fitted = result$fitted,
      nuisance = result$nuisance,
      nuisance_vec_path = result$nuisance_vec_path,
      execution_report = result$execution_report,
      nu = control$nu,
      mstop = control$mstop,
      family = family,
      control = control,
      formula = formula,
      data = data,
      learner_names = c(if (intercept) "(Intercept)", learner_names),
      intercept = intercept,
      center = isTRUE(center),
      x_matrix = x_matrix,
      centering_means = col_means,
      prediction_terms = parsed$prediction_terms,
      xlevels = parsed$xlevels,
      contrasts = parsed$contrasts,
      n = n,
      weights = w,
      weight_policy = normalized$policy,
      call = cl
    ),
    class = "mboostrs_glmboost"
  )
}

# A two-class response given as labels, as mboost takes it: a factor, a
# logical or a character vector. The binary losses (Binomial, AdaExp,
# HingeLoss, AUC) read it as 0 for the first level and 1 for the second, the
# order mboost's `c(-1, 1)[as.integer(y)]` uses. NULL when the response is
# numeric or the family is not a binary loss; three or more labels refuse.
.binary_label_response <- function(raw, family_name) {
  if (!(family_name %in% c("binomial", "adaexp", "hingeloss", "auc"))) return(NULL)
  if (!(is.factor(raw) || is.logical(raw) || is.character(raw))) return(NULL)
  f <- if (is.factor(raw)) droplevels(raw) else factor(raw)
  if (nlevels(f) != 2L) {
    stop("a ", family_name, " response needs two classes; it has ", nlevels(f),
         " (", paste(levels(f), collapse = ", "), ")", call. = FALSE)
  }
  as.double(as.integer(f) - 1L)
}

# The raw left-hand side of a formula, before model.response codes it.
.raw_response <- function(formula, data) {
  tryCatch(eval(formula[[2L]], envir = as.list(data), enclos = environment(formula)),
           error = function(e) NULL)
}

# The core decides the class; R restores the response's factor labels.
.class_prediction <- function(object, eta) {
  indices <- .Call("wrap__family_class_r", object$family$name,
                   object$family$params %||% list(), as.double(eta),
                   PACKAGE = "mboostrs")
  raw <- .raw_response(object$formula, object$data)
  if (is.null(raw) || is.matrix(raw)) {
    stop("type = 'class' requires a stored two-class response", call. = FALSE)
  }
  labels <- if (is.factor(raw)) levels(droplevels(raw)) else levels(factor(raw))
  if (length(labels) != 2L) {
    stop("type = 'class' requires two response levels", call. = FALSE)
  }
  factor(labels[indices], levels = labels)
}

# `selected` in the core's numbering, where the first learner is the
# intercept: shifted by one for a fit without one (`y ~ x - 1`).
# Refuse arguments a method does not take. S3 methods carry `...` for the
# generic's signature, and an argument mboost accepts but this method does
# not (`aggregate`, a misspelled `which`) was silently dropped, so the call
# returned something other than what was asked for.
.refuse_unused <- function(fn, ...) {
  dots <- list(...)
  if (!length(dots)) return(invisible(NULL))
  labels <- names(dots)
  if (is.null(labels)) labels <- rep("", length(dots))
  labels[!nzchar(labels)] <- "<unnamed>"
  hint <- if ("aggregate" %in% labels) {
    paste0(" For the prediction after k steps use mstop = k; for each ",
           "learner's contribution use which =.")
  } else ""
  stop(fn, "() does not take ", paste(labels, collapse = ", "), ".", hint,
       call. = FALSE)
}

.glm_core_selected <- function(object) {
  if (isFALSE(object$intercept)) object$selected + 1L else object$selected
}

#' Predict from a fitted GLMBoost model
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param newdata Optional new data frame. If NULL, uses training data.
#' @param mstop Number of iterations to predict at. NULL uses the full model.
#' @param type "link" (linear predictor), "response" (inverse-link transformed),
#'   or "class" (a factor with the training response's labels for Binomial,
#'   AdaExp, HingeLoss and AUC). A tied score belongs to the first level.
#' @param which Learners (names or positions, the intercept first) whose
#'   contributions to return: a matrix with one column per learner holding
#'   its link-scale contribution without the offset, as \code{mboost}'s
#'   \code{predict(which = )} returns.
#' @param offset The new rows' offset, one value per row, required with
#'   `newdata` when the model was fitted with a per-row offset
#'   (`log(exposure)`); `NULL` reuses the fitted offset.
#' @param ... Not used.
#' @return A numeric vector of predicted values, or a factor for `type = "class"`.
#' @details
#' If `newdata` is NULL, predictions are computed on the training data.
#' The `type` argument controls the output scale: `"link"` returns the raw
#' linear predictor, while `"response"` applies the family's inverse link
#' function (e.g. `exp` for Poisson, `plogis` for Binomial).
#' @seealso [glmboost()], [fitted.mboostrs_glmboost()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' head(predict(fit))
#' newdf <- data.frame(x = c(0, 1, 2))
#' predict(fit, newdata = newdf)
#' @export
predict.mboostrs_glmboost <- function(object, newdata = NULL, mstop = NULL,
                                        type = c("link", "response", "class"),
                                        offset = NULL, which = NULL, ...) {
  .refuse_unused("predict", ...)
  type <- match.arg(type)
  if (type == "class") {
    if (!is.null(which)) stop("type = 'class' requires the whole model", call. = FALSE)
    return(.class_prediction(object, predict(object, newdata = newdata,
                                             mstop = mstop, type = "link", offset = offset)))
  }
  mstop_val <- if (is.null(mstop)) 0L else as.integer(mstop)
  start <- .prediction_offset(object, newdata, offset)
  if (!is.null(which) && type != "link") {
    stop("predict(which = ) returns each learner's contribution on the link ",
         "scale; a response-scale value exists only for the whole model",
         call. = FALSE)
  }

  if (is.null(newdata)) {
    x_matrix <- object$x_matrix
  } else if (!is.null(object$prediction_terms)) {
    design <- stats::model.matrix(stats::delete.response(object$prediction_terms),
                                  data = newdata, xlev = object$xlevels,
                                  contrasts.arg = object$contrasts)
    colnames(design) <- .unquote_matrix_label(colnames(design))
    names <- setdiff(object$learner_names, "(Intercept)")
    x_matrix <- design[, names, drop = FALSE]
  } else {
    has_survival <- object$family$name %in% c("coxph", "weibull", "loglog", "lognormal", "gehan", "cindex")
    parsed <- .parse_formula(object$formula, newdata, has_survival, need_response = FALSE,
                             allow_no_intercept = TRUE)
    x_matrix <- parsed$x_matrix
  }

  if (!is.null(which)) {
    # mboost's predict(which = ) returns a matrix, one column per learner,
    # holding that learner's contribution (beta_j times its centered column)
    # without the offset.
    names_vec <- object$learner_names
    idx <- if (is.character(which)) match(which, names_vec) else as.integer(which)
    if (anyNA(idx) || any(idx < 1L | idx > length(names_vec))) {
      stop("unknown learner in which", call. = FALSE)
    }
    k <- if (mstop_val <= 0L) length(object$selected) else min(mstop_val, length(object$selected))
    selected_k <- object$selected[seq_len(k)]
    core_k <- .glm_core_selected(object)[seq_len(k)]
    coefs_k <- object$coefs[seq_len(k)]
    out <- vapply(idx, function(j) {
      keep <- selected_k == j
      .Call("wrap__predict_glmboost", 0, core_k[keep], coefs_k[keep], object$nu,
            as.double(t(x_matrix)), as.double(object$centering_means), 0L,
            PACKAGE = "mboostrs")
    }, double(nrow(x_matrix)))
    out <- matrix(out, nrow = nrow(x_matrix), ncol = length(idx))
    colnames(out) <- names_vec[idx]
    return(out)
  }

  eta <- .Call("wrap__predict_glmboost",
    start,
    .glm_core_selected(object),
    object$coefs,
    object$nu,
    as.double(t(x_matrix)),  # transpose: R is col-major, Rust reads row-major
    as.double(object$centering_means),
    mstop_val,
    PACKAGE = "mboostrs"
  )

  if (type == "response") {
    if (object$family$name == "propodds") {
      # PropOdds response: cumulative-logit probabilities from eta + K-1 thresholds.
      # Use path-specific thresholds when available so truncated predict(mstop=k)
      # matches a separate mstop=k fit.
      k_effective <- if (is.null(mstop)) length(object$selected) else as.integer(mstop)
      np <- object$nuisance_vec_path
      path_index <- k_effective + 1L  # R list index for zero-based boosting iteration
      if (!is.null(np) && path_index >= 1L && length(np) >= path_index) {
        sigma <- np[[path_index]]
      } else {
        sigma <- object$nuisance
      }
      if (!is.null(sigma) && length(sigma) > 0) {
        n_pred <- length(eta)
        eta <- .Call("wrap__propodds_response_r",
          as.double(eta),
          as.double(sigma),
          PACKAGE = "mboostrs"
        )
        return(matrix(eta, nrow = n_pred,
                      ncol = length(sigma) + 1, byrow = TRUE))
      }
    }
    eta <- .inverse_link(eta, object$family)
  }
  eta
}

#' Print method for mboostrs_glmboost
#'
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param ... Not used.
#' @return Invisibly returns the fitted model object.
#' @seealso [glmboost()], [summary.mboostrs_glmboost()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' print(fit)
#' @export
print.mboostrs_glmboost <- function(x, ...) {
  cat("GLMBoost model\n")
  cat("  Formula:", deparse(x$formula), "\n")
  cat("  Family:", x$family$name, "\n")
  cat("  mstop:", x$mstop, "  nu:", x$nu, "\n")
  cat("  Offset:", .offset_label(x$offset), "\n")
  invisible(x)
}

#' Boosting risk path (generic)
#'
#' @param object A fitted model object.
#' @param ... Not used.
#' @return A numeric vector of risk values, one per boosting iteration
#'   (length `mstop + 1`, including the initial value).
#' @seealso [glmboost()], [plot_risk()], [cvrisk()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' head(risk(fit))
#' @export
risk <- function(object, ...) UseMethod("risk")

#' Structured numerical execution report
#'
#' Returns the versioned engine route, fast-path eligibility and rejection
#' reasons, dimensions, thread count, learner/solver mix including explicit
#' regularization jitter, factor and cross-Gram cache bytes, active
#' family-calibration, calibration-stop, exact tie-selection, and numerical
#' fallback counters, and optional phase timings recorded while fitting.
#' `total_ns` covers the Rust core fit. `basis_ns` and `factorization_ns` split
#' learner preparation, while `binding_ns` covers the remaining frontend
#' conversion; these intervals are exclusive. Route-specific phases that did
#' not run are `NA`.
#' @param object A fitted mboostrs model.
#' @param ... Not used.
#' @return A named list with execution metadata.
#' @export
execution_report <- function(object, ...) UseMethod("execution_report")

#' Iteration-level boosting path
#'
#' Derives one row per learner update from retained model state. Cyclic LSS
#' risk is recorded only on the row that completes a parameter cycle;
#' non-cyclic LSS and standard fits record risk on every row.
#' @param object A fitted mboostrs model.
#' @param ... Not used.
#' @return A data frame with learner identity, risk reduction, coefficient
#'   norms, parameter/cycle identifiers, and nuisance state.
#' @export
path_frame <- function(object, ...) UseMethod("path_frame")

#' @export
execution_report.mboostrs_glmboost <- function(object, ...) object$execution_report

#' @noRd
.render_execution_report <- function(report) {
  fallbacks <- if (length(report$fallback_reasons)) {
    paste(report$fallback_reasons, collapse = ",")
  } else {
    "none"
  }
  message(sprintf(
    paste0(
      "mboostrs execution: route=%s iterations=%d/%d learners=%d ",
      "basis_columns=%d threads=%d\n",
      "  gram_eligible=%s fallbacks=%s nonfinite_fallbacks=%d"
    ),
    report$route, report$completed_mstop, report$requested_mstop,
    report$n_learners, report$total_basis_columns, report$effective_threads,
    tolower(as.character(report$gram_eligible)), fallbacks,
    report$nonfinite_candidate_fallbacks
  ))
  message(sprintf(
    paste0(
      "  learners: scalar=%d block=%d tree=%d; solvers: cholesky=%d lu=%d ",
      "regularized=%d max_jitter=%.3e; ",
      paste0(
        "cache_kib: factors=%.1f gram=%.1f; calibrations=%d ",
        "converged=%d limit=%d failed=%d; tie_selections=%d"
      )
    ),
    report$scalar_learner_count, report$block_learner_count,
    report$tree_learner_count, report$cholesky_solver_count,
    report$lu_solver_count, report$regularized_solver_count,
    if (is.na(report$max_solver_jitter)) 0 else report$max_solver_jitter,
    report$factor_cache_bytes / 1024,
    report$gram_cache_bytes / 1024, report$calibration_count,
    report$calibration_converged_count, report$calibration_limit_count,
    report$calibration_failure_count, report$tie_selection_count
  ))
  if (!is.na(report$total_ns)) {
    message(sprintf(
      paste0(
        "  timing_ms: setup=%.3f basis=%.3f factorization=%.3f ",
        "cross_gram=%.3f iterations=%.3f ",
        "reconstruction=%.3f binding=%.3f total=%.3f"
      ),
      report$setup_ns / 1e6,
      if (is.na(report$basis_ns)) 0 else report$basis_ns / 1e6,
      if (is.na(report$factorization_ns)) 0 else report$factorization_ns / 1e6,
      if (is.na(report$cross_gram_ns)) 0 else report$cross_gram_ns / 1e6,
      report$iteration_ns / 1e6,
      if (is.na(report$reconstruction_ns)) 0 else report$reconstruction_ns / 1e6,
      if (is.na(report$binding_ns)) 0 else report$binding_ns / 1e6,
      report$total_ns / 1e6
    ))
  }
  invisible(report)
}

#' Optimal / current number of boosting iterations
#'
#' @param object An object with an mstop method (e.g. a \code{cvrisk} result).
#' @param ... Not used.
#' @return An integer giving the optimal number of boosting iterations. For an
#'   LSS model, as upstream gamboostLSS: a named integer vector with each
#'   parameter's number of updates for a cyclic fit; for a non-cyclic fit the
#'   total number of updates, with their division between parameters in
#'   \code{attr(, "partitions")}.
#' @seealso [cvrisk()], [cv_weights()], [boost_control()]
#' @examples
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n)
#' y <- 2 * x1 - x2 + rnorm(n)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, control = boost_control(mstop = 100))
#' cr <- cvrisk(fit, b = 5)
#' mstop(cr)
#' @export
mstop <- function(object, ...) UseMethod("mstop")

#' @rdname mstop
#' @param value New mstop value (positive integer).
#' @export
`mstop<-` <- function(object, ..., value) UseMethod("mstop<-")

#' @rdname mstop
#' @export
mstop.mboostrs_glmboost <- function(object, ...) object$mstop

#' @rdname mstop
#' @export
mstop.mboostrs_gamboost <- function(object, ...) object$mstop

# ─── Package generics (must be defined before method files load) ──

#' Variable importance
#'
#' @param object A fitted model.
#' @param ... Not used.
#' @return A named numeric vector of variable importance scores, sorted in
#'   descending order.
#' @seealso [glmboost()], [plot_varimp()], [selected()]
#' @examples
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n)
#' y <- 2 * x1 - x2 + rnorm(n)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, control = boost_control(mstop = 50))
#' varimp(fit)
#' @export
varimp <- function(object, ...) UseMethod("varimp")

#' Extract model components
#'
#' @param object A fitted model.
#' @param what What to extract.
#' @param which Which learners to include (by name or index), or NULL for all.
#' @param ... Not used.
#' @return A numeric vector or matrix of the requested model component, filtered
#'   to the specified learners.
#' @seealso [glmboost()], [coef_path()], [varimp()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' extract(fit, "coefficients")
#' @export
extract <- function(object, what, which = NULL, ...) UseMethod("extract")

#' Coefficient paths over boosting iterations
#'
#' @param object A fitted model.
#' @param which Which learners to include (by name or index), or NULL for all.
#' @param ... Not used.
#' @return A matrix of coefficients with one row per boosting iteration and one
#'   column per learner.
#' @seealso [glmboost()], [coef.mboostrs_glmboost()], [plot_coef_path()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' str(coef_path(fit))
#' @export
coef_path <- function(object, which = NULL, ...) UseMethod("coef_path")

#' Selected learners with optional filtering
#'
#' @param object A fitted model.
#' @param which Which learners to include (by name or index), or NULL for all.
#' @param ... Not used.
#' @return A character vector of learner names selected at least once during
#'   boosting, or a data frame with selection counts.
#' @seealso [glmboost()], [varimp()], [plot_selected()]
#' @examples
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n)
#' y <- 2 * x1 - x2 + rnorm(n)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, control = boost_control(mstop = 50))
#' selected(fit)
#' @export
selected <- function(object, which = NULL, ...) UseMethod("selected")

#' Model weights
#'
#' @param object A fitted model.
#' @param ... Not used.
#' @return Numeric vector of observation weights used by the fit.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @examples
#' set.seed(42)
#' n <- 50
#' dat <- data.frame(y = rnorm(n), x = rnorm(n))
#' fit <- glmboostLSS(y ~ x, data = dat, control = boost_control(mstop = 10))
#' head(model_weights(fit))
#' @export
model_weights <- function(object, ...) UseMethod("model_weights")

#' Predicted LSS parameters
#'
#' @param object A fitted LSS model.
#' @param ... Additional arguments passed to methods.
#' @return A named list of parameter vectors.
#' @seealso [glmboostLSS()], [gamboostLSS()]
#' @export
lss_params <- function(object, ...) UseMethod("lss_params")

#' All-parameter predictions
#'
#' @param object A fitted model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame or list with predicted parameters.
#' @seealso [lss_params()]
#' @export
predictAll <- function(object, ...) UseMethod("predictAll")

#' All-parameter coefficients
#'
#' @param object A fitted model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame or list with coefficients by parameter.
#' @seealso [coef()]
#' @export
coefAll <- function(object, ...) UseMethod("coefAll")

#' Per-observation deviance contribution
#'
#' @param object A fitted model.
#' @param ... Additional arguments passed to methods.
#' @return Numeric vector of per-observation deviance contributions.
#' @seealso [lss_params()]
#' @export
deviance_increment <- function(object, ...) UseMethod("deviance_increment")

#' Probability integral transform values
#'
#' @param object A fitted LSS model.
#' @param ... Additional arguments passed to methods.
#' @return Numeric PIT vector.
#' @seealso [lss_cdf()], [quantile_residuals()]
#' @export
pit <- function(object, ...) UseMethod("pit")

#' LSS prediction intervals
#'
#' @param object A fitted LSS model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame with interval endpoints.
#' @seealso [glmboostLSS()], [gamboostLSS()], [lss_quantile()]
#' @export
prediction_interval <- function(object, ...) UseMethod("prediction_interval")

#' Alias for LSS prediction intervals
#'
#' @param object A fitted LSS model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame with interval endpoints.
#' @seealso [prediction_interval()]
#' @export
predint <- function(object, ...) UseMethod("predint")

#' Alias for LSS prediction intervals
#'
#' @param object A fitted LSS model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame with interval endpoints.
#' @seealso [prediction_interval()]
#' @export
PI <- function(object, ...) UseMethod("PI")

#' LSS centile table
#'
#' @param object A fitted LSS model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame with centile columns.
#' @seealso [lss_quantile()]
#' @export
centiles <- function(object, ...) UseMethod("centiles")

#' LSS randomized quantile residuals
#'
#' @param object A fitted LSS model.
#' @param ... Additional arguments passed to methods.
#' @return Numeric residual vector.
#' @seealso [lss_cdf()]
#' @export
quantile_residuals <- function(object, ...) UseMethod("quantile_residuals")

#' Diagnostic residual table
#'
#' @param object A fitted model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame with residual diagnostics.
#' @seealso [quantile_residuals()], [pit()], [deviance_increment()]
#' @export
residual_frame <- function(object, ...) UseMethod("residual_frame")

#' Binned quantile-residual summaries
#'
#' @param object A fitted model.
#' @param ... Additional arguments passed to methods.
#' @return A data frame of residual moments by interval.
#' @seealso [quantile_residuals()]
#' @export
q_stats <- function(object, ...) UseMethod("q_stats")

#' Partial effects for gamboost
#'
#' @param object A fitted model.
#' @param ... Not used.
#' @return A list of partial effect vectors, one per smooth learner.
#' @seealso [gamboost()], [plot.mboostrs_gamboost()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- sin(x) + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- gamboost(y ~ bbs(x), data = df, family = Gaussian(),
#'                 control = boost_control(mstop = 50))
#' str(partial_effects(fit))
#' @export
partial_effects <- function(object, ...) UseMethod("partial_effects")

# ─── Plot generics ────────────────────────────────────────────────

#' Plot risk path
#'
#' @param x A fitted model object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [risk()], [glmboost()], [plot_diagnostics()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' plot_risk(fit)
#' }
#' @export
plot_risk <- function(x, ...) UseMethod("plot_risk")

#' Plot variable importance
#'
#' @param x A fitted model object.
#' @param sort Logical: sort by importance? (default TRUE).
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [varimp()], [glmboost()], [plot_selected()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n)
#' y <- 2 * x1 - x2 + rnorm(n)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, control = boost_control(mstop = 50))
#' plot_varimp(fit)
#' }
#' @export
plot_varimp <- function(x, sort = TRUE, ...) UseMethod("plot_varimp")

#' Plot selected learners
#'
#' @param x A fitted model object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [selected()], [glmboost()], [plot_varimp()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n)
#' y <- 2 * x1 - x2 + rnorm(n)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, control = boost_control(mstop = 50))
#' plot_selected(fit)
#' }
#' @export
plot_selected <- function(x, ...) UseMethod("plot_selected")

#' Plot residuals
#'
#' @param x A fitted model object.
#' @param type Residual type: "response" (default) or "pearson".
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [glmboost()], [plot_diagnostics()], [plot_risk()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' plot_residuals(fit)
#' }
#' @export
plot_residuals <- function(x, type = "response", ...) UseMethod("plot_residuals")

#' Plot coefficient paths
#'
#' @param x A fitted model object.
#' @param which Which learners to include. NULL = all active.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [coef_path()], [glmboost()], [plot_risk()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' plot_coef_path(fit)
#' }
#' @export
plot_coef_path <- function(x, which = NULL, ...) UseMethod("plot_coef_path")

#' Diagnostic plots
#'
#' @param x A fitted model object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   multi-panel diagnostic plot.
#' @seealso [glmboost()], [plot_risk()], [plot_residuals()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' plot_diagnostics(fit)
#' }
#' @export
plot_diagnostics <- function(x, ...) UseMethod("plot_diagnostics")

#' Plot LSS model
#'
#' @param x A fitted LSS model object.
#' @param parameter Parameter name (e.g. "mu"); NULL = all.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [glmboostLSS()], [plot_lss_risk()], [plot_risk()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- rep(1:3, length.out = n) + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboostLSS(y ~ x, data = df, families = list(mu = Gaussian(), sigma = Gaussian()),
#'                    control = boost_control(mstop = 50))
#' plot_lss(fit)
#' }
#' @export
plot_lss <- function(x, parameter = NULL, ...) UseMethod("plot_lss")

#' Plot LSS risk per parameter
#'
#' @param x A fitted LSS model object.
#' @param parameter Parameter name (e.g. "mu"); NULL = all.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [glmboostLSS()], [plot_lss()], [risk()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- rep(1:3, length.out = n) + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboostLSS(y ~ x, data = df, families = list(mu = Gaussian(), sigma = Gaussian()),
#'                    control = boost_control(mstop = 50))
#' plot_lss_risk(fit)
#' }
#' @export
plot_lss_risk <- function(x, parameter = NULL, ...) UseMethod("plot_lss_risk")

#' Plot observed and fitted LSS distributions
#'
#' @param x A fitted LSS model.
#' @param ... Additional arguments passed to the LSS method.
#' @return Invisibly returns NULL after drawing.
#' @seealso [density_overlay_data()]
#' @export
plot_distribution_density <- function(x, ...) UseMethod("plot_distribution_density")

#' Plot an LSS centile fan
#'
#' @param x A fitted LSS model.
#' @param xvar Numeric covariate used to order the fan.
#' @param ... Additional arguments passed to the LSS method.
#' @return Invisibly returns NULL after drawing.
#' @seealso [centile_fan_data()]
#' @export
plot_centile_fan <- function(x, xvar, ...) UseMethod("plot_centile_fan")

#' Plot a joint LSS deviance profile
#'
#' @param x A fitted LSS model.
#' @param ... Additional arguments passed to the LSS method.
#' @return Invisibly returns NULL after drawing.
#' @seealso [deviance_profile()]
#' @export
plot_deviance_profile <- function(x, ...) UseMethod("plot_deviance_profile")

#' Plot multinomial class probabilities
#'
#' @param x A fitted model object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [Multinomial()], [gamboost()], [plot_varimp()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 150; x <- rnorm(n)
#' eta <- 2 * x
#' y <- factor(cut(eta + rnorm(n), breaks = c(-Inf, -0.5, 0.5, Inf),
#'                 labels = c("A", "B", "C")))
#' df <- data.frame(y = y, x = x)
#' fit <- gamboost(y ~ bols(x), data = df, family = Multinomial(),
#'                 control = boost_control(mstop = 100))
#' plot_multinomial(fit)
#' }
#' @export
plot_multinomial <- function(x, ...) UseMethod("plot_multinomial")

#' Plot stability selection results
#'
#' @param x A stabsel result list.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   barplot of selection frequencies.
#' @seealso [varimp()], [glmboost()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n)
#' y <- 2 * x1 - x2 + rnorm(n)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- glmboost(y ~ x1 + x2, data = df, control = boost_control(mstop = 50))
#' ss <- stabsel(fit, PFER = 1)
#' plot_stabsel(ss)
#' }
#' @export
plot_stabsel <- function(x, ...) {
  max_phat <- x$max
  sel <- x$selected
  max_phat <- sort(max_phat, decreasing = TRUE)
  cols <- ifelse(names(max_phat) %in% sel, "steelblue", "grey70")
  barplot(max_phat, horiz = TRUE, las = 1, col = cols,
          main = "Stability Selection",
          xlab = "Selection Frequency")
  abline(v = x$cutoff, lty = 2, col = "red")
  invisible(NULL)
}

#' Plot LSS cross-validated risk
#'
#' @param x A cvrisk result matrix (B x n_grid) for LSS.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns NULL. Called for its side effect of producing a
#'   plot.
#' @seealso [cvrisk()], [plot_lss_risk()], [mstop()]
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- rnorm(n) + x
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 100))
#' cr <- cvrisk(fit, b = 5)
#' # For LSS models, plot_lss_cvrisk takes the risk matrix directly
#' }
#' @export
plot_lss_cvrisk <- function(x, ...) {
  if (!is.matrix(x)) stop("Expected a B x n_grid risk matrix")
  B <- nrow(x)
  n_grid <- ncol(x)
  agg <- colMeans(x)
  ylim <- range(x)
  plot(1:n_grid, agg, type = "b", lwd = 2, pch = 19, col = "black",
       ylim = ylim, xlab = "Grid Point", ylab = "Risk",
       main = "LSS Cross-validated Risk", ...)
  for (i in 1:B) {
    lines(1:n_grid, x[i, ], col = rgb(0, 0, 0, 0.15))
  }
  min_idx <- which.min(agg)
  points(min_idx, agg[min_idx], pch = 17, col = "red", cex = 1.5)
  invisible(NULL)
}

#' Risk path
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param ... Not used.
#' @return A numeric vector of risk values of length `mstop + 1` (including
#'   the initial risk before any boosting iteration).
#' @seealso [glmboost()], [plot_risk()], [cvrisk()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' head(risk(fit))
#' @export
risk.mboostrs_glmboost <- function(object, ...) {
  object$risk
}

#' Extract coefficients
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param off2int Logical: add offset to intercept? Default FALSE.
#' @param which Character vector of learner names, or NULL for all active.
#' @param ... Not used.
#' @return A named numeric vector of coefficients, summed over all boosting
#'   iterations. The offset is stored as the `"offset"` attribute.
#' @seealso [glmboost()], [coef_path()], [predict.mboostrs_glmboost()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 50))
#' coef(fit)
#' coef(fit, off2int = TRUE)
#' @export
coef.mboostrs_glmboost <- function(object, off2int = FALSE, which = NULL, ...) {
  .refuse_unused("coef", ...)
  # The core aggregates the path, corrects the intercept for the centered
  # predictors (mboost's coef.glmboost subtracts sum(beta_j * mean_j)) and
  # keeps the selected learners, as Python's coef() does.
  if (off2int && length(object$offset) > 1L) {
    warning("a per-row offset has no single value to add to the intercept; ",
            "off2int is ignored", call. = FALSE)
    off2int <- FALSE
  }
  has_intercept <- isTRUE(object$intercept %||% TRUE)
  names_vec <- object$learner_names
  # A fit without an intercept (`y ~ x - 1`) gets an unselected intercept slot
  # in the core's numbering, dropped below: mboost reports such a fit's raw
  # coefficients.
  shift <- if (has_intercept) 0L else 1L
  ms <- object$mstop
  which_pos <- integer(0)
  if (!is.null(which)) {
    which_pos <- if (is.character(which)) match(which, names_vec) else as.integer(which)
    if (anyNA(which_pos)) stop("unknown learner in which", call. = FALSE)
    which_pos <- which_pos + shift
  }
  res <- .Call("wrap__coef_glmboost_r",
               as.integer(object$selected[seq_len(ms)] + shift),
               as.double(unlist(object$coefs)[seq_len(ms)]),
               as.double(object$nu), as.integer(length(names_vec) + shift),
               as.double(object$centering_means %||% numeric(0)),
               isTRUE(off2int) && has_intercept,
               as.double(object$offset[1]), as.integer(which_pos),
               PACKAGE = "mboostrs")
  keep <- res$indices > shift
  result <- setNames(as.double(res$values[keep]), names_vec[res$indices[keep] - shift])
  attr(result, "offset") <- object$offset
  result
}

#' @rdname concurvity
#' @export
concurvity.mboostrs_glmboost <- function(object, pairwise = FALSE, ...) {
  .refuse_unused("concurvity", ...)
  out <- .Call("wrap__glmboost_concurvity_r",
    as.double(t(object$x_matrix)),
    as.integer(nrow(object$x_matrix)),
    as.integer(ncol(object$x_matrix)),
    as.double(object$weights),
    as.double(object$nu),
    as.integer(object$selected),
    as.double(object$coefs),
    as.double(object$centering_means),
    isTRUE(object$intercept),
    PACKAGE = "mboostrs"
  )
  .boosted_concurvity_frame(out, object$learner_names, pairwise)
}

#' Slice model at iteration k
#'
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param k Iteration number to slice at (1 to mstop).
#' @return A modified copy of the model object truncated to `k` boosting
#'   iterations, with updated `fitted`, `selected`, `coefs`, `risk`, and
#'   `mstop` fields.
#' @seealso [glmboost()], [predict.mboostrs_glmboost()], [boost_control()]
#' @examples
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- 2 * x + rnorm(n)
#' df <- data.frame(y = y, x = x)
#' fit <- glmboost(y ~ x, data = df, control = boost_control(mstop = 100))
#' fit_50 <- fit[50]
#' mstop(fit_50)
#' @export
`[.mboostrs_glmboost` <- function(x, k) {
  if (k <= 0 || k > x$mstop) stop("k must be between 1 and mstop")
  k <- as.integer(k)

  eta <- predict(x, mstop = k, type = "link")

  x$fitted <- eta
  x$selected <- x$selected[1:k]
  x$coefs <- x$coefs[1:k]
  x$risk <- x$risk[1:(k + 1)]
  x$mstop <- k
  if (!is.null(x$nuisance_vec_path)) {
    x$nuisance_vec_path <- x$nuisance_vec_path[seq_len(k + 1)]
  }
  if (!is.null(x$execution_report)) {
    x$execution_report$completed_mstop <- k
  }
  x
}

#' @rdname mstop
#' @export
`mstop<-.mboostrs_glmboost` <- function(object, ..., value) {
  object[value]
}

#' Inverse link function
#' @noRd
.inverse_link <- function(eta, family) {
  name <- family$name
  if (name %in% c("gaussian", "gaussreg")) return(eta)
  if (name %in% c("binomial", "adaexp")) return(1 / (1 + exp(-2 * eta)))
  if (name %in% c("poisson", "gammasreg", "nbinomial", "hurdle")) return(exp(eta))
  if (name == "propodds") {
    # PropOdds response handled in predict.mboostrs_glmboost/gamboost via
    # propodds_response_r (cumulative logit with per-iteration thresholds).
    # Fallback: naive single sigmoid (correct only for K=2).
    return(1 / (1 + exp(-eta)))
  }
  # CoxPH/Weibull/Loglog are log-link (AFT/relative-risk): response = exp(eta).
  # Loglog is log-logistic AFT, NOT a logistic model -- exp(eta), not sigmoid.
  if (name %in% c("coxph", "weibull", "loglog", "lognormal")) return(exp(eta))
  eta
}

#' Parse a formula and extract model frame/design matrix
#'
#' @param need_response If TRUE (fit), the response must be present and is
#'   extracted. If FALSE (predict on newdata), the response is dropped from the
#'   terms so the design can be built without a response column.
#' @noRd
.parse_formula <- function(formula, data, has_survival = FALSE, need_response = TRUE,
                           multivariate = FALSE, allow_no_intercept = FALSE) {
  mt <- terms(formula, data = data)

  if (need_response) {
    mf <- model.frame(formula, data = data)
    # model.frame retains makepredictcall() parameters for scale(), poly()
    # and other supported formula transforms. Replay those at prediction.
    mt <- attr(mf, "terms")
    if (has_survival) {
      y_raw <- model.response(mf)
      if (!inherits(y_raw, "Surv")) {
        stop("Survival families require a Surv() response")
      }
      y <- c(as.double(y_raw[, 1]), as.double(y_raw[, 2]))
      n <- length(y_raw[, 1])
    } else if (multivariate) {
      # cbind(y1, ..., yK) response: n x K matrix flattened column-major to
      # the concatenated [col0.., .., col_{K-1}..] slice the Rust family expects.
      y_raw <- as.matrix(model.response(mf))
      y <- as.double(y_raw)
      n <- nrow(y_raw)
    } else {
      y <- as.double(model.response(mf))
      n <- length(y)
    }
  } else {
    # Prediction: no response column in newdata.
    y <- NULL
    n <- nrow(data)
    mt <- stats::delete.response(mt)
  }

  # `- 1` / `+ 0` sets this to 0: R's model.matrix then has no intercept
  # column, and the fit offers no constant learner (mboost's behaviour).
  intercept <- !identical(as.integer(attr(mt, "intercept")), 0L)
  if (!intercept && !allow_no_intercept) {
    stop("intercept control ('- 1' / '+ 0') is available in glmboost() only; ",
         "this model always carries an intercept.", call. = FALSE)
  }

  x_matrix <- model.matrix(mt, data = data)
  contrasts <- attr(x_matrix, "contrasts")
  # model.matrix writes a non-syntactic variable name in backticks; the
  # coefficient is reported under the column's own name.
  colnames(x_matrix) <- .unquote_matrix_label(colnames(x_matrix))
  learner_names <- colnames(x_matrix)

  # Remove intercept column (handled separately in Rust)
  if ("(Intercept)" %in% learner_names) {
    idx <- which(learner_names == "(Intercept)")
    x_matrix <- x_matrix[, -idx, drop = FALSE]
    learner_names <- colnames(x_matrix)
  }

  list(y = y, n = n, x_matrix = x_matrix, learner_names = learner_names,
       intercept = intercept,
       prediction_terms = if (need_response) mt else NULL,
       xlevels = if (need_response) stats::.getXlevels(mt, mf) else NULL,
       contrasts = contrasts)
}


# `offset=` as the Rust binding takes it: NaN for none, one number, or one
# finite entry per row.
.offset_arg <- function(offset, n) {
  if (is.null(offset)) return(NaN)
  offset <- as.double(offset)
  if (length(offset) == 1L) return(offset)
  if (length(offset) != n) {
    stop("offset has ", length(offset), " entries but the data have ", n,
         " rows; give one number or one known contribution per row", call. = FALSE)
  }
  if (any(!is.finite(offset))) {
    stop("offset must be finite in every row; log(0) of a zero exposure is -Inf. ",
         "Drop rows with no exposure or give them weight zero", call. = FALSE)
  }
  offset
}

# The linear predictor's start for a prediction. Training rows reuse the
# fitted offset; new rows need their own per-row offset when the model was
# fitted with one, because the fitted vector describes other rows.
.prediction_offset <- function(object, newdata, offset) {
  if (!is.null(offset)) return(as.double(offset))
  if (length(object$offset) > 1L && !is.null(newdata)) {
    stop("the model was fitted with a per-row offset; predict() on new rows ",
         "needs offset = (for example log(exposure) of the new rows)", call. = FALSE)
  }
  as.double(object$offset)
}

.offset_label <- function(offset) {
  if (length(offset) > 1L) sprintf("per-row (%d values)", length(offset))
  else formatC(offset, digits = 6)
}

# A boosting step moves nu along the least-squares fit to the negative
# gradient; for a strongly curved loss (Poisson with small counts at
# nu = 0.5) it can overshoot, and the training risk then oscillates while
# the fit still returns. The core checks material step increases and profiled
# likelihoods ending above their path minimum. Changing adaptive losses such
# as Huber with an estimated delta remain omitted.
.warn_if_risk_rose <- function(risk, control, family_name, family_params) {
  if (!identical(control$risk %||% "inbag", "inbag") || length(risk) < 3L) return(invisible())
  if (!identical(control$step %||% "fixed", "fixed")) return(invisible())
  found <- .Call("wrap__r_risk_path_rises", family_name, family_params,
                 as.double(risk), PACKAGE = "mboostrs")
  if (is.null(found)) return(invisible())
  rises <- if (found[1] > 0) sprintf(
    "the training risk rose on %d of %d boosting steps (largest rise %.3g of the risk before it)",
    as.integer(found[1]), length(risk) - 1L, found[2]
  ) else "the training risk rose above an earlier minimum"
  warning(sprintf(paste0(
    "%s; the final risk is %.3f%% above its minimum at step %d: ",
    "the step nu=%g is too large for this loss's ",
    "curvature, so the path does not consistently descend. Use a smaller nu ",
    "or boost_control(step = 'line_search')."),
    rises, 100 * found[3],
    as.integer(found[4]), control$nu), call. = FALSE)
  invisible()
}

# A per-parameter nu applies to distributional fits only.
.require_scalar_nu <- function(control) {
  if (length(control$nu) != 1L) {
    stop("boost_control(nu = ...) has ", length(control$nu), " values; a ",
         "per-parameter step applies to glmboostLSS() and gamboostLSS() only",
         call. = FALSE)
  }
  invisible(NULL)
}
