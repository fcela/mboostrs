# ─── Missing functionality: AIC, hatvalues, confint, stabsel, make_grid_lss ──
#
# Pure-R implementations of features that exist in the Python API but were
# absent or stubbed in the R package.

# ─── hatvalues (glmboost) ──────────────────────────────────────────

#' Hat matrix diagonal for a fitted glmboost model
#'
#' Computes the exact diagonal of the boosting hat matrix by iteratively
#' applying the rank-1 projection updates. For n > 500 this can be slow;
#' set \code{approximate = TRUE} for a fast trace-based approximation.
#'
#' @param model A fitted \code{mboostrs_glmboost} object.
#' @param approximate If TRUE, return an approximate hat diagonal (fast).
#' @param ... Not used.
#' @return A numeric vector of length n (hat values / leverages).
#' @examples
#' set.seed(42)
#' n <- 50
#' x <- matrix(rnorm(n * 3), n, 3)
#' colnames(x) <- c("x1", "x2", "x3")
#' dat <- data.frame(y = rnorm(n), x)
#' fit <- glmboost(y ~ ., data = dat, family = Gaussian())
#' h <- hatvalues(fit)
#' sum(h)
#' @seealso [glmboost()], [AIC.mboostrs_glmboost()]
#' @export
hatvalues.mboostrs_glmboost <- function(model, approximate = FALSE, ...) {
  if (approximate) {
    X <- cbind(1, sweep(model$x_matrix, 2, model$centering_means))
    return(.hat_approx_glmboost(model, X))
  }
  if (model$mstop == 0) return(rep(0, model$n))
  .core_hat(model)$diag
}

#' Hat matrix diagonal for a fitted gamboost model
#'
#' @param model A fitted \code{mboostrs_gamboost} object.
#' @param approximate If TRUE, return an approximate hat diagonal (fast).
#' @param ... Not used.
#' @return A numeric vector of length n (hat values / leverages).
#' @examples
#' set.seed(42)
#' n <- 50
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' fit <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat)
#' h <- hatvalues(fit, approximate = TRUE)
#' summary(h)
#' @seealso [gamboost()]
#' @export
hatvalues.mboostrs_gamboost <- function(model, approximate = FALSE, ...) {
  if (approximate) {
    return(.hat_approx(model))
  }
  if (model$mstop == 0) return(rep(0, model$n))
  .core_hat(model)$diag
}

# The boosting hat matrix, computed by the Rust core from the learners the fit
# used: `trace` after every step and the final `diag`. Binomial and Poisson
# weight each step by the IRLS weights at the fit after that step, as
# mboost's `hatvalues` does; other families use the unweighted matrix.
.core_hat <- function(model) {
  ms <- model$mstop
  sel <- as.integer(model$selected[seq_len(ms)])
  coefs <- lapply(seq_len(ms), function(m) as.double(model$coefs[[m]]))
  fam <- model$family$name
  res <- if (inherits(model, "mboostrs_glmboost")) {
    .Call("wrap__glmboost_hat_r",
          as.double(t(model$x_matrix)), as.integer(model$n), as.integer(ncol(model$x_matrix)),
          as.double(model$weights), as.double(model$centering_means),
          isTRUE(model$intercept %||% TRUE), sel, coefs, as.double(model$offset),
          as.double(model$nu), fam, PACKAGE = "mboostrs")
  } else {
    if (is.null(model$rust_specs)) {
      stop("this model was fitted by an older mboostrs and keeps no learner ",
           "specification; refit it to compute the hat matrix", call. = FALSE)
    }
    .Call("wrap__gamboost_hat_r",
          model$rust_specs, as.double(model$weights),
          as.integer(model$multinomial_n_classes %||% 0L), sel, coefs,
          as.double(model$offset), as.double(model$nu), fam, PACKAGE = "mboostrs")
  }
  if (is.null(res)) {
    stop("a selected learner (btree or bmono) has no hat matrix, so hat values ",
         "and the hat-trace AIC are unavailable", call. = FALSE)
  }
  res
}

.hat_approx_glmboost <- function(model, X) {
  n <- model$n
  nu <- model$nu
  h <- rep(1 / n, n)  # intercept contribution
  for (step in seq_along(model$selected)) {
    j <- model$selected[step]
    xj <- X[, j]
    xtx <- sum(xj^2)
    if (xtx < .Machine$double.eps) next
    h <- h + nu * xj^2 / xtx
  }
  pmin(pmax(h, 0), 1)
}

# First-order approximation of the hat diagonal, computed by the core
# (`mboostrs_core::engine::try_approximate_hat_diag`): 1/n plus nu times the
# selected learner's leverage at each step, over every learner kind that has
# a hat matrix. Multi-column learners (splines, factor blocks, brandom) used
# to be skipped here.
.hat_approx <- function(model) {
  if (is.null(model$rust_specs)) {
    stop("this model was fitted by an older mboostrs and keeps no learner ",
         "specification; refit it to compute the hat matrix", call. = FALSE)
  }
  h <- .Call("wrap__gamboost_hat_approx_r", model$rust_specs,
             as.double(model$weights), as.integer(model$multinomial_n_classes %||% 0L),
             as.integer(model$selected[seq_len(model$mstop)]), as.double(model$nu),
             PACKAGE = "mboostrs")
  if (is.null(h)) {
    stop("a selected learner (btree or bmono) has no hat matrix, so the ",
         "approximate hat values are unavailable", call. = FALSE)
  }
  h
}

# ─── AIC ──────────────────────────────────────────────────────────

# Families whose loss is not a log-likelihood, so an AIC computed from it is
# not an information criterion. `mboost` warns for these ("AIC might not be
# reasonable for family ...") and then errors on the corrected method; both
# mboostrs frontends used to return a full, plausible-looking path instead, so
# a user selecting mstop by AIC on a quantile model got a curve and a minimum
# where upstream would have stopped them.
.AIC_NON_LIKELIHOOD_FAMILIES <- c(
  gammasreg = "GammaReg (normalized likelihood is unavailable)",
  quantreg = "Quantile Regression",
  expectreg = "Expectile Regression",
  huber = "Huber Regression",
  laplace = "Laplace Regression",
  adaexp = "AdaExp",
  hinge = "Hinge Loss",
  auc = "AUC",
  gehan = "Gehan",
  cindex = "Concordance Index",
  multinomial = "Multinomial"
)

# Families for which the AIC is a genuine likelihood-based criterion. The
# corrected criterion additionally assumes a Gaussian scale, which is why
# `mboost` restricts it further than the warning.
.AIC_GAUSSIAN_FAMILIES <- "gaussian"

.check_aic_family <- function(object, method) {
  fname <- object$family$name
  label <- if (fname %in% names(.AIC_NON_LIKELIHOOD_FAMILIES)) {
    .AIC_NON_LIKELIHOOD_FAMILIES[[fname]]
  } else NULL
  if (!is.null(label)) {
    stop("AIC is unavailable for non-likelihood family ", label,
         "; use cvrisk() instead", call. = FALSE)
  }
  if (method == "classical" && fname %in% c("gaussian", "gaussreg")) {
    stop("classical AIC method not implemented for Gaussian squared-error loss",
         call. = FALSE)
  }
  if (method %in% c("corrected", "gmdl") && !(fname %in% .AIC_GAUSSIAN_FAMILIES)) {
    stop(method, " AIC method is only implemented for Gaussian; use classical for a likelihood family or cvrisk()",
         call. = FALSE)
  }
  invisible(NULL)
}

# One AIC path, computed by the Rust core.
#
# `risk` has length mstop + 1 (index 1 is the offset-only state) and `df` has
# length mstop. The core returns the criterion `mboost` reports and the 1-based
# optimal iteration, so R, Python, and the engine now agree by construction
# rather than by three implementations happening to match.
.core_aic <- function(object, df_path, method, k) {
  if (length(k) != 1L || !is.numeric(k) || !is.finite(k) || k <= 0) {
    stop("k must be a positive finite number", call. = FALSE)
  }
  if (method != "classical" && k != 2) {
    stop("k only applies to classical AIC; ", method, " has a fixed penalty", call. = FALSE)
  }
  if (!identical(object$risk_kind, "inbag")) {
    stop("AIC requires a fit recording risk='inbag'; refit models with missing risk metadata", call. = FALSE)
  }
  ms <- object$mstop
  risk <- as.double(object$risk[seq_len(ms + 1L)])
  w <- object$weights
  sumw <- if (is.null(w)) as.double(object$n) else sum(w)
  y <- object$data[[.response_name(object)]]
  # mboost::AICboost uses the raw unweighted response sum of squares.
  y_ss <- if (is.numeric(y)) sum(y^2) else 0
  res <- .Call("wrap__compute_aic_r", method, risk, as.double(df_path),
               as.double(sumw), as.double(k), as.double(y_ss))
  path <- as.double(res$aic)
  best <- as.integer(res$mstop)
  # `stats::AIC` is documented to return one value per model, and `AIC(m1, m2)`
  # model comparison depends on it; upstream `mboost` returns a length-1
  # `gbAIC`. This used to return the whole path as a bare numeric vector,
  # which satisfies neither. The value is the criterion at its own optimum,
  # and the path stays reachable as the `AIC` attribute, matching `mboost`.
  value <- path[best]
  attr(value, "AIC") <- path
  attr(value, "criterion") <- paste0("mboost_", method)
  attr(value, "normalization") <- if (method == "classical") "total" else "per_observation"
  attr(value, "df_convention") <- "conditional_hat_trace"
  attr(value, "penalty") <- if (method == "classical") k else NA_real_
  attr(value, "df") <- df_path
  attr(value, "nobs") <- object$n
  attr(value, "mstop") <- best
  attr(value, "corrected") <- identical(method, "corrected")
  attr(value, "method") <- method
  class(value) <- c("mboostrs_gbAIC", "AIC")
  value
}

#' Optimal number of boosting iterations from an AIC object
#'
#' @param object An object returned by [AIC.mboostrs_glmboost()] or
#'   [AIC.mboostrs_gamboost()].
#' @param ... Not used.
#' @return The 1-based iteration minimising the criterion.
#' @export
mstop.mboostrs_gbAIC <- function(object, ...) attr(object, "mstop")

#' The full AIC path behind an AIC object
#'
#' `AIC()` returns the criterion at its optimum, one value, as
#' `stats::AIC` requires. This returns the whole path, one entry per boosting
#' iteration, which is what `plot_aic()` draws.
#'
#' @param object An object returned by [AIC.mboostrs_glmboost()] or
#'   [AIC.mboostrs_gamboost()].
#' @return A numeric vector of length `mstop`.
#' @export
aic_path <- function(object) {
  path <- attr(object, "AIC")
  if (is.null(path)) {
    stop("not an mboostrs AIC object", call. = FALSE)
  }
  path
}

#' @export
print.mboostrs_gbAIC <- function(x, ...) {
  best <- attr(x, "mstop")
  cat(format(as.numeric(x)), "\n")
  cat("Optimal number of boosting iterations:", best, "\n")
  cat("Degrees of freedom (for mstop =", paste0(best, "):"),
      format(attr(x, "df")[best]), "\n")
  invisible(x)
}

.response_name <- function(object) {
  trimws(strsplit(deparse(object$formula), "~", fixed = TRUE)[[1]][1])
}

#' AIC for a fitted glmboost model
#'
#' Computes an upstream-compatible stopping criterion. The corrected
#' Gaussian criterion is per-observation, not a total GAIC. Its conditional
#' hat-matrix trace does not include model-search degrees of freedom.
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param method "corrected" (default), "classical", or "gmdl".
#' @param k Positive finite penalty multiplier for the classical method.
#'   Corrected and gmdl criteria have fixed penalties and require k = 2.
#' @param ... Not used.
#' @return A length-1 numeric: the criterion at its own optimal iteration,
#'   as `stats::AIC` requires and as upstream `mboost` returns. The full path
#'   is the `AIC` attribute, reachable with [aic_path()]; the optimal iteration
#'   is the `mstop` attribute, reachable with `mstop()`. Has class
#'   `mboostrs_gbAIC`.
#' @examples
#' set.seed(42)
#' n <- 50
#' x <- matrix(rnorm(n * 3), n, 3)
#' colnames(x) <- c("x1", "x2", "x3")
#' dat <- data.frame(y = rnorm(n), x)
#' fit <- glmboost(y ~ ., data = dat, family = Gaussian())
#' aic_vals <- AIC(fit)
#' mstop(aic_vals)
#' @seealso [logLik.mboostrs_glmboost()], [hatvalues.mboostrs_glmboost()], [plot_aic()]
#' @export
AIC.mboostrs_glmboost <- function(object, method = c("corrected", "classical", "gmdl"),
                                   k = 2, ...) {
  method <- match.arg(method)
  .check_aic_family(object, method)
  .core_aic(object, .hat_trace_path_glmboost(object), method, k)
}

#' AIC for a fitted gamboost model
#'
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param method "corrected" (default), "classical", or "gmdl".
#' @param k Positive finite penalty multiplier for the classical method.
#'   Corrected and gmdl criteria have fixed penalties and require k = 2.
#' @param ... Not used.
#' @return A length-1 numeric: the criterion at its own optimal iteration,
#'   as `stats::AIC` requires and as upstream `mboost` returns. The full path
#'   is the `AIC` attribute, reachable with [aic_path()]; the optimal iteration
#'   is the `mstop` attribute, reachable with `mstop()`. Has class
#'   `mboostrs_gbAIC`.
#' @examples
#' set.seed(42)
#' n <- 50
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' fit <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat)
#' aic_vals <- AIC(fit)
#' mstop(aic_vals)
#' @seealso [gamboost()], [AIC.mboostrs_glmboost()]
#' @export
AIC.mboostrs_gamboost <- function(object, method = c("corrected", "classical", "gmdl"),
                                   k = 2, ...) {
  method <- match.arg(method)
  .check_aic_family(object, method)
  .core_aic(object, .hat_trace_path_gamboost(object), method, k)
}

# Hat matrix trace after each of the first mstop steps.
.hat_trace_path_glmboost <- function(model) {
  if (model$mstop == 0) return(numeric(0))
  .core_hat(model)$trace
}

.hat_trace_path_gamboost <- .hat_trace_path_glmboost

# Log-likelihood path at each mstop
.loglik_path_glmboost <- function(model) {
  n <- model$n
  ms <- model$mstop
  fname <- model$family$name

  if (fname %in% c("gaussian", "gaussreg")) {
    y <- .get_response(model)
    # Compute fitted at each mstop and RSS
    ll <- numeric(ms)
    for (m in 1:ms) {
      fitted_m <- predict(model, mstop = m, type = "link")
      rss <- sum((y - fitted_m)^2)
      sigma2 <- rss / n
      if (sigma2 <= 0) {
        ll[m] <- -Inf
      } else {
        ll[m] <- -n / 2 * log(2 * pi * sigma2) - rss / (2 * sigma2)
      }
    }
    return(ll)
  }

  # For non-Gaussian, risk = empirical risk (negative log-likelihood scale)
  risk <- model$risk[1:(ms + 1)]
  -risk[-1]
}

.loglik_path_gamboost <- function(model) {
  n <- model$n
  ms <- model$mstop
  risk <- model$risk[1:(ms + 1)]
  -risk[-1]
}

# ─── confint (bootstrap CIs) ──────────────────────────────────────

# Run `expr` under `seed` and put the caller's RNG stream back afterwards. The
# seed is an argument of the interval, not a side effect on the session: a
# `set.seed()` here used to reset the global stream, so a script that seeded
# before `confint()` and drew after it got different numbers than it asked for.
.with_preserved_seed <- function(seed, expr) {
  env <- .GlobalEnv
  had_state <- exists(".Random.seed", envir = env, inherits = FALSE)
  if (had_state) old_state <- get(".Random.seed", envir = env, inherits = FALSE)
  on.exit({
    if (had_state) {
      assign(".Random.seed", old_state, envir = env)
    } else if (exists(".Random.seed", envir = env, inherits = FALSE)) {
      rm(".Random.seed", envir = env)
    }
  }, add = TRUE)
  set.seed(seed)
  force(expr)
}

# Lower and upper percentile bounds of bootstrap draws, one row per draw and
# one column per coefficient. The type-7 quantiles are computed by the Rust
# core (`mboostrs_core::inference::percentile_interval`), the same function the
# Python frontend uses, so both report the same bounds.
.percentile_bounds <- function(draws, level) {
  if (!is.matrix(draws)) draws <- matrix(draws, ncol = 1L)
  out <- .Call("wrap__r_percentile_interval", as.double(t(draws)),
               as.integer(nrow(draws)), as.integer(ncol(draws)),
               as.double(level), PACKAGE = "mboostrs")
  cbind(lower = out$lower, upper = out$upper)
}

# A refit that errors leaves no draw; say so instead of returning an interval
# built from fewer replicates than the caller asked for.
.report_refit_failures <- function(what, n_failed, B) {
  if (n_failed == 0L) return(invisible(NULL))
  warning(sprintf("%s: %d of %d bootstrap refits failed and were excluded",
                  what, n_failed, B), call. = FALSE)
  invisible(NULL)
}

#' Bootstrap confidence intervals for glmboost coefficients
#'
#' Refits the model on B bootstrap resamples and computes percentile
#' confidence intervals for the coefficients.
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param parm Which coefficients (names or indices); NULL = all.
#' @param level Confidence level (default 0.95).
#' @param B Number of bootstrap replications (default 100).
#' @param seed Random seed. The caller's \code{.Random.seed} is restored when
#'   the call returns, so the interval does not advance the global stream.
#' @param ... Not used.
#' @return A matrix with columns giving lower and upper confidence bounds. The
#'   \code{B}, \code{level}, \code{n_used} (replicates that refitted) and
#'   \code{n_failed} attributes report how many draws the bounds came from.
#' @examples
#' set.seed(42)
#' n <- 50
#' x <- matrix(rnorm(n * 3), n, 3)
#' colnames(x) <- c("x1", "x2", "x3")
#' dat <- data.frame(y = rnorm(n), x)
#' fit <- glmboost(y ~ ., data = dat, family = Gaussian())
#' confint(fit, B = 10)
#' @seealso [coef.mboostrs_glmboost()], [plot_confint()]
#' @export
confint.mboostrs_glmboost <- function(object, parm = NULL, level = 0.95,
                                       B = 100, seed = 42L, ...) {
  n <- object$n
  alpha <- (1 - level) / 2
  learner_names <- object$learner_names
  weights <- object$weights %||% rep(1, n)

  coef_mat <- matrix(NA_real_, nrow = B, ncol = length(learner_names))
  colnames(coef_mat) <- learner_names
  n_failed <- 0L
  n_used <- 0L

  .with_preserved_seed(seed, {
    for (b in seq_len(B)) {
      idx <- sample.int(n, replace = TRUE)
      boot_data <- object$data[idx, , drop = FALSE]
      # The refit keeps the fit's observation weights: the resampled rows carry
      # their own weight, so the composition is the weights at the drawn rows,
      # not an unweighted refit of the resampled data.
      fit_b <- tryCatch(
        glmboost(object$formula, data = boot_data, family = object$family,
                 control = .resampling_control(object), weights = weights[idx]),
        error = function(e) NULL
      )
      if (is.null(fit_b)) {
        n_failed <- n_failed + 1L
        next
      }
      coefs_b <- coef(fit_b)
      coef_mat[b, names(coefs_b)] <- coefs_b
      n_used <- n_used + 1L
    }
  })

  .report_refit_failures("confint", n_failed, B)
  coef_mat <- coef_mat[rowSums(!is.na(coef_mat)) > 0, , drop = FALSE]

  bounds <- .percentile_bounds(coef_mat, level)
  result <- cbind(lower = bounds[, 1], upper = bounds[, 2])
  colnames(result) <- paste0(c(alpha, 1 - alpha) * 100, "%")
  rownames(result) <- learner_names

  if (!is.null(parm)) {
    if (is.numeric(parm)) parm <- learner_names[parm]
    result <- result[parm, , drop = FALSE]
  }

  attr(result, "B") <- B
  attr(result, "level") <- level
  attr(result, "n_used") <- n_used
  attr(result, "n_failed") <- n_failed
  class(result) <- c("mboostrs_confint", "matrix")
  result
}

#' Bootstrap confidence intervals for gamboost coefficients
#'
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param parm Which learners; NULL = all.
#' @param level Confidence level (default 0.95).
#' @param B Number of bootstrap replications (default 100).
#' @param seed Random seed. The caller's \code{.Random.seed} is restored when
#'   the call returns, so the interval does not advance the global stream.
#' @param ... Not used.
#' @return A list of matrices (one per learner) with lower/upper bounds,
#'   or a single matrix for scalar learners. A replicate that never selects a
#'   learner contributes a zero coefficient vector, because an unselected
#'   learner has no contribution; it is not dropped as a missing draw. The
#'   \code{B}, \code{level}, \code{n_used} and \code{n_failed} attributes
#'   report how many draws the bounds came from.
#' @examples
#' set.seed(42)
#' n <- 50
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' fit <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat)
#' confint(fit, B = 5)
#' @seealso [coef.mboostrs_gamboost()], [plot_confint()]
#' @export
confint.mboostrs_gamboost <- function(object, parm = NULL, level = 0.95,
                                       B = 100, seed = 42L, ...) {
  if (isTRUE(object$family$name == "multinomial"))
    stop("confint is not supported for the Multinomial family")
  n <- object$n
  alpha <- (1 - level) / 2
  # Keyed like `coef()`: two learners on one column are "x (bols)" and
  # "x (bbs)", and resolving only the raw name lost one of them.
  learner_names <- .base_learner_names(object)
  weights <- object$weights %||% rep(1, n)

  # The coefficient count `coef()` reports for each learner in the fit's own
  # parameterization. A centered `bols()` is reported as intercept + slope, and
  # a centered `bbs()` carries `p - d` coefficients, so the design width
  # `learner_ncol` is only a fallback.
  widths <- as.integer(object$learner_ncol)
  original <- coef(object)
  for (i in seq_along(learner_names)) {
    if (length(original[[i]])) widths[i] <- length(original[[i]])
  }

  draws <- setNames(vector("list", length(learner_names)), learner_names)
  n_failed <- 0L
  n_used <- 0L

  .with_preserved_seed(seed, {
    for (b in seq_len(B)) {
      idx <- sample.int(n, replace = TRUE)
      boot_data <- object$data[idx, , drop = FALSE]
      # The original fit already reported any penalization drift; its B
      # bootstrap refits at the same mstop would repeat it.
      fit_b <- tryCatch(
        .quiet_penalty_drift(
          gamboost(object$formula, data = boot_data, family = object$family,
                   control = .resampling_control(object),
                   weights = weights[idx])),
        error = function(e) NULL
      )
      if (is.null(fit_b)) {
        n_failed <- n_failed + 1L
        next
      }
      n_used <- n_used + 1L
      cb <- coef(fit_b)
      for (i in seq_along(learner_names)) {
        value <- cb[[learner_names[i]]]
        # An unselected learner's contribution is zero. `coef()` returns
        # `numeric(0)` for it, and `rbind` with that silently dropped the whole
        # replicate, removing exactly the draws where the effect is zero.
        if (length(value) == 0L) value <- numeric(widths[i])
        draws[[i]] <- c(draws[[i]], list(as.numeric(value)))
      }
    }
  })

  .report_refit_failures("confint", n_failed, B)

  result <- list()
  for (i in seq_along(learner_names)) {
    rows <- draws[[i]]
    width <- if (length(rows)) max(widths[i], max(lengths(rows))) else widths[i]
    if (!length(rows)) {
      result[[learner_names[i]]] <- matrix(NA_real_, nrow = width, ncol = 2)
    } else {
      mat <- matrix(0.0, nrow = length(rows), ncol = width)
      for (r in seq_along(rows)) {
        value <- rows[[r]]
        n_take <- min(length(value), width)
        mat[r, seq_len(n_take)] <- value[seq_len(n_take)]
      }
      result[[learner_names[i]]] <- .percentile_bounds(mat, level)
    }
    colnames(result[[learner_names[i]]]) <- paste0(c(alpha, 1 - alpha) * 100, "%")
  }

  if (!is.null(parm)) {
    if (is.numeric(parm)) parm <- learner_names[parm]
    result <- result[parm]
  }

  attr(result, "B") <- B
  attr(result, "level") <- level
  attr(result, "n_used") <- n_used
  attr(result, "n_failed") <- n_failed
  class(result) <- c("mboostrs_confint", "list")
  result
}

#' Plot confidence intervals
#'
#' @param x A \code{mboostrs_confint} object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns `NULL`.
#' @examples
#' set.seed(42)
#' n <- 50
#' x <- matrix(rnorm(n * 3), n, 3)
#' colnames(x) <- c("x1", "x2", "x3")
#' dat <- data.frame(y = rnorm(n), x)
#' fit <- glmboost(y ~ ., data = dat, family = Gaussian())
#' ci <- confint(fit, B = 10)
#' if (interactive()) plot_confint(ci)
#' @export
plot_confint <- function(x, ...) UseMethod("plot_confint")

#' @rdname plot_confint
#' @export
plot_confint.mboostrs_confint <- function(x, ...) {
  if (is.matrix(x)) {
    # glmboost
    coefs <- rowMeans(x, na.rm = TRUE)
    nms <- rownames(x)
    yr <- range(x, na.rm = TRUE)
    plot(seq_along(nms), coefs, pch = 19, col = "steelblue",
         ylim = yr, xaxt = "n", xlab = "", ylab = "Coefficient",
         main = "Bootstrap Confidence Intervals", ...)
    axis(1, at = seq_along(nms), labels = nms, las = 2, cex.axis = 0.8)
    arrows(seq_along(nms), x[, 1], seq_along(nms), x[, 2],
           angle = 90, code = 3, length = 0.05, col = "grey50")
    abline(h = 0, lty = 2, col = "red")
  }
  invisible(NULL)
}

# ─── stabsel (full stability selection) ───────────────────────────

#' Stability selection for glmboost
#'
#' Performs stability selection by subsampling: refits the model on each
#' subsample and records which learners are selected. Returns selection
#' frequencies and the stable set.
#'
#' Each subsample enters the refit as a 0/1 in-bag weight vector, so every
#' fold sees the same rows in the same order and only their weights differ.
#' Under the default \code{sampling_type = "SS"} the run draws \code{B}
#' half-samples and also fits each one's exact complement, which is what the
#' Shah-Samworth error bound is derived on; the run therefore performs
#' \code{2 * B} fits. \code{sampling_type = "MB"} draws \code{B} independent
#' half-samples and fits \code{B} models. The bound is computed on \code{B},
#' and both counts are on the result as \code{B} and \code{n_fits}.
#'
#' @param formula A formula object.
#' @param data A data frame.
#' @param family A family object.
#' @param control A boost_control list.
#' @param cutoff Stability cutoff (0.6-0.9 typical, default 0.7).
#' @param q Maximum number of selected variables per subsample.
#' @param PFER Per-family error rate: an upper bound on the expected number of
#'   false selections, not a probability.
#' @param B Number of initial subsamples (default 50): complementary pairs
#'   under \code{"SS"}, independent half-samples under \code{"MB"}.
#' @param prob Subsampling proportion (default 0.5). Only \code{0.5} is
#'   meaningful under \code{"SS"}.
#' @param assumption "unimodal" or "rconcave".
#' @param sampling_type \code{"SS"} for complementary pairs (default) or
#'   \code{"MB"} for independent half-samples.
#' @param folds An \code{n x B} matrix of in-bag weights holding the
#'   \emph{initial} subsamples, or \code{NULL} to generate them. Under
#'   \code{"SS"} the exact complements are appended, so \code{B} columns
#'   produce \code{2 * B} fits.
#' @param seed Random seed.
#' @param offset Optional offset passed to every subsample fit: one number or
#'   one known contribution per row (`log(exposure)`), as in [glmboost()].
#' @param weights Optional observation weights, as in [glmboost()]. Each
#'   subsample carries the fit's weights on its in-bag rows, and under
#'   \code{"SS"} each complement carries the rest of them, so a pair sums to
#'   the full weight vector. Supplied \code{folds} are taken to hold these
#'   weights already.
#' @param ... Not used. The function signals an error if any argument is
#'   supplied.
#' @return A list with \code{selected} (stable variables), \code{max}
#'   (selection frequencies), and stability parameters.
#' @examples
#' set.seed(42)
#' n <- 100
#' p <- 15
#' x <- matrix(rnorm(n * p), n, p)
#' colnames(x) <- paste0("x", 1:p)
#' dat <- as.data.frame(x)
#' dat$y <- 2 * dat$x1 - 1.5 * dat$x3 + rnorm(n)
#' sel <- stabsel(y ~ ., data = dat, family = Gaussian(),
#'                control = boost_control(mstop = 50))
#' sel$selected
#' @seealso [stabsel_params()], [plot_stabsel()]
#' @export
stabsel <- function(formula, data, family = Gaussian(),
                    control = boost_control(), cutoff = 0.7, q = -1L,
                    PFER = -1.0, B = 50L, prob = 0.5,
                    assumption = "unimodal", sampling_type = "SS",
                    folds = NULL, seed = 42L, offset = NULL, weights = NULL, ...) {
  .refuse_unused("stabsel", ...)
  n <- nrow(data)
  sampling_type <- .stabsel_sampling_type(sampling_type)

  # Every fit below goes through glmboost(), which takes plain predictors. A
  # base-learner formula reached `glmboost` and failed inside the term
  # evaluation with "could not find function bols", which names neither the
  # limitation nor the alternative.
  rhs <- paste(deparse(formula[[3]]), collapse = " ")
  learner_call <- "\\b(bols|bbs|bmono|bspatial|brandom|bmrf|bns|btree|bkernel|buser|bconstrained)[[:space:]]*\\("
  if (grepl(learner_call, rhs)) {
    stop("stabsel() runs its subsample fits through glmboost(), so a formula ",
         "with base-learner terms such as bols() or bbs() is not supported ",
         "here. Write the predictors plainly, as in y ~ x1 + x2, for a linear ",
         "stability run; smooth-learner stability selection is available in ",
         "the Python frontend, whose stabsel() takes a fitted gamboost model.",
         call. = FALSE)
  }

  # Fit once to get p (number of learners) and their names.
  initial_control <- control
  initial_control$mstop <- 1L
  fit0 <- glmboost(formula, data = data, family = family, offset = offset,
                   weights = weights, control = initial_control)
  p <- length(fit0$learner_names)
  # The weights the fit used (rescaled as glmboost rescales them); the folds
  # below carry them, so each subsample fit is weighted.
  fold_weights <- if (is.null(weights)) NULL else fit0$weights

  # If q is unspecified, default to min(p-1, 5) — a small value that works
  # with standard cutoffs (0.6-0.9). The user can override for specific needs.
  if (q <= 0L) q <- min(p - 1L, 5L)

  if (is.null(folds)) {
    folds <- stabsel_folds(n, B = B, sampling_type = sampling_type,
                           weights = fold_weights, prob = prob, seed = seed)
    n_fits <- ncol(folds)
    B <- as.integer(B)
  } else {
    prepared <- .stabsel_fit_folds(folds, sampling_type, n, weights = fold_weights)
    folds <- prepared$folds; B <- prepared$B; n_fits <- prepared$n_fits
  }

  par <- tryCatch(
    stabsel_params(p = p, b = B, cutoff = cutoff,
                   q = q, pfer = PFER, assumption = assumption,
                   sampling_type = sampling_type),
    error = function(e) {
      stop("Invalid stabsel parameters (cutoff=", cutoff, ", q=", q, ", p=", p, "): ",
           e$message, ". Try increasing p or adjusting cutoff/q.",
           call. = FALSE)
    }
  )
  cutoff_val <- par$cutoff
  q_val <- par$q

  # One fit per fold column, run by the core. The fold is an in-bag weight
  # vector, so every fit sees the same rows and only their weights change;
  # row subsetting would break the correspondence between a subsample and its
  # complement. Each column gets the scale glmboost(weights = ) gives it.
  folds <- apply(as.matrix(folds), 2, .rescale_weights)
  if (!is.matrix(folds)) folds <- matrix(folds, nrow = n)
  has_survival <- family$name %in% c("coxph", "weibull", "loglog", "lognormal", "gehan", "cindex")
  parsed <- .parse_formula(formula, data, has_survival, allow_no_intercept = TRUE)
  y <- parsed$y
  labelled <- .binary_label_response(.raw_response(formula, data), family$name)
  if (!is.null(labelled)) y <- labelled
  res <- .Call("wrap__stabsel_glmboost_r",
    as.double(t(fit0$x_matrix)), as.integer(n), as.integer(ncol(fit0$x_matrix)),
    as.double(y), as.double(fit0$weights), as.integer(control$mstop),
    as.double(control$nu), family$name, family$params %||% list(),
    colnames(fit0$x_matrix) %||% character(0), as.double(folds),
    as.integer(n_fits), as.double(fit0$centering_means), .offset_arg(offset, n),
    isTRUE(fit0$intercept), as.integer(q_val),
    { s <- control$subsample %||% NA_real_; if (is.na(s)) 0 else as.double(s) },
    as.double(control$subsample_seed %||% 0),
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
    PACKAGE = "mboostrs")
  if (res$failed > 0L) {
    warning(res$failed, " of ", n_fits, " subsample fits failed and contribute no ",
            "selections; the reported frequencies are computed over all ",
            n_fits, " fits.", call. = FALSE)
  }

  phat <- matrix(res$phat, nrow = p, ncol = control$mstop)
  max_phat <- res$max
  names(max_phat) <- fit0$learner_names
  selected <- names(max_phat[max_phat >= cutoff_val])

  result <- list(
    selected = selected,
    phat = phat,
    max = max_phat,
    cutoff = cutoff_val,
    q = q_val,
    PFER = par$pfer,
    p = p,
    # Initial subsamples the error bound was computed on: complementary pairs
    # under "SS", independent half-samples under "MB".
    B = as.integer(B),
    # Models actually fitted: 2 * B under "SS", B under "MB".
    n_fits = as.integer(n_fits),
    sampling_type = sampling_type,
    assumption = assumption,
    param = par
  )
  class(result) <- "mboostrs_stabsel"
  result
}

#' Print stability selection results
#'
#' @param x A \code{mboostrs_stabsel} object.
#' @param ... Not used.
#' @return Invisibly returns the object.
#' @examples
#' set.seed(42)
#' n <- 100
#' p <- 10
#' xm <- matrix(rnorm(n * p), n, p)
#' colnames(xm) <- paste0("x", 1:p)
#' dat <- as.data.frame(xm)
#' dat$y <- 2 * dat$x1 + rnorm(n)
#' sel <- stabsel(y ~ ., data = dat, family = Gaussian(),
#'                control = boost_control(mstop = 50))
#' print(sel)
#' @export
print.mboostrs_stabsel <- function(x, ...) {
  cat("Stability Selection\n")
  cat("  Cutoff:", x$cutoff, "\n")
  cat("  q:", x$q, "\n")
  cat("  PFER:", formatC(x$PFER, digits = 4), "\n")
  label <- if (identical(x$sampling_type, "MB")) "subsamples" else "complementary pairs"
  cat("  B:", x$B, label, "->", x$n_fits, "fits\n")
  cat("  Selected variables:", length(x$selected), "\n")
  if (length(x$selected) > 0) {
    cat("\n  Stable variables:\n")
    freqs <- sort(x$max[x$max >= x$cutoff], decreasing = TRUE)
    for (nm in names(freqs)) {
      cat(sprintf("    %s: %.3f\n", nm, freqs[nm]))
    }
  }
  invisible(x)
}

# ─── make_grid_lss ────────────────────────────────────────────────

#' Generate an mstop grid for GAMLSS cross-validation
#'
#' Creates a grid of per-parameter mstop combinations for use with
#' \code{cvrisk_glmboostLSS()} or \code{cvrisk_gamboostLSS()}.
#'
#' @param mstop_vector A named vector of per-parameter mstop values, or a
#'   scalar applied to all parameters.
#' @param K Number of parameters.
#' @param param_names Parameter names (optional).
#' @param max_combinations Maximum grid size (default 100).
#' @return An n_grid x K matrix of per-parameter mstop values.
#' @examples
#' grid <- make_grid_lss(c(mu = 20, sigma = 20), K = 2)
#' head(grid)
#' @seealso [cvrisk_glmboostLSS()]
#' @export
make_grid_lss <- function(mstop_vector, K = 2, param_names = NULL,
                           max_combinations = 100) {
  if (length(mstop_vector) == 1) {
    mstop_vector <- rep(mstop_vector, K)
  }
  if (length(mstop_vector) != K)
    stop("mstop_vector length must be 1 or K")

  # Generate a reasonable grid: all combinations from 1..mstop_k for each param
  # but capped at max_combinations
  grids <- lapply(mstop_vector, function(m) {
    step <- max(1L, floor(m / 10))
    seq.int(1L, as.integer(m), by = step)
  })

  grid <- as.matrix(expand.grid(grids))
  colnames(grid) <- if (!is.null(param_names)) param_names else paste0("V", seq_len(K))

  if (nrow(grid) > max_combinations) {
    idx <- round(seq(1, nrow(grid), length.out = max_combinations))
    grid <- grid[idx, , drop = FALSE]
  }

  # Ensure the full mstop combination is in the grid
  full_row <- matrix(as.integer(mstop_vector), nrow = 1)
  colnames(full_row) <- colnames(grid)
  grid <- unique(rbind(grid, full_row))

  as.matrix(grid)
}

# ─── plot_aic ─────────────────────────────────────────────────────

#' Plot AIC path
#'
#' @param x A fitted model with an AIC method, or an AIC vector.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns `NULL`.
#' @examples
#' set.seed(42)
#' n <- 50
#' x <- matrix(rnorm(n * 3), n, 3)
#' colnames(x) <- c("x1", "x2", "x3")
#' dat <- data.frame(y = rnorm(n), x)
#' fit <- glmboost(y ~ ., data = dat, family = Gaussian())
#' if (interactive()) plot_aic(fit)
#' @seealso [AIC.mboostrs_glmboost()]
#' @export
plot_aic <- function(x, ...) UseMethod("plot_aic")

#' @rdname plot_aic
#' @export
plot_aic.mboostrs_glmboost <- function(x, ...) {
  aic_vals <- aic_path(AIC(x))
  ms <- length(aic_vals)
  plot(1:ms, aic_vals, type = "l", lwd = 2, col = "darkgreen",
       xlab = "mstop", ylab = "AIC", main = "AIC Path", ...)
  optimal <- which.min(aic_vals)
  points(optimal, aic_vals[optimal], pch = 19, col = "red", cex = 1.5)
  abline(v = optimal, lty = 2, col = "red")
  text(optimal, aic_vals[optimal], labels = paste("mstop =", optimal),
       pos = 4, cex = 0.8)
  invisible(NULL)
}

#' @export
plot_aic.mboostrs_gamboost <- plot_aic.mboostrs_glmboost

# ─── confint generic (if not already defined) ─────────────────────

#' Confidence intervals for model coefficients
#'
#' @param object A fitted model.
#' @param parm Parameters to include.
#' @param level Confidence level.
#' @param ... Additional arguments.
#' @return Depends on the method dispatched. See [confint.mboostrs_glmboost()] or [confint.mboostrs_gamboost()].
#' @seealso [confint.mboostrs_glmboost()]
#' @examples
#' # See method documentation for full examples:
#' # confint.mboostrs_glmboost, confint.mboostrs_gamboost
#' @export
confint <- function(object, parm = NULL, level = 0.95, ...) {
  UseMethod("confint")
}

# Resampling keeps the requested step rule and other fitting controls.
.resampling_control <- function(object) {
  control <- object$control %||% boost_control(object$mstop, object$nu)
  control$mstop <- object$mstop
  control
}
