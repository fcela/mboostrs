# ─── S3 methods for mboostrs_glmboost and mboostrs_gamboost ───────────
#
# Mirrors the Python GLMBoostModel / GAMBoostModel API.
# Pure R — no Rust .Call needed (all data is already in the model object).

# ─── hatvalues ─────────────────────────────────────────────────────
# Implemented in mboostrs_extra.R (exact hat matrix diagonal computation)

# ─── residuals / resid ─────────────────────────────────────────────

#' Residuals for a fitted glmboost model
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param type "response" (default) or "pearson".
#' @param ... Not used.
#' @return A numeric vector of residuals.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' head(residuals(mod))
#' @export
residuals.mboostrs_glmboost <- function(object, type = "response", ...) {
  fam <- object$family$name
  if (fam %in% c("coxph", "weibull", "loglog", "lognormal")) {
    stop("Response residuals are undefined for family '", fam,
         "': the response is censored and the location eta is on the ",
         "log-time scale, so y - mu is not a residual. Use ",
         "residuals(type='working') for the negative gradient, or residual ",
         "quantiles (residual_frame(kind='quantile')) where available.")
  }
  if (fam == "adaexp") {
    stop("Response and Pearson residuals are undefined for family 'adaexp': ",
         "the exponential loss produces a score, not a mu. Use ",
         "residuals(type='working') for the negative gradient.")
  }
  y <- .get_response(object)
  mu <- predict(object, type = "response")
  if (type == "response") return(y - mu)
  if (type == "pearson") return(.pearson_resid(y, mu, object$family))
  stop("type must be 'response' or 'pearson', got '", type, "'")
}

#' Residuals for a fitted gamboost model
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param type "response" (default) or "pearson".
#' @param ... Not used.
#' @return A numeric vector of residuals.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' head(residuals(mod))
#' @export
residuals.mboostrs_gamboost <- function(object, type = "response", ...) {
  fam <- object$family$name
  # Recover raw response from the model data
  if (fam %in% c("coxph", "weibull", "loglog", "lognormal")) {
    stop("Response residuals are undefined for family '", fam,
         "': the response is censored and the location eta is on the ",
         "log-time scale, so y - mu is not a residual. Use ",
         "residuals(type='working') for the negative gradient, or residual ",
         "quantiles (residual_frame(kind='quantile')) where available.")
  }
  if (fam == "multinomial") {
    stop("residuals() is not supported for Multinomial; use predict(type='response').")
  }
  if (fam == "adaexp") {
    stop("Response and Pearson residuals are undefined for family 'adaexp': ",
         "the exponential loss produces a score, not a mu. Use ",
         "residuals(type='working') for the negative gradient.")
  }
  # The response as the fit saw it: a two-class factor coded 0/1 and a
  # transformation such as log(y) applied.
  y_raw <- .get_response(object)
  mu <- predict(object, type = "response")
  if (type == "response") return(y_raw - mu)
  if (type == "pearson") return(.pearson_resid(y_raw, mu, object$family))
  stop("type must be 'response' or 'pearson', got '", type, "'")
}

#' @rdname residuals.mboostrs_glmboost
#' @export
resid.mboostrs_glmboost <- residuals.mboostrs_glmboost

#' @rdname residuals.mboostrs_gamboost
#' @export
resid.mboostrs_gamboost <- residuals.mboostrs_gamboost

# Pearson residuals helper
.pearson_resid <- function(y, mu, family) {
  name <- family$name
  if (name %in% c("gaussian", "gaussreg")) return(y - mu)
  if (name == "binomial") return((y - mu) / sqrt(pmax(mu * (1 - mu), 1e-15)))
  if (name == "poisson") return((y - mu) / sqrt(pmax(mu, 1e-15)))
  if (name == "gammasreg") return((y - mu) / pmax(mu, 1e-15))
  stop("Pearson residuals are undefined for family '", name,
       "': they are (y - mu) / sqrt(V(mu)), and this family declares no ",
       "variance function (only gaussian, gaussreg, binomial, poisson, and ",
"gammasreg do). Laplace, quantile, expectile and Huber losses are not ",
        "probability models; their gradients live under ",
       "residuals(type='working').")
}

# ─── aic ────────────────────────────────────────────────────────────
# AIC methods are implemented in mboostrs_extra.R

# ─── logLik ─────────────────────────────────────────────────────────
#
# For Gaussian this returns the actual Gaussian log-likelihood, recomputed from
# the RSS with the MLE variance, rather than the negated risk. The other
# likelihood families return -risk, which is what `mboost` returns for all
# families; a robust or ranking loss (Laplace, Quantile, Huber, AUC, ...) and
# Multinomial are refused, as in the Python frontend, where `mboost` returns
# the negated loss. Both apply to glmboost and gamboost fits.
#
# That is a deliberate departure and it is worth stating where a reader meets
# the number: a Gaussian `logLik` here is smaller than `mboost`'s by the
# normalizing constant (-220.86 against -166.93 on one 150-point fit), while
# Poisson and Binomial agree with upstream to all printed digits. The value
# returned is a real log-likelihood rather than a negated loss; it is not
# comparable with a `logLik` quoted from `mboost` for the same Gaussian model.
# Both mboostrs frontends agree with each other.

# Families whose loss has no probability model behind it, with the reason and
# the alternative: the families the Python frontend's FAMILY_LIKELIHOOD table
# (python/mboostrs/_likelihood.py) gives no log-likelihood source, with its
# wording. logLik() refuses them rather than return a negated loss that
# compares like a log-likelihood.
.LOGLIK_UNAVAILABLE <- c(
  laplace = paste("absolute-error loss is not a log-likelihood: it targets the",
                  "conditional median with no distributional assumption behind it.",
                  "Compare fits by cross-validated risk (cvrisk)"),
  quantreg = paste("the check loss is not a log-likelihood: it targets a conditional",
                   "quantile with no distributional assumption behind it. Compare fits",
                   "by cross-validated risk (cvrisk) at the same tau"),
  expectreg = paste("the asymmetric squared-error loss is not a log-likelihood: it",
                    "targets a conditional expectile. Compare fits by cross-validated",
                    "risk (cvrisk) at the same tau"),
  huber = paste("the Huber loss is not a log-likelihood: it is a robustified",
                "squared-error loss whose changeover point is re-estimated as the fit",
                "proceeds. Compare fits by cross-validated risk (cvrisk)"),
  adaexp = paste("the exponential loss is not a log-likelihood. Use Binomial for a",
                 "classification model with one, or compare fits by cross-validated",
                 "risk (cvrisk)"),
  hingeloss = paste("the hinge loss is not a log-likelihood: it is a margin loss with",
                    "no probability model. Use Binomial for a classification model",
                    "with one, or compare fits by cross-validated risk (cvrisk)"),
  auc = paste("the smoothed 1 - AUC loss is a ranking objective, not a",
              "log-likelihood: it depends only on the order of the fitted values.",
              "Compare fits by cross-validated risk (cvrisk)"),
  gehan = paste("the Gehan rank loss is a ranking objective, not a log-likelihood.",
                "Use Weibull, Loglog or Lognormal for an AFT model with one, or CoxPH",
                "for a partial likelihood"),
  cindex = paste("the smoothed concordance loss is a ranking objective, not a",
                 "log-likelihood: it depends only on the order of the fitted values.",
                 "Use Weibull, Loglog or Lognormal for an AFT model with one, or",
                 "CoxPH for a partial likelihood"),
  multinomial = paste("the multinomial risk omits every observation in the reference",
                      "class, because its loss is summed over the K-1 non-reference",
                      "columns where such an observation is zero, so -risk is not the",
                      "multinomial log-likelihood")
)

# The log-likelihood at the fitted mstop. The Gaussian risk is the residual
# sum of squares, so the Gaussian log-likelihood is rebuilt with the MLE
# sigma^2 = RSS / n (-Inf for a perfect fit); for the other likelihood
# families the risk is the complete negative log-likelihood and the value is
# -risk.
.boost_loglik_value <- function(object) {
  fname <- object$family$name
  if (fname %in% names(.LOGLIK_UNAVAILABLE)) {
    stop("logLik is not available for family '", fname, "': ",
         .LOGLIK_UNAVAILABLE[[fname]], ". -risk(object) is the quantity the fit ",
         "minimised, negated.", call. = FALSE)
  }
  if (fname %in% c("gaussian", "gaussreg")) {
    n <- object$n
    rss <- sum((.get_response(object) - object$fitted)^2)
    sigma2 <- rss / n
    if (sigma2 <= 0) return(-Inf)
    return(-n / 2.0 * log(2 * pi * sigma2) - rss / (2 * sigma2))
  }
  -object$risk[object$mstop + 1]
}

#' Log-likelihood for a fitted glmboost model
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param ... Not used.
#' @return An object of class \code{logLik} with \code{df} and \code{nobs} attributes.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' logLik(mod)
#' @export
logLik.mboostrs_glmboost <- function(object, ...) {
  n <- object$n
  ms <- object$mstop
  val <- .boost_loglik_value(object)
  if (identical(val, -Inf)) return(-Inf)
  trace <- .hat_trace_path_glmboost(object)
  attr(val, "df") <- if (ms > 0) trace[ms] else 1.0
  attr(val, "nobs") <- n
  class(val) <- "logLik"
  val
}

#' Log-likelihood for a fitted gamboost model
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param ... Not used.
#' @return An object of class \code{logLik} with \code{df} and \code{nobs} attributes.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' logLik(mod)
#' @export
logLik.mboostrs_gamboost <- function(object, ...) {
  n <- object$n
  ms <- object$mstop
  ll <- .boost_loglik_value(object)
  if (identical(ll, -Inf)) return(-Inf)
  trace <- .hat_trace_path_gamboost(object)
  attr(ll, "df") <- if (ms > 0) trace[ms] else 1.0
  attr(ll, "nobs") <- n
  class(ll) <- "logLik"
  ll
}

# ─── summary ────────────────────────────────────────────────────────

#' Documented differences from expected fit behavior
#'
#' Some families and learners have behavior that differs from their usual
#' interpretation or from upstream packages. For example, a loss may not be a
#' likelihood, or a learner may use a different penalty than its upstream
#' counterpart.
#'
#' Each note describes the computation, its consequence for interpretation,
#' whether the difference is also present upstream, and supporting evidence.
#'
#' The notes are generated from a registry shared with the Python frontend.
#'
#' @param object A fitted mboostrs model.
#' @param include_global Include package-wide notes in addition to notes for
#'   this fit's family and learners.
#' @return A data frame with one row per note: \code{name},
#'   \code{mechanism}, \code{consequence}, \code{shared_with_upstream}, and
#'   \code{evidence}.
#' @examples
#' set.seed(1)
#' d <- data.frame(x = rnorm(50))
#' d$y <- d$x + rnorm(50, sd = 0.3)
#' fit <- glmboost(y ~ x, data = d, control = boost_control(mstop = 10))
#' parity_notes(fit)
#' @export
parity_notes <- function(object, include_global = FALSE) {
  owners <- .parity_note_owners(object)
  keep <- vapply(.parity_note_rows, function(row) {
    (identical(row$kind, "global") && isTRUE(include_global)) ||
      (row$kind %in% c("family", "learner") && row$owner %in% owners)
  }, logical(1))
  rows <- .parity_note_rows[keep]
  if (length(rows) == 0L) {
    return(data.frame(name = character(0), mechanism = character(0),
                      consequence = character(0),
                      shared_with_upstream = logical(0),
                      evidence = character(0),
                      stringsAsFactors = FALSE))
  }
  data.frame(
    name = vapply(rows, function(r) r$name, character(1)),
    mechanism = vapply(rows, function(r) r$mechanism, character(1)),
    consequence = vapply(rows, function(r) r$consequence, character(1)),
    shared_with_upstream = vapply(rows, function(r)
      identical(r$shared_with_upstream, "yes"), logical(1)),
    evidence = vapply(rows, function(r) r$evidence, character(1)),
    stringsAsFactors = FALSE
  )
}

# The family and learner names a fit's notes could be attached to. The family
# is stored by its Python class name; learner types are the spec `type` field,
# which is lower case and needs the class-name spelling the registry uses.
.parity_note_owners <- function(object) {
  family <- object$family$name %||% ""
  canonical <- vapply(.parity_note_rows, function(r) r$owner, character(1))
  family_owner <- canonical[tolower(canonical) == tolower(family)]
  learner_types <- vapply(object$learner_specs %||% list(),
                          function(s) s$type %||% "", character(1))
  # A spec's `type` is the engine's word for the learner, which is not always
  # the formula spelling the registry is keyed by: a `btree(x)` term arrives
  # as `"tree"`, so its note was silently unreachable from R.
  aliases <- c(tree = "btree", scalar = "bols", block = "bols")
  learner_types <- ifelse(learner_types %in% names(aliases),
                          aliases[learner_types], learner_types)
  learner_owners <- canonical[tolower(canonical) %in% tolower(learner_types)]
  unique(c(family_owner, learner_owners))
}

#' Summary for a fitted glmboost model
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param ... Not used.
#' @return Invisibly returns the model object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' summary(mod)
#' @export
summary.mboostrs_glmboost <- function(object, ...) {
  ms <- object$mstop
  names <- object$learner_names
  selected <- object$selected[1:ms]

  cat("GLMBoost Model\n")
  cat("  Family:", object$family$name, "\n")
  cat("  Formula:", deparse(object$formula), "\n")
  cat("  mstop:", ms, " nu:", object$nu, "\n")
  cat("  Offset:", .offset_label(object$offset), "\n")
  cat("  Risk at mstop:", formatC(object$risk[ms + 1], digits = 6), "\n")

  # Selection frequency
  freq <- table(factor(names[selected], levels = names))
  freq <- sort(freq, decreasing = TRUE)
  freq <- freq[freq > 0]
  cat("\n  Selection frequency:\n")
  for (nm in names(freq)) {
    cat(sprintf("    %s: %d/%d (%.1f%%)\n", nm, freq[nm], ms, 100 * freq[nm] / ms))
  }

  # Non-zero coefficients
  coefs <- coef(object)
  nonzero <- coefs[abs(coefs) > 1e-12]
  if (length(nonzero) > 0) {
    nonzero <- sort(abs(nonzero), decreasing = TRUE)
    cat("\n  Non-zero coefficients:\n")
    for (nm in names(nonzero)) {
      cat(sprintf("    %s: %.6f\n", nm, coefs[nm]))
    }
  }

  .cat_parity_pointer(object)
  invisible(object)
}

# The same first-contact line Python's `summary()` prints. Without it, a
# documented difference between a fit and what its name suggests was reachable
# from one frontend and not the other.
.cat_parity_pointer <- function(object) {
  notes <- tryCatch(parity_notes(object), error = function(e) NULL)
  if (is.null(notes) || nrow(notes) == 0L) return(invisible(NULL))
  cat(sprintf(
    "\n  This fit carries %d documented parity note(s): %s.\n  Call parity_notes() for what each one does and what follows from it.\n",
    nrow(notes), paste(notes$name, collapse = "; ")))
  invisible(NULL)
}

#' Summary for a fitted gamboost model
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param ... Not used.
#' @return Invisibly returns the model object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' summary(mod)
#' @export
summary.mboostrs_gamboost <- function(object, ...) {
  ms <- object$mstop
  names <- object$learner_names
  selected <- object$selected[1:ms]

  cat("GAMBoost Model\n")
  cat("  Family:", object$family$name, "\n")
  cat("  Formula:", deparse(object$formula), "\n")
  cat("  mstop:", ms, " nu:", object$nu, "\n")
  cat("  Offset:", .offset_label(object$offset), "\n")
  cat("  Number of base-learners:", length(names), "\n")
  cat("  Risk at mstop:", formatC(object$risk[ms + 1], digits = 6), "\n")

  # Selection frequency
  freq <- table(factor(names[selected], levels = names))
  freq <- sort(freq, decreasing = TRUE)
  freq <- freq[freq > 0]
  cat("\n  Selection frequency:\n")
  for (nm in names(freq)) {
    cat(sprintf("    %s: %d/%d (%.1f%%)\n", nm, freq[nm], ms, 100 * freq[nm] / ms))
  }

  .cat_parity_pointer(object)
  invisible(object)
}

# ─── varimp ─────────────────────────────────────────────────────────

#' Variable importance for glmboost
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param percent Logical. TRUE (default) divides each reduction by the sum of their absolute values, so the reductions are shares that sum to 1. FALSE returns the in-bag risk reduction per observation, which is what mboost's \code{varimp()} returns.
#' @param ... Not used.
#' @return A list with \code{reduction} (risk reduction per learner) and \code{selfreq} (selection frequency per learner).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' varimp(mod)
#' @export
varimp.mboostrs_glmboost <- function(object, percent = TRUE, ...) {
  risk <- object$risk
  if (length(risk) < 2) return(list(reduction = numeric(0), selfreq = numeric(0)))
  # Same-named learners sum, as the Python varimp does; the arithmetic is the
  # core's (varimp_attribution).
  names <- object$learner_names
  keys <- unique(names)
  selected <- object$selected[seq_len(object$mstop)]
  .varimp_core(-1.0 * diff(risk) / object$n, match(names[selected], keys) - 1L,
               keys, percent)
}

#' Variable importance for gamboost
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param percent Logical. TRUE (default) divides each reduction by the sum of their absolute values, so the reductions are shares that sum to 1. FALSE returns the in-bag risk reduction per observation, which is what mboost's \code{varimp()} returns.
#' @param ... Not used.
#' @return A list with \code{reduction} (risk reduction per learner) and \code{selfreq} (selection frequency per learner).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' varimp(mod)
#' @export
varimp.mboostrs_gamboost <- function(object, percent = TRUE, ...) {
  risk <- object$risk
  if (length(risk) < 2) return(list(reduction = numeric(0), selfreq = numeric(0)))
  if (isTRUE(object$family$name == "multinomial")) {
    # The in-bag risk is mboost's loss, which leaves the reference class out;
    # a step that lowers the full negative log-likelihood can raise it. Each
    # step is credited with its change in the full likelihood instead.
    risk <- .multinomial_nll_path(object)
  }
  # One entry per base-learner, as mboost reports; learners that share a
  # column are named "x (bols)", "x (bbs)" (.base_learner_names).
  keys <- .base_learner_names(object)
  selected <- object$selected[seq_len(object$mstop)]
  .varimp_core(-1.0 * diff(risk) / object$n, selected - 1L, keys, percent)
}

# Attribute each update's risk change to its learner through the core, and
# name the result. `groups` is the 0-based learner of each update.
.varimp_core <- function(changes, groups, keys, percent) {
  out <- .Call("wrap__varimp_attribution_r", as.double(changes),
               as.integer(groups), as.integer(length(keys)), isTRUE(percent))
  list(reduction = setNames(out$reduction, keys),
       selfreq = setNames(out$selfreq, keys))
}

# ─── extract ────────────────────────────────────────────────────────

#' Extract model components for glmboost
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param what What to extract: "design", "coefficients", "residuals",
#'   "variable.names", "offset", "nuisance", "weights", or "control".
#' @param which Learner names or positions (the intercept first), or NULL
#'   for selected learners, plus the intercept for a centered model.
#' @param asmatrix Keep a single design column as a matrix instead of a vector.
#' @param ... Not used.
#' @return The extracted component (type depends on \code{what}).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' extract(mod, what = "coefficients")
#' extract(mod, what = "offset")
#' @export
extract.mboostrs_glmboost <- function(object, what, which = NULL, asmatrix = FALSE, ...) {
  .refuse_unused("extract", ...)
  valid <- c("design", "coefficients", "residuals", "variable.names",
             "offset", "nuisance", "weights", "control")
  if (!(what %in% valid)) stop("what must be one of ", paste(valid, collapse = ", "))

  if (what == "coefficients") return(coef(object, which = which))
  if (what == "residuals") return(residuals(object))
  if (what == "offset") return(object$offset)
  if (what == "nuisance") return(nuisance(object))
  if (what == "control") return(object$control)
  if (what %in% c("design", "variable.names")) {
    names_vec <- object$learner_names
    if (is.null(which)) {
      idx <- sort(unique(object$selected[seq_len(object$mstop)]))
      centered <- object$center %||% any(object$centering_means != 0)
      if (isTRUE(object$intercept) && centered) idx <- sort(unique(c(1L, idx)))
    } else {
      idx <- if (is.character(which)) match(which, names_vec) else as.integer(which)
      if (anyNA(idx) || any(idx < 1L | idx > length(names_vec))) {
        stop("unknown learner in which", call. = FALSE)
      }
    }
    if (what == "variable.names") return(setNames(names_vec[idx], names_vec[idx]))
    x <- object$x_matrix
    values <- .Call("wrap__glmboost_design_r", as.double(t(x)),
                    as.integer(nrow(x)), as.integer(ncol(x)),
                    as.double(object$centering_means), isTRUE(object$intercept),
                    PACKAGE = "mboostrs")
    design <- matrix(values, nrow = object$n, ncol = length(names_vec), byrow = TRUE,
                     dimnames = list(rownames(x), names_vec))
    return(design[, idx, drop = !asmatrix])
  }
  if (what == "weights") return(object$weights %||% rep(1, object$n))
  NULL
}

#' Extract model components for gamboost
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param what What to extract.
#' @param which Which learners (by name or index), or NULL for all.
#' @param ... Not used.
#' @return The extracted component (type depends on \code{what}).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' extract(mod, what = "coefficients")
#' extract(mod, what = "selected")
#' @export
extract.mboostrs_gamboost <- function(object, what, which = NULL, ...) {
  .refuse_unused("extract", ...)
  valid <- c("design", "coefficients", "residuals", "offset", "weights",
             "penalty", "lambda", "df", "selected", "risk", "formula",
             "nuisance", "variable.names")
  if (!(what %in% valid)) stop("what must be one of ", paste(valid, collapse = ", "))

  if (what == "coefficients") return(coef(object, which = which))
  if (what == "residuals") return(residuals(object))
  if (what == "offset") return(object$offset)
  if (what == "weights") return(object$weights %||% rep(1, object$n))
  if (what == "nuisance") return(nuisance(object))
  if (what == "variable.names") {
    keys <- .base_learner_names(object)
    columns <- vapply(object$learner_specs, function(sp) {
      paste(c(sp$name, sp$name2), collapse = ", ")
    }, character(1))
    out <- setNames(columns, keys)
    return(if (is.null(which)) out else out[.which_learner_indices(object, which)])
  }
  if (what == "selected") return(selected(object, which = which))
  if (what == "risk") return(object$risk)
  if (what == "formula") return(object$formula)

  # design, penalty, lambda and df come from the learners the core prepared,
  # in mboost's layout: one entry per selected learner (or per `which`),
  # named as the learners are.
  if (is.null(object$rust_specs)) {
    stop("this model was fitted by an older mboostrs and keeps no learner ",
         "specification; refit it to extract ", what, call. = FALSE)
  }
  idx <- if (is.null(which)) sort(unique(unlist(object$selected[seq_len(object$mstop)])))
         else .which_learner_indices(object, which)
  info <- .Call("wrap__gamboost_learner_info_r", object$rust_specs,
                as.double(object$weights %||% rep(1, object$n)),
                as.integer(object$multinomial_n_classes %||% 0L),
                as.integer(idx), PACKAGE = "mboostrs")
  out <- lapply(info, function(item) {
    # A centered bols(x) is stored as the one column x - c; mboost's bols(x)
    # is the block [1, x] with an unpenalized ridge, and extract() reports
    # that form. The core stores lambda * K; mboost reports K, and the
    # identity for an unpenalized learner.
    centered <- is.finite(item$centering) && item$centering != 0
    switch(what,
      design = if (centered) cbind(1, item$design[, 1] + item$centering) else item$design,
      penalty = {
        pen <- item$penalty
        if (is.null(pen) || !is.finite(item$lambda)) pen
        else if (centered) diag(2)
        else if (item$lambda > 0) pen / item$lambda
        else diag(nrow(pen))
      },
      lambda = item$lambda,
      df = item$df)
  })
  names(out) <- .base_learner_names(object)[idx]
  out
}

# ─── coef_path ──────────────────────────────────────────────────────

.path_risk_values <- function(risk, index) {
  if (is.na(index) || index < 1L || index > length(risk)) return(c(NA_real_, NA_real_))
  current <- as.double(risk[index])
  previous <- if (index > 1L) as.double(risk[index - 1L]) else NA_real_
  if (!is.finite(current)) current <- NA_real_
  if (!is.finite(previous)) previous <- NA_real_
  c(current, if (is.na(current) || is.na(previous)) NA_real_ else previous - current)
}

.standard_path_frame <- function(object) {
  n_steps <- length(object$selected)
  rows <- lapply(seq_len(n_steps), function(step) {
    learner_index <- as.integer(object$selected[step])
    coefficient <- as.double(object$coefs[[step]])
    risk_values <- .path_risk_values(object$risk, step + 1L)
    nuisance <- if (!is.null(object$nuisance_vec_path) &&
                    length(object$nuisance_vec_path) >= step + 1L) {
      as.double(object$nuisance_vec_path[[step + 1L]])
    } else numeric(0)
    data.frame(
      iteration = step,
      cycle = NA_integer_,
      parameter_index = NA_integer_,
      parameter = NA_character_,
      learner_index = learner_index,
      learner = object$learner_names[learner_index],
      risk = risk_values[1],
      risk_change = risk_values[2],
      coefficient_norm = sqrt(sum(coefficient * coefficient)),
      scaled_coefficient_norm = object$nu * sqrt(sum(coefficient * coefficient)),
      nuisance = I(list(nuisance)),
      stringsAsFactors = FALSE
    )
  })
  out <- if (length(rows)) do.call(rbind, rows) else data.frame(
    iteration = integer(), cycle = integer(), parameter_index = integer(),
    parameter = character(),
    learner_index = integer(), learner = character(), risk = double(),
    risk_change = double(), coefficient_norm = double(),
    scaled_coefficient_norm = double(), nuisance = I(list())
  )
  rownames(out) <- NULL
  out
}

#' @rdname path_frame
#' @export
path_frame.mboostrs_glmboost <- function(object, ...) .standard_path_frame(object)

#' @rdname path_frame
#' @export
path_frame.mboostrs_gamboost <- function(object, ...) .standard_path_frame(object)

.lss_path_frame <- function(object) {
  if (is.null(object$selection_order)) {
    stop("path_frame() requires selection_order; refit this legacy LSS model")
  }
  order <- as.integer(object$selection_order)
  cursors <- integer(length(object$param_names))
  noncyclic <- identical(object$method, "noncyclic")
  rows <- vector("list", length(order))
  for (step in seq_along(order)) {
    parameter_index <- order[step]
    cursors[parameter_index] <- cursors[parameter_index] + 1L
    local_step <- cursors[parameter_index]
    cycle_ends <- step == length(order) || order[step + 1L] <= parameter_index
    risk_index <- if (noncyclic) step + 1L else if (cycle_ends) local_step + 1L else NA_integer_
    risk_values <- .path_risk_values(object$risk, risk_index)
    param <- object$params[[parameter_index]]
    learner_index <- as.integer(param$selected[local_step])
    if (inherits(object, "mboostrs_gamboostLSS")) {
      coefficient <- as.double(param$coefs[[local_step]])
      learner_names <- param$learner_names
      if (is.null(learner_names)) {
        # Legacy saved fit predating stored learner_names; fall back to the
        # bbs-only parser used before mixed-learner gamboostLSS support.
        learner_names <- vapply(.parse_bbs_formula(object$formulas[[parameter_index]]),
                                function(spec) spec$name, character(1))
      }
    } else {
      coefficient <- as.double(param$coefs[[local_step]])
      parsed <- .parse_formula(object$formulas[[parameter_index]], object$data,
                               need_response = FALSE)
      learner_names <- c("(Intercept)", parsed$learner_names)
    }
    coefficient_norm <- sqrt(sum(coefficient * coefficient))
    rows[[step]] <- data.frame(
      iteration = step,
      cycle = if (noncyclic) NA_integer_ else local_step,
      parameter_index = parameter_index,
      parameter = object$param_names[parameter_index],
      learner_index = learner_index,
      learner = learner_names[learner_index],
      risk = risk_values[1],
      risk_change = risk_values[2],
      coefficient_norm = coefficient_norm,
      scaled_coefficient_norm = object$nu * coefficient_norm,
      nuisance = I(list(numeric(0))),
      stringsAsFactors = FALSE
    )
  }
  out <- if (length(rows)) do.call(rbind, rows) else data.frame(
    iteration = integer(), cycle = integer(), parameter_index = integer(),
    parameter = character(),
    learner_index = integer(), learner = character(), risk = double(),
    risk_change = double(), coefficient_norm = double(),
    scaled_coefficient_norm = double(), nuisance = I(list())
  )
  rownames(out) <- NULL
  out
}

#' @rdname path_frame
#' @export
path_frame.mboostrs_glmboostLSS <- function(object, ...) .lss_path_frame(object)

#' @rdname path_frame
#' @export
path_frame.mboostrs_gamboostLSS <- function(object, ...) .lss_path_frame(object)

#' Coefficient paths for glmboost
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param which Which learners to include (by name or index), or NULL for all.
#' @param ... Not used.
#' @return A named list of coefficient paths (cumulative).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' cp <- coef_path(mod)
#' @export
coef_path.mboostrs_glmboost <- function(object, which = NULL, ...) {
  ms <- object$mstop
  names <- object$learner_names
  selected <- object$selected[1:ms]
  coefs <- object$coefs[1:ms]

  active <- if (is.null(which)) unique(names[selected]) else which
  paths <- setNames(lapply(active, function(nm) numeric(ms + 1)), active)
  for (i in seq_along(selected)) {
    nm <- names[selected[i]]
    if (nm %in% active) {
      paths[[nm]][i + 1] <- object$nu * coefs[i]
    }
  }
  for (nm in active) paths[[nm]] <- cumsum(paths[[nm]])
  paths
}

#' Coefficient paths for gamboost
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param which Which learners to include (by name or index), or NULL for all.
#' @param ... Not used.
#' @return A named list of coefficient paths (cumulative).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' cp <- coef_path(mod)
#' @export
coef_path.mboostrs_gamboost <- function(object, which = NULL, ...) {
  ms <- object$mstop
  names <- object$learner_names
  selected <- object$selected[1:ms]

  # For gamboost, coefs is a list of vectors per iteration
  # Aggregate into per-learner coefficient paths
  active <- if (is.null(which)) unique(names[selected]) else which

  # Each learner has a vector of coefficients (smooth = multiple cols)
  # Build per-learner cumulative coefficient matrices
  paths <- list()
  for (nm in active) {
    idx <- which(names == nm)
    if (length(idx) == 0) next
    # Determine coefficient dimension from first occurrence
    first_occ <- which(selected == idx)[1]
    if (is.na(first_occ)) next
    p_len <- length(object$coefs[[first_occ]])
    mat <- matrix(0.0, nrow = ms + 1, ncol = p_len)
    for (i in seq_along(selected)) {
      if (selected[i] == idx) {
        mat[i + 1, ] <- object$nu * object$coefs[[i]]
      }
    }
    paths[[nm]] <- apply(mat, 2, cumsum)
  }
  paths
}

# ─── selected(which=) ───────────────────────────────────────────────

#' Selected learners for glmboost
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param which Which learners to filter by (by name or index), or NULL for all.
#' @param ... Not used.
#' @return A character vector of selected learner names.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' selected(mod)
#' @export
selected.mboostrs_glmboost <- function(object, which = NULL, ...) {
  sel_names <- object$learner_names[object$selected[1:object$mstop]]
  if (!is.null(which)) {
    if (is.numeric(which)) {
      sel_names <- sel_names[sel_names %in% object$learner_names[which]]
    } else {
      sel_names <- sel_names[sel_names %in% which]
    }
  }
  sel_names
}

#' Selected learners for gamboost
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param which Which learners to filter by (by name or index), or NULL for all.
#' @param ... Not used.
#' @return A character vector of selected learner names.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' selected(mod)
#' @export
selected.mboostrs_gamboost <- function(object, which = NULL, ...) {
  sel_names <- object$learner_names[object$selected[1:object$mstop]]
  if (!is.null(which)) {
    if (is.numeric(which)) {
      sel_names <- sel_names[sel_names %in% object$learner_names[which]]
    } else {
      sel_names <- sel_names[sel_names %in% which]
    }
  }
  sel_names
}

# ─── fitted with mstop ─────────────────────────────────────────────

#' Fitted values at a specific mstop
#'
#' @param object A fitted model.
#' @param mstop Iteration to predict at. NULL = full model.
#' @param ... Not used.
#' @return A numeric vector of fitted values at the specified mstop.
#' @seealso [glmboost()], [predict.mboostrs_glmboost()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' head(fitted(mod))
#' @export
fitted.mboostrs_glmboost <- function(object, mstop = NULL, ...) {
  if (is.null(mstop)) return(object$fitted)
  predict(object, mstop = mstop, type = "link")
}

#' Fitted values for gamboost at a specific mstop
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param mstop Iteration to extract fitted values at. NULL = full model.
#' @param ... Not used.
#' @return A numeric vector of fitted values at the specified mstop.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' head(fitted(mod))
#' @export
fitted.mboostrs_gamboost <- function(object, mstop = NULL, ...) {
  if (is.null(mstop)) return(object$fitted)
  predict(object, mstop = mstop, type = "link")
}

# ─── update ─────────────────────────────────────────────────────────

#' Update a glmboost model (change mstop)
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param mstop New number of iterations (must be <= current mstop).
#' @param ... Not used.
#' @return An updated model object sliced to the new mstop.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' mod20 <- update(mod, mstop = 20)
#' @export
update.mboostrs_glmboost <- function(object, mstop = NULL, ...) {
  if (is.null(mstop)) return(object)
  if (mstop > object$mstop) stop("mstop cannot exceed the fitted mstop (", object$mstop, ")")
  object[mstop]
}

#' Update a gamboost model (change mstop)
#'
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param mstop New number of iterations (must be <= current mstop).
#' @param ... Not used.
#' @return An updated model object sliced to the new mstop.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' mod20 <- update(mod, mstop = 20)
#' @export
update.mboostrs_gamboost <- function(object, mstop = NULL, ...) {
  if (is.null(mstop)) return(object)
  if (mstop > object$mstop) stop("mstop cannot exceed the fitted mstop (", object$mstop, ")")
  object[mstop]
}

# ─── partial_effects (gamboost only) ────────────────────────────────

#' Partial effects for gamboost
#'
#' Computes the marginal effect of each smooth base-learner, holding
#' other variables at their training-data means.
#'
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param which Learner name(s). NULL = all.
#' @param n_points Number of evaluation points (default 100).
#' @param ... Not used.
#' @return A list of data frames, one per learner, with columns x and effect.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' pe <- partial_effects(mod, which = "x1")
#' head(pe[[1]])
#' @export
partial_effects.mboostrs_gamboost <- function(object, which = NULL, n_points = 100, ...) {
  specs <- object$learner_specs
  names <- object$learner_names
  # `object$selected` records the learner's position in the *unfiltered* list, so
  # the loop below has to keep that position. Renumbering the kept learners from
  # 1 made a filtered call aggregate whichever learner happened to sit at the new
  # index — a different learner's coefficients against this learner's design.
  keep <- seq_along(specs)
  if (!is.null(which)) keep <- .which_learner_indices(object, which)
  # Results are keyed by .base_learner_names(): two learners on one column
  # ("x (bols)", "x (bbs)") were both stored under "x", the second
  # overwriting the first.
  keys <- .base_learner_names(object)

  # A learner's coefficients have to match the design this function rebuilds. A
  # `center = TRUE` spline is the case where they do not: the spectral centering
  # transform drops the penalty's null space, so the fit carries `p - d`
  # coefficients while `wrap__r_bbs_design` returns all `p` basis columns, and R
  # has no helper that re-applies the transform on new x (the same gap that makes
  # `predict.mboostrs_gamboost` skip such a block). Accumulating them anyway
  # recycles the shorter vector and returns a curve unrelated to the fit —
  # measured on `bbs(x, df = 4, center = TRUE)`, a span of 0.29 against the
  # model's own 1.88. Refuse the learner instead, and say why.
  no_curve <- function(label, ncol) {
    warning(sprintf(
      paste("partial_effects: no effect curve for %s. Its coefficients do not",
            "match its %d-column design, which a centered learner does",
            "(center = TRUE removes the penalty's null space from the design,",
            "and R cannot yet rebuild that transform on new x). Refit with",
            "center = FALSE for the curve, or read the term through predict()."),
      label, ncol), call. = FALSE)
    NULL
  }
  steps <- as.integer(object$selected)
  coefs <- lapply(object$coefs, as.double)
  # The learner's contribution at the rows of `design` (n x p), computed by
  # the core: (design - centering) times the summed, nu-scaled coefficients.
  # NULL, with a warning, when the coefficients do not have p entries.
  effect_at <- function(design, centering, index, label) {
    p <- ncol(design)
    if (length(centering) != p || all(centering == 0)) centering <- numeric(0)
    effect <- .Call("wrap__r_learner_effect", as.double(t(design)), as.integer(p),
                    as.double(centering), steps, coefs, as.integer(index),
                    as.double(object$nu), PACKAGE = "mboostrs")
    if (is.null(effect)) no_curve(label, p) else effect
  }

  result <- list()
  for (i in keep) {
    sp <- specs[[i]]
    nm <- keys[i]

    if (sp$type %in% c("bbs", "bmono", "bns")) {
      x_raw <- sp$x
      x_seq <- seq(min(x_raw), max(x_raw), length.out = n_points)
      # The design at the new x points, with the training boundary knots.
      bd <- if (sp$type == "bns") {
        .Call("wrap__r_bns_design", as.double(x_seq), sp$bk_lo, sp$bk_hi,
              as.integer(sp$knots), as.integer(sp$degree), PACKAGE = "mboostrs")
      } else if (identical(sp$type, "bbs") && identical(sp$knot_placement, "quantile") &&
                 !is.null(sp$knots_vec)) {
        .Call("wrap__r_bbs_ospline_design", as.double(x_seq), as.double(sp$knots_vec),
              sp$bk_lo, sp$bk_hi, as.integer(sp$degree), as.integer(sp$deriv %||% 0),
              isTRUE(sp$center), PACKAGE = "mboostrs")
      } else {
        .Call("wrap__r_bbs_design", as.double(x_seq), sp$bk_lo, sp$bk_hi,
              as.integer(sp$knots), as.integer(sp$degree),
              as.integer(sp$deriv %||% 0), PACKAGE = "mboostrs")
      }
      design_mat <- matrix(bd$design, ncol = bd$p, byrow = TRUE)
      effect <- effect_at(design_mat, object$learner_centering[[i]], i, nm)
      if (is.null(effect)) next
      result[[nm]] <- data.frame(x = x_seq, effect = effect)
    } else if (sp$type == "scalar") {
      x_raw <- sp$design
      # The core centers at the weighted mean, as mboost's bols(x) does
      # through its own intercept, and the fit stores the constant it used.
      x_seq <- seq(min(x_raw), max(x_raw), length.out = n_points)
      effect <- effect_at(matrix(x_seq, ncol = 1), object$learner_centering[[i]][1], i, nm)
      if (is.null(effect)) next
      result[[nm]] <- data.frame(x = x_seq, effect = effect)
    } else if (sp$type %in% c("brandom", "bmrf")) {
      # One value per group: b_g - sum_j c_j b_j, with the column centering
      # c the core applied to the fit (nonzero for center = TRUE). When the
      # object stores a centering with one entry per group, that stored vector
      # is the one `predict(newdata =)` subtracts, so the effect uses it too;
      # otherwise the core rebuilds the learner from the fit's specification.
      stored <- object$learner_centering[[i]]
      if (length(stored) == sp$ncol && any(stored != 0)) {
        b <- numeric(sp$ncol)
        usable <- TRUE
        for (k in which(steps == i)) {
          if (length(coefs[[k]]) != sp$ncol) { usable <- FALSE; break }
          b <- b + as.double(object$nu) * coefs[[k]]
        }
        effect <- if (usable) {
          .Call("wrap__r_centered_group_effects", b, as.double(stored),
                PACKAGE = "mboostrs")
        } else {
          NULL
        }
      } else {
        effect <- .Call("wrap__gamboost_group_effect_r", object$rust_specs,
                        as.double(object$weights),
                        as.integer(object$multinomial_n_classes %||% 0L),
                        as.integer(i), as.integer(sp$ncol), steps, coefs,
                        as.double(object$nu), PACKAGE = "mboostrs")
      }
      if (is.null(effect)) {
        no_curve(nm, sp$ncol)
        next
      }
      result[[nm]] <- data.frame(x = seq_len(sp$ncol), effect = effect)
    }
  }
  result
}

# The full Multinomial negative log-likelihood on the training rows after each
# step 0..mstop, computed in the Rust core.
.multinomial_nll_path <- function(object) {
  K <- object$multinomial_n_classes
  built <- .multinomial_x1_designs(object, object$data)
  resp_var <- all.vars(object$formula[[2]])[1]
  codes <- match(as.character(object$data[[resp_var]]), object$multinomial_levels)
  indicator <- diag(K)[codes, -K, drop = FALSE]
  w <- object$weights %||% rep(1, length(codes))
  .Call("wrap__multinomial_staged_nll", as.double(object$offset[1]),
        as.integer(object$selected[seq_len(object$mstop)]),
        object$coefs[seq_len(object$mstop)], as.double(object$nu),
        built$designs, as.integer(built$ncol), as.integer(K - 1L),
        as.double(t(indicator)), as.double(w), PACKAGE = "mboostrs")
}

#' Joint support of new rows
#'
#' Whether each row of `newdata` lies where the training data has rows. For
#' the model's numeric predictors it gives the squared Mahalanobis distance
#' from the training rows (training mean, pseudo-inverse of the training
#' covariance) and the share of training rows at least that far; a row whose
#' share is below `threshold` is flagged. A row inside every predictor's range
#' can still be outside: a high `x1` with a low `x2` when the two move
#' together in the data. Categorical values the training data never had are
#' listed as `column=level`. The distance is linear in the predictors, so a
#' curved or clustered training cloud can leave gaps it does not see.
#' Computed by the core, as the Python method of the same name.
#'
#' @param object A fitted `mboostrs_gamboost` or `mboostrs_glmboost` model.
#' @param newdata Data frame of new rows.
#' @param threshold Share of training rows below which a row is flagged.
#' @return A data frame with one row per new observation: `row`,
#'   `distance_sq`, `training_share_beyond`, `outside_joint_support` and
#'   `unseen_levels`.
#' @examples
#' set.seed(1)
#' d <- data.frame(x1 = runif(100))
#' d$x2 <- d$x1 + rnorm(100, sd = 0.03)
#' d$y <- d$x1 + d$x2 + rnorm(100)
#' fit <- gamboost(y ~ bbs(x1) + bbs(x2), data = d)
#' support_report(fit, data.frame(x1 = c(0.5, 0.9), x2 = c(0.5, 0.1)))
#' @export
support_report <- function(object, newdata, threshold = 0.01) {
  if (!inherits(object, c("mboostrs_gamboost", "mboostrs_glmboost"))) {
    stop("support_report() takes a gamboost or glmboost model", call. = FALSE)
  }
  train <- object$data
  vars <- setdiff(all.vars(object$formula[[3L]]), all.vars(object$formula[[2L]]))
  vars <- intersect(intersect(vars, names(train)), names(newdata))
  is_cat <- vapply(vars, function(v) {
    is.factor(train[[v]]) || is.character(train[[v]]) || is.logical(train[[v]])
  }, logical(1))
  numeric_vars <- vars[!is_cat]
  categorical <- vars[is_cat]
  m <- nrow(newdata)
  distance <- rep(NA_real_, m)
  share <- rep(NA_real_, m)
  if (length(numeric_vars) && m > 0L) {
    xt <- as.matrix(as.data.frame(lapply(train[numeric_vars], as.double)))
    xt <- xt[stats::complete.cases(xt), , drop = FALSE]
    xn <- as.matrix(as.data.frame(lapply(newdata[numeric_vars], as.double)))
    res <- .Call("wrap__r_joint_support", as.double(xt), nrow(xt), as.double(xn),
                 nrow(xn), ncol(xt), PACKAGE = "mboostrs")
    distance <- res$distance_sq
    share <- res$training_share_beyond
  }
  unseen <- vapply(seq_len(m), function(i) {
    hits <- vapply(categorical, function(v) {
      value <- as.character(newdata[[v]][i])
      if (value %in% as.character(unique(train[[v]]))) "" else paste0(v, "=", value)
    }, character(1))
    paste(hits[nzchar(hits)], collapse = ", ")
  }, character(1))
  data.frame(row = seq_len(m), distance_sq = distance,
             training_share_beyond = share,
             outside_joint_support = share < threshold,
             unseen_levels = unseen, stringsAsFactors = FALSE)
}
