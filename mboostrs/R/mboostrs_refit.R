# REML refit of a boosted additive model.
#
# The numerics (term designs, sum-to-zero constraints, the REML fit and the
# effects at new rows) are in the Rust core (`mboostrs_core::gam`); this file
# parses the formula's terms into basis recipes and wraps the result.

.refit_families <- c(gaussian = "Gaussian", binomial = "Binomial",
                     poisson = "Poisson", nbinomial = "NBinomial",
                     gammasreg = "GammaReg", coxph = "CoxPH")

.refit_links <- c(Gaussian = "identity", Binomial = "logit", Poisson = "log",
                  NBinomial = "log", GammaReg = "log", CoxPH = "log")

# One formula term -> list(name, basis, vars, by, var_names, by_name,
# level_maps). `level_maps` records, for each brandom variable, the labels
# its integer codes stand for, so new data are coded the same way.
.refit_recipe <- function(label, data) {
  label <- trimws(label)
  # A bare numeric column (a glmboost-style linear term) is refitted as bols.
  if (grepl("^[A-Za-z_.][A-Za-z0-9_.]*$", label) ||
        (startsWith(label, "`") && grepl("^x+$", .mask_backticks(label)))) {
    return(.refit_recipe(paste0("bols(", label, ")"), data))
  }
  # A row-wise tensor product of two margins, as in the fit: each margin's
  # basis recipe, combined.
  if (grepl("%X%", label, fixed = TRUE)) {
    sides <- strsplit(label, "%X%", fixed = TRUE)[[1]]
    if (length(sides) != 2L) stop("gam_refit cannot refit the term ", label, call. = FALSE)
    left <- .refit_recipe(sides[1], data)
    right <- .refit_recipe(sides[2], data)
    if (!is.null(left$by_name) || !is.null(right$by_name)) {
      stop("gam_refit does not refit a by= column inside an %X% margin: ", label,
           call. = FALSE)
    }
    return(list(name = label,
                basis = list(kind = "tensor", left = left$basis, right = right$basis),
                vars = c(left$vars, right$vars), by = NULL,
                var_names = c(left$var_names, right$var_names), by_name = NULL,
                level_maps = c(left$level_maps, right$level_maps)))
  }
  m <- regmatches(label, regexec("^\\s*([A-Za-z_.]+)\\s*\\((.*)\\)\\s*$", label))[[1]]
  if (length(m) != 3L) {
    stop("gam_refit cannot refit the term ", label, call. = FALSE)
  }
  kind <- m[2]
  args <- .parse_bl_args(m[3])
  named <- args$named
  by_name <- named$by
  num <- function(name, default) as.double(named[[name]] %||% default)
  bbs_basis <- function(x, knots, degree, differences, boundary = NULL) {
    b <- if (is.null(boundary)) range(x) else boundary
    list(kind = "bbs", boundary = as.double(b), n_knots = knots,
         degree = degree, differences = differences)
  }
  column <- function(name) {
    if (is.null(data[[name]])) stop(.missing_column_message(name, data), call. = FALSE)
    data[[name]]
  }
  level_codes <- function(values, labels) {
    codes <- match(as.character(values), labels)
    codes[is.na(codes)] <- 0          # not a recorded level
    as.double(codes)
  }
  level_maps <- list()
  if (kind == "bbs") {
    var <- args$positional[1]
    x <- as.double(column(var))
    bk <- named$boundary.knots
    boundary <- if (is.null(bk)) NULL else as.double(eval(parse(text = bk)))
    basis <- bbs_basis(x, num("knots", 20), num("degree", 3), num("differences", 2), boundary)
    vars <- list(x)
    var_names <- var
  } else if (kind == "bols") {
    var <- args$positional[1]
    x <- column(var)
    if (!is.numeric(x)) {
      stop("gam_refit refits numeric bols terms; ", label, " is a factor",
           call. = FALSE)
    }
    intercept <- as.logical(named$intercept %||% TRUE)
    basis <- list(kind = "bols", intercept = intercept)
    vars <- list(as.double(x))
    var_names <- var
  } else if (kind == "brandom") {
    var <- args$positional[1]
    g <- column(var)
    labels <- sort(unique(as.character(g)))
    if (is.factor(g)) labels <- levels(droplevels(g))
    basis <- list(kind = "brandom", levels = as.double(seq_along(labels)))
    vars <- list(level_codes(g, labels))
    var_names <- var
    level_maps[[var]] <- labels
  } else if (kind == "bspatial") {
    v1 <- args$positional[1]
    v2 <- args$positional[2]
    x1 <- as.double(column(v1))
    x2 <- as.double(column(v2))
    knots <- num("knots", 10)
    degree <- num("degree", 3)
    differences <- num("differences", 2)
    basis <- list(kind = "tensor",
                  left = bbs_basis(x1, knots, degree, differences),
                  right = bbs_basis(x2, knots, degree, differences))
    vars <- list(x1, x2)
    var_names <- c(v1, v2)
  } else {
    stop("gam_refit supports bbs, bols, brandom and bspatial terms; cannot refit ",
         label, call. = FALSE)
  }
  by <- if (is.null(by_name)) NULL else as.double(column(by_name))
  list(name = label, basis = basis, vars = vars, by = by,
       var_names = var_names, by_name = by_name, level_maps = level_maps)
}

# The recipe's variables at new rows.
.refit_vars <- function(recipe, newdata) {
  lapply(recipe$var_names, function(v) {
    if (is.null(newdata[[v]])) stop(.missing_column_message(v, newdata), call. = FALSE)
    labels <- recipe$level_maps[[v]]
    if (is.null(labels)) return(as.double(newdata[[v]]))
    codes <- match(as.character(newdata[[v]]), labels)
    codes[is.na(codes)] <- 0
    as.double(codes)
  })
}

#' Refit a boosted additive model with REML smoothing parameters
#'
#' Refits the chosen terms of a \code{gamboost} model once, as a penalized
#' GAM on the terms' own B-spline, linear or random-effect bases. Ordinary
#' smooths are constrained to sum to zero over the weighted fitting rows;
#' nonconstant numeric by terms retain their constant-in-x direction. Each
#' penalty gets a smoothing parameter estimated by restricted maximum
#' likelihood (REML), and intervals come from the Bayesian covariance of the coefficients. It
#' is the construction of \code{mgcv::gam(method = "REML")}.
#'
#' Boosting stopped early shrinks every effect towards zero, and a bootstrap
#' of the boosted fit reproduces that shrinkage, so its bands under-cover.
#' The refit estimates the selected terms without the shrinkage. Its
#' intervals are conditional on the chosen terms: when they were selected on
#' the same data, the intervals do not account for the selection.
#'
#' Non-Gaussian families are fitted by penalized iteratively reweighted least
#' squares, and their smoothing parameters minimize the Laplace-approximate
#' REML criterion (the negative binomial theta and the gamma shape are
#' estimated by the same criterion). Effects are on the
#' canonical link scale: log-odds for Binomial, the log of the mean for
#' Poisson, NBinomial (dispersion taken from the boosted fit) and GammaReg,
#' and the log hazard ratio for CoxPH (fitted through the Poisson form of
#' the Cox likelihood with a Breslow baseline hazard).
#'
#' @param model A \code{gamboost} fit with the Gaussian, Binomial, Poisson,
#'   NBinomial, GammaReg or CoxPH family.
#' @param terms Character vector of formula terms to refit. By default the
#'   terms whose learners were selected (\code{which = "selected"}) or all of
#'   them (\code{which = "all"}).
#' @param which \code{"selected"} or \code{"all"}.
#' @return An object of class \code{"mboostrs_gam_refit"} with components
#'   \code{edf} (effective degrees of freedom per term), \code{smoothing}
#'   (smoothing parameters per term, one per penalty), \code{sigma}, \code{link},
#'   \code{reml_score}, \code{convergence}, \code{conditional_on_selection} and,
#'   for NBinomial, \code{theta} (estimated by the same criterion, starting
#'   from the boosted fit's value).
#'   Use \code{effects()} for each term's effect with a pointwise interval and
#'   \code{predict()} for predictions.
#' @examples
#' set.seed(1)
#' n <- 300
#' d <- data.frame(x1 = runif(n, -2, 2), x2 = runif(n, -2, 2))
#' d$y <- 0.5 * d$x1 + sin(1.5 * d$x2) + rnorm(n, sd = 0.5)
#' fit <- gamboost(y ~ bbs(x1) + bbs(x2), data = d)
#' refit <- gam_refit(fit)
#' refit$edf
#' str(effects(refit, data.frame(x1 = 0, x2 = c(-1, 0, 1)))[["bbs(x2)"]])
#' @export
gam_refit <- function(model, terms = NULL, which = c("selected", "all")) {
  which <- match.arg(which)
  if (!inherits(model, "mboostrs_gamboost")) {
    stop("gam_refit works on gamboost models", call. = FALSE)
  }
  family <- .refit_families[model$family$name]
  if (is.na(family)) {
    stop("gam_refit supports Gaussian, Binomial, Poisson, NBinomial, GammaReg ",
         "and CoxPH; this model is ", model$family$name, call. = FALSE)
  }
  family <- unname(family)
  data <- model$data
  labels <- attr(stats::terms(model$formula, data = data), "term.labels")
  # Refitting every term of the formula is refitting a prespecified model; any
  # other set was chosen, here or by the caller, from these data.
  conditional <- !(is.null(terms) && which == "all")
  if (is.null(terms)) {
    used <- if (which == "all") seq_along(labels)
            else sort(unique(model$selected[seq_len(model$mstop)]))
    terms <- labels[used]
  }
  if (length(terms) == 0L) stop("no terms to refit", call. = FALSE)
  recipes <- lapply(terms, .refit_recipe, data = data)

  lhs <- model$formula[[2L]]
  time <- numeric(0)
  if (family == "CoxPH") {
    if (!is.call(lhs) || !identical(as.character(lhs[[1L]]), "Surv")) {
      stop("a CoxPH refit needs a Surv(time, status) response", call. = FALSE)
    }
    time <- as.double(eval(lhs[[2L]], data, environment(model$formula)))
    y <- as.double(eval(lhs[[3L]], data, environment(model$formula)))
  } else {
    y <- as.double(eval(lhs, data, environment(model$formula)))
  }
  theta <- 1
  if (family == "NBinomial") {
    # The dispersion at the model's step count, as mgcv's nb() would
    # estimate it; the refit uses it as known. The path is aligned with the
    # risk (entry 1 is the offset-only model).
    path <- model$nuisance_vec_path
    theta <- if (length(path)) path[[min(length(path), model$mstop + 1L)]]
             else model$nuisance
    theta <- as.double(theta)
    if (length(theta) != 1L || !is.finite(theta)) {
      stop("the NBinomial model carries no dispersion estimate", call. = FALSE)
    }
  }
  w <- model$weights %||% rep(1, length(y))
  core_terms <- lapply(recipes, function(r) r[c("name", "basis", "vars", "by")])
  fit <- .Call("wrap__r_gam_refit", core_terms, y, as.double(w), numeric(0),
               family, theta, time, PACKAGE = "mboostrs")
  edf <- vapply(fit$blocks, function(b) b$edf, numeric(1))
  names(edf) <- terms
  smoothing <- lapply(fit$blocks, function(b) b$lambda)
  names(smoothing) <- terms
  structure(
    list(
      terms = terms,
      edf = edf,
      smoothing = smoothing,
      sigma = sqrt(fit$scale),
      theta = if (family == "NBinomial") fit$theta else NULL,
      link = fit$link,
      family = family,
      reml_score = fit$reml_score,
      loglik = fit$loglik,
      df = fit$df,
      convergence = fit[c("converged", "newton_iterations", "pirls_iterations",
                          "gradient_norm", "method")],
      conditional_on_selection = conditional,
      residual_df = if (is.nan(fit$residual_df)) NULL else fit$residual_df,
      coefficients = fit$coef,
      vcov = fit$cov,
      vcov_corrected = fit$cov_corrected,
      blocks = fit$blocks,
      recipes = recipes,
      data = data
    ),
    class = "mboostrs_gam_refit"
  )
}

#' Effects of a REML refit's terms
#'
#' @param object A \code{mboostrs_gam_refit}.
#' @param newdata Data frame of rows at which to evaluate the effects;
#'   defaults to the fitting data.
#' @param level Coverage of the pointwise intervals.
#' @param unseen_levels \code{"error"} or \code{"zero_effect"}: a
#'   \code{brandom} group the refit never saw gets zero effect, the average
#'   over groups, which is the prediction for a new group that lme4
#'   (\code{allow.new.levels}) and mgcv (\code{exclude=}) make. Its interval
#'   is that of the average; a new group's own effect varies around it by
#'   the random-effect standard deviation, which the interval leaves out.
#' @param corrected If \code{TRUE}, the standard errors use the covariance
#'   corrected, to first order, for the uncertainty of the estimated
#'   smoothing parameters (mgcv's \code{unconditional = TRUE} without its
#'   second-order term); by default they are conditional on the smoothing
#'   parameters.
#' @param ... Not used.
#' @return A named list with one data frame per term: \code{estimate},
#'   \code{se}, \code{lower} and \code{upper}. Ordinary smooth effects are
#'   centred over the weighted fitting rows. Numeric by effects need not
#'   have mean zero.
#' @export
effects.mboostrs_gam_refit <- function(object, newdata = NULL, level = 0.9,
                                       unseen_levels = c("error", "zero_effect"),
                                       corrected = FALSE, ...) {
  unseen_levels <- match.arg(unseen_levels)
  if (is.null(newdata)) newdata <- object$data
  z <- stats::qnorm(0.5 + level / 2)
  V <- if (isTRUE(corrected)) object$vcov_corrected else object$vcov
  out <- lapply(seq_along(object$recipes), function(i) {
    recipe <- object$recipes[[i]]
    block <- object$blocks[[i]]
    cols <- (block$start + 1L):block$end
    by <- if (is.null(recipe$by_name)) NULL else as.double(newdata[[recipe$by_name]])
    r <- .Call("wrap__r_gam_refit_effect", recipe$basis, .refit_vars(recipe, newdata),
               block$householder, object$coefficients[cols],
               V[cols, cols, drop = FALSE], block$has_by, by,
               unseen_levels == "zero_effect", PACKAGE = "mboostrs")
    data.frame(estimate = r$estimate, se = r$se,
               lower = r$estimate - z * r$se, upper = r$estimate + z * r$se)
  })
  names(out) <- object$terms
  out
}

#' Predictions from a REML refit
#'
#' @param object A \code{mboostrs_gam_refit}.
#' @param newdata Data frame; defaults to the fitting data.
#' @param type \code{"link"} or \code{"response"} (the mean, or the relative
#'   hazard for CoxPH).
#' @param unseen_levels As in \code{effects.mboostrs_gam_refit}.
#' @param ... Not used.
#' @return A numeric vector.
#' @export
predict.mboostrs_gam_refit <- function(object, newdata = NULL, type = c("link", "response"),
                                       unseen_levels = c("error", "zero_effect"), ...) {
  type <- match.arg(type)
  unseen_levels <- match.arg(unseen_levels)
  eff <- effects.mboostrs_gam_refit(object, newdata, unseen_levels = unseen_levels)
  eta <- object$coefficients[1] + Reduce(`+`, lapply(eff, `[[`, "estimate"))
  if (type == "link") return(eta)
  switch(object$link, identity = eta, logit = stats::plogis(eta), log = exp(eta))
}

#' Log-likelihood of a REML refit
#'
#' The log-likelihood of the refitted model as mgcv's \code{logLik.gam}
#' computes it (Gaussian with the maximum-likelihood variance, Cox the
#' partial likelihood), with the effective degrees of freedom plus one for
#' an estimated scale, theta or gamma shape as its \code{df}; \code{AIC()}
#' then gives the conditional AIC. mgcv's \code{AIC(gam)} counts the degrees
#' of freedom corrected for smoothing-parameter uncertainty (its
#' \code{edf2}), which are larger, so its AIC is somewhat larger.
#'
#' @param object A \code{mboostrs_gam_refit}.
#' @param ... Not used.
#' @return An object of class \code{"logLik"}.
#' @export
logLik.mboostrs_gam_refit <- function(object, ...) {
  structure(object$loglik, df = object$df, nobs = nrow(as.data.frame(object$data)),
            class = "logLik")
}

#' Plot a REML refit's effects
#'
#' Draws each term's effect with its pointwise interval over the range of
#' its variable in the fitting data, with the other variables at their
#' medians. Only one-variable terms are drawn.
#'
#' @param x A \code{mboostrs_gam_refit}.
#' @param level Coverage of the intervals.
#' @param corrected Use the covariance corrected for smoothing-parameter
#'   uncertainty.
#' @param n_points Number of grid points per term.
#' @param ... Passed to \code{plot}.
#' @return Invisibly, the list of effect data frames drawn.
#' @export
plot.mboostrs_gam_refit <- function(x, level = 0.9, corrected = FALSE, n_points = 100, ...) {
  data <- as.data.frame(x$data)
  typical <- lapply(data, function(v) {
    if (is.numeric(v)) rep(stats::median(v), n_points) else rep(v[1], n_points)
  })
  drawn <- list()
  for (i in seq_along(x$recipes)) {
    recipe <- x$recipes[[i]]
    if (length(recipe$var_names) != 1L || length(recipe$level_maps)) next
    v <- recipe$var_names
    grid <- as.data.frame(typical, stringsAsFactors = FALSE)
    grid[[v]] <- seq(min(data[[v]]), max(data[[v]]), length.out = n_points)
    if (!is.null(recipe$by_name)) grid[[recipe$by_name]] <- 1
    e <- effects.mboostrs_gam_refit(x, grid, level = level, corrected = corrected)[[i]]
    plot(grid[[v]], e$estimate, type = "l", ylim = range(e$lower, e$upper),
         xlab = v, ylab = paste("effect of", x$terms[i]), ...)
    graphics::lines(grid[[v]], e$lower, lty = 2)
    graphics::lines(grid[[v]], e$upper, lty = 2)
    drawn[[x$terms[i]]] <- cbind(grid[v], e)
  }
  invisible(drawn)
}

#' @export
coef.mboostrs_gam_refit <- function(object, ...) object$coefficients

#' Covariance of a REML refit's coefficients
#'
#' @param object A \code{mboostrs_gam_refit}.
#' @param type \code{"bayesian"} (conditional on the smoothing parameters)
#'   or \code{"corrected"} (with the first-order correction for their
#'   uncertainty).
#' @param ... Not used.
#' @return The covariance matrix of \code{coef(object)}.
#' @export
vcov.mboostrs_gam_refit <- function(object, type = c("bayesian", "corrected"), ...) {
  type <- match.arg(type)
  if (type == "corrected") object$vcov_corrected else object$vcov
}

#' @export
print.mboostrs_gam_refit <- function(x, ...) {
  cat("REML refit (", x$family, ", ", x$link, " link) of ", length(x$terms),
      " term(s)\n", sep = "")
  print(summary(x)$terms, row.names = FALSE)
  if (x$link == "identity" || x$family == "GammaReg") {
    cat("scale: ", format(x$sigma^2, digits = 4), "\n", sep = "")
  }
  if (isTRUE(x$conditional_on_selection)) {
    cat("Intervals are conditional on the selected terms.\n")
  }
  if (!isTRUE(x$convergence$converged)) cat("Warning: the refit did not converge.\n")
  invisible(x)
}

#' @export
summary.mboostrs_gam_refit <- function(object, ...) {
  lambda <- vapply(object$smoothing, function(l) {
    if (length(l) == 0L) "" else paste(format(l, digits = 4), collapse = ", ")
  }, character(1))
  # Wood's (2013) approximate test of each term, as mgcv's summary.gam gives
  # it: F when the scale is estimated, chi-square otherwise.
  f_test <- !is.null(object$residual_df)
  ref_df <- vapply(object$blocks, function(b) b$ref_df %||% NA_real_, numeric(1))
  stat <- vapply(object$blocks, function(b) b$statistic %||% NA_real_, numeric(1))
  p_value <- vapply(object$blocks, function(b) b$p_value %||% NA_real_, numeric(1))
  terms <- data.frame(term = object$terms, edf = round(object$edf, 3),
                      ref_df = round(ref_df, 3), stringsAsFactors = FALSE)
  terms[[if (f_test) "F" else "chi_sq"]] <- round(if (f_test) stat / ref_df else stat, 3)
  terms$p_value <- signif(p_value, 4)
  terms$lambda <- lambda
  structure(list(
    terms = terms, sigma = object$sigma, reml_score = object$reml_score,
    convergence = object$convergence,
    conditional_on_selection = object$conditional_on_selection
  ), class = "summary.mboostrs_gam_refit")
}

#' @export
print.summary.mboostrs_gam_refit <- function(x, ...) {
  print(x$terms, row.names = FALSE)
  if (isTRUE(x$conditional_on_selection)) {
    cat("p-values are conditional on terms selected from the same data.\n")
  }
  invisible(x)
}

#' Refit a distributional model with REML smoothing parameters
#'
#' Refits the terms of every distribution parameter of a \code{gamboostLSS}
#' model jointly, as one penalized likelihood: each term keeps its basis;
#' ordinary smooths have a weighted sum-to-zero constraint, while nonconstant
#' numeric by terms retain their constant-in-x direction. All coefficients are
#' estimated by penalized maximum likelihood, and the smoothing parameters
#' minimize the Laplace-approximate restricted likelihood (Wood, Pya and
#' Saefken 2016), as \code{mgcv::gam(list(...), family = gaulss())} does for
#' the Gaussian location-scale model.
#'
#' @param model A \code{gamboostLSS} or \code{glmboostLSS} fit (linear terms
#'   of a \code{glmboostLSS} formula are refitted as unpenalized \code{bols}).
#' @param terms Optional named list of character vectors: the terms to refit
#'   for each distribution parameter. By default the terms whose learners were
#'   selected (\code{which = "selected"}) or all of them (\code{which = "all"}).
#'   A parameter with no terms keeps an intercept.
#' @param which \code{"selected"} or \code{"all"}.
#' @return An object of class \code{"mboostrs_gamlss_refit"} with components
#'   \code{edf} and \code{smoothing} (per parameter, per term),
#'   \code{reml_score}, \code{convergence} and
#'   \code{conditional_on_selection}. Use \code{effects()} and
#'   \code{predict()}; \code{vcov(type = "corrected")} adds the first-order
#'   uncertainty of the smoothing parameters.
#' @examples
#' set.seed(1)
#' n <- 400
#' d <- data.frame(x1 = runif(n), x2 = runif(n))
#' d$y <- sin(6 * d$x1) + exp(-0.7 + d$x2) * rnorm(n)
#' fit <- gamboostLSS(list(mu = y ~ bbs(x1), sigma = y ~ bbs(x2)), data = d,
#'                    families = GaussianLSS())
#' refit <- gamlss_refit(fit)
#' refit$edf
#' head(predict(refit, d, type = "response")$sigma)
#' @export
gamlss_refit <- function(model, terms = NULL, which = c("selected", "all")) {
  which <- match.arg(which)
  if (!inherits(model, c("mboostrs_gamboostLSS", "mboostrs_glmboostLSS"))) {
    stop("gamlss_refit works on gamboostLSS and glmboostLSS models", call. = FALSE)
  }
  linear <- inherits(model, "mboostrs_glmboostLSS")
  pnames <- model$param_names
  data <- model$data
  conditional <- !(is.null(terms) && which == "all")
  chosen <- lapply(pnames, function(p) {
    labels <- attr(stats::terms(model$formulas[[p]], data = data), "term.labels")
    if (!is.null(terms)) return(as.character(terms[[p]] %||% character(0)))
    if (which == "all") return(labels)
    if (linear) {
      # glmboostLSS selects model-matrix columns; a numeric term's column
      # carries the term's own name.
      columns <- colnames(stats::model.matrix(model$formulas[[p]], data = data))
      used <- columns[sort(unique(model$params[[p]]$selected))]
      return(labels[labels %in% used])
    }
    learners <- model$params[[p]]$learner_names
    if (length(learners) != length(labels)) {
      stop("the ", p, " formula's terms do not map one to one onto its learners; ",
           "pass terms= explicitly", call. = FALSE)
    }
    labels[sort(unique(model$params[[p]]$selected))]
  })
  names(chosen) <- pnames
  recipes <- lapply(chosen, function(ts) lapply(ts, .refit_recipe, data = data))
  lhs <- model$formulas[[1]][[2L]]
  y <- as.double(eval(lhs, data, environment(model$formulas[[1]])))
  w <- model$weights %||% rep(1, length(y))
  start <- predict(model, type = "link")
  core_terms <- lapply(recipes, function(rs) lapply(rs, function(r) r[c("name", "basis", "vars", "by")]))
  fit <- .Call("wrap__r_gamlss_refit", unname(core_terms), pnames,
               model$family$rust_name, y, as.double(w),
               unname(lapply(pnames, function(p) as.double(start[[p]]))),
               PACKAGE = "mboostrs")
  edf <- lapply(fit$params, function(q) {
    v <- vapply(q$blocks, function(b) b$edf, numeric(1))
    names(v) <- vapply(q$blocks, function(b) b$name, character(1))
    v
  })
  smoothing <- lapply(fit$params, function(q) {
    v <- lapply(q$blocks, function(b) b$lambda)
    names(v) <- vapply(q$blocks, function(b) b$name, character(1))
    v
  })
  names(edf) <- names(smoothing) <- pnames
  structure(
    list(terms = chosen, edf = edf, smoothing = smoothing,
         family = model$family$name, rust_family = model$family$rust_name,
         param_names = pnames, reml_score = fit$reml_score,
         convergence = fit[c("converged", "iterations", "gradient_norm",
                             "information_floored")],
         conditional_on_selection = conditional,
         coefficients = fit$coef, vcov = fit$cov, vcov_corrected = fit$cov_corrected,
         params = fit$params, recipes = recipes, data = data, state = fit$state),
    class = "mboostrs_gamlss_refit"
  )
}

#' Effects of a GAMLSS refit's terms
#'
#' @param object A \code{mboostrs_gamlss_refit}.
#' @param newdata Data frame; defaults to the fitting data.
#' @param parameter Optional parameter name; by default all parameters.
#' @param level Coverage of the pointwise intervals.
#' @param unseen_levels As in \code{effects.mboostrs_gam_refit}.
#' @param corrected If \code{TRUE}, standard errors include the first-order
#'   uncertainty of the smoothing parameters.
#' @param ... Not used.
#' @return A named list per parameter of named lists per term of data frames
#'   with \code{estimate}, \code{se}, \code{lower} and \code{upper}, on the
#'   parameter's link scale; with \code{parameter}, that parameter's list.
#' @export
effects.mboostrs_gamlss_refit <- function(object, newdata = NULL, parameter = NULL,
                                          level = 0.9,
                                          unseen_levels = c("error", "zero_effect"),
                                          corrected = FALSE, ...) {
  unseen_levels <- match.arg(unseen_levels)
  if (is.null(newdata)) newdata <- object$data
  z <- stats::qnorm(0.5 + level / 2)
  V <- if (isTRUE(corrected)) object$vcov_corrected else object$vcov
  want <- if (is.null(parameter)) object$param_names else parameter
  bad <- setdiff(want, object$param_names)
  if (length(bad)) stop("unknown parameter: ", paste(bad, collapse = ", "), call. = FALSE)
  out <- lapply(want, function(p) {
    q <- match(p, object$param_names)
    res <- lapply(seq_along(object$recipes[[p]]), function(i) {
      recipe <- object$recipes[[p]][[i]]
      block <- object$params[[q]]$blocks[[i]]
      cols <- (block$start + 1L):block$end
      by <- if (is.null(recipe$by_name)) NULL else as.double(newdata[[recipe$by_name]])
      r <- .Call("wrap__r_gam_refit_effect", recipe$basis, .refit_vars(recipe, newdata),
                 block$householder, object$coefficients[cols],
                 V[cols, cols, drop = FALSE], block$has_by, by,
                 unseen_levels == "zero_effect", PACKAGE = "mboostrs")
      data.frame(estimate = r$estimate, se = r$se,
                 lower = r$estimate - z * r$se, upper = r$estimate + z * r$se)
    })
    names(res) <- object$terms[[p]]
    res
  })
  names(out) <- want
  if (is.null(parameter) || length(parameter) > 1L) out else out[[1]]
}

#' Predictions from a GAMLSS refit
#'
#' @param object A \code{mboostrs_gamlss_refit}.
#' @param newdata Data frame; defaults to the fitting data.
#' @param parameter Optional parameter name.
#' @param type \code{"link"} or \code{"response"} (the parameter on its
#'   natural scale, through the family's inverse link).
#' @param unseen_levels As in \code{effects.mboostrs_gam_refit}.
#' @param ... Not used.
#' @return A named list of numeric vectors, one per parameter; with
#'   \code{parameter}, that parameter's vector.
#' @export
predict.mboostrs_gamlss_refit <- function(object, newdata = NULL, parameter = NULL,
                                          type = c("link", "response"),
                                          unseen_levels = c("error", "zero_effect"), ...) {
  type <- match.arg(type)
  unseen_levels <- match.arg(unseen_levels)
  if (is.null(newdata)) newdata <- object$data
  want <- if (is.null(parameter)) object$param_names else parameter
  n <- nrow(as.data.frame(newdata))
  K <- length(object$param_names)
  out <- lapply(want, function(p) {
    q <- match(p, object$param_names)
    eta <- rep(object$coefficients[object$params[[q]]$start + 1L], n)
    eff <- effects.mboostrs_gamlss_refit(object, newdata, parameter = p,
                                         unseen_levels = unseen_levels)
    for (e in eff) eta <- eta + e$estimate
    if (type == "link") eta
    else .Call("wrap__r_lss_response", object$rust_family, K, q, eta, PACKAGE = "mboostrs")
  })
  names(out) <- want
  if (is.null(parameter)) out else out[[1]]
}

#' @export
coef.mboostrs_gamlss_refit <- function(object, ...) object$coefficients

#' @export
vcov.mboostrs_gamlss_refit <- function(object, type = c("bayesian", "corrected"), ...) {
  type <- match.arg(type)
  if (type == "corrected") object$vcov_corrected else object$vcov
}

#' @export
print.mboostrs_gamlss_refit <- function(x, ...) {
  cat("GAMLSS REML refit (", x$family, ")\n", sep = "")
  for (p in x$param_names) {
    cat("  ", p, ": ", sep = "")
    if (length(x$edf[[p]]) == 0L) {
      cat("intercept only\n")
    } else {
      cat(paste0(names(x$edf[[p]]), " (edf ", format(x$edf[[p]], digits = 3), ")",
                 collapse = ", "), "\n")
    }
  }
  if (isTRUE(x$conditional_on_selection)) {
    cat("Intervals are conditional on the selected terms.\n")
  }
  if (!isTRUE(x$convergence$converged)) cat("Warning: the refit did not converge.\n")
  invisible(x)
}

#' Posterior draws of a GAMLSS refit's distribution parameters
#'
#' Draws every distribution parameter, on its natural scale, at the rows of
#' \code{newdata} from the approximate posterior of the joint coefficients.
#' A quantity that combines parameters, such as a centile of the response
#' distribution, is computed draw by draw, and its interval taken from the
#' quantiles over the draws; the draws keep the correlation between the
#' parameters' estimates.
#'
#' @param object A \code{mboostrs_gamlss_refit}.
#' @param newdata Data frame of rows at which to draw.
#' @param n_draws Number of draws.
#' @param seed Random seed.
#' @param corrected If \code{TRUE}, draw from the covariance corrected for
#'   smoothing-parameter uncertainty.
#' @param unseen_levels As in \code{effects.mboostrs_gam_refit}.
#' @return A named list of \code{n_draws} by \code{nrow(newdata)} matrices,
#'   one per parameter.
#' @examples
#' set.seed(1)
#' n <- 400
#' d <- data.frame(x = runif(n))
#' d$y <- sin(6 * d$x) + exp(-0.7 + d$x) * rnorm(n)
#' fit <- gamboostLSS(list(mu = y ~ bbs(x), sigma = y ~ bbs(x)), data = d,
#'                    families = GaussianLSS())
#' draws <- gamlss_draws(gamlss_refit(fit), data.frame(x = c(0.2, 0.8)), 500)
#' q90 <- draws$mu + qnorm(0.9) * draws$sigma
#' apply(q90, 2, quantile, c(0.05, 0.95))
#' @export
gamlss_draws <- function(object, newdata, n_draws = 1000, seed = 0, corrected = FALSE,
                         unseen_levels = c("error", "zero_effect")) {
  unseen_levels <- match.arg(unseen_levels)
  terms <- lapply(object$param_names, function(p) {
    lapply(object$recipes[[p]], function(recipe) {
      list(basis = recipe$basis, vars = .refit_vars(recipe, newdata),
           by = if (is.null(recipe$by_name)) NULL else as.double(newdata[[recipe$by_name]]),
           zero = unseen_levels == "zero_effect")
    })
  })
  out <- .Call("wrap__r_gamlss_draws", object$state, object$rust_family, terms,
               as.integer(n_draws), as.double(seed), isTRUE(corrected), PACKAGE = "mboostrs")
  names(out) <- object$param_names
  out
}

#' Concurvity of a model's terms
#'
#' Concurvity is the additive-model form of collinearity: how far a term's
#' fitted function could be reproduced by the other terms. When it is high
#' the partial effects are not separately identified, so their shapes,
#' intervals and the selection that chose them are unstable. The three
#' measures are mgcv's \code{concurvity(full = TRUE)} (Wood, Generalized
#' Additive Models, 2nd ed., section 5.6.2), each between 0 (the term is
#' orthogonal to the others) and 1 (it lies in their span): \code{worst}
#' over every coefficient vector of the term, \code{observed} at the fitted
#' coefficients, and \code{estimate}, an average over coefficient
#' directions.
#'
#' For a \code{gam_refit()} result the measures are computed in the Rust
#' core from the refit's model matrix, the intercept included among the
#' other terms. For a \code{gamboost()} fit they are computed from the
#' fit's own learner designs and boosting path, no refit needed; terms are
#' reduced by the constant direction before measuring, as mgcv measures
#' its constrained smooths, and trees and \code{\%O\%} Kronecker learners,
#' which expose no function-space basis for this measure, are absent from
#' the frame, as is every term of a Multinomial fit, whose store is
#' Kronecker-wrapped. \code{worst} can overshoot 1 slightly from the
#' numerics for near-collinear terms.
#'
#' A \code{glmboost()} term is one column, so its rows are ordinary
#' collinearity: the term's centered column measured against the intercept
#' and every other centered column, and the three measures coincide up to
#' rounding because a single column has one coefficient direction. The
#' formula's intercept learner is constant after centering and reports NaN,
#' and every pairwise row naming it too.
#'
#' @param object A fitted model. For a \code{gam_refit()} result, the
#'   measures above; for a \code{gamboost()} or \code{glmboost()} fit, the
#'   same measures computed from the fit's own designs and boosting path;
#'   any other object is passed to \code{mgcv::concurvity} when mgcv is
#'   installed.
#' @param pairwise FALSE for one row per term against the rest of the
#'   model, TRUE for one row per ordered pair of terms (boosted fits only).
#' @param ... Passed on to \code{mgcv::concurvity} for other objects.
#' @return A data frame with columns \code{term}, \code{worst},
#'   \code{observed} and \code{estimate}; with \code{pairwise = TRUE} an
#'   \code{against} column comes first among the measures.
#' @export
concurvity <- function(object, ...) UseMethod("concurvity")

#' @rdname concurvity
#' @export
concurvity.default <- function(object, ...) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("concurvity() is defined here for gam_refit() results; install mgcv ",
         "for other models", call. = FALSE)
  }
  mgcv::concurvity(object, ...)
}

#' @rdname concurvity
#' @export
concurvity.mboostrs_gam_refit <- function(object, ...) {
  blocks <- object$blocks
  data.frame(
    term = object$terms,
    worst = vapply(blocks, function(b) b$concurvity_worst %||% NA_real_, numeric(1)),
    observed = vapply(blocks, function(b) b$concurvity_observed %||% NA_real_, numeric(1)),
    estimate = vapply(blocks, function(b) b$concurvity_estimate %||% NA_real_, numeric(1)),
    stringsAsFactors = FALSE
  )
}

# Shared by the boosted-fit concurvity methods: maps the binding's store
# indices to learner names and builds the summary or the pairwise frame.
# Terms the measure cannot represent (trees, `%O%` Kronecker learners,
# and the Multinomial family's wrapped store) are absent from `included`,
# so they are absent from the frame too.
.boosted_concurvity_frame <- function(out, learner_names, pairwise) {
  empty <- data.frame(stringsAsFactors = FALSE)
  if (length(out$included) == 0L) {
    if (pairwise) {
      return(data.frame(term = character(), against = character(),
                        worst = numeric(), observed = numeric(),
                        estimate = numeric(), stringsAsFactors = FALSE))
    }
    return(data.frame(term = character(), worst = numeric(),
                      observed = numeric(), estimate = numeric(),
                      stringsAsFactors = FALSE))
  }
  terms <- learner_names[out$included]
  if (!pairwise) {
    return(data.frame(term = terms, worst = out$worst,
                      observed = out$observed, estimate = out$estimate,
                      stringsAsFactors = FALSE))
  }
  # The core's `pairwise[i][j]` measures term `pair_j` against term
  # `pair_i`, matching Python's `term`/`against` orientation.
  data.frame(term = terms[out$pair_j], against = terms[out$pair_i],
             worst = out$pair_worst, observed = out$pair_observed,
             estimate = out$pair_estimate, stringsAsFactors = FALSE)
}

