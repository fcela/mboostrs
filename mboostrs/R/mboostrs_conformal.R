# Distribution-free prediction intervals for glmboost and gamboost fits.
#
# The numerics are in the Rust core (`mboostrs_core::conformal`): the
# finite-sample conformal quantile, the split, CQR and CV+ scores, the band
# widening and the CV+ order statistics. This file draws the calibration
# split and folds, refits the model on row subsets and predicts the held-out
# rows, then passes the predictions to the core.
#
# The random split and fold assignment use R's sample.int, so for a given seed
# the rows chosen differ from the Python package's numpy generator. Each
# method's guarantee holds for either choice of rows.

#' Conformal prediction intervals for a boosted model
#'
#' Splits the training rows into proper-training and calibration sets (or
#' into CV+ folds), refits the model on the proper-training rows, and widens
#' the model's predictions by the conformal quantile of the calibration
#' residuals or scores. The intervals have finite-sample marginal coverage of
#' at least \code{level} when the training and new rows are exchangeable.
#'
#' @param model A fitted \code{mboostrs_glmboost}, \code{mboostrs_gamboost},
#'   or GaussianLSS model for split/distributional calibration.
#' @param newdata Rows to predict, as a data frame with the predictor
#'   columns. \code{NULL} predicts the training rows.
#' @param level Nominal marginal coverage in (0, 1).
#' @param method \code{"split"} (one refit; constant width for absolute scores), \code{"cv+"}
#'   (K refits; every row calibrates) or \code{"cqr"} (two quantile refits;
#'   width varies with the predictors).
#' @param seed Integer seed for the random split or fold assignment. Count-rank
#'   randomization uses a recorded shared-core seed with \code{0 <= seed <= 2^53-1}. With
#'   \code{NULL}, the split uses the current R RNG state and auxiliary count
#'   uniforms receive a fresh seed, recorded in the result.
#' @param calib_fraction Share of training rows held out for calibration
#'   (\code{"split"}, \code{"cqr"}).
#' @param n_folds Number of folds (\code{"cv+"}).
#' @param mstop \code{NULL} uses each refit's fitted step count, except that
#'   \code{"cqr"} tunes each quantile fit's step count by 5-fold
#'   cross-validation on its own proper-training rows. A positive integer
#'   predicts every refit at that step count.
#' @param calibration Optional data frame of rows held out before the model
#'   was chosen, with the response. With \code{method = "split"} the model is
#'   used as fitted and no refit is made.
#' @param score Split-calibration score: \code{"auto"} (default),
#'   \code{"absolute"}, \code{"poisson_normalized"}, \code{"poisson_quantile"}
#'   or \code{"count_rank"}. Automatic scoring uses count ranks for standard
#'   Poisson/NBinomial fits with count training responses. The reference and
#'   eligibility use proper-training rows only. A residual/reference variance
#'   ratio above 2 uses normalized scores instead; other families retain
#'   absolute scores. Explicit legacy scores retain their definitions. Count scores
#'   require nonnegative integer responses and finite nonnegative fitted means.
#'   use a fitted reference, Poisson scale or central quantiles; their marginal
#'   guarantee does not require a correct reference distribution. Endpoints
#'   remain real-valued. Count-rank endpoints represent the nonnegative-integer
#'   support intersection; a singleton below zero represents an empty count set.
#'   Count-rank coverage averages over observations and auxiliary randomness;
#'   its correction is in rank units. The result records support, count size,
#'   zero probability, reference dispersion and randomization seed.
#'   The normalized correction uses standardized residual units; the other
#'   corrections use response units. \code{predicted} remains the fitted
#'   mean and need not be the centre of a Poisson quantile band.
#'   The experimental \code{gaussian_normalized} score for GaussianLSS divides
#'   absolute residuals by its fitted positive scale and inverts in shared Rust
#'   core. It needs an untransformed response and reports a correction in
#'   standardized units. The automatic Gaussian default retains central-band
#'   calibration until its frozen confirmation gate passes.
#' @param count_reference \code{"zero_inflated_poisson"} fits a mean-preserving
#'   zero probability on proper-training rows. \code{"poisson"} fixes it at zero.
#'
#' @return An object of class \code{mboostrs_conformal} with \code{lower},
#'   \code{upper}, \code{predicted} (the interval centre), \code{level},
#'   \code{method}, \code{n_calibration}, \code{quantile},
#'   \code{guaranteed_coverage}, \code{guarantee}, \code{assumptions},
#'   \code{n_training}, \code{n_folds}, \code{n_set_aside}, \code{mstop_source},
#'   \code{calibration_source} and \code{fitted_mean} (the full model's
#'   prediction at the same rows).
#' @export
conformal_interval <- function(model, newdata = NULL, level = 0.9,
                               method = c("split", "cv+", "cqr", "distributional", "adaptive"), seed = NULL,
                               calib_fraction = 0.5, n_folds = 10L, mstop = NULL,
                               calibration = NULL, score = "auto",
                               count_reference = "zero_inflated_poisson",
                               decay = 0.99, time = NULL) {
  method <- match.arg(method)
  if (!is.numeric(decay) || length(decay) != 1L || !is.finite(decay) || decay <= 0 || decay > 1) {
    stop("decay must be a finite number in (0, 1]", call. = FALSE)
  }
  requested_score <- score
  score <- match.arg(score, c("auto", "absolute", "poisson_normalized", "poisson_quantile", "count_rank", "gaussian_normalized"))
  count_reference <- match.arg(count_reference, c("zero_inflated_poisson", "poisson"))
  is_lss <- inherits(model, c("mboostrs_glmboostLSS", "mboostrs_gamboostLSS"))
  if (score == "auto") {
    score <- if (method == "split" && !is_lss) "count_auto" else "absolute"
  }
  if (is_lss) {
    if (!identical(model$family$rust_name, "gaussianlss") ||
        !(method %in% c("split", "distributional", "adaptive")) ||
        score != "gaussian_normalized") {
      stop("R LSS calibration currently needs GaussianLSS and score=gaussian_normalized", call. = FALSE)
    }
    res_lss <- .conformal_gaussian_lss(model, newdata, level, seed, calib_fraction, mstop,
                                       calibration, score, requested_score)
    res_lss$reliability <- .conformal_assess_reliability(res_lss, model, newdata)
    return(res_lss)
  }
  if (score == "gaussian_normalized" || method == "distributional") {
    stop("Gaussian distributional calibration needs a GaussianLSS fit", call. = FALSE)
  }
  if (score != "absolute" && !(method %in% c("split"))) {
    stop("Poisson scores are available only for method='split'", call. = FALSE)
  }
  if (!inherits(model, c("mboostrs_glmboost", "mboostrs_gamboost"))) {
    stop("conformal_interval() needs a glmboost or gamboost fit; a distributional ",
         "(LSS) fit is not supported in R yet. Use the Python package for ",
         "method = 'distributional'.", call. = FALSE)
  }
  .conformal_min_size(level)  # refuses a level outside (0, 1)
  if (!is.null(mstop) && !(is.numeric(mstop) && length(mstop) == 1L && mstop >= 1)) {
    stop("mstop must be NULL or a single positive integer; mstop = 'tune' is ",
         "available in the Python package", call. = FALSE)
  }
  # An unweighted fit stores unit weights, which are not case weights.
  if (!is.null(model$weights) && any(model$weights != 1)) {
    warning("this model was fitted with case weights. Calibration scores are ",
            "unweighted, so the reported guarantee applies only if rows remain ",
            "exchangeable for the target population and the weights are fixed inputs.",
            call. = FALSE)
  }

  result <- if (method == "adaptive") {
    .conformal_adaptive(model, newdata, level, calib_fraction, mstop, calibration, decay, time)
  } else if (!is.null(calibration)) {
    if (method != "split") {
      stop("calibration= supplies held-out rows for split conformal; cv+ and cqr ",
           "refit on subsets of the training rows instead", call. = FALSE)
    }
    .conformal_held_out(model, newdata, level, mstop, calibration, score, count_reference, seed)
  } else if (method == "split") {
    .conformal_split(model, newdata, level, seed, calib_fraction, mstop, score, count_reference)
  } else if (method == "cv+") {
    .conformal_cv_plus(model, newdata, level, seed, n_folds, mstop)
  } else {
    .conformal_cqr(model, newdata, level, seed, calib_fraction, mstop)
  }
  result$fitted_mean <- as.double(predict(model, newdata = newdata, type = "response"))
  result$requested_score <- requested_score
  if (!is.null(result$score) && result$score %in% c("count_rank", "poisson_normalized", "poisson_quantile")) {
    result$support <- "nonnegative_integers"
    result$count_size <- .Call("wrap__r_conformal_count_cardinalities", as.double(result$lower),
                               as.double(result$upper), PACKAGE="mboostrs")
    result$empty_count_set <- result$count_size == 0
  }
  result$reliability <- .conformal_assess_reliability(result, model, newdata)
  structure(result, class = "mboostrs_conformal")
}


# A GaussianLSS specification is refitted on fitting rows only. All score,
# order-statistic and inversion numerics are core calls.
.conformal_gaussian_lss <- function(model, newdata, level, seed, calib_fraction,
                                    steps, calibration, score, requested_score) {
  .conformal_min_size(level)
  if (!is.null(steps) && !(is.numeric(steps) && length(steps) == 1L &&
                           is.finite(steps) && steps >= 1 && steps == floor(steps))) {
    stop("mstop must be NULL or a positive integer", call. = FALSE)
  }
  lhs <- stats::as.formula(model$formulas[[1L]])[[2L]]
  if (!is.symbol(lhs)) stop("Gaussian normalized calibration needs an untransformed response", call. = FALSE)
  response <- as.character(lhs)
  fit <- model
  if (is.null(calibration)) {
    ncal <- .conformal_calibration_count(model$n, calib_fraction)
    rows <- .conformal_permutation(model$n, seed)
    held <- rows[seq_len(ncal)]
    training <- rows[-seq_len(ncal)]
    calibration <- .conformal_rows(model$data, held)
    refit <- model
    refit$data <- .conformal_rows(model$data, training)
    refit$weights <- model$weights[training]
    refit$n <- length(training)
    budget <- if (is.null(steps)) .lss_mstop(model) else steps
    fit <- update(refit, mstop = budget)
    source <- "random split; specification refitted on proper-training rows only"
  } else {
    source <- "caller-supplied rows held out before specification and fitting"
  }
  .conformal_check_calibration_size(nrow(calibration), level)
  y <- calibration[[response]]
  if (is.null(y) || !is.numeric(y) || any(!is.finite(y))) {
    stop("calibration needs the finite numeric response column", call. = FALSE)
  }
  target <- if (is.null(newdata)) model$data else newdata
  cal <- predict(fit, calibration, type = "response", mstop = steps)
  query <- predict(fit, target, type = "response", mstop = steps)
  scores <- .Call("wrap__r_conformal_location_scale_scores", as.double(y),
                  as.double(cal$mu), as.double(cal$sigma), PACKAGE = "mboostrs")
  q <- .conformal_quantile(scores, level)
  band <- .Call("wrap__r_conformal_location_scale_band", as.double(query$mu),
                as.double(query$sigma), q, PACKAGE = "mboostrs")
  point <- .Call("wrap__r_conformal_clip_to_band", as.double(query$mu), as.double(band$lower),
                 as.double(band$upper), PACKAGE = "mboostrs")
  fields <- .conformal_fields(list(lower = band$lower, upper = band$upper, predicted = point,
    fitted_mean = predict(model, target, type = "response")$mu,
    level = level, method = "distributional", n_calibration = length(y), quantile = q,
    score = score, requested_score = requested_score, guaranteed_coverage = level,
    guarantee = "finite-sample marginal split-conformal coverage", n_training = fit$n,
    calibration_source = source, mstop_source = "fitted path chosen without calibration responses"))
  structure(fields, class = "mboostrs_conformal")
}

#' @export
print.mboostrs_conformal <- function(x, ...) {
  cat("<mboostrs_conformal> method = ", x$method, ", level = ", x$level,
      ", n = ", length(x$lower), ", n_calibration = ", x$n_calibration,
      ", mean width = ", format(mean(x$upper - x$lower), digits = 4), "\n", sep = "")
  invisible(x)
}

# --------------------------------------------------------------------------
# Core calls
# --------------------------------------------------------------------------

.conformal_min_size <- function(level) {
  .Call("wrap__r_conformal_min_calibration_size", as.double(level), PACKAGE = "mboostrs")
}

.conformal_calibration_count <- function(n, fraction) {
  as.integer(.Call("wrap__r_conformal_calibration_count", as.integer(n),
                   as.double(fraction), PACKAGE = "mboostrs"))
}

.conformal_quantile <- function(scores, level) {
  .Call("wrap__r_conformal_quantile", as.double(scores), as.double(level), PACKAGE = "mboostrs")
}

.conformal_residuals <- function(y, predicted) {
  .Call("wrap__r_conformal_absolute_residuals", as.double(y), as.double(predicted),
        PACKAGE = "mboostrs")
}

.conformal_band <- function(point, q) {
  .Call("wrap__r_conformal_symmetric_band", as.double(point), as.double(q), PACKAGE = "mboostrs")
}

.conformal_split_scores <- function(y, predicted, level, score) {
  if (score == "absolute") return(.conformal_residuals(y, predicted))
  .Call("wrap__r_conformal_poisson_scores", as.double(y), as.double(predicted),
        as.double(level), score, PACKAGE = "mboostrs")
}

.conformal_split_band <- function(point, q, level, score) {
  if (score == "absolute") return(.conformal_band(point, q))
  .Call("wrap__r_conformal_poisson_band", as.double(point), as.double(level),
        as.double(q), score, PACKAGE = "mboostrs")
}

.conformal_calibrated_split <- function(fit, y_cal, calib_pred, point, level, seed,
                                        mstop, score, count_reference) {
  metadata <- list()
  if (score == "count_auto") {
    score <- .Call("wrap__r_conformal_count_default_score", fit$family$name,
                    as.double(.conformal_response(fit, fit$data)), PACKAGE="mboostrs")
  }
  if (score %in% c("count_auto", "count_rank")) {
    y <- as.double(.conformal_response(fit, fit$data))
    weights <- fit$weights
    if (is.null(weights)) weights <- rep(1, length(y))
    reference <- .Call("wrap__r_conformal_count_reference", y,
      as.double(predict(fit, mstop=mstop, type="response")), as.double(weights),
      count_reference == "zero_inflated_poisson", PACKAGE="mboostrs")
    metadata <- list(reference_distribution=count_reference,
                     zero_probability=reference$zero_probability,
                     reference_dispersion=reference$dispersion_ratio)
    if (score == "count_auto") score <- reference$automatic_score
    if (score == "count_rank") {
      if (!is.null(seed) && !(is.numeric(seed) && length(seed)==1L && is.finite(seed) &&
                              seed>=0 && seed<=2^53-1 && seed==floor(seed))) {
        stop("count randomization seed must be an integer in [0, 2^53-1]", call.=FALSE)
      }
      aux <- .Call("wrap__r_conformal_count_uniforms", as.integer(length(y_cal)),
        if (is.null(seed)) NA_real_ else as.double(seed), 0L, 0L, PACKAGE="mboostrs")
      query <- .Call("wrap__r_conformal_count_uniforms", as.integer(length(point)),
        aux$seed, 1L, 0L, PACKAGE="mboostrs")
      scores <- .Call("wrap__r_conformal_count_rank_scores", as.double(y_cal),
        as.double(calib_pred), reference$zero_probability, aux$values, as.double(level), PACKAGE="mboostrs")
      q <- .conformal_quantile(scores, level)
      band <- .Call("wrap__r_conformal_count_rank_band", as.double(point), reference$zero_probability,
        q, query$values, as.double(level), PACKAGE="mboostrs")
      metadata$randomization_seed <- aux$seed
      metadata$support <- "nonnegative_integers"
      return(c(band, list(q=q, score=score, metadata=metadata)))
    }
  }
  q <- .conformal_quantile(.conformal_split_scores(y_cal, calib_pred, level, score), level)
  c(.conformal_split_band(point, q, level, score), list(q=q, score=score, metadata=metadata))
}

.conformal_ordered <- function(a, b) {
  .Call("wrap__r_conformal_ordered_pair", as.double(a), as.double(b), PACKAGE = "mboostrs")
}

.conformal_scores <- function(lower, upper, y) {
  .Call("wrap__r_conformal_quantile_scores", as.double(lower), as.double(upper),
        as.double(y), PACKAGE = "mboostrs")
}

.conformal_widen <- function(lower, upper, q) {
  .Call("wrap__r_conformal_widen_band", as.double(lower), as.double(upper), as.double(q),
        PACKAGE = "mboostrs")
}

.conformal_midpoint <- function(lower, upper) {
  .Call("wrap__r_conformal_band_midpoint", as.double(lower), as.double(upper),
        PACKAGE = "mboostrs")
}

.conformal_cv_plus_bounds <- function(residuals, fold_of, fold_predictions, n_folds, level) {
  .Call("wrap__r_conformal_cv_plus_bounds", as.double(residuals), as.integer(fold_of),
        as.double(fold_predictions), as.integer(n_folds), as.double(level),
        PACKAGE = "mboostrs")
}

.conformal_guarantee <- function(n, k, level) {
  value <- .Call("wrap__r_conformal_cv_plus_guaranteed_coverage", as.integer(n),
                 as.integer(k), as.double(level), PACKAGE = "mboostrs")
  if (is.nan(value)) NULL else value
}

.conformal_mstop_start <- function(model_mstop) {
  .Call("wrap__r_conformal_tuned_mstop_start", as.integer(model_mstop), PACKAGE = "mboostrs")
}

.conformal_mstop_next <- function(largest, cap) {
  .Call("wrap__r_conformal_tuned_mstop_next", as.double(largest), as.double(cap),
        PACKAGE = "mboostrs")
}

.conformal_worst_bin <- function(x, lower, upper, y, level, bins) {
  .Call("wrap__r_conformal_worst_covariate_bin", as.double(x), as.double(lower),
        as.double(upper), as.double(y), as.double(level), as.integer(bins),
        PACKAGE = "mboostrs")
}

.conformal_mstop_choice <- function(optimal) {
  .Call("wrap__r_conformal_tuned_mstop_choice", as.double(optimal), PACKAGE = "mboostrs")
}

# --------------------------------------------------------------------------
# Refits and rows
# --------------------------------------------------------------------------

.conformal_cap <- 10000

# Rows of a training frame or a named list of columns.
.conformal_rows <- function(data, rows) {
  if (is.data.frame(data)) return(data[rows, , drop = FALSE])
  lapply(data, function(column) column[rows])
}

.conformal_response <- function(model, data) {
  as.double(data[[.response_name(model)]])
}

# The model's own specification refitted on `rows` of its training data,
# with the weights subset alongside and an optional quantile family or step
# count. The fit carries every argument the original fit did that R exposes.
.conformal_refit <- function(model, rows, family = NULL, steps = NULL) {
  control <- model$control
  if (!is.null(steps)) control$mstop <- as.integer(steps)
  weighted <- !is.null(model$weights) && any(model$weights != 1)
  weights <- if (weighted) model$weights[rows] else NULL
  args <- list(formula = model$formula,
               data = .conformal_rows(model$data, rows),
               family = family %||% model$family,
               control = control,
               weights = weights)
  if (inherits(model, "mboostrs_glmboost")) {
    args$center <- model$center
    do.call(glmboost, args)
  } else {
    do.call(gamboost, args)
  }
}

# Stopping step for a quantile fit: a 5-fold cross-validation on the fit's own
# rows, the grid doubling while the risk is still falling at its end. Returns
# the fitted refit at the cross-validated step count.
.conformal_tuned_refit <- function(model, rows, family, seed) {
  largest <- .conformal_mstop_start(model$control$mstop)
  folds <- cv_weights(length(rows), "kfold", 5L,
                      seed = if (is.null(seed)) 0L else as.integer(seed))
  repeat {
    fit <- .conformal_refit(model, rows, family = family, steps = largest)
    cv <- cvrisk(fit, folds = folds)
    at_boundary <- cv$optimal_mstop == length(cv$aggregate) - 1
    if (!at_boundary || largest >= .conformal_cap) {
      return(list(fit = fit, steps = as.integer(.conformal_mstop_choice(cv$optimal_mstop))))
    }
    largest <- .conformal_mstop_next(largest, .conformal_cap)
  }
}

# A permutation of 1..n from `seed`, leaving the global RNG state as it was.
.conformal_permutation <- function(n, seed) {
  if (is.null(seed)) return(sample.int(n))
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = globalenv())
  on.exit({
    if (had_seed) assign(".Random.seed", old_seed, envir = globalenv())
    else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
      rm(".Random.seed", envir = globalenv())
  }, add = TRUE)
  set.seed(seed)
  sample.int(n)
}

.conformal_target <- function(model, newdata) if (is.null(newdata)) model$data else newdata

.conformal_check_calibration_size <- function(n_calib, level) {
  need <- .conformal_min_size(level)
  if (n_calib < need) {
    stop(sprintf("level = %s needs at least %d calibration points; this split has %d. ",
                 format(level), need, n_calib),
         "Lower the level, raise calib_fraction, or use method = 'cv+'.", call. = FALSE)
  }
}

.conformal_fields <- function(fields) {
  if (is.null(fields$mstop_source)) fields$mstop_source <- ""
  if (is.null(fields$n_set_aside)) fields$n_set_aside <- 0L
  fields$assumptions <- c("training and future rows are exchangeable",
                          "the fitting algorithm is permutation symmetric",
                          "the specification, transformations, hyperparameters and mstop were fixed without calibration responses")
  if (identical(fields$score, "count_rank")) {
    fields$assumptions <- c(fields$assumptions,
      "auxiliary uniforms are independent of responses; coverage averages over randomization")
    fields$guarantee <- "finite-sample marginal split-conformal coverage over observations and auxiliary randomness"
  }
  fields
}

# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------

# Columns of a covariate, cut into thirds of the calibration rows and scored by
# how far each third's coverage falls below nominal. Two views are checked: the
# covariate itself, and its distance from its median, so noise that grows toward
# both tails is also seen. Mirrors python/mboostrs/conformal.py.
.uneven_coverage_z <- -4
.covariate_bins <- 3L
.bin_names <- c("lowest", "middle", "highest")

.warn_uneven_coverage <- function(model, calib_data, lower, upper, level) {
  name <- .response_name(model)
  y <- as.double(calib_data[[name]])
  flagged <- character()
  for (covariate in setdiff(names(calib_data), name)) {
    x <- calib_data[[covariate]]
    if (!is.numeric(x) || any(!is.finite(x))) next
    x <- as.double(x)
    views <- list(x, abs(x - stats::median(x)))
    labels <- c(covariate, paste0("|", covariate, " - median|"))
    for (v in seq_along(views)) {
      w <- .conformal_worst_bin(views[[v]], lower, upper, y, level, .covariate_bins)
      if (w[1] < .uneven_coverage_z) {
        flagged <- c(flagged, sprintf("%s (%s third: coverage %.2f over %d rows)",
                                      labels[v], .bin_names[w[2] + 1], w[3], as.integer(w[4])))
      }
    }
  }
  if (length(flagged)) {
    warning(sprintf(paste0("split-conformal coverage is uneven across covariate values, ",
                           "against a nominal %s: %s. The band has one width, so it does ",
                           "not follow noise that changes with a covariate. Use ",
                           "method='cqr' for that case."),
                    format(level), paste(flagged, collapse = "; ")), call. = FALSE)
  }
  invisible(NULL)
}

.conformal_split <- function(model, newdata, level, seed, calib_fraction, mstop, score="absolute",
                             count_reference="zero_inflated_poisson") {
  data <- model$data
  y <- .conformal_response(model, data)
  n <- length(y)
  perm <- .conformal_permutation(n, seed)
  n_calib <- .conformal_calibration_count(n, calib_fraction)
  .conformal_check_calibration_size(n_calib, level)
  calib <- perm[seq_len(n_calib)]
  train <- perm[-seq_len(n_calib)]

  fit <- .conformal_refit(model, train, steps = mstop)
  calib_data <- .conformal_rows(data, calib)
  calib_pred <- as.double(predict(fit, calib_data, mstop = mstop, type = "response"))
  point <- as.double(predict(fit, .conformal_target(model, newdata), mstop = mstop,
                             type = "response"))
  band <- .conformal_calibrated_split(fit, y[calib], calib_pred, point, level, seed,
                                      mstop, score, count_reference)
  if (band$score == "absolute") {
    calib_band <- .conformal_band(calib_pred, band$q)
    .warn_uneven_coverage(model, calib_data, calib_band$lower, calib_band$upper, level)
  }
  .conformal_fields(c(list(
    lower = band$lower, upper = band$upper, predicted = point, level = level,
    method = "split", n_calibration = n_calib, quantile = band$q, score=band$score,
    guaranteed_coverage = level,
    guarantee = "finite-sample marginal split-conformal coverage",
    n_training = n - n_calib,
    calibration_source = sprintf("random split of the model's %d training rows (seed=%s)",
                                 n, if (is.null(seed)) "NULL" else seed)), band$metadata))
}

.conformal_held_out <- function(model, newdata, level, mstop, calibration, score="absolute",
                               count_reference="zero_inflated_poisson", seed=NULL) {
  name <- .response_name(model)
  if (is.null(calibration[[name]])) {
    stop("calibration rows have no response column ", name, call. = FALSE)
  }
  y_cal <- as.double(calibration[[name]])
  n_calib <- length(y_cal)
  .conformal_check_calibration_size(n_calib, level)
  if (!is.null(mstop) && mstop > model$control$mstop) {
    stop(sprintf(paste0("mstop=%d is beyond the model's fitted %d steps. A held-out ",
                        "calibration uses the model as fitted and cannot add steps; ",
                        "refit the model with that mstop, or use method='split' or ",
                        "'cv+', which refit with it."),
                 as.integer(mstop), as.integer(model$control$mstop)), call. = FALSE)
  }
  calib_pred <- as.double(predict(model, calibration, mstop=mstop, type="response"))
  point <- as.double(predict(model, .conformal_target(model, newdata), mstop = mstop,
                             type = "response"))
  band <- .conformal_calibrated_split(model, y_cal, calib_pred, point, level, seed,
                                      mstop, score, count_reference)
  steps <- if (is.null(mstop)) model$control$mstop else mstop
  .conformal_fields(c(list(
    lower = band$lower, upper = band$upper, predicted = point, level = level,
    method = "split", n_calibration = n_calib, quantile = band$q, score=band$score,
    guaranteed_coverage = level,
    guarantee = "finite-sample marginal split-conformal coverage",
    n_training = NULL,
    calibration_source = "rows supplied by the caller, held out before the model was specified and fitted",
    mstop_source = sprintf("the model as fitted (%d steps), chosen without the calibration rows",
                           as.integer(steps))), band$metadata))
}

.conformal_cv_plus <- function(model, newdata, level, seed, n_folds, mstop) {
  data <- model$data
  y <- .conformal_response(model, data)
  n <- length(y)
  n_folds <- as.integer(n_folds)
  if (n_folds < 2L || n_folds > n) {
    stop(sprintf("n_folds must be in [2, n=%d], got %d", n, n_folds), call. = FALSE)
  }
  .conformal_check_calibration_size(n, level)

  # Equal-size folds: when n is not a multiple of K, n mod K random rows sit
  # out of the calibration, as in the Python method.
  fold_size <- n %/% n_folds
  kept <- .conformal_permutation(n, seed)[seq_len(fold_size * n_folds)]
  folds <- matrix(kept, nrow = fold_size, ncol = n_folds)
  n_kept <- length(kept)
  n_set_aside <- n - n_kept

  residual_blocks <- vector("list", n_folds)
  fold_predictions <- vector("list", n_folds)
  for (k in seq_len(n_folds)) {
    held <- folds[, k]
    rows <- sort(setdiff(kept, held))
    fit <- .conformal_refit(model, rows, steps = mstop)
    held_data <- .conformal_rows(data, held)
    residual_blocks[[k]] <- .conformal_residuals(
      y[held], predict(fit, held_data, mstop = mstop, type = "response"))
    fold_predictions[[k]] <- predict(fit, .conformal_target(model, newdata),
                                     mstop = mstop, type = "response")
  }
  residuals <- unlist(residual_blocks, use.names = FALSE)
  fold_of <- rep(seq_len(n_folds) - 1L, each = fold_size)
  preds <- do.call(rbind, fold_predictions)  # K x m, one row per fold
  bounds <- .conformal_cv_plus_bounds(residuals, fold_of, preds, n_folds, level)
  point <- as.double(predict(model, .conformal_target(model, newdata), mstop = mstop,
                             type = "response"))

  .conformal_fields(list(
    lower = bounds$lower, upper = bounds$upper, predicted = point, level = level,
    method = "cv+", n_calibration = n_kept, quantile = NULL,
    guaranteed_coverage = .conformal_guarantee(n_kept, n_folds, level),
    guarantee = "balanced K-fold finite-sample CV+ lower bound",
    n_training = n_kept - fold_size,
    n_folds = n_folds, n_set_aside = n_set_aside,
    calibration_source = sprintf("%d of the model's %d training rows in %d folds of %d (seed=%s)%s",
                                 n_kept, n, n_folds, fold_size,
                                 if (is.null(seed)) "NULL" else seed,
                                 if (n_set_aside) sprintf("; %d rows set aside so the folds have equal size", n_set_aside) else "")))
}

.conformal_cqr <- function(model, newdata, level, seed, calib_fraction, mstop) {
  data <- model$data
  y <- .conformal_response(model, data)
  n <- length(y)
  alpha <- 1 - level
  perm <- .conformal_permutation(n, seed)
  n_calib <- .conformal_calibration_count(n, calib_fraction)
  .conformal_check_calibration_size(n_calib, level)
  calib <- perm[seq_len(n_calib)]
  train <- perm[-seq_len(n_calib)]

  # Each quantile fit gets its own cross-validated step count when mstop is
  # NULL; an integer mstop refits each quantile fit to that many steps.
  quantile_fit <- function(tau) {
    family <- Quantile(tau = tau)
    if (is.null(mstop)) {
      tuned <- .conformal_tuned_refit(model, train, family, seed)
      list(fit = tuned$fit, mstop = tuned$steps)
    } else {
      list(fit = .conformal_refit(model, train, family = family, steps = mstop), mstop = mstop)
    }
  }
  lo_fit <- quantile_fit(alpha / 2)
  hi_fit <- quantile_fit(1 - alpha / 2)

  calib_data <- .conformal_rows(data, calib)
  y_calib <- y[calib]
  lo_c <- predict(lo_fit$fit, calib_data, mstop = lo_fit$mstop, type = "response")
  hi_c <- predict(hi_fit$fit, calib_data, mstop = hi_fit$mstop, type = "response")
  ordered_c <- .conformal_ordered(lo_c, hi_c)
  scores <- .conformal_scores(ordered_c$lower, ordered_c$upper, y_calib)
  q <- .conformal_quantile(scores, level)

  target <- .conformal_target(model, newdata)
  lo_t <- predict(lo_fit$fit, target, mstop = lo_fit$mstop, type = "response")
  hi_t <- predict(hi_fit$fit, target, mstop = hi_fit$mstop, type = "response")
  ordered_t <- .conformal_ordered(lo_t, hi_t)
  band <- .conformal_widen(ordered_t$lower, ordered_t$upper, q)
  mstop_source <- if (is.null(mstop)) {
    "tuned by 5-fold cross-validation inside each proper-training set; calibration responses unused"
  } else {
    sprintf("fixed by the caller (mstop=%d)", as.integer(mstop))
  }
  .conformal_fields(list(
    lower = band$lower, upper = band$upper,
    predicted = .conformal_midpoint(band$lower, band$upper), level = level,
    method = "cqr", n_calibration = n_calib, quantile = q,
    guaranteed_coverage = level,
    guarantee = "finite-sample marginal split-CQR coverage",
    n_training = n - n_calib,
    calibration_source = sprintf("random split of the model's %d training rows (seed=%s)",
                                 n, if (is.null(seed)) "NULL" else seed),
    mstop_source = mstop_source))
}

.adaptive_gamma_r <- function(scores, decay = 0.99) {
  decay <- as.double(decay)
  if (decay == 1.0 || decay != 0.99) return(decay)
  n_cal <- length(scores)
  half <- n_cal %/% 2L
  if (half < 5L) return(0.5 ^ (1.0 / max(10.0, 0.5 * max(n_cal, 1L))))
  early <- scores[seq_len(half)]
  late <- scores[(half + 1L):n_cal]
  early_rms <- sqrt(mean(early * early))
  late_rms <- sqrt(mean(late * late))
  if (early_rms <= 1e-12 || late_rms <= early_rms) return(1.0)
  ratio <- late_rms / early_rms
  if (ratio <= 1.15) return(0.5 ^ (1.0 / max(25.0, 0.5 * n_cal)))
  neff_fn <- function(g) {
    w <- g ^ rev(seq_len(n_cal) - 1L)
    (sum(w) ^ 2) / max(sum(w * w), 1e-15)
  }
  target_neff <- if (n_cal > 300L) max(40.0, 0.12 * n_cal) else min(40.0, 0.85 * n_cal)
  gamma_lo <- max(0.96, 1.0 - 2.0 / target_neff - 0.015 * min(2.0, ratio - 1.15))
  if (neff_fn(gamma_lo) >= target_neff) return(gamma_lo)
  lo <- gamma_lo
  hi <- 0.9999
  for (iter in seq_len(24L)) {
    mid <- 0.5 * (lo + hi)
    if (neff_fn(mid) >= target_neff) hi <- mid else lo <- mid
  }
  min(0.9999, max(0.96, hi))
}

.weighted_conformal_quantile_r <- function(scores, level, gamma) {
  n_cal <- length(scores)
  w <- gamma ^ rev(seq_len(n_cal) - 1L)
  ord <- order(scores)
  s_sorted <- scores[ord]
  w_sorted <- w[ord] / (sum(w) + 1.0)
  cs <- cumsum(w_sorted)
  idx <- which(cs >= level)
  if (length(idx) == 0L) Inf else as.double(s_sorted[idx[1L]])
}

.temporal_scale_ratio_r <- function(scores) {
  n_cal <- length(scores)
  if (n_cal < 10L) return(NA_real_)
  half <- n_cal %/% 2L
  s1 <- scores[seq_len(half)]
  s2 <- scores[(half + 1L):n_cal]
  sd1 <- max(stats::sd(s1), 1e-12)
  sd2 <- max(stats::sd(s2), 1e-12)
  med1 <- max(stats::median(s1), 1e-12)
  med2 <- max(stats::median(s2), 1e-12)
  max(sd2 / sd1, sd1 / sd2, med2 / med1, med1 / med2)
}

.conformal_adaptive <- function(model, newdata, level, calib_fraction, mstop,
                                calibration, decay, time) {
  target <- .conformal_target(model, newdata)
  if (!is.null(calibration)) {
    if (!is.null(time) && is.character(time) && length(time) == 1L && !is.null(calibration[[time]])) {
      calibration <- .conformal_rows(calibration, order(calibration[[time]]))
    }
    y_cal <- .conformal_response(model, calibration)
    .conformal_check_calibration_size(length(y_cal), level)
    fit <- model
    n_train <- model$n
    source_desc <- "caller-supplied chronological calibration rows"
  } else {
    data <- model$data
    n_total <- model$n
    if (n_total < 25L) {
      stop("method='adaptive' needs at least 25 training rows", call. = FALSE)
    }
    if (!is.null(time) && is.character(time) && length(time) == 1L && !is.null(data[[time]])) {
      ord <- order(data[[time]])
    } else {
      ord <- seq_len(n_total)
    }
    frac <- min(as.double(calib_fraction), 0.25)
    n_cal <- if (n_total >= 40L) max(20L, min(as.integer(round(n_total * frac)), n_total - 20L)) else max(10L, n_total %/% 4L)
    .conformal_check_calibration_size(n_cal, level)
    train_idx <- ord[seq_len(n_total - n_cal)]
    cal_idx <- ord[(n_total - n_cal + 1L):n_total]
    fit <- .conformal_refit(model, train_idx, steps = mstop)
    calibration <- .conformal_rows(data, cal_idx)
    y_cal <- .conformal_response(model, calibration)
    n_train <- length(train_idx)
    source_desc <- sprintf("chronological last %d of %d training rows", n_cal, n_total)
  }
  cal_pred <- as.double(predict(fit, calibration, mstop = mstop, type = "response"))
  scores <- abs(y_cal - cal_pred)
  gamma <- .adaptive_gamma_r(scores, decay)
  q <- .weighted_conformal_quantile_r(scores, level, gamma)
  point <- as.double(predict(fit, target, mstop = mstop, type = "response"))
  band <- .conformal_band(point, q)
  res <- .conformal_fields(list(
    lower = band$lower, upper = band$upper, predicted = point, level = level,
    method = "adaptive", n_calibration = length(scores), quantile = q,
    score = "absolute", guaranteed_coverage = level,
    guarantee = sprintf("weighted non-exchangeable conformal calibration (Barber et al., 2023; gamma=%.4g)", gamma),
    assumptions = "exponentially weighted calibration over chronological nonconformity scores",
    n_training = n_train,
    calibration_source = sprintf("%s (gamma=%.4g)", source_desc, gamma),
    mstop_source = if (is.null(mstop)) "fitted step count" else sprintf("fixed (mstop=%d)", as.integer(mstop))
  ))
  res$.temporal_scale_ratio <- .temporal_scale_ratio_r(scores)
  res
}

.conformal_assess_reliability <- function(result, model, newdata) {
  n_cal <- as.integer(result$n_calibration %||% 0L)
  lvl <- as.double(result$level %||% 0.9)
  cov_se <- sqrt(lvl * (1.0 - lvl) / max(n_cal + 1.0, 1.0))
  warnings <- character(0)
  if (cov_se > 0.05) {
    warnings <- c(warnings, sprintf("small calibration split (n_calibration=%d, coverage SE=+/-%.3f)", n_cal, cov_se))
  }
  tsr <- result$.temporal_scale_ratio
  has_shift <- !is.null(tsr) && is.finite(tsr) && tsr >= 1.4
  if (has_shift) {
    warnings <- c(warnings, sprintf("temporal_scale_shift: second-half calibration residual scale differs from first-half by %.2fx", tsr))
  }
  list(
    status = if (length(warnings) > 0L) "caution" else "reliable",
    warnings = warnings,
    n_calibration = n_cal,
    calibration_coverage_se = round(cov_se, 4L),
    temporal_scale_shift = isTRUE(has_shift)
  )
}

#' Online Adaptive Conformal Inference (Gibbs & Candes, 2021)
#'
#' Updates the effective miscoverage target sequentially along a time-ordered
#' forecast stream to maintain long-run coverage under distribution shift.
#'
#' @param predicted Numeric vector of sequential point predictions.
#' @param observed Numeric vector of sequential observed outcomes (same length).
#' @param half_width Baseline interval half-width (scalar or vector of same length).
#' @param level Nominal target coverage in (0, 1), default 0.9.
#' @param lr Step-size learning rate for the online alpha update, default 0.02.
#' @return A list with \code{lower}, \code{upper}, \code{coverage}, \code{alpha_path},
#'   \code{multipliers}, \code{level}, and \code{lr}.
#' @export
adaptive_conformal_stream <- function(predicted, observed, half_width,
                                      level = 0.9, lr = 0.02) {
  pred <- as.double(predicted)
  obs <- as.double(observed)
  n <- length(pred)
  if (length(obs) != n || n < 1L) {
    stop("predicted and observed must be non-empty numeric vectors of equal length", call. = FALSE)
  }
  hw <- if (length(half_width) == 1L) rep(as.double(half_width), n) else as.double(half_width)
  if (length(hw) != n || any(!is.finite(hw)) || any(hw < 0)) {
    stop("half_width must be non-negative finite scalar or vector matching predicted", call. = FALSE)
  }
  if (!is.numeric(level) || length(level) != 1L || level <= 0 || level >= 1) {
    stop("level must be in (0, 1)", call. = FALSE)
  }
  if (!is.numeric(lr) || length(lr) != 1L || !is.finite(lr) || lr <= 0) {
    stop("lr must be a positive finite number", call. = FALSE)
  }
  target_alpha <- 1.0 - as.double(level)
  z_base <- stats::qnorm(0.5 + 0.5 * level)
  alpha_t <- target_alpha
  lower <- numeric(n)
  upper <- numeric(n)
  alpha_path <- numeric(n)
  multipliers <- numeric(n)
  covered <- logical(n)
  for (t in seq_len(n)) {
    a_clamped <- min(max(alpha_t, 1e-6), 0.999)
    mult <- max(0.0, stats::qnorm(1.0 - 0.5 * a_clamped) / z_base)
    half_t <- mult * hw[t]
    lower[t] <- pred[t] - half_t
    upper[t] <- pred[t] + half_t
    alpha_path[t] <- alpha_t
    multipliers[t] <- mult
    err_t <- if (obs[t] < lower[t] || obs[t] > upper[t]) 1.0 else 0.0
    covered[t] <- (err_t == 0.0)
    alpha_t <- min(max(alpha_t + lr * (target_alpha - err_t), 1e-5), 0.999)
  }
  list(
    lower = lower,
    upper = upper,
    coverage = mean(covered),
    alpha_path = alpha_path,
    multipliers = multipliers,
    level = as.double(level),
    lr = as.double(lr)
  )
}
