# Read a portable model artifact written by Python `mb.save_model`.
#
# The artifact is schema-versioned JSON, independent of Python and the compiled
# extension. This reader loads any artifact for inspection and scores linear
# (glmboost) models directly in R. Models whose prediction needs the spline /
# tensor / transform machinery (gamboost, interactions, inline transforms) are
# loaded but not scored here; use the Python reader for those.

SCHEMA_VERSION <- 1L

# ── typed-value decoding ─────────────────────────────────────────────────────
# The Python writer tags non-trivial values as {"__type__": ...}. jsonlite
# parses the JSON into nested lists (with simplifyVector = FALSE); this walks
# that structure back into plain R values.

.decode <- function(x) {
  if (is.list(x) && !is.null(x[["__type__"]])) {
    kind <- x[["__type__"]]
    if (kind == "ndarray") {
      data <- as.numeric(unlist(x$data))
      shape <- as.integer(unlist(x$shape))
      if (length(shape) <= 1L) return(data)
      # column-major fill to match numpy's row-major requires a transpose;
      # the reader only needs 1-D arrays, so keep this simple.
      return(array(data, dim = rev(shape)))
    }
    if (kind == "dict") {
      items <- x$items
      return(lapply(items, .decode))
    }
    if (kind == "tuple") return(lapply(x$items, .decode))
    if (kind == "pairs") return(lapply(x$items, function(kv) lapply(kv, .decode)))
    if (kind == "family") return(list(name = x$name, params = .decode(x$params)))
    if (kind == "lss_family") return(list(name = x$name,
                                          params = .decode(x$params),
                                          param_names = unlist(x$param_names)))
    if (kind == "learner") return(list(name = x$name, params = .decode(x$params)))
    if (kind == "execution_report") return(.decode(x$fields))
    if (kind == "transform_recipe") return(list(steps = lapply(x$steps, .decode)))
    if (kind == "transform_step") return(.decode(x$fields))
    if (kind == "interaction") return(list(name = x$name, fields = .decode(x$fields)))
    stop(sprintf("unknown encoded type '%s' in artifact", kind))
  }
  if (is.list(x)) return(lapply(x, .decode))
  x
}

# ── inverse links (response scale) ───────────────────────────────────────────
# Keyed by the artifact family name (the Python class name). A distinct name
# avoids the package's own `.inverse_link(eta, family)` helper.

.reader_response_link <- list(
  Gaussian = function(f) f,
  Laplace  = function(f) f,
  Huber    = function(f) f,
  Quantile = function(f) f,
  Expectile = function(f) f,
  Poisson  = function(f) exp(f),
  GammaReg = function(f) exp(f),
  Binomial = function(f) 1 / (1 + exp(-2 * f))
)

# Model classes this reader can load. Python's `save_model` also writes
# `LSSModel`, which carries one learner block per distribution parameter and
# has no representation here.
.PORTABLE_MODEL_CLASSES <- c("GLMBoostModel", "GAMBoostModel", "LSSModel")


# Family parameter domains, as the constructors enforce them. Keyed by the
# artifact's family name, which is the Python class name, lowercased.
.FAMILY_PARAM_CHECKS <- list(
  quantile    = list(tau = "unit", qoffset = "unit"),
  expectile   = list(tau = "unit"),
  huber       = list(delta = "positive"),
  cindex      = list(sigma = "positive"),
  propodds    = list(n_levels = "at_least_3"),
  multinomial = list(n_classes = "at_least_3")
)

.check_family_params <- function(name, params) {
  rules <- .FAMILY_PARAM_CHECKS[[tolower(name %||% "")]]
  if (is.null(rules) || is.null(params)) return(invisible(NULL))
  for (field in names(rules)) {
    value <- params[[field]]
    if (is.null(value) || length(value) != 1L || !is.numeric(value) ||
          !is.finite(value)) {
      next
    }
    rule <- rules[[field]]
    bad <- switch(rule,
      unit = value <= 0 || value >= 1,
      positive = value <= 0,
      at_least_3 = value < 3,
      FALSE)
    if (bad) {
      expected <- switch(rule,
        unit = "in (0, 1)",
        positive = "positive",
        at_least_3 = "at least 3",
        "valid")
      stop(sprintf(
        "the artifact's %s family has %s = %s, which must be %s. The file is ",
        name, field, format(value), expected),
        "not a valid model; it was not written by a checked constructor.",
        call. = FALSE)
    }
  }
  invisible(NULL)
}

#' Load a portable model artifact
#'
#' Reads a JSON artifact written by Python `mboostrs.save_model`. Any artifact
#' can be loaded and inspected; [predict()][predict.mboostrs_model] scores
#' linear (`glmboost`) models. Spline, interaction, and transformed models are
#' loaded but not scored in R.
#'
#' @param path Path to the JSON artifact.
#' @return An object of class `"mboostrs_model"`.
#' @export
load_model <- function(path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("load_model() requires the 'jsonlite' package")
  }
  doc <- jsonlite::fromJSON(path, simplifyVector = FALSE)

  version <- doc$schema_version
  if (is.null(version) || version != SCHEMA_VERSION) {
    stop(sprintf(
      "artifact schema version %s is not supported (expected %d); re-export it",
      as.character(version), SCHEMA_VERSION))
  }

  # Check the model class before decoding the state, so the refusal names
  # what the reader has no shape for. Distributional artifacts ARE read:
  # their per-parameter learner blocks and `lss_family` tag decode below,
  # for inspection; prediction stays with the Python reader, which owns the
  # spline/tensor machinery.
  if (!isTRUE(doc$model_class %in% .PORTABLE_MODEL_CLASSES)) {
    stop(sprintf(
      paste0("this artifact is a %s; the R reader loads glmboost, gamboost ",
             "and gamboostLSS/glmboostLSS models."),
      as.character(doc$model_class)))
  }

  state <- .decode(doc$state)
  recipe <- if (is.null(doc$transform_recipe)) NULL else .decode(doc$transform_recipe)

  # An artifact is a file, and a file can say anything. The Python reader
  # rejects a family parameter outside its domain; this one accepted the same
  # payload -- a `tau = 1.5` QuantReg loaded and predicted, with the check
  # loss minimised by any fit below every observation. Reading a file is not
  # evidence that the writer validated it.
  .check_family_params(state$family$name, state$family$params)

  if (identical(doc$model_class, "LSSModel")) {
    params <- lapply(state$`_params`, function(p) list(
      kind = p$kind,
      offset = as.numeric(p$offset),
      mstop = length(unlist(p$selected)),
      selected = as.integer(unlist(p$selected)),
      learner_names = if (is.null(p$learners)) character(0)
        else vapply(p$learners, function(l) l$name, character(1)),
      fitted = as.numeric(unlist(p$fitted))
    ))
    names(params) <- if (is.null(state$`_family`$param_names))
      paste0("parameter", seq_along(params))
      else as.character(state$`_family`$param_names)
    return(structure(
      list(
        model_class = doc$model_class,
        mboostrs_version = doc$mboostrs_version,
        family = state$`_family`$name,
        param_names = names(params),
        parameters = params,
        response_name = state$`_response_name`,
        transform_recipe = recipe,
        response_transform = if (is.null(state$`_response_transform`)) NULL
          else .decode(state$`_response_transform`),
        scoring_only = isTRUE(state$scoring_only),
        execution_report = state$`_execution_report`
      ),
      class = "mboostrs_model"
    ))
  }

  structure(
    list(
      model_class = doc$model_class,
      mboostrs_version = doc$mboostrs_version,
      formula = state$formula,
      family = state$family$name,
      offset = as.numeric(state$offset),
      nu = as.numeric(state$nu),
      learner_names = unlist(state$learner_names),
      predictor_names = if (is.null(state$predictor_names)) character(0)
        else unlist(state$predictor_names),
      centering_means = if (is.null(state$centering_means)) numeric(0)
        else as.numeric(unlist(state$centering_means)),
      selected = as.integer(unlist(state$selected)),
      coefficients = lapply(state$coefficients, function(c) as.numeric(c)[1]),
      response_name = state$response_name,
      transform_recipe = recipe,
      execution_report = state$execution_report
    ),
    class = "mboostrs_model"
  )
}

#' @export
print.mboostrs_model <- function(x, ...) {
  cat("mboostrs_model (loaded artifact)\n")
  cat("  class:  ", x$model_class, "\n", sep = "")
  cat("  family: ", x$family, "\n", sep = "")
  if (identical(x$model_class, "LSSModel")) {
    for (name in x$param_names) {
      cat("  ", name, ": mstop ", x$parameters[[name]]$mstop,
          ", offset ", round(x$parameters[[name]]$offset, 4), "\n", sep = "")
    }
    if (isTRUE(x$scoring_only))
      cat("  scoring-only: no response or refit state retained\n")
    return(invisible(x))
  }
  cat("  formula:", x$formula, "\n")
  cat("  mstop:  ", length(x$selected), "\n", sep = "")
  invisible(x)
}

.is_linear_scorable <- function(object) {
  object$model_class == "GLMBoostModel" &&
    is.null(object$transform_recipe) &&
    is.character(object$response_name) && length(object$response_name) == 1L
}

#' Predict from a loaded model artifact
#'
#' Scores linear (`glmboost`) models in R. The aggregated per-learner
#' coefficient path, offset, and predictor centering reproduce the Python
#' prediction exactly. Non-linear models raise an error.
#'
#' @param object A `"mboostrs_model"` from [load_model()].
#' @param newdata A data frame or named list with the model's predictors.
#' @param type `"link"` (default) or `"response"`.
#' @param ... Unused.
#' @return A numeric vector of predictions.
#' @export
predict.mboostrs_model <- function(object, newdata, type = c("link", "response"), ...) {
  type <- match.arg(type)
  if (!.is_linear_scorable(object)) {
    stop(sprintf(
      paste0("the R reader scores linear glmboost models only; this is a %s%s. ",
             "Load it with the Python reader (mboostrs.load_model) to predict."),
      object$model_class,
      if (!is.null(object$transform_recipe)) " with a transform recipe" else ""))
  }

  names_l <- object$learner_names
  # aggregate nu * coef per learner across the selected path
  agg <- setNames(numeric(length(names_l)), names_l)
  for (i in seq_along(object$selected)) {
    j <- object$selected[i] + 1L                 # Python index is 0-based
    agg[j] <- agg[j] + object$nu * object$coefficients[[i]]
  }

  n <- if (is.data.frame(newdata)) nrow(newdata) else length(newdata[[object$predictor_names[1]]])
  if (length(object$offset) > 1L) {
    stop("this model was fitted with a per-row offset, which new rows must ",
         "supply; the artifact reader does not take one", call. = FALSE)
  }
  link <- rep(object$offset, n)
  centers <- setNames(object$centering_means, object$predictor_names)

  for (lname in names_l) {
    if (lname == "(Intercept)") {
      link <- link + agg[[lname]]                # design column of ones
    } else {
      col <- newdata[[lname]]
      if (is.null(col)) stop(sprintf("newdata is missing predictor '%s'", lname))
      center <- if (lname %in% names(centers)) centers[[lname]] else 0
      link <- link + agg[[lname]] * (as.numeric(col) - center)
    }
  }

  if (type == "link") return(as.numeric(link))

  inv <- .reader_response_link[[object$family]]
  if (is.null(inv)) {
    stop(sprintf("response-scale prediction is not implemented for family '%s' in the R reader",
                 object$family))
  }
  as.numeric(inv(link))
}
