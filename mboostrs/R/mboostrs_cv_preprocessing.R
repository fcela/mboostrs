# Frontend fold recipes; the core schedules folds and computes family risks.
.cvrisk_refitted <- function(object, folds) {
  if (is.null(object$data)) stop("refitted preprocessing requires the training data", call. = FALSE)
  if (identical(object$family$name, "multinomial")) {
    stop("refitted preprocessing for Multinomial requires a retained class-coordinate mapping",
         call. = FALSE)
  }
  lhs <- paste(deparse(object$formula[[2L]]), collapse = "")
  if (grepl("scale\\(|poly\\(|ecdf\\(", lhs)) {
    stop("refitted preprocessing cannot aggregate a response transform whose scale changes between folds",
         call. = FALSE)
  }
  if (inherits(object, "mboostrs_gamboost")) {
    kinds <- vapply(object$learner_specs, function(sp) sp$type, character(1))
    if (any(kinds %in% c("buser", "user", "bmrf", "mrf"))) {
      stop("refitted preprocessing cannot subset a user design or a neighbourhood graph; use fixed preprocessing",
           call. = FALSE)
    }
  }
  w <- object$weights %||% rep(1, object$n)
  if (identical(object$weight_policy %||% "upstream", "upstream")) {
    # Bare fold matrices follow the model's weight-scale policy, as in Python.
    count <- ncol(folds)
    folds <- vapply(seq_len(count), function(b) .rescale_weights(folds[, b]),
                    double(nrow(folds)))
    dim(folds) <- c(length(w), count)
  }
  ms <- object$mstop
  ctrl <- object$control %||% boost_control(ms, object$nu)
  ctrl$mstop <- ms
  ctrl$risk <- "inbag"
  ctrl$trace <- FALSE
  provide_fold <- function(bb) {
    training <- which(folds[, bb] > 0 & w > 0)
    held <- which(folds[, bb] == 0 & w > 0)
    if (!length(training) || !length(held)) {
      stop("fold ", bb, " needs positive-weight training and evaluation rows", call. = FALSE)
    }
    train <- object$data[training, , drop = FALSE]
    held_data <- object$data[held, , drop = FALSE]
    offset <- object$user_offset
    per_row <- length(offset) > 1L
    args <- list(formula = object$formula, data = train, family = object$family,
                 control = ctrl, weights = folds[training, bb], weight_policy = "raw",
                 offset = if (per_row) offset[training] else offset)
    is_glm <- inherits(object, "mboostrs_glmboost")
    fit <- if (is_glm) {
      args$center <- object$center %||% any(object$centering_means != 0)
      do.call(glmboost, args)
    } else do.call(gamboost, args)
    raw <- .raw_response(object$formula, held_data)
    if (object$family$name %in% c("binomial", "adaexp", "hingeloss", "auc") &&
        (is.factor(raw) || is.character(raw) || is.logical(raw))) {
      training_response <- .raw_response(object$formula, train)
      levels <- if (is.factor(training_response)) levels(droplevels(training_response))
                else levels(factor(training_response))
      response <- as.double(as.integer(factor(raw, levels = levels)) - 1L)
    } else response <- as.double(raw)
    predictions <- lapply(seq.int(0L, ms), function(k) {
      if (k == 0L) return(as.double(if (per_row) offset[held] else rep(fit$offset, length(held))))
      options <- list(object = fit, newdata = held_data, mstop = k, type = "link")
      if (per_row) options$offset <- offset[held]
      if (!is_glm) {
        options$unseen_levels <- "zero_effect"
        # The fitted boundary is the same throughout the path; report its
        # extrapolation warning once per fold rather than at every step.
        options$extrapolation <- if (k == 1L) "warn" else "allow"
      }
      as.double(do.call(predict, options))
    })
    list(as.double(response), as.double(w[held]), predictions,
         lapply(fit$nuisance_vec_path %||% list(), as.double))
  }
  result <- .Call("wrap__cvrisk_foldwise_r", object$family$name,
                  object$family$params %||% list(), as.integer(ncol(folds)),
                  as.integer(ms), provide_fold, PACKAGE = "mboostrs")
  result$risk_matrix <- matrix(result$risk_matrix, nrow = result$n_folds,
                               ncol = result$n_cols, byrow = TRUE)
  result$preprocessing <- "refit"
  structure(result, class = "mboostrs_cvrisk")
}
