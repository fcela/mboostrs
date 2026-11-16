# Rearrangement of separately fitted quantile or expectile levels
# (Chernozhukov, Fernandez-Val and Galichon, 2010). The sort and the crossing
# statistics are computed in the shared Rust core.

#' Sort quantile or expectile predictions across levels
#'
#' Each \code{Quantile(tau)} or \code{Expectile(tau)} fit estimates one level
#' of the conditional distribution, so separately fitted curves can cross: at
#' some rows the predicted \code{tau = 0.9} value lies below the predicted
#' \code{tau = 0.5} value. \code{rearrange_quantiles()} predicts every fit on
#' \code{newdata} and, for each row, sorts the predictions into increasing
#' order of \code{tau} (the rearrangement of Chernozhukov, Fernandez-Val and
#' Galichon, 2010, Econometrica 78(3)). Rows without a crossing are
#' unchanged. The result also reports, from the unsorted predictions, the
#' share of rows with a crossing, the number of crossings per pair of adjacent
#' levels and the largest crossing.
#'
#' Use it when two or more \code{Quantile} fits, or two or more
#' \code{Expectile} fits, are read together as a conditional distribution or
#' as the edges of a prediction band. The identity link makes the predictions
#' of these families response-scale values. When the true curve is increasing
#' in \code{tau}, the sorted estimate is no farther from it than the unsorted
#' estimate in every \eqn{L^p} norm (Chernozhukov et al., 2010). The fits are
#' not modified and nothing is refit.
#'
#' @param models A list of fitted \code{mboostrs_glmboost} or
#'   \code{mboostrs_gamboost} objects, all with \code{Quantile} families or
#'   all with \code{Expectile} families, at distinct \code{tau}. Its names
#'   are ignored.
#' @param newdata A data frame with the predictor columns every fit needs.
#'   \code{NULL} predicts the training rows; all fits must then share them.
#' @param levels Optional numeric vector with the \code{tau} of each fit in
#'   \code{models} order, checked against the \code{tau} stored in each
#'   fit's family.
#'
#' @return A list of class \code{mboostrs_rearranged} with \code{kind}
#'   (\code{"quantile"} or \code{"expectile"}), \code{levels} (increasing
#'   \code{tau}), \code{predictions} (a data frame with one column per level,
#'   named \code{q10}, \code{q50}, ... for quantiles and \code{e10}, ... for
#'   expectiles, each row non-decreasing), \code{unsorted} (the same columns
#'   before sorting), \code{crossing_rows}, \code{crossing_share},
#'   \code{pair_crossings} (a data frame with \code{lower}, \code{upper},
#'   \code{n_crossing} and \code{share} for each pair of adjacent levels),
#'   \code{max_crossing} (the largest amount by which a lower level was
#'   predicted above the next higher level, in response units) and \code{n}.
#' @references Chernozhukov, V., Fernandez-Val, I. and Galichon, A. (2010).
#'   Quantile and probability curves without crossing. Econometrica 78(3),
#'   1093-1125.
#' @examples
#' set.seed(1)
#' n <- 200; x <- runif(n)
#' d <- data.frame(x = x, y = x + (0.2 + x) * rnorm(n))
#' fits <- lapply(c(0.1, 0.5, 0.9), function(t)
#'   glmboost(y ~ x, data = d, family = Quantile(tau = t),
#'            control = boost_control(mstop = 10)))
#' r <- rearrange_quantiles(fits, d)
#' r$crossing_share
#' head(r$predictions)
#' @export
rearrange_quantiles <- function(models, newdata = NULL, levels = NULL) {
  if (inherits(models, c("mboostrs_glmboost", "mboostrs_gamboost"))) {
    stop("rearrange_quantiles() needs a list of at least two fits at different levels",
         call. = FALSE)
  }
  models <- unname(as.list(models))
  if (length(models) < 2L) {
    stop("rearrange_quantiles() needs at least two models at different levels",
         call. = FALSE)
  }
  info <- lapply(seq_along(models), function(i) {
    fam <- models[[i]]$family
    nm <- if (is.null(fam)) NULL else fam$name
    if (identical(nm, "quantreg")) return(list(kind = "quantile", tau = fam$params$tau))
    if (identical(nm, "expectreg")) return(list(kind = "expectile", tau = fam$params$tau))
    stop(sprintf("models[[%d]] has family %s; rearrange_quantiles() needs Quantile or Expectile fits",
                 i, if (is.null(nm)) "unknown" else nm), call. = FALSE)
  })
  kinds <- unique(vapply(info, function(z) z$kind, character(1)))
  if (length(kinds) > 1L) {
    stop("models mix Quantile and Expectile families; rearrange one kind at a time",
         call. = FALSE)
  }
  kind <- kinds
  taus <- vapply(info, function(z) as.double(z$tau), numeric(1))
  if (!is.null(levels)) {
    levels <- as.double(levels)
    if (length(levels) != length(taus) || any(abs(levels - taus) > 1e-12)) {
      stop(sprintf("levels=c(%s) do not match the tau read from the fits' families c(%s)",
                   paste(format(levels), collapse = ", "),
                   paste(format(taus), collapse = ", ")), call. = FALSE)
    }
  }
  if (anyDuplicated(taus)) {
    stop(sprintf("two models share a tau; taus are c(%s)",
                 paste(format(taus), collapse = ", ")), call. = FALSE)
  }
  ord <- order(taus)
  cols <- lapply(ord, function(i) as.double(predict(models[[i]], newdata = newdata, type = "link")))
  lens <- vapply(cols, length, integer(1))
  if (length(unique(lens)) != 1L) {
    stop(sprintf("models predicted different numbers of rows: %s",
                 paste(lens, collapse = ", ")), call. = FALSE)
  }
  n <- lens[[1L]]
  k <- length(cols)
  res <- .Call("wrap__r_rearrange_levels", unlist(cols), as.integer(n), as.integer(k),
               PACKAGE = "mboostrs")
  sorted <- matrix(res$sorted, nrow = n, ncol = k)
  unsorted <- matrix(unlist(cols), nrow = n, ncol = k)
  stau <- taus[ord]
  prefix <- if (kind == "quantile") "q" else "e"
  nms <- gsub(".", "_", paste0(prefix, formatC(100 * stau, format = "g", digits = 6, width = 1)), fixed = TRUE)
  colnames(sorted) <- nms
  colnames(unsorted) <- nms
  structure(list(
    kind = kind,
    levels = stau,
    predictions = as.data.frame(sorted),
    unsorted = as.data.frame(unsorted),
    crossing_rows = res$crossing_rows,
    crossing_share = res$crossing_share,
    pair_crossings = data.frame(
      lower = stau[-k], upper = stau[-1L],
      n_crossing = res$pair_crossings,
      share = if (n > 0L) res$pair_crossings / n else rep(0, k - 1L)),
    max_crossing = res$max_crossing,
    n = n
  ), class = "mboostrs_rearranged")
}
