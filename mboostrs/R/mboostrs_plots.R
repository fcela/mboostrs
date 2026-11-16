# ─── Plotting functions for mboostrs R package ───────────────────────
#
# Mirrors python/mboostrs/plot.py. Uses base R graphics (no ggplot2
# dependency) so the package stays lightweight. All functions return
# invisible(NULL) and draw to the current device, matching R's
# plot-method convention.

# ─── Shared palette ──────────────────────────────────────────────────
#
# The same ten categorical colours and greys `mboostrs.plot.PALETTE` and
# `mboostrs.plot.theme()` use in Python. Before this, the two frontends drew
# the same fitted model in unrelated colours — `steelblue` for coefficient
# paths and bars, `darkred` for the risk path, `rainbow()` for categorical
# series, `red` for reference lines — so a reader comparing the two saw output
# from what looked like two different packages, while the numbers agreed to
# 1e-8. Base R graphics cannot reproduce the typography, but the hues cost
# nothing to share.
#
# If `plot.py`'s PALETTE changes, change these to match; there is no build step
# that could derive one from the other across the language boundary.

#' The mboostrs categorical palette
#'
#' The ten-colour sequence used by both frontends' charts, matching
#' \code{mboostrs.plot.PALETTE} in Python.
#'
#' @return A character vector of ten hex colours.
#' @export
mboostrs_palette <- function() {
  c("#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac")
}

# `n` colours from the palette, recycling past ten the way Python's
# categorical scale does.
.mboostrs_cols <- function(n, alpha = NULL) {
  base <- mboostrs_palette()
  out <- base[(seq_len(max(n, 1L)) - 1L) %% length(base) + 1L]
  if (!is.null(alpha)) out <- grDevices::adjustcolor(out, alpha.f = alpha)
  out
}

# Named theme colours, matching plot.py's grid/tick/domain and text greys.
.MBOOSTRS_PRIMARY   <- "#4e79a7"  # PALETTE[1]
.MBOOSTRS_ALERT     <- "#e15759"  # PALETTE[3], reference lines that mark a problem
.MBOOSTRS_RISK      <- "#e15759"  # the risk path
.MBOOSTRS_RULE      <- "#c6c6c6"  # tick/domain grey
.MBOOSTRS_INK       <- "#161616"  # title/aggregate line

# ─── plot.mboostrs_glmboost (coef path) ─────────────────────────────

#' Plot coefficient paths for a glmboost model
#'
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param which Predictor names to include. NULL = all active.
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' if (interactive()) plot(mod)
#' @export
plot.mboostrs_glmboost <- function(x, which = NULL, main = "Coefficient Paths", ...) {
  paths <- coef_path(x, which = which)
  if (length(paths) == 0) { cat("No active predictors.\n"); return(invisible(NULL)) }

  ms <- x$mstop
  ylim <- range(unlist(paths))
  cols <- .mboostrs_cols(length(paths))

  plot(0:ms, paths[[1]], type = "l", col = cols[1], lwd = 2,
       xlab = "mstop", ylab = "Coefficient", ylim = ylim,
 main = main, ...)
  for (i in seq_along(paths)[-1]) {
    lines(0:ms, paths[[i]], col = cols[i], lwd = 2)
  }
  abline(h = 0, lty = 3, col = .MBOOSTRS_RULE)
  legend("topright", legend = names(paths), col = cols, lwd = 2, cex = 0.7)
  invisible(NULL)
}

# ─── plot.mboostrs_gamboost ─────────────────────────────────────────

#' Plot smooth effects for a gamboost model
#'
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param which Learner name(s) to plot. NULL = all.
#' @param n_points Number of evaluation points per smooth (default 100).
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- runif(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- gamboost(y ~ bbs(x1) + bols(x2), data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot(fit)
#' }
#' @export
plot.mboostrs_gamboost <- function(x, which = NULL, n_points = 100, ...) {
  pe <- partial_effects(x, which = which, n_points = n_points)
  if (length(pe) == 0) { cat("No partial effects.\n"); return(invisible(NULL)) }

  n_plots <- length(pe)
  nrow <- ceiling(sqrt(n_plots))
  ncol <- ceiling(n_plots / nrow)
  op <- par(mfrow = c(nrow, ncol), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))
  on.exit(par(op))

  for (nm in names(pe)) {
    df <- pe[[nm]]
    plot(df$x, df$effect, type = "l", lwd = 2, col = .MBOOSTRS_PRIMARY,
         xlab = nm, ylab = "Effect", main = nm, ...)
    abline(h = 0, lty = 3, col = .MBOOSTRS_RULE)
  }
  mtext("Partial Effects", outer = TRUE, cex = 1.2)
  invisible(NULL)
}

# ─── plot_risk ──────────────────────────────────────────────────────

#' Plot risk path for glmboost
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' if (interactive()) plot_risk(mod)
#' @export
plot_risk.mboostrs_glmboost <- function(x, main = "Empirical Risk", ...) {
  risk <- x$risk
  plot(0:(length(risk) - 1), risk, type = "l", lwd = 2, col = .MBOOSTRS_RISK,
       xlab = "mstop", ylab = "Risk", main = main, ...)
  invisible(NULL)
}

#' Plot risk path for gamboost
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- runif(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- gamboost(y ~ bbs(x1) + bols(x2), data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot_risk(fit)
#' }
#' @export
plot_risk.mboostrs_gamboost <- function(x, main = "Empirical Risk", ...) {
  risk <- x$risk
  plot(0:(length(risk) - 1), risk, type = "l", lwd = 2, col = .MBOOSTRS_RISK,
       xlab = "mstop", ylab = "Risk", main = main, ...)
  invisible(NULL)
}

# ─── plot_cvrisk ────────────────────────────────────────────────────

#' Plot cross-validated risk
#'
#' @param x A \code{mboostrs_cvrisk} object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- rnorm(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- glmboost(y ~ x1 + x2, data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' cr <- cvrisk(fit, b=5)
#' plot(cr)
#' }
#' @export
plot.mboostrs_cvrisk <- function(x, ...) {
  mat <- x$risk_matrix
  m <- ncol(mat)
  if (is.null(colnames(mat))) colnames(mat) <- 1:m

  # Fold lines (semi-transparent)
  cols <- .mboostrs_cols(nrow(mat), alpha = 0.3)
  ylim <- range(mat)
  plot(1:m, x$aggregate, type = "n", ylim = ylim,
       xlab = "mstop", ylab = "Risk", main = "Cross-validated Risk", ...)
  for (i in 1:nrow(mat)) {
    lines(1:m, mat[i, ], col = cols[i])
  }
  # Aggregate (solid black)
  lines(1:m, x$aggregate, lwd = 2, col = .MBOOSTRS_INK)
  # Optimal mstop
  abline(v = x$optimal_mstop, lty = 2, col = .MBOOSTRS_PRIMARY)
  invisible(NULL)
}

# ─── plot_varimp ────────────────────────────────────────────────────

#' Plot variable importance for glmboost
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param sort Logical: sort by importance? (default TRUE).
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' if (interactive()) plot_varimp(mod)
#' @export
plot_varimp.mboostrs_glmboost <- function(x, sort = TRUE, main = "Variable Importance (% reduction)", ...) {
  vi <- varimp(x, percent = TRUE)
  # varimp(percent = TRUE) returns shares that sum to 1, not percentages
  # that sum to 100 (matching Python's mboostrs.varimp()); scale for display
  # so the "(%)" axis label and title are actually true.
  red <- vi$reduction * 100
  if (sort) red <- sort(red, decreasing = TRUE)
  red <- red[red != 0]
  if (length(red) == 0) { cat("No variable importance.\n"); return(invisible(NULL)) }
  barplot(red, horiz = TRUE, las = 1, col = .MBOOSTRS_PRIMARY,
          main = main, xlab = "Reduction (%)", ...)
  invisible(NULL)
}

#' Plot variable importance for gamboost
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param sort Logical: sort by importance? (default TRUE).
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- runif(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- gamboost(y ~ bbs(x1) + bols(x2), data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot_varimp(fit)
#' }
#' @export
plot_varimp.mboostrs_gamboost <- function(x, sort = TRUE, main = "Variable Importance (% reduction)", ...) {
  vi <- varimp(x, percent = TRUE)
  # varimp(percent = TRUE) returns shares that sum to 1, not percentages
  # that sum to 100 (matching Python's mboostrs.varimp()); scale for display
  # so the "(%)" axis label and title are actually true.
  red <- vi$reduction * 100
  if (sort) red <- sort(red, decreasing = TRUE)
  red <- red[red != 0]
  if (length(red) == 0) { cat("No variable importance.\n"); return(invisible(NULL)) }
  barplot(red, horiz = TRUE, las = 1, col = .MBOOSTRS_PRIMARY,
          main = main, xlab = "Reduction (%)", ...)
  invisible(NULL)
}

# ─── plot_selected ──────────────────────────────────────────────────

#' Plot selected base-learners for glmboost
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' if (interactive()) plot_selected(mod)
#' @export
plot_selected.mboostrs_glmboost <- function(x, main = "Selected Base-Learners", ...) {
  ms <- x$mstop
  names <- x$learner_names
  sel <- x$selected[1:ms]
  sel_names <- names[sel]
  unique_names <- unique(sel_names)
  cols <- .mboostrs_cols(length(unique_names))
  names(cols) <- unique_names

  plot(1:ms, rep(NA, ms), ylim = c(0.5, length(unique_names) + 0.5),
       xlab = "Iteration", ylab = "", yaxt = "n",
       main = main, ...)
  axis(2, at = seq_along(unique_names), labels = unique_names, las = 2, cex.axis = 0.7)
  for (i in 1:ms) {
    nm <- sel_names[i]
    points(i, match(nm, unique_names), pch = 15, col = cols[nm], cex = 0.8)
  }
  invisible(NULL)
}

#' Plot selected base-learners for gamboost
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- runif(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- gamboost(y ~ bbs(x1) + bols(x2), data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot_selected(fit)
#' }
#' @export
plot_selected.mboostrs_gamboost <- function(x, main = "Selected Base-Learners", ...) {
  ms <- x$mstop
  names <- x$learner_names
  sel <- x$selected[1:ms]
  sel_names <- names[sel]
  unique_names <- unique(sel_names)
  cols <- .mboostrs_cols(length(unique_names))
  names(cols) <- unique_names

  plot(1:ms, rep(NA, ms), ylim = c(0.5, length(unique_names) + 0.5),
       xlab = "Iteration", ylab = "", yaxt = "n",
       main = main, ...)
  axis(2, at = seq_along(unique_names), labels = unique_names, las = 2, cex.axis = 0.7)
  for (i in 1:ms) {
    nm <- sel_names[i]
    points(i, match(nm, unique_names), pch = 15, col = cols[nm], cex = 0.8)
  }
  invisible(NULL)
}

# ─── plot_residuals ─────────────────────────────────────────────────

#' Plot residuals vs fitted for glmboost
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param type Residual type: "response" (default) or "pearson".
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' if (interactive()) plot_residuals(mod)
#' @export
plot_residuals.mboostrs_glmboost <- function(x, type = "response", main = "Residuals vs Fitted", ...) {
  r <- residuals(x, type = type)
  mu <- predict(x, type = "response")
  plot(mu, r, pch = 20, col = rgb(0, 0, 0, 0.5),
       xlab = "Fitted", ylab = paste0(type, " Residuals"),
       main = main, ...)
  abline(h = 0, lty = 2, col = .MBOOSTRS_ALERT)
  invisible(NULL)
}

#' Plot residuals vs fitted for gamboost
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param type Residual type: "response" (default) or "pearson".
#' @param main Plot title.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- runif(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- gamboost(y ~ bbs(x1) + bols(x2), data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot_residuals(fit)
#' }
#' @export
plot_residuals.mboostrs_gamboost <- function(x, type = "response", main = "Residuals vs Fitted", ...) {
  r <- residuals(x, type = type)
  mu <- predict(x, type = "response")
  plot(mu, r, pch = 20, col = rgb(0, 0, 0, 0.5),
       xlab = "Fitted", ylab = paste0(type, " Residuals"),
       main = main, ...)
  abline(h = 0, lty = 2, col = .MBOOSTRS_ALERT)
  invisible(NULL)
}

# ─── plot_coef_path ─────────────────────────────────────────────────

#' Plot coefficient paths for glmboost
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param which Which learners to include. NULL = all active.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- x + rnorm(n)
#' df <- data.frame(y=y, x=x)
#' fit <- glmboost(y ~ x, data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot_coef_path(fit)
#' }
#' @export
plot_coef_path.mboostrs_glmboost <- function(x, which = NULL, ...) {
  plot(x, which = which, ...)
}

#' Plot coefficient paths for gamboost
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param which Which learners to include. NULL = all active.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- runif(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- gamboost(y ~ bbs(x1) + bols(x2), data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot_coef_path(fit)
#' }
#' @export
plot_coef_path.mboostrs_gamboost <- function(x, which = NULL, ...) {
  plot(x, which = which, ...)
}

# ─── plot_diagnostics ──────────────────────────────────────────────

#' Diagnostic plots for glmboost
#' @param x A fitted \code{mboostrs_glmboost} object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' if (interactive()) plot_diagnostics(mod)
#' @export
plot_diagnostics.mboostrs_glmboost <- function(x, ...) {
  op <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  on.exit(par(op))
  plot_risk(x, main = "Risk Path")
  plot_residuals(x, main = "Residuals")
  plot_varimp(x, main = "Variable Importance")
  plot_selected(x, main = "Selected Learners")
  invisible(NULL)
}

#' Diagnostic plots for gamboost
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x1 <- rnorm(n); x2 <- runif(n); y <- x1 + x2 + rnorm(n)
#' df <- data.frame(y=y, x1=x1, x2=x2)
#' fit <- gamboost(y ~ bbs(x1) + bols(x2), data=df, family=Gaussian(),
#'                 control=boost_control(mstop=50))
#' plot_diagnostics(fit)
#' }
#' @export
plot_diagnostics.mboostrs_gamboost <- function(x, ...) {
  op <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  on.exit(par(op))
  plot_risk(x, main = "Risk Path")
  plot_residuals(x, main = "Residuals")
  plot_varimp(x, main = "Variable Importance")
  plot_selected(x, main = "Selected Learners")
  invisible(NULL)
}

# ─── plot_lss ──────────────────────────────────────────────────────

#' Plot LSS model for glmboostLSS
#' @param x A fitted \code{mboostrs_glmboostLSS} object.
#' @param parameter Parameter name (e.g. "mu"); NULL = all.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- x + rnorm(n)
#' dat <- data.frame(y=y, x=x)
#' mod <- glmboostLSS(y ~ x, data=dat, families=GaussianLSS(),
#'                    control=boost_control(mstop=50))
#' plot_lss(mod)
#' }
#' @export
plot_lss.mboostrs_glmboostLSS <- function(x, parameter = NULL, ...) {
  # For linear LSS, just show the coefficients
  pnames <- x$param_names
  if (is.null(parameter)) parameter <- pnames
  for (pn in parameter) {
    p <- x$params[[pn]]
    cat(sprintf("Parameter: %s, mstop=%d, offset=%.4f\n",
                pn, length(p$selected), p$offset))
  }
  invisible(NULL)
}

#' Plot LSS model for gamboostLSS
#' @param x A fitted \code{mboostrs_gamboostLSS} object.
#' @param parameter Parameter name (e.g. "mu"); NULL = all.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- x + rnorm(n)
#' dat <- data.frame(y=y, x=x)
#' mod <- gamboostLSS(y ~ bbs(x, df=4), data=dat, families=GaussianLSS(),
#'                    control=boost_control(mstop=50))
#' plot_lss(mod)
#' }
#' @export
plot_lss.mboostrs_gamboostLSS <- function(x, parameter = NULL, ...) {
  # For smooth LSS: partial effects are in-sample only
  cat("gamboostLSS plotting: in-sample fitted values per parameter\n")
  pnames <- x$param_names
  if (is.null(parameter)) parameter <- pnames
  for (pn in parameter) {
    eta <- x$params[[pn]]$fitted
    plot(eta, type = "l", main = paste("Fitted", pn),
         ylab = pn, xlab = "Observation", col = .MBOOSTRS_PRIMARY, lwd = 2)
  }
  invisible(NULL)
}

# ─── plot_lss_risk ──────────────────────────────────────────────────

#' Plot LSS risk per parameter for glmboostLSS
#' @param x A fitted \code{mboostrs_glmboostLSS} object.
#' @param parameter Parameter name (e.g. "mu"); NULL = all.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- x + rnorm(n)
#' dat <- data.frame(y=y, x=x)
#' mod <- glmboostLSS(y ~ x, data=dat, families=GaussianLSS(),
#'                    control=boost_control(mstop=50))
#' plot_lss_risk(mod)
#' }
#' @export
plot_lss_risk.mboostrs_glmboostLSS <- function(x, parameter = NULL, ...) {
  risk <- x$risk
  if (!is.list(risk)) {
    plot(risk, type = "l", lwd = 2, col = .MBOOSTRS_RISK,
         xlab = "mstop", ylab = "Risk", main = "LSS Risk")
  } else {
    pnames <- names(risk)
    if (is.null(parameter)) parameter <- pnames
    n_plots <- length(parameter)
    op <- par(mfrow = c(n_plots, 1), mar = c(4, 4, 2, 1))
    on.exit(par(op))
    for (pn in parameter) {
      r <- risk[[pn]]
      plot(0:(length(r) - 1), r, type = "l", lwd = 2, col = .MBOOSTRS_RISK,
           xlab = "mstop", ylab = "Risk", main = paste("Risk:", pn))
    }
  }
  invisible(NULL)
}

#' Plot LSS risk per parameter for gamboostLSS
#' @param x A fitted \code{mboostrs_gamboostLSS} object.
#' @param parameter Parameter name (e.g. "mu"); NULL = all.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n); y <- x + rnorm(n)
#' dat <- data.frame(y=y, x=x)
#' mod <- gamboostLSS(y ~ bbs(x, df=4), data=dat, families=GaussianLSS(),
#'                    control=boost_control(mstop=50))
#' plot_lss_risk(mod)
#' }
#' @export
plot_lss_risk.mboostrs_gamboostLSS <- function(x, parameter = NULL, ...) {
  plot_lss_risk.mboostrs_glmboostLSS(x, parameter = parameter, ...)
}

.plot_distribution_density_lss <- function(x, newdata = NULL, mstop = NULL,
                                           grid = NULL, n_grid = 200L,
                                           weights = NULL, ...) {
  frame <- density_overlay_data(
    x, newdata = newdata, mstop = mstop, grid = grid,
    n_grid = n_grid, weights = weights
  )
  has_atom <- any(frame$kind == "atom")
  if (has_atom) {
    op <- graphics::par(mfrow = c(1, 2))
    on.exit(graphics::par(op))
  }
  kind <- if (any(frame$kind == "mass")) "mass" else "density"
  curves <- frame[frame$kind == kind, , drop = FALSE]
  observed <- curves[curves$series == "observed", , drop = FALSE]
  fitted <- curves[curves$series == "fitted", , drop = FALSE]
  yrange <- range(curves$value, finite = TRUE)
  graphics::plot(
    observed$x, observed$value, type = if (kind == "mass") "b" else "l",
    col = mboostrs_palette()[2], lty = 2, lwd = 2,
    xlab = "Response", ylab = if (kind == "mass") "Probability mass" else "Density",
    ylim = yrange, main = "Observed and average fitted distribution", ...
  )
  graphics::lines(
    fitted$x, fitted$value, type = if (kind == "mass") "b" else "l",
    col = .MBOOSTRS_PRIMARY, lty = 1, lwd = 2
  )
  graphics::legend(
    "topright", legend = c("observed", "fitted"),
    col = c(mboostrs_palette()[2], .MBOOSTRS_PRIMARY), lty = c(2, 1), lwd = 2,
    bty = "n"
  )
  if (has_atom) {
    atom <- frame[frame$kind == "atom", , drop = FALSE]
    graphics::barplot(
      atom$value, names.arg = atom$series,
      col = c(mboostrs_palette()[2], .MBOOSTRS_PRIMARY), ylim = c(0, 1),
      ylab = "Probability mass", main = paste("Atom at", format(atom$x[1]))
    )
  }
  invisible(NULL)
}

#' Plot observed and fitted distributions for glmboostLSS
#' @param x A fitted LSS model.
#' @param newdata Optional data frame containing response and predictors.
#' @param mstop Optional scalar or per-parameter stopping point.
#' @param grid Optional increasing response grid.
#' @param n_grid Number of default grid points.
#' @param weights Optional evaluation weights.
#' @param ... Additional base-graphics arguments.
#' @return Invisibly returns NULL after drawing.
#' @export
plot_distribution_density.mboostrs_glmboostLSS <- function(
    x, newdata = NULL, mstop = NULL, grid = NULL, n_grid = 200L,
    weights = NULL, ...) {
  .plot_distribution_density_lss(x, newdata, mstop, grid, n_grid, weights, ...)
}

#' Plot observed and fitted distributions for gamboostLSS
#' @inheritParams plot_distribution_density.mboostrs_glmboostLSS
#' @export
plot_distribution_density.mboostrs_gamboostLSS <-
  plot_distribution_density.mboostrs_glmboostLSS

.plot_centile_fan_lss <- function(x, xvar,
                                  probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
                                  split_by = NULL, newdata = NULL, mstop = NULL, ...) {
  frame <- centile_fan_data(
    x, xvar = xvar, probs = probs, split_by = split_by,
    newdata = newdata, mstop = mstop
  )
  probs <- sort(unique(as.double(probs)))
  lower <- probs[probs < 0.5]
  groups <- unique(frame$split)
  if (length(groups) > 1L) {
    op <- graphics::par(mfrow = c(1, length(groups)))
    on.exit(graphics::par(op))
  }
  for (group in groups) {
    panel <- frame[frame$split == group, , drop = FALSE]
    quantile_columns <- vapply(probs, .lss_prob_col, character(1))
    yrange <- range(c(panel$observed, unlist(panel[quantile_columns])), finite = TRUE)
    graphics::plot(
      panel$x, panel$observed, type = "n", xlab = xvar, ylab = "Response",
      ylim = yrange,
      main = if (is.null(split_by)) "Fitted distribution centiles" else paste(split_by, "=", group),
      ...
    )
    for (i in seq_along(lower)) {
      lo <- .lss_prob_col(lower[i])
      hi <- .lss_prob_col(1 - lower[i])
      alpha <- 0.10 + 0.25 * i / length(lower)
      graphics::polygon(
        c(panel$x, rev(panel$x)), c(panel[[lo]], rev(panel[[hi]])),
        col = grDevices::adjustcolor(.MBOOSTRS_PRIMARY, alpha.f = alpha), border = NA
      )
    }
    graphics::points(panel$x, panel$observed, pch = 16, cex = 0.45,
                     col = grDevices::adjustcolor(.MBOOSTRS_INK, alpha.f = 0.4))
    graphics::lines(panel$x, panel[[.lss_prob_col(0.5)]],
                    col = .MBOOSTRS_PRIMARY, lwd = 2)
  }
  invisible(NULL)
}

#' Plot a centile fan for glmboostLSS
#' @param x A fitted LSS model.
#' @param xvar Numeric covariate used to order the fan.
#' @param probs Symmetric probabilities containing 0.5.
#' @param split_by Optional column used for panels; this does not refit models.
#' @param newdata Optional data frame containing response and predictors.
#' @param mstop Optional scalar or per-parameter stopping point.
#' @param ... Additional base-graphics arguments.
#' @return Invisibly returns NULL after drawing.
#' @export
plot_centile_fan.mboostrs_glmboostLSS <- function(
    x, xvar, probs = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
    split_by = NULL, newdata = NULL, mstop = NULL, ...) {
  .plot_centile_fan_lss(x, xvar, probs, split_by, newdata, mstop, ...)
}

#' Plot a centile fan for gamboostLSS
#' @inheritParams plot_centile_fan.mboostrs_glmboostLSS
#' @export
plot_centile_fan.mboostrs_gamboostLSS <- plot_centile_fan.mboostrs_glmboostLSS

.plot_deviance_profile_lss <- function(x, xvar = NULL, bins = 10L,
                                       newdata = NULL, mstop = NULL,
                                       weights = NULL, ...) {
  frame <- deviance_profile(
    x, xvar = xvar, bins = bins, newdata = newdata,
    mstop = mstop, weights = weights
  )
  yrange <- range(c(frame$mean_deviance, frame$q90_deviance), finite = TRUE)
  graphics::plot(
    frame$x_mid, frame$mean_deviance, type = "b", pch = 16,
    col = .MBOOSTRS_PRIMARY, lwd = 2, ylim = yrange,
    xlab = if (is.null(xvar)) "Observation order" else xvar,
    ylab = "Joint deviance", main = "Joint deviance profile", ...
  )
  graphics::lines(frame$x_mid, frame$q90_deviance, col = mboostrs_palette()[2],
                  lwd = 2, lty = 2)
  graphics::legend(
    "topright", legend = c("mean", "90th percentile"),
    col = c(.MBOOSTRS_PRIMARY, mboostrs_palette()[2]), lty = c(1, 2), lwd = 2,
    bty = "n"
  )
  invisible(NULL)
}

#' Plot a joint deviance profile for glmboostLSS
#' @param x A fitted LSS model.
#' @param xvar Optional numeric covariate; row order is used when NULL.
#' @param bins Number of equal-count bins.
#' @param newdata Optional data frame containing response and predictors.
#' @param mstop Optional scalar or per-parameter stopping point.
#' @param weights Optional evaluation weights.
#' @param ... Additional base-graphics arguments.
#' @return Invisibly returns NULL after drawing.
#' @export
plot_deviance_profile.mboostrs_glmboostLSS <- function(
    x, xvar = NULL, bins = 10L, newdata = NULL, mstop = NULL,
    weights = NULL, ...) {
  .plot_deviance_profile_lss(x, xvar, bins, newdata, mstop, weights, ...)
}

#' Plot a joint deviance profile for gamboostLSS
#' @inheritParams plot_deviance_profile.mboostrs_glmboostLSS
#' @export
plot_deviance_profile.mboostrs_gamboostLSS <-
  plot_deviance_profile.mboostrs_glmboostLSS

# ─── plot_multinomial ──────────────────────────────────────────────

#' Plot multinomial class probabilities
#' @param x A fitted \code{mboostrs_gamboost} object with Multinomial family.
#' @param ... Additional graphical parameters.
#' @return Invisibly returns \code{NULL}.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100; x <- rnorm(n)
#' y <- factor(cut(2*x + rnorm(n), breaks=c(-Inf,-0.5,0.5,Inf),
#'                 labels=c("A","B","C")))
#' df <- data.frame(y=y, x=x)
#' fit <- gamboost(y ~ bols(x), data=df, family=Multinomial(),
#'                 control=boost_control(mstop=50))
#' plot_multinomial(fit)
#' }
#' @export
plot_multinomial.mboostrs_gamboost <- function(x, ...) {
  if (x$family$name != "multinomial")
    stop("plot_multinomial is only for Multinomial family models")
  probs <- predict(x, type = "response")
  n <- nrow(probs)
  K <- ncol(probs)
  cols <- .mboostrs_cols(K)
  plot(1:n, probs[, 1], type = "l", lwd = 2, col = cols[1],
       ylim = c(0, 1), ylab = "Probability", xlab = "Observation",
       main = "Class Probabilities", ...)
  for (k in 2:K) lines(1:n, probs[, k], lwd = 2, col = cols[k])
  legend("topright", legend = colnames(probs), col = cols, lwd = 2, cex = 0.7)
  invisible(NULL)
}
