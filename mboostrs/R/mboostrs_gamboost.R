

# ─── gamboost ─────────────────────────────────────────────────────────

#' Fit a GAMBoost model
#'
#' @param formula A formula with base-learner specifications (e.g., \code{y ~ bbs(x1, df=4) + bols(x2)}).
#'   Supported base-learners: \code{bbs()} (smooth P-spline), \code{bols()} (linear),
#'   \code{btree()} (tree stumps), \code{bmono()} (monotone P-spline), \code{bspatial()}
#'   (tensor-product 2-D surface), \code{brandom()} (random intercept), \code{bmrf()}
#'   (Markov random field), \code{bns()} (unpenalized B-spline).
#'
#'   \code{bmono()} takes \code{constraint} (\code{"increasing"} by default,
#'   also \code{"decreasing"}, \code{"convex"}, \code{"concave"},
#'   \code{"positive"}, \code{"negative"}) and \code{type}, which selects the
#'   inner constrained solver. \code{"quad.prog"} (the default here and in
#'   \code{mboost}) solves the constrained quadratic programme exactly, so the
#'   fitted curve satisfies the constraint at every iteration count.
#'   \code{"iterative"} is a faster violation-penalty approximation whose
#'   per-step error carries a consistent sign, so the cumulative fit drifts out
#'   of the constrained region as the iteration count grows.
#'   \code{solver} is accepted as a synonym for \code{type}.
#' @param data A data frame.
#' @param family A family object.
#' @param control A boost_control list.
#' @param weights Optional observation weights, one per row, non-negative.
#'   Integer weights are frequencies -- weight 3 stands for three identical
#'   rows -- and are used at the scale given; fractional weights are rescaled
#'   (see `weight_policy`). A zero weight excludes a row while keeping its
#'   position. An all-zero vector is rejected: the fit it produces reports a
#'   completed path over an empty sample.
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
#' @param bins Reserved for Python-interface parity. The R frontend refuses
#'   non-NULL values because quantile-binned designs are not implemented here.
#' @param .oob_weights Internal: held-out risk weights for one
#'   cross-validation fold, supplied by \code{cvrisk()}. Leave unset.
#' @return An object of class \code{"mboostrs_gamboost"}. Its
#'   \code{$penalty_drift} is \code{NULL} unless some learner was selected
#'   often enough (\code{nu} times its selections at least 50) to have
#'   drifted; then it is a data frame with one row per penalized learner:
#'   \code{selections}, \code{single_step_df} (the trace of the learner's
#'   smoother \eqn{S}), \code{boosted_df} (\eqn{\sum_k 1 - (1 - \nu s_k)^m}
#'   over the eigenvalues \eqn{s_k} of \eqn{S} after \eqn{m} selections),
#'   \code{unpenalized_df} (the rank of the learner's basis, which the boosted
#'   df approaches as selections grow) and \code{drift_share}, the part of the
#'   gap from the one-step df to the unpenalized df that boosting has used.
#'   The fit warns when a learner's share reaches one third: a penalty's
#'   \code{df} then no longer sets the term's flexibility (Schulte and
#'   Ruegamer, 2025).
#' @seealso [glmboost()], [boost_control()], [Gaussian()], [predict.mboostrs_gamboost()], [bbs()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(
#'   y = rnorm(n),
#'   x1 = rnorm(n),
#'   x2 = runif(n)
#' )
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' head(predict(mod))
#' coef(mod)
#' @export
gamboost <- function(formula, data, family = Gaussian(), control = boost_control(),
                     weights = NULL, offset = NULL, weight_policy = "upstream",
                     bins = NULL, .oob_weights = NULL) {
  cl <- match.call()
  # A family may set its own step length for a fit given no control
  # (Quantile: 0.3, measured on 18 quantile fits; mboost uses 0.1).
  if (missing(control) && !is.null(family$default_nu))
    control <- boost_control(nu = family$default_nu)
  .require_scalar_nu(control)

  # Accepted so the refusal names the feature. Without the argument, a reader
  # who found `bins=` in the Python documentation got R's generic
  # `unused argument (bins = 32)`, which says nothing about which frontend
  # has it.
  if (!is.null(bins)) {
    stop("bins= is not supported by the mboostrs R gamboost frontend. ",
         "Quantile binning of continuous designs is available from the ",
         "Python frontend.", call. = FALSE)
  }

  n <- nrow(data)
  normalized <- .normalize_weights(weights, n, weight_policy)
  w <- normalized$fit

  # Parse formula to extract base-learner specs
  spec_list <- .parse_gamboost_formula(formula, data,
                                      vector_offset = length(offset) > 1L,
                                      family_name = family$name)
  response <- spec_list$response
  labelled <- .binary_label_response(.raw_response(formula, data), family$name)
  if (!is.null(labelled)) response <- labelled
  learner_specs <- spec_list$learner_specs
  learner_designs <- spec_list$learner_designs
  learner_names <- spec_list$learner_names
  learner_ncol <- spec_list$learner_ncol
  learner_centering <- spec_list$learner_centering
  n_learners <- length(learner_specs)
  for (sp in spec_list$learner_specs) {
    if (!is.null(sp$by_levels) && length(sp$by_levels) < 2L) {
      stop("by = ", sp$by_name, " has a single level; a factor by= codes the ",
           "difference from its first level, so two are needed", call. = FALSE)
    }
    floor <- sp$by_factor$floor
    if (!is.null(floor) && !is.null(sp$df) && sp$df == floor) {
      # mboost warns at the same boundary: lambda grows without bound and df=
      # no longer sets the term's flexibility.
      warning("bbs(", sp$name, ", by = ", sp$by_name, "): df = ", sp$df,
              " equals the dimension of the stacked penalty's unpenalized ",
              "part, so the term is fitted at maximum shrinkage. Use a larger df.",
              call. = FALSE)
    }
  }

  family_name <- family$name
  family_params <- family$params %||% list()
  offset_val <- .offset_arg(offset, nrow(data))

  # Multinomial: dummy-encode the factor response (drop the last/reference class)
  # into an observation-major (n*(K-1)) vector and pass the class count K. Only
  # bols-style learners are supported (matching R's bols(x) %O% bols(class)).
  multinomial_n_classes <- 1L
  multinomial_levels <- NULL
  if (family_name == "multinomial") {
    if (length(offset_val) > 1L) {
      stop("Multinomial has one linear predictor per class, so a per-row offset ",
           "names no single vector; fit without offset", call. = FALSE)
    }
    resp_var <- all.vars(formula[[2]])[1]
    yfac <- as.factor(data[[resp_var]])
    multinomial_levels <- levels(yfac)
    K <- nlevels(yfac)
    if (K < 3) stop("Multinomial requires at least 3 classes; use Binomial for 2.")
    codes <- as.integer(yfac)                        # 1..K
    indicator <- diag(K)[codes, -K, drop = FALSE]     # (n, K-1), reference dropped
    response <- as.double(t(indicator))               # observation-major flat
    family_params$n_classes <- as.numeric(K)
    multinomial_n_classes <- as.integer(K)
    # Supported: numeric bols (type "scalar", linear), bbs (type "bbs",
    # smooth, built as a Kronecker bbs learner) and brandom over a factor
    # (type "brandom", wrapped over its level indicators). Trees and monotone
    # splines are not supported for Multinomial yet.
    for (sp in learner_specs) {
      if (!(sp$type %in% c("scalar", "bbs", "brandom")))
        stop("Multinomial currently supports bols(), bbs() and brandom() base-learners; ",
             "got a '", sp$type, "' term.")
      if (identical(sp$type, "brandom") && (!is.null(sp$by) || isTRUE(sp$center)))
        stop("Multinomial does not support brandom() with by= or center=TRUE.")
    }
  }

  # Build the list of learner specs for Rust
  rust_specs <- vector("list", n_learners)
  for (i in seq_len(n_learners)) {
    sp <- learner_specs[[i]]
    rust_specs[[i]] <- switch(sp$type,
      "scalar" = list(
        type = "scalar",
        name = sp$name,
        design = as.double(sp$design),
        lambda = sp$lambda,
        center = sp$center
      ),
      "block" = list(
        type = "block",
        name = sp$name,
        design = as.double(sp$design),
        penalty = as.double(sp$penalty),
        np = as.integer(sp$dims),
        center = sp$center,
        # A factor by= asks the binding to calibrate lambda from df.
        df = if (is.null(sp$df)) -1 else as.double(sp$df)
      ),
      "bbs" = list(
        type = "bbs",
        name = sp$name,
        center = sp$center,
        x = as.double(sp$x),
        knots = as.double(sp$knots),
        degree = as.double(sp$degree),
        differences = as.double(sp$differences),
        df = as.double(sp$df),
        lambda = as.double(sp$lambda %||% -1),
        deriv = as.double(sp$deriv %||% 0),
        bk_lo = as.double(sp$bk_lo),
        bk_hi = as.double(sp$bk_hi),
        knot_placement = as.character(sp$knot_placement %||% "equidistant"),
        bins = if (is.null(sp$bins)) NULL else as.integer(sp$bins),
        by_values = if (is.null(sp$by_values)) NULL else as.double(sp$by_values)
      ),
      "tree" = list(
        type = "tree",
        name = sp$name,
        design = as.double(sp$design),
        nq = as.integer(sp$dims),
        categorical = as.integer(sp$categorical %||% integer(0)),
        selection = as.character(sp$selection %||% "ctree"),
        split_method = as.character(sp$split_method %||% "exact"),
        lambda_leaf = as.double(sp$lambda_leaf %||% 0.0),
        colsample = as.double(sp$colsample %||% 1.0),
        colsample_seed = as.integer(sp$colsample_seed %||% 0L),
        minbucket = as.integer(sp$minbucket %||% 4L),
        minsplit = as.integer(sp$minsplit %||% 10L),
        maxdepth = as.integer(sp$maxdepth %||% 1L),
        by_values = if (is.null(sp$by_values)) NULL else as.double(sp$by_values)
      ),
      "bmono" = list(
        type = "bmono",
        name = sp$name,
        x = as.double(sp$x),
        knots = as.double(sp$knots),
        degree = as.double(sp$degree),
        differences = as.double(sp$differences),
        df = as.double(sp$df),
        solver = sp$solver %||% "quad.prog",
        lambda2 = sp$lambda2 %||% 1e6,
        niter = as.integer(sp$niter %||% 10L),
        constraint = sp$constraint %||% "increasing",
        center = sp$center %||% FALSE,
        by_values = if (is.null(sp$by_values)) NULL else as.double(sp$by_values)
      ),
      # The binding had an arm for it, but nothing forwarded it here, so
      # gamboost refused every bconstrained() term as an unknown type.
      "bconstrained" = list(
        type = "bconstrained",
        name = sp$name,
        x = as.double(sp$x),
        C = as.double(sp$C),
        C_dims = as.integer(sp$C_dims),
        knots = as.double(sp$knots),
        degree = as.double(sp$degree),
        differences = as.double(sp$differences),
        df = as.double(sp$df),
        lambda = as.double(sp$lambda %||% -1),
        solver = sp$solver %||% "quad.prog",
        lambda2 = sp$lambda2 %||% 1e6,
        niter = as.integer(sp$niter %||% 10L)
      ),
      "bspatial" = list(
        type = "bspatial",
        name = sp$name,
        name1 = sp$name1,
        name2 = sp$name2,
        x1 = as.double(sp$x1),
        x2 = as.double(sp$x2),
        knots = as.double(sp$knots %||% 10),
        degree = as.double(sp$degree %||% 3),
        differences = as.double(sp$differences %||% 2),
        df = as.double(sp$df %||% 6),
        lambda = as.double(sp$lambda %||% -1),
        center = sp$center %||% TRUE,
        by_values = if (is.null(sp$by_values)) NULL else as.double(sp$by_values)
      ),
      "brandom" = list(
         type = "brandom",
         name = sp$name,
         x = as.double(sp$x),
         design = as.double(sp$design),
         np = as.integer(sp$dims),
         n_levels = as.integer(sp$n_levels),
         df = as.double(sp$df),
         lambda = as.double(sp$lambda %||% -1),
         center = sp$center %||% FALSE,
         by = sp$by,
         by_values = sp$by_values,
         ncol = as.integer(sp$n_levels)
       ),
       "bmrf" = list(
         type = "bmrf",
         name = sp$name,
         levels = sp$levels,
         n_levels = as.integer(sp$n_levels),
         K = sp$K,
         df = as.double(sp$df),
         lambda = as.double(sp$lambda %||% -1),
         center = sp$center %||% FALSE,
         by = sp$by,
         by_values = if (is.null(sp$by_values)) NULL else as.double(sp$by_values),
         ncol = as.integer(sp$n_levels)
       ),
      "bns" = list(
        type = "bns",
        name = sp$name,
        x = as.double(sp$x),
        knots = as.double(sp$knots %||% 20),
        degree = as.double(sp$degree %||% 3),
        df = as.double(sp$df),
        center = sp$center %||% FALSE,
        bk_lo = as.double(sp$bk_lo),
        bk_hi = as.double(sp$bk_hi),
        by = sp$by,
        by_values = if (is.null(sp$by_values)) NULL else as.double(sp$by_values)
      ),
      stop("Unknown learner type: ", sp$type)
    )
  }

  if (any(!is.finite(response))) stop("y contains NA, NaN, or Inf values")
  if (any(!is.finite(w))) stop("weights contains NA, NaN, or Inf values")

  result <- .Call("wrap__fit_gamboost",
    rust_specs,
    as.double(response),
    as.double(w),
    as.integer(control$mstop),
    as.double(control$nu),
    family_name,
    family_params,
    offset_val,
    control$risk,
    control$trace %||% FALSE,
    multinomial_n_classes,
    # extendr rejects NA for a scalar double; 0 is the "off" sentinel
    { s <- control$subsample %||% NA_real_; if (is.na(s)) 0 else as.double(s) },
    as.double(control$subsample_seed %||% 0),
    # Held-out risk weights for a cvrisk fold; empty means weight one.
    as.double(.oob_weights %||% numeric(0)),
    control$step %||% "fixed", as.integer(control$step_iterations %||% 4L),
    PACKAGE = "mboostrs"
  )
  # Block, scalar, brandom and bmrf learners centre on the fit weights inside
  # the core (brandom and bmrf are block learners there); the binding returns
  # the constants it used, which replace the parse-time values. Keeping the
  # parse-time zeros of a `center = TRUE` brandom or bmrf left
  # `predict(newdata =)` without the centering shift. A Multinomial fit wraps
  # its learners and returns none.
  for (i in seq_along(learner_specs)) {
    if (learner_specs[[i]]$type %in% c("block", "scalar", "brandom", "bmrf") &&
        !is.null(result$learner_centering) &&
        length(result$learner_centering[[i]]) > 0L) {
      learner_centering[[i]] <- result$learner_centering[[i]]
    }
  }
  if (isTRUE(control$trace)) .render_execution_report(result$execution_report)
  .warn_if_risk_rose(result$risk, control, family_name, family_params)
  drift <- .penalty_drift_frame(result$penalty_drift, learner_names, learner_specs)
  # A cross-validation fold refit (held-out weights given) does not warn.
  if (is.null(.oob_weights)) .warn_if_penalty_drifted(drift)

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
      increments = result$increments,
      nuisance = result$nuisance,
      nuisance_vec_path = result$nuisance_vec_path,
      execution_report = result$execution_report,
      penalty_drift = drift,
      nu = control$nu,
      mstop = control$mstop,
      family = family,
      control = control,
      formula = formula,
      data = data,
      learner_names = learner_names,
      learner_specs = learner_specs,
      rust_specs = rust_specs,
      learner_designs = learner_designs,
      learner_ncol = learner_ncol,
      learner_centering = learner_centering,
      n = n,
      weights = w,
      weight_policy = normalized$policy,
      multinomial_n_classes = multinomial_n_classes,
      multinomial_levels = multinomial_levels,
      call = cl
    ),
    class = "mboostrs_gamboost"
  )
}

#' Parse a gamboost formula with base-learner specifications
#'
#' Handles: bbs(), bols(), btree(), bmono(), bspatial(), brandom(), bmrf(), bns()
#' Returns the response, a list of learner specs, design matrices,
#' names, column counts, and centering vectors.
#'
#' @noRd
.parse_gamboost_formula <- function(formula, data, vector_offset = FALSE,
                                    family_name = "gaussian", replay_specs = NULL) {
  tf <- terms(formula, data = data)
  # Evaluate ONLY the response (LHS). The RHS terms are base-learner calls such as
  # bols()/bbs() that are parsed as strings by .parse_bl_term, not real R
  # functions, so model.frame() on the full formula would fail to evaluate them.
  # When called from predict() on newdata, the response column is absent; the
  # response is not needed to build designs, so tolerate a failed LHS eval.
  resp_raw <- tryCatch(
    eval(formula[[2L]], envir = as.list(data), enclos = environment(formula)),
    error = function(e) rep(NA_real_, nrow(data))
  )
  response <- if (is.numeric(resp_raw)) as.double(resp_raw)
              else as.double(as.integer(as.factor(resp_raw)))

  # Extract the RHS terms
  rhs_labels <- attr(tf, "term.labels")
  if (length(rhs_labels) == 0) {
    stop("gamboost formula must have at least one base-learner")
  }

  # For each term, check if it's wrapped in a base-learner function
  learner_specs <- list()
  learner_designs <- list()
  learner_names <- character(0)
  learner_ncol <- integer(0)
  learner_centering <- list()

  parsed_terms <- lapply(seq_along(rhs_labels), function(i) {
    if (!is.null(replay_specs) && !is.null(replay_specs[[i]]$radial_basis)) {
      sp <- replay_specs[[i]]
      design <- .rebuild_block_design(sp, data)
      sp$design <- as.double(t(design)); sp$dims <- as.integer(dim(design))
      return(list(spec=sp, design=sp$design, name=sp$name, ncol=ncol(design),
                  centering=rep(0,ncol(design))))
    }
    .parse_bl_term(rhs_labels[[i]], data, enclos = environment(formula))
  })

  # Which form a `bols` takes is a property of the whole menu, not of one
  # term. The centered single-column form takes its constant from the model
  # offset; the block form `[1, x]` carries its own. Both are correct alone,
  # and mixing them represents the constant twice, so the two compete along
  # the path: measured on `bols(x, lambda = 2) + bols(z)`, fitted values
  # differed from `mboost` by 1.1e-3 while an all-centered or all-block menu
  # agreed to 3e-15, with an identical selection sequence throughout. If any
  # `bols` was forced to the block form -- by a lambda -- the rest follow.
  #
  # A term written `intercept = FALSE` forces the same choice. Upstream gives
  # every other `bols` in the menu its own `[1, x]` intercept, refitted each
  # time that learner is selected; the centered form takes one constant from
  # the fixed offset instead. Those agree when every term is centered and do
  # not when one is not: `bols(x) + bols(z, intercept = FALSE)` sat 0.0502
  # from `mboost`, again with an identical selection sequence.
  #
  # Only a menu whose `bols` terms *disagree* needs converting. One `bols` is
  # consistent with itself whatever form it takes, and a menu where every one
  # is already centered, or every one already uncentered, has nothing to
  # reconcile. Converting those anyway is not merely wasted: it turns a lone
  # `bols(z, intercept = FALSE)` into a block spec, which the distributional
  # frontend does not accept.
  bols_form <- function(term) {
    # A factor bols is a block with its own intercept whatever the menu; with
    # two levels in the data it also has two columns, and counting it here
    # made the form depend on which levels the rows held: new data with two
    # of five levels switched every bols to the block form at prediction.
    if (identical(term$spec$type, "block") && is.null(term$spec$factor_levels) &&
          identical(term$spec$name, term$name) && term$ncol == 2L) {
      return("block")
    }
    if (!identical(term$spec$type, "scalar")) return(NA_character_)
    if (!isTRUE(term$spec$intercept %||% TRUE)) "uncentered" else "centered"
  }
  forms <- unique(stats::na.omit(vapply(parsed_terms, bols_form, character(1))))
  # A `by = z` learner's columns are scaled by z, so its span lacks the
  # constant and the residual mean drifts as it is selected; upstream's
  # `bols(x)` re-absorbs that drift through its own intercept and a centered
  # column cannot (`bbs(x, by = z) + bols(w)`: 0.015 from mboost centered,
  # 2.3e-13 as blocks).
  has_by <- any(vapply(parsed_terms, function(term) !is.null(term$spec$by_name),
                       logical(1)))
  # A per-row offset is not a constant the centered form can borrow, so every
  # `bols` carries its own intercept, as the Python frontend does.
  # A ridge-penalized brandom (or a bkernel) shrinks the constant direction
  # too, so the residual mean drifts as it is selected and a centered bols
  # cannot follow: `brandom(g) + bols(w)` sat 7e-4 from mboost centered,
  # 3e-16 as blocks. An unpenalized brandom keeps the mean.
  shrinks_mean <- any(vapply(parsed_terms, function(term) {
    sp <- term$spec
    (identical(sp$type, "brandom") && !identical(as.double(sp$lambda %||% -1), 0)) ||
      identical(sp$type, "bkernel") || !is.null(sp$radial_basis)
  }, logical(1)))
  # The centered form relies on the residual mean staying at zero, which only
  # a squared-error loss keeps; any other loss moves the mean of its negative
  # gradient, and mboost's bols(x) re-fits it through its intercept column
  # (bbs(x) + bols(z) was 6.4e-4 from mboost for Binomial, 0.053 for Laplace).
  # Multinomial rebuilds a centered bols as [1, x] itself.
  non_squared_loss <- !(family_name %in% c("gaussian", "gaussreg", "multinomial"))
  if (length(forms) > 1L ||
      ((vector_offset || has_by || shrinks_mean || non_squared_loss) &&
       length(forms) > 0L)) {
    parsed_terms <- lapply(parsed_terms, .bols_as_block)
  }

  for (spec in parsed_terms) {
    learner_specs <- c(learner_specs, list(spec$spec))
    learner_designs <- c(learner_designs, list(spec$design))
    learner_names <- c(learner_names, spec$name)
    learner_ncol <- c(learner_ncol, spec$ncol)
    learner_centering <- c(learner_centering, list(spec$centering))
  }

  list(
    response = response,
    learner_specs = learner_specs,
    learner_designs = learner_designs,
    learner_names = learner_names,
    learner_ncol = learner_ncol,
    learner_centering = learner_centering
  )
}

# Convert an unpenalized scalar `bols` term to the block form, so a menu that
# already contains a block `bols` does not mix the two. Anything that is not a
# scalar `bols` passes through: only that learner has two forms.
#
# The conversion changes how a term is represented and must not change what it
# can fit. `bols(z, intercept = FALSE)` spans `{z}`; giving it the two-column
# `[1, z]` hands back the constant the caller removed. On
# `bols(x, lambda = 2) + bols(z, intercept = FALSE)` that moved fitted values
# 0.0307 away from `mboost` while the selection sequence stayed identical --
# the same shape of defect as the mixed-form one above, and invisible to any
# test that compares `selected()`.
.bols_as_block <- function(term) {
  if (!identical(term$spec$type, "scalar")) return(term)
  x <- as.double(term$spec$design)
  intercept <- isTRUE(term$spec$intercept %||% TRUE)
  design <- if (intercept) cbind(rep(1.0, length(x)), x) else cbind(x)
  p <- ncol(design)
  # The penalty travels with the term. A scalar `bols` records `lambda = -1`
  # when it has none; anything else is a ridge the block form has to keep, or
  # the conversion silently turns a penalized learner into least squares --
  # `bols(x, lambda = 1, intercept = FALSE)` moved 0.0102 from `mboost` that
  # way.
  lambda <- term$spec$lambda %||% -1
  penalty <- if (is.numeric(lambda) && length(lambda) == 1L && lambda >= 0) {
    as.double(lambda) * diag(p)
  } else {
    matrix(0.0, p, p)
  }
  term$spec <- list(
    type = "block",
    name = term$spec$name,
    design = as.double(t(design)),
    penalty = as.double(penalty),
    dims = c(length(x), p),
    center = FALSE,
    bols_block = TRUE,
    intercept = intercept
  )
  # Row-major, as every reader of `learner_designs` (the prediction bridge,
  # the hat-matrix helpers) takes it.
  term$design <- as.double(t(design))
  term$ncol <- p
  term$centering <- 0.0
  term
}


# A term naming a call rather than a column. The R frontend reads a formula
# term as a column name, so `bols(log(x))` reported "Variable not found:
# log(x)" -- true, and it reads as a typo in the data rather than as a
# frontend that does not evaluate transforms. Python stores the transform
# recipe with the model and replays it at prediction time.
.missing_column_message <- function(var_name, data) {
  hint <- .backtick_hint(var_name)
  if (grepl("[()]", var_name)) {
    return(paste0("`", var_name, "` is an expression, and the mboostrs R ",
                  "frontend reads formula terms as column names. Inline ",
                  "transforms are available from the Python frontend; add ",
                  "the computed column to `data` and name it here.", hint))
  }
  paste0("Variable not found: ", var_name,
         ". Columns available: ", paste(names(data), collapse = ", "), hint)
}

# The sentence added to a missing-column message when the name could be a
# column written without the backticks its characters need.
.backtick_hint <- function(var_name) {
  if (identical(make.names(var_name), var_name)) return("")
  paste0(" If ", sQuote(var_name, FALSE), " is a column name, write it in ",
         "backticks: ", .quote_name(var_name), ". Names with spaces, operators ",
         "(+ - * / : ^ % ~), parentheses, commas, quotes or a leading digit ",
         "need backticks, as in bbs(`house age`) + bols(`price/sqft`).")
}

# A column name as formula text: bare when it is syntactic, otherwise in
# backticks with R's escapes (deparse's rule).
.quote_name <- function(x) {
  deparse(as.name(x), backtick = TRUE)
}

# The column name a formula token stands for. A token in backticks is read by
# R's own parser, so its escapes (a backslash before a backtick or a
# backslash) follow R; any other token is returned trimmed.
.unquote_name <- function(x) {
  x <- trimws(x)
  if (length(x) != 1L || is.na(x) || nchar(x) < 2L ||
        !startsWith(x, "`") || !endsWith(x, "`")) return(x)
  tryCatch(as.character(str2lang(x)), error = function(e) substr(x, 2L, nchar(x) - 1L))
}

# A model.matrix column label with the backticks R puts around a non-syntactic
# variable name removed: "`house age`" is "house age", "`my f`B" is "my fB".
.unquote_matrix_label <- function(x) {
  m <- regmatches(x, regexec("^(`(?:[^`\\\\]|\\\\.)+`)(.*)$", x, perl = TRUE))
  vapply(seq_along(x), function(i) {
    if (length(m[[i]]) == 3L) paste0(.unquote_name(m[[i]][2L]), m[[i]][3L]) else x[i]
  }, character(1))
}

#' Parse a single base-learner term
#'
#' @noRd
.parse_bl_term <- function(label, data, enclos = parent.frame()) {
  # An infix operator between two learners has to be refused before the
  # learner regex sees the label. `^(bbs)\\((.+)\\)$` matches
  # `bbs(x, df = 4) %X% bols(z)` with the argument group swallowing
  # `x, df = 4) %X% bols(z`, so the term parsed as a plain `bbs(x)` and fitted
  # a one-dimensional smooth -- 24 coefficients named `x`, the interaction
  # silently gone, and nothing but an "NAs introduced by coercion" warning
  # from the argument reader. Python fits the tensor product, so a ported
  # script got a different model and no error.
  unticked <- .mask_backticks(label)
  if (grepl("%[A-Za-z+]*%", unticked)) {
    stop("the mboostrs R frontend does not support the ",
         regmatches(unticked, regexpr("%[A-Za-z+]*%", unticked)),
         " operator in `", label, "`. Tensor interactions are available from ",
         "the Python frontend; write the interacting term as a pre-computed ",
         "column here.", call. = FALSE)
  }
  # Try to match bbs(...), bols(...), btree(...), bmono(...)
  bl_match <- regmatches(label, regexec("^(bbs|bols|btree|bmono|bconstrained|bspatial|brandom|bmrf|bns|bkernel|buser|brad)\\((.+)\\)$", label))
  if (length(bl_match[[1]]) < 3) {
    # Not a base-learner call — treat as bols (linear)
    var_name <- .unquote_name(label)
    x <- as.double(data[[var_name]])
    if (is.null(x) || length(x) == 0) {
      stop("Variable not found or empty design: ", var_name, .backtick_hint(var_name))
    }
    return(list(
      spec = list(type = "scalar", name = var_name, design = x, lambda = -1, center = TRUE),
      design = x,
      name = var_name,
      ncol = 1L,
      centering = mean(x)
    ))
  }

  bl_type <- bl_match[[1]][2]
  args_str <- bl_match[[1]][3]

  # `by = z` with a numeric z multiplies the learner's design row by row, as
  # mboost does: `bbs(x, by = z)` fits f(x) * z, and `bols(x, by = z)` fits
  # the block [z, x * z]. Other learners and a factor `by` fit a different
  # design and are refused rather than fitted without the modifier.
  parsed_args <- .parse_bl_args(args_str)
  if (!is.null(parsed_args$named$by)) {
    return(.parse_with_by(bl_type, args_str, parsed_args, data))
  }

  switch(bl_type,
    "bols" = .parse_bols(args_str, data),
    "bbs"  = .parse_bbs(args_str, data),
    "btree" = .parse_btree(args_str, data),
    "bmono" = .parse_bmono(args_str, data),
    "bconstrained" = .parse_bconstrained(args_str, data, enclos = enclos),
    "bspatial" = .parse_bspatial(args_str, data),
    "brandom" = .parse_brandom(args_str, data),
    "bmrf" = .parse_bmrf(args_str, data, enclos = enclos),
    "bns" = .parse_bns(args_str, data),
    "bkernel" = .parse_bkernel(args_str, data),
    "buser" = .parse_buser(args_str, data, enclos = enclos),
    "brad" = .parse_brad(args_str, data),
    stop("Unknown base-learner: ", bl_type)
  )
}


.parse_with_by <- function(bl_type, args_str, parsed_args, data) {
  by_name <- if ("by" %in% parsed_args$ticked) parsed_args$named$by
             else gsub("[\"' ]", "", parsed_args$named$by)
  z <- data[[by_name]]
  if (is.null(z)) stop(.missing_column_message(by_name, data))
  if (identical(bl_type, "btree")) {
    # btree(..., by = z) fits f(t) * z by least squares: a tree grown on
    # u / z with weights w * z^2. For mboost's 0/1 indicator or two-level
    # factor (coded 0/1 by its levels) that is mboost's rule, the tree grown
    # on the rows where z is 1; a continuous z (which mboost refuses) gives a
    # one-split varying slope.
    by_levels <- NULL
    if (is.factor(z) || is.character(z) || is.logical(z)) {
      by_levels <- levels(as.factor(z))
      if (length(by_levels) != 2L) {
        stop("btree(by = ", by_name, ") needs a factor with two levels; it has ",
             length(by_levels), ".", call. = FALSE)
      }
      z <- as.double(match(as.character(z), by_levels) - 1L)
    } else {
      z <- as.double(z)
    }
    parsed <- .parse_btree(args_str, data)
    parsed$spec$by_values <- z
    parsed$spec$by_name <- by_name
    parsed$spec$by_levels <- by_levels
    return(parsed)
  }
  if (is.factor(z) || is.character(z) || is.logical(z)) {
    return(.parse_factor_by(bl_type, args_str, parsed_args, data, z, by_name))
  }
  z <- as.double(z)
  if (identical(bl_type, "brandom")) {
    # The indicator design scaled by z row by row, as mboost builds
    # brandom(g, by = z); df is calibrated on the scaled design.
    parsed <- .parse_brandom(args_str, data)
    p <- parsed$ncol
    parsed$design <- as.double(parsed$design * rep(z, each = p))
    parsed$spec$design <- parsed$design
    parsed$spec$by_values <- z
    parsed$spec$by_name <- by_name
    return(parsed)
  }
  if (identical(bl_type, "bmrf")) {
    # The binding scales the region indicators by z row by row, as mboost
    # builds bmrf(region, by = z), and calibrates df on the scaled design.
    parsed <- .parse_bmrf(args_str, data)
    parsed$spec$by_values <- z
    parsed$spec$by_name <- by_name
    return(parsed)
  }
  if (bl_type %in% c("bbs", "bmono", "bns", "bspatial")) {
    # The binding multiplies the basis by z row by row; the stored design,
    # which in-sample replay reads, is scaled the same way.
    parsed <- switch(bl_type,
      bbs = .parse_bbs(args_str, data), bmono = .parse_bmono(args_str, data),
      bns = .parse_bns(args_str, data), bspatial = .parse_bspatial(args_str, data))
    parsed$spec$by_values <- z
    parsed$spec$by_name <- by_name
    p <- parsed$ncol
    parsed$design <- as.double(parsed$design * rep(z, each = p))
    return(parsed)
  }
  if (identical(bl_type, "bols")) {
    var_name <- parsed_args$positional[1]
    x <- data[[var_name]]
    if (is.null(x)) stop(.missing_column_message(var_name, data))
    if (is.factor(x) || is.character(x) || is.logical(x)) {
      stop("bols(", var_name, ", by = ", by_name, ") on a factor is not built ",
           "by the R frontend yet.", call. = FALSE)
    }
    intercept <- as.logical(parsed_args$named$intercept %||% TRUE)
    if (!is.null(parsed_args$named$df)) {
      stop("bols(", var_name, ", by = ", by_name, ", df = ...) needs a df -> ",
           "lambda calibration the R frontend does not build; give lambda=.",
           call. = FALSE)
    }
    design <- .bols_by_design(as.double(x), z, intercept)
    pcols <- ncol(design)
    lambda <- parsed_args$named$lambda
    penalty <- if (is.null(lambda)) matrix(0.0, pcols, pcols) else as.double(lambda) * diag(pcols)
    return(list(
      spec = list(type = "block", name = var_name, design = as.double(t(design)),
                  penalty = as.double(penalty), dims = c(nrow(design), pcols),
                  center = FALSE, by_name = by_name, bols_by = TRUE,
                  intercept = intercept),
      design = as.double(t(design)), name = var_name, ncol = pcols, centering = 0.0))
  }
  stop("by= is not yet supported for ", bl_type, "() by the R frontend; it ",
       "is supported for a numeric by on bbs(), bols(), bmono(), bns(), ",
       "bspatial(), brandom(), bmrf() and btree(), a two-level factor by on ",
       "btree(), and a factor by on bbs() and bols().",
       call. = FALSE)
}

# A factor by= (mboost codes it with treatment contrasts): the learner's design
# X is stacked once per non-reference level, [d_2 X, ..., d_L X], where d_j is
# the indicator of level j, so the term fits each level's difference from the
# first. The penalty is block diagonal, one copy of the learner's per block,
# and df= is calibrated on the whole stacked design by the binding (the block
# spec's $df). Built for bbs() and bols(), as the Python frontend builds it.
.parse_factor_by <- function(bl_type, args_str, parsed_args, data, z, by_name) {
  # Fewer than two levels is refused by gamboost(); predict() re-parses the
  # formula on new rows, which may hold a single level, and replaces this
  # design with one built from the training levels.
  lv <- .by_levels(z)
  if (identical(bl_type, "bbs")) {
    parsed <- .parse_bbs(args_str, data)
    sp <- parsed$spec
    unsupported <- c(center = isTRUE(sp$center), deriv = (sp$deriv %||% 0L) != 0L,
                     cyclic = isTRUE(as.logical(parsed_args$named$cyclic %||% FALSE)))
    if (any(unsupported)) {
      stop("bbs(", sp$name, ", by = ", by_name, ") with ",
           paste(names(unsupported)[unsupported], collapse = ", "),
           ": a factor by= stacks the plain basis per level and does not carry ",
           "that option. Drop it, or use a numeric by=.", call. = FALSE)
    }
    inner <- matrix(parsed$design, ncol = parsed$ncol, byrow = TRUE)
    k_inner <- matrix(.Call("wrap__r_diff_penalty", as.integer(parsed$ncol),
                            as.integer(sp$differences), PACKAGE = "mboostrs"),
                      parsed$ncol)
    var_name <- sp$name
    lambda <- if (sp$lambda >= 0) sp$lambda else NULL
    df <- if (is.null(lambda)) sp$df else NULL
    floor <- as.integer(sp$differences) * (length(lv) - 1L)
    if (!is.null(df) && df < floor) {
      stop("bbs(", var_name, ", by = ", by_name, "): df = ", df, " is below the ",
           floor, "-dimensional unpenalized part of the stacked penalty, which ",
           "no lambda can shrink. Use df >= ", floor, ".", call. = FALSE)
    }

    base <- list(base = "bbs", x_name = var_name, bk_lo = sp$bk_lo, bk_hi = sp$bk_hi,
                 knots = sp$knots, degree = sp$degree, floor = floor)
  } else if (identical(bl_type, "bols")) {
    var_name <- parsed_args$positional[1]
    x <- data[[var_name]]
    if (is.null(x)) stop(.missing_column_message(var_name, data))
    if (is.factor(x) || is.character(x) || is.logical(x)) {
      stop("bols(", var_name, ", by = ", by_name, ") on a factor is not built ",
           "by the R frontend yet.", call. = FALSE)
    }
    intercept <- as.logical(parsed_args$named$intercept %||% TRUE)
    inner <- .bols_by_design(as.double(x), 1.0, intercept)
    lambda <- parsed_args$named$lambda
    df <- parsed_args$named$df
    # R bols: unpenalized unless df= or lambda= asks for a ridge on every column.
    k_inner <- diag(ncol(inner))
    if (is.null(lambda) && is.null(df)) lambda <- 0
    if (!is.null(lambda)) { lambda <- as.double(lambda); df <- NULL } else df <- as.double(df)
    base <- list(base = "bols", x_name = var_name, intercept = intercept)
  } else {
    stop("by= with a factor is supported for bbs() and bols() by the R ",
         "frontend; ", bl_type, "() is not.", call. = FALSE)
  }
  design <- if (length(lv) >= 2L) .factor_by_design(inner, z, lv)
            else matrix(0.0, nrow(inner), 0L)
  penalty <- kronecker(diag(max(length(lv) - 1L, 0L)), k_inner)
  if (!is.null(lambda)) penalty <- lambda * penalty
  pcols <- ncol(design)
  list(
    spec = c(list(type = "block", name = var_name, design = as.double(t(design)),
                  penalty = as.double(penalty), dims = c(nrow(design), pcols),
                  center = FALSE, df = df, by_name = by_name, by_levels = lv),
             list(by_factor = base)),
    design = as.double(t(design)), name = var_name, ncol = pcols, centering = 0.0)
}

# A block learner's design at new rows, from what its spec recorded at fit
# time: a factor by= (training levels and boundary knots, then stacked per
# level), a numeric bols by=, a factor bols, or a plain bols in the block
# form. NULL for a block whose design cannot be rebuilt from columns.
# `extrap` is predict()'s extrapolation policy, applied to a spline's x.
.rebuild_block_design <- function(sp, newdata, extrap = NULL, unseen = "error") {
  column <- function(name) {
    v <- newdata[[name]]
    if (is.null(v)) stop(.missing_column_message(name, newdata))
    v
  }
  if (!is.null(sp$radial_basis)) {
    recipe <- sp$radial_basis
    x <- as.matrix(as.data.frame(lapply(recipe$names, function(name) as.double(column(name)))))
    flat <- .Call("wrap__r_brad_design", as.double(t(x)), as.integer(nrow(x)),
                  as.integer(ncol(x)), recipe$centres, recipe$whitening,
                  as.integer(recipe$k), recipe$theta, PACKAGE="mboostrs")
    return(matrix(flat,ncol=recipe$k,byrow=TRUE))
  }
  if (!is.null(sp$by_factor)) {
    bf <- sp$by_factor
    if (identical(bf$base, "bbs")) {
      xn <- as.double(column(bf$x_name))
      if (!is.null(extrap)) xn <- extrap$check(xn, bf$bk_lo, bf$bk_hi, bf$x_name)
      bd <- .Call("wrap__r_bbs_design", xn, bf$bk_lo, bf$bk_hi,
                  as.integer(bf$knots), as.integer(bf$degree), 0L,
                  PACKAGE = "mboostrs")
      inner <- matrix(bd$design, ncol = bd$p, byrow = TRUE)
    } else {
      inner <- .bols_by_design(as.double(column(bf$x_name)), 1.0,
                               isTRUE(bf$intercept))
    }
    return(.factor_by_design(inner, column(sp$by_name), sp$by_levels, unseen))
  }
  if (!is.null(sp$brandom_levels)) {
    # brandom(g, by = z): level indicators, coded against the training
    # labels, scaled by z.
    out <- .level_indicators(column(sp$name), sp$brandom_labels, sp$brandom_levels,
                             paste0("brandom(", sp$name, ")"), unseen)
    return(out * as.double(column(sp$by_name)))
  }
  if (isTRUE(sp$bols_by)) {
    return(.bols_by_design(as.double(column(sp$name)), as.double(column(sp$by_name)),
                           isTRUE(sp$intercept)))
  }
  if (!is.null(sp$factor_levels)) {
    return(.bols_factor_design(column(sp$name), sp$factor_levels,
                               isTRUE(sp$intercept), sp$name, unseen))
  }
  if (isTRUE(sp$bols_block)) {
    x <- as.double(column(sp$name))
    return(if (isTRUE(sp$intercept)) cbind(1.0, x) else cbind(x))
  }
  NULL
}

# The levels of a factor by=: a factor's own levels, the sorted values of a
# character column, FALSE/TRUE for a logical one.
.by_levels <- function(z) {
  if (is.factor(z)) return(levels(z))
  if (is.logical(z)) return(c("FALSE", "TRUE"))
  sort(unique(as.character(z)))
}

# Each row's design placed in the block of its level; the first level is the
# reference and has no block. A level outside `lv` is refused.
.factor_by_design <- function(inner, z, lv, unseen = "error") {
  codes <- match(as.character(z), lv)
  if (anyNA(codes) && identical(unseen, "error")) {
    new <- unique(as.character(z)[is.na(codes)])
    stop("by= column has level(s) ", paste(new, collapse = ", "),
         " not present when the model was fitted (levels: ",
         paste(lv, collapse = ", "), "). Pass unseen_levels = \"average\" or ",
         "\"zero_effect\" to predict them.", call. = FALSE)
  }
  p <- ncol(inner)
  out <- matrix(0.0, nrow(inner), p * (length(lv) - 1L))
  for (j in seq_along(lv)[-1L]) {
    rows <- !is.na(codes) & codes == j
    out[rows, (j - 2L) * p + seq_len(p)] <- inner[rows, , drop = FALSE]
    # An unseen level under "average": the mean over the levels, each block
    # weighted 1/L (the reference level's block is zero).
    if (identical(unseen, "average")) {
      new <- is.na(codes)
      out[new, (j - 2L) * p + seq_len(p)] <- inner[new, , drop = FALSE] / length(lv)
    }
  }
  out
}

# Indicator rows for a categorical learner (brandom, bmrf) at new data: the
# values are coded against the training labels when the fit recorded them,
# otherwise read as the integer codes the fit used. A level the fit never saw
# raises under "error", gets a zero row under "zero_effect" and the mean of
# the level rows (1/L each) under "average".
.level_indicators <- function(values, labels, n_levels, name, unseen = "error") {
  codes <- if (!is.null(labels)) {
    match(as.character(values), labels)
  } else {
    v <- suppressWarnings(as.integer(values))
    v[!is.na(v) & (v < 1L | v > n_levels)] <- NA_integer_
    v
  }
  new <- is.na(codes) & !is.na(values)
  if (any(new) && identical(unseen, "error")) {
    stop(name, " has level(s) ", paste(unique(as.character(values[new])), collapse = ", "),
         " the fit did not see. Pass unseen_levels = \"average\" or \"zero_effect\" ",
         "to predict them.", call. = FALSE)
  }
  out <- matrix(0.0, length(codes), n_levels)
  seen <- which(!is.na(codes))
  out[cbind(seen, codes[seen])] <- 1.0
  if (any(new) && identical(unseen, "average")) out[new, ] <- 1 / n_levels
  out
}

# mboost's bols(x, by = z): the design [1, x] (or [x] without intercept)
# multiplied row by row by z.
.bols_by_design <- function(x, z, intercept) {
  base <- if (intercept) cbind(1.0, x) else matrix(x, ncol = 1)
  base * z
}

#' Parse a radial basis using the shared core; retain its replay recipe.
#' @noRd
.parse_brad <- function(args_str, data) {
  args <- .parse_bl_args(args_str)
  names_vec <- args$positional
  if (!length(names_vec)) stop("brad requires at least one predictor")
  missing <- setdiff(names_vec, names(data))
  if (length(missing)) stop("brad predictor(s) not found in data: ", paste(missing, collapse=", "))
  x <- as.matrix(as.data.frame(lapply(names_vec, function(name) as.double(data[[name]]))))
  knots <- as.integer(.check_int_at_least("knots", as.double(args$named$knots %||% 100L), 1L))
  df <- .check_positive("df", as.double(args$named$df %||% 4))
  theta <- if (is.null(args$named$theta)) -1 else .check_positive("theta", as.double(args$named$theta))
  lambda <- if (is.null(args$named$lambda)) -1 else as.double(args$named$lambda)
  if (!is.finite(lambda) || (lambda < 0 && !is.null(args$named$lambda))) stop("brad lambda must be finite and nonnegative")
  built <- .Call("wrap__r_brad_basis", as.double(t(x)), as.integer(nrow(x)),
                 as.integer(ncol(x)), knots, as.double(theta), as.double(lambda), PACKAGE="mboostrs")
  name <- paste(names_vec, collapse=",")
  recipe <- list(names=names_vec, centres=built$centres, whitening=built$whitening,
                 theta=built$theta, k=built$k, p=built$p)
  list(spec=list(type="block", name=name, design=built$design, dims=c(nrow(x),built$k),
                 penalty=built$penalty, df=if (lambda < 0) df else NULL,
                 center=as.logical(args$named$center %||% FALSE), radial_basis=recipe),
       design=built$design, name=name, ncol=built$k, centering=rep(0,built$k))
}

#' Parse the explicit mboostrs kernel-ridge learner
#' @noRd
.parse_bkernel <- function(args_str, data) {
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  x <- as.double(data[[var_name]])
  if (is.null(x) || length(x) == 0L) stop(.missing_column_message(var_name, data))
  bandwidth <- as.double(args$named$bandwidth %||% 1.0)
  df <- as.double(args$named$df %||% 4.0)
  kernel <- as.character(args$named$kernel %||% "gaussian")
  center <- as.logical(args$named$center %||% FALSE)
  landmarks <- if (is.null(args$named$landmarks)) NULL else as.integer(args$named$landmarks)
  if (!is.null(landmarks)) {
    if (length(landmarks) == 0L || is.na(landmarks) || landmarks < 1L)
      stop("bkernel landmarks must be a positive integer or NULL")
  }
  if (!is.finite(bandwidth) || bandwidth <= 0.0)
    stop("bkernel bandwidth must be finite and positive")
  if (!is.finite(df) || df <= 0.0)
    stop("bkernel df must be finite and positive")
  if (!(kernel %in% c("gaussian", "epanechnikov", "uniform")))
    stop("bkernel kernel must be gaussian, epanechnikov, or uniform")
  request <- list(type = "bkernel", name = var_name, x = x,
                  bandwidth = bandwidth, df = df, kernel = kernel,
                  center = center)
  if (!is.null(landmarks))
    request$landmarks <- landmarks
  list(
    spec = request,
    design = x, name = var_name, ncol = 1L, centering = 0.0
  )
}

#' Parse a caller-supplied design and penalty learner
#' @noRd
.parse_buser <- function(args_str, data, enclos = parent.frame()) {
  args <- .parse_bl_args(args_str)
  learner_name <- args$positional[1]
  x_expr <- args$named$X
  k_expr <- args$named$K
  if (is.null(x_expr)) stop("buser() requires X=<design matrix>")
  if (is.null(k_expr)) stop("buser() requires K=<penalty matrix>")
  X <- tryCatch(
    as.matrix(eval(str2lang(x_expr), envir = as.list(data), enclos = enclos)),
    error = function(error) {
      stop("could not resolve buser X expression '", x_expr, "': ",
           conditionMessage(error))
    }
  )
  K <- tryCatch(
    as.matrix(eval(str2lang(k_expr), envir = as.list(data), enclos = enclos)),
    error = function(error) {
      stop("could not resolve buser K expression '", k_expr, "': ",
           conditionMessage(error))
    }
  )
  if (!is.numeric(X) || nrow(X) != nrow(data) || ncol(X) < 1L ||
      any(!is.finite(X))) {
    stop("buser X must be a finite numeric matrix with one row per observation")
  }
  if (!is.numeric(K) || nrow(K) != ncol(X) || ncol(K) != ncol(X) ||
      any(!is.finite(K))) {
    stop("buser K must be a finite square matrix with one row per X column")
  }
  df <- as.double(args$named$df %||% 4.0)
  lambda <- args$named$lambda
  center <- as.logical(args$named$center %||% FALSE)
  if (!is.finite(df) || df <= 0.0) stop("buser df must be finite and positive")
  if (!is.null(lambda) && (!is.finite(as.double(lambda)) || as.double(lambda) < 0.0))
    stop("buser lambda must be finite and non-negative")
  list(
    spec = list(type = "buser", name = learner_name, X = X,
                dims = as.integer(dim(X)), K = K,
                K_dims = as.integer(dim(K)), df = df,
                lambda = if (is.null(lambda)) -1.0 else as.double(lambda),
                center = center),
    design = as.double(t(X)), name = learner_name, ncol = ncol(X),
    centering = if (center) colMeans(X) else rep(0.0, ncol(X))
  )
}

#' Parse bbs() arguments
#' @noRd
.parse_bbs <- function(args_str, data) {
  # Parse: x, df=4, degree=3, differences=2, center=TRUE, lambda=NULL, deriv=0
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  x <- as.double(data[[var_name]])
  if (is.null(x) || length(x) == 0) stop(.missing_column_message(var_name, data))

  df <- as.double(args$named$df %||% 4)
  degree <- as.integer(args$named$degree %||% 3)
  differences <- as.integer(args$named$differences %||% 2)
  # mboost's bbs default is center = FALSE; match it.
  center <- as.logical(args$named$center %||% FALSE)
  knots <- as.integer(args$named$knots %||% 20)   # inner knots (mboost default)
  lambda <- args$named$lambda  # NULL means use df
  deriv <- as.integer(args$named$deriv %||% 0)
  knot_placement <- gsub("[\"']", "", as.character(args$named$knot_placement %||% "equidistant"))
  if (!knot_placement %in% c("equidistant", "quantile")) {
    stop("bbs knot_placement must be 'equidistant' or 'quantile'", call. = FALSE)
  }
  bins <- args$named$bins
  if (!is.null(bins)) {
    bins <- as.integer(bins)
    if (is.na(bins) || bins < 1L) stop("bbs bins must be a positive integer", call. = FALSE)
  }

  # Boundary knots: user-specified or training data range
  bk_str <- args$named$boundary.knots
  if (!is.null(bk_str)) {
    # Parse "c(0, 10)" -> c(0, 10)
    bk_vals <- tryCatch(
      eval(parse(text = bk_str)),
      error = function(e) NULL
    )
    if (!is.null(bk_vals) && length(bk_vals) == 2) {
      bk_lo <- as.double(bk_vals[1])
      bk_hi <- as.double(bk_vals[2])
    } else {
      bk_lo <- min(x)
      bk_hi <- max(x)
    }
  } else {
    bk_lo <- min(x)
    bk_hi <- max(x)
  }

  # Build the design with the CORE B-spline machinery (the same basis the fit
  # uses via prepare_bbs_with_lambda / bbs_ospline), rather than splines::bs.
  knots_vec <- NULL
  if (identical(knot_placement, "quantile")) {
    knots_vec <- .Call("wrap__r_bbs_ospline_knots", x, bk_lo, bk_hi, knots,
                       degree, PACKAGE = "mboostrs")
    bd <- .Call("wrap__r_bbs_ospline_design", x, as.double(knots_vec), bk_lo, bk_hi,
                degree, as.integer(deriv), isTRUE(center), PACKAGE = "mboostrs")
  } else {
    bd <- .Call("wrap__r_bbs_design", x, bk_lo, bk_hi, knots, degree,
                as.integer(deriv), PACKAGE = "mboostrs")
  }
  design <- bd$design          # observation-major flat, length n*p
  p <- bd$p

  # Type "bbs": carries the core design AND the raw spline parameters
  list(
    spec = list(
      type = "bbs",
      name = var_name,
      center = center,
      x = x,
      knots = as.double(knots),
      degree = as.double(degree),
      differences = as.double(differences),
      df = df,
      lambda = if (!is.null(lambda)) as.double(lambda) else -1,
      deriv = deriv,
      bk_lo = bk_lo,
      bk_hi = bk_hi,
      knot_placement = knot_placement,
      knots_vec = knots_vec,
      bins = bins
    ),
    design = design,
    name = var_name,
    ncol = p,
    centering = rep(0, p)
  )
}

#' Parse bols() arguments
#' @noRd
.parse_bols <- function(args_str, data, levels = NULL) {
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  raw <- data[[var_name]]
  if (is.null(raw) || length(raw) == 0) stop(.missing_column_message(var_name, data))
  intercept <- as.logical(args$named$intercept %||% TRUE)
  if (is.factor(raw) || is.character(raw) || is.logical(raw)) {
    return(.bols_factor_spec(var_name, raw, intercept, args$named$lambda,
                             args$named$df, levels))
  }
  x <- as.double(raw)
  # `center=` is retained as a compatibility alias for the historical
  # mboostrs R frontend. The upstream bols option is `intercept=`.
  center <- as.logical(args$named$center %||% intercept)
  lambda <- args$named$lambda
  df <- args$named$df

  # `bols(x, intercept=TRUE)` has two engine forms, and which one is correct
  # depends on whether the learner is penalized.
  #
  # Unpenalized, `gamboost` puts the constant in the model offset and fits one
  # weight-centered column. That is exactly equivalent: a constant is available
  # either way, from the learner or from the offset.
  #
  # Penalized, it is not. R's `bols(x, lambda = 1)` shrinks the learner's own
  # intercept along with its slope, and a constant supplied by the offset is
  # not shrinkable -- so the centered single column fits a different model.
  # This frontend sent the centered form regardless, which is why a penalized
  # `bols` diverged from `mboost` while every unpenalized one matched: on a
  # ten-row Gaussian fixture with `bols(x, lambda = 1)`, `fitted()[1]` was
  # -0.2348387 against mboost's -0.2317270, with the gap opening at the first
  # iteration and growing with mstop. Python's spec builder has carried this
  # rule since the compact learner boundary was written; R had not.
  penalized <- !is.null(lambda) || !is.null(df)

  if (penalized && intercept) {
    if (!is.null(df)) {
      # The block spec carries a penalty matrix, not a df, and the R frontend
      # has no df -> lambda calibration for it. Refuse rather than silently
      # fitting the wrong penalty, which is what sending the centered scalar
      # form amounted to.
      stop("bols(", var_name, ", df = ", df, ") is not supported by the R ",
           "frontend: a df-calibrated bols needs the block form with a ",
           "solved lambda, which this binding does not build. Give lambda= ",
           "directly, or use the Python frontend, which does calibrate it.",
           call. = FALSE)
    }
    # The two-column block `[1, x]` with a ridge penalty on both coefficients,
    # matching `mboost::bols(x, lambda = ...)`. The block spec takes the
    # penalty as-is -- there is no separate lambda field -- so lambda is folded
    # into it here.
    design <- cbind(rep(1.0, length(x)), x)
    return(list(
      spec = list(
        type = "block",
        name = var_name,
        design = as.double(t(design)),
        penalty = as.double(as.double(lambda) * diag(2)),
        dims = c(length(x), 2L),
        center = FALSE,
        bols_block = TRUE,
        intercept = TRUE
      ),
      # Row-major, as every reader of `learner_designs` takes it.
      design = as.double(t(design)),
      name = var_name,
      ncol = 2L,
      centering = 0.0
    ))
  }

  # bols is a scalar (1-column) learner
  list(
    spec = list(
      type = "scalar",
      name = var_name,
      design = x,
      lambda = if (!is.null(lambda)) as.double(lambda) else -1,
      df = if (!is.null(df)) as.double(df) else -1,
      intercept = intercept,
      center = center
    ),
    design = x,
    name = var_name,
    ncol = 1L,
    centering = if (center) mean(x) else 0.0
  )
}

# `bols(f)` for a factor: mboost's design, the treatment-contrast model
# matrix -- an intercept and one indicator per non-reference level, the
# intercept column dropped with `intercept = FALSE` -- fitted as one block. This
# frontend read the column with as.double(), so a factor became a single
# slope on its integer codes (fitted values 0.078 from mboost on a four-level
# example). `levels` are the training levels when a design is rebuilt for new
# data.
.bols_factor_design <- function(raw, levels, intercept, var_name, unseen = "error") {
  f <- factor(as.character(raw), levels = levels)
  new <- is.na(f) & !is.na(raw)
  if (any(new) && identical(unseen, "error")) {
    stop("bols(", var_name, ") has levels the fit did not see: ",
         paste(setdiff(unique(as.character(raw)), levels), collapse = ", "),
         ". Pass unseen_levels = \"average\" or \"zero_effect\" to predict them.",
         call. = FALSE)
  }
  indicators <- vapply(levels, function(l) as.double(f == l), numeric(length(f)))
  indicators <- matrix(indicators, nrow = length(f))
  indicators[is.na(indicators)] <- 0
  # An unseen level: the mean of the level rows under "average", the
  # reference row (no level effect) under "zero_effect".
  if (any(new) && identical(unseen, "average")) indicators[new, ] <- 1 / length(levels)
  # mboost's `intercept = FALSE` drops the intercept column of the same
  # treatment-contrast matrix; the reference level stays the zero row.
  contrasts <- indicators[, -1, drop = FALSE]
  if (intercept) cbind(1.0, contrasts) else contrasts
}

.bols_factor_spec <- function(var_name, raw, intercept, lambda, df, levels = NULL) {
  if (!is.null(df)) {
    stop("bols(", var_name, ", df = ", df, ") on a factor is not supported by ",
         "the R frontend, which has no df -> lambda calibration for a block ",
         "learner. Give lambda= directly, or use the Python frontend.",
         call. = FALSE)
  }
  levels <- levels %||% (if (is.factor(raw)) levels(droplevels(raw)) else
                           sort(unique(as.character(raw))))
  design <- .bols_factor_design(raw, levels, intercept, var_name)
  p <- ncol(design)
  penalty <- if (is.null(lambda)) matrix(0.0, p, p) else as.double(lambda) * diag(p)
  list(
    spec = list(
      type = "block",
      name = var_name,
      design = as.double(t(design)),
      penalty = as.double(penalty),
      dims = c(nrow(design), p),
      center = FALSE,
      factor_levels = levels,
      intercept = intercept
    ),
    design = as.double(t(design)),
    name = var_name,
    ncol = p,
    centering = 0.0
  )
}

#' Parse btree() arguments
#' @noRd
.parse_btree <- function(args_str, data) {
  args <- .parse_bl_args(args_str)
  cols <- unlist(args$positional)
  for (col in cols) {
    if (is.null(data[[col]])) stop(.missing_column_message(col, data))
  }
  # A factor or character column is coded 0..L-1 by its training levels; a
  # tree splits it into two sets of levels.
  levels <- lapply(cols, function(col) {
    v <- data[[col]]
    if (is.factor(v) || is.character(v)) levels(as.factor(v)) else NULL
  })
  sp_levels <- list(names = cols, levels = levels)
  design <- .tree_design(sp_levels, data)

  minbucket <- as.integer(args$named$minbucket %||% 4L)
  minsplit <- as.integer(args$named$minsplit %||% 10L)
  maxdepth <- as.integer(args$named$maxdepth %||% 1L)
  selection <- gsub("[\"']", "", as.character(args$named$selection %||% "ctree"))
  if (!selection %in% c("sse", "ctree", "ctree_pvalue")) {
    stop("btree selection must be 'sse', 'ctree' or 'ctree_pvalue'")
  }
  split_method <- gsub("[\"']", "", as.character(args$named$split_method %||% "exact"))
  if (!split_method %in% c("exact", "histogram")) {
    stop("btree split_method must be 'exact' or 'histogram'", call. = FALSE)
  }
  lambda_leaf <- as.double(args$named$lambda_leaf %||% 0.0)
  if (!is.finite(lambda_leaf) || lambda_leaf < 0.0) {
    stop("btree lambda_leaf must be a finite non-negative number", call. = FALSE)
  }
  colsample <- as.double(args$named$colsample %||% 1.0)
  if (!is.finite(colsample) || colsample <= 0.0 || colsample > 1.0) {
    stop("btree colsample must be in (0, 1]", call. = FALSE)
  }
  colsample_seed <- as.integer(args$named$colsample_seed %||% 0L)
  if (is.na(colsample_seed) || colsample_seed < 0L) {
    stop("btree colsample_seed must be a non-negative integer", call. = FALSE)
  }
  n <- nrow(data)

  list(
    spec = list(
      type = "tree",
      name = paste(cols, collapse = ", "),
      names = cols,
      levels = levels,
      categorical = vapply(levels, function(l) if (is.null(l)) 0L else length(l), integer(1)),
      design = design,
      dims = c(n, length(cols)),
      minbucket = minbucket,
      minsplit = minsplit,
      maxdepth = maxdepth,
      selection = selection,
      split_method = split_method,
      lambda_leaf = lambda_leaf,
      colsample = colsample,
      colsample_seed = colsample_seed
    ),
    design = design,
    name = paste(cols, collapse = ", "),
    ncol = length(cols),
    centering = numeric(0)
  )
}

# A tree learner's row-major (n, q) design at the rows of `source_data`:
# numeric columns as they are, factor columns as their training-level codes,
# -1 for a level the fit never saw (the tree routes it to the child that held
# more training weight).
.tree_design <- function(sp, source_data) {
  cols <- sp$names %||% sp$name
  mat <- vapply(seq_along(cols), function(j) {
    v <- source_data[[cols[j]]]
    lv <- sp$levels[[j]]
    if (is.null(lv)) return(as.double(v))
    codes <- match(as.character(v), lv) - 1L
    codes[is.na(codes)] <- -1L
    as.double(codes)
  }, numeric(nrow(as.data.frame(source_data))))
  if (is.null(dim(mat))) mat <- matrix(mat, ncol = length(cols))
  as.double(t(mat))
}

# Whether a tree's row-major design holds a level the fit never saw: a
# factor column (one with recorded levels) coded -1.
.tree_has_unseen <- function(sp, design) {
  factor_columns <- which(vapply(sp$levels %||% list(), Negate(is.null), logical(1)))
  if (!length(factor_columns)) return(FALSE)
  q <- length(sp$names %||% sp$name)
  mat <- matrix(design, ncol = q, byrow = TRUE)
  any(mat[, factor_columns, drop = FALSE] < 0)
}

# The modifier of btree(..., by = z) at the training rows, or at the rows of
# `newdata` (a two-level factor coded 0/1 against the training levels).
.tree_by <- function(sp, newdata) {
  if (is.null(newdata)) return(as.double(sp$by_values))
  z <- newdata[[sp$by_name]]
  if (is.null(z)) stop(.missing_column_message(sp$by_name, newdata))
  if (!is.null(sp$by_levels)) {
    codes <- match(as.character(z), sp$by_levels) - 1L
    if (anyNA(codes)) {
      stop("btree(by = ", sp$by_name, "): newdata has levels the fit did not see.",
           call. = FALSE)
    }
    return(as.double(codes))
  }
  as.double(z)
}

#' Parse bmono() arguments
#' @noRd
.parse_bmono <- function(args_str, data) {
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  x <- as.double(data[[var_name]])
  if (is.null(x) || length(x) == 0) stop(.missing_column_message(var_name, data))

  df <- as.double(args$named$df %||% 4)
  degree <- as.integer(args$named$degree %||% 3)
  differences <- as.integer(args$named$differences %||% 2)
  constraint <- args$named$constraint %||% "increasing"
  # The solver is spelled `type=` by both upstream mboost and the mboostrs
  # Python frontend; `solver=` is the older mboostrs-R spelling and stays
  # accepted. Reading only `solver=` meant a `type=` written by anyone
  # following mboost's own documentation was silently dropped and the caller
  # got a solver they did not ask for without being told.
  if (!is.null(args$named$type) && !is.null(args$named$solver) &&
      !identical(args$named$type, args$named$solver)) {
    stop("bmono(): `type` and `solver` both given with different values ('",
         args$named$type, "' vs '", args$named$solver,
         "'); supply only one (`type` is the mboost spelling)")
  }
  solver <- args$named$type %||% args$named$solver %||% "quad.prog"
  lambda2 <- as.double(args$named$lambda2 %||% 1e6)
  niter <- as.integer(args$named$niter %||% 10L)
  center <- as.logical(args$named$center %||% FALSE)
  knots <- as.integer(args$named$knots %||% 20)   # inner knots (mboost default)

  # Boundary knots are the TRAINING range; prediction rebuilds the basis against
  # them. The constraint shapes the coefficients, so the prediction *design* is
  # the same raw B-spline basis as bbs (built by the core via r_bbs_design).
  bk_lo <- min(x)
  bk_hi <- max(x)
  bd <- .Call("wrap__r_bbs_design", x, bk_lo, bk_hi, knots, degree, 0L, PACKAGE = "mboostrs")
  design <- bd$design
  p <- bd$p

  list(
    spec = list(
      type = "bmono",
      name = var_name,
      x = x,
      knots = as.double(knots),
      degree = as.double(degree),
      differences = as.double(differences),
      df = df,
      solver = solver,
      lambda2 = lambda2,
      niter = niter,
      constraint = constraint,
      center = center,
      bk_lo = bk_lo,
      bk_hi = bk_hi
    ),
    design = design,
    name = var_name,
    ncol = p,
    centering = rep(0, p)
  )
}

#' Parse bconstrained() arguments
#' @noRd
.parse_bconstrained <- function(args_str, data, enclos = parent.frame()) {
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  x <- as.double(data[[var_name]])
  if (is.null(x) || length(x) == 0) stop(.missing_column_message(var_name, data))

  knots <- as.integer(args$named$knots %||% 20)
  degree <- as.integer(args$named$degree %||% 3)
  differences <- as.integer(args$named$differences %||% 2)
  df <- as.double(args$named$df %||% 4)
  lambda <- args$named$lambda
  lambda2 <- as.double(args$named$lambda2 %||% 1e6)
  niter <- as.integer(args$named$niter %||% 10L)
  c_expr <- args$named$C
  if (is.null(c_expr)) stop("bconstrained() requires C=<constraint matrix>")
  C <- tryCatch(
    as.matrix(eval(str2lang(c_expr), envir = as.list(data), enclos = enclos)),
    error = function(error) {
      stop("could not resolve bconstrained C expression '", c_expr, "': ",
           conditionMessage(error))
    }
  )
  p <- knots + degree + 1L
  if (!is.numeric(C) || nrow(C) < 1L || ncol(C) != p || any(!is.finite(C))) {
    stop("bconstrained C must be a non-empty finite matrix with ", p, " columns")
  }
  bk_lo <- min(x)
  bk_hi <- max(x)
  bd <- .Call("wrap__r_bbs_design", x, bk_lo, bk_hi, knots, degree, 0L,
              PACKAGE = "mboostrs")

  list(
    spec = list(
      type = "bconstrained", name = var_name, x = x, C = C,
      C_dims = as.integer(dim(C)),
      knots = as.double(knots), degree = as.double(degree),
      differences = as.double(differences), df = df,
      lambda = if (!is.null(lambda)) as.double(lambda) else -1,
      lambda2 = lambda2, niter = niter,
      # "quad.prog" (exact, the default) or "iterative", as for bmono.
      solver = gsub("[\"' ]", "", args$named$type %||% "quad.prog"),
      bk_lo = bk_lo, bk_hi = bk_hi
    ),
    design = bd$design,
    name = var_name,
    ncol = bd$p,
    centering = rep(0, bd$p)
  )
}

#' Parse bspatial() arguments
#' @noRd
.parse_bspatial <- function(args_str, data) {
  args <- .parse_bl_args(args_str)
  var_name1 <- args$positional[1]
  var_name2 <- args$positional[2]
  if (is.null(var_name1) || is.na(var_name1)) {
    stop("bspatial() requires two positional arguments (x1, x2)")
  }
  if (is.null(var_name2) || is.na(var_name2)) {
    stop("bspatial() requires two positional arguments (x1, x2)")
  }
  x1 <- as.double(data[[var_name1]])
  x2 <- as.double(data[[var_name2]])
  if (is.null(x1) || length(x1) == 0) stop("Variable not found: ", var_name1)
  if (is.null(x2) || length(x2) == 0) stop("Variable not found: ", var_name2)

  df <- as.double(args$named$df %||% 6)
  degree <- as.integer(args$named$degree %||% 3)
  differences <- as.integer(args$named$differences %||% 2)
  # mboost's defaults (and Python's): 20 inner knots per margin, center =
  # FALSE. This frontend used 10 and TRUE, so a default bspatial fitted a
  # different, smaller and null-space-free surface.
  center <- as.logical(args$named$center %||% FALSE)
  knots <- as.integer(args$named$knots %||% 20)
  lambda <- args$named$lambda

  # Boundary knots from training data
  bk1_lo <- min(x1); bk1_hi <- max(x1)
  bk2_lo <- min(x2); bk2_hi <- max(x2)

  # Build design via Rust
  bd <- .Call("wrap__r_bspatial_design", x1, x2,
              bk1_lo, bk1_hi, bk2_lo, bk2_hi,
              knots, degree, differences, center,
              PACKAGE = "mboostrs")

  list(
    spec = list(
      type = "bspatial",
      name = paste0(var_name1, ",", var_name2),
      name1 = var_name1,
      name2 = var_name2,
      x1 = x1,
      x2 = x2,
      knots = as.double(knots),
      degree = as.double(degree),
      differences = as.double(differences),
      df = df,
      lambda = if (!is.null(lambda)) as.double(lambda) else -1,
      center = center,
      bk1_lo = bk1_lo, bk1_hi = bk1_hi,
      bk2_lo = bk2_lo, bk2_hi = bk2_hi
    ),
    design = bd$design,
    name = paste0(var_name1, ",", var_name2),
    ncol = bd$p,
    centering = rep(0, bd$p)
  )
}

#' Parse brandom() arguments
#' @noRd
.parse_brandom <- function(args_str, data) {
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  raw <- data[[var_name]]
  if (is.null(raw) || length(raw) == 0) stop(.missing_column_message(var_name, data))
  # A character or logical column is a factor whose levels are its sorted
  # values; its labels are recorded below, so new data is coded against them.
  if (is.character(raw) || is.logical(raw)) {
    raw <- factor(raw)
    data[[var_name]] <- raw
  }
  x <- as.integer(raw)

  df <- as.double(args$named$df %||% 4)
  lambda <- args$named$lambda
  center <- as.logical(args$named$center %||% FALSE)
  by_name <- args$named$by

  # Build indicator design in row-major order (observation-major flat vector)
  n <- length(x)
  n_levels <- max(x)
  design <- double(n * n_levels)
  for (i in seq_len(n)) {
    design[(i - 1L) * n_levels + x[i]] <- 1.0
  }

  list(
    spec = list(
      type = "brandom",
      name = var_name,
      x = as.double(x),
      design = design,
      dims = c(n, n_levels),
      n_levels = n_levels,
      ncol = n_levels,
      df = df,
      lambda = if (!is.null(lambda)) as.double(lambda) else -1,
      center = center,
      by = by_name,
      by_values = if (is.null(by_name)) NULL else as.double(data[[by_name]]),
      # The labels behind the codes, so new data is coded against them.
      levels = if (is.factor(data[[var_name]])) levels(data[[var_name]]) else NULL
    ),
    design = design,
    name = var_name,
    ncol = n_levels,
    centering = rep(0, n_levels)
  )
}

#' Parse bmrf() arguments
#' @noRd
.parse_bmrf <- function(args_str, data, enclos = parent.frame()) {
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  if (is.null(data[[var_name]]) || length(data[[var_name]]) == 0) {
    stop(.missing_column_message(var_name, data))
  }
  if (is.character(data[[var_name]])) data[[var_name]] <- factor(data[[var_name]])
  x <- as.integer(data[[var_name]])

  df <- as.double(args$named$df %||% 4)
  lambda <- args$named$lambda
  center <- as.logical(args$named$center %||% FALSE)
  by_name <- args$named$by

  n <- length(x)
  n_levels <- if (is.factor(data[[var_name]])) nlevels(data[[var_name]]) else max(x)
  levels <- x - 1L  # R factor/region codes are 1-based; Rust uses 0-based codes.
  k_expr <- args$named$K %||% args$named$bnd
  K <- if (is.null(k_expr)) {
    NULL
  } else {
    value <- tryCatch(
      eval(str2lang(k_expr), envir = as.list(data), enclos = enclos),
      error = function(error) {
        stop("could not resolve bmrf K/bnd expression '", k_expr, "': ",
             conditionMessage(error))
      }
    )
    as.matrix(value)
  }
  if (!is.null(K) && !identical(dim(K), c(n_levels, n_levels))) {
    stop("bmrf K must have shape ", n_levels, " x ", n_levels)
  }

  list(
    spec = list(
      type = "bmrf",
      name = var_name,
      levels = levels,
      n_levels = n_levels,
      ncol = n_levels,
      K = K,
      df = df,
      lambda = if (!is.null(lambda)) as.double(lambda) else -1,
      center = center,
      by = by_name,
      # The region labels behind the codes, so new data is coded against them.
      # (`levels` above holds the 0-based codes the core reads.)
      level_labels = if (is.factor(data[[var_name]])) levels(data[[var_name]]) else NULL
    ),
    design = as.double(levels),
    name = var_name,
    ncol = n_levels,
    centering = rep(0, n_levels)
  )
}

#' Parse bns() arguments
#' @noRd
.parse_bns <- function(args_str, data) {
  args <- .parse_bl_args(args_str)
  var_name <- args$positional[1]
  x <- as.double(data[[var_name]])
  if (is.null(x) || length(x) == 0) stop(.missing_column_message(var_name, data))

  knots <- as.integer(args$named$knots %||% 20)
  degree <- as.integer(args$named$degree %||% 3)
  df <- args$named$df
  center <- as.logical(args$named$center %||% FALSE)
  by_name <- args$named$by

  bk_lo <- min(x)
  bk_hi <- max(x)

  bd <- .Call("wrap__r_bns_design", x, bk_lo, bk_hi, knots, degree,
              PACKAGE = "mboostrs")
  design <- bd$design
  p <- bd$p

  list(
    spec = list(
      type = "bns",
      name = var_name,
      x = x,
      knots = as.double(knots),
      degree = as.double(degree),
      df = if (!is.null(df)) as.double(df) else as.double(p),
      center = center,
      bk_lo = bk_lo,
      bk_hi = bk_hi,
      by = by_name
    ),
    design = design,
    name = var_name,
    ncol = p,
    centering = rep(0, p)
  )
}

#' Build a difference penalty matrix of given order
#' @noRd
.build_diff_penalty <- function(p, differences = 2) {
  D <- diag(p)
  for (d in seq_len(differences)) {
    D <- diff(D)
  }
  K <- crossprod(D)
  K
}

#' Build monotone constraint matrix D
#' @noRd
.build_mono_D <- function(p, constraint = "increasing") {
  D <- diff(diag(p))
  if (constraint == "decreasing") {
    D <- -D
  }
  D
}

#' Simple argument parser for base-learner calls
#'
#' Splits "x, df=4, degree=3" into positional and named args.
#' @noRd
.parse_bl_args <- function(args_str) {
  # Tokenize by comma respecting parentheses depth (handles c(0, 10) etc.)
  # and backticked names (a comma or parenthesis inside a name belongs to it).
  parts <- character(0)
  current <- character(0)
  depth <- 0L
  in_name <- FALSE
  chars <- strsplit(args_str, "")[[1]]
  i <- 1L
  while (i <= length(chars)) {
    ch <- chars[i]
    if (in_name) {
      current <- c(current, ch)
      if (ch == "\\" && i < length(chars)) {
        i <- i + 1L
        current <- c(current, chars[i])
      } else if (ch == "`") {
        in_name <- FALSE
      }
    } else if (ch == "`") {
      in_name <- TRUE
      current <- c(current, ch)
    } else if (ch == "(" || ch == "[") {
      depth <- depth + 1L
      current <- c(current, ch)
    } else if (ch == ")" || ch == "]") {
      depth <- depth - 1L
      current <- c(current, ch)
    } else if (ch == "," && depth == 0L) {
      parts <- c(parts, paste(current, collapse = ""))
      current <- character(0)
    } else {
      current <- c(current, ch)
    }
    i <- i + 1L
  }
  if (length(current) > 0) {
    parts <- c(parts, paste(current, collapse = ""))
  }

  positional <- character(0)
  named <- list()
  ticked <- character(0)

  for (part in parts) {
    part <- trimws(part)
    # The first "=" outside backticks separates a name from its value.
    eq_pos <- regexpr("=", .mask_backticks(part), fixed = TRUE)
    if (eq_pos > 0) {
      key <- trimws(substr(part, 1, eq_pos - 1))
      val <- trimws(substr(part, eq_pos + 1, nchar(part)))
      if (startsWith(val, "`")) {
        # A column name in backticks, as in by = `my group`.
        val <- .unquote_name(val)
        ticked <- c(ticked, key)
      } else {
        # Remove surrounding quotes
        val <- gsub('^"|"$', "", val)
      }
      named[[key]] <- val
    } else {
      positional <- c(positional, .unquote_name(part))
    }
  }

  list(positional = positional, named = named, ticked = ticked)
}

# `x` with every backticked name (backticks included) replaced by "x"
# characters, so that an operator, comma or parenthesis that belongs to a name
# is not read as syntax. The result has as many characters as `x`.
.mask_backticks <- function(x) {
  chars <- strsplit(x, "")[[1]]
  out <- chars
  i <- 1L
  n <- length(chars)
  while (i <= n) {
    if (chars[i] == "`") {
      j <- i + 1L
      while (j <= n && chars[j] != "`") j <- j + (if (chars[j] == "\\") 2L else 1L)
      end <- min(j, n)
      out[i:end] <- "x"
      i <- end + 1L
    } else {
      i <- i + 1L
    }
  }
  paste(out, collapse = "")
}

# ─── gamboost S3 methods ───────────────────────────────────────────

# The x1 design of each Kronecker learner of a Multinomial fit at the rows of
# `source_data`: row-major (n, p1) vectors and their column counts.
.multinomial_x1_designs <- function(object, source_data) {
  n_learners <- length(object$learner_specs)
  x1_designs <- vector("list", n_learners)
  x1_ncol <- integer(n_learners)
  for (i in seq_len(n_learners)) {
    sp <- object$learner_specs[[i]]
    if (identical(sp$type, "scalar")) {
      xn <- as.double(source_data[[sp$name]])
      if (isTRUE(sp$center)) {
        x1_designs[[i]] <- as.double(rbind(1, xn))  # row-major (n, 2)
        x1_ncol[i] <- 2L
      } else {
        x1_designs[[i]] <- xn
        x1_ncol[i] <- 1L
      }
    } else if (identical(sp$type, "bbs")) {
      xn <- as.double(source_data[[sp$name]])
      deriv <- as.integer(sp$deriv %||% 0)
      # A centred term replays the fit's reparametrization.
      bd <- if (isTRUE(sp$center)) {
        .Call("wrap__r_bbs_design_centered", xn, sp$bk_lo, sp$bk_hi,
              as.integer(sp$knots), as.integer(sp$degree),
              as.integer(sp$differences %||% 2), deriv, PACKAGE = "mboostrs")
      } else {
        .Call("wrap__r_bbs_design", xn, sp$bk_lo, sp$bk_hi,
              as.integer(sp$knots), as.integer(sp$degree),
              deriv, PACKAGE = "mboostrs")
      }
      x1_designs[[i]] <- bd$design
      x1_ncol[i] <- bd$p
    } else if (identical(sp$type, "brandom")) {
      # Level indicators, coded against the training labels; a label the
      # fit never saw gets a zero row, the average level's effect.
      raw <- source_data[[sp$name]]
      codes <- if (!is.null(sp$levels)) match(as.character(raw), sp$levels) else as.integer(raw)
      n_new <- length(codes)
      design_mat <- matrix(0.0, nrow = n_new, ncol = sp$n_levels)
      for (j in seq_len(n_new)) {
        if (!is.na(codes[j]) && codes[j] >= 1L && codes[j] <= sp$n_levels)
          design_mat[j, codes[j]] <- 1.0
      }
      x1_designs[[i]] <- as.double(t(design_mat))
      x1_ncol[i] <- as.integer(sp$n_levels)
    } else {
      stop("Multinomial predict with newdata/mstop does not support learner type: ", sp$type)
    }
  }
  list(designs = x1_designs, ncol = x1_ncol)
}

#' Predict from a fitted GAMBoost model
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param newdata Optional new data frame.
#' @param mstop Number of iterations to predict at.
#' @param type "link", "response", or "class" (a factor with the training
#'   response's labels for Binomial, AdaExp, HingeLoss and AUC). A tied score
#'   belongs to the first level.
#' @param offset The new rows' offset, one value per row, required with
#'   `newdata` when the model was fitted with a per-row offset
#'   (`log(exposure)`); `NULL` reuses the fitted offset.
#' @param which Learners (names or positions) whose contributions to return.
#'   With \code{which}, the result is a matrix with one column per learner
#'   holding its summed contributions on the link scale, without the
#'   offset, as \code{mboost}'s \code{predict(which = )} returns.
#' @param unseen_levels What a term over a categorical column does with a
#'   level in \code{newdata} the fit never saw. \code{"error"} (default)
#'   refuses. \code{"average"} predicts the row as a member of an average
#'   group: each term that reads the column (\code{brandom}, \code{bmrf},
#'   \code{bols} over a factor, a factor \code{by=}, a \code{btree} split)
#'   contributes the mean of its contributions over the fitted levels.
#'   \code{"zero_effect"} gives the level no effect of its own (a
#'   \code{btree} sends it to the branch that held more training weight).
#'   New data are coded against the training labels, so a factor holding a
#'   subset of the levels is read correctly.
#' @param extrapolation What to do when \code{newdata} holds values of a
#'   spline learner's variable (\code{bbs}, \code{bmono}, \code{bns},
#'   \code{bspatial}) outside the range the basis was fitted on. A spline
#'   basis continues linearly by its edge slope there, and nothing in the fit
#'   constrains that continuation. \code{"warn"} (default) predicts and warns
#'   once, naming each learner, how many rows fall outside and the worst
#'   distance in training ranges, when some row is at least 5\% of the
#'   training range outside. \code{"error"} refuses any value strictly
#'   outside. \code{"clip"} evaluates the basis at the nearest training
#'   boundary, so the effect is held flat outside the range. \code{"allow"}
#'   predicts silently. Linear and categorical terms are not checked.
#' @param ... Not used.
#' @return A numeric vector of predicted values, or a factor for `type = "class"`.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' head(predict(mod))
#' newdat <- data.frame(x1 = rnorm(10), x2 = runif(10))
#' head(predict(mod, newdata = newdat))
#' @export
predict.mboostrs_gamboost <- function(object, newdata = NULL, mstop = NULL,
                                      type = c("link", "response", "class"),
                                      offset = NULL,
                                      extrapolation = c("warn", "error", "clip", "allow"),
                                      unseen_levels = c("error", "zero_effect", "average"),
                                      which = NULL,
                                      ...) {
  .refuse_unused("predict", ...)
  type <- match.arg(type)
  if (type == "class") {
    if (!is.null(which)) stop("type = 'class' requires the whole model", call. = FALSE)
    return(.class_prediction(object, predict(object, newdata = newdata,
        mstop = mstop, type = "link", offset = offset,
        extrapolation = extrapolation, unseen_levels = unseen_levels)))
  }
  extrapolation <- match.arg(extrapolation)
  unseen_levels <- match.arg(unseen_levels)
  mstop_val <- if (is.null(mstop)) 0L else as.integer(mstop)
  start <- .prediction_offset(object, newdata, offset)
  if (!is.null(which)) {
    if (isTRUE(object$family$name == "multinomial")) {
      stop("predict(which = ) is not available for the Multinomial family",
           call. = FALSE)
    }
    if (type != "link") {
      stop("predict(which = ) returns each learner's contribution on the link ",
           "scale; a response-scale value exists only for the whole model",
           call. = FALSE)
    }
  }

  # Multinomial: the fitted predictor is an observation-major (n*(K-1)) vector;
  # reshape to the (n, K-1) link matrix or the (n, K) softmax probability matrix
  # (reference class last).
  if (isTRUE(object$family$name == "multinomial")) {
    nr <- object$multinomial_n_classes - 1L
    # Fast path: in-sample, full mstop — use stored fitted values
    if (is.null(newdata) && (is.null(mstop) || mstop_val <= 0L ||
                             mstop_val >= length(object$selected))) {
      eta <- matrix(object$fitted, ncol = nr, byrow = TRUE)   # (n, K-1)
      if (type == "link") return(eta)
      ef <- exp(pmin(pmax(eta, -36), 36))
      denom <- 1 + rowSums(ef)
      probs <- cbind(ef / denom, 1 / denom)                   # (n, K), reference last
      if (!is.null(object$multinomial_levels)) colnames(probs) <- object$multinomial_levels
      return(probs)
    }
    # Build x1 designs for each KroneckerBols learner from newdata (or training data)
    built <- .multinomial_x1_designs(object, if (is.null(newdata)) object$data else newdata)
    x1_designs <- built$designs
    x1_ncol <- built$ncol
    mstop_predict <- if (mstop_val <= 0L) 0L else mstop_val
    eta <- .Call("wrap__predict_gamboost_kronecker",
      object$offset,
      object$selected,
      object$coefs,
      object$nu,
      x1_designs,
      as.integer(x1_ncol),
      as.integer(nr),
      as.integer(mstop_predict),
      PACKAGE = "mboostrs"
    )
    eta <- matrix(eta, ncol = nr, byrow = TRUE)               # (n, K-1)
    if (type == "link") return(eta)
    ef <- exp(pmin(pmax(eta, -36), 36))
    denom <- 1 + rowSums(ef)
    probs <- cbind(ef / denom, 1 / denom)                     # (n, K), reference last
    if (!is.null(object$multinomial_levels)) colnames(probs) <- object$multinomial_levels
    return(probs)
  }

  # In-sample at the full path the stored fitted values are the prediction.
  # A user offset= replaces the fitted offset, so that case takes the
  # accumulating path below, which starts from it.
  if (is.null(which) && is.null(newdata) && is.null(offset) &&
      (is.null(mstop) || mstop_val <= 0L || mstop_val >= length(object$selected))) {
    eta <- object$fitted
    if (type == "response") eta <- .gamboost_response(object, eta, mstop)
    return(eta)
  }

  if (is.null(newdata)) {
    # Use stored learner designs
    designs_list <- object$learner_designs
  } else {
    # Re-build designs from newdata
    spec_list <- .parse_gamboost_formula(object$formula, newdata,
                                         vector_offset = length(object$offset) > 1L,
                                         family_name = object$family$name, replay_specs = object$learner_specs)
    designs_list <- spec_list$learner_designs
    extrap <- .extrapolation_policy(extrapolation)
    # bbs designs must use the TRAINING boundary knots (mboost evaluates the basis
    # against the fitting range), not the newdata range, so rebuild them here.
    for (i in seq_along(object$learner_specs)) {
      sp <- object$learner_specs[[i]]
      if (!is.null(sp$by_factor)) {
        designs_list[[i]] <- as.double(t(.rebuild_block_design(sp, newdata, extrap,
                                                               unseen_levels)))
        next
      }
      # bbs, bmono, bconstrained and bspatial predict through the raw
      # B-spline basis,
      # evaluated against the TRAINING boundary knots.
      if (sp$type %in% c("bbs", "bmono", "bconstrained")) {
        xn <- extrap$check(as.double(newdata[[sp$name]]), sp$bk_lo, sp$bk_hi,
                           sp$name)
        deriv <- if (identical(sp$type, "bbs")) as.integer(sp$deriv %||% 0) else 0L
        # A `center = TRUE` fit removed the penalty's null space by a spectral
        # decomposition, so its coefficients live in a `p - d` space while the
        # raw basis has `p` columns. The transform depends only on the penalty,
        # not on the data, so applying it to a basis built at the new x values
        # lands in the same reduced space -- which is what makes new-data
        # prediction possible here at all. Rebuilding the raw basis instead is
        # what made this refuse.
        centered <- isTRUE(sp$center)
        bd <- if (identical(sp$type, "bbs") && identical(sp$knot_placement, "quantile") &&
                  !is.null(sp$knots_vec)) {
          .Call("wrap__r_bbs_ospline_design", xn, as.double(sp$knots_vec), sp$bk_lo, sp$bk_hi,
                as.integer(sp$degree), deriv, centered, PACKAGE = "mboostrs")
        } else if (centered) {
          .Call("wrap__r_bbs_design_centered", xn, sp$bk_lo, sp$bk_hi,
                as.integer(sp$knots), as.integer(sp$degree),
                as.integer(sp$differences %||% 2), deriv, PACKAGE = "mboostrs")
        } else {
          .Call("wrap__r_bbs_design", xn, sp$bk_lo, sp$bk_hi,
                as.integer(sp$knots), as.integer(sp$degree),
                deriv, PACKAGE = "mboostrs")
        }
        designs_list[[i]] <- bd$design
        # The guard below compares the coefficient length against this count,
        # so it has to be the rebuilt design's, not the training spec's.
        object$learner_ncol[[i]] <- bd$p
      } else if (identical(sp$type, "bspatial")) {
        xn1 <- extrap$check(as.double(newdata[[sp$name1]]), sp$bk1_lo, sp$bk1_hi,
                            sprintf("bspatial(%s, %s)", sp$name1, sp$name2), sp$name1)
        xn2 <- extrap$check(as.double(newdata[[sp$name2]]), sp$bk2_lo, sp$bk2_hi,
                            sprintf("bspatial(%s, %s)", sp$name1, sp$name2), sp$name2)
        bd <- .Call("wrap__r_bspatial_design", xn1, xn2,
                    sp$bk1_lo, sp$bk1_hi, sp$bk2_lo, sp$bk2_hi,
                    as.integer(sp$knots), as.integer(sp$degree),
                    as.integer(sp$differences), isTRUE(sp$center),
                    PACKAGE = "mboostrs")
        designs_list[[i]] <- bd$design
        # A centered surface has fewer columns than the raw tensor basis, and
        # the guard below compares against this count.
        object$learner_ncol[[i]] <- bd$p
      } else if (identical(sp$type, "bns")) {
        xn <- extrap$check(as.double(newdata[[sp$name]]), sp$bk_lo, sp$bk_hi,
                           sp$name)
        bd <- .Call("wrap__r_bns_design", xn, sp$bk_lo, sp$bk_hi,
                    as.integer(sp$knots), as.integer(sp$degree),
                    PACKAGE = "mboostrs")
        designs_list[[i]] <- bd$design
      } else if (identical(sp$type, "block") && !is.null(sp$radial_basis)) {
        designs_list[[i]] <- as.double(t(.rebuild_block_design(sp, newdata)))
      } else if (identical(sp$type, "block") && !is.null(sp$factor_levels)) {
        design_mat <- .bols_factor_design(newdata[[sp$name]], sp$factor_levels,
                                          isTRUE(sp$intercept), sp$name, unseen_levels)
        designs_list[[i]] <- as.double(t(design_mat))   # row-major
      } else if (identical(sp$type, "brandom") || identical(sp$type, "bmrf")) {
        # Coded against the training labels: a factor's own integer codes
        # follow its level set, so new data holding a subset of the levels
        # (or a level the fit never saw) was read as the wrong groups.
        if (is.null(newdata[[sp$name]])) stop(.missing_column_message(sp$name, newdata))
        labels <- if (identical(sp$type, "bmrf")) sp$level_labels else sp$levels
        design_mat <- .level_indicators(newdata[[sp$name]], labels, sp$n_levels,
                                        paste0(sp$type, "(", sp$name, ")"), unseen_levels)
        designs_list[[i]] <- as.double(t(design_mat))   # row-major
      }
    }
    extrap$report()
  }

  # Build centering list
  centering_list <- object$learner_centering

  # Spline `by = z` designs above are rebuilt raw, so the modifier is
  # applied in the accumulator; a `bols` by-block is rebuilt already scaled.
  by_values <- lapply(seq_along(designs_list), function(i) {
    sp <- object$learner_specs[[i]]
    if (!is.null(newdata) && sp$type %in% c("bbs", "bmono", "bns", "bspatial", "brandom", "bmrf") &&
        !is.null(sp$by_name)) {
      as.double(newdata[[sp$by_name]])
    } else numeric(0)
  })

  # A rebuilt design must have the columns the fit's coefficients index; a
  # mismatch means the new data were parsed into another form than the fit.
  if (!is.null(newdata)) {
    n_new <- nrow(as.data.frame(newdata))
    for (i in seq_along(designs_list)) {
      sp <- object$learner_specs[[i]]
      # Only the bols forms are re-derived from the data; other learners'
      # entries are raw values the accumulator expands.
      is_bols <- identical(sp$type, "scalar") ||
        (identical(sp$type, "block") && (isTRUE(sp$bols_block) || !is.null(sp$factor_levels)))
      if (!is_bols) next
      if (length(designs_list[[i]]) != n_new * object$learner_ncol[[i]]) {
        stop("predict: the new data rebuilt learner ", sp$name %||% i, " with ",
             length(designs_list[[i]]) / max(n_new, 1L), " columns where the fit has ",
             object$learner_ncol[[i]], "; this is an internal error, please report it",
             call. = FALSE)
      }
    }
  }

  # Fail closed before entering the Rust accumulator. The bridge can safely
  # replay scalar learners and simple block learners where the rebuilt design
  # column count matches the coefficient vector. Other selected learners (trees,
  # centered spectral smooths, some tensor bases) used to be silently skipped.
  k <- if (mstop_val <= 0L) length(object$selected) else min(mstop_val, length(object$selected))
  # predict(which = ) accumulates only the requested learners other than
  # brandom and bmrf, whose contributions the core forms from group indicators.
  accumulated <- if (is.null(which)) seq_along(object$learner_specs) else {
    requested <- .which_learner_indices(object, which)
    requested[!vapply(object$learner_specs[requested],
                      function(sp) sp$type %in% c("brandom", "bmrf"), logical(1))]
  }
  if (k > 0L) {
    for (step in seq_len(k)) {
      learner_idx <- object$selected[[step]]
      if (!(learner_idx %in% accumulated)) next
      sp <- object$learner_specs[[learner_idx]]
      coefs_vec <- object$coefs[[step]]
      nc <- object$learner_ncol[[learner_idx]]
      learner_type <- sp$type %||% "unknown"
      learner_name <- sp$name %||% paste0("learner ", learner_idx)

      if (learner_type %in% c("btree", "tree")) next
      if (!((nc == 1L && length(coefs_vec) == 1L) ||
            (nc > 1L && length(coefs_vec) == nc))) {
        stop("R predict.mboostrs_gamboost cannot safely predict selected learner ",
             learner_name, " (", learner_type, "): coefficient length ",
             length(coefs_vec), " does not match rebuilt design columns ", nc,
             ". This usually indicates centered or transformed basis metadata ",
             "that is not available in the R prediction bridge yet.")
      }
    }
  }

  # Tree steps are evaluated from their stored trees; the accumulator takes
  # the other steps.
  # `keep` selects the steps to accumulate: all of them for the model's
  # prediction, one learner's for predict(which = ).
  accumulate <- function(keep, start) {
  steps <- seq_len(k)[keep]
  is_tree <- vapply(steps, function(step) {
    (object$learner_specs[[object$selected[[step]]]]$type %||% "") %in% c("btree", "tree")
  }, logical(1))
  eta <- .Call("wrap__predict_gamboost",
    start,
    object$selected[steps][!is_tree],
    object$coefs[steps][!is_tree],
    object$nu,
    designs_list,
    as.integer(object$learner_ncol),
    centering_list,
    0L,
    by_values,
    PACKAGE = "mboostrs"
  )
  if (any(is_tree)) {
    source_data <- if (is.null(newdata)) object$data else newdata
    tree_designs <- list()
    for (step in steps[is_tree]) {
      idx <- object$selected[[step]]
      key <- as.character(idx)
      sp <- object$learner_specs[[idx]]
      if (is.null(tree_designs[[key]])) {
        tree_designs[[key]] <- .tree_design(sp, source_data)
        # .tree_design codes a level the fit never saw as -1.
        if (identical(unseen_levels, "error") && !is.null(newdata) &&
            .tree_has_unseen(sp, tree_designs[[key]])) {
          stop("btree(", sp$name, ") has a level the fit did not see. Pass ",
               "unseen_levels = \"average\" or \"zero_effect\" to predict it.",
               call. = FALSE)
        }
      }
      # "zero_effect" sends an unseen level to the branch that held more
      # training weight; "average" takes the mean over the levels.
      contribution <- .Call("wrap__r_tree_predict", as.double(object$coefs[[step]]),
                            tree_designs[[key]], as.integer(length(sp$names)),
                            as.integer(sp$categorical %||% integer(0)),
                            identical(unseen_levels, "average"),
                            PACKAGE = "mboostrs")
      if (!is.null(sp$by_values)) {
        contribution <- contribution * .tree_by(sp, if (is.null(newdata)) NULL else newdata)
      }
      if (length(eta) != length(contribution)) eta <- rep_len(eta, length(contribution))
      eta <- eta + object$nu * contribution
    }
  }
  eta
  }

  if (!is.null(which)) {
    # mboost's predict(which = ) returns a matrix, one column per learner,
    # holding that learner's summed contributions without the offset.
    idx <- .which_learner_indices(object, which)
    selected_k <- unlist(object$selected[seq_len(k)])
    columns <- lapply(idx, function(j) {
      sp <- object$learner_specs[[j]]
      if (sp$type %in% c("brandom", "bmrf")) .grouped_contribution(object, j, k, newdata, unseen_levels)
      else accumulate(selected_k == j, 0)
    })
    n_rows <- max(vapply(columns, length, integer(1)))
    out <- vapply(columns, function(col) rep_len(as.double(col), n_rows), double(n_rows))
    out <- matrix(out, nrow = n_rows, ncol = length(idx))
    colnames(out) <- .base_learner_names(object)[idx]
    return(out)
  }
  eta <- accumulate(rep(TRUE, k), start)

  if (type == "response") eta <- .gamboost_response(object, eta, mstop)
  eta
}

# The contribution of grouped learner `j` (brandom, bmrf) after `k` steps at
# the rows of `newdata` (the training rows when NULL). The core forms it from
# the rows' group indicators: b_g - sum_j c_j b_j for a centered brandom,
# with the centering the fit applied, and the projection onto the penalized
# directions of K for bmrf(center = TRUE).
.grouped_contribution <- function(object, j, k, newdata, unseen_levels) {
  sp <- object$learner_specs[[j]]
  source <- if (is.null(newdata)) object$data else newdata
  if (is.null(source[[sp$name]])) stop(.missing_column_message(sp$name, source))
  label <- paste0(sp$type, "(", sp$name, ")")
  labels <- if (identical(sp$type, "bmrf")) sp$level_labels else sp$levels
  indicators <- .level_indicators(source[[sp$name]], labels, sp$n_levels, label,
                                  unseen_levels)
  by <- if (is.null(sp$by_name)) numeric(0) else as.double(source[[sp$by_name]])
  steps <- seq_len(k)
  effect <- .Call("wrap__gamboost_grouped_contribution_r", object$rust_specs,
                  as.double(object$weights),
                  as.integer(object$multinomial_n_classes %||% 0L),
                  as.integer(j), as.double(t(indicators)), as.integer(sp$n_levels),
                  by, as.integer(unlist(object$selected[steps])),
                  lapply(object$coefs[steps], as.double), as.double(object$nu),
                  PACKAGE = "mboostrs")
  if (is.null(effect)) {
    stop("predict(which = ): the coefficients of ", label, " do not match its ",
         sp$n_levels, "-column design; this is an internal error, please report it",
         call. = FALSE)
  }
  effect
}

# Response-scale prediction from the link. PropOdds returns the (n, K) matrix
# of category probabilities from eta and the thresholds at step `mstop`, so
# predict(mstop = k) matches a separate mstop = k fit; every other family
# applies its inverse link.
.gamboost_response <- function(object, eta, mstop) {
  if (object$family$name == "propodds") {
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
      probs <- .Call("wrap__propodds_response_r",
        as.double(eta),
        as.double(sigma),
        PACKAGE = "mboostrs"
      )
      return(matrix(probs, nrow = n_pred,
                    ncol = length(sigma) + 1, byrow = TRUE))
    }
  }
  .inverse_link(eta, object$family)
}

#' Risk path for gamboost
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param ... Not used.
#' @return A numeric vector of risk values (length mstop + 1).
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' plot(risk(mod))
#' @export
risk.mboostrs_gamboost <- function(object, ...) {
  object$risk
}

#' @export
execution_report.mboostrs_gamboost <- function(object, ...) object$execution_report

#' Coefficients for gamboost
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param which Which learners (by name or index).
#' @param ... Not used.
#' @return A named list of coefficient vectors, one per base-learner.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' coef(mod)
#' @export
coef.mboostrs_gamboost <- function(object, which = NULL, ...) {
  .refuse_unused("coef", ...)
  if (isTRUE(object$family$name == "multinomial")) {
    # The Multinomial coefficients are Kronecker (class x covariate) blocks; the
    # per-learner aggregation below does not represent them, so fail loudly rather
    # than return a misleading value.
    stop("coef() is not supported for the Multinomial family (Kronecker-structured ",
         "coefficients); use predict(type = \"response\") for class probabilities.")
  }
  # Aggregate per-iteration coefs by learner index. Two learners can read one
  # column (`bols(x) + bbs(x, center = TRUE)`) and share its name; indexing by
  # name added the second learner's steps into the first's entry.
  keys <- .base_learner_names(object)
  coefs_agg <- setNames(rep(list(numeric(0)), length(keys)), keys)

  for (i in seq_along(object$selected)) {
    idx <- object$selected[i]
    step_coefs <- object$nu * object$coefs[[i]]
    coefs_agg[[idx]] <- if (length(coefs_agg[[idx]]) == 0) step_coefs
                        else coefs_agg[[idx]] + step_coefs
  }

  # A centered bols(x) fits b * (x - c), c the weighted mean of x. mboost's
  # bols(x) fits [1, x] and reports both coefficients, so report the
  # intercept -b * c the centered column implies, under mboost's names.
  for (i in seq_along(keys)) {
    rs <- object$rust_specs[[i]]
    if (is.null(rs) || !identical(rs$type, "scalar") || !isTRUE(rs$center)) next
    x <- as.double(rs$design)
    w <- object$weights %||% rep(1, length(x))
    slope <- if (length(coefs_agg[[i]])) coefs_agg[[i]][1] else 0
    coefs_agg[[i]] <- setNames(c(-slope * sum(w * x) / sum(w), slope),
                               c("(Intercept)", object$learner_names[i]))
  }

  if (!is.null(which)) {
    coefs_agg <- coefs_agg[.which_learner_indices(object, which)]
  }

  attr(coefs_agg, "offset") <- object$offset
  coefs_agg
}

#' @rdname concurvity
#' @export
concurvity.mboostrs_gamboost <- function(object, pairwise = FALSE, ...) {
  .refuse_unused("concurvity", ...)
  out <- .Call("wrap__gamboost_concurvity_r",
    object$rust_specs,
    as.double(object$weights),
    as.double(object$nu),
    as.integer(object$selected),
    object$coefs,
    as.integer(object$multinomial_n_classes %||% 1L),
    PACKAGE = "mboostrs"
  )
  .boosted_concurvity_frame(out, object$learner_names, pairwise)
}

#' Print method for gamboost
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param ... Not used.
#' @return Invisibly returns the model object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' print(mod)
#' @export
print.mboostrs_gamboost <- function(x, ...) {
  cat("GAMBoost model\n")
  cat("  Formula:", deparse(x$formula), "\n")
  cat("  Family:", x$family$name, "\n")
  cat("  mstop:", x$mstop, " nu:", x$nu, "\n")
  cat("  Offset:", .offset_label(x$offset), "\n")
  cat("  Number of base-learners:", length(x$learner_names), "\n")
  invisible(x)
}

#' Slice gamboost model at iteration k
#' @param x A fitted \code{mboostrs_gamboost} object.
#' @param k Iteration number to slice at (1 to mstop).
#' @return A sliced model object containing only the first \code{i} iterations.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 50))
#' mod20 <- mod[20]
#' @export
`[.mboostrs_gamboost` <- function(x, k) {
  if (k <= 0 || k > x$mstop) stop("k must be between 1 and mstop")
  k <- as.integer(k)

  if (isTRUE(x$family$name == "multinomial")) {
    if (k < x$mstop)
      warning("Multinomial subsetting to truncated mstop is approximate; ",
              "in-sample fitted values are not recomputed.")
    x$fitted <- x$fitted
  } else {
    eta <- predict(x, mstop = k, type = "link")
    x$fitted <- eta
  }
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
`mstop<-.mboostrs_gamboost` <- function(object, ..., value) {
  object[value]
}

# ─── cv / cvrisk ───────────────────────────────────────────────────

#' Generate cross-validation weight matrix
#'
#' @param n Number of observations.
#' @param fold_type "kfold", "bootstrap", or "subsampling".
#' @param b Number of folds / resampling iterations.
#' @param prob Proportion for subsampling (default: 0.5).
#' @param seed Random seed.
#' @return A matrix of weights (n x b).
#' @seealso [cvrisk()]
#' @examples
#' w <- cv_weights(n = 100, b = 5)
#' dim(w)
#' @export
cv_weights <- function(n, fold_type = "subsampling", b = 25, prob = 0.5, seed = 42L) {
  result <- .Call("wrap__cv_weights",
    as.integer(n),
    fold_type,
    as.integer(b),
    as.double(prob),
    as.integer(seed),
    PACKAGE = "mboostrs"
  )
  folds <- result$folds
  # Convert list of vectors to matrix
  do.call(cbind, folds)
}

#' Leave-one-out fold matrix
#'
#' Fold \code{i} holds out row \code{i} alone and refits on all the others,
#' so there are as many folds as rows with positive weight. This is mboost's
#' \code{cv(weights, type = "kfold", B = n)} with the columns in row order.
#' Every row is judged by a fit to all the others, so the estimate of
#' prediction error has little bias, at the cost of one fit per row.
#'
#' @param n Number of observations.
#' @param weights Optional observation weights; a row with zero weight is out
#'   of the sample and is held out by no fold.
#' @return An \code{n} x \code{B} matrix of in-bag weights, one column per
#'   held-out row, for \code{cvrisk(folds = )}.
#' @seealso [cv_weights()], [cvrisk()]
#' @examples
#' folds <- loo_folds(5)
#' folds
#' @export
loo_folds <- function(n, weights = NULL) {
  n <- as.integer(n)
  w <- if (is.null(weights)) rep(1, n) else as.double(weights)
  if (length(w) != n) stop("weights has length ", length(w), " but n = ", n, call. = FALSE)
  in_sample <- which(w > 0)
  if (length(in_sample) < 2L) {
    stop("leave-one-out needs at least 2 rows with positive weight", call. = FALSE)
  }
  if (length(in_sample) > 5000L) {
    warning("leave-one-out over ", length(in_sample), " rows costs ",
            length(in_sample), " fits; k-fold with b = 10 usually estimates ",
            "the same risk at a hundredth of the cost", call. = FALSE)
  }
  folds <- matrix(w, nrow = n, ncol = length(in_sample))
  folds[cbind(in_sample, seq_along(in_sample))] <- 0
  folds
}

# Normalise the sampling-type spelling used across the stabsel entry points.
.stabsel_sampling_type <- function(sampling_type) {
  switch(tolower(sampling_type),
    "ss" =, "subsampling" = "SS",
    "mb" =, "no" = "MB",
    stop("sampling_type must be 'SS' or 'MB'", call. = FALSE)
  )
}

# Append each column's exact complement, giving an n x 2B matrix in which
# column b and column B + b sum elementwise to the full weight vector. This is
# `folds <- cbind(folds, weights - folds)` in `stabs:::check_folds`, extended
# to non-unit weights (identical to upstream when every weight is 1).
.stabsel_append_complements <- function(folds, weights = NULL) {
  n <- nrow(folds)
  w <- if (is.null(weights)) rep(1.0, n) else as.double(weights)
  if (length(w) != n) {
    stop("weights has length ", length(w), " but folds has ", n, " rows",
         call. = FALSE)
  }
  cbind(folds, w - folds)
}

#' In-bag weight matrix for a stability-selection run
#'
#' `sampling_type = "SS"` draws `B` half-samples and appends their exact
#' complements, so the result has `2 * B` columns and column `b` pairs with
#' column `B + b`. `sampling_type = "MB"` draws `B` independent half-samples
#' and returns them unchanged.
#'
#' `B` counts initial subsamples — complementary pairs under `"SS"` — and is
#' the value the Shah-Samworth error bound is computed on. The column count is
#' the number of model fits the run performs.
#'
#' @param n Number of observations.
#' @param B Number of initial subsamples (pairs under `"SS"`).
#' @param sampling_type `"SS"` for complementary pairs or `"MB"` for
#'   independent half-samples. `"subsampling"` and `"no"` are accepted aliases.
#' @param weights Model weights, or `NULL` for unit weights. Each `"SS"` pair
#'   sums elementwise to this vector.
#' @param prob Subsample proportion. Only `0.5` is meaningful under `"SS"`,
#'   because the complement of a subsample is a half-sample only then.
#' @param seed Random seed.
#' @return A matrix of in-bag weights, `n x 2B` for `"SS"` and `n x B` for
#'   `"MB"`.
#' @seealso [cv_weights()], [stabsel()]
#' @examples
#' f <- stabsel_folds(n = 20, B = 3)
#' dim(f)
#' all.equal(f[, 1] + f[, 4], rep(1, 20))
#' @export
stabsel_folds <- function(n, B = 50L, sampling_type = "SS", weights = NULL,
                          prob = 0.5, seed = 42L) {
  sampling_type <- .stabsel_sampling_type(sampling_type)
  B <- as.integer(B)
  if (is.na(B) || B < 1L) stop("B must be at least 1", call. = FALSE)
  if (sampling_type == "SS" && !isTRUE(all.equal(prob, 0.5))) {
    stop("complementary-pairs sampling requires prob = 0.5; the complement of ",
         "a subsample of proportion ", prob, " is not a half-sample. Use ",
         "sampling_type = \"MB\" for other proportions.", call. = FALSE)
  }
  folds <- cv_weights(n, "subsampling", b = B, prob = prob, seed = seed)
  if (!is.null(weights)) folds <- folds * as.double(weights)
  if (sampling_type == "SS") folds <- .stabsel_append_complements(folds, weights)
  folds
}

# Turn caller-supplied `folds` into the matrix that is actually fitted.
#
# Caller-supplied `folds` are always the *initial* subsamples: B columns, one
# per subsample under "MB" and one per pair under "SS". Under "SS" the
# complements are appended here, so a caller passing B columns gets 2 * B fits.
# Pairing is never inferred from an even column count.
.stabsel_fit_folds <- function(folds, sampling_type, n, weights = NULL) {
  if (!is.matrix(folds)) folds <- as.matrix(folds)
  if (nrow(folds) != n || ncol(folds) < 1L) {
    stop("folds must be an n x B matrix with at least one column", call. = FALSE)
  }
  if (any(!is.finite(folds)) || any(folds < 0)) {
    stop("fold weights must be finite and non-negative", call. = FALSE)
  }
  B <- ncol(folds)
  if (sampling_type == "SS") {
    folds <- .stabsel_append_complements(folds, weights)
  }
  list(folds = folds, B = B, n_fits = ncol(folds))
}

#' Cross-validated risk for glmboost
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param folds A weight matrix (n x B) from \code{cv_weights()}, or NULL to auto-generate.
#' @param fold_type Fold type (if folds is NULL).
#' @param b Number of folds (if folds is NULL).
#' @param prob Subsampling probability (if folds is NULL).
#' @param seed Random seed (if folds is NULL).
#' @param preprocessing "fixed" (default) reuses the full-data design,
#'   matching mboost. "refit" learns the formula transformations and design
#'   on each fold's training rows and replays them on held-out rows. Refitted
#'   folds are processed sequentially; native code scores their risk paths.
#' @param ... Generic: arguments passed on to the method. The mboostrs
#'   glmboost and gamboost methods accept none and signal an error if any is
#'   supplied; for other model classes they are passed to \code{mboost::cvrisk()}.
#' @return An object of class \code{mboostrs_cvrisk}.
#' @seealso [cv_weights()], [mstop.mboostrs_cvrisk()]
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' cr <- cvrisk(mod, b = 5)
#' mstop(cr)
#' @export
cvrisk <- function(object, folds = NULL, fold_type = "subsampling", b = 25,
                   prob = 0.5, seed = 42L, preprocessing = c("fixed", "refit"), ...) {
  UseMethod("cvrisk")
}

# A 0/1 fold mask given to a model with non-constant weights refits each fold
# with the model's weights (the mask times the weights); a matrix that already
# carries the weights, and any count or fractional matrix, is used as given.
# The Python frontend applies the same rule.
.fold_mask_weights <- function(folds, w) {
  if (is.null(w) || length(w) != nrow(folds) || length(unique(w)) == 1L) return(folds)
  in_bag <- folds > 0
  if (!all(folds[in_bag] == 1)) return(folds)
  carried <- ifelse(in_bag, w, 0)
  dim(carried) <- dim(folds)
  if (isTRUE(all(carried == folds))) return(folds)
  carried
}

#' Cross-validated risk for glmboost
#'
#' @param object A fitted \code{mboostrs_glmboost} object.
#' @param folds A weight matrix (n x B) from \code{cv_weights()}, or NULL to auto-generate.
#' @param fold_type Fold type (if folds is NULL).
#' @param b Number of folds (if folds is NULL).
#' @param prob Subsampling probability (if folds is NULL).
#' @param seed Random seed (if folds is NULL).
#' @param preprocessing "fixed" (default) reuses the full-data design,
#'   matching mboost. "refit" learns formula transformations and the design
#'   inside each training fold and replays them on held-out rows.
#' @param ... Not used. The method signals an error if any argument is supplied.
#' @return An object of class \code{"mboostrs_cvrisk"}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' cr <- cvrisk(mod, b = 5)
#' mstop(cr)
#' @export
cvrisk.mboostrs_glmboost <- function(object, folds = NULL, fold_type = "subsampling",
                                      b = 25, prob = 0.5, seed = 42L,
                                      preprocessing = c("fixed", "refit"), ...) {
  .refuse_unused("cvrisk", ...)
  preprocessing <- match.arg(preprocessing)
  if (is.null(folds)) {
    folds <- cv_weights(object$n, fold_type, b, prob, seed)
  }

  if (!is.matrix(folds)) {
    folds <- do.call(cbind, folds)
  }
  folds <- .fold_mask_weights(folds, object$weights)
  if (preprocessing == "refit") return(.cvrisk_refitted(object, folds))

  # Convert matrix to list of column vectors
  folds_list <- lapply(seq_len(ncol(folds)), function(j) folds[, j])

  result <- .Call("wrap__cvrisk_glmboost",
    as.double(t(object$x_matrix)),
    as.integer(object$n),
    as.integer(ncol(object$x_matrix)),
    as.double(.get_response(object)),
    as.double(object$weights %||% rep(1, object$n)),
    as.integer(object$mstop),
    as.double(object$nu),
    object$family$name,
    object$family$params %||% list(),
    setdiff(object$learner_names, "(Intercept)"),
    folds_list,
    as.double(object$centering_means),
    # A fitted scalar is re-estimated per fold; a per-row offset is known.
    if (length(object$offset) > 1L) object$offset else NaN,
    "oobag",
    !isFALSE(object$intercept),
    object$control$step %||% "fixed", as.integer(object$control$step_iterations %||% 4L),
    PACKAGE = "mboostrs"
  )

  # Reshape risk matrix from flat vector to B x n_cols matrix
  risk_mat <- matrix(result$risk_matrix, nrow = result$n_folds, ncol = result$n_cols, byrow = TRUE)

  structure(
    list(
      risk_matrix = risk_mat,
      n_folds = result$n_folds,
      n_cols = result$n_cols,
      optimal_mstop = result$optimal_mstop,
      aggregate = result$aggregate,
      preprocessing = preprocessing
    ),
    class = "mboostrs_cvrisk"
  )
}

#' Cross-validated risk for gamboost
#'
#' Refits the model once per fold on that fold's in-bag weights and reads the
#' out-of-bag risk path the engine records along the fit, which is the quantity
#' `mboost::cvrisk` reports. A fold column is an in-bag weight vector, so each
#' fold recalibrates `df -> lambda` and centering on its own weights while the
#' knots stay those of the full data, as `mboost`'s `dpp` does. The risk path
#' starts at the offset-only state, so the returned matrix has `mstop + 1`
#' columns and `optimal_mstop` is a number of boosting steps.
#'
#' For a model fitted with weights, each fold refits with the model's weights
#' on its in-bag rows (a 0/1 fold mask is multiplied by them) and the held-out
#' risk weights each held-out row by its model weight, divided by their total,
#' as the Python frontend's `cvrisk` does.
#'
#' @param object A fitted \code{mboostrs_gamboost} object.
#' @param folds A weight matrix (n x B) from \code{cv_weights()}, or NULL to auto-generate.
#' @param fold_type Fold type (if folds is NULL).
#' @param b Number of folds (if folds is NULL).
#' @param prob Subsampling probability (if folds is NULL).
#' @param seed Random seed (if folds is NULL).
#' @param preprocessing "fixed" (default) reuses full-data transformations
#'   and spline boundaries, matching mboost. "refit" rebuilds them using each
#'   fold's training rows and predicts held-out rows through that fitted recipe.
#'   Refitted folds are processed sequentially and the core scores their paths.
#' @param ... Not used. The method signals an error if any argument is supplied.
#' @return An object of class \code{"mboostrs_cvrisk"}.
#' @examples
#' set.seed(42)
#' n <- 50
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = runif(n))
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 20))
#' cr <- cvrisk(mod, b = 3)
#' mstop(cr)
#' @export
cvrisk.mboostrs_gamboost <- function(object, folds = NULL, fold_type = "subsampling",
                                      b = 25, prob = 0.5, seed = 42L,
                                      preprocessing = c("fixed", "refit"), ...) {
  .refuse_unused("cvrisk", ...)
  preprocessing <- match.arg(preprocessing)
  if (is.null(folds)) {
    folds <- cv_weights(object$n, fold_type, b, prob, seed)
  }
  if (!is.matrix(folds)) folds <- do.call(cbind, folds)

  if (nrow(folds) != object$n) {
    stop("folds has ", nrow(folds), " rows but the model has ", object$n,
         " observations", call. = FALSE)
  }

  w <- object$weights %||% rep(1, object$n)
  folds <- .fold_mask_weights(folds, w)
  if (preprocessing == "refit") return(.cvrisk_refitted(object, folds))

  B <- ncol(folds)
  ms <- object$mstop

  risk_mat <- matrix(NA_real_, nrow = B, ncol = ms + 1L)
  ctrl <- object$control %||% boost_control(mstop = ms, nu = object$nu)
  ctrl$mstop <- ms
  ctrl$risk <- "oobag"

  for (bb in seq_len(B)) {
    # Held-out rows are scored with the model's weights, as upstream's
    # `OOBweights` are; rows the model gives zero weight are in no fold.
    oob_w <- ifelse(folds[, bb] == 0, w, 0)
    n_oobag <- sum(oob_w)
    if (n_oobag == 0) {
      stop("fold ", bb, " keeps every observation in bag, so it has no OOB ",
           "rows to evaluate", call. = FALSE)
    }

    # The fold column is the in-bag weight vector, so the fit sees every row
    # and only the weights change. Row subsetting would move the knots and the
    # centering onto the subsample, which is not what upstream cross-validates.
    fit_b <- tryCatch(
      gamboost(object$formula, data = object$data, family = object$family,
               control = ctrl, weights = folds[, bb], offset = object$user_offset,
               .oob_weights = oob_w),
      error = function(e) NULL
    )
    if (is.null(fit_b)) next

    # `risk = "oobag"` records the risk on the zero-weight rows at every
    # iteration, offset state first, weighted by `oob_w`. Upstream divides by
    # the out-of-bag weight total.
    risk_mat[bb, ] <- as.numeric(fit_b$risk) / n_oobag
  }

  failed <- sum(apply(risk_mat, 1, function(row) all(is.na(row))))
  if (failed > 0) {
    warning(failed, " of ", B, " folds failed to fit; the risk path averages ",
            B - failed, " folds", call. = FALSE)
  }
  if (failed == B) {
    stop("every fold failed to fit", call. = FALSE)
  }

  aggregate <- colMeans(risk_mat, na.rm = TRUE)
  optimal <- which.min(aggregate) - 1L

  structure(
    list(
      risk_matrix = risk_mat,
      n_folds = B,
      n_cols = ms + 1L,
      optimal_mstop = optimal,
      aggregate = aggregate,
      preprocessing = preprocessing
    ),
    class = "mboostrs_cvrisk"
  )
}

#' Get response from a fitted model
#' @noRd
.get_response <- function(object) {
  raw <- .raw_response(object$formula, object$data)
  labelled <- .binary_label_response(raw, object$family$name %||% "")
  if (!is.null(labelled)) return(labelled)
  if (inherits(object, "mboostrs_gamboost")) {
    # A gamboost right-hand side holds base-learner calls that model.frame()
    # cannot evaluate, so the response is the evaluated left-hand side (a
    # transformation such as log(y) applied), coded as the fit coded it.
    if (is.null(raw)) stop("the model's response could not be evaluated from its data",
                           call. = FALSE)
    return(if (is.numeric(raw)) as.double(raw) else as.double(as.integer(as.factor(raw))))
  }
  mf <- model.frame(object$formula, data = object$data)
  as.double(model.response(mf))
}

#' Print cvrisk result
#' @param x A \code{mboostrs_cvrisk} object.
#' @param ... Not used.
#' @return Invisibly returns the object.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' cr <- cvrisk(mod, b = 5)
#' print(cr)
#' @export
print.mboostrs_cvrisk <- function(x, ...) {
  cat("Cross-validated risk\n")
  cat("  Preprocessing:", x$preprocessing %||% "fixed", "\n")
  cat("  Number of folds:", x$n_folds, "\n")
  cat("  Optimal mstop:", x$optimal_mstop, "\n")
  cat("  Minimum aggregate risk:", formatC(min(x$aggregate), digits = 6), "\n")
  invisible(x)
}

#' Extract optimal mstop from cvrisk
#' @param object A \code{mboostrs_cvrisk} object.
#' @param ... Not used.
#' @return An integer scalar.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(y = rnorm(n), x1 = rnorm(n), x2 = rnorm(n))
#' mod <- glmboost(y ~ x1 + x2, data = dat, control = boost_control(mstop = 50))
#' cr <- cvrisk(mod, b = 5)
#' mstop(cr)
#' @export
mstop.mboostrs_cvrisk <- function(object, ...) {
  object$optimal_mstop
}

# ─── stabsel ───────────────────────────────────────────────────────

#' Stability selection parameter computation
#'
#' @param p Number of base-learners.
#' @param b Number of resampling iterations.
#' @param cutoff Cutoff for stability selection (0.6-0.9 typical).
#' @param q Number of variables selected per subsample (optional).
#' @param pfer Upper bound on the expected number of falsely selected
#'   learners (optional). A count, not a probability or a rate, despite
#'   the name it carries in the literature.
#' @param assumption `"none"`, `"unimodal"`, or `"r-concave"`
#'   (`"rconcave"` is accepted as a compatibility alias).
#' @param sampling_type `"SS"` for complementary-pairs stability selection or
#'   `"MB"` for the original bound. The legacy aliases `"subsampling"` and
#'   `"no"` are also accepted.
#' @return A list with computed parameters.
#' @seealso [stabsel()]
#' @examples
#' params <- stabsel_params(p = 10, b = 50, q = 5)
#' params
#' @export
stabsel_params <- function(p, b, cutoff = 0.7, q = -1L, pfer = -1.0,
                           assumption = "unimodal", sampling_type = "subsampling") {
  sampling_key <- tolower(sampling_type)
  sampling_type <- switch(sampling_key,
    "ss" =, "subsampling" = "SS",
    "mb" =, "no" = "MB",
    stop("sampling_type must be 'SS' or 'MB'", call. = FALSE)
  )
  if (identical(assumption, "rconcave")) assumption <- "r-concave"
  result <- .Call("wrap__stabsel_params_rust",
    as.integer(p),
    as.integer(b),
    as.double(cutoff),
    as.integer(q),
    as.double(pfer),
    assumption,
    sampling_type,
    PACKAGE = "mboostrs"
  )

  result
}

# The `extrapolation=` policy of predict(): `check()` measures one variable
# against its training range and returns the values to evaluate the basis at
# (clamped under "clip"); `report()` emits one warning naming every learner
# that was asked about values outside its range. As in Python, a warning
# needs some row at least 5% of the training span outside; "error" and "clip"
# act on every value strictly outside.
.extrapolation_policy <- function(policy) {
  rows <- list()
  check <- function(x, lo, hi, learner, variable = learner) {
    if (identical(policy, "allow") || is.null(lo) || is.null(hi)) return(x)
    finite <- x[is.finite(x)]
    below <- sum(finite < lo)
    above <- sum(finite > hi)
    if (below + above == 0L) return(x)
    span <- hi - lo
    worst <- if (span > 0) {
      max((lo - min(finite)) / span, (max(finite) - hi) / span, 0)
    } else Inf
    where <- c(if (below > 0L) sprintf("%d below %.6g", below, lo),
               if (above > 0L) sprintf("%d above %.6g", above, hi))
    label <- if (identical(learner, variable)) learner else
      sprintf("%s (%s)", learner, variable)
    text <- sprintf("%s: %s (worst %.2g training ranges out)", label,
                    paste(where, collapse = " and "), worst)
    if (identical(policy, "error")) {
      stop("predict(newdata=) asks for values outside the training range of ",
           "a spline learner. ", text, ". Pass extrapolation = \"clip\" to ",
           "hold the effect flat outside the range or \"allow\" to continue ",
           "the edge slope.", call. = FALSE)
    }
    rows[[length(rows) + 1L]] <<- list(text = text, worst = worst)
    if (identical(policy, "clip")) x <- pmin(pmax(x, lo), hi)
    x
  }
  report <- function() {
    worst <- vapply(rows, `[[`, numeric(1), "worst")
    material <- rows[worst >= 0.05][order(-worst[worst >= 0.05])]
    if (!length(material)) return(invisible(NULL))
    texts <- paste(vapply(material, `[[`, "", "text"), collapse = "; ")
    advice <- if (identical(policy, "clip")) {
      paste("Those rows were clipped to the training range, so the effect is",
            "held flat outside it rather than continuing at the edge slope.")
    } else {
      paste("A spline basis continues by its edge slope outside its boundary",
            "knots, which the fit does not constrain. Pass extrapolation =",
            "\"allow\" to accept it, \"clip\" to hold the effect flat",
            "outside the range, or \"error\" to refuse.")
    }
    warning(length(material), " learner(s) are asked to predict outside the ",
            "range they were fitted on. ", texts, ". ", advice, call. = FALSE)
    invisible(NULL)
  }
  list(check = check, report = report)
}

# Penalization drift (Schulte and Ruegamer, 2025). A penalized learner selected
# m times at step nu has accumulated I - (I - nu S)^m on its basis; its df,
# sum_k 1 - (1 - nu s_k)^m over the eigenvalues of the smoother S, rises
# towards the rank of the basis, the df of the unpenalized fit. The core
# computes it during the fit, only when some learner has nu * selections >= 50
# (otherwise `drift` is NULL). drift_share is the part of the gap from one
# step's df, tr(S), to the unpenalized df that boosting has used.
.penalty_drift_frame <- function(drift, learner_names, learner_specs) {
  if (is.null(drift)) return(NULL)
  # The term as written, so bbs(x) and bols(x) stay apart.
  kind <- vapply(learner_specs, function(sp) {
    t <- sp$type %||% ""
    if (t %in% c("scalar", "block")) "bols" else t
  }, character(1))
  learner_names <- ifelse(nzchar(kind), sprintf("%s(%s)", kind, learner_names),
                          learner_names)
  gap <- drift$unpenalized_df - drift$single_step_df
  share <- ifelse(is.finite(gap) & gap > 1e-8,
                  pmax((drift$boosted_df - drift$single_step_df) / gap, 0), 0)
  frame <- data.frame(learner = learner_names,
                      selections = as.integer(drift$selections),
                      single_step_df = drift$single_step_df,
                      boosted_df = drift$boosted_df,
                      unpenalized_df = drift$unpenalized_df,
                      drift_share = share,
                      stringsAsFactors = FALSE)
  frame[is.finite(frame$boosted_df), , drop = FALSE]
}

# Warns when a learner has used at least a third of that gap, as Python's
# warn_if_penalty_drifted does.
.warn_if_penalty_drifted <- function(drift) {
  if (is.null(drift) || !nrow(drift)) return(invisible(NULL))
  hit <- drift[drift$drift_share >= 1 / 3, , drop = FALSE]
  if (!nrow(hit)) return(invisible(NULL))
  hit <- hit[order(-hit$drift_share), , drop = FALSE]
  parts <- sprintf("%s: %.3g of %.3g df after %d selections (one step: %.3g)",
                   hit$learner, hit$boosted_df, hit$unpenalized_df,
                   hit$selections, hit$single_step_df)
  warning("boosting has undone much of the penalty of ", nrow(hit),
          " learner(s): ", paste(parts, collapse = "; "), ". Each selection ",
          "fits a shrunken step, and the steps add up to the unpenalized fit on ",
          "the learner's basis, so a penalty's df no longer sets the term's ",
          "flexibility. Choose mstop by cross-validation (cvrisk) rather than ",
          "fitting this long. The model's $penalty_drift lists every learner.",
          call. = FALSE)
  invisible(NULL)
}

# Evaluates `expr` with the penalization-drift warning muffled, for refits of a
# model whose own fit already reported it.
.quiet_penalty_drift <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    if (startsWith(conditionMessage(w), "boosting has undone much of the penalty"))
      invokeRestart("muffleWarning")
  })
}

# Learner names that identify learners: the column name, with the learner type
# appended where two learners share it ("x (bols)", "x (bbs)"), as Python's
# base_learner_names() does.
.base_learner_names <- function(object) {
  names <- object$learner_names
  kinds <- vapply(object$learner_specs, function(sp) {
    t <- sp$type %||% ""
    if (t %in% c("scalar", "block")) "bols" else t
  }, character(1))
  shared <- names %in% names[duplicated(names)]
  ifelse(shared, sprintf("%s (%s)", names, kinds), names)
}

# Learner indices named by `which`: an index, a column name (every learner on
# it) or a name from .base_learner_names().
.which_learner_indices <- function(object, which) {
  if (is.numeric(which)) return(as.integer(which))
  keys <- .base_learner_names(object)
  unlist(lapply(which, function(w) {
    found <- which(object$learner_names == w | keys == w)
    if (!length(found)) stop("unknown base-learner '", w, "'; the model has ",
                             paste(keys, collapse = ", "), call. = FALSE)
    found
  }))
}
