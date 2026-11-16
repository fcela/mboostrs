# Model descriptions written to be read by a language model.
#
# `llm_explain_prompt()` renders a fit as one prompt: how component-wise boosting
# produced it, which base learners it kept, each learner's share of the in-bag
# risk reduction, the shape of every effect it estimated, and which questions the
# fit cannot answer. It sends nothing anywhere. `llm_describe()` is the one
# function here that makes a request: it posts that prompt to an
# OpenAI-compatible chat endpoint and returns the reply.
#
# The wording of the fixed sections matches the Python frontend, so a prompt
# built in either language reads the same to the model receiving it.

.LLM_SECTIONS <- c("reading", "facts", "importance", "effects", "caveats", "task")

.LLM_DEFAULT_TASK <- paste(
  "Write the brief a statistician would hand a colleague who has already",
  "read the tables above. Structure the reply in four labelled parts.",
  "**Findings**: the two or three results that matter most, each with the",
  "numbers that establish it, in the units named above. **Surprising",
  "patterns**: what runs against expectation for this subject matter, with",
  "the numbers that make it notable. **What could mislead**: features of",
  "the fit that could turn a finding into an artefact - a sparse or",
  "extrapolated covariate range, two learners carrying the same signal, an",
  "effect standing in for an omitted variable, a stopping iteration that",
  "may be too early or too late. **Next check**: the one analysis or datum",
  "that would most change your reading. Keep the whole reply under 300",
  "words. A reply that walks the effects one by one, or that rebuilds a",
  "table the description already contains, fails this task however",
  "accurate it is; cite a number only in support of a point. Describe",
  "associations: this is a predictive fit on observational data unless the",
  "notes above say otherwise, so write what moves with what, not what",
  "causes what. Where the fit does not support a claim, say so instead of",
  "qualifying it. Markdown with bold labels and bullets; no code fence."
)

.LLM_SYSTEM <- paste0(
  "You are %s. You read the global structure of a fitted component-wise ",
  "boosting model \\u2014 an additive model whose terms were selected one at a ",
  "time \\u2014 and you answer from the numbers you are given, the context ",
  "supplied with them, and your knowledge of the subject matter."
)

.LLM_READING <- paste(
  "How to read what follows.",
  "",
  "The fit is an additive model. The prediction for one observation is an offset",
  "plus one effect per base learner, summed on the link scale named below.",
  "Component-wise boosting built it by repeating one step: fit every candidate",
  "base learner to the current negative gradient of the loss, keep the single",
  "learner that reduces the loss most, and add `nu` times that fit to the model.",
  "A learner picked in many steps accumulated a large effect. A learner never",
  "picked contributes exactly zero \\u2014 it is absent from the model, not merely",
  "small.",
  "",
  "Each effect below is centered, so the offset carries the overall level and only",
  "differences within an effect are meaningful. An effect is printed as",
  "`covariate value: contribution` pairs; a key in parentheses is an interval, and",
  "a quoted key is a category label.",
  sep = "\n"
)

.LLM_CAVEATS <- paste(
  "Limits of this description.",
  "",
  "- The fit reports no standard errors, confidence intervals or p-values. Each",
  "  base learner was selected and estimated on the same data, so the usual",
  "  interval formulas do not apply to it. Treat every number here as a point",
  "  estimate with unquantified uncertainty.",
  "- Importance is the share of in-bag risk reduction credited to a learner over",
  "  the path. Two correlated covariates split that credit, so a small share can",
  "  mean redundancy rather than irrelevance, and a learner's share depends on",
  "  which other learners were offered.",
  "- Effect shapes hold over the covariate range printed with them. Outside it the",
  "  learner extrapolates by construction, not from data.",
  "- Everything here is conditional on the stopping iteration `mstop`. The same",
  "  data stopped at a different iteration gives different effects, and a longer",
  "  path gives larger ones.",
  "- No cross-validated risk is included, so nothing here says whether the fit",
  "  stopped too early or too late.",
  sep = "\n"
)

.LLM_PROBE_SURPRISES <- paste(
  "Now examine the patterns you reported and the fit behind them, and",
  "surface what a first reading misses: regions surprising or",
  "counterintuitive for this subject matter, effects that look like data",
  "artefacts rather than mechanisms, and importance shares or update",
  "counts that could mislead - credit split between correlated covariates,",
  "many updates buying little risk reduction, a learner absorbing the",
  "signal of a variable absent from the model. For each item say which",
  "numbers make it notable, which explanations could produce it - a real",
  "mechanism, a data artefact such as a sparse covariate range, a",
  "covariate standing in for another, confounding by something not in the",
  "model, or the path having stopped where it did - which one you consider",
  "most likely on subject-matter grounds, and what observation, analysis",
  "or experiment would settle it. The fit alone cannot settle it, so",
  "commit to a judgment instead of listing possibilities."
)

.LLM_PROBE_CONDENSE <- paste(
  "Now write the brief for a reader who has the charts and tables in hand:",
  "at most %d sentences. Open with the finding that matters most, give the",
  "other findings worth acting on, name the issues that could make them",
  "artefacts, and end with the single next check to run. A sentence that",
  "only reports what a table already shows is a wasted sentence: spend none",
  "on the method and none on walking the effects one by one. Write it in",
  "Markdown with bold labels or bullets where they help. Do not wrap it in",
  "a code fence."
)

# What one unit of the additive predictor means, per family. Families whose
# inverse link this package defines make a claim; the rest say they do not.
.LLM_RESPONSE_SCALE <- list(
  gaussian  = c("the response itself (identity link)",
                "A contribution of +1 adds 1 to the predicted mean."),
  gaussreg  = c("the response itself (identity link)",
                "A contribution of +1 adds 1 to the predicted mean."),
  laplace   = c("the response itself (identity link)",
                "A contribution of +1 adds 1 to the predicted median."),
  huber     = c("the response itself (identity link)",
                "A contribution of +1 adds 1 to the predicted centre."),
  quantreg  = c("the response itself (identity link)",
                "A contribution of +1 adds 1 to the predicted quantile."),
  expectreg = c("the response itself (identity link)",
                "A contribution of +1 adds 1 to the predicted expectile."),
  binomial  = c("half the log-odds of the positive class",
                paste("The response is 1/(1 + exp(-2*eta)), so a contribution",
                      "of +1 multiplies the odds by exp(2) = 7.39.")),
  adaexp    = c("half the log-odds of the positive class",
                "The response is 1/(1 + exp(-2*eta))."),
  poisson   = c("the log of the expected count",
                "A contribution of +1 multiplies the expected count by exp(1) = 2.72."),
  nbinomial = c("the log of the expected count",
                "A contribution of +1 multiplies the expected count by 2.72."),
  gammasreg = c("the log of the expected response",
                "A contribution of +1 multiplies the expected response by 2.72."),
  coxph     = c("the log relative risk",
                paste("A contribution of +1 multiplies the hazard by exp(1) =",
                      "2.72; the baseline hazard is not part of the additive",
                      "predictor.")),
  weibull   = c("the log of the expected survival time",
                "A contribution of +1 multiplies expected time by 2.72."),
  loglog    = c("the log of the expected survival time",
                "A contribution of +1 multiplies expected time by 2.72."),
  lognormal = c("the log of the expected survival time",
                "A contribution of +1 multiplies expected time by 2.72.")
)

# --- formatting -----------------------------------------------------

.llm_num <- function(value, digits = 4) {
  if (length(value) != 1 || !is.finite(value)) return("NA")
  # `width = 1` because formatC otherwise pads to a common width, and these
  # numbers are read inside a sentence, not down a column.
  text <- formatC(value, digits = digits, format = "g", width = 1)
  if (text %in% c("-0", "-0.0")) "0" else text
}

.llm_pct <- function(share) {
  if (!is.finite(share) || share == 0) return("0.0%")
  if (abs(share) < 0.001) return(if (share > 0) "<0.1%" else ">-0.1%")
  sprintf("%.1f%%", 100 * share)
}

# Decimals that keep about three significant digits of an effect's span. An
# effect moving by 0.004 needs more decimals than one moving by 40, and a fixed
# width either loses the first or pads the second with noise.
.llm_precision <- function(values) {
  values <- values[is.finite(values)]
  if (!length(values)) return(3L)
  span <- max(values) - min(values)
  if (span <= 0) return(3L)
  precision <- 1L
  fraction <- span / 100
  while (fraction < 1 && precision < 8L) {
    fraction <- fraction * 10
    precision <- precision + 1L
  }
  precision
}

.llm_pairs <- function(keys, values, precision) {
  rounded <- vapply(
    round(values, precision),
    function(value) formatC(value, format = "fg", width = 1, drop0trailing = TRUE),
    character(1)
  )
  paste0("{", paste(sprintf('"%s": %s', keys, rounded), collapse = ", "), "}")
}

# One sentence on the monotonicity and turning points of an effect. A step under
# one per cent of the effect's own span is flat for reading purposes; without
# that, every rounding wiggle counts as a turn.
.llm_shape <- function(x, y) {
  ordering <- order(x)
  x <- x[ordering]
  y <- y[ordering]
  span <- if (length(y)) max(y) - min(y) else 0
  if (length(y) < 3 || span <= 0) {
    return(if (span <= 0) "flat" else "too few points to characterize")
  }
  steps <- diff(y)
  live <- sign(steps)
  live[abs(steps) <= 0.01 * span] <- 0
  live <- live[live != 0]
  turns <- if (length(live) > 1) sum(live[-1] != live[-length(live)]) else 0
  if (turns == 0) {
    direction <- if (length(live) && live[1] > 0) "increasing" else "decreasing"
    return(paste0("monotone ", direction, " over the observed range"))
  }
  sprintf(
    "%d turning point%s; highest near %s, lowest near %s",
    turns, if (turns > 1) "s" else "",
    .llm_num(x[which.max(y)]), .llm_num(x[which.min(y)])
  )
}

# Reduce (x, effect) to at most `n_points` labelled values. Few distinct
# covariate values are reported as they are, which is what a factor, an indicator
# or a stump needs; a continuous covariate with more is summarized by equal-count
# intervals carrying the mean effect of their rows.
.llm_condense <- function(x, y, n_points, levels = NULL) {
  unique_x <- sort(unique(x))
  if (length(unique_x) <= max(n_points, 2)) {
    values <- vapply(unique_x, function(value) mean(y[x == value]), numeric(1))
    keys <- if (!is.null(levels) && all(unique_x %% 1 == 0) &&
                min(unique_x) >= 1 && max(unique_x) <= length(levels)) {
      levels[unique_x]
    } else {
      vapply(unique_x, .llm_num, character(1))
    }
    return(list(keys = keys, values = values, x = unique_x, y = values))
  }
  edges <- unique(stats::quantile(x, probs = seq(0, 1, length.out = n_points + 1),
                                  names = FALSE, type = 7))
  keys <- character(0)
  values <- numeric(0)
  centres <- numeric(0)
  for (i in seq_len(length(edges) - 1)) {
    lo <- edges[i]
    hi <- edges[i + 1]
    rows <- if (i == length(edges) - 1) x >= lo & x <= hi else x >= lo & x < hi
    if (!any(rows)) next
    keys <- c(keys, sprintf("(%s, %s)", .llm_num(lo), .llm_num(hi)))
    values <- c(values, mean(y[rows]))
    centres <- c(centres, (lo + hi) / 2)
  }
  list(keys = keys, values = values, x = centres, y = values)
}

# --- model facts ----------------------------------------------------

.llm_response_scale <- function(object) {
  name <- object$family$name
  scale <- .LLM_RESPONSE_SCALE[[name]]
  if (is.null(scale)) {
    return(c(
      sprintf("the %s family's link scale", name),
      paste("The relation between a contribution and the response is the",
            "family's inverse link, which is not summarized here.")
    ))
  }
  scale
}

.llm_risk_facts <- function(object) {
  risk <- object$risk
  start <- risk[1]
  end <- risk[min(object$mstop + 1, length(risk))]
  list(start = start, end = end,
       reduction = if (start != 0) (start - end) / abs(start) else NA_real_)
}

.llm_facts_block <- function(object, kind, response_description) {
  facts <- .llm_risk_facts(object)
  scale <- .llm_response_scale(object)
  family <- object$family$name
  lines <- c(
    # R stores the family name as the engine spells it (`gaussian`); the heading
    # reads better with the capital the Python frontend's class name carries.
    sprintf("%s boosting results", sub("^(.)", "\\U\\1", family, perl = TRUE)),
    sprintf("Model: %s", kind),
    sprintf("Family: %s", object$family$name),
    sprintf("Formula: %s", paste(deparse(object$formula), collapse = " ")),
    sprintf("No. observations: %d", object$n),
    sprintf("Base learners: %d", length(object$learner_names)),
    sprintf("Iterations (mstop): %d", object$mstop),
    sprintf("Step length (nu): %s", .llm_num(object$nu)),
    sprintf("Offset: %s", .offset_label(object$offset)),
    sprintf("Risk at offset: %s", .llm_num(facts$start, 6)),
    sprintf("Risk at mstop: %s", .llm_num(facts$end, 6)),
    sprintf("Risk reduction: %s",
            if (is.finite(facts$reduction)) sprintf("%.1f%%", 100 * facts$reduction) else "NA"),
    "",
    sprintf("Response scale: %s. %s", scale[1], scale[2])
  )
  if (!is.null(response_description)) {
    lines <- c(lines, sprintf("Response meaning: %s", response_description))
  }
  paste(lines, collapse = "\n")
}

# --- learner mechanics ----------------------------------------------

.llm_ordinal <- function(k) {
  switch(as.character(as.integer(k)),
         "1" = "first", "2" = "second", "3" = "third",
         paste0(as.integer(k), "th"))
}

.llm_smoothing <- function(spec) {
  df <- spec$df
  lambda <- spec$lambda
  if (!is.null(df) && length(df) == 1 && is.finite(df) && df > 0) {
    return(sprintf("df=%s", .llm_num(df)))
  }
  if (!is.null(lambda) && length(lambda) == 1 && is.finite(lambda) && lambda >= 0) {
    return(sprintf("lambda=%s", .llm_num(lambda)))
  }
  "unpenalized"
}

.llm_learner_note <- function(spec) {
  if (is.null(spec)) return("linear term, unpenalized")
  smoothing <- .llm_smoothing(spec)
  note <- switch(
    spec$type,
    scalar = paste0("linear term, ", smoothing),
    bbs = sprintf(
      "B-spline basis of degree %s on %s inner knots, %s, %s-difference penalty",
      .llm_num(spec$degree), .llm_num(spec$knots), smoothing,
      .llm_ordinal(spec$differences)
    ),
    bns = sprintf("unpenalized B-spline basis of degree %s on %s inner knots, %s",
                  .llm_num(spec$degree), .llm_num(spec$knots), smoothing),
    bmono = sprintf(
      "B-spline basis under a monotonicity constraint (%s by construction), %s",
      if (is.null(spec$constraint)) "increasing" else spec$constraint, smoothing
    ),
    brandom = paste0("ridge-penalized effect over the levels of the column, ", smoothing),
    bmrf = paste0("Markov random field over a supplied neighbourhood matrix, ", smoothing),
    bspatial = paste0("tensor-product spline over two covariates, ", smoothing),
    btree = "regression tree over one or more covariates",
    buser = paste0("caller-supplied design and penalty matrices, ", smoothing),
    smoothing
  )
  if (isTRUE(spec$center)) note <- paste0(note, "; centered")
  paste0(spec$type, ": ", note)
}

# What a learner can represent when its `df` sits at the shrinkage floor.
#
# A P-spline penalty of difference order d leaves a d-dimensional unpenalized
# subspace, and df(lambda) falls to d and stops. A learner asked for df = d is
# fitted at maximum shrinkage and confined to that subspace: second differences
# leave straight lines, first differences leave constants, and a bspatial tensor
# leaves the product of the marginal null spaces. The effect still has a shape,
# but not one the data chose, and a reader told only "monotone increasing" would
# take it for one. `center = TRUE` projects the null space out, so no floor
# applies. Same rule the learner constructors warn about at fit time.
.llm_null_space_note <- function(spec) {
  if (is.null(spec) || isTRUE(spec$center)) return(NULL)
  differences <- spec$differences
  if (is.null(differences) || !is.finite(differences)) return(NULL)
  null_rank <- switch(spec$type,
                      bbs = as.integer(differences),
                      bmono = as.integer(differences),
                      bspatial = as.integer(differences)^2,
                      NULL)
  df <- spec$df
  if (is.null(null_rank) || is.null(df) || !is.finite(df) || df > null_rank) {
    return(NULL)
  }
  span <- if (spec$type == "bspatial") {
    sprintf("the %d-dimensional unpenalized subspace of the tensor penalty",
            null_rank)
  } else {
    switch(as.character(null_rank),
           "1" = "constants", "2" = "straight lines", "3" = "quadratics",
           sprintf("the %d-dimensional unpenalized subspace", null_rank))
  }
  sprintf(paste("At the shrinkage floor: df=%s is the rank of this penalty's",
                "null space, which no finite lambda reaches, so the learner is",
                "fitted at maximum shrinkage and can only represent %s."),
          .llm_num(df), span)
}

# Risk reduction credited to each learner *index*, normalized over the path.
# Several learners can share a name (`bbs(x) + bmono(x)`), and a name-keyed share
# cannot separate them, which is why this does not call `varimp`.
.llm_importance <- function(object) {
  risk <- object$risk
  selected <- object$selected[seq_len(object$mstop)]
  n_learners <- length(object$learner_names)
  if (length(risk) < 2 || !length(selected) || !n_learners) return(NULL)
  riskdiff <- -1 * diff(risk)
  steps <- min(length(selected), length(riskdiff))
  reduction <- rep(0, n_learners)
  for (i in seq_len(steps)) {
    reduction[selected[i]] <- reduction[selected[i]] + riskdiff[i]
  }
  total <- sum(abs(reduction))
  if (total > 0) reduction / total else reduction
}

.llm_selection_facts <- function(object, index) {
  selected <- object$selected[seq_len(object$mstop)]
  hits <- which(selected == index)
  if (!length(hits)) return(NULL)
  list(updates = length(hits), mstop = length(selected),
       share = length(hits) / length(selected), first = hits[1])
}

# How one learner is named. A learner over several columns, and a name two
# learners share, both carry the learner type so the blocks stay distinguishable.
.llm_label <- function(object, index) {
  name <- object$learner_names[index]
  specs <- object$learner_specs
  if (is.null(specs)) return(name)
  if (sum(object$learner_names == name) > 1) {
    return(sprintf("%s (%s)", name, specs[[index]]$type))
  }
  name
}

.llm_selection_lines <- function(object, index, importance) {
  facts <- .llm_selection_facts(object, index)
  if (is.null(facts)) {
    return("Never selected: this term contributes exactly zero.")
  }
  lines <- sprintf(
    "Selected in %d of %d updates (%.1f%%), first at iteration %d",
    facts$updates, facts$mstop, 100 * facts$share, facts$first
  )
  if (!is.null(importance)) {
    lines <- c(lines, sprintf("Importance: %s of total risk reduction",
                              .llm_pct(importance)))
  }
  lines
}

.llm_levels_of <- function(object, column) {
  data <- object$data
  if (is.null(data) || !(column %in% names(data))) return(NULL)
  values <- data[[column]]
  if (is.factor(values)) return(levels(values))
  if (is.character(values)) return(sort(unique(values)))
  NULL
}

# --- importance table -----------------------------------------------

.llm_importance_block <- function(object, shares) {
  names <- object$learner_names
  rows <- lapply(seq_along(names), function(index) {
    facts <- .llm_selection_facts(object, index)
    list(label = .llm_label(object, index),
         share = if (is.null(shares)) NULL else shares[index],
         updates = if (is.null(facts)) 0L else facts$updates,
         first = if (is.null(facts)) NA_integer_ else facts$first)
  })
  order_by <- vapply(rows, function(row) {
    -(if (is.null(row$share)) 0 else abs(row$share))
  }, numeric(1))
  rows <- rows[order(order_by, -vapply(rows, function(r) r$updates, numeric(1)))]

  lines <- "Base learners, ranked:"
  for (row in rows) {
    parts <- paste0("  ", row$label)
    if (!is.null(row$share)) {
      parts <- c(parts, paste0("importance ", .llm_pct(row$share)))
    }
    parts <- c(parts, if (row$updates) {
      sprintf("%d update%s", row$updates, if (row$updates == 1) "" else "s")
    } else {
      "never selected (contributes zero)"
    })
    if (!is.na(row$first)) {
      parts <- c(parts, sprintf("entered at iteration %d", row$first))
    }
    lines <- c(lines, paste(parts, collapse = " \\u2014 "))
  }
  paste(lines, collapse = "\n")
}

# --- effect blocks --------------------------------------------------

.llm_effect_block_gam <- function(object, index, importance, effect, n_points) {
  spec <- object$learner_specs[[index]]
  header <- c(
    sprintf("Learner: %s  [%s]", .llm_label(object, index), .llm_learner_note(spec)),
    .llm_selection_lines(object, index, importance),
    .llm_null_space_note(spec)
  )
  if (is.null(effect)) {
    return(paste(c(header, paste(
      "No effect curve is available for this learner type in the R frontend;",
      "its contribution is in predict(type = \"link\") but is not decomposed here."
    )), collapse = "\n"))
  }
  column <- spec$name
  levels <- .llm_levels_of(object, column)
  header <- c(header, if (!is.null(levels) && spec$type %in% c("brandom", "bmrf")) {
    sprintf("Covariate: %s (categorical, %d levels)", column, length(levels))
  } else {
    sprintf("Covariate: %s (observed range [%s, %s])", column,
            .llm_num(min(effect$x)), .llm_num(max(effect$x)))
  })
  condensed <- .llm_condense(effect$x, effect$effect, n_points, levels)
  precision <- .llm_precision(condensed$values)
  header <- c(
    header,
    paste0("Contribution: ", .llm_pairs(condensed$keys, condensed$values, precision)),
    sprintf("Contribution range: [%s, %s] (span %s)",
            .llm_num(min(effect$effect)), .llm_num(max(effect$effect)),
            .llm_num(max(effect$effect) - min(effect$effect)))
  )
  if (is.null(levels)) {
    header <- c(header, sprintf("Shape: %s", .llm_shape(condensed$x, condensed$y)))
  }
  paste(header, collapse = "\n")
}

.llm_effect_block_glm <- function(object, index, importance) {
  name <- object$learner_names[index]
  coefficients <- coef(object)
  coefficient <- if (name %in% names(coefficients)) as.numeric(coefficients[name]) else 0
  lines <- c(sprintf("Term: %s  [linear, unpenalized]", name))
  if (name %in% names(object$data)) {
    x <- as.numeric(object$data[[name]])
    contributions <- coefficient * x
    lines <- c(
      lines,
      sprintf("Coefficient: %s  (contribution = %s * %s)",
              .llm_num(coefficient, 6), .llm_num(coefficient, 6), name),
      sprintf("Covariate range: [%s, %s]", .llm_num(min(x)), .llm_num(max(x))),
      sprintf("Contribution over that range: [%s, %s]",
              .llm_num(min(contributions)), .llm_num(max(contributions)))
    )
  } else {
    lines <- c(lines, sprintf("Coefficient: %s", .llm_num(coefficient, 6)))
  }
  paste(c(lines, .llm_selection_lines(object, index, importance)), collapse = "\n")
}

.llm_which_indices <- function(object, which, top, shares) {
  names <- object$learner_names
  if (is.null(which)) {
    indices <- unique(object$selected[seq_len(object$mstop)])
  } else if (is.numeric(which)) {
    indices <- as.integer(which)
    if (any(indices < 1 | indices > length(names))) {
      stop("learner index out of range", call. = FALSE)
    }
  } else {
    missing <- setdiff(which, names)
    if (length(missing)) {
      stop("unknown base learner: ", paste(missing, collapse = ", "), call. = FALSE)
    }
    indices <- match(which, names)
  }
  if (!is.null(shares)) indices <- indices[order(-abs(shares[indices]))]
  if (!is.null(top)) indices <- utils::head(indices, as.integer(top))
  indices
}

.llm_effects_block <- function(object, shares, which, top, n_points, effects) {
  indices <- .llm_which_indices(object, which, top, shares)
  if (!length(indices)) {
    return(paste("Effects: the path selected no base learner, so the model is",
                 "the offset alone."))
  }
  total <- length(unique(object$selected[seq_len(object$mstop)]))
  blocks <- vapply(indices, function(index) {
    importance <- if (is.null(shares)) NULL else shares[index]
    if (is.null(effects)) {
      .llm_effect_block_glm(object, index, importance)
    } else {
      name <- object$learner_names[index]
      .llm_effect_block_gam(object, index, importance, effects[[name]], n_points)
    }
  }, character(1))
  heading <- sprintf(
    "Effects (%d of %d selected learners, most important first):",
    length(indices), total
  )
  paste(c(heading, blocks), collapse = "\n\n")
}

# --- the prompt -----------------------------------------------------

.llm_assemble <- function(object, kind, effects, which, top, n_points,
                          dataset_description, response_description, task,
                          include, expert, output, num_sentences) {
  unknown <- setdiff(include, .LLM_SECTIONS)
  if (length(unknown)) {
    stop("unknown section(s): ", paste(unknown, collapse = ", "),
         "; choose from ", paste(.LLM_SECTIONS, collapse = ", "), call. = FALSE)
  }
  output <- match.arg(output, c("text", "messages", "turns"))

  shares <- .llm_importance(object)
  parts <- character(0)
  if ("reading" %in% include) parts <- c(parts, .LLM_READING)
  if ("facts" %in% include) {
    parts <- c(parts, .llm_facts_block(object, kind, response_description))
  }
  if ("importance" %in% include) {
    parts <- c(parts, .llm_importance_block(object, shares))
  }
  if ("effects" %in% include) {
    parts <- c(parts, .llm_effects_block(object, shares, which, top, n_points, effects))
  }
  if (!is.null(dataset_description) && "facts" %in% include) {
    parts <- c(parts, paste0("What the data are:\n", dataset_description))
  }
  if ("caveats" %in% include) parts <- c(parts, .LLM_CAVEATS)
  if ("task" %in% include) {
    parts <- c(parts, if (is.null(task)) .LLM_DEFAULT_TASK else task)
  }

  text <- paste(parts, collapse = "\n\n")
  if (output == "text") return(text)
  system <- sprintf(.LLM_SYSTEM, expert)
  if (output == "turns") {
    return(list(
      system = system,
      turns = c(text, .LLM_PROBE_SURPRISES,
                sprintf(.LLM_PROBE_CONDENSE, as.integer(num_sentences)))
    ))
  }
  list(
    list(role = "system", content = system),
    list(role = "user", content = text)
  )
}

#' Render a fitted model as a prompt for a language model
#'
#' Builds one prompt describing how the fit was produced, what each retained
#' base learner contributes over the observed covariate range, how much of the
#' risk it removed, and which questions the fit cannot answer. Nothing is sent
#' anywhere: the return value is text (or chat messages) for the caller to pass
#' to a model of their choosing. See \code{\link{llm_describe}} to send it.
#'
#' @param object A fitted \code{mboostrs_glmboost} or \code{mboostrs_gamboost} object.
#' @param ... Passed to the method.
#' @return A character string, or a list of chat messages when
#'   \code{output = "messages"}, or a list of \code{system} and \code{turns}
#'   when \code{output = "turns"}.
#' @examples
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(x1 = runif(n, 0, 10), x2 = runif(n))
#' dat$y <- sin(dat$x1) + 2 * dat$x2 + rnorm(n, 0, 0.3)
#' mod <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = dat,
#'                 control = boost_control(mstop = 60, nu = 0.1))
#' cat(llm_explain_prompt(mod, include = c("importance", "effects")))
#' @export
llm_explain_prompt <- function(object, ...) UseMethod("llm_explain_prompt")

#' @rdname llm_explain_prompt
#' @param which Base learners to describe, by name or index. NULL = every
#'   selected learner.
#' @param top Keep only this many learners, most important first.
#' @param n_points Values per effect: the grid size for a smooth effect, or the
#'   number of equal-count intervals anything else is condensed to.
#' @param dataset_description What the data are, where they came from, what the
#'   columns mean. The fit cannot carry this, and the model reads it as context.
#' @param response_description What the response measures, in your own terms.
#' @param task The closing instruction. Default: an analyst's brief in four
#'   parts - findings, surprising patterns, what could mislead, next check.
#' @param include Sections to write, from \code{"reading"}, \code{"facts"},
#'   \code{"importance"}, \code{"effects"}, \code{"caveats"}, \code{"task"}.
#' @param expert Who the reader is told to be, in the system message.
#' @param output \code{"text"} for one string, \code{"messages"} for a
#'   system/user pair, \code{"turns"} for the three-turn reasoning sequence.
#' @param num_sentences Sentence budget for the closing turn of
#'   \code{output = "turns"}.
#' @export
llm_explain_prompt.mboostrs_glmboost <- function(object, which = NULL, top = NULL,
                                                 n_points = 12,
                                                 dataset_description = NULL,
                                                 response_description = NULL,
                                                 task = NULL,
                                                 include = c("reading", "facts",
                                                             "importance", "effects",
                                                             "caveats", "task"),
                                                 expert = "an expert statistician and applied data analyst",
                                                 output = c("text", "messages", "turns"),
                                                 num_sentences = 7, ...) {
  .llm_assemble(object, "glmboost", NULL, which, top, n_points,
                dataset_description, response_description, task, include, expert,
                output, num_sentences)
}

#' @rdname llm_explain_prompt
#' @export
llm_explain_prompt.mboostrs_gamboost <- function(object, which = NULL, top = NULL,
                                                 n_points = 12,
                                                 dataset_description = NULL,
                                                 response_description = NULL,
                                                 task = NULL,
                                                 include = c("reading", "facts",
                                                             "importance", "effects",
                                                             "caveats", "task"),
                                                 expert = "an expert statistician and applied data analyst",
                                                 output = c("text", "messages", "turns"),
                                                 num_sentences = 7, ...) {
  # Ask for every learner's effect in one call: the per-learner curves are keyed
  # by name, and a filtered call would have to re-derive the same list.
  effects <- partial_effects(object, n_points = n_points)
  .llm_assemble(object, "gamboost", effects, which, top, n_points,
                dataset_description, response_description, task, include, expert,
                output, num_sentences)
}

# --- sending the prompt ---------------------------------------------

#' Default endpoint for \code{\link{llm_describe}}
#'
#' NVIDIA's hosted inference endpoint, which speaks the OpenAI
#' chat-completions protocol. Any endpoint that speaks it works: pass
#' \code{base_url} for a local NIM container, vLLM, llama.cpp or Ollama.
#' @export
LLM_DEFAULT_BASE_URL <- "https://integrate.api.nvidia.com/v1"

#' Default model id for \code{\link{llm_describe}}
#' @export
LLM_DEFAULT_MODEL <- "nvidia/nemotron-3-ultra-550b-a55b"

.llm_from_env <- function(names) {
  for (name in names) {
    value <- Sys.getenv(name, unset = "")
    if (nzchar(value)) return(value)
  }
  ""
}

.llm_reply_text <- function(payload) {
  choices <- payload$choices
  if (is.null(choices) || !length(choices)) {
    stop("the endpoint returned no choices", call. = FALSE)
  }
  message <- choices[[1]]$message
  content <- if (is.null(message$content)) "" else trimws(message$content)
  if (nzchar(content)) return(content)
  reasoning <- if (is.null(message$reasoning_content)) "" else trimws(message$reasoning_content)
  if (nzchar(reasoning)) {
    stop("the model spent the whole token budget reasoning and returned no ",
         "answer; raise max_tokens", call. = FALSE)
  }
  stop("the endpoint returned an empty reply", call. = FALSE)
}

# One chat-completions request. The `curl` package is used when installed; the
# `curl` command-line binary otherwise, so a POST needs no new hard dependency.
.llm_post_chat <- function(messages, llm, base_url, api_key, max_tokens,
                           temperature, timeout) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("llm_describe() needs the jsonlite package to build the request",
         call. = FALSE)
  }
  url <- paste0(sub("/+$", "", base_url), "/chat/completions")
  body <- jsonlite::toJSON(
    list(model = llm, messages = messages,
         max_tokens = as.integer(max_tokens), temperature = as.numeric(temperature)),
    auto_unbox = TRUE
  )

  if (requireNamespace("curl", quietly = TRUE)) {
    handle <- curl::new_handle()
    curl::handle_setheaders(
      handle,
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json",
      Accept = "application/json"
    )
    curl::handle_setopt(handle, post = TRUE, postfields = body,
                        timeout = as.integer(timeout))
    response <- curl::curl_fetch_memory(url, handle = handle)
    text <- rawToChar(response$content)
    if (response$status_code != 200L) {
      stop(sprintf("%s returned HTTP %d for model '%s': %s", url,
                   response$status_code, llm, substr(text, 1, 600)), call. = FALSE)
    }
    return(jsonlite::fromJSON(text, simplifyVector = FALSE))
  }

  if (!nzchar(Sys.which("curl"))) {
    stop("llm_describe() needs either the curl R package or the curl ",
         "command-line tool. Use llm_explain_prompt() to get the prompt text ",
         "without sending it anywhere.", call. = FALSE)
  }
  body_file <- tempfile(fileext = ".json")
  on.exit(unlink(body_file), add = TRUE)
  writeLines(as.character(body), body_file, useBytes = TRUE)
  output <- system2(
    "curl",
    c("-sS", "--max-time", as.character(as.integer(timeout)),
      "-w", shQuote("\n%{http_code}"),
      "-H", shQuote(paste("Authorization: Bearer", api_key)),
      "-H", shQuote("Content-Type: application/json"),
      "-H", shQuote("Accept: application/json"),
      "--data-binary", shQuote(paste0("@", body_file)), shQuote(url)),
    stdout = TRUE, stderr = TRUE
  )
  status <- utils::tail(output, 1)
  text <- paste(utils::head(output, -1), collapse = "\n")
  if (!identical(status, "200")) {
    stop(sprintf("%s returned HTTP %s for model '%s': %s", url, status, llm,
                 substr(text, 1, 600)), call. = FALSE)
  }
  jsonlite::fromJSON(text, simplifyVector = FALSE)
}

#' Send a model's description to a language model and return the reply
#'
#' The one function here that makes a network request. It builds the prompt with
#' \code{\link{llm_explain_prompt}}, posts it to an OpenAI-compatible
#' chat-completions endpoint, and returns the text that comes back. The default
#' endpoint is NVIDIA's hosted inference API and the default model is
#' \code{"nvidia/nemotron-3-ultra-550b-a55b"}; the API key is read from the environment.
#'
#' @param object A fitted \code{mboostrs_glmboost} or \code{mboostrs_gamboost} object.
#' @param ... Passed to \code{\link{llm_explain_prompt}}
#'   (\code{dataset_description}, \code{top}, \code{n_points}, ...).
#' @return A Markdown-formatted string, or the whole conversation when
#'   \code{return_conversation = TRUE}. Render it with a Markdown-capable
#'   renderer rather than printing it as plain text.
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 100
#' dat <- data.frame(x1 = runif(n, 0, 10))
#' dat$y <- sin(dat$x1) + rnorm(n, 0, 0.3)
#' mod <- gamboost(y ~ bbs(x1, df = 4), data = dat,
#'                 control = boost_control(mstop = 60, nu = 0.1))
#' cat(llm_describe(mod, dataset_description = "Simulated sine wave."))
#' }
#' @export
llm_describe <- function(object, ...) UseMethod("llm_describe")

#' @rdname llm_describe
#' @param llm Model id the endpoint serves.
#' @param base_url Endpoint root, without \code{/chat/completions}. Default: the
#'   environment variable \code{MBOOSTRS_LLM_BASE_URL} or \code{NVIDIA_BASE_URL},
#'   else \code{LLM_DEFAULT_BASE_URL}.
#' @param api_key Bearer token. Default: the first of
#'   \code{MBOOSTRS_LLM_API_KEY}, \code{NVIDIA_API_KEY}, \code{OPENAI_API_KEY}
#'   that is set.
#' @param reasoning FALSE sends one request. TRUE runs the three-turn sequence:
#'   describe the effects, name what is surprising and why, then compress both
#'   into \code{num_sentences}.
#' @param num_sentences Sentence budget for the closing turn when
#'   \code{reasoning = TRUE}.
#' @param max_tokens,temperature,timeout Per-request generation and socket settings.
#' @param return_conversation Return the full message list instead of the final reply.
#' @export
llm_describe.mboostrs_gamboost <- function(object, llm = LLM_DEFAULT_MODEL,
                                           base_url = NULL, api_key = NULL,
                                           reasoning = FALSE, num_sentences = 7,
                                           max_tokens = 4000, temperature = 0.7,
                                           timeout = 180,
                                           return_conversation = FALSE, ...) {
  if (is.null(api_key)) {
    api_key <- .llm_from_env(c("MBOOSTRS_LLM_API_KEY", "NVIDIA_API_KEY",
                               "OPENAI_API_KEY"))
  }
  if (!nzchar(api_key)) {
    stop("llm_describe() needs an API key: pass api_key, or set one of ",
         "MBOOSTRS_LLM_API_KEY, NVIDIA_API_KEY, OPENAI_API_KEY. Use ",
         "llm_explain_prompt() to get the prompt text without sending it ",
         "anywhere.", call. = FALSE)
  }
  if (is.null(base_url)) {
    base_url <- .llm_from_env(c("MBOOSTRS_LLM_BASE_URL", "NVIDIA_BASE_URL"))
  }
  if (!nzchar(base_url)) base_url <- LLM_DEFAULT_BASE_URL

  send <- function(conversation) {
    .llm_reply_text(.llm_post_chat(conversation, llm, base_url, api_key,
                                   max_tokens, temperature, timeout))
  }

  # A prompt with no closing instruction spends a request and comes back asking
  # what the reader wanted, so the task section is not optional when sending.
  prompt_args <- list(...)
  if (!is.null(prompt_args$include) && !("task" %in% prompt_args$include)) {
    prompt_args$include <- c(prompt_args$include, "task")
  }

  if (isTRUE(reasoning)) {
    plan <- do.call(llm_explain_prompt, c(
      list(object, output = "turns", num_sentences = num_sentences), prompt_args
    ))
    conversation <- list(list(role = "system", content = plan$system))
    for (turn in plan$turns) {
      conversation <- c(conversation, list(list(role = "user", content = turn)))
      conversation <- c(conversation,
                        list(list(role = "assistant", content = send(conversation))))
    }
  } else {
    conversation <- do.call(llm_explain_prompt,
                            c(list(object, output = "messages"), prompt_args))
    conversation <- c(conversation,
                      list(list(role = "assistant", content = send(conversation))))
  }

  if (isTRUE(return_conversation)) return(conversation)
  conversation[[length(conversation)]]$content
}

#' @rdname llm_describe
#' @export
# The two model classes take the same arguments and the same path through
# `llm_explain_prompt`, so one function serves both names rather than a wrapper
# whose signature would have to be kept in step with the other's.
llm_describe.mboostrs_glmboost <- llm_describe.mboostrs_gamboost
