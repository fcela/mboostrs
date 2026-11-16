# Centered linear designs and binary class labels against mboost 2.9.11.
if (!requireNamespace("mboost", quietly = TRUE)) quit(status = 0)
suppressMessages(library(mboostrs))
suppressMessages(library(mboost))
ok <- function(value, message) if (!isTRUE(value)) stop(message, call. = FALSE)
set.seed(17)
d <- data.frame(x = rnorm(80, 3), z = rnorm(80, -2))
d$y <- 2 * d$x + rnorm(80)
wt <- runif(80, 0.5, 2)
for (center in c(TRUE, FALSE)) for (intercept in c(TRUE, FALSE)) {
  formula <- if (intercept) y ~ x + z else y ~ x + z - 1
  for (weighted in c(TRUE, FALSE)) {
    w <- if (weighted) wt else NULL
    up <- suppressWarnings(mboost::glmboost(formula, d, center = center, weights = w,
                  control = mboost::boost_control(mstop = 1)))
    ours <- suppressWarnings(mboostrs::glmboost(formula, d, center = center, weights = w,
                  control = mboostrs::boost_control(mstop = 1)))
    for (which in list(NULL, seq_along(ours$learner_names), "z")) {
      a <- mboost::extract(up, "design", which = which, asmatrix = TRUE)
      b <- mboostrs::extract(ours, "design", which = which, asmatrix = TRUE)
      ok(identical(dim(a), dim(b)) && max(abs(a - b)) < 1e-14, "glmboost design differs")
    }
    ok(is.vector(extract(ours, "design", which = "z")), "single design column must drop by default")
    ok(identical(unname(mboost::extract(up, "variable.names")),
                 unname(mboostrs::extract(ours, "variable.names"))), "design names differ")
    ok(max(abs(mboost::extract(up, "weights") - extract(ours, "weights"))) < 1e-14,
       "fit weights differ")
    ok(max(abs(residuals(up) - residuals(ours))) < 1e-12, "Gaussian residuals differ")
  }
}

set.seed(21)
d <- data.frame(x = rnorm(100))
# Reverse alphabetic order to check that prediction follows the factor levels.
d$y <- factor(ifelse(rbinom(100, 1, plogis(d$x)), "case", "control"),
              levels = c("control", "case"))
nd <- data.frame(x = c(-2, -0.5, 0.5, 2))
for (fit_name in c("glmboost", "gamboost")) {
  formula <- if (fit_name == "gamboost") y ~ bols(x) else y ~ x
  fit_up <- get(fit_name, asNamespace("mboost"))
  fit_ours <- get(fit_name, asNamespace("mboostrs"))
  for (family_name in c("Binomial", "AdaExp", "HingeLoss", "AUC")) {
    up <- fit_up(formula, d, family = get(family_name, asNamespace("mboost"))(),
                 control = mboost::boost_control(mstop = 30))
    ours <- fit_ours(formula, d, family = get(family_name, asNamespace("mboostrs"))(),
                     control = mboostrs::boost_control(mstop = 30))
    for (k in c(5L, 30L)) {
      mboost::mstop(up) <- k
      for (newdata in list(NULL, nd)) {
        a <- predict(up, newdata, type = "class")
        b <- predict(ours, newdata, mstop = k, type = "class")
        ok(identical(levels(a), levels(b)) && identical(as.character(a), as.character(b)),
           paste(fit_name, family_name, "class predictions differ"))
      }
    }
  }
  tie <- fit_ours(formula, data.frame(x = rep(0, 10), y = factor(rep(c("a", "b"), 5))),
                  family = mboostrs::Binomial(), control = mboostrs::boost_control(mstop = 1))
  ok(all(predict(tie, type = "class") == "a"), "tied scores must choose the first level")
  regression <- fit_ours(formula, data.frame(x = 1:10, y = 1:10),
                         control = mboostrs::boost_control(mstop = 1))
  error <- tryCatch(predict(regression, type = "class"), error = conditionMessage)
  ok(is.character(error) && grepl("binary family", error), "regression class prediction must refuse")
}
cat("glmboost extract and binary class parity: PASS\n")
