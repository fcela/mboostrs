# Regression: GAMBoost R prediction must fail closed when the R bridge cannot
# faithfully replay a selected learner. Silent partial predictions are worse
# than an explicit unsupported-case error.
library(mboostrs)

expect_error <- function(expr, pattern) {
  ok <- FALSE
  msg <- NULL
  tryCatch(
    expr,
    error = function(e) {
      ok <<- grepl(pattern, conditionMessage(e), fixed = TRUE)
      msg <<- conditionMessage(e)
    }
  )
  if (!ok) {
    stop(sprintf("expected error containing %s, got %s", pattern, msg %||% "no error"))
  }
}

`%||%` <- function(x, y) if (is.null(x)) y else x

set.seed(24)
n <- 80
x <- runif(n)
y <- ifelse(x > 0.5, 1.0, -1.0) + rnorm(n, sd = 0.1)
d <- data.frame(x = x, y = y)

tree_fit <- gamboost(y ~ btree(x), data = d, family = Gaussian(),
                     control = boost_control(mstop = 20, nu = 0.1))
if (!any(tree_fit$selected == 1L)) stop("test setup did not select btree learner")
stopifnot(isTRUE(all.equal(as.numeric(predict(tree_fit)), as.numeric(fitted(tree_fit)))))
# Tree steps are evaluated from their stored trees, so new-data prediction
# at the training rows reproduces the fitted values (it used to be refused).
stopifnot(isTRUE(all.equal(as.numeric(predict(tree_fit, newdata = d)),
                           as.numeric(fitted(tree_fit)))))
stopifnot(isTRUE(all.equal(as.numeric(predict(tree_fit, newdata = d, mstop = 10)),
                           as.numeric(predict(tree_fit, mstop = 10)))))

# A centered smooth used to be refused here: the bridge rebuilt the raw
# `p`-column basis against a fit whose coefficients had `p - d` entries. It now
# rebuilds through the same centering transform the fit used, so the refusal is
# gone and the replay is exact -- which is the property worth pinning, because
# a rebuild that runs but lands in a *different* reduced space would look like
# a poor fit rather than an error.
center_fit <- gamboost(y ~ bbs(x, center = TRUE), data = d, family = Gaussian(),
                       control = boost_control(mstop = 20, nu = 0.1))
if (!any(center_fit$selected == 1L)) stop("test setup did not select centered bbs learner")
replayed <- predict(center_fit, newdata = d)
if (max(abs(replayed - fitted(center_fit))) > 1e-10) {
  stop(sprintf("centered bbs newdata replay is off by %.3e; it must reproduce fitted()",
               max(abs(replayed - fitted(center_fit)))))
}

# The guard itself still has to fire on a genuine mismatch. `object$coefs` is
# what it compares against, so a truncated coefficient vector is the minimal
# way to produce one without a learner that legitimately cannot be replayed.
damaged <- center_fit
damaged$coefs[[1]] <- damaged$coefs[[1]][-1]
expect_error(
  predict(damaged, newdata = d),
  "cannot safely predict selected learner"
)

cat("gamboost R predict guards: PASS\n")
