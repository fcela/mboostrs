# CV recipes fitted inside training folds; scoring and aggregation use the core.
suppressMessages(library(mboostrs))
set.seed(71)
d <- data.frame(x = rnorm(36), z = rnorm(36))
d$y <- 0.5 * d$x + d$z + rnorm(36)
folds <- outer(seq_len(36) %% 3, 0:2, "!=") * 1
for (kind in c("glmboost", "gamboost")) {
  fit <- get(kind, asNamespace("mboostrs"))
  formula <- if (kind == "glmboost") y ~ scale(x) + z else y ~ bbs(x, knots = 4) + bols(z)
  model <- fit(formula, d, control = boost_control(8, 0.02))
  cv <- cvrisk(model, folds = folds, preprocessing = "refit")
  stopifnot(cv$preprocessing == "refit", mstop(cv) > 0)
  for (b in 1:3) {
    train <- folds[, b] > 0
    held <- !train
    reference <- fit(formula, d[train, ], control = boost_control(8, 0.02))
    for (k in 0:8) {
      eta <- if (k == 0) rep(reference$offset, sum(held))
             else predict(reference, d[held, ], mstop = k)
      expected <- mean((d$y[held] - eta)^2)
      stopifnot(abs(cv$risk_matrix[b, k + 1] - expected) < 1e-10)
    }
  }
  stopifnot(cvrisk(model, folds = folds)$preprocessing == "fixed")
}

# R's model.frame stores scale() parameters in predvars for prediction replay.
train <- data.frame(x = 0:11, y = sin(0:11))
held <- data.frame(x = c(1000, 2000))
fit <- glmboost(y ~ scale(x), train, control = boost_control(4))
eta <- predict(fit, held)
manual <- held
manual$x <- (manual$x - mean(train$x)) / sd(train$x)
direct <- glmboost(y ~ x, transform(train, x = (x - mean(x)) / sd(x)), control = boost_control(4))
stopifnot(max(abs(eta - predict(direct, manual))) < 1e-10)
err <- tryCatch(cvrisk(fit, folds = matrix(1, 12, 1), preprocessing = "refit"), error = conditionMessage)
stopifnot(is.character(err), grepl("training and evaluation", err))
cat("CV preprocessing: PASS\n")
