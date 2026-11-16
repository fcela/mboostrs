suppressMessages(library(mboostrs))
set.seed(0)
d <- data.frame(x = runif(250, -1, 1))
d$y <- rnbinom(250, size = 2, mu = exp(.5 + d$x))
w <- sample(1:4, nrow(d), replace = TRUE)
for (kind in c("glm", "gam")) {
  fit <- if (kind == "glm") glmboost else gamboost
  formula <- if (kind == "glm") y ~ x else y ~ bols(x)
  model <- fit(formula, d, weights = w, weight_policy = "raw",
                family = NBinomial(), control = boost_control(100))
  longer <- fit(formula, d, weights = w, weight_policy = "raw",
                family = NBinomial(), control = boost_control(120))
  for (current in list(model, model[20], longer)) {
    stopifnot(is.numeric(size(current)), length(size(current)) == 1,
              size(current) > 0, identical(size(current), nuisance(current)))
    mean <- predict(current, type = "response")
    expected <- sum(w * dnbinom(d$y, size = size(current), mu = mean, log = TRUE))
    stopifnot(abs(as.numeric(logLik(current)) - expected) < 1e-8)
  }
  stopifnot(size(model[20]) != size(model))
  path <- tempfile(fileext = ".rds")
  saveRDS(model[20], path)
  stopifnot(size(readRDS(path)) == size(model[20]))
  unlink(path)
  for (family in list(Poisson(), Hurdle())) {
    positive <- d; positive$y <- positive$y + 1
    other <- fit(formula, positive, family = family, control = boost_control(10))
    error <- tryCatch(size(other), error = identity)
    stopifnot(inherits(error, "error"), grepl("NBinomial", conditionMessage(error)))
  }
}
for (argument in c("alpha", "dispersion", "size")) {
  error <- tryCatch(do.call(NBinomial, setNames(list(.5), argument)), error = identity)
  stopifnot(inherits(error, "error"), grepl("alpha = 1/size", conditionMessage(error)))
}
cat("NBinomial named size and density: PASS\n")
