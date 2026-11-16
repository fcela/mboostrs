suppressMessages(library(mboostrs))
set.seed(44)
d <- data.frame(x = rnorm(120))
d$y <- rgamma(120, shape = 20, scale = 5000 * exp(.2 * d$x) / 20)
w <- rep(1, 120); w[1:8] <- seq(500, 1200, 100)
for (kind in c("glm", "gam")) {
  fit <- if (kind == "glm") glmboost else gamboost
  formula <- if (kind == "glm") y ~ x else y ~ bols(x)
  model <- fit(formula, d, family = GammaReg(), weights = w, weight_policy = "raw",
                control = boost_control(120, .5, step = "line_search"))
  stopifnot(max(diff(model$risk)) < 1e-7, model$control$step == "line_search")
  stopifnot(max(abs(predict(model, d) - fitted(model))) < 1e-10)
  folds <- outer(seq_len(120) %% 3, 0:2, "!=") * 1
  result <- cvrisk(model, folds = folds)
  stopifnot(all(is.finite(result$risk_matrix)))
}
g <- glmboost(y ~ x, d, weights = w, family = GammaReg(), weight_policy = "raw",
                control = boost_control(1000, 1, step = "line_search"))
reference <- stats::glm(y ~ x, d, family = stats::Gamma(link = "log"), weights = w)
stopifnot(max(abs(predict(g) - predict(reference))) < 1e-5)

# The step option applies to the existing distributional core too.
normal <- data.frame(x = seq(-1, 1, length.out = 60))
normal$y <- 1 + normal$x + rnorm(60, sd = .5)
for (kind in c("glm", "gam")) {
  fit <- if (kind == "glm") glmboostLSS else gamboostLSS
  formula <- if (kind == "glm") y ~ x else y ~ bols(x)
  lss <- fit(formula, normal, families = GaussianLSS(), method = "cyclic",
                      control = boost_control(30, .1, step = "line_search"))
  stopifnot(all(is.finite(unlist(predict(lss)))))
  cv <- if (kind == "glm") cvrisk_glmboostLSS else cvrisk_gamboostLSS
  folds <- outer(seq_len(nrow(normal)) %% 3, 0:2, "!=") * 1
  curve <- cv(formula, normal, families = GaussianLSS(), folds = folds,
              grid = rbind(c(10L, 10L), c(20L, 20L)),
              control = boost_control(30, .1, step = "line_search"))
  stopifnot(all(is.finite(curve)))
}
# Non-cyclic selection compares the joint risk of candidate updates and the
# core deliberately rejects a separate per-direction line search.
error <- tryCatch(glmboostLSS(y ~ x, normal, families = GaussianLSS(),
                             method = "noncyclic",
                             control = boost_control(30, .1, step = "line_search")),
                  error = identity)
stopifnot(inherits(error, "error"), grepl("cyclic LSS", conditionMessage(error)))
cat("Gamma line search and step propagation: PASS\n")
