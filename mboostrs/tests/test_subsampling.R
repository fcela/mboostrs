# Stochastic component subsampling in the R frontend.
# Off by default; seeded for reproducibility; reported in the execution report.
# Run with: Rscript mboostrs/tests/test_subsampling.R

library(mboostrs)

passed <- 0L
failed <- 0L
errors <- list()

check <- function(name, expr) {
  result <- tryCatch(
    { stopifnot(isTRUE(expr)); TRUE },
    error = function(e) { errors[[name]] <<- conditionMessage(e); FALSE },
    warning = function(w) { errors[[name]] <<- conditionMessage(w); FALSE }
  )
  if (result) {
    passed <<- passed + 1L
    cat("  PASS: ", name, "\n", sep = "")
  } else {
    failed <<- failed + 1L
    cat("  FAIL: ", name, " — ", errors[[name]], "\n", sep = "")
  }
}

set.seed(1)
n <- 800
p <- 30
X <- matrix(rnorm(n * p), n, p)
y <- 1.5 * X[, 1] - 0.9 * X[, 2] + 0.6 * X[, 3] + rnorm(n, sd = 0.3)
d <- as.data.frame(X)
names(d) <- paste0("x", seq_len(p))
d$y <- y
form <- as.formula(paste("y ~", paste0("x", seq_len(p), collapse = " + ")))

# --- boost_control validates the fraction ---
check("invalid fraction rejected (>=1)",
      inherits(try(boost_control(mstop = 10, subsample = 1.0), silent = TRUE), "try-error"))
check("invalid fraction rejected (<=0)",
      inherits(try(boost_control(mstop = 10, subsample = 0.0), silent = TRUE), "try-error"))
check("valid fraction accepted",
      is.list(boost_control(mstop = 10, subsample = 0.5, subsample_seed = 1)))

# --- off by default ---
m_off <- glmboost(form, data = d, control = boost_control(mstop = 200, nu = 0.1))
check("off by default reports zero subsampled iterations",
      execution_report(m_off)$component_subsample_iterations == 0)

# --- subsampled fit ---
ctrl <- boost_control(mstop = 200, nu = 0.1, subsample = 0.3, subsample_seed = 42)
m1 <- glmboost(form, data = d, control = ctrl)
m2 <- glmboost(form, data = d, control = ctrl)

check("same seed is deterministic",
      isTRUE(all.equal(fitted(m1), fitted(m2))))
check("report counts subsampled iterations",
      execution_report(m1)$component_subsample_iterations == 200)

m_seed7 <- glmboost(form, data = d,
                    control = boost_control(mstop = 200, nu = 0.1,
                                            subsample = 0.3, subsample_seed = 7))
check("different seed changes the path",
      !isTRUE(all.equal(fitted(m1), fitted(m_seed7))))

# signal columns dominate the coefficients
co <- abs(unlist(coef(m1)))
top3 <- sort(names(sort(co, decreasing = TRUE))[1:3])
check("subsampled fit recovers the signal columns",
      identical(top3, c("x1", "x2", "x3")))

# --- gamboost path ---
dg <- as.data.frame(matrix(runif(n * 6, -2, 2), n, 6))
names(dg) <- paste0("z", 1:6)
dg$y <- sin(dg$z1) + 0.5 * dg$z2 + rnorm(n, sd = 0.3)
form_g <- as.formula(paste("y ~", paste0("bbs(z", 1:6, ", df = 4)", collapse = " + ")))
mg1 <- gamboost(form_g, data = dg,
                control = boost_control(mstop = 120, nu = 0.1, subsample = 0.5, subsample_seed = 3))
mg2 <- gamboost(form_g, data = dg,
                control = boost_control(mstop = 120, nu = 0.1, subsample = 0.5, subsample_seed = 3))
check("gamboost subsampling is deterministic",
      isTRUE(all.equal(fitted(mg1), fitted(mg2))))
check("gamboost subsampling routes off the Gram fast path",
      execution_report(mg1)$route == "general")

cat("\n", passed, " passed, ", failed, " failed\n", sep = "")
if (failed > 0L) quit(status = 1L)
