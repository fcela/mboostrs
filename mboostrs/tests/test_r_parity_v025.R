# R frontend parity tests for v0.2.5 additions (btree histogram/regularization,
# bbs knot_placement/bins validation, and adaptive_conformal_stream).

set.seed(42)

# 1. Online Adaptive Conformal Inference stream under 2.5x scale shift
n <- 400L
obs <- rnorm(n, sd = 1.5)
pred <- rep(0.0, n)
stream_res <- adaptive_conformal_stream(pred, obs, half_width = 0.75, level = 0.80, lr = 0.03)
stopifnot(
  is.list(stream_res),
  length(stream_res$lower) == n,
  length(stream_res$upper) == n,
  stream_res$coverage > 0.72,
  stream_res$coverage < 0.90
)

# 2. Formula parser checks for btree(split_method, lambda_leaf, colsample)
dat <- data.frame(
  y = rnorm(80),
  x1 = rnorm(80),
  x2 = runif(80),
  cat = factor(sample(c("a", "b", "c"), 80, replace = TRUE))
)
parsed_tree <- mboostrs:::.parse_btree(
  "x1, x2, cat, maxdepth = 3, selection = 'sse', split_method = 'histogram', lambda_leaf = 1.5, colsample = 0.75, colsample_seed = 9",
  dat
)
stopifnot(
  identical(parsed_tree$spec$split_method, "histogram"),
  identical(parsed_tree$spec$lambda_leaf, 1.5),
  identical(parsed_tree$spec$colsample, 0.75),
  identical(parsed_tree$spec$colsample_seed, 9L),
  identical(parsed_tree$spec$categorical, c(0L, 0L, 3L))
)
