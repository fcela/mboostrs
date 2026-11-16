library(mboostrs)

# The core transform kernels against base R: mean/sd, median, type-7 quantiles
# (the linear-interpolation definition), approx() with the end values held.
call <- function(name, ...) .Call(paste0("wrap__r_transform_", name), ..., PACKAGE = "mboostrs")

set.seed(11)
x <- rnorm(400, 3, 2)
ms <- call("mean_sd", x)
stopifnot(abs(ms[1] - mean(x)) < 1e-12, abs(ms[2] - sd(x)) < 1e-12)
stopifnot(is.na(call("mean_sd", 2)[2]))

for (n in c(1, 2, 3, 4, 7, 100)) {
  v <- rnorm(n)
  stopifnot(identical(call("median", v), median(v)))
}

for (v in list(x, c(1, 4), c(5, 1, 3), c(rep(0, 200), rnorm(20)),
               as.double(sample(0:3, 300, replace = TRUE)))) {
  p <- call("ecdf_params", v)
  grid <- seq(0, 1, length.out = min(257, length(v)))
  q <- quantile(v, grid, type = 7, names = FALSE)
  # R's type-7 quantile differs from the core's by at most one ulp.
  stopifnot(max(abs(p$knots - unique(round(q, 12)))) < 1e-11)
  want_p <- vapply(p$knots, function(k) mean(grid[abs(q - k) < 1e-12]), 0)
  stopifnot(max(abs(p$probabilities - want_p)) < 1e-12)
  new <- c(min(v) - 1, max(v) + 1, seq(min(v), max(v), length.out = 50))
  got <- call("forward", "ecdf", c(p$knots, p$probabilities), new)
  want <- approx(p$knots, p$probabilities, new, rule = 2)$y
  stopifnot(max(abs(got - want)) < 1e-12)
}

r <- tryCatch(call("ecdf_params", rep(2, 5)), error = function(e) conditionMessage(e))
stopifnot(is.character(r), grepl("at least two distinct", r))

pos <- runif(50, 0.1, 10)
stopifnot(max(abs(call("forward", "log", numeric(), pos) - log(pos))) < 1e-12)
stopifnot(max(abs(call("inverse", "scale", c(1.5, 2), c(0.2, -1)) - (c(0.2, -1) * 2 + 1.5))) < 1e-12)
stopifnot(max(abs(call("jacobian", "sqrt", numeric(), pos) - 0.5 / sqrt(pos))) < 1e-12)

cat("R transform kernels passed\n")
