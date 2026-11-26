# rearrange_quantiles(): row-wise sort of separately fitted Quantile or
# Expectile levels, with the crossing report (issue 103).
library(mboostrs)
# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)
fails <- function(expr, pattern) {
  msg <- tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
  ok(!is.na(msg) && grepl(pattern, msg), paste("expected error matching", pattern, "got", msg))
}

# MBOOSTRS_REARRANGE_IN names a CSV with columns x and y; the Python test sets
# it so both languages fit the same rows.
in_file <- Sys.getenv("MBOOSTRS_REARRANGE_IN")
if (nzchar(in_file)) {
  d <- read.csv(in_file)
} else {
  set.seed(3)
  x <- runif(300, -1, 1)
  d <- data.frame(x = x, y = 2 * x + (0.05 + abs(x)) * rnorm(300))
}
n <- nrow(d)
taus <- c(0.1, 0.5, 0.9)
fit_all <- function(fam) lapply(taus, function(t)
  # R's start at the median for every level, so these short fits cross.
  glmboost(y ~ x, data = d,
           family = if (identical(fam, Quantile)) fam(tau = t, qoffset = 0.5) else fam(tau = t),
           control = boost_control(mstop = 3, nu = 0.1)))

q <- fit_all(Quantile)
r <- rearrange_quantiles(q, d)
ok(r$kind == "quantile", "kind")
ok(identical(names(r$predictions), c("q10", "q50", "q90")), "column names")
m <- as.matrix(r$predictions)
ok(all(m[, -1] >= m[, -3]), "sorted rows are monotone")
u <- as.matrix(r$unsorted)
cross <- u[, -3] > u[, -1]
ok(r$crossing_rows == sum(rowSums(cross) > 0), "crossing_rows")
ok(r$crossing_rows > 0, "fixture has crossings")
ok(abs(r$crossing_share - r$crossing_rows / n) < 1e-15, "crossing_share")
ok(identical(as.integer(r$pair_crossings$n_crossing), as.integer(colSums(cross))), "pair counts")
ok(abs(r$max_crossing - max(0, max(u[, -3] - u[, -1]))) < 1e-12, "max_crossing")
ok(max(abs(m - t(apply(u, 1, sort)))) == 0, "sorted equals row sort")

e <- fit_all(Expectile)
re <- rearrange_quantiles(rev(e), d, levels = c(0.9, 0.5, 0.1))
ok(re$kind == "expectile" && identical(names(re$predictions), c("e10", "e50", "e90")), "expectile")

fails(rearrange_quantiles(list(q[[1]], e[[2]]), d), "mix Quantile and Expectile")
fails(rearrange_quantiles(list(q[[1]], q[[1]]), d), "share a tau")
fails(rearrange_quantiles(list(q[[1]]), d), "at least two")
fails(rearrange_quantiles(q, d, levels = c(0.2, 0.5, 0.9)), "do not match")
g <- glmboost(y ~ x, data = d, family = Gaussian(), control = boost_control(mstop = 3))
fails(rearrange_quantiles(list(q[[1]], g), d), "needs Quantile or Expectile")

# Cross-language parity: the data and the sorted predictions are written to
# MBOOSTRS_REARRANGE_OUT so the Python test can compare them at 1e-12.
out <- Sys.getenv("MBOOSTRS_REARRANGE_OUT")
if (nzchar(out)) write.csv(cbind(d, r$predictions), out, row.names = FALSE)
cat("rearrange_quantiles R checks passed\n")
