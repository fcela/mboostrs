library(mboostrs)

# Both frontends call these Rust kernels. Compare against base R statistics;
# no high-level R split-refit or discovery API is introduced here.
call <- function(name, ...) .Call(paste0("wrap__r_", name), ..., PACKAGE = "mboostrs")
near <- function(a, b, tolerance = 1e-12) {
  stopifnot(isTRUE(all.equal(a, b, tolerance = tolerance, check.attributes = FALSE)))
}
refuses <- function(expr) stopifnot(inherits(tryCatch(expr, error = identity), "error"))

p <- c(0.01, 0.02, 0.03, 0.04)
for (gamma in c(0.1, 0.25, 0.5, 0.9)) {
  near(call("aggregate_split_p", p, gamma), min(1, quantile(p / gamma, gamma)))
}
refuses(call("aggregate_split_p", numeric(0), 0.5))
refuses(call("aggregate_split_p", c(0.1, NaN), 0.5))

z <- qnorm(0.975)
near(call("normal_interval_p", 2, 2 - z, 2 + z, 0.95), 2 * pnorm(-2))
near(call("normal_interval_p", 10, 10 - z, 10 + z, 0.95) / (2 * pnorm(-10)), 1)
stopifnot(call("normal_interval_p", 0, 0, 0, 0.95) == 1)
near(call("draw_p", c(-2, -1, 1, 2, 3)), 0.8)

a <- matrix(c(-2, 0, 0, 2, 0, 0, 2, 4), nrow = 4)
b <- matrix(0, nrow = 4, ncol = 2)
contrast <- call("draw_contrast", c(2, 4), c(0, 0), a, b, 0.5)
near(unlist(contrast), c(3, -0.25, 1.5, 1))
refuses(call("draw_contrast", c(2, 4), 0, a, b, 0.5))

before <- c(2, 4, 6, 0)
after <- c(1, 2, 3, 0)
gain <- call("paired_mean_se", before, after)
near(gain$gain, mean(before - after))
near(gain$se, sd(before - after) / sqrt(length(before)))
stopifnot(is.nan(call("paired_mean_se", 1, 0)$se))

x <- cbind(seq(-2, 2, length.out = 40), rep(c(0, 1), 20))
residual <- exp(0.2 * x[, 1] + 0.4 * x[, 2]) * (1 + seq_len(40) %% 7)
residual[c(2, 3)] <- c(0, NA)
x[4, 2] <- NA
centered <- sweep(x, 2, colMeans(x, na.rm = TRUE))
y <- log(residual^2)
keep <- is.finite(y) & apply(is.finite(x), 1, all)
design <- centered[keep, , drop = FALSE]
y <- y[keep] - mean(y[keep])
inverse <- solve(crossprod(design))
slopes <- drop(inverse %*% crossprod(design, y))
screen <- call("log_variance_screen", residual, x)
near(screen$slopes, slopes)
near(screen$t_stats, slopes / sqrt(pi^2 / 2 * diag(inverse)))
near(screen$r_squared, 1 - sum((y - design %*% slopes)^2) / sum(y^2))
stopifnot(screen$n_used == sum(keep))
stopifnot(is.null(call("log_variance_screen", rep(1, 40), matrix(1, 40, 2))))
