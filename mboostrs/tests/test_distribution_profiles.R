#@validates contract.r.density_overlay
#@validates contract.r.centile_fan
#@validates contract.r.deviance_profile
#@validates smoke.r.lss_distribution_profile_plots

library(mboostrs)

set.seed(90)
n <- 100L
x <- seq(-1.5, 1.5, length.out = n)
group <- ifelse(x < 0, "lower", "upper")
y <- 1 + 0.7 * x + rnorm(n, sd = exp(-0.4 + 0.25 * x))
dat <- data.frame(y = y, x = x, group = group)
fit <- glmboostLSS(
  list(mu = y ~ x, sigma = y ~ x), data = dat, families = GaussianLSS(),
  control = boost_control(mstop = 12, nu = 0.08)
)

grid <- seq(-5, 6, length.out = 301L)
overlay <- density_overlay_data(fit, grid = grid)
stopifnot(identical(unique(overlay$kind), "density"))
fitted_curve <- overlay[overlay$series == "fitted", ]
observed_curve <- overlay[overlay$series == "observed", ]
params <- lss_params(fit)
for (index in c(1L, 76L, 151L, 226L, 301L)) {
  expected <- mean(lss_pdf(GaussianLSS(), rep(grid[index], n), params))
  stopifnot(abs(fitted_curve$value[index] - expected) < 1e-13)
}
trapz <- function(x, value) sum(diff(x) * (head(value, -1L) + tail(value, -1L)) / 2)
stopifnot(abs(trapz(grid, fitted_curve$value) - 1) < 2e-6)
stopifnot(abs(trapz(grid, observed_curve$value) - 1) < 2e-6)

fan <- centile_fan_data(fit, "x", probs = c(0.1, 0.5, 0.9), split_by = "group")
stopifnot(nrow(fan) == n, all(c("q10", "q50", "q90") %in% names(fan)))
stopifnot(all(fan$q10 < fan$q50), all(fan$q50 < fan$q90))
stopifnot(identical(unique(fan$split), c("lower", "upper")))

profile <- deviance_profile(fit, xvar = "x", bins = 7L)
stopifnot(sum(profile$n) == n)
stopifnot(abs(sum(profile$total_deviance) - sum(deviance_increment(fit))) < 1e-10)

# Bin sizes follow the rule the Python frontend uses: the first `n %% bins`
# intervals take one row more than the rest. `cut(seq_len(n), breaks = bins)`,
# used before, spreads the remainder over different intervals, so the same
# model reported different rows in R and in Python whenever `n %% bins != 0`.
for (bins in c(7L, 9L, 10L, 13L)) {
  sizes <- deviance_profile(fit, xvar = "x", bins = bins)$n
  base <- n %/% bins
  remainder <- n %% bins
  expected <- c(rep(base + 1L, remainder), rep(base, bins - remainder))
  stopifnot(identical(as.integer(sizes), as.integer(expected)))
  stopifnot(sum(sizes) == n)
}

set.seed(92)
x_mix <- rnorm(180)
is_zero <- runif(180) < 0.3
y_mix <- ifelse(is_zero, 0, rgamma(180, shape = 4, scale = 0.5))
mix <- glmboostLSS(
  list(mu = y ~ x, sigma = y ~ x, nu = y ~ x),
  data = data.frame(y = y_mix, x = x_mix), families = ZeroAdjustedGammaLSS(),
  control = boost_control(mstop = 10, nu = 0.08)
)
mixed <- density_overlay_data(mix, grid = seq(0.001, 12, length.out = 350L))
atom <- mixed[mixed$kind == "atom", ]
stopifnot(abs(atom$value[atom$series == "observed"] - mean(is_zero)) < 1e-14)
stopifnot(abs(atom$value[atom$series == "fitted"] - mean(lss_params(mix)$nu)) < 1e-14)

plot_file <- tempfile(fileext = ".pdf")
grDevices::pdf(plot_file)
plot_distribution_density(fit, grid = grid)
plot_centile_fan(fit, "x", probs = c(0.1, 0.5, 0.9), split_by = "group")
plot_deviance_profile(fit, xvar = "x", bins = 7L)
grDevices::dev.off()
stopifnot(file.exists(plot_file), file.info(plot_file)$size > 0)
unlink(plot_file)

cat("R LSS distribution profiles passed\n")
