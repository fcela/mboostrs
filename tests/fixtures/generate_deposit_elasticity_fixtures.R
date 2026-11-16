#!/usr/bin/env Rscript
# R reference fixtures for the deposit-elasticity case study.
#
# Run from the repo root:  Rscript tests/fixtures/generate_deposit_elasticity_fixtures.R
# Output: tests/fixtures/deposit_elasticity_*.json
#
# The designs here are the ones the case study fits, on the packaged data, so
# the parity test covers what the notebook actually claims: centered P-splines
# at df = 1, a shape-constrained spline, bmono, and a varying-coefficient
# interaction through `by =`.

suppressMessages(library(mboost))
suppressMessages(library(jsonlite))

OUT <- file.path("tests", "fixtures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

macro <- read.csv(file.path("python", "mboostrs", "datasets", "usmacro.csv"))
macro <- macro[macro$year <= 1993, ]
n <- nrow(macro)
real_m1 <- macro$m1 / (macro$cpi / 100)
y <- log(real_m1 / macro$realgdp)
r <- macro$tbilrate
r_mean <- mean(r)
dat <- data.frame(
    y = y,
    r_c = r - r_mean,
    t_c = seq_len(n) - (n + 1) / 2,
    infl_c = macro$infl - mean(macro$infl)
)

# theta(r) = -d log Q / d r, read off a grid where only r moves.
GRID <- c(3, 5, 8, 11, 14)
CURVE <- seq(2.5, 15, length.out = 26)
theta_of <- function(mod, pts, h = 0.05) {
    frame_at <- function(v) data.frame(r_c = v - r_mean, t_c = 0, infl_c = 0)
    up <- as.numeric(predict(mod, newdata = frame_at(pts + h)))
    dn <- as.numeric(predict(mod, newdata = frame_at(pts - h)))
    -(up - dn) / (2 * h)
}

record <- function(mod, label, mstop) {
    list(
        label = label,
        mstop = mstop,
        nu = 0.05,
        n = n,
        risk = as.numeric(risk(mod)),
        fitted = as.numeric(predict(mod, type = "link")),
        selected = as.numeric(selected(mod)) - 1L,
        theta_grid = GRID,
        theta = theta_of(mod, GRID),
        curve_grid = CURVE,
        curve_theta = theta_of(mod, CURVE),
        resid_sd = sd(dat$y - fitted(mod)),
        offset = as.numeric(mod$offset)
    )
}

MSTOP <- 172
free <- gamboost(
    y ~ bols(r_c, intercept = FALSE) + bbs(r_c, center = TRUE, df = 1) +
        bols(t_c, intercept = FALSE) + bols(infl_c, intercept = FALSE) +
        bbs(infl_c, center = TRUE, df = 1),
    data = dat, control = boost_control(mstop = MSTOP, nu = 0.05))
write_json(record(free, "centered_df1", MSTOP),
           file.path(OUT, "deposit_elasticity_centered_df1.json"),
           auto_unbox = TRUE, digits = 15)

constrained <- gamboost(
    y ~ bols(r_c, intercept = FALSE) +
        bbs(r_c, center = TRUE, df = 1, constraint = "decreasing") +
        bols(t_c, intercept = FALSE) + bols(infl_c, intercept = FALSE) +
        bbs(infl_c, center = TRUE, df = 1),
    data = dat, control = boost_control(mstop = MSTOP, nu = 0.05))
write_json(record(constrained, "constrained_decreasing", MSTOP),
           file.path(OUT, "deposit_elasticity_constrained.json"),
           auto_unbox = TRUE, digits = 15)

mono <- gamboost(
    y ~ bols(r_c) + bmono(r_c, constraint = "decreasing", df = 2) +
        bols(t_c) + bols(infl_c) + bbs(infl_c, center = TRUE, df = 2),
    data = dat, control = boost_control(mstop = MSTOP, nu = 0.05))
write_json(record(mono, "bmono_df2", MSTOP),
           file.path(OUT, "deposit_elasticity_bmono.json"),
           auto_unbox = TRUE, digits = 15)

# Differenced varying-coefficient model: the impact response of deposit growth
# to a rate change, allowed to vary with the level of the rate.
dy <- diff(dat$y)
dr <- diff(r)
dinfl <- diff(dat$infl_c)
r_lag <- head(r, -1)
vc <- data.frame(
    dy = dy[-1], dr = dr[-1], dinfl = dinfl[-1],
    dy_lag = head(dy, -1), r_lag_c = r_lag[-1] - mean(r_lag[-1])
)
vc_fit <- gamboost(
    dy ~ bols(dr, intercept = FALSE) + bols(r_lag_c, by = dr, intercept = FALSE) +
        bbs(r_lag_c, by = dr, center = TRUE, df = 1) +
        bols(dy_lag, intercept = FALSE) + bols(dinfl, intercept = FALSE),
    data = vc, control = boost_control(mstop = 500, nu = 0.05))
beta_at <- function(mod, pts) {
    base <- data.frame(dr = 0, dinfl = 0, dy_lag = 0,
                       r_lag_c = pts - mean(r_lag[-1]))
    up <- base; up$dr <- 1
    -(as.numeric(predict(mod, newdata = up)) - as.numeric(predict(mod, newdata = base)))
}
write_json(list(
    label = "varying_coefficient", mstop = 500, nu = 0.05, n = nrow(vc),
    risk = as.numeric(risk(vc_fit)),
    fitted = as.numeric(predict(vc_fit, type = "link")),
    selected = as.numeric(selected(vc_fit)) - 1L,
    beta_grid = GRID, beta = beta_at(vc_fit, GRID),
    offset = as.numeric(vc_fit$offset)
), file.path(OUT, "deposit_elasticity_vc.json"), auto_unbox = TRUE, digits = 15)

# Modern monthly spread design, Huber family, as the case study fits it.
dm <- read.csv(file.path("python", "mboostrs", "datasets", "deposit_market.csv"))
dm <- dm[!is.na(dm$fedfunds) & !is.na(dm$deposits) & !is.na(dm$personal_income) &
         !is.na(dm$natl_mmda) & !is.na(dm$cpi), ]
m <- nrow(dm)
spread <- dm$fedfunds - dm$natl_mmda
modern <- data.frame(
    y = log(dm$deposits / dm$personal_income),
    s_c = spread - mean(spread),
    bs_c = log(dm$walcl / (dm$personal_income * 1000)) -
           mean(log(dm$walcl / (dm$personal_income * 1000))),
    t_c = seq_len(m) - (m + 1) / 2,
    covid = as.numeric(dm$date >= "2020-03-01" & dm$date <= "2020-07-01"),
    svb = as.numeric(dm$date >= "2023-03-01" & dm$date <= "2023-05-01")
)
modern_fit <- gamboost(
    y ~ bols(s_c, intercept = FALSE) + bbs(s_c, center = TRUE, df = 1) +
        bols(bs_c, intercept = FALSE) + bbs(bs_c, center = TRUE, df = 1) +
        bols(t_c, intercept = FALSE) + bols(covid, intercept = FALSE) +
        bols(svb, intercept = FALSE),
    data = modern, control = boost_control(mstop = 518, nu = 0.05))
s_grid <- c(0, 1, 2, 3, 4.5)
theta_modern <- function(mod, pts, h = 0.05) {
    frame_at <- function(v) data.frame(s_c = v - mean(spread), bs_c = 0, t_c = 0,
                                       covid = 0, svb = 0)
    -(as.numeric(predict(mod, newdata = frame_at(pts + h))) -
      as.numeric(predict(mod, newdata = frame_at(pts - h)))) / (2 * h)
}
write_json(list(
    label = "modern_spread", mstop = 518, nu = 0.05, n = m,
    risk = as.numeric(risk(modern_fit)),
    fitted = as.numeric(predict(modern_fit, type = "link")),
    selected = as.numeric(selected(modern_fit)) - 1L,
    theta_grid = s_grid, theta = theta_modern(modern_fit, s_grid),
    offset = as.numeric(modern_fit$offset)
), file.path(OUT, "deposit_elasticity_modern.json"), auto_unbox = TRUE, digits = 15)


# A single constrained learner on a smooth decreasing curve, which isolates the
# shape-constraint machinery from the selection path.
set.seed(7)
iso_x <- seq(0, 10, length.out = 60)
iso_y <- -0.3 * iso_x + sin(iso_x / 3) + rnorm(60, sd = 0.1)
iso <- data.frame(x = iso_x, y = iso_y)
iso_fits <- list()
for (cfg in list(list(center = TRUE, df = 1), list(center = FALSE, df = 4),
                 list(center = TRUE, df = 4))) {
    fit <- suppressWarnings(gamboost(
        y ~ bbs(x, center = cfg$center, df = cfg$df, constraint = "decreasing"),
        data = iso, control = boost_control(mstop = 50, nu = 0.5)))
    key <- paste0("center", cfg$center, "_df", cfg$df)
    iso_fits[[key]] <- as.numeric(predict(fit, type = "link"))
}
write_json(list(label = "constrained_single_learner", mstop = 50, nu = 0.5,
                x = iso_x, y = iso_y, fitted = iso_fits),
           file.path(OUT, "deposit_elasticity_constraint_isolated.json"),
           auto_unbox = TRUE, digits = 15)

cat("wrote deposit-elasticity fixtures with mboost",
    as.character(packageVersion("mboost")), "on", R.version.string, "\n")
