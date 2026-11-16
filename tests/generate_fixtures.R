#!/usr/bin/env Rscript
# Generate R reference fixtures for weighted GLMBoost, GAMBoost, and GAMBoostLSS.
#
# Run from repo root: Rscript tests/generate_fixtures.R
# Output: tests/fixtures/*.json
#
# Each fixture JSON contains everything needed to write a portable Python test
# that compares mboostrs output against R without requiring rpy2.

suppressMessages(library(mboost))
suppressMessages(library(jsonlite))

OUT <- file.path("tests", "fixtures")
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

set.seed(20250609)
n <- 120
x1 <- rnorm(n)
x2 <- rnorm(n)
w <- sample(1:4, n, replace=TRUE)
mstop <- 50
nu <- 0.1

# ---------------------------------------------------------------
# Helper: extract GLMBoost results
# ---------------------------------------------------------------
extract <- function(fit, y, x1, x2, w, mstop, nu, family_name) {
    cc_nooff <- coef(fit, off2int=FALSE)
    off <- attr(cc_nooff, "offset")
    if (is.null(off)) off <- 0.0
    list(
        family   = family_name,
        mstop    = mstop,
        nu       = nu,
        n        = length(y),
        seed     = 20250609,
        offset   = as.numeric(off),
        risk     = as.numeric(risk(fit)),
        fitted   = as.numeric(predict(fit, type="link")),
        selected = as.numeric(selected(fit)) - 1L,
        coef     = as.numeric(cc_nooff),
        x1       = as.numeric(x1),
        x2       = as.numeric(x2),
        y        = as.numeric(y),
        w        = as.numeric(w)
    )
}

# ---------------------------------------------------------------
# 1. GLMBoost — all families
# ---------------------------------------------------------------
cat("Generating GLMBoost fixtures ...\n")

y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Gaussian(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
write_json(extract(fit, y, x1, x2, w, mstop, nu, "Gaussian"),
    file.path(OUT, "glmboost_gaussian_weighted.json"), digits=15, auto_unbox=TRUE)

y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=GaussReg(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
write_json(extract(fit, y, x1, x2, w, mstop, nu, "GaussReg"),
    file.path(OUT, "glmboost_gaussreg_weighted.json"), digits=15, auto_unbox=TRUE)

y <- factor(rbinom(n, 1, plogis(0.5*x1 - 0.3*x2)), levels=0:1)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Binomial(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
r <- extract(fit, as.numeric(as.character(y)), x1, x2, w, mstop, nu, "Binomial")
r$y_levels <- levels(y)
write_json(r, file.path(OUT, "glmboost_binomial_weighted.json"), digits=15, auto_unbox=TRUE)

y <- rpois(n, exp(0.5 + 0.3*x1 - 0.2*x2))
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Poisson(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
write_json(extract(fit, y, x1, x2, w, mstop, nu, "Poisson"),
    file.path(OUT, "glmboost_poisson_weighted.json"), digits=15, auto_unbox=TRUE)

# Laplace needs odd n for unique L1 median
set.seed(20250609)
n_odd <- 121
x1o <- rnorm(n_odd); x2o <- rnorm(n_odd)
yo <- 1 + 0.7*x1o - 0.4*x2o + rnorm(n_odd, sd=0.5)
wo <- sample(1:4, n_odd, replace=TRUE)
fit <- glmboost(yo ~ x1o + x2o, data=data.frame(x1o=x1o, x2o=x2o, yo=yo),
    weights=wo, family=Laplace(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
r <- extract(fit, yo, x1o, x2o, wo, mstop, nu, "Laplace")
r$seed <- 20250609
write_json(r, file.path(OUT, "glmboost_laplace_weighted.json"), digits=15, auto_unbox=TRUE)

y <- rgamma(n, shape=2, scale=exp(0.3 + 0.2*x1 - 0.1*x2) / 2)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=GammaReg(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
write_json(extract(fit, y, x1, x2, w, mstop, nu, "GammaReg"),
    file.path(OUT, "glmboost_gammareg_weighted.json"), digits=15, auto_unbox=TRUE)

for (tau in c(0.5, 0.3, 0.8)) {
    set.seed(20250609)
    y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
    fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
        weights=w, family=ExpectReg(tau=tau),
        center=TRUE, control=boost_control(mstop=mstop, nu=nu))
    r <- extract(fit, y, x1, x2, w, mstop, nu, "ExpectReg")
    r$tau <- tau
    write_json(r, file.path(OUT, sprintf("glmboost_expectreg_tau%.1f_weighted.json", tau)),
        digits=15, auto_unbox=TRUE)
}

for (tau in c(0.5, 0.3, 0.8)) {
    set.seed(20250609)
    y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
    fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
        weights=w, family=QuantReg(tau=tau),
        center=TRUE, control=boost_control(mstop=mstop, nu=nu))
    r <- extract(fit, y, x1, x2, w, mstop, nu, "QuantReg")
    r$tau <- tau
    write_json(r, file.path(OUT, sprintf("glmboost_quantreg_tau%.1f_weighted.json", tau)),
        digits=15, auto_unbox=TRUE)
}

y <- factor(rbinom(n, 1, plogis(0.5*x1 - 0.3*x2)), levels=0:1)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=AdaExp(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
r <- extract(fit, as.numeric(as.character(y)), x1, x2, w, mstop, nu, "AdaExp")
write_json(r, file.path(OUT, "glmboost_adaexp_weighted.json"), digits=15, auto_unbox=TRUE)

y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Huber(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
write_json(extract(fit, y, x1, x2, w, mstop, nu, "Huber"),
    file.path(OUT, "glmboost_huber_weighted.json"), digits=15, auto_unbox=TRUE)

y <- rnbinom(n, mu=exp(0.5 + 0.3*x1 - 0.2*x2), size=2)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=NBinomial(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
write_json(extract(fit, y, x1, x2, w, mstop, nu, "NBinomial"),
    file.path(OUT, "glmboost_nbinomial_weighted.json"), digits=15, auto_unbox=TRUE)

y <- sample(1:3, n, replace=TRUE, prob=c(0.3, 0.4, 0.3))
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=PropOdds(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
r <- extract(fit, y, x1, x2, w, mstop, nu, "PropOdds")
write_json(r, file.path(OUT, "glmboost_propodds_weighted.json"), digits=15, auto_unbox=TRUE)

cat(sprintf("  %d glmboost fixtures\n", length(grep("glmboost_", list.files(OUT)))))

# ---------------------------------------------------------------
# 2. GAMBoost — all families with bbs(x1, df=4) + bbs(x2, df=4)
# ---------------------------------------------------------------
cat("Generating GAMBoost fixtures ...\n")

set.seed(20250609)
x1 <- rnorm(n); x2 <- rnorm(n)
w <- sample(1:4, n, replace=TRUE)
mstop <- 100

extract_gam <- function(fit, y, x1, x2, w, mstop, nu, family_name) {
    off <- fit$offset
    if (is.null(off)) off <- 0.0
    list(
        family   = family_name,
        mstop    = mstop,
        nu       = nu,
        n        = length(y),
        seed     = 20250609,
        offset   = as.numeric(off),
        risk     = as.numeric(risk(fit)),
        fitted   = as.numeric(predict(fit, type="link")),
        selected = as.numeric(selected(fit)) - 1L,
        x1       = as.numeric(x1),
        x2       = as.numeric(x2),
        y        = as.numeric(y),
        w        = as.numeric(w)
    )
}

y <- 1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Gaussian(),
    control=boost_control(mstop=mstop, nu=nu))
write_json(extract_gam(fit, y, x1, x2, w, mstop, nu, "Gaussian"),
    file.path(OUT, "gamboost_gaussian_weighted.json"), digits=15, auto_unbox=TRUE)

y <- factor(rbinom(n, 1, plogis(0.5*sin(x1) - 0.3*x2)), levels=0:1)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Binomial(),
    control=boost_control(mstop=mstop, nu=nu))
r <- extract_gam(fit, as.numeric(as.character(y)), x1, x2, w, mstop, nu, "Binomial")
write_json(r, file.path(OUT, "gamboost_binomial_weighted.json"), digits=15, auto_unbox=TRUE)

y <- rpois(n, exp(0.5 + 0.3*sin(x1) - 0.2*x2))
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Poisson(),
    control=boost_control(mstop=mstop, nu=nu))
write_json(extract_gam(fit, y, x1, x2, w, mstop, nu, "Poisson"),
    file.path(OUT, "gamboost_poisson_weighted.json"), digits=15, auto_unbox=TRUE)

set.seed(20250609)
n_odd <- 121
x1o <- rnorm(n_odd); x2o <- rnorm(n_odd)
yo <- 1 + 0.7*sin(x1o) - 0.4*x2o^2 + rnorm(n_odd, sd=0.5)
wo <- sample(1:4, n_odd, replace=TRUE)
fit <- gamboost(yo ~ bbs(x1o, df=4) + bbs(x2o, df=4),
    data=data.frame(x1o=x1o, x2o=x2o, yo=yo),
    weights=wo, family=Laplace(),
    control=boost_control(mstop=mstop, nu=nu))
r <- extract_gam(fit, yo, x1o, x2o, wo, mstop, nu, "Laplace")
r$seed <- 20250609
write_json(r, file.path(OUT, "gamboost_laplace_weighted.json"), digits=15, auto_unbox=TRUE)

y <- rgamma(n, shape=2, scale=exp(0.3 + 0.2*sin(x1) - 0.1*x2) / 2)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=GammaReg(),
    control=boost_control(mstop=mstop, nu=nu))
write_json(extract_gam(fit, y, x1, x2, w, mstop, nu, "GammaReg"),
    file.path(OUT, "gamboost_gammareg_weighted.json"), digits=15, auto_unbox=TRUE)

y <- 1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=ExpectReg(tau=0.5),
    control=boost_control(mstop=mstop, nu=nu))
r <- extract_gam(fit, y, x1, x2, w, mstop, nu, "ExpectReg")
r$tau <- 0.5
write_json(r, file.path(OUT, "gamboost_expectreg_tau0.5_weighted.json"), digits=15, auto_unbox=TRUE)

y <- 1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=QuantReg(tau=0.5),
    control=boost_control(mstop=mstop, nu=nu))
r <- extract_gam(fit, y, x1, x2, w, mstop, nu, "QuantReg")
r$tau <- 0.5
write_json(r, file.path(OUT, "gamboost_quantreg_tau0.5_weighted.json"), digits=15, auto_unbox=TRUE)

y <- factor(rbinom(n, 1, plogis(0.5*sin(x1) - 0.3*x2)), levels=0:1)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=AdaExp(),
    control=boost_control(mstop=mstop, nu=nu))
r <- extract_gam(fit, as.numeric(as.character(y)), x1, x2, w, mstop, nu, "AdaExp")
write_json(r, file.path(OUT, "gamboost_adaexp_weighted.json"), digits=15, auto_unbox=TRUE)

y <- 1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Huber(),
    control=boost_control(mstop=mstop, nu=nu))
write_json(extract_gam(fit, y, x1, x2, w, mstop, nu, "Huber"),
    file.path(OUT, "gamboost_huber_weighted.json"), digits=15, auto_unbox=TRUE)

y <- 1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=GaussReg(),
    control=boost_control(mstop=mstop, nu=nu))
write_json(extract_gam(fit, y, x1, x2, w, mstop, nu, "GaussReg"),
    file.path(OUT, "gamboost_gaussreg_weighted.json"), digits=15, auto_unbox=TRUE)

cat(sprintf("  %d gamboost fixtures\n", length(grep("gamboost_", list.files(OUT)))))

# ---------------------------------------------------------------
# 3. GAMBoostLSS — key families
# ---------------------------------------------------------------
cat("Generating GAMBoostLSS fixtures ...\n")

suppressMessages(library(gamboostLSS))

set.seed(20250609)
n <- 120
x1 <- rnorm(n); x2 <- rnorm(n)
y <- rnorm(n, mean=1 + 0.5*x1 - 0.3*x2, sd=exp(0.1 + 0.2*x1))
w <- sample(1:4, n, replace=TRUE)
mstop <- 50

# GaussianLSS
fit <- glmboostLSS(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, families=GaussianLSS(),
    control=boost_control(mstop=mstop, nu=nu))
risk_mat <- risk(fit)
write_json(list(
    family = "GaussianLSS", mstop=mstop, nu=nu, n=length(y),
    risk = as.numeric(risk_mat),
    fitted_mu = as.numeric(fitted(fit, parameter="mu")),
    fitted_sigma = as.numeric(fitted(fit, parameter="sigma")),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)
), file.path(OUT, "gamboostlss_gaussian_weighted.json"), digits=15, auto_unbox=TRUE)

set.seed(20250609)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- rexp(n, rate=exp(-0.5 - 0.3*x1 + 0.2*x2))
w <- sample(1:4, n, replace=TRUE)

fit <- glmboostLSS(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, families=GammaLSS(),
    control=boost_control(mstop=mstop, nu=nu))
risk_mat <- risk(fit)
write_json(list(
    family = "GammaLSS", mstop=mstop, nu=nu, n=length(y),
    risk = as.numeric(risk_mat),
    fitted_mu = as.numeric(fitted(fit, parameter="mu")),
    fitted_sigma = as.numeric(fitted(fit, parameter="sigma")),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)
), file.path(OUT, "gamboostlss_gamma_weighted.json"), digits=15, auto_unbox=TRUE)

cat(sprintf("  %d gamboostlss fixtures\n", length(grep("gamboostlss_", list.files(OUT)))))

cat(sprintf("\nDone. %d total fixtures.\n", length(list.files(OUT, pattern="\\.json$"))))
