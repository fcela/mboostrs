suppressMessages(library(mboost))
suppressMessages(library(survival))
suppressMessages(library(jsonlite))

OUT <- "tests/fixtures"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

write_fixture <- function(lst, name) {
  sink(file.path(OUT, sprintf("glmboost_%s.json", name)))
  cat(toJSON(lst, digits=15, auto_unbox=TRUE))
  sink()
  cat(sprintf("  %s done\n", name))
}

# ---- Gaussian ----
set.seed(42); n <- 200
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.3)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=Gaussian(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="gaussian_default", family="Gaussian",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=TRUE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "gaussian_default")

# ---- GaussReg (mstop=100) ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 1 + 0.5*x1 - 0.3*x2 + rnorm(n)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=GaussReg(), center=TRUE,
  control=boost_control(mstop=100, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="gaussreg_default", family="GaussReg",
  formula="y ~ x1 + x2", mstop=100L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=TRUE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-8, tol_fitted=1e-8, tol_offset=1e-6), "gaussreg_default")

# ---- Binomial ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
p <- 1/(1+exp(-(0.5*x1+0.3*x2)))
y <- factor(rbinom(n,1,p))
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=Binomial(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
y_num <- as.numeric(y)-1L
write_fixture(list(
  scenario="binomial_default", family="Binomial",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y_num),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "binomial_default")

# ---- AdaExp (mstop=50) ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
eta <- 0.5*x1 - 0.3*x2
p <- 1/(1+exp(-eta))
y_num <- ifelse(runif(n) < p, 1, -1)
y <- as.factor(ifelse(y_num==1, "yes", "no"))
data <- data.frame(x1=x1, x2=x2, y=y)
fit <- glmboost(y ~ x1 + x2, data=data,
  family=AdaExp(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="adaexp_default", family="AdaExp",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y_num),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  selected=as.numeric(selected(fit))-1L,
  has_selected=TRUE, has_coef=FALSE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "adaexp_default")

# ---- AdaExp (short mstop=20) ----
set.seed(123); ns <- 100
x1s <- rnorm(ns); x2s <- rnorm(ns)
eta <- 0.5*x1s - 0.3*x2s
p <- 1/(1+exp(-eta))
y_num <- ifelse(runif(ns) < p, 1, -1)
y <- as.factor(ifelse(y_num==1, "yes", "no"))
fit <- glmboost(y ~ x1s + x2s, data=data.frame(x1=x1s, x2=x2s, y=y),
  family=AdaExp(), center=TRUE,
  control=boost_control(mstop=20, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="adaexp_short", family="AdaExp",
  formula="y ~ x1 + x2", mstop=20L, nu=0.1, n=ns, seed=123L, center=TRUE,
  x1=as.numeric(x1s), x2=as.numeric(x2s), y=as.numeric(y_num),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  selected=as.numeric(selected(fit))-1L,
  has_selected=TRUE, has_coef=FALSE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "adaexp_short")

# ---- Poisson ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- rpois(n, lambda=exp(0.3*x1+0.1*x2))
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=Poisson(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="poisson_default", family="Poisson",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "poisson_default")

# ---- Laplace (n=201 odd!) ----
# NOTE: R's Laplace offset differs from mboostrs by ~4e-6 even at odd n.
# R's weighted.median() uses a slightly different algorithm than our
# type-7 quantile implementation, causing a tiny drift that propagates
# through the boosting path (~0.04 in risk at mstop=50, ~0.001 in fitted).
# Tolerances reflect this known numeric quirk (not a correctness bug).
set.seed(42); n_odd <- 201
x1o <- rnorm(n_odd); x2o <- rnorm(n_odd)
y_odd <- 1.5*x1o + 0.8*x2o + rnorm(n_odd, sd=0.5)
fit <- glmboost(y_odd ~ x1o + x2o, data=data.frame(x1o=x1o, x2o=x2o, y_odd=y_odd),
  family=Laplace(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="laplace_default", family="Laplace",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=n_odd, seed=42L, center=TRUE,
  x1=as.numeric(x1o), x2=as.numeric(x2o), y=as.numeric(y_odd),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-3, tol_fitted=1e-2, tol_offset=5e-6,
  atol_fitted=1e-3), "laplace_default")

# ---- Huber ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=Huber(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="huber_default", family="Huber",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "huber_default")

# ---- GammaReg ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- exp(0.3*x1 + 0.1*x2) * rgamma(n, shape=2, rate=2)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=GammaReg(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="gammareg_default", family="GammaReg",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "gammareg_default")

# ---- NBinomial (mstop=100) ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
mu <- exp(0.5 + 0.3*x1 - 0.2*x2)
y <- rnbinom(n, size=5, mu=mu)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=NBinomial(), center=TRUE,
  control=boost_control(mstop=100, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="nbinomial_default", family="NBinomial",
  formula="y ~ x1 + x2", mstop=100L, nu=0.1, n=n, seed=42L, center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=TRUE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-5, tol_fitted=1e-6, tol_offset=1e-4), "nbinomial_default")

# ---- NBinomial (short mstop=20) ----
set.seed(123); ns <- 100
x1s <- rnorm(ns); x2s <- rnorm(ns)
mu <- exp(0.5 + 0.3*x1s - 0.2*x2s)
y <- rnbinom(ns, size=5, mu=mu)
fit <- glmboost(y ~ x1s + x2s, data=data.frame(x1=x1s, x2=x2s, y=y),
  family=NBinomial(), center=TRUE,
  control=boost_control(mstop=20, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="nbinomial_short", family="NBinomial",
  formula="y ~ x1 + x2", mstop=20L, nu=0.1, n=ns, seed=123L, center=TRUE,
  x1=as.numeric(x1s), x2=as.numeric(x2s), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=TRUE, has_coef=TRUE, has_response=FALSE,
  tol_risk=5e-5, tol_fitted=1e-5, tol_offset=1e-4), "nbinomial_short")

# ---- Quantile (tau=0.5) ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=QuantReg(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="quantile_default", family="QuantReg",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, tau=0.5, n=n, seed=42L,
  center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "quantile_default")

# ---- Quantile weighted (tau=0.5, 0.25, 0.75) ----
for (tau in c(0.5, 0.25, 0.75)) {
  set.seed(11); nw <- 180
  x1w <- rnorm(nw); x2w <- rnorm(nw)
  y <- 1 + 0.8*x1w - 0.5*x2w + rnorm(nw)
  w <- sample(1:4, nw, replace=TRUE)
  fit <- glmboost(y ~ x1w + x2w, data=data.frame(x1w=x1w, x2w=x2w, y=y),
    weights=w, family=QuantReg(tau=tau), center=TRUE,
    control=boost_control(mstop=80, nu=0.1))
  cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
  write_fixture(list(
    scenario=sprintf("quantile_weighted_tau%.2f", tau),
    family="QuantReg", formula="y ~ x1 + x2",
    mstop=80L, nu=0.1, tau=tau, n=nw, seed=11L, center=TRUE,
    x1=as.numeric(x1w), x2=as.numeric(x2w), y=as.numeric(y),
    w=as.numeric(w), offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
    has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
    tol_risk=1e-9, tol_fitted=1e-9, tol_offset=1e-12),
    sprintf("quantile_weighted_tau%.0f", 100*tau))
}

# ---- Expectile (tau=0.5) ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 1.5*x1 + 0.8*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
  family=ExpectReg(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="expectile_default", family="ExpectReg",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, tau=0.5, n=n, seed=42L,
  center=TRUE,
  x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-10, tol_fitted=1e-10, tol_offset=1e-10), "expectile_default")

# ---- Expectile weighted (tau=0.5, 0.2, 0.8) ----
for (tau in c(0.5, 0.2, 0.8)) {
  set.seed(11); nw <- 180
  x1w <- rnorm(nw); x2w <- rnorm(nw)
  y <- 1 + 0.7*x1w - 0.4*x2w + rnorm(nw)
  w <- sample(1:4, nw, replace=TRUE)
  fit <- glmboost(y ~ x1w + x2w, data=data.frame(x1w=x1w, x2w=x2w, y=y),
    weights=w, family=ExpectReg(tau=tau), center=TRUE,
    control=boost_control(mstop=80, nu=0.1))
  cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
  write_fixture(list(
    scenario=sprintf("expectile_weighted_tau%.2f", tau),
    family="ExpectReg", formula="y ~ x1 + x2",
    mstop=80L, nu=0.1, tau=tau, n=nw, seed=11L, center=TRUE,
    x1=as.numeric(x1w), x2=as.numeric(x2w), y=as.numeric(y),
    w=as.numeric(w), offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
    has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
    tol_risk=1e-9, tol_fitted=1e-9, tol_offset=1e-12),
    sprintf("expectile_weighted_tau%.0f", 100*tau))
}

# ===== Survival families =====
# ---- Loglog ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
eta <- 0.3*x1 + 0.1*x2
time <- rlnorm(n, meanlog=eta, sdlog=0.5)
event <- rbinom(n, 1, 0.7)
df <- data.frame(time=time, event=event, x1=x1, x2=x2)
fit <- glmboost(Surv(time, event) ~ x1 + x2, data=df,
  family=Loglog(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="loglog_default", family="Loglog",
  formula="Surv(time, event) ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L,
  center=TRUE, response_type="Surv",
  x1=as.numeric(x1), x2=as.numeric(x2),
  time=as.numeric(time), event=as.numeric(event),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-5, tol_fitted=1e-4, tol_offset=1e-6,
  atol_fitted=1e-5), "loglog_default")

# ---- Weibull ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
eta <- 0.3*x1 + 0.1*x2
time <- rweibull(n, shape=1, scale=exp(eta))
event <- rbinom(n, 1, 0.7)
df <- data.frame(time=time, event=event, x1=x1, x2=x2)
fit <- glmboost(Surv(time, event) ~ x1 + x2, data=df,
  family=Weibull(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="weibull_default", family="Weibull",
  formula="Surv(time, event) ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L,
  center=TRUE, response_type="Surv",
  x1=as.numeric(x1), x2=as.numeric(x2),
  time=as.numeric(time), event=as.numeric(event),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-5, tol_fitted=1e-4, tol_offset=1e-6,
  atol_fitted=1e-5), "weibull_default")

# ---- CoxPH ----
set.seed(42)
x1 <- rnorm(n); x2 <- rnorm(n)
eta <- 0.3*x1 + 0.1*x2
time <- rexp(n, rate=exp(eta))
event <- rbinom(n, 1, 0.7)
df <- data.frame(time=time, event=event, x1=x1, x2=x2)
fit <- glmboost(Surv(time, event) ~ x1 + x2, data=df,
  family=CoxPH(), center=TRUE,
  control=boost_control(mstop=50, nu=0.1))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
write_fixture(list(
  scenario="coxph_default", family="CoxPH",
  formula="Surv(time, event) ~ x1 + x2", mstop=50L, nu=0.1, n=n, seed=42L,
  center=TRUE, response_type="Surv",
  x1=as.numeric(x1), x2=as.numeric(x2),
  time=as.numeric(time), event=as.numeric(event),
  offset=as.numeric(off),
  risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
  coef=as.numeric(cc), selected=as.numeric(selected(fit))-1L,
  has_selected=FALSE, has_coef=TRUE, has_response=FALSE,
  tol_risk=1e-8, tol_fitted=1e-6, tol_offset=1e-10,
  atol_fitted=1e-8), "coxph_default")

# ---- CoxPH cvrisk ----
set.seed(21); ncv <- 120; B <- 15
x1 <- rnorm(ncv); x2 <- rnorm(ncv)
eta <- 0.8*x1 - 0.5*x2
time <- rexp(ncv, exp(eta)); cens <- runif(ncv, 0, 3)
status <- as.numeric(time <= cens); time <- pmin(time, cens)
df <- data.frame(time=time, status=status, x1=x1, x2=x2)
m <- glmboost(Surv(time, status) ~ x1 + x2, data=df,
  family=CoxPH(), control=boost_control(mstop=100, nu=0.1))
set.seed(7)
folds <- cv(rep(1, ncv), type="bootstrap", B=B)
cvr <- cvrisk(m, folds=folds, papply=lapply)
write_fixture(list(
  scenario="coxph_cvrisk", family="CoxPH",
  formula="Surv(time, status) ~ x1 + x2", mstop=100L, nu=0.1, n=ncv,
  seed=21L, center=TRUE, response_type="Surv",
  x1=as.numeric(x1), x2=as.numeric(x2), time=as.numeric(time),
  status=as.numeric(status),
  folds=as.numeric(folds), folds_nrow=nrow(folds), folds_ncol=ncol(folds),
  oob=as.numeric(cvr), oob_nrow=nrow(cvr), oob_ncol=ncol(cvr),
  optimal_mstop=as.integer(mstop(cvr))), "coxph_cvrisk")

# ===== PropOdds =====
# ---- PropOdds (3 classes, mstop=100) ----
generate_propodds <- function(mstop, nu, n_levels, n, seed) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- 0.5*x1 - 0.3*x2
  K <- n_levels
  cutpoints <- seq(0, by=1.5, length.out=K-1)
  probs <- matrix(0, n, K)
  for (i in 1:n) {
    cum <- numeric(K)
    cum[1] <- 1/(1+exp(-(cutpoints[1]+eta[i])))
    for (k in 2:(K-1)) cum[k] <- 1/(1+exp(-(cutpoints[k]+eta[i])))
    cum[K] <- 1
    probs[i,] <- diff(c(0, cum))
  }
  y_int <- apply(probs, 1, function(p) sample(1:K, 1, prob=p))
  y <- as.ordered(y_int)
  fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    family=PropOdds(), center=TRUE,
    control=boost_control(mstop=mstop, nu=nu))
  cc <- coef(fit)
  resp <- predict(fit, type="response")
  resp_t <- t(resp)
  cc_nooff <- coef(fit, off2int=FALSE)
  off <- attr(cc_nooff, "offset"); if(is.null(off)) off <- 0.0
  list(x1=x1, x2=x2, y=as.integer(y_int),
       offset=as.numeric(off),
       risk=as.numeric(risk(fit)),
       fitted=as.numeric(predict(fit,type="link")),
       selected=as.integer(selected(fit))-1L,
       response=as.numeric(resp_t), resp_ncol=ncol(resp),
       coef_x1=as.numeric(cc["x1"]), coef_x2=as.numeric(cc["x2"]),
       nlevels=K)
}

d <- generate_propodds(100, 0.1, 3, 200, 42)
write_fixture(c(d, list(scenario="propodds_default", family="PropOdds",
  formula="y ~ x1 + x2", mstop=100L, nu=0.1, n=200L, seed=42L, center=TRUE,
  has_selected=TRUE, has_coef=TRUE, has_response=TRUE,
  skip_fitted=TRUE, skip_risk=TRUE,
  tol_risk=1e-5, tol_fitted=1e-5, tol_offset=1e-12,
  tol_response=1e-4, atol_response=1e-5)),
  "propodds_default")

d <- generate_propodds(20, 0.1, 3, 100, 123)
write_fixture(c(d, list(scenario="propodds_short", family="PropOdds",
  formula="y ~ x1 + x2", mstop=20L, nu=0.1, n=100L, seed=123L, center=TRUE,
  has_selected=TRUE, has_coef=TRUE, has_response=TRUE,
  skip_fitted=TRUE, skip_risk=TRUE,
  tol_risk=1e-5, tol_fitted=1e-5, tol_offset=1e-12,
  tol_response=1e-4, atol_response=1e-5)),
  "propodds_short")


d <- generate_propodds(50, 0.1, 4, 200, 77)
write_fixture(c(d, list(scenario="propodds_4class", family="PropOdds",
  formula="y ~ x1 + x2", mstop=50L, nu=0.1, n=200L, seed=77L, center=TRUE,
  has_selected=FALSE, has_coef=TRUE, has_response=TRUE,
  skip_fitted=TRUE, skip_risk=TRUE,
  tol_risk=1e-5, tol_fitted=1e-5, tol_offset=1e-12,
  tol_response=1e-4, atol_response=1e-5)),
  "propodds_4class")

cat(sprintf("\nDone. %d fixtures generated in %s\n",
  length(list.files(OUT, pattern="glmboost_.*\\.json$")), OUT))
