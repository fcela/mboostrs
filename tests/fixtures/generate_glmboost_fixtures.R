suppressMessages(library(mboost))
suppressMessages(library(jsonlite))

OUT <- "tests/fixtures"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)
mstop <- 50; nu <- 0.1

# --- Gaussian ---
set.seed(20250609)
n <- 120; x1 <- rnorm(n); x2 <- rnorm(n)
w <- sample(1:4, n, replace=TRUE)
y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Gaussian(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE)
off <- attr(cc, "offset"); if (is.null(off)) off <- 0.0
sink(file.path(OUT, "glmboost_gaussian_weighted.json"))
cat(toJSON(list(
    family="Gaussian", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)),
    fitted=as.numeric(predict(fit, type="link")),
    selected=as.numeric(selected(fit))-1L,
    coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)
), digits=15, auto_unbox=TRUE))
sink()
cat("gaussian done\n")

# --- GaussReg ---
y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=GaussReg(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_gaussreg_weighted.json"))
cat(toJSON(list(family="GaussReg", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("gaussreg done\n")

# --- Poisson ---
y <- rpois(n, exp(0.5 + 0.3*x1 - 0.2*x2))
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Poisson(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_poisson_weighted.json"))
cat(toJSON(list(family="Poisson", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("poisson done\n")

# --- Binomial ---
y <- factor(rbinom(n, 1, plogis(0.5*x1 - 0.3*x2)), levels=0:1)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Binomial(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_binomial_weighted.json"))
cat(toJSON(list(family="Binomial", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(as.character(y)), w=as.numeric(w),
    y_levels=levels(y)), digits=15, auto_unbox=TRUE))
sink()
cat("binomial done\n")

# --- AdaExp ---
y <- factor(rbinom(n, 1, plogis(0.5*x1 - 0.3*x2)), levels=0:1)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=AdaExp(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_adaexp_weighted.json"))
cat(toJSON(list(family="AdaExp", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(as.character(y)), w=as.numeric(w),
    y_levels=levels(y)), digits=15, auto_unbox=TRUE))
sink()
cat("adaexp done\n")

# --- Laplace ---
set.seed(20250609)
n_odd <- 121
x1o <- rnorm(n_odd); x2o <- rnorm(n_odd)
yo <- 1 + 0.7*x1o - 0.4*x2o + rnorm(n_odd, sd=0.5)
wo <- sample(1:4, n_odd, replace=TRUE)
fit <- glmboost(yo ~ x1o + x2o, data=data.frame(x1o=x1o, x2o=x2o, yo=yo),
    weights=wo, family=Laplace(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_laplace_weighted.json"))
cat(toJSON(list(family="Laplace", mstop=mstop, nu=nu, n=n_odd, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1o), x2=as.numeric(x2o), y=as.numeric(yo), w=as.numeric(wo)),
    digits=15, auto_unbox=TRUE))
sink()
cat("laplace done\n")

# --- GammaReg ---
y <- rgamma(n, shape=2, scale=exp(0.3 + 0.2*x1 - 0.1*x2)/2)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=GammaReg(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_gammareg_weighted.json"))
cat(toJSON(list(family="GammaReg", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("gammareg done\n")

# --- Huber ---
y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Huber(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_huber_weighted.json"))
cat(toJSON(list(family="Huber", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("huber done\n")

# --- NBinomial ---
y <- rnbinom(n, mu=exp(0.5 + 0.3*x1 - 0.2*x2), size=2)
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=NBinomial(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_nbinomial_weighted.json"))
cat(toJSON(list(family="NBinomial", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("nbinomial done\n")

# --- PropOdds ---
y <- sample(1:3, n, replace=TRUE, prob=c(0.3,0.4,0.3))
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=PropOdds(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
sink(file.path(OUT, "glmboost_propodds_weighted.json"))
cat(toJSON(list(family="PropOdds", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L,
    coef=as.numeric(cc),  # PropOdds coef is a flat vector
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("propodds done\n")

# --- ExpectReg: tau = 0.5, 0.3, 0.8 ---
for (tau in c(0.5, 0.3, 0.8)) {
    set.seed(20250609)
    y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
    fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
        weights=w, family=ExpectReg(tau=tau),
        center=TRUE, control=boost_control(mstop=mstop, nu=nu))
    cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
    sink(file.path(OUT, sprintf("glmboost_expectreg_tau%.1f_weighted.json", tau)))
    cat(toJSON(list(family="ExpectReg", mstop=mstop, nu=nu, n=n, tau=tau, offset=as.numeric(off),
        risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
        selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
        x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
        digits=15, auto_unbox=TRUE))
    sink()
    cat(sprintf("expectreg tau=%.1f done\n", tau))
}

# --- QuantReg: tau = 0.5, 0.3, 0.8 ---
for (tau in c(0.5, 0.3, 0.8)) {
    set.seed(20250609)
    y <- 1 + 0.7*x1 - 0.4*x2 + rnorm(n, sd=0.5)
    fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
        weights=w, family=QuantReg(tau=tau),
        center=TRUE, control=boost_control(mstop=mstop, nu=nu))
    cc <- coef(fit, off2int=FALSE); off <- attr(cc,"offset"); if(is.null(off))off<-0
    sink(file.path(OUT, sprintf("glmboost_quantreg_tau%.1f_weighted.json", tau)))
    cat(toJSON(list(family="QuantReg", mstop=mstop, nu=nu, n=n, tau=tau, offset=as.numeric(off),
        risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
        selected=as.numeric(selected(fit))-1L, coef=as.numeric(cc),
        x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
        digits=15, auto_unbox=TRUE))
    sink()
    cat(sprintf("quantreg tau=%.1f done\n", tau))
}

cat(sprintf("\nDone. %d fixtures.\n", length(list.files(OUT, pattern="\\.json$"))))
