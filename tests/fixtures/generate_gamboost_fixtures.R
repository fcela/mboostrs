suppressMessages(library(mboost))
suppressMessages(library(jsonlite))

OUT <- "tests/fixtures"
mstop <- 100; nu <- 0.1
set.seed(20250609)
n <- 120; x1 <- rnorm(n); x2 <- rnorm(n)
w <- sample(1:4, n, replace=TRUE)

write_fixture <- function(fam_name, fam, y, extra=list()) {
    fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
        data=data.frame(x1=x1, x2=x2, y=y),
        weights=w, family=fam, control=boost_control(mstop=mstop, nu=nu))
    off <- fit$offset; if (is.null(off)) off <- 0.0
    d <- c(list(family=fam_name, mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
        risk=as.numeric(risk(fit)),
        fitted=as.numeric(predict(fit, type="link")),
        selected=as.numeric(selected(fit)) - 1L,
        x1=as.numeric(x1), x2=as.numeric(x2),
        y=as.numeric(y), w=as.numeric(w)), extra)
    write_json(d, file.path(OUT, sprintf("gamboost_%s_weighted.json", fam_name)),
        digits=15, auto_unbox=TRUE)
    cat(sprintf("  %s\n", fam_name))
}

write_fixture("gaussian", Gaussian(),
    1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5))
write_fixture("gaussreg", GaussReg(),
    1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5))
write_fixture("poisson", Poisson(),
    rpois(n, exp(0.5 + 0.3*sin(x1) - 0.2*x2)))
write_fixture("gammareg", GammaReg(),
    rgamma(n, shape=2, scale=exp(0.3+0.2*sin(x1)-0.1*x2)/2))
write_fixture("huber", Huber(),
    1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5))
write_fixture("expectreg", ExpectReg(tau=0.5),
    1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5),
    extra=list(tau=0.5))
write_fixture("quantreg", QuantReg(tau=0.5),
    1 + 0.7*sin(x1) - 0.4*x2^2 + rnorm(n, sd=0.5),
    extra=list(tau=0.5))

# Binomial/AdaExp: factor response
y <- factor(rbinom(n, 1, plogis(0.5*sin(x1)-0.3*x2)), levels=0:1)
fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=Binomial(), control=boost_control(mstop=mstop, nu=nu))
off <- fit$offset; if (is.null(off)) off <- 0.0
write_json(list(family="Binomial", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L,
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(as.character(y)), w=as.numeric(w)),
    file.path(OUT, "gamboost_binomial_weighted.json"), digits=15, auto_unbox=TRUE)
cat("  binomial\n")

fit <- gamboost(y ~ bbs(x1, df=4) + bbs(x2, df=4),
    data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=AdaExp(), control=boost_control(mstop=mstop, nu=nu))
off <- fit$offset; if (is.null(off)) off <- 0.0
write_json(list(family="AdaExp", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L,
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(as.character(y)), w=as.numeric(w)),
    file.path(OUT, "gamboost_adaexp_weighted.json"), digits=15, auto_unbox=TRUE)
cat("  adaexp\n")

# Laplace (odd n)
set.seed(20250609)
n_odd <- 121; x1o <- rnorm(n_odd); x2o <- rnorm(n_odd)
wo <- sample(1:4, n_odd, replace=TRUE)
yo <- 1 + 0.7*sin(x1o) - 0.4*x2o^2 + rnorm(n_odd, sd=0.5)
fit <- gamboost(yo ~ bbs(x1o, df=4) + bbs(x2o, df=4),
    data=data.frame(x1o=x1o, x2o=x2o, yo=yo),
    weights=wo, family=Laplace(), control=boost_control(mstop=mstop, nu=nu))
off <- fit$offset; if (is.null(off)) off <- 0.0
write_json(list(family="Laplace", mstop=mstop, nu=nu, n=n_odd, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L,
    x1=as.numeric(x1o), x2=as.numeric(x2o), y=as.numeric(yo), w=as.numeric(wo)),
    file.path(OUT, "gamboost_laplace_weighted.json"), digits=15, auto_unbox=TRUE)
cat("  laplace\n")

cat(sprintf("Done. %d GAMBoost fixtures.\n", length(grep("gamboost_",list.files(OUT)))))
