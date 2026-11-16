suppressMessages(library(mboost))
suppressMessages(library(gamboostLSS))
suppressMessages(library(jsonlite))

OUT <- "tests/fixtures"

# GaussianLSS
set.seed(20250609)
n <- 120; x1 <- rnorm(n); x2 <- rnorm(n)
y <- rnorm(n, mean=1+0.5*x1-0.3*x2, sd=exp(0.1+0.2*x1))
w <- sample(1:4, n, replace=TRUE)
fit <- glmboostLSS(y ~ x1 + x2, data=data.frame(x1=x1,x2=x2,y=y),
    weights=w, families=GaussianLSS(),
    control=boost_control(mstop=50, nu=0.1))
r <- risk(fit)
sink(file.path(OUT, "gamboostlss_gaussian_weighted.json"))
cat(toJSON(list(family="GaussianLSS", mstop=50, nu=0.1, n=n,
    risk_mu=as.numeric(r$mu), risk_sigma=as.numeric(r$sigma),
    fitted_mu=as.numeric(fitted(fit,parameter="mu")),
    fitted_sigma=as.numeric(fitted(fit,parameter="sigma")),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("gaussianlss done\n")

# GammaLSS
set.seed(20250609)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- rexp(n, rate=exp(-0.5-0.3*x1+0.2*x2))
w <- sample(1:4, n, replace=TRUE)
fit <- glmboostLSS(y ~ x1 + x2, data=data.frame(x1=x1,x2=x2,y=y),
    weights=w, families=GammaLSS(),
    control=boost_control(mstop=50, nu=0.1))
r <- risk(fit)
sink(file.path(OUT, "gamboostlss_gamma_weighted.json"))
cat(toJSON(list(family="GammaLSS", mstop=50, nu=0.1, n=n,
    risk_mu=as.numeric(r$mu), risk_sigma=as.numeric(r$sigma),
    fitted_mu=as.numeric(fitted(fit,parameter="mu")),
    fitted_sigma=as.numeric(fitted(fit,parameter="sigma")),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(y), w=as.numeric(w)),
    digits=15, auto_unbox=TRUE))
sink()
cat("gammalss done\n")

cat("Done.\n")
