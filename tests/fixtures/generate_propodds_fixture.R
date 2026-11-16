suppressMessages(library(mboost))
suppressMessages(library(jsonlite))

OUT <- "tests/fixtures"
mstop <- 50; nu <- 0.1
set.seed(20250609)
n <- 120; x1 <- rnorm(n); x2 <- rnorm(n)
w <- sample(1:4, n, replace=TRUE)

y <- ordered(sample(1:3, n, replace=TRUE, prob=c(0.3,0.4,0.3)))
fit <- glmboost(y ~ x1 + x2, data=data.frame(x1=x1, x2=x2, y=y),
    weights=w, family=PropOdds(),
    center=TRUE, control=boost_control(mstop=mstop, nu=nu))
cc <- coef(fit, off2int=FALSE)
off <- attr(cc,"offset"); if(is.null(off)) off <- 0.0
sink(file.path(OUT, "glmboost_propodds_weighted.json"))
cat(toJSON(list(family="PropOdds", mstop=mstop, nu=nu, n=n, offset=as.numeric(off),
    risk=as.numeric(risk(fit)), fitted=as.numeric(predict(fit,type="link")),
    selected=as.numeric(selected(fit))-1L, coef=as.numeric(unlist(cc)),
    x1=as.numeric(x1), x2=as.numeric(x2), y=as.numeric(as.character(y)), w=as.numeric(w),
    y_levels=levels(y)), digits=15, auto_unbox=TRUE))
sink()
cat("propodds done\n")
