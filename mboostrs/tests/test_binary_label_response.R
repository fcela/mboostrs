# A two-class factor, logical or character response for the binary losses, as
# mboost takes it (first level 0, second 1). It was passed to the core as its
# integer codes 1/2 and refused ("y must be in {-1, +1}, {0, 1} ..., got 2").
suppressMessages(library(mboostrs))
set.seed(0); n <- 200
d <- data.frame(x = rnorm(n), z = rnorm(n))
d$b <- factor(ifelse(rbinom(n, 1, plogis(d$x)) == 1, "yes", "no"))
d$b01 <- as.integer(d$b) - 1L
d$l <- d$b == "yes"
d$s <- as.character(d$b)
for (fam in list(Binomial(), AdaExp())) {
  ref <- glmboost(b01 ~ x + z, data = d, family = fam, control = boost_control(mstop = 40))
  for (resp in c("b", "l", "s")) {
    fit <- glmboost(as.formula(paste(resp, "~ x + z")), data = d, family = fam,
                    control = boost_control(mstop = 40))
    stopifnot(max(abs(fit$fitted - ref$fitted)) < 1e-12)
  }
  g <- gamboost(b ~ bols(x), data = d, family = fam, control = boost_control(mstop = 40))
  g01 <- gamboost(b01 ~ bols(x), data = d, family = fam, control = boost_control(mstop = 40))
  stopifnot(max(abs(g$fitted - g01$fitted)) < 1e-12)
}
cv <- cvrisk(glmboost(b ~ x + z, data = d, family = Binomial(), control = boost_control(mstop = 40)),
             folds = cv_weights(n, "kfold", 3))
stopifnot(all(is.finite(as.matrix(cv$risk))))
d$three <- factor(sample(c("a", "b", "c"), n, TRUE))
stopifnot(inherits(try(glmboost(three ~ x, data = d, family = Binomial()), silent = TRUE), "try-error"))
cat("test_binary_label_response.R: OK\n")
