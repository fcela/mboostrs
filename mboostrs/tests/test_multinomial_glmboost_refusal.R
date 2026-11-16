# glmboost refuses Multinomial and names gamboost (issue #98).
suppressMessages(library(mboostrs))
msg <- tryCatch(glmboost(Species ~ Sepal.Length, data = iris, family = Multinomial()),
                error = conditionMessage)
stopifnot(is.character(msg), grepl("glmboost does not fit family = Multinomial", msg))
fit <- gamboost(Species ~ bols(Sepal.Length), data = iris, family = Multinomial(),
                control = boost_control(mstop = 20))
p <- predict(fit, type = "response")
stopifnot(max(abs(rowSums(p) - 1)) < 1e-12)
cat("test_multinomial_glmboost_refusal.R: OK\n")
