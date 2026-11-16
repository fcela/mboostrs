# A Multinomial model takes a factor predictor through its level indicators
# (brandom), fits, and predicts new rows coded against the training labels;
# an unseen label predicts at the average level.
suppressMessages(library(mboostrs))
set.seed(1); n <- 900
g <- factor(sample(c("a", "b", "c", "d"), n, TRUE)); x <- runif(n, -1, 1)
eff <- c(a = 1.2, b = 0, c = -0.8, d = 0.4)
e1 <- eff[as.character(g)] + sin(2 * x); e2 <- 0.8 * x
P <- cbind(1, exp(e1), exp(e2)); P <- P / rowSums(P)
y <- factor(apply(P, 1, function(p) sample(1:3, 1, prob = p)))
d <- data.frame(y = y, g = g, x = x)
fit <- gamboost(y ~ brandom(g) + bbs(x), data = d, family = Multinomial(),
                control = boost_control(mstop = 200))
nd <- data.frame(g = c("a", "c", "z"), x = c(0, 0, 0))
p <- predict(fit, newdata = nd, type = "response")
stopifnot(all(abs(rowSums(p) - 1) < 1e-12))
stopifnot(p[1, 2] > p[2, 2])
bad <- tryCatch(gamboost(y ~ btree(x), data = d, family = Multinomial()),
                error = function(e) conditionMessage(e))
stopifnot(grepl("brandom", bad))
# Predicting the training rows through newdata reproduces the fitted
# probabilities, for bols (an intercept-plus-slope design per row) and for a
# centred bbs term (the fit's reparametrization replayed at prediction).
fit2 <- gamboost(y ~ bols(x) + bbs(x, center = TRUE, df = 1), data = d,
                 family = Multinomial(), control = boost_control(mstop = 100))
p_fit <- predict(fit2, type = "response")
p_new <- predict(fit2, newdata = d["x"], type = "response")
stopifnot(max(abs(unname(p_fit) - unname(p_new))) < 1e-10)
# Variable importance credits each step with its change in the full
# likelihood, which every step lowers, so no share is negative.
vi <- varimp(fit2)$reduction
stopifnot(all(vi >= 0), abs(sum(vi) - 1) < 1e-8)
cat("ok\n")
