# A fixed mstop in conformal_interval() must refit to that many steps.
suppressMessages(library(mboostrs))
ns <- asNamespace("mboostrs")

sim <- function(seed, n = 1000) {
  set.seed(seed)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  data.frame(y = sin(2 * pi * x1) + 0.5 * x2 + rnorm(n), x1 = x1, x2 = x2)
}
d0 <- sim(0)
fit0 <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d0, family = Gaussian())

# 3. Fixed mstop: the split interval's point prediction equals a model fitted
#    directly to 300 steps on the same training rows.
n <- nrow(d0)
perm <- ns$.conformal_permutation(n, 1)
n_calib <- ns$.conformal_calibration_count(n, 0.5)
train <- perm[-seq_len(n_calib)]
ref <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = d0[train, ], family = Gaussian(),
                control = boost_control(mstop = 300))
ci <- conformal_interval(fit0, level = 0.9, seed = 1, method = "split", mstop = 300)
stopifnot(max(abs(ci$predicted - as.double(predict(ref, newdata = d0, type = "response")))) < 1e-8)

# 4. Held-out calibration refuses more steps than the model was fitted with.
cal <- d0[sample(n, 300), ]
err <- tryCatch(conformal_interval(fit0, level = 0.9, calibration = cal, mstop = 300),
                error = function(e) conditionMessage(e))
stopifnot(grepl("beyond the model's fitted 100 steps", err, fixed = TRUE))

cat("test_conformal_fixed_mstop.R: all checks passed\n")
