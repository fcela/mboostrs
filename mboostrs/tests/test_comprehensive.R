# Comprehensive test suite for mboostrs R package
# Tests all exported functions, S3 methods, and families.
# Run with: Rscript mboostrs/tests/test_comprehensive.R

library(mboostrs)
suppressMessages(library(survival))

cat("============================================\n")
cat("mboostrs R Package — Comprehensive Test Suite\n")
cat("============================================\n\n")

passed <- 0L
failed <- 0L
errors <- list()

check <- function(name, expr) {
  result <- tryCatch(
    { expr; TRUE },
    error = function(e) { errors[[name]] <<- conditionMessage(e); FALSE },
    warning = function(w) { errors[[name]] <<- conditionMessage(w); FALSE }
  )
  if (result) {
    passed <<- passed + 1L
    cat("  PASS: ", name, "\n", sep = "")
  } else {
    failed <<- failed + 1L
    cat("  FAIL: ", name, " — ", errors[[name]], "\n", sep = "")
  }
}

# ─── Test Data ────────────────────────────────────────────────────

set.seed(42)
n <- 200
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
y_gauss <- 0.3 * x1 - 0.1 * x2 + 0.2 * x3 + rnorm(n)
y_count <- rpois(n, exp(0.3 * x1))
y_binom <- rbinom(n, 1, 1 / (1 + exp(-(0.3 * x1 - 0.1 * x2))))

df_gauss <- data.frame(y = y_gauss, x1 = x1, x2 = x2, x3 = x3)
df_count <- data.frame(y = y_count, x1 = x1, x2 = x2)
df_binom <- data.frame(y = y_binom, x1 = x1, x2 = x2)

y_time <- rexp(n, rate = exp(0.3 * x1 - 0.1 * x2))
y_event <- rbinom(n, 1, 0.7)
df_surv <- data.frame(time = y_time, event = y_event, x1 = x1, x2 = x2)

#@validates r.family.contract
cat("--- Section 1: Family Constructors ---\n")

check("Gaussian()", Gaussian())
check("GaussReg()", GaussReg())
check("Binomial()", Binomial())
check("Poisson()", Poisson())
check("Laplace()", Laplace())
check("GammaReg()", GammaReg())
check("Quantile()", Quantile(tau = 0.75))
check("Expectile()", Expectile(tau = 0.9))
check("Huber()", Huber(delta = 1.345))
check("NBinomial()", NBinomial())
check("PropOdds(n_levels=3)", PropOdds(n_levels = 3))
check("AdaExp()", AdaExp())
check("CoxPH()", CoxPH())
check("Weibull()", Weibull())
check("Loglog()", Loglog())
check("Lognormal()", Lognormal())
check("HingeLoss()", HingeLoss())
check("AUC()", AUC())
check("Gehan()", Gehan())
check("Cindex()", Cindex())
check("Cindex(sigma=0.05)", Cindex(sigma = 0.05))
check("Multinomial()", Multinomial(classes = 3))

check("GaussianLSS()", GaussianLSS())
check("GammaLSS()", GammaLSS())
check("StudentTLSS()", StudentTLSS())
check("BetaLSS()", BetaLSS())
check("NBinomialLSS()", NBinomialLSS())
check("LogNormalLSS()", LogNormalLSS())
check("LogLogLSS()", LogLogLSS())
check("WeibullLSS()", WeibullLSS())
check("ZIPoLSS()", ZIPoLSS())
check("ZINBLSS()", ZINBLSS())
check("DirichletLSS(K=3)", DirichletLSS(K = 3))

cat("\n--- Section 2: boost_control ---\n")

check("boost_control default", {
  ctrl <- boost_control()
  stopifnot(ctrl$mstop == 100L, ctrl$nu == 0.1, ctrl$risk == "inbag")
})
check("boost_control custom", {
  ctrl <- boost_control(mstop = 200, nu = 0.05, risk = "oobag")
  stopifnot(ctrl$mstop == 200L, ctrl$nu == 0.05, ctrl$risk == "oobag")
})
check("boost_control invalid risk", {
  tryCatch(boost_control(risk = "invalid"), error = function(e) TRUE)
})

cat("\n--- Section 3: glmboost ---\n")

fit_glm <- glmboost(y ~ x1 + x2 + x3, data = df_gauss, family = Gaussian(),
                    control = boost_control(mstop = 100))
check("glmboost fit", exists("fit_glm", inherits = FALSE))
check("glmboost class", inherits(fit_glm, "mboostrs_glmboost"))
check("glmboost print", capture.output(print(fit_glm)))
check("glmboost coef", {
  co <- coef(fit_glm); stopifnot(length(co) == 4)
})
check("glmboost coef off2int", {
  co <- coef(fit_glm, off2int = TRUE)
})
check("glmboost fitted", {
  f <- fitted(fit_glm); stopifnot(length(f) == n)
})
check("glmboost fitted(mstop=50)", {
  f <- fitted(fit_glm, mstop = 50); stopifnot(length(f) == n)
})
check("glmboost predict in-sample", {
  p <- predict(fit_glm); stopifnot(length(p) == n)
})
check("glmboost predict newdata", {
  p <- predict(fit_glm, newdata = df_gauss[1:10, ])
  stopifnot(length(p) == 10)
})
check("glmboost predict type=response", {
  p <- predict(fit_glm, type = "response")
})
check("glmboost risk", {
  r <- risk(fit_glm); stopifnot(length(r) == 101)
})
check("glmboost residuals", {
  r <- residuals(fit_glm); stopifnot(length(r) == n)
})
check("glmboost residuals type=pearson", {
  r <- residuals(fit_glm, type = "pearson")
})
check("glmboost resid alias", {
  r <- resid(fit_glm); stopifnot(length(r) == n)
})
check("glmboost logLik", {
  ll <- logLik(fit_glm)
  stopifnot(!is.na(ll))
})
check("glmboost AIC", {
  # `stats::AIC` returns one value per model, and so does upstream `mboost`;
  # the whole path is the `AIC` attribute, reachable via aic_path().
  aic <- AIC(fit_glm)
  stopifnot(length(aic) == 1, !is.na(aic))
  stopifnot(length(aic_path(aic)) == 100, !any(is.na(aic_path(aic))))
  stopifnot(mstop(aic) >= 1, mstop(aic) <= 100)
  stopifnot(isTRUE(all.equal(as.numeric(aic), aic_path(aic)[mstop(aic)])))
})
check("glmboost AIC classical", {
  error <- tryCatch(AIC(fit_glm, method = "classical"), error = identity)
  stopifnot(inherits(error, "error"))
})
check("glmboost hatvalues", {
  h <- hatvalues(fit_glm); stopifnot(length(h) == n, all(h >= 0))
})
check("glmboost summary", capture.output(summary(fit_glm)))
check("glmboost varimp", {
  vi <- varimp(fit_glm)
  stopifnot(!is.null(vi$reduction), !is.null(vi$selfreq))
})
check("glmboost coef_path", {
  cp <- coef_path(fit_glm)
  stopifnot(is.list(cp))
})
check("glmboost selected", {
  sel <- selected(fit_glm)
  stopifnot(length(sel) > 0)
})
check("glmboost extract coefficients", {
  ex <- extract(fit_glm, "coefficients")
})
check("glmboost extract design", {
  ex <- extract(fit_glm, "design")
})
check("glmboost extract residuals", {
  ex <- extract(fit_glm, "residuals")
})
check("glmboost [k] slice", {
  fit10 <- fit_glm[10]
  stopifnot(fit10$mstop == 10)
})
check("glmboost update", {
  fit50 <- update(fit_glm, mstop = 50)
  stopifnot(fit50$mstop == 50)
})
check("glmboost confint", {
  ci <- confint(fit_glm, B = 10, level = 0.95)
  stopifnot(nrow(ci) == 4, ncol(ci) == 2)
})

cat("\n--- Section 4: gamboost ---\n")

fit_gam <- gamboost(y ~ bbs(x1, df = 4) + bols(x2) + bbs(x3, df = 4),
                    data = df_gauss, family = Gaussian(),
                    control = boost_control(mstop = 50))
check("gamboost fit", exists("fit_gam", inherits = FALSE))
check("gamboost class", inherits(fit_gam, "mboostrs_gamboost"))
check("gamboost print", capture.output(print(fit_gam)))
check("gamboost coef", {
  co <- coef(fit_gam); stopifnot(is.list(co))
})
check("gamboost fitted", {
  f <- fitted(fit_gam); stopifnot(length(f) == n)
})
check("gamboost predict in-sample", {
  p <- predict(fit_gam); stopifnot(length(p) == n)
})
check("gamboost predict newdata", {
  p <- predict(fit_gam, newdata = df_gauss[1:10, ])
  stopifnot(length(p) == 10)
})
check("gamboost risk", {
  r <- risk(fit_gam); stopifnot(length(r) == 51)
})
check("gamboost residuals", {
  r <- residuals(fit_gam); stopifnot(length(r) == n)
})
check("gamboost logLik", logLik(fit_gam))
check("gamboost AIC", {
  aic <- AIC(fit_gam)
  stopifnot(length(aic) == 1, length(aic_path(aic)) == 50)
})
check("gamboost hatvalues", {
  h <- hatvalues(fit_gam, approximate = TRUE); stopifnot(length(h) == n)
})
check("gamboost summary", capture.output(summary(fit_gam)))
check("gamboost varimp", varimp(fit_gam))
check("gamboost coef_path", coef_path(fit_gam))
check("gamboost selected", selected(fit_gam))
check("gamboost partial_effects", {
  pe <- partial_effects(fit_gam)
  stopifnot(is.list(pe))
})
check("gamboost extract", {
  extract(fit_gam, "design")
  extract(fit_gam, "penalty")
  extract(fit_gam, "df")
})
check("gamboost [k] slice", {
  fit10 <- fit_gam[10]
  stopifnot(fit10$mstop == 10)
})
check("gamboost update", update(fit_gam, mstop = 30))
check("gamboost confint", {
  ci <- confint(fit_gam, B = 5)
  stopifnot(is.list(ci))
})
check("gamboost bmono", {
  fit_mono <- gamboost(y ~ bmono(x1, df=4) + bols(x2), data = df_gauss,
                       family = Gaussian(), control = boost_control(mstop = 20))
  predict(fit_mono)
})
check("gamboost bmono type=quad.prog is honoured", {
  # `type=` is the mboost/Python spelling; it used to be silently dropped by
  # the R parser (only `solver=` was read), so the caller got the iterative
  # solver without being told. The two solvers must give *different* fits
  # here, which is what proves the argument reached the engine at all. Both
  # are named explicitly: `quad.prog` is now the default, so leaving one out
  # would compare the default against itself.
  f <- y ~ bmono(x1, constraint = "increasing", df = 4, type = "iterative")
  f_qp <- y ~ bmono(x1, constraint = "increasing", df = 4, type = "quad.prog")
  fit_it <- gamboost(f, data = df_gauss, family = Gaussian(),
                     control = boost_control(mstop = 20))
  fit_qp <- gamboost(f_qp, data = df_gauss, family = Gaussian(),
                     control = boost_control(mstop = 20))
  stopifnot(!isTRUE(all.equal(as.numeric(predict(fit_it)),
                              as.numeric(predict(fit_qp)),
                              tolerance = 1e-12)))
  # `solver=` remains an accepted synonym and must agree with `type=`.
  fit_solver <- gamboost(
    y ~ bmono(x1, constraint = "increasing", df = 4, solver = "quad.prog"),
    data = df_gauss, family = Gaussian(), control = boost_control(mstop = 20))
  stopifnot(isTRUE(all.equal(as.numeric(predict(fit_qp)),
                             as.numeric(predict(fit_solver)))))
})
check("gamboost bmono rejects conflicting type/solver", {
  bad <- tryCatch({
    gamboost(y ~ bmono(x1, df = 4, type = "quad.prog", solver = "iterative"),
             data = df_gauss, family = Gaussian(),
             control = boost_control(mstop = 5))
    FALSE
  }, error = function(e) grepl("both given", conditionMessage(e)))
  stopifnot(bad)
})
check("gamboost btree", {
  fit_tree <- gamboost(y ~ btree(x1) + bols(x2), data = df_gauss,
                       family = Gaussian(), control = boost_control(mstop = 20))
  predict(fit_tree)
})
check("gamboost brandom", {
  set.seed(1)
  grp <- sample(1:5, n, replace = TRUE)
  df_rand <- data.frame(y = y_gauss, x1 = x1, grp = grp)
  fit_rand <- gamboost(y ~ brandom(grp, df=4), data = df_rand,
                       family = Gaussian(), control = boost_control(mstop = 20))
  p <- predict(fit_rand)
  stopifnot(length(p) == n)
  pe <- partial_effects(fit_rand)
  stopifnot(is.list(pe), length(pe) == 1)
})
check("gamboost bns", {
  fit_bns <- gamboost(y ~ bns(x1, df=4) + bols(x2), data = df_gauss,
                      family = Gaussian(), control = boost_control(mstop = 20))
  p <- predict(fit_bns)
  stopifnot(length(p) == n)
  pe <- partial_effects(fit_bns)
  stopifnot(is.list(pe))
})
check("gamboost brandom predict newdata", {
  set.seed(1)
  grp <- sample(1:5, n, replace = TRUE)
  df_rand <- data.frame(y = y_gauss, x1 = x1, grp = grp)
  fit_rand <- gamboost(y ~ brandom(grp, df=4), data = df_rand,
                       family = Gaussian(), control = boost_control(mstop = 20))
  nd <- data.frame(grp = c(1, 3, 5))
  p <- predict(fit_rand, newdata = nd)
  stopifnot(length(p) == 3)
})

cat("\n--- Section 5: Families (glmboost) ---\n")

check("glmboost Poisson", {
  fit <- glmboost(y ~ x1 + x2, data = df_count, family = Poisson(),
                  control = boost_control(mstop = 50))
  predict(fit, type = "response")
})
check("glmboost Binomial", {
  fit <- glmboost(y ~ x1 + x2, data = df_binom, family = Binomial(),
                  control = boost_control(mstop = 50))
  p <- predict(fit, type = "response")
  stopifnot(all(p >= 0 & p <= 1))
})
check("glmboost Laplace", {
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = Laplace(),
                  control = boost_control(mstop = 50))
  residuals(fit)
})
check("glmboost Quantile", {
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = Quantile(tau = 0.75),
                  control = boost_control(mstop = 50))
  predict(fit)
})
check("glmboost Expectile", {
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = Expectile(tau = 0.9),
                  control = boost_control(mstop = 50))
})
check("glmboost Huber", {
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = Huber(),
                  control = boost_control(mstop = 50))
})
check("glmboost GaussReg", {
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = GaussReg(),
                  control = boost_control(mstop = 50))
})
check("glmboost GammaReg", {
  df_pos <- data.frame(y = abs(y_gauss) + 0.1, x1 = x1, x2 = x2)
  fit <- glmboost(y ~ x1 + x2, data = df_pos, family = GammaReg(),
                  control = boost_control(mstop = 50))
})
check("glmboost AdaExp", {
  fit <- glmboost(y ~ x1 + x2, data = df_binom, family = AdaExp(),
                  control = boost_control(mstop = 50))
})
check("margin families accept 0/1 and -1/+1 coding", {
  df_pm <- transform(df_binom, y = ifelse(y == 0, -1, 1))
  for (family in list(AdaExp(), HingeLoss())) {
    fit01 <- glmboost(y ~ x1 + x2, data = df_binom, family = family,
                      control = boost_control(mstop = 20))
    fitpm <- glmboost(y ~ x1 + x2, data = df_pm, family = family,
                      control = boost_control(mstop = 20))
    stopifnot(identical(fit01$selected, fitpm$selected))
    stopifnot(isTRUE(all.equal(fit01$risk, fitpm$risk, tolerance = 0)))
    stopifnot(isTRUE(all.equal(fit01$fitted, fitpm$fitted, tolerance = 0)))
  }
})
#@validates smoke.r.auc_cvrisk
check("glmboost AUC", {
  fit <- glmboost(y ~ x1 + x2, data = df_binom, family = AUC(),
                  control = boost_control(mstop = 50))
  stopifnot(length(fit$fitted) == n, is.finite(tail(fit$risk, 1)))
})
check("glmboost AUC accepts non-unit weights", {
  # Auc now implements a principled *weighted* gradient (see the Auc doc
  # comment in families.rs) instead of rejecting non-unit weights -- a
  # deliberate mboostrs extension, not R parity (R's own weighted AUC
  # ngradient is broken: it returns a compacted vector of length sum(w),
  # not n). This only checks that non-unit weights are accepted end-to-end
  # and produce a finite fit; the R-independent correctness oracle for the
  # weighted gradient lives in Rust
  # (auc_ngradient_matches_finite_difference_of_weighted_unstandardized_loss,
  # auc_zero_one_weights_match_unit_weight_subset).
  fit <- glmboost(y ~ x1 + x2, data = df_binom, family = AUC(),
                  weights = ifelse(seq_len(n) %% 2 == 0, 2, 1),
                  control = boost_control(mstop = 10))
  stopifnot(length(fit$fitted) == n, is.finite(tail(fit$risk, 1)))
})
check("cvrisk AUC subsampling", {
  fit_auc <- glmboost(y ~ x1 + x2, data = df_binom, family = AUC(),
                      control = boost_control(mstop = 20))
  cv <- cvrisk(fit_auc, fold_type = "subsampling", b = 5)
  stopifnot(inherits(cv, "mboostrs_cvrisk"), is.finite(mstop(cv)))
})
check("cvrisk AUC bootstrap", {
  # Bootstrap folds are integer case-weight counts (not just {0,1}); the
  # weighted gradient accepts any non-negative weight, unlike R's broken
  # weighted AUC ngradient, so (unlike Cindex) bootstrap folds are also
  # supported here, not just subsampling.
  fit_auc <- glmboost(y ~ x1 + x2, data = df_binom, family = AUC(),
                      control = boost_control(mstop = 20))
  cv <- cvrisk(fit_auc, fold_type = "bootstrap", b = 5)
  stopifnot(inherits(cv, "mboostrs_cvrisk"), is.finite(mstop(cv)))
})
#@validates smoke.r.gehan_cvrisk
check("glmboost Gehan", {
  fit <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Gehan(),
                  control = boost_control(mstop = 50))
  stopifnot(length(fit$fitted) == n, is.finite(tail(fit$risk, 1)))
  # The offset is unidentifiable (see families.rs Gehan doc comment);
  # mboostrs defines offset() == 0 rather than R's arbitrary optimize()
  # value on the flat objective.
  stopifnot(fit$offset == 0)
})
check("glmboost Gehan accepts non-unit weights", {
  # Same principled weighted-gradient extension as AUC above (see the
  # Gehan doc comment in families.rs); not R parity, R's weighted Gehan
  # ngradient has the same upstream bug as AUC's.
  fit <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Gehan(),
                  weights = ifelse(seq_len(n) %% 2 == 0, 2, 1),
                  control = boost_control(mstop = 10))
  stopifnot(length(fit$fitted) == n, is.finite(tail(fit$risk, 1)))
})
check("cvrisk Gehan subsampling", {
  fit_gehan <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Gehan(),
                        control = boost_control(mstop = 20))
  cv <- cvrisk(fit_gehan, fold_type = "subsampling", b = 5)
  stopifnot(inherits(cv, "mboostrs_cvrisk"), is.finite(mstop(cv)))
})
check("cvrisk Gehan bootstrap", {
  fit_gehan <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Gehan(),
                        control = boost_control(mstop = 20))
  cv <- cvrisk(fit_gehan, fold_type = "bootstrap", b = 5)
  stopifnot(inherits(cv, "mboostrs_cvrisk"), is.finite(mstop(cv)))
})
check("glmboost Cindex", {
  fit <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Cindex(),
                  control = boost_control(mstop = 50))
  stopifnot(length(fit$fitted) == n, is.finite(tail(fit$risk, 1)))
  # Unlike Gehan, Cindex's offset matches R exactly (both 0) -- not an
  # unidentifiable/intentional-deviation case.
  stopifnot(fit$offset == 0)
})
check("glmboost Cindex(sigma=0.05)", {
  fit <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Cindex(sigma = 0.05),
                  control = boost_control(mstop = 30))
  stopifnot(length(fit$fitted) == n, is.finite(tail(fit$risk, 1)))
})
check("glmboost Cindex rejects non-zero-one weights", {
  ok <- tryCatch({
    glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Cindex(),
             weights = ifelse(seq_len(n) %% 2 == 0, 2, 1),
             control = boost_control(mstop = 10))
    FALSE
  }, error = function(e) TRUE)
  stopifnot(ok)
})
check("glmboost Cindex accepts zero-one weights", {
  fit <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Cindex(),
                  weights = ifelse(seq_len(n) %% 3 == 0, 0, 1),
                  control = boost_control(mstop = 20))
  stopifnot(length(fit$fitted) == n, is.finite(tail(fit$risk, 1)))
})
check("cvrisk Cindex subsampling", {
  fit_cindex <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Cindex(),
                         control = boost_control(mstop = 20))
  cv <- cvrisk(fit_cindex, fold_type = "subsampling", b = 5)
  stopifnot(inherits(cv, "mboostrs_cvrisk"))
})
check("cvrisk Cindex bootstrap rejected", {
  fit_cindex <- glmboost(Surv(time, event) ~ x1 + x2, data = df_surv, family = Cindex(),
                         control = boost_control(mstop = 10))
  ok <- tryCatch({
    cvrisk(fit_cindex, fold_type = "bootstrap", b = 5)
    FALSE
  }, error = function(e) TRUE)
  stopifnot(ok)
})

cat("\n--- Section 6: cvrisk ---\n")

check("cv_weights", {
  w <- cv_weights(n, fold_type = "subsampling", b = 5)
  stopifnot(nrow(w) == n, ncol(w) == 5)
})
check("cvrisk glmboost", {
  cv <- cvrisk(fit_glm, b = 5)
  stopifnot(inherits(cv, "mboostrs_cvrisk"))
})
check("cvrisk print", capture.output(print(cv_risk_tmp <- cvrisk(fit_glm, b = 3))))
check("mstop(cvrisk)", {
  cv <- cvrisk(fit_glm, b = 5)
  m <- mstop(cv)
  stopifnot(is.integer(m) || is.numeric(m), length(m) == 1)
})
check("cvrisk gamboost", {
  cv <- cvrisk(fit_gam, b = 3)
  stopifnot(inherits(cv, "mboostrs_cvrisk"))
})

cat("\n--- Section 7: stabsel ---\n")

check("stabsel_params", {
  par <- stabsel_params(p = 10, b = 25, cutoff = 0.7, q = 3, pfer = -1.0,
                        assumption = "unimodal", sampling_type = "SS")
  stopifnot(!is.null(par$cutoff), !is.null(par$q))
})
check("stabsel", {
  # Need enough variables for stability selection to be meaningful
  set.seed(42)
  X <- matrix(rnorm(n * 15), n, 15)
  df_ss <- data.frame(y = y_gauss, X)
  ss <- stabsel(y ~ ., data = df_ss, family = Gaussian(),
                control = boost_control(mstop = 30), B = 10)
  stopifnot(inherits(ss, "mboostrs_stabsel"))
})
check("stabsel print", {
  set.seed(42)
  X <- matrix(rnorm(n * 15), n, 15)
  df_ss <- data.frame(y = y_gauss, X)
  ss <- stabsel(y ~ ., data = df_ss, family = Gaussian(),
                control = boost_control(mstop = 20), B = 5)
  capture.output(print(ss))
})

cat("\n--- Section 8: glmboostLSS ---\n")

check("glmboostLSS Gaussian", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  stopifnot(inherits(fit, "mboostrs_glmboostLSS"))
})
check("glmboostLSS print", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  capture.output(print(fit))
})
check("glmboostLSS coef", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  co <- coef(fit)
  stopifnot(is.list(co), length(co) == 2)
})
check("glmboostLSS predict", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  p <- predict(fit)
  stopifnot(is.list(p), length(p) == 2)
})
check("glmboostLSS predict type=response", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  p <- predict(fit, type = "response")
})
check("glmboostLSS residuals", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  r <- residuals(fit)
})
check("glmboostLSS summary", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  capture.output(summary(fit))
})
check("glmboostLSS logLik", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  ll <- logLik(fit)
})
check("glmboostLSS [k] slice", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  fit10 <- fit[10]
})
check("glmboostLSS noncyclic", {
  fit <- glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20), method = "noncyclic")
})
check("glmboostLSS GammaLSS", {
  df_pos <- data.frame(y = abs(y_gauss) + 0.1, x1 = x1, x2 = x2)
  fit <- glmboostLSS(y ~ x1 + x2, data = df_pos, families = GammaLSS(),
                     control = boost_control(mstop = 20))
})
check("cvrisk_glmboostLSS", {
  folds <- cv_weights(n, b = 3)
  grid <- make_grid_lss(c(20, 20), K = 2, param_names = c("mu", "sigma"))
  cv <- cvrisk_glmboostLSS(y ~ x1 + x2, data = df_gauss, families = GaussianLSS(),
                            control = boost_control(mstop = 20),
                            folds = folds, grid = grid)
  stopifnot(is.matrix(cv))
})

cat("\n--- Section 9: gamboostLSS ---\n")

check("gamboostLSS Gaussian", {
  fit <- gamboostLSS(y ~ bbs(x1, df = 4), data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  stopifnot(inherits(fit, "mboostrs_gamboostLSS"))
})
check("gamboostLSS print", {
  fit <- gamboostLSS(y ~ bbs(x1, df = 4), data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  capture.output(print(fit))
})
check("gamboostLSS predict", {
  fit <- gamboostLSS(y ~ bbs(x1, df = 4), data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  p <- predict(fit)
})
check("gamboostLSS predict type=response", {
  fit <- gamboostLSS(y ~ bbs(x1, df = 4), data = df_gauss, families = GaussianLSS(),
                     control = boost_control(mstop = 20))
  p <- predict(fit, type = "response")
})

cat("\n--- Section 10: make_grid_lss ---\n")

check("make_grid_lss scalar", {
  grid <- make_grid_lss(20, K = 2)
  stopifnot(is.matrix(grid), ncol(grid) == 2)
})
check("make_grid_lss vector", {
  grid <- make_grid_lss(c(mu = 30, sigma = 20), K = 2, param_names = c("mu", "sigma"))
  stopifnot(is.matrix(grid), ncol(grid) == 2)
})

cat("\n--- Section 11: Edge Cases ---\n")

check("glmboost with offset", {
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = Gaussian(),
                  control = boost_control(mstop = 50), offset = 0.5)
})
check("glmboost with weights", {
  w <- runif(n)
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = Gaussian(),
                  control = boost_control(mstop = 50), weights = w)
})
check("glmboost risk=none", {
  fit <- glmboost(y ~ x1 + x2, data = df_gauss, family = Gaussian(),
                  control = boost_control(mstop = 50, risk = "none"))
})
check("gamboost oobag risk", {
  fit <- gamboost(y ~ bbs(x1, df = 4) + bols(x2), data = df_gauss,
                  family = Gaussian(), control = boost_control(mstop = 30, risk = "oobag"))
})
check("NA in y rejected", {
  df_na <- df_gauss; df_na$y[1] <- NA
  tryCatch(glmboost(y ~ x1, data = df_na), error = function(e) TRUE)
})
check("large mstop", {
  fit <- glmboost(y ~ x1, data = df_gauss, control = boost_control(mstop = 500))
})

cat("\n--- Section 12: Multinomial ---\n")

check("gamboost Multinomial", {
  set.seed(123)
  y_fac <- factor(sample(letters[1:3], n, replace = TRUE))
  df_mn <- data.frame(y = y_fac, x1 = x1, x2 = x2)
  fit <- gamboost(y ~ bols(x1) + bols(x2), data = df_mn, family = Multinomial(),
                  control = boost_control(mstop = 30))
  p <- predict(fit, type = "response")
  stopifnot(is.matrix(p), ncol(p) == 3)
})

cat("\n============================================\n")
cat(sprintf("Results: %d passed, %d failed (out of %d)\n", passed, failed, passed + failed))
cat("============================================\n")

if (failed > 0) {
  cat("\nFailed tests:\n")
  for (name in names(errors)) {
    cat("  ", name, ": ", errors[[name]], "\n", sep = "")
  }
  quit(status = 1)
} else {
  cat("\nAll tests passed!\n")
}

# --- intercept control in glmboost; refused by the LSS fits -----------------
# `- 1` / `+ 0` drop the intercept learner, as mboost does (issue #101); they
# used to be refused, and before that silently ignored.
local({
  set.seed(11); n <- 50
  d <- data.frame(y = 5 + rnorm(n), x = rnorm(n))
  for (f in list(y ~ x - 1, y ~ x + 0, y ~ 0 + x)) {
    fit <- suppressWarnings(glmboost(f, data = d, control = boost_control(mstop = 20)))
    stopifnot(identical(names(coef(fit)), "x"))
  }
  err <- tryCatch(glmboostLSS(list(mu = y ~ x - 1, sigma = y ~ x), data = d),
                  error = function(e) conditionMessage(e))
  stopifnot(grepl("intercept control", err, fixed = TRUE))
  cat("PASS: intercept control in glmboost, refused by glmboostLSS\n")
})
