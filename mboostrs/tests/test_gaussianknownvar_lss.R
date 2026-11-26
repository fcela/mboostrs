# GaussianKnownVarLSS in the R frontend, matching the Python one.
#
# `v_i = s2_i + exp(eta_sigma_i)`: the known sampling variance plus a boosted
# excess variance component. The response is `cbind(outcome, s2)`, which
# flattens column-major to the `[outcome..., s2...]` layout the Rust family
# reads -- the same path `DirichletLSS` already used, which is why exposing
# this needed no new response machinery.
#
# This family has **no gamboostLSS counterpart**, so there is no upstream
# oracle. The reference is the Python frontend, which is the implementation
# the study it was written for used; the two must agree because they reach the
# same engine, and a disagreement means a frontend is mis-assembling the call.
#@validates parity.r.gaussianknownvarlss
library(mboostrs)

# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
ok <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

set.seed(1)
n <- 200
x <- rnorm(n)
s2 <- runif(n, 0.05, 0.5)
y <- rnorm(n, 1 + x, sqrt(s2 + 0.25))
d <- data.frame(y = y, s2 = s2, x = x)

fit <- glmboostLSS(cbind(y, s2) ~ x, data = d,
                   families = GaussianKnownVarLSS(),
                   control = boost_control(mstop = 40, nu = 0.1))

# Values from the Python frontend on this fixture:
#   mb.glmboostLSS("cbind(y, s2) ~ x", data=d,
#                  families=mb.GaussianKnownVarLSS(),
#                  control=mb.boost_control(40, 0.1))
PYTHON_SLOPE <- 1.011145
PYTHON_RISK <- 242.572815
PYTHON_OFFSET_MU <- 1.0477000701156733
PYTHON_FITTED_HEAD <- c(0.378329, 1.197454, 0.166823)

slope <- as.numeric(coef(fit)$mu)[2]
ok(abs(slope - PYTHON_SLOPE) < 1e-6,
   sprintf("mu slope %.8f differs from Python's %.8f", slope, PYTHON_SLOPE))

risk <- fit$risk[length(fit$risk)]
ok(abs(risk - PYTHON_RISK) < 1e-4,
   sprintf("final risk %.6f differs from Python's %.6f", risk, PYTHON_RISK))

ok(abs(fit$params$mu$offset - PYTHON_OFFSET_MU) < 1e-9,
   sprintf("mu offset %.10f differs from Python's %.10f",
           fit$params$mu$offset, PYTHON_OFFSET_MU))

predicted <- predict(fit, type = "link")
ok(max(abs(predicted$mu[1:3] - PYTHON_FITTED_HEAD)) < 1e-6,
   sprintf("fitted mu head %s differs from Python's %s",
           paste(round(predicted$mu[1:3], 6), collapse = " "),
           paste(PYTHON_FITTED_HEAD, collapse = " ")))

# The family's own contract: only sigma's excess component is boosted, so
# fitting one with a large known variance should leave less excess to find
# than the same data with a small one.
s2_large <- rep(4.0, n)
d_large <- data.frame(y = y, s2 = s2_large, x = x)
fit_large <- glmboostLSS(cbind(y, s2) ~ x, data = d_large,
                         families = GaussianKnownVarLSS(),
                         control = boost_control(mstop = 40, nu = 0.1))
excess_small <- mean(exp(predict(fit, type = "link")$sigma))
excess_large <- mean(exp(predict(fit_large, type = "link")$sigma))
ok(excess_large < excess_small,
   sprintf("a larger known variance should leave less excess: %.4f vs %.4f",
           excess_large, excess_small))

# The response must be the two-column form; a plain one is refused.
refused <- try(glmboostLSS(y ~ x, data = d, families = GaussianKnownVarLSS(),
                           control = boost_control(mstop = 5)), silent = TRUE)
ok(inherits(refused, "try-error"),
   "a plain response should be refused for a known-variance family")

cat("GaussianKnownVarLSS (R): PASS\n")
