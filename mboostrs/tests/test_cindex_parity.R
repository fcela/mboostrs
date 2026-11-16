#@validates parity.r.cindex
# Parity check: mboostrs::glmboost(..., family = Cindex()) vs
# mboost::glmboost(..., family = Cindex()) (the real R package). Unlike
# Gehan, Cindex's offset matches R exactly (both 0), so raw fitted values
# are compared directly -- no centering needed.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping Cindex glmboost parity check")
  quit(save = "no")
}
library(mboostrs)
suppressWarnings(suppressMessages(library(mboost)))
suppressWarnings(suppressMessages(library(survival)))

chk <- function(a, b, what, tol = 1e-8) {
  md <- max(abs(as.numeric(a) - as.numeric(b)))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-32s maxdiff=%.2e\n", what, md))
}

set.seed(11)
n <- 150
x1 <- rnorm(n); x2 <- rnorm(n)
eta <- 0.35 * x1 - 0.2 * x2
time <- rexp(n, rate = exp(eta))
event <- rbinom(n, 1, 0.7)
df <- data.frame(time = time, event = event, x1 = x1, x2 = x2)

ctrl_m <- mboostrs::boost_control(mstop = 60, nu = 0.1)
ctrl_r <- mboost::boost_control(mstop = 60, nu = 0.1)

# --- default sigma ---
r_fit <- mboost::glmboost(Surv(time, event) ~ x1 + x2, data = df,
                          family = mboost::Cindex(), center = TRUE, control = ctrl_r)
m_fit <- mboostrs::glmboost(Surv(time, event) ~ x1 + x2, data = df,
                            family = mboostrs::Cindex(), control = ctrl_m)
chk(fitted(m_fit), as.numeric(predict(r_fit, type = "link")), "Cindex fitted (sigma=0.1)")
chk(mboostrs::risk(m_fit), as.numeric(mboost::risk(r_fit)), "Cindex risk path (sigma=0.1)")
chk(m_fit$offset, 0, "Cindex offset (sigma=0.1)")
stopifnot(identical(as.integer(m_fit$selected), as.integer(r_fit$xselect())))

# --- non-default sigma ---
r_fit2 <- mboost::glmboost(Surv(time, event) ~ x1 + x2, data = df,
                           family = mboost::Cindex(sigma = 0.05), center = TRUE,
                           control = ctrl_r)
m_fit2 <- mboostrs::glmboost(Surv(time, event) ~ x1 + x2, data = df,
                             family = mboostrs::Cindex(sigma = 0.05), control = ctrl_m)
chk(fitted(m_fit2), as.numeric(predict(r_fit2, type = "link")), "Cindex fitted (sigma=0.05)")
chk(mboostrs::risk(m_fit2), as.numeric(mboost::risk(r_fit2)), "Cindex risk path (sigma=0.05)")

# --- {0,1} case weights (the IPCW-on-subset path) ---
w01 <- ifelse(seq_len(n) %% 4 == 0, 0, 1)
r_fit3 <- mboost::glmboost(Surv(time, event) ~ x1 + x2, data = df,
                           family = mboost::Cindex(), weights = w01, center = TRUE,
                           control = ctrl_r)
m_fit3 <- mboostrs::glmboost(Surv(time, event) ~ x1 + x2, data = df,
                             family = mboostrs::Cindex(), weights = w01, control = ctrl_m)
chk(fitted(m_fit3), as.numeric(predict(r_fit3, type = "link")), "Cindex fitted ({0,1} weights)")
chk(mboostrs::risk(m_fit3), as.numeric(mboost::risk(r_fit3)), "Cindex risk path ({0,1} weights)")

cat("Cindex glmboost R-binding parity: PASS\n")
