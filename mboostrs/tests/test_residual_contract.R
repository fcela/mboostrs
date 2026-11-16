# Residual kinds are the quantities their names refer to, or refused.
#
# Mirrors tests/test_residual_contract.py: standard-model families show the
# kinds the family likelihood table declares (robust losses have no Pearson
# residual -- that vector is the working gradient), censored responses refuse a
# response residual (CoxPH, AFT, survival LSS), and LSS location families show
# response/pearson only where a variance function exists.
library(mboostrs)
library(survival)

chk <- function(what, ok, detail = "") {
  if (!ok) stop(sprintf("FAIL %s%s", what, if (nzchar(detail)) paste0(": ", detail) else ""))
  cat(sprintf("OK   %s\n", what))
}

err <- function(expr) tryCatch({ force(expr); NULL }, error = identity)

set.seed(12); n <- 160
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 0.5 * x1 + rnorm(n)
d <- data.frame(y = y, x1 = x1, x2 = x2)

# ── LSS: names, values, refusal surface ─────────────────────────────
lss <- glmboostLSS(y ~ x1 + x2, data = d, families = GaussianLSS(),
                   control = boost_control(mstop = 25, nu = 0.1))
r <- residuals(lss)
chk("LSS response residual is y - mu under the location name", 
    is.list(r) && identical(names(r), "mu"))
mu <- predict(lss, parameter = "mu", type = "response")
chk("LSS response residual value", isTRUE(all.equal(r$mu, y - mu, tolerance = 1e-10)))
pr <- residuals(lss, parameter = "mu", type = "pearson")
sigma <- predict(lss, parameter = "sigma", type = "response")
chk("LSS Gaussian pearson is (y-mu)/sigma",
    isTRUE(all.equal(pr, (y - mu) / pmax(sigma, 1e-10), tolerance = 1e-10)))
chk("LSS pearson for a non-location parameter is refused",
    inherits(err(residuals(lss, parameter = "sigma", type = "pearson")), "error"))

# A family without a variance function must refuse, naming what pearson is.
st <- glmboostLSS(y ~ x1 + x2, data = d, families = StudentTLSS(),
                  control = boost_control(mstop = 20, nu = 0.1))
pe <- err(residuals(st, type = "pearson"))
chk("Student-t LSS pearson refuses naming the variance function",
    inherits(pe, "error") && grepl("variance function", conditionMessage(pe)))

dcount <- data.frame(y = rpois(n, exp(0.1 * x1 + 0.05 * x2)), x1 = x1, x2 = x2)
psc <- glmboostLSS(y ~ x1 + x2, data = dcount, families = PoissonLSS(),
                   control = boost_control(mstop = 20, nu = 0.1))
muc <- predict(psc, parameter = "mu", type = "response")
prc <- residuals(psc, parameter = "mu", type = "pearson")
chk("LSS Poisson pearson is (y-mu)/sqrt(mu)",
    isTRUE(all.equal(prc, (dcount$y - muc) / sqrt(pmax(muc, 1e-10)), tolerance = 1e-10)))

# Weighted LSS fit: residuals stay length-n and keep the y-mu value (weights
# enter the fit, not the residual definition).
w <- sample(c(0.5, 1, 2), n, replace = TRUE)
lw <- glmboostLSS(y ~ x1 + x2, data = d, families = GaussianLSS(),
                  weights = w, control = boost_control(mstop = 25, nu = 0.1))
rw <- residuals(lw)
muw <- predict(lw, parameter = "mu", type = "response")
chk("weighted LSS response residual length and value",
    length(rw$mu) == n && isTRUE(all.equal(rw$mu, y - muw, tolerance = 1e-10)))

# Survival and multivariate responses have no response residual.
t <- exp(0.3 * x1 + rnorm(n)); stv <- rbinom(n, 1, 0.7)
ds <- data.frame(t = t, st = stv, x1 = x1, x2 = x2)
slss <- glmboostLSS(Surv(t, st) ~ x1 + x2, data = ds, families = LogNormalLSS(),
                    control = boost_control(mstop = 20, nu = 0.1))
sre <- err(residuals(slss))
chk("survival LSS response residual refused as censored",
    inherits(sre, "error") && grepl("censored", conditionMessage(sre)))
sgl <- gamboostLSS(Surv(t, st) ~ x1 + x2, data = ds, families = LogNormalLSS(),
                   control = boost_control(mstop = 20, nu = 0.1))
sge <- err(residuals(sgl))
chk("gamboost survival LSS equally refuses", inherits(sge, "error"))

comp <- t(apply(matrix(rgamma(n * 3, shape = c(3, 4, 5)), n, 3), 1,
                function(r) r / sum(r)))
dc <- data.frame(y1 = comp[, 1], y2 = comp[, 2], y3 = comp[, 3], x1 = x1, x2 = x2)
dlss <- tryCatch(
  glmboostLSS(cbind(y1, y2, y3) ~ x1 + x2, data = dc, families = DirichletLSS(3),
              control = boost_control(mstop = 15, nu = 0.1)),
  error = function(e) e)
if (!inherits(dlss, "error")) {
  dre <- err(residuals(dlss))
  chk("Dirichlet LSS response residual refused as multivariate",
      inherits(dre, "error") && grepl("multivariate|composition", conditionMessage(dre)))
} else {
  cat("SKIP DirichletLSS fit unavailable:", conditionMessage(dlss), "\n")
}

# ── standard model: refusal surface ─────────────────────────────────
gl <- glmboost(y ~ x1 + x2, data = d, family = Laplace(),
               control = boost_control(mstop = 20, nu = 0.1))
pe <- err(residuals(gl, type = "pearson"))
chk("Laplace pearson refused, naming working",
    inherits(pe, "error") && grepl("working", conditionMessage(pe)))
chk("Laplace response residual still available",
    length(residuals(gl)) == n)

gc <- gamboost(Surv(t, st) ~ x1 + x2, data = ds, family = CoxPH(),
               control = boost_control(mstop = 10, nu = 0.1))
cre <- err(residuals(gc))
chk("CoxPH gamboost response residual refused as censored",
    inherits(cre, "error") && grepl("censored", conditionMessage(cre)))

glo <- glmboost(Surv(t, st) ~ x1 + x2, data = ds, family = CoxPH(),
                control = boost_control(mstop = 10, nu = 0.1))
cre <- err(residuals(glo))
chk("CoxPH glmboost response residual refused as censored",
    inherits(cre, "error") && grepl("censored", conditionMessage(cre)))

cat("residual contract (R): PASS\n")