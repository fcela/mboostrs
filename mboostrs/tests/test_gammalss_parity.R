#@validates parity.r.gammalss
# Parity check: our GammaLSS (glmboostLSS + gamboostLSS) vs R's gamboostLSS.
# Compared with R mboost/gamboostLSS (or with values recorded from them), so
# fitted on the raw response as R does (see ?gamboostLSS, response_scale).
options(mboostrs.response_scale = "none")
if (!requireNamespace("gamboostLSS", quietly = TRUE)) {
  message("gamboostLSS not installed; skipping GammaLSS parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(gamboostLSS))

set.seed(1); n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
mu <- exp(0.5 + 0.3 * x1); sig <- exp(0.2 + 0.2 * x2)
y <- rgamma(n, shape = sig, scale = mu / sig)
d <- data.frame(y = y, x1 = x1, x2 = x2)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-26s maxdiff=%.2e\n", what, md))
}

# glmboostLSS (linear). Namespace-qualify "ours" (gamboostLSS is attached too).
ref <- gamboostLSS::glmboostLSS(y ~ x1 + x2, data = d,
                               families = gamboostLSS::GammaLSS(),
                               control = mboost::boost_control(mstop = 100, nu = 0.1))
# R searches the log shape offset over [0, max(y)]; here the marginal shape is
# below 1, so R's offset is the boundary (about 0) and mboostrs's, which opens
# the interval in that case, is -0.107 (see "GammaLSS offset search" in the
# book's parity chapter). The boosting path is compared from R's offsets.
r_offsets <- c(ref$mu$offset, ref$sigma$offset)
ours <- mboostrs::glmboostLSS(y ~ x1 + x2, data = d,
                             families = mboostrs::GammaLSS(),
                             control = mboostrs::boost_control(mstop = 100, nu = 0.1),
                             offset = r_offsets)
chk(predict(ours, parameter = "mu", type = "link"),
    as.numeric(predict(ref, parameter = "mu", type = "link")), "glm mu (link)")
chk(predict(ours, parameter = "sigma", type = "link"),
    as.numeric(predict(ref, parameter = "sigma", type = "link")), "glm sigma (link)")
chk(predict(ours, parameter = "sigma", type = "response"),
    as.numeric(predict(ref, parameter = "sigma", type = "response")), "glm sigma (response)")

# gamboostLSS (smooth, bbs).
xg <- runif(n, -1, 1)
mug <- exp(0.5 + 0.5 * sin(2 * xg)); sigg <- exp(0.3 + 0.3 * xg)
yg <- rgamma(n, shape = sigg, scale = mug / sigg)
dg <- data.frame(y = yg, x = xg)
refg <- gamboostLSS::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                                families = gamboostLSS::GammaLSS(),
                                control = mboost::boost_control(mstop = 100, nu = 0.1))
oursg <- mboostrs::gamboostLSS(y ~ bbs(x, knots = 20, df = 4), data = dg,
                              families = mboostrs::GammaLSS(),
                              control = mboostrs::boost_control(mstop = 100, nu = 0.1),
                              offset = c(refg$mu$offset, refg$sigma$offset))
chk(predict(oursg, parameter = "mu", type = "link"),
    as.numeric(predict(refg, parameter = "mu", type = "link")), "gam mu (link)")
chk(predict(oursg, parameter = "sigma", type = "link"),
    as.numeric(predict(refg, parameter = "sigma", type = "link")), "gam sigma (link)")

cat("GammaLSS R-binding parity: PASS\n")
