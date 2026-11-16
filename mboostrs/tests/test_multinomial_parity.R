# Parity check: our Multinomial (gamboost, bols) vs R's mboost.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping Multinomial parity check")
  quit(save = "no")
}
library(mboostrs)
suppressMessages(library(mboost)); data(iris)

chk <- function(a, b, what, tol = 1e-6) {
  md <- max(abs(a - b))
  if (md >= tol) stop(sprintf("%s mismatch: maxdiff=%.3e", what, md))
  cat(sprintf("OK %-22s maxdiff=%.2e\n", what, md))
}

# Reference: R mboost multinomial via bols(x) %O% bols(class).
myiris <- as.list(iris)
myiris$class <- factor(levels(iris$Species)[-nlevels(iris$Species)])
ref <- mboost(Species ~ bols(Sepal.Length, df = 2) %O%
              bols(class, df = 2, contrasts.arg = "contr.dummy"),
              data = myiris, family = Multinomial(),
              control = boost_control(mstop = 20, nu = 0.1))
rrisk <- as.numeric(risk(ref))
rfit  <- as.numeric(fitted(ref))                 # column-major (n, K-1)
rresp <- matrix(as.numeric(predict(ref, type = "response")), ncol = 3)

# Ours.
m <- mboostrs::gamboost(Species ~ bols(Sepal.Length), data = iris,
                        family = mboostrs::Multinomial(),
                        control = mboostrs::boost_control(mstop = 20, nu = 0.1))
nr <- 2
ofit_cm <- as.numeric(matrix(m$fitted, ncol = nr, byrow = TRUE))   # -> column-major

chk(m$risk, rrisk, "bols risk path")
chk(ofit_cm, rfit, "bols fitted")
chk(predict(m, type = "response"), rresp, "bols response")

# predict(type="link") returns an (n, K-1) matrix; response rows sum to 1.
stopifnot(all(dim(predict(m, type = "link")) == c(150, 2)))
stopifnot(all(abs(rowSums(predict(m, type = "response")) - 1) < 1e-10))

# Smooth (bbs) multinomial: matches R's bbs(x) %O% bols(class).
refb <- mboost(Species ~ bbs(Sepal.Length, df = 4) %O%
               bols(class, df = 2, contrasts.arg = "contr.dummy"),
               data = myiris, family = Multinomial(),
               control = boost_control(mstop = 20, nu = 0.1))
mb_ <- mboostrs::gamboost(Species ~ bbs(Sepal.Length, df = 4), data = iris,
                          family = mboostrs::Multinomial(),
                          control = mboostrs::boost_control(mstop = 20, nu = 0.1))
chk(mb_$risk, as.numeric(risk(refb)), "bbs risk path", tol = 1e-4)
chk(as.numeric(matrix(mb_$fitted, ncol = nr, byrow = TRUE)),
    as.numeric(fitted(refb)), "bbs fitted", tol = 1e-5)
chk(predict(mb_, type = "response"),
    matrix(as.numeric(predict(refb, type = "response")), ncol = 3),
    "bbs response", tol = 1e-5)

# Weighted fits apply each observation's weight once. The response is stored
# flat, K-1 entries per observation, with that observation's weight repeated
# over its block; indexing those weights by observation gave observation i the
# weight of observation i/(K-1). On this fixture that reported 220.2557 where
# upstream reports 236.4396.
set.seed(1)
wts <- as.numeric(sample(1:4, nrow(iris), replace = TRUE))
refw <- mboost(Species ~ bols(Sepal.Length, df = 2) %O%
               bols(class, df = 2, contrasts.arg = "contr.dummy"),
               data = myiris, family = Multinomial(), weights = wts,
               control = boost_control(mstop = 20, nu = 0.1))
mw <- mboostrs::gamboost(Species ~ bols(Sepal.Length), data = iris,
                         family = mboostrs::Multinomial(), weights = wts,
                         control = mboostrs::boost_control(mstop = 20, nu = 0.1))
chk(mw$risk, as.numeric(risk(refw)), "weighted risk path")
chk(as.numeric(matrix(mw$fitted, ncol = nr, byrow = TRUE)),
    as.numeric(fitted(refw)), "weighted fitted")

# Unsupported learners (e.g. btree) are rejected for Multinomial.
err <- tryCatch({
  mboostrs::gamboost(Species ~ btree(Sepal.Length), data = iris,
                     family = mboostrs::Multinomial(),
                     control = mboostrs::boost_control(mstop = 5, nu = 0.1))
  "no error"
}, error = function(e) conditionMessage(e))
if (!grepl("Multinomial", err)) stop("expected a clear rejection error, got: ", err)
cat("OK unsupported learner rejected for Multinomial\n")

cat("Multinomial R-binding parity: PASS\n")
