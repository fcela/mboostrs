# mboost and mboostrs both define mstop, risk, varimp, nuisance, extract and
# cvrisk as generics. With both packages attached, the one attached last used
# to mask the other's generic, and calls on the other package's models failed
# with "no applicable method". Each order is checked in a fresh R process.
if (!requireNamespace("mboost", quietly = TRUE)) {
  message("mboost not installed; skipping shared generic check")
  quit(save = "no")
}
script <- '
args <- commandArgs(TRUE)
if (args[1] == "mboostrs_last") suppressMessages({library(mboost); library(mboostrs)}) else
  suppressMessages({library(mboostrs); library(mboost)})
set.seed(1); d <- data.frame(x = rnorm(60), z = rnorm(60)); d$y <- d$x + rnorm(60)
up <- mboost::glmboost(y ~ x + z, data = d)
ours <- mboostrs::glmboost(y ~ x + z, data = d)
folds <- mboost::cv(model.weights(up), type = "kfold", B = 3)
for (obj in list(up, ours)) {
  mstop(obj); risk(obj); varimp(obj); nuisance(obj)
  extract(obj, "coefficients"); cvrisk(obj, folds = folds)
}
stopifnot(mstop(up) == 100, mstop(ours) == 100)
cat("ok\n")
'
path <- tempfile(fileext = ".R")
writeLines(script, path)
rscript <- file.path(R.home("bin"), "Rscript")
for (order in c("mboostrs_last", "mboostrs_first")) {
  out <- suppressWarnings(system2(rscript, c(path, order), stdout = TRUE, stderr = TRUE))
  if (!any(out == "ok")) stop("shared generics fail with ", order, ":\n",
                              paste(out, collapse = "\n"), call. = FALSE)
  cat("shared generics,", order, ": ok\n")
}
