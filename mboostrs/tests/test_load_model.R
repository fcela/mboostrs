# Cross-language artifact reader: load a model saved by Python `save_model`
# and score it in R. Fixtures in tests/fixtures/ were written by the Python
# writer; expected.json holds the predictions Python produced for the same
# newdata. The R predictions must match exactly.
#
# Run with: Rscript mboostrs/tests/test_load_model.R

library(mboostrs)

passed <- 0L
failed <- 0L

check <- function(name, ok, detail = "") {
  if (isTRUE(ok)) {
    passed <<- passed + 1L
    cat("  PASS: ", name, "\n", sep = "")
  } else {
    failed <<- failed + 1L
    cat("  FAIL: ", name, " ", detail, "\n", sep = "")
  }
}

# Resolve the fixtures directory whether run from the repo root or tests/.
fx <- "mboostrs/tests/fixtures"
if (!dir.exists(fx)) fx <- "fixtures"
if (!dir.exists(fx)) fx <- file.path(getwd(), "tests", "fixtures")
stopifnot(dir.exists(fx))

exp <- jsonlite::fromJSON(file.path(fx, "expected.json"))
newdata <- as.data.frame(exp$newdata)

for (fam in c("gaussian", "poisson", "binomial")) {
  m <- load_model(file.path(fx, sprintf("glm_%s.json", fam)))
  check(sprintf("%s loads as GLMBoostModel", fam), m$model_class == "GLMBoostModel")
  for (ty in c("link", "response")) {
    got <- predict(m, newdata = newdata, type = ty)
    want <- as.numeric(exp$expected[[fam]][[ty]])
    md <- max(abs(got - want))
    check(sprintf("%s %s matches Python", fam, ty), md < 1e-9,
          sprintf("(maxdiff=%.2e)", md))
  }
}

# A spline model loads but is not scorable in R.
gam <- load_model(file.path(fx, "gam_spline.json"))
check("gamboost artifact loads", gam$model_class == "GAMBoostModel")
refused <- tryCatch({
  predict(gam, newdata = newdata)
  FALSE
}, error = function(e) grepl("linear glmboost models only", conditionMessage(e)))
check("gamboost prediction is refused with a clear message", refused)

# Unsupported schema version is rejected.
tmp <- tempfile(fileext = ".json")
doc <- jsonlite::fromJSON(file.path(fx, "glm_gaussian.json"), simplifyVector = FALSE)
doc$schema_version <- 999L
writeLines(jsonlite::toJSON(doc, auto_unbox = TRUE, digits = NA), tmp)
bad <- tryCatch({ load_model(tmp); FALSE },
                error = function(e) grepl("schema version", conditionMessage(e)))
check("unsupported schema version rejected", bad)

cat("\n", passed, " passed, ", failed, " failed\n", sep = "")
if (failed > 0L) quit(status = 1L)
