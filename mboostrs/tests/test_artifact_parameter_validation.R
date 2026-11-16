#@validates contract.r.artifact_family_params
# The R artifact reader refuses what the Python one refuses.
#
# P0.3's closure condition: "Audit the Rust and R artifact readers separately;
# the Python artifact test is not evidence that those readers reject the same
# invalid payload before numerical dispatch."
#
# It was not. A hand-edited artifact with `tau = 1.5` on a `Quantile` family
# was refused by `mboostrs.load_model` in Python and **loaded** in R. The
# check loss for `tau = 1.5` is minimised by any fit below every observation,
# so the model that came back was not merely odd, it was unbounded -- and it
# predicted.
#
# An artifact is a file, and a file can say anything. Reading one is not
# evidence that whatever wrote it validated the contents.
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  message("jsonlite not installed; skipping artifact validation check")
  quit(save = "no")
}
library(mboostrs)

scratch <- tempfile("mboostrs-artifact-")
dir.create(scratch)
on.exit(unlink(scratch, recursive = TRUE), add = TRUE)

# A valid artifact to edit. The R frontend reads artifacts and does not write
# them, so this is a committed fixture from the Python writer -- the same
# source as the other reader tests use.
valid <- file.path("fixtures", "glm_quantile.json")
if (!file.exists(valid)) {
  valid <- file.path("mboostrs", "tests", "fixtures", "glm_quantile.json")
}
if (!file.exists(valid)) {
  message("glm_quantile.json fixture not found; skipping")
  quit(save = "no")
}

if (inherits(tryCatch(load_model(valid), error = function(e) e), "error")) {
  stop("the valid artifact should load")
}
cat("OK valid artifact loads\n")

corrupt <- function(field, value, name) {
  doc <- jsonlite::fromJSON(valid, simplifyVector = FALSE)
  doc$state$items$family$name <- name
  doc$state$items$family$params$items[[field]] <- value
  path <- file.path(scratch, paste0(name, "-", field, ".json"))
  writeLines(jsonlite::toJSON(doc, auto_unbox = TRUE, digits = NA), path)
  path
}

cases <- list(
  list("Quantile", "tau", 1.5, "in \\(0, 1\\)"),
  list("Quantile", "tau", 0.0, "in \\(0, 1\\)"),
  list("Quantile", "qoffset", 1.0, "in \\(0, 1\\)"),
  list("Expectile", "tau", -0.2, "in \\(0, 1\\)"),
  list("Huber", "delta", -1.0, "must be positive"),
  list("Cindex", "sigma", 0.0, "must be positive"),
  list("PropOdds", "n_levels", 2, "at least 3"),
  list("Multinomial", "n_classes", 1, "at least 3")
)

for (case in cases) {
  name <- case[[1]]; field <- case[[2]]; value <- case[[3]]; expect <- case[[4]]
  path <- corrupt(field, value, name)
  message_text <- tryCatch({
    load_model(path)
    NULL
  }, error = function(e) conditionMessage(e))
  if (is.null(message_text)) {
    stop(sprintf("%s with %s = %s loaded, and should not have",
                 name, field, format(value)))
  }
  if (!grepl(expect, message_text)) {
    stop(sprintf("%s/%s: the refusal does not say %s; got: %s",
                 name, field, sQuote(expect), message_text))
  }
  cat(sprintf("OK %-12s %-10s = %-6s refused\n", name, field, format(value)))
}

# A value inside its domain still loads, so the check is not simply refusing
# every edited file.
inside <- corrupt("tau", 0.9, "Quantile")
if (inherits(tryCatch(load_model(inside), error = function(e) e), "error")) {
  stop("tau = 0.9 is valid and should load")
}
cat("OK Quantile     tau        = 0.9    loads\n")

# A family with no recorded parameter domain is not blocked by the check.
untouched <- jsonlite::fromJSON(valid, simplifyVector = FALSE)
untouched$state$items$family$name <- "Gaussian"
gaussian_path <- file.path(scratch, "gaussian.json")
writeLines(jsonlite::toJSON(untouched, auto_unbox = TRUE, digits = NA),
           gaussian_path)
if (inherits(tryCatch(load_model(gaussian_path), error = function(e) e),
             "error")) {
  stop("a family with no parameter domain should still load")
}
cat("OK Gaussian     (no params)         loads\n")

cat("artifact parameter validation: R refuses what Python refuses\n")
