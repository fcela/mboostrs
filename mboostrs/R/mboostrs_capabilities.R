# Public views of the generated capability contract.
#
# The row data in aaa-capabilities-generated.R is generated from the canonical
# python/mboostrs/capability_manifest.json file. Numerical dispatch remains in
# the implementation; these functions only expose its checked support claims.

#' Native build identity
#'
#' Reports compile-time identity embedded in the loaded Rust library. Package
#' version identifies the public release; `build_id` additionally distinguishes
#' source changes, compiler, target features, target, profile, and ABI schema.
#' The function does not inspect the current source checkout.
#' @return A named list describing the loaded native build.
#' @export
mboostrs_build_info <- function() {
  .Call("wrap__build_info_r", PACKAGE = "mboostrs")
}

.capability_df <- function(rows) {
  if (length(rows) == 0L) return(data.frame())
  columns <- names(rows[[1L]])
  out <- as.data.frame(
    stats::setNames(lapply(columns, function(column) {
      vapply(rows, function(row) {
        value <- row[[column]]
        if (is.null(value)) "" else as.character(value)
      }, character(1))
    }), columns),
    stringsAsFactors = FALSE
  )
  rownames(out) <- NULL
  out
}

.capability_output <- function(rows, as.data.frame) {
  if (!as.data.frame) return(rows)
  .capability_df(rows)
}

#' Family capability and validation matrix
#'
#' Legacy yes/no columns are retained for compatibility. Frontend-specific
#' status columns use `verified`, `smoke`, `intentional-deviation`,
#' `unsupported`, or `unverified`, and `validation_ids` points into the
#' generated validation ledger.
#'
#' @param as.data.frame Return a data frame. If FALSE, returns a list of rows.
#' @return Support metadata for standard and LSS families.
#' @export
family_capabilities <- function(as.data.frame = TRUE) {
  .capability_output(.capability_family_rows, as.data.frame)
}

#' Learner capability and validation matrix
#'
#' @param as.data.frame Return a data frame. If FALSE, returns a list of rows.
#' @return Support metadata for base learners and related R features.
#' @export
learner_capabilities <- function(as.data.frame = TRUE) {
  .capability_output(.capability_learner_rows, as.data.frame)
}

#' R method capability and validation matrix
#'
#' @param as.data.frame Return a data frame. If FALSE, returns a list of rows.
#' @return Support metadata for public R model and result methods.
#' @export
method_capabilities <- function(as.data.frame = TRUE) {
  .capability_output(.capability_method_rows, as.data.frame)
}

#' Capability validation ledger
#'
#' @param as.data.frame Return a data frame. If FALSE, returns a list of rows.
#' @return Explicit test IDs, oracles, compared outputs, and tolerances backing
#'   advertised capabilities.
#' @export
validation_capabilities <- function(as.data.frame = TRUE) {
  .capability_output(.capability_validation_rows, as.data.frame)
}

#' Full mboostrs capability matrix
#'
#' @param as.data.frame Return data frames. If FALSE, returns lists of rows.
#' @return A list with `families`, `learners`, and `methods`.
#' @export
mboostrs_capabilities <- function(as.data.frame = TRUE) {
  list(
    families = family_capabilities(as.data.frame = as.data.frame),
    learners = learner_capabilities(as.data.frame = as.data.frame),
    methods = method_capabilities(as.data.frame = as.data.frame)
  )
}
