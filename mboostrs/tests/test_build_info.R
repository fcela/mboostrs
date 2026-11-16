#@validates contract.r.native_build_identity

library(mboostrs)

info <- mboostrs_build_info()
required <- c(
  "package_version", "source_digest", "git_revision", "git_dirty", "rustc",
  "target", "profile", "target_features", "crate_features", "abi_version",
  "build_id"
)
stopifnot(identical(names(info), required))
stopifnot(info$package_version == as.character(utils::packageVersion("mboostrs")))
stopifnot(grepl("^sha256:[0-9a-f]{64}$", info$source_digest))
stopifnot(grepl("^sha256:[0-9a-f]{64}$", info$build_id))
stopifnot(info$build_id != info$package_version)
stopifnot(info$abi_version == 1L)
stopifnot(nzchar(info$target), nzchar(info$profile), grepl("rustc", info$rustc))

cat("R native build identity passed\n")
