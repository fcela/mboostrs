//! Compile-time identity of the loaded native implementation.
//!
//! A package version identifies an API release; it cannot distinguish a
//! portable wheel from a local `target-cpu=native` build, or two dirty builds
//! made from different Rust sources. [`build_info`] reports that identity from
//! constants embedded by `build.rs`, so it describes the loaded binary rather
//! than the source checkout from which a caller happens to run Python or R.

/// Schema of the native ABI/build-identity record.
pub const NATIVE_ABI_VERSION: u32 = 1;

/// Immutable build identity embedded in a native mboostrs library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildInfo {
    /// Workspace package version compiled into the native crate.
    pub package_version: &'static str,
    /// Digest of Rust sources and Cargo configuration used for this build.
    pub source_digest: &'static str,
    /// Git revision visible at build time, or `"unknown"` for an unpacked source tree.
    pub git_revision: &'static str,
    /// Whether native source/configuration paths were dirty at build time.
    pub git_dirty: bool,
    /// Full `rustc -vV` output, flattened to one line.
    pub rustc: &'static str,
    /// Cargo target triple.
    pub target: &'static str,
    /// Cargo profile, normally `release` for wheels and R packages.
    pub profile: &'static str,
    /// CPU features enabled for the compiled target.
    pub target_features: &'static str,
    /// Enabled `mboostrs-core` Cargo features.
    pub crate_features: &'static str,
    /// ABI/build-record schema version.
    pub abi_version: u32,
    /// Digest of all preceding identity fields.
    pub build_id: &'static str,
}

/// Return identity embedded in the currently loaded native library.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        package_version: env!("CARGO_PKG_VERSION"),
        source_digest: env!("MBOOSTRS_SOURCE_DIGEST"),
        git_revision: env!("MBOOSTRS_GIT_REVISION"),
        git_dirty: env!("MBOOSTRS_GIT_DIRTY") == "true",
        rustc: env!("MBOOSTRS_RUSTC"),
        target: env!("MBOOSTRS_TARGET"),
        profile: env!("MBOOSTRS_PROFILE"),
        target_features: env!("MBOOSTRS_TARGET_FEATURES"),
        crate_features: env!("MBOOSTRS_CRATE_FEATURES"),
        abi_version: NATIVE_ABI_VERSION,
        build_id: env!("MBOOSTRS_BUILD_ID"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_identity_is_complete_and_not_the_release_version() {
        let info = build_info();
        assert_eq!(info.package_version, env!("CARGO_PKG_VERSION"));
        assert!(info.source_digest.starts_with("sha256:"));
        assert_eq!(info.source_digest.len(), 71);
        assert!(info.build_id.starts_with("sha256:"));
        assert_eq!(info.build_id.len(), 71);
        assert_ne!(info.build_id, info.package_version);
        assert!(!info.rustc.is_empty());
        assert!(!info.target.is_empty());
        assert!(!info.profile.is_empty());
        assert_eq!(info.abi_version, 1);
    }
}
