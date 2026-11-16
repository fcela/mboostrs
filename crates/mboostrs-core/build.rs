use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn collect_sources(root: &Path, directory: &Path, paths: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_sources(root, &path, paths);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            paths.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
        }
    }
}

fn command(root: &Path, program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!value.is_empty()).then_some(value)
}

/// Lowercase hex of a digest. sha2 0.11 returns a `hybrid-array` output
/// block, which implements no `LowerHex`, so the encoding is done here.
fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn source_digest(root: &Path) -> String {
    println!("cargo:rerun-if-changed={}", root.join("crates").display());
    let mut paths = Vec::new();
    collect_sources(root, &root.join("crates"), &mut paths);
    for name in ["Cargo.toml", "Cargo.lock", ".cargo/config.toml"] {
        let path = PathBuf::from(name);
        if root.join(&path).is_file() {
            paths.push(path);
        }
    }
    paths.sort();
    let mut digest = Sha256::new();
    for relative in paths {
        let absolute = root.join(&relative);
        println!("cargo:rerun-if-changed={}", absolute.display());
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        if let Ok(contents) = fs::read(absolute) {
            digest.update(contents);
        }
        digest.update([0]);
    }
    format!("sha256:{}", hex(&digest.finalize()))
}

fn watch_git_identity(root: &Path) {
    let git = root.join(".git");
    let head = git.join("HEAD");
    let index = git.join("index");
    println!("cargo:rerun-if-changed={}", head.display());
    println!("cargo:rerun-if-changed={}", index.display());
    if let Ok(value) = fs::read_to_string(&head) {
        if let Some(reference) = value.trim().strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed={}", git.join(reference).display());
        }
    }
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let root = manifest.ancestors().nth(2).unwrap_or(&manifest);
    watch_git_identity(root);
    let digest = source_digest(root);
    let revision = command(root, "git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = command(
        root,
        "git",
        &[
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            ".cargo",
        ],
    )
    .is_some_and(|value| !value.is_empty());
    let rustc = env::var("RUSTC")
        .ok()
        .and_then(|compiler| command(root, &compiler, &["-vV"]))
        .unwrap_or_else(|| "unknown".into())
        .replace('\n', " | ");
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let mut crate_features: Vec<_> = env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .collect();
    crate_features.sort();
    let crate_features = crate_features
        .join(",")
        .to_ascii_lowercase()
        .replace('_', "-");
    let identity_material = format!(
        "{digest}\0{revision}\0{dirty}\0{rustc}\0{target}\0{profile}\0{target_features}\0{crate_features}\0abi=1"
    );
    let build_id = format!(
        "sha256:{}",
        hex(&Sha256::digest(identity_material.as_bytes()))
    );

    for (name, value) in [
        ("MBOOSTRS_SOURCE_DIGEST", digest),
        ("MBOOSTRS_GIT_REVISION", revision),
        ("MBOOSTRS_GIT_DIRTY", dirty.to_string()),
        ("MBOOSTRS_RUSTC", rustc),
        ("MBOOSTRS_TARGET", target),
        ("MBOOSTRS_PROFILE", profile),
        ("MBOOSTRS_TARGET_FEATURES", target_features),
        ("MBOOSTRS_CRATE_FEATURES", crate_features),
        ("MBOOSTRS_BUILD_ID", build_id),
    ] {
        println!("cargo:rustc-env={name}={value}");
    }
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
