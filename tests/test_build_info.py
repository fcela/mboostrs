"""The loaded native binary has an identity distinct from package version."""

from types import MappingProxyType

import pytest

import mboostrs as mb


REQUIRED = {
    "package_version", "source_digest", "git_revision", "git_dirty", "rustc",
    "target", "profile", "target_features", "crate_features", "abi_version",
    "build_id",
}


def test_build_info_describes_the_loaded_extension():
    info = mb.build_info()
    assert isinstance(info, MappingProxyType)
    assert set(info) == REQUIRED
    assert info["package_version"] == mb.__version__
    assert info["source_digest"].startswith("sha256:")
    assert len(info["source_digest"]) == 71
    assert info["build_id"].startswith("sha256:")
    assert len(info["build_id"]) == 71
    assert info["build_id"] != info["package_version"]
    assert info["abi_version"] == 1
    assert info["profile"] in {"debug", "release"}
    assert info["target"]
    assert "rustc" in info["rustc"]
    with pytest.raises(TypeError):
        info["build_id"] = "changed"


def test_same_release_can_be_detected_as_a_different_native_build():
    first = dict(mb.build_info())
    second = {**first, "build_id": "sha256:" + "0" * 64}
    assert first["package_version"] == second["package_version"]
    assert first["build_id"] != second["build_id"]


def test_benchmark_metadata_records_loaded_native_identity():
    from benches.benchmark_contract import machine_metadata

    metadata = machine_metadata()
    assert metadata["native_build"] == dict(mb.build_info())
