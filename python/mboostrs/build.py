"""Identity of the native binary loaded by this Python process."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def build_info() -> Mapping[str, Any]:
    """Return immutable compile-time identity for the loaded Rust extension.

    ``mboostrs.__version__`` identifies the public package release. This record
    distinguishes binaries carrying that same version: it includes a digest of
    native sources and Cargo configuration, the build-time Git revision and
    dirty flag, compiler, target, enabled CPU and crate features, profile, and
    ABI schema. It is embedded in the extension and therefore does not inspect
    the current checkout at call time.
    """
    from mboostrs._rust import build_info as _native_build_info

    return MappingProxyType(dict(_native_build_info()))
