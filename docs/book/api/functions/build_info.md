# `build_info`

Defined in `mboostrs.build`; exported as `mboostrs.build_info`.


*Function*

```python
build_info() -> collections.abc.Mapping[str, Any]
```

Return immutable compile-time identity for the loaded Rust extension.

`mboostrs.__version__` identifies the public package release. This record
distinguishes binaries carrying that same version: it includes a digest of
native sources and Cargo configuration, the build-time Git revision and
dirty flag, compiler, target, enabled CPU and crate features, profile, and
ABI schema. It is embedded in the extension and therefore does not inspect
the current checkout at call time.

Source: [`python/mboostrs/build.py` lines 10-22](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/build.py#L10-L22)
