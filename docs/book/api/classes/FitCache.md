# `FitCache`

Defined in `mboostrs.policy`; exported as `mboostrs.FitCache`.


*Class*

```python
FitCache(*, build=None, enabled=True)
```

Memo for evaluated specifications within a Python process.

The key carries the right-hand side, a content hash of the data, the
family and step count, the fold assignment, the weights, the transform
and the build identity. Two calls that differ in any of these are
different computations and cannot share an entry; a hit therefore
returns the value the same computation produced. Opaque configuration
(such as a callable whose closure cannot be fingerprinted) bypasses the
cache. Prepared learners are not shared between distinct specifications.

Source: [`python/mboostrs/policy.py` lines 217-277](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L217-L277)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`key`](#fitcache-key) | Method | Key the full computation, or return ``None`` to bypass caching. |
| [`get_or_run`](#fitcache-get_or_run) | Method | Return the cached value for `key`, computing it on a miss. |
| [`clear`](#fitcache-clear) | Method | Drop every entry and reset the counters. |
| [`to_dict`](#fitcache-to_dict) | Method | Cache size and hit/miss counts, as a JSON-safe record. |

(fitcache-key)=
## `FitCache.key`

*Method* of `FitCache`

```python
FitCache.key(
    rhs,
    data,
    family,
    mstop,
    folds,
    weights=None,
    transform=None,
    response=None,
    seed=0,
    metric='mse',
    na_action='error',
    unseen_levels='error',
    fit_kwargs=None,
    score_parameter=None
)
```

Key the full computation, or return `None` to bypass caching.

Source: [`python/mboostrs/policy.py` lines 238-253](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L238-L253)

(fitcache-get_or_run)=
## `FitCache.get_or_run`

*Method* of `FitCache`

```python
FitCache.get_or_run(key, compute)
```

Return the cached value for `key`, computing it on a miss.

Source: [`python/mboostrs/policy.py` lines 255-266](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L255-L266)

(fitcache-clear)=
## `FitCache.clear`

*Method* of `FitCache`

```python
FitCache.clear()
```

Drop every entry and reset the counters.

Source: [`python/mboostrs/policy.py` lines 268-272](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L268-L272)

(fitcache-to_dict)=
## `FitCache.to_dict`

*Method* of `FitCache`

```python
FitCache.to_dict(*, include_arrays=False)
```

Cache size and hit/miss counts, as a JSON-safe record.

Source: [`python/mboostrs/policy.py` lines 274-277](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/policy.py#L274-L277)
