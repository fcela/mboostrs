# `collect_warnings`

Defined in `mboostrs._warning_collect`; exported as `mboostrs.collect_warnings`.


*Function*

```python
collect_warnings()
```

Show each repeated warning once, with a count, when the block ends.

Use it around a loop or a script that calls mboostrs many times:

```python
with mb.collect_warnings():
    for seed in range(50):
        result = mb.ci_test(data, "x", "y", given=("z",), seed=seed)
```

Every warning raised inside the block is recorded, from mboostrs or from
any other package. At exit, warnings with the same wording (numbers aside)
are grouped. A warning that occurred once is issued as it was, with its
original location. A group of several is issued once, with the wording of
its first occurrence (so the numbers are those of the first call) and a
note of how many times it occurred. Outside a block, mboostrs warnings are
issued as they occur, except that the extrapolation summary of
`ci_test` is shown once per process.

Source: [`python/mboostrs/_warning_collect.py` lines 40-67](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/_warning_collect.py#L40-L67)
