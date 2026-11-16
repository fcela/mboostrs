# `label_evidence`

Defined in `mboostrs.terms`; exported as `mboostrs.label_evidence`.


*Function*

```python
label_evidence(name, values, response)
```

Read a whole-number column against the response; see `LabelEvidence`.

Returns `None` when the question does not arise: a column that is not
whole numbers, has at most `CATEGORICAL_MAX_LEVELS` values (those are
already read as levels) or more than `LABEL_MAX_LEVELS` (too many levels
to fit one column each), or has too few repeated values to compare level
means. The statistics use the values seen at least twice, and only when
those values hold at least `LABEL_MIN_COVERAGE` of the rows and average
two or more rows each: a district code where a few districts appear once
is still read, while a column of mostly unique integers is not.

Source: [`python/mboostrs/terms.py` lines 216-270](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L216-L270)
