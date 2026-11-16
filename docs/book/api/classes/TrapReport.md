# `TrapReport`

Defined in `mboostrs.causal`; exported as `mboostrs.TrapReport`.


*Class*

```python
TrapReport(
    findings: tuple[mboostrs.causal.TrapFinding, ...] = (),
    spec: dict = <factory>
) -> None
```

Findings from checking a declared causal design against the supplied data.

Source: [`python/mboostrs/causal.py` lines 105-145](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L105-L145)

## Members

| Member | Kind | Summary |
|---|---|---|
| [`blocking`](#trapreport-blocking) | Property | Findings that leave the estimand unidentified. |
| [`advisory`](#trapreport-advisory) | Property | Findings that change which population the estimate describes. |
| [`clean`](#trapreport-clean) | Property | True when the design raises no finding at all. |
| [`to_dict`](#trapreport-to_dict) | Method | A JSON-safe record of the findings. |

(trapreport-blocking)=
## `TrapReport.blocking`

*Property* of `TrapReport`

```python
TrapReport.blocking
```

Findings that leave the estimand unidentified.

Source: [`python/mboostrs/causal.py` lines 112-115](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L112-L115)

(trapreport-advisory)=
## `TrapReport.advisory`

*Property* of `TrapReport`

```python
TrapReport.advisory
```

Findings that change which population the estimate describes.

Source: [`python/mboostrs/causal.py` lines 117-120](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L117-L120)

(trapreport-clean)=
## `TrapReport.clean`

*Property* of `TrapReport`

```python
TrapReport.clean
```

True when the design raises no finding at all.

Source: [`python/mboostrs/causal.py` lines 122-125](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L122-L125)

(trapreport-to_dict)=
## `TrapReport.to_dict`

*Method* of `TrapReport`

```python
TrapReport.to_dict(*, include_arrays=False)
```

A JSON-safe record of the findings.

Source: [`python/mboostrs/causal.py` lines 127-140](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/causal.py#L127-L140)
