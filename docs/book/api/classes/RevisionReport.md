# `RevisionReport`

Defined in `mboostrs.residual_checks`; exported as `mboostrs.RevisionReport`.


*Class*

```python
RevisionReport(findings: tuple[mboostrs.residual_checks.RevisionFinding, ...]) -> None
```

The three checks run together, in the order cluster, time, influence.

A check that was not asked for (no `cluster=` or no `time=`) is absent
from `findings`. `revisions` lists the revision of every flagged check.

Source: [`python/mboostrs/residual_checks.py` lines 125-156](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L125-L156)

## Members

| Member | Kind | Summary |
|---|---|---|
| `flagged` | Property | The findings whose pattern is present at ``alpha``. |
| `revisions` | Property | The revision of each flagged finding, in finding order. |
| `summary` | Property | Every finding, one block each. |

(revisionreport-flagged)=
## `RevisionReport.flagged`

*Property* of `RevisionReport`

```python
RevisionReport.flagged  # -> tuple[RevisionFinding, ...]
```

The findings whose pattern is present at `alpha`.

Source: [`python/mboostrs/residual_checks.py` lines 135-138](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L135-L138)

(revisionreport-revisions)=
## `RevisionReport.revisions`

*Property* of `RevisionReport`

```python
RevisionReport.revisions  # -> tuple[str, ...]
```

The revision of each flagged finding, in finding order.

Source: [`python/mboostrs/residual_checks.py` lines 140-143](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L140-L143)

(revisionreport-summary)=
## `RevisionReport.summary`

*Property* of `RevisionReport`

```python
RevisionReport.summary  # -> str
```

Every finding, one block each.

Source: [`python/mboostrs/residual_checks.py` lines 145-150](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/residual_checks.py#L145-L150)
