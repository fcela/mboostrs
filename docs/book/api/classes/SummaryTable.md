# `SummaryTable`

Defined in `mboostrs.summary`; exported as `mboostrs.SummaryTable`.


*Class*

```python
SummaryTable(title, headers, rows, align=None, note=None)
```

A titled table of rows, rendered as text or HTML.

`align` gives one of "l" or "r" per column; numeric columns read better
right-aligned and names left-aligned. Cells keep their values in `rows`;
they are formatted only when the table is rendered, so `as_dict()` holds
numbers.

Source: [`python/mboostrs/summary.py` lines 69-148](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/summary.py#L69-L148)

## Members

| Member | Kind | Summary |
|---|---|---|
| `widths` | Method |  |
| `as_text` | Method |  |
| `as_html` | Method |  |
| `as_dict` | Method |  |

(summarytable-widths)=
## `SummaryTable.widths`

*Method* of `SummaryTable`

```python
SummaryTable.widths()
```

*No description in the docstring.*

Source: [`python/mboostrs/summary.py` lines 88-93](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/summary.py#L88-L93)

(summarytable-as_text)=
## `SummaryTable.as_text`

*Method* of `SummaryTable`

```python
SummaryTable.as_text()
```

*No description in the docstring.*

Source: [`python/mboostrs/summary.py` lines 95-112](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/summary.py#L95-L112)

(summarytable-as_html)=
## `SummaryTable.as_html`

*Method* of `SummaryTable`

```python
SummaryTable.as_html()
```

*No description in the docstring.*

Source: [`python/mboostrs/summary.py` lines 114-144](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/summary.py#L114-L144)

(summarytable-as_dict)=
## `SummaryTable.as_dict`

*Method* of `SummaryTable`

```python
SummaryTable.as_dict()
```

*No description in the docstring.*

Source: [`python/mboostrs/summary.py` lines 146-148](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/summary.py#L146-L148)
