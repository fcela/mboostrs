# `suggest_terms`

Defined in `mboostrs.terms`; exported as `mboostrs.suggest_terms`.


*Function*

```python
suggest_terms(data, response=None, df=4.0, smooth='bbs')
```

The main-effect right-hand side suggested for `data`.

**Examples**

```python
>>> suggest_terms({"age": [30, 41, 52], "sex": [0, 1, 0], "y": [1.0, 2.0, 3.0]},
...               response="y")
'bbs(age, df=4.0) + bols(sex)'
```

Source: [`python/mboostrs/terms.py` lines 623-632](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/terms.py#L623-L632)
