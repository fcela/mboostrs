# `GaussReg`

Defined in `mboostrs.families`; exported as `mboostrs.GaussReg`.


*Class*

```python
GaussReg()
```

Squared error loss with an identity link, without R's `mboost` equivalent.

Fits the same model as `Gaussian`. It exists because `mboost` names
its squared-error family `Gaussian` while some references reserve that name
for the likelihood rather than the loss; both are provided so a translated
script keeps working.

Source: [`python/mboostrs/families.py` lines 67-74](https://github.com/fcela/mboostrs/blob/main/python/mboostrs/families.py#L67-L74)
